"""
This script implements run_evaluation using Morph Cloud.
It builds a snapshot via a chain of .asetup() commands
(including repository cloning and checkout using the
environment_setup_commit), starting an instance, applying 
a patch, running tests and generating a report.
"""

import time
import json
import asyncio
import logging
import traceback

from typing import cast
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Configure logging (adjust level and format as needed)
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s"
)

try:
    from morphcloud.api import MorphCloudClient
except ImportError:
    raise ImportError("Please install the morphcloud package: pip install morphcloud")

from swebench.harness.reporting import make_run_report
from swebench.harness.utils import (
    load_swebench_dataset,
    get_predictions_from_file,
    str2bool,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.constants import (
    RUN_EVALUATION_LOG_DIR,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    LOG_REPORT,
)
from swebench.harness.docker_build import setup_logger
from swebench.harness.utils import EvaluationError
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
)


@dataclass
class TestOutput:
    instance_id: str
    test_output: str
    report_json_str: str
    run_instance_log: str
    patch_diff: str
    log_dir: Path
    errored: bool


client = MorphCloudClient()


@asynccontextmanager
async def base_snapshot_context(test_spec: TestSpec):
    """
    Build and yield a base snapshot that contains all common installation steps.
    These steps run once and are cached.
    """
    snapshot = client.snapshots.create(
        vcpus=4,
        memory=16384,  # Increased from 8192
        disk_size=100000,  # Increased from 20000
        digest="swebench-base",
    )
    # Common steps executed once
    snapshot = await snapshot.asetup("apt-get update -q")
    snapshot = await snapshot.asetup(
        "export DEBIAN_FRONTEND=noninteractive && export TZ='Etc/UTC'"
    )
    snapshot = await snapshot.asetup(
        "apt install -y wget git build-essential libffi-dev libtiff-dev jq curl locales locales-all tzdata patch"
    )
    # Install Miniconda
    snapshot = await snapshot.asetup(
        "wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh' -O miniconda.sh"
    )
    snapshot = await snapshot.asetup("bash miniconda.sh -b -p /opt/miniconda3")
    snapshot = await snapshot.asetup(
        "echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc"
    )
    snapshot = await snapshot.asetup("/opt/miniconda3/bin/conda init --all")
    snapshot = await snapshot.asetup(
        "/opt/miniconda3/bin/conda config --append channels conda-forge"
    )
    snapshot = await snapshot.asetup(
        "adduser --disabled-password --gecos 'dog' nonroot"
    )
    snapshot = await snapshot.asetup("mkdir -p /testbed")

    env_script = test_spec.setup_env_script
    if env_script:
        snapshot = await snapshot.asetup(
            f"""
            cat > /root/setup_env.sh <<'EOF'
{env_script}
EOF
        """
        )
        snapshot = await snapshot.asetup("chmod +x /root/setup_env.sh")
        snapshot = await snapshot.asetup(
            "bash -c 'source ~/.bashrc && /root/setup_env.sh'"
        )
        snapshot = await snapshot.asetup(
            "echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc"
        )

    # Inline the repository installation script from TestSpec.
    repo_script = test_spec.install_repo_script
    if repo_script:
        snapshot = await snapshot.asetup(
            f"""
            cat > /root/setup_repo.sh <<'EOF' 
{repo_script}
EOF
        """
        )
        snapshot = await snapshot.asetup("chmod +x /root/setup_repo.sh")
        snapshot = await snapshot.asetup("bash /root/setup_repo.sh")

    with client.instances.start(snapshot.id, ttl_seconds=3600) as instance:
        try:
            yield instance
        finally:
            pass


async def get_log_dir(pred: dict, run_id: str, instance_id: str) -> Path:
    model_name_or_path = cast(
        str, pred.get("model_name_or_path", "None").replace("/", "__")
    )
    return RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id


async def process_instance_morph(test_spec, pred, run_id) -> TestOutput:
    """
    Do the remaining work (patch application, running eval, logging, reporting)
    on the Morph Cloud instance yielded by base_snapshot_context.
    """
    instance_id = test_spec.instance_id
    this_pred = pred[instance_id]
    # Setup logging directory:
    log_dir = await get_log_dir(this_pred, run_id, instance_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_instance.log"
    logger = setup_logger(instance_id, log_file, add_stdout=True)

    # Retrieve any patch diff from the prediction:
    patch_diff = this_pred.get("model_patch", "")

    try:
        async with base_snapshot_context(test_spec) as morphvm:
            if patch_diff:
                # Write the patch to /tmp/patch.diff
                res = await morphvm.aexec(
                    command=f"""
                    cat > /tmp/patch.diff <<'EOF'
{patch_diff}
EOF
                """
                )
                logger.info(
                    f"Wrote patch file: stdout: {res.stdout}; stderr: {res.stderr}"
                )

                # Attempt to apply the patch
                apply_patch_resp = await morphvm.aexec(
                    command="cd /testbed && git apply -v /tmp/patch.diff"
                )
                apply_patch_output = (
                    apply_patch_resp.stdout + "\n" + apply_patch_resp.stderr
                )
                returncode = apply_patch_resp.exit_code

                if returncode != 0:
                    logger.info("Failed to apply patch to container, trying again...")

                    apply_patch_resp = await morphvm.aexec(
                        command="cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff"
                    )
                    apply_patch_output = (
                        apply_patch_resp.stdout + "\n" + apply_patch_resp.stderr
                    )
                    returncode = apply_patch_resp.exit_code

                    if returncode != 0:
                        logger.info(f"{APPLY_PATCH_FAIL}:\n{apply_patch_output}")
                        raise EvaluationError(
                            instance_id,
                            f"{APPLY_PATCH_FAIL}:\n{apply_patch_output}",
                            logger,
                        )
                    else:
                        logger.info(f"{APPLY_PATCH_PASS}:\n{apply_patch_output}")
                else:
                    logger.info(f"{APPLY_PATCH_PASS}:\n{apply_patch_output}")

            # Run git diff before evaluation.
            git_diff_resp = await morphvm.aexec(command="cd /testbed && git diff")
            git_diff_output_before = git_diff_resp.stdout
            logger.info(f"Git diff before:\n{git_diff_output_before}")

            # Write and prepare evaluation script with the django hack.
            eval_script = test_spec.eval_script
            # Apply django hack
            eval_script = eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")

            res_eval_setup = await morphvm.aexec(
                command=f"""
                cat > /root/eval.sh <<'EOF'
{eval_script}
EOF
                chmod +x /root/eval.sh
            """
            )
            logger.info(
                f"Eval script setup: stdout: {res_eval_setup.stdout}; stderr: {res_eval_setup.stderr}"
            )

            start_time = time.time()

            # Run command with test markers
            run_command = "cd /testbed"
            # Add pylint hack
            if "pylint" in test_spec.instance_id:
                run_command += " && PYTHONPATH="
            # increase recursion limit for testing
            run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
            # Add start marker
            run_command += f" && echo '{START_TEST_OUTPUT}'"
            # run eval script
            run_command += " && /bin/bash /root/eval.sh"
            # Add end marker
            run_command += f" && echo '{END_TEST_OUTPUT}'"

            eval_resp = await morphvm.aexec(command=run_command)
            test_output = eval_resp.stdout
            total_runtime = time.time() - start_time
            logger.info(f"Test runtime: {total_runtime:.2f} seconds")

            # Get git diff after running eval script
            git_diff_resp = await morphvm.aexec(command="cd /testbed && git diff")
            git_diff_output_after = git_diff_resp.stdout

            # Check if git diff changed after running eval script
            logger.info(f"Git diff after:\n{git_diff_output_after}")
            if git_diff_output_after != git_diff_output_before:
                logger.info("Git diff changed after running eval script")

            # Write all log files immediately in this process

            # Write test output file
            test_output_path = log_dir / "test_output.txt"
            with open(test_output_path, "w", encoding="utf-8") as f:
                f.write(test_output)
                logger.info(
                    f"Test output for {instance_id} written to {test_output_path}"
                )
                print(f"Test output for {instance_id} written to {test_output_path}")

            # Get report from test output with logging
            logger.info(f"Grading answer for {instance_id}...")
            report = get_eval_report(
                test_spec=test_spec,
                prediction=this_pred,
                test_log_path=test_output_path,
                include_tests_status=True,
            )
            logger.info(
                f"report: {report}\n"
                f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
            )

            # Write report.json file
            report_path = log_dir / "report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
                logger.info(f"Report for {instance_id} written to {report_path}")

            # Write the patch file
            patch_path = log_dir / "patch.diff"
            with open(patch_path, "w", encoding="utf-8") as f:
                f.write(patch_diff)
                logger.info(f"Patch for {instance_id} written to {patch_path}")

            # Write run_instance.log
            # This will copy the current logger's content to make sure we have a complete log
            run_log_content = log_file.read_text() if log_file.exists() else ""

            return TestOutput(
                instance_id=test_spec.instance_id,
                test_output=test_output,
                report_json_str=json.dumps(report, indent=4),
                patch_diff=patch_diff,
                run_instance_log=run_log_content,
                log_dir=log_dir,
                errored=False,
            )
    except EvaluationError:
        error_msg = traceback.format_exc()
        logger.info(error_msg)

        # Write error log and an empty report for failed runs
        error_report = {
            instance_id: {
                "patch_is_None": False,
                "patch_exists": True,
                "patch_successfully_applied": False,
                "resolved": False,
                "error": True,
            }
        }

        report_path = log_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(error_report, f, indent=4)
            logger.info(f"Error report for {instance_id} written to {report_path}")

        patch_path = log_dir / "patch.diff"
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(patch_diff)

        run_log_content = log_file.read_text()

        return TestOutput(
            instance_id=instance_id,
            test_output="",
            report_json_str=json.dumps(error_report, indent=4),
            run_instance_log=run_log_content,
            patch_diff=patch_diff,
            log_dir=log_dir,
            errored=True,
        )
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({log_file}) for more information."
        )
        logger.error(error_msg)

        # Write error log and an empty report for failed runs
        error_report = {
            instance_id: {
                "patch_is_None": False,
                "patch_exists": True,
                "patch_successfully_applied": False,
                "resolved": False,
                "error": True,
            }
        }

        report_path = log_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(error_report, f, indent=4)
            logger.info(f"Error report for {instance_id} written to {report_path}")

        patch_path = log_dir / "patch.diff"
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(patch_diff)

        run_log_content = log_file.read_text()

        return TestOutput(
            instance_id=instance_id,
            test_output="",
            report_json_str=json.dumps(error_report, indent=4),
            run_instance_log=run_log_content,
            patch_diff=patch_diff,
            log_dir=log_dir,
            errored=True,
        )


async def process_instances_distributed(
    predictions, dataset, full_dataset, run_id, max_workers
):
    """
    Create a process pool to process the test specifications and run each instance on Morph Cloud.
    """
    run_test_specs = []
    test_specs = list(map(make_test_spec, dataset))

    # Check for instances that have already been run
    for test_spec in test_specs:
        log_dir = await get_log_dir(
            predictions[test_spec.instance_id], run_id, test_spec.instance_id
        )
        if log_dir.exists():
            continue
        run_test_specs.append(test_spec)

    results = []
    if run_test_specs:
        # Process instances concurrently but with limited concurrency
        tasks = []
        for test_spec in run_test_specs:
            task = asyncio.create_task(
                process_instance_morph(
                    test_spec,
                    predictions,
                    run_id,
                )
            )
            tasks.append(task)

            # Limit concurrency by processing in batches
            if len(tasks) >= max_workers:
                completed, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                results.extend([t.result() for t in completed])

        # Wait for any remaining tasks
        if tasks:
            completed, _ = await asyncio.wait(tasks)
            results.extend([t.result() for t in completed])

    # No need to save logs here as it's already done in process_instance_morph
    # Just print a summary
    for result in results:
        result = cast(TestOutput, result)
        print(f"Instance {result.instance_id} completed (errored: {result.errored})")

    make_run_report(predictions, full_dataset, run_id)


async def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,
    max_workers: int,
    run_id: str,
    namespace: str | None,
    rewrite_reports: bool,
    report_dir: str = ".",
):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    namespace = None if namespace == "" else namespace

    if dataset_name == "princeton-nlp/SWE-bench_Multimodal" and split == "test":
        print(
            "⚠️ Local evaluation for the test split of SWE-bench Multimodal is not supported. "
            "Please check out sb-cli (https://github.com/swe-bench/sb-cli/) for instructions on how to submit predictions."
        )
        return

    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    if report_dir is not None:
        report_dir_path = Path(report_dir)
        if not report_dir_path.exists():
            report_dir_path.mkdir(parents=True)

    # load predictions as map of instance_id to prediction
    predictions = get_predictions_from_file(predictions_path, dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    dataset = get_dataset_from_preds(
        dataset_name, split, instance_ids, predictions, run_id, rewrite_reports
    )
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)
    return await process_instances_distributed(
        predictions, dataset, full_dataset, run_id, max_workers
    )


if __name__ == "__main__":

    def get_dataset_from_preds(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions: dict,
        run_id: str,
        rewrite_reports: bool,
        exclude_completed: bool = True,
    ):
        """
        Return only instances that have predictions and are in the dataset.
        If instance_ids is provided, only return instances with those IDs.
        If exclude_completed is True, only return instances that have not been run yet.
        """
        # load dataset
        dataset = load_swebench_dataset(dataset_name, split)
        dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

        if instance_ids:
            # check that all instance IDs have predictions
            missing_preds = set(instance_ids) - set(predictions.keys())
            if missing_preds:
                print(
                    f"Warning: Missing predictions for {len(missing_preds)} instance IDs."
                )

        # check that all prediction IDs are in the dataset
        prediction_ids = set(predictions.keys())
        if prediction_ids - dataset_ids:
            raise ValueError(
                (
                    "Some prediction IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
                )
            )
        if instance_ids:
            dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

        if rewrite_reports:
            # we only return instances that have existing test outputs
            test_output_ids = set()
            for instance in dataset:
                if instance[KEY_INSTANCE_ID] not in predictions:
                    continue
                prediction = predictions[instance[KEY_INSTANCE_ID]]
                test_output_file = (
                    RUN_EVALUATION_LOG_DIR
                    / run_id
                    / prediction.get("model_name_or_path", "None").replace("/", "__")
                    / prediction[KEY_INSTANCE_ID]
                    / "test_output.txt"
                )
                if test_output_file.exists():
                    test_output_ids.add(instance[KEY_INSTANCE_ID])
            dataset = [
                i
                for i in dataset
                if i[KEY_INSTANCE_ID] in prediction_ids
                and i[KEY_INSTANCE_ID] in test_output_ids
            ]
            return dataset

        # check which instance IDs have already been run
        completed_ids = set()
        for instance in dataset:
            if instance[KEY_INSTANCE_ID] not in prediction_ids:
                # skip instances without predictions
                continue
            prediction = predictions[instance[KEY_INSTANCE_ID]]
            report_file = (
                RUN_EVALUATION_LOG_DIR
                / run_id
                / prediction.get("model_name_or_path", "None").replace("/", "__")
                / prediction[KEY_INSTANCE_ID]
                / LOG_REPORT
            )
            if report_file.exists():
                completed_ids.add(instance[KEY_INSTANCE_ID])

        if completed_ids and exclude_completed:
            # filter dataset to only instances that have not been run
            print(f"{len(completed_ids)} instances already run, skipping...")
            dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

        empty_patch_ids = {
            k
            for k, v in predictions.items()
            if v[KEY_PREDICTION] == "" or v[KEY_PREDICTION] is None
        }

        # filter dataset to only instances with predictions
        dataset = [
            i
            for i in dataset
            if i[KEY_INSTANCE_ID] in prediction_ids
            and i[KEY_INSTANCE_ID] not in empty_patch_ids
        ]
        return dataset

    parser = ArgumentParser(
        description="Run evaluation harness for the given dataset and predictions.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers",
    )
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Lite",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    parser.add_argument(
        "--rewrite_reports",
        type=str2bool,
        default=False,
        help="Doesn't run new instances, only writes reports for instances with existing test outputs",
    )
    parser.add_argument(
        "--report_dir", type=str, default="logs", help="Directory to write reports to"
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Path to predictions file - if 'gold', uses gold predictions",
        required=True,
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--namespace", type=str, default="swebench", help="Namespace for images"
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
