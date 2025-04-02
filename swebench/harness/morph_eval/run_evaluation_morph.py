"""
This script implements run_evaluation using Morph Cloud.
It builds a snapshot via a chain of .asetup() commands
(including repository cloning and checkout using the
environment_setup_commit), starting an instance, applying 
a patch, running tests and generating a report.
"""

import os
import time
import json
import base64
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager
from typing import cast
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

from swebench.harness.docker_build import setup_logger

# Configure logging (adjust level and format as needed)
logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from morphcloud.api import MorphCloudClient
except ImportError:
    raise ImportError(
        "Please install the morphcloud package: pip install morphcloud"
    )

from swebench.harness.reporting import make_run_report
from swebench.harness.utils import (
    load_swebench_dataset,
    get_predictions_from_file,
    str2bool
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.constants import RUN_EVALUATION_LOG_DIR, KEY_INSTANCE_ID, KEY_PREDICTION, LOG_REPORT, KEY_MODEL

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
        memory=8192,
        disk_size=20000,
        digest="swebench-base"
    )
    # Common steps executed once
    snapshot = await snapshot.asetup("apt-get update -q")
    snapshot = await snapshot.asetup("export DEBIAN_FRONTEND=noninteractive && export TZ='Etc/UTC'")
    snapshot = await snapshot.asetup("apt install -y wget git build-essential libffi-dev libtiff-dev jq curl locales locales-all tzdata patch")
    # Install Miniconda
    snapshot = await snapshot.asetup("wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh' -O miniconda.sh")
    snapshot = await snapshot.asetup("bash miniconda.sh -b -p /opt/miniconda3")
    snapshot = await snapshot.asetup("echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc")
    snapshot = await snapshot.asetup("/opt/miniconda3/bin/conda init --all")
    snapshot = await snapshot.asetup("/opt/miniconda3/bin/conda config --append channels conda-forge")
    snapshot = await snapshot.asetup("adduser --disabled-password --gecos 'dog' nonroot")
    snapshot = await snapshot.asetup("mkdir -p /testbed")
    env_script = test_spec.setup_env_script
    if env_script:
        snapshot = await snapshot.asetup(f"cat <<'EOF' > /root/setup_env.sh\n{env_script}\nEOF")
        snapshot = await snapshot.asetup("chmod +x /root/setup_env.sh")
        snapshot = await snapshot.asetup("bash -c 'source ~/.bashrc && /root/setup_env.sh'")
        snapshot = await snapshot.asetup("echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc")
        # Inline the repository installation script from TestSpec.
    repo_script = test_spec.install_repo_script
    if repo_script:
        snapshot = await snapshot.asetup(f"cat <<'EOF' > /root/setup_repo.sh\n{repo_script}\nEOF")
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


async def process_instances_distributed(predictions, dataset, full_dataset, run_id, max_workers):
    """
    Create an async queue over the test specifications and run each instance on Morph Cloud.
    """
    test_queue = asyncio.Queue()
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

    if run_test_specs:
        # Run instances that haven't been run yet
        for test_spec in run_test_specs:
            test_queue.put_nowait(test_spec)
    results = []

    async def process_instance(pred: dict, run_id: str) -> None:
        """
        Do the remaining work (patch application, running eval, logging, reporting)
        on the Morph Cloud instance yielded by base_snapshot_context.
        Appends TestOutput to the global results list.
        """
        while not test_queue.empty():
            try:
                test_spec = await test_queue.get()
                instance_id = test_spec.instance_id
                this_pred = pred[instance_id]
                # Setup logging directory:
                log_dir = RUN_EVALUATION_LOG_DIR / run_id / test_spec.repo.replace("/", "__") / instance_id
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "run_instance.log"

                logger = setup_logger(instance_id, log_file, add_stdout=True)

                # Retrieve any patch diff from the prediction:
                patch_diff = this_pred.get("model_patch", "")
                try:
                    async with base_snapshot_context(test_spec) as morphvm:
                        if patch_diff:
                            patch_bytes = patch_diff.encode('utf-8')
                            patch_b64 = base64.b64encode(patch_bytes).decode('utf-8')

                            result = await morphvm.aexec(command=f"echo '{patch_b64}' | base64 -d > /tmp/patch.diff")
                            if result.exit_code != 0:
                                raise Exception(f"Error writing patch file:\n{result.stdout}\n{result.stderr}")

                            apply_patch_resp = await morphvm.aexec(command="cd /testbed && git apply -v /tmp/patch.diff")
                            if apply_patch_resp.exit_code != 0:
                                apply_patch_resp = await morphvm.aexec(command="cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff")
                                if apply_patch_resp.exit_code != 0:
                                    raise Exception(f"Patch failed:\n{apply_patch_resp.stdout}\n{apply_patch_resp.stderr}")
                        result = await morphvm.aexec(command="cd /testbed && git diff")
                        if result.exit_code != 0:
                            raise Exception(f"Error getting git diff:\n{result.stdout}\n{result.stderr}")

                        eval_script_bytes = test_spec.eval_script.encode('utf-8')
                        eval_script_b64 = base64.b64encode(eval_script_bytes).decode('utf-8')
                        result = await morphvm.aexec(command=f"bash -c 'echo \"{eval_script_b64}\" | base64 -d > /root/eval.sh && chmod +x /root/eval.sh'")
                        if result.exit_code != 0:
                            raise Exception(f"Error getting git diff:\n{result.stdout}\n{result.stderr}")
                        start_time = time.time()
                        run_command = (
                            "cd /testbed && python3 -c 'import sys; sys.setrecursionlimit(10000)' "
                            "&& /bin/bash /root/eval.sh"
                        )
                        eval_resp = await morphvm.aexec(command=run_command)
                        if eval_resp.exit_code != 0:
                            raise Exception(f"Error running eval script:\n{eval_resp.stdout}\n{eval_resp.stderr}")
                        test_output = eval_resp.stdout
                        total_runtime = time.time() - start_time
                        result = await morphvm.aexec(command="cd /testbed && git diff")
                        if result.exit_code != 0:
                            raise Exception(f"Error getting git diff:\n{result.stdout}\n{result.stderr}")
                        test_output_path = log_dir / "test_output.txt"
                        with open(test_output_path, "w", encoding="utf-8") as f:
                            f.write(test_output)
                        report = get_eval_report(
                            test_spec=test_spec,
                            prediction=this_pred,
                            test_log_path=test_output_path,
                            include_tests_status=True,
                        )
                        results.append(TestOutput(
                            instance_id=test_spec.instance_id,
                            test_output=test_output,
                            report_json_str=json.dumps(report, indent=4),
                            run_instance_log=log_file.read_text(),
                            patch_diff=patch_diff,
                            log_dir=log_dir,
                            errored=False,
                        ))
                except Exception:
                    error_msg = traceback.format_exc()
                    with open(log_file, "w", encoding="utf-8") as lf:
                        lf.write(error_msg)
                    logger.error(f"Error processing test_spec {test_spec.instance_id}", exc_info=True)
                    results.append(TestOutput(
                        instance_id=test_spec.instance_id,
                        test_output="",
                        report_json_str="",
                        run_instance_log=log_file.read_text(),
                        patch_diff=patch_diff,
                        log_dir=log_dir,
                        errored=True,
                    ))
            except Exception as e:
                logger.error("Error in process_instance loop", exc_info=True)
                continue

    # Run workers concurrently
    await asyncio.gather(*[process_instance(predictions, run_id) for _ in range(max_workers)])
    for result in results:
        result = cast(TestOutput, result)
        # Save logs locally
        log_dir = result.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "run_instance.log", "w") as f:
            f.write(result.run_instance_log)
        with open(log_dir / "test_output.txt", "w") as f:
            f.write(result.test_output)
        with open(log_dir / "patch.diff", "w") as f:
            f.write(result.patch_diff)
        with open(log_dir / "report.json", "w") as f:
            try:
                # Only load JSON if there is content; otherwise create an empty dict.
                if result.report_json_str and result.report_json_str.strip():
                    report_json = json.loads(result.report_json_str)
                else:
                    report_json = {}
                json.dump(report_json, f, indent=4)
            except Exception:
                logger.error(f"{result.instance_id}: Error writing report.json", exc_info=True)
                print(f"{result.instance_id}: no report.json")

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
        report_dir = Path(report_dir)
        if not report_dir.exists():
            report_dir.mkdir(parents=True)

    # load predictions as map of instance_id to prediction
    predictions = get_predictions_from_file(predictions_path, dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    dataset = get_dataset_from_preds(
        dataset_name, split, instance_ids, predictions, run_id, rewrite_reports
    )
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)
    return await process_instances_distributed(predictions, dataset, full_dataset, run_id, max_workers)


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
                    / prediction["model_name_or_path"].replace("/", "__")
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
                / prediction[KEY_MODEL].replace("/", "__")
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
