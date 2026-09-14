"""
joblib/subprocess dispatch glue used by the scripts/run_*.py entry points to
fan a doce experiment out into one subprocess per parameter combination.
"""
import datetime
import itertools
import json
import os
import subprocess


def dispatch_speaker_run(speaker, log_dir, params, driver_script, gpu_id=None):
    """Like run_experiment_combinations, but pins ``params["speaker"]`` to a single speaker first."""
    for key, value in params.items():
        if not isinstance(value, list):
            params[key] = [value]

    params["speaker"] = [speaker]

    return run_experiment_combinations(params, log_dir=log_dir, driver_script=driver_script, gpu_id=gpu_id)


def run_experiment_combinations(params, log_dir="logs", driver_script=None, gpu_id=None):
    """
    Run one subprocess per combination of ``params`` (the cartesian product of
    every list-valued entry), invoking ``python3 -u <driver_script> -s
    <plan>/<json-encoded-params> -c`` for each.

    Args:
        params (dict): dictionary of parameter lists. Must include a "plan" key.
        log_dir (str or None): directory to store logs (None = no logging)
        driver_script (str): path to the doce driver script to run (e.g.
            "granspeechmask_paper/experiments/monospeaker.py"), relative to
            wherever this process itself is run from.
        gpu_id (int or None): if set, the subprocess only sees this GPU
            (via CUDA_VISIBLE_DEVICES); if None, the subprocess inherits
            the current environment as-is
    """
    if driver_script is None:
        raise ValueError("driver_script is required")

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Create log directory only if needed
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    keys = list(params.keys())
    value_lists = [params[k] for k in keys]

    for combo in itertools.product(*value_lists):
        combo_dict = dict(zip(keys, combo))

        # Extract plan separately
        if "plan" not in combo_dict:
            raise ValueError("Parameter 'plan' is required")

        plan = combo_dict.pop("plan")
        json_part = json.dumps(combo_dict)

        cmd = [
            "python3",
            "-u",
            driver_script,
            "-s",
            f"{plan}/{json_part}",
            "-c",
        ]

        print(cmd)
        print("=" * 80)
        print(f"Running: {' '.join(cmd)}")

        if log_dir is not None:
            log_name = "_".join(f"{k[:2]}-{v}" for k, v in combo_dict.items())
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{plan}_{log_name}_{timestamp}.log")

            print(f"Logging to: {log_file}")
            print("=" * 80)

            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                for line in process.stdout:
                    print(line, end="")  # live
                    f.write(line)        # save
                process.wait()
        else:
            print("(no log file)")
            print("=" * 80)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            for line in process.stdout:
                print(line, end="")
            process.wait()

        if process.returncode != 0:
            print(f"FAILED (exit code {process.returncode}): {' '.join(cmd)}")
        else:
            print(f"OK: {' '.join(cmd)}")
