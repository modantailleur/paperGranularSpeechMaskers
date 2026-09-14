import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so granspeechmask_paper is importable without installing it

from granspeechmask_paper.experiments.common import multispeakers_list
from granspeechmask_paper.utils.dispatch import dispatch_speaker_run
import os
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import time

log_dir = "../speechConcealerExp/logs-multispeaker/"
speakers_list_pruned = multispeakers_list  # all couples except the first 10

driver_script = "granspeechmask_paper/experiments/multispeaker.py"

start_time = time.time()

base_params_list = []

params = {
    "plan": ["baseline"],
    "model": ["oracle", "whitenoise"],
    "dataset": ["ebr-low-asr-low", "ebr-mid-asr-mid", "ebr-high-asr-high"],
}

base_params_list.append(params)

params = {
    "plan": ["simplelist"],
    "model": ["simplelist"],
    "dataset": ["ebr-low-asr-low", "ebr-mid-asr-mid", "ebr-high-asr-high"],
    "svad": ["ten"],
    "cvad": ["ten"],
    "vad_thresh": 0.5,
    "exptype": ["normal", "oracle"],
    "nr": ["metricgan"],
    "qc": ["none"],
    "rr": ["none"],
    "max_len_strat": ["queue"],
    "fe": ["logmel"],
    "dw": [300],
}

base_params_list.append(params)

# ======================================================================
# PHASE 1: generate — CPU only, up to 20 jobs in parallel
# ======================================================================
n_jobs = min(20, max(1, int(0.8 * os.cpu_count())))
generate_params_list = [dict(p, step=["generate"]) for p in base_params_list]

Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(dispatch_speaker_run)(speaker, log_dir, params, driver_script)
    for speaker in speakers_list_pruned for params in generate_params_list
)

print("All 'generate' jobs finished.")

# ======================================================================
# PHASE 2: evaluate — 2 jobs on GPU 0 + 2 jobs on GPU 1, running
# together (4 total), each GPU never running more than 2 at once.
# ======================================================================
evaluate_params_list = [dict(p, step=["evaluate"]) for p in base_params_list]
evaluate_tasks = [
    (speaker, params)
    for speaker in speakers_list_pruned for params in evaluate_params_list
]

def run_evaluate_on_gpu(tasks, gpu_id):
    Parallel(n_jobs=2, backend="loky")(
        delayed(dispatch_speaker_run)(speaker, log_dir, params, driver_script, gpu_id=gpu_id)
        for speaker, params in tasks
    )

tasks_gpu0 = evaluate_tasks[0::2]
tasks_gpu1 = evaluate_tasks[1::2]

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(run_evaluate_on_gpu, tasks_gpu0, 0),
        executor.submit(run_evaluate_on_gpu, tasks_gpu1, 1),
    ]
    for future in futures:
        future.result()

print("All 'evaluate' jobs finished.")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
