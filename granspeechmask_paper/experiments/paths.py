"""
Per-machine experiment/dataset path resolution, shared by every experiment
driver in this package. Each driver (monospeaker.py, delay.py, etc.) defines
its own HOST_PATHS/DEFAULT_PATHS - the exact directory values legitimately
differ per experiment (different dataset variants, different scratch
locations) - and calls resolve_experiment_paths() to turn that into
(exp_dir, dataset_dir) for whichever machine it's currently running on.
Transcripts are expected under a "transc" subdirectory of dataset_dir.

The EXP_DIR/DATASET_DIR environment variables always take priority over
HOST_PATHS/DEFAULT_PATHS, so a new machine never requires editing these
files: export the ones you need before running a script, e.g.

    EXP_DIR=/data/speechConcealerExp \\
    DATASET_DIR=/data/SOS-1SP \\
    python3 scripts/run_ablation.py

HOST_PATHS/DEFAULT_PATHS remain as convenience defaults for machines the
paper's authors already use.
"""
import os
import socket


def resolve_experiment_paths(host_paths, default_paths):
    """
    Resolve (exp_dir, dataset_dir) for the current machine, creating exp_dir
    if it doesn't exist yet.

    Args:
        host_paths: dict mapping hostname -> (exp_dir, dataset_dir).
        default_paths: (exp_dir, dataset_dir) fallback used when the current
            hostname isn't a key in host_paths.

    The EXP_DIR / DATASET_DIR environment variables, when set, override the
    corresponding value regardless of hostname.
    """
    hostname = socket.gethostname()
    print("HOSTNAME:", hostname)

    exp_dir, dataset_dir = host_paths.get(hostname, default_paths)

    exp_dir = os.environ.get("EXP_DIR", exp_dir)
    dataset_dir = os.environ.get("DATASET_DIR", dataset_dir)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    return exp_dir, dataset_dir
