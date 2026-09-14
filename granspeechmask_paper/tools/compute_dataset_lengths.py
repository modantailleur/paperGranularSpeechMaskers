"""
Computes total audio duration of the SOS-1SP and
SOS-2SP datasets (evaluation split), broken down
per ASR-EBR condition. Run on the server where the datasets are mounted.

Layout expected: DATASET_DIR/<condition>/evaluation/pan_0/<speaker>/*.{wav,mp3,flac}
"""
import os
import socket
import soundfile as sf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so granspeechmask_paper is importable without installing it

from granspeechmask_paper.experiments.common import speakers_list, multispeakers_list

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac')
CONDITIONS = ["ebr-low-asr-low", "ebr-mid-asr-mid", "ebr-high-asr-high"]

hostname = socket.gethostname()

if hostname in ('pc-ls2n-lagrange', 'serveur-ls2n-cpu1', 'serveur-ls2n-gpu3'):
    MONO_DATASET_DIR = '../SOS-1SP/'
    MULTI_DATASET_DIR = '../SOS-2SP/'
elif hostname == 'po-tailleur-840g8':
    MONO_DATASET_DIR = '/media/user/EXTbackup/speech_dataset/SOS-1SP/'
    MULTI_DATASET_DIR = '/media/user/EXTbackup/speech_dataset/SOS-2SP/'
else:
    MONO_DATASET_DIR = '/media/user/MT-SSD-3/0-PROJETS_INFO/Post-doc/datasets/SOS-1SP/'
    MULTI_DATASET_DIR = '/media/user/MT-SSD-3/0-PROJETS_INFO/Post-doc/datasets/SOS-2SP/'

# Override with env vars if the hostname above doesn't match this machine.
MONO_DATASET_DIR = os.environ.get('MONO_DATASET_DIR', MONO_DATASET_DIR)
MULTI_DATASET_DIR = os.environ.get('MULTI_DATASET_DIR', MULTI_DATASET_DIR)


def format_duration(seconds):
    hours = seconds / 3600
    return f"{seconds:.2f} s ({hours:.2f} h)"


def compute_condition_stats(dataset_dir, condition, speakers):
    n_files = 0
    total_duration = 0.0
    for speaker in speakers:
        speaker_dir = os.path.join(dataset_dir, condition, "evaluation", "pan_0", speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for root, _, files in os.walk(speaker_dir):
            for fname in files:
                if fname.lower().endswith(AUDIO_EXTENSIONS):
                    fpath = os.path.join(root, fname)
                    info = sf.info(fpath)
                    total_duration += info.frames / info.samplerate
                    n_files += 1
    return n_files, total_duration


def compute_dataset_stats(name, dataset_dir, speakers):
    print(f"\n=== {name} ===")
    print(f"Dataset dir: {dataset_dir}")

    dataset_n_files = 0
    dataset_duration = 0.0

    for condition in CONDITIONS:
        n_files, duration = compute_condition_stats(dataset_dir, condition, speakers)
        dataset_n_files += n_files
        dataset_duration += duration
        print(f"  Condition '{condition}': {n_files} files, {format_duration(duration)}")

    print(f"  TOTAL: {dataset_n_files} files, {format_duration(dataset_duration)}")
    return dataset_n_files, dataset_duration


if __name__ == "__main__":
    mono_n_files, mono_duration = compute_dataset_stats(
        "SOS-1SP", MONO_DATASET_DIR, speakers_list
    )
    multi_n_files, multi_duration = compute_dataset_stats(
        "SOS-2SP", MULTI_DATASET_DIR, multispeakers_list
    )

    print("\n=== SUMMARY ===")
    print(f"MONOSPEAKER:  {mono_n_files} files, {format_duration(mono_duration)}")
    print(f"MULTISPEAKER: {multi_n_files} files, {format_duration(multi_duration)}")
    print(f"COMBINED:     {mono_n_files + multi_n_files} files, "
          f"{format_duration(mono_duration + multi_duration)}")
