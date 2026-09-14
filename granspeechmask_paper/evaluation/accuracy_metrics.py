
import os
import numpy as np
import pandas as pd
from granspeechmask_paper.stream import extract_turns_id, compute_speaker_timeline, dominant_speaker_in_range
from amods.config import load_config

def base_without_ext(name, audio_exts=('.wav', '.flac', '.mp3')):
    bn = os.path.basename(name)
    for ext in audio_exts:
        if bn.lower().endswith(ext):
            return bn[:-len(ext)]
    # if no known audio extension, keep full basename
    return bn

def _extract_speaker_id(voice_name):
    # 'spk_p226__p225-p226_conv0021_mic1__...' -> 'p226'
    return voice_name.split('__')[0].split('_')[1]

def run_accuracy_metrics(args):

    # If a "mix" folder exists inside output_audio_dir, use it instead
    concealer_dir = os.path.join(args.output_audio_dir, "concealer")
    if os.path.isdir(concealer_dir):

        stream_config = load_config("stream", args.streamconfig)
        sr = stream_config["sr"]
        buffer_size = int(sr * stream_config["buffer_duration"])

        scores = []
        for eval_audio_file in args.eval_audio_files:
            prefix = base_without_ext(eval_audio_file)
            metadata_path = os.path.join(concealer_dir, f"{prefix}_concealer_metadata.npy")
            if not os.path.isfile(metadata_path):
                continue
            meta = np.load(metadata_path, allow_pickle=True)

            # Re-open the ground-truth turns file for this conversation (same one
            # used at generation time) instead of persisting the ground truth again.
            turns_id = extract_turns_id(eval_audio_file)
            turns_file_path = os.path.join(args.turns_path, f"{turns_id}.csv")
            turns_df = pd.read_csv(turns_file_path)

            n_blocks = len(meta)
            speaker_timeline = compute_speaker_timeline(turns_df, n_blocks * buffer_size, sr)

            file_scores = []
            for b, voice_name in enumerate(meta):
                if voice_name == '':
                    continue
                gt_speaker = dominant_speaker_in_range(speaker_timeline, b * buffer_size, (b + 1) * buffer_size)
                if gt_speaker is None:
                    continue
                file_scores.append(1 if _extract_speaker_id(voice_name) == gt_speaker else 0)

            print(f"Scored {len(file_scores)} concealer block(s) for {prefix}")
            scores += file_scores

        scores = np.array(scores)
        if len(scores) > 0:
            print(f"Accuracy: {np.mean(scores):.4f}")
        else:
            print("Accuracy: N/A (no scoreable concealer blocks found)")

        np.save(os.path.join(args.accuracy_dir, f"{args.setting_identifier}_accuracy.npy"), scores)
    else:
        scores = np.array([])
        np.save(os.path.join(args.accuracy_dir, f"{args.setting_identifier}_accuracy.npy"), scores)
