import os
import numpy as np
import librosa
import h5py
from joblib import Parallel, delayed

from amods.models.vad import WebrtcVAD, SileroVAD, TenVAD

WINDOW_MS = 50
SAMPLE_RATE = 16000

DATASET_DIR = '../SOS-1SP/'
OUT_DIR = "../vadConcealerExp/"
os.makedirs(OUT_DIR, exist_ok=True)

sub_datasets = ['ebr-high-asr-high', 'ebr-mid-asr-mid', 'ebr-low-asr-low']

speakers_list = [
    'p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266',
    'p267','p268','p269','p270','p271','p272','p273','p274','p275','p276',
    'p277','p278','p279','p280','p281','p282','p283','p284','p285','p286',
    'p287','p288','p292','p293','p294','p295','p297','p298','p299','p300',
    'p301','p302','p303','p304','p305','p306','p307','p308','p310','p311',
    'p312','p313','p314','p315','p316','p317','p318','p323','p326','p329',
    'p330','p333','p334','p335','p336','p339','p340','p341','p343','p345',
    'p347','p351','p360','p361','p362','p363','p364','p374','p376','s5'
]

vad_models_names = ["webrtc1", "webrtc2", "webrtc3", "silero", "ten"]

def build_vad(vad_name):
    if "webrtc" in vad_name:
        return WebrtcVAD(sr=SAMPLE_RATE, logit_threshold=None, aggressiveness=int(vad_name[-1]))
    elif vad_name == "silero":
        return SileroVAD(sr=SAMPLE_RATE, logit_threshold=None)
    elif vad_name == "ten":
        return TenVAD(sr=SAMPLE_RATE, logit_threshold=None)
    else:
        raise ValueError(vad_name)

def build_file_list():
    all_files = []
    splits = ["evaluation", "calibration"]

    for sub_dataset in sub_datasets:
        for split in splits:
            for speaker in speakers_list:

                full_path = f"{DATASET_DIR}/{sub_dataset}/{split}/pan_0/{speaker}"

                if not os.path.exists(full_path):
                    continue

                for file in os.listdir(full_path):
                    if file.endswith(".flac") and "mic1" in file:
                        file_path = os.path.join(full_path, file)
                        all_files.append((file_path, file, sub_dataset, split))

    return all_files

def reset_output():
    print("Cleaning old H5 files...")
    for f in os.listdir(OUT_DIR):
        if f.endswith(".h5"):
            os.remove(os.path.join(OUT_DIR, f))


def process_model_dataset(vad_name, sub_dataset, all_files):

    out_path = os.path.join(OUT_DIR, f"{vad_name}_{sub_dataset}.h5")

    if os.path.exists(out_path):
        os.remove(out_path)

    vad = build_vad(vad_name)
    print(f"VAD built: {vad_name}")

    with h5py.File(out_path, "w") as f:

        ds_logits = None
        ds_frame_idx = None
        ds_file_name = None
        ds_file_idx = None

        for file_idx, (file_path, file_name, ds_name, split) in enumerate(all_files):

            if ds_name != sub_dataset:
                continue

            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            audio = audio.astype(np.float32)

            win_len = int(round(SAMPLE_RATE * WINDOW_MS / 1000.0))
            n_frames = int(np.ceil(len(audio) / win_len))

            pad_len = n_frames * win_len - len(audio)
            if pad_len > 0:
                audio = np.pad(audio, (0, pad_len), mode="constant")

            frames = audio.reshape(n_frames, win_len)

            logits = np.array(
                [vad.predict(frame) for frame in frames],
                dtype=np.float32
            )

            frame_idx = np.arange(n_frames, dtype=np.int32)

            # init datasets
            if ds_logits is None:

                ds_logits = f.create_dataset(
                    "logits",
                    shape=(0, n_frames),
                    maxshape=(None, n_frames),
                    dtype=np.float32,
                    chunks=(1, n_frames),
                    compression="gzip"
                )

                ds_frame_idx = f.create_dataset(
                    "frame_idx",
                    shape=(0, n_frames),
                    maxshape=(None, n_frames),
                    dtype=np.int32,
                    chunks=(1, n_frames),
                    compression="gzip"
                )

                ds_file_name = f.create_dataset(
                    "file_name",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype("utf-8")
                )

                ds_file_idx = f.create_dataset(
                    "file_idx",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int32
                )

            # resize
            new_i = ds_logits.shape[0]

            ds_logits.resize((new_i + 1, n_frames))
            ds_frame_idx.resize((new_i + 1, n_frames))
            ds_file_name.resize((new_i + 1,))
            ds_file_idx.resize((new_i + 1,))

            # write
            ds_logits[new_i] = logits
            ds_frame_idx[new_i] = frame_idx
            ds_file_name[new_i] = file_name
            ds_file_idx[new_i] = file_idx

            print(f"[{vad_name} | {sub_dataset}] {file_idx+1}/{len(all_files)}")


if __name__ == "__main__":

    reset_output()

    all_files = build_file_list()
    print("Total files:", len(all_files))

    jobs = [
        (vad, ds, all_files)
        for vad in vad_models_names
        for ds in sub_datasets
    ]

    Parallel(n_jobs=len(vad_models_names) * len(sub_datasets), backend="loky")(
        delayed(process_model_dataset)(vad, ds, all_files)
        for vad, ds, all_files in jobs
    )

    print("Done.")