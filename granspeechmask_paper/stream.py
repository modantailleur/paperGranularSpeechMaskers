"""
Paper-specific real-time simulation harness, built on top of amods.stream.Stream.

`Stream` here adds two things amods's own Stream doesn't have: a simulated
concealer computational delay (`delay_samples`, exercised by the delay
experiment) and a ground-truth speaker passthrough (`gt_speaker`, exercised
by the multispeaker oracle experiment). Everything else in this file -
folder-batch orchestration with calibration, the whitenoise baseline, and
the turn-taking bookkeeping - is genuinely paper-specific evaluation harness
with no amods equivalent.
"""
import argparse
import numpy as np
import soundfile as sf
import librosa
from collections import Counter
import os
import json
import re
import pandas as pd

# Importing amods's Stream directly (not the whole amods.stream module) keeps
# what's paper-only in this file explicit; granspeechmask_paper.concealer is
# imported for its @register_concealer("granspeechmask_paper") side effect,
# so amods.models.concealer.select_concealer_model (used internally by
# amods.stream.Stream) can find it.
from amods.audio import soft_clip
from amods.stream import Stream as AmodsStream
from granspeechmask_paper import concealer as _concealer  # noqa: F401
from granspeechmask_paper.utils.acoustics import compute_LAeq

np.random.seed(0)


class Stream(AmodsStream):
    """
    amods.stream.Stream plus a simulated concealer computational delay and a
    ground-truth speaker passthrough to the concealer - both exercised only
    by this paper's evaluation harness (see module docstring).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simulated computational delay (in samples), measured from the end of
        # the buffer that produced it (the earliest instant the concealer
        # could causally react, since that's when the buffer finishes filling).
        # delay_samples == 0 means the concealer plays right as its causing
        # buffer ends. Negative values push it earlier, down to the start of
        # that same buffer (delay_samples == -frames); nothing earlier than
        # that is reachable since prior buffers are already finalized.
        self.delay_samples = 0

    def _process_chunk(self, x, n_out_channels, gt_speaker=None):
        forecast = self.forecaster.predict(x)
        frames = len(x)

        voice_activity = self.source_vad.predict(x)
        if voice_activity:
            concealer_audio, voice_name = self.concealer.get_concealer(forecast, gt_speaker=gt_speaker)
            concealer_audio = (concealer_audio * self.stream_config["conc_multiplier"]).astype(np.float32, copy=False)
            raw_start = self.delay_samples + frames
            if raw_start < 0:
                raise ValueError(
                    f"delay_samples={self.delay_samples} is more negative than -frames "
                    f"(-{frames}): the concealer would need to reach into an already "
                    f"finalized past buffer, which isn't supported. The earliest reachable "
                    f"delay is delay_samples == -frames."
                )
            start = min(raw_start, len(self.pending_conc))
            end = min(start + len(concealer_audio), len(self.pending_conc))
            n = end - start
            if n > 0:
                self.pending_conc[start:end] += concealer_audio[:n]
        else:
            self.concealer.refresh(x)
            voice_name = ""

        conc_block = self.pending_conc[:frames].copy()
        self.pending_conc[:-frames] = self.pending_conc[frames:]
        self.pending_conc[-frames:] = 0.0

        play_mic = self.stream_config["monitor_gain"] * x
        play_mix = soft_clip(play_mic + conc_block, limit=0.95)

        rec_mic = self.stream_config["record_mic_gain"] * x
        rec_mix_mono = soft_clip(rec_mic + conc_block, limit=0.95)

        self.rec_original.append(x.copy())
        self.rec_concealer.append(conc_block.copy())
        self.rec_concealer_metadata.append(voice_name)
        self.rec_mix.append(np.tile(rec_mix_mono[:, np.newaxis], (1, n_out_channels)))

        return play_mix

    def callback(self, indata, outdata, frames, time, status, gt_speaker=None):
        x = indata[:, 0].astype(np.float32, copy=False)
        play_mix = self._process_chunk(x, outdata.shape[1], gt_speaker=gt_speaker)
        outdata[:] = play_mix[:, np.newaxis]

    def copy(self, freeze_learning=False):
        new_stream = type(self)(
            self.source_vad_config_name,
            self.concealer_config_name,
            self.stream_config_name,
            self.forecaster_config_name,
            self.is_stream,
            freeze_learning=freeze_learning
        )

        # Copy runtime state explicitly
        new_stream.concealer = self.concealer.copy()
        # concealer.copy() rebuilds the model from its own (stale) self.config,
        # which silently discards the freeze_learning value requested here - force it.
        new_stream.concealer.freeze_learning = freeze_learning
        new_stream.delay_samples = self.delay_samples

        return new_stream


def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# -----------------------
# File mode: simulate the callback over a folder of eval files, with
# calibration, ground-truth-speaker-aware oracle selection, and shortlist
# dumping. No amods equivalent (amods.stream.run_file_mode handles one file,
# with no calibration/oracle/shortlist-dump support).
# -----------------------
def run_file_mode_via_callback_on_folder(args, stream=None):

    # Simulated computational delay (seconds), e.g. to reproduce real-world
    # concealer compute latency measured on a given machine.
    delay = getattr(args, "delay", None)
    delay = delay if delay is not None else 0

    if stream is None:

        if not args.replace_existing:
            count = 0
            for infile_path in args.eval_audio_files:
                infile = os.path.basename(infile_path)

                out_prefix = infile.split(".")[0]
                out_dir = f"{args.output_audio_dir}"
                mix_path = f"{out_dir}/mix/{out_prefix}_mix.flac"
                if os.path.exists(mix_path):
                    count += 1
            if len(args.eval_audio_files) == count:
                print(f"All mix files already exist in {args.output_audio_dir}/mix/. Skipping processing due to replace_existing=False.")
                return

        stream = Stream(args.sourcevadconfig, args.concealerconfig, args.streamconfig, args.forecasterconfig, is_stream=False)
        stream.delay_samples = int(round(delay * stream.stream_config["sr"]))
        for calib_file in args.calib_audio_files:
            print(f'CALIBRATING: {calib_file}')
            # Load as mono, resample to sr
            y, _ = librosa.load(calib_file, sr=stream.stream_config["sr"], mono=True)
            fname_calib_file = os.path.splitext(os.path.basename(calib_file))[0]
            y = y.astype(np.float32, copy=False)

            # The calib file mixes both speakers, so tag each learned concealer with
            # the actual speaker talking at that time (see learn()/gt_speaker), for
            # bookkeeping purposes. This is done regardless of exptype: it only
            # labels memory entries, it never influences which concealer gets
            # picked at inference time (that's gated on exptype=="oracle" below).
            # Only multispeaker settings define turns_path (calib files there mix
            # several speakers); monospeaker calib files have a single speaker, so
            # there's nothing to tag.
            turns_path = getattr(args, "turns_path", None)
            if turns_path is not None:
                turns_id = extract_turns_id(calib_file)
                turns_file_path = os.path.join(turns_path, f"{turns_id}.csv")
                turns_df = pd.read_csv(turns_file_path)
                gt_speaker_timeline = compute_speaker_timeline(turns_df, len(y), stream.stream_config["sr"])
            else:
                gt_speaker_timeline = None

            stream.concealer.learn(y, fname_calib_file, gt_speaker=gt_speaker_timeline)

    # For each concealer stored in memory after calibration: which speaker it
    # was tagged with, and its feature vector.
    shortlist_entries = [
        {
            "speaker": extract_speaker_id(voice_name),
            "features": concealer_feat.tolist(),
        }
        for concealer, concealer_feat, _, voice_name in stream.concealer.memory
    ]

    shortlist_path = os.path.join(
        args.shortlist_dir,
        f"{args.setting_identifier}_shortlist.json"
    )

    with open(shortlist_path, "w") as f:
        json.dump(shortlist_entries, f, indent=2)

    for infile_path in args.eval_audio_files:
        print(f"\nProcessing: {infile_path}")
        infile = os.path.basename(infile_path)

        out_prefix = infile.split(".")[0]
        out_dir = f"{args.output_audio_dir}"
        os.makedirs(out_dir, exist_ok=True)
        mix_path = f"{out_dir}/mix/{out_prefix}_mix.flac"
        if os.path.exists(mix_path) and not args.replace_existing:
            print(f"Mix file already exists at {mix_path}. Skipping due to replace_existing=False.")
            continue

        stream = stream.copy(freeze_learning=True)
        stream.delay_samples = int(round(delay * stream.stream_config["sr"]))
        stream.reset_state()  # reset state but keep concealer model state (calibration)

        # Load as mono, resample to sr
        y, _ = librosa.load(infile_path, sr=stream.stream_config["sr"], mono=True)
        y = y.astype(np.float32, copy=False)

        if getattr(args, "exptype", None) == "oracle":
            # Load the turns file for this specific conversation
            turns_id = extract_turns_id(infile_path)
            turns_file_path = os.path.join(args.turns_path, f"{turns_id}.csv")
            turns_df = pd.read_csv(turns_file_path)
        else:
            turns_df = None

        original, concealer, concealer_metadata, mix, stream, n_speaker = run_fake_stream(y, stream, turns_df=turns_df)

        laeq_mix = compute_LAeq(np.mean(mix, axis=-1), stream.stream_config["sr"])
        laeq_original = compute_LAeq(original, stream.stream_config["sr"])
        diff_laeq = laeq_mix - laeq_original
        print(f"Difference in LAeq (mix - original): {diff_laeq:.2f} dB")

        if args.save_original:
            os.makedirs(f"{out_dir}/original/", exist_ok=True)
            sf.write(f"{out_dir}/original/{out_prefix}_original.flac", original, stream.stream_config["sr"], format='FLAC', subtype='PCM_24')
        if args.save_concealer:
            os.makedirs(f"{out_dir}/concealer/", exist_ok=True)
            sf.write(f"{out_dir}/concealer/{out_prefix}_concealer.flac", concealer, stream.stream_config["sr"], format='FLAC', subtype='PCM_24')
            # Save concealer metadata
            concealer_metadata_arr = np.array(concealer_metadata, dtype=object)
            n_nonempty = sum(1 for v in concealer_metadata_arr if v != '')
            print(f"Concealer metadata for {out_prefix}: {n_nonempty}/{len(concealer_metadata_arr)} non-empty block(s)")
            np.save(
                os.path.join(f"{out_dir}/concealer/", f"{out_prefix}_concealer_metadata.npy"),
                concealer_metadata_arr,
                allow_pickle=True,
            )

        os.makedirs(f"{out_dir}/mix/", exist_ok=True)
        sf.write(mix_path, mix, stream.stream_config["sr"], format='FLAC', subtype='PCM_24')
        out_dir = f"{args.output_difflaeq_dir}"
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"{out_dir}/{out_prefix}_difflaeq.npy", diff_laeq)


# -----------------------
# Whitenoise baseline: masks with speech-shaped noise instead of any learned
# concealer. No amods equivalent (amods has no baseline/adversary-free mode).
# -----------------------
def run_whitenoise_file_mode_via_callback_on_folder(args):
    sr = 48000

    for infile_path in args.eval_audio_files:
        print(f"\nProcessing: {infile_path}")
        infile = os.path.basename(infile_path)

        y, _ = librosa.load(infile_path, sr=sr, mono=True)
        y = y.astype(np.float32, copy=False)

        wn = generate_speech_shaped_noise(
            args.calib_audio_files,
            num_samples=len(y),
            sr=sr
        )

        db_diff = 0
        y_rms = np.sqrt(np.mean(y**2))
        wn_rms = np.sqrt(np.mean(wn**2))
        wn = wn * (y_rms * 10**(db_diff/20)) / wn_rms

        concealer = wn
        mix = soft_clip(y + concealer, limit=0.95)
        original = y

        laeq_mix = compute_LAeq(mix, sr)
        laeq_original = compute_LAeq(original, sr)
        diff_laeq = laeq_mix - laeq_original
        print(f"Difference in LAeq (mix - original): {diff_laeq:.2f} dB")

        out_prefix = infile.split(".")[0]
        out_dir = f"{args.output_audio_dir}"
        os.makedirs(out_dir, exist_ok=True)

        if args.save_original:
            os.makedirs(f"{out_dir}/original/", exist_ok=True)
            sf.write(f"{out_dir}/original/{out_prefix}_original.flac", original, sr, format='FLAC', subtype='PCM_24')
        if args.save_concealer:
            os.makedirs(f"{out_dir}/concealer/", exist_ok=True)
            sf.write(f"{out_dir}/concealer/{out_prefix}_concealer.flac", concealer, sr, format='FLAC', subtype='PCM_24')
        os.makedirs(f"{out_dir}/mix/", exist_ok=True)
        sf.write(f"{out_dir}/mix/{out_prefix}_mix.flac", mix, sr, format='FLAC', subtype='PCM_24')
        out_dir = f"{args.output_difflaeq_dir}"
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"{out_dir}/{out_prefix}_difflaeq.npy", diff_laeq)


def load_speech_files(files, sr=16000, max_files=None):
    if max_files:
        files = files[:max_files]

    signals = []
    for f in files:
        y, _ = librosa.load(f, sr=sr)
        signals.append(y)

    return signals, sr


def generate_speech_shaped_noise(files, num_samples, sr=48000):
    # 1. Load and concatenate speech
    signals, sr = load_speech_files(files, sr=sr)
    full_signal = np.concatenate(signals)

    # 2. Generate SSN via phase randomization
    X = np.fft.rfft(full_signal)
    random_phase = np.exp(2j * np.pi * np.random.rand(len(X)))
    noise_full = np.fft.irfft(np.abs(X) * random_phase)

    # 3. Normalize (important for RMS scaling later)
    noise_full = noise_full / (np.sqrt(np.mean(noise_full**2)) + 1e-8)

    # 4. Match requested length
    if len(noise_full) >= num_samples:
        return noise_full[:num_samples]
    else:
        repeats = (num_samples // len(noise_full)) + 1
        return np.tile(noise_full, repeats)[:num_samples]


def add_noise_at_snr(clean, noise, snr_db):
    clean = clean.astype(np.float64)
    noise = noise.astype(np.float64)

    min_len = min(len(clean), len(noise))
    clean = clean[:min_len]
    noise = noise[:min_len]

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    snr_linear = 10 ** (snr_db / 10)
    desired_noise_power = clean_power / snr_linear

    noise = noise * np.sqrt(desired_noise_power / (noise_power + 1e-12))

    mixed = clean + noise
    mixed = mixed / np.max(np.abs(mixed)) * 0.95

    return mixed


def apply_speech_shaped_noise(speech_folder, input_file, output_file="output_noisy.wav", snr_db=0, sr=16000):
    print("Loading input audio...")
    clean, _ = librosa.load(input_file, sr=sr)

    print("Generating speech-shaped noise...")
    noise = generate_speech_shaped_noise(speech_folder, num_samples=len(clean), sr=sr)

    print(f"Mixing at {snr_db} dB SNR...")
    noisy = add_noise_at_snr(clean, noise, snr_db)

    print(f"Saving to {output_file}")
    sf.write(output_file, noisy, sr)


# -----------------------
# Turn-taking / ground-truth speaker bookkeeping (two-speaker conversational
# simulation). Also reused directly by accuracy_metrics.py.
# -----------------------
def compute_speaker_timeline(turns_df, n_samples, sr):
    """
    Per-sample dominant speaker (dtype=object array of length n_samples, entries
    are speaker labels or None). Turns only carry a start timestamp, so each turn
    is assumed to run until the next turn starts (last turn runs until n_samples).
    Returns None if turns_df is None.
    """
    if turns_df is None:
        return None

    sorted_turns = turns_df.sort_values("timestamp").reset_index(drop=True)
    turn_starts = sorted_turns["timestamp"].to_numpy()
    turn_ends = np.append(turn_starts[1:], n_samples / sr)
    turn_speakers = sorted_turns["speaker"].to_numpy()

    timeline = np.full(n_samples, None, dtype=object)
    for start, end, speaker in zip(turn_starts, turn_ends, turn_speakers):
        start_sample = max(0, int(round(start * sr)))
        end_sample = min(n_samples, int(round(end * sr)))
        if end_sample > start_sample:
            timeline[start_sample:end_sample] = speaker
    return timeline


def dominant_speaker_in_range(timeline, start_sample, end_sample):
    """Majority speaker (ignoring None) in timeline[start_sample:end_sample], or None."""
    if timeline is None:
        return None
    counts = Counter(s for s in timeline[start_sample:end_sample] if s is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def run_fake_stream(y, stream, turns_df=None):
    """Simulate the callback over a whole file, block by block, optionally with a per-block ground-truth speaker."""
    n_orig = len(y)

    n_blocks = int(np.ceil(n_orig / stream.buffer_size))
    pad = n_blocks * stream.buffer_size - n_orig
    if pad > 0:
        y = np.pad(y, (0, pad), mode="constant")

    speaker_timeline = compute_speaker_timeline(turns_df, n_orig, stream.stream_config["sr"])
    if speaker_timeline is not None:
        n_speaker = np.array([
            dominant_speaker_in_range(speaker_timeline, b * stream.buffer_size, (b + 1) * stream.buffer_size)
            for b in range(n_blocks)
        ], dtype=object)
    else:
        n_speaker = None

    for b in range(n_blocks):
        block = y[b * stream.buffer_size: (b + 1) * stream.buffer_size]
        indata = block.reshape(-1, 1)
        outdata = np.zeros((stream.buffer_size, 2), dtype=np.float32)
        gt_speaker = n_speaker[b] if n_speaker is not None else None
        stream.callback(indata, outdata, stream.buffer_size, None, None, gt_speaker=gt_speaker)

    original = np.concatenate(stream.rec_original, axis=0)[:n_orig]
    concealer = np.concatenate(stream.rec_concealer, axis=0)[:n_orig]
    concealer_metadata = stream.rec_concealer_metadata
    mix = np.concatenate(stream.rec_mix, axis=0)[:n_orig]

    return original, concealer, concealer_metadata, mix, stream, n_speaker


def extract_speaker_id(filepath):
    filename = os.path.basename(filepath)
    # Example: spk_92__auntcretesemancipation_01_hill_0262__pan_0.wav
    first_part = filename.split("__")[0]   # spk_92
    spk_id = first_part.split("_")[1]  # 92
    return spk_id


def extract_turns_id(filepath):
    filename = os.path.basename(filepath)
    # Example: spk_p225-p226__p225-p226_conv0001_mic1__pan_0__ebr-high-asr-high.flac
    # -> turns file is named p225-p226_conv0001.csv
    conv_part = filename.split("__")[1]   # p225-p226_conv0001_mic1
    turns_id = re.sub(r"_mic\d+$", "", conv_part)  # p225-p226_conv0001
    return turns_id


# -----------------------
# Standalone CLI (mic stream / single file) - kept as a thin wrapper for
# ad-hoc manual testing of this paper's Stream (delay/gt_speaker); for
# anything not needing those, use the published `amods` CLI directly.
# -----------------------
if __name__ == "__main__":
    from amods.stream import run_file_mode, run_stream_mode

    parser = argparse.ArgumentParser(description="Run concealer on mic stream or on a wav file, using the SAME callback.")
    parser.add_argument("--infile", type=str, default=None, help="Input wav for file mode. If unset, stream mode is used directly from microphone.")
    parser.add_argument("--indir", type=str, default="./audios/", help="Input directory for file mode (e.g., ./audios/)")
    parser.add_argument("--outprefix", type=str, default="", help="Output prefix for file mode")
    parser.add_argument("--outdir", type=str, default="./output/", help="Output directory for file mode")
    parser.add_argument("--concealerconfig", type=str, default="default", help="Concealer config file name. Check folder configs/concealer/ to see available configs.")
    parser.add_argument("--streamconfig", type=str, default="default", help="Stream config file name. Check folder configs/stream/ to see available configs.")
    parser.add_argument("--sourcevadconfig", type=str, default="default_source", help="Source VAD config file name. Check folder configs/vad/ to see available configs.")
    parser.add_argument("--forecasterconfig", type=str, default="default", help="Forecaster config file name. Check folder configs/forecaster/ to see available configs.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")

    args = parser.parse_args()

    if args.infile is not None:
        run_file_mode(
            indir=args.indir, infile=args.infile, outdir=args.outdir, outprefix=args.outprefix,
            concealerconfig=args.concealerconfig, streamconfig=args.streamconfig,
            sourcevadconfig=args.sourcevadconfig, forecasterconfig=args.forecasterconfig, debug=args.debug,
        )
    else:
        run_stream_mode(
            outdir=args.outdir, outprefix=args.outprefix, concealerconfig=args.concealerconfig,
            streamconfig=args.streamconfig, sourcevadconfig=args.sourcevadconfig,
            forecasterconfig=args.forecasterconfig,
        )
