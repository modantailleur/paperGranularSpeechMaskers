"""
Paper-specific concealer algorithm, built on top of the published `amods`
package rather than reimplementing it. `amods.models.concealer.GranSpeechMask`
covers the general algorithm; the extensions here (ground-truth speaker
restriction for the multispeaker oracle experiments, and DNSMOS
quality-gating) are research-only knobs specific to this paper's ablations,
so they live here rather than in the general-purpose package.
"""
from collections import Counter
import numpy as np
import warnings

from amods.audio import apply_fade
from amods.models.concealer import register_concealer
from amods.models.concealer.granspeechmask import GranSpeechMask

from granspeechmask_paper.evaluation.quality_metrics import DNSMOSMetric


def _extract_speaker_id_from_voice_name(voice_name):
    # Mirrors conceal.py's extract_speaker_id: 'spk_p225__...' -> 'p225'.
    # voice_name comes from a calibration filename stem, so entries that don't
    # follow that convention (e.g. "", "stream_3") simply won't match any gt_speaker.
    if not voice_name:
        return None
    try:
        first_part = voice_name.split("__")[0]  # spk_p225
        return first_part.split("_")[1]  # p225
    except IndexError:
        return None


def _dominant_label(labels):
    # Majority vote over a per-sample speaker timeline slice, ignoring None (silence/no turn).
    counts = Counter(l for l in labels if l is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _substitute_speaker_in_voice_name(voice_name, speaker_id):
    # 'spk_p225-p226__p225-p226_conv0021_mic1__...' -> 'spk_p226__p225-p226_conv0021_mic1__...'
    if not voice_name:
        return voice_name
    parts = voice_name.split("__")
    parts[0] = f"spk_{speaker_id}"
    return "__".join(parts)


@register_concealer("granspeechmask_paper")
class GranSpeechMaskPaper(GranSpeechMask):
    """
    amods.models.concealer.GranSpeechMask plus this paper's ablation knobs:
    DNSMOS quality-gating of new candidate clips, ground-truth speaker
    restriction (multispeaker oracle experiments), and a zero-warm-up
    concealing threshold (concealing starts as soon as memory has at least
    one candidate, unlike amods's own min_memory_to_conceal default of 20 -
    this paper's original behavior, never configured otherwise here).
    """

    def __init__(self, sr, config_path, config):
        super().__init__(sr, config_path, config)
        self.quality_threshold = config.get("quality_threshold", None)
        if self.quality_threshold is not None:
            self.dnsmos_metric = DNSMOSMetric()

    def get_concealer(self, x, gt_speaker=None):
        self.update_memory(x)
        self.update(x)  # extends pending_voice with x, so it's included below

        should_conceal = (not self.cur_concealing) and (len(self.memory) > 0)

        if should_conceal:
            self.cur_concealing = True

            # The decision of which concealer matches the current buffer isn't
            # limited to the raw buffer x: it's made over the trailing
            # decision_win seconds of audio (buffer x plus whatever context
            # precedes it), a window that slides forward by one buffer every
            # callback. pending_voice already holds x (see self.update(x)
            # above) plus prior history, so slicing its tail gives exactly
            # that sliding window.
            decision_win_samples = int(self.sr * self.decision_win)
            decision_audio = np.array(self.pending_voice)[-decision_win_samples:]
            x_feat = self._feature_extractor(decision_audio, norm=True)

            # Restrict candidates to the gt_speaker's own concealers, if available
            candidate_indices = range(len(self.memory))
            if gt_speaker is not None:
                gt_indices = [
                    i for i, (_, _, _, voice_name) in enumerate(self.memory)
                    if _extract_speaker_id_from_voice_name(voice_name) == gt_speaker
                ]
                if len(gt_indices) == 0:
                    warnings.warn(
                        f"No concealer found in memory for ground-truth speaker '{gt_speaker}'. "
                        "Falling back to the entire memory."
                    )
                else:
                    candidate_indices = gt_indices

            # Avoid reusing same concealer too often using cooldown
            length_condition = max(0, len(self.memory) - self.max_concealer_distance_to_buffer)

            if self.distance_type == "p-correlation":
                mel_distances = [
                    1 - np.corrcoef(x_feat, self.memory[i][1])[0, 1] if self.memory[i][2] == 0 and i < length_condition else np.inf
                    for i in candidate_indices
                ]
            elif self.distance_type == "cosine":
                x_feat_norm = np.linalg.norm(x_feat)
                mel_distances = [
                    1 - (np.dot(x_feat, self.memory[i][1]) / (x_feat_norm * (np.linalg.norm(self.memory[i][1]) + 1e-10))) if self.memory[i][2] == 0 and i < length_condition else np.inf
                    for i in candidate_indices
                ]
            else:
                raise ValueError(f"Unknown distance_type: {self.distance_type}")

            min_distance = np.inf
            best_idx = 0
            for local_i, distance in enumerate(mel_distances):
                if distance < min_distance:
                    min_distance = distance
                    best_idx = candidate_indices[local_i]
            if min_distance == np.inf:
                eligible_candidates = [i for i in candidate_indices if i < length_condition]
                random_pool = eligible_candidates if len(eligible_candidates) > 0 else list(candidate_indices)
                best_idx = int(np.random.choice(random_pool)) if len(random_pool) > 0 else 0

            # Set chosen concealer cooldown
            concealer, concealer_feat, _, voice_name = self.memory[best_idx]
            self.memory[best_idx] = (concealer, concealer_feat, self.max_countdown_reuse, voice_name)

            # Normalize energy to match input buffer
            concealer = concealer * (np.sqrt(np.mean(np.array(self.pending_voice)**2)) / np.sqrt(np.mean(concealer**2)))

            concealing_min_timeout_size = int(len(concealer) * self.concealing_min_timeout_ratio)
            concealing_max_timeout_size = int(len(concealer) * self.concealing_max_timeout_ratio)
            self.concealing_countdown = np.random.randint(concealing_min_timeout_size, concealing_max_timeout_size) \
                if concealing_max_timeout_size > concealing_min_timeout_size else \
                concealing_min_timeout_size
        else:
            # Return silence
            concealer = np.zeros(self.pending_conc_max_size, dtype=np.float32)
            voice_name = ""
        return concealer, voice_name

    def _feed_memory(self, y, voice_name=None, gt_speaker=None):
        new_memory = []

        pending_audio = np.array(y)

        if self.denoise:
            pending_audio_denoised = self.denoiser.predict(pending_audio)
        else:
            pending_audio_denoised = pending_audio

        if len(pending_audio_denoised) > len(pending_audio):
            pending_audio_denoised = pending_audio_denoised[:len(pending_audio)]
        elif len(pending_audio_denoised) < len(pending_audio):
            pending_audio_denoised = np.pad(pending_audio_denoised, (0, len(pending_audio) - len(pending_audio_denoised)), mode='constant')

        new_concealers = np.array_split(pending_audio_denoised, self.n_concealer_before_denoise)
        # Same split, on the original (non-denoised) audio: pending_audio and
        # pending_audio_denoised are the same length (see padding/truncation
        # above), so array_split produces identical segment boundaries for
        # both - this just gives each denoised chunk its original counterpart.
        new_concealers_original = np.array_split(pending_audio, self.n_concealer_before_denoise)

        if gt_speaker is not None:
            gt_speaker_splits = np.array_split(np.asarray(gt_speaker, dtype=object), self.n_concealer_before_denoise)
        else:
            gt_speaker_splits = [None] * len(new_concealers)

        fade_size = int(self.sr * self.fade_duration)
        fade_size = fade_size if fade_size > self.sr // 100 else self.sr // 100  # min 10ms of fade

        for concealer, concealer_original, concealer_gt_speaker in zip(new_concealers, new_concealers_original, gt_speaker_splits):
            concealer = apply_fade(concealer, fade_size)
            if self.random_reverse:
                concealer = self.am.predict(concealer)
            # Features come from the original (non-denoised) segment, not the
            # denoised/faded/reversed one played back below: the live query
            # side (pending_voice in get_concealer) is raw mic audio that's
            # never denoised, so matching against denoised candidate features
            # would compare across two different audio domains.
            concealer_feat = self._feature_extractor(concealer_original, norm=True)
            if self.vad.predict(concealer):
                if self.quality_threshold is not None:
                    dnsmos_score = self.dnsmos_metric.calculate(concealer, sr=self.sr)[-1]
                    if dnsmos_score < self.quality_threshold:
                        continue
                concealer_voice_name = voice_name if voice_name is not None else ""
                if concealer_gt_speaker is not None:
                    dominant_speaker = _dominant_label(concealer_gt_speaker)
                    if dominant_speaker is not None:
                        concealer_voice_name = _substitute_speaker_in_voice_name(concealer_voice_name, dominant_speaker)
                new_memory.append((concealer.astype(np.float32, copy=False), concealer_feat.astype(np.float32, copy=False), 0, concealer_voice_name))

        # Fill memory up to max length
        while len(self.memory) < self.memory_maxlen and len(new_memory) > 0:
            self.memory.append(new_memory.pop(0))

        if self.max_len_strat == "queue":
            self.memory = self.memory + new_memory
            # Enforce max length manually (drop oldest)
            while len(self.memory) > self.memory_maxlen:
                self.memory.pop(0)

        elif self.max_len_strat == "diversity_marginal_gain":
            # Decide to keep new concealers based on greedy marginal gain in diversity (log-Mel frequencies correlation)
            if len(new_memory) > 0:
                if self.cross_corr is None:
                    if self.distance_type == "p-correlation":
                        self.cross_corr = self._compute_correlation_matrix(self.memory)  # shape: (k, k)
                    if self.distance_type == "cosine":
                        self.cross_corr = self._compute_cosine_similarity_matrix(self.memory)  # shape: (k, k)
                for e in new_memory:
                    self.memory, self.cross_corr = self._compute_swap_marginal_gain(self.memory, self.cross_corr, e, distance_type=self.distance_type)

        if not self.is_stream:
            self.pending_voice.clear()
        self.stop_processing = False

    def learn(self, y, voice_name=None, gt_speaker=None):
        pending_audio = np.array(y)
        chunk_size = int(self.pending_voice_max_duration * self.sr)

        for i in range(0, len(pending_audio), chunk_size):
            chunk = pending_audio[i:i + chunk_size]
            if len(chunk) == chunk_size:
                chunk_gt_speaker = gt_speaker[i:i + chunk_size] if gt_speaker is not None else None
                self._feed_memory(chunk, voice_name=voice_name, gt_speaker=chunk_gt_speaker)

    def copy(self):
        new_cm = GranSpeechMaskPaper(self.sr, self.config_path, self.config)
        new_cm.memory = [(concealer, concealer_feat, 0, voice_name)
                         for concealer, concealer_feat, _, voice_name in self.memory]
        return new_cm
