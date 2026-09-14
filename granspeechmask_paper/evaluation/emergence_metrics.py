from torch import randn
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio import SignalNoiseRatio
import librosa
import torch
import os
import torchaudio 
import numpy as np
from matplotlib.patches import Rectangle

class EchoicLogSurprise:
    """
    Formula from the paper:
    The robustness of echoic log-surprise auditory saliency detection.
    """
    def __init__(self):
        pass
    def compute_mel_spectrogram(self, audio, sr, n_mels=40, win_length=0.02, hop_length=0.01):

        n_fft = int(sr * win_length)
        hop = int(sr * hop_length)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            window='hamming',
            power=2.0
        )

        return mel_spec.T   # shape: (frames, mel_bands)
    
    def running_stats(self, X, N):
        frames, bands = X.shape
        mu = np.zeros_like(X)
        var = np.zeros_like(X)

        for n in range(frames):
            start = max(0, n - N + 1)
            buffer = X[start:n+1]

            mu[n] = np.mean(buffer, axis=0)
            var[n] = np.var(buffer, axis=0) + 1e-10

        return mu, var
    
    def log_surprise(self, mu_n, mu_prev, var_n, var_prev):

        term1 = (mu_n - mu_prev)**2 / (2 * var_prev)

        ratio = var_n / var_prev
        term2 = 0.5 * (ratio - 1 - np.log(ratio))

        dkl = term1 + term2

        return np.log(dkl + 1e-10)
    
    # ------------------------------------------------------------
    # 4. Single-scale saliency s(n)
    # ------------------------------------------------------------
    def saliency_single_scale(self, X, memory):

        frames, bands = X.shape

        mu, var = self.running_stats(X, memory)

        s = np.zeros(frames)

        for n in range(1, frames):

            d = self.log_surprise(
                mu[n], mu[n-1],
                var[n], var[n-1]
            )

            s[n] = np.mean(d)

        return s


    # ------------------------------------------------------------
    # 5. Multi-scale saliency
    # ------------------------------------------------------------
    def multi_scale_saliency(self, X, depth=5, N1=2):

        saliency_signals = []

        memory = N1

        for i in range(depth):

            s = self.saliency_single_scale(X, memory)
            saliency_signals.append(s)

            memory *= 2   # Ni = 2 * Ni-1

        return np.array(saliency_signals)


    # ------------------------------------------------------------
    # 6. Running histogram
    # ------------------------------------------------------------
    def running_histogram(self, signal, M=100, bins=100, value_range=(0, 5)):

        frames = len(signal)
        histograms = []

        for n in range(frames):

            start = max(0, n - M + 1)
            window = signal[start:n+1]

            hist, _ = np.histogram(
                window,
                bins=bins,
                range=value_range,
                density=True
            )

            histograms.append(hist + 1e-10)

        return np.array(histograms)


    # ------------------------------------------------------------
    # 7. Jensen-Shannon divergence fusion
    # ------------------------------------------------------------
    def js_divergence(self, hists):

        mean_hist = np.mean(hists, axis=0)

        js = 0

        for h in hists:
            js += entropy(h, mean_hist)

        return js / len(hists)


    def echoic_fusion(self, saliency_signals, M=100):

        depth, frames = saliency_signals.shape

        # ------------------------------------------------------------
        # Compute global range 
        # ------------------------------------------------------------
        global_min = np.percentile(saliency_signals, 1)
        global_max = np.percentile(saliency_signals, 99)

        # Optional safety (avoid degenerate case)
        if global_max <= global_min:
            global_max = global_min + 1e-6

        value_range = (global_min, global_max)

        histograms = [
            self.running_histogram(saliency_signals[i], M, value_range=value_range)
            for i in range(depth)
        ]

        # Histograms shape: (depth, frames, histogram bins)
        histograms = np.array(histograms)

        # # Plot only once to avoid opening many figures in repeated calls
        # first_hist = histograms[0, 100, :]
        # x_bins = np.arange(first_hist.size)

        # plt.figure(figsize=(8, 3))
        # plt.bar(x_bins, first_hist, color="steelblue", edgecolor="black", linewidth=0.5)
        # plt.title("First running histogram (scale 0, frame 0)")
        # plt.xlabel("Bin index")
        # plt.ylabel("Density")
        # plt.tight_layout()
        # plt.show()



        sechoic = np.zeros(frames)

        for n in range(frames):

            h = histograms[:, n, :]
            sechoic[n] = self.js_divergence(h)

        return sechoic


    # ------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------
    def echoic_log_surprise(self, audio, sr, n_mels=40, depth=5, N1=2, M=100):
        """
        From: The robustness of echoic log-surprise auditory saliency detection.
        """
        X = self.compute_mel_spectrogram(audio, sr, n_mels)

        multi_sal = self.multi_scale_saliency(X, depth, N1)

        final_saliency = self.echoic_fusion(multi_sal, M)

        return final_saliency

    def diff_echoic_log_surprise(self, audio_input, audio_ref, sr, n_mels=40, depth=5, N1=2, M=100):
        """
        From: The robustness of echoic log-surprise auditory saliency detection.
        But this time we use a reference audio, so it's new.
        """
        X_input = self.compute_mel_spectrogram(audio_input, sr, n_mels)
        X_ref = self.compute_mel_spectrogram(audio_ref, sr, n_mels)
        X_diff = X_input - X_ref

        multi_sal = self.multi_scale_saliency(X_diff, depth, N1)

        final_saliency = self.echoic_fusion(multi_sal, M)

        return final_saliency

    # ------------------------------------------------------------
    # Full pipeline but without echoic fusion
    # ------------------------------------------------------------
    def basic_log_surprise(self, audio, sr, n_mels=40, depth=5, N1=2):
        """
        From: The robustness of echoic log-surprise auditory saliency detection.
        Normally the initial parameters are N1=2 and depth=20, but in our case depth=20 means
        having an audio of minimum length 2^(20-1)*N1 = 2^19*2 = 1048576 frames, which is around 
        65 seconds at 16 kHz, so we use depth=5 by default.
        """
        X = self.compute_mel_spectrogram(audio, sr, n_mels)

        multi_sal = self.multi_scale_saliency(X, depth, N1)

        return multi_sal

    def diff_basic_log_surprise(self, audio_input, audio_ref, sr, n_mels=40, depth=5, N1=2):
        """
        From: The robustness of echoic log-surprise auditory saliency detection.
        But this time we use a reference audio, so it's new.
        """

        X_input = self.compute_mel_spectrogram(audio_input, sr, n_mels)
        X_ref = self.compute_mel_spectrogram(audio_ref, sr, n_mels)
        X_diff = X_input - X_ref
        # X_diff[X_diff < 0] = 0  # keep only non-negative differences

        multi_sal = self.multi_scale_saliency(X_diff, depth, N1)

        return multi_sal

class DiffSpectroSaliency:
    """
    Very simple formula that computes the differences of spectrograms.
    What's interesting to compare here is that whenever the noise is 
    shifted far away in time, if the source at this time is silence, 
    then you can shift further away without affecting the saliency.
    With the log-surprise model, the long-term memory makes it so that even 
    if you shift the noise far away, it still affects the saliency.
    """
    def __init__(self):
        pass

    def compute_mel_spectrogram(self, audio, sr, n_mels=40, win_length=0.02, hop_length=0.01):

        n_fft = int(sr * win_length)
        hop = int(sr * hop_length)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            window='hamming',
            power=2.0
        )

        return mel_spec.T   # shape: (frames, mel_bands)

    def diff_spectro_saliency(self, audio_input, audio_ref, sr, n_mels=32, depth=5, N1=2):
        """
        From: The robustness of echoic log-surprise auditory saliency detection.
        But this time we use a reference audio, so it's new.
        """

        X_input = self.compute_mel_spectrogram(audio_input, sr, n_mels)
        X_ref = self.compute_mel_spectrogram(audio_ref, sr, n_mels)
        X_diff = librosa.power_to_db(X_input + 1e-10, ref=1.0) - librosa.power_to_db(X_ref + 1e-10, ref=1.0)

        saliency = np.mean(X_diff)

        return saliency

    
class GammaBayesianSurprise:
    """
    Formula from the paper:
    “WOW!” BAYESIAN SURPRISE FOR SALIENT ACOUSTIC EVENT DETECTION
    """
    def __init__(self):
        pass

    # ------------------------------------------------------------
    # 1. Spectrogram (STFT power spectrogram)
    # ------------------------------------------------------------
    def compute_spectrogram(self, audio, sr, win_length=0.02, hop_length=0.01):

        n_fft = int(sr * win_length)
        hop = int(sr * hop_length)

        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop,
            window='hamming'
        )

        G = np.abs(stft) ** 2   # power spectrogram

        return G.T   # shape: (frames, freq_bins)

    # ------------------------------------------------------------
    # 3. Gaussian KL divergence per band
    # ------------------------------------------------------------
    def gaussian_kl(self, mu_post, mu_prior, var_post, var_prior):

        term1 = np.log(var_prior / var_post)

        term2 = var_post / var_prior

        term3 = (mu_post - mu_prior)**2 / var_prior

        return 0.5 * (term1 + term2 + term3 - 1)

    def update_gaussian(self, mu, var, x, zeta=0.9):

        mu_new = zeta * mu + (1 - zeta) * x

        var_new = zeta * var + (1 - zeta) * (x - mu_new)**2

        return mu_new, var_new

    def acoustic_surprise_gaussian(self, X, zeta=0.9):

        frames, bands = X.shape

        mu = X[0].copy()
        var = np.ones_like(mu) * 1e-6

        saliency = np.zeros(frames)

        for t in range(1, frames):

            x = X[t]

            # posterior
            mu_post, var_post = self.update_gaussian(mu, var, x, zeta)

            # KL divergence (per band)
            kl = self.gaussian_kl(mu_post, mu, var_post, var)

            # average across frequencies
            saliency[t] = np.mean(kl)

            # update prior
            mu, var = mu_post, var_post

        return saliency

    def acoustic_surprise(self, audio, sr):

        X = self.compute_spectrogram(audio, sr)

        saliency = self.acoustic_surprise_gaussian(X)

        return saliency


class GaussianBayesianSurprise:
    """
    Implementation of:
    "WOW! Bayesian Surprise for Salient Acoustic Event Detection"

    Gaussian model ONLY (paper-faithful version)
    """

    def __init__(self):
        pass

    def compute_mel_spectrogram(self, audio, sr, n_mels=40, win_length=0.02, hop_length=0.01):

        n_fft = int(sr * win_length)
        hop = int(sr * hop_length)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            window='hamming',
            power=2.0
        )

        return mel_spec.T   # shape: (frames, mel_bands)

    # ------------------------------------------------------------
    # 1. Spectrogram G(t, ω) = |F(t, ω)|^2
    # ------------------------------------------------------------
    def compute_spectrogram(self, audio, sr, win_length=0.02, hop_length=0.01):

        n_fft = int(sr * win_length)
        hop = int(sr * hop_length)

        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop,
            window='hamming'
        )

        G = np.abs(stft) ** 2  # power spectrogram

        return G.T  # shape: (frames, freq_bins)

    # ------------------------------------------------------------
    # 2. Gaussian KL divergence (Eq. 2 simplified for diagonal Σ)
    # ------------------------------------------------------------
    def gaussian_kl(self, mu_post, mu_prior, var_post, var_prior):

        # Avoid numerical issues
        var_post = np.maximum(var_post, 1e-10)
        var_prior = np.maximum(var_prior, 1e-10)

        term1 = np.log(var_prior / var_post)
        term2 = var_post / var_prior
        term3 = (mu_post - mu_prior) ** 2 / var_prior

        return 0.5 * (term1 + term2 + term3 - 1)

    # ------------------------------------------------------------
    # 3. Main acoustic surprise computation
    # ------------------------------------------------------------
    def acoustic_surprise(self, audio, sr, N=5, win_length=0.02, hop_length=0.01):

        # X = self.compute_spectrogram(audio, sr, win_length=win_length, hop_length=hop_length)
        X = self.compute_mel_spectrogram(audio, sr, win_length=win_length, hop_length=hop_length)
        # X = librosa.power_to_db(X)  # Convert to log-energy (dB) for better numerical stability

        # Compute and plot spectrogram
        # plt.figure(figsize=(12, 6))
        # plt.imshow(librosa.power_to_db(X.T, ref=np.max), aspect='auto', origin='lower', cmap='viridis')
        # plt.colorbar(label='Power (dB)')
        # plt.xlabel('Time (frames)')
        # plt.ylabel('Frequency (mel bands)')
        # plt.title('Mel Spectrogram')
        # plt.tight_layout()
        # plt.show()


        frames, bands = X.shape
        saliency = np.zeros(frames)

        for t in range(N, frames):

            # ----------------------------------------------------
            # Prior: [t-N, ..., t-1]
            # Posterior: [t-N, ..., t]
            # ----------------------------------------------------
            window_prior = X[t - N:t]
            window_post = X[t - N:t + 1]

            # Compute Gaussian parameters
            mu_prior = np.mean(window_prior, axis=0)
            var_prior = np.var(window_prior, axis=0)

            mu_post = np.mean(window_post, axis=0)
            var_post = np.var(window_post, axis=0)

            # KL divergence per frequency
            kl = self.gaussian_kl(mu_post, mu_prior, var_post, var_prior)

            # ----------------------------------------------------
            # 3.3 Across-frequency averaging (Eq. 11)
            # ----------------------------------------------------
            saliency[t] = np.mean(kl)

        # # Plot saliency superposed with waveform
        # time_frames = np.arange(len(saliency))
        # time_seconds = librosa.frames_to_time(time_frames, sr=sr, hop_length=int(sr * hop_length))

        # fig, ax1 = plt.subplots(figsize=(14, 5))

        # # Plot waveform on left y-axis
        # time_audio = np.linspace(0, len(audio) / sr, len(audio))
        # ax1.plot(time_audio, audio, color='steelblue', alpha=0.6, linewidth=0.5)
        # ax1.set_xlabel('Time (s)')
        # ax1.set_ylabel('Amplitude', color='steelblue')
        # ax1.tick_params(axis='y', labelcolor='steelblue')

        # # Plot saliency on right y-axis
        # ax2 = ax1.twinx()
        # ax2.plot(time_seconds, saliency, color='red', alpha=0.8, linewidth=2, label='Saliency')
        # ax2.set_ylabel('Saliency', color='red')
        # ax2.tick_params(axis='y', labelcolor='red')

        # fig.tight_layout()
        # plt.show()

        # saliency = np.log(saliency)  # Log-transform for better visualization

        return saliency


class AcousticEmergence:
    """
    Implementation of Acoustic Emergence
    """

    def __init__(self):
        pass

    def compute_mel_spectrogram(self, audio, sr, n_mels=40, win_length=0.02, hop_length=0.01):

        n_fft = int(sr * win_length)
        hop = int(sr * hop_length)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            window='hamming',
            power=2.0
        )

        return mel_spec.T   # shape: (frames, mel_bands)
    
    def acoustic_emergence(self, audio_input, audio_ref, sr,
                        n_mels=40, win_length=0.02, hop_length=0.01):
        """
        Acoustic emergence:
        E(t) = L_input(t) - L_ref(t)
        """

        # ------------------------------------------------------------
        # 1. Cochleograms (power)
        # ------------------------------------------------------------
        # X_input = self.compute_mel_spectrogram(audio_input, sr, n_mels,
        #                                 win_length, hop_length)
        # X_ref = self.compute_mel_spectrogram(audio_ref, sr, n_mels,
        #                                 win_length, hop_length)

        # # ------------------------------------------------------------
        # # 2. Convert to broadband energy per frame
        # # ------------------------------------------------------------
        # # mean over frequency bands
        # energy_input = np.mean(X_input, axis=1)
        # energy_ref   = np.mean(X_ref, axis=1)

        # # ------------------------------------------------------------
        # # 3. Convert to dB
        # # ------------------------------------------------------------
        # eps = 1e-10
        # L_input = 10 * np.log10(energy_input + eps)
        # L_ref   = 10 * np.log10(energy_ref + eps)

        
        # ------------------------------------------------------------
        # 4. Emergence
        # ------------------------------------------------------------
        # emergence = L_input - L_ref
        # emergence = np.mean(emergence)

        # Compute loudness directly from waveforms
        eps = 1e-10
        energy_input = np.mean(audio_input ** 2)
        energy_ref = np.mean(audio_ref ** 2)

        L_input = 10 * np.log10(energy_input + eps)
        L_ref = 10 * np.log10(energy_ref + eps)

        emergence = L_input - L_ref

        return emergence


def run_emergence_metrics(args):
    emergence_path = os.path.join(args.emergence_dir, f"{args.setting_identifier}_emergence.npy")
    fnames_path = os.path.join(args.emergence_dir, f"{args.setting_identifier}_fnames.npy")

    if not args.replace_existing and os.path.exists(emergence_path) and os.path.exists(fnames_path):
        print(f"Skipping {args.setting_identifier}; outputs already exist.")
        return

    # If a "mix" folder exists inside output_audio_dir, use it instead
    mix_dir = os.path.join(args.output_audio_dir, "mix")
    if os.path.isdir(mix_dir):
        output_audio_dir = mix_dir
        suffix = "_mix"
    else:
        output_audio_dir = args.output_audio_dir
        suffix = ""

    ref_audio_files = [os.path.basename(f) for f in args.eval_audio_files]
    eval_audio_files = [os.path.splitext(f)[0] + suffix + os.path.splitext(f)[1] for f in ref_audio_files]

    emergence_metric = DiffSpectroSaliency()

    emergence_scores = []
    fnames = []
    print('Calculating ermergence...')

    for ref_audio_file, audio_file in zip(ref_audio_files, eval_audio_files):
        file_path = os.path.join(output_audio_dir, audio_file)
        audio, sr = librosa.load(file_path, sr=16000)
        ref_audio_path = os.path.join(args.eval_dataset_dir, ref_audio_file)
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000)
        emergence_score = emergence_metric.diff_spectro_saliency(audio, ref_audio, sr)  #
        emergence_scores.append(emergence_score)
        fnames.append(audio_file)

    np.save(emergence_path, np.array(emergence_scores))
    np.save(fnames_path, np.array(fnames))
