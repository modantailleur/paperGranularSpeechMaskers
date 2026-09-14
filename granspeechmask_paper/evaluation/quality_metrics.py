from torch import randn
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio import SignalNoiseRatio
import librosa
import torch
import os
import torchaudio 
import numpy as np

class DNSMOSMetric():
    def __init__(self, device=torch.device("cpu"), num_threads=1):
        self.device = device
        self.sr = 8000
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(self.sr, False, num_threads=num_threads).to(self.device)

    def calculate(self, x, sr):

        if sr != self.sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.sr)

        x_torch = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            d = self.dnsmos(x_torch)

        return d

class SNRMetric():
    def __init__(self):
        self.snr = SignalNoiseRatio(self.sr, False)

    def calculate(self, x_preds, x_target):
        return self.snr(x_preds, x_target)

class UTMOSMetric():
    def __init__(self):
        self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    def calculate(self, x, sr):

        x_input = torch.from_numpy(x).float().unsqueeze(0)

        score = self.predictor(x_input, sr).item()

        return(score)

def run_quality_metrics(args):
    device = args.device

    # If a "mix" folder exists inside output_audio_dir, use it instead
    mix_dir = os.path.join(args.output_audio_dir, "mix")
    if os.path.isdir(mix_dir):
        output_audio_dir = mix_dir
    else:
        output_audio_dir = args.output_audio_dir

    audio_files = [f for f in os.listdir(output_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    dnsmos_metric = DNSMOSMetric(device=device)

    dnsmosp808_scores = []
    dnsmossig_scores = []
    dnsmosbak_scores = []
    dnsmosovr_scores = []
    fnames = []
    for audio_file in audio_files:
        file_path = os.path.join(output_audio_dir, audio_file)
        audio, sr = librosa.load(file_path, sr=8000)
        dnsmos_score = dnsmos_metric.calculate(audio, sr)  #
        dnsmosp808_scores.append(dnsmos_score[0])
        dnsmossig_scores.append(dnsmos_score[1])
        dnsmosbak_scores.append(dnsmos_score[2])
        dnsmosovr_scores.append(dnsmos_score[3])
        fnames.append(audio_file)

    np.save(os.path.join(args.dnsmos_dir, f"{args.setting_identifier}_dnsmosp808.npy"), np.array(dnsmosp808_scores))
    np.save(os.path.join(args.dnsmos_dir, f"{args.setting_identifier}_dnsmossig.npy"), np.array(dnsmossig_scores))
    np.save(os.path.join(args.dnsmos_dir, f"{args.setting_identifier}_dnsmosbak.npy"), np.array(dnsmosbak_scores))
    np.save(os.path.join(args.dnsmos_dir, f"{args.setting_identifier}_dnsmosovr.npy"), np.array(dnsmosovr_scores))
    np.save(os.path.join(args.dnsmos_dir, f"{args.setting_identifier}_fnames.npy"), np.array(fnames))
