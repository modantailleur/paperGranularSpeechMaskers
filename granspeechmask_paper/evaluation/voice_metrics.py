import jiwer
import librosa
import torch
import torchaudio
import os
import numpy as np
from pystoi import stoi
import scipy
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from speechbrain.inference.interfaces import Pretrained
import numpy as np
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
import time

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

class WerMetricW2V2():
    def __init__(self, device=torch.device("cpu")):
        self.device = device

        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        self.model = self.bundle.get_model().to(device)

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.decoder = GreedyCTCDecoder(labels=self.bundle.get_labels())

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        reference_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio.to(self.device)
        if sr != self.bundle.sample_rate:
            audio_input = torchaudio.functional.resample(audio, sr, self.bundle.sample_rate).to(self.device)

        with torch.inference_mode():
            emission, _ = self.model(audio_input)

        transcript = self.decoder(emission[0])
        transcript = transcript.replace("|", " ")

        return transcript
    
    def get_wer(self, transcript_target, transcript_test):
        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        reference_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )
        return(wer)

class WerMetricFastConfo():
    def __init__(self, device=torch.device("cpu")):
        self.device = device

        import nemo.collections.asr as nemo_asr
        import librosa
        self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/stt_en_fastconformer_ctc_large", device=device)

        audio, _ = librosa.load('./audios/office_audio_LJSpeech.wav', sr=16000)

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        reference_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio
        if sr != self.sr:
            audio_input = torchaudio.functional.resample(audio, sr, self.sr)

        # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(dim=0, keepdim=True)
        
        audio_input = audio_input.to(self.device)

        # Generate transcription
        with torch.no_grad():
            output = self.model.transcribe(audio)
            transcription = output[0].text
        
        return transcription
    
    def get_wer(self, transcript_target, transcript_test):
        try:
            wer = jiwer.wer(
                transcript_target,
                transcript_test,
                reference_transform=self.transforms,
                hypothesis_transform=self.transforms,
            )
        except ValueError:
            print('ValueError: one of the transcript might be empty')
            wer = 1.0
        return(wer)


class WerMetricWhisper():
    def __init__(self, device=torch.device("cpu"), model_type="whisper-large-v3"):
        # callers in this repo sometimes pass device as a plain string (e.g. "cuda:0")
        # rather than a torch.device, so normalize before relying on .type/.index below.
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        model_prefix = "openai/"

        # Load the Whisper model and processor
        model_name = f"openai/{model_type}"
        if model_type not in ["whisper-large-v3", "whisper-large-v2", \
                              "whisper-large", "whisper-medium", "whisper-small", "whisper-base", "whisper-tiny"]:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        # .float(): some transformers versions default to the checkpoint's native dtype
        # (float16 for whisper-large-v3), which conv1d can't mix with the float32
        # features the processor produces. Cast after loading to avoid relying on the
        # from_pretrained dtype kwarg, whose name changed across transformers versions
        # (torch_dtype vs dtype).
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device).float()

        if device.type == "cpu":
            # Dynamic int8 quantization of the linear layers: decoding is autoregressive
            # (hundreds of sequential forward passes per chunk) so it's the bottleneck on
            # CPU, and quantizing just the Linear layers is a large, low-effort speedup
            # there with only a modest hit to transcription quality.
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        # Long-form transcription via the pipeline's fixed-size chunk_length_s=30
        # chunking, as recommended on the model card, instead of Whisper's built-in
        # seek/timestamp-based long-form generate(): the seek algorithm's chunk
        # boundaries depend on the model's own (sometimes unreliable) timestamp
        # predictions, which can blow up runtime on hard audio; fixed-size chunking
        # has predictable, linear-in-duration cost and is batchable.
        # stride_length_s=5.0: overlap on each side, just enough to give the stitching
        # algorithm context across a chunk boundary, instead of the default ~5s-each-side
        # overlap (a 60s clip now costs ~65s of audio instead of ~90s).
        pipeline_device = -1 if device.type == "cpu" else (device.index if device.index is not None else 0)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            stride_length_s=5.0,
            device=pipeline_device,
            generate_kwargs={"language": "en", "task": "transcribe"},
        )

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        reference_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio
        if sr != self.sr:
            audio_input = torchaudio.functional.resample(audio, sr, self.sr)

        # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(dim=0, keepdim=True)

        audio_input = audio_input.squeeze().to(torch.device("cpu")).numpy()

        transcription = self.pipe({"raw": audio_input, "sampling_rate": self.sr})["text"]
        return transcription
    
    def get_wer(self, transcript_target, transcript_test):
        try:
            wer = jiwer.wer(
                transcript_target,
                transcript_test,
                reference_transform=self.transforms,
                hypothesis_transform=self.transforms,
            )
        except ValueError:
            print('ValueError: one of the transcript might be empty')
            wer = 1.0
        return(wer)

class WerMetricSPT():
    def __init__(self, device=torch.device("cpu")):
        # self.device = torch.device("cpu")
        self.device = device
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr").to(self.device)

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        reference_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio
        if sr != self.sr:
            audio_input = torchaudio.functional.resample(audio, sr, self.sr)

        # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(dim=0, keepdim=True)
        
        # audio_input = audio_input.to(torch.device("cpu"))

        # inputs = self.processor(audio_input.squeeze(), sampling_rate=self.sr, return_tensors="pt", device=self.device).to(self.device)
        inputs = self.processor(audio_input.detach().cpu().squeeze(0).numpy(), sampling_rate=self.sr, return_tensors="pt")

        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]

        # Generate the transcription
        generated_ids = self.model.generate(input_features.to(self.device), attention_mask=attention_mask.to(self.device))

        # Decode the generated ids to get the transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return transcription[0]
    
    def get_wer(self, transcript_target, transcript_test):
        try:
            wer = jiwer.wer(
                transcript_target,
                transcript_test,
                reference_transform=self.transforms,
                hypothesis_transform=self.transforms,
            )
        except ValueError:
            print('ValueError: one of the transcript might be empty')
            wer = 1.0
        return(wer)

class WerMetricSB():
    def __init__(self, device=torch.device("cpu"), asr_short_name="ct"):
        # self.device = torch.device("cpu")
        self.device = device
        # asr_name = "asr-wav2vec2-commonvoice-14-en"
        # asr_name = "asr-streaming-conformer-librispeech"
        # asr_name = "asr-conformersmall-transformerlm-librispeech"
        # asr_name = "asr-crdnn-commonvoice-14-en"
        # asr_name = "asr-crdnn-rnnlm-librispeech"
        if asr_short_name == "cst":
            asr_name = "asr-conformersmall-transformerlm-librispeech"
        if asr_short_name == "sc":
            asr_name = "asr-streaming-conformer-librispeech"
        if asr_short_name == "ct":
            asr_name = "asr-conformer-transformerlm-librispeech"
        if asr_short_name == "cr":
            asr_name = "asr-crdnn-rnnlm-librispeech"
            
        self.asr_model = SpeechbrainEncoderDecoderASR.from_hparams(source=f"speechbrain/{asr_name}", savedir=f"pretrained_models/{asr_name}", run_opts={"device":device})

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        reference_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio

        if sr != self.sr:
            audio_input = torchaudio.functional.resample(audio, sr, self.sr)

        # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(dim=0, keepdim=True)
        
        # audio_input = audio_input.to(torch.device("cpu"))
        transcription = self.asr_model.transcribe_file(audio_input)
        
        return transcription
    
    def get_wer(self, transcript_target, transcript_test):
        try:
            wer = jiwer.wer(
                transcript_target,
                transcript_test,
                reference_transform=self.transforms,
                hypothesis_transform=self.transforms,
            )
        except ValueError:
            print('ValueError: one of the transcript might be empty')
            wer = 1.0
        return(wer)

class SpeechbrainEncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contain the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.inference.ASR import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP
    >>> asr_model.transcribe_file("tests/samples/single-mic/example2.flac")  # doctest: +SKIP
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        self.transducer_beam_search = False
        self.transformer_beam_search = False
        if hasattr(self.hparams, "transducer_beam_search"):
            self.transducer_beam_search = self.hparams.transducer_beam_search
        if hasattr(self.hparams, "transformer_beam_search"):
            self.transformer_beam_search = self.hparams.transformer_beam_search

    def transcribe_file(self, waveform, **kwargs):
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        # waveform = self.load_audio(path, **kwargs)
        # Fake a batch:
        # audio = torch.from_numpy(waveform)
        # batch = audio.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.transcribe_batch(
            waveform, rel_length
        )
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        if self.transformer_beam_search:
            encoder_out = self.mods.transformer.encode(encoder_out, wav_lens)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            if self.transducer_beam_search:
                inputs = [encoder_out]
            else:
                inputs = [encoder_out, wav_lens]
            predicted_tokens, _, _, _ = self.mods.decoder(*inputs)
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens)

class STOIMetric():
    def __init__(self, device=torch.device("cpu")):
        self.device = device
        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):

        d = stoi(audio_target[0], audio_test[0], sr, extended=False)
        
        return d

def load_wer_metrics(device):
    wer_w2v2 = WerMetricW2V2(device=device)
    wer_whisper = WerMetricWhisper(device=device)
    wer_spt = WerMetricSPT(device=device)
    # wer_sb_ct = WerMetricSB(device=device, asr_short_name="ct")
    # wer_sb_sc = WerMetricSB(device=device, asr_short_name="sc")
    # wer_sb_cst = WerMetricSB(device=device, asr_short_name="cst")
    wer_sb_cr = WerMetricSB(device=device, asr_short_name="cr")

    wer_dict = {
        "w2v2": wer_w2v2,
        "whisper": wer_whisper,
        "spt": wer_spt,
        # "sbct": wer_sb_ct, #unused in previous experiments, but can be added back if needed
        # "sbsc": wer_sb_sc, #unused in previous experiments, but can be added back if needed
        # "sbcst": wer_sb_cst, #unused in previous experiments, but can be added back if needed
        "sbcr": wer_sb_cr,
        }
    
    return wer_dict

def run_voice_metrics(args):

    start_time = time.time()

    device = args.device
    # wer_w2v2 = WerMetricW2V2(device=device)
    # wer_whisper = WerMetricWhisper(device=device, model_type="whisper-tiny")
    wer_whisperlarge = WerMetricWhisper(device=device, model_type="whisper-large-v3")
    # wer_spt = WerMetricSPT(device=device)
    # wer_sb_cr = WerMetricSB(device=device, asr_short_name="cr")
    # wer_sb_ct = WerMetricSB(device=device, asr_short_name="ct")
    # wer_sb_sc = WerMetricSB(device=device, asr_short_name="sc")
    # wer_sb_cst = WerMetricSB(device=device, asr_short_name="cst")

    wer_dict = {
        # "w2v2": wer_w2v2,
        # "whisper": wer_whisper,
        "whisperlarge": wer_whisperlarge,
        # "spt": wer_spt,
        # "sbcr": wer_sb_cr,
        # "fastconfo": WerMetricFastConfo(device=device),
        
        # NOT WORKING
        # "sbct": wer_sb_ct, #unused in previous experiments, but can be added back if needed
        # "sbsc": wer_sb_sc, #unused in previous experiments, but can be added back if needed
        # "sbcst": wer_sb_cst, #unused in previous experiments, but can be added back if needed

        }
    
    end_time = time.time()
    print(f"Initialization time: {end_time - start_time:.2f} seconds")

    mix_dir = os.path.join(args.output_audio_dir, "mix")
    if os.path.isdir(mix_dir):
        output_audio_dir = mix_dir
    else:
        output_audio_dir = args.output_audio_dir
    audio_files = [f for f in os.listdir(output_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]

    for metric_name, wer_metric in wer_dict.items():

        print('Calculating WER for metric:', metric_name)
        wer_path = os.path.join(args.wer_dir, f"{args.setting_identifier}_{metric_name}_wer.npy")
        fnames_path = os.path.join(args.wer_dir, f"{args.setting_identifier}_{metric_name}_fnames.npy")
        if not args.replace_existing and os.path.exists(wer_path) and os.path.exists(fnames_path):
            print(f"Skipping {metric_name}; outputs already exist.")
            continue

        wers = []
        fnames = []

        for audio_file in audio_files:
            file_path = os.path.join(output_audio_dir, audio_file)
            audio, sr = torchaudio.load(file_path)  # Add batch dimension
            audio = audio.mean(dim=0, keepdim=True)
            audio = audio.to(device)
            transcript = wer_metric.get_transcript(audio, sr=sr)
            transc_file = audio_file.split("__")[1]
            transcript_gt = open(os.path.join(args.transc_dataset_dir, transc_file + ".txt")).read()
            wer = wer_metric.get_wer(transcript_gt, transcript)
            wer = min(wer, 1.0)  # Ensure WER does not exceed 1.0
            wers.append(wer)
            fnames.append(audio_file)
            # print("Average WER:", np.mean(wers))
        
        np.save(wer_path, np.array(wers))
        np.save(fnames_path, np.array(fnames))