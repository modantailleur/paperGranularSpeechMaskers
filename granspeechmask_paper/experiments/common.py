import os
import torch
import pandas as pd
from granspeechmask_paper.config_generation import create_config_files

LATENCIES_PATH = os.path.join(os.path.dirname(__file__), "data", "latencies.csv")


def _lookup_vad_delay(vad_name):
    """
    Average measured computational delay (seconds) for a VAD, read from
    utils/latencies.csv. A column matches if vad_name (e.g. "webrtc") appears
    in its header (e.g. "WebRTC + Mel + Cosine"); the "Average" row holds the
    delay across all measured machines, in ms.
    """
    if not vad_name:
        return 0.0
    latencies_df = pd.read_csv(LATENCIES_PATH, index_col=0, decimal=",")
    for col in latencies_df.columns:
        if vad_name.lower() in col.lower():
            return float(latencies_df.loc["Average", col]) / 1000.0 # convert ms to seconds
    return 0.0

speakers_list = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', \
                'p232', 'p233', 'p234', 'p236', 'p237', 'p238', 'p239', \
                'p240', 'p241', 'p243', 'p244', 'p245', 'p246', 'p247',\
                'p248', 'p249', 'p250', 'p251', 'p252', 'p253', 'p254',\
                'p255', 'p256', 'p257', 'p258', 'p259', 'p260', 'p261',\
                'p262', 'p263', 'p264', 'p265', 'p266', 'p267', 'p268', \
                'p269', 'p270', 'p271', 'p272', 'p273', 'p274', 'p275', \
                'p276', 'p277', 'p278', 'p279', 'p280', 'p281', 'p282', \
                'p283', 'p284', 'p285', 'p286', 'p287', 'p288', 'p292', \
                'p293', 'p294', 'p295', 'p297', 'p298', 'p299', 'p300', \
                'p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', \
                'p308', 'p310', 'p311', 'p312', 'p313', 'p314', 'p315', \
                'p316', 'p317', 'p318', 'p323', 'p326', 'p329', 'p330', \
                'p333', 'p334', 'p335', 'p336', 'p339', 'p340', 'p341', \
                'p343', 'p345', 'p347', 'p351', 'p360', 'p361', 'p362', \
                'p363', 'p364', 'p374', 'p376', 's5']

multispeakers_list = [
    'p225-p226', 'p233-p234', 'p243-p244', 'p251-p252', 'p259-p260', 'p267-p268', 'p275-p276', 'p283-p284', 'p294-p295', 'p303-p304', 'p312-p313', 'p329-p330', 'p341-p343', 'p363-p364',
    'p227-p228', 'p236-p237', 'p245-p246', 'p253-p254', 'p261-p262', 'p269-p270', 'p277-p278', 'p285-p286', 'p297-p298', 'p305-p306', 'p314-p316', 'p333-p334', 'p345-p347', 'p374-p376',
    'p229-p230', 'p238-p239', 'p247-p248', 'p255-p256', 'p263-p264', 'p271-p272', 'p279-p280', 'p287-p288', 'p299-p300', 'p307-p308', 'p317-p318', 'p335-p336', 'p351-p360',
    'p231-p232', 'p240-p241', 'p249-p250', 'p257-p258', 'p265-p266', 'p273-p274', 'p281-p282', 'p292-p293', 'p301-p302', 'p310-p311', 'p323-p326', 'p339-p340', 'p361-p362'
]

class TemplateSettingExp:
    def __init__(self, setting, experiment, dataset_dir, config_dir):
        """
        Represents an experimental setting for the transcoder project. Some attributes are
        directly from doce settings, others are created in the initialisation, and depend
        on the values of the doce settings.
        
        NB: Doce setting attributes have the particularity of being "objects" types, this lead to a lot of bugs,
        especially when trying to store their values into yaml files. In this class, correct
        data types are reattributed to every doce setting attribute. 

        Args:
            setting: A doce setting object.
            experiment: The doce experiment object.
            project_data_path: Path to the project's data directory. This path must contain the data created with create_mel_tho_dataset.
            force_cpu: Whether to force CPU usage instead of GPU (default: False).
        """
        self.setting_doce = setting
        self.setting_identifier = setting.identifier()

        self.step = self._normalize(getattr(setting, 'step', None))
        self.model = self._normalize(getattr(setting, 'model', None))
        self.plan = self._normalize(getattr(setting, 'plan', None))
        self.dataset = self._normalize(getattr(setting, 'dataset', None))
        self.speaker = self._normalize(getattr(setting, 'speaker', None))
        self.qc = self._normalize(getattr(setting, 'qc', None))
        self.rr = self._normalize(getattr(setting, 'rr', None))
        self.nr = self._normalize(getattr(setting, 'nr', None))
        self.source_vad = self._normalize(getattr(setting, 'svad', None))
        self.concealer_vad = self._normalize(getattr(setting, 'cvad', None))
        self.vad_thresh = self._normalize(getattr(setting, 'vad_thresh', None))
        self.max_len_strat = self._normalize(getattr(setting, 'max_len_strat', None))
        self.feature_extractor = self._normalize(getattr(setting, 'fe', None))
        decision_win_ms = self._normalize(getattr(setting, 'dw', None))  # doce factor is in ms, like "delay"
        self.decision_win = decision_win_ms / 1000 if decision_win_ms is not None else None  # seconds

        if self.step == "generate":
            self.generate_setting_identifier = self.setting_identifier
        else:
            self.generate_setting_identifier = self.setting_identifier.replace(self.step, "generate")

        self.output_audio_dir = experiment.path.audio + "/" + self.generate_setting_identifier
        self.output_difflaeq_dir = experiment.path.difflaeq + "/" + self.generate_setting_identifier
        self.duration_dir = experiment.path.duration 
        self.wer_dir = experiment.path.wer 
        self.dnsmos_dir = experiment.path.dnsmos
        self.emergence_dir = experiment.path.emergence 
        self.shortlist_dir = experiment.path.shortlist
        self.config_dir = experiment.path.config

        self.audio_extensions = ('.mp3', '.wav', '.flac')

        self.eval_dataset_dir = f"{dataset_dir}/{self.dataset}/evaluation/pan_0/{self.speaker}/"
        self.audio_files = []
        for root, dirs, files in os.walk(self.eval_dataset_dir):
            for file in files:
                if file.lower().endswith(self.audio_extensions):
                    self.audio_files.append(os.path.join(root, file))

        # self.calib_dataset_dir = f"{DATASET_DIR}/{self.dataset}/calibration/pan_0/{self.speaker}/"
        self.transc_dataset_dir = os.path.join(dataset_dir, "transc")
        self.replace_existing = False

        if self.model == "oracle":
            self.output_audio_dir = self.eval_dataset_dir
        elif self.model == "whitenoise":
            pass
        else:
            self.streamconfig, self.concealerconfig, self.sourcevadconfig, self.concealervadconfig, self.forecasterconfig = create_config_files(
                config_template_dir=config_dir,
                config_dir=os.path.join(experiment.path.config, self.setting_identifier),
                model_name=self.model,
                source_vad=self.source_vad,
                concealer_vad=self.concealer_vad,
                vad_thresh=self.vad_thresh,
                nr=self.nr,
                qc=self.qc,
                rr=self.rr,
                max_len_strat=self.max_len_strat,
                feature_extractor=self.feature_extractor or "logmel",
                decision_win=self.decision_win,
            )

        self.save_concealer = True
        self.save_original = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.delay = _lookup_vad_delay(self.source_vad)

    def _normalize(self, value):
        if value is None:
            return None
        if value == "none":
            return None
        if hasattr(value, "item"):  # numpy scalar
            return value.item()
        return value