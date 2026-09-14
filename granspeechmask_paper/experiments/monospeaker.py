import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so granspeechmask_paper is importable without installing it

import time
import torch
import doce

from granspeechmask_paper.stream import run_file_mode_via_callback_on_folder, run_whitenoise_file_mode_via_callback_on_folder
from granspeechmask_paper.evaluation.voice_metrics import run_voice_metrics
from granspeechmask_paper.evaluation.emergence_metrics import run_emergence_metrics
from granspeechmask_paper.experiments.common import TemplateSettingExp, speakers_list
from granspeechmask_paper.experiments.paths import resolve_experiment_paths

torch.manual_seed(71)

if torch.cuda.is_available():
    # Set the random seed for GPU (if available)
    torch.cuda.manual_seed(0)

exp_name = "SpeechConcealerExp_20260529"

# define the experiment
experiment = doce.Experiment(
  name = exp_name,
  purpose = 'experiment for making speech concealers',
  author = 'Modan Tailleur',
  address = 'modan.tailleur@ls2n.fr',
)
# newer doce versions default this to '->', but all metrics in this project
# are saved with an underscore between the setting identifier and the metric
# name (see evaluation/voice_metrics.py / emergence_metrics.py), so
# get_output() must be told to look for the same separator, otherwise it
# matches 0 files.
experiment.metric_delimiter = '_'

#########################
#########################
# EXPERIMENT PATH AND DATASET PATH

# NOTE: no Jean Zay entry yet - add one here (exp_dir, dataset_dir) if/when
# running this on that cluster.
HOST_PATHS = {
    'pc-ls2n-lagrange': ('../speechConcealerExp/', '../SOS-1SP/'),
    'serveur-ls2n-cpu1': ('../speechConcealerExp/', '../SOS-1SP/'),
    'serveur-ls2n-gpu3': ('../speechConcealerExp/', '../SOS-1SP/'),
    # for ssd from pc-lagrange
    'po-tailleur-840g8': ('/media/user/MT-SSD-3/0-PROJETS_INFO/Post-doc/speechConcealerExp/', '/media/user/EXTbackup/speech_dataset/SOS-1SP/'),
}
DEFAULT_PATHS = (
    '../SpeechConcealer-data/',
    '/media/user/MT-SSD-3/0-PROJETS_INFO/Post-doc/datasets/SOS-1SP/',
)

EXP_DIR, DATASET_DIR = resolve_experiment_paths(HOST_PATHS, DEFAULT_PATHS)

#########################
######################################
# CONFIG PATH

CONFIG_DIR = './configs/'

experiment.set_path('audio', EXP_DIR+"/"+experiment.name+'/audio', force=True)
experiment.set_path('difflaeq', EXP_DIR+"/"+experiment.name+'/difflaeq/', force=True)
experiment.set_path('duration', EXP_DIR+"/"+experiment.name+'/duration/', force=True)
experiment.set_path('wer', EXP_DIR+"/"+experiment.name+'/wer/', force=True)
experiment.set_path('dnsmos', EXP_DIR+"/"+experiment.name+'/dnsmos/', force=True)
experiment.set_path('emergence', EXP_DIR+"/"+experiment.name+'/emergence/', force=True)
experiment.set_path('config', EXP_DIR+"/"+experiment.name+'/config/', force=True)
experiment.set_path('embs', EXP_DIR+"/"+experiment.name+'/embs/', force=True)
experiment.set_path('shortlist', EXP_DIR+"/"+experiment.name+'/shortlist/', force=True)

experiment.add_plan('simplelist',
  plan = ["simplelist"],
  model = ["simplelist"],
  svad = ["webrtc", "silero", "ten", "none"],
  cvad = ["webrtc", "silero", "ten", "none"],
  vad_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  nr = ["none", "metricgan"],
  qc = ["none", "dnsmosovr"],
  rr = ["none", "half"],
  fe = ["logmel", "mfcc"],
  dw = [50, 300], # in ms
  dataset = ["ebr-low-asr-low", "ebr-mid-asr-mid", "ebr-high-asr-high"],
  speaker = speakers_list,
  step = ['generate', 'evaluate'],
)

experiment.add_plan('baseline',
  plan = ["baseline"],
  model = ["oracle", "whitenoise"],
  dataset = ["ebr-low-asr-low", "ebr-mid-asr-mid", "ebr-high-asr-high"],
  speaker = speakers_list,
  step = ['generate', 'evaluate'],
)

class SettingExp(TemplateSettingExp):
    def __init__(self, setting, experiment, dataset_dir, config_dir):
        super().__init__(setting, experiment, dataset_dir, config_dir)
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

        num_calib_files = 2
        self.speakers_list = speakers_list
        self.calib_audio_files = self.audio_files[:num_calib_files] 
        self.eval_audio_files = self.audio_files[num_calib_files:]

def step(setting, experiment):
    
    print('XXXXXXXX ONGOING SETTING XXXXXXXX')
    print(setting.identifier())
    start_time = time.time()

    setting_exp = SettingExp(setting=setting, experiment=experiment, dataset_dir=DATASET_DIR, config_dir=CONFIG_DIR)
    
    if setting_exp.step == "generate":
        if setting_exp.model not in ["oracle", "oracle3db", "whitenoise"]:
            run_file_mode_via_callback_on_folder(setting_exp)
        elif setting_exp.model == "whitenoise":
            run_whitenoise_file_mode_via_callback_on_folder(setting_exp)

    elif setting_exp.step == "evaluate":
        run_voice_metrics(setting_exp)
        run_emergence_metrics(setting_exp)

    end_time = time.time()
    duration = end_time - start_time
    
    print(setting_exp.setting_identifier)
    print(f"Duration of the step: {duration:.2f} seconds")

# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment, func=step)