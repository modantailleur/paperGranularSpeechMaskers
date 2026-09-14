import os
import yaml

def create_config_files(config_template_dir, config_dir, model_name,
                  source_vad="ten", concealer_vad="ten", vad_thresh=0.5, nr=None, qc=None, rr=None, max_len_strat="queue", feature_extractor="logmel", decision_win=None):

    #######################
    # Stream config
    ######################
    stream_config = {
        "conc_multiplier": 1.0,
        "monitor_gain": 0.0,
        "record_mic_gain": 1.0,
        "max_level": 1.0,
        "sr": 48000,
        "channels_in": 1,
        "channels_out": 2,
        "device_in": "default",   # 15 when my computer is used alone - 15 default, ALSA (64 in, 64 out); 4 when used with dock - 4 sof-hda-dsp: - (hw:0,6), ALSA (4 in, 0 out)
        "device_out": "default",  # 15 when my computer is used alone - 15 default, ALSA (64 in, 64 out); 17 when used with dock - 17 default, ALSA (64 in, 64 out)
        "dtype": "float32",
        "buffer_duration": 0.050,
    }

    # Decision window: how much audio (in seconds) the concealer looks at to
    # decide which candidate matches the current buffer. Defaults to exactly
    # one buffer (current behavior); anything larger adds that much trailing
    # context from before the buffer. The buffer itself (buffer_duration
    # above, i.e. how much audio streams in per callback) is unaffected.
    if decision_win is None:
        decision_win = stream_config["buffer_duration"]

    #######################
    # Concealer config
    ######################
    if model_name == "simplelist":
        concealer_config = {
            "concealer_type": "granspeechmask_paper",
            "concealer_vad_config": f"default_concealer.yaml",
            "memory_maxlen": 100,
            "max_countdown_reuse": 10,  # in number of uses
            "max_concealer_distance_to_buffer": 5,  # in number of concealers
            "concealer_duration": 0.300,
            "concealing_min_timeout_ratio": 0.60,
            "concealing_max_timeout_ratio": 1.0,
            "fade_duration": 0.050,
            "decision_win": decision_win,  # seconds
        }

    if max_len_strat is not None:
        concealer_config["max_len_strat"] = max_len_strat
    else:
        concealer_config["max_len_strat"] = "queue"

    if nr is not None:
        concealer_config["denoise"] = True
    else:
        concealer_config["denoise"] = False

    if qc is not None:
        concealer_config["quality_threshold"] = 2.0  # DNSMOS score threshold (None disables quality check)
    else:
        concealer_config["quality_threshold"] = None

    if rr is not None:
        concealer_config["random_reverse"] = True
    else:
        concealer_config["random_reverse"] = False

    #######################
    # VAD config
    ######################
    source_vad_name = "none" if source_vad is None else source_vad
    concealer_vad_name = "none" if concealer_vad is None else concealer_vad

    if source_vad_name == "webrtc":
        source_vad_config = {
            "vad_type": "webrtc",
            #MT: just because webrtc doesn't have the same logic as the other VADs, we need to adapt its 
            # parameters. There are 4 levels of aggressivness, so we trigger them with vad_thresh values of 0.1, 0.2, 0.3, 0.4 
            # (which correspond to webrtc_mode 0, 1, 2, 3).
            "aggressiveness": min(int(vad_thresh*10-1), 3),  # 0-3, more aggressive in filtering out non-speech
            "logit_threshold": 0.5,  # threshold for converting VAD probabilities to binary decisions
        }
    else:
        source_vad_config = {
            "vad_type": source_vad_name,
            "logit_threshold": vad_thresh,
        }

    if concealer_vad_name == "webrtc":
        concealer_vad_config = {
            "vad_type": "webrtc",
            #MT: just because webrtc doesn't have the same logic as the other VADs, we need to adapt its 
            # parameters. There are 4 levels of aggressivness, so we trigger them with vad_thresh values of 0.1, 0.2, 0.3, 0.4 
            # (which correspond to webrtc_mode 0, 1, 2, 3).
            "aggressiveness": min(int(vad_thresh*10-1), 3),  # 0-3, more aggressive in filtering out non-speech
            "logit_threshold": 0.5,  # threshold for converting VAD probabilities to binary decisions
        }
    else:
        concealer_vad_config = {
            "vad_type": concealer_vad_name,
            "logit_threshold": vad_thresh,
        }

    #######################
    # Forecaster config
    ######################
    forecaster_config = {
        "forecaster_type": "identity"
    }

    streamconfig = os.path.join(config_dir, "stream/default.yaml")
    concealerconfig = os.path.join(config_dir, "concealer/default.yaml")
    sourcevadconfig = os.path.join(config_dir, "vad/default_source.yaml")
    forecasterconfig = os.path.join(config_dir, "forecaster/default.yaml")
    concealervadconfig = os.path.join(config_dir, "vad/default_concealer.yaml")

    os.makedirs(os.path.dirname(streamconfig), exist_ok=True)
    os.makedirs(os.path.dirname(concealerconfig), exist_ok=True)
    os.makedirs(os.path.dirname(sourcevadconfig), exist_ok=True)
    os.makedirs(os.path.dirname(forecasterconfig), exist_ok=True)
    os.makedirs(os.path.dirname(concealervadconfig), exist_ok=True)

    # Save configs to experiment paths
    with open(streamconfig, 'w') as f:
        yaml.dump(stream_config, f)
    with open(concealerconfig, 'w') as f:
        yaml.dump(concealer_config, f)
    with open(sourcevadconfig, 'w') as f:
        yaml.dump(source_vad_config, f)
    with open(concealervadconfig, 'w') as f:
        yaml.dump(concealer_vad_config, f)
    with open(forecasterconfig, 'w') as f:
        yaml.dump(forecaster_config, f)

    return streamconfig, concealerconfig, sourcevadconfig, concealervadconfig, forecasterconfig