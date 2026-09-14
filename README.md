# Low Latency Intelligibility Reduction of Distracting Conversations Using Granular Speech Maskers

Research code accompanying the following paper:

**TO CITE**

This repository builds on the [`amods`](https://pypi.org/project/amods/) package, which provides the general speech-concealer, VAD, and streaming infrastructure. It adds the research code specific to this paper in the `granspeechmask_paper` package.

Companion page is available at:
https://modantailleur.github.io/paperGranularSpeechMaskers/

## Setup

This is not a Python package — it's a plain research-code repository. Create a
**Python 3.9+** environment and install the pinned dependencies (including `amods`):

```bash
pip install -r requirements.txt
```

If `ten-vad` does not work correctly, it can be force-reinstalled with:

```bash
pip install -U --force-reinstall -v git+https://github.com/TEN-framework/ten-vad.git
```

If `doce` does not work correctly, reinstall the version used for the experiments with:

```bash
pip install git+https://github.com/mathieulagrange/doce.git@a04e56b85cae656f5c947fd3e2ec7fb5868c17ee
```

Download the **SOS-1SP** and **SOS-2SP** datasets from the following Zenodo repository:

https://zenodo.org/records/22687159

## Reproducing the Paper's Experiments

Each experiment is implemented as a [doce](https://github.com/mathieulagrange/doce) driver under `granspeechmask_paper/experiments/` and can be run using the corresponding script in `scripts/`. For example:

```bash
python3 scripts/run_ablation.py
```

Run the scripts from anywhere in the cloned repo. The experiment and dataset directories can be specified using the `EXP_DIR` and `DATASET_DIR` environment variables:

```bash
EXP_DIR=/data/speechConcealerExp \
DATASET_DIR=/data/SOS-1SP \
python3 scripts/run_ablation.py
```

`DATASET_DIR` should point to the appropriate dataset variant for each experiment (1SP or 2SP).

The table below maps the experiments reported in the paper to the scripts required to reproduce them.

| Paper section                                     | Run script                                                                                | Plot script                                     | Output                                                                               |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| §4 Selecting VAD model — mAP, Table 1             | `granspeechmask_paper/vad_benchmark/compute_vad_on_dataset.py`, then `.../compute_map.py` | — (prints AP directly)                          | mAP per VAD                                                                          |
| §4 Selecting VAD model — Fig. 2                   | `scripts/run_vad.py`                                                                      | `granspeechmask_paper/plotting/vad.py`          | `plots/vad/vad_scatter_pareto.{png,pdf}`                                             |
| §5 Experiments — ablation, Table 2                | `scripts/run_ablation.py`                                                                 | `granspeechmask_paper/plotting/ablation.py`     | `plots/ablation/results_table_overall.{png,tex}`                                     |
| §5 Experiments — EBR-SAR conditions               | `scripts/run_ablation.py`                                                                 | `granspeechmask_paper/plotting/ablation.py`     | `plots/ablation/results_table_by_dataset.png`                                        |
| §5 Experiments — influence of latency             | `scripts/run_delay.py`                                                                    | `granspeechmask_paper/plotting/delay.py`        | `plots/delay/delay_scatter.png` and `results_table_delay_overall.png`                |
| §5 Experiments — multispeaker robustness, Table 3 | `scripts/run_multispeaker.py`                                                             | `granspeechmask_paper/plotting/multispeaker.py` | `plots/multispeaker/results_table_multispeaker_overall.png` (including SPI accuracy) |
