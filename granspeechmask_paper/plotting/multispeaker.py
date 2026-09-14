import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import doce
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so granspeechmask_paper is importable without installing it

from granspeechmask_paper.experiments import multispeaker
from granspeechmask_paper.experiments.multispeaker import EXP_DIR
from granspeechmask_paper.experiments.common import multispeakers_list

"""
This file allows the plot of training curve. Please uncomment the row of
the model you want to plot and modify the parameters as you want.
Note that the plan must be specified explicitely (hybridts, cnn or ts).
"""

print(EXP_DIR)

plot_path = "./plots/multipseaker/"
os.makedirs(plot_path, exist_ok=True)

# =========================
# GLOBAL STORAGE FOR SCATTER
# =========================
global_points = []

def parse_header(header_str):
    return dict(
        item.split("=") for item in header_str.replace(" ", "").split(",")
    )

def process_settings_data(selector, metric_names, metric_paths, optional_metrics=("accuracy",)):

    data = None
    header_index = None  # header string -> row index in `data`
    ref_metric_name = None

    for metric_name, metric_path in zip(metric_names, metric_paths):

        (metric_s, metric_h, _) = multispeaker.experiment.get_output(
            output=metric_name,
            selector=selector,
            path=metric_path,
            plan=selector['plan']
        )

        print(
            f"[plan={selector['plan']} model={selector.get('model')} dataset={selector.get('dataset')}] "
            f"metric={metric_name!r} path={metric_path!r}: "
            f"{len(metric_h)} headers / {len(metric_s)} values"
        )

        if data is None:
            data = [
                {
                    "plan": selector['plan'],
                    "model": selector.get('model', selector['plan']),
                    **parse_header(h)
                }
                for h in metric_h
            ]
            header_index = {h: i for i, h in enumerate(metric_h)}
            ref_metric_name = metric_name

        else:
            ref_set = set(header_index)
            cur_set = set(metric_h)

            missing_here = ref_set - cur_set
            extra_here = cur_set - ref_set

            if missing_here or extra_here:
                print(
                    f"  WARNING: {metric_name!r} does not line up with "
                    f"{ref_metric_name!r} ({len(cur_set)} vs {len(ref_set)} entries)."
                )
                if missing_here:
                    print(f"  -> present in {ref_metric_name!r} but MISSING in {metric_name!r} (will be dropped):")
                    for h in sorted(missing_here):
                        print(f"       {h}")
                if extra_here:
                    print(f"  -> present in {metric_name!r} but has no match in {ref_metric_name!r} (ignored):")
                    for h in sorted(extra_here):
                        print(f"       {h}")

        # match on header string rather than position, so extra/unmatched
        # entries for this metric are simply ignored instead of crashing.
        for h, s in zip(metric_h, metric_s):
            idx = header_index.get(h)
            if idx is None:
                continue
            data[idx][metric_name] = s

    # drop rows that ended up missing one of the requested REQUIRED metrics
    # (e.g. an entry present in "wer" but never matched in "emergence").
    # optional metrics (e.g. "accuracy" for models that don't have it, such
    # as the "oracle"/"whitenoise" baselines) are kept with an empty array
    # placeholder instead, so they still show up in the table as "nan".
    complete_data = []
    for row in data:
        missing_metrics = [m for m in metric_names if m not in row]
        required_missing = [m for m in missing_metrics if m not in optional_metrics]
        if required_missing:
            print(f"  WARNING: dropping entry {row} — missing required metrics {required_missing}")
            continue
        for m in missing_metrics:
            row[m] = np.array([])
        complete_data.append(row)

    return complete_data


def is_list_column(df, col):
    return df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any()

def concat_lists(series):
    arrays = []
    for item in series:
        if isinstance(item, np.ndarray):
            arrays.append(item.ravel())
        elif isinstance(item, list):
            arrays.append(np.array(item).ravel())
        else:
            arrays.append(np.array([item]))
    return np.concatenate(arrays)

# MT: old version
# metric_names = ["w2v2_wer", "spt_wer", "sbcr_wer", "whisper_wer", "emergence", "accuracy"]
# metric_paths = ["wer", "wer", "wer", "wer", "emergence", "accuracy"]

# MT: new version with only whisper large v3
metric_names = ["whisperlarge_wer", "emergence", "accuracy"]
metric_paths = ["wer", "emergence", "accuracy"]


datasets = ["ebr-low-asr-low", "ebr-mid-asr-mid", "ebr-high-asr-high"]
speakers_list_pruned = multispeaker.multispeakers_list[:10]  # first 10 speaker couples (test-sized slice)


def load_speaker_genders():
    """Maps speaker ID (e.g. "p225") to gender ("M"/"F"), read from speaker-info.txt."""
    speaker_info_path = os.path.join(multispeaker.DATASET_DIR, "speaker-info.txt")
    genders = {}
    with open(speaker_info_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip the "ID AGE GENDER ACCENTS REGION COMMENTS" header
        parts = line.split()
        if len(parts) < 3:
            continue
        speaker_id, _age, gender = parts[0], parts[1], parts[2]
        genders[speaker_id] = gender
    return genders


SPEAKER_GENDERS = load_speaker_genders()


def couple_gender_type(couple):
    """"pXXX-pYYY" -> "mixed" (one M, one F), "same" (both same gender), or "unknown"."""
    ids = couple.split("-")
    if len(ids) != 2:
        return "unknown"
    g1 = SPEAKER_GENDERS.get(ids[0])
    g2 = SPEAKER_GENDERS.get(ids[1])
    if g1 is None or g2 is None:
        return "unknown"
    return "mixed" if g1 != g2 else "same"


mixed_gender_speakers = [c for c in speakers_list_pruned if couple_gender_type(c) == "mixed"]
same_gender_speakers = [c for c in speakers_list_pruned if couple_gender_type(c) == "same"]


def safe_ttest_rel(a, b, metric_label, row, config_cols):
    try:
        return stats.ttest_rel(a, b)
    except ValueError as e:
        config_str = ", ".join(f"{c}={row[c]}" for c in config_cols)
        print(
            f"  WARNING: paired t-test failed for metric {metric_label!r} "
            f"({config_str}) — len(best)={len(a)} vs len(row)={len(b)} ({e}). "
            f"Falling back to nan (entry left un-bolded)."
        )
        return np.nan, np.nan


def build_and_save_table(dataset_value, out_name, collect_scatter=False, speakers=None):
    pooled = isinstance(dataset_value, list)
    speakers = speakers_list_pruned if speakers is None else speakers

    selector_full = {
        "plan": "simplelist",
        "model": "simplelist",
        "svad": "ten",
        "cvad": "ten",
        "nr": "metricgan",
        "qc": "none",
        "rr": "none",
        "dataset": dataset_value,
        "step": "evaluate",
        "speaker": speakers
    }

    selector_abvad = {
        "plan": "simplelist",
        "model": "simplelist",
        "svad": "none",
        "cvad": "none",
        "nr": "metricgan",
        "qc": "none",
        "rr": "none",
        "dataset": dataset_value,
        "step": "evaluate",
        "speaker": speakers
    }

    selector_abnr = {
        "plan": "simplelist",
        "model": "simplelist",
        "svad": "ten",
        "cvad": "ten",
        "nr": "none",
        "qc": "none",
        "rr": "none",
        "dataset": dataset_value,
        "step": "evaluate",
        "speaker": speakers
    }

    # qc/rr ablation is not wired up: doing so would require qc="dnsmosovr"
    # and rr="half" data, which hasn't been generated for the multispeaker
    # experiment (only qc="none"/rr="none" exists). See plot_output_ablation.py
    # for the same pattern, disabled for the same reason.
    # selector_abqc = selector_full.copy()
    # selector_abqc.update({"qc": "none"})
    #
    # selector_abrr = selector_full.copy()
    # selector_abrr.update({"rr": "none"})

    selector_baseline = {
        "plan": "baseline",
        "model": ["whitenoise", "oracle"],
        "dataset": dataset_value,
        "step": "evaluate",
        "speaker": speakers
    }

    data_full = process_settings_data(selector_full, metric_names, metric_paths)
    data_abvad = process_settings_data(selector_abvad, metric_names, metric_paths)
    data_abnr = process_settings_data(selector_abnr, metric_names, metric_paths)
    data_baseline = process_settings_data(selector_baseline, metric_names, metric_paths)

    # ------------------------
    # annotate treatments
    # ------------------------
    # qc/rr are always "none" in every selector above (see comment there), so
    # they're labeled False everywhere: the generated data never actually
    # turns quality control or repetition removal on.
    for d in data_full:
        d["vad"] = True
        d["nr"] = True
        d["qc"] = False
        d["rr"] = False

    for d in data_abvad:
        d["vad"] = False
        d["nr"] = True
        d["qc"] = False
        d["rr"] = False

    for d in data_abnr:
        d["vad"] = True
        d["nr"] = False
        d["qc"] = False
        d["rr"] = False

    data = data_full + data_abvad + data_abnr + data_baseline

    df = pd.DataFrame(data)

    if "speaker" in df.columns:
        df = df.drop(columns=["speaker"])

    # pool across datasets: drop the "dataset" column itself so entries that
    # only differ by dataset get merged together in the groupby below
    if pooled and "dataset" in df.columns:
        df = df.drop(columns=["dataset"])

    list_cols = [col for col in df.columns if is_list_column(df, col)]
    non_list_cols = [col for col in df.columns if col not in list_cols]

    df = (
        df
        .groupby(non_list_cols, dropna=False)
        .agg({col: concat_lists for col in list_cols})
        .reset_index()
    )

    metric_cols = [col for col in df.columns if col in metric_names]
    config_cols = [col for col in df.columns if col not in metric_cols]

    rows = []

    for _, row in df.iterrows():
        base_info = {k: row[k] for k in config_cols}

        wer_arrays = []
        accuracy_arrays = []

        for metric in metric_cols:
            values = row[metric]

            if len(values) == 0:
                mean = np.nan
                ci95 = np.nan
            else:
                mean = np.mean(values)
                ci95 = 1.96 * np.std(values) / np.sqrt(len(values))

            base_info[f"{metric}_avg"] = mean
            base_info[f"{metric}_ci95"] = ci95

            if "wer" in metric:
                wer_arrays.append(values)

            if "accuracy" in metric:
                accuracy_arrays.append(values)

            if metric == "emergence":
                base_info["emergence_values"] = values

        wer = np.concatenate(wer_arrays) if len(wer_arrays) > 0 else np.array([])
        if len(wer) > 0:
            base_info["wer_avg"] = np.mean(wer)
            base_info["wer_ci95"] = 1.96 * np.std(wer) / np.sqrt(len(wer))
            intelligibility = 1 - wer
            base_info["intelligibility_avg"] = np.mean(intelligibility)
            base_info["intelligibility_ci95"] = 1.96 * np.std(intelligibility) / np.sqrt(len(intelligibility))
        else:
            base_info["wer_avg"] = np.nan
            base_info["wer_ci95"] = np.nan
            intelligibility = np.array([])
            base_info["intelligibility_avg"] = np.nan
            base_info["intelligibility_ci95"] = np.nan
        base_info["wer_values"] = wer
        base_info["intelligibility_values"] = intelligibility

        accuracy = np.concatenate(accuracy_arrays) if len(accuracy_arrays) > 0 else np.array([])
        if len(accuracy) > 0:
            base_info["accuracy_avg"] = np.mean(accuracy)
            base_info["accuracy_ci95"] = 1.96 * np.std(accuracy) / np.sqrt(len(accuracy))
        else:
            base_info["accuracy_avg"] = np.nan
            base_info["accuracy_ci95"] = np.nan
        base_info["accuracy_values"] = accuracy

        rows.append(base_info)

    df_stats = pd.DataFrame(rows)

    # =========================
    # COLLECT GLOBAL SCATTER DATA
    # =========================
    if collect_scatter:
        for _, row in df_stats.iterrows():
            global_points.append({
                "model": row["model"],
                "dataset": dataset_value,
                "wer": row["wer_avg"],
                "emergence": row["emergence_avg"],
                "accuracy": row["accuracy_avg"],
                **{k: row[k] for k in config_cols if "vad" in k}
            })

    # -------------------------
    # Paired t-tests
    # -------------------------
    df_eval = df_stats[df_stats["model"] != "oracle"].copy()

    best_intel_idx = df_eval["intelligibility_avg"].idxmin()
    best_intel_values = df_eval.loc[best_intel_idx, "intelligibility_values"]
    df_stats["intelligibility_bold"] = False

    for idx, row in df_eval.iterrows():
        if idx == best_intel_idx:
            df_stats.loc[idx, "intelligibility_bold"] = True
            continue

        t_stat, p_val = safe_ttest_rel(best_intel_values, row["intelligibility_values"], "intelligibility", row, config_cols)
        if p_val > 0.05:
            df_stats.loc[idx, "intelligibility_bold"] = True

    df_stats["accuracy_bold"] = False
    valid_accuracy = df_eval["accuracy_avg"].notna()

    if valid_accuracy.any():
        best_accuracy_idx = df_eval.loc[valid_accuracy, "accuracy_avg"].idxmax()
        best_accuracy_values = df_eval.loc[best_accuracy_idx, "accuracy_values"]

        for idx, row in df_eval.iterrows():
            if idx == best_accuracy_idx:
                df_stats.loc[idx, "accuracy_bold"] = True
                continue

            if len(row["accuracy_values"]) == 0:
                continue

            t_stat, p_val = safe_ttest_rel(best_accuracy_values, row["accuracy_values"], "accuracy", row, config_cols)
            if p_val > 0.05:
                df_stats.loc[idx, "accuracy_bold"] = True

    best_em_idx = df_eval["emergence_avg"].idxmin()
    best_em_values = df_eval.loc[best_em_idx, "emergence_values"]
    df_stats["emergence_bold"] = False

    for idx, row in df_eval.iterrows():
        if idx == best_em_idx:
            df_stats.loc[idx, "emergence_bold"] = True
            continue
        t_stat, p_val = safe_ttest_rel(best_em_values, row["emergence_values"], "emergence", row, config_cols)
        if p_val > 0.05:
            df_stats.loc[idx, "emergence_bold"] = True

    # -------------------------
    # TABLE
    # -------------------------
    metrics_to_display = ["emergence", "intelligibility", "accuracy"]

    for metric in metrics_to_display:
        def format_metric(row, metric=metric):
            avg = row[f"{metric}_avg"]
            if pd.isna(avg):
                return "nan"
            s = f"{avg:.2f}±{row[f'{metric}_ci95']:.2f}"
            if row[f"{metric}_bold"]:
                s = f"$\\bf{{{s}}}$"
            return s
        df_stats[metric] = df_stats.apply(format_metric, axis=1)

    final_columns = config_cols + metrics_to_display
    df_display = df_stats[final_columns]

    df_display = df_display.sort_values(by="plan")
    df_display = df_display.drop(columns=["plan"])

    fig, ax = plt.subplots(
        figsize=(max(12, len(final_columns)*1.5), 0.5*len(df_display)+2)
    )

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.savefig(f'{plot_path}/results_table_{out_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -------------------------
    # LATEX TABLE EXPORT
    # -------------------------
    def format_latex_metric(row, metric):
        avg = row[f"{metric}_avg"]
        if pd.isna(avg):
            return "nan"
        s = f"{avg:.2f}±{row[f'{metric}_ci95']:.2f}"
        if row.get(f"{metric}_bold", False):
            s = r"\bf{" + s + "}"
        return s

    df_latex = df_stats.copy()

    for metric in metrics_to_display:
        df_latex[metric] = df_latex.apply(lambda r, metric=metric: format_latex_metric(r, metric), axis=1)

    df_latex = df_latex[final_columns]
    df_latex = df_latex.sort_values(by="plan")
    df_latex = df_latex.drop(columns=["plan"])

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format="l" * len(df_latex.columns)
    )

    with open(f'{plot_path}/results_table_{out_name}.tex', "w") as f:
        f.write(latex_table)

    print(f"Saved table for {out_name}")


for dataset in datasets:
    build_and_save_table(dataset, dataset, collect_scatter=True)

# overall table, pooling all datasets together
build_and_save_table(datasets, "multispeaker_overall", collect_scatter=False)

# mixed-gender couples (1 male + 1 female) vs same-gender couples, pooled
# across all datasets, to check whether mixed-gender pairs get better accuracy
build_and_save_table(datasets, "overall_mixedgender", collect_scatter=False, speakers=mixed_gender_speakers)
build_and_save_table(datasets, "overall_samegender", collect_scatter=False, speakers=same_gender_speakers)

# =========================
# GLOBAL SCATTER PLOT (ALL DATASETS)
# =========================

df_global = pd.DataFrame(global_points)

size_map = {
    "ebr-low-asr-low": 40,
    "ebr-mid-asr-mid": 120,
    "ebr-high-asr-high": 250,
}

markers = {
    "oracle": "o",
    "whitenoise": "s",
    "simplelist": "^"
}

# -------------------------
# VAD encoding (colors)
# -------------------------
def get_vad_label(row):
    vad_values = []
    for k in row.keys():
        if "vad" in k:
            v = row[k]
            if isinstance(v, float) and np.isnan(v):
                continue
            if v is None:
                continue
            vad_values.append(str(v))
    return "-".join(vad_values) if len(vad_values) > 0 else "none"

df_global["vad"] = df_global.apply(get_vad_label, axis=1)

unique_vads = sorted(df_global["vad"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vads)))
vad_color_map = {v: c for v, c in zip(unique_vads, colors)}

# -------------------------
# PARETO TRANSFORM
# -------------------------
df_global["x"] = 1 - df_global["wer"]
df_global["y"] = df_global["emergence"]

plt.figure(figsize=(10, 7))

# -------------------------
# SCATTER PLOT
# -------------------------
for _, row in df_global.iterrows():

    x = row["x"]
    y = row["y"]
    model = row["model"]
    dataset = row["dataset"]
    vad = row["vad"]

    plt.scatter(
        x,
        y,
        s=size_map.get(dataset, 80),
        marker=markers.get(model, "o"),
        color=vad_color_map[vad],
        alpha=0.75
    )

# -------------------------
# PARETO FRONTIER
# -------------------------
sorted_df = df_global.sort_values(by="x", ascending=False)

pareto_x = []
pareto_y = []

best_y = np.inf

for _, row in sorted_df.iterrows():
    if row["y"] < best_y:
        pareto_x.append(row["x"])
        pareto_y.append(row["y"])
        best_y = row["y"]

plt.plot(pareto_x, pareto_y, linestyle="--", color="black", linewidth=2, label="Pareto front")

# -------------------------
# LEGENDS
# -------------------------

# Models
model_handles = [
    plt.Line2D(
        [0], [0],
        marker=markers[m],
        color="w",
        label=m,
        markerfacecolor="black",
        markersize=8
    )
    for m in markers
]

# place dataset legend in the upper-right but slightly below the VAD legend
legend1 = plt.legend(
    handles=model_handles,
    title="Models",
    loc="upper right",
    bbox_to_anchor=(1, 0.80)
)

plt.gca().add_artist(legend1)

# Dataset sizes
dataset_handles = [
    plt.scatter([], [], s=size_map[d], color="gray", alpha=0.5, label=d)
    for d in size_map
]

# place dataset legend in the upper-right but slightly below the VAD legend
legend2 = plt.legend(
    handles=dataset_handles,
    title="Dataset size",
    loc="upper right",
    bbox_to_anchor=(1, 0.62)
)

plt.gca().add_artist(legend2)

# VAD colors
vad_handles = [
    plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        label=v,
        markerfacecolor=vad_color_map[v],
        markersize=8
    )
    for v in unique_vads
]

plt.legend(handles=vad_handles, title="VAD config", loc="upper right")

# -------------------------
# AXES
# -------------------------
plt.xlabel("Intelligibility (1 - WER/100) ↓")
plt.ylabel("Emergence ↓")
plt.title("Intelligibility vs Emergence")
plt.grid(True)
plt.tight_layout()

plt.savefig(f'{plot_path}/scatter_pareto_ALL.png', dpi=150)
plt.close()

print("Saved scatter_pareto_ALL.png")