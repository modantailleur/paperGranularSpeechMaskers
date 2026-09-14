import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import doce
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so granspeechmask_paper is importable without installing it

from granspeechmask_paper.experiments import delay
from granspeechmask_paper.experiments.delay import EXP_DIR
from granspeechmask_paper.experiments.common import speakers_list

"""
This file allows the plot of training curve. Please uncomment the row of
the model you want to plot and modify the parameters as you want.
Note that the plan must be specified explicitely (hybridts, cnn or ts).
"""

print(EXP_DIR)

plt.rcParams["font.family"] = "serif"
# fall back to metric-compatible serif fonts if Times New Roman isn't
# installed on this machine (e.g. missing the msttcorefonts package)
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 21
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["legend.title_fontsize"] = 21

plot_path = "./plots/delay/"
os.makedirs(plot_path, exist_ok=True)

# =========================
# GLOBAL STORAGE FOR SCATTER
# =========================
global_points = []
delay_points = []
whitenoise_emergence_by_dataset = {}

def parse_header(header_str):
    return dict(
        item.split("=") for item in header_str.replace(" ", "").split(",")
    )

def process_settings_data(selector, metric_names, metric_paths):

    data = None

    for metric_name, metric_path in zip(metric_names, metric_paths):

        (metric_s, metric_h, _) = delay.experiment.get_output(
            output=metric_name,
            selector=selector,
            path=metric_path,
            plan=selector['plan']
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


        for i, s in enumerate(metric_s):
            data[i][metric_name] = s

    return data


def is_list_column(col):
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


metric_names = ["whisperlarge_wer", "emergence"]
metric_paths = ["wer", "emergence"]

datasets = ["ebr-low-asr-low", "ebr-mid-asr-mid", "ebr-high-asr-high"]
speakers_list_pruned = speakers_list[10:20]  # speakers 10-19 (a 10-speaker slice)

for dataset in datasets:

    selector_simplelist = {
        "plan": "simplelist",
        "model": "simplelist",
        "svad": ["ten"],
        "cvad": ["ten"],
        "vad_thresh": 0.5,
        "nr": "metricgan",
        "qc": "none",
        "rr": "none",
        "dataset": dataset,
        "speaker": speakers_list_pruned,
        "step": "evaluate"
    }

    selector_baseline = {
        "plan": "baseline",
        "model": ["whitenoise", "oracle"],
        "dataset": dataset,
        "speaker": speakers_list_pruned,
        "step": "evaluate"
    }

    data_simplelist = process_settings_data(selector_simplelist, metric_names, metric_paths)
    data_baseline = process_settings_data(selector_baseline, metric_names, metric_paths)

    data = data_simplelist + data_baseline

    df = pd.DataFrame(data)
    
    if "speaker" in df.columns:
        df = df.drop(columns=["speaker"])

    # if "plan" in df.columns:
    #     df = df.drop(columns=["plan"])

    list_cols = [col for col in df.columns if is_list_column(col)]
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

        for metric in metric_cols:
            values = row[metric]

            mean = np.mean(values)
            ci95 = 1.96 * np.std(values) / np.sqrt(len(values))

            base_info[f"{metric}_avg"] = mean
            base_info[f"{metric}_ci95"] = ci95

            if "wer" in metric:
                wer_arrays.append(values)

            if metric == "emergence":
                base_info["emergence_values"] = values

        if len(wer_arrays) > 0:
            wer = np.concatenate(wer_arrays)
            base_info["wer_avg"] = np.mean(wer)
            base_info["wer_ci95"] = 1.96 * np.std(wer) / np.sqrt(len(wer))
            base_info["wer_values"] = wer
            intelligibility = 1 - wer
            base_info["intelligibility_avg"] = np.mean(intelligibility)
            base_info["intelligibility_ci95"] = 1.96 * np.std(intelligibility) / np.sqrt(len(intelligibility))
            base_info["intelligibility_values"] = intelligibility

        rows.append(base_info)

    df_stats = pd.DataFrame(rows)

    # =========================
    # COLLECT GLOBAL SCATTER DATA
    # =========================
    for _, row in df_stats.iterrows():
        global_points.append({
            "model": row["model"],
            "dataset": dataset,
            "wer": row["wer_avg"],
            "emergence": row["emergence_avg"],
            **{k: row[k] for k in config_cols if "vad" in k}
        })
        if row["model"] == "whitenoise":
            whitenoise_emergence_by_dataset[dataset] = row["emergence_avg"]
        if "delay" in config_cols and pd.notna(row["delay"]):
            delay_points.append({
                "model": row["model"],
                "dataset": dataset,
                "wer": row["wer_avg"],
                "emergence": row["emergence_avg"],
                "delay": row["delay"],
            })

    # -------------------------
    # Paired t-tests (UNCHANGED)
    # -------------------------
    df_eval = df_stats[df_stats["model"] != "oracle"].copy()

    best_intel_idx = df_eval["intelligibility_avg"].idxmin()
    best_intel_values = df_eval.loc[best_intel_idx, "intelligibility_values"]
    df_stats["intelligibility_bold"] = False

    for idx, row in df_eval.iterrows():
        if idx == best_intel_idx:
            df_stats.loc[idx, "intelligibility_bold"] = True
            continue
        t_stat, p_val = stats.ttest_rel(best_intel_values, row["intelligibility_values"])
        if p_val > 0.05:
            df_stats.loc[idx, "intelligibility_bold"] = True

    best_em_idx = df_eval["emergence_avg"].idxmin()
    best_em_values = df_eval.loc[best_em_idx, "emergence_values"]
    df_stats["emergence_bold"] = False

    for idx, row in df_eval.iterrows():
        if idx == best_em_idx:
            df_stats.loc[idx, "emergence_bold"] = True
            continue
        t_stat, p_val = stats.ttest_rel(best_em_values, row["emergence_values"])
        if p_val > 0.05:
            df_stats.loc[idx, "emergence_bold"] = True

    # -------------------------
    # TABLE (UNCHANGED)
    # -------------------------
    metrics_to_display = ["emergence", "intelligibility"]

    for metric in metrics_to_display:
        def format_metric(row):
            s = f"{row[f'{metric}_avg']:.2f}±{row[f'{metric}_ci95']:.2f}"
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

    plt.savefig(f'{plot_path}/results_table_{dataset}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -------------------------
    # LATEX TABLE EXPORT
    # -------------------------
    def format_latex_metric(row, metric):
        s = f"{row[f'{metric}_avg']:.2f}±{row[f'{metric}_ci95']:.2f}"
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

    with open(f'{plot_path}/results_table_{dataset}.tex', "w") as f:
        f.write(latex_table)

    print(f"Saved table for {dataset}")

# =========================================================
# OVERALL TABLE (pools all datasets together)
# =========================================================

selector_simplelist_overall = {
    "plan": "simplelist",
    "model": "simplelist",
    "svad": ["ten"],
    "cvad": ["ten"],
    "vad_thresh": 0.5,
    "nr": "metricgan",
    "qc": "none",
    "rr": "none",
    "dataset": datasets,
    "speaker": speakers_list_pruned,
    "step": "evaluate"
}

selector_baseline_overall = {
    "plan": "baseline",
    "model": ["whitenoise", "oracle"],
    "dataset": datasets,
    "speaker": speakers_list_pruned,
    "step": "evaluate"
}

data_simplelist_overall = process_settings_data(selector_simplelist_overall, metric_names, metric_paths)
data_baseline_overall = process_settings_data(selector_baseline_overall, metric_names, metric_paths)

data_overall = data_simplelist_overall + data_baseline_overall

df_overall = pd.DataFrame(data_overall)

if "speaker" in df_overall.columns:
    df_overall = df_overall.drop(columns=["speaker"])

# pool across datasets: drop the "dataset" column itself so entries that only
# differ by dataset get merged together in the groupby below
if "dataset" in df_overall.columns:
    df_overall = df_overall.drop(columns=["dataset"])

df = df_overall
list_cols = [col for col in df.columns if is_list_column(col)]
non_list_cols = [col for col in df.columns if col not in list_cols]

df_overall = (
    df
    .groupby(non_list_cols, dropna=False)
    .agg({col: concat_lists for col in list_cols})
    .reset_index()
)

metric_cols = [col for col in df_overall.columns if col in metric_names]
config_cols = [col for col in df_overall.columns if col not in metric_cols]

rows = []

for _, row in df_overall.iterrows():
    base_info = {k: row[k] for k in config_cols}

    wer_arrays = []

    for metric in metric_cols:
        values = row[metric]

        mean = np.mean(values)
        ci95 = 1.96 * np.std(values) / np.sqrt(len(values))

        base_info[f"{metric}_avg"] = mean
        base_info[f"{metric}_ci95"] = ci95

        if "wer" in metric:
            wer_arrays.append(values)

        if metric == "emergence":
            base_info["emergence_values"] = values

    if len(wer_arrays) > 0:
        wer = np.concatenate(wer_arrays)
        base_info["wer_avg"] = np.mean(wer)
        base_info["wer_ci95"] = 1.96 * np.std(wer) / np.sqrt(len(wer))
        base_info["wer_values"] = wer
        intelligibility = 1 - wer
        base_info["intelligibility_avg"] = np.mean(intelligibility)
        base_info["intelligibility_ci95"] = 1.96 * np.std(intelligibility) / np.sqrt(len(intelligibility))
        base_info["intelligibility_values"] = intelligibility

    rows.append(base_info)

df_stats = pd.DataFrame(rows)

df_eval = df_stats[df_stats["model"] != "oracle"].copy()

best_intel_idx = df_eval["intelligibility_avg"].idxmin()
best_intel_values = df_eval.loc[best_intel_idx, "intelligibility_values"]
df_stats["intelligibility_bold"] = False

for idx, row in df_eval.iterrows():
    if idx == best_intel_idx:
        df_stats.loc[idx, "intelligibility_bold"] = True
        continue
    t_stat, p_val = stats.ttest_rel(best_intel_values, row["intelligibility_values"])
    if p_val > 0.05:
        df_stats.loc[idx, "intelligibility_bold"] = True

best_em_idx = df_eval["emergence_avg"].idxmin()
best_em_values = df_eval.loc[best_em_idx, "emergence_values"]
df_stats["emergence_bold"] = False

for idx, row in df_eval.iterrows():
    if idx == best_em_idx:
        df_stats.loc[idx, "emergence_bold"] = True
        continue
    t_stat, p_val = stats.ttest_rel(best_em_values, row["emergence_values"])
    if p_val > 0.05:
        df_stats.loc[idx, "emergence_bold"] = True

metrics_to_display = ["emergence", "intelligibility"]

for metric in metrics_to_display:
    def format_metric(row):
        s = f"{row[f'{metric}_avg']:.2f}±{row[f'{metric}_ci95']:.2f}"
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

plt.savefig(f'{plot_path}/results_table_delay_overall.png', dpi=150, bbox_inches='tight')
plt.close()

def format_latex_metric(row, metric):
    s = f"{row[f'{metric}_avg']:.2f}±{row[f'{metric}_ci95']:.2f}"
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

with open(f'{plot_path}/results_table_delay_overall.tex', "w") as f:
    f.write(latex_table)

print("Saved overall table (pooled across all datasets)")

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

model_display_names = {
    "oracle": "oracle",
    "whitenoise": "SSN",
    "simplelist": "simplelist"
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
        label=model_display_names.get(m, m),
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

# =========================
# INTELLIGIBILITY vs EMERGENCE, COLORED BY DELAY
# (one plot per dataset + one pooled over all datasets)
# =========================

df_delay = pd.DataFrame(delay_points)
df_delay["delay"] = df_delay["delay"].astype(float)
df_delay["x"] = 1 - df_delay["wer"]
df_delay["y"] = df_delay["emergence"]

unique_delays = sorted(df_delay["delay"].unique())
delay_norm = plt.Normalize(vmin=min(unique_delays), vmax=max(unique_delays))
delay_cmap = plt.cm.viridis

def plot_scatter_by_delay(df, out_path, size=None, whitenoise_emergence=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        point_size = size if size is not None else size_map.get(row["dataset"], 80)
        ax.scatter(
            row["x"],
            row["y"],
            s=point_size,
            marker=markers.get(row.get("model", "simplelist"), "o"),
            color=delay_cmap(delay_norm(row["delay"])),
            alpha=0.85
        )

    if whitenoise_emergence is not None:
        ax.axhline(
            whitenoise_emergence,
            color="red",
            linestyle="--",
            linewidth=2,
            label="SSN"
        )
        ax.legend(loc="upper right")

    sm = plt.cm.ScalarMappable(norm=delay_norm, cmap=delay_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Delay (ms)")

    ax.set_xlabel("Intelligibility ↓")
    ax.set_ylabel("Emergence ↓")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

for dataset in datasets:
    df_delay_dataset = df_delay[df_delay["dataset"] == dataset]
    if len(df_delay_dataset) == 0:
        continue
    plot_scatter_by_delay(
        df_delay_dataset,
        f'{plot_path}/scatter_delay_{dataset}.png',
        whitenoise_emergence=whitenoise_emergence_by_dataset.get(dataset)
    )

# "ALL" plot: one point per delay value, averaging wer/emergence across datasets
# (rather than showing all dataset x delay combinations as separate points)
df_delay_overall = df_delay.groupby("delay", as_index=False)[["wer", "emergence"]].mean()
df_delay_overall["x"] = 1 - df_delay_overall["wer"]
df_delay_overall["y"] = df_delay_overall["emergence"]

overall_whitenoise_emergence = (
    np.mean(list(whitenoise_emergence_by_dataset.values()))
    if whitenoise_emergence_by_dataset else None
)

plot_scatter_by_delay(
    df_delay_overall,
    f'{plot_path}/delay_scatter.png',
    size=150,
    whitenoise_emergence=overall_whitenoise_emergence,
)