import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import doce
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so granspeechmask_paper is importable without installing it

from granspeechmask_paper.experiments import monospeaker
from granspeechmask_paper.experiments.monospeaker import EXP_DIR
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
plt.rcParams["font.size"] = 30
plt.rcParams["axes.labelsize"] = 33
plt.rcParams["axes.titlesize"] = 33
plt.rcParams["xtick.labelsize"] = 29
plt.rcParams["ytick.labelsize"] = 29
plt.rcParams["legend.fontsize"] = 29
plt.rcParams["legend.title_fontsize"] = 30


def savefig_all(path_png, **kwargs):
    """Save the current figure as both PNG and PDF."""
    plt.savefig(path_png, **kwargs)
    plt.savefig(os.path.splitext(path_png)[0] + ".pdf", **kwargs)


def plot_capped_point(ax, x, y, cap, arrow_length=0.8, label_dx=0.015, **scatter_kwargs):
    """
    Scatter a single point, clipping it to `cap` on the y-axis when it
    exceeds it. A clipped point is drawn at y=cap with an upward arrow
    starting from the point and a text label showing its true value.
    """
    if y <= cap:
        ax.scatter(x, y, **scatter_kwargs)
        return

    ax.scatter(x, cap, **scatter_kwargs)
    ax.annotate(
        "",
        xy=(x, cap + arrow_length),
        xytext=(x, cap),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        zorder=6,
    )
    ax.text(
        x + label_dx,
        cap + arrow_length,
        f"{y:.1f}",
        fontsize=plt.rcParams["font.size"],
        va="center",
        ha="left",
    )


def style_discrete_grid(ax):
    """Dashed grey grid with the top and right spines removed."""
    ax.grid(True, which="major", linestyle="--", color="grey", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plot_path = "./plots/vad/"
os.makedirs(plot_path, exist_ok=True)

# =========================
# GLOBAL STORAGE FOR SCATTER
# =========================
global_points = []

def parse_header(header_str):
    return dict(
        item.split("=") for item in header_str.replace(" ", "").split(",")
    )

def process_settings_data(selector, metric_names, metric_paths):

    data = None
    header_index = None  # header string -> row index in `data`
    ref_metric_name = None

    for metric_name, metric_path in zip(metric_names, metric_paths):

        (metric_s, metric_h, _) = monospeaker.experiment.get_output(
            output=metric_name,
            selector=selector,
            path=metric_path,
            plan=selector['plan']
        )

        print(
            f"[plan={selector['plan']} dataset={selector.get('dataset')}] "
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

    # drop rows that ended up missing one of the requested metrics
    # (e.g. an entry present in "wer" but never matched in "emergence")
    complete_data = []
    for row in data:
        missing_metrics = [m for m in metric_names if m not in row]
        if missing_metrics:
            print(f"  WARNING: dropping entry {row} — missing metrics {missing_metrics}")
            continue
        complete_data.append(row)

    return complete_data


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
speakers_list_pruned = speakers_list[:10]  # for testing, use only the first 10 speakers

for dataset in datasets:

    selector_simplelist = {
        "plan": "simplelist",
        "model": "simplelist",
        "svad": ["webrtc", "silero", "ten"],
        "cvad": ["webrtc", "silero", "ten"],
        # "svad": ["webrtc", "silero"],
        # "cvad": ["webrtc", "silero"],
        "nr": "metricgan",
        "qc": "none",
        "rr": "none",
        "dataset": dataset,
        "speaker": speakers_list_pruned,
        "step": "evaluate",
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

    # -------------------------
    # Paired t-tests
    # -------------------------
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

    savefig_all(f'{plot_path}/results_table_{dataset}.png', dpi=150, bbox_inches='tight')
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

# =========================
# GLOBAL SCATTER PLOT (ALL DATASETS)
# =========================

df_global = pd.DataFrame(global_points)

# Baseline models (oracle/whitenoise) are excluded entirely from these plots
# and legends — only "ours" (simplelist) is shown.
if 'model' in df_global.columns:
    df_global = df_global[df_global['model'] == 'simplelist'].reset_index(drop=True)
    df_global['model'] = df_global['model'].replace('simplelist', 'ours')

size_map = {
    "ebr-low-asr-low": 40,
    "ebr-mid-asr-mid": 120,
    "ebr-high-asr-high": 250,
}

# Fixed color+marker per VAD so they stay consistent across every figure,
# instead of being reassigned dynamically based on what happens to be present.
VAD_STYLE = {
    "webrtc": {"color": plt.cm.tab10(0), "marker": "o"},
    "silero": {"color": plt.cm.tab10(1), "marker": "s"},
    "ten": {"color": plt.cm.tab10(2), "marker": "^"},
}


def get_point_style(row):
    """Marker/color for a scatter point, coded by VAD."""
    style = VAD_STYLE.get(row["vad"], {"color": "gray", "marker": "o"})
    return style["marker"], style["color"]

# -------------------------
# VAD encoding (colors)
# -------------------------
def get_vad_label(row):
    vad_values = []
    for k in row.keys():
        # check if the last 3 characters of the key are 'vad'
        if isinstance(k, str) and k.endswith("vad"):
            v = row[k]
            if isinstance(v, float) and np.isnan(v):
                continue
            if v is None:
                continue
            vad_values.append(str(v))
    if len(vad_values) == 0:
        return "N/A"
    if len(set(vad_values)) == 1:
        return vad_values[0]
    return "-".join(vad_values)

def get_point_size(thresh):

    if pd.isna(thresh):
        return 120

    return 10 + 250 * float(thresh)

df_global["vad"] = df_global.apply(get_vad_label, axis=1)
# only known VAD types get a legend entry
unique_vads = [v for v in VAD_STYLE if v in set(df_global["vad"])]

# =========================
# ONE FIGURE PER DATASET
# =========================

# -------------------------
# GROUPING (ONLY INSIDE DATASET LOOP)
# -------------------------
group_cols = [
    "model",
    "svad",
    "cvad",
    "vad_thresh",
    "vad"
]

df_global["vad_thresh"] = pd.to_numeric(
    df_global["vad_thresh"],
    errors="coerce"
)

# Remap WebRTC thresholds
webrtc_mask = (
    (df_global["svad"] == "webrtc") &
    (df_global["cvad"] == "webrtc")
)

webrtc_thresh_map = {
    0.1: 0.1,
    0.2: 0.3,
    0.3: 0.6,
    0.4: 0.9,
}

df_global.loc[webrtc_mask, "vad_thresh"] = (
    df_global.loc[webrtc_mask, "vad_thresh"]
    .astype(float)
    .replace(webrtc_thresh_map)
)

for dataset in df_global["dataset"].unique():

    df_d = df_global[df_global["dataset"] == dataset].copy()

    print(df_d)

    # average over speakers/runs etc. BUT KEEP dataset fixed
    df_d = (
        df_d
        .groupby(group_cols, dropna=False, as_index=False)
        .agg({
            "wer": "mean",
            "emergence": "mean"
        })
    )

    # Pareto transform
    df_d["x"] = 1 - df_d["wer"]
    df_d["y"] = df_d["emergence"]

    plt.figure(figsize=(10, 7))

    # -------------------------
    # SCATTER
    # -------------------------
    for _, row in df_d.iterrows():

        marker, color = get_point_style(row)
        plt.scatter(
            row["x"],
            row["y"],
            s=get_point_size(row["vad_thresh"]),
            marker=marker,
            color=color,
            alpha=0.75
        )

    # -------------------------
    # PARETO FRONT
    # -------------------------
    sorted_df = df_d.sort_values(by="x", ascending=False)

    pareto_x = []
    pareto_y = []

    best_y = np.inf

    for _, row in sorted_df.iterrows():
        if row["y"] < best_y:
            pareto_x.append(row["x"])
            pareto_y.append(row["y"])
            best_y = row["y"]

    plt.plot(
        pareto_x,
        pareto_y,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Pareto front"
    )

    # -------------------------
    # LEGENDS
    # -------------------------

    ax = plt.gca()
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()

    # VAD legend (placed first/topmost)
    vad_handles = [
        plt.Line2D(
            [0], [0],
            marker=VAD_STYLE[v]["marker"],
            color="w",
            label=v,
            markerfacecolor=VAD_STYLE[v]["color"],
            markersize=16
        )
        for v in unique_vads
    ]

    legend_vad = ax.legend(
        handles=vad_handles,
        title="VAD",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.3
    )
    ax.add_artist(legend_vad)

    fig.canvas.draw()
    bbox_vad = legend_vad.get_window_extent(renderer).transformed(ax.transAxes.inverted())

    # Threshold / aggressiveness legend
    thresh_to_agr = {0.1: 0, 0.3: 1, 0.6: 2, 0.9: 3}
    size_handles = [
        plt.scatter(
            [],
            [],
            s=get_point_size(t),
            color="gray",
            alpha=0.75,
            label=f"{t:.1f} / {agr}"
        )
        for t, agr in thresh_to_agr.items()
    ]

    legend_thresh = ax.legend(
        handles=size_handles,
        title="VAD thr. / agr.",
        loc="upper left",
        bbox_to_anchor=(1.02, bbox_vad.y0 - 0.015),
        borderaxespad=0.,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.3
    )
    ax.add_artist(legend_thresh)

    # -------------------------
    # AXES
    # -------------------------
    plt.xlabel("Intelligibility (1-WER)")
    plt.ylabel("Emergence (LMSD)")
    plt.xlim(0, 1)

    plt.grid(True)

    out_path = f"{plot_path}/scatter_pareto_{dataset}.png"
    savefig_all(
        out_path,
        dpi=150,
        bbox_inches='tight',
        bbox_extra_artists=(legend_vad, legend_thresh)
    )
    plt.close()

    print(f"Saved {out_path}")

# =========================
# OVERALL (AVERAGED OVER DATASETS)
# =========================

df_overall = df_global.copy()

# same grouping as per dataset plots
group_cols = [
    "model",
    "svad",
    "cvad",
    "vad_thresh",
    "vad"
]

df_overall = (
    df_overall
    .groupby(group_cols, dropna=False, as_index=False)
    .agg({
        "wer": "mean",
        "emergence": "mean"
    })
)

print(df_overall)

# Pareto transform
df_overall["x"] = 1 - df_overall["wer"]
df_overall["y"] = df_overall["emergence"]

fig = plt.figure(figsize=(10, 7))
ax = plt.gca()

OVERALL_Y_CAP = 11.0

# -------------------------
# SCATTER
# -------------------------
for _, row in df_overall.iterrows():

    marker, color = get_point_style(row)
    plot_capped_point(
        ax,
        row["x"],
        row["y"],
        cap=OVERALL_Y_CAP,
        s=get_point_size(row["vad_thresh"]),
        marker=marker,
        color=color,
        alpha=0.75
    )

# -------------------------
# PARETO FRONTIER
# -------------------------
sorted_df = df_overall.sort_values(by="x", ascending=False)

pareto_x = []
pareto_y = []

best_y = np.inf

for _, row in sorted_df.iterrows():
    if row["y"] < best_y:
        pareto_x.append(row["x"])
        pareto_y.append(row["y"])
        best_y = row["y"]

plt.plot(
    pareto_x,
    np.minimum(pareto_y, OVERALL_Y_CAP),
    linestyle="--",
    color="black",
    linewidth=2,
    label="Pareto front"
)

# -------------------------
# LEGENDS
# -------------------------

ax = plt.gca()
fig = plt.gcf()
renderer = fig.canvas.get_renderer()

# VAD legend (placed first/topmost)
vad_handles = [
    plt.Line2D(
        [0], [0],
        marker=VAD_STYLE[v]["marker"],
        color="w",
        label=v,
        markerfacecolor=VAD_STYLE[v]["color"],
        markersize=16
    )
    for v in unique_vads
]

legend_vad = ax.legend(
    handles=vad_handles,
    title="VAD",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.,
    frameon=False,
    handletextpad=0.4,
    labelspacing=0.3
)
ax.add_artist(legend_vad)

fig.canvas.draw()
bbox_vad = legend_vad.get_window_extent(renderer).transformed(ax.transAxes.inverted())

# Threshold / aggressiveness legend
thresh_to_agr = {0.1: 0, 0.3: 1, 0.6: 2, 0.9: 3}
size_handles = [
    plt.scatter(
        [],
        [],
        s=get_point_size(t),
        color="gray",
        alpha=0.75,
        label=f"{t:.1f} / {agr}"
    )
    for t, agr in thresh_to_agr.items()
]

legend_thresh = ax.legend(
    handles=size_handles,
    title="VAD thr. / agr.",
    loc="upper left",
    bbox_to_anchor=(1.02, bbox_vad.y0 - 0.015),
    borderaxespad=0.,
    frameon=False,
    handletextpad=0.4,
    labelspacing=0.3
)
ax.add_artist(legend_thresh)

# -------------------------
# AXES
# -------------------------
plt.xlabel("Intelligibility (1-WER)")
plt.ylabel("Emergence (LMSD)")
# plt.xlim(0, 1)
# ax.set_ylim(top=OVERALL_Y_CAP + 1.5)
style_discrete_grid(ax)

out_path = f"{plot_path}/vad_scatter_pareto.png"
savefig_all(
    out_path,
    dpi=150,
    bbox_inches='tight',
    bbox_extra_artists=(legend_vad, legend_thresh)
)
plt.close()

print(f"Saved {out_path}")