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

print(EXP_DIR)

plot_path = "./plots/ablation/"
os.makedirs(plot_path, exist_ok=True)

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

    # sanity check: does the selector actually restrict the returned entries
    # to the requested values, or does e.g. vad_thresh=0.8 also let 0.5 through?
    for key, expected in selector.items():
        if key in ("plan", "model", "speaker", "step", "dataset"):
            continue
        expected_values = {str(v) for v in (expected if isinstance(expected, list) else [expected])}
        actual_values = {str(row[key]) for row in complete_data if key in row}
        unexpected = actual_values - expected_values
        if unexpected:
            print(
                f"  !! SELECTOR LEAK: requested {key}={expected!r} but returned "
                f"entries also contain {key} in {sorted(unexpected)} "
                f"(selector filtering is not restricting this key as expected)."
            )

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
speakers_list_pruned = speakers_list[10:20]  # speakers 10-19 (a 10-speaker slice)

for dataset in datasets:

    selector_full = {
        "plan": "simplelist",
        "model": "simplelist",
        "svad": "ten",
        "cvad": "ten",
        "vad_thresh": 0.5,
        "nr": "metricgan",
        "qc": "none",
        "rr": "none",
        "dataset": dataset,
        "speaker": speakers_list_pruned,
        "step": "evaluate"
    }

    selector_abvad = selector_full.copy()
    selector_abvad.update({"svad": "none", "cvad": "none"})

    selector_abnr = selector_full.copy()
    selector_abnr.update({"nr": "none"})

    # selector_abqc = selector_full.copy()
    # selector_abqc.update({"qc": "none"})

    # selector_abrr = selector_full.copy()
    # selector_abrr.update({"rr": "none"})

    selector_baseline = {
        "plan": "baseline",
        "model": ["whitenoise", "oracle"],
        "dataset": dataset,
        "speaker": speakers_list_pruned,
        "step": "evaluate"
    }

    data_full = process_settings_data(selector_full, metric_names, metric_paths)
    data_abvad = process_settings_data(selector_abvad, metric_names, metric_paths)
    data_abnr = process_settings_data(selector_abnr, metric_names, metric_paths)
    # data_abqc = process_settings_data(selector_abqc, metric_names, metric_paths)
    # data_abrr = process_settings_data(selector_abrr, metric_names, metric_paths)
    data_baseline = process_settings_data(selector_baseline, metric_names, metric_paths)

    # -------------------------
    # annotations
    # -------------------------
    for d in data_full:
        d["vad"] = True; d["nr"] = True; d["qc"] = True; d["rr"] = True
    for d in data_abvad:
        d["vad"] = False; d["nr"] = True; d["qc"] = True; d["rr"] = True
    for d in data_abnr:
        d["vad"] = True; d["nr"] = False; d["qc"] = True; d["rr"] = True
    # for d in data_abqc:
    #     d["vad"] = True; d["nr"] = True; d["qc"] = False; d["rr"] = True
    # for d in data_abrr:
    #     d["vad"] = True; d["nr"] = True; d["qc"] = True; d["rr"] = False

    # data = data_full + data_abvad + data_abnr + data_abqc + data_abrr + data_baseline
    data = data_full + data_abvad + data_abnr + data_baseline

    df = pd.DataFrame(data)

    if "speaker" in df.columns:
        df = df.drop(columns=["speaker"])

    list_cols = [col for col in df.columns if is_list_column(col)]
    non_list_cols = [col for col in df.columns if col not in list_cols]

    df = (
        df.groupby(non_list_cols, dropna=False)
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
            intelligibility = 1 - wer
            base_info["intelligibility_avg"] = np.mean(intelligibility)
            base_info["intelligibility_ci95"] = 1.96 * np.std(intelligibility) / np.sqrt(len(intelligibility))
            base_info["intelligibility_values"] = intelligibility

        rows.append(base_info)

    df_stats = pd.DataFrame(rows)

    # =========================================================
    # 🔥 RESTORED BOLD STATISTICS (CRITICAL FIX)
    # =========================================================

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

    # intelligibility bold (min is best)
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

    # emergence bold (min is best)
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

    # =========================================================
    # PNG TABLE (WITH BOLD RESTORED)
    # =========================================================

    def format_metric(row, metric):
        s = f"{row[f'{metric}_avg']:.2f}±{row[f'{metric}_ci95']:.2f}"
        if row.get(f"{metric}_bold", False):
            s = r"$\bf{" + s + "}$"
        return s

    for metric in ["emergence", "intelligibility"]:
        df_stats[metric] = df_stats.apply(lambda r: format_metric(r, metric), axis=1)

    df_display = df_stats[config_cols + ["emergence", "intelligibility"]]
    df_display = df_display.sort_values(by="model")

    fig, ax = plt.subplots(figsize=(12, 6))
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

    # =========================================================
    # LATEX TABLE EXPORT (WITH BOLD RESTORED)
    # =========================================================

    def ablation_symbol(x):
        if pd.isna(x):
            return "-"
        return r"$\checkmark$" if x else r"$\times$"

    def format_latex_metric(row, metric):
        s = f"{row[f'{metric}_avg']:.2f}±{row[f'{metric}_ci95']:.2f}"
        if row.get(f"{metric}_bold", False):
            s = r"\bf{" + s + "}"
        return s

    df_latex = df_stats.copy()

    for col in ["vad", "nr", "qc", "rr"]:
        if col in df_latex.columns:
            df_latex[col] = df_latex[col].apply(ablation_symbol)

    for metric in ["emergence", "intelligibility"]:
        df_latex[metric] = df_latex.apply(lambda r: format_latex_metric(r, metric), axis=1)

    df_latex = df_latex[config_cols + ["emergence", "intelligibility"]]
    df_latex = df_latex.drop(columns=["plan"], errors="ignore")

    df_oracle = df_latex[df_latex["model"] == "oracle"]
    df_noise = df_latex[df_latex["model"] == "whitenoise"]
    df_ablation = df_latex[~df_latex["model"].isin(["oracle", "whitenoise"])]

    midrule = pd.DataFrame([{col: "" for col in df_latex.columns}])
    midrule.iloc[0, 0] = r"\midrule"

    df_final = pd.concat([df_oracle, df_noise, midrule, df_ablation], ignore_index=True)

    latex_table = df_final.to_latex(
        index=False,
        escape=False,
        column_format="l" * len(df_final.columns)
    )

    with open(f'{plot_path}/results_table_{dataset}.tex', "w") as f:
        f.write(latex_table)

    print(f"Saved table for {dataset}")

# =========================================================
# OVERALL TABLE (pools all datasets together)
# =========================================================

selector_full = {
    "plan": "simplelist",
    "model": "simplelist",
    "svad": "ten",
    "cvad": "ten",
    "vad_thresh": 0.5,
    "nr": "metricgan",
    "qc": "none",
    "rr": "none",
    "dataset": datasets,
    "speaker": speakers_list_pruned,
    "step": "evaluate"
}

selector_abvad = selector_full.copy()
selector_abvad.update({"svad": "none", "cvad": "none"})

selector_abnr = selector_full.copy()
selector_abnr.update({"nr": "none"})

# qc/rr are off in selector_full, so their ablation arms turn each one ON
# (rather than "removing" it from an already-off baseline)
selector_abqc = selector_full.copy()
selector_abqc.update({"qc": "dnsmosovr"})

selector_abrr = selector_full.copy()
selector_abrr.update({"rr": "half"})

selector_baseline = {
    "plan": "baseline",
    "model": ["whitenoise", "oracle"],
    "dataset": datasets,
    "speaker": speakers_list_pruned,
    "step": "evaluate"
}

data_full = process_settings_data(selector_full, metric_names, metric_paths)
data_abvad = process_settings_data(selector_abvad, metric_names, metric_paths)
data_abnr = process_settings_data(selector_abnr, metric_names, metric_paths)
data_abqc = process_settings_data(selector_abqc, metric_names, metric_paths)
data_abrr = process_settings_data(selector_abrr, metric_names, metric_paths)
data_baseline = process_settings_data(selector_baseline, metric_names, metric_paths)

# -------------------------
# annotations
# -------------------------
for d in data_full:
    d["vad"] = True; d["nr"] = True; d["qc"] = False; d["rr"] = False
for d in data_abvad:
    d["vad"] = False; d["nr"] = True; d["qc"] = False; d["rr"] = False
for d in data_abnr:
    d["vad"] = True; d["nr"] = False; d["qc"] = False; d["rr"] = False
for d in data_abqc:
    d["vad"] = True; d["nr"] = True; d["qc"] = True; d["rr"] = False
for d in data_abrr:
    d["vad"] = True; d["nr"] = True; d["qc"] = False; d["rr"] = True

data = data_full + data_abvad + data_abnr + data_abqc + data_abrr + data_baseline

df = pd.DataFrame(data)

if "speaker" in df.columns:
    df = df.drop(columns=["speaker"])

# pool across datasets: drop the "dataset" column itself so entries that only
# differ by dataset get merged together in the groupby below
if "dataset" in df.columns:
    df = df.drop(columns=["dataset"])

list_cols = [col for col in df.columns if is_list_column(col)]
non_list_cols = [col for col in df.columns if col not in list_cols]

df = (
    df.groupby(non_list_cols, dropna=False)
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
        intelligibility = 1 - wer
        base_info["intelligibility_avg"] = np.mean(intelligibility)
        base_info["intelligibility_ci95"] = 1.96 * np.std(intelligibility) / np.sqrt(len(intelligibility))
        base_info["intelligibility_values"] = intelligibility

    rows.append(base_info)

df_stats = pd.DataFrame(rows)

df_eval = df_stats[df_stats["model"] != "oracle"].copy()

# intelligibility bold (min is best)
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

# emergence bold (min is best)
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

for metric in ["emergence", "intelligibility"]:
    df_stats[metric] = df_stats.apply(lambda r: format_metric(r, metric), axis=1)

df_display = df_stats[config_cols + ["emergence", "intelligibility"]]
df_display = df_display.sort_values(by="model")

fig, ax = plt.subplots(figsize=(12, 6))
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

plt.savefig(f'{plot_path}/results_table_overall.png', dpi=150, bbox_inches='tight')
plt.close()

df_latex = df_stats.copy()

for col in ["vad", "nr", "qc", "rr"]:
    if col in df_latex.columns:
        df_latex[col] = df_latex[col].apply(ablation_symbol)

for metric in ["emergence", "intelligibility"]:
    df_latex[metric] = df_latex.apply(lambda r: format_latex_metric(r, metric), axis=1)

df_latex = df_latex[config_cols + ["emergence", "intelligibility"]]
df_latex = df_latex.drop(columns=["plan"], errors="ignore")

df_oracle = df_latex[df_latex["model"] == "oracle"]
df_noise = df_latex[df_latex["model"] == "whitenoise"]
df_ablation = df_latex[~df_latex["model"].isin(["oracle", "whitenoise"])]

midrule = pd.DataFrame([{col: "" for col in df_latex.columns}])
midrule.iloc[0, 0] = r"\midrule"

df_final = pd.concat([df_oracle, df_noise, midrule, df_ablation], ignore_index=True)

latex_table = df_final.to_latex(
    index=False,
    escape=False,
    column_format="l" * len(df_final.columns)
)

with open(f'{plot_path}/results_table_overall.tex', "w") as f:
    f.write(latex_table)

print("Saved overall table (pooled across all datasets)")

# =========================================================
# SIMPLELIST-ONLY TABLE, BROKEN DOWN BY DATASET
# =========================================================

dataset_rows = []

for dataset in datasets:

    selector_full = {
        "plan": "simplelist",
        "model": "simplelist",
        "svad": "ten",
        "cvad": "ten",
        "vad_thresh": 0.5,
        "nr": "metricgan",
        "qc": "none",
        "rr": "none",
        "dataset": dataset,
        "speaker": speakers_list_pruned,
        "step": "evaluate"
    }

    data_full = process_settings_data(selector_full, metric_names, metric_paths)

    wer = concat_lists([d["whisperlarge_wer"] for d in data_full])
    emergence = concat_lists([d["emergence"] for d in data_full])
    intelligibility = 1 - wer

    dataset_rows.append({
        "dataset": dataset,
        "intelligibility_avg": np.mean(intelligibility),
        "intelligibility_ci95": 1.96 * np.std(intelligibility) / np.sqrt(len(intelligibility)),
        "emergence_avg": np.mean(emergence),
        "emergence_ci95": 1.96 * np.std(emergence) / np.sqrt(len(emergence)),
    })

df_by_dataset = pd.DataFrame(dataset_rows)

for metric in ["intelligibility", "emergence"]:
    df_by_dataset[metric] = df_by_dataset.apply(
        lambda r, metric=metric: f"{r[f'{metric}_avg']:.2f}±{r[f'{metric}_ci95']:.2f}", axis=1
    )

df_by_dataset_display = df_by_dataset[["dataset", "intelligibility", "emergence"]]

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=df_by_dataset_display.values,
    colLabels=df_by_dataset_display.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

plt.savefig(f'{plot_path}/results_table_by_dataset.png', dpi=150, bbox_inches='tight')
plt.close()

latex_table = df_by_dataset_display.to_latex(
    index=False,
    escape=False,
    column_format="l" * len(df_by_dataset_display.columns)
)

with open(f'{plot_path}/results_table_by_dataset.tex', "w") as f:
    f.write(latex_table)

print("Saved SimpleList-by-dataset table")