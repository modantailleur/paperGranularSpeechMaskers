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

"""
This file allows the plot of training curve. Please uncomment the row of
the model you want to plot and modify the parameters as you want.
Note that the plan must be specified explicitely (hybridts, cnn or ts).
"""

class OutputExtractor:
    def __init__(self, experiment, selector):
        self.experiment = experiment
        self.selector = selector

    def get_output(self, model_name, output_names, path_names):
        output = {}
        self.selector["model"] = model_name
        for output_name, path_name in zip(output_names, path_names):
            output_i, _, _ = self.experiment.get_output(
                output=output_name,
                selector=self.selector,
                path=path_name,
            )
            if len(output_i) != 1:
                raise ValueError(f"Expected only one output for {output_name}, but got {len(output_i)}")
            output[output_name] = output_i[0]
        return output

    def get_models_output(self, models_names, output_names, path_names):
        output = {}
        for model_name in models_names:
            output[model_name] = self.get_output(model_name, output_names, path_names)
        return output

class MetaOutputExtractor:
    def __init__(self, experiment, selector):
        self.experiment = experiment
        self.selector = selector

    def get_output(self, model_name, output_names, path_names):
        output = {}
        self.selector["model"] = model_name
        for output_name, path_name in zip(output_names, path_names):
            full_path_name = getattr(monospeaker.experiment.path, path_name) + "/" + doce.Setting(self.experiment.plan, ["generate", model_name], positional=False).identifier()
            outputs = []
            for file_name in os.listdir(full_path_name):
                if file_name.endswith(f'{output_name}.npy'):
                    file_path = os.path.join(full_path_name, file_name)
                    outputs.append(np.load(file_path, allow_pickle=True))
            output[output_name] = np.array(outputs)
        return output
    
    def get_models_output(self, models_names, output_names, path_names):
        output = {}
        for model_name in models_names:
            output[model_name] = self.get_output(model_name, output_names, path_names)
        return output


output_names = ["w2v2_wer", "dnsmosp808", "dnsmossig", "dnsmosbak", "dnsmosovr", "emergence"]
path_names = ["wer", "dnsmos", "dnsmos", "dnsmos", "dnsmos", "emergence"]
model_names = ["oracle", "whitenoise", "simplelistdenoised", "simplelistdenoisedqualitycheck"]

ext_oracle = OutputExtractor(monospeaker.experiment, {"step": "evaluate"})
output = ext_oracle.get_models_output(
    models_names=model_names,
    output_names=output_names,
    path_names=path_names,
)

output_names_meta = ["difflaeq"]
path_names_meta = ["difflaeq"]
ext_meta = MetaOutputExtractor(monospeaker.experiment, {"step": "generate"})
meta_model_names = ["whitenoise", "simplelistdenoised", "simplelistdenoisedqualitycheck"]
output_meta = ext_meta.get_models_output(
    models_names=meta_model_names,
    output_names=output_names_meta,
    path_names=path_names_meta,
)

for model_name in output.keys():
    print(model_name)
    if "oracle" not in model_name:
        output[model_name]["difflaeq"] = output_meta[model_name]["difflaeq"]
    if model_name == "oracle":
        output[model_name]["difflaeq"] = np.array([0])
    if model_name == "oracle3db":
        output[model_name]["difflaeq"] = np.array([3])
    output[model_name]["intelligibility"] = 1 - output[model_name]["w2v2_wer"]

output_names = ["intelligibility" if n == "w2v2_wer" else n for n in output_names] + ["difflaeq"]
output_sign = ['low', 'up', 'up', 'up', 'up', 'low', 'no']


for model_name in output.keys():
    output[model_name]["dnsmosp808_avg"] = np.mean(output[model_name]["dnsmosp808"])
    output[model_name]["dnsmosp808_ci95"] = 1.96 * np.std(output[model_name]["dnsmosp808"]) / np.sqrt(len(output[model_name]["dnsmosp808"]))
    output[model_name]["dnsmossig_avg"] = np.mean(output[model_name]["dnsmossig"])
    output[model_name]["dnsmossig_ci95"] = 1.96 * np.std(output[model_name]["dnsmossig"]) / np.sqrt(len(output[model_name]["dnsmossig"]))
    output[model_name]["dnsmosbak_avg"] = np.mean(output[model_name]["dnsmosbak"])
    output[model_name]["dnsmosbak_ci95"] = 1.96 * np.std(output[model_name]["dnsmosbak"]) / np.sqrt(len(output[model_name]["dnsmosbak"]))
    output[model_name]["dnsmosovr_avg"] = np.mean(output[model_name]["dnsmosovr"])
    output[model_name]["dnsmosovr_ci95"] = 1.96 * np.std(output[model_name]["dnsmosovr"]) / np.sqrt(len(output[model_name]["dnsmosovr"]))
    output[model_name]["intelligibility_avg"] = np.mean(output[model_name]["intelligibility"])
    output[model_name]["intelligibility_ci95"] = 1.96 * np.std(output[model_name]["intelligibility"]) / np.sqrt(len(output[model_name]["intelligibility"]))
    output[model_name]["difflaeq_avg"] = np.mean(output[model_name]["difflaeq"])
    output[model_name]["difflaeq_ci95"] = 1.96 * np.std(output[model_name]["difflaeq"]) / np.sqrt(len(output[model_name]["difflaeq"]))
    output[model_name]["emergence_avg"] = np.mean(output[model_name]["emergence"])
    output[model_name]["emergence_ci95"] = 1.96 * np.std(output[model_name]["emergence"]) / np.sqrt(len(output[model_name]["emergence"]))

for metric, sign in zip(output_names, output_sign):
    models = [m for m in output.keys() if "oracle" not in m]
    if sign == "no":
        for model in models:
            output[model][f"{metric}_significant"] = False
        continue
    if sign == "up":
        best_model = max(models, key=lambda m: output[m][f"{metric}_avg"])
    if sign == "low":
        best_model = min(models, key=lambda m: output[m][f"{metric}_avg"])

    print("DNSMOS Significance Tests:")
    print(f"Best model for {metric}: {best_model}\n")
    output[best_model][f"{metric}_significant"] = True  # Best model is significant by definition

    for model in models:
        if model != best_model:
            t_stat, p_val = stats.ttest_rel(output[best_model][metric], output[model][metric])
            print(f"{best_model} vs {model}: t={t_stat:.4f}, p={p_val:.4f}")
            output[model][f"{metric}_significant"] = p_val > 0.05

# Perform significance tests (t-tests) for each metric
# models = [m for m in output.keys() if m != "oracle"]
# best_model_dnsmos = max(models, key=lambda m: output[m]["dnsmosp808_avg"])
# best_model_wer = max(models, key=lambda m: output[m]["w2v2_wer_avg"])

# print("DNSMOS Significance Tests:")
# print(f"Best model: {best_model_dnsmos}\n")
# output[best_model_dnsmos]["dnsmosp808_significant"] = True  # Best model is significant by definition

# for model in models:
#     if model != best_model_dnsmos:
#         t_stat, p_val = stats.ttest_rel(output[best_model_dnsmos]["dnsmosp808"], output[model]["dnsmosp808"])
#         print(f"{best_model_dnsmos} vs {model}: t={t_stat:.4f}, p={p_val:.4f}")
#         output[model]["dnsmosp808_significant"] = p_val > 0.05

# print("\nW2V2 WER Significance Tests:")
# print(f"Best model: {best_model_wer}\n")
# output[best_model_wer]["w2v2_wer_significant"] = True  # Best model is significant by definition

# for model in models:
#     if model != best_model_wer:
#         t_stat, p_val = stats.ttest_rel(output[best_model_wer]["w2v2_wer"], output[model]["w2v2_wer"])
#         print(f"{best_model_wer} vs {model}: t={t_stat:.4f}, p={p_val:.4f}")
#         output[model]["w2v2_wer_significant"] = p_val > 0.05

df_dict = {f"{metric}_avg": [output[model][f"{metric}_avg"] for model in output.keys()] for metric in output_names}
df_dict.update({f"{metric}_ci95": [output[model][f"{metric}_ci95"] for model in output.keys()] for metric in output_names})
df = pd.DataFrame(df_dict, index=output.keys())

# df = pd.DataFrame({
#     "dnsmosp808_avg": [output[model]["dnsmosp808_avg"] for model in output.keys()],
#     "dnsmosp808_ci95": [output[model]["dnsmosp808_ci95"] for model in output.keys()],
#     "w2v2_wer_avg": [output[model]["w2v2_wer_avg"] for model in output.keys()],
#     "w2v2_wer_ci95": [output[model]["w2v2_wer_ci95"] for model in output.keys()]
# }, index=output.keys())

# Create a formatted table for visualization
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

model_names_mapping = {
    "oracle": "Oracle",
    "whitenoise": "White Noise",
    "simplelistdenoised": "Simple List (Denoised)",
    "simplelistdenoisedqualitycheck": "Simple List (Denoised + Quality Check)"
}

table_data = []
for model in df.index:
    metrics = []
    for metric in output_names:
        metric_str = f"{df.loc[model, f'{metric}_avg']:.2f}±{df.loc[model, f'{metric}_ci95']:.2f}"
        if output[model].get(f"{metric}_significant", False):
            metric_str = f"$\\bf{{{metric_str}}}$"
        metrics.append(metric_str)
    table_data.append([model_names_mapping.get(model, model)] + metrics)

    # dnsmos_str = f"{df.loc[model, 'dnsmosp808_avg']:.2f}±{df.loc[model, 'dnsmosp808_ci95']:.2f}"
    # wer_str = f"{df.loc[model, 'w2v2_wer_avg']:.2f}±{df.loc[model, 'w2v2_wer_ci95']:.2f}"
    
    # # Make significant metrics bold
    # if output[model].get("dnsmosp808_significant", False):
    #     dnsmos_str = f"$\\bf{{{dnsmos_str}}}$"
    # if output[model].get("w2v2_wer_significant", False):
    #     wer_str = f"$\\bf{{{wer_str}}}$"
    
    # table_data.append([model_names_mapping.get(model, model), dnsmos_str, wer_str])


# table_data = []
# for model in df.index:
#     dnsmos_str = f"{df.loc[model, 'dnsmosp808_avg']:.2f}±{df.loc[model, 'dnsmosp808_ci95']:.2f}"
#     wer_str = f"{df.loc[model, 'w2v2_wer_avg']:.2f}±{df.loc[model, 'w2v2_wer_ci95']:.2f}"
    
#     # Make significant metrics bold
#     if output[model].get("dnsmosp808_significant", False):
#         dnsmos_str = f"$\\bf{{{dnsmos_str}}}$"
#     if output[model].get("w2v2_wer_significant", False):
#         wer_str = f"$\\bf{{{wer_str}}}$"
    
#     table_data.append([model_names_mapping.get(model, model), dnsmos_str, wer_str])


colLabels = ['Model', 'Intelligibility ↓', 'DNSMOS p808 ↑', "DNSMOS Sig ↑", "DNSMOS Bak ↑", "DNSMOS Ovr ↑", "Emergence ↓", "Diff LAeq"]
table = ax.table(cellText=table_data, colLabels=colLabels, 
                cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 2)

plt.savefig('results_table.png', dpi=150, bbox_inches='tight')

# -------------------------
# LATEX TABLE EXPORT
# -------------------------
latex_table_data = []
for model in df.index:
    metrics = []
    for metric in output_names:
        metric_str = f"{df.loc[model, f'{metric}_avg']:.2f}±{df.loc[model, f'{metric}_ci95']:.2f}"
        if output[model].get(f"{metric}_significant", False):
            metric_str = r"\bf{" + metric_str + "}"
        metrics.append(metric_str)
    latex_table_data.append([model_names_mapping.get(model, model)] + metrics)

df_latex = pd.DataFrame(latex_table_data, columns=colLabels)

latex_table = df_latex.to_latex(
    index=False,
    escape=False,
    column_format="l" * len(colLabels)
)

with open('results_table.tex', "w") as f:
    f.write(latex_table)

plt.show()