from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score

# -----------------------------
# PATHS
# -----------------------------
H5_DIR = "../vadConcealerExp"
VAD_GT_DIR = "../SOS-1SP/vadgt"

h5_path = Path(H5_DIR)
vad_path = Path(VAD_GT_DIR)

# -----------------------------
# LOAD GROUND TRUTH
# -----------------------------
print("Loading ground truth...")
gt_cache = {}

for gt_file in sorted(vad_path.glob("*.npy")):
    gt_cache[gt_file.stem] = np.load(gt_file).reshape(-1)

gt_keys_sorted = sorted(gt_cache.keys())
print(f"Loaded {len(gt_cache)} GT files")

# -----------------------------
# STORAGE
# -----------------------------
result_dict = {}
global_scores = {}
global_labels = {}

# -----------------------------
# PARSE FILENAME
# -----------------------------
def parse_filename(filename):
    name = filename.replace(".h5", "")
    parts = name.split("_", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]

# -----------------------------
# NORMALIZE MODELS
# -----------------------------
def normalize_model(name):
    if "webrtc1" in name:
        return "webrtc", 1  # aggressiveness=1
    if "webrtc2" in name:
        return "webrtc", 2  # aggressiveness=2
    if "webrtc3" in name:
        return "webrtc", 3  # aggressiveness=3
    return name, None

# -----------------------------
# MAIN LOOP
# -----------------------------
for file in sorted(h5_path.glob("*.h5")):

    raw_model, dataset_name = parse_filename(file.name)
    if raw_model is None:
        continue

    model_name, aggressiveness = normalize_model(raw_model)  # aggressiveness currently unused, kept for future breakdowns

    print(f"Processing {file.name}")

    with h5py.File(file, "r") as f:
        probs_all = f["logits"]
        names_all = f["file_name"]

        # Decode and sort filenames
        names_decoded = [
            name.decode('utf-8') if isinstance(name, bytes) else name
            for name in names_all
        ]
        sort_indices = np.argsort(names_decoded)

        probs_all = np.asarray(probs_all)[sort_indices]
        names_all = [names_decoded[i] for i in sort_indices]

        result_dict.setdefault(dataset_name, {})
        result_dict[dataset_name].setdefault(model_name, [])

        # -------------------------
        # FLATTEN ALL SCORES
        # -------------------------
        all_scores = []
        all_labels = []

        for i, key in enumerate(gt_keys_sorted):
            if i >= probs_all.shape[0]:
                break

            scores = probs_all[i].reshape(-1)
            labels = gt_cache[key].reshape(-1)

            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # -------------------------
        # SKLEARN PR CURVE + AP
        # -------------------------
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = average_precision_score(all_labels, all_scores)

        result_dict[dataset_name][model_name].append(
            (0.5, pr_auc, precision, recall)
        )

        # -------------------------
        # GLOBAL STORAGE
        # -------------------------
        global_scores.setdefault(model_name, [])
        global_labels.setdefault(model_name, [])

        global_scores[model_name].extend(all_scores.tolist())
        global_labels[model_name].extend(all_labels.tolist())

# -----------------------------
# PLOTTING PER DATASET
# -----------------------------
for dataset_name, models in result_dict.items():

    plt.figure()

    for model_name, values in models.items():

        _, pr_auc, precision, recall = values[0]

        plt.plot(recall, precision, label=f"{model_name} (AP={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve - {dataset_name}")
    plt.legend()
    plt.grid()
    plt.show()

# -----------------------------
# GLOBAL PR-AUC (AP)
# -----------------------------
print("\n========== GLOBAL PR-AUC (Average Precision) ==========")

for model_name in global_scores:

    scores = np.array(global_scores[model_name])
    labels = np.array(global_labels[model_name])

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    print(f"{model_name}: AP = {pr_auc:.4f}")