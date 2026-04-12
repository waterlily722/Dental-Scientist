import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_NAME = "dental_pa"


def find_runs(base_dir: Path):
    runs = []
    for path in sorted(base_dir.glob("run_*")):
        if path.is_dir() and (path / "final_info.json").exists() and (path / "all_results.npy").exists():
            runs.append(path.name)
    return runs


def load_run(run_name: str):
    with open(Path(run_name) / "final_info.json", "r", encoding="utf-8") as handle:
        final_info = json.load(handle)
    results = np.load(Path(run_name) / "all_results.npy", allow_pickle=True).item()
    return final_info, results


def main():
    base_dir = Path.cwd()
    runs = find_runs(base_dir)
    if not runs:
        print("No run directories found.")
        return

    for run_name in runs:
        final_info, results = load_run(run_name)
        dataset_info = final_info.get(DATASET_NAME, {})
        metrics = dataset_info.get("means", {})

        train_key = f"{DATASET_NAME}_0_train_log_info"
        val_key = f"{DATASET_NAME}_0_val_log_info"
        train_logs = results.get(train_key, [])
        val_logs = results.get(val_key, [])

        if train_logs:
            epochs = [entry["epoch"] + 1 for entry in train_logs]
            train_loss = [entry["loss"] for entry in train_logs]
            train_auc = [entry["auc"] for entry in train_logs]
            plt.figure(figsize=(10, 4))
            plt.plot(epochs, train_loss, marker="o", label="Train loss")
            plt.plot(epochs, train_auc, marker="o", label="Train AUC")
            plt.title(f"{run_name} training curves")
            plt.xlabel("Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(base_dir / run_name / "train_curves.png", dpi=150)
            plt.close()

        if val_logs:
            epochs = [entry["epoch"] + 1 for entry in val_logs]
            val_loss = [entry["loss"] for entry in val_logs]
            val_auc = [entry["auc"] for entry in val_logs]
            val_f1 = [entry["f1"] for entry in val_logs]
            plt.figure(figsize=(10, 4))
            plt.plot(epochs, val_loss, marker="o", label="Val loss")
            plt.plot(epochs, val_auc, marker="o", label="Val AUC")
            plt.plot(epochs, val_f1, marker="o", label="Val F1")
            plt.title(f"{run_name} validation curves")
            plt.xlabel("Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(base_dir / run_name / "val_curves.png", dpi=150)
            plt.close()

        if metrics:
            names = [
                "best_test_accuracy",
                "best_test_auc",
                "best_test_f1",
                "best_test_sensitivity",
                "best_test_specificity",
                "best_test_ece",
            ]
            values = [metrics.get(name, 0.0) for name in names]
            plt.figure(figsize=(10, 4))
            plt.bar(names, values)
            plt.xticks(rotation=30, ha="right")
            plt.title(f"{run_name} test metrics")
            plt.tight_layout()
            plt.savefig(base_dir / run_name / "test_metrics.png", dpi=150)
            plt.close()

        print(f"Saved plots for {run_name}")


if __name__ == "__main__":
    main()