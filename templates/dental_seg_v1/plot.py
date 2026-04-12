import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["dice", "iou", "boundary_f1"]


def _load_run(run_dir: Path):
    final_info_path = run_dir / "final_info.json"
    all_results_path = run_dir / "all_results.npy"
    if not final_info_path.exists() or not all_results_path.exists():
        return None
    final_info = json.loads(final_info_path.read_text(encoding="utf-8"))
    dataset_name = next(iter(final_info.keys()))
    payload = final_info[dataset_name].get("means", {})
    all_results = np.load(all_results_path, allow_pickle=True).item()
    return dataset_name, payload, all_results


def _collect_runs(base_dir: Path):
    runs = []
    for run_dir in sorted(base_dir.glob("run_*")):
        loaded = _load_run(run_dir)
        if loaded is None:
            continue
        runs.append((run_dir.name, *loaded))
    return runs


def _plot_primary_curves(runs, out_dir: Path):
    plt.figure(figsize=(8, 5))
    for run_name, dataset_name, _payload, all_results in runs:
        train_key = f"{dataset_name}_0_train_log_info"
        val_key = f"{dataset_name}_0_val_log_info"
        train_log = all_results.get(train_key, [])
        val_log = all_results.get(val_key, [])
        if not train_log or not val_log:
            continue
        plt.plot([x["epoch"] for x in train_log], [x.get("loss", float("nan")) for x in train_log], linestyle="--", label=f"{run_name} train loss")
        plt.plot([x["epoch"] for x in val_log], [x.get("primary_metric", float("nan")) for x in val_log], label=f"{run_name} val dice")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Dental Segmentation Training Loss and Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "primary_metric_curves.png", dpi=200)
    plt.close()


def _plot_metric_curves(runs, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for metric_name, ax in zip(METRICS, axes):
        for run_name, dataset_name, _payload, all_results in runs:
            val_key = f"{dataset_name}_0_val_log_info"
            val_log = all_results.get(val_key, [])
            if not val_log:
                continue
            ax.plot([x["epoch"] for x in val_log], [x.get(metric_name, float("nan")) for x in val_log], label=run_name)
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("Dental Segmentation Validation Metrics")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "reported_metrics_curves.png", dpi=200)
    plt.close(fig)


def _plot_summary(runs, out_dir: Path):
    names = [run_name for run_name, *_ in runs]
    if not names:
        return
    x = np.arange(len(names))
    width = 0.22
    plt.figure(figsize=(max(8, len(names) * 1.4), 5))
    offsets = np.linspace(-width, width, len(METRICS))
    for offset, metric_name in zip(offsets, METRICS):
        values = [payload.get("best_test_reported_metrics", {}).get(metric_name, float("nan")) for _, _, payload, _ in runs]
        plt.bar(x + offset, values, width=width, label=metric_name)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Dental Segmentation Test Metrics by Run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_metrics.png", dpi=200)
    plt.close()


def main():
    base_dir = Path(".")
    runs = _collect_runs(base_dir)
    if not runs:
        raise RuntimeError("No valid run_* folders found.")
    _plot_primary_curves(runs, base_dir)
    _plot_metric_curves(runs, base_dir)
    _plot_summary(runs, base_dir)
    print("Saved plots: primary_metric_curves.png, reported_metrics_curves.png, summary_metrics.png")


if __name__ == "__main__":
    main()
