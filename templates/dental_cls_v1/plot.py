import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_final_info(final_info_path: Path) -> Tuple[str, Dict[str, object]]:
    data = json.loads(final_info_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid final_info.json: {final_info_path}")
    dataset_name = next(iter(data.keys()))
    payload = data[dataset_name].get("means", {})
    return dataset_name, payload


def _load_all_results(all_results_path: Path) -> Dict[str, object]:
    data = np.load(all_results_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"Invalid all_results.npy: {all_results_path}")
    return data


def _safe_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_curve(logs: List[Dict[str, object]]) -> Tuple[List[int], List[float]]:
    epochs = []
    values = []
    for row in logs:
        epochs.append(int(row.get("epoch", len(epochs))))
        values.append(_safe_float(row.get("primary_metric")))
    return epochs, values


def _collect_run_dirs(template_dir: Path) -> List[Path]:
    return sorted([p for p in template_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])


def _plot_training_curve(run_dir: Path, dataset_name: str, all_results: Dict[str, object]) -> None:
    train_key = f"{dataset_name}_0_train_log_info"
    val_key = f"{dataset_name}_0_val_log_info"

    if train_key not in all_results or val_key not in all_results:
        return

    train_logs = all_results[train_key]
    val_logs = all_results[val_key]

    train_epochs, train_values = _extract_curve(train_logs)
    val_epochs, val_values = _extract_curve(val_logs)

    plt.figure(figsize=(7, 5))
    plt.plot(train_epochs, train_values, label="train primary")
    plt.plot(val_epochs, val_values, label="val primary")
    plt.xlabel("Epoch")
    plt.ylabel("Primary metric")
    plt.title(f"{dataset_name}: training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curve.png", dpi=200)
    plt.close()


def _plot_summary_bar(template_dir: Path, summary_rows: List[Dict[str, object]]) -> None:
    if not summary_rows:
        return

    run_names = [row["run_name"] for row in summary_rows]
    val_scores = [row["best_val_primary"] for row in summary_rows]
    test_scores = [row["best_test_primary"] for row in summary_rows]

    x = np.arange(len(run_names))
    width = 0.35

    plt.figure(figsize=(max(8, len(run_names) * 1.2), 5))
    plt.bar(x - width / 2, val_scores, width, label="best val primary")
    plt.bar(x + width / 2, test_scores, width, label="best test primary")
    plt.xticks(x, run_names, rotation=30, ha="right")
    plt.ylabel("Metric")
    plt.title("Dental CLS run summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(template_dir / "run_summary.png", dpi=200)
    plt.close()


def main() -> None:
    template_dir = Path(__file__).resolve().parent
    run_dirs = _collect_run_dirs(template_dir)

    if not run_dirs:
        print("No run_* directories found.")
        return

    summary_rows: List[Dict[str, object]] = []

    for run_dir in run_dirs:
        final_info_path = run_dir / "final_info.json"
        all_results_path = run_dir / "all_results.npy"

        if not final_info_path.exists():
            print(f"[Skip] Missing {final_info_path}")
            continue

        dataset_name, final_info = _load_final_info(final_info_path)

        if all_results_path.exists():
            try:
                all_results = _load_all_results(all_results_path)
                _plot_training_curve(run_dir, dataset_name, all_results)
            except Exception as exc:
                print(f"[Warn] Failed plotting training curve for {run_dir.name}: {exc}")

        scorecard = final_info.get("scorecard", {})
        best_test_metrics = final_info.get("best_test_metrics", {})

        summary_rows.append(
            {
                "run_name": run_dir.name,
                "dataset_name": dataset_name,
                "primary_metric_name": scorecard.get(
                    "primary_metric_name",
                    final_info.get("primary_metric_name", "unknown"),
                ),
                "best_val_primary": _safe_float(scorecard.get("best_val_primary")),
                "best_test_primary": _safe_float(scorecard.get("best_test_primary")),
                "best_epoch": _safe_float(scorecard.get("best_epoch")),
                "total_train_time": _safe_float(final_info.get("total_train_time")),
                "test_accuracy": _safe_float(best_test_metrics.get("accuracy")),
                "test_auc": _safe_float(best_test_metrics.get("auc", best_test_metrics.get("macro_auc"))),
                "test_f1": _safe_float(best_test_metrics.get("f1", best_test_metrics.get("f1_macro"))),
                "test_ece": _safe_float(best_test_metrics.get("ece")),
            }
        )

    (template_dir / "run_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_summary_bar(template_dir, summary_rows)

    print(f"Processed {len(summary_rows)} runs.")
    print(f"Saved summary json to: {template_dir / 'run_summary.json'}")
    print(f"Saved summary figure to: {template_dir / 'run_summary.png'}")


if __name__ == "__main__":
    main()