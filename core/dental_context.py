import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.registry import resolve_task_spec
from core.result_writer import extract_final_info_payload
from core.task_spec import infer_preprocess_mode, task_spec_summary


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _load_baseline_snapshot(base_dir: Path) -> Dict[str, Any]:
    baseline_path = base_dir / "run_0" / "final_info.json"
    if not baseline_path.exists():
        return {
            "available": False,
            "path": str(baseline_path),
            "primary_metric_name": "",
            "best_val_primary": None,
            "best_test_primary": None,
        }

    baseline_wrapped = _load_json_if_exists(baseline_path)
    if not baseline_wrapped:
        return {
            "available": False,
            "path": str(baseline_path),
            "primary_metric_name": "",
            "best_val_primary": None,
            "best_test_primary": None,
        }

    dataset_name = next(iter(baseline_wrapped.keys()))
    payload = extract_final_info_payload(baseline_wrapped, dataset_name)
    scorecard = payload.get("scorecard", {}) if isinstance(payload, dict) else {}
    return {
        "available": True,
        "path": str(baseline_path),
        "dataset_name": dataset_name,
        "primary_metric_name": scorecard.get(
            "primary_metric_name",
            payload.get("primary_metric_name", ""),
        ),
        "best_val_primary": scorecard.get("best_val_primary"),
        "best_test_primary": scorecard.get("best_test_primary"),
        "best_epoch": scorecard.get("best_epoch"),
    }


def build_dental_task_context(
    repo_root: Path,
    base_dir: Path,
    task_name: str,
    data_root_override: str = "",
    split_file_override: str = "",
) -> Dict[str, Any]:
    task_spec = resolve_task_spec(
        root=repo_root,
        task_name=task_name,
        data_root_override=data_root_override,
        split_file_override=split_file_override,
    )

    split_payload = _load_json_if_exists(task_spec.split_file)
    split_counts = split_payload.get("counts", {}) if isinstance(split_payload, dict) else {}
    class_distribution = split_payload.get("class_distribution", {}) if isinstance(split_payload, dict) else {}

    return {
        "template": base_dir.name,
        "research_domain": "dental_imaging",
        "research_mode": "task_driven_exploration",
        "resource_profile": "low_resource_demo",
        "task_name": task_spec.task_name,
        "dataset_name": task_spec.dataset_name,
        "task_spec": task_spec_summary(task_spec),
        "data_paths": {
            "data_root": str(task_spec.data_root),
            "split_file": str(task_spec.split_file),
        },
        "split_counts": split_counts,
        "class_distribution": class_distribution,
        "baseline": _load_baseline_snapshot(base_dir),
        "recommended_defaults": {
            "preprocess_mode": infer_preprocess_mode(task_spec),
            "backbone_name": "efficientnet_b0",
            "loss_name": "weighted_ce",
            "use_pretrained": True,
        },
        "allowed_modification_axes": [
            "preprocessing",
            "augmentation_policy",
            "backbone_selection_within_existing_choices",
            "loss_design",
            "imbalance_handling",
            "threshold_selection",
            "calibration_analysis",
        ],
        "discouraged_axes": [
            "changing_benchmark_split_protocol",
            "using_external_data",
            "task_specific_label_leakage",
            "augmentations_forbidden_by_benchmark_policy",
            "unbounded_model_scaling",
        ],
        "demo_success_criteria": [
            "idea is clinically meaningful for dental imaging",
            "idea can be implemented inside the current template",
            "idea preserves benchmark fairness and reproducibility",
            "idea fits a low-resource experimental budget",
        ],
    }


def write_dental_task_context(
    repo_root: Path,
    base_dir: Path,
    task_name: str,
    data_root_override: str = "",
    split_file_override: str = "",
) -> Path:
    context = build_dental_task_context(
        repo_root=repo_root,
        base_dir=base_dir,
        task_name=task_name,
        data_root_override=data_root_override,
        split_file_override=split_file_override,
    )
    output_path = base_dir / "task_context.json"
    output_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
