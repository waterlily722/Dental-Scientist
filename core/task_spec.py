from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class BenchmarkEntry:
    task_name: str
    data_root: Path
    split_file: Path
    modality: str
    task_type: str
    label_level: str
    target_name: str
    class_names: List[str]
    clinical_goal: str
    primary_metric: str
    secondary_metrics: List[str]
    augmentation_allowed: List[str]
    augmentation_disallowed: List[str]
    imbalance_expected: str
    notes: List[str]


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    dataset_name: str
    data_root: Path
    split_file: Path
    modality: str
    task_type: str
    label_level: str
    target_name: str
    class_names: List[str]
    clinical_goal: str
    primary_metric: str
    secondary_metrics: List[str]
    augmentation_allowed: List[str]
    augmentation_disallowed: List[str]
    imbalance_expected: str
    notes: List[str]


def normalize_task_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def infer_task_family(task_type: str) -> str:
    value = str(task_type).strip().lower().replace(" ", "")
    if "classification" in value and "segmentation" not in value and "detection" not in value:
        return "classification"
    if "segmentation" in value:
        return "segmentation"
    if "detection" in value:
        return "detection"
    if any(token in value for token in ["keypoint", "landmark", "tooth_numbering", "numbering"]):
        return "keypoint"
    return "unknown"


def infer_preprocess_mode(task_spec: TaskSpec) -> str:
    """Infer a robust preprocess mode from modality."""
    modality = str(task_spec.modality or "").upper()
    if "IOP" in modality:
        return "identity"
    if any(token in modality for token in ["PA", "PAN", "OPG", "X", "XRAY"]):
        return "auto_equalize"
    return "identity"


def infer_image_size(task_spec: TaskSpec, fallback: int = 224) -> int:
    return int(fallback)


def task_spec_summary(task_spec: TaskSpec) -> Dict[str, object]:
    """Compact summary for agents/planners to consume consistently."""
    return {
        "task_name": task_spec.task_name,
        "modality": task_spec.modality,
        "task_type": task_spec.task_type,
        "label_level": task_spec.label_level,
        "target_name": task_spec.target_name,
        "primary_metric": task_spec.primary_metric,
        "secondary_metrics": task_spec.secondary_metrics,
        "recommended_preprocess_mode": infer_preprocess_mode(task_spec),
        "recommended_image_size": infer_image_size(task_spec),
        "allowed_augmentations": task_spec.augmentation_allowed,
        "disallowed_augmentations": task_spec.augmentation_disallowed,
        "imbalance_expected": task_spec.imbalance_expected,
        "notes": task_spec.notes,
    }


def task_spec_to_dict(task_spec: TaskSpec) -> Dict[str, object]:
    data = asdict(task_spec)
    data["data_root"] = str(task_spec.data_root)
    data["split_file"] = str(task_spec.split_file)
    return data