import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from core.task_spec import BenchmarkEntry, TaskSpec, normalize_task_name


_REGISTRY_CACHE: Optional[Dict[str, BenchmarkEntry]] = None


def repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in [current, current.parent, *current.parents]:
        if (candidate / "benchmark").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Unable to infer repository root containing benchmark/ and data/")


def _read_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _to_str_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _resolve_path(root: Path, raw_path: str, fallback_rel_path: Path) -> Path:
    candidate = Path(raw_path).expanduser() if raw_path else fallback_rel_path
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _build_registry(root: Optional[Path] = None) -> Dict[str, BenchmarkEntry]:
    root = repo_root(root)
    benchmark_root = root / "benchmark"
    registry: Dict[str, BenchmarkEntry] = {}

    for ds_dir in sorted(p for p in benchmark_root.iterdir() if p.is_dir()):
        card_path = ds_dir / "dataset_card.yaml"
        split_path = ds_dir / "splits.json"
        if not card_path.exists() or not split_path.exists():
            continue

        card = _read_yaml(card_path)
        split_data = json.loads(split_path.read_text(encoding="utf-8"))

        task_name = normalize_task_name(ds_dir.name)
        dataset_name = str(split_data.get("dataset_name") or card.get("name") or ds_dir.name)

        data_root_path = _resolve_path(
            root,
            str(split_data.get("data_root") or card.get("data_root") or "").strip(),
            Path("data") / ds_dir.name,
        )
        split_file_path = _resolve_path(
            root,
            str(card.get("split_file", "")).strip(),
            Path("benchmark") / ds_dir.name / "splits.json",
        )

        entry = BenchmarkEntry(
            task_name=task_name,
            data_root=data_root_path,
            split_file=split_file_path,
            modality=str(card.get("modality", "")).strip(),
            task_type=str(split_data.get("task_type") or card.get("task_type") or "").strip(),
            label_level=str(card.get("label_level", "")).strip(),
            target_name=str(card.get("target_name", "")).strip(),
            class_names=[str(x).strip() for x in (split_data.get("class_names") or card.get("class_names") or [])],
            clinical_goal=str(card.get("clinical_goal", "")).strip(),
            primary_metric=str(card.get("primary_metric", "")).strip(),
            secondary_metrics=_to_str_list(card.get("secondary_metrics")),
            augmentation_allowed=_to_str_list((card.get("augmentation") or {}).get("allowed")),
            augmentation_disallowed=_to_str_list((card.get("augmentation") or {}).get("disallowed")),
            imbalance_expected=str((card.get("imbalance") or {}).get("expected", "")).strip(),
            notes=_to_str_list(card.get("notes")),
        )

        for alias in {task_name, normalize_task_name(dataset_name)}:
            registry[alias] = entry

    return registry


def _registry() -> Dict[str, BenchmarkEntry]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()
    return _REGISTRY_CACHE


def get_entry(task_name: str) -> BenchmarkEntry:
    key = normalize_task_name(task_name)
    registry = _registry()
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown task_name '{task_name}'. Available task names: {available}")
    return registry[key]


def list_task_names() -> List[str]:
    return sorted(_registry().keys())


def resolve_relative_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_task_spec(
    root: Path,
    task_name: str,
    data_root_override: str = "",
    split_file_override: str = "",
) -> TaskSpec:
    entry = get_entry(task_name)
    data_root = resolve_relative_path(root, data_root_override) if data_root_override else entry.data_root
    split_file = resolve_relative_path(root, split_file_override) if split_file_override else entry.split_file

    return TaskSpec(
        task_name=entry.task_name,
        dataset_name=entry.task_name,
        data_root=data_root,
        split_file=split_file,
        modality=entry.modality,
        task_type=entry.task_type,
        label_level=entry.label_level,
        target_name=entry.target_name,
        class_names=[str(x).strip() for x in entry.class_names],
        clinical_goal=entry.clinical_goal,
        primary_metric=entry.primary_metric,
        secondary_metrics=[str(x) for x in entry.secondary_metrics],
        augmentation_allowed=[str(x) for x in entry.augmentation_allowed],
        augmentation_disallowed=[str(x) for x in entry.augmentation_disallowed],
        imbalance_expected=entry.imbalance_expected,
        notes=[str(x) for x in entry.notes],
    )