import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from registry import load_registry


REQUIRED_TOP_LEVEL = [
    "dataset_name",
    "data_root",
    "task_type",
    "class_names",
    "counts",
    "class_distribution",
    "splits",
]

REQUIRED_SPLIT_KEYS = ["train", "val", "test"]
REQUIRED_SAMPLE_KEYS = [
    "id",
    "img_path",
    "mask_path",
    "disease_dict",
    "tooth_dict",
    "structure_dict",
    "therapy_dict",
    "class_name",
    "label",
]


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, current.parent, *current.parents]:
        if (candidate / "benchmark").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Unable to infer repository root with benchmark/ and data/")


def _validate_sample(sample: Dict[str, object], class_names: List[str], where: str) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_SAMPLE_KEYS:
        if key not in sample:
            errors.append(f"{where}: missing sample key '{key}'")

    if "class_name" in sample and sample["class_name"] not in class_names:
        errors.append(f"{where}: class_name '{sample['class_name']}' not in class_names")

    if "label" in sample:
        label = sample["label"]
        if not isinstance(label, int):
            errors.append(f"{where}: label must be int")
        elif not (0 <= label < max(len(class_names), 1)):
            errors.append(f"{where}: label {label} out of range [0, {len(class_names) - 1}]")

    return errors


def _sample_uid(sample: Dict[str, object]) -> str:
    sample_id = str(sample.get("id", "")).strip()
    image_path = str(sample.get("img_path", "")).strip()
    if sample_id:
        return f"id:{sample_id}"
    if image_path:
        return f"img:{image_path}"
    return ""


def validate_split_file(split_path: Path) -> Tuple[List[str], Dict[str, int]]:
    errors: List[str] = []
    data = json.loads(split_path.read_text(encoding="utf-8"))

    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            errors.append(f"{split_path}: missing top-level key '{key}'")

    if errors:
        return errors, {}

    class_names = [str(x) for x in data.get("class_names", [])]
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    splits = data.get("splits", {})
    declared_data_root = str(data.get("data_root", "")).strip()
    if not declared_data_root:
        errors.append(f"{split_path}: data_root is empty")
    else:
        data_root_path = _repo_root(split_path.parent) / declared_data_root
        if not data_root_path.exists():
            errors.append(f"{split_path}: data_root does not exist -> {declared_data_root}")

    for split_key in REQUIRED_SPLIT_KEYS:
        if split_key not in splits:
            errors.append(f"{split_path}: missing split '{split_key}'")

    if errors:
        return errors, {}

    # sample schema and file existence
    ids_seen: Set[str] = set()
    split_seen: Dict[str, str] = {}
    seen_train_classes: Set[str] = set()
    seen_val_classes: Set[str] = set()
    seen_test_classes: Set[str] = set()

    split_class_sets = {
        "train": seen_train_classes,
        "val": seen_val_classes,
        "test": seen_test_classes,
    }

    for split_key in REQUIRED_SPLIT_KEYS:
        samples = splits.get(split_key, [])
        if not isinstance(samples, list):
            errors.append(f"{split_path}:{split_key} must be a list")
            continue
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                errors.append(f"{split_path}:{split_key}[{idx}] must be an object")
                continue
            where = f"{split_path}:{split_key}[{idx}]"
            errors.extend(_validate_sample(sample, class_names, where))

            class_name = str(sample.get("class_name", ""))
            label = sample.get("label")
            if class_name:
                split_class_sets[split_key].add(class_name)

            if class_name in class_to_label and isinstance(label, int):
                expected = class_to_label[class_name]
                if label != expected:
                    errors.append(
                        f"{where}: class_name='{class_name}' expects label={expected}, got {label}"
                    )

            sample_id = str(sample.get("id", "")).strip()
            if sample_id:
                if sample_id in ids_seen:
                    errors.append(f"{where}: duplicate id '{sample_id}' across dataset")
                else:
                    ids_seen.add(sample_id)

            uid = _sample_uid(sample)
            if uid:
                seen_in = split_seen.get(uid)
                if seen_in is not None and seen_in != split_key:
                    errors.append(
                        f"{where}: sample appears in multiple splits ({seen_in} and {split_key})"
                    )
                else:
                    split_seen[uid] = split_key

            if declared_data_root:
                img_path = str(sample.get("img_path", "")).strip()
                if img_path:
                    image_file = data_root_path / img_path
                    if not image_file.exists():
                        errors.append(f"{where}: img_path not found -> {img_path}")
                mask_path = str(sample.get("mask_path", "")).strip()
                if mask_path:
                    mask_file = data_root_path / mask_path
                    if not mask_file.exists():
                        errors.append(f"{where}: mask_path not found -> {mask_path}")

    if len(splits.get("train", [])) == 0:
        errors.append(f"{split_path}: train split is empty")

    missing_in_train = set(class_names) - seen_train_classes
    if missing_in_train:
        errors.append(
            f"{split_path}: class_names not covered by train split -> {sorted(missing_in_train)}"
        )

    unseen_in_val = seen_val_classes - seen_train_classes
    if unseen_in_val:
        errors.append(
            f"{split_path}: val contains classes unseen in train -> {sorted(unseen_in_val)}"
        )

    unseen_in_test = seen_test_classes - seen_train_classes
    if unseen_in_test:
        errors.append(
            f"{split_path}: test contains classes unseen in train -> {sorted(unseen_in_test)}"
        )

    # counts consistency
    counts = data.get("counts", {})
    expected_counts = {k: len(splits.get(k, [])) for k in REQUIRED_SPLIT_KEYS}
    for k in REQUIRED_SPLIT_KEYS:
        if counts.get(k) != expected_counts[k]:
            errors.append(
                f"{split_path}: counts.{k}={counts.get(k)} but actual={expected_counts[k]}"
            )

    # class distribution consistency
    dist = data.get("class_distribution", {})
    for split_key in REQUIRED_SPLIT_KEYS:
        actual = {cn: 0 for cn in class_names}
        for sample in splits.get(split_key, []):
            cn = sample.get("class_name")
            if cn in actual:
                actual[cn] += 1
        reported = dist.get(split_key, {})
        if reported != actual:
            errors.append(f"{split_path}: class_distribution.{split_key} mismatch")

    return errors, expected_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark split manifests")
    parser.add_argument("--repo_root", type=str, default=".")
    args = parser.parse_args()

    root = _repo_root(Path(args.repo_root))
    registry = load_registry(root)

    # Deduplicate by split file path because aliases point to same entry.
    unique_split_files = sorted({entry.split_file for entry in registry.values()})

    all_errors: List[str] = []
    for split_rel in unique_split_files:
        split_path = root / split_rel
        errors, counts = validate_split_file(split_path)
        if errors:
            all_errors.extend(errors)
            print(f"[FAIL] {split_rel}")
        else:
            print(f"[OK] {split_rel} counts={counts}")

    if all_errors:
        print("\nValidation errors:")
        for err in all_errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("\nBenchmark validation passed.")


if __name__ == "__main__":
    main()
