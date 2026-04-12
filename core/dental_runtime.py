import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from core.registry import list_task_names


def env_flag(name: str) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def load_local_task_context(template_file: str) -> Dict[str, object]:
    context_path = Path(template_file).resolve().parent / "task_context.json"
    if not context_path.exists():
        return {}
    try:
        with context_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def context_data_paths(task_context: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(task_context, dict):
        return {}
    data_paths = task_context.get("data_paths", {})
    return data_paths if isinstance(data_paths, dict) else {}


def resolve_default_task_name(
    preferred_task_name: str,
    task_context: Optional[Dict[str, object]] = None,
) -> str:
    task_name = str((task_context or {}).get("task_name", "")).strip()
    if task_name:
        return task_name
    task_names = list_task_names()
    if not task_names:
        raise FileNotFoundError("No tasks found in benchmark registry")
    return preferred_task_name if preferred_task_name in task_names else task_names[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_image_relpath(sample: Dict[str, object]) -> str:
    if "path" in sample and sample["path"]:
        return str(sample["path"])
    if "img_path" in sample and sample["img_path"]:
        return str(sample["img_path"])
    raise KeyError("Sample must contain 'path' or 'img_path'")


def load_manifest_from_file(manifest_path: Path) -> Dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "splits" not in manifest:
        raise ValueError(f"Invalid split file: missing 'splits' in {manifest_path}")
    return manifest


def radiograph_preprocess(image: Image.Image, mode: str) -> Image.Image:
    if mode == "identity":
        return image
    if mode == "gray":
        return ImageOps.grayscale(image).convert("RGB")
    if mode == "autocontrast":
        return ImageOps.autocontrast(ImageOps.grayscale(image)).convert("RGB")
    if mode == "equalize":
        return ImageOps.equalize(ImageOps.grayscale(image)).convert("RGB")
    if mode == "auto_equalize":
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        gray = ImageOps.equalize(gray)
        return gray.convert("RGB")
    raise ValueError(f"Unsupported preprocess_mode: {mode}")


def normalize_policy_tokens(values: Sequence[str]) -> List[str]:
    return [
        str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for value in values
        if str(value).strip()
    ]


def contains_any(tokens: Sequence[str], keywords: Sequence[str]) -> bool:
    for token in tokens:
        for keyword in keywords:
            if keyword in token:
                return True
    return False
