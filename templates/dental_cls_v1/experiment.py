import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "benchmark").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Unable to infer repository root containing benchmark/ and data/")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dental_runtime import (  # noqa: E402
    contains_any,
    context_data_paths,
    env_flag,
    env_int,
    load_manifest_from_file as load_manifest_from_file_base,
    load_local_task_context,
    normalize_policy_tokens,
    radiograph_preprocess,
    resolve_default_task_name,
    sample_image_relpath,
    seed_everything,
)
from core.registry import resolve_task_spec  # noqa: E402
from core.result_writer import write_run_outputs  # noqa: E402
from core.task_spec import (  # noqa: E402
    TaskSpec,
    infer_image_size,
    infer_preprocess_mode as infer_preprocess_mode_from_spec,
    infer_task_family,
    task_spec_to_dict,
)


def precheck_enabled() -> bool:
    return env_flag("AI_SCIENTIST_PRECHECK")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def compute_split_counts(splits: Dict[str, List[Dict[str, object]]]) -> Dict[str, int]:
    return {split_name: len(entries) for split_name, entries in splits.items()}


def compute_class_counts(entries: List[Dict[str, object]], class_names: List[str]) -> Dict[str, int]:
    counter = Counter(str(entry["class_name"]) for entry in entries)
    return {class_name: int(counter.get(class_name, 0)) for class_name in class_names}


def enrich_manifest_stats(manifest: Dict[str, object]) -> Dict[str, object]:
    class_names = [str(x) for x in manifest.get("class_names", [])]
    splits = manifest.get("splits", {})
    for key in ["train", "val", "test"]:
        splits.setdefault(key, [])
    manifest["splits"] = splits

    if not class_names:
        raise ValueError("split manifest must include explicit class_names from benchmark")

    manifest["num_classes"] = len(class_names)
    manifest["counts"] = compute_split_counts(splits)

    all_split_samples = [sample for split_entries in splits.values() for sample in split_entries]
    manifest["class_counts"] = compute_class_counts(all_split_samples, class_names)
    return manifest


def load_manifest_from_file(manifest_path: Path) -> Dict[str, object]:
    return enrich_manifest_stats(load_manifest_from_file_base(manifest_path))


def _limit_entries_for_precheck(entries: List[Dict[str, object]], max_samples: int) -> List[Dict[str, object]]:
    if len(entries) <= max_samples:
        return entries

    by_label: Dict[int, List[Dict[str, object]]] = {}
    for entry in entries:
        by_label.setdefault(int(entry["label"]), []).append(entry)

    selected: List[Dict[str, object]] = []
    for label_entries in by_label.values():
        selected.append(label_entries[0])
        if len(selected) >= max_samples:
            return selected[:max_samples]

    remaining = []
    seen_ids = {id(item) for item in selected}
    for entry in entries:
        if id(entry) not in seen_ids:
            remaining.append(entry)

    budget = max_samples - len(selected)
    selected.extend(remaining[:budget])
    return selected


def maybe_reduce_manifest_for_precheck(manifest: Dict[str, object]) -> Dict[str, object]:
    if not precheck_enabled():
        return manifest

    max_samples = env_int("AI_SCIENTIST_PRECHECK_MAX_SAMPLES", 48)
    reduced = dict(manifest)
    reduced_splits: Dict[str, List[Dict[str, object]]] = {}
    for split_name, entries in manifest["splits"].items():
        reduced_splits[split_name] = _limit_entries_for_precheck(list(entries), max_samples)
    reduced["splits"] = reduced_splits
    reduced["precheck"] = {
        "enabled": True,
        "max_samples_per_split": max_samples,
    }
    return enrich_manifest_stats(reduced)


class RadiographPreprocessTransform:
    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, image: Image.Image) -> Image.Image:
        return radiograph_preprocess(image, self.mode)


class AddGaussianNoise:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.std * torch.randn_like(tensor)


def infer_augmentation_policy(task_spec: TaskSpec) -> Dict[str, bool]:
    allowed = normalize_policy_tokens(list(getattr(task_spec, "augmentation_allowed", []) or []))
    disallowed = normalize_policy_tokens(list(getattr(task_spec, "augmentation_disallowed", []) or []))

    policy = {
        "enable_random_resized_crop": True,
        "enable_rotation": True,
        "enable_horizontal_flip": False,
        "enable_color_jitter": False,
        "enable_gaussian_noise": False,
    }

    if allowed:
        policy["enable_random_resized_crop"] = contains_any(allowed, ["crop", "resized"])
        policy["enable_rotation"] = contains_any(allowed, ["rotation", "rotate"])
        policy["enable_horizontal_flip"] = contains_any(allowed, ["flip", "horizontal_flip"])
        policy["enable_color_jitter"] = contains_any(allowed, ["color_jitter", "brightness", "contrast"])
        policy["enable_gaussian_noise"] = contains_any(allowed, ["gaussian_noise", "noise"])

    if contains_any(disallowed, ["left_right_flip", "flip", "tooth_number", "tooth_id", "quadrant", "laterality"]):
        policy["enable_horizontal_flip"] = False
    if contains_any(disallowed, ["rotation", "label_breaking_geometry"]):
        policy["enable_rotation"] = False
    if contains_any(disallowed, ["color_jitter", "brightness", "contrast"]):
        policy["enable_color_jitter"] = False
    if contains_any(disallowed, ["gaussian_noise", "noise"]):
        policy["enable_gaussian_noise"] = False

    return policy


def build_transforms(
    image_size: int = 224,
    preprocess_mode: str = "identity",
    augmentation_policy: Optional[Dict[str, bool]] = None,
):
    preprocess = RadiographPreprocessTransform(preprocess_mode)
    policy = augmentation_policy or {}

    enable_crop = bool(policy.get("enable_random_resized_crop", True))
    enable_rotation = bool(policy.get("enable_rotation", True))
    enable_hflip = bool(policy.get("enable_horizontal_flip", False))
    enable_color_jitter = bool(policy.get("enable_color_jitter", False))
    enable_gaussian_noise = bool(policy.get("enable_gaussian_noise", False))

    train_ops: List[object] = [preprocess, transforms.Resize((256, 256))]
    if enable_crop:
        train_ops.append(transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)))
    else:
        train_ops.append(transforms.CenterCrop(image_size))
    if enable_rotation:
        train_ops.append(transforms.RandomRotation(5))
    if enable_hflip:
        train_ops.append(transforms.RandomHorizontalFlip())
    if enable_color_jitter:
        train_ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))

    train_ops.append(transforms.ToTensor())
    if enable_gaussian_noise:
        train_ops.append(AddGaussianNoise(std=0.02))
    train_ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)))

    eval_ops = [
        preprocess,
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    ]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class DentalRadiographDataset(Dataset):
    def __init__(self, root: Path, entries: List[Dict[str, object]], transform=None):
        self.root = root
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image_path = self.root / sample_image_relpath(entry)
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(int(entry["label"]), dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_model(
    backbone_name: str,
    num_classes: int,
    use_pretrained: bool,
    dropout_p: float,
) -> nn.Module:
    backbone_name = backbone_name.strip().lower()

    if backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    if backbone_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        hidden_features = model.classifier[0].out_features
        input_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported backbone_name: {backbone_name}")


def compute_class_weights(entries: List[Dict[str, object]], num_classes: int) -> torch.Tensor:
    counts = np.bincount([int(entry["label"]) for entry in entries], minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (float(num_classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(entries: List[Dict[str, object]], num_classes: int) -> WeightedRandomSampler:
    labels = [int(entry["label"]) for entry in entries]
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    sample_weights = torch.tensor([1.0 / class_counts[label] for label in labels], dtype=torch.float32)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("class_counts", class_counts.float().clamp(min=1.0))
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        adjusted_logits = logits + self.class_counts.log().to(logits.device)
        return F.cross_entropy(adjusted_logits, targets, label_smoothing=self.label_smoothing)


def build_criterion(config, class_weights: torch.Tensor, class_counts: torch.Tensor):
    if config.loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    if config.loss_name == "weighted_ce":
        weight = class_weights if config.use_class_weights else None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)

    if config.loss_name == "focal":
        alpha = class_weights if config.use_class_weights else None
        return FocalLoss(alpha=alpha, gamma=config.focal_gamma)

    if config.loss_name == "balanced_softmax":
        return BalancedSoftmaxLoss(class_counts=class_counts, label_smoothing=config.label_smoothing)

    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)

    i = 0
    rank = 1.0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1.0)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += j - i
        i = j

    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def auprc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = int(y_true.sum())
    if pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(pos)

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    delta_recall = np.diff(recall)
    return float(np.sum(delta_recall * precision[1:]))


def _ece_score(y_true: np.ndarray, y_pred: np.ndarray, y_conf: np.ndarray) -> float:
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end == 1.0:
            mask = (y_conf >= start) & (y_conf <= end)
        else:
            mask = (y_conf >= start) & (y_conf < end)
        if not np.any(mask):
            continue
        bin_confidence = float(y_conf[mask].mean())
        bin_accuracy = float((y_true[mask] == y_pred[mask]).mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim == 1:
        y_prob = np.stack([1.0 - y_prob, y_prob], axis=1)

    num_classes = y_prob.shape[1]
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)

    metrics: Dict[str, float] = {
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else float("nan"),
        "ece": _ece_score(y_true, y_pred, confidence),
        "confidence": float(np.mean(confidence)) if len(confidence) else float("nan"),
    }

    if num_classes == 2:
        positive_prob = y_prob[:, 1]
        binary_pred = (positive_prob >= threshold).astype(np.int64)

        tp = int(((binary_pred == 1) & (y_true == 1)).sum())
        tn = int(((binary_pred == 0) & (y_true == 0)).sum())
        fp = int(((binary_pred == 1) & (y_true == 0)).sum())
        fn = int(((binary_pred == 0) & (y_true == 1)).sum())

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)

        metrics.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
                "auc": float(auc_score(y_true, positive_prob)),
                "auprc": float(auprc_score(y_true, positive_prob)),
            }
        )
        return metrics

    per_class_f1 = []
    aucs = []
    auprcs = []
    for class_idx in range(num_classes):
        one_vs_rest_true = (y_true == class_idx).astype(np.int64)
        one_vs_rest_pred = (y_pred == class_idx).astype(np.int64)

        tp = int(((one_vs_rest_pred == 1) & (one_vs_rest_true == 1)).sum())
        fp = int(((one_vs_rest_pred == 1) & (one_vs_rest_true == 0)).sum())
        fn = int(((one_vs_rest_pred == 0) & (one_vs_rest_true == 1)).sum())

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        per_class_f1.append(f1)

        auc_value = auc_score(one_vs_rest_true, y_prob[:, class_idx])
        if np.isfinite(auc_value):
            aucs.append(float(auc_value))

        auprc_value = auprc_score(one_vs_rest_true, y_prob[:, class_idx])
        if np.isfinite(auprc_value):
            auprcs.append(float(auprc_value))

    metrics.update(
        {
            "f1_macro": float(np.mean(per_class_f1)) if per_class_f1 else float("nan"),
            "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        }
    )
    return metrics


def resolve_primary_metric_name(task_spec: TaskSpec, num_classes: int) -> str:
    raw = str(task_spec.primary_metric or "").strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "auroc": "auc",
        "roc_auc": "auc",
        "pr_auc": "auprc",
        "auc_pr": "auprc",
        "sensitivity": "recall",
        "tpr": "recall",
    }
    metric = alias.get(raw, raw)

    if not metric:
        metric = "auc" if num_classes == 2 else "macro_auc"

    if num_classes > 2:
        if metric == "auc":
            return "macro_auc"
        if metric == "auprc":
            return "macro_auprc"
        if metric == "f1":
            return "f1_macro"
    return metric


def run_epoch(model, loader, criterion, optimizer, device, threshold: float):
    model.train()
    total_loss = 0.0
    total_examples = 0
    y_true: List[int] = []
    y_prob: List[List[float]] = []

    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_TRAIN_BATCHES", 2) if precheck_enabled() else None

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_examples += int(labels.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(torch.softmax(logits.detach(), dim=1).cpu().numpy().tolist())

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    metrics = classification_metrics(np.array(y_true), np.array(y_prob), threshold=threshold)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, device, threshold: float):
    model.eval()
    y_true: List[int] = []
    y_prob: List[List[float]] = []

    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_EVAL_BATCHES", 2) if precheck_enabled() else None

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    return classification_metrics(np.array(y_true), np.array(y_prob), threshold=threshold)


def infer_imbalance_expected(task_spec: TaskSpec) -> bool:
    value = str(getattr(task_spec, "imbalance_expected", "") or "").strip().lower()
    return value in {"yes", "true", "likely", "expected", "1"}


def resolve_bool(value: int, default: bool) -> bool:
    if value < 0:
        return default
    return bool(value)


@dataclass
class ClassificationConfig:
    task_name: str
    out_dir: str

    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    image_size: int = 224
    backbone_name: str = "efficientnet_b0"
    use_pretrained: bool = True
    dropout_p: float = 0.2

    preprocess_mode: str = "identity"
    loss_name: str = "weighted_ce"
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    threshold: float = 0.5
    augmentation_policy: Optional[Dict[str, bool]] = None
    auto_policy: Optional[Dict[str, object]] = None
    run_mode: str = "full"


def resolve_classification_config(args, task_spec: TaskSpec) -> ClassificationConfig:
    preprocess_mode = args.preprocess_mode if args.preprocess_mode != "auto" else infer_preprocess_mode_from_spec(task_spec)
    image_size = args.image_size if args.image_size > 0 else infer_image_size(task_spec)

    imbalance_default = infer_imbalance_expected(task_spec)
    use_class_weights = resolve_bool(args.use_class_weights, imbalance_default)
    use_weighted_sampler = resolve_bool(args.use_weighted_sampler, imbalance_default)
    augmentation_policy = infer_augmentation_policy(task_spec)

    run_mode = "precheck" if precheck_enabled() else "full"
    epochs = min(args.epochs, env_int("AI_SCIENTIST_PRECHECK_EPOCHS", 1)) if precheck_enabled() else args.epochs
    batch_size = min(args.batch_size, 8) if precheck_enabled() else args.batch_size

    auto_policy = {
        "template": "dental_cls_v1",
        "baseline": "efficientnet_b0 + weighted_ce",
        "run_mode": run_mode,
        "preprocess_mode": preprocess_mode,
        "image_size": image_size,
        "imbalance_expected": getattr(task_spec, "imbalance_expected", ""),
        "use_class_weights": use_class_weights,
        "use_weighted_sampler": use_weighted_sampler,
        "augmentation_policy": augmentation_policy,
    }

    return ClassificationConfig(
        task_name=task_spec.task_name,
        out_dir=args.out_dir,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=epochs,
        seed=args.seed,
        num_workers=0 if precheck_enabled() else args.num_workers,
        image_size=image_size,
        backbone_name=args.backbone_name,
        use_pretrained=bool(args.use_pretrained),
        dropout_p=args.dropout_p,
        preprocess_mode=preprocess_mode,
        loss_name=args.loss_name,
        use_class_weights=use_class_weights,
        use_weighted_sampler=use_weighted_sampler,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        threshold=args.threshold,
        augmentation_policy=augmentation_policy,
        auto_policy=auto_policy,
        run_mode=run_mode,
    )


def make_loaders(config: ClassificationConfig, task_spec: TaskSpec):
    manifest = load_manifest_from_file(task_spec.split_file)
    manifest = maybe_reduce_manifest_for_precheck(manifest)

    train_transform, eval_transform = build_transforms(
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        augmentation_policy=config.augmentation_policy,
    )

    train_dataset = DentalRadiographDataset(task_spec.data_root, manifest["splits"]["train"], transform=train_transform)
    val_dataset = DentalRadiographDataset(task_spec.data_root, manifest["splits"]["val"], transform=eval_transform)
    test_dataset = DentalRadiographDataset(task_spec.data_root, manifest["splits"]["test"], transform=eval_transform)

    train_sampler = None
    if config.use_weighted_sampler:
        train_sampler = build_weighted_sampler(manifest["splits"]["train"], int(manifest["num_classes"]))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return manifest, train_loader, val_loader, test_loader


def default_task_name() -> str:
    task_context = load_local_task_context(__file__)
    return resolve_default_task_name("dental_caries_classificationv3", task_context)


def parse_args():
    task_context = load_local_task_context(__file__)
    data_paths = context_data_paths(task_context)

    parser = argparse.ArgumentParser(description="Dental Scientist classification template")
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--task_name", type=str, default=default_task_name())
    parser.add_argument("--data_root", type=str, default=str(data_paths.get("data_root", "")))
    parser.add_argument("--split_file", type=str, default=str(data_paths.get("split_file", "")))

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--image_size", type=int, default=-1)
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "mobilenet_v3_small"],
    )
    parser.add_argument("--use_pretrained", type=int, default=1)
    parser.add_argument("--dropout_p", type=float, default=0.2)

    parser.add_argument("--preprocess_mode", type=str, default="auto")
    parser.add_argument(
        "--loss_name",
        type=str,
        default="weighted_ce",
        choices=["ce", "weighted_ce", "focal", "balanced_softmax"],
    )
    parser.add_argument("--use_class_weights", type=int, default=-1)
    parser.add_argument("--use_weighted_sampler", type=int, default=-1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def run_classification_experiment(config: ClassificationConfig, task_spec: TaskSpec):
    seed_everything(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)

    manifest, train_loader, val_loader, test_loader = make_loaders(config, task_spec)
    dataset_name = str(manifest.get("dataset_name") or task_spec.dataset_name)
    class_names = [str(name) for name in (manifest.get("class_names") or task_spec.class_names)]
    num_classes = int(manifest.get("num_classes", len(class_names)))

    primary_metric = resolve_primary_metric_name(task_spec, num_classes=num_classes)

    print(
        f"Dataset {dataset_name}: modality={task_spec.modality}, task_type={task_spec.task_type}, "
        f"classes={class_names}, num_classes={num_classes}, split_counts={manifest.get('counts', {})}"
    )

    class_weights = compute_class_weights(manifest["splits"]["train"], num_classes).to(config.device)
    class_counts = torch.tensor(
        [manifest["class_counts"][class_name] for class_name in class_names],
        dtype=torch.float32,
        device=config.device,
    )

    model = build_model(
        backbone_name=config.backbone_name,
        num_classes=num_classes,
        use_pretrained=config.use_pretrained,
        dropout_p=config.dropout_p,
    ).to(config.device)

    criterion = build_criterion(config, class_weights, class_counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_log_info = []
    val_log_info = []

    best_epoch = -1
    best_val_score = -math.inf
    best_ckpt_path = os.path.join(config.out_dir, "best_model.pth")

    start_time = time.time()
    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, config.device, config.threshold)
        val_metrics = evaluate_model(model, val_loader, config.device, threshold=config.threshold)
        scheduler.step()

        train_primary = float(train_metrics.get(primary_metric, float("nan")))
        val_primary = float(val_metrics.get(primary_metric, float("nan")))

        train_log_info.append(
            {
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "primary_metric": train_primary,
                "accuracy": train_metrics.get("accuracy", float("nan")),
                "auc": train_metrics.get("auc", train_metrics.get("macro_auc", float("nan"))),
                "f1": train_metrics.get("f1", train_metrics.get("f1_macro", float("nan"))),
            }
        )
        val_log_info.append(
            {
                "epoch": epoch,
                "primary_metric": val_primary,
                "accuracy": val_metrics.get("accuracy", float("nan")),
                "auc": val_metrics.get("auc", val_metrics.get("macro_auc", float("nan"))),
                "f1": val_metrics.get("f1", val_metrics.get("f1_macro", float("nan"))),
                "ece": val_metrics.get("ece", float("nan")),
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train {primary_metric} {train_primary:.4f} | "
            f"val {primary_metric} {val_primary:.4f}"
        )

        if np.isfinite(val_primary) and val_primary >= best_val_score:
            best_val_score = float(val_primary)
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

    total_train_time = time.time() - start_time
    model.load_state_dict(torch.load(best_ckpt_path, map_location=config.device, weights_only=False))

    final_val_metrics = evaluate_model(model, val_loader, config.device, threshold=config.threshold)
    final_test_metrics = evaluate_model(model, test_loader, config.device, threshold=config.threshold)
    dataset_summary = manifest.get("dataset_summary", {})

    final_info = {
        "best_epoch": best_epoch,
        "run_mode": config.run_mode,
        "dataset_name": dataset_name,
        "task_name": task_spec.task_name,
        "modality": task_spec.modality,
        "task_type": task_spec.task_type,
        "class_names": class_names,
        "num_classes": num_classes,
        "primary_metric": task_spec.primary_metric,
        "primary_metric_name": primary_metric,
        "clinical_goal": task_spec.clinical_goal,
        "default_val_metrics": final_val_metrics,
        "default_test_metrics": final_test_metrics,
        "best_val_metrics": final_val_metrics,
        "best_test_metrics": final_test_metrics,
        "total_train_time": total_train_time,
        "scorecard": {
            "primary_metric_name": primary_metric,
            "best_val_primary": float(final_val_metrics.get(primary_metric, float("nan"))),
            "best_test_primary": float(final_test_metrics.get(primary_metric, float("nan"))),
            "best_epoch": float(best_epoch),
        },
        "auto_policy": config.auto_policy,
        "config": asdict(config),
        "task_spec": task_spec_to_dict(task_spec),
        "dataset_summary": dataset_summary,
    }

    write_run_outputs(
        out_dir=config.out_dir,
        dataset_name=dataset_name,
        final_info=final_info,
        train_log_info=train_log_info,
        val_log_info=val_log_info,
        dataset_summary=dataset_summary,
    )
    print(f"Saved results to {config.out_dir}")


def main():
    args = parse_args()

    task_spec = resolve_task_spec(
        root=ROOT,
        task_name=args.task_name,
        data_root_override=args.data_root,
        split_file_override=args.split_file,
    )

    task_family = infer_task_family(task_spec.task_type)
    if task_family != "classification":
        raise ValueError(
            f"Task '{task_spec.task_name}' has task_type='{task_spec.task_type}', "
            f"resolved family='{task_family}', not classification."
        )

    if not task_spec.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {task_spec.data_root}")
    if not task_spec.split_file.exists():
        raise FileNotFoundError(f"split_file not found: {task_spec.split_file}")

    config = resolve_classification_config(args, task_spec)
    print(f"[AutoPolicy] {json.dumps(config.auto_policy, ensure_ascii=False)}")
    run_classification_experiment(config=config, task_spec=task_spec)


if __name__ == "__main__":
    main()
