import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset


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
    load_manifest_from_file,
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
    infer_preprocess_mode,
    infer_task_family,
    task_spec_to_dict,
)


def precheck_enabled() -> bool:
    return env_flag("AI_SCIENTIST_PRECHECK")


def infer_segmentation_augmentation_policy(task_spec: TaskSpec) -> Dict[str, bool]:
    allowed = normalize_policy_tokens(list(getattr(task_spec, "augmentation_allowed", []) or []))
    disallowed = normalize_policy_tokens(list(getattr(task_spec, "augmentation_disallowed", []) or []))

    policy = {
        "enable_horizontal_flip": False,
        "enable_color_jitter": False,
    }
    if allowed:
        policy["enable_horizontal_flip"] = contains_any(allowed, ["flip", "horizontal_flip"])
        policy["enable_color_jitter"] = contains_any(allowed, ["color_jitter", "brightness", "contrast"])
    if contains_any(disallowed, ["flip", "left_right_flip", "tooth_number", "quadrant"]):
        policy["enable_horizontal_flip"] = False
    if contains_any(disallowed, ["color_jitter", "brightness", "contrast"]):
        policy["enable_color_jitter"] = False
    return policy


def _polygon_to_tuples(polygon: object) -> List[Tuple[float, float]]:
    if not isinstance(polygon, list):
        return []
    points: List[Tuple[float, float]] = []
    for point in polygon:
        if isinstance(point, list) and len(point) == 2:
            points.append((float(point[0]), float(point[1])))
    return points


def extract_segmentation_polygons(sample: Dict[str, object], task_spec: TaskSpec) -> List[List[Tuple[float, float]]]:
    polygons: List[List[Tuple[float, float]]] = []
    target_set = set(task_spec.class_names)
    for field_name in ["disease_dict", "tooth_dict", "structure_dict", "therapy_dict"]:
        annotations = sample.get(field_name, {})
        if not isinstance(annotations, dict):
            continue
        for class_name, annotation in annotations.items():
            if class_name not in target_set:
                continue
            if not isinstance(annotation, dict):
                continue
            segmentations = annotation.get("segmentation", [])
            if not isinstance(segmentations, list):
                continue
            for polygon in segmentations:
                points = _polygon_to_tuples(polygon)
                if len(points) >= 3:
                    polygons.append(points)
    return polygons


def compute_dataset_summary(manifest: Dict[str, object], task_spec: TaskSpec) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "task_name": task_spec.task_name,
        "dataset_name": str(manifest.get("dataset_name") or task_spec.dataset_name),
        "split_counts": {name: len(entries) for name, entries in manifest.get("splits", {}).items()},
        "class_names": list(task_spec.class_names),
    }
    polygon_counts = {"train": 0, "val": 0, "test": 0}
    for split_name, entries in manifest.get("splits", {}).items():
        count = 0
        for sample in entries:
            count += len(extract_segmentation_polygons(sample, task_spec))
        polygon_counts[split_name] = count
    summary["polygon_counts"] = polygon_counts
    return summary


def maybe_reduce_manifest_for_precheck(manifest: Dict[str, object]) -> Dict[str, object]:
    if not precheck_enabled():
        return manifest
    max_samples = env_int("AI_SCIENTIST_PRECHECK_MAX_SAMPLES", 16)
    reduced = dict(manifest)
    reduced["splits"] = {
        split_name: list(entries)[:max_samples]
        for split_name, entries in manifest.get("splits", {}).items()
    }
    return reduced


def default_task_name() -> str:
    task_context = load_local_task_context(__file__)
    return resolve_default_task_name("AlphaDent", task_context)


def parse_args():
    task_context = load_local_task_context(__file__)
    data_paths = context_data_paths(task_context)

    parser = argparse.ArgumentParser(description="Dental Scientist segmentation template")
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--task_name", type=str, default=default_task_name())
    parser.add_argument("--data_root", type=str, default=str(data_paths.get("data_root", "")))
    parser.add_argument("--split_file", type=str, default=str(data_paths.get("split_file", "")))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--preprocess_mode", type=str, default="auto")
    return parser.parse_args()


@dataclass
class SegmentationConfig:
    task_name: str
    out_dir: str
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 12
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 256
    preprocess_mode: str = "identity"
    augmentation_policy: Optional[Dict[str, bool]] = None
    auto_policy: Optional[Dict[str, object]] = None
    run_mode: str = "full"


def resolve_segmentation_config(args, task_spec: TaskSpec) -> SegmentationConfig:
    preprocess_mode = args.preprocess_mode if args.preprocess_mode != "auto" else infer_preprocess_mode(task_spec)
    run_mode = "precheck" if precheck_enabled() else "full"
    epochs = min(args.epochs, env_int("AI_SCIENTIST_PRECHECK_EPOCHS", 1)) if precheck_enabled() else args.epochs
    batch_size = min(args.batch_size, 2) if precheck_enabled() else args.batch_size
    num_workers = 0 if precheck_enabled() else args.num_workers
    augmentation_policy = infer_segmentation_augmentation_policy(task_spec)
    auto_policy = {
        "template": "dental_seg_v1",
        "baseline": "simple_unet",
        "run_mode": run_mode,
        "preprocess_mode": preprocess_mode,
        "image_size": args.image_size,
        "augmentation_policy": augmentation_policy,
    }
    return SegmentationConfig(
        task_name=task_spec.task_name,
        out_dir=args.out_dir,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=epochs,
        seed=args.seed,
        num_workers=num_workers,
        image_size=args.image_size,
        preprocess_mode=preprocess_mode,
        augmentation_policy=augmentation_policy,
        auto_policy=auto_policy,
        run_mode=run_mode,
    )


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._block(3, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block(64, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


class DentalSegmentationDataset(Dataset):
    def __init__(
        self,
        root: Path,
        entries: List[Dict[str, object]],
        task_spec: TaskSpec,
        image_size: int,
        preprocess_mode: str,
        is_train: bool,
        augmentation_policy: Optional[Dict[str, bool]] = None,
    ):
        self.root = root
        self.entries = entries
        self.task_spec = task_spec
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.is_train = is_train
        self.augmentation_policy = augmentation_policy or {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image_path = self.root / sample_image_relpath(entry)
        image = Image.open(image_path).convert("RGB")
        image = radiograph_preprocess(image, self.preprocess_mode)
        polygons = extract_segmentation_polygons(entry, self.task_spec)

        orig_w, orig_h = image.size
        mask = Image.new("L", (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            draw.polygon(polygon, outline=1, fill=1)

        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        if self.is_train and self.augmentation_policy.get("enable_horizontal_flip", False) and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        if self.is_train and self.augmentation_policy.get("enable_color_jitter", False):
            factor = 1.0 + random.uniform(-0.15, 0.15)
            image_tensor = torch.clamp(image_tensor * factor, 0.0, 1.0)
        mask_tensor = torch.from_numpy((np.array(mask, dtype=np.uint8) > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


def make_loaders(config: SegmentationConfig, task_spec: TaskSpec):
    manifest = load_manifest_from_file(task_spec.split_file)
    manifest = maybe_reduce_manifest_for_precheck(manifest)
    manifest["dataset_summary"] = compute_dataset_summary(manifest, task_spec)

    train_dataset = DentalSegmentationDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("train", [])),
        task_spec=task_spec,
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        is_train=True,
        augmentation_policy=config.augmentation_policy,
    )
    val_dataset = DentalSegmentationDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("val", [])),
        task_spec=task_spec,
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        is_train=False,
    )
    test_dataset = DentalSegmentationDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("test", [])),
        task_spec=task_spec,
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return manifest, train_loader, val_loader, test_loader


def dice_score(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.reshape(prob.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    return (2.0 * inter + eps) / (union + eps)


def iou_score(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.reshape(prob.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1) - inter
    return (inter + eps) / (union + eps)


def boundary_f1(prob_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    prob_edges = torch.abs(prob_mask[:, :, 1:, :] - prob_mask[:, :, :-1, :]).sum() + torch.abs(prob_mask[:, :, :, 1:] - prob_mask[:, :, :, :-1]).sum()
    target_edges = torch.abs(target_mask[:, :, 1:, :] - target_mask[:, :, :-1, :]).sum() + torch.abs(target_mask[:, :, :, 1:] - target_mask[:, :, :, :-1]).sum()
    overlap = torch.minimum(prob_edges, target_edges)
    denom = (prob_edges + target_edges).clamp(min=1e-6)
    return float((2.0 * overlap / denom).item())


def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = 1.0 - dice_score(prob, target).mean()
    return bce + dice


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_TRAIN_BATCHES", 2) if precheck_enabled() else None

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = compute_loss(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    dice_values: List[float] = []
    iou_values: List[float] = []
    boundary_values: List[float] = []
    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_EVAL_BATCHES", 2) if precheck_enabled() else None

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        probs = (torch.sigmoid(logits) > 0.5).float()

        dice_values.extend(dice_score(probs, masks).cpu().tolist())
        iou_values.extend(iou_score(probs, masks).cpu().tolist())
        boundary_values.append(boundary_f1(probs, masks))

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    return {
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "iou": float(np.mean(iou_values)) if iou_values else 0.0,
        "boundary_f1": float(np.mean(boundary_values)) if boundary_values else 0.0,
    }


def run_segmentation_experiment(config: SegmentationConfig, task_spec: TaskSpec):
    seed_everything(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)

    manifest, train_loader, val_loader, test_loader = make_loaders(config, task_spec)
    dataset_name = str(manifest.get("dataset_name") or task_spec.dataset_name)
    requested_report_metrics = ["dice", "iou", "boundary_f1"]

    print(
        f"Dataset {dataset_name}: modality={task_spec.modality}, task_type={task_spec.task_type}, "
        f"split_counts={manifest.get('dataset_summary', {}).get('split_counts', {})}"
    )

    model = SimpleUNet().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_log_info = []
    val_log_info = []
    best_epoch = -1
    best_val_dice = -float("inf")
    best_ckpt_path = os.path.join(config.out_dir, "best_model.pth")

    start_time = time.time()
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config.device)
        val_metrics = evaluate_model(model, val_loader, config.device)
        scheduler.step()

        train_log_info.append(
            {
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "primary_metric": float(val_metrics["dice"]),
                "dice": float("nan"),
                "iou": float("nan"),
                "boundary_f1": float("nan"),
            }
        )
        val_log_info.append(
            {
                "epoch": epoch,
                "primary_metric": float(val_metrics["dice"]),
                "dice": float(val_metrics["dice"]),
                "iou": float(val_metrics["iou"]),
                "boundary_f1": float(val_metrics["boundary_f1"]),
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val dice {val_metrics['dice']:.4f} | val iou {val_metrics['iou']:.4f}"
        )

        if np.isfinite(val_metrics["dice"]) and val_metrics["dice"] >= best_val_dice:
            best_val_dice = float(val_metrics["dice"])
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

    total_train_time = time.time() - start_time
    model.load_state_dict(torch.load(best_ckpt_path, map_location=config.device, weights_only=False))
    best_val_metrics = evaluate_model(model, val_loader, config.device)
    best_test_metrics = evaluate_model(model, test_loader, config.device)

    final_info = {
        "best_epoch": best_epoch,
        "run_mode": config.run_mode,
        "dataset_name": dataset_name,
        "task_name": task_spec.task_name,
        "modality": task_spec.modality,
        "task_type": task_spec.task_type,
        "label_level": task_spec.label_level,
        "clinical_goal": task_spec.clinical_goal,
        "primary_metric": "dice",
        "secondary_metrics": ["iou", "boundary_f1"],
        "primary_metric_name": "dice",
        "requested_report_metrics": requested_report_metrics,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "best_val_reported_metrics": dict(best_val_metrics),
        "best_test_reported_metrics": dict(best_test_metrics),
        "scorecard": {
            "primary_metric_name": "dice",
            "best_val_primary": float(best_val_metrics.get("dice", float("nan"))),
            "best_test_primary": float(best_test_metrics.get("dice", float("nan"))),
            "best_epoch": float(best_epoch),
            "best_test_iou": float(best_test_metrics.get("iou", float("nan"))),
            "best_test_boundary_f1": float(best_test_metrics.get("boundary_f1", float("nan"))),
        },
        "total_train_time": total_train_time,
        "auto_policy": config.auto_policy,
        "config": asdict(config),
        "task_spec": task_spec_to_dict(task_spec),
        "dataset_summary": manifest.get("dataset_summary", {}),
    }

    write_run_outputs(
        out_dir=config.out_dir,
        dataset_name=dataset_name,
        final_info=final_info,
        train_log_info=train_log_info,
        val_log_info=val_log_info,
        dataset_summary=manifest.get("dataset_summary", {}),
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

    task_type_lower = str(task_spec.task_type).lower()
    if "segmentation" not in task_type_lower:
        raise ValueError(
            f"Task '{task_spec.task_name}' has task_type='{task_spec.task_type}', which is not suitable for the segmentation template."
        )
    if not task_spec.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {task_spec.data_root}")
    if not task_spec.split_file.exists():
        raise FileNotFoundError(f"split_file not found: {task_spec.split_file}")

    config = resolve_segmentation_config(args, task_spec)
    print(f"[AutoPolicy] {json.dumps(config.auto_policy, ensure_ascii=False)}")
    run_segmentation_experiment(config=config, task_spec=task_spec)


if __name__ == "__main__":
    main()
