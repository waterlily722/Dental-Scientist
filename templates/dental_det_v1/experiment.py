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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.transforms import functional as TF


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


def infer_detection_augmentation_policy(task_spec: TaskSpec) -> Dict[str, bool]:
    allowed = normalize_policy_tokens(list(getattr(task_spec, "augmentation_allowed", []) or []))
    disallowed = normalize_policy_tokens(list(getattr(task_spec, "augmentation_disallowed", []) or []))

    policy = {
        "enable_horizontal_flip": False,
        "enable_color_jitter": False,
        "enable_gaussian_noise": False,
    }

    if allowed:
        policy["enable_horizontal_flip"] = contains_any(allowed, ["flip", "horizontal_flip"])
        policy["enable_color_jitter"] = contains_any(allowed, ["color_jitter", "brightness", "contrast"])
        policy["enable_gaussian_noise"] = contains_any(allowed, ["gaussian_noise", "noise"])

    if contains_any(disallowed, ["left_right_flip", "flip", "quadrant", "tooth_id", "tooth_number"]):
        policy["enable_horizontal_flip"] = False
    if contains_any(disallowed, ["color_jitter", "brightness", "contrast"]):
        policy["enable_color_jitter"] = False
    if contains_any(disallowed, ["gaussian_noise", "noise"]):
        policy["enable_gaussian_noise"] = False

    return policy


def _xyxy_from_bbox(raw_bbox: object) -> Optional[List[float]]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 2:
        return None
    pt1, pt2 = raw_bbox
    if not (
        isinstance(pt1, list)
        and isinstance(pt2, list)
        and len(pt1) == 2
        and len(pt2) == 2
    ):
        return None
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    if (x_max - x_min) < 1 or (y_max - y_min) < 1:
        return None
    return [x_min, y_min, x_max, y_max]


def extract_detection_instances(sample: Dict[str, object], task_spec: TaskSpec) -> Tuple[List[List[float]], List[int]]:
    class_to_idx = {name: idx + 1 for idx, name in enumerate(task_spec.class_names)}
    boxes: List[List[float]] = []
    labels: List[int] = []

    for field_name in ["disease_dict", "tooth_dict", "structure_dict", "therapy_dict"]:
        annotations = sample.get(field_name, {})
        if not isinstance(annotations, dict):
            continue
        for class_name, annotation in annotations.items():
            if class_name not in class_to_idx:
                continue
            if not isinstance(annotation, dict):
                continue
            raw_boxes = annotation.get("bbox", [])
            if not isinstance(raw_boxes, list):
                continue
            for raw_bbox in raw_boxes:
                box = _xyxy_from_bbox(raw_bbox)
                if box is None:
                    continue
                boxes.append(box)
                labels.append(class_to_idx[class_name])
    return boxes, labels


def compute_dataset_summary(manifest: Dict[str, object], task_spec: TaskSpec) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "task_name": task_spec.task_name,
        "dataset_name": str(manifest.get("dataset_name") or task_spec.dataset_name),
        "split_counts": {name: len(entries) for name, entries in manifest.get("splits", {}).items()},
        "num_classes": len(task_spec.class_names),
        "class_names": list(task_spec.class_names),
    }

    instance_counts = {class_name: 0 for class_name in task_spec.class_names}
    split_instance_counts: Dict[str, Dict[str, int]] = {}
    for split_name, entries in manifest.get("splits", {}).items():
        split_counts = {class_name: 0 for class_name in task_spec.class_names}
        for sample in entries:
            _, labels = extract_detection_instances(sample, task_spec)
            for label in labels:
                class_name = task_spec.class_names[label - 1]
                split_counts[class_name] += 1
                instance_counts[class_name] += 1
        split_instance_counts[split_name] = split_counts

    summary["instance_counts"] = instance_counts
    summary["split_instance_counts"] = split_instance_counts
    return summary


def _limit_entries_for_precheck(entries: List[Dict[str, object]], max_samples: int) -> List[Dict[str, object]]:
    return entries[:max_samples]


def maybe_reduce_manifest_for_precheck(manifest: Dict[str, object]) -> Dict[str, object]:
    if not precheck_enabled():
        return manifest
    max_samples = env_int("AI_SCIENTIST_PRECHECK_MAX_SAMPLES", 24)
    reduced = dict(manifest)
    reduced["splits"] = {
        split_name: _limit_entries_for_precheck(list(entries), max_samples)
        for split_name, entries in manifest.get("splits", {}).items()
    }
    return reduced


@dataclass
class DetectionConfig:
    task_name: str
    out_dir: str
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 8
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 512
    use_pretrained: bool = True
    preprocess_mode: str = "identity"
    score_threshold: float = 0.05
    augmentation_policy: Optional[Dict[str, bool]] = None
    auto_policy: Optional[Dict[str, object]] = None
    run_mode: str = "full"


def default_task_name() -> str:
    task_context = load_local_task_context(__file__)
    return resolve_default_task_name("Dental_Radiography", task_context)


def parse_args():
    task_context = load_local_task_context(__file__)
    data_paths = context_data_paths(task_context)

    parser = argparse.ArgumentParser(description="Dental Scientist detection template")
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--task_name", type=str, default=default_task_name())
    parser.add_argument("--data_root", type=str, default=str(data_paths.get("data_root", "")))
    parser.add_argument("--split_file", type=str, default=str(data_paths.get("split_file", "")))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--use_pretrained", type=int, default=1)
    parser.add_argument("--preprocess_mode", type=str, default="auto")
    parser.add_argument("--score_threshold", type=float, default=0.05)
    return parser.parse_args()


def resolve_detection_config(args, task_spec: TaskSpec) -> DetectionConfig:
    preprocess_mode = args.preprocess_mode if args.preprocess_mode != "auto" else infer_preprocess_mode(task_spec)
    run_mode = "precheck" if precheck_enabled() else "full"
    epochs = min(args.epochs, env_int("AI_SCIENTIST_PRECHECK_EPOCHS", 1)) if precheck_enabled() else args.epochs
    batch_size = 1 if precheck_enabled() else args.batch_size
    num_workers = 0 if precheck_enabled() else args.num_workers
    augmentation_policy = infer_detection_augmentation_policy(task_spec)
    auto_policy = {
        "template": "dental_det_v1",
        "baseline": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "run_mode": run_mode,
        "preprocess_mode": preprocess_mode,
        "image_size": args.image_size,
        "augmentation_policy": augmentation_policy,
    }
    return DetectionConfig(
        task_name=task_spec.task_name,
        out_dir=args.out_dir,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=epochs,
        seed=args.seed,
        num_workers=num_workers,
        image_size=args.image_size,
        use_pretrained=bool(args.use_pretrained),
        preprocess_mode=preprocess_mode,
        score_threshold=args.score_threshold,
        augmentation_policy=augmentation_policy,
        auto_policy=auto_policy,
        run_mode=run_mode,
    )


class DentalDetectionDataset(Dataset):
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

        orig_w, orig_h = image.size
        boxes, labels = extract_detection_instances(entry, self.task_spec)

        image = image.resize((self.image_size, self.image_size))
        scale_x = self.image_size / max(orig_w, 1)
        scale_y = self.image_size / max(orig_h, 1)

        resized_boxes: List[List[float]] = []
        for x1, y1, x2, y2 in boxes:
            resized_boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])

        if self.is_train and self.augmentation_policy.get("enable_horizontal_flip", False) and random.random() < 0.5:
            image = ImageOps.mirror(image)
            flipped_boxes: List[List[float]] = []
            for x1, y1, x2, y2 in resized_boxes:
                flipped_boxes.append([self.image_size - x2, y1, self.image_size - x1, y2])
            resized_boxes = flipped_boxes

        image_tensor = TF.to_tensor(image)
        if self.augmentation_policy.get("enable_color_jitter", False) and self.is_train:
            brightness = 1.0 + random.uniform(-0.15, 0.15)
            contrast = 1.0 + random.uniform(-0.15, 0.15)
            image_tensor = TF.adjust_brightness(image_tensor, brightness)
            image_tensor = TF.adjust_contrast(image_tensor, contrast)
        if self.augmentation_policy.get("enable_gaussian_noise", False) and self.is_train:
            image_tensor = torch.clamp(image_tensor + 0.02 * torch.randn_like(image_tensor), 0.0, 1.0)

        boxes_tensor = (
            torch.tensor(resized_boxes, dtype=torch.float32)
            if resized_boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64)
        )

        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes_tensor) > 0
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_model(num_classes: int, use_pretrained: bool):
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if use_pretrained else None
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_pretrained else None
    try:
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )
    except Exception as exc:
        if use_pretrained:
            print(f"[Warn] Failed to load pretrained detection weights, falling back to random init: {exc}")
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )


def make_loaders(config: DetectionConfig, task_spec: TaskSpec):
    manifest = load_manifest_from_file(task_spec.split_file)
    manifest = maybe_reduce_manifest_for_precheck(manifest)
    dataset_summary = compute_dataset_summary(manifest, task_spec)
    manifest["dataset_summary"] = dataset_summary

    train_dataset = DentalDetectionDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("train", [])),
        task_spec=task_spec,
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        is_train=True,
        augmentation_policy=config.augmentation_policy,
    )
    val_dataset = DentalDetectionDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("val", [])),
        task_spec=task_spec,
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        is_train=False,
    )
    test_dataset = DentalDetectionDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("test", [])),
        task_spec=task_spec,
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    return manifest, train_loader, val_loader, test_loader


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def evaluate_detection(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    score_threshold: float,
) -> Dict[str, float]:
    gt_count = {class_id: 0 for class_id in range(1, num_classes + 1)}
    pred_by_class = {class_id: [] for class_id in range(1, num_classes + 1)}
    gt_by_image_class: Dict[Tuple[int, int], Dict[str, object]] = {}

    for image_idx, target in enumerate(targets):
        boxes = target["boxes"].cpu()
        labels = target["labels"].cpu()
        for class_id in range(1, num_classes + 1):
            mask = labels == class_id
            class_boxes = boxes[mask]
            gt_count[class_id] += int(mask.sum().item())
            gt_by_image_class[(image_idx, class_id)] = {
                "boxes": class_boxes,
            }

    for image_idx, pred in enumerate(predictions):
        boxes = pred.get("boxes", torch.zeros((0, 4))).detach().cpu()
        labels = pred.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu()
        scores = pred.get("scores", torch.zeros((0,), dtype=torch.float32)).detach().cpu()
        keep = scores >= score_threshold
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        for box, label, score in zip(boxes, labels, scores):
            pred_by_class[int(label.item())].append((image_idx, float(score.item()), box))

    ap50_values: List[float] = []
    map_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []

    for class_id in range(1, num_classes + 1):
        if gt_count[class_id] == 0:
            continue
        entries = sorted(pred_by_class[class_id], key=lambda x: x[1], reverse=True)
        ap_per_threshold: List[float] = []
        precision_at_50 = 0.0
        recall_at_50 = 0.0

        for threshold in np.arange(0.5, 1.0, 0.05):
            matched = {
                key: torch.zeros((len(value["boxes"]),), dtype=torch.bool)
                for key, value in gt_by_image_class.items()
                if key[1] == class_id
            }
            tp = np.zeros((len(entries),), dtype=np.float32)
            fp = np.zeros((len(entries),), dtype=np.float32)

            for idx, (image_idx, _score, box) in enumerate(entries):
                gt_key = (image_idx, class_id)
                gt_info = gt_by_image_class.get(gt_key, {"boxes": torch.zeros((0, 4))})
                gt_boxes = gt_info["boxes"]
                if gt_boxes.numel() == 0:
                    fp[idx] = 1.0
                    continue
                ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_idx = (ious.max(dim=0) if ious.numel() else (torch.tensor(0.0), torch.tensor(0)))
                if float(best_iou.item()) >= float(threshold) and not matched[gt_key][int(best_idx.item())]:
                    tp[idx] = 1.0
                    matched[gt_key][int(best_idx.item())] = True
                else:
                    fp[idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / max(gt_count[class_id], 1)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
            ap = _compute_ap(recalls, precisions) if len(recalls) > 0 else 0.0
            ap_per_threshold.append(ap)

            if abs(threshold - 0.5) < 1e-8:
                precision_at_50 = float(precisions[-1]) if len(precisions) > 0 else 0.0
                recall_at_50 = float(recalls[-1]) if len(recalls) > 0 else 0.0

        ap50_values.append(ap_per_threshold[0])
        map_values.append(float(np.mean(ap_per_threshold)))
        precision_values.append(precision_at_50)
        recall_values.append(recall_at_50)

    return {
        "mAP": float(np.mean(map_values)) if map_values else 0.0,
        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall": float(np.mean(recall_values)) if recall_values else 0.0,
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_TRAIN_BATCHES", 2) if precheck_enabled() else None

    for batch_idx, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        losses = model(images, targets)
        loss = sum(losses.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes: int, score_threshold: float):
    model.eval()
    predictions: List[Dict[str, torch.Tensor]] = []
    targets_all: List[Dict[str, torch.Tensor]] = []
    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_EVAL_BATCHES", 2) if precheck_enabled() else None

    for batch_idx, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        outputs = model(images)
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets_all.extend([{k: v.detach().cpu() for k, v in target.items()} for target in targets])
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    return evaluate_detection(predictions, targets_all, num_classes=num_classes, score_threshold=score_threshold)


def run_detection_experiment(config: DetectionConfig, task_spec: TaskSpec):
    seed_everything(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)

    manifest, train_loader, val_loader, test_loader = make_loaders(config, task_spec)
    dataset_name = str(manifest.get("dataset_name") or task_spec.dataset_name)
    num_classes = len(task_spec.class_names)
    requested_report_metrics = ["mAP", "mAP50", "recall", "precision"]

    print(
        f"Dataset {dataset_name}: modality={task_spec.modality}, task_type={task_spec.task_type}, "
        f"classes={list(task_spec.class_names)}, num_classes={num_classes}, split_counts={manifest.get('dataset_summary', {}).get('split_counts', {})}"
    )

    model = build_model(num_classes=num_classes + 1, use_pretrained=config.use_pretrained).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_log_info = []
    val_log_info = []
    best_epoch = -1
    best_val_primary = -float("inf")
    best_ckpt_path = os.path.join(config.out_dir, "best_model.pth")

    start_time = time.time()
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config.device)
        val_metrics = evaluate_model(model, val_loader, config.device, num_classes, config.score_threshold)
        scheduler.step()

        train_log = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "primary_metric": float(val_metrics["mAP"]),
        }
        train_log.update({name: float("nan") for name in requested_report_metrics})
        train_log_info.append(train_log)

        val_log = {"epoch": epoch, "primary_metric": float(val_metrics["mAP"])}
        val_log.update({name: float(val_metrics.get(name, float("nan"))) for name in requested_report_metrics})
        val_log_info.append(val_log)

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val mAP {val_metrics['mAP']:.4f} | val mAP50 {val_metrics['mAP50']:.4f}"
        )

        if np.isfinite(val_metrics["mAP"]) and val_metrics["mAP"] >= best_val_primary:
            best_val_primary = float(val_metrics["mAP"])
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

    total_train_time = time.time() - start_time

    model.load_state_dict(torch.load(best_ckpt_path, map_location=config.device, weights_only=False))
    best_val_metrics = evaluate_model(model, val_loader, config.device, num_classes, config.score_threshold)
    best_test_metrics = evaluate_model(model, test_loader, config.device, num_classes, config.score_threshold)

    final_info = {
        "best_epoch": best_epoch,
        "run_mode": config.run_mode,
        "dataset_name": dataset_name,
        "task_name": task_spec.task_name,
        "modality": task_spec.modality,
        "task_type": task_spec.task_type,
        "label_level": task_spec.label_level,
        "clinical_goal": task_spec.clinical_goal,
        "primary_metric": task_spec.primary_metric,
        "secondary_metrics": task_spec.secondary_metrics,
        "class_names": list(task_spec.class_names),
        "num_classes": num_classes,
        "primary_metric_name": "mAP",
        "requested_report_metrics": requested_report_metrics,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "best_val_reported_metrics": dict(best_val_metrics),
        "best_test_reported_metrics": dict(best_test_metrics),
        "scorecard": {
            "primary_metric_name": "mAP",
            "best_val_primary": float(best_val_metrics.get("mAP", float("nan"))),
            "best_test_primary": float(best_test_metrics.get("mAP", float("nan"))),
            "best_epoch": float(best_epoch),
            "best_test_mAP50": float(best_test_metrics.get("mAP50", float("nan"))),
            "best_test_recall": float(best_test_metrics.get("recall", float("nan"))),
            "best_test_precision": float(best_test_metrics.get("precision", float("nan"))),
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

    task_family = infer_task_family(task_spec.task_type)
    if task_family not in {"detection", "unknown"} and "detection" not in str(task_spec.task_type).lower():
        raise ValueError(
            f"Task '{task_spec.task_name}' has task_type='{task_spec.task_type}', which is not suitable for the detection template."
        )

    if not task_spec.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {task_spec.data_root}")
    if not task_spec.split_file.exists():
        raise FileNotFoundError(f"split_file not found: {task_spec.split_file}")

    config = resolve_detection_config(args, task_spec)
    print(f"[AutoPolicy] {json.dumps(config.auto_policy, ensure_ascii=False)}")
    run_detection_experiment(config=config, task_spec=task_spec)


if __name__ == "__main__":
    main()
