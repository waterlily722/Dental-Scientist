import argparse
import json
import math
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
from PIL import Image, ImageOps
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
from core.task_spec import TaskSpec, infer_preprocess_mode, task_spec_to_dict  # noqa: E402


def precheck_enabled() -> bool:
    return env_flag("AI_SCIENTIST_PRECHECK")


def infer_keypoint_augmentation_policy(task_spec: TaskSpec) -> Dict[str, bool]:
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


def _bbox_center(raw_bbox: object) -> Optional[Tuple[float, float]]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 2:
        return None
    pt1, pt2 = raw_bbox
    if not (isinstance(pt1, list) and isinstance(pt2, list) and len(pt1) == 2 and len(pt2) == 2):
        return None
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def extract_keypoints(sample: Dict[str, object]) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    tooth_dict = sample.get("tooth_dict", {})
    if isinstance(tooth_dict, dict):
        for _tooth_name, annotation in tooth_dict.items():
            if not isinstance(annotation, dict):
                continue
            bboxes = annotation.get("bbox", [])
            if not isinstance(bboxes, list):
                continue
            for bbox in bboxes:
                center = _bbox_center(bbox)
                if center is not None:
                    centers.append(center)
    return centers


def compute_dataset_summary(manifest: Dict[str, object], task_spec: TaskSpec) -> Dict[str, object]:
    summary = {
        "task_name": task_spec.task_name,
        "dataset_name": str(manifest.get("dataset_name") or task_spec.dataset_name),
        "split_counts": {name: len(entries) for name, entries in manifest.get("splits", {}).items()},
        "class_names": list(task_spec.class_names),
    }
    keypoint_counts = {}
    for split_name, entries in manifest.get("splits", {}).items():
        keypoint_counts[split_name] = int(sum(len(extract_keypoints(sample)) for sample in entries))
    summary["keypoint_counts"] = keypoint_counts
    return summary


def maybe_reduce_manifest_for_precheck(manifest: Dict[str, object]) -> Dict[str, object]:
    if not precheck_enabled():
        return manifest
    max_samples = env_int("AI_SCIENTIST_PRECHECK_MAX_SAMPLES", 16)
    reduced = dict(manifest)
    reduced["splits"] = {split_name: list(entries)[:max_samples] for split_name, entries in manifest.get("splits", {}).items()}
    return reduced


def default_task_name() -> str:
    task_context = load_local_task_context(__file__)
    return resolve_default_task_name("Dentistry_Computer_Vision_Dataset", task_context)


def parse_args():
    task_context = load_local_task_context(__file__)
    data_paths = context_data_paths(task_context)

    parser = argparse.ArgumentParser(description="Dental Scientist keypoint template")
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
    parser.add_argument("--heatmap_sigma", type=float, default=2.5)
    return parser.parse_args()


@dataclass
class KeypointConfig:
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
    heatmap_sigma: float = 2.5
    augmentation_policy: Optional[Dict[str, bool]] = None
    auto_policy: Optional[Dict[str, object]] = None
    run_mode: str = "full"


def resolve_keypoint_config(args, task_spec: TaskSpec) -> KeypointConfig:
    preprocess_mode = args.preprocess_mode if args.preprocess_mode != "auto" else infer_preprocess_mode(task_spec)
    run_mode = "precheck" if precheck_enabled() else "full"
    epochs = min(args.epochs, env_int("AI_SCIENTIST_PRECHECK_EPOCHS", 1)) if precheck_enabled() else args.epochs
    batch_size = min(args.batch_size, 2) if precheck_enabled() else args.batch_size
    num_workers = 0 if precheck_enabled() else args.num_workers
    augmentation_policy = infer_keypoint_augmentation_policy(task_spec)
    auto_policy = {
        "template": "dental_keypoint_v1",
        "baseline": "simple_heatmap_net",
        "run_mode": run_mode,
        "preprocess_mode": preprocess_mode,
        "image_size": args.image_size,
        "heatmap_sigma": args.heatmap_sigma,
        "augmentation_policy": augmentation_policy,
    }
    return KeypointConfig(
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
        heatmap_sigma=args.heatmap_sigma,
        augmentation_policy=augmentation_policy,
        auto_policy=auto_policy,
        run_mode=run_mode,
    )


class SimpleHeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _make_gaussian_heatmap(image_size: int, centers: List[Tuple[float, float]], sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    for cx, cy in centers:
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        heatmap = np.maximum(heatmap, np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32))
    return heatmap


class DentalKeypointDataset(Dataset):
    def __init__(
        self,
        root: Path,
        entries: List[Dict[str, object]],
        image_size: int,
        preprocess_mode: str,
        sigma: float,
        is_train: bool,
        augmentation_policy: Optional[Dict[str, bool]] = None,
    ):
        self.root = root
        self.entries = entries
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.sigma = sigma
        self.is_train = is_train
        self.augmentation_policy = augmentation_policy or {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image_path = self.root / sample_image_relpath(entry)
        image = Image.open(image_path).convert("RGB")
        image = radiograph_preprocess(image, self.preprocess_mode)
        centers = extract_keypoints(entry)

        orig_w, orig_h = image.size
        sx = self.image_size / max(orig_w, 1)
        sy = self.image_size / max(orig_h, 1)
        resized_centers = [(cx * sx, cy * sy) for cx, cy in centers]

        image = image.resize((self.image_size, self.image_size))
        if self.is_train and self.augmentation_policy.get("enable_horizontal_flip", False) and random.random() < 0.5:
            image = ImageOps.mirror(image)
            resized_centers = [(self.image_size - cx, cy) for cx, cy in resized_centers]

        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        if self.is_train and self.augmentation_policy.get("enable_color_jitter", False):
            factor = 1.0 + random.uniform(-0.15, 0.15)
            image_tensor = torch.clamp(image_tensor * factor, 0.0, 1.0)

        heatmap = _make_gaussian_heatmap(self.image_size, resized_centers, sigma=self.sigma)
        target_heatmap = torch.from_numpy(heatmap).unsqueeze(0)
        points_tensor = torch.tensor(resized_centers, dtype=torch.float32) if resized_centers else torch.zeros((0, 2), dtype=torch.float32)
        return image_tensor, target_heatmap, points_tensor


def collate_fn(batch):
    images, heatmaps, points = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(heatmaps, dim=0), list(points)


def make_loaders(config: KeypointConfig, task_spec: TaskSpec):
    manifest = load_manifest_from_file(task_spec.split_file)
    manifest = maybe_reduce_manifest_for_precheck(manifest)
    manifest["dataset_summary"] = compute_dataset_summary(manifest, task_spec)

    train_dataset = DentalKeypointDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("train", [])),
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        sigma=config.heatmap_sigma,
        is_train=True,
        augmentation_policy=config.augmentation_policy,
    )
    val_dataset = DentalKeypointDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("val", [])),
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        sigma=config.heatmap_sigma,
        is_train=False,
    )
    test_dataset = DentalKeypointDataset(
        root=task_spec.data_root,
        entries=list(manifest.get("splits", {}).get("test", [])),
        image_size=config.image_size,
        preprocess_mode=config.preprocess_mode,
        sigma=config.heatmap_sigma,
        is_train=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)
    return manifest, train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_TRAIN_BATCHES", 2) if precheck_enabled() else None
    for batch_idx, (images, heatmaps, _points) in enumerate(loader):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        preds = model(images)
        loss = F.mse_loss(preds, heatmaps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    return {"loss": total_loss / max(total_batches, 1)}


def _heatmap_peak(heatmap: torch.Tensor) -> Tuple[float, float]:
    flat_idx = int(torch.argmax(heatmap).item())
    h, w = heatmap.shape
    y = flat_idx // w
    x = flat_idx % w
    return float(x), float(y)


@torch.no_grad()
def evaluate_model(model, loader, device, image_size: int):
    model.eval()
    distances: List[float] = []
    pck_hits = 0
    total_points = 0
    max_batches = env_int("AI_SCIENTIST_PRECHECK_MAX_EVAL_BATCHES", 2) if precheck_enabled() else None
    threshold = 0.05 * image_size

    for batch_idx, (images, _heatmaps, points) in enumerate(loader):
        images = images.to(device)
        preds = torch.sigmoid(model(images)).cpu()
        for pred_heatmap, gt_points in zip(preds, points):
            if gt_points.numel() == 0:
                continue
            pred_x, pred_y = _heatmap_peak(pred_heatmap[0])
            gt_center = gt_points.mean(dim=0)
            dist = float(torch.norm(gt_center - torch.tensor([pred_x, pred_y])).item())
            distances.append(dist)
            total_points += 1
            if dist <= threshold:
                pck_hits += 1
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    mean_distance = float(np.mean(distances)) if distances else float("nan")
    pck = float(pck_hits / max(total_points, 1))
    return {
        "mean_distance": mean_distance,
        "pck@0.05": pck,
        "score": float(1.0 / (1.0 + mean_distance)) if np.isfinite(mean_distance) else 0.0,
    }


def run_keypoint_experiment(config: KeypointConfig, task_spec: TaskSpec):
    seed_everything(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)

    manifest, train_loader, val_loader, test_loader = make_loaders(config, task_spec)
    dataset_name = str(manifest.get("dataset_name") or task_spec.dataset_name)
    requested_report_metrics = ["mean_distance", "pck@0.05"]

    print(
        f"Dataset {dataset_name}: modality={task_spec.modality}, task_type={task_spec.task_type}, "
        f"split_counts={manifest.get('dataset_summary', {}).get('split_counts', {})}"
    )

    model = SimpleHeatmapNet().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_log_info = []
    val_log_info = []
    best_epoch = -1
    best_val_score = -float("inf")
    best_ckpt_path = os.path.join(config.out_dir, "best_model.pth")

    start_time = time.time()
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config.device)
        val_metrics = evaluate_model(model, val_loader, config.device, config.image_size)
        scheduler.step()

        train_log_info.append(
            {
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "primary_metric": float(val_metrics["score"]),
                "mean_distance": float("nan"),
                "pck@0.05": float("nan"),
            }
        )
        val_log_info.append(
            {
                "epoch": epoch,
                "primary_metric": float(val_metrics["score"]),
                "mean_distance": float(val_metrics["mean_distance"]),
                "pck@0.05": float(val_metrics["pck@0.05"]),
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val score {val_metrics['score']:.4f} | val distance {val_metrics['mean_distance']:.2f}"
        )

        if np.isfinite(val_metrics["score"]) and val_metrics["score"] >= best_val_score:
            best_val_score = float(val_metrics["score"])
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

    total_train_time = time.time() - start_time
    model.load_state_dict(torch.load(best_ckpt_path, map_location=config.device, weights_only=False))
    best_val_metrics = evaluate_model(model, val_loader, config.device, config.image_size)
    best_test_metrics = evaluate_model(model, test_loader, config.device, config.image_size)

    final_info = {
        "best_epoch": best_epoch,
        "run_mode": config.run_mode,
        "dataset_name": dataset_name,
        "task_name": task_spec.task_name,
        "modality": task_spec.modality,
        "task_type": task_spec.task_type,
        "label_level": task_spec.label_level,
        "clinical_goal": task_spec.clinical_goal,
        "primary_metric": "score",
        "secondary_metrics": requested_report_metrics,
        "primary_metric_name": "score",
        "requested_report_metrics": requested_report_metrics,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "best_val_reported_metrics": dict(best_val_metrics),
        "best_test_reported_metrics": dict(best_test_metrics),
        "scorecard": {
            "primary_metric_name": "score",
            "best_val_primary": float(best_val_metrics.get("score", float("nan"))),
            "best_test_primary": float(best_test_metrics.get("score", float("nan"))),
            "best_epoch": float(best_epoch),
            "best_test_mean_distance": float(best_test_metrics.get("mean_distance", float("nan"))),
            "best_test_pck@0.05": float(best_test_metrics.get("pck@0.05", float("nan"))),
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

    if "keypoint" not in str(task_spec.task_type).lower() and "number" not in str(task_spec.target_name).lower() and "tooth" not in str(task_spec.target_name).lower():
        print(f"[Warn] task_type='{task_spec.task_type}' is not explicitly keypoint-specific; using tooth bbox centers as landmark targets.")
    if not task_spec.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {task_spec.data_root}")
    if not task_spec.split_file.exists():
        raise FileNotFoundError(f"split_file not found: {task_spec.split_file}")

    config = resolve_keypoint_config(args, task_spec)
    print(f"[AutoPolicy] {json.dumps(config.auto_policy, ensure_ascii=False)}")
    run_keypoint_experiment(config=config, task_spec=task_spec)


if __name__ == "__main__":
    main()
