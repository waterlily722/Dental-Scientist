import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DATASET_NAME = "dental_pa"
CLASS_NAMES = ["healthy", "caries"]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "data").exists() and (candidate / "templates").exists():
            return candidate
    for candidate in [current.parent, *current.parents]:
        if (candidate / "data" / "dental_caries_classificationv3").exists():
            return candidate
    raise FileNotFoundError("Unable to infer repository root containing data directory")


def default_data_root() -> Path:
    return repo_root() / "data" / "dental_caries_classificationv3"


def split_manifest_path(root: Path) -> Path:
    return root / "splits.json"


def scan_images(root: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append(
                    {
                        "path": image_path.relative_to(root).as_posix(),
                        "label": CLASS_TO_LABEL[class_name],
                        "class_name": class_name,
                    }
                )
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples


def stratified_split(samples: List[Dict[str, object]], seed: int, val_ratio: float, test_ratio: float) -> Dict[str, object]:
    per_class: Dict[int, List[Dict[str, object]]] = {0: [], 1: []}
    for sample in samples:
        per_class[int(sample["label"])].append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for items in per_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        while n_test + n_val >= n and n > 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        n_train = max(1, n - n_val - n_test)
        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val : n_train + n_val + n_test]
        if len(train_items) + len(val_items) + len(test_items) < n:
            train_items = items[: n - len(val_items) - len(test_items)]
        splits["train"].extend(train_items)
        splits["val"].extend(val_items)
        splits["test"].extend(test_items)

    for key in splits:
        rng.shuffle(splits[key])

    return {
        "dataset_name": DATASET_NAME,
        "seed": seed,
        "class_names": CLASS_NAMES,
        "class_to_label": CLASS_TO_LABEL,
        "counts": {key: len(value) for key, value in splits.items()},
        "splits": splits,
    }


def load_or_create_manifest(root: Path, seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, object]:
    manifest_path = split_manifest_path(root)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    manifest = stratified_split(scan_images(root), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


class DentalRadiographDataset(Dataset):
    def __init__(self, root: Path, entries: List[Dict[str, object]], transform=None):
        self.root = root
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image_path = self.root / entry["path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(entry["label"]), dtype=torch.long)
        return image, label


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_transform, eval_transform


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Unexpected MobileNetV3 classifier head")
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
    return model


def make_loaders(root: Path, batch_size: int, num_workers: int, seed: int):
    manifest = load_or_create_manifest(root, seed=seed)
    train_transform, eval_transform = build_transforms()

    train_dataset = DentalRadiographDataset(root, manifest["splits"]["train"], transform=train_transform)
    val_dataset = DentalRadiographDataset(root, manifest["splits"]["val"], transform=eval_transform)
    test_dataset = DentalRadiographDataset(root, manifest["splits"]["test"], transform=eval_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return manifest, train_loader, val_loader, test_loader


def compute_class_weights(entries: List[Dict[str, object]]) -> torch.Tensor:
    counts = np.bincount([int(entry["label"]) for entry in entries], minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


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


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    auc = auc_score(y_true, y_prob)

    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end == 1.0:
            mask = (y_prob >= start) & (y_prob <= end)
        else:
            mask = (y_prob >= start) & (y_prob < end)
        if not np.any(mask):
            continue
        bin_confidence = float(y_prob[mask].mean())
        bin_accuracy = float((y_true[mask] == y_pred[mask]).mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc": float(auc),
        "ece": float(ece),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    y_true: List[int] = []
    y_prob: List[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        predictions = logits.detach().argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_examples += int(labels.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(probabilities.detach().cpu().numpy().tolist())

    metrics = binary_metrics(np.array(y_true), np.array(y_prob))
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["accuracy"] = total_correct / max(total_examples, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    y_true: List[int] = []
    y_prob: List[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_examples += int(labels.size(0))
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probabilities.cpu().numpy().tolist())

    metrics = binary_metrics(np.array(y_true), np.array(y_prob))
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["accuracy"] = total_correct / max(total_examples, 1)
    return metrics


@dataclass
class Config:
    data_root: str
    out_dir: str
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_run(config: Config):
    seed_everything(config.seed)
    root = Path(config.data_root)
    os.makedirs(config.out_dir, exist_ok=True)

    manifest, train_loader, val_loader, test_loader = make_loaders(
        root=root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    model = build_model(num_classes=2).to(config.device)
    class_weights = compute_class_weights(manifest["splits"]["train"]).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_log_info = []
    val_log_info = []
    best_val_auc = -math.inf
    best_val_metrics = None
    best_test_metrics = None
    best_epoch = -1

    start_time = time.time()
    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, config.device)
        val_metrics = evaluate(model, val_loader, criterion, config.device)
        scheduler.step()

        train_log_info.append(
            {
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "acc": train_metrics["accuracy"],
                "auc": train_metrics["auc"],
                "f1": train_metrics["f1"],
            }
        )
        val_log_info.append(
            {
                "epoch": epoch,
                "loss": val_metrics["loss"],
                "acc": val_metrics["accuracy"],
                "auc": val_metrics["auc"],
                "f1": val_metrics["f1"],
                "ece": val_metrics["ece"],
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} auc {train_metrics['auc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} auc {val_metrics['auc']:.4f}"
        )

        if np.isfinite(val_metrics["auc"]) and val_metrics["auc"] >= best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metrics = evaluate(model, test_loader, criterion, config.device)
            torch.save(model.state_dict(), os.path.join(config.out_dir, "best_model.pth"))

    total_train_time = time.time() - start_time
    if best_val_metrics is None or best_test_metrics is None:
        best_val_metrics = evaluate(model, val_loader, criterion, config.device)
        best_test_metrics = evaluate(model, test_loader, criterion, config.device)
        best_epoch = config.epochs - 1

    final_info = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_metrics["auc"],
        "best_val_f1": best_val_metrics["f1"],
        "best_test_accuracy": best_test_metrics["accuracy"],
        "best_test_auc": best_test_metrics["auc"],
        "best_test_f1": best_test_metrics["f1"],
        "best_test_sensitivity": best_test_metrics["recall"],
        "best_test_specificity": best_test_metrics["specificity"],
        "best_test_ece": best_test_metrics["ece"],
        "total_train_time": total_train_time,
        "config": {
            "data_root": config.data_root,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "seed": config.seed,
            "device": config.device,
        },
    }

    all_results = {
        f"{DATASET_NAME}_0_train_log_info": train_log_info,
        f"{DATASET_NAME}_0_val_log_info": val_log_info,
        f"{DATASET_NAME}_0_final_info": final_info,
    }

    final_info_wrapped = {
        DATASET_NAME: {
            "means": final_info,
            "stderrs": {key: 0.0 for key, value in final_info.items() if isinstance(value, (int, float))},
            "final_info_dict": final_info,
        }
    }

    with open(os.path.join(config.out_dir, "final_info.json"), "w", encoding="utf-8") as handle:
        json.dump(final_info_wrapped, handle, indent=2)

    with open(os.path.join(config.out_dir, "all_results.npy"), "wb") as handle:
        np.save(handle, all_results)

    print(f"Saved baseline results to {config.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dental periapical radiograph classifier")
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--data_root", type=str, default=str(default_data_root()))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        data_root=args.data_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    train_one_run(config)


if __name__ == "__main__":
    main()