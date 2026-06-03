"""Train the custom CNN architecture for 3-class chest X-ray classification.

Expected pre-split data layout:
    data/
      train/NORMAL/*.jpg
      train/PNEUMONIA_BACTERIAL/*.jpg
      train/PNEUMONIA_VIRUS/*.jpg
      val/...
      test/...

The model follows the custom architecture described in the project document:
four Conv-BN-ReLU-MaxPool blocks, global average pooling, a dense hidden layer,
dropout, and a 3-neuron classification head trained with CrossEntropyLoss.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = ("NORMAL", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRUS")


class CustomChestXrayCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.5, feature_dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.feature_dropout = nn.Dropout2d(feature_dropout) if feature_dropout > 0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.feature_dropout(x)
        x = self.pool(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom CNN for 3-class chest X-ray classification")
    parser.add_argument("--data_dir", type=str, default="/workspace/data")
    parser.add_argument("--output_dir", type=str, default="/workspace/output")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feature_dropout", type=float, default=0.2)
    parser.add_argument("--random_erasing_p", type=float, default=0.5)
    parser.add_argument("--random_erasing_scale_min", type=float, default=0.02)
    parser.add_argument("--random_erasing_scale_max", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--selection_metric", choices=("val_loss", "val_balanced_accuracy"), default="val_balanced_accuracy")
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_cam", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transforms(args: argparse.Namespace):
    # Trénovacie transformácie - zamerané na odstrihnutie rohov a robustnosť
    train_steps = [
        # ZMENA: Namiesto Resize použijeme RandomResizedCrop. 
        # To náhodne vyberie 75% až 100% plochy obrázka a zväčší ju na 224x224.
        # Väčšina rohov so značkami (R/L) tak ostane mimo "zorného poľa".
        transforms.RandomResizedCrop(args.resize, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(15), # Mierne zvýšená rotácia
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Pomáha ignorovať intenzitu značiek
        transforms.ToTensor(),
    ]
    
    if args.random_erasing_p > 0:
        train_steps.append(
            transforms.RandomErasing(
                p=args.random_erasing_p,
                scale=(0.05, 0.25), # ZMENA: Väčšie vymazávacie plochy, ktoré prekryjú značky
                ratio=(0.3, 3.3),
                value=0.0,
            )
        )
    train_steps.append(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )
    train_transform = transforms.Compose(train_steps)

    # Validačné a testovacie transformácie
    eval_transform = transforms.Compose([
        # ZMENA: Obrázok najprv zväčšíme a potom urobíme CenterCrop.
        # Tým sa zbavíme okrajov, kde sú značky, na všetkých testovacích fotkách.
        transforms.Resize(args.resize + 32), 
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def split_dir(data_dir: Path, *names: str) -> Path:
    for name in names:
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing split folder in {data_dir}: {names}")


def build_loaders(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    train_transform, eval_transform = build_transforms(args)
    train_ds = datasets.ImageFolder(split_dir(data_dir, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(split_dir(data_dir, "val", "validation"), transform=eval_transform)
    test_ds = datasets.ImageFolder(split_dir(data_dir, "test"), transform=eval_transform)

    expected = list(CLASS_NAMES)
    for split_name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if dataset.classes != expected:
            raise ValueError(f"{split_name} classes must be {expected}, got {dataset.classes}")

    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                              pin_memory=pin_memory, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                             pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def class_weights_from_dataset(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    targets = np.array(dataset.targets, dtype=np.int64)
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    print("Class counts:", dict(zip(CLASS_NAMES, counts.astype(int).tolist())))
    print("Class weights:", dict(zip(CLASS_NAMES, np.round(weights, 4).tolist())))
    return torch.tensor(weights, dtype=torch.float32, device=device)


class LiveTelemetry:
    def __init__(self, output_dir: Path, fieldnames: list[str]) -> None:
        self.path = output_dir / "telemetry.csv"
        self.fieldnames = ["epoch", "timestamp"] + fieldnames
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> dict[str, float]:
        row = {"epoch": epoch, "timestamp": time.time()}
        row.update(metrics)
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)
        return row


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels_all, preds_all, probs_all = [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        total_loss += loss.item() * images.size(0)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    labels_np = np.array(labels_all)
    preds_np = np.array(preds_all)
    probs_np = np.array(probs_all)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": sk_metrics.accuracy_score(labels_np, preds_np),
        "balanced_accuracy": sk_metrics.balanced_accuracy_score(labels_np, preds_np),
        "precision_macro": sk_metrics.precision_score(labels_np, preds_np, average="macro", zero_division=0),
        "recall_macro": sk_metrics.recall_score(labels_np, preds_np, average="macro", zero_division=0),
        "f1_macro": sk_metrics.f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "auc_roc_ovr_macro": sk_metrics.roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro"),
    }
    return metrics, labels_all, preds_all, probs_all


def plot_history(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Val")
    axes[1].plot(epochs, [row["test_acc"] for row in history], label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_accuracy.png", dpi=150)
    plt.close(fig)


def plot_metrics(history: list[dict[str, float]], output_dir: Path) -> None:
    metric_names = ["balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_roc_ovr_macro"]
    epochs = [row["epoch"] for row in history]
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8), squeeze=False)
    for index, metric_name in enumerate(metric_names):
        ax = axes[index // cols][index % cols]
        ax.plot(epochs, [row[f"val_{metric_name}"] for row in history], marker="o", markersize=3, label="Val")
        ax.plot(epochs, [row[f"test_{metric_name}"] for row in history], marker="o", markersize=3, label="Test")
        ax.set_title(metric_name.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1][-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, output_dir: Path) -> None:
    cm = sk_metrics.confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = sk_metrics.ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    display.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=20)
    ax.set_title("Confusion Matrix")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc(labels, probs, output_dir: Path) -> None:
    labels_np = np.array(labels)
    probs_np = np.array(probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_labels = (labels_np == class_index).astype(int)
        fpr, tpr, _ = sk_metrics.roc_curve(binary_labels, probs_np[:, class_index])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={sk_metrics.auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def generate_gradcam(model, dataloader, device, output_dir: Path, num_correct: int = 5, num_incorrect: int = 5, seed: int = 27) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as exc:
        print(f"Grad-CAM skipped: {exc}")
        return

    target_layer = [model.features[-1][0]]
    cam = GradCAM(model=model, target_layers=target_layer)
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rng = random.Random(seed)
    selected = {"correct": [], "incorrect": []}
    seen = {"correct": 0, "incorrect": 0}
    limits = {"correct": num_correct, "incorrect": num_incorrect}

    with torch.no_grad():
        for images, labels in dataloader:
            images_device = images.to(device)
            logits = model(images_device)
            probs = torch.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)
            for index in range(images.size(0)):
                label_index = int(labels[index].item())
                pred_index = int(preds[index].detach().cpu().item())
                bucket = "correct" if pred_index == label_index else "incorrect"
                if limits[bucket] <= 0:
                    continue
                seen[bucket] += 1
                sample = {
                    "image": images[index].detach().cpu(),
                    "label": label_index,
                    "pred": pred_index,
                    "confidence": float(confidences[index].detach().cpu().item()),
                }
                if len(selected[bucket]) < limits[bucket]:
                    selected[bucket].append(sample)
                else:
                    replace_index = rng.randrange(seen[bucket])
                    if replace_index < limits[bucket]:
                        selected[bucket][replace_index] = sample

    def safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    for bucket in ("correct", "incorrect"):
        if not selected[bucket]:
            print(f"Grad-CAM: no {bucket} test predictions found.")
            continue
        for index, sample in enumerate(selected[bucket], start=1):
            input_tensor = sample["image"].unsqueeze(0).to(device)
            target = [ClassifierOutputTarget(sample["pred"])]
            grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
            rgb = sample["image"].numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb * std + mean, 0, 1).astype(np.float32)
            overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            true_name = CLASS_NAMES[sample["label"]]
            pred_name = CLASS_NAMES[sample["pred"]]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(overlay)
            ax.set_title(f"Grad-CAM - true {true_name} / pred {pred_name}")
            ax.axis("off")
            filename = f"gradcam_{bucket}_{index:02d}_true_{safe_name(true_name)}_pred_{safe_name(pred_name)}.png"
            fig.savefig(output_dir / filename, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"Grad-CAM saved: {filename} (confidence={sample['confidence']:.3f})")


def export_results_excel(history: list[dict[str, float]], final_metrics: dict[str, float],
                         val_labels, val_preds, test_labels, test_preds, output_dir: Path) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        print(f"Excel export skipped: {exc}")
        return

    path = output_dir / "results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(history).to_excel(writer, sheet_name="Epoch History", index=False)
        pd.DataFrame([final_metrics]).to_excel(writer, sheet_name="Final Metrics", index=False)
        pd.DataFrame({
            "actual": val_labels,
            "predicted": val_preds,
            "actual_class": [CLASS_NAMES[index] for index in val_labels],
            "predicted_class": [CLASS_NAMES[index] for index in val_preds],
        }).to_excel(writer, sheet_name="Val Predictions", index=False)
        pd.DataFrame({
            "actual": test_labels,
            "predicted": test_preds,
            "actual_class": [CLASS_NAMES[index] for index in test_labels],
            "predicted_class": [CLASS_NAMES[index] for index in test_preds],
        }).to_excel(writer, sheet_name="Test Predictions", index=False)
    print(f"Excel report saved: {path}")


def generate_report(args: argparse.Namespace, history: list[dict[str, float]], final_metrics: dict[str, float],
                    output_dir: Path, train_size: int, val_size: int, test_size: int) -> None:
    path = output_dir / "REPORT.md"
    flags = [f"Early Stopping (patience={args.patience})"]
    if args.grad_cam:
        flags.append("Grad-CAM")
    if args.label_smoothing > 0:
        flags.append(f"Label Smoothing ({args.label_smoothing})")
    if args.random_erasing_p > 0:
        flags.append(f"Random Erasing (p={args.random_erasing_p})")
    if not args.no_class_weights:
        flags.append("Class Weights")

    lines = [
        "# SCOUT Training Report",
        "",
        "**Task:** Classification",
        "**Model:** custom_cnn_3class",
        f"**Optimizer:** adamw  |  **LR:** {args.lr}",
        f"**Batch Size:** {args.batch_size}  |  **Epochs (ran):** {len(history)}",
        "",
        f"**Classes ({len(CLASS_NAMES)}):** {', '.join(CLASS_NAMES)}",
        "",
        f"**Dataset:** {train_size} train / {val_size} validation / {test_size} test images  ",
        f"**Split mode:** pre-split folders  |  **Seed:** {args.seed}",
        "",
        "**Flags:** " + ", ".join(flags),
        "",
        "## Final Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for key, value in final_metrics.items():
        lines.append(f"| {key} | {value:.4f} |")
    lines += [
        "",
        "## Training Summary",
        "",
        f"- **Best val loss:** {min(row['val_loss'] for row in history):.4f}",
        f"- **Best test loss:** {min(row['test_loss'] for row in history):.4f}",
        f"- **Best val accuracy:** {max(row['val_acc'] for row in history):.4f}",
        f"- **Best test accuracy:** {max(row['test_acc'] for row in history):.4f}",
        f"- **Best val balanced accuracy:** {max(row['val_balanced_accuracy'] for row in history):.4f}",
        f"- **Final train loss:** {history[-1]['train_loss']:.4f}",
        f"- **Final test loss:** {history[-1]['test_loss']:.4f}",
        "",
        "## Output Files",
        "",
    ]
    for output_file in sorted(output_dir.iterdir()):
        if output_file.is_file():
            size_kb = output_file.stat().st_size / 1024
            lines.append(f"- `{output_file.name}` ({size_kb:.0f} KB)")
    lines += ["", "---", "*Generated by SCOUT - Vast.ai Training Orchestrator*"]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {path}")


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "telemetry.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label_smoothing must be in [0, 1).")
    if not 0.0 <= args.dropout < 1.0 or not 0.0 <= args.feature_dropout < 1.0:
        raise ValueError("--dropout and --feature_dropout must be in [0, 1).")
    if not 0.0 <= args.random_erasing_p <= 1.0:
        raise ValueError("--random_erasing_p must be in [0, 1].")
    if args.random_erasing_scale_min <= 0 or args.random_erasing_scale_max < args.random_erasing_scale_min:
        raise ValueError("Random erasing scale must be positive and min <= max.")

    train_loader, val_loader, test_loader = build_loaders(args)
    model = CustomChestXrayCNN(
        num_classes=len(CLASS_NAMES),
        dropout=args.dropout,
        feature_dropout=args.feature_dropout,
    ).to(device)
    class_weights = None if args.no_class_weights else class_weights_from_dataset(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    maximize_metric = args.selection_metric == "val_balanced_accuracy"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize_metric else "min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    print(
        f"Regularization: dropout={args.dropout}, feature_dropout={args.feature_dropout}, "
        f"random_erasing_p={args.random_erasing_p}, label_smoothing={args.label_smoothing}, "
        f"class_weights={not args.no_class_weights}, selection_metric={args.selection_metric}"
    )

    best_score = -float("inf") if maximize_metric else float("inf")
    patience_counter = 0
    history = []
    best_val_metrics, best_test_metrics = {}, {}
    best_val_labels, best_val_preds = [], []
    best_test_labels, best_test_preds, best_test_probs = [], [], []
    telemetry_fields = [
        "lr", "train_loss", "val_loss", "test_loss", "train_acc", "val_acc", "test_acc",
        "val_balanced_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro", "val_auc_roc_ovr_macro",
        "test_balanced_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro", "test_auc_roc_ovr_macro",
    ]
    telemetry = LiveTelemetry(output_dir, telemetry_fields)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_metrics, val_labels, val_preds, _ = evaluate(model, val_loader, criterion, device)
        test_metrics, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device)
        score = val_metrics["balanced_accuracy"] if maximize_metric else val_metrics["loss"]
        scheduler.step(score)

        telemetry_row = {
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "test_loss": test_metrics["loss"],
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
            "test_acc": test_metrics["accuracy"],
            **{f"val_{key}": value for key, value in val_metrics.items() if key not in ("loss", "accuracy")},
            **{f"test_{key}": value for key, value in test_metrics.items() if key not in ("loss", "accuracy")},
        }
        history.append(telemetry.log_epoch(epoch, telemetry_row))
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f}"
        )

        improved = score > best_score if maximize_metric else score < best_score
        if improved:
            best_score = score
            patience_counter = 0
            best_val_metrics, best_test_metrics = val_metrics, test_metrics
            best_val_labels, best_val_preds = val_labels, val_preds
            best_test_labels, best_test_preds, best_test_probs = test_labels, test_preds, test_probs
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), output_dir / "final_model.pth")
    save_history(history, output_dir)
    plot_history(history, output_dir)
    plot_metrics(history, output_dir)
    plot_confusion_matrix(best_test_labels, best_test_preds, output_dir)
    plot_roc(best_test_labels, best_test_probs, output_dir)
    best_model_path = output_dir / "best_model.pth"
    if best_model_path.is_file():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    if args.grad_cam:
        generate_gradcam(model, test_loader, device, output_dir, seed=args.seed)
    final_metrics = {
        "val_loss": best_val_metrics.get("loss", 0.0),
        "test_loss": best_test_metrics.get("loss", 0.0),
        "val_accuracy": best_val_metrics.get("accuracy", 0.0),
        "test_accuracy": best_test_metrics.get("accuracy", 0.0),
    }
    final_metrics.update({f"val_{key}": value for key, value in best_val_metrics.items() if key not in ("loss", "accuracy")})
    final_metrics.update({f"test_{key}": value for key, value in best_test_metrics.items() if key not in ("loss", "accuracy")})
    export_results_excel(history, final_metrics, best_val_labels, best_val_preds, best_test_labels, best_test_preds, output_dir)
    generate_report(args, history, final_metrics, output_dir, len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())