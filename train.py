"""Training script supporting both classification (image) and regression (tabular).

Classification:
    Uses ImageFolder dataset with pretrained CNN models.
Regression:
    Loads CSV/Excel from data_dir, trains a simple MLP or fine-tuned model.
    --target_column specifies which column to predict.
    --feature_columns (comma-sep) specifies input columns (default: all except target).

All results (metrics, plots, live telemetry CSV) are saved in output_dir.
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Training Script (Classification / Regression)")
    p.add_argument("--task", type=str, default="classification", choices=["classification", "regression"])
    p.add_argument("--model", type=str, default="resnet50",
                   choices=["resnet50", "densenet121", "efficientnet_b0", "convnext"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--data_dir", type=str, default="/workspace/data")
    p.add_argument("--test_dir", type=str, default="")
    p.add_argument("--output_dir", type=str, default="/workspace/output")
    p.add_argument("--train_split", type=float, default=0.8)
    # Regression specific
    p.add_argument("--target_column", type=str, default="")
    p.add_argument("--feature_columns", type=str, default="")
    # Augmentation (classification)
    p.add_argument("--random_rotation", action="store_true")
    p.add_argument("--horizontal_flip", action="store_true")
    p.add_argument("--random_erasing", action="store_true")
    # Feature flags
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--grad_cam", action="store_true")
    p.add_argument("--lr_scheduler", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--label_smoothing", action="store_true")
    # Metrics
    p.add_argument("--metrics", type=str, default="accuracy,loss,precision,recall,f1_score,auc_roc,confusion_matrix")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════
def train_classification(args):
    from torchvision import datasets, models, transforms
    from sklearn import metrics as sk_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classification] Device: {device}")

    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    print(f"Metrics: {requested_metrics}")

    train_tfm, val_tfm = _build_cls_transforms(args)

    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_tfm)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}  ({num_classes} classes)")

    if args.test_dir and os.path.isdir(args.test_dir):
        train_ds = full_dataset
        val_ds = datasets.ImageFolder(args.test_dir, transform=val_tfm)
        print(f"Separate test dir: {args.test_dir} ({len(val_ds)} images)")
    else:
        val_size = int((1 - args.train_split) * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        val_ds.dataset = copy.copy(full_dataset)
        val_ds.dataset.transform = val_tfm
        print(f"Split: {train_size} train / {val_size} test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = _build_cls_model(args.model, num_classes).to(device)
    smoothing = 0.1 if args.label_smoothing else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    opt = _build_optimizer(args, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.lr_scheduler else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    metric_history = {m: [] for m in requested_metrics if m not in ("loss", "confusion_matrix")}

    best_val_loss = float("inf")
    patience_counter = 0
    epoch_metrics = {}

    # ── Live telemetry CSV ──
    telemetry_fields = ["train_loss", "val_loss", "train_acc", "val_acc"] + list(metric_history.keys())
    telemetry = LiveTelemetry(args.output_dir, telemetry_fields)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if args.mixup:
                mixed, ta, tb, lam = _mixup_data(imgs, labels)
                opt.zero_grad()
                out = model(mixed)
                loss = lam * criterion(out, ta) + (1 - lam) * criterion(out, tb)
            else:
                opt.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
            loss.backward(); opt.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        if scheduler:
            scheduler.step()

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                running_loss += loss.item() * imgs.size(0)
                correct += (out.argmax(1) == labels).sum().item()
                total += imgs.size(0)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(out.argmax(1).cpu().tolist())
                all_probs.extend(torch.softmax(out, dim=1).cpu().tolist())

        val_loss = running_loss / total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_metrics = _compute_cls_metrics(all_labels, all_preds, all_probs, requested_metrics, num_classes)
        for m in metric_history:
            metric_history[m].append(epoch_metrics.get(m, 0.0))

        metric_str = "  ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
        print(f"Epoch {epoch}/{args.epochs}  TrainLoss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"ValLoss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  {metric_str}")
        sys.stdout.flush()

        # ── Write live telemetry ──
        telem_row = {"train_loss": train_loss, "val_loss": val_loss,
                     "train_acc": train_acc, "val_acc": val_acc}
        telem_row.update(epoch_metrics)
        telemetry.log_epoch(epoch, telem_row)

        if args.early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss; patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}."); break
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

    # ── Plots ──
    epochs_range = range(1, len(history["train_loss"]) + 1)
    _plot_loss_acc(history, epochs_range, args.output_dir)
    _plot_scalar_metrics(metric_history, epochs_range, args.output_dir)

    if "confusion_matrix" in requested_metrics:
        _plot_confusion_matrix(all_labels, all_preds, class_names, args.output_dir)
    if "auc_roc" in requested_metrics:
        _plot_roc(all_labels, all_probs, class_names, num_classes, args.output_dir)
    if args.grad_cam:
        try:
            _generate_gradcam(model, val_loader, device, args.output_dir, class_names, num_images=5)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

    # ── Excel export ──
    _export_cls_excel(history, metric_history, epoch_metrics, class_names,
                      all_labels, all_preds, args.output_dir)

    # ── Markdown report ──
    _generate_report(
        "classification", args, epoch_metrics, history, args.output_dir,
        class_names=class_names,
        extra_info=f"**Dataset:** {len(train_loader.dataset)} train / {len(val_loader.dataset)} test images",
    )

    _print_summary(epoch_metrics, args.output_dir)


# ══════════════════════════════════════════════════════════════════════
#  REGRESSION
# ══════════════════════════════════════════════════════════════════════
def train_regression(args):
    import pandas as pd
    from sklearn import metrics as sk_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Regression] Device: {device}")

    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    print(f"Metrics: {requested_metrics}")

    # ── Load data ──
    df = _load_tabular_data(args.data_dir)
    if args.test_dir and os.path.isdir(args.test_dir):
        df_test = _load_tabular_data(args.test_dir)
    else:
        df_test = None

    target_col = args.target_column
    if not target_col:
        raise ValueError("--target_column is required for regression")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    # Feature columns
    if args.feature_columns:
        feature_cols = [c.strip() for c in args.feature_columns.split(",")]
    else:
        feature_cols = [c for c in df.columns if c != target_col]

    print(f"Target: {target_col}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Samples: {len(df)}")

    # Prepare tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

    if df_test is not None:
        X_test = torch.tensor(df_test[feature_cols].values, dtype=torch.float32)
        y_test = torch.tensor(df_test[target_col].values, dtype=torch.float32).unsqueeze(1)
        train_ds = TensorDataset(X, y)
        val_ds = TensorDataset(X_test, y_test)
        print(f"Separate test set: {len(df_test)} samples")
    else:
        full_ds = TensorDataset(X, y)
        val_size = int((1 - args.train_split) * len(full_ds))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        print(f"Split: {train_size} train / {val_size} test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Build MLP
    n_features = len(feature_cols)
    model = nn.Sequential(
        nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    criterion = nn.MSELoss()
    opt = _build_optimizer(args, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.lr_scheduler else None

    history = {"train_loss": [], "val_loss": []}
    metric_history = {m: [] for m in requested_metrics if m != "loss"}

    best_val_loss = float("inf")
    patience_counter = 0
    epoch_metrics = {}

    # ── Live telemetry CSV ──
    telemetry_fields = ["train_loss", "val_loss"] + list(metric_history.keys())
    telemetry = LiveTelemetry(args.output_dir, telemetry_fields)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, total = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward(); opt.step()
            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        train_loss = running_loss / total
        if scheduler:
            scheduler.step()

        # Validation
        model.eval()
        running_loss, total = 0.0, 0
        all_true, all_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                running_loss += loss.item() * xb.size(0)
                total += xb.size(0)
                all_true.extend(yb.cpu().numpy().flatten().tolist())
                all_pred.extend(pred.cpu().numpy().flatten().tolist())
        val_loss = running_loss / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_metrics = _compute_reg_metrics(all_true, all_pred, requested_metrics)
        for m in metric_history:
            metric_history[m].append(epoch_metrics.get(m, 0.0))

        metric_str = "  ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
        print(f"Epoch {epoch}/{args.epochs}  TrainLoss: {train_loss:.4f}  |  "
              f"ValLoss: {val_loss:.4f}  |  {metric_str}")
        sys.stdout.flush()

        # ── Write live telemetry ──
        telem_row = {"train_loss": train_loss, "val_loss": val_loss}
        telem_row.update(epoch_metrics)
        telemetry.log_epoch(epoch, telem_row)

        if args.early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss; patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}."); break
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

    # ── Plots ──
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss.png"), dpi=150)
    plt.close()

    _plot_scalar_metrics(metric_history, epochs_range, args.output_dir)

    # Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true, all_pred, alpha=0.5, s=10)
    mn, mx = min(all_true + all_pred), max(all_true + all_pred)
    ax.plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pred_vs_actual.png"), dpi=150)
    plt.close()

    # Residual plot
    residuals = np.array(all_true) - np.array(all_pred)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(all_pred, residuals, alpha=0.5, s=10)
    ax.axhline(0, color="r", linestyle="--", alpha=0.7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "residuals.png"), dpi=150)
    plt.close()

    # ── Excel export ──
    _export_reg_excel(history, metric_history, epoch_metrics,
                      feature_cols, target_col, all_true, all_pred, args.output_dir)

    # ── Markdown report ──
    _generate_report(
        "regression", args, epoch_metrics, history, args.output_dir,
        extra_info=f"**Target:** `{target_col}`  |  **Features:** {len(feature_cols)}  |  **Samples:** {len(df)}",
    )

    _print_summary(epoch_metrics, args.output_dir)


# ══════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════
def _build_optimizer(args, model):
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr)
    return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


# ── Live Telemetry CSV ───────────────────────────────────────────────
class LiveTelemetry:
    """Writes metrics to a CSV file after each epoch for real-time GUI polling."""

    def __init__(self, output_dir: str, fieldnames: list):
        self.path = os.path.join(output_dir, "telemetry.csv")
        self.fieldnames = ["epoch", "timestamp"] + fieldnames
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        row = {"epoch": epoch, "timestamp": time.time()}
        row.update(metrics)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ── Markdown Report ──────────────────────────────────────────────────
def _generate_report(
    task_type: str,
    args,
    final_metrics: dict,
    history: dict,
    output_dir: str,
    class_names: list = None,
    extra_info: str = "",
):
    """Generate a Markdown report summarising the training run."""
    path = os.path.join(output_dir, "REPORT.md")
    lines = [
        "# SCOUT Training Report",
        "",
        f"**Task:** {task_type.title()}",
        f"**Model:** {args.model}",
        f"**Optimizer:** {args.optimizer}  |  **LR:** {args.lr}",
        f"**Batch Size:** {args.batch_size}  |  **Epochs (ran):** {len(history['train_loss'])}",
        "",
    ]
    if class_names:
        lines.append(f"**Classes ({len(class_names)}):** {', '.join(class_names)}")
        lines.append("")
    if extra_info:
        lines.append(extra_info)
        lines.append("")
    # Feature flags
    flags = []
    if args.early_stopping:
        flags.append(f"Early Stopping (patience={args.patience})")
    if args.lr_scheduler:
        flags.append("Cosine LR Scheduler")
    if getattr(args, "grad_cam", False):
        flags.append("Grad-CAM")
    if getattr(args, "mixup", False):
        flags.append("Mixup Augmentation")
    if getattr(args, "label_smoothing", False):
        flags.append("Label Smoothing (0.1)")
    if flags:
        lines.append("**Flags:** " + ", ".join(flags))
        lines.append("")

    # Final metrics table
    lines.append("## Final Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in final_metrics.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    # Training summary
    lines.append("## Training Summary")
    lines.append("")
    lines.append(f"- **Best val loss:** {min(history['val_loss']):.4f}")
    if "val_acc" in history:
        lines.append(f"- **Best val accuracy:** {max(history['val_acc']):.4f}")
    lines.append(f"- **Final train loss:** {history['train_loss'][-1]:.4f}")
    lines.append("")

    # Output files
    lines.append("## Output Files")
    lines.append("")
    output_files = sorted(os.listdir(output_dir))
    for f in output_files:
        size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
        icon = "📊" if f.endswith(".png") else "📦" if f.endswith(".pth") else "📄"
        lines.append(f"- {icon} `{f}` ({size_kb:.0f} KB)")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by SCOUT — Vast.ai Training Orchestrator*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {path}")


def _print_summary(metrics_dict, output_dir):
    print("\n" + "=" * 50)
    print("FINAL METRICS (last epoch):")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 50)


# ── Classification helpers ───────────────────────────────────────────
def _build_cls_transforms(args):
    from torchvision import transforms
    train_tfms = [transforms.Resize((224, 224))]
    if args.random_rotation:
        train_tfms.append(transforms.RandomRotation(15))
    if args.horizontal_flip:
        train_tfms.append(transforms.RandomHorizontalFlip())
    train_tfms += [transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if args.random_erasing:
        train_tfms.append(transforms.RandomErasing())
    val_tfms = [transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(train_tfms), transforms.Compose(val_tfms)


def _build_cls_model(name, num_classes):
    from torchvision import models
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def _mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _compute_cls_metrics(all_labels, all_preds, all_probs, metric_names, num_classes):
    from sklearn import metrics as sk_metrics
    results = {}
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)

    if "accuracy" in metric_names:
        results["accuracy"] = sk_metrics.accuracy_score(labels_np, preds_np)
    if "precision" in metric_names:
        avg = "binary" if num_classes == 2 else "macro"
        results["precision"] = sk_metrics.precision_score(labels_np, preds_np, average=avg, zero_division=0)
    if "recall" in metric_names or "sensitivity" in metric_names:
        avg = "binary" if num_classes == 2 else "macro"
        rec = sk_metrics.recall_score(labels_np, preds_np, average=avg, zero_division=0)
        if "recall" in metric_names: results["recall"] = rec
        if "sensitivity" in metric_names: results["sensitivity"] = rec
    if "f1_score" in metric_names:
        avg = "binary" if num_classes == 2 else "macro"
        results["f1_score"] = sk_metrics.f1_score(labels_np, preds_np, average=avg, zero_division=0)
    if "specificity" in metric_names:
        if num_classes == 2:
            tn, fp, fn, tp = sk_metrics.confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()
            results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            cm = sk_metrics.confusion_matrix(labels_np, preds_np)
            specs = []
            for i in range(num_classes):
                tp_i = cm[i, i]; fp_i = cm[:, i].sum() - tp_i
                tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + tp_i
                specs.append(tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0)
            results["specificity"] = np.mean(specs)
    if "auc_roc" in metric_names:
        try:
            probs_np = np.array(all_probs)
            if num_classes == 2:
                results["auc_roc"] = sk_metrics.roc_auc_score(labels_np, probs_np[:, 1])
            else:
                results["auc_roc"] = sk_metrics.roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
        except Exception:
            results["auc_roc"] = 0.0
    if "cohen_kappa" in metric_names:
        results["cohen_kappa"] = sk_metrics.cohen_kappa_score(labels_np, preds_np)
    return results


# ── Regression helpers ───────────────────────────────────────────────
def _load_tabular_data(path):
    """Load CSV or Excel file(s) from a directory or single file."""
    import pandas as pd
    if os.path.isfile(path):
        if path.endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)
    # Directory — find first CSV/Excel
    for f in sorted(os.listdir(path)):
        fp = os.path.join(path, f)
        if f.endswith(".csv"):
            print(f"Loading: {fp}")
            return pd.read_csv(fp)
        if f.endswith((".xlsx", ".xls")):
            print(f"Loading: {fp}")
            return pd.read_excel(fp)
    raise FileNotFoundError(f"No CSV/Excel files found in {path}")


def _compute_reg_metrics(all_true, all_pred, metric_names):
    from sklearn import metrics as sk_metrics
    t = np.array(all_true)
    p = np.array(all_pred)
    results = {}
    if "mse" in metric_names:
        results["mse"] = sk_metrics.mean_squared_error(t, p)
    if "rmse" in metric_names:
        results["rmse"] = np.sqrt(sk_metrics.mean_squared_error(t, p))
    if "mae" in metric_names:
        results["mae"] = sk_metrics.mean_absolute_error(t, p)
    if "r2" in metric_names:
        results["r2"] = sk_metrics.r2_score(t, p)
    if "explained_variance" in metric_names:
        results["explained_variance"] = sk_metrics.explained_variance_score(t, p)
    if "mape" in metric_names:
        mask = t != 0
        if mask.any():
            results["mape"] = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100
        else:
            results["mape"] = 0.0
    return results


# ── Plotting helpers ─────────────────────────────────────────────────
def _plot_loss_acc(history, epochs_range, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, history["train_loss"], label="Train")
    ax1.plot(epochs_range, history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs_range, history["train_acc"], label="Train")
    ax2.plot(epochs_range, history["val_acc"], label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_accuracy.png"), dpi=150)
    plt.close()


def _plot_scalar_metrics(metric_history, epochs_range, output_dir):
    plotable = {k: v for k, v in metric_history.items() if v}
    if not plotable:
        return
    n = len(plotable)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, (name, values) in enumerate(plotable.items()):
        ax = axes[idx // cols][idx % cols]
        ax.plot(epochs_range, values, marker="o", markersize=3)
        ax.set_title(name.replace("_", " ").upper() if name in ("mse", "rmse", "mae", "r2", "mape") else name.replace("_", " ").title())
        ax.set_xlabel("Epoch")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics.png"), dpi=150)
    plt.close()


def _plot_confusion_matrix(all_labels, all_preds, class_names, output_dir):
    from sklearn import metrics as sk_metrics
    cm = sk_metrics.confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = sk_metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_roc(all_labels, all_probs, class_names, num_classes, output_dir):
    from sklearn import metrics as sk_metrics
    try:
        probs_np = np.array(all_probs)
        fig, ax = plt.subplots(figsize=(6, 5))
        if num_classes == 2:
            fpr, tpr, _ = sk_metrics.roc_curve(all_labels, probs_np[:, 1])
            ax.plot(fpr, tpr, label=f"AUC = {sk_metrics.auc(fpr, tpr):.3f}")
        else:
            for i, cls in enumerate(class_names):
                bl = (np.array(all_labels) == i).astype(int)
                fpr, tpr, _ = sk_metrics.roc_curve(bl, probs_np[:, i])
                ax.plot(fpr, tpr, label=f"{cls} (AUC={sk_metrics.auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"ROC curve failed: {e}")


def _generate_gradcam(model, dataloader, device, output_dir, class_names, num_images=4):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if hasattr(model, "layer4"):
        target_layer = [model.layer4[-1]]
    elif hasattr(model, "features"):
        target_layer = [model.features[-1]]
    else:
        print("Grad-CAM: cannot determine target layer, skipping."); return

    cam = GradCAM(model=model, target_layers=target_layer)
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    done = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        for i in range(imgs.size(0)):
            if done >= num_images:
                return
            inp = imgs[i].unsqueeze(0)
            target = [ClassifierOutputTarget(labels[i].item())]
            gc = cam(input_tensor=inp, targets=target)[0]
            rgb = imgs[i].cpu().numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb * std + mean, 0, 1).astype(np.float32)
            overlay = show_cam_on_image(rgb, gc, use_rgb=True)
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(overlay)
            ax.set_title(f"Grad-CAM — {class_names[labels[i].item()]}")
            ax.axis("off")
            plt.savefig(os.path.join(output_dir, f"gradcam_{done}.png"), dpi=100, bbox_inches="tight")
            plt.close()
            done += 1


# ── Excel export ─────────────────────────────────────────────────────
def _export_cls_excel(history, metric_history, final_metrics, class_names,
                      all_labels, all_preds, output_dir):
    """Save classification results to Excel (multiple sheets)."""
    import pandas as pd
    path = os.path.join(output_dir, "results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        # Epoch history
        df_hist = pd.DataFrame({
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
        })
        for m, vals in metric_history.items():
            df_hist[m] = vals
        df_hist.to_excel(w, sheet_name="Epoch History", index=False)

        # Final metrics
        df_final = pd.DataFrame([final_metrics])
        df_final.to_excel(w, sheet_name="Final Metrics", index=False)

        # Predictions
        df_preds = pd.DataFrame({
            "actual": all_labels,
            "predicted": all_preds,
            "actual_class": [class_names[i] for i in all_labels],
            "predicted_class": [class_names[i] for i in all_preds],
        })
        df_preds.to_excel(w, sheet_name="Predictions", index=False)

    print(f"Excel report saved: {path}")


def _export_reg_excel(history, metric_history, final_metrics,
                      feature_cols, target_col, all_true, all_pred, output_dir):
    """Save regression results to Excel (multiple sheets)."""
    import pandas as pd
    path = os.path.join(output_dir, "results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        # Epoch history
        df_hist = pd.DataFrame({
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        })
        for m, vals in metric_history.items():
            df_hist[m] = vals
        df_hist.to_excel(w, sheet_name="Epoch History", index=False)

        # Final metrics
        df_final = pd.DataFrame([final_metrics])
        df_final.to_excel(w, sheet_name="Final Metrics", index=False)

        # Predictions
        df_preds = pd.DataFrame({
            f"actual_{target_col}": all_true,
            f"predicted_{target_col}": all_pred,
            "residual": np.array(all_true) - np.array(all_pred),
        })
        df_preds.to_excel(w, sheet_name="Predictions", index=False)

        # Column info
        df_info = pd.DataFrame({
            "feature_columns": feature_cols + [""] * max(0, 1 - len(feature_cols)),
        })
        df_info["target_column"] = ""
        df_info.loc[0, "target_column"] = target_col
        df_info.to_excel(w, sheet_name="Column Info", index=False)

    print(f"Excel report saved: {path}")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    if args.task == "regression":
        train_regression(args)
    else:
        train_classification(args)
