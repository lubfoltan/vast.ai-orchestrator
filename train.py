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
import csv
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pre_split_data", action="store_true",
                   help="data_dir already contains train/val/test folders")
    # Regression specific
    p.add_argument("--target_column", type=str, default="")
    p.add_argument("--feature_columns", type=str, default="")
    # Augmentation (classification)
    p.add_argument("--random_rotation", action="store_true")
    p.add_argument("--horizontal_flip", action="store_true")
    p.add_argument("--random_erasing", action="store_true")
    p.add_argument("--resize_width", type=int, default=224)
    p.add_argument("--resize_height", type=int, default=224)
    p.add_argument("--no_resize", action="store_true")
    # Feature flags
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--grad_cam", action="store_true")
    p.add_argument("--lr_scheduler", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--label_smoothing", action="store_true")
    # Test mode
    p.add_argument("--use_builtin", action="store_true", help="Use built-in CIFAR-100 dataset")
    # Metrics
    p.add_argument("--metrics", type=str, default="accuracy,loss,precision,recall,f1_score,auc_roc,confusion_matrix")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════
def train_classification(args):
    from torchvision import datasets, transforms

    os.makedirs(args.output_dir, exist_ok=True)
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classification] Device: {device}")
    print(f"Seed: {args.seed}")

    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    print(f"Metrics: {requested_metrics}")
    if args.no_resize:
        print("Resize: disabled")
    else:
        print(f"Resize: {args.resize_width}x{args.resize_height}")

    train_tfm, val_tfm = _build_cls_transforms(args)

    if getattr(args, "use_builtin", False):
        print("[Test Mode] Using built-in CIFAR-100 dataset")
        cifar_train_steps = _resize_steps(args, transforms)
        cifar_train_steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]
        cifar_val_steps = _resize_steps(args, transforms)
        cifar_val_steps += [
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]
        cifar_train_tfm = transforms.Compose(cifar_train_steps)
        cifar_val_tfm = transforms.Compose(cifar_val_steps)
        train_source_ds = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=cifar_train_tfm)
        eval_source_ds = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=cifar_val_tfm)
        test_ds = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=cifar_val_tfm)
        train_indices, val_indices, _ = _stratified_split_indices(
            train_source_ds.targets,
            args.train_split,
            args.val_split,
            0.0,
            args.seed,
            include_test=False,
        )
        train_ds = Subset(train_source_ds, train_indices)
        val_ds = Subset(eval_source_ds, val_indices)
        num_classes = 100
        class_names = train_source_ds.classes
        split_mode = "built-in CIFAR-100"
        print(f"CIFAR-100: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test ({num_classes} classes)")
    else:
        if args.pre_split_data:
            train_dir = _resolve_split_dir(args.data_dir, ("train",))
            val_dir = _resolve_split_dir(args.data_dir, ("val", "validation"))
            test_dir = _resolve_split_dir(args.data_dir, ("test",))
            train_ds = datasets.ImageFolder(train_dir, transform=train_tfm)
            val_ds = datasets.ImageFolder(val_dir, transform=val_tfm)
            test_ds = datasets.ImageFolder(test_dir, transform=val_tfm)
            class_names = train_ds.classes
            _ensure_class_names_match(class_names, val_ds.classes, "validation")
            _ensure_class_names_match(class_names, test_ds.classes, "test")
            num_classes = len(class_names)
            split_mode = "pre-split folders"
            print(f"Pre-split dataset: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
        elif args.test_dir and os.path.isdir(args.test_dir):
            train_source_ds = datasets.ImageFolder(args.data_dir, transform=train_tfm)
            eval_source_ds = datasets.ImageFolder(args.data_dir, transform=val_tfm)
            test_ds = datasets.ImageFolder(args.test_dir, transform=val_tfm)
            class_names = train_source_ds.classes
            _ensure_class_names_match(class_names, test_ds.classes, "test")
            num_classes = len(class_names)
            train_indices, val_indices, _ = _stratified_split_indices(
                train_source_ds.targets,
                args.train_split,
                args.val_split,
                0.0,
                args.seed,
                include_test=False,
            )
            train_ds = Subset(train_source_ds, train_indices)
            val_ds = Subset(eval_source_ds, val_indices)
            split_mode = "train/val split with separate test folder"
            print(f"Separate test dir: {args.test_dir} ({len(test_ds)} images)")
            print(f"Split source data: {len(train_ds)} train / {len(val_ds)} val")
        else:
            train_source_ds = datasets.ImageFolder(args.data_dir, transform=train_tfm)
            eval_source_ds = datasets.ImageFolder(args.data_dir, transform=val_tfm)
            class_names = train_source_ds.classes
            num_classes = len(class_names)
            train_indices, val_indices, test_indices = _stratified_split_indices(
                train_source_ds.targets,
                args.train_split,
                args.val_split,
                args.test_split,
                args.seed,
                include_test=True,
            )
            train_ds = Subset(train_source_ds, train_indices)
            val_ds = Subset(eval_source_ds, val_indices)
            test_ds = Subset(eval_source_ds, test_indices)
            split_mode = "seeded random train/val/test split"
            print(f"Split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

        print(f"Classes: {class_names}  ({num_classes} classes)")

    _ensure_non_empty_split(train_ds, "train")
    _ensure_non_empty_split(val_ds, "validation")
    _ensure_non_empty_split(test_ds, "test")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )

    model = _build_cls_model(args.model, num_classes).to(device)
    smoothing = 0.1 if args.label_smoothing else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    opt = _build_optimizer(args, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.lr_scheduler else None

    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc": [], "val_acc": [], "test_acc": [],
    }
    scalar_metrics = [metric_name for metric_name in requested_metrics if metric_name not in ("loss", "confusion_matrix")]
    val_metric_history = {metric_name: [] for metric_name in scalar_metrics}
    test_metric_history = {metric_name: [] for metric_name in scalar_metrics}

    best_val_loss = float("inf")
    patience_counter = 0
    val_metrics = {}
    test_metrics = {}
    val_labels, val_preds, val_probs = [], [], []
    test_labels, test_preds, test_probs = [], [], []

    # ── Live telemetry CSV ──
    telemetry_fields = [
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc",
    ]
    telemetry_fields += [f"val_{metric_name}" for metric_name in scalar_metrics]
    telemetry_fields += [f"test_{metric_name}" for metric_name in scalar_metrics]
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

        val_loss, val_acc, val_metrics, val_labels, val_preds, val_probs = _evaluate_cls(
            model, val_loader, criterion, device, requested_metrics, num_classes
        )
        test_loss, test_acc, test_metrics, test_labels, test_preds, test_probs = _evaluate_cls(
            model, test_loader, criterion, device, requested_metrics, num_classes
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        for metric_name in scalar_metrics:
            val_metric_history[metric_name].append(val_metrics.get(metric_name, 0.0))
            test_metric_history[metric_name].append(test_metrics.get(metric_name, 0.0))

        val_metric_str = "  ".join(f"val_{key}: {value:.4f}" for key, value in val_metrics.items())
        test_metric_str = "  ".join(f"test_{key}: {value:.4f}" for key, value in test_metrics.items())
        print(f"Epoch {epoch}/{args.epochs}  TrainLoss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"ValLoss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"TestLoss: {test_loss:.4f}  Acc: {test_acc:.4f}  |  "
              f"{val_metric_str}  {test_metric_str}")
        sys.stdout.flush()

        # ── Write live telemetry ──
        telem_row = {
            "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
            "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
        }
        telem_row.update({f"val_{metric_name}": val_metrics.get(metric_name, 0.0) for metric_name in scalar_metrics})
        telem_row.update({f"test_{metric_name}": test_metrics.get(metric_name, 0.0) for metric_name in scalar_metrics})
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
    _plot_cls_scalar_metrics(val_metric_history, test_metric_history, epochs_range, args.output_dir)

    if "confusion_matrix" in requested_metrics:
        _plot_confusion_matrix(test_labels, test_preds, class_names, args.output_dir)
    if "auc_roc" in requested_metrics:
        _plot_roc(test_labels, test_probs, class_names, num_classes, args.output_dir)
    if args.grad_cam:
        try:
            _generate_gradcam(model, test_loader, device, args.output_dir, class_names, num_images=5)
        except Exception as e:
            import traceback
            print(f"Grad-CAM failed: {e}")
            traceback.print_exc()

    final_metrics = {
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    final_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
    final_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})

    # ── Excel export ──
    _export_cls_excel(
        history, val_metric_history, test_metric_history, final_metrics, class_names,
        val_labels, val_preds, test_labels, test_preds, args.output_dir,
    )

    # ── Markdown report ──
    _generate_report(
        "classification", args, final_metrics, history, args.output_dir,
        class_names=class_names,
        extra_info=(
            f"**Dataset:** {len(train_loader.dataset)} train / {len(val_loader.dataset)} validation / "
            f"{len(test_loader.dataset)} test images  \n"
            f"**Split mode:** {split_mode}  |  **Seed:** {args.seed}"
        ),
    )

    _print_summary(final_metrics, args.output_dir)


# ══════════════════════════════════════════════════════════════════════
#  REGRESSION
# ══════════════════════════════════════════════════════════════════════
def train_regression(args):
    import pandas as pd
    from sklearn import metrics as sk_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    _set_seed(args.seed)
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
        split_generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=split_generator)
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
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_split_dir(root_dir: str, names: tuple) -> str:
    for dirname in names:
        candidate = os.path.join(root_dir, dirname)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Missing split folder in {root_dir}: expected one of {names}")


def _ensure_class_names_match(reference_classes: list, candidate_classes: list, split_name: str) -> None:
    if list(reference_classes) != list(candidate_classes):
        raise ValueError(
            f"Class folders in {split_name} do not match train classes. "
            f"train={reference_classes}, {split_name}={candidate_classes}"
        )


def _ensure_non_empty_split(dataset, split_name: str) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{split_name} split is empty. Adjust ratios or provide more data.")


def _stratified_split_indices(
    targets: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_test: bool,
) -> tuple:
    ratio_sum = train_ratio + val_ratio + (test_ratio if include_test else 0.0)
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    normalized_train = train_ratio / ratio_sum
    normalized_val = val_ratio / ratio_sum
    rng = random.Random(seed)
    class_to_indices = {}
    for sample_index, target in enumerate(targets):
        class_to_indices.setdefault(int(target), []).append(sample_index)

    train_indices, val_indices, test_indices = [], [], []
    for class_id in sorted(class_to_indices):
        class_indices = list(class_to_indices[class_id])
        rng.shuffle(class_indices)
        total_count = len(class_indices)
        train_count = round(total_count * normalized_train)
        if include_test:
            val_count = round(total_count * normalized_val)
            if train_count + val_count > total_count:
                val_count = max(0, total_count - train_count)
            test_count = total_count - train_count - val_count
        else:
            val_count = total_count - train_count
            test_count = 0

        train_indices.extend(class_indices[:train_count])
        val_indices.extend(class_indices[train_count:train_count + val_count])
        if include_test:
            test_indices.extend(class_indices[train_count + val_count:train_count + val_count + test_count])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def _evaluate_cls(model, dataloader, criterion, device, requested_metrics, num_classes):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(1)
            running_loss += loss.item() * imgs.size(0)
            correct += (predictions == labels).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().tolist())

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")
    loss_value = running_loss / total
    accuracy = correct / total
    metrics = _compute_cls_metrics(all_labels, all_preds, all_probs, requested_metrics, num_classes)
    return loss_value, accuracy, metrics, all_labels, all_preds, all_probs


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
    if "test_loss" in history:
        lines.append(f"- **Best test loss:** {min(history['test_loss']):.4f}")
    if "val_acc" in history:
        lines.append(f"- **Best val accuracy:** {max(history['val_acc']):.4f}")
    if "test_acc" in history:
        lines.append(f"- **Best test accuracy:** {max(history['test_acc']):.4f}")
    lines.append(f"- **Final train loss:** {history['train_loss'][-1]:.4f}")
    if "test_loss" in history:
        lines.append(f"- **Final test loss:** {history['test_loss'][-1]:.4f}")
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
    train_tfms = _resize_steps(args, transforms)
    if args.random_rotation:
        train_tfms.append(transforms.RandomRotation(15))
    if args.horizontal_flip:
        train_tfms.append(transforms.RandomHorizontalFlip())
    train_tfms += [transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if args.random_erasing:
        train_tfms.append(transforms.RandomErasing())
    val_tfms = _resize_steps(args, transforms)
    val_tfms += [transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(train_tfms), transforms.Compose(val_tfms)


def _resize_steps(args, transforms):
    if getattr(args, "no_resize", False):
        return []
    if args.resize_width <= 0 or args.resize_height <= 0:
        raise ValueError("resize_width and resize_height must be positive integers, or use --no_resize")
    return [transforms.Resize((args.resize_height, args.resize_width))]


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
    if "balanced_accuracy" in metric_names:
        results["balanced_accuracy"] = sk_metrics.balanced_accuracy_score(labels_np, preds_np)
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
    if history.get("test_loss"):
        ax1.plot(epochs_range, history["test_loss"], label="Test")
    loss_values = history["train_loss"] + history["val_loss"] + history.get("test_loss", [])
    loss_upper = max(loss_values) * 1.08 if loss_values and max(loss_values) > 0 else 1.0
    ax1.set_ylim(0, loss_upper)
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.25)
    ax2.plot(epochs_range, history["train_acc"], label="Train")
    ax2.plot(epochs_range, history["val_acc"], label="Val")
    if history.get("test_acc"):
        ax2.plot(epochs_range, history["test_acc"], label="Test")
    ax2.set_ylim(0, 1)
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_accuracy.png"), dpi=150)
    plt.close()


def _plot_cls_scalar_metrics(val_metric_history, test_metric_history, epochs_range, output_dir):
    metric_names = [
        metric_name for metric_name in val_metric_history
        if val_metric_history.get(metric_name) or test_metric_history.get(metric_name)
    ]
    if not metric_names:
        return
    cols = min(3, len(metric_names))
    rows = (len(metric_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for index, metric_name in enumerate(metric_names):
        ax = axes[index // cols][index % cols]
        val_values = val_metric_history.get(metric_name, [])
        test_values = test_metric_history.get(metric_name, [])
        if val_values:
            ax.plot(epochs_range, val_values, marker="o", markersize=3, label="Val")
        if test_values:
            ax.plot(epochs_range, test_values, marker="o", markersize=3, label="Test")
        title = metric_name.replace("_", " ").title()
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if metric_name == "cohen_kappa":
            ax.set_ylim(-1, 1)
        else:
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
    for index in range(len(metric_names), rows * cols):
        axes[index // cols][index % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics.png"), dpi=150)
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
    per_class_limit = max(1, int(np.ceil(num_images / max(len(class_names), 1))))
    class_counts = {index: 0 for index in range(len(class_names))}
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = model(imgs).argmax(1).cpu().tolist()
        for i in range(imgs.size(0)):
            if done >= num_images:
                return
            true_index = labels[i].item()
            if class_counts.get(true_index, 0) >= per_class_limit:
                continue
            inp = imgs[i].unsqueeze(0)
            pred_index = preds[i]
            target = [ClassifierOutputTarget(pred_index)]
            gc = cam(input_tensor=inp, targets=target)[0]
            rgb = imgs[i].cpu().numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb * std + mean, 0, 1).astype(np.float32)
            overlay = show_cam_on_image(rgb, gc, use_rgb=True)
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(overlay)
            ax.set_title(f"Grad-CAM — true {class_names[true_index]} / pred {class_names[pred_index]}")
            ax.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"gradcam_{done}_true_{class_names[true_index]}_pred_{class_names[pred_index]}.png"),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()
            class_counts[true_index] = class_counts.get(true_index, 0) + 1
            done += 1


# ── Excel export ─────────────────────────────────────────────────────
def _export_cls_excel(history, val_metric_history, test_metric_history, final_metrics, class_names,
                      val_labels, val_preds, test_labels, test_preds, output_dir):
    """Save classification results to Excel (multiple sheets)."""
    import pandas as pd
    path = os.path.join(output_dir, "results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        # Epoch history
        df_hist = pd.DataFrame({
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "test_loss": history["test_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
            "test_acc": history["test_acc"],
        })
        for metric_name, values in val_metric_history.items():
            df_hist[f"val_{metric_name}"] = values
        for metric_name, values in test_metric_history.items():
            df_hist[f"test_{metric_name}"] = values
        df_hist.to_excel(w, sheet_name="Epoch History", index=False)

        # Final metrics
        df_final = pd.DataFrame([final_metrics])
        df_final.to_excel(w, sheet_name="Final Metrics", index=False)

        # Predictions
        df_val_preds = pd.DataFrame({
            "actual": val_labels,
            "predicted": val_preds,
            "actual_class": [class_names[index] for index in val_labels],
            "predicted_class": [class_names[index] for index in val_preds],
        })
        df_val_preds.to_excel(w, sheet_name="Validation Predictions", index=False)

        df_test_preds = pd.DataFrame({
            "actual": test_labels,
            "predicted": test_preds,
            "actual_class": [class_names[index] for index in test_labels],
            "predicted_class": [class_names[index] for index in test_preds],
        })
        df_test_preds.to_excel(w, sheet_name="Test Predictions", index=False)

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
