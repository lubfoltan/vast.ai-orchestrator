"""Finish SCOUT classification outputs after a disconnected training run.

This script is intended to run on the remote Vast.ai instance. It reuses the
plotting, metric, Excel/report, and Grad-CAM helpers from /workspace/train.py
so recovered artifacts match the normal pipeline output style.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize SCOUT pipeline artifacts from a saved checkpoint")
    parser.add_argument("--data_dir", default="/workspace/data")
    parser.add_argument("--output_dir", default="/workspace/output")
    parser.add_argument("--train_script", default="/workspace/train.py")
    parser.add_argument("--checkpoint", default="/workspace/output/best_model.pth")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--grad_cam_images", type=int, default=5)
    parser.add_argument(
        "--metrics",
        default="accuracy,loss,precision,recall,f1_score,auc_roc,confusion_matrix",
    )
    return parser.parse_args()


def import_train_module(train_script: str):
    spec = importlib.util.spec_from_file_location("scout_train", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load train.py from {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_model_name(state_dict: dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("features.denseblock") for key in keys):
        return "densenet121"
    if any(key.startswith("layer4.") for key in keys):
        return "resnet50"
    if any(key.startswith("features.7.") for key in keys) and any(key.startswith("classifier.1.") for key in keys):
        return "efficientnet_b0"
    if any(key.startswith("features.7.") for key in keys) and any(key.startswith("classifier.2.") for key in keys):
        return "convnext"
    if any(key.startswith("features.0.0.") for key in keys) and any(key.startswith("classifier.4.") for key in keys):
        return "custom_cnn"
    raise ValueError("Could not infer model architecture from checkpoint. Pass --model explicitly.")


def load_telemetry(output_dir: Path) -> list[dict[str, str]]:
    telemetry_path = output_dir / "telemetry.csv"
    if not telemetry_path.is_file():
        raise FileNotFoundError(f"Missing telemetry file: {telemetry_path}")
    with telemetry_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_float(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def build_history(rows: list[dict[str, str]], metrics: list[str]) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    history = {
        "train_loss": [get_float(row, "train_loss") for row in rows],
        "val_loss": [get_float(row, "val_loss") for row in rows],
        "test_loss": [get_float(row, "test_loss") for row in rows],
        "train_acc": [get_float(row, "train_acc", "train_accuracy") for row in rows],
        "val_acc": [get_float(row, "val_acc", "val_accuracy") for row in rows],
        "test_acc": [get_float(row, "test_acc", "test_accuracy") for row in rows],
    }
    scalar_metrics = [metric for metric in metrics if metric not in ("loss", "confusion_matrix")]
    val_metric_history = {
        metric: [get_float(row, f"val_{metric}") for row in rows]
        for metric in scalar_metrics
    }
    test_metric_history = {
        metric: [get_float(row, f"test_{metric}") for row in rows]
        for metric in scalar_metrics
    }
    return history, val_metric_history, test_metric_history


def build_pipeline_args(args: argparse.Namespace, epochs_ran: int, model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        task="classification",
        model=model_name,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=epochs_ran,
        optimizer=args.optimizer,
        data_dir=args.data_dir,
        test_dir="",
        output_dir=args.output_dir,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        seed=args.seed,
        pre_split_data=True,
        random_rotation=False,
        horizontal_flip=False,
        random_erasing=False,
        resize_width=224,
        resize_height=224,
        no_resize=False,
        early_stopping=True,
        patience=args.patience,
        grad_cam=True,
        lr_scheduler=False,
        mixup=False,
        label_smoothing=False,
        use_builtin=False,
        amp=False,
        metrics=args.metrics,
        target_column="",
        feature_columns="",
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    train_module = import_train_module(args.train_script)
    train_module._set_seed(args.seed)

    rows = load_telemetry(output_dir)
    if not rows:
        raise ValueError("telemetry.csv has no epoch rows")
    requested_metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    history, val_metric_history, test_metric_history = build_history(rows, requested_metrics)

    state_dict = torch.load(checkpoint, map_location="cpu")
    model_name = infer_model_name(state_dict) if args.model == "auto" else args.model
    pipeline_args = build_pipeline_args(args, len(rows), model_name)

    _, val_transform = train_module._build_cls_transforms(pipeline_args)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_transform)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=val_transform)
    class_names = val_ds.classes
    if class_names != test_ds.classes:
        raise ValueError(f"Validation/test class mismatch: {class_names} vs {test_ds.classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_module._build_cls_model(model_name, len(class_names)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    pin_memory = torch.cuda.is_available()
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, val_metrics, val_labels, val_preds, _ = train_module._evaluate_cls(
        model, val_loader, criterion, device, requested_metrics, len(class_names)
    )
    test_loss, test_acc, test_metrics, test_labels, test_preds, test_probs = train_module._evaluate_cls(
        model, test_loader, criterion, device, requested_metrics, len(class_names)
    )

    epochs_range = range(1, len(rows) + 1)
    train_module._plot_loss_acc(history, epochs_range, args.output_dir)
    train_module._plot_cls_scalar_metrics(val_metric_history, test_metric_history, epochs_range, args.output_dir)
    if "confusion_matrix" in requested_metrics:
        train_module._plot_confusion_matrix(test_labels, test_preds, class_names, args.output_dir)
    if "auc_roc" in requested_metrics:
        train_module._plot_roc(test_labels, test_probs, class_names, len(class_names), args.output_dir)
    if args.grad_cam_images > 0:
        train_module._generate_gradcam(model, test_loader, device, args.output_dir, class_names, num_images=args.grad_cam_images)

    final_metrics = {
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    final_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
    final_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})

    train_module._export_cls_excel(
        history,
        val_metric_history,
        test_metric_history,
        final_metrics,
        class_names,
        val_labels,
        val_preds,
        test_labels,
        test_preds,
        args.output_dir,
    )
    train_module._generate_report(
        "classification",
        pipeline_args,
        final_metrics,
        history,
        args.output_dir,
        class_names=class_names,
        extra_info=(
            f"**Dataset:** telemetry recovered / {len(val_ds)} validation / {len(test_ds)} test images  \n"
            f"**Split mode:** pre-split folders  |  **Seed:** {args.seed}  |  **Checkpoint:** {checkpoint.name}"
        ),
    )
    with (output_dir / "finalized_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=2)

    train_module._print_summary(final_metrics, args.output_dir)
    print(f"Recovered pipeline-style artifacts for {model_name} with {len(class_names)} classes into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())