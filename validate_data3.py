"""Validate ResNet50 (3-class) on DATA3 dataset.

Classes: NORMAL | PNEUMONIA_BACTERIAL | PNEUMONIA_VIRUS

Data layout expected:
    DATA3/NORMAL/         *.jpeg  (1583 files)
    DATA3/PNEUMONIA/      person*_bacteria_*.jpeg  (2780 files)
                          person*_virus_*.jpeg     (1493 files)

Usage:
    python validate_data3.py           # full run  (~900 images, all metrics + plots)
    python validate_data3.py --test    # quick test  (10 images, console-only)

Outputs (saved to OUTPUT_DIR):
    validation_metrics.txt
    confusion_matrix.png
    roc_curve.png
    gradcam_correct_1..5_true_X_pred_X.png
    gradcam_incorrect_1..5_true_X_pred_Y.png
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics as sk_metrics

# ── Default Paths (DATA3) ──────────────────────────────────────────────
DATA3_NORMAL    = r"C:\Skola\4 letny semester\HSU\DATA3\NORMAL"
DATA3_PNEUMONIA = r"C:\Skola\4 letny semester\HSU\DATA3\PNEUMONIA"
MODEL_PATH      = r"C:\Skola\4 letny semester\HSU\224x224_ResNet_3_classes\final_model.pth"
OUTPUT_DIR      = r"C:\Skola\4 letny semester\HSU\224x224_ResNet_3_classes\validation_data3"
# ──────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["NORMAL", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRUS"]
N_PER_CLASS  = 300   # images per class for the full run
N_TEST_TOTAL = 10    # total images for --test quick check
BATCH_SIZE   = 16
SEED         = 27
NUM_GRADCAM  = 5     # correct  +  incorrect  (each)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class DATA3Dataset(Dataset):
    """(path, label) dataset — returns image tensor + label + path string."""

    def __init__(self, samples: list[tuple[Path, int]], transform: T.Compose) -> None:
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


def collect_samples(
    n_per_class: int,
    seed: int,
    normal_dir: str | None = None,
    pneumonia_dir: str | None = None,
) -> list[tuple[Path, int]]:
    """Return a shuffled list of (path, class_id) tuples.

    Skips files smaller than 1 KB (macOS ._* resource forks, .DS_Store, etc.).
    When n_per_class <= 0 all available images are used.
    """
    rng = random.Random(seed)
    exts = ("*.jpeg", "*.jpg", "*.png")
    MIN_BYTES = 1024  # skip resource-fork stubs

    def glob_all(folder: str) -> list[Path]:
        p = Path(folder)
        files: list[Path] = []
        for ext in exts:
            files.extend(p.glob(ext))
        # filter out macOS metadata stubs and hidden files
        files = [f for f in files if not f.name.startswith("._") and f.stat().st_size >= MIN_BYTES]
        return sorted(files)

    src_normal    = normal_dir    if normal_dir    else DATA3_NORMAL
    src_pneumonia = pneumonia_dir if pneumonia_dir else DATA3_PNEUMONIA

    # Class 0 – NORMAL
    normal_files = glob_all(src_normal)
    if n_per_class > 0 and len(normal_files) > n_per_class:
        normal_files = rng.sample(normal_files, n_per_class)

    # Class 1 – PNEUMONIA_BACTERIAL  (filename contains "bacteria")
    # Class 2 – PNEUMONIA_VIRUS      (filename contains "virus")
    all_pneumonia  = glob_all(src_pneumonia)
    bacteria_files = sorted([f for f in all_pneumonia if "bacteria" in f.name.lower()])
    virus_files    = sorted([f for f in all_pneumonia if "virus"    in f.name.lower()])
    if n_per_class > 0 and len(bacteria_files) > n_per_class:
        bacteria_files = rng.sample(bacteria_files, n_per_class)
    if n_per_class > 0 and len(virus_files) > n_per_class:
        virus_files = rng.sample(virus_files, n_per_class)

    samples: list[tuple[Path, int]] = (
        [(f, 0) for f in normal_files] +
        [(f, 1) for f in bacteria_files] +
        [(f, 2) for f in virus_files]
    )
    rng.shuffle(samples)
    return samples


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    model = tvm.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    all_labels: list[int]         = []
    all_preds:  list[int]         = []
    all_probs:  list[list[float]] = []
    all_paths:  list[str]         = []

    with torch.no_grad():
        for batch_idx, (imgs, labels, paths) in enumerate(loader):
            imgs = imgs.to(device)
            out  = model(imgs)
            preds  = out.argmax(1)
            probs  = torch.softmax(out, dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_paths.extend(paths)
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} / {len(loader.dataset)}", end="\r")

    print()
    return all_labels, all_preds, all_probs, all_paths


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs) -> dict:
    probs_np = np.array(all_probs)
    acc      = sk_metrics.accuracy_score(all_labels, all_preds)
    bal_acc  = sk_metrics.balanced_accuracy_score(all_labels, all_preds)
    prec_mac = sk_metrics.precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_mac  = sk_metrics.recall_score(   all_labels, all_preds, average="macro", zero_division=0)
    f1_mac   = sk_metrics.f1_score(       all_labels, all_preds, average="macro", zero_division=0)
    prec_cls = sk_metrics.precision_score(all_labels, all_preds, average=None,    zero_division=0)
    rec_cls  = sk_metrics.recall_score(   all_labels, all_preds, average=None,    zero_division=0)
    f1_cls   = sk_metrics.f1_score(       all_labels, all_preds, average=None,    zero_division=0)
    try:
        auc = sk_metrics.roc_auc_score(all_labels, probs_np, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    cm = sk_metrics.confusion_matrix(all_labels, all_preds)
    report = sk_metrics.classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0
    )
    return {
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "precision_macro":   prec_mac,
        "recall_macro":      rec_mac,
        "f1_macro":          f1_mac,
        "precision_cls":     prec_cls,
        "recall_cls":        rec_cls,
        "f1_cls":            f1_cls,
        "auc_roc_macro":     auc,
        "confusion_matrix":  cm,
        "report":            report,
    }


def print_metrics(m: dict) -> None:
    print("\n" + "=" * 55)
    print("VALIDATION METRICS")
    print("=" * 55)
    print(f"  Accuracy:           {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy:  {m['balanced_accuracy']:.4f}  ({m['balanced_accuracy']*100:.2f}%)")
    print(f"  Precision (macro):  {m['precision_macro']:.4f}")
    print(f"  Recall (macro):     {m['recall_macro']:.4f}")
    print(f"  F1 Score (macro):   {m['f1_macro']:.4f}")
    print(f"  AUC-ROC (macro):    {m['auc_roc_macro']:.4f}")
    print()
    print(m["report"])
    print("Confusion Matrix (rows=True, cols=Predicted):")
    print(m["confusion_matrix"])


# ──────────────────────────────────────────────────────────────────────
# Output: metrics text file
# ──────────────────────────────────────────────────────────────────────

def save_metrics_txt(m: dict, n_samples: int, run_type: str, out_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(out_dir, "validation_metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"DATA3 Validation Results — ResNet50 (3-class)\n")
        f.write(f"{'=' * 55}\n")
        f.write(f"Date:           {ts}\n")
        f.write(f"Run type:       {run_type}\n")
        f.write(f"Model:          {MODEL_PATH}\n")
        f.write(f"Dataset:        DATA3  (NORMAL / PNEUMONIA_BACTERIAL / PNEUMONIA_VIRUS)\n")
        f.write(f"Total samples:  {n_samples}\n")
        f.write(f"  Per class:    {sum(1 for _ in range(n_samples))} total  "
                f"(approx {n_samples // 3} each)\n\n")

        f.write("=" * 55 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 55 + "\n")
        f.write(f"Accuracy:           {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)\n")
        f.write(f"Balanced Accuracy:  {m['balanced_accuracy']:.4f}  ({m['balanced_accuracy']*100:.2f}%)\n")
        f.write(f"Precision (macro):  {m['precision_macro']:.4f}\n")
        f.write(f"Recall (macro):     {m['recall_macro']:.4f}\n")
        f.write(f"F1 Score (macro):   {m['f1_macro']:.4f}\n")
        f.write(f"AUC-ROC (macro):    {m['auc_roc_macro']:.4f}\n\n")

        f.write("=" * 55 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("=" * 55 + "\n")
        header = f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}\n"
        f.write(header)
        f.write("-" * 58 + "\n")
        for i, cls in enumerate(CLASS_NAMES):
            p  = m["precision_cls"][i] if i < len(m["precision_cls"]) else float("nan")
            r  = m["recall_cls"][i]    if i < len(m["recall_cls"])    else float("nan")
            fi = m["f1_cls"][i]        if i < len(m["f1_cls"])        else float("nan")
            f.write(f"{cls:<25} {p:>10.4f} {r:>10.4f} {fi:>10.4f}\n")

        f.write("\n")
        f.write("=" * 55 + "\n")
        f.write("FULL CLASSIFICATION REPORT\n")
        f.write("=" * 55 + "\n")
        f.write(m["report"])
        f.write("\n")

        f.write("=" * 55 + "\n")
        f.write("CONFUSION MATRIX  (rows=True label, cols=Predicted)\n")
        f.write("=" * 55 + "\n")
        col_w = 22
        f.write(f"{'':>25}")
        for cls in CLASS_NAMES:
            f.write(f"{cls[:col_w]:>{col_w}}")
        f.write("\n")
        f.write("-" * (25 + col_w * len(CLASS_NAMES)) + "\n")
        for i, row_cls in enumerate(CLASS_NAMES):
            f.write(f"{row_cls:<25}")
            for val in m["confusion_matrix"][i]:
                f.write(f"{val:>{col_w}}")
            f.write("\n")

    print(f"\nMetrics saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# Output: plots
# ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, out_dir: str = OUTPUT_DIR) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = sk_metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASS_NAMES
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Validation (ResNet50)")
    plt.tight_layout()
    out = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


def plot_roc(all_labels: list, all_probs: list, out_dir: str = OUTPUT_DIR) -> None:
    probs_np = np.array(all_probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASS_NAMES):
        bl = (np.array(all_labels) == i).astype(int)
        try:
            fpr, tpr, _ = sk_metrics.roc_curve(bl, probs_np[:, i])
            auc_val = sk_metrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls}  (AUC = {auc_val:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_title("ROC Curve — Validation (ResNet50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ──────────────────────────────────────────────────────────────────────
# Output: Grad-CAM
# ──────────────────────────────────────────────────────────────────────

def generate_gradcam(
    model: nn.Module,
    all_labels: list,
    all_preds:  list,
    all_probs:  list,
    all_paths:  list,
    device: torch.device,
    out_dir: str = OUTPUT_DIR,
) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("\npytorch-grad-cam not installed — skipping Grad-CAM.")
        print("Install with:  pip install grad-cam")
        return

    target_layer = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    transform = get_transform()

    # Split into correct / incorrect
    correct_idxs   = [i for i in range(len(all_labels)) if all_labels[i] == all_preds[i]]
    incorrect_idxs = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]

    rng = random.Random(SEED)
    correct_sample   = rng.sample(correct_idxs,   min(NUM_GRADCAM, len(correct_idxs)))
    incorrect_sample = rng.sample(incorrect_idxs, min(NUM_GRADCAM, len(incorrect_idxs)))

    def _save_cam(idx: int, tag: str, cnt: int) -> None:
        img_path  = all_paths[idx]
        true_cls  = CLASS_NAMES[all_labels[idx]]
        pred_cls  = CLASS_NAMES[all_preds[idx]]
        conf      = all_probs[idx][all_preds[idx]]

        raw = Image.open(img_path).convert("RGB").resize((224, 224))
        inp = transform(raw).unsqueeze(0).to(device)

        gc      = cam(input_tensor=inp, targets=[ClassifierOutputTarget(all_labels[idx])])[0]
        rgb_f32 = np.array(raw, dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb_f32, gc, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(raw)
        axes[0].set_title("Original X-ray")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM\nTrue: {true_cls}\nPred: {pred_cls}  (conf {conf:.2f})")
        axes[1].axis("off")
        verdict = "CORRECT" if tag == "correct" else "INCORRECT"
        plt.suptitle(f"{verdict} prediction #{cnt + 1}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fname = f"gradcam_{tag}_{cnt + 1}_true_{true_cls}_pred_{pred_cls}.png"
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {fpath}")

    print(f"\nGenerating Grad-CAM ({NUM_GRADCAM} correct, {NUM_GRADCAM} incorrect)…")
    for cnt, idx in enumerate(correct_sample):
        _save_cam(idx, "correct", cnt)
    if incorrect_sample:
        for cnt, idx in enumerate(incorrect_sample):
            _save_cam(idx, "incorrect", cnt)
    else:
        print("  No incorrect predictions found — skipping incorrect Grad-CAMs.")


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ResNet50 on chest X-ray data (NORMAL / BACTERIA / VIRUS)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick sanity-check: load ~10 images, run inference, print metrics (no plots saved)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help=(
            "Path to a folder containing NORMAL/ and PNEUMONIA/ subfolders. "
            "When provided, ALL available images are used (no per-class cap). "
            "Outputs are saved next to that folder in a validation_resnet50/ subfolder."
        ),
    )
    args = parser.parse_args()

    # ── Resolve paths based on --test_dir ────────────────────────────
    if args.test_dir:
        test_dir      = Path(args.test_dir)
        normal_dir    = str(test_dir / "NORMAL")
        pneumonia_dir = str(test_dir / "PNEUMONIA")
        out_dir       = str(test_dir.parent / "validation_resnet50")
        n_per_class   = 0   # 0 = use all
    else:
        normal_dir    = None
        pneumonia_dir = None
        out_dir       = OUTPUT_DIR
        n_per_class   = N_PER_CLASS

    set_seed(SEED)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test:
        run_type = "TEST RUN (~10 images)"
    elif args.test_dir:
        run_type = f"FULL RUN — {args.test_dir}"
    else:
        run_type = f"FULL RUN (~{N_PER_CLASS * 3} images, DATA3)"

    print(f"Device  : {device}")
    print(f"Run     : {run_type}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Output  : {out_dir}")

    # ── Collect samples ──────────────────────────────────────────────
    if args.test:
        n = max(4, N_TEST_TOTAL // len(CLASS_NAMES))
        samples = collect_samples(n_per_class=n, seed=SEED,
                                  normal_dir=normal_dir, pneumonia_dir=pneumonia_dir)
        samples = samples[:N_TEST_TOTAL]
    else:
        samples = collect_samples(n_per_class=n_per_class, seed=SEED,
                                  normal_dir=normal_dir, pneumonia_dir=pneumonia_dir)

    n0 = sum(1 for _, l in samples if l == 0)
    n1 = sum(1 for _, l in samples if l == 1)
    n2 = sum(1 for _, l in samples if l == 2)
    print(f"\nSamples : {len(samples)} total  |  "
          f"NORMAL={n0}  BACTERIA={n1}  VIRUS={n2}")

    # ── Load model ───────────────────────────────────────────────────
    print("\nLoading model…")
    model = build_model(device)
    print("Model loaded.")

    # ── DataLoader ───────────────────────────────────────────────────
    dataset = DATA3Dataset(samples, get_transform())
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Inference ────────────────────────────────────────────────────
    print("\nRunning inference…")
    all_labels, all_preds, all_probs, all_paths = run_inference(model, loader, device)

    # ── Compute & display metrics ────────────────────────────────────
    m = compute_metrics(all_labels, all_preds, all_probs)
    print_metrics(m)

    if args.test:
        print("\n[TEST COMPLETE]  Re-run without --test for full run + saved plots.")
        return

    # ── Save outputs ─────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    save_metrics_txt(m, len(samples), run_type, out_dir)
    plot_confusion_matrix(m["confusion_matrix"], out_dir)
    plot_roc(all_labels, all_probs, out_dir)
    generate_gradcam(model, all_labels, all_preds, all_probs, all_paths, device, out_dir)
    print(f"\nAll outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
