"""Smart Estimator — pre-flight analysis of dataset and resource estimation.

Scans the local dataset to compute file count, total size, and average image
dimensions. Combines this with historical benchmarks to estimate VRAM needs
and training time for common models.
"""

import os
import struct
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

LogCallback = Optional[Callable[[str], None]]

# ---------------------------------------------------------------------------
# Historical benchmarks (images/sec on batch_size=32, single GPU)
# (model_name → {vram_gb, img_per_sec_map: {gpu_tier → ips}})
# ---------------------------------------------------------------------------
_BENCHMARKS = {
    "resnet50": {
        "vram_base_gb": 4.0,       # VRAM for model + overhead
        "vram_per_k_images": 0.3,  # Additional VRAM per 1k batch images (roughly)
        "ips": {"low": 120, "mid": 300, "high": 700},  # images/sec by GPU tier
    },
    "densenet121": {
        "vram_base_gb": 3.5,
        "vram_per_k_images": 0.25,
        "ips": {"low": 100, "mid": 260, "high": 550},
    },
    "efficientnet_b0": {
        "vram_base_gb": 3.0,
        "vram_per_k_images": 0.2,
        "ips": {"low": 150, "mid": 350, "high": 800},
    },
    "convnext": {
        "vram_base_gb": 4.5,
        "vram_per_k_images": 0.35,
        "ips": {"low": 90, "mid": 220, "high": 500},
    },
    "custom_cnn": {
        "vram_base_gb": 2.0,
        "vram_per_k_images": 0.15,
        "ips": {"low": 180, "mid": 420, "high": 900},
    },
}

# GPU tier classification by VRAM
_GPU_TIERS = {
    "low": (0, 8),      # <=8 GB — e.g. RTX 2060/3060
    "mid": (8, 24),      # 8–24 GB — e.g. RTX 3090, A5000
    "high": (24, 999),   # 24+ GB — e.g. A100, H100
}


def _gpu_tier(vram_gb: float) -> str:
    for tier, (lo, hi) in _GPU_TIERS.items():
        if lo < vram_gb <= hi:
            return tier
    return "mid"


def _fast_image_dimensions(filepath: str) -> Optional[Tuple[int, int]]:
    """Read image width × height from header bytes without loading full image."""
    try:
        with open(filepath, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return w, h
            if head[:2] == b"\xff\xd8":  # JPEG
                f.seek(2); f.read(2)
                while True:
                    marker, size = struct.unpack(">HH", f.read(4))
                    if 0xFFC0 <= marker <= 0xFFC2:
                        f.read(1)
                        h, w = struct.unpack(">HH", f.read(4))
                        return w, h
                    f.seek(size - 2, 1)
    except Exception:
        pass
    return None


class DatasetProfile:
    """Summary statistics of a local dataset."""

    def __init__(self):
        self.total_files: int = 0
        self.total_bytes: int = 0
        self.image_files: int = 0
        self.num_classes: int = 0
        self.class_names: list = []
        self.avg_width: int = 0
        self.avg_height: int = 0
        self.samples_per_class: Dict[str, int] = {}

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)


def profile_dataset(
    path: str,
    task_type: str = "classification",
    log_cb: LogCallback = None,
) -> DatasetProfile:
    """Scan a local dataset directory and return a DatasetProfile.

    For classification: counts images per class sub-folder.
    For regression: counts CSV/Excel rows.
    """
    prof = DatasetProfile()
    root = Path(path)

    if not root.is_dir():
        if log_cb:
            log_cb(f"Dataset path not found: {path}")
        return prof

    _IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    if task_type == "classification":
        # Check for sub-folders
        subdirs = sorted(
            [d for d in root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        if subdirs:
            # Organized dataset
            prof.class_names = [d.name for d in subdirs]
            prof.num_classes = len(subdirs)
            widths, heights, sampled = [], [], 0
            for cls_dir in subdirs:
                imgs = [f for f in cls_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in _IMG_EXT]
                count = len(imgs)
                prof.samples_per_class[cls_dir.name] = count
                prof.image_files += count
                for img_f in imgs:
                    prof.total_bytes += img_f.stat().st_size
                    if sampled < 50:
                        dims = _fast_image_dimensions(str(img_f))
                        if dims:
                            widths.append(dims[0])
                            heights.append(dims[1])
                            sampled += 1
            if widths:
                prof.avg_width = int(sum(widths) / len(widths))
                prof.avg_height = int(sum(heights) / len(heights))
        else:
            # Flat folder — count all image files
            imgs = [f for f in root.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMG_EXT]
            prof.image_files = len(imgs)
            for img_f in imgs:
                prof.total_bytes += img_f.stat().st_size

        prof.total_files = prof.image_files

    else:
        # Regression — count CSV/Excel rows
        for f in root.iterdir():
            if f.is_file():
                prof.total_files += 1
                prof.total_bytes += f.stat().st_size
                if f.suffix.lower() in (".csv", ".xlsx", ".xls"):
                    try:
                        import csv
                        with open(str(f), newline="", encoding="utf-8") as csvf:
                            prof.image_files = sum(1 for _ in csvf) - 1  # minus header
                    except Exception:
                        pass

    return prof


def estimate_resources(
    profile: DatasetProfile,
    model_name: str = "resnet50",
    batch_size: int = 32,
    epochs: int = 50,
    gpu_vram_gb: float = 16.0,
    log_cb: LogCallback = None,
) -> Dict[str, object]:
    """Estimate required VRAM, training time, and recommended GPU config.

    Returns a dict with:
        - recommended_vram_gb
        - estimated_time_minutes
        - estimated_upload_minutes (at ~50 MB/s)
        - gpu_tier
        - dataset_mb
        - images_per_epoch
    """
    bench = _BENCHMARKS.get(model_name, _BENCHMARKS["resnet50"])
    tier = _gpu_tier(gpu_vram_gb)

    images_per_epoch = max(profile.image_files, profile.total_files)

    # VRAM estimate: base + batch contribution
    vram_needed = bench["vram_base_gb"] + (batch_size / 32) * 1.5

    # Training time: images × epochs ÷ throughput
    ips = bench["ips"].get(tier, 200)
    total_images = images_per_epoch * epochs
    train_seconds = total_images / ips if ips > 0 else 9999
    train_minutes = train_seconds / 60

    # Upload time estimate (assume ~50 MB/s effective throughput over SSH)
    upload_minutes = (profile.total_mb / 50) / 60 if profile.total_mb > 0 else 0

    result = {
        "recommended_vram_gb": round(vram_needed, 1),
        "estimated_time_minutes": round(train_minutes, 1),
        "estimated_upload_minutes": round(upload_minutes, 1),
        "gpu_tier": tier,
        "dataset_mb": round(profile.total_mb, 1),
        "images_per_epoch": images_per_epoch,
    }

    if log_cb:
        log_cb("─── Smart Estimator ───")
        log_cb(f"  Dataset: {profile.image_files} images, {profile.total_mb:.1f} MB")
        if profile.num_classes > 0:
            log_cb(f"  Classes: {profile.num_classes} — {profile.class_names[:8]}{'…' if profile.num_classes > 8 else ''}")
            imbalance = _class_imbalance(profile.samples_per_class)
            if imbalance > 3.0:
                log_cb(f"  ⚠ Class imbalance detected (ratio {imbalance:.1f}:1)")
        if profile.avg_width:
            log_cb(f"  Avg image: {profile.avg_width}×{profile.avg_height} px")
        log_cb(f"  Recommended VRAM: ≥{vram_needed:.0f} GB")
        log_cb(f"  Est. training time: ~{train_minutes:.0f} min ({tier}-tier GPU @ ~{ips} img/s)")
        log_cb(f"  Est. upload time: ~{max(upload_minutes, 0.1):.1f} min")
        log_cb("───────────────────────")

    return result


def _class_imbalance(samples: Dict[str, int]) -> float:
    """Return max/min class ratio. 1.0 = perfectly balanced."""
    if not samples:
        return 1.0
    counts = list(samples.values())
    mn = min(counts) if counts else 1
    mx = max(counts) if counts else 1
    return mx / max(mn, 1)
