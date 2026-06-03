"""High-level orchestration logic that ties Vast.ai API and SSH together."""

import logging
import os
import threading
import time
from typing import Callable, Optional

from config import ExperimentConfig
from estimator import profile_dataset, estimate_resources
from ssh_manager import SSHManager, SSHError
from vast_api import VastAPI, VastAPIError

logger = logging.getLogger(__name__)

LogCallback = Optional[Callable[[str], None]]

# System dependencies needed by OpenCV/Grad-CAM in the PyTorch Docker image.
_SYSTEM_DEPS = (
    "if command -v apt-get >/dev/null 2>&1; then "
    "apt-get update -qq && "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
    "libgl1 libglib2.0-0; "
    "else echo 'apt-get not found; skipping OpenCV system dependencies'; fi"
)

# Dependencies to install inside the container
# NOTE: torch + torchvision are already present in pytorch/pytorch:latest —
#       do NOT re-install them here or pip will downgrade to the CPU/stable
#       build and break CUDA support before the fix script can run.
_REMOTE_DEPS = (
    "pip install --quiet 'numpy<2' timm scikit-learn matplotlib "
    "grad-cam pillow tqdm pandas openpyxl"
)

_PRE_SPLIT_VALIDATE_SCRIPT = r'''
import os, re, shutil, sys
from collections import defaultdict

root = "/workspace/data"
three_class = __THREE_CLASS__
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
split_names = ("train", "val", "test")
required_three_classes = ("NORMAL", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRUS")

def resolve_split(names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return name, path
    print(f"ERROR: Missing split folder. Expected one of: {names}")
    sys.exit(1)

def iter_images(path):
    for current, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            if filename.lower().endswith(exts):
                yield os.path.join(current, filename)

def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    suffix = 1
    while True:
        candidate = f"{base}_{suffix}{ext}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1

def target_class_name(cls, source_path):
    filename = os.path.basename(source_path).lower()
    text = f"{cls}/{source_path}".lower()
    cls_upper = cls.upper()
    if three_class:
        if "bacteria" in text or "bacterial" in text:
            return "PNEUMONIA_BACTERIAL"
        if "virus" in text or "viral" in text:
            return "PNEUMONIA_VIRUS"
    if "normal" in text or filename.startswith("im-") or filename.startswith("normal2-im-"):
        return "NORMAL"
    if cls_upper in {"NORMAL", "PNEUMONIA", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRUS"}:
        return cls_upper
    return cls

def cleanup_empty_dirs(path):
    for current, dirs, _ in os.walk(path, topdown=False):
        for dirname in dirs:
            candidate = os.path.join(current, dirname)
            try:
                if not os.listdir(candidate):
                    os.rmdir(candidate)
            except OSError:
                pass

def organize_split(split_dir):
    files_by_class = defaultdict(list)
    for entry in sorted(os.listdir(split_dir)):
        entry_path = os.path.join(split_dir, entry)
        if os.path.isdir(entry_path):
            for source_path in iter_images(entry_path):
                files_by_class[target_class_name(entry, source_path)].append(source_path)
        elif os.path.isfile(entry_path) and entry.lower().endswith(exts):
            files_by_class[target_class_name("UNKNOWN", entry_path)].append(entry_path)
    if not files_by_class:
        print(f"ERROR: No image files found in {split_dir}")
        sys.exit(1)
    for class_name, source_paths in sorted(files_by_class.items()):
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for source_path in source_paths:
            if os.path.abspath(os.path.dirname(source_path)) == os.path.abspath(class_dir):
                continue
            dest_path = unique_path(os.path.join(class_dir, os.path.basename(source_path)))
            shutil.move(source_path, dest_path)
    cleanup_empty_dirs(split_dir)

def image_files(path):
    return sorted(
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename)) and filename.lower().endswith(exts)
    )

def count_by_class(split_dir):
    counts = {}
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = len(image_files(class_dir))
        if count > 0:
            counts[class_name] = count
    return counts

def group_key(source_path):
    stem = os.path.splitext(os.path.basename(source_path))[0]
    stem = re.sub(r'_aug_\d+$', '', stem, flags=re.I)
    for pattern in (r'(person\d+)', r'(NORMAL2-IM-\d+|IM-\d+)', r'((?:BACTERIA|VIRUS)-\d+)'):
        match = re.search(pattern, stem, re.I)
        if match:
            return match.group(1).lower()
    return stem.lower()

def validate_no_overlap(split_dirs):
    groups = defaultdict(lambda: defaultdict(set))
    for split_name, split_dir in split_dirs.items():
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for source_path in image_files(class_dir):
                groups[class_name][split_name].add(group_key(source_path))
    for class_name, by_split in groups.items():
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = by_split[left] & by_split[right]
            print(f"{class_name} {left}/{right} group overlap: {len(overlap)}")
            if overlap:
                print(f"ERROR: Data leakage detected for {class_name}: {sorted(overlap)[:10]}")
                sys.exit(1)

split_dirs = {
    "train": resolve_split(("train",))[1],
    "val": resolve_split(("val", "validation"))[1],
    "test": resolve_split(("test",))[1],
}
for split_dir in split_dirs.values():
    organize_split(split_dir)

counts = {name: count_by_class(path) for name, path in split_dirs.items()}
classes = {name: sorted(values) for name, values in counts.items()}
if not classes["train"]:
    print("ERROR: No class folders found in train split.")
    sys.exit(1)
if classes["train"] != classes["val"] or classes["train"] != classes["test"]:
    print(f"ERROR: Class folders differ: {classes}")
    sys.exit(1)
if three_class and classes["train"] != list(required_three_classes):
    print(f"ERROR: Three-class mode requires {list(required_three_classes)}, got {classes['train']}")
    sys.exit(1)
validate_no_overlap(split_dirs)

total = sum(sum(values.values()) for values in counts.values())
for split_name in split_names:
    split_total = sum(counts[split_name].values())
    pct = (split_total / total * 100.0) if total else 0.0
    print(f"{split_name}: {split_total}/{total} images ({pct:.2f}%) per class {counts[split_name]}")
print(f"Pre-split dataset OK. Classes: {classes['train']}")
'''

_AUTO_SPLIT_CLASSIFICATION_SCRIPT = r'''
import math, os, random, re, shutil, sys
from collections import defaultdict

root = "/workspace/data"
tmp_root = "/workspace/data_split_tmp"
seed = __SEED__
three_class = __THREE_CLASS__
ratios = {
    "train": __TRAIN_RATIO__,
    "val": __VAL_RATIO__,
    "test": __TEST_RATIO__,
}
split_names = ("train", "val", "test")
split_aliases = {"train", "val", "validation", "test"}
input_split_options = {
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test",),
}
required_three_classes = ("NORMAL", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRUS")
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def image_files(path):
    return sorted(
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename)) and filename.lower().endswith(exts)
    )

def iter_images(path):
    for current, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            if filename.lower().endswith(exts):
                yield os.path.join(current, filename)

def target_class_name(cls, source_path):
    filename = os.path.basename(source_path).lower()
    text = f"{cls}/{source_path}".lower()
    cls_upper = cls.upper()
    if three_class:
        if "bacteria" in text or "bacterial" in text:
            return "PNEUMONIA_BACTERIAL"
        if "virus" in text or "viral" in text:
            return "PNEUMONIA_VIRUS"
    if "normal" in text or filename.startswith("im-") or filename.startswith("normal2-im-"):
        return "NORMAL"
    if cls_upper in {"NORMAL", "PNEUMONIA", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRUS"}:
        return cls_upper
    return cls

def collect_class_files():
    resolved_splits = {}
    for split_name, aliases in input_split_options.items():
        for alias in aliases:
            candidate = os.path.join(root, alias)
            if os.path.isdir(candidate):
                resolved_splits[split_name] = candidate
                break

    class_files = defaultdict(list)
    if resolved_splits:
        missing = [name for name in split_names if name not in resolved_splits]
        if missing:
            print(f"ERROR: Found partial split folders, missing: {missing}")
            print("Provide train/val/test folders, or provide only class folders like NORMAL/PNEUMONIA.")
            sys.exit(1)
        print("Pooling existing train/val/test folders by class before re-splitting.")
        for split_name in split_names:
            split_dir = resolved_splits[split_name]
            for entry in sorted(os.listdir(split_dir)):
                entry_path = os.path.join(split_dir, entry)
                if os.path.isdir(entry_path):
                    for source_path in iter_images(entry_path):
                        class_files[target_class_name(entry, source_path)].append(source_path)
                elif os.path.isfile(entry_path) and entry.lower().endswith(exts):
                    class_files[target_class_name("UNKNOWN", entry_path)].append(entry_path)
        return class_files

    for source_path in image_files(root):
        class_files[target_class_name("UNKNOWN", source_path)].append(source_path)
    for entry in sorted(os.listdir(root)):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path) or entry in split_aliases:
            continue
        for source_path in iter_images(entry_path):
            class_files[target_class_name(entry, source_path)].append(source_path)
    return class_files

def group_key(source_path):
    stem = os.path.splitext(os.path.basename(source_path))[0]
    stem = re.sub(r'_aug_\d+$', '', stem, flags=re.I)
    for pattern in (r'(person\d+)', r'(NORMAL2-IM-\d+|IM-\d+)', r'((?:BACTERIA|VIRUS)-\d+)'):
        match = re.search(pattern, stem, re.I)
        if match:
            return match.group(1).lower()
    return stem.lower()

def split_counts(total):
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        print("ERROR: Split ratios must sum to a positive value.")
        sys.exit(1)
    normalized = [ratios[name] / ratio_sum for name in split_names]
    positive = [index for index, ratio in enumerate(normalized) if ratio > 0]
    if total < len(positive):
        print(f"ERROR: Only {total} items available for {len(positive)} non-empty splits.")
        sys.exit(1)
    raw = [total * ratio for ratio in normalized]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (raw[index] - counts[index], normalized[index]), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    for index in positive:
        if counts[index] == 0:
            donors = [candidate for candidate in positive if counts[candidate] > 1]
            if not donors:
                print("ERROR: Cannot keep all requested splits non-empty.")
                sys.exit(1)
            donor = max(donors, key=lambda candidate: counts[candidate])
            counts[donor] -= 1
            counts[index] += 1
    return dict(zip(split_names, counts))

def split_groups_for_class(class_name, files):
    groups_by_key = defaultdict(list)
    for source_path in files:
        groups_by_key[group_key(source_path)].append(source_path)

    positive_splits = [name for name in split_names if ratios[name] > 0]
    if len(groups_by_key) < len(positive_splits):
        print(
            f"ERROR: Class {class_name} has only {len(groups_by_key)} patient/source groups, "
            f"but {len(positive_splits)} non-empty splits are requested."
        )
        sys.exit(1)

    groups = sorted(groups_by_key.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
    target_counts = split_counts(len(files))
    assigned = {name: [] for name in split_names}
    assigned_counts = {name: 0 for name in split_names}

    ordered_splits = sorted(positive_splits, key=lambda name: target_counts[name], reverse=True)
    if ordered_splits:
        split_name, (_, paths) = ordered_splits[0], groups.pop(0)
        assigned[split_name].extend(paths)
        assigned_counts[split_name] += len(paths)
        for split_name in sorted(ordered_splits[1:], key=lambda name: target_counts[name]):
            _, paths = groups.pop()
            assigned[split_name].extend(paths)
            assigned_counts[split_name] += len(paths)

    rng = random.Random(f"{seed}:{class_name}:groups")
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    for _, paths in groups:
        best_split = max(
            positive_splits,
            key=lambda name: (
                (target_counts[name] - assigned_counts[name]) / max(target_counts[name], 1),
                target_counts[name] - assigned_counts[name],
            ),
        )
        assigned[best_split].extend(paths)
        assigned_counts[best_split] += len(paths)

    print(f"{class_name}: {len(files)} images in {len(groups_by_key)} groups -> {assigned_counts}")
    return assigned

def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    suffix = 1
    while True:
        candidate = f"{base}_{suffix}{ext}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1

def count_by_class(split_dir):
    counts = {}
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = len(image_files(class_dir))
        if count > 0:
            counts[class_name] = count
    return counts

def validate_split_dirs():
    counts = {}
    classes = None
    total = 0
    groups = defaultdict(lambda: defaultdict(set))
    for split_name in split_names:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            print(f"ERROR: Missing generated split folder: {split_dir}")
            sys.exit(1)
        split_counts_by_class = count_by_class(split_dir)
        if not split_counts_by_class:
            print(f"ERROR: Split {split_name} is empty.")
            sys.exit(1)
        split_classes = sorted(split_counts_by_class)
        if classes is None:
            classes = split_classes
        elif classes != split_classes:
            print(f"ERROR: Class folders differ in {split_name}: expected {classes}, got {split_classes}")
            sys.exit(1)
        counts[split_name] = split_counts_by_class
        total += sum(split_counts_by_class.values())
        for class_name in split_classes:
            class_dir = os.path.join(split_dir, class_name)
            for source_path in image_files(class_dir):
                groups[class_name][split_name].add(group_key(source_path))

    if three_class and classes != list(required_three_classes):
        print(f"ERROR: Three-class mode requires {list(required_three_classes)}, got {classes}")
        sys.exit(1)

    for class_name, by_split in groups.items():
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = by_split[left] & by_split[right]
            print(f"{class_name} {left}/{right} group overlap: {len(overlap)}")
            if overlap:
                print(f"ERROR: Data leakage detected for {class_name}: {sorted(overlap)[:10]}")
                sys.exit(1)

    ratio_sum = sum(ratios.values())
    for split_name in split_names:
        split_total = sum(counts[split_name].values())
        actual_pct = split_total / total * 100.0
        requested_pct = ratios[split_name] / ratio_sum * 100.0
        print(f"{split_name}: {split_total}/{total} images ({actual_pct:.2f}%, requested {requested_pct:.2f}%) per class {counts[split_name]}")
    return counts

class_files = collect_class_files()
class_files = {class_name: paths for class_name, paths in class_files.items() if paths}
if not class_files:
    print("ERROR: No class folders found. Expected folders like NORMAL/ and PNEUMONIA/.")
    sys.exit(1)
if three_class:
    missing = [class_name for class_name in required_three_classes if not class_files.get(class_name)]
    if missing:
        print(f"ERROR: Missing three-class targets after inference: {missing}")
        print("PNEUMONIA images must include bacteria/bacterial or virus/viral tokens in filename/path.")
        sys.exit(1)

class_assignments = {
    class_name: split_groups_for_class(class_name, files)
    for class_name, files in sorted(class_files.items())
}

shutil.rmtree(tmp_root, ignore_errors=True)
for split_name in split_names:
    os.makedirs(os.path.join(tmp_root, split_name), exist_ok=True)

for class_name in sorted(class_assignments):
    assignments = class_assignments[class_name]
    for split_name in split_names:
        dest_dir = os.path.join(tmp_root, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for source_path in assignments[split_name]:
            shutil.move(source_path, unique_path(os.path.join(dest_dir, os.path.basename(source_path))))

for entry in os.listdir(root):
    path = os.path.join(root, entry)
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
for split_name in split_names:
    shutil.move(os.path.join(tmp_root, split_name), os.path.join(root, split_name))
shutil.rmtree(tmp_root, ignore_errors=True)

validate_split_dirs()
print("Auto split dataset OK.")
'''

# Subsample remote dataset to N images per class (0 = disabled)
_SUBSAMPLE_SCRIPT_TEMPLATE = r'''
import os, random
data_dir = "/workspace/data"
max_per_class = {max_n}
seed = {seed}
if max_per_class <= 0:
    print("Subsampling disabled.")
    exit(0)
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
for cls in sorted(os.listdir(data_dir)):
    cls_path = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    files = [f for f in os.listdir(cls_path) if f.lower().endswith(exts)]
    if len(files) <= max_per_class:
        print(f"  {{cls}}: {{len(files)}} images — keeping all")
        continue
    random.seed(seed)
    to_keep = set(random.sample(files, max_per_class))
    removed = 0
    for f in files:
        if f not in to_keep:
            os.remove(os.path.join(cls_path, f))
            removed += 1
    print(f"  {{cls}}: kept {{max_per_class}}/{{len(files)}} (removed {{removed}})")
print("Subsampling done.")
'''

# CUDA fixup for newer GPUs (RTX 5000-series / Blackwell)
_CUDA_FIX_SCRIPT = r'''
import subprocess, sys, re
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap",
                                   "--format=csv,noheader"], text=True)
    print(f"GPU: {out.strip()}")
    cap = re.search(r"(\d+\.\d+)", out)
    if cap and float(cap.group(1)) >= 12.0:
        print("Detected Blackwell+ GPU (sm_120) — replacing PyTorch with cu128 build…")
        # Uninstall first — the conda-managed torch would otherwise shadow the
        # pip-installed one and still crash with "no kernel image" at runtime.
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                         "torch", "torchvision", "torchaudio"])
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                                   "--index-url", "https://download.pytorch.org/whl/cu128",
                                   "torch", "torchvision", "torchaudio"])
            import importlib, importlib.util
            spec = importlib.util.find_spec("torch")
            print(f"PyTorch cu128 installed → {spec.origin if spec else '?'}")
        except subprocess.CalledProcessError:
            print("Stable cu128 failed — trying nightly…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                                   "--index-url", "https://download.pytorch.org/whl/nightly/cu128",
                                   "torch", "torchvision", "torchaudio"])
            print("PyTorch nightly cu128 installed.")
    elif cap and float(cap.group(1)) >= 8.9:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available — upgrading PyTorch for SM89+ support…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                                   "--upgrade", "torch", "torchvision"])
            print("PyTorch upgraded.")
        else:
            print(f"CUDA OK: torch {torch.__version__}, CUDA {torch.version.cuda}")
    else:
        print("Standard GPU — no CUDA fix needed.")
except Exception as e:
    print(f"CUDA check skipped: {e}")
'''


class Orchestrator:
    """Coordinates the full lifecycle: rent → setup → train → download → destroy."""

    def __init__(self, config: ExperimentConfig, log_cb: LogCallback = None,
                 telemetry_cb: Optional[Callable] = None):
        self.config = config
        self.log_cb = log_cb or (lambda msg: None)
        self.telemetry_cb = telemetry_cb  # GUI callback for live chart data
        self.vast: Optional[VastAPI] = None
        self.ssh: Optional[SSHManager] = None
        self.instance_id: Optional[int] = None
        self._cancel = threading.Event()
        self._telemetry_stop = threading.Event()
        self._attached = False  # True when manually connected to existing instance

    # ------------------------------------------------------------------
    # Public entry-point (runs in a background thread)
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full pipeline. Intended to be called from a thread."""
        try:
            if not self.config.use_builtin:
                self._step_estimate()
                if self._cancelled():
                    return
            else:
                self.log_cb("=" * 60)
                self.log_cb("[0/6] Skipping estimation (built-in CIFAR-100 mode).")
            self._step_search_and_rent()
            if self._cancelled():
                return
            self._step_connect_ssh()
            if self._cancelled():
                return
            self._step_setup_environment()
            if self._cancelled():
                return
            if not self.config.use_builtin:
                self._step_upload_data()
            else:
                self._step_upload_script_only()
            if self._cancelled():
                return
            self._step_run_training()
            if self._cancelled():
                return
            self._step_download_results()
            self.log_cb("=" * 60)
            self.log_cb("Pipeline finished successfully.")
            self.log_cb("You can now terminate the instance to stop billing.")
            self.log_cb("SSH Console is available for interactive commands.")
        except (VastAPIError, SSHError) as exc:
            self.log_cb(f"\n*** ERROR: {exc}")
            logger.exception("Orchestration error")
        except Exception as exc:
            self.log_cb(f"\n*** UNEXPECTED ERROR: {exc}")
            logger.exception("Unexpected orchestration error")
        finally:
            # Keep SSH alive for interactive console — disconnect on destroy
            pass

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------
    # Manual attach (connect to existing instance)
    # ------------------------------------------------------------------
    def attach(self, ssh_host: str, ssh_port: int) -> None:
        """Connect to an already-running instance via SSH.

        Sets ``_attached = True`` so the pipeline can skip provisioning and
        conditionally skip the upload step.
        """
        self.log_cb("=" * 60)
        self.log_cb(f"Attaching to existing instance at {ssh_host}:{ssh_port}…")
        self._ssh_host = ssh_host
        self._ssh_port = ssh_port
        self._attached = True

        self.ssh = SSHManager(
            host=ssh_host,
            port=ssh_port,
            key_filename=self.config.ssh_key_path,
        )
        self.ssh.connect(log_cb=self.log_cb)
        self.log_cb("Attached successfully.")

    def run_attached(self) -> None:
        """Execute the pipeline on a manually-attached instance.

        Skips provisioning [1/6]. Skips upload [4/6] if /workspace/data
        already has data. Always re-uploads train.py and runs setup.
        """
        try:
            if not self.ssh or not self.ssh.is_connected:
                raise SSHError("Not attached — call attach() first.")

            # Step 0: optionally estimate (skip for builtin)
            if not self.config.use_builtin:
                self._step_estimate()
                if self._cancelled():
                    return
            else:
                self.log_cb("=" * 60)
                self.log_cb("[0/6] Skipping estimation (built-in CIFAR-100 mode).")

            # Step 1: skip — already attached
            self.log_cb("=" * 60)
            self.log_cb("[1/6] Skipped provisioning (attached to existing instance).")

            # Step 2: skip — already connected
            self.log_cb("=" * 60)
            self.log_cb("[2/6] SSH already connected.")

            # Step 3: setup
            self._step_setup_environment()
            if self._cancelled():
                return

            # Step 4: check if data exists, skip upload if so
            if self.config.use_builtin:
                self._step_upload_script_only()
            else:
                remote_status = self.ssh.verify_remote_data(log_cb=self.log_cb)
                if remote_status.get("data"):
                    self.log_cb("=" * 60)
                    self.log_cb("[4/6] Dataset already present — skipping upload.")
                    # Still re-upload train.py (could have been modified)
                    self._upload_train_script()
                    if self.config.task_type == "classification":
                        self._prepare_remote_classification_data()
                else:
                    self._step_upload_data()
            if self._cancelled():
                return

            # Step 5: train
            self._step_run_training()
            if self._cancelled():
                return

            # Step 6: download
            self._step_download_results()

            self.log_cb("=" * 60)
            self.log_cb("Pipeline finished successfully (attached mode).")
            self.log_cb("SSH Console is available for interactive commands.")
        except (VastAPIError, SSHError) as exc:
            self.log_cb(f"\n*** ERROR: {exc}")
            logger.exception("Orchestration error (attached)")
        except Exception as exc:
            self.log_cb(f"\n*** UNEXPECTED ERROR: {exc}")
            logger.exception("Unexpected error (attached)")

    def _local_subsample(self, src_dir: str, n: int, seed: int):
        """Copy up to *n* images per class-subfolder into a temp directory.

        Returns (tmp_path, tmp_path) so the caller uses tmp_path for upload
        and cleans it up afterwards. Requires per-class sub-directories.
        """
        import random
        import shutil
        import tempfile

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        tmp = tempfile.mkdtemp(prefix="vast_subsample_")
        total = 0
        for cls in os.listdir(src_dir):
            cls_src = os.path.join(src_dir, cls)
            if not os.path.isdir(cls_src):
                continue
            files = [f for f in os.listdir(cls_src)
                     if os.path.splitext(f)[1].lower() in exts]
            random.seed(seed)
            selected = random.sample(files, min(n, len(files)))
            cls_dst = os.path.join(tmp, cls)
            os.makedirs(cls_dst, exist_ok=True)
            for f in selected:
                shutil.copy2(os.path.join(cls_src, f), os.path.join(cls_dst, f))
            total += len(selected)
            self.log_cb(f"  {cls}: selected {len(selected)}/{len(files)} images")
        self.log_cb(f"Local subsample: {total} images staged for upload.")
        return tmp, tmp

    def _builtin_training_script(self) -> tuple[str, str]:
        """Return local and remote script paths for the selected built-in pipeline."""
        if (
            self.config.task_type == "classification"
            and self.config.model_name == "custom_cnn"
            and not self.config.use_builtin
        ):
            script_name = "custom_cnn_3class.py"
            return os.path.join(os.path.dirname(__file__), script_name), f"/workspace/{script_name}"
        return os.path.join(os.path.dirname(__file__), "train.py"), "/workspace/train.py"

    def _upload_train_script(self) -> None:
        """Upload just the training script (no dataset)."""
        if self.config.custom_script_path and os.path.isfile(self.config.custom_script_path):
            self.log_cb(f"Uploading custom script: {os.path.basename(self.config.custom_script_path)}…")
            self.ssh.upload_file(self.config.custom_script_path, "/workspace/train.py")
        else:
            local_script, remote_script = self._builtin_training_script()
            if not os.path.isfile(local_script):
                raise SSHError(f"Training script not found: {local_script}")
            self.log_cb(f"Uploading {os.path.basename(local_script)}…")
            self.ssh.upload_file(local_script, remote_script)

    def _prepare_remote_classification_data(self) -> None:
        """Validate or create the remote train/val/test classification split."""
        if self.config.pre_split_data:
            self.log_cb("Validating pre-split train/val/test dataset...")
            validate_script = _PRE_SPLIT_VALIDATE_SCRIPT.replace(
                "__THREE_CLASS__", repr(bool(self.config.three_class_classification))
            )
            rc = self.ssh.exec_command(
                f"python3 -c {self._shell_quote(validate_script)}",
                log_cb=self.log_cb,
            )
            if rc != 0:
                raise SSHError("Failed to validate pre-split dataset.")
            return

        self.log_cb(
            "Splitting uploaded class folders into train/val/test "
            f"({self.config.train_split:.2f}/{self.config.val_split:.2f}/{self.config.test_split:.2f})..."
        )
        split_script = (
            _AUTO_SPLIT_CLASSIFICATION_SCRIPT
            .replace("__SEED__", repr(int(self.config.seed)))
            .replace("__THREE_CLASS__", repr(bool(self.config.three_class_classification)))
            .replace("__TRAIN_RATIO__", repr(float(self.config.train_split)))
            .replace("__VAL_RATIO__", repr(float(self.config.val_split)))
            .replace("__TEST_RATIO__", repr(float(self.config.test_split)))
        )
        rc = self.ssh.exec_command(
            f"python3 -c {self._shell_quote(split_script)}",
            log_cb=self.log_cb,
        )
        if rc != 0:
            raise SSHError("Failed to split dataset into train/val/test folders.")
        self.config.pre_split_data = True

    def _cancelled(self) -> bool:
        if self._cancel.is_set():
            self.log_cb("Pipeline cancelled by user.")
            return True
        return False

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _step_estimate(self) -> None:
        """Step 0: Pre-flight dataset analysis and resource estimation."""
        self.log_cb("=" * 60)
        self.log_cb("[0/6] Pre-flight analysis (Smart Estimator)…")
        profile_path = self.config.data_path
        if self.config.task_type == "classification" and self.config.pre_split_data:
            train_path = os.path.join(self.config.data_path, "train")
            if os.path.isdir(train_path):
                profile_path = train_path
                self.log_cb("Pre-split dataset detected — estimating on train split.")
        profile = profile_dataset(
            profile_path,
            task_type=self.config.task_type,
            log_cb=self.log_cb,
        )
        self._dataset_profile = profile
        self._estimate = estimate_resources(
            profile,
            model_name=self.config.model_name,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            gpu_vram_gb=self.config.min_gpu_ram,
            log_cb=self.log_cb,
        )

    def _step_search_and_rent(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[1/6] Searching for GPU instances on Vast.ai…")
        self.vast = VastAPI(self.config.api_key)

        offers = self.vast.search_offers(
            min_gpu_ram=self.config.min_gpu_ram,
            max_price=self.config.max_price,
            log_cb=self.log_cb,
        )
        if not offers:
            raise VastAPIError(
                "No offers found matching the criteria. "
                "Try increasing max price or lowering GPU RAM requirement."
            )

        best = offers[0]
        gpu_name = best.get("gpu_name", "unknown")
        price = best.get("dph_total", "?")
        offer_id = best["id"]
        self.log_cb(f"Best value: {gpu_name} — ${price}/hr (offer #{offer_id})")

        self.log_cb("Renting instance…")
        result = self.vast.create_instance(
            offer_id=offer_id,
            image="pytorch/pytorch:latest",
            disk_gb=30,
        )
        # Extract instance id
        self.instance_id = result.get("new_contract") or result.get("id")
        if self.instance_id is None:
            # Try parsing from raw text
            raw = str(result)
            for token in raw.split():
                if token.isdigit():
                    self.instance_id = int(token)
                    break
        if self.instance_id is None:
            raise VastAPIError(f"Could not determine instance ID from: {result}")

        self.log_cb(f"Instance #{self.instance_id} created. Waiting for it to start…")
        info = self.vast.wait_for_instance_ready(
            self.instance_id,
            timeout_seconds=600,
            callback=self.log_cb,
        )

        self._ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
        self._ssh_port = int(info.get("ssh_port", info.get("ports", {}).get("22/tcp", [{}])[0].get("HostPort", 22)))
        self.log_cb(f"Instance ready at {self._ssh_host}:{self._ssh_port}")

    def _step_connect_ssh(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[2/6] Connecting via SSH…")
        self.ssh = SSHManager(
            host=self._ssh_host,
            port=self._ssh_port,
            key_filename=self.config.ssh_key_path,
        )
        self.ssh.connect(log_cb=self.log_cb)

    def _step_setup_environment(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[3/6] Setting up remote environment…")
        rc_system = self.ssh.exec_command(_SYSTEM_DEPS, log_cb=self.log_cb)
        if rc_system != 0:
            raise SSHError(f"System dependency installation failed (exit code {rc_system})")
        rc = self.ssh.exec_command(_REMOTE_DEPS, log_cb=self.log_cb)
        if rc != 0:
            raise SSHError(f"Dependency installation failed (exit code {rc})")
        self.ssh.exec_command("mkdir -p /workspace/data /workspace/test_data /workspace/output", log_cb=self.log_cb)
        # CUDA compatibility check for newer GPUs
        self.log_cb("Checking CUDA compatibility…")
        rc_cuda = self.ssh.exec_command(
            f"python3 -c {self._shell_quote(_CUDA_FIX_SCRIPT)}",
            log_cb=self.log_cb,
        )
        if rc_cuda != 0:
            raise SSHError("CUDA fix script failed — cannot continue.")

    def _step_upload_data(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[4/6] Uploading dataset…")

        # ── Local subsampling: build a temp folder with only N files/class ──
        upload_path = self.config.data_path
        _tmp_dir = None
        if (
            self.config.max_samples_per_class > 0
            and self.config.task_type == "classification"
            and not self.config.pre_split_data
        ):
            upload_path, _tmp_dir = self._local_subsample(
                self.config.data_path, self.config.max_samples_per_class, self.config.seed
            )

        try:
            # Try rsync first (resume + checksum), then fall back to tar/SFTP
            if not self.ssh.upload_rsync(
                upload_path, "/workspace/data", log_cb=self.log_cb,
            ):
                self.ssh.upload_directory(
                    local_path=upload_path,
                    remote_path="/workspace/data",
                    log_cb=self.log_cb,
                )
        finally:
            if _tmp_dir is not None:
                import shutil as _shutil
                _shutil.rmtree(_tmp_dir, ignore_errors=True)

        self._upload_train_script()

        # Auto-organize flat image folder (classification only)
        if self.config.task_type == "classification":
            self.log_cb("Organizing dataset into class folders…")
            organize_script = r'''
import os, re, shutil
data_dir = "/workspace/data"
subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
if subdirs:
    has_images = any(
        any(f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))
            for f in os.listdir(os.path.join(data_dir, sd)))
        for sd in subdirs
    )
    if has_images:
        print(f"Already organized: {subdirs}")
        exit(0)
files = [f for f in os.listdir(data_dir)
         if os.path.isfile(os.path.join(data_dir, f))
         and f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
if not files:
    print("ERROR: No image files found in data directory!")
    exit(1)
classes = {}
for f in files:
    name = os.path.splitext(f)[0]
    match = re.match(r'^([A-Za-z]+)', name)
    label = match.group(1).upper() if match else "UNKNOWN"
    classes.setdefault(label, []).append(f)
print(f"Detected {len(classes)} classes: {dict((k, len(v)) for k, v in classes.items())}")
for label, flist in classes.items():
    class_dir = os.path.join(data_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    for f in flist:
        shutil.move(os.path.join(data_dir, f), os.path.join(class_dir, f))
print("Dataset organized successfully.")
'''
            if self.config.pre_split_data:
                self.log_cb("Validating pre-split train/val/test dataset…")
                pre_split_script = r'''
import os, re, shutil, sys
root = "/workspace/data"
exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')

def resolve_split(names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return name, path
    print(f"ERROR: Missing split folder. Expected one of: {names}")
    sys.exit(1)

def organize(data_dir):
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if subdirs:
        has_images = any(
            any(f.lower().endswith(exts) for f in os.listdir(os.path.join(data_dir, sd)))
            for sd in subdirs
        )
        if has_images:
            print(f"{os.path.basename(data_dir)} already organized: {subdirs}")
            return
    files = [f for f in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, f)) and f.lower().endswith(exts)]
    if not files:
        print(f"ERROR: No image files found in {data_dir}")
        sys.exit(1)
    classes = {}
    for filename in files:
        name = os.path.splitext(filename)[0]
        match = re.match(r'^([A-Za-z]+)', name)
        label = match.group(1).upper() if match else "UNKNOWN"
        classes.setdefault(label, []).append(filename)
    for label, filenames in classes.items():
        class_dir = os.path.join(data_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        for filename in filenames:
            shutil.move(os.path.join(data_dir, filename), os.path.join(class_dir, filename))
    print(f"{os.path.basename(data_dir)} organized: {dict((k, len(v)) for k, v in classes.items())}")

def class_names(data_dir):
    return sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))

_, train_dir = resolve_split(("train",))
_, val_dir = resolve_split(("val", "validation"))
_, test_dir = resolve_split(("test",))
for split_dir in (train_dir, val_dir, test_dir):
    organize(split_dir)
train_classes = class_names(train_dir)
val_classes = class_names(val_dir)
test_classes = class_names(test_dir)
if not train_classes:
    print("ERROR: No class folders found in train split.")
    sys.exit(1)
if train_classes != val_classes or train_classes != test_classes:
    print(f"ERROR: Class folders differ: train={train_classes}, val={val_classes}, test={test_classes}")
    sys.exit(1)
print(f"Pre-split dataset OK. Classes: {train_classes}")
'''
                rc = self.ssh.exec_command(
                    f"python3 -c {self._shell_quote(pre_split_script)}",
                    log_cb=self.log_cb,
                )
                if rc != 0:
                    raise SSHError("Failed to validate pre-split dataset.")
            else:
                rc = self.ssh.exec_command(
                    f"python3 -c {self._shell_quote(organize_script)}",
                    log_cb=self.log_cb,
                )
                if rc != 0:
                    raise SSHError("Failed to organize dataset into class folders.")

                # Subsample training set to N images per class
                if self.config.max_samples_per_class > 0:
                    self.log_cb(
                        f"Subsampling training data to "
                        f"{self.config.max_samples_per_class} images/class…"
                    )
                    sub_script = _SUBSAMPLE_SCRIPT_TEMPLATE.format(
                        max_n=self.config.max_samples_per_class,
                        seed=self.config.seed,
                    )
                    rc_sub = self.ssh.exec_command(
                        f"python3 -c {self._shell_quote(sub_script)}",
                        log_cb=self.log_cb,
                    )
                    if rc_sub != 0:
                        raise SSHError("Subsampling step failed.")

                # Upload separate test data if provided
                if self.config.test_data_path and os.path.isdir(self.config.test_data_path):
                    self.log_cb("Uploading separate test dataset…")
                    if not self.ssh.upload_rsync(
                        self.config.test_data_path, "/workspace/test_data", log_cb=self.log_cb,
                    ):
                        self.ssh.upload_directory(
                            local_path=self.config.test_data_path,
                            remote_path="/workspace/test_data",
                            log_cb=self.log_cb,
                        )
                    self.log_cb("Organizing test dataset into class folders…")
                    organize_test = organize_script.replace(
                        'data_dir = "/workspace/data"',
                        'data_dir = "/workspace/test_data"',
                    )
                    rc2 = self.ssh.exec_command(
                        f"python3 -c {self._shell_quote(organize_test)}",
                        log_cb=self.log_cb,
                    )
                    if rc2 != 0:
                        raise SSHError("Failed to organize test dataset into class folders.")
        else:
            # Regression — just upload test data if provided (no re-organizing needed)
            if self.config.test_data_path and os.path.isdir(self.config.test_data_path):
                self.log_cb("Uploading separate test dataset…")
                if not self.ssh.upload_rsync(
                    self.config.test_data_path, "/workspace/test_data", log_cb=self.log_cb,
                ):
                    self.ssh.upload_directory(
                        local_path=self.config.test_data_path,
                        remote_path="/workspace/test_data",
                        log_cb=self.log_cb,
                    )

        if self.config.task_type == "classification":
            self._prepare_remote_classification_data()

        # Diagnostic: list final data structure
        self.log_cb("Verifying dataset structure…")
        self.ssh.exec_command(
            "find /workspace/data -maxdepth 2 -type d && "
            "echo '---' && "
            "find /workspace/data -type f | head -10",
            log_cb=self.log_cb,
        )

    @staticmethod
    def _shell_quote(script: str) -> str:
        """Quote a Python script for passing as shell argument."""
        import shlex
        return shlex.quote(script)

    def _step_upload_script_only(self) -> None:
        """Step 4 (test mode): Skip dataset upload, just upload train.py and create output dir."""
        self.log_cb("=" * 60)
        self.log_cb("[4/6] Skipping dataset upload (built-in CIFAR-100 mode)…")
        self.ssh.exec_command("mkdir -p /workspace/output", log_cb=self.log_cb)
        self._upload_train_script()
        self.log_cb("Ready for CIFAR-100 training.")

    def _step_run_training(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[5/6] Starting training…")
        if self.config.custom_script_path:
            # Custom script — run it directly
            cmd = "cd /workspace && python train.py"
            self.log_cb("Running custom script…")
        else:
            cmd = self.config.build_train_command()
        self.log_cb(f"Command: {cmd}")

        # Start telemetry polling in background
        self._telemetry_stop.clear()
        telem_thread = threading.Thread(target=self._poll_telemetry, daemon=True)
        telem_thread.start()

        rc = self.ssh.exec_command(cmd, log_cb=self.log_cb)

        # Stop telemetry polling
        self._telemetry_stop.set()
        telem_thread.join(timeout=5)

        if rc != 0:
            raise SSHError(f"Training script exited with code {rc}")
        self.log_cb("Training completed.")

    def _poll_telemetry(self) -> None:
        """Periodically download telemetry.csv and feed it to the GUI callback."""
        if not self.telemetry_cb:
            return

        import csv
        import io
        import tempfile

        local_tmp = os.path.join(tempfile.gettempdir(), "scout_telemetry.csv")
        remote_csv = "/workspace/output/telemetry.csv"

        while not self._telemetry_stop.is_set():
            self._telemetry_stop.wait(10)  # poll every 10 seconds
            if self._telemetry_stop.is_set():
                break
            try:
                self.ssh._sftp.get(remote_csv, local_tmp)
                with open(local_tmp, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    self.telemetry_cb(rows)
            except Exception:
                pass  # CSV may not exist yet early in training

    def _step_download_results(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[6/6] Downloading results (Harvest)…")

        # Try rsync first, then tar/SFTP fallback
        if not self.ssh.download_rsync(
            "/workspace/output", self.config.output_path, log_cb=self.log_cb,
        ):
            self.ssh.download_directory(
                remote_path="/workspace/output",
                local_path=self.config.output_path,
                log_cb=self.log_cb,
            )

    # ------------------------------------------------------------------
    # Instance destruction
    # ------------------------------------------------------------------
    def destroy_instance(self) -> None:
        """Destroy (delete) the rented instance completely."""
        self._disconnect_ssh()
        if self.vast and self.instance_id:
            self.log_cb(f"Destroying instance #{self.instance_id}…")
            msg = self.vast.destroy_instance(self.instance_id)
            self.log_cb(msg)
            self.instance_id = None
        else:
            self.log_cb("No active instance to destroy.")

    def stop_instance(self) -> None:
        """Stop the instance (pause billing, keep data)."""
        if self.vast and self.instance_id:
            self.log_cb(f"Stopping instance #{self.instance_id}…")
            msg = self.vast.stop_instance(self.instance_id)
            self.log_cb(msg)
        else:
            self.log_cb("No active instance to stop.")

    def _disconnect_ssh(self) -> None:
        if self.ssh:
            self.ssh.disconnect()
            self.ssh = None
