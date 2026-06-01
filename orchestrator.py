"""High-level orchestration logic that ties Vast.ai API and SSH together."""

import logging
import os
import threading
import time
from pathlib import Path
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

root = "/workspace/data"
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

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

def count_by_class(data_dir):
    counts = {}
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        count = sum(1 for f in os.listdir(cls_path) if f.lower().endswith(exts))
        if count > 0:
            counts[cls] = count
    return counts

split_dirs = {
    "train": resolve_split(("train",))[1],
    "val": resolve_split(("val", "validation"))[1],
    "test": resolve_split(("test",))[1],
}
for split_dir in split_dirs.values():
    organize(split_dir)

counts = {name: count_by_class(path) for name, path in split_dirs.items()}
classes = {name: sorted(values) for name, values in counts.items()}
if not classes["train"]:
    print("ERROR: No class folders found in train split.")
    sys.exit(1)
if classes["train"] != classes["val"] or classes["train"] != classes["test"]:
    print(f"ERROR: Class folders differ: {classes}")
    sys.exit(1)

total = sum(sum(values.values()) for values in counts.values())
for split_name in ("train", "val", "test"):
    split_total = sum(counts[split_name].values())
    pct = (split_total / total * 100.0) if total else 0.0
    print(f"{split_name}: {split_total}/{total} images ({pct:.2f}%) per class {counts[split_name]}")
print(f"Pre-split dataset OK. Classes: {classes['train']}")
'''

_AUTO_SPLIT_CLASSIFICATION_SCRIPT = r'''
import math, os, random, re, shutil, sys

root = "/workspace/data"
tmp_root = "/workspace/data_split_tmp"
seed = __SEED__
ratios = {
    "train": __TRAIN_RATIO__,
    "val": __VAL_RATIO__,
    "test": __TEST_RATIO__,
}
split_names = ("train", "val", "test")
split_aliases = {"train", "val", "validation", "test"}
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def image_files(path):
    return sorted(
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(exts)
    )

def organize_flat_files():
    files = image_files(root)
    if not files:
        return
    classes = {}
    for filename in files:
        name = os.path.splitext(filename)[0]
        match = re.match(r'^([A-Za-z]+)', name)
        label = match.group(1).upper() if match else "UNKNOWN"
        classes.setdefault(label, []).append(filename)
    print(f"Detected flat files; organizing into class folders: {dict((k, len(v)) for k, v in classes.items())}")
    for label, filenames in classes.items():
        class_dir = os.path.join(root, label)
        os.makedirs(class_dir, exist_ok=True)
        for filename in filenames:
            shutil.move(os.path.join(root, filename), os.path.join(class_dir, filename))

def split_counts(total):
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        print("ERROR: Split ratios must sum to a positive value.")
        sys.exit(1)
    normalized = [ratios[name] / ratio_sum for name in split_names]
    positive = [i for i, ratio in enumerate(normalized) if ratio > 0]
    if total < len(positive):
        print(f"ERROR: Class has only {total} images, but {len(positive)} non-empty splits are requested.")
        sys.exit(1)
    raw = [total * ratio for ratio in normalized]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i], normalized[i]), reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    if total >= len(positive):
        for i in positive:
            if counts[i] == 0:
                donors = [j for j in positive if counts[j] > 1]
                if not donors:
                    print("ERROR: Cannot keep all requested splits non-empty.")
                    sys.exit(1)
                donor = max(donors, key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] += 1
    return dict(zip(split_names, counts))

def split_totals(class_split_counts):
    return {
        split_name: sum(counts[split_name] for counts in class_split_counts.values())
        for split_name in split_names
    }

def min_count_for(class_size, split_name):
    positive_split_count = sum(1 for name in split_names if ratios[name] > 0)
    if ratios[split_name] > 0 and class_size >= positive_split_count:
        return 1
    return 0

def rebalance_to_global_targets(class_split_counts, class_sizes, target_totals):
    current = split_totals(class_split_counts)
    guard = 0
    while current != target_totals:
        guard += 1
        if guard > 100000:
            print("ERROR: Split rebalancing did not converge.")
            sys.exit(1)
        moved = False
        surplus_splits = [name for name in split_names if current[name] > target_totals[name]]
        deficit_splits = [name for name in split_names if current[name] < target_totals[name]]
        if not surplus_splits or not deficit_splits:
            break
        for src_split in surplus_splits:
            while current[src_split] > target_totals[src_split]:
                deficit_splits = [name for name in split_names if current[name] < target_totals[name]]
                if not deficit_splits:
                    break
                dst_split = max(deficit_splits, key=lambda name: target_totals[name] - current[name])
                donors = [
                    cls for cls, counts in class_split_counts.items()
                    if counts[src_split] > min_count_for(class_sizes[cls], src_split)
                ]
                if not donors:
                    break
                cls = max(
                    donors,
                    key=lambda name: class_split_counts[name][src_split] - min_count_for(class_sizes[name], src_split),
                )
                class_split_counts[cls][src_split] -= 1
                class_split_counts[cls][dst_split] += 1
                current[src_split] -= 1
                current[dst_split] += 1
                moved = True
        if not moved:
            print(f"ERROR: Cannot rebalance split counts from {current} to requested {target_totals}.")
            print("Try adding more images per class or using less extreme split ratios.")
            sys.exit(1)

def validate_split_dirs(expected_totals=None):
    counts = {}
    classes = None
    total = 0
    for split_name in split_names:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            print(f"ERROR: Missing generated split folder: {split_dir}")
            sys.exit(1)
        split_counts_by_class = {}
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            count = len(image_files(cls_path))
            if count > 0:
                split_counts_by_class[cls] = count
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
    ratio_sum = sum(ratios.values())
    for split_name in split_names:
        split_total = sum(counts[split_name].values())
        if expected_totals and split_total != expected_totals[split_name]:
            print(f"ERROR: {split_name} has {split_total} images, expected {expected_totals[split_name]}.")
            sys.exit(1)
        actual_pct = split_total / total * 100.0
        requested_pct = ratios[split_name] / ratio_sum * 100.0
        print(f"{split_name}: {split_total}/{total} images ({actual_pct:.2f}%, requested {requested_pct:.2f}%) per class {counts[split_name]}")
    return counts

existing_splits = [name for name in split_names if os.path.isdir(os.path.join(root, name))]
if existing_splits:
    print("Dataset already contains split folders on remote; validating existing split.")
    validate_split_dirs()
    sys.exit(0)

organize_flat_files()
class_dirs = [
    d for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d)) and d not in split_aliases
]
if not class_dirs:
    print("ERROR: No class folders found. Expected folders like NORMAL/ and PNEUMONIA/.")
    sys.exit(1)

class_files = {}
for cls in class_dirs:
    files = image_files(os.path.join(root, cls))
    if not files:
        print(f"ERROR: Class folder {cls} has no supported image files.")
        sys.exit(1)
    class_files[cls] = files

total_images = sum(len(files) for files in class_files.values())
target_totals = split_counts(total_images)
for split_name in split_names:
    if ratios[split_name] > 0 and target_totals[split_name] < len(class_dirs):
        print(
            f"ERROR: Requested {split_name} ratio gives only {target_totals[split_name]} images, "
            f"but {len(class_dirs)} classes must be represented."
        )
        print("Add more images or increase this split ratio.")
        sys.exit(1)

class_sizes = {cls: len(files) for cls, files in class_files.items()}
class_split_counts = {cls: split_counts(len(files)) for cls, files in class_files.items()}
rebalance_to_global_targets(class_split_counts, class_sizes, target_totals)

shutil.rmtree(tmp_root, ignore_errors=True)
for split_name in split_names:
    os.makedirs(os.path.join(tmp_root, split_name), exist_ok=True)

for cls in class_dirs:
    src_dir = os.path.join(root, cls)
    files = class_files[cls]
    rng = random.Random(f"{seed}:{cls}")
    rng.shuffle(files)
    counts = class_split_counts[cls]
    print(f"{cls}: splitting {len(files)} images -> {counts}")
    offset = 0
    for split_name in split_names:
        count = counts[split_name]
        dest_dir = os.path.join(tmp_root, split_name, cls)
        os.makedirs(dest_dir, exist_ok=True)
        for filename in files[offset:offset + count]:
            shutil.move(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))
        offset += count

for entry in os.listdir(root):
    path = os.path.join(root, entry)
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
for split_name in split_names:
    shutil.move(os.path.join(tmp_root, split_name), os.path.join(root, split_name))
shutil.rmtree(tmp_root, ignore_errors=True)

validate_split_dirs(target_totals)
print("Auto split dataset OK.")
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

    def _upload_train_script(self) -> None:
        """Upload just the training script (no dataset)."""
        if self.config.custom_script_path and os.path.isfile(self.config.custom_script_path):
            self.log_cb(f"Uploading custom script: {os.path.basename(self.config.custom_script_path)}…")
            self.ssh.upload_file(self.config.custom_script_path, "/workspace/train.py")
        else:
            train_script = os.path.join(os.path.dirname(__file__), "train.py")
            if os.path.isfile(train_script):
                self.log_cb("Re-uploading train.py…")
                self.ssh.upload_file(train_script, "/workspace/train.py")

    def _prepare_remote_classification_data(self) -> None:
        """Validate or create the remote train/val/test folder split."""
        if self.config.pre_split_data:
            self.log_cb("Validating pre-split train/val/test dataset…")
            rc = self.ssh.exec_command(
                f"python3 -c {self._shell_quote(_PRE_SPLIT_VALIDATE_SCRIPT)}",
                log_cb=self.log_cb,
            )
            if rc != 0:
                raise SSHError("Failed to validate pre-split dataset.")
            return

        self.log_cb(
            "Splitting uploaded class folders into train/val/test "
            f"({self.config.train_split:.2f}/{self.config.val_split:.2f}/{self.config.test_split:.2f})…"
        )
        split_script = (
            _AUTO_SPLIT_CLASSIFICATION_SCRIPT
            .replace("__SEED__", repr(int(self.config.seed)))
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
        # Re-install grad-cam AFTER the CUDA fix so it links against the
        # correct torch build (the fix may have replaced torch/torchvision).
        rc_gc = self.ssh.exec_command(
            "pip install --quiet --no-deps --force-reinstall grad-cam",
            log_cb=self.log_cb,
        )
        if rc_gc != 0:
            self.log_cb("WARNING: grad-cam reinstall failed — Grad-CAM may not work.")

    def _step_upload_data(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[4/6] Uploading dataset…")
        self.ssh.exec_command(
            "rm -rf /workspace/data /workspace/test_data && "
            "mkdir -p /workspace/data /workspace/test_data /workspace/output",
            log_cb=self.log_cb,
        )

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

        # Upload training script
        if self.config.custom_script_path and os.path.isfile(self.config.custom_script_path):
            self.log_cb(f"Uploading custom script: {os.path.basename(self.config.custom_script_path)}…")
            self.ssh.upload_file(self.config.custom_script_path, "/workspace/train.py")
        else:
            train_script = os.path.join(os.path.dirname(__file__), "train.py")
            if os.path.isfile(train_script):
                self.log_cb("Uploading train.py…")
                self.ssh.upload_file(train_script, "/workspace/train.py")

        if self.config.task_type == "classification":
            self._prepare_remote_classification_data()
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
        if self.config.custom_script_path and os.path.isfile(self.config.custom_script_path):
            self.log_cb(f"Uploading custom script: {os.path.basename(self.config.custom_script_path)}…")
            self.ssh.upload_file(self.config.custom_script_path, "/workspace/train.py")
        else:
            train_script = os.path.join(os.path.dirname(__file__), "train.py")
            if os.path.isfile(train_script):
                self.log_cb("Uploading train.py…")
                self.ssh.upload_file(train_script, "/workspace/train.py")
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
        self.log_cb(f"Local output path: {self.config.output_path}")

        remote_files = self._remote_output_files()
        if remote_files:
            png_count = sum(1 for item in remote_files if item.lower().endswith(".png"))
            gradcam_count = sum(1 for item in remote_files if "gradcam" in item.lower() and item.lower().endswith(".png"))
            self.log_cb(
                f"Remote output contains {len(remote_files)} files "
                f"({png_count} PNG, {gradcam_count} Grad-CAM)."
            )
        else:
            self.log_cb("WARNING: /workspace/output is empty or could not be scanned before download.")

        # Try rsync first, then tar/SFTP fallback
        if not self.ssh.download_rsync(
            "/workspace/output", self.config.output_path, log_cb=self.log_cb,
        ):
            self.ssh.download_directory(
                remote_path="/workspace/output",
                local_path=self.config.output_path,
                log_cb=self.log_cb,
            )
        self._log_local_output_summary()

    def download_results(self) -> None:
        """Public wrapper used by the GUI to download current remote results."""
        if not self.ssh or not self.ssh.is_connected:
            raise SSHError("No active SSH connection for result download.")
        self._step_download_results()

    def _remote_output_files(self) -> list[str]:
        """Return relative file paths currently present in /workspace/output."""
        try:
            tree = self.ssh.get_remote_file_structure("/workspace/output", max_depth=3)
        except Exception as exc:
            self.log_cb(f"WARNING: remote output scan failed: {exc}")
            return []

        files: list[str] = []

        def collect(nodes: list) -> None:
            for node in nodes:
                if node.get("is_dir"):
                    collect(node.get("children", []))
                else:
                    files.append(str(node.get("path", "")).replace("/workspace/output/", "", 1))

        collect(tree)
        return sorted(item for item in files if item)

    def _log_local_output_summary(self) -> None:
        local = Path(self.config.output_path)
        if not local.exists():
            self.log_cb(f"WARNING: local output path was not created: {local}")
            return

        files = [p for p in local.rglob("*") if p.is_file()]
        pngs = [p for p in files if p.suffix.lower() == ".png"]
        gradcams = [p for p in pngs if "gradcam" in p.name.lower()]
        self.log_cb(
            f"Local output now has {len(files)} files "
            f"({len(pngs)} PNG, {len(gradcams)} Grad-CAM)."
        )
        if pngs:
            preview = ", ".join(p.name for p in sorted(pngs)[:8])
            self.log_cb(f"Downloaded PNGs: {preview}")

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
