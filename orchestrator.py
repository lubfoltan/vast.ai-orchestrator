"""High-level orchestration logic that ties Vast.ai API and SSH together."""

import logging
import os
import threading
from typing import Callable, Optional

from config import ExperimentConfig
from ssh_manager import SSHManager, SSHError
from vast_api import VastAPI, VastAPIError

logger = logging.getLogger(__name__)

LogCallback = Optional[Callable[[str], None]]

# Dependencies to install inside the container
_REMOTE_DEPS = (
    "pip install --quiet 'numpy<2' torchvision timm scikit-learn matplotlib "
    "grad-cam pillow tqdm pandas openpyxl"
)


class Orchestrator:
    """Coordinates the full lifecycle: rent → setup → train → download → destroy."""

    def __init__(self, config: ExperimentConfig, log_cb: LogCallback = None):
        self.config = config
        self.log_cb = log_cb or (lambda msg: None)
        self.vast: Optional[VastAPI] = None
        self.ssh: Optional[SSHManager] = None
        self.instance_id: Optional[int] = None
        self._cancel = threading.Event()

    # ------------------------------------------------------------------
    # Public entry-point (runs in a background thread)
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full pipeline. Intended to be called from a thread."""
        try:
            self._step_search_and_rent()
            if self._cancelled():
                return
            self._step_connect_ssh()
            if self._cancelled():
                return
            self._step_setup_environment()
            if self._cancelled():
                return
            self._step_upload_data()
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

    def _cancelled(self) -> bool:
        if self._cancel.is_set():
            self.log_cb("Pipeline cancelled by user.")
            return True
        return False

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _step_search_and_rent(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[1/5] Searching for GPU instances on Vast.ai…")
        self.vast = VastAPI(self.config.api_key)

        offers = self.vast.search_offers(
            min_gpu_ram=self.config.min_gpu_ram,
            max_price=self.config.max_price,
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
        self.log_cb(f"Best offer: {gpu_name} — ${price}/hr (offer #{offer_id})")

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
        self.log_cb("[2/5] Connecting via SSH…")
        self.ssh = SSHManager(
            host=self._ssh_host,
            port=self._ssh_port,
            key_filename=self.config.ssh_key_path,
        )
        self.ssh.connect(log_cb=self.log_cb)

    def _step_setup_environment(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[3/5] Setting up remote environment…")
        rc = self.ssh.exec_command(_REMOTE_DEPS, log_cb=self.log_cb)
        if rc != 0:
            raise SSHError(f"Dependency installation failed (exit code {rc})")
        self.ssh.exec_command("mkdir -p /workspace/data /workspace/test_data /workspace/output", log_cb=self.log_cb)

    def _step_upload_data(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[4/5] Uploading dataset…")
        self.ssh.upload_directory(
            local_path=self.config.data_path,
            remote_path="/workspace/data",
            log_cb=self.log_cb,
        )

        # Upload training script
        if self.config.custom_script_path and os.path.isfile(self.config.custom_script_path):
            self.log_cb(f"Uploading custom script: {os.path.basename(self.config.custom_script_path)}…")
            self.ssh.upload_file(self.config.custom_script_path, "/workspace/train.py")
        else:
            train_script = os.path.join(os.path.dirname(__file__), "train.py")
            if os.path.isfile(train_script):
                self.log_cb("Uploading train.py…")
                self.ssh.upload_file(train_script, "/workspace/train.py")

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
            rc = self.ssh.exec_command(
                f"python3 -c {self._shell_quote(organize_script)}",
                log_cb=self.log_cb,
            )
            if rc != 0:
                raise SSHError("Failed to organize dataset into class folders.")

            # Upload separate test data if provided
            if self.config.test_data_path and os.path.isdir(self.config.test_data_path):
                self.log_cb("Uploading separate test dataset…")
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

    def _step_run_training(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("[5/5] Starting training…")
        if self.config.custom_script_path:
            # Custom script — run it directly
            cmd = "cd /workspace && python train.py"
            self.log_cb("Running custom script…")
        else:
            cmd = self.config.build_train_command()
        self.log_cb(f"Command: {cmd}")
        rc = self.ssh.exec_command(cmd, log_cb=self.log_cb)
        if rc != 0:
            raise SSHError(f"Training script exited with code {rc}")
        self.log_cb("Training completed.")

    def _step_download_results(self) -> None:
        self.log_cb("=" * 60)
        self.log_cb("Downloading results…")
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
