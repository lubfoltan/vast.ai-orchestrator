"""SSH connection and file transfer manager using paramiko."""

import logging
import os
import stat
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import paramiko

logger = logging.getLogger(__name__)

LogCallback = Optional[Callable[[str], None]]


class SSHError(Exception):
    """Raised on SSH / SCP failures."""


class SSHManager:
    """Manages SSH connections, command execution, and file transfers."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str = "root",
        key_filename: Optional[str] = None,
        password: Optional[str] = None,
        connect_timeout: int = 30,
        max_retries: int = 3,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.key_filename = key_filename
        self.password = password
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self._client: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self, log_cb: LogCallback = None) -> None:
        """Establish an SSH connection with retries."""
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if log_cb:
                    log_cb(f"SSH connect attempt {attempt}/{self.max_retries} → {self.host}:{self.port}")
                kwargs = {
                    "hostname": self.host,
                    "port": self.port,
                    "username": self.username,
                    "timeout": self.connect_timeout,
                    "allow_agent": False,
                    "look_for_keys": False,
                }
                if self.key_filename:
                    kwargs["key_filename"] = self.key_filename
                if self.password:
                    kwargs["password"] = self.password

                self._client.connect(**kwargs)
                self._sftp = self._client.open_sftp()
                if log_cb:
                    log_cb("SSH connected successfully.")
                return
            except Exception as exc:
                last_err = exc
                if log_cb:
                    log_cb(f"SSH attempt {attempt} failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(5 * attempt)

        raise SSHError(f"Failed to connect after {self.max_retries} attempts: {last_err}")

    def disconnect(self) -> None:
        """Close SSH and SFTP sessions."""
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return transport is not None and transport.is_active()

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            raise SSHError("SSH not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def exec_command(
        self,
        command: str,
        log_cb: LogCallback = None,
        timeout: int = 3600,
    ) -> int:
        """Execute a command and stream stdout/stderr to *log_cb* line-by-line.

        Returns the exit status code.
        """
        self._ensure_connected()
        if log_cb:
            log_cb(f"$ {command}")

        transport = self._client.get_transport()
        channel = transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(command)

        # Stream output
        stdout_buf = ""
        stderr_buf = ""
        while True:
            if channel.recv_ready():
                chunk = channel.recv(4096).decode("utf-8", errors="replace")
                stdout_buf += chunk
                while "\n" in stdout_buf:
                    line, stdout_buf = stdout_buf.split("\n", 1)
                    if log_cb:
                        log_cb(line)
            if channel.recv_stderr_ready():
                chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                stderr_buf += chunk
                while "\n" in stderr_buf:
                    line, stderr_buf = stderr_buf.split("\n", 1)
                    if log_cb:
                        log_cb(f"[stderr] {line}")
            if channel.exit_status_ready():
                # Drain remaining
                while channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    if log_cb and chunk.strip():
                        for l in chunk.strip().split("\n"):
                            log_cb(l)
                while channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                    if log_cb and chunk.strip():
                        for l in chunk.strip().split("\n"):
                            log_cb(f"[stderr] {l}")
                break
            time.sleep(0.1)

        # Flush remaining partial lines
        if stdout_buf.strip() and log_cb:
            log_cb(stdout_buf.strip())
        if stderr_buf.strip() and log_cb:
            log_cb(f"[stderr] {stderr_buf.strip()}")

        return channel.recv_exit_status()

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------
    def _remote_mkdir_p(self, remote_dir: str) -> None:
        """Recursively create remote directories (like mkdir -p)."""
        dirs_to_create = []
        current = remote_dir
        while current and current != "/":
            try:
                self._sftp.stat(current)
                break
            except FileNotFoundError:
                dirs_to_create.append(current)
                current = str(PurePosixPath(current).parent)
        for d in reversed(dirs_to_create):
            self._sftp.mkdir(d)

    def upload_directory(
        self,
        local_path: str,
        remote_path: str,
        log_cb: LogCallback = None,
    ) -> None:
        """Recursively upload a local directory to the remote server."""
        self._ensure_connected()
        local = Path(local_path)
        if not local.is_dir():
            raise SSHError(f"Local path is not a directory: {local_path}")

        # Count files for progress
        all_files = [f for f in local.rglob("*") if f.is_file()]
        total = len(all_files)
        if log_cb:
            log_cb(f"Uploading {total} files from {local_path} → {remote_path}")

        self._remote_mkdir_p(remote_path)

        for idx, file in enumerate(all_files, 1):
            rel = file.relative_to(local)
            remote_file = str(PurePosixPath(remote_path) / rel.as_posix())
            remote_dir = str(PurePosixPath(remote_file).parent)
            self._remote_mkdir_p(remote_dir)

            if log_cb and idx % max(1, total // 20) == 0:
                log_cb(f"  [{idx}/{total}] {rel.as_posix()}")

            self._sftp.put(str(file), remote_file)

        if log_cb:
            log_cb(f"Upload complete: {total} files transferred.")

    def download_directory(
        self,
        remote_path: str,
        local_path: str,
        log_cb: LogCallback = None,
    ) -> None:
        """Recursively download a remote directory to a local path."""
        self._ensure_connected()
        local = Path(local_path)
        local.mkdir(parents=True, exist_ok=True)

        if log_cb:
            log_cb(f"Downloading {remote_path} → {local_path}")

        self._download_recursive(remote_path, local, log_cb)

        if log_cb:
            log_cb("Download complete.")

    def _download_recursive(
        self,
        remote_dir: str,
        local_dir: Path,
        log_cb: LogCallback,
    ) -> None:
        for entry in self._sftp.listdir_attr(remote_dir):
            remote_entry = f"{remote_dir}/{entry.filename}"
            local_entry = local_dir / entry.filename

            if stat.S_ISDIR(entry.st_mode):
                local_entry.mkdir(parents=True, exist_ok=True)
                self._download_recursive(remote_entry, local_entry, log_cb)
            else:
                if log_cb:
                    log_cb(f"  ↓ {remote_entry}")
                self._sftp.get(remote_entry, str(local_entry))

    def upload_file(self, local_file: str, remote_file: str) -> None:
        """Upload a single file."""
        self._ensure_connected()
        remote_dir = str(PurePosixPath(remote_file).parent)
        self._remote_mkdir_p(remote_dir)
        self._sftp.put(local_file, remote_file)
