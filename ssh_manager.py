"""SSH connection and file transfer manager using paramiko."""

import logging
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import shlex

import paramiko

logger = logging.getLogger(__name__)

# Tuned transport defaults — paramiko's stock 32 KB window cripples throughput
# on high-latency links. 128 MB window + 32 KB packets ≈ saturates a 1 Gbps link.
_SSH_WINDOW_SIZE = 128 * 1024 * 1024  # 128 MB
_SSH_MAX_PACKET = 32 * 1024  # 32 KB (paramiko hard cap)

LogCallback = Optional[Callable[[str], None]]


def _flatten(tree: list) -> list:
    """Flatten a nested file-tree list into a single list of entries."""
    for node in tree:
        yield node
        if node.get("children"):
            yield from _flatten(node["children"])


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
        banner_timeout: int = 60,
        max_retries: int = 12,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.key_filename = key_filename
        self.password = password
        self.connect_timeout = connect_timeout
        self.banner_timeout = banner_timeout
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
                    "banner_timeout": self.banner_timeout,
                    "allow_agent": False,
                    "look_for_keys": False,
                }
                if self.key_filename:
                    kwargs["key_filename"] = self.key_filename
                if self.password:
                    kwargs["password"] = self.password

                self._client.connect(**kwargs)
                # Tune transport window/packet sizes BEFORE opening any channels —
                # paramiko's default 32 KB window is the main reason SFTP feels slow.
                transport = self._client.get_transport()
                if transport is not None:
                    try:
                        transport.default_window_size = _SSH_WINDOW_SIZE
                        transport.default_max_packet_size = _SSH_MAX_PACKET
                        # Disable rekey churn on long large transfers
                        transport.packetizer.REKEY_BYTES = pow(2, 40)
                        transport.packetizer.REKEY_PACKETS = pow(2, 40)
                    except Exception:
                        pass
                self._sftp = self._client.open_sftp()
                if log_cb:
                    log_cb("SSH connected successfully.")
                return
            except Exception as exc:
                last_err = exc
                if log_cb:
                    log_cb(f"SSH attempt {attempt} failed: {exc}")
                if attempt < self.max_retries:
                    wait = min(15 * attempt, 90)  # 15s, 30s, 45s … capped at 90s
                    if log_cb:
                        log_cb(f"Retrying in {wait}s (instance may still be booting)…")
                    time.sleep(wait)

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
        """Recursively upload a local directory to the remote server.

        Strategy (fastest → slowest fallback):
          1. **Streaming tar over SSH exec** — pipes an uncompressed tar stream
             directly into ``tar -xf -`` on the remote. No intermediate archive
             file, no gzip cost (images don't compress), no SFTP window stalls.
             Typically 5–15× faster than SFTP for image datasets.
          2. **tar.gz staged via SFTP** — old path, kept as a safety net if
             streaming exec fails (e.g. remote tar missing, channel closed).
          3. **Per-file SFTP** — last resort.
        """
        self._ensure_connected()
        local = Path(local_path)
        if not local.is_dir():
            raise SSHError(f"Local path is not a directory: {local_path}")

        all_files = [f for f in local.rglob("*") if f.is_file()]
        total = len(all_files)
        if total == 0:
            if log_cb:
                log_cb("No files to upload.")
            return

        total_bytes = sum(f.stat().st_size for f in all_files)
        total_mb = total_bytes / (1024 * 1024)
        if log_cb:
            log_cb(f"Uploading {total} files ({total_mb:.1f} MB) → {remote_path}")
            log_cb("Using streaming tar (uncompressed, direct pipe)…")

        # ---- Primary: streaming tar over exec ---------------------------------
        try:
            self._remote_mkdir_p(remote_path)
            self._upload_directory_streaming_tar(
                local, remote_path, all_files, total_bytes, log_cb
            )
            if log_cb:
                log_cb(f"Upload complete: {total} files transferred.")
            return
        except Exception as exc:
            if log_cb:
                log_cb(f"Streaming tar failed ({exc}), falling back to staged tar.gz…")

        # ---- Fallback 1: staged tar.gz via SFTP -------------------------------
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name
            with tarfile.open(tmp_path, "w:gz") as tar:
                for file in all_files:
                    arcname = file.relative_to(local).as_posix()
                    tar.add(str(file), arcname=arcname)

            archive_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            if log_cb:
                log_cb(f"Archive: {archive_size_mb:.1f} MB ({total} files compressed)")
            remote_archive = f"{remote_path}/_upload.tar.gz"
            if log_cb:
                log_cb(f"Uploading archive → {remote_path}…")
            self._sftp.put(
                tmp_path,
                remote_archive,
                callback=self._make_progress_cb(log_cb, archive_size_mb),
            )
            if log_cb:
                log_cb("Extracting on remote server…")
            rc = self.exec_command(
                f"cd {remote_path} && tar xzf _upload.tar.gz && rm _upload.tar.gz",
                log_cb=log_cb,
            )
            if rc != 0:
                raise SSHError("Remote tar extraction failed")
            if log_cb:
                log_cb(f"Upload complete: {total} files transferred.")
            return
        except Exception as exc:
            if log_cb:
                log_cb(f"Staged tar.gz failed ({exc}), falling back to per-file SFTP…")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # ---- Fallback 2: per-file SFTP ----------------------------------------
        self._upload_directory_sftp(local_path, remote_path, log_cb)

    def _upload_directory_streaming_tar(
        self,
        local: Path,
        remote_path: str,
        all_files: list,
        total_bytes: int,
        log_cb: LogCallback,
    ) -> None:
        """Pipe an uncompressed tar stream into ``tar -xf -`` on the remote.

        This bypasses SFTP entirely for the data path. The SSH exec channel
        carries the tar bytes; the remote ``tar`` command writes the files
        as the stream arrives. Memory use is O(1).
        """
        self._ensure_connected()
        transport = self._client.get_transport()
        channel = transport.open_session()
        # Larger timeout — big uploads can take a while
        channel.settimeout(None)

        quoted_remote = shlex.quote(remote_path)
        # -m: don't restore mtimes (faster, avoids clock-skew warnings)
        remote_cmd = f"tar -xmf - -C {quoted_remote}"
        channel.exec_command(remote_cmd)

        bytes_sent = [0]
        last_pct = [-1]
        last_log_time = [time.monotonic()]
        start_time = time.monotonic()

        class _ProgressStream:
            """File-like wrapper that forwards writes to the SSH channel and
            reports progress to ``log_cb``. tarfile only needs write/flush."""

            def __init__(self, ch):
                self._ch = ch

            def write(self, data):
                # paramiko channel.sendall blocks until all bytes are queued
                self._ch.sendall(data)
                bytes_sent[0] += len(data)
                if log_cb and total_bytes:
                    pct = int(bytes_sent[0] / total_bytes * 100)
                    now = time.monotonic()
                    # Throttle: log on 10% step OR every 5s, whichever first
                    if pct // 10 > last_pct[0] // 10 or (now - last_log_time[0]) >= 5.0:
                        last_pct[0] = pct
                        last_log_time[0] = now
                        elapsed = now - start_time
                        mb_done = bytes_sent[0] / (1024 * 1024)
                        speed = (mb_done / elapsed) if elapsed > 0 else 0.0
                        log_cb(
                            f"  ↑ {mb_done:.1f} / {total_bytes / (1024 * 1024):.1f} MB "
                            f"({pct}%, {speed:.1f} MB/s)"
                        )
                return len(data)

            def flush(self):
                pass

        stream = _ProgressStream(channel)
        try:
            # mode "w|" = streaming tar, uncompressed, no seek required
            with tarfile.open(fileobj=stream, mode="w|") as tar:
                for f in all_files:
                    arcname = f.relative_to(local).as_posix()
                    # recursive=False — we already enumerated every file
                    tar.add(str(f), arcname=arcname, recursive=False)
        finally:
            # Signal EOF to remote tar so it can finish and exit
            try:
                channel.shutdown_write()
            except Exception:
                pass

        # Wait for remote tar to finish
        exit_code = channel.recv_exit_status()
        # Drain stderr for diagnostics on failure
        err = b""
        try:
            while channel.recv_stderr_ready():
                err += channel.recv_stderr(4096)
        except Exception:
            pass
        try:
            channel.close()
        except Exception:
            pass

        if exit_code != 0:
            err_text = err.decode("utf-8", errors="replace").strip()
            raise SSHError(
                f"Remote tar extract failed (exit {exit_code}): {err_text or 'no stderr'}"
            )

        if log_cb:
            elapsed = time.monotonic() - start_time
            mb = total_bytes / (1024 * 1024)
            speed = mb / elapsed if elapsed > 0 else 0.0
            log_cb(f"Streamed {mb:.1f} MB in {elapsed:.1f}s ({speed:.1f} MB/s)")

    @staticmethod
    def _make_progress_cb(log_cb: LogCallback, total_mb: float):
        """Return a paramiko progress callback that logs every ~10%."""
        if not log_cb or total_mb < 1:
            return None
        total_bytes = int(total_mb * 1024 * 1024)
        last_pct = [-1]  # mutable for closure

        def _progress(transferred: int, total: int) -> None:
            pct = int(transferred / total * 100) if total else 0
            # Log at every 10% step
            if pct // 10 > last_pct[0] // 10:
                last_pct[0] = pct
                mb_done = transferred / (1024 * 1024)
                log_cb(f"  ↑ {mb_done:.1f} / {total_mb:.1f} MB ({pct}%)")
        return _progress

    def _upload_directory_sftp(
        self,
        local_path: str,
        remote_path: str,
        log_cb: LogCallback = None,
    ) -> None:
        """Fallback: recursively upload a local directory file-by-file via SFTP."""
        local = Path(local_path)
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
        """Recursively download a remote directory to a local path.

        Strategy:
          1. **Streaming tar from exec stdout** — runs ``tar -cf - .`` on
             remote and pipes stdout straight into a local extractor. No
             intermediate file, no remote disk pressure.
          2. **Staged tar.gz via SFTP** — old path, kept as fallback.
          3. **Per-file SFTP** — last resort.
        """
        self._ensure_connected()
        local = Path(local_path)
        local.mkdir(parents=True, exist_ok=True)

        if log_cb:
            log_cb(f"Downloading {remote_path} → {local_path}")

        # ---- Primary: streaming tar from exec ---------------------------------
        try:
            self._download_directory_streaming_tar(remote_path, local, log_cb)
            if log_cb:
                log_cb("Download complete.")
            return
        except Exception as exc:
            if log_cb:
                log_cb(f"Streaming download failed ({exc}), falling back to staged tar.gz…")

        # ---- Fallback 1: staged tar.gz ----------------------------------------
        try:
            remote_archive = f"{remote_path}/_download.tar.gz"
            if log_cb:
                log_cb("Packing results on remote server…")
            rc = self.exec_command(
                f"cd {remote_path} && tar czf _download.tar.gz --exclude=_download.tar.gz .",
                log_cb=log_cb,
            )
            if rc != 0:
                raise SSHError("Remote tar packing failed")

            tmp_path = os.path.join(tempfile.gettempdir(), "_download.tar.gz")
            if log_cb:
                log_cb("Downloading archive…")
            self._sftp.get(remote_archive, tmp_path)
            self.exec_command(f"rm -f {remote_archive}")

            archive_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            if log_cb:
                log_cb(f"Extracting archive ({archive_size_mb:.1f} MB)…")
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=str(local), filter="data")
            os.unlink(tmp_path)
            if log_cb:
                log_cb("Download complete.")
            return
        except Exception as exc:
            if log_cb:
                log_cb(f"Staged download failed ({exc}), falling back to per-file SFTP…")

        # ---- Fallback 2: per-file SFTP ----------------------------------------
        self._download_recursive(remote_path, local, log_cb)
        if log_cb:
            log_cb("Download complete.")

    def _download_directory_streaming_tar(
        self,
        remote_path: str,
        local: Path,
        log_cb: LogCallback,
    ) -> None:
        """Run ``tar -cf - .`` on remote and extract its stdout locally."""
        self._ensure_connected()
        transport = self._client.get_transport()
        channel = transport.open_session()
        channel.settimeout(None)

        quoted_remote = shlex.quote(remote_path)
        # Uncompressed stream — model artifacts are mostly already-binary;
        # for text/CSV the savings don't justify CPU on both ends.
        remote_cmd = f"cd {quoted_remote} && tar -cf - ."
        channel.exec_command(remote_cmd)

        bytes_recv = [0]
        last_log_time = [time.monotonic()]
        start_time = time.monotonic()

        class _ReadStream:
            """File-like wrapper around channel.recv for tarfile streaming."""

            def __init__(self, ch):
                self._ch = ch
                self._buf = b""

            def read(self, n=-1):
                if n is None or n < 0:
                    # Read everything (tarfile shouldn't request this in stream mode)
                    chunks = [self._buf]
                    self._buf = b""
                    while True:
                        data = self._ch.recv(65536)
                        if not data:
                            break
                        chunks.append(data)
                        bytes_recv[0] += len(data)
                    return b"".join(chunks)
                # Buffered read of exactly n bytes (or less at EOF)
                while len(self._buf) < n:
                    data = self._ch.recv(max(65536, n - len(self._buf)))
                    if not data:
                        break
                    self._buf += data
                    bytes_recv[0] += len(data)
                    if log_cb:
                        now = time.monotonic()
                        if (now - last_log_time[0]) >= 5.0:
                            last_log_time[0] = now
                            elapsed = now - start_time
                            mb = bytes_recv[0] / (1024 * 1024)
                            speed = mb / elapsed if elapsed > 0 else 0.0
                            log_cb(f"  ↓ {mb:.1f} MB ({speed:.1f} MB/s)")
                out, self._buf = self._buf[:n], self._buf[n:]
                return out

        stream = _ReadStream(channel)
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            tar.extractall(path=str(local), filter="data")

        exit_code = channel.recv_exit_status()
        err = b""
        try:
            while channel.recv_stderr_ready():
                err += channel.recv_stderr(4096)
        except Exception:
            pass
        try:
            channel.close()
        except Exception:
            pass

        if exit_code != 0:
            err_text = err.decode("utf-8", errors="replace").strip()
            raise SSHError(
                f"Remote tar pack failed (exit {exit_code}): {err_text or 'no stderr'}"
            )

        if log_cb:
            elapsed = time.monotonic() - start_time
            mb = bytes_recv[0] / (1024 * 1024)
            speed = mb / elapsed if elapsed > 0 else 0.0
            log_cb(f"Streamed {mb:.1f} MB in {elapsed:.1f}s ({speed:.1f} MB/s)")

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
        local_path = Path(local_file)
        if not local_path.is_file():
            raise SSHError(f"Local file is not a file: {local_file}")

        remote_dir = str(PurePosixPath(remote_file).parent)
        rc = self.exec_command(f"mkdir -p {shlex.quote(remote_dir)}", log_cb=None, timeout=30)
        if rc != 0:
            raise SSHError(f"Failed to create remote directory: {remote_dir}")

        try:
            self._upload_file_streaming(local_path, remote_file)
        except Exception as exc:
            logger.debug("Streaming file upload failed, falling back to SFTP: %s", exc)
            self._remote_mkdir_p(remote_dir)
            self._sftp.put(str(local_path), remote_file)

    def _upload_file_streaming(self, local_path: Path, remote_file: str) -> None:
        """Upload a small/medium file over a fresh exec channel."""
        transport = self._client.get_transport()
        channel = transport.open_session()
        channel.settimeout(None)
        channel.exec_command(f"cat > {shlex.quote(remote_file)}")

        try:
            with open(local_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    channel.sendall(chunk)
        finally:
            try:
                channel.shutdown_write()
            except Exception:
                pass

        exit_code = channel.recv_exit_status()
        err = b""
        try:
            while channel.recv_stderr_ready():
                err += channel.recv_stderr(4096)
        except Exception:
            pass
        try:
            channel.close()
        except Exception:
            pass

        if exit_code != 0:
            err_text = err.decode("utf-8", errors="replace").strip()
            raise SSHError(
                f"Remote file write failed (exit {exit_code}): {err_text or 'no stderr'}"
            )

    def remote_path_exists(self, remote_path: str, timeout: int = 15) -> bool:
        """Return True if a remote path exists.

        This intentionally uses a short-lived SSH exec channel instead of the
        long-lived SFTP client. In attached mode the SFTP channel can occasionally
        hang after large installs or transfers, while opening a fresh exec channel
        for ``test -e`` stays responsive.
        """
        self._ensure_connected()
        channel = None
        try:
            transport = self._client.get_transport()
            channel = transport.open_session()
            channel.settimeout(1)
            channel.exec_command(f"test -e {shlex.quote(remote_path)}")
            deadline = time.monotonic() + timeout
            while not channel.exit_status_ready():
                if time.monotonic() >= deadline:
                    channel.close()
                    return False
                time.sleep(0.05)
            return channel.recv_exit_status() == 0
        except Exception as exc:
            logger.debug("Remote path check failed for %s: %s", remote_path, exc)
            return False
        finally:
            if channel is not None:
                try:
                    channel.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Remote inspection helpers
    # ------------------------------------------------------------------
    def verify_remote_data(self, log_cb: LogCallback = None) -> dict:
        """Check which workspace folders exist remotely.

        Returns a dict like ``{"data": True, "output": True, "train.py": True}``.
        """
        self._ensure_connected()
        result = {}
        for name, path in [
            ("data", "/workspace/data"),
            ("output", "/workspace/output"),
            ("train.py", "/workspace/train.py"),
        ]:
            exists = self.remote_path_exists(path)
            result[name] = exists
            if log_cb:
                mark = "✓" if exists else "✗"
                state = "exists" if exists else "not found"
                log_cb(f"  {mark} {path} {state}")
        return result

    def get_remote_file_structure(
        self,
        remote_path: str = "/workspace",
        max_depth: int = 3,
        log_cb: LogCallback = None,
    ) -> list:
        """Return a list of dicts representing the remote file tree.

        Each entry: ``{"path": "/workspace/data", "name": "data",
                       "is_dir": True, "size": 0, "children": [...]}``.
        """
        self._ensure_connected()

        def _walk(path: str, depth: int) -> list:
            if depth > max_depth:
                return []
            entries = []
            try:
                items = self._sftp.listdir_attr(path)
            except PermissionError:
                return []
            except Exception:
                return []
            for item in sorted(items, key=lambda a: a.filename):
                full = f"{path}/{item.filename}"
                is_dir = stat.S_ISDIR(item.st_mode)
                node = {
                    "path": full,
                    "name": item.filename,
                    "is_dir": is_dir,
                    "size": item.st_size if not is_dir else 0,
                    "children": [],
                }
                if is_dir:
                    node["children"] = _walk(full, depth + 1)
                entries.append(node)
            return entries

        if log_cb:
            log_cb(f"Scanning remote: {remote_path} (depth={max_depth})…")
        tree = _walk(remote_path, 1)
        if log_cb:
            log_cb(f"Found {sum(1 for _ in _flatten(tree))} items.")
        return tree

    def delete_remote_path(self, remote_path: str, log_cb: LogCallback = None) -> None:
        """Recursively delete a remote file or directory."""
        self._ensure_connected()
        if log_cb:
            log_cb(f"Deleting {remote_path}…")
        rc = self.exec_command(f"rm -rf {remote_path}", log_cb=log_cb)
        if rc != 0:
            raise SSHError(f"Failed to delete {remote_path}")

    # ------------------------------------------------------------------
    # Rsync-based transfer (resume + checksum)
    # ------------------------------------------------------------------
    @staticmethod
    def _has_rsync() -> bool:
        """Check if rsync is available locally (never on Windows — Git rsync mis-handles paths)."""
        if platform.system() == "Windows":
            return False
        return shutil.which("rsync") is not None

    def upload_rsync(
        self,
        local_path: str,
        remote_path: str,
        log_cb: LogCallback = None,
    ) -> bool:
        """Upload via rsync over SSH. Returns True on success.

        Rsync provides:
        - Resume on interruption (--partial)
        - Checksum verification (--checksum)
        - Compression during transfer (-z)
        - Delta transfer (only changed bytes for re-uploads)
        """
        if not self._has_rsync():
            if log_cb:
                log_cb("rsync not found locally — falling back to tar upload.")
            return False

        ssh_cmd = f"ssh -p {self.port} -o StrictHostKeyChecking=no"
        if self.key_filename:
            ssh_cmd += f" -i {self.key_filename}"

        # Ensure trailing slash to sync contents (not the dir itself into a subdir)
        src = local_path.rstrip("/\\") + "/"
        dst = f"{self.username}@{self.host}:{remote_path}/"

        cmd = [
            "rsync", "-avz",
            "--partial", "--progress", "--checksum",
            "-e", ssh_cmd,
            src, dst,
        ]

        if log_cb:
            log_cb(f"Rsync: {src} → {dst}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line and log_cb:
                    log_cb(f"  [rsync] {line}")
            rc = proc.wait()
            if rc != 0:
                if log_cb:
                    log_cb(f"rsync exited with code {rc}")
                return False
            if log_cb:
                log_cb("Rsync upload complete.")
            return True
        except Exception as exc:
            if log_cb:
                log_cb(f"rsync error: {exc}")
            return False

    def download_rsync(
        self,
        remote_path: str,
        local_path: str,
        log_cb: LogCallback = None,
    ) -> bool:
        """Download via rsync over SSH. Returns True on success."""
        if not self._has_rsync():
            if log_cb:
                log_cb("rsync not found locally — falling back to tar download.")
            return False

        ssh_cmd = f"ssh -p {self.port} -o StrictHostKeyChecking=no"
        if self.key_filename:
            ssh_cmd += f" -i {self.key_filename}"

        src = f"{self.username}@{self.host}:{remote_path}/"
        dst = local_path.rstrip("/\\") + "/"
        os.makedirs(dst, exist_ok=True)

        cmd = [
            "rsync", "-avz",
            "--partial", "--progress", "--checksum",
            "-e", ssh_cmd,
            src, dst,
        ]

        if log_cb:
            log_cb(f"Rsync: {src} → {dst}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line and log_cb:
                    log_cb(f"  [rsync] {line}")
            rc = proc.wait()
            if rc != 0:
                if log_cb:
                    log_cb(f"rsync exited with code {rc}")
                return False
            if log_cb:
                log_cb("Rsync download complete.")
            return True
        except Exception as exc:
            if log_cb:
                log_cb(f"rsync error: {exc}")
            return False
