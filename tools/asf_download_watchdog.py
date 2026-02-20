#!/usr/bin/env python3
import argparse
import os
import pty
import select
import signal
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_ASC_SCRIPT = (
    "/home/dusen/Geokinesia/Early-warning-model-compare/"
    "ministro_insar_processing/asc/data/download-all-ASC_ministro_chili.py"
)
DEFAULT_DSC_SCRIPT = (
    "/home/dusen/Geokinesia/Early-warning-model-compare/"
    "ministro_insar_processing/des/data/download-all-DSC_ministro_chili.py"
)

# User-provided credentials.
DEFAULT_USERNAME = "SenDu"
DEFAULT_PASSWORD = "Dusen130730199206031838"


def directory_total_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_file():
                    total += file_path.stat().st_size
            except OSError:
                continue
    return total


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class DownloadTask:
    name: str
    script: Path
    username: str
    password: str
    monitor_interval: int
    progress_threshold_bytes: int
    restart_delay: int
    proc: Optional[subprocess.Popen] = None
    master_fd: Optional[int] = None
    reader_thread: Optional[threading.Thread] = None
    stop_reader: bool = False
    done: bool = False
    last_size: int = 0
    next_check_ts: float = 0.0
    log_path: Path = field(default_factory=lambda: Path("."))

    def __post_init__(self) -> None:
        self.workdir = self.script.parent
        self.log_path = self.workdir / f"watchdog-{self.name.lower()}.log"

    def start(self) -> None:
        if self.done:
            return
        self.stop_reader = False
        self.last_size = directory_total_bytes(self.workdir)
        self.next_check_ts = time.time() + self.monitor_interval

        master_fd, slave_fd = pty.openpty()
        cmd = [sys.executable, "-u", str(self.script)]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.workdir),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        self.master_fd = master_fd

        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        self._log(f"started pid={self.proc.pid} script={self.script}")

    def _log(self, msg: str) -> None:
        line = f"[{now_str()}] [{self.name}] {msg}"
        print(line, flush=True)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _send_line(self, text: str) -> None:
        if self.master_fd is None:
            return
        try:
            os.write(self.master_fd, (text + "\n").encode("utf-8"))
        except OSError:
            pass

    def _find_corrupt_slc_zips(self) -> list[Path]:
        bad_files: list[Path] = []
        for zip_path in sorted(self.workdir.glob("S1?_IW_SLC__1SDV_*.zip")):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    # testzip() returns the first bad member name, or None if all good.
                    if zf.testzip() is not None:
                        bad_files.append(zip_path)
            except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError):
                bad_files.append(zip_path)
        return bad_files

    def _reader_loop(self) -> None:
        sent_username = False
        sent_password = False

        while not self.stop_reader and self.master_fd is not None:
            try:
                rlist, _, _ = select.select([self.master_fd], [], [], 0.5)
                if not rlist:
                    if self.proc is not None and self.proc.poll() is not None:
                        break
                    continue
                data = os.read(self.master_fd, 4096)
                if not data:
                    break
            except OSError:
                break

            text = data.decode("utf-8", errors="ignore")
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(text)
            except OSError:
                pass

            if "Username:" in text:
                self._send_line(self.username)
                sent_username = True
                self._log("credentials: username sent")

            if "Password (will not be displayed):" in text:
                self._send_line(self.password)
                sent_password = True
                self._log("credentials: password sent")

            if sent_username and sent_password:
                sent_username = False
                sent_password = False

    def check_and_handle(self) -> None:
        if self.done:
            return

        if self.proc is None:
            self.start()
            return

        exit_code = self.proc.poll()
        if exit_code is not None:
            self._cleanup_fds()
            if exit_code == 0:
                bad_files = self._find_corrupt_slc_zips()
                if bad_files:
                    for bad in bad_files:
                        try:
                            bad.unlink()
                            self._log(f"removed corrupt zip: {bad.name}")
                        except OSError as exc:
                            self._log(f"failed to remove corrupt zip {bad.name}: {exc}")
                    self._log(
                        f"integrity check failed: {len(bad_files)} bad zip(s), restarting after {self.restart_delay}s"
                    )
                    time.sleep(self.restart_delay)
                    self.start()
                    return
                self.done = True
                self._log("completed successfully (zip integrity check passed)")
            else:
                self._log(f"exited with code={exit_code}, restarting after {self.restart_delay}s")
                time.sleep(self.restart_delay)
                self.start()
            return

        current_ts = time.time()
        if current_ts < self.next_check_ts:
            return

        current_size = directory_total_bytes(self.workdir)
        delta = current_size - self.last_size
        if delta > self.progress_threshold_bytes:
            self._log(f"progress ok: +{delta} bytes in last {self.monitor_interval}s")
            self.last_size = current_size
            self.next_check_ts = current_ts + self.monitor_interval
            return

        self._log(
            f"no progress in last {self.monitor_interval}s (delta={delta}), restarting process"
        )
        self.restart()
        self.next_check_ts = time.time() + self.monitor_interval

    def restart(self) -> None:
        self.stop()
        time.sleep(self.restart_delay)
        self.start()

    def stop(self) -> None:
        if self.proc is None:
            self._cleanup_fds()
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            pass

        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        self.proc = None
        self._cleanup_fds()

    def _cleanup_fds(self) -> None:
        self.stop_reader = True
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
        self.master_fd = None
        self.reader_thread = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch ASC/DSC ASF downloads and auto-restart if stalled."
    )
    parser.add_argument("--asc-script", default=DEFAULT_ASC_SCRIPT, help="ASC script path")
    parser.add_argument("--dsc-script", default=DEFAULT_DSC_SCRIPT, help="DSC script path")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="ASF username")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="ASF password")
    parser.add_argument(
        "--check-interval",
        type=int,
        default=1800,
        help="Progress check interval in seconds (default: 1800)",
    )
    parser.add_argument(
        "--progress-threshold-bytes",
        type=int,
        default=1,
        help="Minimum size growth considered as progress (default: 1)",
    )
    parser.add_argument(
        "--restart-delay",
        type=int,
        default=10,
        help="Delay before restart in seconds (default: 10)",
    )
    return parser.parse_args()


def validate_script(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"script not found: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    return path


def main() -> int:
    args = parse_args()
    asc_script = validate_script(args.asc_script)
    dsc_script = validate_script(args.dsc_script)

    tasks = [
        DownloadTask(
            name="ASC",
            script=asc_script,
            username=args.username,
            password=args.password,
            monitor_interval=args.check_interval,
            progress_threshold_bytes=args.progress_threshold_bytes,
            restart_delay=args.restart_delay,
        ),
        DownloadTask(
            name="DSC",
            script=dsc_script,
            username=args.username,
            password=args.password,
            monitor_interval=args.check_interval,
            progress_threshold_bytes=args.progress_threshold_bytes,
            restart_delay=args.restart_delay,
        ),
    ]

    stop = False

    def handle_signal(sig_num, _frame) -> None:
        nonlocal stop
        print(f"\n[{now_str()}] received signal {sig_num}, stopping...", flush=True)
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for task in tasks:
        task.start()

    try:
        while not stop:
            all_done = True
            for task in tasks:
                task.check_and_handle()
                if not task.done:
                    all_done = False
            if all_done:
                print(f"[{now_str()}] all download tasks completed", flush=True)
                return 0
            time.sleep(5)
    finally:
        for task in tasks:
            task.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
