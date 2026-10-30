"""Download the Lab 4 CIHP competition files one by one.

Kaggle serves no bundled archive for this competition (the download-all endpoint
and folder downloads return 404), so this script lists every file through the
API and fetches each one with a thread pool. It is resumable: files that already
exist with the expected size are skipped, and 429/5xx responses trigger a shared
back-off so the whole pool slows down together.

Usage:
    python scripts/tools/download_data.py --out data --workers 8 --rps 5
    python scripts/tools/download_data.py --out data --limit 200   # smoke test

Kaggle answers bursts of per-file downloads with 404 instead of 429, so 404 is
treated as a quota signal (retry with a pool-wide pause), and --rps caps the
request rate. 400/401/403 stay fatal.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from kagglehub.clients import build_kaggle_client
from kagglesdk.competitions.types.competition_api_service import (
    ApiDownloadDataFileRequest,
    ApiListDataFilesRequest,
)
from tqdm import tqdm

SLUG: str = "2026-lab-4-semantic-segmentation-on-cihp-639401"
MANIFEST_NAME: str = ".kaggle_manifest.json"
FAILURES_NAME: str = ".download_failures.txt"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int


class Stats:
    """Pool-wide counters so rate limiting is visible in the progress bar."""

    def __init__(self) -> None:
        self.retries: int = 0
        self.rate_limited: int = 0
        self.size_mismatch: int = 0
        self.not_found: int = 0
        self._lock = threading.Lock()

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {"retry": self.retries, "429": self.rate_limited, "404": self.not_found, "mismatch": self.size_mismatch}


class SizeMismatch(IOError):
    """Downloaded byte count differs from the manifest; carries the response headers."""


class Throttle:
    """Pool-wide pacing: a request-per-second cap plus a shared pause after rate limiting."""

    def __init__(self, rps: float) -> None:
        self._until: float = 0.0
        self._interval: float = 1.0 / rps if rps > 0 else 0.0
        self._next_slot: float = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until this worker may send one more API request."""
        self.wait()
        if self._interval == 0.0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(self._next_slot, now)
            self._next_slot = slot + self._interval
        time.sleep(max(0.0, slot - time.monotonic()))

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, time.monotonic() + seconds)


def retry_after_seconds(exc: Exception, attempt: int) -> float:
    """Back-off delay for one failed attempt, honouring Retry-After when present."""
    response = getattr(exc, "response", None)
    header = response.headers.get("Retry-After") if response is not None else None
    if header and header.isdigit():
        return float(header) + random.uniform(0.0, 1.0)
    return min(2.0**attempt, 120.0) + random.uniform(0.0, 1.0)


def status_of(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return None if response is None else response.status_code


def list_remote_files(manifest_path: Path, refresh: bool) -> list[RemoteFile]:
    """List every competition file, caching the result next to the data."""
    if manifest_path.is_file() and not refresh:
        raw = json.loads(manifest_path.read_text())
        return [RemoteFile(name=r["name"], size=r["bytes"]) for r in raw]

    files: list[RemoteFile] = []
    token: str | None = None
    with build_kaggle_client() as api:
        service = api.competitions.competition_api_client
        while True:
            request = ApiListDataFilesRequest()
            request.competition_name = SLUG
            request.page_size = 1000
            if token:
                request.page_token = token
            for attempt in range(12):
                try:
                    response = service.list_data_files(request=request)
                    break
                except Exception as exc:  # noqa: BLE001
                    delay = retry_after_seconds(exc=exc, attempt=attempt + 2)
                    print(f"list page failed ({status_of(exc=exc)}), retry in {delay:.0f}s", flush=True)
                    time.sleep(delay)
            else:
                raise RuntimeError("could not list competition files")
            files.extend(RemoteFile(name=f.name, size=f.total_bytes) for f in response.files)
            print(f"listed {len(files)} files", flush=True)
            token = getattr(response, "next_page_token", None)
            if not token:
                break
            time.sleep(1.0)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([{"name": f.name, "bytes": f.size} for f in files]))
    return files


_local = threading.local()


def thread_service():  # noqa: ANN201
    """One Kaggle client per worker thread; requests sessions are not thread-safe."""
    if not hasattr(_local, "service"):
        _local.client = build_kaggle_client()
        _local.service = _local.client.competitions.competition_api_client
    return _local.service


def stream_to_file(response: requests.Response, target: Path, expected_size: int) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    written = 0
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 16):
            handle.write(chunk)
            written += len(chunk)
    if written != expected_size:
        tmp.unlink(missing_ok=True)
        headers = {k: response.headers.get(k) for k in ("Content-Type", "Content-Length", "Content-Encoding", "Content-Disposition")}
        raise SizeMismatch(f"size mismatch for {target.name}: got {written}, expected {expected_size}; headers {headers}")
    os.replace(src=tmp, dst=target)


def fetch_one(
    remote: RemoteFile,
    out_dir: Path,
    throttle: Throttle,
    stats: Stats,
    max_attempts: int,
    max_404: int,
) -> str:
    """Download one file; returns 'skipped', 'downloaded' or raises."""
    target = out_dir / remote.name
    if target.is_file() and target.stat().st_size == remote.size:
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    not_found = 0
    for attempt in range(max_attempts):
        throttle.acquire()
        try:
            request = ApiDownloadDataFileRequest()
            request.competition_name = SLUG
            request.file_name = remote.name
            result = thread_service().download_data_file(request=request)
            if isinstance(result, requests.Response):
                response = result
            else:
                response = requests.get(
                    url=result.url,
                    stream=True,
                    timeout=(10, 120),
                )
                response.raise_for_status()
            with response:
                stream_to_file(response=response, target=target, expected_size=remote.size)
            return "downloaded"
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            code = status_of(exc=exc)
            if code is not None and code in (400, 401, 403):
                raise
            if isinstance(exc, SizeMismatch):
                stats.bump(field="size_mismatch")
                if attempt >= 1:
                    raise
            stats.bump(field="retries")
            delay = retry_after_seconds(exc=exc, attempt=attempt + 1)
            if code == 404:
                # Kaggle sometimes answers a burst of file downloads with 404; a few retries with a
                # pool-wide pause recover those. Files the API never serves (see CLAUDE.md) fail fast.
                stats.bump(field="not_found")
                not_found += 1
                if not_found >= max_404:
                    raise
                throttle.pause(seconds=delay)
            elif code == 429:
                stats.bump(field="rate_limited")
                throttle.pause(seconds=delay)
            else:
                time.sleep(delay)
    raise RuntimeError(f"{remote.name}: giving up after {max_attempts} attempts: {last_exc!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"), help="destination folder")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="only the first N files (smoke test)")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--rps", type=float, default=5.0, help="API requests per second across all workers (0 = unlimited)")
    parser.add_argument("--include", nargs="*", default=None, help="only files whose name starts with one of these prefixes, e.g. test/ test_index.csv")
    parser.add_argument("--max-404", type=int, default=3, help="give up on a file after this many 404 answers")
    parser.add_argument("--log-every", type=int, default=200, help="print a plain progress line every N files (for log files)")
    parser.add_argument("--refresh-manifest", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_remote_files(manifest_path=out_dir / MANIFEST_NAME, refresh=args.refresh_manifest)
    total_bytes = sum(f.size for f in files)
    print(f"{len(files)} files, {total_bytes / 1e9:.2f} GB listed", flush=True)
    if args.include:
        files = [f for f in files if any(f.name.startswith(prefix) for prefix in args.include)]
        print(f"{len(files)} files match {args.include}", flush=True)
    if args.limit:
        files = files[: args.limit]

    throttle = Throttle(rps=args.rps)
    stats = Stats()
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_one,
                remote=f,
                out_dir=out_dir,
                throttle=throttle,
                stats=stats,
                max_attempts=args.max_attempts,
                max_404=args.max_404,
            ): f
            for f in files
        }
        with tqdm(total=len(futures), unit="file", mininterval=5.0, smoothing=0.05, file=sys.stderr) as bar:
            for done, future in enumerate(as_completed(futures), start=1):
                remote = futures[future]
                try:
                    counts[future.result()] += 1
                except Exception as exc:  # noqa: BLE001
                    counts["failed"] += 1
                    failures.append(f"{remote.name}\t{exc!r}")
                bar.update(1)
                bar.set_postfix({**counts, **stats.as_dict()}, refresh=False)
                if args.log_every and (done % args.log_every == 0 or done == len(futures)):
                    rate = counts["downloaded"] / max(time.time() - started, 1e-9)
                    print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(futures)} {counts} {stats.as_dict()} {rate:.2f} dl/s", flush=True)

    elapsed = time.time() - started
    print(f"done in {elapsed / 60:.1f} min: {counts} {stats.as_dict()}", flush=True)
    failure_path = out_dir / FAILURES_NAME
    if failures:
        failure_path.write_text("\n".join(failures) + "\n")
        print(f"{len(failures)} failures written to {failure_path}; rerun to retry", flush=True)
        return 1
    failure_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
