#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_WORKSPACE_SYSTEM_ROOT = Path.home() / ".openclaw" / "workspaces" / "pi" / "fenlie-system"
DEFAULT_CHECKPOINT_DIR = (
    DEFAULT_WORKSPACE_SYSTEM_ROOT / "output" / "review" / "local_pi_recovery_checkpoints"
)
DEFAULT_LOCK_TIMEOUT_SEC = 5.0
DEFAULT_LOCK_RETRY_SEC = 0.05


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_utc_compact() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_state_fingerprint(payload: dict[str, Any]) -> str:
    normalized = json.dumps(
        payload if isinstance(payload, dict) else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def acquire_lock(path: Path, *, timeout_sec: float, retry_sec: float) -> tuple[Any | None, bool, float]:
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl  # type: ignore
    except Exception:
        return None, False, max(0.0, time.monotonic() - started)
    lockf = path.open("a+", encoding="utf-8")
    while True:
        try:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lockf, True, max(0.0, time.monotonic() - started)
        except BlockingIOError:
            if (time.monotonic() - started) >= max(0.0, timeout_sec):
                try:
                    lockf.close()
                except Exception:
                    pass
                return None, False, max(0.0, time.monotonic() - started)
            time.sleep(max(0.01, retry_sec))
        except Exception:
            try:
                lockf.close()
            except Exception:
                pass
            return None, False, max(0.0, time.monotonic() - started)


def release_lock(lockf: Any | None) -> None:
    if lockf is None:
        return
    try:
        import fcntl  # type: ignore

        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lockf.close()
    except Exception:
        pass


def tracked_files(workspace_system_root: Path) -> list[tuple[str, Path]]:
    return [
        ("spot_paper_state", workspace_system_root / "output" / "state" / "spot_paper_state.json"),
        (
            "paper_consecutive_loss_ack",
            workspace_system_root / "output" / "state" / "paper_consecutive_loss_ack.json",
        ),
        (
            "paper_consecutive_loss_ack_checksum",
            workspace_system_root / "output" / "state" / "paper_consecutive_loss_ack_checksum.json",
        ),
    ]


def prune_checkpoints(
    checkpoint_root: Path, *, keep: int, max_age_hours: int
) -> tuple[list[str], list[str]]:
    pruned_keep: list[str] = []
    pruned_age: list[str] = []
    if not checkpoint_root.exists():
        return pruned_keep, pruned_age
    dirs = sorted(
        [p for p in checkpoint_root.glob("*_local_pi_recovery_checkpoint") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    cutoff = now_utc() - dt.timedelta(hours=max_age_hours)
    survivors: list[Path] = []
    for path in dirs:
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except Exception:
            continue
        if mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)
    for path in survivors[keep:]:
        shutil.rmtree(path, ignore_errors=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a consistent checkpoint for local PI paper recovery state under run-halfhour-pulse mutex."
    )
    parser.add_argument("--workspace-system-root", default=str(DEFAULT_WORKSPACE_SYSTEM_ROOT))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-keep", type=int, default=12)
    parser.add_argument("--checkpoint-max-age-hours", type=int, default=168)
    parser.add_argument("--pulse-lock-path", default="")
    parser.add_argument("--lock-timeout-sec", type=float, default=DEFAULT_LOCK_TIMEOUT_SEC)
    parser.add_argument("--lock-retry-sec", type=float, default=DEFAULT_LOCK_RETRY_SEC)
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_system_root = Path(args.workspace_system_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    pulse_lock_path = (
        Path(str(args.pulse_lock_path)).expanduser().resolve()
        if str(args.pulse_lock_path).strip()
        else (workspace_system_root / "output" / "state" / "run-halfhour-pulse.lock").resolve()
    )
    note = str(args.note or "").strip() or None
    stamp = now_utc_compact()
    checkpoint_root = checkpoint_dir / f"{stamp}_local_pi_recovery_checkpoint"
    files_root = checkpoint_root / "files"

    out: dict[str, Any] = {
        "action": "snapshot_local_pi_recovery_state",
        "ok": False,
        "status": "initializing",
        "workspace_system_root": str(workspace_system_root),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_root": str(checkpoint_root),
        "pulse_lock_path": str(pulse_lock_path),
        "lock_acquired": None,
        "lock_wait_sec": None,
        "dry_run": bool(args.dry_run),
        "note": note,
        "files": [],
        "state_fingerprint": None,
        "checkpoint_manifest": str(checkpoint_root / "checkpoint.json"),
        "checkpoint_checksum": str(checkpoint_root / "checkpoint_checksum.json"),
        "pruned_keep": [],
        "pruned_age": [],
    }

    if not workspace_system_root.exists():
        out["status"] = "workspace_missing"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    lockf = None
    lock_acquired = False
    lock_wait_sec = 0.0
    if not bool(args.dry_run):
        lockf, lock_acquired, lock_wait_sec = acquire_lock(
            pulse_lock_path,
            timeout_sec=max(0.0, float(args.lock_timeout_sec)),
            retry_sec=max(0.01, float(args.lock_retry_sec)),
        )
        out["lock_acquired"] = lock_acquired
        out["lock_wait_sec"] = round(lock_wait_sec, 6)
        if not lock_acquired:
            out["status"] = "lock_timeout"
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 3

    try:
        entries: list[dict[str, Any]] = []
        state_fingerprint: str | None = None
        for label, src in tracked_files(workspace_system_root):
            rel = src.relative_to(workspace_system_root)
            entry: dict[str, Any] = {
                "label": label,
                "source_path": str(src),
                "relative_path": str(rel),
                "present": src.exists(),
                "size_bytes": int(src.stat().st_size) if src.exists() else 0,
                "sha256": None,
                "checkpoint_path": None,
            }
            if src.exists():
                entry["sha256"] = sha256_file(src)
                if label == "spot_paper_state":
                    state_fingerprint = build_state_fingerprint(load_json(src))
                if not bool(args.dry_run):
                    dst = files_root / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    entry["checkpoint_path"] = str(dst)
            entries.append(entry)

        out["files"] = entries
        out["state_fingerprint"] = state_fingerprint

        if bool(args.dry_run):
            out["ok"] = True
            out["status"] = "dry_run"
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

        checkpoint_root.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_root / "checkpoint.json"
        checksum_path = checkpoint_root / "checkpoint_checksum.json"
        out["generated_at"] = now_utc().isoformat()
        manifest_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        digest = sha256_file(manifest_path)
        checksum_payload = {
            "generated_at": now_utc().isoformat(),
            "files": [
                {
                    "path": str(manifest_path),
                    "sha256": digest,
                    "size_bytes": int(manifest_path.stat().st_size),
                }
            ],
        }
        checksum_path.write_text(
            json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pruned_keep, pruned_age = prune_checkpoints(
            checkpoint_dir,
            keep=max(1, int(args.checkpoint_keep)),
            max_age_hours=max(1, int(args.checkpoint_max_age_hours)),
        )
        out["pruned_keep"] = pruned_keep
        out["pruned_age"] = pruned_age
        out["ok"] = True
        out["status"] = "ok"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    finally:
        release_lock(lockf)


if __name__ == "__main__":
    raise SystemExit(main())
