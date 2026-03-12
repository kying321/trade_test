#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BACKUP_KEEP = 5
DEFAULT_BACKUP_MAX_AGE_HOURS = 168
DEFAULT_LOCK_TIMEOUT_SEC = 5.0
DEFAULT_LOCK_RETRY_SEC = 0.05
BACKUP_PREFIX = "runtime_script_publish_"
EXCLUDED_FILE_NAMES = {".DS_Store"}
EXCLUDED_DIR_NAMES = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


@dataclass
class SyncStats:
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0

    @property
    def changed(self) -> bool:
        return (self.created + self.updated) > 0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_compact() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def iter_source_files(source_root: Path) -> Iterable[Path]:
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.name in EXCLUDED_FILE_NAMES:
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        yield path


def load_manifest_files(manifest_path: Path, source_root: Path) -> list[Path]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest_invalid:{manifest_path}")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"manifest_files_missing:{manifest_path}")
    resolved: list[Path] = []
    seen: set[str] = set()
    for item in files:
        rel = str(item or "").strip()
        if not rel:
            continue
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise ValueError(f"manifest_requires_relative_paths:{rel}")
        if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
            raise ValueError(f"manifest_path_blocked:{rel}")
        if rel_path.name in EXCLUDED_FILE_NAMES or rel_path.suffix in EXCLUDED_SUFFIXES:
            raise ValueError(f"manifest_path_blocked:{rel}")
        key = rel_path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        full = (source_root / rel_path).resolve()
        if not full.exists() or not full.is_file():
            raise FileNotFoundError(f"manifest_source_missing:{full}")
        if source_root not in full.parents:
            raise ValueError(f"manifest_path_escape:{rel}")
        resolved.append(full)
    if not resolved:
        raise ValueError(f"manifest_files_empty:{manifest_path}")
    return sorted(resolved)


def file_bytes_differ(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    if src.stat().st_size != dst.stat().st_size:
        return True
    return src.read_bytes() != dst.read_bytes()


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


def prune_backup_dirs(backups_root: Path, *, keep: int, max_age_hours: int) -> tuple[list[str], list[str]]:
    pruned_keep: list[str] = []
    pruned_age: list[str] = []
    if not backups_root.exists():
        return pruned_keep, pruned_age

    now = now_utc()
    dirs = sorted(
        (
            path
            for path in backups_root.iterdir()
            if path.is_dir() and path.name.startswith(BACKUP_PREFIX)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    expire_before = now - timedelta(hours=max_age_hours)
    survivors: list[Path] = []
    for path in dirs:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime < expire_before:
            shutil.rmtree(path, ignore_errors=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    for path in survivors[keep:]:
        shutil.rmtree(path, ignore_errors=True)
        pruned_keep.append(str(path))

    return pruned_keep, pruned_age


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish repo-managed local PI runtime scripts into ~/.openclaw/workspaces/pi/scripts."
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1] / "runtime" / "pi" / "scripts"),
        help="Repo-managed runtime scripts root.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(Path(__file__).resolve().parents[1] / "runtime" / "pi" / "runtime_manifest.json"),
        help="Manifest of repo-managed PI runtime scripts to publish.",
    )
    parser.add_argument(
        "--target-root",
        default=str(Path.home() / ".openclaw" / "workspaces" / "pi" / "scripts"),
        help="Local PI runtime scripts directory.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path.home() / ".openclaw" / "workspaces" / "pi" / "fenlie-system" / "output"),
        help="Output root used for backups and run-halfhour-pulse lock resolution.",
    )
    parser.add_argument(
        "--backup-keep",
        type=int,
        default=DEFAULT_BACKUP_KEEP,
        help="Keep at most this many runtime_script_publish backups.",
    )
    parser.add_argument(
        "--backup-max-age-hours",
        type=int,
        default=DEFAULT_BACKUP_MAX_AGE_HOURS,
        help="Delete runtime_script_publish backups older than this age.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the publish plan without mutating files.")
    parser.add_argument("--no-backup", action="store_true", help="Disable overwrite backups for this publish run.")
    parser.add_argument(
        "--pulse-lock-path",
        default="",
        help="Override run-halfhour-pulse mutex path. Defaults to <output-root>/state/run_halfhour_pulse.lock.",
    )
    parser.add_argument(
        "--lock-timeout-sec",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_SEC,
        help="Mutex acquisition timeout for non-dry-run publishes.",
    )
    parser.add_argument(
        "--lock-retry-sec",
        type=float,
        default=DEFAULT_LOCK_RETRY_SEC,
        help="Mutex acquisition retry interval for non-dry-run publishes.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    source_root = Path(str(args.source_root)).expanduser().resolve()
    manifest_path = Path(str(args.manifest_path)).expanduser().resolve()
    target_root = Path(str(args.target_root)).expanduser().resolve()
    output_root = Path(str(args.output_root)).expanduser().resolve()
    backup_keep = int(args.backup_keep)
    backup_max_age_hours = int(args.backup_max_age_hours)
    dry_run = bool(args.dry_run)
    backup_enabled = not bool(args.no_backup)
    pulse_lock_path_raw = str(args.pulse_lock_path or "").strip()
    pulse_lock_path = (
        Path(pulse_lock_path_raw).expanduser().resolve()
        if pulse_lock_path_raw
        else (output_root / "state" / "run_halfhour_pulse.lock").resolve()
    )

    out: dict[str, object] = {
        "action": "publish_local_pi_runtime_scripts",
        "ok": False,
        "changed": False,
        "dry_run": dry_run,
        "source_root": str(source_root),
        "manifest_path": str(manifest_path),
        "manifest_file_count": 0,
        "target_root": str(target_root),
        "output_root": str(output_root),
        "backup_enabled": backup_enabled,
        "backup_dir": None,
        "pulse_lock_path": str(pulse_lock_path),
        "lock_acquired": None,
        "lock_wait_sec": None,
        "created_files": [],
        "updated_files": [],
        "skipped_files": [],
        "stats": {},
        "pruned_backups_keep": [],
        "pruned_backups_age": [],
    }

    try:
        if backup_keep < 1:
            raise ValueError("backup_keep_must_be_positive")
        if backup_max_age_hours < 1:
            raise ValueError("backup_max_age_hours_must_be_positive")
        if not source_root.exists():
            raise FileNotFoundError(f"source_root_missing:{source_root}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest_missing:{manifest_path}")
        if target_root == source_root:
            raise ValueError("target_root_must_differ_from_source_root")

        source_files = load_manifest_files(manifest_path, source_root)
        out["manifest_file_count"] = len(source_files)
        if not source_files:
            raise FileNotFoundError(f"source_root_empty:{source_root}")

        backups_root = output_root / "backups"
        backup_dir = backups_root / f"{BACKUP_PREFIX}{now_utc_compact()}"
        stats = SyncStats()
        created_files: list[str] = []
        updated_files: list[str] = []
        skipped_files: list[str] = []
        pruned_keep: list[str] = []
        pruned_age: list[str] = []
        lockf = None
        try:
            if not dry_run:
                lockf, lock_acquired, lock_wait_sec = acquire_lock(
                    pulse_lock_path,
                    timeout_sec=max(0.0, float(args.lock_timeout_sec)),
                    retry_sec=max(0.01, float(args.lock_retry_sec)),
                )
                out["lock_acquired"] = bool(lock_acquired)
                out["lock_wait_sec"] = round(lock_wait_sec, 6)
                if not lock_acquired:
                    raise TimeoutError(f"run-halfhour-pulse mutex timeout: {float(args.lock_timeout_sec):.1f}s")

            for src in source_files:
                rel = src.relative_to(source_root)
                dst = target_root / rel
                if dst.exists() and dst.is_dir():
                    skipped_files.append(str(rel))
                    stats.skipped += 1
                    continue
                if not dst.exists():
                    if not dry_run:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        if dst.suffix == ".sh":
                            dst.chmod(0o755)
                    created_files.append(str(rel))
                    stats.created += 1
                    continue
                if file_bytes_differ(src, dst):
                    if not dry_run:
                        if backup_enabled:
                            backup_target = backup_dir / rel
                            backup_target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(dst, backup_target)
                        shutil.copy2(src, dst)
                        if dst.suffix == ".sh":
                            dst.chmod(0o755)
                    updated_files.append(str(rel))
                    stats.updated += 1
                    continue
                stats.unchanged += 1

            if not dry_run:
                if backup_enabled and not backup_dir.exists():
                    backup_dir = None  # type: ignore[assignment]
                pruned_keep, pruned_age = prune_backup_dirs(
                    backups_root,
                    keep=backup_keep,
                    max_age_hours=backup_max_age_hours,
                )
        finally:
            release_lock(lockf)

        out.update(
            {
                "ok": True,
                "changed": stats.changed,
                "backup_dir": str(backup_dir) if backup_enabled and stats.updated > 0 and not dry_run else None,
                "created_files": created_files,
                "updated_files": updated_files,
                "skipped_files": skipped_files,
                "stats": {
                    "created": stats.created,
                    "updated": stats.updated,
                    "unchanged": stats.unchanged,
                    "skipped": stats.skipped,
                },
                "pruned_backups_keep": pruned_keep,
                "pruned_backups_age": pruned_age,
            }
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        out["error"] = str(exc)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4 if "run-halfhour-pulse mutex timeout" in str(exc) else 2


if __name__ == "__main__":
    raise SystemExit(main())
