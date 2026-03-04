#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import yaml


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _normalize_runtime_values(runtime: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in runtime.items():
        if isinstance(value, bool):
            out[str(key)] = 1.0 if value else 0.0
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
            continue
    return out


@contextmanager
def _run_halfhour_mutex(output_root: Path, timeout_seconds: float):
    lock_path = output_root / "state" / "run-halfhour-pulse.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = lock_path.open("a+", encoding="utf-8")
    try:
        import fcntl

        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"run-halfhour-pulse mutex timeout: {timeout_seconds:.1f}s")
                time.sleep(0.1)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fd.close()


def apply_runtime_profile(
    *,
    output_root: Path,
    profile_file: Path,
    profile_name: str,
    write: bool,
    allow_reflex_lock: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    profiles = _safe_load_yaml(profile_file)
    profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    runtime_raw = profile.get("runtime", {})
    if not isinstance(runtime_raw, dict) or not runtime_raw:
        raise ValueError(f"profile `{profile_name}` runtime is empty or invalid: {profile_file}")

    params_path = output_root / "artifacts" / "params_live.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _safe_load_yaml(params_path)
    runtime = _normalize_runtime_values(runtime_raw)

    changed: dict[str, dict[str, Any]] = {}
    merged = dict(existing)
    for key, value in runtime.items():
        before = merged.get(key, None)
        if before != value:
            changed[key] = {"before": before, "after": value}
        merged[key] = value

    reflex_active = str(existing.get("reflex_lock", "")).strip().upper() == "ACTIVE"
    blocked_by_reflex = bool(reflex_active and write and not allow_reflex_lock)

    out: dict[str, Any] = {
        "status": "dry_run" if not write else ("blocked_reflex_lock" if blocked_by_reflex else "applied"),
        "output_root": str(output_root),
        "params_path": str(params_path),
        "profile_file": str(profile_file),
        "profile_name": str(profile_name),
        "changed_keys": sorted(changed.keys()),
        "changed_count": int(len(changed)),
        "reflex_lock_active": bool(reflex_active),
        "write": bool(write),
        "allow_reflex_lock": bool(allow_reflex_lock),
        "backup_path": "",
    }
    if not write:
        return out
    if blocked_by_reflex:
        return out
    if not changed:
        out["status"] = "noop"
        return out

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = output_root / "artifacts" / f"params_live_backup_profile_{timestamp}.yaml"
    with _run_halfhour_mutex(output_root=output_root, timeout_seconds=timeout_seconds):
        if params_path.exists():
            backup.write_text(params_path.read_text(encoding="utf-8"), encoding="utf-8")
        params_path.write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8")
    out["backup_path"] = str(backup) if backup.exists() else ""
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply runtime profile to output/artifacts/params_live.yaml")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--profile-file", required=True)
    parser.add_argument("--profile", default="stable_profile")
    parser.add_argument("--write", action="store_true", help="Actually write params_live.yaml")
    parser.add_argument("--allow-reflex-lock", action="store_true", help="Allow write when reflex_lock=ACTIVE")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = apply_runtime_profile(
        output_root=Path(args.output_root).resolve(),
        profile_file=Path(args.profile_file).resolve(),
        profile_name=str(args.profile).strip(),
        write=bool(args.write),
        allow_reflex_lock=bool(args.allow_reflex_lock),
        timeout_seconds=float(args.timeout_seconds),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if str(out.get("status", "")) == "blocked_reflex_lock":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
