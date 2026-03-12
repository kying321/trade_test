#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from snapshot_local_pi_recovery_state import (
    DEFAULT_LOCK_TIMEOUT_SEC,
    DEFAULT_LOCK_RETRY_SEC,
    acquire_lock,
    build_state_fingerprint,
    load_json,
    now_utc,
    now_utc_compact,
    release_lock,
    sha256_file,
)


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_WORKSPACE_SYSTEM_ROOT = Path.home() / ".openclaw" / "workspaces" / "pi" / "fenlie-system"


def parse_checkpoint_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint_manifest_invalid")
    return raw


def validate_checkpoint_manifest(manifest_path: Path, checksum_path: Path) -> tuple[bool, str | None]:
    if not manifest_path.exists():
        return False, "checkpoint_manifest_missing"
    if not checksum_path.exists():
        return False, "checkpoint_checksum_missing"
    checksum_payload = load_json(checksum_path)
    expected = str(checksum_payload.get("files", [{}])[0].get("sha256") if isinstance(checksum_payload.get("files"), list) and checksum_payload.get("files") else "" or "").strip()
    if not expected:
        expected = str(checksum_payload.get("sha256") or "").strip()
    actual = sha256_file(manifest_path)
    return expected.lower() == actual.lower(), actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore local PI paper recovery files from a checkpoint manifest under run-halfhour-pulse mutex."
    )
    parser.add_argument("--workspace-system-root", default=str(DEFAULT_WORKSPACE_SYSTEM_ROOT))
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.json or checkpoint dir.")
    parser.add_argument("--pulse-lock-path", default="")
    parser.add_argument("--expected-current-state-fingerprint", default="")
    parser.add_argument("--lock-timeout-sec", type=float, default=DEFAULT_LOCK_TIMEOUT_SEC)
    parser.add_argument("--lock-retry-sec", type=float, default=DEFAULT_LOCK_RETRY_SEC)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_system_root = Path(args.workspace_system_root).expanduser().resolve()
    checkpoint_input = Path(args.checkpoint).expanduser().resolve()
    checkpoint_root = checkpoint_input if checkpoint_input.is_dir() else checkpoint_input.parent
    manifest_path = checkpoint_root / "checkpoint.json" if checkpoint_input.is_dir() else checkpoint_input
    checksum_path = checkpoint_root / "checkpoint_checksum.json"
    pulse_lock_path = (
        Path(str(args.pulse_lock_path)).expanduser().resolve()
        if str(args.pulse_lock_path).strip()
        else (workspace_system_root / "output" / "state" / "run_halfhour_pulse.lock").resolve()
    )
    expected_current_state_fingerprint = (
        str(args.expected_current_state_fingerprint or "").strip() or None
    )

    out: dict[str, Any] = {
        "action": "restore_local_pi_recovery_state",
        "ok": False,
        "status": "initializing",
        "workspace_system_root": str(workspace_system_root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_checksum": str(checksum_path),
        "pulse_lock_path": str(pulse_lock_path),
        "dry_run": bool(args.dry_run or not args.write),
        "write_requested": bool(args.write),
        "expected_current_state_fingerprint": expected_current_state_fingerprint,
        "lock_acquired": None,
        "lock_wait_sec": None,
        "checkpoint_valid": False,
        "checkpoint_manifest_sha256": None,
        "current_state_fingerprint": None,
        "current_state_fingerprint_match": None,
        "pre_restore_backup_dir": None,
        "restore_plan": [],
    }

    checkpoint_valid, manifest_sha256 = validate_checkpoint_manifest(manifest_path, checksum_path)
    out["checkpoint_valid"] = checkpoint_valid
    out["checkpoint_manifest_sha256"] = manifest_sha256
    if not checkpoint_valid:
        out["status"] = "checkpoint_invalid"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    manifest = parse_checkpoint_manifest(manifest_path)
    files = manifest.get("files", [])
    if not isinstance(files, list) or not files:
        out["status"] = "checkpoint_files_missing"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    state_path = workspace_system_root / "output" / "state" / "spot_paper_state.json"
    current_state_fingerprint = build_state_fingerprint(load_json(state_path))
    out["current_state_fingerprint"] = current_state_fingerprint
    out["current_state_fingerprint_match"] = (
        None
        if expected_current_state_fingerprint is None
        else current_state_fingerprint == expected_current_state_fingerprint
    )
    if expected_current_state_fingerprint is not None and current_state_fingerprint != expected_current_state_fingerprint:
        out["status"] = "state_fingerprint_mismatch"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 5

    restore_plan: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        rel = Path(str(entry.get("relative_path") or "").strip())
        target = workspace_system_root / rel
        checkpoint_file = (
            Path(str(entry.get("checkpoint_path") or "").strip()).expanduser().resolve()
            if str(entry.get("checkpoint_path") or "").strip()
            else None
        )
        restore_plan.append(
            {
                "label": entry.get("label"),
                "target_path": str(target),
                "target_exists_before": target.exists(),
                "present_in_checkpoint": bool(entry.get("present")),
                "checkpoint_path": str(checkpoint_file) if checkpoint_file is not None else None,
                "checkpoint_sha256": entry.get("sha256"),
                "action": "restore_file" if bool(entry.get("present")) else "remove_file",
            }
        )
    out["restore_plan"] = restore_plan

    if bool(args.dry_run or not args.write):
        out["ok"] = True
        out["status"] = "dry_run"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    lockf = None
    lock_acquired = False
    lock_wait_sec = 0.0
    try:
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

        backup_dir = (
            workspace_system_root
            / "output"
            / "review"
            / "local_pi_recovery_checkpoints"
            / f"{now_utc_compact()}_restore_backup"
        )
        out["pre_restore_backup_dir"] = str(backup_dir)
        for plan in restore_plan:
            target_path = Path(str(plan["target_path"]))
            if target_path.exists():
                backup_path = backup_dir / target_path.relative_to(workspace_system_root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target_path, backup_path)
                plan["backup_path"] = str(backup_path)
            else:
                plan["backup_path"] = None

        for plan in restore_plan:
            target_path = Path(str(plan["target_path"]))
            if plan["action"] == "restore_file":
                checkpoint_path = Path(str(plan["checkpoint_path"]))
                if not checkpoint_path.exists():
                    out["status"] = "checkpoint_file_missing"
                    out["ok"] = False
                    print(json.dumps(out, ensure_ascii=False, indent=2))
                    return 4
                target_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp_restore")
                shutil.copy2(checkpoint_path, tmp_path)
                tmp_path.replace(target_path)
                plan["restored"] = True
            else:
                if target_path.exists():
                    target_path.unlink()
                    plan["removed"] = True
                else:
                    plan["removed"] = True

        out["ok"] = True
        out["status"] = "ok"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    finally:
        release_lock(lockf)


if __name__ == "__main__":
    raise SystemExit(main())
