#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from paper_consecutive_loss_ack_archive_status import inspect_archive, inspect_live_ack
from paper_consecutive_loss_guardrail_status import (
    build_backfill_preview,
    build_recovery_plan,
    build_state_fingerprint,
    evaluate_manual_ack_status,
    load_json,
    parse_ts,
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
DEFAULT_REVIEW_DIR = DEFAULT_WORKSPACE_SYSTEM_ROOT / "output" / "review"
DEFAULT_CHECKPOINT_DIR = DEFAULT_REVIEW_DIR / "local_pi_recovery_checkpoints"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_handoffs(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_checksum: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    pruned_keep: list[str] = []
    pruned_age: list[str] = []
    review_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        review_dir.glob("*_local_pi_recovery_handoff*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    survivors: list[Path] = []
    protected = {current_artifact.name, current_checksum.name}
    for path in candidates:
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except Exception:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)
    artifact_like = [p for p in survivors if p.name.endswith(".json")]
    for path in artifact_like[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def latest_checkpoint(checkpoint_dir: Path) -> dict[str, Any] | None:
    dirs = sorted(
        [p for p in checkpoint_dir.glob("*_local_pi_recovery_checkpoint") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not dirs:
        return None
    root = dirs[0]
    manifest = root / "checkpoint.json"
    checksum = root / "checkpoint_checksum.json"
    payload = load_json(manifest)
    if not payload:
        return {
            "present": False,
            "checkpoint_root": str(root),
            "checkpoint_manifest": str(manifest),
            "checkpoint_checksum": str(checksum),
        }
    guidance = {
        "dry_run_command": (
            f'cd {SYSTEM_ROOT} && '
            f'LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT="{manifest}" '
            "scripts/openclaw_cloud_bridge.sh rollback-local-pi-recovery-state"
        ),
        "write_command": (
            f'cd {SYSTEM_ROOT} && '
            "LOCAL_PI_RECOVERY_RESTORE_WRITE=true "
            f'LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT="{manifest}" '
            "scripts/openclaw_cloud_bridge.sh rollback-local-pi-recovery-state"
        ),
    }
    return {
        "present": True,
        "checkpoint_root": str(root),
        "checkpoint_manifest": str(manifest),
        "checkpoint_checksum": str(checksum),
        "generated_at": payload.get("generated_at"),
        "note": payload.get("note"),
        "state_fingerprint": payload.get("state_fingerprint"),
        "lock_acquired": payload.get("lock_acquired"),
        "files": payload.get("files"),
        "rollback_guidance": guidance,
    }


def extract_latest_pi_cycle_envelope_from_log(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists() or not log_path.is_file():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        candidate = line.strip()
        if not candidate.startswith("{") or '"domain": "pi_cycle"' not in candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def latest_retro(review_dir: Path) -> dict[str, Any] | None:
    files = sorted(
        review_dir.glob("*_pi_launchd_auto_retro.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    path = files[0]
    payload = load_json(path)
    latest_envelope: dict[str, Any] | None = None
    launchd_log_path = None
    if isinstance(payload, dict):
        launchd_log_value = str(payload.get("launchd_log") or "").strip()
        if launchd_log_value:
            launchd_log_path = Path(launchd_log_value).expanduser()
            latest_envelope = extract_latest_pi_cycle_envelope_from_log(launchd_log_path)
    direct_or_envelope = payload if isinstance(payload, dict) else {}
    if latest_envelope:
        direct_or_envelope = latest_envelope
    return {
        "present": True,
        "path": str(path),
        "launchd_log": None if launchd_log_path is None else str(launchd_log_path),
        "summary_source": "launchd_log_tail" if latest_envelope else "retro_file",
        "ts": direct_or_envelope.get("ts") or payload.get("generated_at_utc"),
        "status": direct_or_envelope.get("status"),
        "duration_sec": direct_or_envelope.get("duration_sec"),
        "core_execution_status": direct_or_envelope.get("core_execution_status"),
        "core_execution_reason": direct_or_envelope.get("core_execution_reason"),
        "core_execution_decision": direct_or_envelope.get("core_execution_decision"),
        "ops_next_action": direct_or_envelope.get("ops_next_action"),
        "ops_next_action_reason": direct_or_envelope.get("ops_next_action_reason"),
    }


def compute_guardrail_status(
    *,
    workspace_system_root: Path,
    now_ts: dt.datetime,
    stop_threshold: int,
    cooldown_hours: float,
) -> dict[str, Any]:
    state_path = workspace_system_root / "output" / "state" / "spot_paper_state.json"
    ack_path = workspace_system_root / "output" / "state" / "paper_consecutive_loss_ack.json"
    checksum_path = workspace_system_root / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"
    ledger_path = workspace_system_root / "output" / "logs" / "paper_execution_ledger.jsonl"
    archive_dir = workspace_system_root / "output" / "review" / "paper_consecutive_loss_ack_archive"
    archive_manifest = archive_dir / "manifest.jsonl"

    state = load_json(state_path)
    streak = int(state.get("consecutive_losses") or 0)
    last_loss_ts_text = state.get("last_loss_ts")
    last_loss_ts = parse_ts(last_loss_ts_text)
    backfill_preview = build_backfill_preview(
        ledger_path=ledger_path,
        current_streak=streak,
        current_last_loss_ts=str(last_loss_ts_text or "") or None,
    )
    manual_ack_eligible, manual_ack_reasons, cooldown_elapsed_hours = evaluate_manual_ack_status(
        now_utc=now_ts,
        streak=streak,
        stop_threshold=stop_threshold,
        last_loss_ts=last_loss_ts,
        cooldown_required_hours=cooldown_hours,
        allow_missing_last_loss_ts=False,
    )
    ack_live = inspect_live_ack(ack_path, checksum_path)
    ack_archive = inspect_archive(
        archive_dir,
        archive_manifest,
        manifest_tail=5,
        archive_keep_files=max(
            1,
            int(os.getenv("LIE_PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES", "12")),
        ),
    )

    ack_payload = ack_live.get("payload") if isinstance(ack_live.get("payload"), dict) else {}
    ack_present = bool(ack_live.get("present"))
    ack_eligible_for_apply_now = (
        ack_present
        and bool(ack_live.get("checksum_valid"))
        and bool(ack_payload.get("active"))
        and int(ack_payload.get("uses_remaining") or 0) > 0
        and int(ack_payload.get("streak_snapshot") or 0) == streak
    )
    recovery_plan = build_recovery_plan(
        system_root=SYSTEM_ROOT,
        manual_ack_eligible=manual_ack_eligible,
        manual_ack_reasons=manual_ack_reasons,
        ack_eligible_for_apply_now=ack_eligible_for_apply_now,
        backfill_preview=backfill_preview,
    )
    return {
        "state_path": str(state_path),
        "state_fingerprint": build_state_fingerprint(state),
        "current_streak": streak,
        "last_loss_ts": fmt_utc(last_loss_ts),
        "guardrail_hit": streak >= stop_threshold,
        "stop_threshold": stop_threshold,
        "cooldown_hours_required": cooldown_hours,
        "cooldown_elapsed_hours": cooldown_elapsed_hours,
        "manual_ack_eligible": manual_ack_eligible,
        "manual_ack_reasons": manual_ack_reasons,
        "backfill_preview": backfill_preview,
        "recovery_plan": recovery_plan,
        "ack": {
            "present": ack_present,
            "checksum_valid": ack_live.get("checksum_valid"),
            "eligible_for_apply_now": ack_eligible_for_apply_now,
            "payload": ack_payload or None,
        },
        "ack_archive": ack_archive,
    }


def build_handoff_summary(
    *,
    guardrail_status: dict[str, Any],
    latest_archive_status: dict[str, Any],
    latest_checkpoint_status: dict[str, Any] | None,
    latest_retro_status: dict[str, Any] | None,
) -> dict[str, Any]:
    ack_payload = (
        guardrail_status.get("ack", {}).get("payload")
        if isinstance(guardrail_status.get("ack"), dict)
        else None
    )
    archive_payload = (
        latest_archive_status.get("latest_archive", {}).get("payload")
        if isinstance(latest_archive_status.get("latest_archive"), dict)
        else None
    )
    live_ack_present = bool(guardrail_status.get("ack", {}).get("present"))
    archive_consumed = (
        isinstance(archive_payload, dict)
        and str(archive_payload.get("archive_reason") or "").strip() == "consumed"
    )

    if live_ack_present:
        handoff_state = "ack_live_ready"
        summary = "A live single-use ack is present and ready for the next full-cycle."
        recommended_action = "Run one controlled full-cycle if you intend to consume the current ack."
    elif archive_consumed:
        handoff_state = "ack_consumed_archived"
        summary = "The latest single-use ack has already been consumed and archived by runtime."
        recommended_action = "No immediate action is required. Only regenerate a new ack if you intentionally want another bypass cycle."
    elif str(guardrail_status.get("recovery_plan", {}).get("next_action") or "") == "write_manual_ack":
        handoff_state = "ack_write_available"
        summary = "last_loss_ts is restored and the next reversible step is writing a manual ack."
        recommended_action = "Write a new single-use ack only if you need one more bypass cycle."
    else:
        handoff_state = "guardrail_review"
        summary = "Recovery state needs review before the next action."
        recommended_action = "Review the current guardrail status and recovery plan."

    operator_commands = [
        f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh local-pi-consecutive-loss-guardrail-status",
        f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh local-pi-ack-archive-status",
    ]
    if isinstance(latest_checkpoint_status, dict) and latest_checkpoint_status.get("present"):
        rollback = latest_checkpoint_status.get("rollback_guidance", {})
        if isinstance(rollback, dict):
            if rollback.get("dry_run_command"):
                operator_commands.append(str(rollback["dry_run_command"]))
            if rollback.get("write_command"):
                operator_commands.append(str(rollback["write_command"]))
    if handoff_state == "ack_live_ready":
        operator_commands.append(
            f"cd {SYSTEM_ROOT} && LOCAL_PI_PREPARE_BEFORE_FULL_SMOKE=false scripts/openclaw_cloud_bridge.sh smoke-local-pi-cycle"
        )

    return {
        "handoff_state": handoff_state,
        "summary": summary,
        "recommended_action": recommended_action,
        "current_next_action": guardrail_status.get("recovery_plan", {}).get("next_action"),
        "latest_retro_status": None if not isinstance(latest_retro_status, dict) else latest_retro_status.get("status"),
        "latest_retro_reason": None if not isinstance(latest_retro_status, dict) else latest_retro_status.get("core_execution_reason"),
        "live_ack_present": live_ack_present,
        "archive_consumed": archive_consumed,
        "rollback_available": bool(
            isinstance(latest_checkpoint_status, dict) and latest_checkpoint_status.get("present")
        ),
        "operator_commands": operator_commands,
        "latest_archive_payload": archive_payload,
        "live_ack_payload": ack_payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a single-file handoff artifact summarizing local PI paper recovery state, rollback path, latest archive, and latest retro."
    )
    parser.add_argument("--workspace-system-root", default=str(DEFAULT_WORKSPACE_SYSTEM_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--stop-threshold", type=int, default=3)
    parser.add_argument("--cooldown-hours", type=float, default=12.0)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_system_root = Path(args.workspace_system_root).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    now_ts = parse_ts(args.now) or now_utc()

    out: dict[str, Any] = {
        "action": "build_local_pi_recovery_handoff",
        "ok": False,
        "status": "initializing",
        "generated_at": fmt_utc(now_ts),
        "workspace_system_root": str(workspace_system_root),
        "review_dir": str(review_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "guardrail_status": None,
        "latest_ack_archive_status": None,
        "latest_checkpoint": None,
        "latest_retro": None,
        "operator_handoff": None,
        "artifact_status_label": None,
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }

    if not workspace_system_root.exists():
        out["status"] = "workspace_missing"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    state_path = workspace_system_root / "output" / "state" / "spot_paper_state.json"
    if not state_path.exists():
        out["status"] = "state_missing"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    guardrail_status = compute_guardrail_status(
        workspace_system_root=workspace_system_root,
        now_ts=now_ts,
        stop_threshold=max(1, int(args.stop_threshold)),
        cooldown_hours=max(0.0, float(args.cooldown_hours)),
    )
    latest_ack_archive_status = guardrail_status["ack_archive"]
    latest_checkpoint_status = latest_checkpoint(checkpoint_dir)
    latest_retro_status = latest_retro(review_dir)
    operator_handoff = build_handoff_summary(
        guardrail_status=guardrail_status,
        latest_archive_status=latest_ack_archive_status,
        latest_checkpoint_status=latest_checkpoint_status,
        latest_retro_status=latest_retro_status,
    )

    out["guardrail_status"] = guardrail_status
    out["latest_ack_archive_status"] = latest_ack_archive_status
    out["latest_checkpoint"] = latest_checkpoint_status
    out["latest_retro"] = latest_retro_status
    out["operator_handoff"] = operator_handoff
    out["ok"] = True
    out["status"] = "ok"
    out["artifact_status_label"] = "handoff-ok"
    out["artifact_label"] = f"local-pi-recovery-handoff:{operator_handoff['handoff_state']}"
    out["artifact_tags"] = [
        "local-pi",
        "recovery-handoff",
        str(operator_handoff["handoff_state"]),
        "read-only",
    ]

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_local_pi_recovery_handoff.json"
    checksum_path = review_dir / f"{stamp}_local_pi_recovery_handoff_checksum.json"
    out["artifact"] = str(artifact_path)
    out["checksum"] = str(checksum_path)
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(artifact_path)
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": fmt_utc(now_ts),
                "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                "files": [
                    {
                        "path": str(artifact_path),
                        "sha256": digest,
                        "size_bytes": int(artifact_path.stat().st_size),
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pruned_keep, pruned_age = prune_handoffs(
        review_dir,
        current_artifact=artifact_path,
        current_checksum=checksum_path,
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    out["pruned_keep"] = pruned_keep
    out["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
