#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any

from backfill_paper_last_loss_ts import DEFAULT_LEDGER_PATH, find_candidates, load_sell_events
from paper_consecutive_loss_ack_archive_status import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_ARCHIVE_MANIFEST_PATH,
    inspect_archive,
    inspect_live_ack,
)


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_STATE_PATH = SYSTEM_ROOT / "output" / "state" / "spot_paper_state.json"
DEFAULT_ACK_PATH = SYSTEM_ROOT / "output" / "state" / "paper_consecutive_loss_ack.json"
DEFAULT_CHECKSUM_PATH = SYSTEM_ROOT / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"


def parse_ts(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def build_state_fingerprint(state: dict[str, Any]) -> str:
    normalized = json.dumps(state if isinstance(state, dict) else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only status summary for the local paper consecutive-loss guardrail and optional ack artifact."
    )
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--ack-path", default=str(DEFAULT_ACK_PATH))
    parser.add_argument("--checksum-path", default=str(DEFAULT_CHECKSUM_PATH))
    parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--archive-manifest-path", default=str(DEFAULT_ARCHIVE_MANIFEST_PATH))
    parser.add_argument(
        "--archive-keep-files",
        type=int,
        default=max(1, int(os.getenv("LIE_PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES", "12"))),
    )
    parser.add_argument("--archive-manifest-tail", type=int, default=5)
    parser.add_argument("--cooldown-hours", type=float, default=12.0)
    parser.add_argument("--stop-threshold", type=int, default=3)
    parser.add_argument("--allow-missing-last-loss-ts", action="store_true")
    parser.add_argument("--now", default="", help="Optional ISO8601 UTC override for deterministic checks.")
    return parser


def build_backfill_preview(
    *,
    ledger_path: Path,
    current_streak: int,
    current_last_loss_ts: str | None,
) -> dict[str, Any]:
    sell_events = load_sell_events(ledger_path)
    candidates = find_candidates(sell_events)
    latest_sell = candidates["latest_sell"]
    latest_negative = candidates["latest_negative_sell"]
    trailing_negative_streak = int(candidates["trailing_negative_streak"])
    base_reasons: list[str] = []
    if current_streak <= 0:
        base_reasons.append("no_active_consecutive_loss_streak")
    if current_last_loss_ts:
        base_reasons.append("last_loss_ts_already_present")
    if int(candidates["sell_events"]) == 0:
        base_reasons.append("ledger_sell_events_missing")
    if latest_negative is None:
        base_reasons.append("ledger_negative_sell_missing")

    strict_reasons = list(base_reasons)
    strict_selected_ts = None
    strict_selected_method = None
    if not strict_reasons:
        if trailing_negative_streak == current_streak and latest_negative is not None:
            strict_selected_method = "ledger_trailing_streak_match"
            strict_selected_ts = str(latest_negative["ts_text"])
        else:
            strict_reasons.append(
                f"trailing_negative_streak_mismatch(expected={current_streak},actual={trailing_negative_streak})"
            )

    fallback_reasons = list(base_reasons)
    fallback_selected_ts = None
    fallback_selected_method = None
    if not fallback_reasons:
        if trailing_negative_streak == current_streak and latest_negative is not None:
            fallback_selected_method = "ledger_trailing_streak_match"
            fallback_selected_ts = str(latest_negative["ts_text"])
        elif latest_negative is not None:
            fallback_selected_method = "ledger_latest_negative_fallback"
            fallback_selected_ts = str(latest_negative["ts_text"])
            fallback_reasons.append("streak_match_bypassed_via_latest_loss_fallback")
        else:
            fallback_reasons.append("ledger_negative_sell_missing")

    strict_eligible = strict_selected_ts is not None and not strict_reasons
    fallback_eligible = fallback_selected_ts is not None and all(
        reason not in {
            "no_active_consecutive_loss_streak",
            "last_loss_ts_already_present",
            "ledger_sell_events_missing",
            "ledger_negative_sell_missing",
        }
        for reason in fallback_reasons
    )

    return {
        "ledger_path": str(ledger_path),
        "sell_event_count": int(candidates["sell_events"]),
        "trailing_negative_streak": trailing_negative_streak,
        "latest_sell": {
            "ts": latest_sell["ts_text"],
            "realized_pnl_change": latest_sell["realized_pnl_change"],
        }
        if isinstance(latest_sell, dict)
        else None,
        "latest_negative_sell": {
            "ts": latest_negative["ts_text"],
            "realized_pnl_change": latest_negative["realized_pnl_change"],
        }
        if isinstance(latest_negative, dict)
        else None,
        "strict_candidate": {
            "eligible": strict_eligible,
            "selected_method": strict_selected_method,
            "selected_last_loss_ts": strict_selected_ts,
            "reasons": strict_reasons,
        },
        "fallback_candidate": {
            "eligible": fallback_eligible,
            "selected_method": fallback_selected_method,
            "selected_last_loss_ts": fallback_selected_ts,
            "reasons": fallback_reasons,
        },
    }


def evaluate_manual_ack_status(
    *,
    now_utc: dt.datetime,
    streak: int,
    stop_threshold: int,
    last_loss_ts: dt.datetime | None,
    cooldown_required_hours: float,
    allow_missing_last_loss_ts: bool,
) -> tuple[bool, list[str], float | None]:
    cooldown_elapsed_hours: float | None = None
    reasons: list[str] = []
    if streak < stop_threshold:
        reasons.append("guardrail_not_hit")
    if last_loss_ts is not None:
        cooldown_elapsed_hours = max(0.0, (now_utc - last_loss_ts).total_seconds() / 3600.0)
        if cooldown_elapsed_hours < cooldown_required_hours:
            reasons.append("cooldown_active")
    elif not allow_missing_last_loss_ts:
        reasons.append("last_loss_ts_missing")
    return len(reasons) == 0, reasons, cooldown_elapsed_hours


def build_recovery_plan(
    *,
    system_root: Path,
    manual_ack_eligible: bool,
    manual_ack_reasons: list[str],
    ack_eligible_for_apply_now: bool,
    backfill_preview: dict[str, Any],
) -> dict[str, Any]:
    root_cmd = f"cd {shlex.quote(str(system_root))}"
    status_cmd = f"{root_cmd} && scripts/openclaw_cloud_bridge.sh local-pi-consecutive-loss-guardrail-status"
    ack_write_cmd = (
        f"{root_cmd} && "
        "LOCAL_PI_CONSECUTIVE_LOSS_ACK_WRITE=true "
        "scripts/openclaw_cloud_bridge.sh ack-local-pi-consecutive-loss-guardrail"
    )
    strict_backfill_write_cmd = (
        f"{root_cmd} && "
        "LOCAL_PI_LAST_LOSS_TS_BACKFILL_WRITE=true "
        "scripts/openclaw_cloud_bridge.sh backfill-local-pi-last-loss-ts"
    )
    fallback_backfill_preview_cmd = (
        f"{root_cmd} && "
        "LOCAL_PI_LAST_LOSS_TS_BACKFILL_ALLOW_LATEST_LOSS_FALLBACK=true "
        "scripts/openclaw_cloud_bridge.sh backfill-local-pi-last-loss-ts"
    )
    fallback_backfill_write_cmd = (
        f"{root_cmd} && "
        "LOCAL_PI_LAST_LOSS_TS_BACKFILL_ALLOW_LATEST_LOSS_FALLBACK=true "
        "LOCAL_PI_LAST_LOSS_TS_BACKFILL_WRITE=true "
        "scripts/openclaw_cloud_bridge.sh backfill-local-pi-last-loss-ts"
    )
    full_smoke_cmd = f"{root_cmd} && scripts/openclaw_cloud_bridge.sh smoke-local-pi-cycle"

    strict_candidate = backfill_preview.get("strict_candidate") if isinstance(backfill_preview, dict) else {}
    fallback_candidate = backfill_preview.get("fallback_candidate") if isinstance(backfill_preview, dict) else {}
    strict_eligible = bool(isinstance(strict_candidate, dict) and strict_candidate.get("eligible"))
    fallback_eligible = bool(isinstance(fallback_candidate, dict) and fallback_candidate.get("eligible"))

    next_action = "observe_only"
    level = "observe"
    reason = "no_recovery_action_required"
    commands = [status_cmd]

    if ack_eligible_for_apply_now:
        next_action = "run_full_cycle_with_existing_ack"
        level = "execute"
        reason = "ack_artifact_ready"
        commands = [status_cmd, full_smoke_cmd]
    elif manual_ack_eligible:
        next_action = "write_manual_ack"
        level = "write"
        reason = "manual_ack_eligible"
        commands = [status_cmd, ack_write_cmd]
    elif "last_loss_ts_missing" in manual_ack_reasons:
        if strict_eligible:
            next_action = "write_strict_last_loss_ts_backfill"
            level = "write"
            reason = "strict_backfill_candidate_available"
            commands = [status_cmd, strict_backfill_write_cmd]
        elif fallback_eligible:
            next_action = "review_fallback_last_loss_ts_backfill"
            level = "review"
            reason = "strict_backfill_unavailable_but_fallback_candidate_available"
            commands = [status_cmd, fallback_backfill_preview_cmd, fallback_backfill_write_cmd]
        else:
            next_action = "wait_for_new_loss_event_or_reset"
            level = "observe"
            reason = "no_backfill_candidate_available"
    elif "cooldown_active" in manual_ack_reasons:
        next_action = "wait_for_cooldown"
        level = "observe"
        reason = "cooldown_active"
    elif "guardrail_not_hit" in manual_ack_reasons:
        next_action = "none_guardrail_not_hit"
        level = "observe"
        reason = "guardrail_not_hit"

    return {
        "next_action": next_action,
        "action_level": level,
        "reason": reason,
        "commands": commands,
    }


def build_write_projection(
    *,
    now_utc: dt.datetime,
    stop_threshold: int,
    cooldown_required_hours: float,
    allow_missing_last_loss_ts: bool,
    ledger_path: Path,
    current_streak: int,
    current_last_loss_ts_raw: str | None,
    ack_eligible_for_apply_now: bool,
    ack_present: bool,
    backfill_preview: dict[str, Any],
) -> dict[str, Any]:
    simulated_last_loss_ts_raw = current_last_loss_ts_raw
    simulated_ack_eligible = ack_eligible_for_apply_now
    simulated_ack_present = ack_present

    manual_ack_eligible, manual_ack_reasons, cooldown_elapsed_hours = evaluate_manual_ack_status(
        now_utc=now_utc,
        streak=current_streak,
        stop_threshold=stop_threshold,
        last_loss_ts=parse_ts(simulated_last_loss_ts_raw),
        cooldown_required_hours=cooldown_required_hours,
        allow_missing_last_loss_ts=allow_missing_last_loss_ts,
    )
    simulated_backfill_preview = backfill_preview
    simulated_recovery_plan = build_recovery_plan(
        system_root=SYSTEM_ROOT,
        manual_ack_eligible=manual_ack_eligible,
        manual_ack_reasons=manual_ack_reasons,
        ack_eligible_for_apply_now=simulated_ack_eligible,
        backfill_preview=simulated_backfill_preview,
    )

    projected_steps: list[dict[str, Any]] = []
    terminal_action = None
    max_steps = 3

    for _ in range(max_steps):
        next_action = str(simulated_recovery_plan.get("next_action") or "").strip()
        if not next_action or next_action in {
            "observe_only",
            "wait_for_cooldown",
            "wait_for_new_loss_event_or_reset",
            "none_guardrail_not_hit",
        }:
            break

        step_payload: dict[str, Any] = {
            "from_next_action": next_action,
            "from_action_level": simulated_recovery_plan.get("action_level"),
            "from_reason": simulated_recovery_plan.get("reason"),
        }

        if next_action == "write_manual_ack":
            if not manual_ack_eligible:
                step_payload["blocked"] = True
                step_payload["blocked_reason"] = "manual_ack_no_longer_eligible"
                projected_steps.append(step_payload)
                break
            step_payload["simulated_step"] = "ack_write"
            simulated_ack_eligible = True
            simulated_ack_present = True
        elif next_action == "write_strict_last_loss_ts_backfill":
            strict_candidate = (
                simulated_backfill_preview.get("strict_candidate")
                if isinstance(simulated_backfill_preview, dict)
                else {}
            )
            if not isinstance(strict_candidate, dict) or not bool(strict_candidate.get("eligible")):
                step_payload["blocked"] = True
                step_payload["blocked_reason"] = "strict_backfill_not_eligible"
                projected_steps.append(step_payload)
                break
            simulated_last_loss_ts_raw = str(strict_candidate.get("selected_last_loss_ts") or "").strip() or None
            step_payload["simulated_step"] = "strict_backfill_write"
            step_payload["selected_last_loss_ts"] = simulated_last_loss_ts_raw
        elif next_action == "review_fallback_last_loss_ts_backfill":
            fallback_candidate = (
                simulated_backfill_preview.get("fallback_candidate")
                if isinstance(simulated_backfill_preview, dict)
                else {}
            )
            if not isinstance(fallback_candidate, dict) or not bool(fallback_candidate.get("eligible")):
                step_payload["blocked"] = True
                step_payload["blocked_reason"] = "fallback_backfill_not_eligible"
                projected_steps.append(step_payload)
                break
            simulated_last_loss_ts_raw = str(fallback_candidate.get("selected_last_loss_ts") or "").strip() or None
            step_payload["simulated_step"] = "fallback_backfill_write"
            step_payload["selected_last_loss_ts"] = simulated_last_loss_ts_raw
        elif next_action == "run_full_cycle_with_existing_ack":
            step_payload["simulated_step"] = "full_cycle_gate_ready"
            step_payload["terminal"] = True
            terminal_action = "full_cycle_gate_ready"
            projected_steps.append(step_payload)
            break
        else:
            step_payload["blocked"] = True
            step_payload["blocked_reason"] = "unsupported_next_action"
            projected_steps.append(step_payload)
            break

        simulated_backfill_preview = build_backfill_preview(
            ledger_path=ledger_path,
            current_streak=current_streak,
            current_last_loss_ts=simulated_last_loss_ts_raw,
        )
        manual_ack_eligible, manual_ack_reasons, cooldown_elapsed_hours = evaluate_manual_ack_status(
            now_utc=now_utc,
            streak=current_streak,
            stop_threshold=stop_threshold,
            last_loss_ts=parse_ts(simulated_last_loss_ts_raw),
            cooldown_required_hours=cooldown_required_hours,
            allow_missing_last_loss_ts=allow_missing_last_loss_ts,
        )
        simulated_recovery_plan = build_recovery_plan(
            system_root=SYSTEM_ROOT,
            manual_ack_eligible=manual_ack_eligible,
            manual_ack_reasons=manual_ack_reasons,
            ack_eligible_for_apply_now=simulated_ack_eligible,
            backfill_preview=simulated_backfill_preview,
        )
        step_payload["projected_last_loss_ts"] = simulated_last_loss_ts_raw
        step_payload["projected_ack_present"] = simulated_ack_present
        step_payload["projected_manual_ack_eligible"] = manual_ack_eligible
        step_payload["projected_manual_ack_reasons"] = manual_ack_reasons
        step_payload["projected_cooldown_elapsed_hours"] = (
            round(cooldown_elapsed_hours, 4) if cooldown_elapsed_hours is not None else None
        )
        step_payload["projected_next_action"] = simulated_recovery_plan.get("next_action")
        step_payload["projected_action_level"] = simulated_recovery_plan.get("action_level")
        projected_steps.append(step_payload)

        if step_payload["projected_next_action"] == next_action:
            break

    final_next_action = None
    final_action_level = None
    if terminal_action is None and isinstance(simulated_recovery_plan, dict):
        final_next_action = simulated_recovery_plan.get("next_action")
        final_action_level = simulated_recovery_plan.get("action_level")

    return {
        "writes_enabled_assumption": True,
        "fallback_write_enabled_assumption": True,
        "max_steps": max_steps,
        "projected_step_count": len(projected_steps),
        "write_chain_possible": len(projected_steps) > 0,
        "would_require_fallback_write": any(
            step.get("simulated_step") == "fallback_backfill_write" for step in projected_steps
        ),
        "would_progress": bool(projected_steps),
        "terminal_action": terminal_action,
        "final_next_action": final_next_action,
        "final_action_level": final_action_level,
        "projected_steps": projected_steps,
    }


def main() -> int:
    args = build_parser().parse_args()
    now_utc = parse_ts(args.now) or dt.datetime.now(dt.timezone.utc)
    state_path = Path(str(args.state_path)).expanduser().resolve()
    ack_path = Path(str(args.ack_path)).expanduser().resolve()
    checksum_path = Path(str(args.checksum_path)).expanduser().resolve()
    ledger_path = Path(str(args.ledger_path)).expanduser().resolve()
    archive_dir = Path(str(args.archive_dir)).expanduser().resolve()
    archive_manifest_path = Path(str(args.archive_manifest_path)).expanduser().resolve()
    cooldown_required_hours = max(0.0, float(args.cooldown_hours))
    stop_threshold = max(1, int(args.stop_threshold))

    state_raw = load_json(state_path)
    state_fingerprint = build_state_fingerprint(state_raw)
    streak = safe_int(state_raw.get("consecutive_losses"), 0)
    last_loss_ts_raw = str(state_raw.get("last_loss_ts") or "").strip() or None
    last_loss_ts = parse_ts(last_loss_ts_raw)
    manual_ack_eligible, manual_ack_reasons, cooldown_elapsed_hours = evaluate_manual_ack_status(
        now_utc=now_utc,
        streak=streak,
        stop_threshold=stop_threshold,
        last_loss_ts=last_loss_ts,
        cooldown_required_hours=cooldown_required_hours,
        allow_missing_last_loss_ts=bool(args.allow_missing_last_loss_ts),
    )

    ack_raw = load_json(ack_path)
    checksum_raw = load_json(checksum_path)
    ack_present = ack_path.exists() and bool(ack_raw)
    checksum_present = checksum_path.exists() and bool(checksum_raw)
    checksum_expected = str(checksum_raw.get("sha256") or "").strip()
    checksum_valid = None
    if ack_present and checksum_present and checksum_expected:
        checksum_valid = checksum_expected == sha256_file(ack_path)
    elif ack_present:
        checksum_valid = False

    ack_expires_at = parse_ts(ack_raw.get("expires_at"))
    ack_expired = None if ack_expires_at is None else now_utc > ack_expires_at
    ack_streak_snapshot = safe_int(ack_raw.get("streak_snapshot"), -1)
    ack_uses_remaining = safe_int(ack_raw.get("uses_remaining"), 0)
    ack_use_limit = safe_int(ack_raw.get("use_limit"), 0)
    ack_active = bool(ack_raw.get("active", False))
    ack_guardrail = str(ack_raw.get("guardrail") or "").strip() or None
    ack_allow_missing_last_loss_ts = bool(ack_raw.get("allow_missing_last_loss_ts", False))
    ack_cooldown_required_hours = safe_float(ack_raw.get("cooldown_hours_required"), cooldown_required_hours)
    ack_note = str(ack_raw.get("note") or "").strip() or None

    ack_apply_reasons: list[str] = []
    ack_streak_matches_current = ack_streak_snapshot == streak
    if not ack_present:
        ack_apply_reasons.append("ack_missing")
    else:
        if not checksum_valid:
            ack_apply_reasons.append("checksum_invalid")
        if ack_guardrail != "consecutive_loss_stop":
            ack_apply_reasons.append("guardrail_mismatch")
        if ack_expired is True:
            ack_apply_reasons.append("ack_expired")
        if not ack_active:
            ack_apply_reasons.append("ack_inactive")
        if ack_uses_remaining <= 0:
            ack_apply_reasons.append("uses_exhausted")
        if not ack_streak_matches_current:
            ack_apply_reasons.append("streak_snapshot_mismatch")
        if streak < stop_threshold:
            ack_apply_reasons.append("guardrail_not_hit")
        if last_loss_ts is not None:
            cooldown_elapsed_hours = max(0.0, (now_utc - last_loss_ts).total_seconds() / 3600.0)
            if cooldown_elapsed_hours < ack_cooldown_required_hours:
                ack_apply_reasons.append("cooldown_active")
        elif not ack_allow_missing_last_loss_ts:
            ack_apply_reasons.append("last_loss_ts_missing")

    backfill_preview = build_backfill_preview(
        ledger_path=ledger_path,
        current_streak=streak,
        current_last_loss_ts=last_loss_ts_raw,
    )
    recovery_plan = build_recovery_plan(
        system_root=SYSTEM_ROOT,
        manual_ack_eligible=manual_ack_eligible,
        manual_ack_reasons=manual_ack_reasons,
        ack_eligible_for_apply_now=len(ack_apply_reasons) == 0,
        backfill_preview=backfill_preview,
    )
    write_projection = build_write_projection(
        now_utc=now_utc,
        stop_threshold=stop_threshold,
        cooldown_required_hours=cooldown_required_hours,
        allow_missing_last_loss_ts=bool(args.allow_missing_last_loss_ts),
        ledger_path=ledger_path,
        current_streak=streak,
        current_last_loss_ts_raw=last_loss_ts_raw,
        ack_eligible_for_apply_now=len(ack_apply_reasons) == 0,
        ack_present=ack_present,
        backfill_preview=backfill_preview,
    )

    out = {
        "action": "paper_consecutive_loss_guardrail_status",
        "ok": True,
        "now_utc": fmt_utc(now_utc),
        "state_path": str(state_path),
        "state_fingerprint": state_fingerprint,
        "ack_path": str(ack_path),
        "checksum_path": str(checksum_path),
        "stop_threshold": stop_threshold,
        "current_streak": streak,
        "guardrail_hit": streak >= stop_threshold,
        "last_loss_ts": fmt_utc(last_loss_ts) if last_loss_ts is not None else None,
        "cooldown_hours_required": round(cooldown_required_hours, 4),
        "cooldown_elapsed_hours": round(cooldown_elapsed_hours, 4) if cooldown_elapsed_hours is not None else None,
        "allow_missing_last_loss_ts": bool(args.allow_missing_last_loss_ts),
        "manual_ack_eligible": manual_ack_eligible,
        "manual_ack_reasons": manual_ack_reasons,
        "backfill_preview": backfill_preview,
        "recovery_plan": recovery_plan,
        "write_projection": write_projection,
        "ack": {
            "present": ack_present,
            "checksum_present": checksum_present,
            "checksum_valid": checksum_valid,
            "guardrail": ack_guardrail,
            "active": ack_active if ack_present else None,
            "use_limit": ack_use_limit if ack_present else None,
            "uses_remaining": ack_uses_remaining if ack_present else None,
            "streak_snapshot": ack_streak_snapshot if ack_present else None,
            "streak_matches_current": ack_streak_matches_current if ack_present else None,
            "expires_at": fmt_utc(ack_expires_at) if ack_expires_at is not None else None,
            "expired": ack_expired,
            "cooldown_hours_required": round(ack_cooldown_required_hours, 4) if ack_present else None,
            "allow_missing_last_loss_ts": ack_allow_missing_last_loss_ts if ack_present else None,
            "eligible_for_apply_now": len(ack_apply_reasons) == 0,
            "apply_reasons": ack_apply_reasons,
            "note": ack_note,
        },
        "ack_archive": {
            "live_ack": inspect_live_ack(ack_path, checksum_path),
            "archive": inspect_archive(
                archive_dir,
                archive_manifest_path,
                manifest_tail=max(1, int(args.archive_manifest_tail)),
                archive_keep_files=max(1, int(args.archive_keep_files)),
            ),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
