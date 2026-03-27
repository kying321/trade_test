#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from operator_context_sections import (
    cross_market_operator_backlog_section_lines as _cross_market_operator_backlog_section_lines,
    cross_market_operator_head_section_lines as _cross_market_operator_head_section_lines,
    cross_market_remote_live_section_lines as _cross_market_remote_live_section_lines,
    cross_market_repair_section_lines as _cross_market_repair_section_lines,
    cross_market_review_section_lines as _cross_market_review_section_lines,
    cross_market_state_lanes_section_lines as _cross_market_state_lanes_section_lines,
    openclaw_orderflow_blueprint_section_lines as _openclaw_orderflow_blueprint_section_lines,
    remote_live_account_scope_section_lines as _remote_live_account_scope_section_lines,
    system_time_sync_repair_plan_section_lines as _system_time_sync_repair_plan_section_lines,
    system_time_sync_repair_verification_section_lines as _system_time_sync_repair_verification_section_lines,
)


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
NEXT_CONTEXT_PATH = DEFAULT_REVIEW_DIR / "NEXT_WINDOW_CONTEXT_LATEST.md"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
SHANGHAI_TZ = dt.timezone(dt.timedelta(hours=8))


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def step_now(base: dt.datetime, offset_seconds: int) -> dt.datetime:
    return base + dt.timedelta(seconds=int(offset_seconds))


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def latest_stamped_artifact(review_dir: Path, suffix: str) -> Path:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        raise FileNotFoundError(f"no_{suffix}")
    return max(candidates, key=lambda path: (artifact_stamp(path), path.stat().st_mtime, path.name))


def write_hot_brief_snapshot(
    review_dir: Path,
    *,
    stamp: str,
    brief_payload: dict[str, Any],
) -> Path:
    snapshot_path = review_dir / f"{stamp}_commodity_paper_execution_refresh_hot_brief_snapshot.json"
    source_path = Path(str(brief_payload.get("artifact") or "")).expanduser()
    if source_path.exists():
        snapshot_text = source_path.read_text(encoding="utf-8")
    else:
        snapshot_text = json.dumps(dict(brief_payload), ensure_ascii=False, indent=2) + "\n"
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    return snapshot_path


def latest_stamped_datetime(review_dir: Path, suffixes: list[str]) -> dt.datetime | None:
    latest_dt: dt.datetime | None = None
    for suffix in suffixes:
        for path in review_dir.glob(f"*_{suffix}.json"):
            stamp_dt = parsed_artifact_stamp(path)
            if stamp_dt is None:
                continue
            if latest_dt is None or stamp_dt > latest_dt:
                latest_dt = stamp_dt
    return latest_dt


def derive_runtime_now(review_dir: Path, requested_now: str | None) -> dt.datetime:
    if str(requested_now or "").strip():
        return parse_now(requested_now)
    runtime_now = now_utc()
    latest_dt = latest_stamped_datetime(
        review_dir,
        [
            "crypto_route_refresh",
            "remote_live_history_audit",
            "remote_live_handoff",
            "live_gate_blocker_report",
            "brooks_structure_refresh",
            "cross_market_operator_state",
            "commodity_paper_execution_queue",
            "commodity_paper_execution_bridge",
            "commodity_paper_execution_review",
            "commodity_paper_execution_retro",
            "commodity_paper_execution_gap_report",
            "hot_universe_operator_brief",
            "commodity_paper_execution_refresh",
        ],
    )
    if latest_dt is not None and latest_dt >= runtime_now:
        return latest_dt + dt.timedelta(seconds=1)
    return runtime_now


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime | None = None,
) -> tuple[list[str], list[str]]:
    effective_now = now_dt or now_utc()
    cutoff = effective_now - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda item: item.stat().st_mtime, reverse=True)

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def run_json_step(*, step_name: str, cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{step_name}_failed: {(proc.stderr or proc.stdout or '').strip() or f'returncode={proc.returncode}'}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{step_name}_invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{step_name}_invalid_payload")
    return payload


def script_path(name: str) -> Path:
    return SYSTEM_ROOT / "scripts" / name


def context_updated_line(runtime_now: dt.datetime) -> str:
    return runtime_now.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d Asia/Shanghai")


def bridge_apply_symbols(payload: dict[str, Any]) -> list[str]:
    applied_ids = {
        str(x).strip() for x in payload.get("applied_execution_ids", []) if str(x).strip()
    }
    symbols: list[str] = []
    for row in payload.get("bridge_items", []):
        if not isinstance(row, dict):
            continue
        execution_id = str(row.get("execution_id") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        if execution_id in applied_ids and symbol:
            symbols.append(symbol)
    return symbols


def _list_text(values: list[str], limit: int = 8) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _mapping_text(values: dict[str, Any], limit: int = 8) -> str:
    items = [
        f"{str(key).strip().upper()}:{str(value).strip()}"
        for key, value in sorted((values or {}).items())
        if str(key).strip() and str(value).strip()
    ]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _watch_items_text(items: list[dict[str, Any]], limit: int = 8) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        signal_date = str(row.get("signal_date") or "").strip()
        age = row.get("signal_age_days")
        if not symbol:
            continue
        detail = symbol
        if age not in (None, ""):
            detail += f":{age}d"
        if signal_date:
            detail += f"@{signal_date}"
        parts.append(detail)
    if not parts:
        return "-"
    if len(items) <= limit:
        return ", ".join(parts)
    return ", ".join(parts) + f" (+{len(items) - limit})"


def _action_checklist_lines(items: list[dict[str, Any]], limit: int = 6) -> list[str]:
    lines: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        rank = row.get("rank") or "-"
        state = str(row.get("state") or "").strip() or "-"
        symbol = str(row.get("symbol") or "").strip().upper() or "-"
        action = str(row.get("action") or "").strip() or "-"
        blocker = str(row.get("blocker_detail") or "").strip() or "-"
        done_when = str(row.get("done_when") or "").strip() or "-"
        lines.append(
            f"- {rank}. `{state}` `{symbol}` `{action}` blocker=`{blocker}` done_when=`{done_when}`"
        )
    return lines


def _fmt_num(raw: Any) -> str:
    try:
        value = float(raw)
    except Exception:
        text = str(raw or "").strip()
        return text or "-"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def render_context_markdown(
    *,
    runtime_now: dt.datetime,
    brief: dict[str, Any],
    review: dict[str, Any],
    retro: dict[str, Any],
    gap: dict[str, Any],
    bridge: dict[str, Any],
    bridge_apply: dict[str, Any] | None,
) -> str:
    stale_symbols = [
        str(x).strip().upper()
        for x in gap.get("queue_symbols_with_stale_directional_signal", [])
        if str(x).strip()
    ]
    stale_signal_dates = {
        str(key).strip().upper(): str(value).strip()
        for key, value in dict(
            brief.get("commodity_execution_bridge_stale_signal_dates")
            or gap.get("queue_symbols_with_stale_directional_signal_dates")
            or {}
        ).items()
        if str(key).strip() and str(value).strip()
    }
    stale_signal_age_days = {
        str(key).strip().upper(): int(value)
        for key, value in dict(
            brief.get("commodity_execution_bridge_stale_signal_age_days")
            or gap.get("queue_symbols_with_stale_directional_signal_age_days")
            or {}
        ).items()
        if str(key).strip() and str(value).strip()
    }
    stale_signal_watch_items = [
        dict(row)
        for row in (
            brief.get("commodity_stale_signal_watch_items")
            or gap.get("stale_directional_signal_watch_items")
            or []
        )
        if isinstance(row, dict)
    ]
    already_bridged_symbols = [
        str(x).strip().upper()
        for x in brief.get("commodity_execution_bridge_already_bridged_symbols", [])
        if str(x).strip()
    ]
    next_fill_symbol = str(
        brief.get("commodity_next_fill_evidence_execution_symbol")
        or retro.get("next_fill_evidence_execution_symbol")
        or review.get("next_fill_evidence_execution_symbol")
        or ""
    ).strip().upper()
    next_fill_target = str(
        brief.get("commodity_next_fill_evidence_execution_id")
        or retro.get("next_fill_evidence_execution_id")
        or review.get("next_fill_evidence_execution_id")
        or ""
    ).strip()
    next_review_close_evidence_symbol = str(
        brief.get("commodity_next_review_close_evidence_execution_symbol")
        or review.get("next_close_evidence_execution_symbol")
        or brief.get("commodity_next_close_evidence_execution_symbol")
        or retro.get("next_close_evidence_execution_symbol")
        or ""
    ).strip().upper()
    next_review_close_evidence_target = str(
        brief.get("commodity_next_review_close_evidence_execution_id")
        or review.get("next_close_evidence_execution_id")
        or brief.get("commodity_next_close_evidence_execution_id")
        or retro.get("next_close_evidence_execution_id")
        or ""
    ).strip()
    next_close_evidence_symbol = str(
        brief.get("commodity_next_close_evidence_execution_symbol")
        or retro.get("next_close_evidence_execution_symbol")
        or ""
    ).strip().upper()
    next_close_evidence_target = str(
        brief.get("commodity_next_close_evidence_execution_id")
        or retro.get("next_close_evidence_execution_id")
        or ""
    ).strip()
    close_evidence_pending_symbols = [
        str(x).strip().upper()
        for x in brief.get("commodity_close_evidence_pending_symbols", []) or retro.get("close_evidence_pending_symbols", [])
        if str(x).strip()
    ]
    commodity_remainder_focus_area = str(brief.get("commodity_remainder_focus_area") or "").strip()
    commodity_remainder_focus_target = str(brief.get("commodity_remainder_focus_target") or "").strip()
    commodity_remainder_focus_action = str(brief.get("commodity_remainder_focus_action") or "").strip()
    commodity_remainder_focus_signal_date = str(brief.get("commodity_remainder_focus_signal_date") or "").strip()
    commodity_remainder_focus_signal_age_days = str(brief.get("commodity_remainder_focus_signal_age_days") or "").strip()
    commodity_stale_signal_watch_next_execution_id = str(
        brief.get("commodity_stale_signal_watch_next_execution_id") or ""
    ).strip()
    commodity_stale_signal_watch_next_symbol = str(
        brief.get("commodity_stale_signal_watch_next_symbol") or ""
    ).strip().upper()
    commodity_stale_signal_watch_next_signal_date = str(
        brief.get("commodity_stale_signal_watch_next_signal_date") or ""
    ).strip()
    commodity_stale_signal_watch_next_signal_age_days = str(
        brief.get("commodity_stale_signal_watch_next_signal_age_days") or ""
    ).strip()
    followup_focus_area = str(brief.get("followup_focus_area") or "").strip()
    followup_focus_target = str(brief.get("followup_focus_target") or "").strip()
    followup_focus_action = str(brief.get("followup_focus_action") or "").strip()
    next_focus_state = str(brief.get("next_focus_state") or "").strip()
    next_focus_blocker_detail = str(brief.get("next_focus_blocker_detail") or "").strip()
    next_focus_done_when = str(brief.get("next_focus_done_when") or "").strip()
    followup_focus_state = str(brief.get("followup_focus_state") or "").strip()
    followup_focus_blocker_detail = str(brief.get("followup_focus_blocker_detail") or "").strip()
    followup_focus_done_when = str(brief.get("followup_focus_done_when") or "").strip()
    commodity_focus_lifecycle_status = str(brief.get("commodity_focus_lifecycle_status") or "").strip()
    commodity_focus_lifecycle_brief = str(brief.get("commodity_focus_lifecycle_brief") or "").strip()
    commodity_focus_lifecycle_blocker_detail = str(
        brief.get("commodity_focus_lifecycle_blocker_detail") or ""
    ).strip()
    commodity_focus_lifecycle_done_when = str(brief.get("commodity_focus_lifecycle_done_when") or "").strip()
    commodity_execution_close_evidence_status = str(
        brief.get("commodity_execution_close_evidence_status") or ""
    ).strip()
    commodity_execution_close_evidence_brief = str(
        brief.get("commodity_execution_close_evidence_brief") or ""
    ).strip()
    commodity_execution_close_evidence_target = str(
        brief.get("commodity_execution_close_evidence_target") or ""
    ).strip()
    commodity_execution_close_evidence_symbol = str(
        brief.get("commodity_execution_close_evidence_symbol") or ""
    ).strip().upper()
    commodity_execution_close_evidence_blocker_detail = str(
        brief.get("commodity_execution_close_evidence_blocker_detail") or ""
    ).strip()
    commodity_execution_close_evidence_done_when = str(
        brief.get("commodity_execution_close_evidence_done_when") or ""
    ).strip()
    operator_focus_slots = [dict(row) for row in brief.get("operator_focus_slots", []) if isinstance(row, dict)]
    operator_focus_slots_brief = str(brief.get("operator_focus_slots_brief") or "").strip()
    operator_focus_slot_sources_brief = str(brief.get("operator_focus_slot_sources_brief") or "").strip()
    operator_focus_slot_status_brief = str(brief.get("operator_focus_slot_status_brief") or "").strip()
    operator_focus_slot_recency_brief = str(brief.get("operator_focus_slot_recency_brief") or "").strip()
    operator_focus_slot_health_brief = str(brief.get("operator_focus_slot_health_brief") or "").strip()
    operator_focus_slot_refresh_backlog_brief = str(
        brief.get("operator_focus_slot_refresh_backlog_brief") or ""
    ).strip()
    operator_focus_slot_refresh_backlog = [
        dict(row) for row in brief.get("operator_focus_slot_refresh_backlog", []) if isinstance(row, dict)
    ]
    operator_focus_slot_refresh_backlog_count_raw = brief.get("operator_focus_slot_refresh_backlog_count")
    operator_focus_slot_refresh_backlog_count = (
        str(operator_focus_slot_refresh_backlog_count_raw).strip()
        if operator_focus_slot_refresh_backlog_count_raw not in (None, "")
        else ""
    )
    operator_focus_slot_ready_count_raw = brief.get("operator_focus_slot_ready_count")
    operator_focus_slot_ready_count = (
        str(operator_focus_slot_ready_count_raw).strip()
        if operator_focus_slot_ready_count_raw not in (None, "")
        else ""
    )
    operator_focus_slot_total_count_raw = brief.get("operator_focus_slot_total_count")
    operator_focus_slot_total_count = (
        str(operator_focus_slot_total_count_raw).strip()
        if operator_focus_slot_total_count_raw not in (None, "")
        else ""
    )
    operator_focus_slot_promotion_gate_brief = str(
        brief.get("operator_focus_slot_promotion_gate_brief") or ""
    ).strip()
    operator_focus_slot_promotion_gate_status = str(
        brief.get("operator_focus_slot_promotion_gate_status") or ""
    ).strip()
    operator_focus_slot_promotion_gate_blocker_detail = str(
        brief.get("operator_focus_slot_promotion_gate_blocker_detail") or ""
    ).strip()
    operator_focus_slot_promotion_gate_done_when = str(
        brief.get("operator_focus_slot_promotion_gate_done_when") or ""
    ).strip()
    operator_focus_slot_actionability_backlog_brief = str(
        brief.get("operator_focus_slot_actionability_backlog_brief") or ""
    ).strip()
    operator_focus_slot_actionability_backlog = [
        dict(row) for row in brief.get("operator_focus_slot_actionability_backlog", []) if isinstance(row, dict)
    ]
    operator_focus_slot_actionability_backlog_count_raw = brief.get(
        "operator_focus_slot_actionability_backlog_count"
    )
    operator_focus_slot_actionability_backlog_count = (
        str(operator_focus_slot_actionability_backlog_count_raw).strip()
        if operator_focus_slot_actionability_backlog_count_raw not in (None, "")
        else ""
    )
    operator_focus_slot_actionable_count_raw = brief.get("operator_focus_slot_actionable_count")
    operator_focus_slot_actionable_count = (
        str(operator_focus_slot_actionable_count_raw).strip()
        if operator_focus_slot_actionable_count_raw not in (None, "")
        else ""
    )
    operator_focus_slot_actionability_gate_brief = str(
        brief.get("operator_focus_slot_actionability_gate_brief") or ""
    ).strip()
    operator_focus_slot_actionability_gate_status = str(
        brief.get("operator_focus_slot_actionability_gate_status") or ""
    ).strip()
    operator_focus_slot_actionability_gate_blocker_detail = str(
        brief.get("operator_focus_slot_actionability_gate_blocker_detail") or ""
    ).strip()
    operator_focus_slot_actionability_gate_done_when = str(
        brief.get("operator_focus_slot_actionability_gate_done_when") or ""
    ).strip()
    operator_focus_slot_readiness_gate_ready_count_raw = brief.get("operator_focus_slot_readiness_gate_ready_count")
    operator_focus_slot_readiness_gate_ready_count = (
        str(operator_focus_slot_readiness_gate_ready_count_raw).strip()
        if operator_focus_slot_readiness_gate_ready_count_raw not in (None, "")
        else ""
    )
    operator_focus_slot_readiness_gate_brief = str(
        brief.get("operator_focus_slot_readiness_gate_brief") or ""
    ).strip()
    operator_focus_slot_readiness_gate_status = str(
        brief.get("operator_focus_slot_readiness_gate_status") or ""
    ).strip()
    operator_focus_slot_readiness_gate_blocking_gate = str(
        brief.get("operator_focus_slot_readiness_gate_blocking_gate") or ""
    ).strip()
    operator_focus_slot_readiness_gate_blocker_detail = str(
        brief.get("operator_focus_slot_readiness_gate_blocker_detail") or ""
    ).strip()
    operator_focus_slot_readiness_gate_done_when = str(
        brief.get("operator_focus_slot_readiness_gate_done_when") or ""
    ).strip()
    operator_research_embedding_quality_status = str(
        brief.get("operator_research_embedding_quality_status") or ""
    ).strip()
    operator_research_embedding_quality_brief = str(
        brief.get("operator_research_embedding_quality_brief") or ""
    ).strip()
    operator_research_embedding_quality_blocker_detail = str(
        brief.get("operator_research_embedding_quality_blocker_detail") or ""
    ).strip()
    operator_research_embedding_quality_done_when = str(
        brief.get("operator_research_embedding_quality_done_when") or ""
    ).strip()
    operator_research_embedding_active_batches = [
        str(x).strip()
        for x in brief.get("operator_research_embedding_active_batches", [])
        if str(x).strip()
    ]
    operator_research_embedding_avoid_batches = [
        str(x).strip()
        for x in brief.get("operator_research_embedding_avoid_batches", [])
        if str(x).strip()
    ]
    operator_research_embedding_zero_trade_deprioritized_batches = [
        str(x).strip()
        for x in brief.get("operator_research_embedding_zero_trade_deprioritized_batches", [])
        if str(x).strip()
    ]
    operator_crypto_route_alignment_status = str(brief.get("operator_crypto_route_alignment_status") or "").strip()
    operator_crypto_route_alignment_focus_area = str(
        brief.get("operator_crypto_route_alignment_focus_area") or ""
    ).strip()
    operator_crypto_route_alignment_focus_slot = str(
        brief.get("operator_crypto_route_alignment_focus_slot") or ""
    ).strip()
    operator_crypto_route_alignment_focus_symbol = str(
        brief.get("operator_crypto_route_alignment_focus_symbol") or ""
    ).strip().upper()
    operator_crypto_route_alignment_focus_action = str(
        brief.get("operator_crypto_route_alignment_focus_action") or ""
    ).strip()
    operator_crypto_route_alignment_brief = str(brief.get("operator_crypto_route_alignment_brief") or "").strip()
    operator_crypto_route_alignment_blocker_detail = str(
        brief.get("operator_crypto_route_alignment_blocker_detail") or ""
    ).strip()
    operator_crypto_route_alignment_done_when = str(
        brief.get("operator_crypto_route_alignment_done_when") or ""
    ).strip()
    operator_crypto_route_alignment_recovery_status = str(
        brief.get("operator_crypto_route_alignment_recovery_status") or ""
    ).strip()
    operator_crypto_route_alignment_recovery_brief = str(
        brief.get("operator_crypto_route_alignment_recovery_brief") or ""
    ).strip()
    operator_crypto_route_alignment_recovery_blocker_detail = str(
        brief.get("operator_crypto_route_alignment_recovery_blocker_detail") or ""
    ).strip()
    operator_crypto_route_alignment_recovery_done_when = str(
        brief.get("operator_crypto_route_alignment_recovery_done_when") or ""
    ).strip()
    operator_crypto_route_alignment_recovery_failed_batch_count_raw = brief.get(
        "operator_crypto_route_alignment_recovery_failed_batch_count"
    )
    operator_crypto_route_alignment_recovery_failed_batch_count = (
        str(operator_crypto_route_alignment_recovery_failed_batch_count_raw).strip()
        if operator_crypto_route_alignment_recovery_failed_batch_count_raw not in (None, "")
        else ""
    )
    operator_crypto_route_alignment_recovery_timed_out_batch_count_raw = brief.get(
        "operator_crypto_route_alignment_recovery_timed_out_batch_count"
    )
    operator_crypto_route_alignment_recovery_timed_out_batch_count = (
        str(operator_crypto_route_alignment_recovery_timed_out_batch_count_raw).strip()
        if operator_crypto_route_alignment_recovery_timed_out_batch_count_raw not in (None, "")
        else ""
    )
    operator_crypto_route_alignment_recovery_zero_trade_batches = [
        str(x).strip()
        for x in brief.get("operator_crypto_route_alignment_recovery_zero_trade_batches", [])
        if str(x).strip()
    ]
    operator_crypto_route_alignment_cooldown_status = str(
        brief.get("operator_crypto_route_alignment_cooldown_status") or ""
    ).strip()
    operator_crypto_route_alignment_cooldown_brief = str(
        brief.get("operator_crypto_route_alignment_cooldown_brief") or ""
    ).strip()
    operator_crypto_route_alignment_cooldown_blocker_detail = str(
        brief.get("operator_crypto_route_alignment_cooldown_blocker_detail") or ""
    ).strip()
    operator_crypto_route_alignment_cooldown_done_when = str(
        brief.get("operator_crypto_route_alignment_cooldown_done_when") or ""
    ).strip()
    operator_crypto_route_alignment_cooldown_last_research_end_date = str(
        brief.get("operator_crypto_route_alignment_cooldown_last_research_end_date") or ""
    ).strip()
    operator_crypto_route_alignment_cooldown_next_eligible_end_date = str(
        brief.get("operator_crypto_route_alignment_cooldown_next_eligible_end_date") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_status = str(
        brief.get("operator_crypto_route_alignment_recipe_status") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_brief = str(
        brief.get("operator_crypto_route_alignment_recipe_brief") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_blocker_detail = str(
        brief.get("operator_crypto_route_alignment_recipe_blocker_detail") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_done_when = str(
        brief.get("operator_crypto_route_alignment_recipe_done_when") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_ready_on_date = str(
        brief.get("operator_crypto_route_alignment_recipe_ready_on_date") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_script = str(
        brief.get("operator_crypto_route_alignment_recipe_script") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_command_hint = str(
        brief.get("operator_crypto_route_alignment_recipe_command_hint") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_expected_status = str(
        brief.get("operator_crypto_route_alignment_recipe_expected_status") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_note = str(
        brief.get("operator_crypto_route_alignment_recipe_note") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_followup_script = str(
        brief.get("operator_crypto_route_alignment_recipe_followup_script") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_followup_command_hint = str(
        brief.get("operator_crypto_route_alignment_recipe_followup_command_hint") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_verify_hint = str(
        brief.get("operator_crypto_route_alignment_recipe_verify_hint") or ""
    ).strip()
    operator_crypto_route_alignment_recipe_window_days_raw = brief.get(
        "operator_crypto_route_alignment_recipe_window_days"
    )
    operator_crypto_route_alignment_recipe_window_days = (
        str(operator_crypto_route_alignment_recipe_window_days_raw).strip()
        if operator_crypto_route_alignment_recipe_window_days_raw not in (None, "")
        else ""
    )
    operator_crypto_route_alignment_recipe_target_batches = [
        str(x).strip()
        for x in brief.get("operator_crypto_route_alignment_recipe_target_batches", [])
        if str(x).strip()
    ]
    operator_source_refresh_queue_brief = str(brief.get("operator_source_refresh_queue_brief") or "").strip()
    operator_source_refresh_queue = [
        dict(row) for row in brief.get("operator_source_refresh_queue", []) if isinstance(row, dict)
    ]
    operator_source_refresh_queue_count_raw = brief.get("operator_source_refresh_queue_count")
    operator_source_refresh_queue_count = (
        str(operator_source_refresh_queue_count_raw).strip()
        if operator_source_refresh_queue_count_raw not in (None, "")
        else ""
    )
    operator_source_refresh_checklist_brief = str(
        brief.get("operator_source_refresh_checklist_brief") or ""
    ).strip()
    operator_source_refresh_checklist = [
        dict(row) for row in brief.get("operator_source_refresh_checklist", []) if isinstance(row, dict)
    ]
    operator_source_refresh_pipeline_steps_brief = str(
        brief.get("operator_source_refresh_pipeline_steps_brief") or ""
    ).strip()
    operator_source_refresh_pipeline_step_checkpoint_brief = str(
        brief.get("operator_source_refresh_pipeline_step_checkpoint_brief") or ""
    ).strip()
    operator_source_refresh_pipeline_pending_brief = str(
        brief.get("operator_source_refresh_pipeline_pending_brief") or ""
    ).strip()
    operator_source_refresh_pipeline_pending_count_raw = brief.get("operator_source_refresh_pipeline_pending_count")
    operator_source_refresh_pipeline_pending_count = (
        str(operator_source_refresh_pipeline_pending_count_raw).strip()
        if operator_source_refresh_pipeline_pending_count_raw not in (None, "")
        else ""
    )
    operator_source_refresh_pipeline_head_rank = str(
        brief.get("operator_source_refresh_pipeline_head_rank") or ""
    ).strip()
    operator_source_refresh_pipeline_head_name = str(
        brief.get("operator_source_refresh_pipeline_head_name") or ""
    ).strip()
    operator_source_refresh_pipeline_head_checkpoint_state = str(
        brief.get("operator_source_refresh_pipeline_head_checkpoint_state") or ""
    ).strip()
    operator_source_refresh_pipeline_head_expected_artifact_kind = str(
        brief.get("operator_source_refresh_pipeline_head_expected_artifact_kind") or ""
    ).strip()
    operator_source_refresh_pipeline_head_current_artifact = str(
        brief.get("operator_source_refresh_pipeline_head_current_artifact") or ""
    ).strip()
    operator_source_refresh_pipeline_relevance_status = str(
        brief.get("operator_source_refresh_pipeline_relevance_status") or ""
    ).strip()
    operator_source_refresh_pipeline_relevance_brief = str(
        brief.get("operator_source_refresh_pipeline_relevance_brief") or ""
    ).strip()
    operator_source_refresh_pipeline_relevance_blocker_detail = str(
        brief.get("operator_source_refresh_pipeline_relevance_blocker_detail") or ""
    ).strip()
    operator_source_refresh_pipeline_relevance_done_when = str(
        brief.get("operator_source_refresh_pipeline_relevance_done_when") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_brief = str(
        brief.get("operator_source_refresh_pipeline_deferred_brief") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_count_raw = brief.get("operator_source_refresh_pipeline_deferred_count")
    operator_source_refresh_pipeline_deferred_count = (
        str(operator_source_refresh_pipeline_deferred_count_raw).strip()
        if operator_source_refresh_pipeline_deferred_count_raw not in (None, "")
        else ""
    )
    operator_source_refresh_pipeline_deferred_status = str(
        brief.get("operator_source_refresh_pipeline_deferred_status") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_until = str(
        brief.get("operator_source_refresh_pipeline_deferred_until") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_reason = str(
        brief.get("operator_source_refresh_pipeline_deferred_reason") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_head_rank = str(
        brief.get("operator_source_refresh_pipeline_deferred_head_rank") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_head_name = str(
        brief.get("operator_source_refresh_pipeline_deferred_head_name") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_head_checkpoint_state = str(
        brief.get("operator_source_refresh_pipeline_deferred_head_checkpoint_state") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_head_expected_artifact_kind = str(
        brief.get("operator_source_refresh_pipeline_deferred_head_expected_artifact_kind") or ""
    ).strip()
    operator_source_refresh_pipeline_deferred_head_current_artifact = str(
        brief.get("operator_source_refresh_pipeline_deferred_head_current_artifact") or ""
    ).strip()
    next_focus_source_kind = str(brief.get("next_focus_source_kind") or "").strip()
    next_focus_source_artifact = str(brief.get("next_focus_source_artifact") or "").strip()
    next_focus_source_status = str(brief.get("next_focus_source_status") or "").strip()
    next_focus_source_as_of = str(brief.get("next_focus_source_as_of") or "").strip()
    next_focus_source_age_minutes_raw = brief.get("next_focus_source_age_minutes")
    next_focus_source_age_minutes = (
        str(next_focus_source_age_minutes_raw).strip()
        if next_focus_source_age_minutes_raw not in (None, "")
        else ""
    )
    next_focus_source_recency = str(brief.get("next_focus_source_recency") or "").strip()
    next_focus_source_health = str(brief.get("next_focus_source_health") or "").strip()
    next_focus_source_refresh_action = str(brief.get("next_focus_source_refresh_action") or "").strip()
    followup_focus_source_kind = str(brief.get("followup_focus_source_kind") or "").strip()
    followup_focus_source_artifact = str(brief.get("followup_focus_source_artifact") or "").strip()
    followup_focus_source_status = str(brief.get("followup_focus_source_status") or "").strip()
    followup_focus_source_as_of = str(brief.get("followup_focus_source_as_of") or "").strip()
    followup_focus_source_age_minutes_raw = brief.get("followup_focus_source_age_minutes")
    followup_focus_source_age_minutes = (
        str(followup_focus_source_age_minutes_raw).strip()
        if followup_focus_source_age_minutes_raw not in (None, "")
        else ""
    )
    followup_focus_source_recency = str(brief.get("followup_focus_source_recency") or "").strip()
    followup_focus_source_health = str(brief.get("followup_focus_source_health") or "").strip()
    followup_focus_source_refresh_action = str(brief.get("followup_focus_source_refresh_action") or "").strip()
    secondary_focus_source_kind = str(brief.get("secondary_focus_source_kind") or "").strip()
    secondary_focus_source_artifact = str(brief.get("secondary_focus_source_artifact") or "").strip()
    secondary_focus_source_status = str(brief.get("secondary_focus_source_status") or "").strip()
    secondary_focus_source_as_of = str(brief.get("secondary_focus_source_as_of") or "").strip()
    secondary_focus_source_age_minutes_raw = brief.get("secondary_focus_source_age_minutes")
    secondary_focus_source_age_minutes = (
        str(secondary_focus_source_age_minutes_raw).strip()
        if secondary_focus_source_age_minutes_raw not in (None, "")
        else ""
    )
    secondary_focus_source_recency = str(brief.get("secondary_focus_source_recency") or "").strip()
    secondary_focus_source_health = str(brief.get("secondary_focus_source_health") or "").strip()
    secondary_focus_source_refresh_action = str(brief.get("secondary_focus_source_refresh_action") or "").strip()
    operator_focus_slot_refresh_head_slot = str(brief.get("operator_focus_slot_refresh_head_slot") or "").strip()
    operator_focus_slot_refresh_head_symbol = str(brief.get("operator_focus_slot_refresh_head_symbol") or "").strip()
    operator_focus_slot_refresh_head_action = str(brief.get("operator_focus_slot_refresh_head_action") or "").strip()
    operator_focus_slot_refresh_head_health = str(brief.get("operator_focus_slot_refresh_head_health") or "").strip()
    operator_source_refresh_next_slot = str(brief.get("operator_source_refresh_next_slot") or "").strip()
    operator_source_refresh_next_symbol = str(brief.get("operator_source_refresh_next_symbol") or "").strip()
    operator_source_refresh_next_action = str(brief.get("operator_source_refresh_next_action") or "").strip()
    operator_source_refresh_next_source_kind = str(
        brief.get("operator_source_refresh_next_source_kind") or ""
    ).strip()
    operator_source_refresh_next_source_health = str(
        brief.get("operator_source_refresh_next_source_health") or ""
    ).strip()
    operator_source_refresh_next_source_artifact = str(
        brief.get("operator_source_refresh_next_source_artifact") or ""
    ).strip()
    operator_source_refresh_next_state = str(brief.get("operator_source_refresh_next_state") or "").strip()
    operator_source_refresh_next_blocker_detail = str(
        brief.get("operator_source_refresh_next_blocker_detail") or ""
    ).strip()
    operator_source_refresh_next_done_when = str(
        brief.get("operator_source_refresh_next_done_when") or ""
    ).strip()
    operator_source_refresh_next_recipe_script = str(
        brief.get("operator_source_refresh_next_recipe_script") or ""
    ).strip()
    operator_source_refresh_next_recipe_command_hint = str(
        brief.get("operator_source_refresh_next_recipe_command_hint") or ""
    ).strip()
    operator_source_refresh_next_recipe_expected_status = str(
        brief.get("operator_source_refresh_next_recipe_expected_status") or ""
    ).strip()
    operator_source_refresh_next_recipe_expected_artifact_kind = str(
        brief.get("operator_source_refresh_next_recipe_expected_artifact_kind") or ""
    ).strip()
    operator_source_refresh_next_recipe_expected_artifact_path_hint = str(
        brief.get("operator_source_refresh_next_recipe_expected_artifact_path_hint") or ""
    ).strip()
    operator_source_refresh_next_recipe_note = str(
        brief.get("operator_source_refresh_next_recipe_note") or ""
    ).strip()
    operator_source_refresh_next_recipe_followup_script = str(
        brief.get("operator_source_refresh_next_recipe_followup_script") or ""
    ).strip()
    operator_source_refresh_next_recipe_followup_command_hint = str(
        brief.get("operator_source_refresh_next_recipe_followup_command_hint") or ""
    ).strip()
    operator_source_refresh_next_recipe_verify_hint = str(
        brief.get("operator_source_refresh_next_recipe_verify_hint") or ""
    ).strip()
    operator_source_refresh_next_recipe_steps_brief = str(
        brief.get("operator_source_refresh_next_recipe_steps_brief") or ""
    ).strip()
    operator_source_refresh_next_recipe_step_checkpoint_brief = str(
        brief.get("operator_source_refresh_next_recipe_step_checkpoint_brief") or ""
    ).strip()
    operator_source_refresh_next_recipe_steps = [
        dict(row) for row in brief.get("operator_source_refresh_next_recipe_steps", []) if isinstance(row, dict)
    ]
    crypto_route_head_source_refresh_status = str(
        brief.get("crypto_route_head_source_refresh_status") or ""
    ).strip()
    crypto_route_head_source_refresh_brief = str(
        brief.get("crypto_route_head_source_refresh_brief") or ""
    ).strip()
    crypto_route_head_source_refresh_slot = str(
        brief.get("crypto_route_head_source_refresh_slot") or ""
    ).strip()
    crypto_route_head_source_refresh_symbol = str(
        brief.get("crypto_route_head_source_refresh_symbol") or ""
    ).strip().upper()
    crypto_route_head_source_refresh_action = str(
        brief.get("crypto_route_head_source_refresh_action") or ""
    ).strip()
    crypto_route_head_source_refresh_source_kind = str(
        brief.get("crypto_route_head_source_refresh_source_kind") or ""
    ).strip()
    crypto_route_head_source_refresh_source_health = str(
        brief.get("crypto_route_head_source_refresh_source_health") or ""
    ).strip()
    crypto_route_head_source_refresh_source_artifact = str(
        brief.get("crypto_route_head_source_refresh_source_artifact") or ""
    ).strip()
    crypto_route_head_source_refresh_blocker_detail = str(
        brief.get("crypto_route_head_source_refresh_blocker_detail") or ""
    ).strip()
    crypto_route_head_source_refresh_done_when = str(
        brief.get("crypto_route_head_source_refresh_done_when") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_script = str(
        brief.get("crypto_route_head_source_refresh_recipe_script") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_command_hint = str(
        brief.get("crypto_route_head_source_refresh_recipe_command_hint") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_expected_status = str(
        brief.get("crypto_route_head_source_refresh_recipe_expected_status") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_expected_artifact_kind = str(
        brief.get("crypto_route_head_source_refresh_recipe_expected_artifact_kind") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_expected_artifact_path_hint = str(
        brief.get("crypto_route_head_source_refresh_recipe_expected_artifact_path_hint") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_note = str(
        brief.get("crypto_route_head_source_refresh_recipe_note") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_followup_script = str(
        brief.get("crypto_route_head_source_refresh_recipe_followup_script") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_followup_command_hint = str(
        brief.get("crypto_route_head_source_refresh_recipe_followup_command_hint") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_verify_hint = str(
        brief.get("crypto_route_head_source_refresh_recipe_verify_hint") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_steps_brief = str(
        brief.get("crypto_route_head_source_refresh_recipe_steps_brief") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_step_checkpoint_brief = str(
        brief.get("crypto_route_head_source_refresh_recipe_step_checkpoint_brief") or ""
    ).strip()
    crypto_route_head_source_refresh_recipe_steps = [
        dict(row)
        for row in brief.get("crypto_route_head_source_refresh_recipe_steps", [])
        if isinstance(row, dict)
    ]
    source_crypto_route_refresh_artifact = str(
        brief.get("source_crypto_route_refresh_artifact") or ""
    ).strip()
    source_crypto_route_refresh_status = str(
        brief.get("source_crypto_route_refresh_status") or ""
    ).strip()
    source_crypto_route_refresh_as_of = str(
        brief.get("source_crypto_route_refresh_as_of") or ""
    ).strip()
    source_crypto_route_refresh_native_mode = str(
        brief.get("source_crypto_route_refresh_native_mode") or ""
    ).strip()
    source_crypto_route_refresh_native_step_count_raw = brief.get(
        "source_crypto_route_refresh_native_step_count"
    )
    source_crypto_route_refresh_native_step_count = (
        str(source_crypto_route_refresh_native_step_count_raw).strip()
        if source_crypto_route_refresh_native_step_count_raw not in (None, "")
        else ""
    )
    source_crypto_route_refresh_reused_native_count_raw = brief.get(
        "source_crypto_route_refresh_reused_native_count"
    )
    source_crypto_route_refresh_reused_native_count = (
        str(source_crypto_route_refresh_reused_native_count_raw).strip()
        if source_crypto_route_refresh_reused_native_count_raw not in (None, "")
        else ""
    )
    source_crypto_route_refresh_missing_reused_count_raw = brief.get(
        "source_crypto_route_refresh_missing_reused_count"
    )
    source_crypto_route_refresh_missing_reused_count = (
        str(source_crypto_route_refresh_missing_reused_count_raw).strip()
        if source_crypto_route_refresh_missing_reused_count_raw not in (None, "")
        else ""
    )
    source_crypto_route_refresh_reuse_status = str(
        brief.get("source_crypto_route_refresh_reuse_status") or ""
    ).strip()
    source_crypto_route_refresh_reuse_brief = str(
        brief.get("source_crypto_route_refresh_reuse_brief") or ""
    ).strip()
    source_crypto_route_refresh_reuse_note = str(
        brief.get("source_crypto_route_refresh_reuse_note") or ""
    ).strip()
    source_crypto_route_refresh_reuse_done_when = str(
        brief.get("source_crypto_route_refresh_reuse_done_when") or ""
    ).strip()
    source_crypto_route_refresh_reuse_level = str(
        brief.get("source_crypto_route_refresh_reuse_level") or ""
    ).strip()
    source_crypto_route_refresh_reuse_gate_status = str(
        brief.get("source_crypto_route_refresh_reuse_gate_status") or ""
    ).strip()
    source_crypto_route_refresh_reuse_gate_brief = str(
        brief.get("source_crypto_route_refresh_reuse_gate_brief") or ""
    ).strip()
    source_crypto_route_refresh_reuse_gate_blocker_detail = str(
        brief.get("source_crypto_route_refresh_reuse_gate_blocker_detail") or ""
    ).strip()
    source_crypto_route_refresh_reuse_gate_done_when = str(
        brief.get("source_crypto_route_refresh_reuse_gate_done_when") or ""
    ).strip()
    source_crypto_route_refresh_reuse_gate_blocking_raw = brief.get(
        "source_crypto_route_refresh_reuse_gate_blocking"
    )
    if isinstance(source_crypto_route_refresh_reuse_gate_blocking_raw, bool):
        source_crypto_route_refresh_reuse_gate_blocking = (
            "true" if source_crypto_route_refresh_reuse_gate_blocking_raw else "false"
        )
    elif source_crypto_route_refresh_reuse_gate_blocking_raw not in (None, ""):
        source_crypto_route_refresh_reuse_gate_blocking = str(
            source_crypto_route_refresh_reuse_gate_blocking_raw
        ).strip()
    else:
        source_crypto_route_refresh_reuse_gate_blocking = ""
    source_brooks_route_report_artifact = str(
        brief.get("source_brooks_route_report_artifact") or ""
    ).strip()
    source_brooks_route_report_status = str(
        brief.get("source_brooks_route_report_status") or ""
    ).strip()
    source_brooks_route_report_as_of = str(
        brief.get("source_brooks_route_report_as_of") or ""
    ).strip()
    source_brooks_route_report_selected_routes_brief = str(
        brief.get("source_brooks_route_report_selected_routes_brief") or ""
    ).strip()
    source_brooks_route_report_candidate_count_raw = brief.get(
        "source_brooks_route_report_candidate_count"
    )
    source_brooks_route_report_candidate_count = (
        str(source_brooks_route_report_candidate_count_raw).strip()
        if source_brooks_route_report_candidate_count_raw not in (None, "")
        else ""
    )
    source_brooks_route_report_head_symbol = str(
        brief.get("source_brooks_route_report_head_symbol") or ""
    ).strip()
    source_brooks_route_report_head_strategy_id = str(
        brief.get("source_brooks_route_report_head_strategy_id") or ""
    ).strip()
    source_brooks_route_report_head_direction = str(
        brief.get("source_brooks_route_report_head_direction") or ""
    ).strip()
    source_brooks_route_report_head_bridge_status = str(
        brief.get("source_brooks_route_report_head_bridge_status") or ""
    ).strip()
    source_brooks_route_report_head_blocker_detail = str(
        brief.get("source_brooks_route_report_head_blocker_detail") or ""
    ).strip()
    source_brooks_execution_plan_artifact = str(
        brief.get("source_brooks_execution_plan_artifact") or ""
    ).strip()
    source_brooks_execution_plan_status = str(
        brief.get("source_brooks_execution_plan_status") or ""
    ).strip()
    source_brooks_execution_plan_as_of = str(
        brief.get("source_brooks_execution_plan_as_of") or ""
    ).strip()
    source_brooks_execution_plan_actionable_count_raw = brief.get(
        "source_brooks_execution_plan_actionable_count"
    )
    source_brooks_execution_plan_actionable_count = (
        str(source_brooks_execution_plan_actionable_count_raw).strip()
        if source_brooks_execution_plan_actionable_count_raw not in (None, "")
        else ""
    )
    source_brooks_execution_plan_blocked_count_raw = brief.get(
        "source_brooks_execution_plan_blocked_count"
    )
    source_brooks_execution_plan_blocked_count = (
        str(source_brooks_execution_plan_blocked_count_raw).strip()
        if source_brooks_execution_plan_blocked_count_raw not in (None, "")
        else ""
    )
    source_brooks_execution_plan_head_symbol = str(
        brief.get("source_brooks_execution_plan_head_symbol") or ""
    ).strip()
    source_brooks_execution_plan_head_strategy_id = str(
        brief.get("source_brooks_execution_plan_head_strategy_id") or ""
    ).strip()
    source_brooks_execution_plan_head_plan_status = str(
        brief.get("source_brooks_execution_plan_head_plan_status") or ""
    ).strip()
    source_brooks_execution_plan_head_execution_action = str(
        brief.get("source_brooks_execution_plan_head_execution_action") or ""
    ).strip()
    source_brooks_execution_plan_head_entry_price_raw = brief.get(
        "source_brooks_execution_plan_head_entry_price"
    )
    source_brooks_execution_plan_head_entry_price = (
        _fmt_num(source_brooks_execution_plan_head_entry_price_raw)
        if source_brooks_execution_plan_head_entry_price_raw not in (None, "")
        else ""
    )
    source_brooks_execution_plan_head_stop_price_raw = brief.get(
        "source_brooks_execution_plan_head_stop_price"
    )
    source_brooks_execution_plan_head_stop_price = (
        _fmt_num(source_brooks_execution_plan_head_stop_price_raw)
        if source_brooks_execution_plan_head_stop_price_raw not in (None, "")
        else ""
    )
    source_brooks_execution_plan_head_target_price_raw = brief.get(
        "source_brooks_execution_plan_head_target_price"
    )
    source_brooks_execution_plan_head_target_price = (
        _fmt_num(source_brooks_execution_plan_head_target_price_raw)
        if source_brooks_execution_plan_head_target_price_raw not in (None, "")
        else ""
    )
    source_brooks_execution_plan_head_rr_ratio_raw = brief.get(
        "source_brooks_execution_plan_head_rr_ratio"
    )
    source_brooks_execution_plan_head_rr_ratio = (
        _fmt_num(source_brooks_execution_plan_head_rr_ratio_raw)
        if source_brooks_execution_plan_head_rr_ratio_raw not in (None, "")
        else ""
    )
    source_brooks_execution_plan_head_blocker_detail = str(
        brief.get("source_brooks_execution_plan_head_blocker_detail") or ""
    ).strip()
    source_brooks_structure_review_queue_artifact = str(
        brief.get("source_brooks_structure_review_queue_artifact") or ""
    ).strip()
    source_brooks_structure_review_queue_status = str(
        brief.get("source_brooks_structure_review_queue_status") or ""
    ).strip()
    source_brooks_structure_review_queue_as_of = str(
        brief.get("source_brooks_structure_review_queue_as_of") or ""
    ).strip()
    source_brooks_structure_review_queue_brief = str(
        brief.get("source_brooks_structure_review_queue_brief") or ""
    ).strip()
    source_brooks_structure_refresh_artifact = str(
        brief.get("source_brooks_structure_refresh_artifact") or ""
    ).strip()
    source_brooks_structure_refresh_status = str(
        brief.get("source_brooks_structure_refresh_status") or ""
    ).strip()
    source_brooks_structure_refresh_as_of = str(
        brief.get("source_brooks_structure_refresh_as_of") or ""
    ).strip()
    source_brooks_structure_refresh_brief = str(
        brief.get("source_brooks_structure_refresh_brief") or ""
    ).strip()
    source_brooks_structure_refresh_queue_count_raw = brief.get(
        "source_brooks_structure_refresh_queue_count"
    )
    source_brooks_structure_refresh_queue_count = (
        str(source_brooks_structure_refresh_queue_count_raw).strip()
        if source_brooks_structure_refresh_queue_count_raw not in (None, "")
        else ""
    )
    source_brooks_structure_refresh_head_symbol = str(
        brief.get("source_brooks_structure_refresh_head_symbol") or ""
    ).strip()
    source_brooks_structure_refresh_head_action = str(
        brief.get("source_brooks_structure_refresh_head_action") or ""
    ).strip()
    source_brooks_structure_refresh_head_priority_score_raw = brief.get(
        "source_brooks_structure_refresh_head_priority_score"
    )
    source_brooks_structure_refresh_head_priority_score = (
        str(source_brooks_structure_refresh_head_priority_score_raw).strip()
        if source_brooks_structure_refresh_head_priority_score_raw not in (None, "")
        else ""
    )
    source_cross_market_operator_state_artifact = str(
        brief.get("source_cross_market_operator_state_artifact") or ""
    ).strip()
    source_cross_market_operator_state_status = str(
        brief.get("source_cross_market_operator_state_status") or ""
    ).strip()
    source_cross_market_operator_state_as_of = str(
        brief.get("source_cross_market_operator_state_as_of") or ""
    ).strip()
    source_cross_market_operator_state_snapshot_brief = str(
        brief.get("source_cross_market_operator_state_snapshot_brief") or ""
    ).strip()
    source_cross_market_operator_state_operator_snapshot_brief = str(
        brief.get("source_cross_market_operator_state_operator_snapshot_brief") or ""
    ).strip()
    source_cross_market_operator_state_review_snapshot_brief = str(
        brief.get("source_cross_market_operator_state_review_snapshot_brief") or ""
    ).strip()
    source_cross_market_operator_state_remote_live_snapshot_brief = str(
        brief.get("source_cross_market_operator_state_remote_live_snapshot_brief") or ""
    ).strip()
    source_cross_market_operator_state_operator_backlog_status = str(
        brief.get("source_cross_market_operator_state_operator_backlog_status") or ""
    ).strip()
    source_cross_market_operator_state_operator_backlog_count_raw = brief.get(
        "source_cross_market_operator_state_operator_backlog_count"
    )
    source_cross_market_operator_state_operator_backlog_count = (
        str(source_cross_market_operator_state_operator_backlog_count_raw).strip()
        if source_cross_market_operator_state_operator_backlog_count_raw not in (None, "")
        else ""
    )
    source_cross_market_operator_state_operator_backlog_brief = str(
        brief.get("source_cross_market_operator_state_operator_backlog_brief") or ""
    ).strip()
    source_cross_market_operator_state_operator_backlog_state_brief = str(
        brief.get("source_cross_market_operator_state_operator_backlog_state_brief") or ""
    ).strip()
    source_cross_market_operator_state_operator_backlog_priority_totals_brief = str(
        brief.get("source_cross_market_operator_state_operator_backlog_priority_totals_brief") or ""
    ).strip()
    source_cross_market_operator_state_operator_head_area = str(
        brief.get("source_cross_market_operator_state_operator_head_area") or ""
    ).strip()
    source_cross_market_operator_state_operator_head_symbol = str(
        brief.get("source_cross_market_operator_state_operator_head_symbol") or ""
    ).strip().upper()
    source_cross_market_operator_state_operator_head_action = str(
        brief.get("source_cross_market_operator_state_operator_head_action") or ""
    ).strip()
    source_cross_market_operator_state_operator_head_state = str(
        brief.get("source_cross_market_operator_state_operator_head_state") or ""
    ).strip()
    source_cross_market_operator_state_operator_head_priority_score_raw = brief.get(
        "source_cross_market_operator_state_operator_head_priority_score"
    )
    source_cross_market_operator_state_operator_head_priority_score = (
        str(source_cross_market_operator_state_operator_head_priority_score_raw).strip()
        if source_cross_market_operator_state_operator_head_priority_score_raw not in (None, "")
        else ""
    )
    source_cross_market_operator_state_operator_head_priority_tier = str(
        brief.get("source_cross_market_operator_state_operator_head_priority_tier") or ""
    ).strip()
    source_cross_market_operator_state_review_backlog_status = str(
        brief.get("source_cross_market_operator_state_review_backlog_status") or ""
    ).strip()
    source_cross_market_operator_state_review_backlog_count_raw = brief.get(
        "source_cross_market_operator_state_review_backlog_count"
    )
    source_cross_market_operator_state_review_backlog_count = (
        str(source_cross_market_operator_state_review_backlog_count_raw).strip()
        if source_cross_market_operator_state_review_backlog_count_raw not in (None, "")
        else ""
    )
    source_cross_market_operator_state_review_backlog_brief = str(
        brief.get("source_cross_market_operator_state_review_backlog_brief") or ""
    ).strip()
    source_cross_market_operator_state_review_head_area = str(
        brief.get("source_cross_market_operator_state_review_head_area") or ""
    ).strip()
    source_cross_market_operator_state_review_head_symbol = str(
        brief.get("source_cross_market_operator_state_review_head_symbol") or ""
    ).strip()
    source_cross_market_operator_state_review_head_action = str(
        brief.get("source_cross_market_operator_state_review_head_action") or ""
    ).strip()
    source_cross_market_operator_state_review_head_priority_score_raw = brief.get(
        "source_cross_market_operator_state_review_head_priority_score"
    )
    source_cross_market_operator_state_review_head_priority_score = (
        str(source_cross_market_operator_state_review_head_priority_score_raw).strip()
        if source_cross_market_operator_state_review_head_priority_score_raw not in (None, "")
        else ""
    )
    source_cross_market_operator_state_review_head_priority_tier = str(
        brief.get("source_cross_market_operator_state_review_head_priority_tier") or ""
    ).strip()
    cross_market_review_head_status = str(brief.get("cross_market_review_head_status") or "").strip()
    cross_market_review_head_brief = str(brief.get("cross_market_review_head_brief") or "").strip()
    cross_market_review_head_area = str(brief.get("cross_market_review_head_area") or "").strip()
    cross_market_review_head_symbol = str(brief.get("cross_market_review_head_symbol") or "").strip().upper()
    cross_market_review_head_action = str(brief.get("cross_market_review_head_action") or "").strip()
    cross_market_review_head_priority_score_raw = brief.get("cross_market_review_head_priority_score")
    cross_market_review_head_priority_score = (
        str(cross_market_review_head_priority_score_raw).strip()
        if cross_market_review_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_review_head_priority_tier = str(
        brief.get("cross_market_review_head_priority_tier") or ""
    ).strip()
    cross_market_review_head_blocker_detail = str(
        brief.get("cross_market_review_head_blocker_detail") or ""
    ).strip()
    cross_market_review_head_done_when = str(
        brief.get("cross_market_review_head_done_when") or ""
    ).strip()
    source_system_time_sync_repair_plan_artifact = str(
        brief.get("source_system_time_sync_repair_plan_artifact") or ""
    ).strip()
    source_system_time_sync_repair_plan_status = str(
        brief.get("source_system_time_sync_repair_plan_status") or ""
    ).strip()
    source_system_time_sync_repair_plan_brief = str(
        brief.get("source_system_time_sync_repair_plan_brief") or ""
    ).strip()
    source_system_time_sync_repair_plan_done_when = str(
        brief.get("source_system_time_sync_repair_plan_done_when") or ""
    ).strip()
    source_system_time_sync_repair_plan_admin_required = brief.get(
        "source_system_time_sync_repair_plan_admin_required"
    )
    source_system_time_sync_repair_verification_artifact = str(
        brief.get("source_system_time_sync_repair_verification_artifact") or ""
    ).strip()
    source_system_time_sync_repair_verification_status = str(
        brief.get("source_system_time_sync_repair_verification_status") or ""
    ).strip()
    source_system_time_sync_repair_verification_brief = str(
        brief.get("source_system_time_sync_repair_verification_brief") or ""
    ).strip()
    source_system_time_sync_repair_verification_cleared = brief.get(
        "source_system_time_sync_repair_verification_cleared"
    )
    source_openclaw_orderflow_blueprint_artifact = str(
        brief.get("source_openclaw_orderflow_blueprint_artifact") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_status = str(
        brief.get("source_openclaw_orderflow_blueprint_status") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_current_life_stage = str(
        brief.get("source_openclaw_orderflow_blueprint_current_life_stage") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_target_life_stage = str(
        brief.get("source_openclaw_orderflow_blueprint_target_life_stage") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_execution_journal_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_execution_journal_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_execution_ack_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_execution_ack_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_title = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_title")
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code")
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_target_artifact = str(
        brief.get(
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_target_artifact"
        )
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_live_boundary_hold_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_live_boundary_hold_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_brief")
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_title = str(
        brief.get(
            "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_title"
        )
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code = str(
        brief.get(
            "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code"
        )
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_target_artifact = str(
        brief.get(
            "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_target_artifact"
        )
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief")
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief")
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_ticket_actionability_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_ticket_actionability_brief")
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_brief")
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_brief = str(
        brief.get(
            "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_brief"
        )
        or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_time_sync_mode = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_time_sync_mode") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_remote_shadow_clock_evidence_brief = str(
        brief.get("source_openclaw_orderflow_blueprint_remote_shadow_clock_evidence_brief") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_top_backlog_title = str(
        brief.get("source_openclaw_orderflow_blueprint_top_backlog_title") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_top_backlog_target_artifact = str(
        brief.get("source_openclaw_orderflow_blueprint_top_backlog_target_artifact") or ""
    ).strip()
    source_openclaw_orderflow_blueprint_top_backlog_why = str(
        brief.get("source_openclaw_orderflow_blueprint_top_backlog_why") or ""
    ).strip()
    cross_market_review_backlog_count_raw = brief.get("cross_market_review_backlog_count")
    cross_market_review_backlog_count = (
        str(cross_market_review_backlog_count_raw).strip()
        if cross_market_review_backlog_count_raw not in (None, "")
        else ""
    )
    cross_market_review_backlog_brief = str(
        brief.get("cross_market_review_backlog_brief") or ""
    ).strip()
    cross_market_operator_head_status = str(brief.get("cross_market_operator_head_status") or "").strip()
    cross_market_operator_head_brief = str(brief.get("cross_market_operator_head_brief") or "").strip()
    cross_market_operator_head_area = str(brief.get("cross_market_operator_head_area") or "").strip()
    cross_market_operator_head_symbol = str(brief.get("cross_market_operator_head_symbol") or "").strip().upper()
    cross_market_operator_head_action = str(brief.get("cross_market_operator_head_action") or "").strip()
    cross_market_operator_head_state = str(brief.get("cross_market_operator_head_state") or "").strip()
    cross_market_operator_head_priority_score_raw = brief.get("cross_market_operator_head_priority_score")
    cross_market_operator_head_priority_score = (
        str(cross_market_operator_head_priority_score_raw).strip()
        if cross_market_operator_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_operator_head_priority_tier = str(
        brief.get("cross_market_operator_head_priority_tier") or ""
    ).strip()
    cross_market_operator_head_blocker_detail = str(
        brief.get("cross_market_operator_head_blocker_detail") or ""
    ).strip()
    cross_market_operator_head_done_when = str(
        brief.get("cross_market_operator_head_done_when") or ""
    ).strip()
    cross_market_remote_live_takeover_gate_status = str(
        brief.get("cross_market_remote_live_takeover_gate_status") or ""
    ).strip()
    cross_market_remote_live_takeover_gate_brief = str(
        brief.get("cross_market_remote_live_takeover_gate_brief") or ""
    ).strip()
    cross_market_remote_live_takeover_gate_blocker_detail = str(
        brief.get("cross_market_remote_live_takeover_gate_blocker_detail") or ""
    ).strip()
    cross_market_remote_live_takeover_gate_done_when = str(
        brief.get("cross_market_remote_live_takeover_gate_done_when") or ""
    ).strip()
    cross_market_remote_live_takeover_clearing_status = str(
        brief.get("cross_market_remote_live_takeover_clearing_status") or ""
    ).strip()
    cross_market_remote_live_takeover_clearing_brief = str(
        brief.get("cross_market_remote_live_takeover_clearing_brief") or ""
    ).strip()
    cross_market_remote_live_takeover_clearing_blocker_detail = str(
        brief.get("cross_market_remote_live_takeover_clearing_blocker_detail") or ""
    ).strip()
    cross_market_remote_live_takeover_clearing_done_when = str(
        brief.get("cross_market_remote_live_takeover_clearing_done_when") or ""
    ).strip()
    remote_live_takeover_repair_queue_status = str(
        brief.get("remote_live_takeover_repair_queue_status") or ""
    ).strip()
    remote_live_takeover_repair_queue_brief = str(
        brief.get("remote_live_takeover_repair_queue_brief") or ""
    ).strip()
    remote_live_takeover_repair_queue_queue_brief = str(
        brief.get("remote_live_takeover_repair_queue_queue_brief") or ""
    ).strip()
    remote_live_takeover_repair_queue_count_raw = brief.get("remote_live_takeover_repair_queue_count")
    remote_live_takeover_repair_queue_count = (
        str(remote_live_takeover_repair_queue_count_raw).strip()
        if remote_live_takeover_repair_queue_count_raw not in (None, "")
        else ""
    )
    remote_live_takeover_repair_queue_head_area = str(
        brief.get("remote_live_takeover_repair_queue_head_area") or ""
    ).strip()
    remote_live_takeover_repair_queue_head_code = str(
        brief.get("remote_live_takeover_repair_queue_head_code") or ""
    ).strip()
    remote_live_takeover_repair_queue_head_action = str(
        brief.get("remote_live_takeover_repair_queue_head_action") or ""
    ).strip()
    remote_live_takeover_repair_queue_head_priority_score_raw = brief.get(
        "remote_live_takeover_repair_queue_head_priority_score"
    )
    remote_live_takeover_repair_queue_head_priority_score = (
        str(remote_live_takeover_repair_queue_head_priority_score_raw).strip()
        if remote_live_takeover_repair_queue_head_priority_score_raw not in (None, "")
        else ""
    )
    remote_live_takeover_repair_queue_head_priority_tier = str(
        brief.get("remote_live_takeover_repair_queue_head_priority_tier") or ""
    ).strip()
    remote_live_takeover_repair_queue_head_command = str(
        brief.get("remote_live_takeover_repair_queue_head_command") or ""
    ).strip()
    remote_live_takeover_repair_queue_head_clear_when = str(
        brief.get("remote_live_takeover_repair_queue_head_clear_when") or ""
    ).strip()
    remote_live_takeover_repair_queue_done_when = str(
        brief.get("remote_live_takeover_repair_queue_done_when") or ""
    ).strip()
    cross_market_operator_repair_head_status = str(
        brief.get("cross_market_operator_repair_head_status") or ""
    ).strip()
    cross_market_operator_repair_head_brief = str(
        brief.get("cross_market_operator_repair_head_brief") or ""
    ).strip()
    cross_market_operator_repair_head_area = str(
        brief.get("cross_market_operator_repair_head_area") or ""
    ).strip()
    cross_market_operator_repair_head_code = str(
        brief.get("cross_market_operator_repair_head_code") or ""
    ).strip().upper()
    cross_market_operator_repair_head_action = str(
        brief.get("cross_market_operator_repair_head_action") or ""
    ).strip()
    cross_market_operator_repair_head_priority_score_raw = brief.get(
        "cross_market_operator_repair_head_priority_score"
    )
    cross_market_operator_repair_head_priority_score = (
        str(cross_market_operator_repair_head_priority_score_raw).strip()
        if cross_market_operator_repair_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_operator_repair_head_priority_tier = str(
        brief.get("cross_market_operator_repair_head_priority_tier") or ""
    ).strip()
    cross_market_operator_repair_head_command = str(
        brief.get("cross_market_operator_repair_head_command") or ""
    ).strip()
    cross_market_operator_repair_head_clear_when = str(
        brief.get("cross_market_operator_repair_head_clear_when") or ""
    ).strip()
    cross_market_operator_repair_head_done_when = str(
        brief.get("cross_market_operator_repair_head_done_when") or ""
    ).strip()
    cross_market_operator_repair_backlog_status = str(
        brief.get("cross_market_operator_repair_backlog_status") or ""
    ).strip()
    cross_market_operator_repair_backlog_brief = str(
        brief.get("cross_market_operator_repair_backlog_brief") or ""
    ).strip()
    cross_market_operator_repair_backlog_count_raw = brief.get(
        "cross_market_operator_repair_backlog_count"
    )
    cross_market_operator_repair_backlog_count = (
        str(cross_market_operator_repair_backlog_count_raw).strip()
        if cross_market_operator_repair_backlog_count_raw not in (None, "")
        else ""
    )
    cross_market_operator_repair_backlog_priority_total_raw = brief.get(
        "cross_market_operator_repair_backlog_priority_total"
    )
    cross_market_operator_repair_backlog_priority_total = (
        str(cross_market_operator_repair_backlog_priority_total_raw).strip()
        if cross_market_operator_repair_backlog_priority_total_raw not in (None, "")
        else ""
    )
    cross_market_operator_repair_backlog_done_when = str(
        brief.get("cross_market_operator_repair_backlog_done_when") or ""
    ).strip()
    cross_market_operator_backlog_count_raw = brief.get("cross_market_operator_backlog_count")
    cross_market_operator_backlog_count = (
        str(cross_market_operator_backlog_count_raw).strip()
        if cross_market_operator_backlog_count_raw not in (None, "")
        else ""
    )
    cross_market_operator_backlog_brief = str(
        brief.get("cross_market_operator_backlog_brief") or ""
    ).strip()
    cross_market_operator_backlog_state_brief = str(
        brief.get("cross_market_operator_backlog_state_brief") or ""
    ).strip()
    cross_market_operator_backlog_priority_totals_brief = str(
        brief.get("cross_market_operator_backlog_priority_totals_brief") or ""
    ).strip()
    cross_market_operator_lane_heads_brief = str(
        brief.get("cross_market_operator_lane_heads_brief") or ""
    ).strip()
    cross_market_operator_lane_priority_order_brief = str(
        brief.get("cross_market_operator_lane_priority_order_brief") or ""
    ).strip()
    cross_market_operator_waiting_lane_status = str(
        brief.get("cross_market_operator_waiting_lane_status") or ""
    ).strip()
    cross_market_operator_waiting_lane_count_raw = brief.get("cross_market_operator_waiting_lane_count")
    cross_market_operator_waiting_lane_count = (
        str(cross_market_operator_waiting_lane_count_raw).strip()
        if cross_market_operator_waiting_lane_count_raw not in (None, "")
        else ""
    )
    cross_market_operator_waiting_lane_brief = str(
        brief.get("cross_market_operator_waiting_lane_brief") or ""
    ).strip()
    cross_market_operator_waiting_lane_priority_total_raw = brief.get("cross_market_operator_waiting_lane_priority_total")
    cross_market_operator_waiting_lane_priority_total = (
        str(cross_market_operator_waiting_lane_priority_total_raw).strip()
        if cross_market_operator_waiting_lane_priority_total_raw not in (None, "")
        else ""
    )
    cross_market_operator_waiting_lane_head_symbol = str(
        brief.get("cross_market_operator_waiting_lane_head_symbol") or ""
    ).strip().upper()
    cross_market_operator_waiting_lane_head_action = str(
        brief.get("cross_market_operator_waiting_lane_head_action") or ""
    ).strip()
    cross_market_operator_waiting_lane_head_priority_score_raw = brief.get(
        "cross_market_operator_waiting_lane_head_priority_score"
    )
    cross_market_operator_waiting_lane_head_priority_score = (
        str(cross_market_operator_waiting_lane_head_priority_score_raw).strip()
        if cross_market_operator_waiting_lane_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_operator_waiting_lane_head_priority_tier = str(
        brief.get("cross_market_operator_waiting_lane_head_priority_tier") or ""
    ).strip()
    cross_market_operator_review_lane_status = str(
        brief.get("cross_market_operator_review_lane_status") or ""
    ).strip()
    cross_market_operator_review_lane_count_raw = brief.get("cross_market_operator_review_lane_count")
    cross_market_operator_review_lane_count = (
        str(cross_market_operator_review_lane_count_raw).strip()
        if cross_market_operator_review_lane_count_raw not in (None, "")
        else ""
    )
    cross_market_operator_review_lane_brief = str(
        brief.get("cross_market_operator_review_lane_brief") or ""
    ).strip()
    cross_market_operator_review_lane_priority_total_raw = brief.get("cross_market_operator_review_lane_priority_total")
    cross_market_operator_review_lane_priority_total = (
        str(cross_market_operator_review_lane_priority_total_raw).strip()
        if cross_market_operator_review_lane_priority_total_raw not in (None, "")
        else ""
    )
    cross_market_operator_review_lane_head_symbol = str(
        brief.get("cross_market_operator_review_lane_head_symbol") or ""
    ).strip().upper()
    cross_market_operator_review_lane_head_action = str(
        brief.get("cross_market_operator_review_lane_head_action") or ""
    ).strip()
    cross_market_operator_review_lane_head_priority_score_raw = brief.get(
        "cross_market_operator_review_lane_head_priority_score"
    )
    cross_market_operator_review_lane_head_priority_score = (
        str(cross_market_operator_review_lane_head_priority_score_raw).strip()
        if cross_market_operator_review_lane_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_operator_review_lane_head_priority_tier = str(
        brief.get("cross_market_operator_review_lane_head_priority_tier") or ""
    ).strip()
    cross_market_operator_watch_lane_status = str(
        brief.get("cross_market_operator_watch_lane_status") or ""
    ).strip()
    cross_market_operator_watch_lane_count_raw = brief.get("cross_market_operator_watch_lane_count")
    cross_market_operator_watch_lane_count = (
        str(cross_market_operator_watch_lane_count_raw).strip()
        if cross_market_operator_watch_lane_count_raw not in (None, "")
        else ""
    )
    cross_market_operator_watch_lane_brief = str(
        brief.get("cross_market_operator_watch_lane_brief") or ""
    ).strip()
    cross_market_operator_watch_lane_priority_total_raw = brief.get("cross_market_operator_watch_lane_priority_total")
    cross_market_operator_watch_lane_priority_total = (
        str(cross_market_operator_watch_lane_priority_total_raw).strip()
        if cross_market_operator_watch_lane_priority_total_raw not in (None, "")
        else ""
    )
    cross_market_operator_watch_lane_head_symbol = str(
        brief.get("cross_market_operator_watch_lane_head_symbol") or ""
    ).strip().upper()
    cross_market_operator_watch_lane_head_action = str(
        brief.get("cross_market_operator_watch_lane_head_action") or ""
    ).strip()
    cross_market_operator_watch_lane_head_priority_score_raw = brief.get(
        "cross_market_operator_watch_lane_head_priority_score"
    )
    cross_market_operator_watch_lane_head_priority_score = (
        str(cross_market_operator_watch_lane_head_priority_score_raw).strip()
        if cross_market_operator_watch_lane_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_operator_watch_lane_head_priority_tier = str(
        brief.get("cross_market_operator_watch_lane_head_priority_tier") or ""
    ).strip()
    cross_market_operator_blocked_lane_status = str(
        brief.get("cross_market_operator_blocked_lane_status") or ""
    ).strip()
    cross_market_operator_blocked_lane_count_raw = brief.get("cross_market_operator_blocked_lane_count")
    cross_market_operator_blocked_lane_count = (
        str(cross_market_operator_blocked_lane_count_raw).strip()
        if cross_market_operator_blocked_lane_count_raw not in (None, "")
        else ""
    )
    cross_market_operator_blocked_lane_brief = str(
        brief.get("cross_market_operator_blocked_lane_brief") or ""
    ).strip()
    cross_market_operator_blocked_lane_priority_total_raw = brief.get("cross_market_operator_blocked_lane_priority_total")
    cross_market_operator_blocked_lane_priority_total = (
        str(cross_market_operator_blocked_lane_priority_total_raw).strip()
        if cross_market_operator_blocked_lane_priority_total_raw not in (None, "")
        else ""
    )
    cross_market_operator_blocked_lane_head_symbol = str(
        brief.get("cross_market_operator_blocked_lane_head_symbol") or ""
    ).strip().upper()
    cross_market_operator_blocked_lane_head_action = str(
        brief.get("cross_market_operator_blocked_lane_head_action") or ""
    ).strip()
    cross_market_operator_blocked_lane_head_priority_score_raw = brief.get(
        "cross_market_operator_blocked_lane_head_priority_score"
    )
    cross_market_operator_blocked_lane_head_priority_score = (
        str(cross_market_operator_blocked_lane_head_priority_score_raw).strip()
        if cross_market_operator_blocked_lane_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_operator_blocked_lane_head_priority_tier = str(
        brief.get("cross_market_operator_blocked_lane_head_priority_tier") or ""
    ).strip()
    cross_market_operator_repair_lane_status = str(
        brief.get("cross_market_operator_repair_lane_status") or ""
    ).strip()
    cross_market_operator_repair_lane_count_raw = brief.get("cross_market_operator_repair_lane_count")
    cross_market_operator_repair_lane_count = (
        str(cross_market_operator_repair_lane_count_raw).strip()
        if cross_market_operator_repair_lane_count_raw not in (None, "")
        else ""
    )
    cross_market_operator_repair_lane_brief = str(
        brief.get("cross_market_operator_repair_lane_brief") or ""
    ).strip()
    cross_market_operator_repair_lane_priority_total_raw = brief.get("cross_market_operator_repair_lane_priority_total")
    cross_market_operator_repair_lane_priority_total = (
        str(cross_market_operator_repair_lane_priority_total_raw).strip()
        if cross_market_operator_repair_lane_priority_total_raw not in (None, "")
        else ""
    )
    cross_market_operator_repair_lane_head_symbol = str(
        brief.get("cross_market_operator_repair_lane_head_symbol") or ""
    ).strip().upper()
    cross_market_operator_repair_lane_head_action = str(
        brief.get("cross_market_operator_repair_lane_head_action") or ""
    ).strip()
    cross_market_operator_repair_lane_head_priority_score_raw = brief.get(
        "cross_market_operator_repair_lane_head_priority_score"
    )
    cross_market_operator_repair_lane_head_priority_score = (
        str(cross_market_operator_repair_lane_head_priority_score_raw).strip()
        if cross_market_operator_repair_lane_head_priority_score_raw not in (None, "")
        else ""
    )
    cross_market_operator_repair_lane_head_priority_tier = str(
        brief.get("cross_market_operator_repair_lane_head_priority_tier") or ""
    ).strip()
    brooks_structure_review_status = str(
        brief.get("brooks_structure_review_status") or ""
    ).strip()
    brooks_structure_review_brief = str(
        brief.get("brooks_structure_review_brief") or ""
    ).strip()
    brooks_structure_review_queue_status = str(
        brief.get("brooks_structure_review_queue_status") or ""
    ).strip()
    brooks_structure_review_queue_count_raw = brief.get(
        "brooks_structure_review_queue_count"
    )
    brooks_structure_review_queue_count = (
        str(brooks_structure_review_queue_count_raw).strip()
        if brooks_structure_review_queue_count_raw not in (None, "")
        else ""
    )
    brooks_structure_review_queue_brief = str(
        brief.get("brooks_structure_review_queue_brief") or ""
    ).strip()
    brooks_structure_review_priority_status = str(
        brief.get("brooks_structure_review_priority_status") or ""
    ).strip()
    brooks_structure_review_priority_brief = str(
        brief.get("brooks_structure_review_priority_brief") or ""
    ).strip()
    brooks_structure_review_queue = [
        dict(row)
        for row in list(brief.get("brooks_structure_review_queue") or [])
        if isinstance(row, dict)
    ]
    brooks_structure_review_head_rank_raw = brief.get(
        "brooks_structure_review_head_rank"
    )
    brooks_structure_review_head_rank = (
        str(brooks_structure_review_head_rank_raw).strip()
        if brooks_structure_review_head_rank_raw not in (None, "")
        else ""
    )
    brooks_structure_review_head_symbol = str(
        brief.get("brooks_structure_review_head_symbol") or ""
    ).strip()
    brooks_structure_review_head_strategy_id = str(
        brief.get("brooks_structure_review_head_strategy_id") or ""
    ).strip()
    brooks_structure_review_head_direction = str(
        brief.get("brooks_structure_review_head_direction") or ""
    ).strip()
    brooks_structure_review_head_tier = str(
        brief.get("brooks_structure_review_head_tier") or ""
    ).strip()
    brooks_structure_review_head_plan_status = str(
        brief.get("brooks_structure_review_head_plan_status") or ""
    ).strip()
    brooks_structure_review_head_action = str(
        brief.get("brooks_structure_review_head_action") or ""
    ).strip()
    brooks_structure_review_head_route_selection_score_raw = brief.get(
        "brooks_structure_review_head_route_selection_score"
    )
    brooks_structure_review_head_route_selection_score = (
        _fmt_num(brooks_structure_review_head_route_selection_score_raw)
        if brooks_structure_review_head_route_selection_score_raw not in (None, "")
        else ""
    )
    brooks_structure_review_head_signal_score_raw = brief.get(
        "brooks_structure_review_head_signal_score"
    )
    brooks_structure_review_head_signal_score = (
        str(brooks_structure_review_head_signal_score_raw).strip()
        if brooks_structure_review_head_signal_score_raw not in (None, "")
        else ""
    )
    brooks_structure_review_head_signal_age_bars_raw = brief.get(
        "brooks_structure_review_head_signal_age_bars"
    )
    brooks_structure_review_head_signal_age_bars = (
        str(brooks_structure_review_head_signal_age_bars_raw).strip()
        if brooks_structure_review_head_signal_age_bars_raw not in (None, "")
        else ""
    )
    brooks_structure_review_head_priority_score_raw = brief.get(
        "brooks_structure_review_head_priority_score"
    )
    brooks_structure_review_head_priority_score = (
        str(brooks_structure_review_head_priority_score_raw).strip()
        if brooks_structure_review_head_priority_score_raw not in (None, "")
        else ""
    )
    brooks_structure_review_head_priority_tier = str(
        brief.get("brooks_structure_review_head_priority_tier") or ""
    ).strip()
    brooks_structure_review_head_blocker_detail = str(
        brief.get("brooks_structure_review_head_blocker_detail") or ""
    ).strip()
    brooks_structure_review_head_done_when = str(
        brief.get("brooks_structure_review_head_done_when") or ""
    ).strip()
    brooks_structure_review_blocker_detail = str(
        brief.get("brooks_structure_review_blocker_detail") or ""
    ).strip()
    brooks_structure_review_done_when = str(
        brief.get("brooks_structure_review_done_when") or ""
    ).strip()
    brooks_structure_operator_status = str(
        brief.get("brooks_structure_operator_status") or ""
    ).strip()
    brooks_structure_operator_brief = str(
        brief.get("brooks_structure_operator_brief") or ""
    ).strip()
    brooks_structure_operator_head_symbol = str(
        brief.get("brooks_structure_operator_head_symbol") or ""
    ).strip()
    brooks_structure_operator_head_strategy_id = str(
        brief.get("brooks_structure_operator_head_strategy_id") or ""
    ).strip()
    brooks_structure_operator_head_direction = str(
        brief.get("brooks_structure_operator_head_direction") or ""
    ).strip()
    brooks_structure_operator_head_action = str(
        brief.get("brooks_structure_operator_head_action") or ""
    ).strip()
    brooks_structure_operator_head_plan_status = str(
        brief.get("brooks_structure_operator_head_plan_status") or ""
    ).strip()
    brooks_structure_operator_head_priority_score_raw = brief.get(
        "brooks_structure_operator_head_priority_score"
    )
    brooks_structure_operator_head_priority_score = (
        str(brooks_structure_operator_head_priority_score_raw).strip()
        if brooks_structure_operator_head_priority_score_raw not in (None, "")
        else ""
    )
    brooks_structure_operator_head_priority_tier = str(
        brief.get("brooks_structure_operator_head_priority_tier") or ""
    ).strip()
    brooks_structure_operator_backlog_count_raw = brief.get(
        "brooks_structure_operator_backlog_count"
    )
    brooks_structure_operator_backlog_count = (
        str(brooks_structure_operator_backlog_count_raw).strip()
        if brooks_structure_operator_backlog_count_raw not in (None, "")
        else ""
    )
    brooks_structure_operator_backlog_brief = str(
        brief.get("brooks_structure_operator_backlog_brief") or ""
    ).strip()
    brooks_structure_operator_blocker_detail = str(
        brief.get("brooks_structure_operator_blocker_detail") or ""
    ).strip()
    brooks_structure_operator_done_when = str(
        brief.get("brooks_structure_operator_done_when") or ""
    ).strip()
    secondary_focus_area = str(brief.get("secondary_focus_area") or "").strip()
    secondary_focus_target = str(brief.get("secondary_focus_target") or "").strip()
    secondary_focus_symbol = str(brief.get("secondary_focus_symbol") or "").strip()
    secondary_focus_action = str(brief.get("secondary_focus_action") or "").strip()
    secondary_focus_reason = str(brief.get("secondary_focus_reason") or "").strip()
    secondary_focus_state = str(brief.get("secondary_focus_state") or "").strip()
    secondary_focus_blocker_detail = str(brief.get("secondary_focus_blocker_detail") or "").strip()
    secondary_focus_done_when = str(brief.get("secondary_focus_done_when") or "").strip()
    secondary_focus_priority_tier = str(brief.get("secondary_focus_priority_tier") or "").strip()
    secondary_focus_priority_score_raw = brief.get("secondary_focus_priority_score")
    secondary_focus_priority_score = (
        str(secondary_focus_priority_score_raw).strip()
        if secondary_focus_priority_score_raw not in (None, "")
        else ""
    )
    secondary_focus_queue_rank_raw = brief.get("secondary_focus_queue_rank")
    secondary_focus_queue_rank = (
        str(secondary_focus_queue_rank_raw).strip()
        if secondary_focus_queue_rank_raw not in (None, "")
        else ""
    )
    crypto_route_shortline_market_state_brief = str(
        brief.get("crypto_route_shortline_market_state_brief") or ""
    ).strip()
    crypto_route_shortline_execution_gate_brief = str(
        brief.get("crypto_route_shortline_execution_gate_brief") or ""
    ).strip()
    crypto_route_shortline_no_trade_rule = str(
        brief.get("crypto_route_shortline_no_trade_rule") or ""
    ).strip()
    crypto_route_shortline_session_map_brief = str(
        brief.get("crypto_route_shortline_session_map_brief") or ""
    ).strip()
    crypto_route_shortline_cvd_semantic_status = str(
        brief.get("crypto_route_shortline_cvd_semantic_status") or ""
    ).strip()
    crypto_route_shortline_cvd_semantic_takeaway = str(
        brief.get("crypto_route_shortline_cvd_semantic_takeaway") or ""
    ).strip()
    crypto_route_shortline_cvd_queue_handoff_status = str(
        brief.get("crypto_route_shortline_cvd_queue_handoff_status") or ""
    ).strip()
    crypto_route_shortline_cvd_queue_handoff_takeaway = str(
        brief.get("crypto_route_shortline_cvd_queue_handoff_takeaway") or ""
    ).strip()
    crypto_route_shortline_cvd_queue_focus_batch = str(
        brief.get("crypto_route_shortline_cvd_queue_focus_batch") or ""
    ).strip()
    crypto_route_shortline_cvd_queue_focus_action = str(
        brief.get("crypto_route_shortline_cvd_queue_focus_action") or ""
    ).strip()
    crypto_route_shortline_cvd_queue_stack_brief = str(
        brief.get("crypto_route_shortline_cvd_queue_stack_brief") or ""
    ).strip()
    crypto_route_focus_execution_state = str(brief.get("crypto_route_focus_execution_state") or "").strip()
    crypto_route_focus_execution_blocker_detail = str(
        brief.get("crypto_route_focus_execution_blocker_detail") or ""
    ).strip()
    crypto_route_focus_execution_done_when = str(
        brief.get("crypto_route_focus_execution_done_when") or ""
    ).strip()
    crypto_route_focus_execution_micro_classification = str(
        brief.get("crypto_route_focus_execution_micro_classification") or ""
    ).strip()
    crypto_route_focus_execution_micro_context = str(
        brief.get("crypto_route_focus_execution_micro_context") or ""
    ).strip()
    crypto_route_focus_execution_micro_trust_tier = str(
        brief.get("crypto_route_focus_execution_micro_trust_tier") or ""
    ).strip()
    crypto_route_focus_execution_micro_veto = str(
        brief.get("crypto_route_focus_execution_micro_veto") or ""
    ).strip()
    crypto_route_focus_execution_micro_locality_status = str(
        brief.get("crypto_route_focus_execution_micro_locality_status") or ""
    ).strip()
    crypto_route_focus_execution_micro_drift_risk = str(
        brief.get("crypto_route_focus_execution_micro_drift_risk") or ""
    ).strip()
    crypto_route_focus_execution_micro_attack_side = str(
        brief.get("crypto_route_focus_execution_micro_attack_side") or ""
    ).strip()
    crypto_route_focus_execution_micro_attack_presence = str(
        brief.get("crypto_route_focus_execution_micro_attack_presence") or ""
    ).strip()
    crypto_route_focus_execution_micro_reasons = [
        str(x).strip()
        for x in brief.get("crypto_route_focus_execution_micro_reasons", [])
        if str(x).strip()
    ]
    crypto_route_focus_review_status = str(brief.get("crypto_route_focus_review_status") or "").strip()
    crypto_route_focus_review_brief = str(brief.get("crypto_route_focus_review_brief") or "").strip()
    crypto_route_focus_review_primary_blocker = str(
        brief.get("crypto_route_focus_review_primary_blocker") or ""
    ).strip()
    crypto_route_focus_review_micro_blocker = str(
        brief.get("crypto_route_focus_review_micro_blocker") or ""
    ).strip()
    crypto_route_focus_review_blocker_detail = str(
        brief.get("crypto_route_focus_review_blocker_detail") or ""
    ).strip()
    crypto_route_focus_review_done_when = str(
        brief.get("crypto_route_focus_review_done_when") or ""
    ).strip()
    crypto_route_focus_review_score_status = str(
        brief.get("crypto_route_focus_review_score_status") or ""
    ).strip()
    crypto_route_focus_review_edge_score = int(brief.get("crypto_route_focus_review_edge_score") or 0)
    crypto_route_focus_review_structure_score = int(brief.get("crypto_route_focus_review_structure_score") or 0)
    crypto_route_focus_review_micro_score = int(brief.get("crypto_route_focus_review_micro_score") or 0)
    crypto_route_focus_review_composite_score = int(brief.get("crypto_route_focus_review_composite_score") or 0)
    crypto_route_focus_review_score_brief = str(
        brief.get("crypto_route_focus_review_score_brief") or ""
    ).strip()
    crypto_route_focus_review_priority_status = str(
        brief.get("crypto_route_focus_review_priority_status") or ""
    ).strip()
    crypto_route_focus_review_priority_score = int(brief.get("crypto_route_focus_review_priority_score") or 0)
    crypto_route_focus_review_priority_tier = str(
        brief.get("crypto_route_focus_review_priority_tier") or ""
    ).strip()
    crypto_route_focus_review_priority_brief = str(
        brief.get("crypto_route_focus_review_priority_brief") or ""
    ).strip()
    crypto_route_review_priority_queue_status = str(
        brief.get("crypto_route_review_priority_queue_status") or ""
    ).strip()
    crypto_route_review_priority_queue_count = int(brief.get("crypto_route_review_priority_queue_count") or 0)
    crypto_route_review_priority_queue_brief = str(
        brief.get("crypto_route_review_priority_queue_brief") or ""
    ).strip()
    crypto_route_review_priority_head_symbol = str(
        brief.get("crypto_route_review_priority_head_symbol") or ""
    ).strip()
    crypto_route_review_priority_head_tier = str(
        brief.get("crypto_route_review_priority_head_tier") or ""
    ).strip()
    crypto_route_review_priority_head_score = int(brief.get("crypto_route_review_priority_head_score") or 0)
    crypto_route_review_priority_queue = [
        dict(row) for row in brief.get("crypto_route_review_priority_queue", []) if isinstance(row, dict)
    ]
    operator_action_queue_brief = str(brief.get("operator_action_queue_brief") or "").strip()
    operator_action_checklist = [
        dict(row) for row in brief.get("operator_action_checklist", []) if isinstance(row, dict)
    ]
    operator_action_checklist_brief = str(brief.get("operator_action_checklist_brief") or "").strip()
    operator_repair_queue = [
        dict(row) for row in brief.get("operator_repair_queue", []) if isinstance(row, dict)
    ]
    operator_repair_queue_brief = str(brief.get("operator_repair_queue_brief") or "").strip()
    operator_repair_queue_count_raw = brief.get("operator_repair_queue_count")
    operator_repair_queue_count = (
        str(operator_repair_queue_count_raw).strip()
        if operator_repair_queue_count_raw not in (None, "")
        else ""
    )
    operator_repair_checklist = [
        dict(row) for row in brief.get("operator_repair_checklist", []) if isinstance(row, dict)
    ]
    operator_repair_checklist_brief = str(brief.get("operator_repair_checklist_brief") or "").strip()
    commodity_focus_evidence_summary = dict(brief.get("commodity_focus_evidence_summary") or {})
    review_pending_symbols = [
        str(x).strip().upper()
        for x in brief.get("commodity_review_pending_symbols", review.get("review_pending_symbols", []))
        if str(x).strip()
    ]
    review_close_evidence_pending_symbols = [
        str(x).strip().upper()
        for x in (
            brief.get("commodity_review_close_evidence_pending_symbols")
            or review.get("close_evidence_pending_symbols", [])
            or brief.get("commodity_close_evidence_pending_symbols")
            or retro.get("close_evidence_pending_symbols", [])
        )
        if str(x).strip()
    ]
    retro_pending_symbols = [
        str(x).strip().upper()
        for x in brief.get("commodity_retro_pending_symbols", retro.get("retro_pending_symbols", []))
        if str(x).strip()
    ]
    fill_evidence_pending_symbols = [
        str(x).strip().upper()
        for x in brief.get(
            "commodity_retro_fill_evidence_pending_symbols",
            retro.get("fill_evidence_pending_symbols", review.get("fill_evidence_pending_symbols", [])),
        )
        if str(x).strip()
    ]
    with_evidence = [
        str(x).strip().upper()
        for x in gap.get("queue_symbols_with_any_evidence", [])
        if str(x).strip()
    ]
    without_evidence = [
        str(x).strip().upper()
        for x in gap.get("queue_symbols_without_any_evidence", [])
        if str(x).strip()
    ]
    applied_symbols = bridge_apply_symbols(bridge_apply or {})
    lines = [
        "# Next Window Context",
        "",
        f"Updated: {context_updated_line(runtime_now)}",
        "",
        "## Current State",
    ]
    if applied_symbols:
        lines.append("- Paper execution evidence was written in this refresh cycle for: `" + _list_text(applied_symbols) + "`")
    elif with_evidence:
        lines.append("- Commodity paper evidence already exists for: `" + _list_text(with_evidence) + "`")
    else:
        lines.append("- No commodity paper execution evidence exists yet.")
    if stale_symbols:
        lines.append("- Commodity stale-signal watch remains active for: `" + _list_text(stale_symbols) + "`")
    else:
        lines.append("- No commodity stale-signal blockers remain in the current queue.")
    if stale_signal_dates:
        lines.append("- Commodity stale-signal dates are: `" + _mapping_text(stale_signal_dates) + "`")
    if stale_signal_age_days:
        lines.append("- Commodity stale-signal ages are: `" + _mapping_text(stale_signal_age_days) + "`")
    if stale_signal_watch_items:
        lines.append("- Commodity stale-signal watch priority is: `" + _watch_items_text(stale_signal_watch_items) + "`")
    if commodity_stale_signal_watch_next_symbol:
        lines.append(
            "- Commodity stale-signal watch head is: `"
            + " ".join(
                [
                    commodity_stale_signal_watch_next_symbol,
                    f"target={commodity_stale_signal_watch_next_execution_id or '-'}",
                    f"date={commodity_stale_signal_watch_next_signal_date or '-'}",
                    f"age={commodity_stale_signal_watch_next_signal_age_days or '-'}d",
                ]
            )
            + "`"
        )
    if followup_focus_area:
        lines.append(
            "- Follow-up after primary focus is: `"
            + " ".join(
                [
                    followup_focus_area,
                    f"target={followup_focus_target or '-'}",
                    f"action={followup_focus_action or '-'}",
                ]
            )
            + "`"
        )
    if next_focus_state:
        lines.append(
            "- Primary focus gate is: `"
            + " | ".join(
                [
                    f"state={next_focus_state or '-'}",
                    f"blocker={next_focus_blocker_detail or '-'}",
                    f"done_when={next_focus_done_when or '-'}",
                ]
            )
            + "`"
        )
    if followup_focus_state:
        lines.append(
            "- Follow-up gate is: `"
            + " | ".join(
                [
                    f"state={followup_focus_state or '-'}",
                    f"blocker={followup_focus_blocker_detail or '-'}",
                    f"done_when={followup_focus_done_when or '-'}",
                ]
            )
            + "`"
        )
    if secondary_focus_state:
        lines.append(
            "- Secondary focus gate is: `"
            + " | ".join(
                [
                    f"state={secondary_focus_state or '-'}",
                    f"blocker={secondary_focus_blocker_detail or '-'}",
                    f"done_when={secondary_focus_done_when or '-'}",
                ]
            )
            + "`"
        )
    if secondary_focus_priority_tier or secondary_focus_priority_score or secondary_focus_queue_rank:
        lines.append(
            "- Secondary focus priority is: `"
            + " | ".join(
                [
                    f"tier={secondary_focus_priority_tier or '-'}",
                    f"score={secondary_focus_priority_score or '-'}",
                    f"queue_rank={secondary_focus_queue_rank or '-'}",
                ]
            )
            + "`"
        )
    if operator_focus_slots_brief:
        lines.append("- Focus slots are: `" + operator_focus_slots_brief + "`")
    if operator_focus_slot_sources_brief:
        lines.append("- Focus slot sources are: `" + operator_focus_slot_sources_brief + "`")
    if operator_focus_slot_status_brief:
        lines.append("- Focus slot source status is: `" + operator_focus_slot_status_brief + "`")
    if operator_focus_slot_recency_brief:
        lines.append("- Focus slot source recency is: `" + operator_focus_slot_recency_brief + "`")
    if operator_focus_slot_health_brief:
        lines.append("- Focus slot source health is: `" + operator_focus_slot_health_brief + "`")
    if operator_focus_slot_refresh_backlog_brief:
        lines.append("- Focus slot refresh backlog is: `" + operator_focus_slot_refresh_backlog_brief + "`")
    if operator_focus_slot_promotion_gate_status:
        lines.append(
            "- Focus slot promotion gate is: `"
            + " | ".join(
                [
                    f"status={operator_focus_slot_promotion_gate_status or '-'}",
                    f"ready={operator_focus_slot_ready_count or '-'}/{operator_focus_slot_total_count or '-'}",
                    f"blocker={operator_focus_slot_promotion_gate_blocker_detail or '-'}",
                    f"done_when={operator_focus_slot_promotion_gate_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_focus_slot_actionability_gate_status:
        lines.append(
            "- Focus slot actionability gate is: `"
            + " | ".join(
                [
                    f"status={operator_focus_slot_actionability_gate_status or '-'}",
                    f"actionable={operator_focus_slot_actionable_count or '-'}/{operator_focus_slot_total_count or '-'}",
                    f"blocker={operator_focus_slot_actionability_gate_blocker_detail or '-'}",
                    f"done_when={operator_focus_slot_actionability_gate_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_focus_slot_readiness_gate_status:
        lines.append(
            "- Focus slot readiness gate is: `"
            + " | ".join(
                [
                    f"status={operator_focus_slot_readiness_gate_status or '-'}",
                    f"blocking_gate={operator_focus_slot_readiness_gate_blocking_gate or '-'}",
                    f"ready={operator_focus_slot_readiness_gate_ready_count or '-'}/{operator_focus_slot_total_count or '-'}",
                    f"blocker={operator_focus_slot_readiness_gate_blocker_detail or '-'}",
                    f"done_when={operator_focus_slot_readiness_gate_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_research_embedding_quality_brief or operator_research_embedding_quality_status:
        lines.append(
            "- Research embedding quality is: `"
            + " | ".join(
                [
                    f"status={operator_research_embedding_quality_status or '-'}",
                    f"brief={operator_research_embedding_quality_brief or '-'}",
                    f"blocker={operator_research_embedding_quality_blocker_detail or '-'}",
                    f"done_when={operator_research_embedding_quality_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_crypto_route_alignment_brief or operator_crypto_route_alignment_status:
        lines.append(
            "- Crypto route alignment is: `"
            + " | ".join(
                [
                    f"area={operator_crypto_route_alignment_focus_area or '-'}",
                    f"slot={operator_crypto_route_alignment_focus_slot or '-'}",
                    f"symbol={operator_crypto_route_alignment_focus_symbol or '-'}",
                    f"action={operator_crypto_route_alignment_focus_action or '-'}",
                    f"status={operator_crypto_route_alignment_status or '-'}",
                    f"brief={operator_crypto_route_alignment_brief or '-'}",
                    f"blocker={operator_crypto_route_alignment_blocker_detail or '-'}",
                    f"done_when={operator_crypto_route_alignment_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_crypto_route_alignment_recovery_brief or operator_crypto_route_alignment_recovery_status:
        lines.append(
            "- Crypto route alignment recovery outcome is: `"
            + " | ".join(
                [
                    f"status={operator_crypto_route_alignment_recovery_status or '-'}",
                    f"brief={operator_crypto_route_alignment_recovery_brief or '-'}",
                    f"failed={operator_crypto_route_alignment_recovery_failed_batch_count or '-'}",
                    f"timed_out={operator_crypto_route_alignment_recovery_timed_out_batch_count or '-'}",
                    f"zero_trade_batches={_list_text(operator_crypto_route_alignment_recovery_zero_trade_batches) or '-'}",
                    f"blocker={operator_crypto_route_alignment_recovery_blocker_detail or '-'}",
                    f"done_when={operator_crypto_route_alignment_recovery_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_crypto_route_alignment_cooldown_brief or operator_crypto_route_alignment_cooldown_status:
        lines.append(
            "- Crypto route alignment cooldown is: `"
            + " | ".join(
                [
                    f"status={operator_crypto_route_alignment_cooldown_status or '-'}",
                    f"brief={operator_crypto_route_alignment_cooldown_brief or '-'}",
                    f"last_end={operator_crypto_route_alignment_cooldown_last_research_end_date or '-'}",
                    f"next_eligible={operator_crypto_route_alignment_cooldown_next_eligible_end_date or '-'}",
                    f"blocker={operator_crypto_route_alignment_cooldown_blocker_detail or '-'}",
                    f"done_when={operator_crypto_route_alignment_cooldown_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_crypto_route_alignment_recipe_brief or operator_crypto_route_alignment_recipe_status:
        lines.append(
            "- Crypto route alignment recovery recipe gate is: `"
            + " | ".join(
                [
                    f"status={operator_crypto_route_alignment_recipe_status or '-'}",
                    f"brief={operator_crypto_route_alignment_recipe_brief or '-'}",
                    f"ready_on={operator_crypto_route_alignment_recipe_ready_on_date or '-'}",
                    f"blocker={operator_crypto_route_alignment_recipe_blocker_detail or '-'}",
                    f"done_when={operator_crypto_route_alignment_recipe_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_crypto_route_alignment_recipe_command_hint or operator_crypto_route_alignment_recipe_script:
        recipe_label = "recovery template" if operator_crypto_route_alignment_recipe_status == "deferred_by_cooldown" else "recovery"
        lines.append(
            f"- Crypto route alignment {recipe_label} is: `"
            + " | ".join(
                [
                    f"script={operator_crypto_route_alignment_recipe_script or '-'}",
                    f"expected_status={operator_crypto_route_alignment_recipe_expected_status or '-'}",
                    f"window_days={operator_crypto_route_alignment_recipe_window_days or '-'}",
                    f"target_batches={_list_text(operator_crypto_route_alignment_recipe_target_batches) or '-'}",
                    f"note={operator_crypto_route_alignment_recipe_note or '-'}",
                ]
            )
            + "`"
        )
    if (
        crypto_route_shortline_market_state_brief
        or crypto_route_focus_execution_state
        or crypto_route_shortline_cvd_queue_handoff_status
        or crypto_route_focus_execution_micro_classification
    ):
        lines.append(
            "- Crypto shortline gate is: `"
            + " | ".join(
                [
                    f"market={crypto_route_shortline_market_state_brief or '-'}",
                    f"focus_state={crypto_route_focus_execution_state or '-'}",
                    f"micro={crypto_route_focus_execution_micro_classification or '-'}:{crypto_route_focus_execution_micro_context or '-'}:{crypto_route_focus_execution_micro_veto or '-'}",
                    f"cvd_queue={crypto_route_shortline_cvd_queue_handoff_status or '-'}:{crypto_route_shortline_cvd_queue_focus_batch or '-'}:{crypto_route_shortline_cvd_queue_focus_action or '-'}",
                ]
            )
            + "`"
        )
    if crypto_route_focus_review_status:
        lines.append(
            "- Crypto review lane is: `"
            + " | ".join(
                [
                    f"status={crypto_route_focus_review_status or '-'}",
                    f"brief={crypto_route_focus_review_brief or '-'}",
                    f"primary_blocker={crypto_route_focus_review_primary_blocker or '-'}",
                    f"micro_blocker={crypto_route_focus_review_micro_blocker or '-'}",
                    f"done_when={crypto_route_focus_review_done_when or '-'}",
                ]
            )
            + "`"
        )
    if crypto_route_focus_review_score_status == "scored":
        lines.append(
            "- Crypto review scores are: `"
            + " | ".join(
                [
                    f"edge={crypto_route_focus_review_edge_score}",
                    f"structure={crypto_route_focus_review_structure_score}",
                    f"micro={crypto_route_focus_review_micro_score}",
                    f"composite={crypto_route_focus_review_composite_score}",
                ]
            )
            + "`"
        )
    if crypto_route_focus_review_priority_status == "ready":
        lines.append(
            "- Crypto review priority is: `"
            + " | ".join(
                [
                    f"tier={crypto_route_focus_review_priority_tier or '-'}",
                    f"score={crypto_route_focus_review_priority_score}",
                    f"brief={crypto_route_focus_review_priority_brief or '-'}",
                ]
            )
            + "`"
        )
    if crypto_route_review_priority_queue_status:
        lines.append(
            "- Crypto review queue is: `"
            + " | ".join(
                [
                    f"status={crypto_route_review_priority_queue_status or '-'}",
                    f"count={crypto_route_review_priority_queue_count}",
                    f"brief={crypto_route_review_priority_queue_brief or '-'}",
                    f"head={crypto_route_review_priority_head_symbol or '-'}:{crypto_route_review_priority_head_tier or '-'}:{crypto_route_review_priority_head_score}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_queue_brief:
        lines.append("- Source refresh queue is: `" + operator_source_refresh_queue_brief + "`")
    if crypto_route_head_source_refresh_status or crypto_route_head_source_refresh_symbol:
        lines.append(
            "- Crypto route head source refresh is: `"
            + " | ".join(
                [
                    f"status={crypto_route_head_source_refresh_status or '-'}",
                    f"brief={crypto_route_head_source_refresh_brief or '-'}",
                    f"slot={crypto_route_head_source_refresh_slot or '-'}",
                    f"symbol={crypto_route_head_source_refresh_symbol or '-'}",
                    f"action={crypto_route_head_source_refresh_action or '-'}",
                    f"kind={crypto_route_head_source_refresh_source_kind or '-'}",
                    f"health={crypto_route_head_source_refresh_source_health or '-'}",
                ]
            )
            + "`"
        )
        lines.append(
            "- Crypto route head source refresh gate is: `"
            + " | ".join(
                [
                    f"blocker={crypto_route_head_source_refresh_blocker_detail or '-'}",
                    f"done_when={crypto_route_head_source_refresh_done_when or '-'}",
                ]
            )
            + "`"
        )
    if crypto_route_head_source_refresh_recipe_script:
        lines.append(
            "- Crypto route head source refresh recipe is: `"
            + " | ".join(
                [
                    f"script={crypto_route_head_source_refresh_recipe_script or '-'}",
                    f"expected_status={crypto_route_head_source_refresh_recipe_expected_status or '-'}",
                    "expected_artifact="
                    + (
                        f"{crypto_route_head_source_refresh_recipe_expected_artifact_kind or '-'}"
                        f"@{crypto_route_head_source_refresh_recipe_expected_artifact_path_hint or '-'}"
                    ),
                    f"note={crypto_route_head_source_refresh_recipe_note or '-'}",
                ]
            )
            + "`"
        )
    if crypto_route_head_source_refresh_recipe_followup_script:
        lines.append(
            "- Crypto route head source refresh follow-up is: `"
            + " | ".join(
                [
                    f"script={crypto_route_head_source_refresh_recipe_followup_script or '-'}",
                    f"verify={crypto_route_head_source_refresh_recipe_verify_hint or '-'}",
                ]
            )
            + "`"
        )
    if crypto_route_head_source_refresh_recipe_steps_brief:
        lines.append(
            "- Crypto route head source refresh pipeline is: `"
            + crypto_route_head_source_refresh_recipe_steps_brief
            + "`"
        )
    if crypto_route_head_source_refresh_recipe_step_checkpoint_brief:
        lines.append(
            "- Crypto route head source refresh checkpoint is: `"
            + crypto_route_head_source_refresh_recipe_step_checkpoint_brief
            + "`"
        )
    if source_crypto_route_refresh_reuse_brief or source_crypto_route_refresh_artifact:
        lines.append(
            "- Crypto route refresh audit is: `"
            + " | ".join(
                [
                    f"brief={source_crypto_route_refresh_reuse_brief or '-'}",
                    f"mode={source_crypto_route_refresh_native_mode or '-'}",
                    f"reused={source_crypto_route_refresh_reused_native_count or '-'}"
                    + f"/{source_crypto_route_refresh_native_step_count or '-'}",
                    f"path={source_crypto_route_refresh_artifact or '-'}",
                ]
            )
            + "`"
        )
    if source_crypto_route_refresh_reuse_gate_brief or source_crypto_route_refresh_artifact:
        lines.append(
            "- Crypto route refresh reuse gate is: `"
            + " | ".join(
                [
                    f"brief={source_crypto_route_refresh_reuse_gate_brief or '-'}",
                    f"level={source_crypto_route_refresh_reuse_level or '-'}",
                    f"blocking={source_crypto_route_refresh_reuse_gate_blocking or '-'}",
                    f"path={source_crypto_route_refresh_artifact or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_pipeline_pending_brief:
        lines.append("- Source refresh pipeline pending is: `" + operator_source_refresh_pipeline_pending_brief + "`")
    if operator_source_refresh_pipeline_relevance_brief or operator_source_refresh_pipeline_relevance_status:
        lines.append(
            "- Source refresh pipeline relevance is: `"
            + " | ".join(
                [
                    f"status={operator_source_refresh_pipeline_relevance_status or '-'}",
                    f"brief={operator_source_refresh_pipeline_relevance_brief or '-'}",
                    f"blocker={operator_source_refresh_pipeline_relevance_blocker_detail or '-'}",
                    f"done_when={operator_source_refresh_pipeline_relevance_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_pipeline_deferred_brief or operator_source_refresh_pipeline_deferred_status:
        lines.append(
            "- Source refresh pipeline deferred is: `"
            + " | ".join(
                [
                    f"status={operator_source_refresh_pipeline_deferred_status or '-'}",
                    f"brief={operator_source_refresh_pipeline_deferred_brief or '-'}",
                    f"until={operator_source_refresh_pipeline_deferred_until or '-'}",
                    f"reason={operator_source_refresh_pipeline_deferred_reason or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_pipeline_step_checkpoint_brief:
        lines.append(
            "- Source refresh pipeline checkpoint is: `"
            + operator_source_refresh_pipeline_step_checkpoint_brief
            + "`"
        )
    if operator_source_refresh_pipeline_head_name:
        lines.append(
            "- Source refresh pipeline head is: `"
            + " ".join(
                [
                    f"step={operator_source_refresh_pipeline_head_rank or '-'}",
                    f"name={operator_source_refresh_pipeline_head_name or '-'}",
                    f"state={operator_source_refresh_pipeline_head_checkpoint_state or '-'}",
                    f"artifact={operator_source_refresh_pipeline_head_expected_artifact_kind or '-'}",
                    f"current={operator_source_refresh_pipeline_head_current_artifact or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_checklist_brief:
        lines.append("- Source refresh checklist is: `" + operator_source_refresh_checklist_brief + "`")
    if operator_focus_slot_refresh_head_symbol:
        lines.append(
            "- Focus slot refresh head is: `"
            + " ".join(
                [
                    operator_focus_slot_refresh_head_symbol,
                    f"slot={operator_focus_slot_refresh_head_slot or '-'}",
                    f"action={operator_focus_slot_refresh_head_action or '-'}",
                    f"health={operator_focus_slot_refresh_head_health or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_next_symbol:
        lines.append(
            "- Next source refresh task is: `"
            + " ".join(
                [
                    operator_source_refresh_next_symbol,
                    f"slot={operator_source_refresh_next_slot or '-'}",
                    f"action={operator_source_refresh_next_action or '-'}",
                    f"kind={operator_source_refresh_next_source_kind or '-'}",
                    f"health={operator_source_refresh_next_source_health or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_next_state:
        lines.append(
            "- Next source refresh gate is: `"
            + " | ".join(
                [
                    f"state={operator_source_refresh_next_state or '-'}",
                    f"blocker={operator_source_refresh_next_blocker_detail or '-'}",
                    f"done_when={operator_source_refresh_next_done_when or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_next_recipe_script:
        lines.append(
            "- Next source refresh recipe is: `"
            + " | ".join(
                [
                    f"script={operator_source_refresh_next_recipe_script or '-'}",
                    f"expected_status={operator_source_refresh_next_recipe_expected_status or '-'}",
                    "expected_artifact="
                    + (
                        f"{operator_source_refresh_next_recipe_expected_artifact_kind or '-'}"
                        f"@{operator_source_refresh_next_recipe_expected_artifact_path_hint or '-'}"
                    ),
                    f"note={operator_source_refresh_next_recipe_note or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_next_recipe_followup_script:
        lines.append(
            "- Next source refresh follow-up is: `"
            + " | ".join(
                [
                    f"script={operator_source_refresh_next_recipe_followup_script or '-'}",
                    f"verify={operator_source_refresh_next_recipe_verify_hint or '-'}",
                ]
            )
            + "`"
        )
    if operator_source_refresh_next_recipe_steps_brief:
        lines.append("- Next source refresh pipeline is: `" + operator_source_refresh_next_recipe_steps_brief + "`")
    if operator_source_refresh_next_recipe_step_checkpoint_brief:
        lines.append(
            "- Next source refresh checkpoint is: `"
            + operator_source_refresh_next_recipe_step_checkpoint_brief
            + "`"
        )
    if operator_action_checklist_brief:
        lines.append("- Action checklist is: `" + operator_action_checklist_brief + "`")
    if operator_repair_checklist_brief:
        lines.append("- Remote repair checklist is: `" + operator_repair_checklist_brief + "`")
    if next_fill_symbol:
        lines.append("- Next commodity fill-evidence remainder is: `" + next_fill_symbol + "`.")
    if next_close_evidence_symbol:
        lines.append("- Next commodity close-evidence target is: `" + next_close_evidence_symbol + "`.")
    if commodity_remainder_focus_signal_date:
        lines.append("- Next commodity remainder signal date is: `" + commodity_remainder_focus_signal_date + "`.")
    if commodity_remainder_focus_signal_age_days:
        lines.append("- Next commodity remainder signal age is: `" + commodity_remainder_focus_signal_age_days + " days`.")
    if commodity_focus_evidence_summary:
        lines.append(
            "- Current commodity paper evidence summary: "
            + "`"
            + " ".join(
                [
                    f"entry={_fmt_num(commodity_focus_evidence_summary.get('paper_entry_price'))}",
                    f"stop={_fmt_num(commodity_focus_evidence_summary.get('paper_stop_price'))}",
                    f"target={_fmt_num(commodity_focus_evidence_summary.get('paper_target_price'))}",
                    f"quote={_fmt_num(commodity_focus_evidence_summary.get('paper_quote_usdt'))}",
                    f"status={commodity_focus_evidence_summary.get('paper_execution_status') or '-'}",
                    f"ref={commodity_focus_evidence_summary.get('paper_signal_price_reference_source') or '-'}",
                ]
            )
            + "`"
        )
    if commodity_focus_lifecycle_brief or commodity_focus_lifecycle_status:
        lines.append(
            "- Commodity focus lifecycle is: `"
            + " | ".join(
                [
                    f"status={commodity_focus_lifecycle_status or '-'}",
                    f"brief={commodity_focus_lifecycle_brief or '-'}",
                    f"blocker={commodity_focus_lifecycle_blocker_detail or '-'}",
                    f"done_when={commodity_focus_lifecycle_done_when or '-'}",
                ]
            )
            + "`"
        )
    if commodity_execution_close_evidence_status or commodity_execution_close_evidence_brief:
        lines.append(
            "- Commodity close-evidence lane is: `"
            + " | ".join(
                [
                    f"status={commodity_execution_close_evidence_status or '-'}",
                    f"brief={commodity_execution_close_evidence_brief or '-'}",
                    f"target={commodity_execution_close_evidence_target or '-'}",
                    f"symbol={commodity_execution_close_evidence_symbol or '-'}",
                    f"blocker={commodity_execution_close_evidence_blocker_detail or '-'}",
                    f"done_when={commodity_execution_close_evidence_done_when or '-'}",
                ]
            )
            + "`"
        )
    lines.append(f"- Current operator state is `{brief.get('operator_status') or '-'}`.")
    lines.append("- Formal live still remains blocked by business gates.")
    lines.append("- No live capital was touched by this refresh flow.")
    lines.extend(
        [
            "",
            "## Primary Artifact To Read First",
            "- Operator brief:",
            f"  - `{brief.get('artifact') or ''}`",
            "",
            "## Current Operator Summary",
            f"- `operator_status = {brief.get('operator_status') or '-'}`",
            f"- `operator_stack_brief = {brief.get('operator_stack_brief') or '-'}`",
            f"- `next_focus_area = {brief.get('next_focus_area') or '-'}`",
            f"- `next_focus_target = {brief.get('next_focus_target') or '-'}`",
            f"- `next_focus_action = {brief.get('next_focus_action') or '-'}`",
            f"- `next_focus_reason = {brief.get('next_focus_reason') or '-'}`",
            f"- `next_focus_state = {next_focus_state or '-'}`",
            f"- `next_focus_blocker_detail = {next_focus_blocker_detail or '-'}`",
            f"- `next_focus_done_when = {next_focus_done_when or '-'}`",
            f"- `followup_focus_area = {followup_focus_area or '-'}`",
            f"- `followup_focus_target = {followup_focus_target or '-'}`",
            f"- `followup_focus_action = {followup_focus_action or '-'}`",
            f"- `followup_focus_state = {followup_focus_state or '-'}`",
            f"- `followup_focus_blocker_detail = {followup_focus_blocker_detail or '-'}`",
            f"- `followup_focus_done_when = {followup_focus_done_when or '-'}`",
            f"- `operator_focus_slots_brief = {operator_focus_slots_brief or '-'}`",
            f"- `operator_focus_slot_sources_brief = {operator_focus_slot_sources_brief or '-'}`",
            f"- `operator_focus_slot_status_brief = {operator_focus_slot_status_brief or '-'}`",
            f"- `operator_focus_slot_recency_brief = {operator_focus_slot_recency_brief or '-'}`",
            f"- `operator_focus_slot_health_brief = {operator_focus_slot_health_brief or '-'}`",
            f"- `operator_focus_slot_refresh_backlog_brief = {operator_focus_slot_refresh_backlog_brief or '-'}`",
            f"- `operator_focus_slot_refresh_backlog_count = {operator_focus_slot_refresh_backlog_count or '-'}`",
            f"- `operator_focus_slot_refresh_backlog = {json.dumps(operator_focus_slot_refresh_backlog, ensure_ascii=False, sort_keys=True)}`",
            f"- `operator_focus_slot_ready_count = {operator_focus_slot_ready_count or '-'}`",
            f"- `operator_focus_slot_total_count = {operator_focus_slot_total_count or '-'}`",
            f"- `operator_focus_slot_promotion_gate_brief = {operator_focus_slot_promotion_gate_brief or '-'}`",
            f"- `operator_focus_slot_promotion_gate_status = {operator_focus_slot_promotion_gate_status or '-'}`",
            f"- `operator_focus_slot_promotion_gate_blocker_detail = {operator_focus_slot_promotion_gate_blocker_detail or '-'}`",
            f"- `operator_focus_slot_promotion_gate_done_when = {operator_focus_slot_promotion_gate_done_when or '-'}`",
            f"- `operator_focus_slot_actionability_backlog_brief = {operator_focus_slot_actionability_backlog_brief or '-'}`",
            f"- `operator_focus_slot_actionability_backlog_count = {operator_focus_slot_actionability_backlog_count or '-'}`",
            f"- `operator_focus_slot_actionability_backlog = {json.dumps(operator_focus_slot_actionability_backlog, ensure_ascii=False, sort_keys=True)}`",
            f"- `operator_focus_slot_actionable_count = {operator_focus_slot_actionable_count or '-'}`",
            f"- `operator_focus_slot_actionability_gate_brief = {operator_focus_slot_actionability_gate_brief or '-'}`",
            f"- `operator_focus_slot_actionability_gate_status = {operator_focus_slot_actionability_gate_status or '-'}`",
            f"- `operator_focus_slot_actionability_gate_blocker_detail = {operator_focus_slot_actionability_gate_blocker_detail or '-'}`",
            f"- `operator_focus_slot_actionability_gate_done_when = {operator_focus_slot_actionability_gate_done_when or '-'}`",
            f"- `operator_focus_slot_readiness_gate_ready_count = {operator_focus_slot_readiness_gate_ready_count or '-'}`",
            f"- `operator_focus_slot_readiness_gate_brief = {operator_focus_slot_readiness_gate_brief or '-'}`",
            f"- `operator_focus_slot_readiness_gate_status = {operator_focus_slot_readiness_gate_status or '-'}`",
            f"- `operator_focus_slot_readiness_gate_blocking_gate = {operator_focus_slot_readiness_gate_blocking_gate or '-'}`",
            f"- `operator_focus_slot_readiness_gate_blocker_detail = {operator_focus_slot_readiness_gate_blocker_detail or '-'}`",
            f"- `operator_focus_slot_readiness_gate_done_when = {operator_focus_slot_readiness_gate_done_when or '-'}`",
            f"- `operator_research_embedding_quality_status = {operator_research_embedding_quality_status or '-'}`",
            f"- `operator_research_embedding_quality_brief = {operator_research_embedding_quality_brief or '-'}`",
            f"- `operator_research_embedding_quality_blocker_detail = {operator_research_embedding_quality_blocker_detail or '-'}`",
            f"- `operator_research_embedding_quality_done_when = {operator_research_embedding_quality_done_when or '-'}`",
            f"- `operator_research_embedding_active_batches = {_list_text(operator_research_embedding_active_batches, limit=20)}`",
            f"- `operator_research_embedding_avoid_batches = {_list_text(operator_research_embedding_avoid_batches, limit=20)}`",
            f"- `operator_research_embedding_zero_trade_deprioritized_batches = {_list_text(operator_research_embedding_zero_trade_deprioritized_batches, limit=20)}`",
            f"- `operator_crypto_route_alignment_focus_area = {operator_crypto_route_alignment_focus_area or '-'}`",
            f"- `operator_crypto_route_alignment_focus_slot = {operator_crypto_route_alignment_focus_slot or '-'}`",
            f"- `operator_crypto_route_alignment_focus_symbol = {operator_crypto_route_alignment_focus_symbol or '-'}`",
            f"- `operator_crypto_route_alignment_focus_action = {operator_crypto_route_alignment_focus_action or '-'}`",
            f"- `operator_crypto_route_alignment_status = {operator_crypto_route_alignment_status or '-'}`",
            f"- `operator_crypto_route_alignment_brief = {operator_crypto_route_alignment_brief or '-'}`",
            f"- `operator_crypto_route_alignment_blocker_detail = {operator_crypto_route_alignment_blocker_detail or '-'}`",
            f"- `operator_crypto_route_alignment_done_when = {operator_crypto_route_alignment_done_when or '-'}`",
            f"- `operator_crypto_route_alignment_recovery_status = {operator_crypto_route_alignment_recovery_status or '-'}`",
            f"- `operator_crypto_route_alignment_recovery_brief = {operator_crypto_route_alignment_recovery_brief or '-'}`",
            f"- `operator_crypto_route_alignment_recovery_blocker_detail = {operator_crypto_route_alignment_recovery_blocker_detail or '-'}`",
            f"- `operator_crypto_route_alignment_recovery_done_when = {operator_crypto_route_alignment_recovery_done_when or '-'}`",
            f"- `operator_crypto_route_alignment_recovery_failed_batch_count = {operator_crypto_route_alignment_recovery_failed_batch_count or '-'}`",
            f"- `operator_crypto_route_alignment_recovery_timed_out_batch_count = {operator_crypto_route_alignment_recovery_timed_out_batch_count or '-'}`",
            f"- `operator_crypto_route_alignment_recovery_zero_trade_batches = {_list_text(operator_crypto_route_alignment_recovery_zero_trade_batches, limit=20)}`",
            f"- `operator_crypto_route_alignment_cooldown_status = {operator_crypto_route_alignment_cooldown_status or '-'}`",
            f"- `operator_crypto_route_alignment_cooldown_brief = {operator_crypto_route_alignment_cooldown_brief or '-'}`",
            f"- `operator_crypto_route_alignment_cooldown_blocker_detail = {operator_crypto_route_alignment_cooldown_blocker_detail or '-'}`",
            f"- `operator_crypto_route_alignment_cooldown_done_when = {operator_crypto_route_alignment_cooldown_done_when or '-'}`",
            f"- `operator_crypto_route_alignment_cooldown_last_research_end_date = {operator_crypto_route_alignment_cooldown_last_research_end_date or '-'}`",
            f"- `operator_crypto_route_alignment_cooldown_next_eligible_end_date = {operator_crypto_route_alignment_cooldown_next_eligible_end_date or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_status = {operator_crypto_route_alignment_recipe_status or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_brief = {operator_crypto_route_alignment_recipe_brief or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_blocker_detail = {operator_crypto_route_alignment_recipe_blocker_detail or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_done_when = {operator_crypto_route_alignment_recipe_done_when or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_ready_on_date = {operator_crypto_route_alignment_recipe_ready_on_date or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_script = {operator_crypto_route_alignment_recipe_script or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_command_hint = {operator_crypto_route_alignment_recipe_command_hint or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_expected_status = {operator_crypto_route_alignment_recipe_expected_status or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_note = {operator_crypto_route_alignment_recipe_note or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_followup_script = {operator_crypto_route_alignment_recipe_followup_script or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_followup_command_hint = {operator_crypto_route_alignment_recipe_followup_command_hint or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_verify_hint = {operator_crypto_route_alignment_recipe_verify_hint or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_window_days = {operator_crypto_route_alignment_recipe_window_days or '-'}`",
            f"- `operator_crypto_route_alignment_recipe_target_batches = {_list_text(operator_crypto_route_alignment_recipe_target_batches, limit=20)}`",
            f"- `operator_source_refresh_queue_brief = {operator_source_refresh_queue_brief or '-'}`",
            f"- `operator_source_refresh_queue_count = {operator_source_refresh_queue_count or '-'}`",
            f"- `operator_source_refresh_queue = {json.dumps(operator_source_refresh_queue, ensure_ascii=False, sort_keys=True)}`",
            f"- `operator_source_refresh_checklist_brief = {operator_source_refresh_checklist_brief or '-'}`",
            f"- `operator_source_refresh_checklist = {json.dumps(operator_source_refresh_checklist, ensure_ascii=False, sort_keys=True)}`",
            f"- `operator_source_refresh_pipeline_steps_brief = {operator_source_refresh_pipeline_steps_brief or '-'}`",
            f"- `operator_source_refresh_pipeline_step_checkpoint_brief = {operator_source_refresh_pipeline_step_checkpoint_brief or '-'}`",
            f"- `operator_source_refresh_pipeline_pending_brief = {operator_source_refresh_pipeline_pending_brief or '-'}`",
            f"- `operator_source_refresh_pipeline_pending_count = {operator_source_refresh_pipeline_pending_count or '-'}`",
            f"- `operator_source_refresh_pipeline_head_rank = {operator_source_refresh_pipeline_head_rank or '-'}`",
            f"- `operator_source_refresh_pipeline_head_name = {operator_source_refresh_pipeline_head_name or '-'}`",
            f"- `operator_source_refresh_pipeline_head_checkpoint_state = {operator_source_refresh_pipeline_head_checkpoint_state or '-'}`",
            f"- `operator_source_refresh_pipeline_head_expected_artifact_kind = {operator_source_refresh_pipeline_head_expected_artifact_kind or '-'}`",
            f"- `operator_source_refresh_pipeline_head_current_artifact = {operator_source_refresh_pipeline_head_current_artifact or '-'}`",
            f"- `operator_source_refresh_pipeline_relevance_status = {operator_source_refresh_pipeline_relevance_status or '-'}`",
            f"- `operator_source_refresh_pipeline_relevance_brief = {operator_source_refresh_pipeline_relevance_brief or '-'}`",
            f"- `operator_source_refresh_pipeline_relevance_blocker_detail = {operator_source_refresh_pipeline_relevance_blocker_detail or '-'}`",
            f"- `operator_source_refresh_pipeline_relevance_done_when = {operator_source_refresh_pipeline_relevance_done_when or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_brief = {operator_source_refresh_pipeline_deferred_brief or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_count = {operator_source_refresh_pipeline_deferred_count or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_status = {operator_source_refresh_pipeline_deferred_status or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_until = {operator_source_refresh_pipeline_deferred_until or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_reason = {operator_source_refresh_pipeline_deferred_reason or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_head_rank = {operator_source_refresh_pipeline_deferred_head_rank or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_head_name = {operator_source_refresh_pipeline_deferred_head_name or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_head_checkpoint_state = {operator_source_refresh_pipeline_deferred_head_checkpoint_state or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_head_expected_artifact_kind = {operator_source_refresh_pipeline_deferred_head_expected_artifact_kind or '-'}`",
            f"- `operator_source_refresh_pipeline_deferred_head_current_artifact = {operator_source_refresh_pipeline_deferred_head_current_artifact or '-'}`",
            f"- `operator_focus_slots = {json.dumps(operator_focus_slots, ensure_ascii=False, sort_keys=True)}`",
            f"- `next_focus_source_kind = {next_focus_source_kind or '-'}`",
            f"- `next_focus_source_artifact = {next_focus_source_artifact or '-'}`",
            f"- `next_focus_source_status = {next_focus_source_status or '-'}`",
            f"- `next_focus_source_as_of = {next_focus_source_as_of or '-'}`",
            f"- `next_focus_source_age_minutes = {next_focus_source_age_minutes or '-'}`",
            f"- `next_focus_source_recency = {next_focus_source_recency or '-'}`",
            f"- `next_focus_source_health = {next_focus_source_health or '-'}`",
            f"- `next_focus_source_refresh_action = {next_focus_source_refresh_action or '-'}`",
            f"- `followup_focus_source_kind = {followup_focus_source_kind or '-'}`",
            f"- `followup_focus_source_artifact = {followup_focus_source_artifact or '-'}`",
            f"- `followup_focus_source_status = {followup_focus_source_status or '-'}`",
            f"- `followup_focus_source_as_of = {followup_focus_source_as_of or '-'}`",
            f"- `followup_focus_source_age_minutes = {followup_focus_source_age_minutes or '-'}`",
            f"- `followup_focus_source_recency = {followup_focus_source_recency or '-'}`",
            f"- `followup_focus_source_health = {followup_focus_source_health or '-'}`",
            f"- `followup_focus_source_refresh_action = {followup_focus_source_refresh_action or '-'}`",
            f"- `secondary_focus_source_kind = {secondary_focus_source_kind or '-'}`",
            f"- `secondary_focus_source_artifact = {secondary_focus_source_artifact or '-'}`",
            f"- `secondary_focus_source_status = {secondary_focus_source_status or '-'}`",
            f"- `secondary_focus_source_as_of = {secondary_focus_source_as_of or '-'}`",
            f"- `secondary_focus_source_age_minutes = {secondary_focus_source_age_minutes or '-'}`",
            f"- `secondary_focus_source_recency = {secondary_focus_source_recency or '-'}`",
            f"- `secondary_focus_source_health = {secondary_focus_source_health or '-'}`",
            f"- `secondary_focus_source_refresh_action = {secondary_focus_source_refresh_action or '-'}`",
            f"- `operator_focus_slot_refresh_head_slot = {operator_focus_slot_refresh_head_slot or '-'}`",
            f"- `operator_focus_slot_refresh_head_symbol = {operator_focus_slot_refresh_head_symbol or '-'}`",
            f"- `operator_focus_slot_refresh_head_action = {operator_focus_slot_refresh_head_action or '-'}`",
            f"- `operator_focus_slot_refresh_head_health = {operator_focus_slot_refresh_head_health or '-'}`",
            f"- `operator_source_refresh_next_slot = {operator_source_refresh_next_slot or '-'}`",
            f"- `operator_source_refresh_next_symbol = {operator_source_refresh_next_symbol or '-'}`",
            f"- `operator_source_refresh_next_action = {operator_source_refresh_next_action or '-'}`",
            f"- `operator_source_refresh_next_source_kind = {operator_source_refresh_next_source_kind or '-'}`",
            f"- `operator_source_refresh_next_source_health = {operator_source_refresh_next_source_health or '-'}`",
            f"- `operator_source_refresh_next_source_artifact = {operator_source_refresh_next_source_artifact or '-'}`",
            f"- `operator_source_refresh_next_state = {operator_source_refresh_next_state or '-'}`",
            f"- `operator_source_refresh_next_blocker_detail = {operator_source_refresh_next_blocker_detail or '-'}`",
            f"- `operator_source_refresh_next_done_when = {operator_source_refresh_next_done_when or '-'}`",
            f"- `operator_source_refresh_next_recipe_script = {operator_source_refresh_next_recipe_script or '-'}`",
            f"- `operator_source_refresh_next_recipe_command_hint = {operator_source_refresh_next_recipe_command_hint or '-'}`",
            f"- `operator_source_refresh_next_recipe_expected_status = {operator_source_refresh_next_recipe_expected_status or '-'}`",
            f"- `operator_source_refresh_next_recipe_expected_artifact_kind = {operator_source_refresh_next_recipe_expected_artifact_kind or '-'}`",
            f"- `operator_source_refresh_next_recipe_expected_artifact_path_hint = {operator_source_refresh_next_recipe_expected_artifact_path_hint or '-'}`",
            f"- `operator_source_refresh_next_recipe_note = {operator_source_refresh_next_recipe_note or '-'}`",
            f"- `operator_source_refresh_next_recipe_followup_script = {operator_source_refresh_next_recipe_followup_script or '-'}`",
            f"- `operator_source_refresh_next_recipe_followup_command_hint = {operator_source_refresh_next_recipe_followup_command_hint or '-'}`",
            f"- `operator_source_refresh_next_recipe_verify_hint = {operator_source_refresh_next_recipe_verify_hint or '-'}`",
            f"- `operator_source_refresh_next_recipe_steps_brief = {operator_source_refresh_next_recipe_steps_brief or '-'}`",
            f"- `operator_source_refresh_next_recipe_step_checkpoint_brief = {operator_source_refresh_next_recipe_step_checkpoint_brief or '-'}`",
            f"- `operator_source_refresh_next_recipe_steps = {json.dumps(operator_source_refresh_next_recipe_steps, ensure_ascii=False, sort_keys=True)}`",
            f"- `crypto_route_head_source_refresh_status = {crypto_route_head_source_refresh_status or '-'}`",
            f"- `crypto_route_head_source_refresh_brief = {crypto_route_head_source_refresh_brief or '-'}`",
            f"- `crypto_route_head_source_refresh_slot = {crypto_route_head_source_refresh_slot or '-'}`",
            f"- `crypto_route_head_source_refresh_symbol = {crypto_route_head_source_refresh_symbol or '-'}`",
            f"- `crypto_route_head_source_refresh_action = {crypto_route_head_source_refresh_action or '-'}`",
            f"- `crypto_route_head_source_refresh_source_kind = {crypto_route_head_source_refresh_source_kind or '-'}`",
            f"- `crypto_route_head_source_refresh_source_health = {crypto_route_head_source_refresh_source_health or '-'}`",
            f"- `crypto_route_head_source_refresh_source_artifact = {crypto_route_head_source_refresh_source_artifact or '-'}`",
            f"- `crypto_route_head_source_refresh_blocker_detail = {crypto_route_head_source_refresh_blocker_detail or '-'}`",
            f"- `crypto_route_head_source_refresh_done_when = {crypto_route_head_source_refresh_done_when or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_script = {crypto_route_head_source_refresh_recipe_script or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_command_hint = {crypto_route_head_source_refresh_recipe_command_hint or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_expected_status = {crypto_route_head_source_refresh_recipe_expected_status or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_expected_artifact_kind = {crypto_route_head_source_refresh_recipe_expected_artifact_kind or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_expected_artifact_path_hint = {crypto_route_head_source_refresh_recipe_expected_artifact_path_hint or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_note = {crypto_route_head_source_refresh_recipe_note or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_followup_script = {crypto_route_head_source_refresh_recipe_followup_script or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_followup_command_hint = {crypto_route_head_source_refresh_recipe_followup_command_hint or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_verify_hint = {crypto_route_head_source_refresh_recipe_verify_hint or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_steps_brief = {crypto_route_head_source_refresh_recipe_steps_brief or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_step_checkpoint_brief = {crypto_route_head_source_refresh_recipe_step_checkpoint_brief or '-'}`",
            f"- `crypto_route_head_source_refresh_recipe_steps = {json.dumps(crypto_route_head_source_refresh_recipe_steps, ensure_ascii=False, sort_keys=True)}`",
            f"- `source_crypto_route_refresh_artifact = {source_crypto_route_refresh_artifact or '-'}`",
            f"- `source_crypto_route_refresh_status = {source_crypto_route_refresh_status or '-'}`",
            f"- `source_crypto_route_refresh_as_of = {source_crypto_route_refresh_as_of or '-'}`",
            f"- `source_crypto_route_refresh_native_mode = {source_crypto_route_refresh_native_mode or '-'}`",
            f"- `source_crypto_route_refresh_native_step_count = {source_crypto_route_refresh_native_step_count or '-'}`",
            f"- `source_crypto_route_refresh_reused_native_count = {source_crypto_route_refresh_reused_native_count or '-'}`",
            f"- `source_crypto_route_refresh_missing_reused_count = {source_crypto_route_refresh_missing_reused_count or '-'}`",
            f"- `source_crypto_route_refresh_reuse_status = {source_crypto_route_refresh_reuse_status or '-'}`",
            f"- `source_crypto_route_refresh_reuse_brief = {source_crypto_route_refresh_reuse_brief or '-'}`",
            f"- `source_crypto_route_refresh_reuse_note = {source_crypto_route_refresh_reuse_note or '-'}`",
            f"- `source_crypto_route_refresh_reuse_done_when = {source_crypto_route_refresh_reuse_done_when or '-'}`",
            f"- `source_crypto_route_refresh_reuse_level = {source_crypto_route_refresh_reuse_level or '-'}`",
            f"- `source_crypto_route_refresh_reuse_gate_status = {source_crypto_route_refresh_reuse_gate_status or '-'}`",
            f"- `source_crypto_route_refresh_reuse_gate_brief = {source_crypto_route_refresh_reuse_gate_brief or '-'}`",
            f"- `source_crypto_route_refresh_reuse_gate_blocking = {source_crypto_route_refresh_reuse_gate_blocking or '-'}`",
            f"- `source_crypto_route_refresh_reuse_gate_blocker_detail = {source_crypto_route_refresh_reuse_gate_blocker_detail or '-'}`",
            f"- `source_crypto_route_refresh_reuse_gate_done_when = {source_crypto_route_refresh_reuse_gate_done_when or '-'}`",
            f"- `source_remote_live_handoff_artifact = {brief.get('source_remote_live_handoff_artifact') or '-'}`",
            f"- `source_remote_live_handoff_status = {brief.get('source_remote_live_handoff_status') or '-'}`",
            f"- `source_remote_live_handoff_ready_check_scope_brief = {brief.get('source_remote_live_handoff_ready_check_scope_brief') or '-'}`",
            f"- `source_remote_live_handoff_account_scope_alignment_brief = {brief.get('source_remote_live_handoff_account_scope_alignment_brief') or '-'}`",
            f"- `source_live_gate_blocker_artifact = {brief.get('source_live_gate_blocker_artifact') or '-'}`",
            f"- `source_live_gate_blocker_as_of = {brief.get('source_live_gate_blocker_as_of') or '-'}`",
            f"- `source_live_gate_blocker_live_decision = {brief.get('source_live_gate_blocker_live_decision') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_diagnosis_status = {brief.get('source_live_gate_blocker_remote_live_diagnosis_status') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_diagnosis_brief = {brief.get('source_live_gate_blocker_remote_live_diagnosis_brief') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_diagnosis_blocker_detail = {brief.get('source_live_gate_blocker_remote_live_diagnosis_blocker_detail') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_diagnosis_done_when = {brief.get('source_live_gate_blocker_remote_live_diagnosis_done_when') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_operator_alignment_status = {brief.get('source_live_gate_blocker_remote_live_operator_alignment_status') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_operator_alignment_brief = {brief.get('source_live_gate_blocker_remote_live_operator_alignment_brief') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_operator_alignment_blocker_detail = {brief.get('source_live_gate_blocker_remote_live_operator_alignment_blocker_detail') or '-'}`",
            f"- `source_live_gate_blocker_remote_live_operator_alignment_done_when = {brief.get('source_live_gate_blocker_remote_live_operator_alignment_done_when') or '-'}`",
            f"- `source_cross_market_operator_state_remote_live_takeover_gate_status = {brief.get('source_cross_market_operator_state_remote_live_takeover_gate_status') or '-'}`",
            f"- `source_cross_market_operator_state_remote_live_takeover_gate_brief = {brief.get('source_cross_market_operator_state_remote_live_takeover_gate_brief') or '-'}`",
            f"- `source_cross_market_operator_state_remote_live_takeover_gate_blocker_detail = {brief.get('source_cross_market_operator_state_remote_live_takeover_gate_blocker_detail') or '-'}`",
            f"- `source_cross_market_operator_state_remote_live_takeover_gate_done_when = {brief.get('source_cross_market_operator_state_remote_live_takeover_gate_done_when') or '-'}`",
            f"- `source_brooks_route_report_artifact = {source_brooks_route_report_artifact or '-'}`",
            f"- `source_brooks_route_report_status = {source_brooks_route_report_status or '-'}`",
            f"- `source_brooks_route_report_as_of = {source_brooks_route_report_as_of or '-'}`",
            f"- `source_brooks_route_report_selected_routes_brief = {source_brooks_route_report_selected_routes_brief or '-'}`",
            f"- `source_brooks_route_report_candidate_count = {source_brooks_route_report_candidate_count or '-'}`",
            f"- `source_brooks_route_report_head_symbol = {source_brooks_route_report_head_symbol or '-'}`",
            f"- `source_brooks_route_report_head_strategy_id = {source_brooks_route_report_head_strategy_id or '-'}`",
            f"- `source_brooks_route_report_head_direction = {source_brooks_route_report_head_direction or '-'}`",
            f"- `source_brooks_route_report_head_bridge_status = {source_brooks_route_report_head_bridge_status or '-'}`",
            f"- `source_brooks_route_report_head_blocker_detail = {source_brooks_route_report_head_blocker_detail or '-'}`",
            f"- `source_brooks_execution_plan_artifact = {source_brooks_execution_plan_artifact or '-'}`",
            f"- `source_brooks_execution_plan_status = {source_brooks_execution_plan_status or '-'}`",
            f"- `source_brooks_execution_plan_as_of = {source_brooks_execution_plan_as_of or '-'}`",
            f"- `source_brooks_execution_plan_actionable_count = {source_brooks_execution_plan_actionable_count or '-'}`",
            f"- `source_brooks_execution_plan_blocked_count = {source_brooks_execution_plan_blocked_count or '-'}`",
            f"- `source_brooks_execution_plan_head_symbol = {source_brooks_execution_plan_head_symbol or '-'}`",
            f"- `source_brooks_execution_plan_head_strategy_id = {source_brooks_execution_plan_head_strategy_id or '-'}`",
            f"- `source_brooks_execution_plan_head_plan_status = {source_brooks_execution_plan_head_plan_status or '-'}`",
            f"- `source_brooks_execution_plan_head_execution_action = {source_brooks_execution_plan_head_execution_action or '-'}`",
            f"- `source_brooks_execution_plan_head_entry_price = {source_brooks_execution_plan_head_entry_price or '-'}`",
            f"- `source_brooks_execution_plan_head_stop_price = {source_brooks_execution_plan_head_stop_price or '-'}`",
            f"- `source_brooks_execution_plan_head_target_price = {source_brooks_execution_plan_head_target_price or '-'}`",
            f"- `source_brooks_execution_plan_head_rr_ratio = {source_brooks_execution_plan_head_rr_ratio or '-'}`",
            f"- `source_brooks_execution_plan_head_blocker_detail = {source_brooks_execution_plan_head_blocker_detail or '-'}`",
            f"- `source_brooks_structure_review_queue_artifact = {source_brooks_structure_review_queue_artifact or '-'}`",
            f"- `source_brooks_structure_review_queue_status = {source_brooks_structure_review_queue_status or '-'}`",
            f"- `source_brooks_structure_review_queue_as_of = {source_brooks_structure_review_queue_as_of or '-'}`",
            f"- `source_brooks_structure_review_queue_brief = {source_brooks_structure_review_queue_brief or '-'}`",
            f"- `brooks_structure_review_status = {brooks_structure_review_status or '-'}`",
            f"- `brooks_structure_review_brief = {brooks_structure_review_brief or '-'}`",
            f"- `brooks_structure_review_queue_status = {brooks_structure_review_queue_status or '-'}`",
            f"- `brooks_structure_review_queue_count = {brooks_structure_review_queue_count or '-'}`",
            f"- `brooks_structure_review_queue_brief = {brooks_structure_review_queue_brief or '-'}`",
            f"- `brooks_structure_review_priority_status = {brooks_structure_review_priority_status or '-'}`",
            f"- `brooks_structure_review_priority_brief = {brooks_structure_review_priority_brief or '-'}`",
            f"- `brooks_structure_review_queue = {json.dumps(brooks_structure_review_queue, ensure_ascii=False, sort_keys=True)}`",
            f"- `brooks_structure_review_head_rank = {brooks_structure_review_head_rank or '-'}`",
            f"- `brooks_structure_review_head_symbol = {brooks_structure_review_head_symbol or '-'}`",
            f"- `brooks_structure_review_head_strategy_id = {brooks_structure_review_head_strategy_id or '-'}`",
            f"- `brooks_structure_review_head_direction = {brooks_structure_review_head_direction or '-'}`",
            f"- `brooks_structure_review_head_tier = {brooks_structure_review_head_tier or '-'}`",
            f"- `brooks_structure_review_head_plan_status = {brooks_structure_review_head_plan_status or '-'}`",
            f"- `brooks_structure_review_head_action = {brooks_structure_review_head_action or '-'}`",
            f"- `brooks_structure_review_head_route_selection_score = {brooks_structure_review_head_route_selection_score or '-'}`",
            f"- `brooks_structure_review_head_signal_score = {brooks_structure_review_head_signal_score or '-'}`",
            f"- `brooks_structure_review_head_signal_age_bars = {brooks_structure_review_head_signal_age_bars or '-'}`",
            f"- `brooks_structure_review_head_priority_score = {brooks_structure_review_head_priority_score or '-'}`",
            f"- `brooks_structure_review_head_priority_tier = {brooks_structure_review_head_priority_tier or '-'}`",
            f"- `brooks_structure_review_head_blocker_detail = {brooks_structure_review_head_blocker_detail or '-'}`",
            f"- `brooks_structure_review_head_done_when = {brooks_structure_review_head_done_when or '-'}`",
            f"- `brooks_structure_review_blocker_detail = {brooks_structure_review_blocker_detail or '-'}`",
            f"- `brooks_structure_review_done_when = {brooks_structure_review_done_when or '-'}`",
            f"- `brooks_structure_operator_status = {brooks_structure_operator_status or '-'}`",
            f"- `brooks_structure_operator_brief = {brooks_structure_operator_brief or '-'}`",
            f"- `brooks_structure_operator_head_symbol = {brooks_structure_operator_head_symbol or '-'}`",
            f"- `brooks_structure_operator_head_strategy_id = {brooks_structure_operator_head_strategy_id or '-'}`",
            f"- `brooks_structure_operator_head_direction = {brooks_structure_operator_head_direction or '-'}`",
            f"- `brooks_structure_operator_head_action = {brooks_structure_operator_head_action or '-'}`",
            f"- `brooks_structure_operator_head_plan_status = {brooks_structure_operator_head_plan_status or '-'}`",
            f"- `brooks_structure_operator_head_priority_score = {brooks_structure_operator_head_priority_score or '-'}`",
            f"- `brooks_structure_operator_head_priority_tier = {brooks_structure_operator_head_priority_tier or '-'}`",
            f"- `brooks_structure_operator_backlog_count = {brooks_structure_operator_backlog_count or '-'}`",
            f"- `brooks_structure_operator_backlog_brief = {brooks_structure_operator_backlog_brief or '-'}`",
            f"- `brooks_structure_operator_blocker_detail = {brooks_structure_operator_blocker_detail or '-'}`",
            f"- `brooks_structure_operator_done_when = {brooks_structure_operator_done_when or '-'}`",
            f"- `cross_market_operator_head_status = {cross_market_operator_head_status or '-'}`",
            f"- `cross_market_operator_head_brief = {cross_market_operator_head_brief or '-'}`",
            f"- `cross_market_operator_head_area = {cross_market_operator_head_area or '-'}`",
            f"- `cross_market_operator_head_symbol = {cross_market_operator_head_symbol or '-'}`",
            f"- `cross_market_operator_head_action = {cross_market_operator_head_action or '-'}`",
            f"- `cross_market_operator_head_state = {cross_market_operator_head_state or '-'}`",
            f"- `cross_market_operator_head_priority_score = {cross_market_operator_head_priority_score or '-'}`",
            f"- `cross_market_operator_head_priority_tier = {cross_market_operator_head_priority_tier or '-'}`",
            f"- `cross_market_operator_head_blocker_detail = {cross_market_operator_head_blocker_detail or '-'}`",
            f"- `cross_market_operator_head_done_when = {cross_market_operator_head_done_when or '-'}`",
            f"- `cross_market_remote_live_takeover_gate_status = {cross_market_remote_live_takeover_gate_status or '-'}`",
            f"- `cross_market_remote_live_takeover_gate_brief = {cross_market_remote_live_takeover_gate_brief or '-'}`",
            f"- `cross_market_remote_live_takeover_gate_blocker_detail = {cross_market_remote_live_takeover_gate_blocker_detail or '-'}`",
            f"- `cross_market_remote_live_takeover_gate_done_when = {cross_market_remote_live_takeover_gate_done_when or '-'}`",
            f"- `cross_market_remote_live_takeover_clearing_status = {cross_market_remote_live_takeover_clearing_status or '-'}`",
            f"- `cross_market_remote_live_takeover_clearing_brief = {cross_market_remote_live_takeover_clearing_brief or '-'}`",
            f"- `cross_market_remote_live_takeover_clearing_blocker_detail = {cross_market_remote_live_takeover_clearing_blocker_detail or '-'}`",
            f"- `cross_market_remote_live_takeover_clearing_done_when = {cross_market_remote_live_takeover_clearing_done_when or '-'}`",
            f"- `remote_live_takeover_repair_queue_status = {remote_live_takeover_repair_queue_status or '-'}`",
            f"- `remote_live_takeover_repair_queue_brief = {remote_live_takeover_repair_queue_brief or '-'}`",
            f"- `remote_live_takeover_repair_queue_queue_brief = {remote_live_takeover_repair_queue_queue_brief or '-'}`",
            f"- `remote_live_takeover_repair_queue_count = {remote_live_takeover_repair_queue_count or '-'}`",
            f"- `remote_live_takeover_repair_queue_head_area = {remote_live_takeover_repair_queue_head_area or '-'}`",
            f"- `remote_live_takeover_repair_queue_head_code = {remote_live_takeover_repair_queue_head_code or '-'}`",
            f"- `remote_live_takeover_repair_queue_head_action = {remote_live_takeover_repair_queue_head_action or '-'}`",
            f"- `remote_live_takeover_repair_queue_head_priority_score = {remote_live_takeover_repair_queue_head_priority_score or '-'}`",
            f"- `remote_live_takeover_repair_queue_head_priority_tier = {remote_live_takeover_repair_queue_head_priority_tier or '-'}`",
            f"- `remote_live_takeover_repair_queue_head_command = {remote_live_takeover_repair_queue_head_command or '-'}`",
            f"- `remote_live_takeover_repair_queue_head_clear_when = {remote_live_takeover_repair_queue_head_clear_when or '-'}`",
            f"- `remote_live_takeover_repair_queue_done_when = {remote_live_takeover_repair_queue_done_when or '-'}`",
            f"- `cross_market_operator_repair_head_status = {cross_market_operator_repair_head_status or '-'}`",
            f"- `cross_market_operator_repair_head_brief = {cross_market_operator_repair_head_brief or '-'}`",
            f"- `cross_market_operator_repair_head_area = {cross_market_operator_repair_head_area or '-'}`",
            f"- `cross_market_operator_repair_head_code = {cross_market_operator_repair_head_code or '-'}`",
            f"- `cross_market_operator_repair_head_action = {cross_market_operator_repair_head_action or '-'}`",
            f"- `cross_market_operator_repair_head_priority_score = {cross_market_operator_repair_head_priority_score or '-'}`",
            f"- `cross_market_operator_repair_head_priority_tier = {cross_market_operator_repair_head_priority_tier or '-'}`",
            f"- `cross_market_operator_repair_head_command = {cross_market_operator_repair_head_command or '-'}`",
            f"- `cross_market_operator_repair_head_clear_when = {cross_market_operator_repair_head_clear_when or '-'}`",
            f"- `cross_market_operator_repair_head_done_when = {cross_market_operator_repair_head_done_when or '-'}`",
            f"- `cross_market_operator_repair_backlog_status = {cross_market_operator_repair_backlog_status or '-'}`",
            f"- `cross_market_operator_repair_backlog_brief = {cross_market_operator_repair_backlog_brief or '-'}`",
            f"- `cross_market_operator_repair_backlog_count = {cross_market_operator_repair_backlog_count or '-'}`",
            f"- `cross_market_operator_repair_backlog_priority_total = {cross_market_operator_repair_backlog_priority_total or '-'}`",
            f"- `cross_market_operator_repair_backlog_done_when = {cross_market_operator_repair_backlog_done_when or '-'}`",
            f"- `cross_market_operator_backlog_count = {cross_market_operator_backlog_count or '-'}`",
            f"- `cross_market_operator_backlog_brief = {cross_market_operator_backlog_brief or '-'}`",
            f"- `cross_market_operator_backlog_state_brief = {cross_market_operator_backlog_state_brief or '-'}`",
            f"- `cross_market_operator_backlog_priority_totals_brief = {cross_market_operator_backlog_priority_totals_brief or '-'}`",
            f"- `cross_market_operator_lane_heads_brief = {cross_market_operator_lane_heads_brief or '-'}`",
            f"- `cross_market_operator_lane_priority_order_brief = {cross_market_operator_lane_priority_order_brief or '-'}`",
            f"- `cross_market_operator_waiting_lane_status = {cross_market_operator_waiting_lane_status or '-'}`",
            f"- `cross_market_operator_waiting_lane_count = {cross_market_operator_waiting_lane_count or '-'}`",
            f"- `cross_market_operator_waiting_lane_brief = {cross_market_operator_waiting_lane_brief or '-'}`",
            f"- `cross_market_operator_waiting_lane_priority_total = {cross_market_operator_waiting_lane_priority_total or '-'}`",
            f"- `cross_market_operator_waiting_lane_head_symbol = {cross_market_operator_waiting_lane_head_symbol or '-'}`",
            f"- `cross_market_operator_waiting_lane_head_action = {cross_market_operator_waiting_lane_head_action or '-'}`",
            f"- `cross_market_operator_waiting_lane_head_priority_score = {cross_market_operator_waiting_lane_head_priority_score or '-'}`",
            f"- `cross_market_operator_waiting_lane_head_priority_tier = {cross_market_operator_waiting_lane_head_priority_tier or '-'}`",
            f"- `cross_market_operator_review_lane_status = {cross_market_operator_review_lane_status or '-'}`",
            f"- `cross_market_operator_review_lane_count = {cross_market_operator_review_lane_count or '-'}`",
            f"- `cross_market_operator_review_lane_brief = {cross_market_operator_review_lane_brief or '-'}`",
            f"- `cross_market_operator_review_lane_priority_total = {cross_market_operator_review_lane_priority_total or '-'}`",
            f"- `cross_market_operator_review_lane_head_symbol = {cross_market_operator_review_lane_head_symbol or '-'}`",
            f"- `cross_market_operator_review_lane_head_action = {cross_market_operator_review_lane_head_action or '-'}`",
            f"- `cross_market_operator_review_lane_head_priority_score = {cross_market_operator_review_lane_head_priority_score or '-'}`",
            f"- `cross_market_operator_review_lane_head_priority_tier = {cross_market_operator_review_lane_head_priority_tier or '-'}`",
            f"- `cross_market_operator_watch_lane_status = {cross_market_operator_watch_lane_status or '-'}`",
            f"- `cross_market_operator_watch_lane_count = {cross_market_operator_watch_lane_count or '-'}`",
            f"- `cross_market_operator_watch_lane_brief = {cross_market_operator_watch_lane_brief or '-'}`",
            f"- `cross_market_operator_watch_lane_priority_total = {cross_market_operator_watch_lane_priority_total or '-'}`",
            f"- `cross_market_operator_watch_lane_head_symbol = {cross_market_operator_watch_lane_head_symbol or '-'}`",
            f"- `cross_market_operator_watch_lane_head_action = {cross_market_operator_watch_lane_head_action or '-'}`",
            f"- `cross_market_operator_watch_lane_head_priority_score = {cross_market_operator_watch_lane_head_priority_score or '-'}`",
            f"- `cross_market_operator_watch_lane_head_priority_tier = {cross_market_operator_watch_lane_head_priority_tier or '-'}`",
            f"- `cross_market_operator_blocked_lane_status = {cross_market_operator_blocked_lane_status or '-'}`",
            f"- `cross_market_operator_blocked_lane_count = {cross_market_operator_blocked_lane_count or '-'}`",
            f"- `cross_market_operator_blocked_lane_brief = {cross_market_operator_blocked_lane_brief or '-'}`",
            f"- `cross_market_operator_blocked_lane_priority_total = {cross_market_operator_blocked_lane_priority_total or '-'}`",
            f"- `cross_market_operator_blocked_lane_head_symbol = {cross_market_operator_blocked_lane_head_symbol or '-'}`",
            f"- `cross_market_operator_blocked_lane_head_action = {cross_market_operator_blocked_lane_head_action or '-'}`",
            f"- `cross_market_operator_blocked_lane_head_priority_score = {cross_market_operator_blocked_lane_head_priority_score or '-'}`",
            f"- `cross_market_operator_blocked_lane_head_priority_tier = {cross_market_operator_blocked_lane_head_priority_tier or '-'}`",
            f"- `cross_market_operator_repair_lane_status = {cross_market_operator_repair_lane_status or '-'}`",
            f"- `cross_market_operator_repair_lane_count = {cross_market_operator_repair_lane_count or '-'}`",
            f"- `cross_market_operator_repair_lane_brief = {cross_market_operator_repair_lane_brief or '-'}`",
            f"- `cross_market_operator_repair_lane_priority_total = {cross_market_operator_repair_lane_priority_total or '-'}`",
            f"- `cross_market_operator_repair_lane_head_symbol = {cross_market_operator_repair_lane_head_symbol or '-'}`",
            f"- `cross_market_operator_repair_lane_head_action = {cross_market_operator_repair_lane_head_action or '-'}`",
            f"- `cross_market_operator_repair_lane_head_priority_score = {cross_market_operator_repair_lane_head_priority_score or '-'}`",
            f"- `cross_market_operator_repair_lane_head_priority_tier = {cross_market_operator_repair_lane_head_priority_tier or '-'}`",
            f"- `cross_market_review_head_status = {cross_market_review_head_status or '-'}`",
            f"- `cross_market_review_head_brief = {cross_market_review_head_brief or '-'}`",
            f"- `cross_market_review_head_area = {cross_market_review_head_area or '-'}`",
            f"- `cross_market_review_head_symbol = {cross_market_review_head_symbol or '-'}`",
            f"- `cross_market_review_head_action = {cross_market_review_head_action or '-'}`",
            f"- `cross_market_review_head_priority_score = {cross_market_review_head_priority_score or '-'}`",
            f"- `cross_market_review_head_priority_tier = {cross_market_review_head_priority_tier or '-'}`",
            f"- `cross_market_review_head_blocker_detail = {cross_market_review_head_blocker_detail or '-'}`",
            f"- `cross_market_review_head_done_when = {cross_market_review_head_done_when or '-'}`",
            f"- `source_system_time_sync_repair_plan_artifact = {source_system_time_sync_repair_plan_artifact or '-'}`",
            f"- `source_system_time_sync_repair_plan_status = {source_system_time_sync_repair_plan_status or '-'}`",
            f"- `source_system_time_sync_repair_plan_brief = {source_system_time_sync_repair_plan_brief or '-'}`",
            f"- `source_system_time_sync_repair_plan_done_when = {source_system_time_sync_repair_plan_done_when or '-'}`",
            f"- `source_system_time_sync_repair_plan_admin_required = {source_system_time_sync_repair_plan_admin_required}`",
            f"- `source_system_time_sync_repair_verification_artifact = {source_system_time_sync_repair_verification_artifact or '-'}`",
            f"- `source_system_time_sync_repair_verification_status = {source_system_time_sync_repair_verification_status or '-'}`",
            f"- `source_system_time_sync_repair_verification_brief = {source_system_time_sync_repair_verification_brief or '-'}`",
            f"- `source_system_time_sync_repair_verification_cleared = {source_system_time_sync_repair_verification_cleared}`",
            f"- `source_openclaw_orderflow_blueprint_artifact = {source_openclaw_orderflow_blueprint_artifact or '-'}`",
            f"- `source_openclaw_orderflow_blueprint_status = {source_openclaw_orderflow_blueprint_status or '-'}`",
            f"- `source_openclaw_orderflow_blueprint_brief = {source_openclaw_orderflow_blueprint_brief or '-'}`",
            f"- `source_openclaw_orderflow_blueprint_current_life_stage = {source_openclaw_orderflow_blueprint_current_life_stage or '-'}`",
            f"- `source_openclaw_orderflow_blueprint_target_life_stage = {source_openclaw_orderflow_blueprint_target_life_stage or '-'}`",
            f"- `source_openclaw_orderflow_blueprint_top_backlog_title = {source_openclaw_orderflow_blueprint_top_backlog_title or '-'}`",
            f"- `source_openclaw_orderflow_blueprint_top_backlog_target_artifact = {source_openclaw_orderflow_blueprint_top_backlog_target_artifact or '-'}`",
            f"- `source_openclaw_orderflow_blueprint_remote_execution_ack_brief = {source_openclaw_orderflow_blueprint_remote_execution_ack_brief or '-'}`",
            f"- `cross_market_review_backlog_count = {cross_market_review_backlog_count or '-'}`",
            f"- `cross_market_review_backlog_brief = {cross_market_review_backlog_brief or '-'}`",
            f"- `operator_action_queue_brief = {operator_action_queue_brief or '-'}`",
            f"- `operator_action_checklist_brief = {operator_action_checklist_brief or '-'}`",
            f"- `operator_repair_queue_brief = {operator_repair_queue_brief or '-'}`",
            f"- `operator_repair_queue_count = {operator_repair_queue_count or '-'}`",
            f"- `operator_repair_checklist_brief = {operator_repair_checklist_brief or '-'}`",
            f"- `commodity_execution_review_status = {brief.get('commodity_execution_review_status') or '-'}`",
            f"- `commodity_execution_retro_status = {brief.get('commodity_execution_retro_status') or '-'}`",
            f"- `commodity_execution_bridge_status = {brief.get('commodity_execution_bridge_status') or '-'}`",
            f"- `commodity_execution_gap_status = {brief.get('commodity_execution_gap_status') or '-'}`",
            f"- `commodity_remainder_focus_area = {commodity_remainder_focus_area or '-'}`",
            f"- `commodity_remainder_focus_target = {commodity_remainder_focus_target or '-'}`",
            f"- `commodity_remainder_focus_action = {commodity_remainder_focus_action or '-'}`",
            f"- `commodity_remainder_focus_signal_date = {commodity_remainder_focus_signal_date or '-'}`",
            f"- `commodity_remainder_focus_signal_age_days = {commodity_remainder_focus_signal_age_days or '-'}`",
            f"- `commodity_execution_bridge_stale_signal_dates = {json.dumps(stale_signal_dates, ensure_ascii=False, sort_keys=True)}`",
            f"- `commodity_execution_bridge_stale_signal_age_days = {json.dumps(stale_signal_age_days, ensure_ascii=False, sort_keys=True)}`",
            f"- `commodity_stale_signal_watch_brief = {brief.get('commodity_stale_signal_watch_brief') or '-'}`",
            f"- `commodity_stale_signal_watch_next_execution_id = {commodity_stale_signal_watch_next_execution_id or '-'}`",
            f"- `commodity_stale_signal_watch_next_symbol = {commodity_stale_signal_watch_next_symbol or '-'}`",
            f"- `commodity_stale_signal_watch_next_signal_date = {commodity_stale_signal_watch_next_signal_date or '-'}`",
            f"- `commodity_stale_signal_watch_next_signal_age_days = {commodity_stale_signal_watch_next_signal_age_days or '-'}`",
            f"- `commodity_focus_evidence_summary = {json.dumps(commodity_focus_evidence_summary, ensure_ascii=False, sort_keys=True)}`",
            f"- `commodity_focus_lifecycle_status = {commodity_focus_lifecycle_status or '-'}`",
            f"- `commodity_focus_lifecycle_brief = {commodity_focus_lifecycle_brief or '-'}`",
            f"- `commodity_focus_lifecycle_blocker_detail = {commodity_focus_lifecycle_blocker_detail or '-'}`",
            f"- `commodity_focus_lifecycle_done_when = {commodity_focus_lifecycle_done_when or '-'}`",
            f"- `commodity_execution_close_evidence_status = {commodity_execution_close_evidence_status or '-'}`",
            f"- `commodity_execution_close_evidence_brief = {commodity_execution_close_evidence_brief or '-'}`",
            f"- `commodity_execution_close_evidence_target = {commodity_execution_close_evidence_target or '-'}`",
            f"- `commodity_execution_close_evidence_symbol = {commodity_execution_close_evidence_symbol or '-'}`",
            f"- `commodity_execution_close_evidence_blocker_detail = {commodity_execution_close_evidence_blocker_detail or '-'}`",
            f"- `commodity_execution_close_evidence_done_when = {commodity_execution_close_evidence_done_when or '-'}`",
            f"- `commodity_review_close_evidence_pending_symbols = {_list_text(review_close_evidence_pending_symbols)}`",
            f"- `commodity_next_review_close_evidence_execution_symbol = {next_review_close_evidence_symbol or '-'}`",
            f"- `commodity_next_review_close_evidence_execution_id = {next_review_close_evidence_target or '-'}`",
            f"- `secondary_focus_area = {secondary_focus_area or '-'}`",
            f"- `secondary_focus_target = {secondary_focus_target or '-'}`",
            f"- `secondary_focus_symbol = {secondary_focus_symbol or '-'}`",
            f"- `secondary_focus_action = {secondary_focus_action or '-'}`",
            f"- `secondary_focus_reason = {secondary_focus_reason or '-'}`",
            f"- `secondary_focus_state = {secondary_focus_state or '-'}`",
            f"- `secondary_focus_blocker_detail = {secondary_focus_blocker_detail or '-'}`",
            f"- `secondary_focus_done_when = {secondary_focus_done_when or '-'}`",
            f"- `secondary_focus_priority_tier = {secondary_focus_priority_tier or '-'}`",
            f"- `secondary_focus_priority_score = {secondary_focus_priority_score or '-'}`",
            f"- `secondary_focus_queue_rank = {secondary_focus_queue_rank or '-'}`",
            f"- `crypto_route_focus_review_status = {crypto_route_focus_review_status or '-'}`",
            f"- `crypto_route_focus_review_brief = {crypto_route_focus_review_brief or '-'}`",
            f"- `crypto_route_focus_review_primary_blocker = {crypto_route_focus_review_primary_blocker or '-'}`",
            f"- `crypto_route_focus_review_micro_blocker = {crypto_route_focus_review_micro_blocker or '-'}`",
            f"- `crypto_route_focus_review_blocker_detail = {crypto_route_focus_review_blocker_detail or '-'}`",
            f"- `crypto_route_focus_review_done_when = {crypto_route_focus_review_done_when or '-'}`",
            f"- `crypto_route_focus_review_score_status = {crypto_route_focus_review_score_status or '-'}`",
            f"- `crypto_route_focus_review_edge_score = {crypto_route_focus_review_edge_score}`",
            f"- `crypto_route_focus_review_structure_score = {crypto_route_focus_review_structure_score}`",
            f"- `crypto_route_focus_review_micro_score = {crypto_route_focus_review_micro_score}`",
            f"- `crypto_route_focus_review_composite_score = {crypto_route_focus_review_composite_score}`",
            f"- `crypto_route_focus_review_score_brief = {crypto_route_focus_review_score_brief or '-'}`",
            f"- `crypto_route_focus_review_priority_status = {crypto_route_focus_review_priority_status or '-'}`",
            f"- `crypto_route_focus_review_priority_score = {crypto_route_focus_review_priority_score}`",
            f"- `crypto_route_focus_review_priority_tier = {crypto_route_focus_review_priority_tier or '-'}`",
            f"- `crypto_route_focus_review_priority_brief = {crypto_route_focus_review_priority_brief or '-'}`",
            f"- `crypto_route_review_priority_queue_status = {crypto_route_review_priority_queue_status or '-'}`",
            f"- `crypto_route_review_priority_queue_count = {crypto_route_review_priority_queue_count}`",
            f"- `crypto_route_review_priority_queue_brief = {crypto_route_review_priority_queue_brief or '-'}`",
            f"- `crypto_route_review_priority_head_symbol = {crypto_route_review_priority_head_symbol or '-'}`",
            f"- `crypto_route_review_priority_head_tier = {crypto_route_review_priority_head_tier or '-'}`",
            f"- `crypto_route_review_priority_head_score = {crypto_route_review_priority_head_score}`",
            "",
            "## Commodity Route",
            f"- review pending symbols: `{_list_text(review_pending_symbols)}`",
            f"- review close-evidence next symbol: `{next_review_close_evidence_symbol or '-'}`",
            f"- review close-evidence next target: `{next_review_close_evidence_target or '-'}`",
            f"- review close-evidence pending symbols: `{_list_text(review_close_evidence_pending_symbols)}`",
            f"- retro pending symbols: `{_list_text(retro_pending_symbols)}`",
            f"- fill evidence next symbol: `{next_fill_symbol or '-'}`",
            f"- fill evidence next target: `{next_fill_target or '-'}`",
            f"- close evidence next symbol: `{next_close_evidence_symbol or '-'}`",
            f"- close evidence next target: `{next_close_evidence_target or '-'}`",
            f"- close evidence pending symbols: `{_list_text(close_evidence_pending_symbols)}`",
            f"- fill evidence pending symbols: `{_list_text(fill_evidence_pending_symbols)}`",
            f"- already bridged symbols: `{_list_text(already_bridged_symbols)}`",
            f"- evidence present symbols: `{_list_text(with_evidence)}`",
            f"- evidence missing symbols: `{_list_text(without_evidence)}`",
            f"- stale directional symbols: `{_list_text(stale_symbols)}`",
            f"- stale directional signal dates: `{_mapping_text(stale_signal_dates)}`",
            f"- stale directional signal ages: `{_mapping_text(stale_signal_age_days)}`",
            f"- stale directional watch priority: `{_watch_items_text(stale_signal_watch_items)}`",
            f"- stale directional watch head: `{commodity_stale_signal_watch_next_symbol or '-'} | {commodity_stale_signal_watch_next_execution_id or '-'} | {commodity_stale_signal_watch_next_signal_date or '-'} | {commodity_stale_signal_watch_next_signal_age_days or '-'}d`",
            f"- commodity focus lifecycle: `{commodity_focus_lifecycle_brief or '-'}`",
            f"- commodity close-evidence lane: `{commodity_execution_close_evidence_brief or '-'}`",
            f"- action queue: `{operator_action_queue_brief or '-'}`",
            f"- action checklist: `{operator_action_checklist_brief or '-'}`",
            f"- repair queue: `{operator_repair_queue_brief or '-'}`",
            f"- repair checklist: `{operator_repair_checklist_brief or '-'}`",
            f"- focus slot refresh backlog: `{operator_focus_slot_refresh_backlog_brief or '-'}`",
            f"- focus slot promotion gate: `{operator_focus_slot_promotion_gate_brief or '-'}`",
            f"- focus slot actionability gate: `{operator_focus_slot_actionability_gate_brief or '-'}`",
            f"- focus slot readiness gate: `{operator_focus_slot_readiness_gate_brief or '-'}`",
            f"- research embedding quality: `{operator_research_embedding_quality_brief or '-'}`",
            f"- crypto route alignment: `{operator_crypto_route_alignment_brief or '-'}`",
            f"- crypto route alignment area: `{operator_crypto_route_alignment_focus_area or '-'}`",
            f"- crypto route alignment slot: `{operator_crypto_route_alignment_focus_slot or '-'}`",
            f"- crypto route alignment symbol: `{operator_crypto_route_alignment_focus_symbol or '-'}`",
            f"- crypto route alignment action: `{operator_crypto_route_alignment_focus_action or '-'}`",
            f"- crypto route alignment recovery outcome: `{operator_crypto_route_alignment_recovery_brief or '-'}`",
            f"- crypto route alignment cooldown: `{operator_crypto_route_alignment_cooldown_brief or '-'}`",
            f"- crypto route alignment recovery recipe gate: `{operator_crypto_route_alignment_recipe_brief or '-'}`",
            f"- crypto route alignment recovery: `{_list_text(operator_crypto_route_alignment_recipe_target_batches) or '-'}@{operator_crypto_route_alignment_recipe_window_days or '-'}d`",
            f"- crypto shortline market state: `{crypto_route_shortline_market_state_brief or '-'}`",
            f"- crypto shortline trigger stack: `{crypto_route_shortline_execution_gate_brief or '-'}`",
            f"- crypto shortline no-trade rule: `{crypto_route_shortline_no_trade_rule or '-'}`",
            f"- crypto shortline sessions: `{crypto_route_shortline_session_map_brief or '-'}`",
            f"- crypto shortline cvd semantic: `{(crypto_route_shortline_cvd_semantic_status or '-') + ' | ' + (crypto_route_shortline_cvd_semantic_takeaway or '-')}`",
            f"- crypto shortline cvd queue: `{(crypto_route_shortline_cvd_queue_handoff_status or '-') + ' | ' + (crypto_route_shortline_cvd_queue_focus_batch or '-') + ' | ' + (crypto_route_shortline_cvd_queue_focus_action or '-') + ' | ' + (crypto_route_shortline_cvd_queue_stack_brief or '-')}`",
            f"- crypto shortline focus execution: `{(crypto_route_focus_execution_state or '-') + ' | ' + (crypto_route_focus_execution_blocker_detail or '-') + ' | ' + (crypto_route_focus_execution_done_when or '-')}`",
            f"- crypto shortline micro gate: `{(crypto_route_focus_execution_micro_classification or '-') + ' | ' + (crypto_route_focus_execution_micro_context or '-') + ' | ' + (crypto_route_focus_execution_micro_trust_tier or '-') + ' | ' + (crypto_route_focus_execution_micro_veto or '-') + ' | ' + (_list_text(crypto_route_focus_execution_micro_reasons) or '-')}`",
            f"- crypto shortline micro locality: `{(crypto_route_focus_execution_micro_locality_status or '-') + ' | drift=' + (crypto_route_focus_execution_micro_drift_risk or '-') + ' | attack=' + ((crypto_route_focus_execution_micro_attack_side or '-') + ':' + (crypto_route_focus_execution_micro_attack_presence or '-'))}`",
            f"- crypto review lane: `{(crypto_route_focus_review_status or '-') + ' | ' + (crypto_route_focus_review_primary_blocker or '-') + ' | ' + (crypto_route_focus_review_micro_blocker or '-') + ' | ' + (crypto_route_focus_review_done_when or '-')}`",
            f"- crypto head source refresh: `{crypto_route_head_source_refresh_brief or '-'}`",
            f"- crypto route refresh audit: `{(source_crypto_route_refresh_reuse_brief or '-') + ' | mode=' + (source_crypto_route_refresh_native_mode or '-') + ' | reused=' + (source_crypto_route_refresh_reused_native_count or '-') + '/' + (source_crypto_route_refresh_native_step_count or '-') + ' | path=' + (source_crypto_route_refresh_artifact or '-')}`",
            f"- crypto route refresh reuse gate: `{(source_crypto_route_refresh_reuse_gate_brief or '-') + ' | level=' + (source_crypto_route_refresh_reuse_level or '-') + ' | blocking=' + (source_crypto_route_refresh_reuse_gate_blocking or '-') + ' | path=' + (source_crypto_route_refresh_artifact or '-')}`",
            f"- remote live account scope: `{(brief.get('source_remote_live_handoff_account_scope_alignment_brief') or '-') + ' | scope=' + (brief.get('source_remote_live_handoff_ready_check_scope_brief') or '-')}`",
            f"- remote live diagnosis: `{brief.get('source_live_gate_blocker_remote_live_diagnosis_brief') or '-'}`",
            f"- source refresh queue: `{operator_source_refresh_queue_brief or '-'}`",
            f"- source refresh checklist: `{operator_source_refresh_checklist_brief or '-'}`",
            f"- source refresh pipeline: `{operator_source_refresh_pipeline_pending_brief or '-'}`",
            f"- source refresh pipeline relevance: `{operator_source_refresh_pipeline_relevance_brief or '-'}`",
            f"- source refresh pipeline deferred: `{operator_source_refresh_pipeline_deferred_brief or '-'}`",
            "",
            "## Crypto Shortline Gate",
            f"- market state: `{crypto_route_shortline_market_state_brief or '-'}`",
            f"- trigger stack: `{crypto_route_shortline_execution_gate_brief or '-'}`",
            f"- no-trade rule: `{crypto_route_shortline_no_trade_rule or '-'}`",
            f"- session liquidity map: `{crypto_route_shortline_session_map_brief or '-'}`",
            f"- cvd semantic: `{(crypto_route_shortline_cvd_semantic_status or '-') + ' | ' + (crypto_route_shortline_cvd_semantic_takeaway or '-')}`",
            f"- cvd queue handoff: `{(crypto_route_shortline_cvd_queue_handoff_status or '-') + ' | ' + (crypto_route_shortline_cvd_queue_focus_batch or '-') + ' | ' + (crypto_route_shortline_cvd_queue_focus_action or '-') + ' | ' + (crypto_route_shortline_cvd_queue_stack_brief or '-')}`",
            f"- focus execution gate: `{(crypto_route_focus_execution_state or '-') + ' | ' + (crypto_route_focus_execution_blocker_detail or '-') + ' | ' + (crypto_route_focus_execution_done_when or '-')}`",
            f"- micro classification: `{(crypto_route_focus_execution_micro_classification or '-') + ' | ' + (crypto_route_focus_execution_micro_context or '-') + ' | ' + (crypto_route_focus_execution_micro_trust_tier or '-') + ' | ' + (crypto_route_focus_execution_micro_veto or '-') + ' | ' + (_list_text(crypto_route_focus_execution_micro_reasons) or '-')}`",
            f"- micro locality: `{(crypto_route_focus_execution_micro_locality_status or '-') + ' | drift=' + (crypto_route_focus_execution_micro_drift_risk or '-') + ' | attack=' + ((crypto_route_focus_execution_micro_attack_side or '-') + ':' + (crypto_route_focus_execution_micro_attack_presence or '-'))}`",
            f"- review lane: `{(crypto_route_focus_review_status or '-') + ' | ' + (crypto_route_focus_review_primary_blocker or '-') + ' | ' + (crypto_route_focus_review_micro_blocker or '-') + ' | ' + (crypto_route_focus_review_done_when or '-')}`",
            "",
            "## Focus Slot Artifacts",
            f"- primary source: `{next_focus_source_kind or '-'} | {next_focus_source_status or '-'} | {next_focus_source_recency or '-'} | {next_focus_source_health or '-'} | {next_focus_source_refresh_action or '-'} | {next_focus_source_age_minutes or '-'}m | {next_focus_source_as_of or '-'} | {next_focus_source_artifact or '-'}`",
            f"- followup source: `{followup_focus_source_kind or '-'} | {followup_focus_source_status or '-'} | {followup_focus_source_recency or '-'} | {followup_focus_source_health or '-'} | {followup_focus_source_refresh_action or '-'} | {followup_focus_source_age_minutes or '-'}m | {followup_focus_source_as_of or '-'} | {followup_focus_source_artifact or '-'}`",
            f"- secondary source: `{secondary_focus_source_kind or '-'} | {secondary_focus_source_status or '-'} | {secondary_focus_source_recency or '-'} | {secondary_focus_source_health or '-'} | {secondary_focus_source_refresh_action or '-'} | {secondary_focus_source_age_minutes or '-'}m | {secondary_focus_source_as_of or '-'} | {secondary_focus_source_artifact or '-'}`",
            "",
            "## Focus Slot Refresh Backlog",
        ]
    )
    if crypto_route_focus_review_score_status == "scored":
        lines.append(
            "- crypto review scores: `"
            + " | ".join(
                [
                    f"edge={crypto_route_focus_review_edge_score}",
                    f"structure={crypto_route_focus_review_structure_score}",
                    f"micro={crypto_route_focus_review_micro_score}",
                    f"composite={crypto_route_focus_review_composite_score}",
                ]
            )
            + "`"
        )
        lines.append(
            "- review scores: `"
            + " | ".join(
                [
                    f"edge={crypto_route_focus_review_edge_score}",
                    f"structure={crypto_route_focus_review_structure_score}",
                    f"micro={crypto_route_focus_review_micro_score}",
                    f"composite={crypto_route_focus_review_composite_score}",
                ]
            )
            + "`"
        )
    if crypto_route_focus_review_priority_status == "ready":
        lines.append(
            "- crypto review priority: `"
            + " | ".join(
                [
                    f"tier={crypto_route_focus_review_priority_tier or '-'}",
                    f"score={crypto_route_focus_review_priority_score}",
                    f"brief={crypto_route_focus_review_priority_brief or '-'}",
                ]
            )
            + "`"
        )
    if crypto_route_review_priority_queue_status:
        lines.append(
            "- crypto review queue: `"
            + " | ".join(
                [
                    f"status={crypto_route_review_priority_queue_status or '-'}",
                    f"count={crypto_route_review_priority_queue_count}",
                    f"brief={crypto_route_review_priority_queue_brief or '-'}",
                    f"head={crypto_route_review_priority_head_symbol or '-'}:{crypto_route_review_priority_head_tier or '-'}:{crypto_route_review_priority_head_score}",
                ]
            )
            + "`"
        )
    if operator_focus_slot_refresh_backlog:
        for row in operator_focus_slot_refresh_backlog:
            slot = str(row.get("slot") or "").strip() or "-"
            symbol = str(row.get("symbol") or "").strip().upper() or "-"
            action = str(row.get("action") or "").strip() or "-"
            source_kind = str(row.get("source_kind") or "").strip() or "-"
            source_health = str(row.get("source_health") or "").strip() or "-"
            source_status = str(row.get("source_status") or "").strip() or "-"
            source_recency = str(row.get("source_recency") or "").strip() or "-"
            source_age_minutes_raw = row.get("source_age_minutes")
            source_age_minutes = (
                str(source_age_minutes_raw).strip()
                if source_age_minutes_raw not in (None, "")
                else "-"
            )
            source_as_of = str(row.get("source_as_of") or "").strip() or "-"
            source_artifact = str(row.get("source_artifact") or "").strip() or "-"
            lines.append(
                f"- `{slot}` `{symbol}` action=`{action}` kind=`{source_kind}` "
                f"health=`{source_health}` status=`{source_status}` recency=`{source_recency}` "
                f"age=`{source_age_minutes}m` as_of=`{source_as_of}` path=`{source_artifact}`"
            )
    else:
        lines.append("- No focus slot source refresh backlog remains.")
    lines.extend(
        [
            "",
            "## Research Embedding Quality",
            f"- status=`{operator_research_embedding_quality_status or '-'}` brief=`{operator_research_embedding_quality_brief or '-'}`",
            f"- blocker=`{operator_research_embedding_quality_blocker_detail or '-'}`",
            f"- done_when=`{operator_research_embedding_quality_done_when or '-'}`",
            f"- active_batches=`{_list_text(operator_research_embedding_active_batches)}`",
            f"- avoid_batches=`{_list_text(operator_research_embedding_avoid_batches)}`",
            f"- zero_trade_deprioritized_batches=`{_list_text(operator_research_embedding_zero_trade_deprioritized_batches)}`",
            "",
            "## Crypto Route Alignment",
            f"- area=`{operator_crypto_route_alignment_focus_area or '-'}` slot=`{operator_crypto_route_alignment_focus_slot or '-'}` symbol=`{operator_crypto_route_alignment_focus_symbol or '-'}` action=`{operator_crypto_route_alignment_focus_action or '-'}` status=`{operator_crypto_route_alignment_status or '-'}` brief=`{operator_crypto_route_alignment_brief or '-'}`",
            f"- blocker=`{operator_crypto_route_alignment_blocker_detail or '-'}`",
            f"- done_when=`{operator_crypto_route_alignment_done_when or '-'}`",
            f"- recovery_outcome=`{operator_crypto_route_alignment_recovery_status or '-'} | {operator_crypto_route_alignment_recovery_brief or '-'} | failed={operator_crypto_route_alignment_recovery_failed_batch_count or '-'} | timed_out={operator_crypto_route_alignment_recovery_timed_out_batch_count or '-'} | zero_trade_batches={_list_text(operator_crypto_route_alignment_recovery_zero_trade_batches) or '-'}`",
            f"- recovery_outcome_blocker=`{operator_crypto_route_alignment_recovery_blocker_detail or '-'}`",
            f"- recovery_outcome_done_when=`{operator_crypto_route_alignment_recovery_done_when or '-'}`",
            f"- cooldown=`{operator_crypto_route_alignment_cooldown_status or '-'} | {operator_crypto_route_alignment_cooldown_brief or '-'} | last_end={operator_crypto_route_alignment_cooldown_last_research_end_date or '-'} | next_eligible={operator_crypto_route_alignment_cooldown_next_eligible_end_date or '-'} | blocker={operator_crypto_route_alignment_cooldown_blocker_detail or '-'} | done_when={operator_crypto_route_alignment_cooldown_done_when or '-'}`",
            f"- recovery_recipe_gate=`{operator_crypto_route_alignment_recipe_status or '-'} | {operator_crypto_route_alignment_recipe_brief or '-'} | ready_on={operator_crypto_route_alignment_recipe_ready_on_date or '-'} | blocker={operator_crypto_route_alignment_recipe_blocker_detail or '-'} | done_when={operator_crypto_route_alignment_recipe_done_when or '-'}`",
            f"- recovery=`{_list_text(operator_crypto_route_alignment_recipe_target_batches) or '-'}@{operator_crypto_route_alignment_recipe_window_days or '-'}d`",
            f"- recovery_script=`{operator_crypto_route_alignment_recipe_script or '-'}`",
            f"- recovery_command=`{operator_crypto_route_alignment_recipe_command_hint or '-'}`",
            f"- recovery_followup=`{operator_crypto_route_alignment_recipe_followup_command_hint or '-'}`",
            f"- recovery_verify=`{operator_crypto_route_alignment_recipe_verify_hint or '-'}`",
            "",
            "## Crypto Route Head Source Refresh",
            f"- status=`{crypto_route_head_source_refresh_status or '-'}` brief=`{crypto_route_head_source_refresh_brief or '-'}` slot=`{crypto_route_head_source_refresh_slot or '-'}` symbol=`{crypto_route_head_source_refresh_symbol or '-'}` action=`{crypto_route_head_source_refresh_action or '-'}`",
            f"- source=`{crypto_route_head_source_refresh_source_kind or '-'} | {crypto_route_head_source_refresh_source_health or '-'} | {crypto_route_head_source_refresh_source_artifact or '-'}`",
            f"- blocker=`{crypto_route_head_source_refresh_blocker_detail or '-'}`",
            f"- done_when=`{crypto_route_head_source_refresh_done_when or '-'}`",
            f"- recipe=`{(crypto_route_head_source_refresh_recipe_script or '-') + ' | ' + (crypto_route_head_source_refresh_recipe_expected_status or '-') + ' | ' + (crypto_route_head_source_refresh_recipe_expected_artifact_kind or '-') + '@' + (crypto_route_head_source_refresh_recipe_expected_artifact_path_hint or '-')}`",
            f"- recipe_followup=`{(crypto_route_head_source_refresh_recipe_followup_script or '-') + ' | ' + (crypto_route_head_source_refresh_recipe_verify_hint or '-')}`",
            f"- recipe_pipeline=`{crypto_route_head_source_refresh_recipe_steps_brief or '-'}`",
            f"- recipe_checkpoint=`{crypto_route_head_source_refresh_recipe_step_checkpoint_brief or '-'}`",
            "",
            "## Crypto Route Refresh Audit",
            f"- status=`{source_crypto_route_refresh_status or '-'}` as_of=`{source_crypto_route_refresh_as_of or '-'}` path=`{source_crypto_route_refresh_artifact or '-'}`",
            f"- native_mode=`{source_crypto_route_refresh_native_mode or '-'}` reuse=`{source_crypto_route_refresh_reuse_status or '-'} | {source_crypto_route_refresh_reuse_brief or '-'}`",
            f"- counts=`reused={source_crypto_route_refresh_reused_native_count or '-'} | missing={source_crypto_route_refresh_missing_reused_count or '-'} | native_steps={source_crypto_route_refresh_native_step_count or '-'}`",
            f"- note=`{source_crypto_route_refresh_reuse_note or '-'}`",
            f"- done_when=`{source_crypto_route_refresh_reuse_done_when or '-'}`",
            f"- reuse_gate=`{source_crypto_route_refresh_reuse_gate_status or '-'} | {source_crypto_route_refresh_reuse_gate_brief or '-'} | level={source_crypto_route_refresh_reuse_level or '-'} | blocking={source_crypto_route_refresh_reuse_gate_blocking or '-'}`",
            f"- reuse_gate_blocker=`{source_crypto_route_refresh_reuse_gate_blocker_detail or '-'}`",
            f"- reuse_gate_done_when=`{source_crypto_route_refresh_reuse_gate_done_when or '-'}`",
            "",
            "## Remote Live History Audit",
            f"- status=`{brief.get('source_remote_live_history_audit_status') or '-'}` as_of=`{brief.get('source_remote_live_history_audit_as_of') or '-'}` market=`{brief.get('source_remote_live_history_audit_market') or '-'}` path=`{brief.get('source_remote_live_history_audit_artifact') or '-'}`",
            f"- windows=`{brief.get('source_remote_live_history_audit_window_brief') or '-'}`",
            f"- snapshot=`quote_available={brief.get('source_remote_live_history_audit_quote_available')} | open_positions={brief.get('source_remote_live_history_audit_open_positions')} | blocked_candidate={brief.get('source_remote_live_history_audit_blocked_candidate_symbol') or '-'}`",
            f"- risk_guard=`{brief.get('source_remote_live_history_audit_risk_guard_status') or '-'} | {_list_text(brief.get('source_remote_live_history_audit_risk_guard_reasons') or [])}`",
            f"- pnl_30d_by_symbol=`{brief.get('source_remote_live_history_audit_30d_symbol_pnl_brief') or '-'}`",
            f"- pnl_30d_by_day=`{brief.get('source_remote_live_history_audit_30d_day_pnl_brief') or '-'}`",
            "",
            "## Remote Live Diagnosis",
            f"- status=`{brief.get('source_live_gate_blocker_remote_live_diagnosis_status') or '-'}` brief=`{brief.get('source_live_gate_blocker_remote_live_diagnosis_brief') or '-'}` path=`{brief.get('source_live_gate_blocker_artifact') or '-'}`",
            f"- live_decision=`{brief.get('source_live_gate_blocker_live_decision') or '-'}` as_of=`{brief.get('source_live_gate_blocker_as_of') or '-'}`",
            f"- blocker=`{brief.get('source_live_gate_blocker_remote_live_diagnosis_blocker_detail') or '-'}`",
            f"- done_when=`{brief.get('source_live_gate_blocker_remote_live_diagnosis_done_when') or '-'}`",
            "",
            "## Remote Live Operator Alignment",
            f"- status=`{brief.get('source_live_gate_blocker_remote_live_operator_alignment_status') or '-'}` brief=`{brief.get('source_live_gate_blocker_remote_live_operator_alignment_brief') or '-'}`",
            f"- blocker=`{brief.get('source_live_gate_blocker_remote_live_operator_alignment_blocker_detail') or '-'}`",
            f"- done_when=`{brief.get('source_live_gate_blocker_remote_live_operator_alignment_done_when') or '-'}`",
            "",
            "## Brooks Structure Route",
            f"- route_report=`{source_brooks_route_report_status or '-'} | {source_brooks_route_report_as_of or '-'} | {source_brooks_route_report_artifact or '-'}`",
            f"- selected_routes=`{source_brooks_route_report_selected_routes_brief or '-'}`",
            f"- candidate_count=`{source_brooks_route_report_candidate_count or '-'}`",
            f"- route_head=`{(source_brooks_route_report_head_symbol or '-') + ' | ' + (source_brooks_route_report_head_strategy_id or '-') + ' | ' + (source_brooks_route_report_head_direction or '-') + ' | ' + (source_brooks_route_report_head_bridge_status or '-')}`",
            f"- route_blocker=`{source_brooks_route_report_head_blocker_detail or '-'}`",
            f"- execution_plan=`{source_brooks_execution_plan_status or '-'} | {source_brooks_execution_plan_as_of or '-'} | {source_brooks_execution_plan_artifact or '-'}`",
            f"- execution_counts=`actionable={source_brooks_execution_plan_actionable_count or '-'} | blocked={source_brooks_execution_plan_blocked_count or '-'}`",
            f"- execution_head=`{(source_brooks_execution_plan_head_symbol or '-') + ' | ' + (source_brooks_execution_plan_head_strategy_id or '-') + ' | ' + (source_brooks_execution_plan_head_plan_status or '-') + ' | ' + (source_brooks_execution_plan_head_execution_action or '-')}`",
            f"- execution_prices=`{('entry=' + (source_brooks_execution_plan_head_entry_price or '-')) + ' | ' + ('stop=' + (source_brooks_execution_plan_head_stop_price or '-')) + ' | ' + ('target=' + (source_brooks_execution_plan_head_target_price or '-')) + ' | ' + ('rr=' + (source_brooks_execution_plan_head_rr_ratio or '-'))}`",
            f"- execution_blocker=`{source_brooks_execution_plan_head_blocker_detail or '-'}`",
            "",
            "## Brooks Structure Review Queue",
            f"- queue_source=`{(source_brooks_structure_review_queue_status or '-') + ' | ' + (source_brooks_structure_review_queue_as_of or '-') + ' | ' + (source_brooks_structure_review_queue_artifact or '-')}`",
            f"- refresh_source=`{(source_brooks_structure_refresh_status or '-') + ' | ' + (source_brooks_structure_refresh_as_of or '-') + ' | ' + (source_brooks_structure_refresh_artifact or '-')}`",
            f"- refresh=`{(source_brooks_structure_refresh_brief or '-') + ' | queue=' + (source_brooks_structure_refresh_queue_count or '-') + ' | head=' + (source_brooks_structure_refresh_head_symbol or '-') + ':' + (source_brooks_structure_refresh_head_action or '-') + ':' + (source_brooks_structure_refresh_head_priority_score or '-')}`",
            f"- status=`{brooks_structure_review_status or '-'}` brief=`{brooks_structure_review_brief or '-'}`",
            f"- queue=`{(brooks_structure_review_queue_status or '-') + ' | count=' + (brooks_structure_review_queue_count or '-') + ' | ' + (brooks_structure_review_queue_brief or '-')}`",
            f"- priority=`{(brooks_structure_review_priority_status or '-') + ' | ' + (brooks_structure_review_priority_brief or '-')}`",
            f"- head=`{(brooks_structure_review_head_symbol or '-') + ' | ' + (brooks_structure_review_head_strategy_id or '-') + ' | ' + (brooks_structure_review_head_direction or '-') + ' | ' + (brooks_structure_review_head_tier or '-') + ' | ' + (brooks_structure_review_head_plan_status or '-') + ' | ' + (brooks_structure_review_head_action or '-') + ' | rank=' + (brooks_structure_review_head_rank or '-') + ' | route_score=' + (brooks_structure_review_head_route_selection_score or '-') + ' | signal_score=' + (brooks_structure_review_head_signal_score or '-') + ' | age_bars=' + (brooks_structure_review_head_signal_age_bars or '-') + ' | priority_score=' + (brooks_structure_review_head_priority_score or '-') + ' | priority_tier=' + (brooks_structure_review_head_priority_tier or '-')}`",
            f"- blocker=`{brooks_structure_review_head_blocker_detail or brooks_structure_review_blocker_detail or '-'}`",
            f"- done_when=`{brooks_structure_review_head_done_when or brooks_structure_review_done_when or '-'}`",
            "",
            "## Brooks Structure Operator Lane",
            f"- status=`{brooks_structure_operator_status or '-'}` brief=`{brooks_structure_operator_brief or '-'}`",
            f"- head=`{(brooks_structure_operator_head_symbol or '-') + ' | ' + (brooks_structure_operator_head_strategy_id or '-') + ' | ' + (brooks_structure_operator_head_direction or '-') + ' | ' + (brooks_structure_operator_head_action or '-') + ' | ' + (brooks_structure_operator_head_plan_status or '-') + ' | priority_score=' + (brooks_structure_operator_head_priority_score or '-') + ' | priority_tier=' + (brooks_structure_operator_head_priority_tier or '-')}`",
            f"- backlog=`{(brooks_structure_operator_backlog_count or '-') + ' | ' + (brooks_structure_operator_backlog_brief or '-')}`",
            f"- blocker=`{brooks_structure_operator_blocker_detail or '-'}`",
            f"- done_when=`{brooks_structure_operator_done_when or '-'}`",
            "",
            *_cross_market_operator_head_section_lines(
                status=cross_market_operator_head_status,
                brief=cross_market_operator_head_brief,
                area=cross_market_operator_head_area,
                symbol=cross_market_operator_head_symbol,
                action=cross_market_operator_head_action,
                state=cross_market_operator_head_state,
                priority_score=cross_market_operator_head_priority_score,
                priority_tier=cross_market_operator_head_priority_tier,
                backlog_count=cross_market_operator_backlog_count,
                backlog_brief=cross_market_operator_backlog_brief,
                blocker_detail=cross_market_operator_head_blocker_detail,
                done_when=cross_market_operator_head_done_when,
            ),
            *_cross_market_operator_backlog_section_lines(
                source_status=source_cross_market_operator_state_status,
                source_as_of=source_cross_market_operator_state_as_of,
                source_artifact=source_cross_market_operator_state_artifact,
                snapshot_brief=source_cross_market_operator_state_operator_snapshot_brief
                or source_cross_market_operator_state_snapshot_brief,
                backlog_status=source_cross_market_operator_state_operator_backlog_status,
                backlog_count=source_cross_market_operator_state_operator_backlog_count,
                backlog_state_brief=source_cross_market_operator_state_operator_backlog_state_brief,
                backlog_priority_totals_brief=source_cross_market_operator_state_operator_backlog_priority_totals_brief,
                lane_heads_brief=cross_market_operator_lane_heads_brief,
                lane_priority_order_brief=cross_market_operator_lane_priority_order_brief,
                backlog_brief=source_cross_market_operator_state_operator_backlog_brief,
                head_area=source_cross_market_operator_state_operator_head_area,
                head_symbol=source_cross_market_operator_state_operator_head_symbol,
                head_action=source_cross_market_operator_state_operator_head_action,
                head_state=source_cross_market_operator_state_operator_head_state,
                head_priority_score=source_cross_market_operator_state_operator_head_priority_score,
                head_priority_tier=source_cross_market_operator_state_operator_head_priority_tier,
            ),
            *_cross_market_remote_live_section_lines(
                gate_status=cross_market_remote_live_takeover_gate_status,
                gate_brief=cross_market_remote_live_takeover_gate_brief,
                remote_snapshot_brief=source_cross_market_operator_state_remote_live_snapshot_brief
                or source_cross_market_operator_state_snapshot_brief,
                gate_blocker_detail=cross_market_remote_live_takeover_gate_blocker_detail,
                gate_done_when=cross_market_remote_live_takeover_gate_done_when,
                clearing_status=cross_market_remote_live_takeover_clearing_status,
                clearing_brief=cross_market_remote_live_takeover_clearing_brief,
                clearing_blocker_detail=cross_market_remote_live_takeover_clearing_blocker_detail,
                clearing_done_when=cross_market_remote_live_takeover_clearing_done_when,
            ),
            *_cross_market_repair_section_lines(
                queue_status=remote_live_takeover_repair_queue_status,
                queue_brief=remote_live_takeover_repair_queue_brief,
                queue_count=remote_live_takeover_repair_queue_count,
                queue_head_area=remote_live_takeover_repair_queue_head_area,
                queue_head_code=remote_live_takeover_repair_queue_head_code,
                queue_head_action=remote_live_takeover_repair_queue_head_action,
                queue_head_priority_score=remote_live_takeover_repair_queue_head_priority_score,
                queue_head_priority_tier=remote_live_takeover_repair_queue_head_priority_tier,
                queue_head_command=remote_live_takeover_repair_queue_head_command,
                queue_head_clear_when=remote_live_takeover_repair_queue_head_clear_when,
                queue_done_when=remote_live_takeover_repair_queue_done_when,
                operator_repair_queue_count=operator_repair_queue_count,
                operator_repair_queue_brief=operator_repair_queue_brief,
                operator_repair_checklist_brief=operator_repair_checklist_brief,
                repair_head_status=cross_market_operator_repair_head_status,
                repair_head_brief=cross_market_operator_repair_head_brief,
                repair_head_area=cross_market_operator_repair_head_area,
                repair_head_code=cross_market_operator_repair_head_code,
                repair_head_action=cross_market_operator_repair_head_action,
                repair_head_priority_score=cross_market_operator_repair_head_priority_score,
                repair_head_priority_tier=cross_market_operator_repair_head_priority_tier,
                repair_head_command=cross_market_operator_repair_head_command,
                repair_head_clear_when=cross_market_operator_repair_head_clear_when,
                repair_backlog_status=cross_market_operator_repair_backlog_status,
                repair_backlog_count=cross_market_operator_repair_backlog_count,
                repair_backlog_priority_total=cross_market_operator_repair_backlog_priority_total,
                repair_backlog_brief=cross_market_operator_repair_backlog_brief,
                repair_head_done_when=cross_market_operator_repair_head_done_when,
                repair_backlog_done_when=cross_market_operator_repair_backlog_done_when,
            ),
            *_cross_market_state_lanes_section_lines(
                waiting_status=cross_market_operator_waiting_lane_status,
                waiting_count=cross_market_operator_waiting_lane_count,
                waiting_priority_total=cross_market_operator_waiting_lane_priority_total,
                waiting_head_symbol=cross_market_operator_waiting_lane_head_symbol,
                waiting_head_action=cross_market_operator_waiting_lane_head_action,
                waiting_head_priority_score=cross_market_operator_waiting_lane_head_priority_score,
                waiting_head_priority_tier=cross_market_operator_waiting_lane_head_priority_tier,
                waiting_brief=cross_market_operator_waiting_lane_brief,
                review_status=cross_market_operator_review_lane_status,
                review_count=cross_market_operator_review_lane_count,
                review_priority_total=cross_market_operator_review_lane_priority_total,
                review_head_symbol=cross_market_operator_review_lane_head_symbol,
                review_head_action=cross_market_operator_review_lane_head_action,
                review_head_priority_score=cross_market_operator_review_lane_head_priority_score,
                review_head_priority_tier=cross_market_operator_review_lane_head_priority_tier,
                review_brief=cross_market_operator_review_lane_brief,
                watch_status=cross_market_operator_watch_lane_status,
                watch_count=cross_market_operator_watch_lane_count,
                watch_priority_total=cross_market_operator_watch_lane_priority_total,
                watch_head_symbol=cross_market_operator_watch_lane_head_symbol,
                watch_head_action=cross_market_operator_watch_lane_head_action,
                watch_head_priority_score=cross_market_operator_watch_lane_head_priority_score,
                watch_head_priority_tier=cross_market_operator_watch_lane_head_priority_tier,
                watch_brief=cross_market_operator_watch_lane_brief,
                blocked_status=cross_market_operator_blocked_lane_status,
                blocked_count=cross_market_operator_blocked_lane_count,
                blocked_priority_total=cross_market_operator_blocked_lane_priority_total,
                blocked_head_symbol=cross_market_operator_blocked_lane_head_symbol,
                blocked_head_action=cross_market_operator_blocked_lane_head_action,
                blocked_head_priority_score=cross_market_operator_blocked_lane_head_priority_score,
                blocked_head_priority_tier=cross_market_operator_blocked_lane_head_priority_tier,
                blocked_brief=cross_market_operator_blocked_lane_brief,
                repair_status=cross_market_operator_repair_lane_status,
                repair_count=cross_market_operator_repair_lane_count,
                repair_priority_total=cross_market_operator_repair_lane_priority_total,
                repair_head_symbol=cross_market_operator_repair_lane_head_symbol,
                repair_head_action=cross_market_operator_repair_lane_head_action,
                repair_head_priority_score=cross_market_operator_repair_lane_head_priority_score,
                repair_head_priority_tier=cross_market_operator_repair_lane_head_priority_tier,
                repair_brief=cross_market_operator_repair_lane_brief,
            ),
            *_cross_market_review_section_lines(
                review_head_status=cross_market_review_head_status,
                review_head_brief=cross_market_review_head_brief,
                review_head_area=cross_market_review_head_area,
                review_head_symbol=cross_market_review_head_symbol,
                review_head_action=cross_market_review_head_action,
                review_head_priority_score=cross_market_review_head_priority_score,
                review_head_priority_tier=cross_market_review_head_priority_tier,
                review_backlog_count=cross_market_review_backlog_count,
                review_backlog_brief=cross_market_review_backlog_brief,
                review_head_blocker_detail=cross_market_review_head_blocker_detail,
                review_head_done_when=cross_market_review_head_done_when,
                source_status=source_cross_market_operator_state_status,
                source_as_of=source_cross_market_operator_state_as_of,
                source_artifact=source_cross_market_operator_state_artifact,
                review_snapshot_brief=source_cross_market_operator_state_review_snapshot_brief
                or source_cross_market_operator_state_snapshot_brief,
                source_backlog_status=source_cross_market_operator_state_review_backlog_status,
                source_backlog_count=source_cross_market_operator_state_review_backlog_count,
                source_backlog_brief=source_cross_market_operator_state_review_backlog_brief,
                source_head_area=source_cross_market_operator_state_review_head_area,
                source_head_symbol=source_cross_market_operator_state_review_head_symbol,
                source_head_action=source_cross_market_operator_state_review_head_action,
                source_head_priority_score=source_cross_market_operator_state_review_head_priority_score,
                source_head_priority_tier=source_cross_market_operator_state_review_head_priority_tier,
            ),
            *_system_time_sync_repair_plan_section_lines(
                status=source_system_time_sync_repair_plan_status,
                brief=source_system_time_sync_repair_plan_brief,
                artifact=source_system_time_sync_repair_plan_artifact,
                admin_required=source_system_time_sync_repair_plan_admin_required,
                done_when=source_system_time_sync_repair_plan_done_when or cross_market_review_head_done_when,
            ),
            *_system_time_sync_repair_verification_section_lines(
                status=source_system_time_sync_repair_verification_status,
                brief=source_system_time_sync_repair_verification_brief,
                artifact=source_system_time_sync_repair_verification_artifact,
                cleared=source_system_time_sync_repair_verification_cleared,
            ),
            *_openclaw_orderflow_blueprint_section_lines(
                status=source_openclaw_orderflow_blueprint_status,
                brief=source_openclaw_orderflow_blueprint_brief,
                artifact=source_openclaw_orderflow_blueprint_artifact,
                current_life_stage=source_openclaw_orderflow_blueprint_current_life_stage,
                target_life_stage=source_openclaw_orderflow_blueprint_target_life_stage,
                intent_queue_brief=str(
                    brief.get("source_openclaw_orderflow_blueprint_remote_intent_queue_brief") or ""
                ),
                intent_queue_recommendation=str(
                    brief.get("source_openclaw_orderflow_blueprint_remote_intent_queue_recommendation") or ""
                ),
                execution_journal_brief=source_openclaw_orderflow_blueprint_remote_execution_journal_brief,
                orderflow_feedback_brief=source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief,
                orderflow_policy_brief=source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief,
                execution_ack_brief=source_openclaw_orderflow_blueprint_remote_execution_ack_brief,
                canary_gate_brief=source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_brief,
                quality_report_brief=source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_brief,
                guardian_clearance_brief=source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_brief,
                guardian_clearance_top=":".join(
                    [
                        part
                        for part in [
                            source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code,
                            source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_target_artifact,
                            source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_title,
                        ]
                        if part
                    ]
                ),
                live_boundary_hold_brief=source_openclaw_orderflow_blueprint_remote_live_boundary_hold_brief,
                promotion_gate_brief=source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_brief,
                promotion_gate_top=":".join(
                    [
                        part
                        for part in [
                            source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code,
                            source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_target_artifact,
                            source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_title,
                        ]
                        if part
                    ]
                ),
                shadow_learning_continuity_brief=source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief,
                promotion_unblock_readiness_brief=source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief,
                ticket_actionability_brief=source_openclaw_orderflow_blueprint_remote_ticket_actionability_brief,
                shortline_backtest_slice_brief=source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_brief,
                shortline_cross_section_backtest_brief=source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_brief,
                time_sync_mode=source_openclaw_orderflow_blueprint_remote_time_sync_mode,
                shadow_clock_evidence_brief=source_openclaw_orderflow_blueprint_remote_shadow_clock_evidence_brief,
                backlog_top=":".join(
                    [
                        part
                        for part in [
                            source_openclaw_orderflow_blueprint_top_backlog_target_artifact,
                            source_openclaw_orderflow_blueprint_top_backlog_title,
                            source_openclaw_orderflow_blueprint_top_backlog_why,
                        ]
                        if part
                    ]
                ),
            ),
            *_remote_live_account_scope_section_lines(
                status=str(brief.get("source_remote_live_handoff_status") or ""),
                as_of=str(brief.get("source_remote_live_handoff_as_of") or ""),
                state=str(brief.get("source_remote_live_handoff_state") or ""),
                artifact=str(brief.get("source_remote_live_handoff_artifact") or ""),
                snapshot=source_cross_market_operator_state_remote_live_snapshot_brief
                or source_cross_market_operator_state_snapshot_brief,
                ready_scope_brief=str(brief.get("source_remote_live_handoff_ready_check_scope_brief") or ""),
                ready_scope_market=str(brief.get("source_remote_live_handoff_ready_check_scope_market") or ""),
                alignment_brief=str(brief.get("source_remote_live_handoff_account_scope_alignment_brief") or ""),
                alignment_blocking=brief.get("source_remote_live_handoff_account_scope_alignment_blocking"),
                alignment_blocker_detail=str(brief.get("source_remote_live_handoff_account_scope_alignment_blocker_detail") or ""),
            ),
            "## Source Refresh Queue",
        ]
    )
    if brooks_structure_review_queue:
        for row in brooks_structure_review_queue:
            rank = str(row.get("rank") or "-")
            symbol = str(row.get("symbol") or "").strip().upper() or "-"
            strategy_id = str(row.get("strategy_id") or "").strip() or "-"
            direction = str(row.get("direction") or "").strip() or "-"
            tier = str(row.get("tier") or "").strip() or "-"
            plan_status = str(row.get("plan_status") or "").strip() or "-"
            action = str(row.get("execution_action") or "").strip() or "-"
            route_score = _fmt_num(row.get("route_selection_score"))
            signal_score = str(row.get("signal_score") or "-")
            age_bars = str(row.get("signal_age_bars") or "-")
            priority_score = str(row.get("priority_score") or "-")
            priority_tier = str(row.get("priority_tier") or "").strip() or "-"
            blocker = str(row.get("blocker_detail") or "").strip() or "-"
            done_when = str(row.get("done_when") or "").strip() or "-"
            lines.append(
                f"- {rank}. `{symbol}` strategy=`{strategy_id}` direction=`{direction}` "
                f"tier=`{tier}` status=`{plan_status}` action=`{action}` "
                f"route_score=`{route_score}` signal_score=`{signal_score}` age_bars=`{age_bars}` "
                f"priority_score=`{priority_score}` priority_tier=`{priority_tier}`"
            )
            lines.append(f"  blocker=`{blocker}` done_when=`{done_when}`")
    else:
        lines.append("- No Brooks review queue items remain.")
    if operator_source_refresh_queue:
        for row in operator_source_refresh_queue:
            rank = str(row.get("rank") or "-")
            slot = str(row.get("slot") or "").strip() or "-"
            symbol = str(row.get("symbol") or "").strip().upper() or "-"
            action = str(row.get("action") or "").strip() or "-"
            source_kind = str(row.get("source_kind") or "").strip() or "-"
            source_health = str(row.get("source_health") or "").strip() or "-"
            source_status = str(row.get("source_status") or "").strip() or "-"
            source_recency = str(row.get("source_recency") or "").strip() or "-"
            source_artifact = str(row.get("source_artifact") or "").strip() or "-"
            lines.append(
                f"- {rank}. `{slot}` `{symbol}` `{action}` kind=`{source_kind}` "
                f"health=`{source_health}` status=`{source_status}` recency=`{source_recency}` "
                f"path=`{source_artifact}`"
            )
    else:
        lines.append("- No source refresh queue remains.")
    lines.extend(
        [
            "",
            "## Source Refresh Pipeline",
        ]
    )
    if operator_source_refresh_pipeline_pending_count not in ("", "0", 0):
        lines.append(
            f"- pending=`{operator_source_refresh_pipeline_pending_brief or '-'}` "
            f"checkpoint=`{operator_source_refresh_pipeline_step_checkpoint_brief or '-'}`"
        )
        lines.append(
            f"- head=`step {operator_source_refresh_pipeline_head_rank or '-'} | "
            f"{operator_source_refresh_pipeline_head_name or '-'} | "
            f"{operator_source_refresh_pipeline_head_checkpoint_state or '-'} | "
            f"{operator_source_refresh_pipeline_head_expected_artifact_kind or '-'} | "
            f"{operator_source_refresh_pipeline_head_current_artifact or '-'}`"
        )
    else:
        lines.append("- No source refresh pipeline pending remains.")
    if operator_source_refresh_pipeline_relevance_brief or operator_source_refresh_pipeline_relevance_status:
        lines.append(
            f"- relevance=`{operator_source_refresh_pipeline_relevance_brief or '-'} | "
            f"status={operator_source_refresh_pipeline_relevance_status or '-'} | "
            f"blocker={operator_source_refresh_pipeline_relevance_blocker_detail or '-'} | "
            f"done_when={operator_source_refresh_pipeline_relevance_done_when or '-'} `"
        )
    if operator_source_refresh_pipeline_deferred_count not in ("", "0", 0):
        lines.append(
            f"- deferred=`{operator_source_refresh_pipeline_deferred_brief or '-'} | "
            f"status={operator_source_refresh_pipeline_deferred_status or '-'} | "
            f"until={operator_source_refresh_pipeline_deferred_until or '-'} | "
            f"reason={operator_source_refresh_pipeline_deferred_reason or '-'} | "
            f"checkpoint={operator_source_refresh_pipeline_step_checkpoint_brief or '-'}`"
        )
        lines.append(
            f"- deferred_head=`step {operator_source_refresh_pipeline_deferred_head_rank or '-'} | "
            f"{operator_source_refresh_pipeline_deferred_head_name or '-'} | "
            f"{operator_source_refresh_pipeline_deferred_head_checkpoint_state or '-'} | "
            f"{operator_source_refresh_pipeline_deferred_head_expected_artifact_kind or '-'} | "
            f"{operator_source_refresh_pipeline_deferred_head_current_artifact or '-'}`"
        )
    lines.extend(
        [
            "",
            "## Source Refresh Checklist",
        ]
    )
    if operator_source_refresh_checklist:
        for row in operator_source_refresh_checklist:
            rank = str(row.get("rank") or "-")
            state = str(row.get("state") or "").strip() or "-"
            symbol = str(row.get("symbol") or "").strip().upper() or "-"
            action = str(row.get("action") or "").strip() or "-"
            blocker_detail = str(row.get("blocker_detail") or "").strip() or "-"
            done_when = str(row.get("done_when") or "").strip() or "-"
            recipe_script = str(row.get("recipe_script") or "").strip() or "-"
            recipe_expected_status = str(row.get("recipe_expected_status") or "").strip() or "-"
            recipe_note = str(row.get("recipe_note") or "").strip() or "-"
            recipe_command_hint = str(row.get("recipe_command_hint") or "").strip() or "-"
            recipe_followup_script = str(row.get("recipe_followup_script") or "").strip() or "-"
            recipe_followup_command_hint = str(row.get("recipe_followup_command_hint") or "").strip() or "-"
            recipe_verify_hint = str(row.get("recipe_verify_hint") or "").strip() or "-"
            recipe_expected_artifact_kind = str(row.get("recipe_expected_artifact_kind") or "").strip() or "-"
            recipe_expected_artifact_path_hint = str(row.get("recipe_expected_artifact_path_hint") or "").strip() or "-"
            recipe_steps_brief = str(row.get("recipe_steps_brief") or "").strip() or "-"
            recipe_step_checkpoint_brief = str(row.get("recipe_step_checkpoint_brief") or "").strip() or "-"
            recipe_steps = [dict(step) for step in row.get("recipe_steps", []) if isinstance(step, dict)]
            lines.append(
                f"- {rank}. `{state}` `{symbol}` `{action}` blocker=`{blocker_detail}` "
                f"done_when=`{done_when}` script=`{recipe_script}` expected_status=`{recipe_expected_status}` "
                f"expected_artifact=`{recipe_expected_artifact_kind}@{recipe_expected_artifact_path_hint}` "
                f"note=`{recipe_note}`"
            )
            lines.append(f"  command=`{recipe_command_hint}`")
            lines.append(f"  followup_command=`{recipe_followup_command_hint}`")
            lines.append(f"  verify=`{recipe_verify_hint}`")
            lines.append(f"  pipeline=`{recipe_steps_brief}`")
            lines.append(f"  checkpoint=`{recipe_step_checkpoint_brief}`")
            for step in recipe_steps:
                step_rank = str(step.get('rank') or '-')
                step_name = str(step.get('name') or '').strip() or '-'
                step_script = str(step.get('script') or '').strip() or '-'
                step_command = str(step.get('command_hint') or '').strip() or '-'
                step_expected_status = str(step.get('expected_status') or '').strip() or '-'
                step_expected_artifact_kind = str(step.get('expected_artifact_kind') or '').strip() or '-'
                step_expected_artifact_path_hint = str(step.get('expected_artifact_path_hint') or '').strip() or '-'
                step_current_artifact = str(step.get('current_artifact') or '').strip() or '-'
                step_current_status = str(step.get('current_status') or '').strip() or '-'
                step_current_recency = str(step.get('current_recency') or '').strip() or '-'
                step_current_age = step.get('current_age_minutes')
                step_current_age_text = str(step_current_age) if step_current_age not in (None, "") else "-"
                step_checkpoint_state = str(step.get('checkpoint_state') or '').strip() or '-'
                lines.append(
                    f"  step_{step_rank}=`{step_name}` script=`{step_script}` "
                    f"expected_status=`{step_expected_status}` "
                    f"output=`{step_expected_artifact_kind}@{step_expected_artifact_path_hint}` "
                    f"current=`{step_checkpoint_state}|{step_current_status}|{step_current_recency}|{step_current_age_text}m|{step_current_artifact}` "
                    f"command=`{step_command}`"
                )
    else:
        lines.append("- No source refresh checklist remains.")
    lines.extend(
        [
            "",
            "## Action Checklist",
        ]
    )
    lines.extend(_action_checklist_lines(operator_action_checklist))
    lines.extend(
        [
            "",
            "## Key Artifacts",
            f"- bridge: `{bridge.get('artifact') or ''}`",
            f"- review: `{review.get('artifact') or ''}`",
            f"- retro: `{retro.get('artifact') or ''}`",
            f"- gap: `{gap.get('artifact') or ''}`",
            f"- brief: `{brief.get('artifact') or ''}`",
        ]
    )
    if bridge_apply and bridge_apply.get("artifact"):
        lines.append(f"- bridge apply: `{bridge_apply.get('artifact')}`")
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            f"- Continue from `{brief.get('next_focus_action') or '-'}` on `{brief.get('next_focus_target') or '-'}`.",
        ]
    )
    if stale_symbols:
        lines.append("- Keep stale queue symbols in watch until they print a fresh directional trigger.")
    lines.extend(
        [
            "- Use `refresh_commodity_paper_execution_state.py` for future refreshes so dependency order stays stable.",
            "",
            "## If Opening A New Window",
            f"- Read `{NEXT_CONTEXT_PATH}` first.",
            f"- Then read `{brief.get('artifact') or ''}`.",
            f"- Then read `{retro.get('artifact') or ''}`.",
            f"- Then read `{gap.get('artifact') or ''}`.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_refresh_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Refresh",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- apply_bridge: `{str(bool(payload.get('apply_bridge', False))).lower()}`",
        f"- context_path: `{payload.get('context_path') or ''}`",
        f"- brief_artifact: `{payload.get('brief_artifact') or ''}`",
        f"- brief_source_artifact: `{payload.get('brief_source_artifact') or ''}`",
        "",
        "## Steps",
    ]
    for row in payload.get("steps", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('name')}`: now=`{row.get('now')}` artifact=`{row.get('artifact') or '-'}` status=`{row.get('status') or '-'}`"
        )
    lines.extend(
        [
            "",
            "## Summary",
            f"- operator_status: `{payload.get('operator_status') or '-'}`",
            f"- operator_stack_brief: `{payload.get('operator_stack_brief') or '-'}`",
            f"- next_focus_action: `{payload.get('next_focus_action') or '-'}`",
            f"- bridge_status: `{payload.get('commodity_execution_bridge_status') or '-'}`",
            f"- review_status: `{payload.get('commodity_execution_review_status') or '-'}`",
            f"- retro_status: `{payload.get('commodity_execution_retro_status') or '-'}`",
            f"- gap_status: `{payload.get('commodity_execution_gap_status') or '-'}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh commodity paper execution artifacts in a stable sequential order.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--context-path", default=str(NEXT_CONTEXT_PATH))
    parser.add_argument("--signal-tickets-json", default="")
    parser.add_argument("--apply-bridge", action="store_true")
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    context_path = Path(args.context_path).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_now = derive_runtime_now(review_dir, args.now)
    python_bin = "python3"
    signal_tickets_json = str(args.signal_tickets_json or "").strip()

    steps: list[dict[str, Any]] = []
    offset = 0

    ticket_lane_now = step_now(runtime_now, offset)
    ticket_lane_payload = run_json_step(
        step_name="ticket_lane_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_ticket_lane.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(ticket_lane_now),
        ],
    )
    steps.append(
        {
            "name": "ticket_lane_refresh",
            "now": fmt_utc(ticket_lane_now),
            "artifact": str(ticket_lane_payload.get("artifact") or ""),
            "status": str(ticket_lane_payload.get("ticket_status") or ""),
        }
    )
    offset += 1

    ticket_book_now = step_now(runtime_now, offset)
    ticket_book_payload = run_json_step(
        step_name="ticket_book_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_ticket_book.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(ticket_book_now),
        ],
    )
    steps.append(
        {
            "name": "ticket_book_refresh",
            "now": fmt_utc(ticket_book_now),
            "artifact": str(ticket_book_payload.get("artifact") or ""),
            "status": str(ticket_book_payload.get("ticket_book_status") or ""),
        }
    )
    offset += 1

    execution_preview_now = step_now(runtime_now, offset)
    execution_preview_payload = run_json_step(
        step_name="execution_preview_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_execution_preview.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(execution_preview_now),
        ],
    )
    steps.append(
        {
            "name": "execution_preview_refresh",
            "now": fmt_utc(execution_preview_now),
            "artifact": str(execution_preview_payload.get("artifact") or ""),
            "status": str(execution_preview_payload.get("execution_preview_status") or ""),
        }
    )
    offset += 1

    execution_artifact_now = step_now(runtime_now, offset)
    execution_artifact_payload = run_json_step(
        step_name="execution_artifact_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_execution_artifact.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(execution_artifact_now),
        ],
    )
    steps.append(
        {
            "name": "execution_artifact_refresh",
            "now": fmt_utc(execution_artifact_now),
            "artifact": str(execution_artifact_payload.get("artifact") or ""),
            "status": str(execution_artifact_payload.get("execution_artifact_status") or ""),
        }
    )
    offset += 1

    execution_queue_now = step_now(runtime_now, offset)
    execution_queue_payload = run_json_step(
        step_name="execution_queue_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_execution_queue.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(execution_queue_now),
        ],
    )
    steps.append(
        {
            "name": "execution_queue_refresh",
            "now": fmt_utc(execution_queue_now),
            "artifact": str(execution_queue_payload.get("artifact") or ""),
            "status": str(execution_queue_payload.get("execution_queue_status") or ""),
        }
    )
    offset += 1

    queue_path = Path(str(execution_queue_payload.get("artifact") or "")).expanduser().resolve()

    bridge_apply_payload: dict[str, Any] | None = None
    if bool(args.apply_bridge):
        bridge_apply_now = step_now(runtime_now, offset)
        bridge_apply_cmd = [
            python_bin,
            str(script_path("bridge_commodity_paper_execution_queue.py")),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            fmt_utc(bridge_apply_now),
            "--execution-queue-json",
            str(queue_path),
            "--apply",
        ]
        if signal_tickets_json:
            bridge_apply_cmd.extend(["--signal-tickets-json", signal_tickets_json])
        bridge_apply_payload = run_json_step(step_name="bridge_apply", cmd=bridge_apply_cmd)
        steps.append(
            {
                "name": "bridge_apply",
                "now": fmt_utc(bridge_apply_now),
                "artifact": str(bridge_apply_payload.get("artifact") or ""),
                "status": str(bridge_apply_payload.get("bridge_status") or ""),
            }
        )
        offset += 1

    bridge_now = step_now(runtime_now, offset)
    bridge_cmd = [
        python_bin,
        str(script_path("bridge_commodity_paper_execution_queue.py")),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--now",
        fmt_utc(bridge_now),
        "--execution-queue-json",
        str(queue_path),
    ]
    if signal_tickets_json:
        bridge_cmd.extend(["--signal-tickets-json", signal_tickets_json])
    bridge_payload = run_json_step(step_name="bridge_refresh", cmd=bridge_cmd)
    steps.append(
        {
            "name": "bridge_refresh",
            "now": fmt_utc(bridge_now),
            "artifact": str(bridge_payload.get("artifact") or ""),
            "status": str(bridge_payload.get("bridge_status") or ""),
        }
    )
    offset += 1

    review_now = step_now(runtime_now, offset)
    review_payload = run_json_step(
        step_name="review_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_execution_review.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(review_now),
            "--execution-queue-json",
            str(queue_path),
        ],
    )
    steps.append(
        {
            "name": "review_refresh",
            "now": fmt_utc(review_now),
            "artifact": str(review_payload.get("artifact") or ""),
            "status": str(review_payload.get("execution_review_status") or ""),
        }
    )
    offset += 1

    retro_now = step_now(runtime_now, offset)
    retro_payload = run_json_step(
        step_name="retro_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_execution_retro.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(retro_now),
            "--execution-review-json",
            str(review_payload.get("artifact") or ""),
        ],
    )
    steps.append(
        {
            "name": "retro_refresh",
            "now": fmt_utc(retro_now),
            "artifact": str(retro_payload.get("artifact") or ""),
            "status": str(retro_payload.get("execution_retro_status") or ""),
        }
    )
    offset += 1

    gap_now = step_now(runtime_now, offset)
    gap_payload = run_json_step(
        step_name="gap_refresh",
        cmd=[
            python_bin,
            str(script_path("build_commodity_paper_execution_gap_report.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(gap_now),
            "--execution-queue-json",
            str(queue_path),
            "--execution-review-json",
            str(review_payload.get("artifact") or ""),
            "--execution-retro-json",
            str(retro_payload.get("artifact") or ""),
            "--execution-bridge-json",
            str(bridge_payload.get("artifact") or ""),
        ],
    )
    steps.append(
        {
            "name": "gap_refresh",
            "now": fmt_utc(gap_now),
            "artifact": str(gap_payload.get("artifact") or ""),
            "status": str(gap_payload.get("gap_status") or ""),
        }
    )
    offset += 1

    brief_now = step_now(runtime_now, offset)
    brief_payload = run_json_step(
        step_name="brief_refresh",
        cmd=[
            python_bin,
            str(script_path("build_hot_universe_operator_brief.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(brief_now),
            "--commodity-execution-queue-json",
            str(queue_path),
            "--commodity-execution-review-json",
            str(review_payload.get("artifact") or ""),
            "--commodity-execution-retro-json",
            str(retro_payload.get("artifact") or ""),
            "--commodity-execution-gap-json",
            str(gap_payload.get("artifact") or ""),
            "--commodity-execution-bridge-json",
            str(bridge_payload.get("artifact") or ""),
        ],
    )
    steps.append(
        {
            "name": "brief_refresh",
            "now": fmt_utc(brief_now),
            "artifact": str(brief_payload.get("artifact") or ""),
            "status": str(brief_payload.get("operator_status") or ""),
        }
    )

    context_text = render_context_markdown(
        runtime_now=brief_now,
        brief=brief_payload,
        review=review_payload,
        retro=retro_payload,
        gap=gap_payload,
        bridge=bridge_payload,
        bridge_apply=bridge_apply_payload,
    )
    context_path.write_text(context_text, encoding="utf-8")

    stamp = brief_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_refresh.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_refresh.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_refresh_checksum.json"
    brief_source_artifact = str(brief_payload.get("artifact") or "")
    brief_snapshot_path = write_hot_brief_snapshot(
        review_dir,
        stamp=stamp,
        brief_payload=brief_payload,
    )
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(brief_now),
        "apply_bridge": bool(args.apply_bridge),
        "context_path": str(context_path),
        "context_sha256": sha256_file(context_path),
        "steps": steps,
        "ticket_lane_artifact": str(ticket_lane_payload.get("artifact") or ""),
        "ticket_book_artifact": str(ticket_book_payload.get("artifact") or ""),
        "execution_preview_artifact": str(execution_preview_payload.get("artifact") or ""),
        "execution_artifact": str(execution_artifact_payload.get("artifact") or ""),
        "execution_queue_artifact": str(execution_queue_payload.get("artifact") or ""),
        "bridge_apply_artifact": str((bridge_apply_payload or {}).get("artifact") or ""),
        "bridge_artifact": str(bridge_payload.get("artifact") or ""),
        "review_artifact": str(review_payload.get("artifact") or ""),
        "retro_artifact": str(retro_payload.get("artifact") or ""),
        "gap_artifact": str(gap_payload.get("artifact") or ""),
        "brief_artifact": str(brief_snapshot_path),
        "brief_snapshot_artifact": str(brief_snapshot_path),
        "brief_source_artifact": brief_source_artifact,
        "operator_status": str(brief_payload.get("operator_status") or ""),
        "operator_stack_brief": str(brief_payload.get("operator_stack_brief") or ""),
        "next_focus_action": str(brief_payload.get("next_focus_action") or ""),
        "commodity_execution_bridge_status": str(brief_payload.get("commodity_execution_bridge_status") or ""),
        "commodity_execution_review_status": str(brief_payload.get("commodity_execution_review_status") or ""),
        "commodity_execution_retro_status": str(brief_payload.get("commodity_execution_retro_status") or ""),
        "commodity_execution_gap_status": str(brief_payload.get("commodity_execution_gap_status") or ""),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_refresh_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": payload["as_of"],
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
            {"path": str(brief_snapshot_path), "sha256": sha256_file(brief_snapshot_path)},
            {"path": str(context_path), "sha256": sha256_file(context_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="commodity_paper_execution_refresh",
        current_paths=[json_path, md_path, checksum_path, brief_snapshot_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=brief_now,
    )
    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["files"][0]["sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
