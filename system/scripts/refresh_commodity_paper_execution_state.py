#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


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
    operator_crypto_route_alignment_focus_slot = str(
        brief.get("operator_crypto_route_alignment_focus_slot") or ""
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
    secondary_focus_area = str(brief.get("secondary_focus_area") or "").strip()
    secondary_focus_target = str(brief.get("secondary_focus_target") or "").strip()
    secondary_focus_symbol = str(brief.get("secondary_focus_symbol") or "").strip()
    secondary_focus_action = str(brief.get("secondary_focus_action") or "").strip()
    secondary_focus_reason = str(brief.get("secondary_focus_reason") or "").strip()
    secondary_focus_state = str(brief.get("secondary_focus_state") or "").strip()
    secondary_focus_blocker_detail = str(brief.get("secondary_focus_blocker_detail") or "").strip()
    secondary_focus_done_when = str(brief.get("secondary_focus_done_when") or "").strip()
    operator_action_queue_brief = str(brief.get("operator_action_queue_brief") or "").strip()
    operator_action_checklist = [
        dict(row) for row in brief.get("operator_action_checklist", []) if isinstance(row, dict)
    ]
    operator_action_checklist_brief = str(brief.get("operator_action_checklist_brief") or "").strip()
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
                    f"slot={operator_crypto_route_alignment_focus_slot or '-'}",
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
    if operator_source_refresh_queue_brief:
        lines.append("- Source refresh queue is: `" + operator_source_refresh_queue_brief + "`")
    if operator_source_refresh_pipeline_pending_brief:
        lines.append("- Source refresh pipeline pending is: `" + operator_source_refresh_pipeline_pending_brief + "`")
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
            f"- `operator_crypto_route_alignment_focus_slot = {operator_crypto_route_alignment_focus_slot or '-'}`",
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
            f"- `operator_action_queue_brief = {operator_action_queue_brief or '-'}`",
            f"- `operator_action_checklist_brief = {operator_action_checklist_brief or '-'}`",
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
            f"- focus slot refresh backlog: `{operator_focus_slot_refresh_backlog_brief or '-'}`",
            f"- focus slot promotion gate: `{operator_focus_slot_promotion_gate_brief or '-'}`",
            f"- focus slot actionability gate: `{operator_focus_slot_actionability_gate_brief or '-'}`",
            f"- focus slot readiness gate: `{operator_focus_slot_readiness_gate_brief or '-'}`",
            f"- research embedding quality: `{operator_research_embedding_quality_brief or '-'}`",
            f"- crypto route alignment: `{operator_crypto_route_alignment_brief or '-'}`",
            f"- crypto route alignment slot: `{operator_crypto_route_alignment_focus_slot or '-'}`",
            f"- crypto route alignment recovery outcome: `{operator_crypto_route_alignment_recovery_brief or '-'}`",
            f"- crypto route alignment cooldown: `{operator_crypto_route_alignment_cooldown_brief or '-'}`",
            f"- crypto route alignment recovery recipe gate: `{operator_crypto_route_alignment_recipe_brief or '-'}`",
            f"- crypto route alignment recovery: `{_list_text(operator_crypto_route_alignment_recipe_target_batches) or '-'}@{operator_crypto_route_alignment_recipe_window_days or '-'}d`",
            f"- source refresh queue: `{operator_source_refresh_queue_brief or '-'}`",
            f"- source refresh checklist: `{operator_source_refresh_checklist_brief or '-'}`",
            f"- source refresh pipeline: `{operator_source_refresh_pipeline_pending_brief or '-'}`",
            f"- source refresh pipeline deferred: `{operator_source_refresh_pipeline_deferred_brief or '-'}`",
            "",
            "## Focus Slot Artifacts",
            f"- primary source: `{next_focus_source_kind or '-'} | {next_focus_source_status or '-'} | {next_focus_source_recency or '-'} | {next_focus_source_health or '-'} | {next_focus_source_refresh_action or '-'} | {next_focus_source_age_minutes or '-'}m | {next_focus_source_as_of or '-'} | {next_focus_source_artifact or '-'}`",
            f"- followup source: `{followup_focus_source_kind or '-'} | {followup_focus_source_status or '-'} | {followup_focus_source_recency or '-'} | {followup_focus_source_health or '-'} | {followup_focus_source_refresh_action or '-'} | {followup_focus_source_age_minutes or '-'}m | {followup_focus_source_as_of or '-'} | {followup_focus_source_artifact or '-'}`",
            f"- secondary source: `{secondary_focus_source_kind or '-'} | {secondary_focus_source_status or '-'} | {secondary_focus_source_recency or '-'} | {secondary_focus_source_health or '-'} | {secondary_focus_source_refresh_action or '-'} | {secondary_focus_source_age_minutes or '-'}m | {secondary_focus_source_as_of or '-'} | {secondary_focus_source_artifact or '-'}`",
            "",
            "## Focus Slot Refresh Backlog",
        ]
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
            f"- slot=`{operator_crypto_route_alignment_focus_slot or '-'}` status=`{operator_crypto_route_alignment_status or '-'}` brief=`{operator_crypto_route_alignment_brief or '-'}`",
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
            "## Source Refresh Queue",
        ]
    )
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
    queue_path = latest_stamped_artifact(review_dir, "commodity_paper_execution_queue")

    steps: list[dict[str, Any]] = []

    bridge_apply_payload: dict[str, Any] | None = None
    offset = 0
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
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(brief_now),
        "apply_bridge": bool(args.apply_bridge),
        "context_path": str(context_path),
        "context_sha256": sha256_file(context_path),
        "steps": steps,
        "bridge_apply_artifact": str((bridge_apply_payload or {}).get("artifact") or ""),
        "bridge_artifact": str(bridge_payload.get("artifact") or ""),
        "review_artifact": str(review_payload.get("artifact") or ""),
        "retro_artifact": str(retro_payload.get("artifact") or ""),
        "gap_artifact": str(gap_payload.get("artifact") or ""),
        "brief_artifact": str(brief_payload.get("artifact") or ""),
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
            {"path": str(context_path), "sha256": sha256_file(context_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="commodity_paper_execution_refresh",
        current_paths=[json_path, md_path, checksum_path],
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
