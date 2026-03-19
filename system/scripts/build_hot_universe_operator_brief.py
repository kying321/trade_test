#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_CRYPTO_ROUTE_REFRESH_BATCHES = ("crypto_hot", "crypto_majors", "crypto_beta")
SOURCE_TEXT = "text"
SOURCE_INT = "int"
SOURCE_RAW = "raw"
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from operator_brief_debug_snapshots import (
    brooks_source_debug_lines as _brooks_source_debug_lines,
    cross_market_runtime_debug_lines as _cross_market_runtime_debug_lines,
    cross_market_source_debug_lines as _cross_market_source_debug_lines,
    crypto_route_source_debug_lines as _crypto_route_source_debug_lines,
    remote_live_source_debug_lines as _remote_live_source_debug_lines,
)
from operator_brief_source_chunks import (
    build_brooks_runtime_chunk as _build_brooks_runtime_chunk,
    build_brooks_source_chunk as _build_brooks_source_chunk,
    build_cross_market_source_snapshot_chunk as _build_cross_market_source_snapshot_chunk,
    build_cross_market_runtime_chunk as _build_cross_market_runtime_chunk,
    build_crypto_route_source_chunk as _build_crypto_route_source_chunk,
    build_research_embedding_quality_chunk as _build_research_embedding_quality_chunk,
    build_remote_live_source_chunk as _build_remote_live_source_chunk,
)

_CROSS_MARKET_SOURCE_MIRROR_SPECS: tuple[tuple[str, str], ...] = (
    ("status", SOURCE_TEXT),
    ("as_of", SOURCE_TEXT),
    ("review_backlog_status", SOURCE_TEXT),
    ("review_backlog_count", SOURCE_INT),
    ("review_backlog_brief", SOURCE_TEXT),
    ("review_head_area", SOURCE_TEXT),
    ("review_head_symbol", SOURCE_TEXT),
    ("review_head_action", SOURCE_TEXT),
    ("review_head_priority_score", SOURCE_RAW),
    ("review_head_priority_tier", SOURCE_TEXT),
    ("operator_backlog_status", SOURCE_TEXT),
    ("operator_backlog_count", SOURCE_INT),
    ("operator_backlog_brief", SOURCE_TEXT),
    ("operator_backlog_state_brief", SOURCE_TEXT),
    ("operator_backlog_priority_totals_brief", SOURCE_TEXT),
    ("operator_state_lane_heads_brief", SOURCE_TEXT),
    ("operator_state_lane_priority_order_brief", SOURCE_TEXT),
    ("remote_live_operator_alignment_status", SOURCE_TEXT),
    ("remote_live_operator_alignment_brief", SOURCE_TEXT),
    ("remote_live_operator_alignment_blocker_detail", SOURCE_TEXT),
    ("remote_live_operator_alignment_done_when", SOURCE_TEXT),
    ("remote_live_takeover_gate_status", SOURCE_TEXT),
    ("remote_live_takeover_gate_brief", SOURCE_TEXT),
    ("remote_live_takeover_gate_blocker_detail", SOURCE_TEXT),
    ("remote_live_takeover_gate_done_when", SOURCE_TEXT),
    ("remote_live_takeover_clearing_status", SOURCE_TEXT),
    ("remote_live_takeover_clearing_brief", SOURCE_TEXT),
    ("remote_live_takeover_clearing_blocker_detail", SOURCE_TEXT),
    ("remote_live_takeover_clearing_done_when", SOURCE_TEXT),
    ("remote_live_takeover_clearing_source_freshness_brief", SOURCE_TEXT),
    ("remote_live_takeover_slot_anomaly_breakdown_status", SOURCE_TEXT),
    ("remote_live_takeover_slot_anomaly_breakdown_brief", SOURCE_TEXT),
    ("remote_live_takeover_slot_anomaly_breakdown_artifact", SOURCE_TEXT),
    ("remote_live_takeover_slot_anomaly_breakdown_repair_focus", SOURCE_TEXT),
    ("operator_head_area", SOURCE_TEXT),
    ("operator_head_symbol", SOURCE_TEXT),
    ("operator_head_action", SOURCE_TEXT),
    ("operator_head_state", SOURCE_TEXT),
    ("operator_head_priority_score", SOURCE_RAW),
    ("operator_head_priority_tier", SOURCE_TEXT),
    ("review_head_lane_status", SOURCE_TEXT),
    ("review_head_lane_brief", SOURCE_TEXT),
    ("operator_head_lane_status", SOURCE_TEXT),
    ("operator_head_lane_brief", SOURCE_TEXT),
    ("operator_repair_head_lane_status", SOURCE_TEXT),
    ("operator_repair_head_lane_brief", SOURCE_TEXT),
    ("operator_repair_queue_brief", SOURCE_TEXT),
    ("operator_repair_queue_count", SOURCE_INT),
    ("operator_repair_checklist_brief", SOURCE_TEXT),
    ("operator_waiting_lane_status", SOURCE_TEXT),
    ("operator_waiting_lane_count", SOURCE_INT),
    ("operator_waiting_lane_brief", SOURCE_TEXT),
    ("operator_waiting_lane_priority_total", SOURCE_INT),
    ("operator_waiting_lane_head_symbol", SOURCE_TEXT),
    ("operator_waiting_lane_head_action", SOURCE_TEXT),
    ("operator_waiting_lane_head_priority_score", SOURCE_INT),
    ("operator_waiting_lane_head_priority_tier", SOURCE_TEXT),
    ("operator_review_lane_status", SOURCE_TEXT),
    ("operator_review_lane_count", SOURCE_INT),
    ("operator_review_lane_brief", SOURCE_TEXT),
    ("operator_review_lane_priority_total", SOURCE_INT),
    ("operator_review_lane_head_symbol", SOURCE_TEXT),
    ("operator_review_lane_head_action", SOURCE_TEXT),
    ("operator_review_lane_head_priority_score", SOURCE_INT),
    ("operator_review_lane_head_priority_tier", SOURCE_TEXT),
    ("operator_watch_lane_status", SOURCE_TEXT),
    ("operator_watch_lane_count", SOURCE_INT),
    ("operator_watch_lane_brief", SOURCE_TEXT),
    ("operator_watch_lane_priority_total", SOURCE_INT),
    ("operator_watch_lane_head_symbol", SOURCE_TEXT),
    ("operator_watch_lane_head_action", SOURCE_TEXT),
    ("operator_watch_lane_head_priority_score", SOURCE_INT),
    ("operator_watch_lane_head_priority_tier", SOURCE_TEXT),
    ("operator_blocked_lane_status", SOURCE_TEXT),
    ("operator_blocked_lane_count", SOURCE_INT),
    ("operator_blocked_lane_brief", SOURCE_TEXT),
    ("operator_blocked_lane_priority_total", SOURCE_INT),
    ("operator_blocked_lane_head_symbol", SOURCE_TEXT),
    ("operator_blocked_lane_head_action", SOURCE_TEXT),
    ("operator_blocked_lane_head_priority_score", SOURCE_INT),
    ("operator_blocked_lane_head_priority_tier", SOURCE_TEXT),
    ("operator_repair_lane_status", SOURCE_TEXT),
    ("operator_repair_lane_count", SOURCE_INT),
    ("operator_repair_lane_brief", SOURCE_TEXT),
    ("operator_repair_lane_priority_total", SOURCE_INT),
    ("operator_repair_lane_head_symbol", SOURCE_TEXT),
    ("operator_repair_lane_head_action", SOURCE_TEXT),
    ("operator_repair_lane_head_priority_score", SOURCE_INT),
    ("operator_repair_lane_head_priority_tier", SOURCE_TEXT),
)


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


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


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


def _mirror_source_prefixed_fields(
    *,
    prefix: str,
    source_payload: dict[str, Any] | None,
    specs: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    payload = dict(source_payload or {})
    mirrored: dict[str, Any] = {}
    for field, kind in specs:
        value = payload.get(field)
        output_key = f"{prefix}{field}"
        if kind == SOURCE_INT:
            mirrored[output_key] = int(value or 0)
        elif kind == SOURCE_RAW:
            mirrored[output_key] = value
        else:
            mirrored[output_key] = str(value or "")
    return mirrored


def _compact_snapshot_parts(*parts: str) -> str:
    return " | ".join([part for part in (str(x).strip() for x in parts) if part and part != "-"])










def _payload_richness(payload: dict[str, Any]) -> int:
    action_ladder = dict(payload.get("research_action_ladder") or {})
    crypto_route_brief = dict(payload.get("crypto_route_brief") or {})
    score = 0
    score += len([x for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]) * 4
    score += len([x for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()]) * 3
    score += len([x for x in action_ladder.get("shadow_only_batches", []) if str(x).strip()]) * 2
    if str(crypto_route_brief.get("operator_status") or "").strip():
        score += 3
    if str(crypto_route_brief.get("route_stack_brief") or "").strip():
        score += 2
    if str(crypto_route_brief.get("next_focus_symbol") or "").strip():
        score += 1
    return score


def _action_richness(payload: dict[str, Any]) -> int:
    action_ladder = dict(payload.get("research_action_ladder") or {})
    score = 0
    score += len([x for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]) * 5
    score += len([x for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()]) * 4
    score += len([x for x in action_ladder.get("research_queue_batches", []) if str(x).strip()]) * 3
    score += len([x for x in action_ladder.get("shadow_only_batches", []) if str(x).strip()]) * 2
    score += len([x for x in action_ladder.get("avoid_batches", []) if str(x).strip()])
    return score


def _crypto_richness(payload: dict[str, Any]) -> int:
    crypto_route_brief = dict(payload.get("crypto_route_brief") or {})
    crypto_route_operator_brief = dict(payload.get("crypto_route_operator_brief") or {})
    score = 0
    if str(crypto_route_brief.get("operator_status") or "").strip():
        score += 4
    if str(crypto_route_brief.get("route_stack_brief") or "").strip():
        score += 3
    if str(crypto_route_brief.get("next_focus_symbol") or "").strip():
        score += 2
    if str(crypto_route_brief.get("next_retest_action") or "").strip():
        score += 1
    if str(crypto_route_operator_brief.get("focus_window_floor") or "").strip():
        score += 2
    if str(crypto_route_operator_brief.get("price_state_window_floor") or "").strip():
        score += 1
    if str(crypto_route_operator_brief.get("comparative_window_takeaway") or "").strip():
        score += 2
    if str(crypto_route_operator_brief.get("xlong_flow_window_floor") or "").strip():
        score += 3
    if str(crypto_route_operator_brief.get("xlong_comparative_window_takeaway") or "").strip():
        score += 2
    return score


def _hot_universe_status_priority(status: str) -> int:
    text = str(status or "").strip()
    if text == "ok":
        return 3
    if text == "partial_failure":
        return 2
    if text == "dry_run":
        return 1
    return 0


def latest_hot_universe_research(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        raise FileNotFoundError("no_hot_universe_research_artifact")
    ordered = sorted(candidates, key=lambda path: artifact_sort_key(path, reference_now), reverse=True)
    preferred: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    all_ranked: list[tuple[int, int, tuple[int, str, float, str], str, Path]] = []
    for path in ordered:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status") or "").strip()
        richness = _payload_richness(payload)
        status_priority = _hot_universe_status_priority(status)
        sort_key = artifact_sort_key(path, reference_now)
        all_ranked.append((status_priority, richness, sort_key, status, path))
        if status != "dry_run":
            preferred.append((status_priority, sort_key, richness, path))
    if preferred:
        best_non_dry = max(preferred, key=lambda item: (item[0], item[1], item[2]))
        best_overall = max(all_ranked, key=lambda item: (item[1], item[0], item[2]))
        # Prefer a richer dry-run artifact over a sparse older "ok" artifact.
        if best_non_dry[2] > 0 or best_overall[3] != "dry_run":
            return best_non_dry[3]
        return best_overall[4]
    return ordered[0]


def latest_hot_universe_action_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        raise FileNotFoundError("no_hot_universe_research_artifact")
    ordered = sorted(candidates, key=lambda path: artifact_sort_key(path, reference_now), reverse=True)
    preferred: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    fallback: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    for path in ordered:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        richness = _action_richness(payload)
        status_priority = _hot_universe_status_priority(str(payload.get("status") or ""))
        sort_key = artifact_sort_key(path, reference_now)
        fallback.append((status_priority, sort_key, richness, path))
        if str(payload.get("status") or "").strip() != "dry_run":
            preferred.append((status_priority, sort_key, richness, path))
    if preferred:
        best_non_dry = max(preferred, key=lambda item: (item[0], item[1], item[2]))
        if best_non_dry[2] > 0:
            return best_non_dry[3]
    return max(fallback, key=lambda item: (item[1], item[0], item[2]))[3]


def latest_hot_universe_crypto_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        raise FileNotFoundError("no_hot_universe_research_artifact")
    ordered = sorted(candidates, key=lambda path: artifact_sort_key(path, reference_now), reverse=True)
    preferred: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    ranked: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    for path in ordered:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status") or "").strip()
        entry = (
            _hot_universe_status_priority(status),
            artifact_sort_key(path, reference_now),
            _crypto_richness(payload),
            path,
        )
        ranked.append(entry)
        if status != "dry_run":
            preferred.append(entry)
    if preferred:
        best_non_dry = max(preferred, key=lambda item: (item[0], item[1], item[2]))
        if best_non_dry[2] > 0:
            return best_non_dry[3]
    return max(ranked, key=lambda item: (item[1], item[0], item[2]))[3]


def latest_crypto_route_focus_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    operator_candidates = list(review_dir.glob("*_crypto_route_operator_brief.json"))
    if operator_candidates:
        return max(operator_candidates, key=lambda path: artifact_sort_key(path, reference_now))
    brief_candidates = list(review_dir.glob("*_crypto_route_brief.json"))
    if brief_candidates:
        return max(brief_candidates, key=lambda path: artifact_sort_key(path, reference_now))
    try:
        return latest_hot_universe_crypto_source(review_dir, reference_now)
    except FileNotFoundError:
        return None


def latest_crypto_route_refresh_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_crypto_route_refresh.json"))
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_remote_live_history_audit_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_remote_live_history_audit.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_remote_live_handoff_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_remote_live_handoff.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_live_gate_blocker_report_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_live_gate_blocker_report.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_brooks_price_action_route_report_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_brooks_price_action_route_report.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_brooks_price_action_execution_plan_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_brooks_price_action_execution_plan.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_brooks_structure_review_queue_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_brooks_structure_review_queue.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_brooks_structure_refresh_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_brooks_structure_refresh.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_cross_market_operator_state_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_cross_market_operator_state.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_system_time_sync_repair_plan_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_system_time_sync_repair_plan.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_system_time_sync_repair_verification_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_system_time_sync_repair_verification_report.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_openclaw_orderflow_blueprint_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = [
        path
        for path in review_dir.glob("*_openclaw_orderflow_blueprint.json")
        if artifact_stamp(path)
    ]
    if not candidates:
        return None
    if reference_now is not None:
        nonfuture_candidates = [
            path
            for path in candidates
            if (stamp_dt := parsed_artifact_stamp(path)) is None or stamp_dt <= reference_now
        ]
        if nonfuture_candidates:
            candidates = nonfuture_candidates
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def _unwrap_remote_live_handoff_operator_payload(
    source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    source = source_payload if isinstance(source_payload, dict) else {}
    nested = source.get("operator_handoff")
    return nested if isinstance(nested, dict) else {}


def latest_commodity_execution_lane_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_execution_lane.json"))
    if candidates:
        return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))
    blocker_candidates = list(review_dir.glob("*_live_gate_blocker_report.json"))
    if blocker_candidates:
        return max(blocker_candidates, key=lambda path: artifact_sort_key(path, reference_now))
    return None


def latest_commodity_paper_ticket_lane_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_ticket_lane.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_ticket_book_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_ticket_book.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_preview_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_preview.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_artifact_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_artifact.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_queue_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_queue.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_review_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_review.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_retro_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_retro.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_gap_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_gap_report.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_bridge_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_bridge.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def resolve_explicit_source(raw: str | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalize_commodity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "commodity_execution_path" not in payload:
        return dict(payload)
    route = dict(payload.get("commodity_execution_path") or {})
    primary = [str(x).strip() for x in route.get("focus_primary_batches", []) if str(x).strip()]
    regime = [str(x).strip() for x in route.get("focus_with_regime_filter_batches", []) if str(x).strip()]
    shadow = [str(x).strip() for x in route.get("shadow_only_batches", []) if str(x).strip()]
    leaders_primary = [str(x).strip().upper() for x in route.get("leader_symbols_primary", []) if str(x).strip()]
    leaders_regime = [str(x).strip().upper() for x in route.get("leader_symbols_regime_filter", []) if str(x).strip()]
    next_focus_batch = primary[0] if primary else (regime[0] if regime else "")
    next_focus_symbols = leaders_primary if next_focus_batch in set(primary) else leaders_regime
    route_stack = []
    if primary:
        route_stack.append("paper-primary:" + ",".join(primary))
    if regime:
        route_stack.append("regime-filter:" + ",".join(regime))
    if shadow:
        route_stack.append("shadow:" + ",".join(shadow))
    return {
        "status": "ok",
        "route_status": "paper-first",
        "execution_mode": str(route.get("execution_mode") or "paper_first"),
        "focus_primary_batches": primary,
        "focus_with_regime_filter_batches": regime,
        "shadow_only_batches": shadow,
        "avoid_batches": [str(x).strip() for x in route.get("avoid_batches", []) if str(x).strip()],
        "leader_symbols_primary": leaders_primary,
        "leader_symbols_regime_filter": leaders_regime,
        "next_focus_batch": next_focus_batch,
        "next_focus_symbols": next_focus_symbols,
        "next_stage": "paper_ticket_lane",
        "route_stack_brief": " | ".join(route_stack),
        "summary_text": "",
    }


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


def _list_text(values: list[str], limit: int = 4) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _mapping_text(values: dict[str, Any], limit: int = 4) -> str:
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


def _remote_live_history_window_map(source_payload: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    payload = dict(source_payload or {})
    rows = payload.get("window_summaries")
    if not isinstance(rows, list):
        rows = []
    window_map: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            window_hours = int(row.get("window_hours") or 0)
        except Exception:
            continue
        if window_hours <= 0:
            continue
        window_map[window_hours] = dict(row)
    return window_map


def _remote_live_history_snapshot_row(window_map: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for window_hours in (24, 168, 720):
        row = window_map.get(window_hours)
        if row:
            return dict(row)
    if not window_map:
        return {}
    latest_window = max(window_map)
    return dict(window_map.get(latest_window) or {})


def _remote_live_history_longest_row(window_map: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not window_map:
        return {}
    return dict(window_map.get(max(window_map)) or {})


def _remote_live_history_window_brief(row: dict[str, Any]) -> str:
    if not row:
        return "-"
    label = str(row.get("history_window_label") or "").strip()
    if not label:
        try:
            hours = int(row.get("window_hours") or 0)
        except Exception:
            hours = 0
        label = "24h" if hours == 24 else "7d" if hours == 168 else f"{int(hours // 24)}d" if hours else "-"
    return (
        f"{label}:{_fmt_num(row.get('closed_pnl'))}pnl/"
        f"{int(row.get('trade_count') or 0)}tr/"
        f"{int(row.get('open_positions') or 0)}open"
    )


def _crypto_route_refresh_reuse_audit(source_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(source_payload or {})
    if not payload:
        return {
            "reuse_status": "",
            "reuse_brief": "",
            "reuse_note": "",
            "done_when": "",
            "native_step_count": None,
            "reused_native_count": None,
            "missing_reused_count": None,
            "native_refresh_mode": "",
        }
    steps = [dict(row) for row in payload.get("steps", []) if isinstance(row, dict)]
    native_steps = [row for row in steps if str(row.get("name") or "").startswith("native_")]
    native_step_count = len(native_steps)
    reused_native_count = sum(
        1 for row in native_steps if str(row.get("status") or "").strip() == "reused_previous_artifact"
    )
    missing_reused_count = sum(
        1 for row in native_steps if str(row.get("status") or "").strip() == "missing_reused_source"
    )
    native_refresh_mode = str(payload.get("native_refresh_mode") or "").strip()
    if native_step_count <= 0:
        reuse_status = "native_audit_unavailable"
    elif reused_native_count == native_step_count and native_step_count > 0:
        reuse_status = "reused_native_inputs"
    elif reused_native_count > 0:
        reuse_status = "mixed_native_inputs"
    elif missing_reused_count > 0:
        reuse_status = "native_reuse_incomplete"
    else:
        reuse_status = "fresh_native_inputs"
    reuse_brief = (
        f"{reuse_status}:{native_refresh_mode or '-'}:{reused_native_count}/{native_step_count}"
        if native_step_count > 0
        else f"{reuse_status}:{native_refresh_mode or '-'}"
    )
    if native_step_count <= 0:
        reuse_note = "crypto_route_refresh did not expose any native_* steps to audit"
        done_when = "record native refresh steps before relying on reuse audit"
    elif reuse_status == "reused_native_inputs":
        reuse_note = (
            f"crypto_route_refresh is currently reusing {reused_native_count}/{native_step_count} "
            f"native inputs via {native_refresh_mode or 'unknown_mode'}"
        )
        done_when = "run full native refresh only when fresh native recomputation is required"
    elif reuse_status == "mixed_native_inputs":
        reuse_note = (
            f"crypto_route_refresh is mixing reused and refreshed native inputs "
            f"({reused_native_count}/{native_step_count} reused)"
        )
        done_when = "stabilize native refresh mode before treating route refresh inputs as uniform"
    elif reuse_status == "native_reuse_incomplete":
        reuse_note = (
            f"crypto_route_refresh could not reuse all expected native inputs "
            f"({missing_reused_count} missing of {native_step_count})"
        )
        done_when = "fill missing native sources or rerun guarded native refresh"
    else:
        reuse_note = f"crypto_route_refresh refreshed native inputs directly ({native_step_count}/{native_step_count})"
        done_when = "keep using the current crypto_route_refresh artifact while it stays fresh enough"
    return {
        "reuse_status": reuse_status,
        "reuse_brief": reuse_brief,
        "reuse_note": reuse_note,
        "done_when": done_when,
        "native_step_count": native_step_count,
        "reused_native_count": reused_native_count,
        "missing_reused_count": missing_reused_count,
        "native_refresh_mode": native_refresh_mode,
    }


def _crypto_route_refresh_reuse_gate(
    source_payload: dict[str, Any] | None,
    fallback_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(source_payload or {})
    audit = dict(fallback_audit or {})
    gate_level = str(payload.get("crypto_route_refresh_reuse_level") or "").strip()
    gate_status = str(payload.get("crypto_route_refresh_reuse_gate_status") or "").strip()
    gate_brief = str(payload.get("crypto_route_refresh_reuse_gate_brief") or "").strip()
    gate_blocker_detail = str(payload.get("crypto_route_refresh_reuse_gate_blocker_detail") or "").strip()
    gate_done_when = str(payload.get("crypto_route_refresh_reuse_gate_done_when") or "").strip()
    gate_blocking_raw = payload.get("crypto_route_refresh_reuse_gate_blocking")
    if gate_status or gate_brief or gate_level or gate_blocker_detail or gate_done_when:
        return {
            "level": gate_level,
            "status": gate_status,
            "brief": gate_brief,
            "blocking": bool(gate_blocking_raw) if gate_blocking_raw is not None else None,
            "blocker_detail": gate_blocker_detail,
            "done_when": gate_done_when,
        }

    reuse_status = str(audit.get("reuse_status") or "").strip()
    reuse_brief = str(audit.get("reuse_brief") or "").strip()
    reuse_note = str(audit.get("reuse_note") or "").strip()
    reuse_done_when = str(audit.get("done_when") or "").strip()
    if reuse_status == "reused_native_inputs":
        return {
            "level": "informational",
            "status": "reuse_non_blocking",
            "brief": reuse_brief.replace("reused_native_inputs", "reuse_non_blocking", 1),
            "blocking": False,
            "blocker_detail": reuse_note,
            "done_when": reuse_done_when,
        }
    if reuse_status == "fresh_native_inputs":
        return {
            "level": "informational",
            "status": "fresh_non_blocking",
            "brief": reuse_brief.replace("fresh_native_inputs", "fresh_non_blocking", 1),
            "blocking": False,
            "blocker_detail": reuse_note,
            "done_when": reuse_done_when,
        }
    if reuse_status == "mixed_native_inputs":
        return {
            "level": "blocking",
            "status": "mixed_requires_full_native_refresh",
            "brief": reuse_brief.replace("mixed_native_inputs", "mixed_requires_full_native_refresh", 1),
            "blocking": True,
            "blocker_detail": reuse_note,
            "done_when": reuse_done_when or "rerun crypto_route_refresh with a full native refresh before promotion",
        }
    if reuse_status == "native_reuse_incomplete":
        return {
            "level": "blocking",
            "status": "audit_missing_requires_full_native_refresh",
            "brief": reuse_brief.replace("native_reuse_incomplete", "audit_missing_requires_full_native_refresh", 1),
            "blocking": True,
            "blocker_detail": reuse_note,
            "done_when": reuse_done_when or "fill missing native sources or rerun guarded native refresh",
        }
    if reuse_status == "native_audit_unavailable":
        return {
            "level": "blocking",
            "status": "audit_missing_requires_full_native_refresh",
            "brief": reuse_brief.replace("native_audit_unavailable", "audit_missing_requires_full_native_refresh", 1),
            "blocking": True,
            "blocker_detail": reuse_note,
            "done_when": reuse_done_when or "record native refresh steps before relying on reuse audit",
        }
    return {
        "level": "",
        "status": "",
        "brief": "",
        "blocking": None,
        "blocker_detail": "",
        "done_when": "",
    }


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


def _review_priority_head_row(
    queue: list[dict[str, Any]],
    *,
    head_symbol: str,
) -> dict[str, Any]:
    symbol_text = str(head_symbol or "").strip().upper()
    if symbol_text:
        for row in queue:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol") or "").strip().upper() == symbol_text:
                return dict(row)
    for row in queue:
        if isinstance(row, dict) and str(row.get("symbol") or "").strip():
            return dict(row)
    return {}


def _find_focus_slot_row(
    items: list[dict[str, Any]],
    *,
    area: str = "",
    symbol: str = "",
    action: str = "",
) -> dict[str, Any]:
    area_text = str(area or "").strip()
    symbol_text = str(symbol or "").strip().upper()
    action_text = str(action or "").strip()
    fallback: dict[str, Any] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        row_area = str(row.get("area") or "").strip()
        row_symbol = str(row.get("symbol") or "").strip().upper()
        row_action = str(row.get("action") or "").strip()
        if area_text and row_area != area_text:
            continue
        if symbol_text and row_symbol != symbol_text:
            continue
        if action_text and row_action == action_text:
            return dict(row)
        if not fallback:
            fallback = dict(row)
    return fallback


def _research_embedding_quality(payload: dict[str, Any]) -> dict[str, Any]:
    action_ladder = dict(payload.get("research_action_ladder") or {})
    batch_summary = dict(payload.get("batch_summary") or {})
    focus_primary = [str(x).strip() for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]
    focus_regime = [
        str(x).strip() for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()
    ]
    research_queue = [str(x).strip() for x in action_ladder.get("research_queue_batches", []) if str(x).strip()]
    avoid_batches = [str(x).strip() for x in action_ladder.get("avoid_batches", []) if str(x).strip()]
    ranked_batches = [dict(row) for row in batch_summary.get("ranked_batches", []) if isinstance(row, dict)]
    deprioritized_batches = [
        str(row.get("batch") or "").strip()
        for row in ranked_batches
        if str(row.get("status_label") or "").strip() == "deprioritize" and str(row.get("batch") or "").strip()
    ]
    zero_trade_deprioritized_batches = [
        str(row.get("batch") or "").strip()
        for row in ranked_batches
        if str(row.get("status_label") or "").strip() == "deprioritize"
        and int(row.get("research_trades", 0) or 0) <= 0
        and int(row.get("accepted_count", 0) or 0) <= 0
        and str(row.get("batch") or "").strip()
    ]
    active_batches: list[str] = []
    for value in [*focus_primary, *focus_regime, *research_queue]:
        if value and value not in active_batches:
            active_batches.append(value)
    avoid_only_batches: list[str] = []
    for value in [*avoid_batches, *deprioritized_batches]:
        if value and value not in avoid_only_batches:
            avoid_only_batches.append(value)

    status = "no_signal"
    brief = "no_signal:-"
    blocker_detail = "latest hot_universe_research has no focus, queue, or avoid classification."
    done_when = "latest hot_universe_research promotes at least one focus_primary or research_queue batch"
    if focus_primary or focus_regime:
        status = "focus_ready"
        brief = f"focus_ready:{_list_text([*focus_primary, *focus_regime], limit=6)}"
        blocker_detail = (
            "latest hot_universe_research keeps focus batches active: "
            + _list_text([*focus_primary, *focus_regime], limit=6)
        )
        done_when = "keep at least one focus batch active in hot_universe_research"
    elif research_queue:
        status = "queue_ready"
        brief = f"queue_ready:{_list_text(research_queue, limit=6)}"
        blocker_detail = "latest hot_universe_research keeps queue batches active: " + _list_text(
            research_queue,
            limit=6,
        )
        done_when = "keep at least one research_queue batch active in hot_universe_research"
    elif avoid_only_batches:
        status = "avoid_only"
        brief = f"avoid_only:{_list_text(avoid_only_batches, limit=6)}"
        blocker_detail = (
            "latest hot_universe_research is fresh and ok, but all tracked batches are avoid/inactive: "
            + _list_text(avoid_only_batches, limit=6)
        )
        if zero_trade_deprioritized_batches:
            blocker_detail += " | zero_trades=" + _list_text(zero_trade_deprioritized_batches, limit=6)

    return {
        "status": status,
        "brief": brief,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "active_batches": active_batches,
        "avoid_batches": avoid_only_batches,
        "zero_trade_deprioritized_batches": zero_trade_deprioritized_batches,
    }


def _crypto_route_embedding_alignment(
    *,
    secondary_focus_area: str,
    secondary_focus_symbol: str,
    secondary_focus_action: str,
    quality_brief: str,
    active_batches: list[str],
) -> dict[str, Any]:
    area_text = str(secondary_focus_area or "").strip()
    symbol_text = str(secondary_focus_symbol or "").strip().upper()
    action_text = str(secondary_focus_action or "").strip()
    quality_brief_text = str(quality_brief or "").strip() or "-"
    active_crypto_batches = [
        str(batch).strip()
        for batch in active_batches
        if str(batch).strip().startswith("crypto_")
    ]
    if area_text != "crypto_route" or not symbol_text:
        return {
            "status": "not_applicable",
            "brief": "not_applicable:-",
            "blocker_detail": "secondary focus is not currently driven by crypto_route.",
            "done_when": "secondary focus returns to crypto_route if crypto alignment becomes relevant again",
        }
    if active_crypto_batches:
        return {
            "status": "aligned",
            "brief": f"aligned:{symbol_text}:{_list_text(active_crypto_batches, limit=6)}",
            "blocker_detail": (
                f"dedicated crypto_route and hot_universe_research both remain usable for {symbol_text}: "
                f"{_list_text(active_crypto_batches, limit=6)}"
            ),
            "done_when": "keep crypto_route and hot_universe_research aligned",
        }
    return {
        "status": "route_ahead_of_embedding",
        "brief": f"route_ahead_of_embedding:{symbol_text}:{quality_brief_text}",
        "blocker_detail": (
            f"dedicated crypto_route still points to {symbol_text}:{action_text or '-'}, "
            f"but latest hot_universe_research has no active crypto batches ({quality_brief_text})."
        ),
        "done_when": (
            "hot_universe_research promotes at least one crypto batch or crypto_route focus degrades to match embedding"
        ),
    }


def _crypto_route_alignment_focus(
    payload: dict[str, Any],
    *,
    focus_slots: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    head_symbol = str(payload.get("crypto_route_review_priority_head_symbol") or "").strip().upper()
    head_action = str(payload.get("crypto_route_review_priority_head_action") or "").strip()
    if head_symbol:
        matched_row = _find_focus_slot_row(
            focus_slots or list(payload.get("operator_focus_slots") or []),
            area="crypto_route",
            symbol=head_symbol,
            action=head_action,
        )
        matched_slot = str(matched_row.get("slot") or "").strip()
        matched_area = str(matched_row.get("area") or "").strip()
        matched_action = str(matched_row.get("action") or "").strip()
        return {
            "slot": matched_slot or "queue_head",
            "area": matched_area or "crypto_route",
            "symbol": head_symbol,
            "action": head_action or matched_action,
        }
    for prefix in ("next_focus", "followup_focus", "secondary_focus"):
        area = str(payload.get(f"{prefix}_area") or "").strip()
        if area != "crypto_route":
            continue
        symbol = str(payload.get(f"{prefix}_symbol") or "").strip().upper()
        action = str(payload.get(f"{prefix}_action") or "").strip()
        if not symbol:
            continue
        return {
            "slot": prefix.removesuffix("_focus"),
            "area": area,
            "symbol": symbol,
            "action": action,
        }
    return {"slot": "", "area": "", "symbol": "", "action": ""}


def _brooks_structure_review_lane(
    *,
    route_report_payload: dict[str, Any] | None,
    execution_plan_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    route_payload = route_report_payload or {}
    execution_payload = execution_plan_payload or {}
    route_candidates = [
        dict(row)
        for row in (route_payload.get("current_candidates") or [])
        if isinstance(row, dict)
    ]
    route_head = dict(route_candidates[0]) if route_candidates else {}
    route_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in route_candidates:
        route_lookup[
            (
                str(row.get("symbol") or "").strip().upper(),
                str(row.get("strategy_id") or "").strip(),
            )
        ] = row
    execution_items = [
        dict(row)
        for row in (execution_payload.get("plan_items") or [])
        if isinstance(row, dict)
    ]
    queue: list[dict[str, Any]] = []
    status = "inactive"
    brief = "inactive:-"
    blocker_detail = "No Brooks structure review item is currently active."
    done_when = "a fresh Brooks route candidate appears with an execution plan head"

    def _priority_for_row(plan_status: str, route_selection_score: float, signal_score: int, signal_age_bars: int) -> tuple[int, str]:
        plan_status_text = str(plan_status or "").strip()
        if plan_status_text == "manual_structure_review_now":
            base = 60
        elif plan_status_text == "blocked_shortline_gate":
            base = 25
        elif plan_status_text == "route_candidate_only":
            base = 15
        else:
            base = 10
        route_component = min(25, max(0, int(round(float(route_selection_score or 0.0) / 4.0))))
        signal_component = min(10, max(0, int(round(int(signal_score or 0) / 10.0))))
        age_component = max(0, 10 - min(max(0, int(signal_age_bars or 0)), 10))
        priority_score = min(100, base + route_component + signal_component + age_component)
        if plan_status_text == "manual_structure_review_now":
            priority_tier = "review_queue_now"
        elif plan_status_text == "blocked_shortline_gate":
            priority_tier = "blocked_review"
        elif plan_status_text == "route_candidate_only":
            priority_tier = "route_candidate_only"
        else:
            priority_tier = "informational_review"
        return priority_score, priority_tier

    if execution_items:
        for rank, plan_item in enumerate(execution_items, start=1):
            plan_status = str(plan_item.get("plan_status") or "").strip()
            if plan_status == "manual_structure_review_now":
                tier = "review_queue_now"
            elif plan_status == "blocked_shortline_gate":
                tier = "blocked_queue"
            else:
                tier = "informational_queue"
            route_row = route_lookup.get(
                (
                    str(plan_item.get("symbol") or "").strip().upper(),
                    str(plan_item.get("strategy_id") or "").strip(),
                ),
                {},
            )
            route_selection_score = float(plan_item.get("route_selection_score") or 0.0)
            signal_score = int(plan_item.get("signal_score") or 0)
            signal_age_bars = int(plan_item.get("signal_age_bars") or 0)
            priority_score, priority_tier = _priority_for_row(
                plan_status,
                route_selection_score,
                signal_score,
                signal_age_bars,
            )
            queue.append(
                {
                    "rank": rank,
                    "symbol": str(plan_item.get("symbol") or "").strip().upper(),
                    "strategy_id": str(plan_item.get("strategy_id") or "").strip(),
                    "tier": tier,
                    "plan_status": plan_status,
                    "execution_action": str(plan_item.get("execution_action") or "").strip(),
                    "direction": str(plan_item.get("direction") or route_row.get("direction") or "").strip(),
                    "route_selection_score": route_selection_score,
                    "signal_score": signal_score,
                    "signal_age_bars": signal_age_bars,
                    "priority_score": priority_score,
                    "priority_tier": priority_tier,
                    "blocker_detail": str(plan_item.get("plan_blocker_detail") or "").strip(),
                    "done_when": str(plan_item.get("plan_done_when") or "").strip()
                    or "Brooks structure item is either executed manually, promoted into an automated bridge, or invalidated.",
                }
            )
        head = dict(queue[0])
        if str(head.get("tier") or "").strip() == "review_queue_now":
            status = "ready"
        elif str(head.get("tier") or "").strip() == "blocked_queue":
            status = "blocked"
        else:
            status = "informational"
        brief = (
            f"{status}:{head.get('symbol') or '-'}:{head.get('strategy_id') or '-'}:{head.get('plan_status') or '-'}"
        )
        blocker_detail = str(head.get("blocker_detail") or "").strip() or blocker_detail
        done_when = str(head.get("done_when") or "").strip() or done_when
    elif route_head:
        route_selection_score = float(route_head.get("route_selection_score") or 0.0)
        signal_score = int(route_head.get("signal_score") or 0)
        signal_age_bars = int(route_head.get("signal_age_bars") or 0)
        priority_score, priority_tier = _priority_for_row(
            "route_candidate_only",
            route_selection_score,
            signal_score,
            signal_age_bars,
        )
        queue.append(
            {
                "rank": 1,
                "symbol": str(route_head.get("symbol") or "").strip().upper(),
                "strategy_id": str(route_head.get("strategy_id") or "").strip(),
                "tier": "route_candidate_only",
                "plan_status": "route_candidate_only",
                "execution_action": "review_route_only",
                "direction": str(route_head.get("direction") or "").strip(),
                "route_selection_score": route_selection_score,
                "signal_score": signal_score,
                "signal_age_bars": signal_age_bars,
                "priority_score": priority_score,
                "priority_tier": priority_tier,
                "blocker_detail": str(route_head.get("route_bridge_blocker_detail") or "").strip(),
                "done_when": "Brooks route candidate receives a concrete execution plan head.",
            }
        )
        status = "route_candidate_only"
        brief = (
            f"route_candidate_only:{queue[0]['symbol'] or '-'}:{queue[0]['strategy_id'] or '-'}"
        )
        blocker_detail = queue[0]["blocker_detail"] or blocker_detail
        done_when = queue[0]["done_when"]

    queue_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('symbol') or '-')}:{str(row.get('tier') or '-')}:{str(row.get('plan_status') or '-')}"
            for row in queue
        ]
    ) or "-"
    head = dict(queue[0]) if queue else {}
    return {
        "status": status,
        "brief": brief,
        "queue_status": "ready" if queue else "inactive",
        "queue_count": len(queue),
        "queue": queue,
        "queue_brief": queue_brief,
        "head": head,
        "priority_status": "ready" if queue else "inactive",
        "priority_brief": (
            f"ready:{str(head.get('symbol') or '-')}:{int(head.get('priority_score') or 0)}:{str(head.get('priority_tier') or '-')}"
            if queue
            else "inactive:-"
        ),
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def _brooks_structure_operator_lane(
    *,
    queue_payload: dict[str, Any] | None,
    fallback_lane: dict[str, Any] | None,
) -> dict[str, Any]:
    source_payload = dict(queue_payload or {})
    queue_rows = [
        dict(row)
        for row in list(source_payload.get("queue") or [])
        if isinstance(row, dict)
    ]
    head = dict(source_payload.get("head") or {}) if isinstance(source_payload.get("head"), dict) else {}
    if not head and queue_rows:
        head = dict(queue_rows[0])
    fallback = dict(fallback_lane or {})
    if not head and isinstance(fallback.get("head"), dict):
        head = dict(fallback.get("head") or {})
    review_status = str(source_payload.get("review_status") or fallback.get("status") or "").strip()
    review_brief = str(source_payload.get("review_brief") or fallback.get("brief") or "").strip()
    blocker_detail = str(source_payload.get("blocker_detail") or fallback.get("blocker_detail") or "").strip()
    done_when = str(source_payload.get("done_when") or fallback.get("done_when") or "").strip()
    if not head:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "head": {},
            "backlog_count": 0,
            "backlog_brief": "-",
            "blocker_detail": blocker_detail or "No Brooks structure operator lane is active.",
            "done_when": done_when or "a fresh Brooks structure review queue item appears",
        }

    backlog = queue_rows[1:] if queue_rows else []
    backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('symbol') or '-')}:{str(row.get('priority_tier') or '-')}:{int(row.get('priority_score') or 0)}"
            for row in backlog[:5]
        ]
    ) or "-"
    head_symbol = str(head.get("symbol") or "").strip().upper() or "-"
    head_action = str(head.get("execution_action") or "").strip() or "-"
    head_priority_score = int(head.get("priority_score") or 0)
    lane_status = review_status or "ready"
    lane_brief = (
        f"{lane_status}:{head_symbol}:{head_action}:{head_priority_score}"
        if head_symbol != "-"
        else (review_brief or "inactive:-")
    )
    return {
        "status": lane_status,
        "brief": lane_brief,
        "head": head,
        "backlog_count": len(backlog),
        "backlog_brief": backlog_brief,
        "blocker_detail": blocker_detail or str(head.get("blocker_detail") or "").strip() or "-",
        "done_when": done_when or str(head.get("done_when") or "").strip() or "-",
    }


def _cross_market_review_head_lane(
    *,
    source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(source_payload or {})
    explicit_lane = dict(payload.get("review_head_lane") or {}) if isinstance(payload.get("review_head_lane"), dict) else {}
    if explicit_lane:
        head = dict(explicit_lane.get("head") or {}) if isinstance(explicit_lane.get("head"), dict) else {}
        return {
            "status": str(explicit_lane.get("status") or "inactive"),
            "brief": str(explicit_lane.get("brief") or "inactive:-"),
            "head": head,
            "head_rank": int(explicit_lane.get("head_rank") or head.get("rank") or 0),
            "target": str(explicit_lane.get("target") or head.get("target") or head.get("symbol") or "-"),
            "reason": str(explicit_lane.get("reason") or head.get("reason") or "-"),
            "backlog_count": int(explicit_lane.get("backlog_count") or 0),
            "backlog_brief": str(explicit_lane.get("backlog_brief") or "-"),
            "blocker_detail": str(explicit_lane.get("blocker_detail") or ""),
            "done_when": str(explicit_lane.get("done_when") or ""),
        }
    backlog = [
        dict(row)
        for row in list(payload.get("review_backlog") or [])
        if isinstance(row, dict)
    ]
    head = dict(payload.get("review_head") or {}) if isinstance(payload.get("review_head"), dict) else {}
    if not head and backlog:
        head = dict(backlog[0])
    if not head:
        head_area = str(payload.get("review_head_area") or "").strip()
        head_symbol = str(payload.get("review_head_symbol") or "").strip().upper()
        head_action = str(payload.get("review_head_action") or "").strip()
        head_priority_score = payload.get("review_head_priority_score")
        head_priority_tier = str(payload.get("review_head_priority_tier") or "").strip()
        if head_area or head_symbol or head_action:
            head = {
                "area": head_area,
                "symbol": head_symbol,
                "action": head_action,
                "priority_score": head_priority_score,
                "priority_tier": head_priority_tier,
            }
    if not head:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "head": {},
            "backlog_count": 0,
            "backlog_brief": "-",
            "blocker_detail": "No cross-market review head is currently active.",
            "done_when": "cross-market operator state produces at least one review backlog item",
        }

    head_area = str(head.get("area") or "").strip() or "-"
    head_symbol = str(head.get("symbol") or "").strip().upper() or "-"
    head_action = str(head.get("action") or "").strip() or "-"
    head_target = str(head.get("target") or head_symbol).strip() or head_symbol
    head_reason = str(head.get("reason") or "").strip() or "cross_market_review_head"
    head_rank = int(head.get("rank") or 1)
    head_priority_score = int(head.get("priority_score") or 0)
    head_status = str(head.get("status") or payload.get("review_backlog_status") or "").strip() or "ready"
    head_blocker_detail = str(head.get("blocker_detail") or "").strip()
    head_done_when = str(head.get("done_when") or "").strip()

    if not backlog and head_area != "-" and head_symbol != "-":
        backlog = [dict(head)]
    backlog_tail = backlog[1:] if backlog else []
    backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('area') or '-')}"
            f":{str(row.get('symbol') or '-')}"
            f":{str(row.get('priority_tier') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in backlog_tail[:8]
        ]
    ) or "-"
    return {
        "status": head_status,
        "brief": f"{head_status}:{head_area}:{head_symbol}:{head_action}:{head_priority_score}",
        "head": head,
        "head_rank": head_rank,
        "target": head_target,
        "reason": head_reason,
        "backlog_count": len(backlog_tail),
        "backlog_brief": backlog_brief,
        "blocker_detail": head_blocker_detail or "Cross-market review head is active and awaiting operator review.",
        "done_when": head_done_when or "cross-market review head is resolved, promoted, or invalidated",
    }


def _cross_market_operator_head_lane(
    *,
    source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(source_payload or {})
    explicit_lane = dict(payload.get("operator_head_lane") or {}) if isinstance(payload.get("operator_head_lane"), dict) else {}
    if explicit_lane:
        head = dict(explicit_lane.get("head") or {}) if isinstance(explicit_lane.get("head"), dict) else {}
        return {
            "status": str(explicit_lane.get("status") or "inactive"),
            "brief": str(explicit_lane.get("brief") or "inactive:-"),
            "head": head,
            "head_rank": int(explicit_lane.get("head_rank") or head.get("rank") or 0),
            "backlog_count": int(explicit_lane.get("backlog_count") or 0),
            "backlog_brief": str(explicit_lane.get("backlog_brief") or "-"),
            "blocker_detail": str(explicit_lane.get("blocker_detail") or ""),
            "done_when": str(explicit_lane.get("done_when") or ""),
            "state": str(explicit_lane.get("state") or head.get("state") or "inactive"),
            "priority_tier": str(explicit_lane.get("priority_tier") or head.get("priority_tier") or ""),
        }
    backlog = [
        dict(row)
        for row in list(payload.get("operator_backlog") or [])
        if isinstance(row, dict)
    ]
    head = dict(payload.get("operator_head") or {}) if isinstance(payload.get("operator_head"), dict) else {}
    if not head and backlog:
        head = dict(backlog[0])
    if not head:
        head_area = str(payload.get("operator_head_area") or "").strip()
        head_symbol = str(payload.get("operator_head_symbol") or "").strip().upper()
        head_action = str(payload.get("operator_head_action") or "").strip()
        head_state = str(payload.get("operator_head_state") or "").strip()
        head_priority_score = payload.get("operator_head_priority_score")
        head_priority_tier = str(payload.get("operator_head_priority_tier") or "").strip()
        if head_area or head_symbol or head_action:
            head = {
                "area": head_area,
                "symbol": head_symbol,
                "action": head_action,
                "state": head_state,
                "priority_score": head_priority_score,
                "priority_tier": head_priority_tier,
            }
    if not head:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "head": {},
            "head_rank": 0,
            "backlog_count": 0,
            "backlog_brief": "-",
            "blocker_detail": "No cross-market operator head is currently active.",
            "done_when": "cross-market operator state produces at least one operator backlog item",
        }

    head_area = str(head.get("area") or "").strip() or "-"
    head_symbol = str(head.get("symbol") or "").strip().upper() or "-"
    head_action = str(head.get("action") or "").strip() or "-"
    head_state = str(head.get("state") or payload.get("operator_backlog_status") or "").strip() or "review"
    head_rank = int(head.get("rank") or 1)
    head_priority_score = int(head.get("priority_score") or 0)
    head_priority_tier = str(head.get("priority_tier") or "").strip() or "-"
    head_blocker_detail = str(head.get("blocker_detail") or "").strip()
    head_done_when = str(head.get("done_when") or "").strip()

    if not backlog and head_area != "-" and head_symbol != "-":
        backlog = [dict(head)]
    backlog_tail = backlog[1:] if backlog else []
    computed_backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('state') or '-')}"
            f":{str(row.get('area') or '-')}"
            f":{str(row.get('symbol') or '-')}"
            f":{str(row.get('action') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in backlog_tail[:8]
        ]
    ) or "-"
    fallback_backlog_count = max(0, int(payload.get("operator_backlog_count") or 0) - 1)
    backlog_count = len(backlog_tail) if backlog_tail else fallback_backlog_count
    backlog_brief = computed_backlog_brief
    if backlog_brief == "-" and backlog_count > 0:
        fallback_brief = str(payload.get("operator_backlog_brief") or "").strip()
        if fallback_brief:
            fallback_parts = [part.strip() for part in fallback_brief.split("|")]
            backlog_brief = " | ".join(fallback_parts[1:]).strip() or fallback_brief
    return {
        "status": head_state,
        "brief": f"{head_state}:{head_area}:{head_symbol}:{head_action}:{head_priority_score}",
        "head": head,
        "head_rank": head_rank,
        "backlog_count": backlog_count,
        "backlog_brief": backlog_brief,
        "blocker_detail": head_blocker_detail or "Cross-market operator head is active and awaiting operator follow-through.",
        "done_when": head_done_when or "cross-market operator head is resolved, promoted, or invalidated",
        "state": head_state,
        "priority_tier": head_priority_tier,
    }


def _cross_market_operator_repair_head_lane(
    *,
    source_payload: dict[str, Any] | None,
    live_gate_blocker_source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(source_payload or {})
    explicit_lane = (
        dict(payload.get("operator_repair_head_lane") or {})
        if isinstance(payload.get("operator_repair_head_lane"), dict)
        else {}
    )
    if explicit_lane:
        head = dict(explicit_lane.get("head") or {}) if isinstance(explicit_lane.get("head"), dict) else {}
        return {
            "status": str(explicit_lane.get("status") or "inactive"),
            "brief": str(explicit_lane.get("brief") or "inactive:-"),
            "head": head,
            "head_rank": int(explicit_lane.get("head_rank") or head.get("rank") or 0),
            "backlog_count": int(explicit_lane.get("backlog_count") or 0),
            "backlog_brief": str(explicit_lane.get("backlog_brief") or "-"),
            "priority_total": int(explicit_lane.get("priority_total") or 0),
            "blocker_detail": str(explicit_lane.get("blocker_detail") or ""),
            "done_when": str(explicit_lane.get("done_when") or ""),
            "command": str(explicit_lane.get("command") or ""),
            "clear_when": str(explicit_lane.get("clear_when") or ""),
        }
    repair_queue = (
        dict((live_gate_blocker_source_payload or {}).get("remote_live_takeover_repair_queue") or {})
        if isinstance((live_gate_blocker_source_payload or {}).get("remote_live_takeover_repair_queue"), dict)
        else {}
    )
    if not repair_queue:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "head": {},
            "head_rank": 0,
            "backlog_count": 0,
            "backlog_brief": "-",
            "blocker_detail": "",
            "done_when": "",
            "command": "",
            "clear_when": "",
        }
    head_area = str(repair_queue.get("head_area") or "").strip()
    head_code = str(repair_queue.get("head_code") or "").strip()
    head_action = str(repair_queue.get("head_action") or "").strip()
    head_priority_score = int(repair_queue.get("head_priority_score") or 0)
    head_priority_tier = str(repair_queue.get("head_priority_tier") or "").strip()
    head = {
        "area": head_area,
        "symbol": head_code.upper(),
        "target": head_code,
        "action": head_action,
        "priority_score": head_priority_score,
        "priority_tier": head_priority_tier,
    }
    queue_brief = str(repair_queue.get("queue_brief") or "").strip()
    backlog_parts = [part.strip() for part in queue_brief.split("|")] if queue_brief else []
    queue_items = [
        dict(row)
        for row in list(repair_queue.get("items") or [])
        if isinstance(row, dict)
    ]
    priority_total = sum(int(row.get("priority_score") or 0) for row in queue_items)
    if priority_total <= 0:
        priority_total = head_priority_score
    return {
        "status": str(repair_queue.get("status") or "ready"),
        "brief": str(repair_queue.get("brief") or f"ready:{head_area or '-'}:{head_code or '-'}:{head_priority_score}"),
        "head": head,
        "head_rank": 1,
        "backlog_count": int(repair_queue.get("count") or 0),
        "backlog_brief": " | ".join(backlog_parts).strip() if backlog_parts else "-",
        "priority_total": priority_total,
        "blocker_detail": str(repair_queue.get("head_clear_when") or "").strip(),
        "done_when": str(repair_queue.get("done_when") or "").strip(),
        "command": str(repair_queue.get("head_command") or "").strip(),
        "clear_when": str(repair_queue.get("head_clear_when") or "").strip(),
    }


def _crypto_route_alignment_recovery_recipe(
    *,
    alignment_status: str,
    source_artifact: str,
    avoid_batches: list[str],
    zero_trade_deprioritized_batches: list[str],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    recipe = {
        "script": "",
        "command_hint": "",
        "expected_status": "",
        "note": "",
        "followup_script": "",
        "followup_command_hint": "",
        "verify_hint": "",
        "window_days": None,
        "target_batches": [],
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return recipe
    artifact_text = str(source_artifact or "").strip()
    artifact_path = Path(artifact_text).expanduser() if artifact_text else Path()
    review_dir = artifact_path.parent if artifact_path.suffix else DEFAULT_REVIEW_DIR
    output_root = review_dir.parent if review_dir.name == "review" else DEFAULT_OUTPUT_ROOT
    context_path = review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"
    research_script_path = Path(__file__).resolve().parent / "run_hot_universe_research.py"
    followup_script_path = Path(__file__).resolve().parent / "refresh_commodity_paper_execution_state.py"
    now_text = reference_now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path.exists() else {}
    except Exception:
        payload = {}
    end_text = str(payload.get("end") or "").strip() or reference_now.date().isoformat()
    start_text = str(payload.get("start") or "").strip()
    try:
        end_date = dt.date.fromisoformat(end_text)
    except Exception:
        end_date = reference_now.date()
    try:
        start_date = dt.date.fromisoformat(start_text) if start_text else end_date - dt.timedelta(days=3)
    except Exception:
        start_date = end_date - dt.timedelta(days=3)
    current_window_days = max(1, (end_date - start_date).days + 1)
    target_window_days = max(21, current_window_days)
    recovery_start_date = end_date - dt.timedelta(days=target_window_days - 1)
    universe_file = _resolve_universe_file_path(
        review_dir=review_dir,
        payload=payload if isinstance(payload, dict) else {},
        reference_now=reference_now,
    )
    target_batches = [
        str(batch).strip()
        for batch in zero_trade_deprioritized_batches
        if str(batch).strip().startswith("crypto_")
    ]
    if not target_batches:
        target_batches = [str(batch).strip() for batch in avoid_batches if str(batch).strip().startswith("crypto_")]
    if not target_batches:
        target_batches = list(DEFAULT_CRYPTO_ROUTE_REFRESH_BATCHES)
    command = [
        "python3",
        str(research_script_path),
        "--output-root",
        str(output_root),
        "--review-dir",
        str(review_dir),
        "--start",
        recovery_start_date.isoformat(),
        "--end",
        end_date.isoformat(),
        "--now",
        now_text,
        "--hours-budget",
        "0.08",
        "--max-trials-per-mode",
        "5",
        "--review-days",
        "7",
        "--run-strategy-lab",
        "--strategy-lab-candidate-count",
        "6",
        "--batch-timeout-seconds",
        "30",
    ]
    if universe_file:
        command.extend(["--universe-file", universe_file])
    for batch_name in target_batches:
        command.extend(["--batch", batch_name])
    followup_command = [
        "python3",
        str(followup_script_path),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--context-path",
        str(context_path),
    ]
    recipe["script"] = str(research_script_path)
    recipe["command_hint"] = shlex.join(command)
    recipe["expected_status"] = "ok"
    recipe["note"] = (
        f"extend crypto embedding window to {target_window_days}d and enable strategy_lab "
        f"because current crypto batches are avoid_only with zero trades"
    )
    recipe["followup_script"] = str(followup_script_path)
    recipe["followup_command_hint"] = shlex.join(followup_command)
    recipe["verify_hint"] = (
        "confirm operator_research_embedding_quality_status leaves avoid_only or "
        "operator_crypto_route_alignment_status leaves route_ahead_of_embedding"
    )
    recipe["window_days"] = target_window_days
    recipe["target_batches"] = target_batches
    return recipe


def _crypto_route_alignment_recovery_outcome(
    *,
    alignment_status: str,
    quality_status: str,
    source_status: str,
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    outcome = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "crypto route alignment recovery outcome is not needed when alignment is not blocked.",
        "done_when": "crypto route alignment becomes blocked again before reassessing recovery outcome",
        "failed_batch_count": 0,
        "timed_out_batch_count": 0,
        "zero_trade_batches": [],
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return outcome

    failed_batch_count = int(source_payload.get("failed_batch_count", 0) or 0)
    timed_out_batch_count = int(source_payload.get("timed_out_batch_count", 0) or 0)
    ranked_batches = [
        dict(row)
        for row in dict(source_payload.get("batch_summary") or {}).get("ranked_batches", [])
        if isinstance(row, dict)
    ]
    zero_trade_batches = [
        str(row.get("batch") or "").strip()
        for row in ranked_batches
        if str(row.get("batch") or "").strip()
        and int(row.get("research_trades", 0) or 0) <= 0
        and int(row.get("accepted_count", 0) or 0) <= 0
    ]
    if not zero_trade_batches:
        zero_trade_batches = [
            str(batch).strip()
            for batch in dict(source_payload.get("research_action_ladder") or {}).get("avoid_batches", [])
            if str(batch).strip()
        ]
    source_status_text = str(source_status or "").strip()
    quality_status_text = str(quality_status or "").strip()

    if source_status_text == "partial_failure" or failed_batch_count > 0 or timed_out_batch_count > 0:
        outcome["status"] = "recovery_partial_failure"
        outcome["brief"] = (
            f"recovery_partial_failure:failed={failed_batch_count}:timed_out={timed_out_batch_count}"
        )
        outcome["blocker_detail"] = (
            "latest crypto alignment recovery artifact still has batch failures: "
            f"failed_batch_count={failed_batch_count}, timed_out_batch_count={timed_out_batch_count}"
        )
        outcome["done_when"] = (
            "latest hot_universe_research finishes with failed_batch_count=0 and timed_out_batch_count=0"
        )
    elif source_status_text == "ok" and quality_status_text == "avoid_only":
        outcome["status"] = "recovery_completed_no_edge"
        outcome["brief"] = (
            "recovery_completed_no_edge:"
            + _list_text(zero_trade_batches or ["no_edge_batches"], limit=6)
        )
        outcome["blocker_detail"] = (
            "latest crypto alignment recovery artifact finished cleanly, but all targeted crypto batches still "
            "show zero research trades and zero accepted strategy_lab candidates: "
            + _list_text(zero_trade_batches, limit=6)
        )
        outcome["done_when"] = (
            "hot_universe_research promotes at least one crypto batch or dedicated crypto_route degrades to match the no-edge embedding"
        )
    elif source_status_text == "ok":
        outcome["status"] = "recovery_succeeded"
        outcome["brief"] = f"recovery_succeeded:{quality_status_text or '-'}"
        outcome["blocker_detail"] = "latest crypto alignment recovery artifact finished cleanly."
        outcome["done_when"] = "keep the recovery artifact fresh while the crypto route remains usable"
    else:
        outcome["status"] = "recovery_not_run"
        outcome["brief"] = f"recovery_not_run:{source_status_text or '-'}"
        outcome["blocker_detail"] = (
            "latest crypto alignment recovery artifact is not yet a fresh non-dry-run hot_universe_research result."
        )
        outcome["done_when"] = "a fresh non-dry-run hot_universe_research artifact becomes available"

    outcome["failed_batch_count"] = failed_batch_count
    outcome["timed_out_batch_count"] = timed_out_batch_count
    outcome["zero_trade_batches"] = zero_trade_batches
    return outcome


def _crypto_route_alignment_cooldown(
    *,
    alignment_status: str,
    recovery_status: str,
    source_status: str,
    source_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    cooldown = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "crypto route alignment cooldown is not needed when recovery has not completed cleanly without edge.",
        "done_when": "crypto route alignment recovery completes without edge before reassessing cooldown",
        "last_research_end_date": "",
        "next_eligible_end_date": "",
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return cooldown
    if str(recovery_status or "").strip() != "recovery_completed_no_edge":
        return cooldown
    if str(source_status or "").strip() != "ok":
        cooldown["status"] = "cooldown_waiting_for_clean_source"
        cooldown["brief"] = f"cooldown_waiting_for_clean_source:{source_status or '-'}"
        cooldown["blocker_detail"] = "latest hot_universe_research source is not a clean ok artifact yet."
        cooldown["done_when"] = "latest hot_universe_research source becomes ok before enforcing cooldown"
        return cooldown

    end_text = str(source_payload.get("end") or "").strip()
    try:
        end_date = dt.date.fromisoformat(end_text) if end_text else reference_now.date()
    except Exception:
        end_date = reference_now.date()
    next_date = end_date + dt.timedelta(days=1)
    cooldown["last_research_end_date"] = end_date.isoformat()
    cooldown["next_eligible_end_date"] = next_date.isoformat()

    if end_date >= reference_now.date():
        cooldown["status"] = "cooldown_active_wait_for_new_market_data"
        cooldown["brief"] = f"cooldown_active_wait_for_new_market_data:>{end_date.isoformat()}"
        cooldown["blocker_detail"] = (
            "latest clean crypto recovery already evaluated data through "
            f"{end_date.isoformat()} and still found no edge; rerunning before a later end date is unlikely to change the outcome"
        )
        cooldown["done_when"] = (
            f"hot_universe_research end date advances beyond {end_date.isoformat()} or crypto_route focus changes"
        )
        return cooldown

    cooldown["status"] = "cooldown_expired_rerun_allowed"
    cooldown["brief"] = f"cooldown_expired_rerun_allowed:{end_date.isoformat()}"
    cooldown["blocker_detail"] = (
        "latest clean crypto recovery found no edge, but its end date is now behind the current reference date."
    )
    cooldown["done_when"] = f"rerun hot_universe_research using data through at least {next_date.isoformat()}"
    return cooldown


def _crypto_route_alignment_recipe_gate(
    *,
    alignment_status: str,
    recipe_script: str,
    cooldown_status: str,
    cooldown_brief: str,
    cooldown_blocker_detail: str,
    cooldown_done_when: str,
    cooldown_next_eligible_end_date: str,
) -> dict[str, str]:
    gate = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "crypto route alignment recovery recipe is not needed when alignment is not blocked.",
        "done_when": "crypto route alignment becomes blocked again before reassessing recovery recipe usage",
        "ready_on_date": "",
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return gate
    if not str(recipe_script or "").strip():
        gate["status"] = "recipe_unavailable"
        gate["brief"] = "recipe_unavailable:-"
        gate["blocker_detail"] = "crypto route alignment recovery recipe is not available."
        gate["done_when"] = "a recovery recipe becomes available"
        return gate

    cooldown_status_text = str(cooldown_status or "").strip()
    next_eligible_text = str(cooldown_next_eligible_end_date or "").strip()
    if cooldown_status_text == "cooldown_active_wait_for_new_market_data":
        gate["status"] = "deferred_by_cooldown"
        gate["brief"] = f"deferred_by_cooldown:{next_eligible_text or '-'}"
        gate["blocker_detail"] = str(cooldown_blocker_detail or "").strip() or str(cooldown_brief or "").strip() or "-"
        gate["done_when"] = str(cooldown_done_when or "").strip() or "wait for cooldown to expire"
        gate["ready_on_date"] = next_eligible_text
        return gate
    if cooldown_status_text == "cooldown_expired_rerun_allowed":
        gate["status"] = "eligible_now"
        gate["brief"] = f"eligible_now:{next_eligible_text or '-'}"
        gate["blocker_detail"] = "recovery recipe is available and cooldown has expired."
        gate["done_when"] = "execute the recovery recipe and refresh the operator handoff"
        gate["ready_on_date"] = next_eligible_text
        return gate

    gate["status"] = "available_now"
    gate["brief"] = "available_now:-"
    gate["blocker_detail"] = "recovery recipe is available for the current crypto alignment blocker."
    gate["done_when"] = "execute the recovery recipe and refresh the operator handoff"
    gate["ready_on_date"] = next_eligible_text
    return gate


def _append_action_queue_item(
    queue: list[dict[str, Any]],
    *,
    area: str,
    target: str,
    symbol: str,
    action: str,
    reason: str,
) -> None:
    area_text = str(area or "").strip() or "-"
    target_text = str(target or "").strip() or "-"
    action_text = str(action or "").strip() or "-"
    symbol_text = str(symbol or "").strip().upper() or "-"
    reason_text = str(reason or "").strip() or "-"
    dedupe_key = (area_text, target_text, action_text)
    if dedupe_key == ("-", "-", "-"):
        return
    for row in queue:
        if (row.get("area"), row.get("target"), row.get("action")) == dedupe_key:
            return
    queue.append(
        {
            "area": area_text,
            "target": target_text,
            "symbol": symbol_text,
            "action": action_text,
            "reason": reason_text,
        }
    )


def _action_queue_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for idx, row in enumerate(items[:limit], start=1):
        if not isinstance(row, dict):
            continue
        parts.append(
            f"{idx}:{row.get('area') or '-'}:{row.get('target') or '-'}:{row.get('action') or '-'}"
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _action_checklist_state(*, area: str, action: str, rank: int) -> str:
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    if area_text in {"ops_live_gate", "risk_guard"} or action_text.startswith("clear_"):
        return "repair"
    if action_text.startswith("watch_"):
        return "watch"
    if action_text in {"review_manual_stop_entry", "review_route_only"}:
        return "review"
    if action_text == "deprioritize_flow":
        return "review"
    if action_text.startswith("wait_"):
        return "waiting"
    if rank == 1:
        return "active"
    if area_text.startswith("commodity_"):
        return "queued"
    return "queued"


def _action_checklist_blocker(
    *,
    area: str,
    action: str,
    symbol: str,
    reason: str,
    row_blocker_detail: str = "",
    stale_signal_dates: dict[str, str],
    stale_signal_age_days: dict[str, int],
    focus_evidence_summary: dict[str, Any],
    focus_area: str,
    focus_symbol: str,
    focus_lifecycle_status: str = "",
    focus_lifecycle_blocker_detail: str = "",
    crypto_focus_execution_blocker_detail: str = "",
    crypto_route_alignment_status: str = "",
    crypto_route_alignment_recovery_status: str = "",
    crypto_route_alignment_recovery_zero_trade_batches: list[str] | None = None,
) -> str:
    symbol_text = str(symbol or "").strip().upper()
    action_text = str(action or "").strip()
    reason_text = str(reason or "").strip()
    row_blocker_text = str(row_blocker_detail or "").strip()
    stale_date = stale_signal_dates.get(symbol_text, "")
    stale_age = stale_signal_age_days.get(symbol_text)

    if row_blocker_text:
        return row_blocker_text
    if action_text == "review_paper_execution_retro":
        evidence_present = (
            area == focus_area
            and symbol_text == focus_symbol
            and bool(focus_evidence_summary.get("paper_execution_evidence_present"))
        )
        if evidence_present:
            if str(focus_lifecycle_status or "").strip() == "open_position_wait_close_evidence":
                return str(focus_lifecycle_blocker_detail or "").strip() or (
                    "paper execution evidence is present, but position is still OPEN; retro should wait for close evidence"
                )
            return "paper execution evidence is present; retro item still pending"
        return "retro item still pending"
    if action_text == "wait_for_paper_execution_close_evidence":
        if area == focus_area and symbol_text == focus_symbol:
            return str(focus_lifecycle_blocker_detail or "").strip() or (
                "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
            )
        return "paper execution close evidence is not written yet"
    if action_text == "review_paper_execution":
        return "review item still pending"
    if action_text == "wait_for_paper_execution_fill_evidence":
        if stale_age not in (None, "") and stale_date:
            return f"paper execution fill evidence not written; stale directional signal {stale_age}d since {stale_date}"
        return "paper execution fill evidence not written"
    if action_text == "restore_commodity_directional_signal":
        if stale_age not in (None, "") and stale_date:
            return f"directional signal is stale {stale_age}d since {stale_date}"
        return "directional signal ticket is missing or not fresh"
    if action_text == "normalize_commodity_execution_price_reference":
        return "execution price is still proxy-reference only"
    if action_text == "deprioritize_flow":
        if str(crypto_focus_execution_blocker_detail or "").strip():
            return str(crypto_focus_execution_blocker_detail or "").strip()
        if reason_text and reason_text != "secondary_focus":
            return reason_text.replace("_", " ")
        return f"{symbol_text or 'SYMBOL'} flow edge remains below review threshold"
    if action_text == "watch_priority_until_long_window_confirms":
        if (
            str(crypto_route_alignment_status or "").strip() == "route_ahead_of_embedding"
            and str(crypto_route_alignment_recovery_status or "").strip() == "recovery_completed_no_edge"
        ):
            return (
                "long-window confirmation still missing; clean crypto recovery still shows no edge in "
                + _list_text(list(crypto_route_alignment_recovery_zero_trade_batches or []), limit=6)
            )
        return "long-window confirmation still missing"
    if action_text == "apply_commodity_execution_bridge":
        return "bridge is ready but paper apply has not been executed"
    if reason_text:
        return reason_text.replace("_", " ")
    return "pending"


def _action_checklist_done_when(
    *,
    action: str,
    symbol: str,
    row_done_when: str = "",
    focus_lifecycle_status: str = "",
    focus_lifecycle_done_when: str = "",
    crypto_focus_execution_done_when: str = "",
    crypto_route_alignment_status: str = "",
    crypto_route_alignment_recovery_status: str = "",
) -> str:
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    action_text = str(action or "").strip()
    row_done_when_text = str(row_done_when or "").strip()
    if row_done_when_text:
        return row_done_when_text
    if action_text == "review_paper_execution_retro":
        if str(focus_lifecycle_status or "").strip() == "open_position_wait_close_evidence":
            return str(focus_lifecycle_done_when or "").strip() or (
                f"{symbol_text} paper_execution_status leaves OPEN and retro can evaluate close outcome"
            )
        return f"{symbol_text} leaves retro_pending_symbols"
    if action_text == "wait_for_paper_execution_close_evidence":
        return str(focus_lifecycle_done_when or "").strip() or (
            f"{symbol_text} paper_execution_status leaves OPEN and close evidence becomes available"
        )
    if action_text == "review_paper_execution":
        return f"{symbol_text} leaves review_pending_symbols"
    if action_text == "wait_for_paper_execution_fill_evidence":
        return f"{symbol_text} gains paper evidence and leaves fill_evidence_pending_symbols"
    if action_text == "restore_commodity_directional_signal":
        return f"{symbol_text} prints a fresh directional signal ticket"
    if action_text == "normalize_commodity_execution_price_reference":
        return f"{symbol_text} receives executable non-proxy price levels"
    if action_text == "deprioritize_flow":
        return str(crypto_focus_execution_done_when or "").strip() or (
            f"{symbol_text} regains a positive ranked flow edge or leaves review"
        )
    if action_text == "apply_commodity_execution_bridge":
        return f"{symbol_text} is written into paper evidence artifacts"
    if action_text == "watch_priority_until_long_window_confirms":
        if (
            str(crypto_route_alignment_status or "").strip() == "route_ahead_of_embedding"
            and str(crypto_route_alignment_recovery_status or "").strip() == "recovery_completed_no_edge"
        ):
            return f"{symbol_text} gains supporting crypto research edge or leaves priority watch"
        return f"{symbol_text} upgrades from priority watch to deploy or leaves priority watch"
    return f"{symbol_text} completes {action_text or 'the current action'}"


def _action_checklist_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slots_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source(payload: dict[str, Any], *, area: str, action: str) -> tuple[str, str]:
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    mapping = [
        ("commodity_execution_close_evidence", "commodity_execution_retro", payload.get("source_commodity_execution_retro_artifact")),
        ("commodity_execution_retro", "commodity_execution_retro", payload.get("source_commodity_execution_retro_artifact")),
        ("commodity_execution_review", "commodity_execution_review", payload.get("source_commodity_execution_review_artifact")),
        ("commodity_fill_evidence", "commodity_execution_review", payload.get("source_commodity_execution_review_artifact")),
        ("commodity_execution_bridge", "commodity_execution_bridge", payload.get("source_commodity_execution_bridge_artifact")),
        ("commodity_execution_gap", "commodity_execution_gap", payload.get("source_commodity_execution_gap_artifact")),
        ("commodity_execution_queue", "commodity_execution_queue", payload.get("source_commodity_execution_queue_artifact")),
        ("commodity_execution_artifact", "commodity_execution_artifact", payload.get("source_commodity_execution_artifact")),
        ("commodity_execution_preview", "commodity_execution_preview", payload.get("source_commodity_execution_preview_artifact")),
        ("commodity_ticket_book", "commodity_ticket_book", payload.get("source_commodity_ticket_book_artifact")),
        ("commodity_route", "commodity_route", payload.get("source_commodity_artifact")),
        ("brooks_structure", "brooks_structure_review_queue", payload.get("source_brooks_structure_review_queue_artifact")),
        ("cross_market_review", "cross_market_operator_state", payload.get("source_cross_market_operator_state_artifact")),
        ("crypto_route", "crypto_route", payload.get("source_crypto_route_artifact") or payload.get("source_crypto_artifact")),
        ("research_queue", "action_research", payload.get("source_action_artifact")),
    ]
    for match_area, source_kind, source_artifact in mapping:
        if area_text == match_area:
            return source_kind, str(source_artifact or "")
    if action_text == "wait_for_paper_execution_close_evidence":
        return "commodity_execution_retro", str(payload.get("source_commodity_execution_retro_artifact") or "")
    if action_text == "wait_for_paper_execution_fill_evidence":
        return "commodity_execution_review", str(payload.get("source_commodity_execution_review_artifact") or "")
    return "-", ""


def _focus_slot_source_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("source_kind") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source_status(payload: dict[str, Any], *, source_kind: str) -> str:
    source_text = str(source_kind or "").strip()
    mapping = {
        "commodity_execution_retro": str(payload.get("source_commodity_execution_retro_status") or ""),
        "commodity_execution_review": str(payload.get("source_commodity_execution_review_status") or ""),
        "commodity_execution_bridge": str(payload.get("source_commodity_execution_bridge_status") or ""),
        "commodity_execution_gap": str(payload.get("source_commodity_execution_gap_status") or ""),
        "commodity_execution_queue": str(payload.get("source_commodity_execution_queue_status") or ""),
        "commodity_execution_artifact": str(payload.get("source_commodity_execution_artifact_status") or ""),
        "commodity_execution_preview": str(payload.get("source_commodity_execution_preview_status") or ""),
        "commodity_ticket_book": str(payload.get("source_commodity_ticket_book_status") or ""),
        "commodity_route": str(payload.get("source_commodity_status") or ""),
        "brooks_structure_review_queue": str(payload.get("source_brooks_structure_review_queue_status") or ""),
        "cross_market_operator_state": str(payload.get("source_cross_market_operator_state_status") or ""),
        "crypto_route": str(payload.get("source_crypto_route_status") or payload.get("source_crypto_status") or ""),
        "action_research": str(payload.get("source_action_status") or ""),
    }
    return mapping.get(source_text, "")


def _focus_slot_source_as_of(source_artifact: str) -> str:
    text = str(source_artifact or "").strip()
    if not text:
        return ""
    stamp_dt = parsed_artifact_stamp(Path(text))
    return fmt_utc(stamp_dt) or ""


def _focus_slot_status_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        source_status = str(row.get("source_status") or "").strip() or "-"
        source_as_of = str(row.get("source_as_of") or "").strip() or "-"
        parts.append(f"{row.get('slot') or '-'}:{source_status}@{source_as_of}")
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source_age_minutes(*, reference_now: dt.datetime, source_as_of: str) -> int | None:
    text = str(source_as_of or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    delta = reference_now - parsed.astimezone(dt.timezone.utc)
    if delta.total_seconds() < 0:
        return 0
    return int(delta.total_seconds() // 60)


def _focus_slot_source_recency(*, source_age_minutes: int | None) -> str:
    if source_age_minutes is None:
        return "unknown"
    if source_age_minutes <= 15:
        return "fresh"
    if source_age_minutes <= 1440:
        return "carry_over"
    return "stale"


def _focus_slot_recency_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        recency = str(row.get("source_recency") or "").strip() or "-"
        age = row.get("source_age_minutes")
        age_text = str(age) if age not in (None, "") else "-"
        parts.append(f"{row.get('slot') or '-'}:{recency}:{age_text}m")
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source_health(
    *,
    source_status: str,
    source_recency: str,
    source_kind: str = "",
    crypto_route_alignment_cooldown_status: str = "",
) -> str:
    status_text = str(source_status or "").strip()
    recency_text = str(source_recency or "").strip()
    if status_text == "ok" and recency_text == "fresh":
        return "ready"
    if (
        status_text == "ok"
        and recency_text == "carry_over"
        and str(source_kind or "").strip() == "crypto_route"
        and str(crypto_route_alignment_cooldown_status or "").strip() == "cooldown_active_wait_for_new_market_data"
    ):
        return "carry_over_ok"
    if status_text == "ok" and recency_text == "carry_over":
        return "carry_over_ok"
    if status_text == "dry_run" and recency_text == "fresh":
        return "dry_run_only"
    if status_text == "dry_run" and recency_text == "carry_over":
        return "refresh_required"
    if status_text == "dry_run":
        return "dry_run_only"
    if status_text:
        return f"status_{status_text}"
    return "unknown"


def _focus_slot_source_refresh_action(
    *,
    source_health: str,
    source_kind: str = "",
    crypto_route_alignment_cooldown_status: str = "",
) -> str:
    health_text = str(source_health or "").strip()
    if health_text == "ready":
        return "read_current_artifact"
    if health_text == "carry_over_ok":
        if (
            str(source_kind or "").strip() == "crypto_route"
            and str(crypto_route_alignment_cooldown_status or "").strip()
            == "cooldown_active_wait_for_new_market_data"
        ):
            return "wait_for_next_eligible_end_date"
        return "consider_refresh_before_promotion"
    if health_text == "dry_run_only":
        return "do_not_promote_without_non_dry_run"
    if health_text == "refresh_required":
        return "refresh_source_before_use"
    return "inspect_source_state"


def _focus_slot_health_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("source_health") or "-"),
                    str(row.get("source_refresh_action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_refresh_backlog(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        refresh_action = str(row.get("source_refresh_action") or "").strip()
        if refresh_action in ("", "read_current_artifact", "wait_for_next_eligible_end_date"):
            continue
        backlog.append(
            {
                "slot": str(row.get("slot") or "").strip() or "-",
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "action": refresh_action,
                "source_kind": str(row.get("source_kind") or "").strip() or "-",
                "source_status": str(row.get("source_status") or "").strip() or "-",
                "source_recency": str(row.get("source_recency") or "").strip() or "-",
                "source_health": str(row.get("source_health") or "").strip() or "-",
                "source_age_minutes": row.get("source_age_minutes"),
                "source_as_of": str(row.get("source_as_of") or "").strip(),
                "source_artifact": str(row.get("source_artifact") or "").strip(),
            }
        )
    return backlog


def _focus_slot_refresh_backlog_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_promotion_gate_status(*, total_count: int, ready_count: int) -> str:
    if total_count <= 0:
        return "unknown"
    if ready_count >= total_count:
        return "promotion_ready"
    return "promotion_guarded_by_source_freshness"


def _focus_slot_promotion_gate_blocker_detail(
    *,
    total_count: int,
    ready_count: int,
    refresh_backlog: list[dict[str, Any]],
) -> str:
    if total_count <= 0:
        return "no focus slots are present"
    if ready_count >= total_count or not refresh_backlog:
        return f"all {total_count} focus slots are covered by current source or cooldown state"
    head = dict(refresh_backlog[0])
    symbol = str(head.get("symbol") or "").strip().upper() or "-"
    slot = str(head.get("slot") or "").strip() or "-"
    action = str(head.get("action") or "").strip() or "-"
    source_kind = str(head.get("source_kind") or "").strip() or "-"
    source_status = str(head.get("source_status") or "").strip() or "-"
    source_recency = str(head.get("source_recency") or "").strip() or "-"
    age = head.get("source_age_minutes")
    age_text = f", age={age}m" if age not in (None, "") else ""
    return (
        f"{symbol} {slot} source requires {action} "
        f"({source_kind}, {source_status}, {source_recency}{age_text})"
    )


def _focus_slot_promotion_gate_done_when(*, total_count: int, ready_count: int) -> str:
    if total_count <= 0:
        return "operator_focus_slots becomes non-empty"
    if ready_count >= total_count:
        return "all focus slots remain covered by current source or cooldown state"
    return "operator_focus_slot_refresh_backlog_count reaches 0"


def _focus_slot_promotion_gate_brief(*, status: str, ready_count: int, total_count: int) -> str:
    status_text = str(status or "").strip() or "-"
    return f"{status_text}:{ready_count}/{total_count}"


def _focus_slot_actionability_backlog(
    items: list[dict[str, Any]],
    *,
    alignment_focus_slot: str,
    alignment_focus_symbol: str,
    alignment_status: str,
    alignment_brief: str,
    alignment_recovery_status: str,
    alignment_recovery_brief: str,
) -> list[dict[str, Any]]:
    status_text = str(alignment_status or "").strip()
    if status_text != "route_ahead_of_embedding":
        return []
    focus_slot_text = str(alignment_focus_slot or "").strip()
    focus_slot_alias = {
        "next": "primary",
        "primary": "primary",
        "followup": "followup",
        "secondary": "secondary",
    }.get(focus_slot_text, focus_slot_text)
    focus_symbol_text = str(alignment_focus_symbol or "").strip().upper()
    backlog: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        area_text = str(row.get("area") or "").strip()
        slot_text = str(row.get("slot") or "").strip()
        symbol_text = str(row.get("symbol") or "").strip().upper()
        if area_text != "crypto_route":
            continue
        if focus_symbol_text:
            if symbol_text != focus_symbol_text:
                continue
        elif focus_slot_alias and slot_text != focus_slot_alias:
            continue
        backlog.append(
            {
                "slot": slot_text or focus_slot_text or "-",
                "symbol": symbol_text or "-",
                "action": str(row.get("action") or "").strip() or "-",
                "state": str(row.get("state") or "").strip() or "-",
                "blocker_detail": str(row.get("blocker_detail") or "").strip()
                or str(alignment_brief or "").strip()
                or "-",
                "done_when": str(row.get("done_when") or "").strip() or "-",
                "alignment_status": status_text,
                "alignment_brief": str(alignment_brief or "").strip() or "-",
                "alignment_recovery_status": str(alignment_recovery_status or "").strip() or "-",
                "alignment_recovery_brief": str(alignment_recovery_brief or "").strip() or "-",
            }
        )
    return backlog


def _focus_slot_actionability_backlog_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("alignment_recovery_status") or row.get("alignment_status") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_actionability_gate_status(*, total_count: int, actionable_count: int) -> str:
    if total_count <= 0:
        return "unknown"
    if actionable_count >= total_count:
        return "actionability_ready"
    return "actionability_guarded_by_content"


def _focus_slot_actionability_gate_blocker_detail(
    *,
    total_count: int,
    actionable_count: int,
    backlog: list[dict[str, Any]],
) -> str:
    if total_count <= 0:
        return "no focus slots are present"
    if actionable_count >= total_count or not backlog:
        return f"all {total_count} focus slots have content-aligned decision state"
    head = dict(backlog[0])
    symbol = str(head.get("symbol") or "").strip().upper() or "-"
    slot = str(head.get("slot") or "").strip() or "-"
    alignment_status = str(head.get("alignment_status") or "").strip() or "-"
    recovery_status = str(head.get("alignment_recovery_status") or "").strip() or "-"
    blocker_detail = str(head.get("blocker_detail") or "").strip() or "-"
    return (
        f"{symbol} {slot} content state remains blocked "
        f"({alignment_status}, {recovery_status}): {blocker_detail}"
    )


def _focus_slot_actionability_gate_done_when(
    *,
    total_count: int,
    actionable_count: int,
    backlog: list[dict[str, Any]],
) -> str:
    if total_count <= 0:
        return "operator_focus_slots becomes non-empty"
    if actionable_count >= total_count or not backlog:
        return "all focus slots continue with content-aligned decision state"
    head = dict(backlog[0])
    return str(head.get("done_when") or "").strip() or "operator_crypto_route_alignment_status leaves route_ahead_of_embedding"


def _focus_slot_actionability_gate_brief(*, status: str, actionable_count: int, total_count: int) -> str:
    status_text = str(status or "").strip() or "-"
    return f"{status_text}:{actionable_count}/{total_count}"


def _crypto_route_focus_review_lane(
    *,
    symbol: str,
    action: str,
    focus_execution_state: str,
    focus_execution_blocker_detail: str,
    focus_execution_done_when: str,
    focus_execution_micro_veto: str,
    alignment_status: str,
    alignment_recovery_status: str,
) -> dict[str, str]:
    symbol_text = str(symbol or "").strip().upper()
    action_text = str(action or "").strip()
    lane = {
        "status": "not_active",
        "brief": "not_active:-",
        "primary_blocker": "not_applicable",
        "micro_blocker": "-",
        "blocker_detail": "crypto review lane is only active when crypto focus is under deprioritize_flow review.",
        "done_when": "crypto focus returns to a deprioritize_flow review state before reassessing",
    }
    if not symbol_text or action_text != "deprioritize_flow":
        return lane

    state_text = str(focus_execution_state or "").strip()
    blocker_detail_text = str(focus_execution_blocker_detail or "").strip()
    done_when_text = str(focus_execution_done_when or "").strip()
    micro_veto_text = str(focus_execution_micro_veto or "").strip()
    alignment_status_text = str(alignment_status or "").strip()
    alignment_recovery_status_text = str(alignment_recovery_status or "").strip()

    status_parts: list[str] = []
    primary_blocker = "review_pending"
    if (
        alignment_status_text == "route_ahead_of_embedding"
        and alignment_recovery_status_text == "recovery_completed_no_edge"
    ):
        primary_blocker = "no_edge"
        status_parts.append("no_edge")
    if state_text == "Bias_Only":
        if primary_blocker == "review_pending":
            primary_blocker = "bias_only"
        status_parts.append("bias_only")
    if micro_veto_text and micro_veto_text != "-":
        status_parts.append("micro_veto")

    lane["status"] = f"review_{'_'.join(status_parts) if status_parts else 'pending'}"
    lane["brief"] = f"{lane['status']}:{symbol_text}"
    lane["primary_blocker"] = primary_blocker
    lane["micro_blocker"] = micro_veto_text or "-"
    lane["blocker_detail"] = blocker_detail_text or f"{symbol_text} remains under flow review."
    lane["done_when"] = done_when_text or f"{symbol_text} regains a positive ranked flow edge or leaves review"
    return lane


def _crypto_route_focus_review_scores(
    *,
    symbol: str,
    action: str,
    focus_execution_state: str,
    focus_execution_micro_classification: str,
    focus_execution_micro_veto: str,
    review_primary_blocker: str,
) -> dict[str, Any]:
    symbol_text = str(symbol or "").strip().upper()
    action_text = str(action or "").strip()
    state_text = str(focus_execution_state or "").strip()
    micro_class_text = str(focus_execution_micro_classification or "").strip()
    micro_veto_text = str(focus_execution_micro_veto or "").strip()
    primary_blocker_text = str(review_primary_blocker or "").strip()
    scores: dict[str, Any] = {
        "status": "not_active",
        "edge_score": 0,
        "structure_score": 0,
        "micro_score": 0,
        "composite_score": 0,
        "brief": "not_active:edge=0|structure=0|micro=0|composite=0",
    }
    if not symbol_text or action_text != "deprioritize_flow":
        return scores

    edge_score = 35
    if primary_blocker_text == "no_edge":
        edge_score = 5
    elif primary_blocker_text == "bias_only":
        edge_score = 15

    structure_score = {
        "Setup_Ready": 100,
        "Bias_Only": 25,
    }.get(state_text, 40 if state_text else 0)

    micro_score = {
        "confirmed": 100,
        "confirm_and_veto_only": 65,
        "watch_only": 35,
    }.get(micro_class_text, 40 if micro_class_text else 0)
    if micro_veto_text == "low_sample_or_gap_risk":
        micro_score = min(micro_score, 20)
    elif micro_veto_text and micro_veto_text != "-":
        micro_score = min(micro_score, 25)

    composite_score = int(round((edge_score + structure_score + micro_score) / 3.0))
    scores.update(
        {
            "status": "scored",
            "edge_score": edge_score,
            "structure_score": structure_score,
            "micro_score": micro_score,
            "composite_score": composite_score,
            "brief": (
                f"scored:{symbol_text}:"
                f"edge={edge_score}|structure={structure_score}|micro={micro_score}|composite={composite_score}"
            ),
        }
    )
    return scores


def _crypto_route_focus_review_priority(scores: dict[str, Any]) -> dict[str, Any]:
    if str(scores.get("status") or "").strip() != "scored":
        return {
            "status": "not_active",
            "score": 0,
            "tier": "-",
            "brief": "not_active:0/100",
        }
    score = int(scores.get("composite_score") or 0)
    if score >= 70:
        tier = "high_priority_review"
    elif score >= 40:
        tier = "medium_priority_review"
    elif score >= 20:
        tier = "low_priority_review"
    else:
        tier = "deprioritized_review"
    return {
        "status": "ready",
        "score": score,
        "tier": tier,
        "brief": f"{tier}:{score}/100",
    }


def _focus_slot_readiness_gate_status(
    *,
    total_count: int,
    promotion_gate_status: str,
    actionability_gate_status: str,
) -> str:
    if total_count <= 0:
        return "unknown"
    if str(promotion_gate_status or "").strip() != "promotion_ready":
        return "readiness_guarded_by_source_freshness"
    if str(actionability_gate_status or "").strip() != "actionability_ready":
        return "readiness_guarded_by_content"
    return "readiness_ready"


def _focus_slot_readiness_gate_blocking_gate(*, status: str) -> str:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return "source_freshness"
    if status_text == "readiness_guarded_by_content":
        return "content_actionability"
    if status_text == "readiness_ready":
        return "none"
    return "unknown"


def _focus_slot_readiness_gate_ready_count(
    *,
    status: str,
    ready_count: int,
    actionable_count: int,
    total_count: int,
) -> int:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return ready_count
    if status_text == "readiness_guarded_by_content":
        return actionable_count
    if status_text == "readiness_ready":
        return total_count
    return 0


def _focus_slot_readiness_gate_blocker_detail(
    *,
    status: str,
    promotion_gate_blocker_detail: str,
    actionability_gate_blocker_detail: str,
    total_count: int,
) -> str:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return str(promotion_gate_blocker_detail or "").strip() or "focus slot source freshness is still blocked"
    if status_text == "readiness_guarded_by_content":
        return str(actionability_gate_blocker_detail or "").strip() or "focus slot content actionability is still blocked"
    if status_text == "readiness_ready":
        return f"all {total_count} focus slots have fresh sources and content-aligned decision state"
    return "focus slot readiness could not be determined"


def _focus_slot_readiness_gate_done_when(
    *,
    status: str,
    promotion_gate_done_when: str,
    actionability_gate_done_when: str,
) -> str:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return str(promotion_gate_done_when or "").strip() or "source freshness blocker clears"
    if status_text == "readiness_guarded_by_content":
        return str(actionability_gate_done_when or "").strip() or "content actionability blocker clears"
    if status_text == "readiness_ready":
        return "all focus slots remain fresh and content-aligned"
    return "focus slot readiness becomes known"


def _focus_slot_readiness_gate_brief(*, status: str, ready_count: int, total_count: int) -> str:
    status_text = str(status or "").strip() or "-"
    return f"{status_text}:{ready_count}/{total_count}"


def _source_refresh_queue(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for index, row in enumerate(items, start=1):
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["rank"] = index
        queue.append(item)
    return queue


def _source_refresh_queue_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("slot") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _source_refresh_checklist_state(*, action: str) -> str:
    action_text = str(action or "").strip()
    if action_text == "consider_refresh_before_promotion":
        return "refresh_recommended"
    if action_text == "refresh_source_before_use":
        return "refresh_required"
    if action_text == "do_not_promote_without_non_dry_run":
        return "blocked_on_non_dry_run_refresh"
    return "inspect_required"


def _source_refresh_checklist_blocker(
    *,
    source_kind: str,
    source_status: str,
    source_recency: str,
    source_age_minutes: Any,
) -> str:
    kind_text = str(source_kind or "").strip() or "source"
    status_text = str(source_status or "").strip() or "unknown"
    recency_text = str(source_recency or "").strip() or "unknown"
    age_suffix = f", age={source_age_minutes}m" if source_age_minutes not in (None, "") else ""
    return f"{kind_text} artifact is {status_text} and {recency_text}{age_suffix}"


def _source_refresh_checklist_done_when(
    *,
    symbol: str,
    source_kind: str,
    source_status: str,
    action: str,
) -> str:
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    kind_text = str(source_kind or "").strip() or "source"
    status_text = str(source_status or "").strip()
    action_text = str(action or "").strip()
    if status_text == "dry_run" or action_text == "refresh_source_before_use":
        return f"{symbol_text} receives a fresh non-dry-run {kind_text} artifact"
    if action_text == "consider_refresh_before_promotion":
        return f"{symbol_text} receives a fresh {kind_text} artifact before promotion"
    return f"{symbol_text} leaves operator_source_refresh_queue"


def _latest_artifact_from_path_hint(path_hint: str, reference_now: dt.datetime) -> Path | None:
    hint_text = str(path_hint or "").strip()
    if not hint_text:
        return None
    hint_path = Path(hint_text).expanduser()
    parent = hint_path.parent if str(hint_path.parent) not in ("", ".") else DEFAULT_REVIEW_DIR
    pattern = hint_path.name or "*"
    candidates = list(parent.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def _artifact_status_from_path(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("status") or "").strip()


def _resolve_universe_file_path(*, review_dir: Path, payload: dict[str, Any], reference_now: dt.datetime) -> str:
    universe_text = str(payload.get("universe_file") or "").strip()
    if universe_text:
        universe_path = Path(universe_text).expanduser()
        if universe_path.exists():
            return str(universe_path)
    latest_universe = _latest_artifact_from_path_hint(
        str(review_dir / "*_hot_research_universe.json"),
        reference_now,
    )
    return str(latest_universe) if latest_universe else ""


def _crypto_route_refresh_batches(universe_file: str) -> list[str]:
    universe_text = str(universe_file or "").strip()
    if universe_text:
        universe_path = Path(universe_text).expanduser()
        if universe_path.exists():
            try:
                payload = json.loads(universe_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            batches = payload.get("batches", {})
            if isinstance(batches, dict):
                selected = [
                    str(name).strip()
                    for name in batches.keys()
                    if str(name).strip().startswith("crypto_")
                ]
                if selected:
                    return selected
    return list(DEFAULT_CRYPTO_ROUTE_REFRESH_BATCHES)


def _recipe_step_checkpoint_state(*, expected_status: str, current_artifact: str, current_status: str, current_recency: str) -> str:
    if not str(current_artifact or "").strip():
        return "missing"
    expected_text = str(expected_status or "").strip()
    current_status_text = str(current_status or "").strip()
    if expected_text and current_status_text and current_status_text != expected_text:
        return f"status_{current_status_text}"
    current_recency_text = str(current_recency or "").strip()
    if current_recency_text == "fresh":
        return "current"
    if current_recency_text == "carry_over":
        return "carry_over"
    if current_recency_text == "stale":
        return "stale"
    return "unknown"


def _recipe_step_with_checkpoint(step: dict[str, Any], *, reference_now: dt.datetime) -> dict[str, Any]:
    item = dict(step)
    current_path = _latest_artifact_from_path_hint(str(item.get("expected_artifact_path_hint") or ""), reference_now)
    current_artifact = str(current_path) if current_path else ""
    current_status = _artifact_status_from_path(current_path)
    current_as_of = _focus_slot_source_as_of(current_artifact)
    current_age_minutes = _focus_slot_source_age_minutes(reference_now=reference_now, source_as_of=current_as_of)
    current_recency = _focus_slot_source_recency(source_age_minutes=current_age_minutes)
    item["current_artifact"] = current_artifact
    item["current_status"] = current_status
    item["current_as_of"] = current_as_of
    item["current_age_minutes"] = current_age_minutes
    item["current_recency"] = current_recency
    item["checkpoint_state"] = _recipe_step_checkpoint_state(
        expected_status=str(item.get("expected_status") or ""),
        current_artifact=current_artifact,
        current_status=current_status,
        current_recency=current_recency,
    )
    return item


def _recipe_step_checkpoint_brief(steps: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for step in steps[:limit]:
        if not isinstance(step, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(step.get("rank") or "-"),
                    str(step.get("checkpoint_state") or "-"),
                    str(step.get("expected_artifact_kind") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(steps) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(steps) - limit}"


def _recipe_pending_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("checkpoint_state") or "").strip() == "current":
            continue
        pending.append(dict(step))
    return pending


def _recipe_deferred_steps(
    steps: list[dict[str, Any]],
    *,
    recipe_gate_status: str,
) -> list[dict[str, Any]]:
    if str(recipe_gate_status or "").strip() != "deferred_by_cooldown":
        return []
    deferred: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("checkpoint_state") or "").strip() == "current":
            continue
        deferred.append(dict(step))
    return deferred


def _recipe_pending_brief(steps: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for step in steps[:limit]:
        if not isinstance(step, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(step.get("rank") or "-"),
                    str(step.get("name") or "-"),
                    str(step.get("checkpoint_state") or "-"),
                    str(step.get("expected_artifact_kind") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(steps) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(steps) - limit}"


def _source_refresh_recipe(
    *,
    source_kind: str,
    source_artifact: str,
    symbol: str = "",
    reference_now: dt.datetime,
) -> dict[str, Any]:
    kind_text = str(source_kind or "").strip()
    artifact_text = str(source_artifact or "").strip()
    artifact_path = Path(artifact_text).expanduser() if artifact_text else Path()
    review_dir = artifact_path.parent if artifact_path.suffix else DEFAULT_REVIEW_DIR
    output_root = review_dir.parent if review_dir.name == "review" else DEFAULT_OUTPUT_ROOT
    context_path = review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"
    now_text = reference_now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recipe = {
        "script": "",
        "command_hint": "",
        "expected_status": "",
        "expected_artifact_kind": "",
        "expected_artifact_path_hint": "",
        "note": "",
        "followup_script": "",
        "followup_command_hint": "",
        "verify_hint": "",
        "steps_brief": "",
        "step_checkpoint_brief": "",
        "steps": [],
    }
    if kind_text != "crypto_route":
        return recipe
    refresh_script_path = Path(__file__).resolve().parent / "refresh_crypto_route_state.py"
    brief_script_path = Path(__file__).resolve().parent / "build_crypto_route_brief.py"
    operator_script_path = Path(__file__).resolve().parent / "build_crypto_route_operator_brief.py"
    script_path = Path(__file__).resolve().parent / "run_hot_universe_research.py"
    followup_script_path = Path(__file__).resolve().parent / "refresh_commodity_paper_execution_state.py"
    start_text = ""
    end_text = ""
    universe_file = ""
    if artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        start_text = str(payload.get("start") or "").strip()
        end_text = str(payload.get("end") or "").strip()
        universe_file = _resolve_universe_file_path(
            review_dir=review_dir,
            payload=payload,
            reference_now=reference_now,
        )
    if not end_text:
        end_text = reference_now.date().isoformat()
    if not start_text:
        start_text = (reference_now.date() - dt.timedelta(days=3)).isoformat()
    if not universe_file:
        universe_file = _resolve_universe_file_path(
            review_dir=review_dir,
            payload={},
            reference_now=reference_now,
        )
    crypto_refresh_batches = _crypto_route_refresh_batches(universe_file)
    refresh_command = [
        "python3",
        str(refresh_script_path),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--now",
        now_text,
    ]
    brief_command = [
        "python3",
        str(brief_script_path),
        "--review-dir",
        str(review_dir),
        "--now",
        now_text,
    ]
    operator_command = [
        "python3",
        str(operator_script_path),
        "--review-dir",
        str(review_dir),
        "--now",
        now_text,
    ]
    command = [
        "python3",
        str(script_path),
        "--output-root",
        str(output_root),
        "--review-dir",
        str(review_dir),
        "--start",
        start_text,
        "--end",
        end_text,
        "--now",
        now_text,
    ]
    if universe_file:
        command.extend(["--universe-file", universe_file])
    for batch_name in crypto_refresh_batches:
        command.extend(["--batch", batch_name])
    followup_command = [
        "python3",
        str(followup_script_path),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--context-path",
        str(context_path),
    ]
    steps = [
        {
            "rank": 1,
            "name": "refresh_crypto_route_brief",
            "script": str(refresh_script_path),
            "command_hint": shlex.join(refresh_command),
            "expected_status": "ok",
            "expected_artifact_kind": "crypto_route_brief",
            "expected_artifact_path_hint": str(review_dir / "*_crypto_route_brief.json"),
        },
        {
            "rank": 2,
            "name": "refresh_crypto_route_operator_brief",
            "script": str(operator_script_path),
            "command_hint": shlex.join(operator_command),
            "expected_status": "ok",
            "expected_artifact_kind": "crypto_route_operator_brief",
            "expected_artifact_path_hint": str(review_dir / "*_crypto_route_operator_brief.json"),
        },
        {
            "rank": 3,
            "name": "refresh_hot_universe_research_embedding",
            "script": str(script_path),
            "command_hint": shlex.join(command),
            "expected_status": "ok",
            "expected_artifact_kind": "hot_universe_research",
            "expected_artifact_path_hint": str(review_dir / "*_hot_universe_research.json"),
        },
        {
            "rank": 4,
            "name": "refresh_commodity_handoff",
            "script": str(followup_script_path),
            "command_hint": shlex.join(followup_command),
            "expected_status": "ok",
            "expected_artifact_kind": "commodity_paper_execution_refresh",
            "expected_artifact_path_hint": str(review_dir / "*_commodity_paper_execution_refresh.json"),
        },
    ]
    steps = [_recipe_step_with_checkpoint(step, reference_now=reference_now) for step in steps]
    recipe["script"] = str(refresh_script_path)
    recipe["command_hint"] = shlex.join(refresh_command)
    recipe["expected_status"] = "ok"
    recipe["expected_artifact_kind"] = "crypto_route_brief"
    recipe["expected_artifact_path_hint"] = str(review_dir / "*_crypto_route_brief.json")
    recipe["note"] = "guarded entrypoint refreshes native crypto route sources before the remaining pipeline steps"
    recipe["followup_script"] = str(followup_script_path)
    recipe["followup_command_hint"] = shlex.join(followup_command)
    recipe["verify_hint"] = _source_refresh_checklist_verify_hint(symbol=symbol)
    recipe["steps_brief"] = " | ".join(f"{step['rank']}:{step['name']}" for step in steps)
    recipe["step_checkpoint_brief"] = _recipe_step_checkpoint_brief(steps)
    recipe["steps"] = steps
    return recipe


def _source_refresh_checklist(items: list[dict[str, Any]], *, reference_now: dt.datetime) -> list[dict[str, Any]]:
    checklist: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        recipe = _source_refresh_recipe(
            source_kind=str(item.get("source_kind") or ""),
            source_artifact=str(item.get("source_artifact") or ""),
            symbol=str(item.get("symbol") or ""),
            reference_now=reference_now,
        )
        item["state"] = _source_refresh_checklist_state(action=str(item.get("action") or ""))
        item["blocker_detail"] = _source_refresh_checklist_blocker(
            source_kind=str(item.get("source_kind") or ""),
            source_status=str(item.get("source_status") or ""),
            source_recency=str(item.get("source_recency") or ""),
            source_age_minutes=item.get("source_age_minutes"),
        )
        item["done_when"] = _source_refresh_checklist_done_when(
            symbol=str(item.get("symbol") or ""),
            source_kind=str(item.get("source_kind") or ""),
            source_status=str(item.get("source_status") or ""),
            action=str(item.get("action") or ""),
        )
        item["recipe_script"] = str(recipe.get("script") or "")
        item["recipe_command_hint"] = str(recipe.get("command_hint") or "")
        item["recipe_expected_status"] = str(recipe.get("expected_status") or "")
        item["recipe_expected_artifact_kind"] = str(recipe.get("expected_artifact_kind") or "")
        item["recipe_expected_artifact_path_hint"] = str(recipe.get("expected_artifact_path_hint") or "")
        item["recipe_note"] = str(recipe.get("note") or "")
        item["recipe_followup_script"] = str(recipe.get("followup_script") or "")
        item["recipe_followup_command_hint"] = str(recipe.get("followup_command_hint") or "")
        item["recipe_verify_hint"] = str(recipe.get("verify_hint") or "")
        item["recipe_steps_brief"] = str(recipe.get("steps_brief") or "")
        item["recipe_step_checkpoint_brief"] = str(recipe.get("step_checkpoint_brief") or "")
        item["recipe_steps"] = list(recipe.get("steps") or [])
        checklist.append(item)
    return checklist


def _source_refresh_checklist_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _crypto_route_head_source_refresh_lane(
    *,
    row: dict[str, Any] | None,
    reference_now: dt.datetime,
    cooldown_next_eligible_end_date: str = "",
    source_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = dict(row or {})
    symbol = str(item.get("symbol") or "").strip().upper()
    if not symbol:
        return {
            "status": "not_applicable",
            "brief": "not_applicable:-",
            "slot": "-",
            "symbol": "-",
            "action": "-",
            "source_kind": "-",
            "source_health": "-",
            "source_artifact": "",
            "blocker_detail": "crypto route head source refresh lane is only active when a crypto review head is present.",
            "done_when": "crypto route head returns before reassessing source refresh",
            "recipe": {},
        }

    action = str(item.get("source_refresh_action") or "").strip()
    source_kind = str(item.get("source_kind") or "").strip()
    source_status = str(item.get("source_status") or "").strip()
    source_recency = str(item.get("source_recency") or "").strip()
    source_health = str(item.get("source_health") or "").strip()
    source_artifact = str(item.get("source_artifact") or "").strip()
    source_age_minutes = item.get("source_age_minutes")

    source_lane = dict(source_payload or {})
    source_lane_symbol = str(source_lane.get("crypto_route_head_source_refresh_symbol") or "").strip().upper()
    source_lane_status = str(source_lane.get("crypto_route_head_source_refresh_status") or "").strip()
    source_lane_action = str(source_lane.get("crypto_route_head_source_refresh_action") or "").strip()
    if (
        symbol
        and symbol == source_lane_symbol
        and action in ("", "read_current_artifact")
        and source_lane_status in {"ready", "deferred_until_next_eligible_end_date"}
    ):
        return {
            "status": source_lane_status,
            "brief": str(source_lane.get("crypto_route_head_source_refresh_brief") or f"{source_lane_status}:{symbol}:{source_lane_action or '-'}"),
            "slot": str(item.get("slot") or "").strip() or "-",
            "symbol": symbol,
            "action": source_lane_action or action or "-",
            "source_kind": str(source_lane.get("crypto_route_head_source_refresh_source_kind") or source_kind or "-"),
            "source_health": str(source_lane.get("crypto_route_head_source_refresh_source_health") or source_health or "-"),
            "source_artifact": str(source_lane.get("crypto_route_head_source_refresh_source_artifact") or source_artifact),
            "blocker_detail": str(source_lane.get("crypto_route_head_source_refresh_blocker_detail") or ""),
            "done_when": str(source_lane.get("crypto_route_head_source_refresh_done_when") or ""),
            "recipe": {},
        }

    if action in ("", "read_current_artifact"):
        status = "ready"
        blocker_detail = (
            f"{symbol} currently uses a readable {source_kind or 'source'} artifact "
            f"({source_status or '-'}, {source_recency or '-'})"
        )
        done_when = f"keep {symbol} on the current {source_kind or 'source'} artifact while it remains usable"
        recipe: dict[str, Any] = {}
    elif action == "wait_for_next_eligible_end_date":
        status = "deferred_until_next_eligible_end_date"
        ready_on = str(cooldown_next_eligible_end_date or "").strip() or "-"
        blocker_detail = f"{symbol} source refresh is deferred until the next eligible end date ({ready_on})"
        done_when = (
            f"{symbol} reaches the next eligible end date ({ready_on}) and still remains the crypto review head"
        )
        recipe = _source_refresh_recipe(
            source_kind=source_kind,
            source_artifact=source_artifact,
            symbol=symbol,
            reference_now=reference_now,
        )
    else:
        status = _source_refresh_checklist_state(action=action)
        blocker_detail = _source_refresh_checklist_blocker(
            source_kind=source_kind,
            source_status=source_status,
            source_recency=source_recency,
            source_age_minutes=source_age_minutes,
        )
        done_when = _source_refresh_checklist_done_when(
            symbol=symbol,
            source_kind=source_kind,
            source_status=source_status,
            action=action,
        )
        recipe = _source_refresh_recipe(
            source_kind=source_kind,
            source_artifact=source_artifact,
            symbol=symbol,
            reference_now=reference_now,
        )

    return {
        "status": status,
        "brief": f"{status}:{symbol}:{action or '-'}",
        "slot": str(item.get("slot") or "").strip() or "-",
        "symbol": symbol,
        "action": action or "-",
        "source_kind": source_kind or "-",
        "source_health": source_health or "-",
        "source_artifact": source_artifact,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "recipe": recipe,
    }


def _source_refresh_pipeline_relevance(
    *,
    crypto_head_source_refresh: dict[str, Any] | None,
    pending_steps: list[dict[str, Any]],
    deferred_steps: list[dict[str, Any]],
) -> dict[str, str]:
    head = dict(crypto_head_source_refresh or {})
    head_status = str(head.get("status") or "").strip()
    head_symbol = str(head.get("symbol") or "").strip().upper() or "-"
    head_action = str(head.get("action") or "").strip() or "-"
    pending_count = len(pending_steps)
    deferred_count = len(deferred_steps)

    if pending_count == 0 and deferred_count == 0:
        return {
            "status": "clear",
            "brief": "clear",
            "blocker_detail": "no source refresh pipeline work remains.",
            "done_when": "keep the source refresh pipeline clear",
        }

    if head_status in {"ready", "deferred_until_next_eligible_end_date"} and head_action in {
        "read_current_artifact",
        "wait_for_next_eligible_end_date",
    }:
        remaining_count = pending_count + deferred_count
        return {
            "status": "non_blocking_for_current_crypto_head",
            "brief": f"non_blocking_for_current_crypto_head:{head_symbol}:{remaining_count}",
            "blocker_detail": (
                f"{head_symbol} current source lane is already {head_status} via {head_action}; "
                "remaining source refresh pipeline work is broader carry-over and does not block the current crypto head."
            ),
            "done_when": (
                "run the remaining source refresh pipeline only when broader refresh freshness is required "
                "or the crypto review queue head changes"
            ),
        }

    return {
        "status": "blocking_for_current_crypto_head",
        "brief": f"blocking_for_current_crypto_head:{head_symbol}:{pending_count + deferred_count}",
        "blocker_detail": (
            f"{head_symbol} current source lane is {head_status or '-'} via {head_action}; "
            "remaining source refresh pipeline work still blocks the current crypto head."
        ),
        "done_when": "clear the remaining source refresh pipeline steps before promoting the current crypto head",
    }


def _source_refresh_checklist_verify_hint(*, symbol: str) -> str:
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    return f"rerun commodity refresh and confirm {symbol_text} leaves operator_source_refresh_queue"


def _fallback_symbol_list(values: list[str], fallback_symbol: str, fallback_count: int) -> list[str]:
    items = [str(v).strip().upper() for v in values if str(v).strip()]
    if items:
        return items
    symbol = str(fallback_symbol or "").strip().upper()
    if symbol and int(fallback_count or 0) > 0:
        return [symbol]
    return []


def _fmt_num(raw: Any) -> str:
    try:
        value = float(raw)
    except Exception:
        text = str(raw or "").strip()
        return text or "-"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _find_execution_item(items: list[dict[str, Any]], *, execution_id: str, symbol: str) -> dict[str, Any]:
    execution_text = str(execution_id or "").strip()
    symbol_text = str(symbol or "").strip().upper()
    for row in items:
        if not isinstance(row, dict):
            continue
        if execution_text and str(row.get("execution_id") or "").strip() == execution_text:
            return dict(row)
    for row in items:
        if not isinstance(row, dict):
            continue
        if symbol_text and str(row.get("symbol") or "").strip().upper() == symbol_text:
            return dict(row)
    return {}


def _extract_paper_evidence_summary(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "paper_execution_evidence_present": bool(row.get("paper_execution_evidence_present")),
        "paper_open_date": row.get("paper_open_date"),
        "paper_signal_date": row.get("paper_signal_date"),
        "paper_execution_side": row.get("paper_execution_side"),
        "paper_execution_status": row.get("paper_execution_status"),
        "paper_runtime_mode": row.get("paper_runtime_mode"),
        "paper_order_mode": row.get("paper_order_mode"),
        "paper_size_pct": row.get("paper_size_pct"),
        "paper_risk_pct": row.get("paper_risk_pct"),
        "paper_entry_price": row.get("paper_entry_price"),
        "paper_stop_price": row.get("paper_stop_price"),
        "paper_target_price": row.get("paper_target_price"),
        "paper_quote_usdt": row.get("paper_quote_usdt"),
        "paper_execution_price_normalization_mode": row.get("paper_execution_price_normalization_mode"),
        "paper_proxy_price_normalized": row.get("paper_proxy_price_normalized"),
        "paper_signal_price_reference_kind": row.get("paper_signal_price_reference_kind"),
        "paper_signal_price_reference_source": row.get("paper_signal_price_reference_source"),
        "paper_signal_price_reference_provider": row.get("paper_signal_price_reference_provider"),
        "paper_signal_price_reference_symbol": row.get("paper_signal_price_reference_symbol"),
    }


def _paper_execution_status(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    evidence_snapshot = row.get("paper_execution_evidence_snapshot")
    executed_plan = evidence_snapshot.get("executed_plan", {}) if isinstance(evidence_snapshot, dict) else {}
    return str(row.get("paper_execution_status") or executed_plan.get("status") or "").strip().upper()


def _commodity_focus_lifecycle_gate(
    *,
    area: str,
    action: str,
    symbol: str,
    focus_evidence_summary: dict[str, Any],
) -> dict[str, str]:
    gate = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "commodity focus lifecycle gate is not needed when the current focus is not a paper review/retro/close-evidence item.",
        "done_when": "current focus becomes a commodity paper review/retro/close-evidence item before reassessing lifecycle state",
    }
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    if area_text not in {
        "commodity_execution_review",
        "commodity_execution_retro",
        "commodity_execution_close_evidence",
    }:
        return gate
    if action_text not in {
        "review_paper_execution",
        "review_paper_execution_retro",
        "wait_for_paper_execution_close_evidence",
    }:
        return gate
    if not isinstance(focus_evidence_summary, dict) or not focus_evidence_summary:
        gate["status"] = "awaiting_execution_evidence_snapshot"
        gate["brief"] = f"awaiting_execution_evidence_snapshot:{symbol_text}"
        gate["blocker_detail"] = "current commodity focus does not yet have a paper execution evidence snapshot."
        gate["done_when"] = f"{symbol_text} gains a paper execution evidence snapshot"
        return gate

    paper_status = str(focus_evidence_summary.get("paper_execution_status") or "").strip().upper()
    open_date = str(focus_evidence_summary.get("paper_open_date") or "").strip()
    if action_text in {"review_paper_execution_retro", "wait_for_paper_execution_close_evidence"} and paper_status == "OPEN":
        blocker = "paper execution evidence is present, but position is still OPEN"
        if open_date:
            blocker += f" since {open_date}"
        if action_text == "wait_for_paper_execution_close_evidence":
            blocker += "; waiting for close evidence"
        else:
            blocker += "; retro should wait for close evidence"
        gate["status"] = "open_position_wait_close_evidence"
        gate["brief"] = f"open_position_wait_close_evidence:{symbol_text}"
        gate["blocker_detail"] = blocker
        if action_text == "wait_for_paper_execution_close_evidence":
            gate["done_when"] = (
                f"{symbol_text} paper_execution_status leaves OPEN and close evidence becomes available"
            )
        else:
            gate["done_when"] = f"{symbol_text} paper_execution_status leaves OPEN and retro can evaluate close outcome"
        return gate

    gate["status"] = "ready_for_pending_focus_action"
    gate["brief"] = f"ready_for_pending_focus_action:{symbol_text}:{paper_status or '-'}"
    if action_text == "review_paper_execution_retro":
        gate["blocker_detail"] = "paper execution evidence is present; retro item still pending"
        gate["done_when"] = f"{symbol_text} leaves retro_pending_symbols"
    elif action_text == "wait_for_paper_execution_close_evidence":
        gate["blocker_detail"] = "paper execution close evidence is present; wait-close item can clear"
        gate["done_when"] = f"{symbol_text} leaves wait_for_paper_execution_close_evidence state"
    else:
        gate["blocker_detail"] = "paper execution evidence is present; review item still pending"
        gate["done_when"] = f"{symbol_text} leaves review_pending_symbols"
    return gate


def build_operator_brief(
    action_payload: dict[str, Any],
    crypto_payload: dict[str, Any],
    crypto_route_payload: dict[str, Any] | None = None,
    commodity_payload: dict[str, Any] | None = None,
    commodity_ticket_payload: dict[str, Any] | None = None,
    commodity_ticket_book_payload: dict[str, Any] | None = None,
    commodity_execution_preview_payload: dict[str, Any] | None = None,
    commodity_execution_artifact_payload: dict[str, Any] | None = None,
    commodity_execution_queue_payload: dict[str, Any] | None = None,
    commodity_execution_review_payload: dict[str, Any] | None = None,
    commodity_execution_retro_payload: dict[str, Any] | None = None,
    commodity_execution_gap_payload: dict[str, Any] | None = None,
    commodity_execution_bridge_payload: dict[str, Any] | None = None,
    cross_market_operator_state_payload: dict[str, Any] | None = None,
    live_gate_blocker_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    partial_bridge_statuses = {
        "bridge_partially_bridged_missing_remainder",
        "bridge_partially_bridged_stale_remainder",
        "bridge_partially_bridged_proxy_remainder",
    }
    pending_review_statuses = {
        "paper-execution-review-pending",
        "paper-execution-review-pending-close-remainder",
        "paper-execution-review-pending-close-fill-remainder",
        "paper-execution-review-pending-fill-remainder",
        "paper-execution-close-evidence-pending",
        "paper-execution-close-evidence-pending-fill-remainder",
    }
    pending_retro_statuses = {
        "paper-execution-retro-pending",
        "paper-execution-retro-pending-fill-remainder",
        "paper-execution-retro-pending-close-remainder",
        "paper-execution-retro-pending-close-fill-remainder",
    }
    bridge_blocked_statuses = {
        "blocked_missing_directional_signal",
        "blocked_stale_directional_signal",
        "blocked_proxy_price_reference_only",
        *partial_bridge_statuses,
    }
    action_ladder = dict(action_payload.get("research_action_ladder") or {})
    embedded_crypto_route_brief = dict(crypto_payload.get("crypto_route_brief") or {})
    embedded_crypto_route_operator_brief = dict(crypto_payload.get("crypto_route_operator_brief") or {})
    dedicated_crypto_route_payload = dict(crypto_route_payload or {})
    cross_market_operator_state_payload = dict(cross_market_operator_state_payload or {})
    dedicated_crypto_route_operator_brief: dict[str, Any] = {}
    dedicated_crypto_route_brief: dict[str, Any] = {}
    if dedicated_crypto_route_payload:
        if any(
            str(dedicated_crypto_route_payload.get(key) or "").strip()
            for key in (
                "operator_status",
                "route_stack_brief",
                "next_focus_symbol",
                "next_focus_action",
                "operator_text",
                "operator_lines",
                "focus_window_floor",
                "xlong_flow_window_floor",
                "price_state_window_floor",
            )
        ):
            dedicated_crypto_route_operator_brief = dedicated_crypto_route_payload
        if any(
            str(dedicated_crypto_route_payload.get(key) or "").strip()
            for key in (
                "brief_text",
                "brief_lines",
                "route_stack_brief",
                "next_focus_symbol",
                "next_focus_action",
            )
        ):
            dedicated_crypto_route_brief = dedicated_crypto_route_payload
    crypto_route_brief = (
        dedicated_crypto_route_brief
        or embedded_crypto_route_brief
        or dedicated_crypto_route_operator_brief
    )
    crypto_route_operator_brief = (
        dedicated_crypto_route_operator_brief
        or embedded_crypto_route_operator_brief
        or dedicated_crypto_route_brief
    )
    commodity_payload = dict(commodity_payload or {})

    focus_primary = [str(x).strip() for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]
    focus_regime = [str(x).strip() for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()]
    shadow_only = [str(x).strip() for x in action_ladder.get("shadow_only_batches", []) if str(x).strip()]
    research_queue = [str(x).strip() for x in action_ladder.get("research_queue_batches", []) if str(x).strip()]
    avoid = [str(x).strip() for x in action_ladder.get("avoid_batches", []) if str(x).strip()]

    commodity_status = str(commodity_payload.get("route_status") or "").strip()
    commodity_execution_mode = str(commodity_payload.get("execution_mode") or "").strip()
    commodity_route_stack = str(commodity_payload.get("route_stack_brief") or "").strip()
    commodity_primary = [str(x).strip() for x in commodity_payload.get("focus_primary_batches", []) if str(x).strip()]
    commodity_regime = [
        str(x).strip() for x in commodity_payload.get("focus_with_regime_filter_batches", []) if str(x).strip()
    ]
    commodity_shadow = [str(x).strip() for x in commodity_payload.get("shadow_only_batches", []) if str(x).strip()]
    commodity_leaders_primary = [
        str(x).strip().upper() for x in commodity_payload.get("leader_symbols_primary", []) if str(x).strip()
    ]
    commodity_leaders_regime = [
        str(x).strip().upper() for x in commodity_payload.get("leader_symbols_regime_filter", []) if str(x).strip()
    ]
    commodity_focus_batch = str(commodity_payload.get("next_focus_batch") or "").strip()
    commodity_focus_symbols = [
        str(x).strip().upper() for x in commodity_payload.get("next_focus_symbols", []) if str(x).strip()
    ]
    commodity_next_stage = str(commodity_payload.get("next_stage") or "").strip()
    commodity_ticket_payload = dict(commodity_ticket_payload or {})
    commodity_ticket_status = str(commodity_ticket_payload.get("ticket_status") or "").strip()
    commodity_ticket_stack = str(commodity_ticket_payload.get("ticket_stack_brief") or "").strip()
    commodity_paper_ready_batches = [
        str(x).strip() for x in commodity_ticket_payload.get("paper_ready_batches", []) if str(x).strip()
    ]
    commodity_ticket_focus_batch = str(commodity_ticket_payload.get("next_ticket_batch") or "").strip()
    commodity_ticket_focus_symbols = [
        str(x).strip().upper() for x in commodity_ticket_payload.get("next_ticket_symbols", []) if str(x).strip()
    ]
    commodity_ticket_count = len([row for row in commodity_ticket_payload.get("tickets", []) if isinstance(row, dict)])
    commodity_ticket_book_payload = dict(commodity_ticket_book_payload or {})
    commodity_ticket_book_status = str(commodity_ticket_book_payload.get("ticket_book_status") or "").strip()
    commodity_ticket_book_stack = str(commodity_ticket_book_payload.get("ticket_book_stack_brief") or "").strip()
    commodity_actionable_batches = [
        str(x).strip() for x in commodity_ticket_book_payload.get("actionable_batches", []) if str(x).strip()
    ]
    commodity_shadow_batches = [
        str(x).strip() for x in commodity_ticket_book_payload.get("shadow_batches", []) if str(x).strip()
    ]
    commodity_next_ticket_id = str(commodity_ticket_book_payload.get("next_ticket_id") or "").strip()
    commodity_next_ticket_symbol = str(commodity_ticket_book_payload.get("next_ticket_symbol") or "").strip().upper()
    commodity_actionable_ticket_count = int(commodity_ticket_book_payload.get("actionable_ticket_count", 0) or 0)
    commodity_execution_preview_payload = dict(commodity_execution_preview_payload or {})
    commodity_execution_preview_status = str(commodity_execution_preview_payload.get("execution_preview_status") or "").strip()
    commodity_execution_preview_stack = str(commodity_execution_preview_payload.get("preview_stack_brief") or "").strip()
    commodity_preview_ready_batches = [
        str(x).strip() for x in commodity_execution_preview_payload.get("preview_ready_batches", []) if str(x).strip()
    ]
    commodity_preview_shadow_batches = [
        str(x).strip() for x in commodity_execution_preview_payload.get("shadow_only_batches", []) if str(x).strip()
    ]
    commodity_next_execution_batch = str(commodity_execution_preview_payload.get("next_execution_batch") or "").strip()
    commodity_next_execution_symbols = [
        str(x).strip().upper() for x in commodity_execution_preview_payload.get("next_execution_symbols", []) if str(x).strip()
    ]
    commodity_next_execution_ticket_ids = [
        str(x).strip() for x in commodity_execution_preview_payload.get("next_execution_ticket_ids", []) if str(x).strip()
    ]
    commodity_next_execution_regime_gate = str(
        commodity_execution_preview_payload.get("next_execution_regime_gate") or ""
    ).strip()
    commodity_next_execution_weight_hint_sum = float(
        commodity_execution_preview_payload.get("next_execution_weight_hint_sum", 0.0) or 0.0
    )
    commodity_execution_artifact_payload = dict(commodity_execution_artifact_payload or {})
    commodity_execution_artifact_status = str(
        commodity_execution_artifact_payload.get("execution_artifact_status") or ""
    ).strip()
    commodity_execution_artifact_stack = str(
        commodity_execution_artifact_payload.get("execution_stack_brief") or ""
    ).strip()
    commodity_execution_batch = str(commodity_execution_artifact_payload.get("execution_batch") or "").strip()
    commodity_execution_symbols = [
        str(x).strip().upper() for x in commodity_execution_artifact_payload.get("execution_symbols", []) if str(x).strip()
    ]
    commodity_execution_ticket_ids = [
        str(x).strip() for x in commodity_execution_artifact_payload.get("execution_ticket_ids", []) if str(x).strip()
    ]
    commodity_execution_regime_gate = str(commodity_execution_artifact_payload.get("execution_regime_gate") or "").strip()
    commodity_execution_weight_hint_sum = float(
        commodity_execution_artifact_payload.get("execution_weight_hint_sum", 0.0) or 0.0
    )
    commodity_execution_item_count = int(commodity_execution_artifact_payload.get("execution_item_count", 0) or 0)
    commodity_actionable_execution_item_count = int(
        commodity_execution_artifact_payload.get("actionable_execution_item_count", 0) or 0
    )
    commodity_execution_queue_payload = dict(commodity_execution_queue_payload or {})
    commodity_execution_queue_status = str(
        commodity_execution_queue_payload.get("execution_queue_status") or ""
    ).strip()
    commodity_execution_queue_stack = str(
        commodity_execution_queue_payload.get("queue_stack_brief") or ""
    ).strip()
    commodity_execution_queue_batch = str(
        commodity_execution_queue_payload.get("execution_batch") or ""
    ).strip()
    commodity_queue_depth = int(commodity_execution_queue_payload.get("queue_depth", 0) or 0)
    commodity_actionable_queue_depth = int(
        commodity_execution_queue_payload.get("actionable_queue_depth", 0) or 0
    )
    commodity_next_queue_execution_id = str(
        commodity_execution_queue_payload.get("next_execution_id") or ""
    ).strip()
    commodity_next_queue_execution_symbol = str(
        commodity_execution_queue_payload.get("next_execution_symbol") or ""
    ).strip().upper()
    commodity_execution_review_payload = dict(commodity_execution_review_payload or {})
    commodity_execution_review_items = [
        row for row in commodity_execution_review_payload.get("review_items", []) if isinstance(row, dict)
    ]
    commodity_execution_review_status = str(
        commodity_execution_review_payload.get("execution_review_status") or ""
    ).strip()
    commodity_execution_review_stack = str(
        commodity_execution_review_payload.get("review_stack_brief") or ""
    ).strip()
    commodity_review_item_count = int(commodity_execution_review_payload.get("review_item_count", 0) or 0)
    commodity_actionable_review_item_count = int(
        commodity_execution_review_payload.get("actionable_review_item_count", 0) or 0
    )
    commodity_review_fill_evidence_pending_count = int(
        commodity_execution_review_payload.get("fill_evidence_pending_count", 0) or 0
    )
    commodity_next_review_execution_id = str(
        commodity_execution_review_payload.get("next_review_execution_id") or ""
    ).strip()
    commodity_next_review_execution_symbol = str(
        commodity_execution_review_payload.get("next_review_execution_symbol") or ""
    ).strip().upper()
    commodity_next_review_fill_evidence_execution_id = str(
        commodity_execution_review_payload.get("next_fill_evidence_execution_id") or ""
    ).strip()
    commodity_next_review_fill_evidence_execution_symbol = str(
        commodity_execution_review_payload.get("next_fill_evidence_execution_symbol") or ""
    ).strip().upper()
    derived_review_close_evidence_items = [
        row
        for row in commodity_execution_review_payload.get("review_items", [])
        if isinstance(row, dict)
        and (
            str(row.get("review_status") or "").strip() == "awaiting_paper_execution_close_evidence"
            or (_paper_execution_status(row) == "OPEN" and bool(row.get("paper_execution_evidence_present")))
        )
    ]
    commodity_review_close_evidence_pending_count = int(
        commodity_execution_review_payload.get("close_evidence_pending_count", len(derived_review_close_evidence_items))
        or 0
    )
    commodity_next_review_close_evidence_execution_id = str(
        commodity_execution_review_payload.get("next_close_evidence_execution_id")
        or (derived_review_close_evidence_items[0].get("execution_id") if derived_review_close_evidence_items else "")
        or ""
    ).strip()
    commodity_next_review_close_evidence_execution_symbol = str(
        commodity_execution_review_payload.get("next_close_evidence_execution_symbol")
        or (derived_review_close_evidence_items[0].get("symbol") if derived_review_close_evidence_items else "")
        or ""
    ).strip().upper()
    commodity_execution_retro_payload = dict(commodity_execution_retro_payload or {})
    commodity_execution_retro_items = [
        row for row in commodity_execution_retro_payload.get("retro_items", []) if isinstance(row, dict)
    ]
    commodity_execution_retro_status = str(
        commodity_execution_retro_payload.get("execution_retro_status") or ""
    ).strip()
    commodity_execution_retro_stack = str(
        commodity_execution_retro_payload.get("retro_stack_brief") or ""
    ).strip()
    commodity_retro_item_count = int(commodity_execution_retro_payload.get("retro_item_count", 0) or 0)
    commodity_actionable_retro_item_count = int(
        commodity_execution_retro_payload.get("actionable_retro_item_count", 0) or 0
    )
    commodity_retro_fill_evidence_pending_count = int(
        commodity_execution_retro_payload.get("fill_evidence_pending_count", 0) or 0
    )
    commodity_next_retro_execution_id = str(
        commodity_execution_retro_payload.get("next_retro_execution_id") or ""
    ).strip()
    commodity_next_retro_execution_symbol = str(
        commodity_execution_retro_payload.get("next_retro_execution_symbol") or ""
    ).strip().upper()
    commodity_next_retro_fill_evidence_execution_id = str(
        commodity_execution_retro_payload.get("next_fill_evidence_execution_id") or ""
    ).strip()
    commodity_next_retro_fill_evidence_execution_symbol = str(
        commodity_execution_retro_payload.get("next_fill_evidence_execution_symbol") or ""
    ).strip().upper()
    derived_close_evidence_items = [
        row
        for row in commodity_execution_retro_items
        if (
            str(row.get("retro_status") or "").strip() == "awaiting_paper_execution_close_evidence"
            or (
                _paper_execution_status(row) == "OPEN"
                and bool(row.get("paper_execution_evidence_present"))
            )
        )
    ]
    commodity_close_evidence_pending_count = int(
        commodity_execution_retro_payload.get("close_evidence_pending_count", len(derived_close_evidence_items)) or 0
    )
    commodity_next_close_evidence_execution_id = str(
        commodity_execution_retro_payload.get("next_close_evidence_execution_id")
        or (derived_close_evidence_items[0].get("execution_id") if derived_close_evidence_items else "")
        or ""
    ).strip()
    commodity_next_close_evidence_execution_symbol = str(
        commodity_execution_retro_payload.get("next_close_evidence_execution_symbol")
        or (derived_close_evidence_items[0].get("symbol") if derived_close_evidence_items else "")
        or ""
    ).strip().upper()
    commodity_review_pending_symbols = _fallback_symbol_list(
        commodity_execution_review_payload.get("review_pending_symbols", []),
        commodity_next_review_execution_symbol,
        commodity_actionable_review_item_count,
    )
    commodity_review_close_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_review_payload.get("close_evidence_pending_symbols", []),
        commodity_next_review_close_evidence_execution_symbol,
        commodity_review_close_evidence_pending_count,
    )
    commodity_review_fill_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_review_payload.get("fill_evidence_pending_symbols", []),
        commodity_next_review_fill_evidence_execution_symbol,
        commodity_review_fill_evidence_pending_count,
    )
    commodity_retro_pending_symbols = _fallback_symbol_list(
        commodity_execution_retro_payload.get("retro_pending_symbols", []),
        commodity_next_retro_execution_symbol,
        commodity_actionable_retro_item_count,
    )
    commodity_close_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_retro_payload.get("close_evidence_pending_symbols", []),
        commodity_next_close_evidence_execution_symbol,
        commodity_close_evidence_pending_count,
    )
    commodity_retro_fill_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_retro_payload.get("fill_evidence_pending_symbols", []),
        commodity_next_retro_fill_evidence_execution_symbol,
        commodity_retro_fill_evidence_pending_count,
    )
    if commodity_close_evidence_pending_symbols:
        close_evidence_symbol_set = set(commodity_close_evidence_pending_symbols)
        commodity_retro_pending_symbols = [
            symbol for symbol in commodity_retro_pending_symbols if symbol not in close_evidence_symbol_set
        ]
        commodity_actionable_retro_item_count = len(commodity_retro_pending_symbols)
        if (
            commodity_next_retro_execution_symbol
            and commodity_next_retro_execution_symbol in close_evidence_symbol_set
            and commodity_actionable_retro_item_count == 0
        ):
            commodity_next_retro_execution_id = ""
            commodity_next_retro_execution_symbol = ""
    if not commodity_close_evidence_pending_symbols and commodity_review_close_evidence_pending_symbols:
        commodity_close_evidence_pending_count = commodity_review_close_evidence_pending_count
        commodity_close_evidence_pending_symbols = list(commodity_review_close_evidence_pending_symbols)
        commodity_next_close_evidence_execution_id = commodity_next_review_close_evidence_execution_id
        commodity_next_close_evidence_execution_symbol = commodity_next_review_close_evidence_execution_symbol
    review_close_evidence_symbol_set = set(commodity_review_close_evidence_pending_symbols)
    if review_close_evidence_symbol_set:
        commodity_review_pending_symbols = [
            symbol for symbol in commodity_review_pending_symbols if symbol not in review_close_evidence_symbol_set
        ]
        commodity_actionable_review_item_count = len(commodity_review_pending_symbols)
        if (
            commodity_next_review_execution_symbol
            and commodity_next_review_execution_symbol in review_close_evidence_symbol_set
            and commodity_actionable_review_item_count == 0
        ):
            commodity_next_review_execution_id = ""
            commodity_next_review_execution_symbol = ""
    if commodity_actionable_review_item_count > 0:
        if commodity_review_close_evidence_pending_count > 0 and commodity_review_fill_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-review-pending-close-fill-remainder"
        elif commodity_review_close_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-review-pending-close-remainder"
        elif commodity_review_fill_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-review-pending-fill-remainder"
        else:
            commodity_execution_review_status = "paper-execution-review-pending"
    elif commodity_review_close_evidence_pending_count > 0:
        if commodity_review_fill_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-close-evidence-pending-fill-remainder"
        else:
            commodity_execution_review_status = "paper-execution-close-evidence-pending"
    elif commodity_review_fill_evidence_pending_count > 0:
        commodity_execution_review_status = "paper-execution-awaiting-fill-evidence"
    else:
        commodity_execution_review_status = "paper-execution-review-empty"
    commodity_execution_gap_payload = dict(commodity_execution_gap_payload or {})
    commodity_execution_gap_status = str(
        commodity_execution_gap_payload.get("gap_status") or ""
    ).strip()
    commodity_execution_gap_decision = str(
        commodity_execution_gap_payload.get("current_decision") or ""
    ).strip()
    commodity_execution_gap_reason_codes = [
        str(x).strip() for x in commodity_execution_gap_payload.get("gap_reason_codes", []) if str(x).strip()
    ]
    commodity_execution_gap_batch = str(
        commodity_execution_gap_payload.get("execution_batch") or ""
    ).strip()
    commodity_execution_gap_next_execution_id = str(
        commodity_execution_gap_payload.get("next_execution_id") or ""
    ).strip()
    commodity_execution_gap_next_execution_symbol = str(
        commodity_execution_gap_payload.get("next_execution_symbol") or ""
    ).strip().upper()
    commodity_execution_gap_root_cause_lines = [
        str(x).strip() for x in commodity_execution_gap_payload.get("root_cause_lines", []) if str(x).strip()
    ]
    commodity_execution_gap_recommended_actions = [
        str(x).strip() for x in commodity_execution_gap_payload.get("recommended_actions", []) if str(x).strip()
    ]
    commodity_stale_signal_watch_items = [
        dict(row)
        for row in commodity_execution_gap_payload.get("stale_directional_signal_watch_items", [])
        if isinstance(row, dict)
    ]
    commodity_execution_bridge_stale_signal_dates = {
        str(key).strip().upper(): str(value).strip()
        for key, value in dict(
            commodity_execution_gap_payload.get("queue_symbols_with_stale_directional_signal_dates", {})
        ).items()
        if str(key).strip() and str(value).strip()
    }
    commodity_execution_bridge_stale_signal_age_days = {
        str(key).strip().upper(): int(value)
        for key, value in dict(
            commodity_execution_gap_payload.get("queue_symbols_with_stale_directional_signal_age_days", {})
        ).items()
        if str(key).strip() and str(value).strip()
    }
    commodity_gap_focus_batch = (
        commodity_execution_gap_batch
        or commodity_execution_batch
        or commodity_execution_queue_batch
        or commodity_next_execution_batch
    )
    commodity_gap_focus_execution_id = commodity_execution_gap_next_execution_id or commodity_next_queue_execution_id
    commodity_gap_focus_symbol = commodity_execution_gap_next_execution_symbol or commodity_next_queue_execution_symbol
    commodity_execution_bridge_payload = dict(commodity_execution_bridge_payload or {})
    commodity_execution_bridge_status = str(
        commodity_execution_bridge_payload.get("bridge_status") or ""
    ).strip()
    commodity_execution_bridge_next_ready_id = str(
        commodity_execution_bridge_payload.get("next_ready_execution_id") or ""
    ).strip()
    commodity_execution_bridge_next_ready_symbol = str(
        commodity_execution_bridge_payload.get("next_ready_symbol") or ""
    ).strip().upper()
    commodity_execution_bridge_next_blocked_id = str(
        commodity_execution_bridge_payload.get("next_blocked_execution_id") or ""
    ).strip()
    commodity_execution_bridge_next_blocked_symbol = str(
        commodity_execution_bridge_payload.get("next_blocked_symbol") or ""
    ).strip().upper()
    commodity_execution_bridge_signal_missing_count = int(
        commodity_execution_bridge_payload.get("signal_missing_count", 0) or 0
    )
    commodity_execution_bridge_signal_stale_count = int(
        commodity_execution_bridge_payload.get("signal_stale_count", 0) or 0
    )
    commodity_execution_bridge_signal_proxy_price_only_count = int(
        commodity_execution_bridge_payload.get("signal_proxy_price_only_count", 0) or 0
    )
    commodity_execution_bridge_already_present_count = int(
        commodity_execution_bridge_payload.get("already_present_count", 0) or 0
    )
    commodity_execution_bridge_items = [
        row for row in commodity_execution_bridge_payload.get("bridge_items", []) if isinstance(row, dict)
    ]
    commodity_execution_bridge_already_bridged_symbols = [
        str(x).strip().upper()
        for x in commodity_execution_bridge_payload.get("already_bridged_symbols", [])
        if str(x).strip()
    ]
    if not commodity_execution_bridge_stale_signal_dates:
        commodity_execution_bridge_stale_signal_dates = {
            str(row.get("symbol") or "").strip().upper(): str(row.get("signal_date") or "").strip()
            for row in commodity_execution_bridge_items
            if str(row.get("symbol") or "").strip()
            and str(row.get("signal_date") or "").strip()
            and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        }
    if not commodity_execution_bridge_stale_signal_age_days:
        commodity_execution_bridge_stale_signal_age_days = {
            str(row.get("symbol") or "").strip().upper(): int(row.get("signal_age_days"))
            for row in commodity_execution_bridge_items
            if str(row.get("symbol") or "").strip()
            and row.get("signal_age_days") is not None
            and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        }
    commodity_next_fill_evidence_execution_id = (
        commodity_next_retro_fill_evidence_execution_id or commodity_next_review_fill_evidence_execution_id
    )
    commodity_next_fill_evidence_execution_symbol = (
        commodity_next_retro_fill_evidence_execution_symbol or commodity_next_review_fill_evidence_execution_symbol
    )
    commodity_fill_evidence_pending_count = max(
        commodity_review_fill_evidence_pending_count,
        commodity_retro_fill_evidence_pending_count,
    )
    bridge_ready = commodity_execution_bridge_status == "bridge_ready"
    actionable_review_pending = (
        commodity_execution_review_status in pending_review_statuses
        and commodity_actionable_review_item_count > 0
    )
    actionable_retro_pending = (
        commodity_execution_retro_status in pending_retro_statuses
        and commodity_actionable_retro_item_count > 0
    )

    crypto_status = str(crypto_route_operator_brief.get("operator_status") or crypto_route_brief.get("operator_status") or "").strip()
    route_stack = str(crypto_route_operator_brief.get("route_stack_brief") or crypto_route_brief.get("route_stack_brief") or "").strip()
    focus_symbol = str(crypto_route_operator_brief.get("next_focus_symbol") or crypto_route_brief.get("next_focus_symbol") or "").strip()
    focus_action = str(crypto_route_operator_brief.get("next_focus_action") or crypto_route_brief.get("next_focus_action") or "").strip()
    focus_reason = str(crypto_route_operator_brief.get("next_focus_reason") or crypto_route_brief.get("next_focus_reason") or "").strip()
    focus_gate = str(crypto_route_operator_brief.get("focus_window_gate") or crypto_route_brief.get("focus_window_gate") or "").strip()
    focus_window = str(crypto_route_operator_brief.get("focus_window_verdict") or crypto_route_brief.get("focus_window_verdict") or "").strip()
    focus_window_floor = str(crypto_route_operator_brief.get("focus_window_floor") or "").strip()
    price_state_window_floor = str(crypto_route_operator_brief.get("price_state_window_floor") or "").strip()
    comparative_window_takeaway = str(crypto_route_operator_brief.get("comparative_window_takeaway") or "").strip()
    xlong_flow_window_floor = str(crypto_route_operator_brief.get("xlong_flow_window_floor") or "").strip()
    xlong_comparative_window_takeaway = str(crypto_route_operator_brief.get("xlong_comparative_window_takeaway") or "").strip()
    focus_brief = str(crypto_route_operator_brief.get("focus_brief") or crypto_route_brief.get("focus_brief") or "").strip()
    next_retest_action = str(crypto_route_operator_brief.get("next_retest_action") or crypto_route_brief.get("next_retest_action") or "").strip()
    next_retest_reason = str(crypto_route_operator_brief.get("next_retest_reason") or crypto_route_brief.get("next_retest_reason") or "").strip()
    crypto_route_shortline_market_state_brief = str(
        crypto_route_operator_brief.get("shortline_market_state_brief")
        or crypto_route_brief.get("shortline_market_state_brief")
        or ""
    ).strip()
    crypto_route_shortline_execution_gate_brief = str(
        crypto_route_operator_brief.get("shortline_execution_gate_brief")
        or crypto_route_brief.get("shortline_execution_gate_brief")
        or ""
    ).strip()
    crypto_route_shortline_no_trade_rule = str(
        crypto_route_operator_brief.get("shortline_no_trade_rule")
        or crypto_route_brief.get("shortline_no_trade_rule")
        or ""
    ).strip()
    crypto_route_shortline_session_map_brief = str(
        crypto_route_operator_brief.get("shortline_session_map_brief")
        or crypto_route_brief.get("shortline_session_map_brief")
        or ""
    ).strip()
    crypto_route_shortline_cvd_semantic_status = str(
        crypto_route_operator_brief.get("shortline_cvd_semantic_status")
        or crypto_route_brief.get("shortline_cvd_semantic_status")
        or ""
    ).strip()
    crypto_route_shortline_cvd_semantic_takeaway = str(
        crypto_route_operator_brief.get("shortline_cvd_semantic_takeaway")
        or crypto_route_brief.get("shortline_cvd_semantic_takeaway")
        or ""
    ).strip()
    crypto_route_shortline_cvd_queue_handoff_status = str(
        crypto_route_operator_brief.get("shortline_cvd_queue_handoff_status")
        or crypto_route_brief.get("shortline_cvd_queue_handoff_status")
        or ""
    ).strip()
    crypto_route_shortline_cvd_queue_handoff_takeaway = str(
        crypto_route_operator_brief.get("shortline_cvd_queue_handoff_takeaway")
        or crypto_route_brief.get("shortline_cvd_queue_handoff_takeaway")
        or ""
    ).strip()
    crypto_route_shortline_cvd_queue_focus_batch = str(
        crypto_route_operator_brief.get("shortline_cvd_queue_focus_batch")
        or crypto_route_brief.get("shortline_cvd_queue_focus_batch")
        or ""
    ).strip()
    crypto_route_shortline_cvd_queue_focus_action = str(
        crypto_route_operator_brief.get("shortline_cvd_queue_focus_action")
        or crypto_route_brief.get("shortline_cvd_queue_focus_action")
        or ""
    ).strip()
    crypto_route_shortline_cvd_queue_stack_brief = str(
        crypto_route_operator_brief.get("shortline_cvd_queue_stack_brief")
        or crypto_route_brief.get("shortline_cvd_queue_stack_brief")
        or ""
    ).strip()
    crypto_route_focus_execution_state = str(
        crypto_route_operator_brief.get("focus_execution_state")
        or crypto_route_brief.get("focus_execution_state")
        or ""
    ).strip()
    crypto_route_focus_execution_blocker_detail = str(
        crypto_route_operator_brief.get("focus_execution_blocker_detail")
        or crypto_route_brief.get("focus_execution_blocker_detail")
        or ""
    ).strip()
    crypto_route_focus_execution_done_when = str(
        crypto_route_operator_brief.get("focus_execution_done_when")
        or crypto_route_brief.get("focus_execution_done_when")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_classification = str(
        crypto_route_operator_brief.get("focus_execution_micro_classification")
        or crypto_route_brief.get("focus_execution_micro_classification")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_context = str(
        crypto_route_operator_brief.get("focus_execution_micro_context")
        or crypto_route_brief.get("focus_execution_micro_context")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_trust_tier = str(
        crypto_route_operator_brief.get("focus_execution_micro_trust_tier")
        or crypto_route_brief.get("focus_execution_micro_trust_tier")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_veto = str(
        crypto_route_operator_brief.get("focus_execution_micro_veto")
        or crypto_route_brief.get("focus_execution_micro_veto")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_locality_status = str(
        crypto_route_operator_brief.get("focus_execution_micro_locality_status")
        or crypto_route_brief.get("focus_execution_micro_locality_status")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_drift_risk = str(
        crypto_route_operator_brief.get("focus_execution_micro_drift_risk")
        or crypto_route_brief.get("focus_execution_micro_drift_risk")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_attack_side = str(
        crypto_route_operator_brief.get("focus_execution_micro_attack_side")
        or crypto_route_brief.get("focus_execution_micro_attack_side")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_attack_presence = str(
        crypto_route_operator_brief.get("focus_execution_micro_attack_presence")
        or crypto_route_brief.get("focus_execution_micro_attack_presence")
        or ""
    ).strip()
    crypto_route_focus_execution_micro_reasons = [
        str(x).strip()
        for x in (
            crypto_route_operator_brief.get("focus_execution_micro_reasons")
            or crypto_route_brief.get("focus_execution_micro_reasons")
            or []
        )
        if str(x).strip()
    ]
    crypto_route_focus_review_status = str(
        crypto_route_operator_brief.get("focus_review_status")
        or crypto_route_brief.get("focus_review_status")
        or ""
    ).strip()
    crypto_route_focus_review_brief = str(
        crypto_route_operator_brief.get("focus_review_brief")
        or crypto_route_brief.get("focus_review_brief")
        or ""
    ).strip()
    crypto_route_focus_review_primary_blocker = str(
        crypto_route_operator_brief.get("focus_review_primary_blocker")
        or crypto_route_brief.get("focus_review_primary_blocker")
        or ""
    ).strip()
    crypto_route_focus_review_micro_blocker = str(
        crypto_route_operator_brief.get("focus_review_micro_blocker")
        or crypto_route_brief.get("focus_review_micro_blocker")
        or ""
    ).strip()
    crypto_route_focus_review_blocker_detail = str(
        crypto_route_operator_brief.get("focus_review_blocker_detail")
        or crypto_route_brief.get("focus_review_blocker_detail")
        or ""
    ).strip()
    crypto_route_focus_review_done_when = str(
        crypto_route_operator_brief.get("focus_review_done_when")
        or crypto_route_brief.get("focus_review_done_when")
        or ""
    ).strip()
    crypto_route_focus_review_score_status = str(
        crypto_route_operator_brief.get("focus_review_score_status")
        or crypto_route_brief.get("focus_review_score_status")
        or ""
    ).strip()
    crypto_route_focus_review_edge_score = int(
        crypto_route_operator_brief.get("focus_review_edge_score")
        or crypto_route_brief.get("focus_review_edge_score")
        or 0
    )
    crypto_route_focus_review_structure_score = int(
        crypto_route_operator_brief.get("focus_review_structure_score")
        or crypto_route_brief.get("focus_review_structure_score")
        or 0
    )
    crypto_route_focus_review_micro_score = int(
        crypto_route_operator_brief.get("focus_review_micro_score")
        or crypto_route_brief.get("focus_review_micro_score")
        or 0
    )
    crypto_route_focus_review_composite_score = int(
        crypto_route_operator_brief.get("focus_review_composite_score")
        or crypto_route_brief.get("focus_review_composite_score")
        or 0
    )
    crypto_route_focus_review_score_brief = str(
        crypto_route_operator_brief.get("focus_review_score_brief")
        or crypto_route_brief.get("focus_review_score_brief")
        or ""
    ).strip()
    crypto_route_focus_review_priority_status = str(
        crypto_route_operator_brief.get("focus_review_priority_status")
        or crypto_route_brief.get("focus_review_priority_status")
        or ""
    ).strip()
    crypto_route_focus_review_priority_score = int(
        crypto_route_operator_brief.get("focus_review_priority_score")
        or crypto_route_brief.get("focus_review_priority_score")
        or 0
    )
    crypto_route_focus_review_priority_tier = str(
        crypto_route_operator_brief.get("focus_review_priority_tier")
        or crypto_route_brief.get("focus_review_priority_tier")
        or ""
    ).strip()
    crypto_route_focus_review_priority_brief = str(
        crypto_route_operator_brief.get("focus_review_priority_brief")
        or crypto_route_brief.get("focus_review_priority_brief")
        or ""
    ).strip()
    crypto_route_review_priority_queue_status = str(
        crypto_route_operator_brief.get("review_priority_queue_status")
        or crypto_route_brief.get("review_priority_queue_status")
        or ""
    ).strip()
    crypto_route_review_priority_queue_count = int(
        crypto_route_operator_brief.get("review_priority_queue_count")
        or crypto_route_brief.get("review_priority_queue_count")
        or 0
    )
    crypto_route_review_priority_queue_brief = str(
        crypto_route_operator_brief.get("review_priority_queue_brief")
        or crypto_route_brief.get("review_priority_queue_brief")
        or ""
    ).strip()
    crypto_route_review_priority_head_symbol = str(
        crypto_route_operator_brief.get("review_priority_head_symbol")
        or crypto_route_brief.get("review_priority_head_symbol")
        or ""
    ).strip()
    crypto_route_review_priority_head_tier = str(
        crypto_route_operator_brief.get("review_priority_head_tier")
        or crypto_route_brief.get("review_priority_head_tier")
        or ""
    ).strip()
    crypto_route_review_priority_head_score = int(
        crypto_route_operator_brief.get("review_priority_head_score")
        or crypto_route_brief.get("review_priority_head_score")
        or 0
    )
    crypto_route_review_priority_queue = [
        dict(row)
        for row in (
            crypto_route_operator_brief.get("review_priority_queue")
            or crypto_route_brief.get("review_priority_queue")
            or []
        )
        if isinstance(row, dict)
    ]
    crypto_route_review_priority_head_row = _review_priority_head_row(
        crypto_route_review_priority_queue,
        head_symbol=crypto_route_review_priority_head_symbol,
    )

    operator_status = "focus-commodities-plus-crypto-watch"
    if not focus_primary and not focus_regime:
        operator_status = "research-queue-only"
    if research_queue and operator_status == "research-queue-only":
        operator_status = "research-queue-plus-crypto-watch"
    if crypto_status.startswith("deploy"):
        operator_status = "focus-commodities-plus-crypto-deploy-watch"
        if not focus_primary and not focus_regime:
            operator_status = "research-queue-plus-crypto-deploy-watch"
    if commodity_status:
        operator_status = "commodity-paper-first"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-first-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-first-plus-research-queue"
    if commodity_ticket_status == "paper-ready" or commodity_ticket_book_status == "paper-ready":
        operator_status = "commodity-paper-ticket-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-ticket-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-ticket-plus-research-queue"
    if commodity_execution_preview_status == "paper-execution-ready":
        operator_status = "commodity-paper-execution-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-plus-research-queue"
    if commodity_execution_artifact_status == "paper-execution-artifact-ready":
        operator_status = "commodity-paper-execution-artifact-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-artifact-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-artifact-plus-research-queue"
    if commodity_execution_queue_status == "paper-execution-queued":
        operator_status = "commodity-paper-execution-queued"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-queued-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-queued-plus-research-queue"
    if commodity_execution_review_status in pending_review_statuses:
        operator_status = "commodity-paper-execution-review-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-review-pending-plus-research-queue"
    if commodity_close_evidence_pending_count > 0:
        operator_status = "commodity-paper-execution-close-evidence-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-research-queue"
    if commodity_execution_retro_status in pending_retro_statuses:
        operator_status = "commodity-paper-execution-retro-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-retro-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-retro-pending-plus-research-queue"
    if commodity_execution_gap_status == "blocking_gap_active":
        operator_status = "commodity-paper-execution-gap-blocked"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-gap-blocked-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-gap-blocked-plus-research-queue"
    if commodity_execution_bridge_status in bridge_blocked_statuses:
        operator_status = "commodity-paper-execution-bridge-blocked"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-bridge-blocked-plus-research-queue"
    elif commodity_execution_bridge_status == "bridge_ready":
        operator_status = "commodity-paper-execution-bridge-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-bridge-ready-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-bridge-ready-plus-research-queue"
    if not bridge_ready and actionable_review_pending:
        operator_status = "commodity-paper-execution-review-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-review-pending-plus-research-queue"
    if not bridge_ready and actionable_retro_pending:
        operator_status = "commodity-paper-execution-retro-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-retro-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-retro-pending-plus-research-queue"
    if not bridge_ready and commodity_close_evidence_pending_count > 0:
        operator_status = "commodity-paper-execution-close-evidence-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-research-queue"

    commodity_route_brief = f"{commodity_status}:{commodity_focus_batch}" if commodity_focus_batch else (commodity_status or "-")
    commodity_ticket_brief = (
        f"{commodity_ticket_status}:{commodity_ticket_focus_batch}"
        if commodity_ticket_focus_batch
        else (commodity_ticket_status or "-")
    )
    commodity_execution_brief = "-"
    if commodity_next_close_evidence_execution_symbol:
        commodity_execution_brief = f"close-evidence:{commodity_next_close_evidence_execution_symbol}"
    elif commodity_next_retro_execution_symbol:
        commodity_execution_brief = f"retro:{commodity_next_retro_execution_symbol}"
    elif commodity_next_review_execution_symbol:
        commodity_execution_brief = f"review:{commodity_next_review_execution_symbol}"
    elif commodity_next_queue_execution_symbol:
        commodity_execution_brief = f"queue:{commodity_next_queue_execution_symbol}"
    elif commodity_execution_symbols:
        commodity_execution_brief = f"execution:{commodity_execution_symbols[0]}"
    elif commodity_next_execution_batch:
        commodity_execution_brief = f"preview:{commodity_next_execution_batch}"
    elif commodity_next_ticket_symbol:
        commodity_execution_brief = f"ticket:{commodity_next_ticket_symbol}"
    if commodity_execution_gap_status == "blocking_gap_active":
        if commodity_gap_focus_symbol:
            commodity_execution_brief = f"gap:{commodity_gap_focus_symbol}"
        elif commodity_gap_focus_batch:
            commodity_execution_brief = f"gap:{commodity_gap_focus_batch}"
        else:
            commodity_execution_brief = "gap"
    if commodity_execution_bridge_status in {
        "blocked_proxy_price_reference_only",
        "bridge_partially_bridged_proxy_remainder",
    }:
        if commodity_execution_bridge_next_blocked_symbol:
            commodity_execution_brief = f"bridge-proxy:{commodity_execution_bridge_next_blocked_symbol}"
        else:
            commodity_execution_brief = "bridge-proxy"
    elif commodity_execution_bridge_status in {
        "blocked_stale_directional_signal",
        "bridge_partially_bridged_stale_remainder",
    }:
        if commodity_execution_bridge_next_blocked_symbol:
            commodity_execution_brief = f"bridge-stale:{commodity_execution_bridge_next_blocked_symbol}"
        else:
            commodity_execution_brief = "bridge-stale"
    elif commodity_execution_bridge_status in {
        "blocked_missing_directional_signal",
        "bridge_partially_bridged_missing_remainder",
    }:
        if commodity_execution_bridge_next_blocked_symbol:
            commodity_execution_brief = f"bridge-blocked:{commodity_execution_bridge_next_blocked_symbol}"
        else:
            commodity_execution_brief = "bridge-blocked"
    elif commodity_execution_bridge_status == "bridge_ready":
        if commodity_execution_bridge_next_ready_symbol:
            commodity_execution_brief = f"bridge-ready:{commodity_execution_bridge_next_ready_symbol}"
        else:
            commodity_execution_brief = "bridge-ready"
    if not bridge_ready and actionable_review_pending:
        commodity_execution_brief = (
            f"review:{commodity_next_review_execution_symbol}" if commodity_next_review_execution_symbol else "review"
        )
    if not bridge_ready and commodity_close_evidence_pending_count > 0:
        commodity_execution_brief = (
            f"close-evidence:{commodity_next_close_evidence_execution_symbol}"
            if commodity_next_close_evidence_execution_symbol
            else "close-evidence"
        )
    if not bridge_ready and actionable_retro_pending:
        commodity_execution_brief = (
            f"retro:{commodity_next_retro_execution_symbol}" if commodity_next_retro_execution_symbol else "retro"
        )
    if not bridge_ready and commodity_close_evidence_pending_count > 0:
        commodity_execution_brief = (
            f"close-evidence:{commodity_next_close_evidence_execution_symbol}"
            if commodity_next_close_evidence_execution_symbol
            else "close-evidence"
        )

    next_focus_area = "-"
    next_focus_target = "-"
    next_focus_symbol = "-"
    next_focus_action = "-"
    next_focus_reason = "-"
    secondary_focus_area = "-"
    secondary_focus_target = "-"
    secondary_focus_symbol = "-"
    secondary_focus_action = "-"
    secondary_focus_reason = "-"
    secondary_focus_priority_tier = "-"
    secondary_focus_priority_score: int | str = "-"
    secondary_focus_queue_rank: int | str = "-"
    commodity_remainder_focus_area = "-"
    commodity_remainder_focus_target = "-"
    commodity_remainder_focus_symbol = "-"
    commodity_remainder_focus_action = "-"
    commodity_remainder_focus_reason = "-"
    commodity_remainder_focus_signal_date = "-"
    commodity_remainder_focus_signal_age_days: int | str = "-"
    commodity_stale_signal_watch_brief = _watch_items_text(commodity_stale_signal_watch_items)
    commodity_stale_signal_watch_next_execution_id = "-"
    commodity_stale_signal_watch_next_symbol = "-"
    commodity_stale_signal_watch_next_signal_date = "-"
    commodity_stale_signal_watch_next_signal_age_days: int | str = "-"
    followup_focus_area = "-"
    followup_focus_target = "-"
    followup_focus_symbol = "-"
    followup_focus_action = "-"
    followup_focus_reason = "-"
    crypto_secondary_target = focus_symbol or "-"
    crypto_secondary_symbol = focus_symbol or "-"
    crypto_secondary_action = focus_action or "-"
    crypto_secondary_reason = focus_reason or "secondary_focus"
    crypto_secondary_blocker_override = ""
    crypto_secondary_done_when_override = ""
    crypto_route_review_priority_head_action = ""
    crypto_route_review_priority_head_reason = ""
    crypto_route_review_priority_head_blocker_detail = ""
    crypto_route_review_priority_head_done_when = ""
    crypto_route_review_priority_head_rank: int | str = "-"
    if crypto_route_review_priority_head_row:
        crypto_secondary_symbol = (
            str(crypto_route_review_priority_head_row.get("symbol") or "").strip().upper()
            or crypto_secondary_symbol
        )
        crypto_secondary_target = crypto_secondary_symbol or "-"
        crypto_secondary_action = (
            str(crypto_route_review_priority_head_row.get("route_action") or "").strip()
            or crypto_secondary_action
        )
        crypto_secondary_reason = (
            str(crypto_route_review_priority_head_row.get("reason") or "").strip()
            or crypto_secondary_reason
        )
        crypto_secondary_blocker_override = str(
            crypto_route_review_priority_head_row.get("blocker_detail") or ""
        ).strip()
        crypto_secondary_done_when_override = str(
            crypto_route_review_priority_head_row.get("done_when") or ""
        ).strip()
        crypto_route_review_priority_head_action = crypto_secondary_action
        crypto_route_review_priority_head_reason = crypto_secondary_reason
        crypto_route_review_priority_head_blocker_detail = crypto_secondary_blocker_override
        crypto_route_review_priority_head_done_when = crypto_secondary_done_when_override
        secondary_focus_priority_tier = (
            str(crypto_route_review_priority_head_row.get("priority_tier") or "").strip()
            or crypto_route_review_priority_head_tier
            or "-"
        )
        secondary_focus_priority_score = int(
            crypto_route_review_priority_head_row.get("priority_score")
            or crypto_route_review_priority_head_score
            or 0
        )
        secondary_focus_queue_rank = (
            crypto_route_review_priority_head_row.get("rank")
            if crypto_route_review_priority_head_row.get("rank") not in (None, "")
            else "-"
        )
    elif crypto_route_review_priority_head_tier:
        secondary_focus_priority_tier = crypto_route_review_priority_head_tier
        secondary_focus_priority_score = crypto_route_review_priority_head_score
        secondary_focus_queue_rank = 1 if crypto_route_review_priority_head_symbol else "-"
    if crypto_route_review_priority_head_symbol and secondary_focus_queue_rank not in (None, ""):
        crypto_route_review_priority_head_rank = secondary_focus_queue_rank
    crypto_route_short_brief_symbol = (
        crypto_secondary_symbol if crypto_secondary_symbol and crypto_secondary_symbol != "-" else focus_symbol
    )
    crypto_route_short_brief_action = (
        crypto_secondary_action if crypto_route_short_brief_symbol == crypto_secondary_symbol else (focus_action or "-")
    )
    crypto_route_short_brief = (
        f"{crypto_route_short_brief_symbol}:{crypto_route_short_brief_action}"
        if crypto_route_short_brief_symbol
        else (route_stack or "-")
    )
    if commodity_stale_signal_watch_items:
        commodity_stale_watch_head = dict(commodity_stale_signal_watch_items[0])
        commodity_stale_signal_watch_next_execution_id = str(
            commodity_stale_watch_head.get("execution_id") or ""
        ).strip() or "-"
        commodity_stale_signal_watch_next_symbol = str(
            commodity_stale_watch_head.get("symbol") or ""
        ).strip().upper() or "-"
        commodity_stale_signal_watch_next_signal_date = str(
            commodity_stale_watch_head.get("signal_date") or ""
        ).strip() or "-"
        commodity_stale_signal_watch_next_signal_age_days = (
            commodity_stale_watch_head.get("signal_age_days")
            if commodity_stale_watch_head.get("signal_age_days") not in (None, "")
            else "-"
        )

    if commodity_execution_bridge_status == "bridge_ready":
        next_focus_area = "commodity_execution_bridge"
        next_focus_target = commodity_execution_bridge_next_ready_id or commodity_gap_focus_execution_id or "-"
        next_focus_symbol = commodity_execution_bridge_next_ready_symbol or commodity_gap_focus_symbol or "-"
        next_focus_action = "apply_commodity_execution_bridge"
        next_focus_reason = "commodity_bridge_ready"
    elif commodity_close_evidence_pending_count > 0:
        next_focus_area = "commodity_execution_close_evidence"
        next_focus_target = (
            commodity_next_close_evidence_execution_id
            or commodity_next_retro_execution_id
            or commodity_next_close_evidence_execution_symbol
            or "-"
        )
        next_focus_symbol = (
            commodity_next_close_evidence_execution_symbol
            or commodity_next_retro_execution_symbol
            or "-"
        )
        next_focus_action = "wait_for_paper_execution_close_evidence"
        next_focus_reason = "paper_execution_close_evidence_pending"
    elif actionable_retro_pending:
        next_focus_area = "commodity_execution_retro"
        next_focus_target = commodity_next_retro_execution_id or commodity_next_retro_execution_symbol or "-"
        next_focus_symbol = commodity_next_retro_execution_symbol or "-"
        next_focus_action = "review_paper_execution_retro"
        next_focus_reason = "paper_execution_retro_pending"
    elif actionable_review_pending:
        next_focus_area = "commodity_execution_review"
        next_focus_target = commodity_next_review_execution_id or commodity_next_review_execution_symbol or "-"
        next_focus_symbol = commodity_next_review_execution_symbol or "-"
        next_focus_action = "review_paper_execution"
        next_focus_reason = "paper_execution_review_pending"
    elif commodity_execution_bridge_status in bridge_blocked_statuses:
        next_focus_area = "commodity_execution_bridge"
        next_focus_target = commodity_execution_bridge_next_blocked_id or commodity_gap_focus_execution_id or "-"
        next_focus_symbol = commodity_execution_bridge_next_blocked_symbol or commodity_gap_focus_symbol or "-"
        if commodity_execution_bridge_status in {
            "blocked_proxy_price_reference_only",
            "bridge_partially_bridged_proxy_remainder",
        }:
            next_focus_action = "normalize_commodity_execution_price_reference"
            next_focus_reason = "commodity_bridge_blocked_proxy_price_reference_only"
        else:
            next_focus_action = "restore_commodity_directional_signal"
        if commodity_execution_bridge_status in {
            "blocked_stale_directional_signal",
            "bridge_partially_bridged_stale_remainder",
        }:
            next_focus_reason = "commodity_bridge_blocked_stale_directional_signal"
        elif commodity_execution_bridge_status in {
            "blocked_missing_directional_signal",
            "bridge_partially_bridged_missing_remainder",
        }:
            next_focus_reason = "commodity_bridge_blocked_missing_directional_signal"
    elif commodity_execution_gap_status == "blocking_gap_active":
        next_focus_area = "commodity_execution_gap"
        next_focus_target = commodity_gap_focus_execution_id or commodity_gap_focus_batch or "-"
        next_focus_symbol = commodity_gap_focus_symbol or "-"
        next_focus_action = "resolve_commodity_paper_execution_gap"
        next_focus_reason = "commodity_execution_gap_active"
    elif commodity_execution_queue_status == "paper-execution-queued":
        next_focus_area = "commodity_execution_queue"
        next_focus_target = commodity_next_queue_execution_id or commodity_next_queue_execution_symbol or "-"
        next_focus_symbol = commodity_next_queue_execution_symbol or "-"
        next_focus_action = "inspect_paper_execution_queue"
        next_focus_reason = "paper_execution_queued"
    elif commodity_execution_artifact_status == "paper-execution-artifact-ready":
        next_focus_area = "commodity_execution_artifact"
        next_focus_target = commodity_execution_batch or "-"
        next_focus_symbol = commodity_execution_symbols[0] if commodity_execution_symbols else "-"
        next_focus_action = "inspect_paper_execution_artifact"
        next_focus_reason = "paper_execution_artifact_ready"
    elif commodity_execution_preview_status == "paper-execution-ready":
        next_focus_area = "commodity_execution_preview"
        next_focus_target = commodity_next_execution_batch or "-"
        next_focus_symbol = commodity_next_execution_symbols[0] if commodity_next_execution_symbols else "-"
        next_focus_action = "prepare_paper_execution"
        next_focus_reason = "paper_execution_ready"
    elif commodity_ticket_book_status == "paper-ready":
        next_focus_area = "commodity_ticket_book"
        next_focus_target = commodity_next_ticket_id or commodity_ticket_focus_batch or "-"
        next_focus_symbol = commodity_next_ticket_symbol or "-"
        next_focus_action = "build_paper_execution_artifact"
        next_focus_reason = "paper_ticket_ready"
    elif commodity_status:
        next_focus_area = "commodity_route"
        next_focus_target = commodity_focus_batch or "-"
        next_focus_symbol = commodity_focus_symbols[0] if commodity_focus_symbols else "-"
        next_focus_action = "build_paper_ticket_lane"
        next_focus_reason = commodity_status
    elif focus_symbol:
        next_focus_area = "crypto_route"
        next_focus_target = focus_symbol
        next_focus_symbol = focus_symbol
        next_focus_action = focus_action or "-"
        next_focus_reason = focus_reason or crypto_status or "-"
    elif research_queue:
        next_focus_area = "research_queue"
        next_focus_target = research_queue[0]
        next_focus_action = "continue_research_queue"
        next_focus_reason = "research_queue_active"

    if next_focus_area.startswith("commodity_") and crypto_secondary_symbol and crypto_secondary_symbol != "-":
        secondary_focus_area = "crypto_route"
        secondary_focus_target = crypto_secondary_target
        secondary_focus_symbol = crypto_secondary_symbol
        secondary_focus_action = crypto_secondary_action
        secondary_focus_reason = crypto_secondary_reason
    elif next_focus_area == "crypto_route" and research_queue:
        secondary_focus_area = "research_queue"
        secondary_focus_target = research_queue[0]
        secondary_focus_symbol = research_queue[0]
        secondary_focus_action = "continue_research_queue"
        secondary_focus_reason = "secondary_focus"
    elif next_focus_area == "research_queue" and crypto_secondary_symbol and crypto_secondary_symbol != "-":
        secondary_focus_area = "crypto_route"
        secondary_focus_target = crypto_secondary_target
        secondary_focus_symbol = crypto_secondary_symbol
        secondary_focus_action = crypto_secondary_action
        secondary_focus_reason = crypto_secondary_reason

    if commodity_next_fill_evidence_execution_id or commodity_next_fill_evidence_execution_symbol:
        commodity_remainder_focus_area = "commodity_fill_evidence"
        commodity_remainder_focus_target = (
            commodity_next_fill_evidence_execution_id or commodity_next_fill_evidence_execution_symbol or "-"
        )
        commodity_remainder_focus_symbol = commodity_next_fill_evidence_execution_symbol or "-"
        commodity_remainder_focus_action = "wait_for_paper_execution_fill_evidence"
        commodity_remainder_focus_reason = "paper_execution_fill_evidence_pending"
    elif commodity_execution_bridge_status in bridge_blocked_statuses:
        commodity_remainder_focus_area = "commodity_execution_bridge"
        commodity_remainder_focus_target = commodity_execution_bridge_next_blocked_id or commodity_gap_focus_execution_id or "-"
        commodity_remainder_focus_symbol = commodity_execution_bridge_next_blocked_symbol or commodity_gap_focus_symbol or "-"
        if commodity_execution_bridge_status in {
            "blocked_proxy_price_reference_only",
            "bridge_partially_bridged_proxy_remainder",
        }:
            commodity_remainder_focus_action = "normalize_commodity_execution_price_reference"
            commodity_remainder_focus_reason = "commodity_bridge_blocked_proxy_price_reference_only"
        else:
            commodity_remainder_focus_action = "restore_commodity_directional_signal"
            commodity_remainder_focus_reason = (
                "commodity_bridge_blocked_stale_directional_signal"
                if commodity_execution_bridge_status in {
                    "blocked_stale_directional_signal",
                    "bridge_partially_bridged_stale_remainder",
                }
                else "commodity_bridge_blocked_missing_directional_signal"
            )
    if commodity_remainder_focus_symbol and commodity_remainder_focus_symbol != "-":
        commodity_remainder_focus_signal_date = (
            commodity_execution_bridge_stale_signal_dates.get(commodity_remainder_focus_symbol) or "-"
        )
        commodity_remainder_focus_signal_age_days = (
            commodity_execution_bridge_stale_signal_age_days.get(commodity_remainder_focus_symbol, "-")
        )

    checklist_research_embedding_quality = _research_embedding_quality(action_payload)
    checklist_alignment_focus = _crypto_route_alignment_focus(
        {
            "next_focus_area": next_focus_area,
            "next_focus_symbol": next_focus_symbol,
            "next_focus_action": next_focus_action,
            "followup_focus_area": followup_focus_area,
            "followup_focus_symbol": followup_focus_symbol,
            "followup_focus_action": followup_focus_action,
            "secondary_focus_area": secondary_focus_area,
            "secondary_focus_symbol": secondary_focus_symbol,
            "secondary_focus_action": secondary_focus_action,
            "crypto_route_review_priority_head_symbol": crypto_route_review_priority_head_symbol,
            "crypto_route_review_priority_head_action": crypto_route_review_priority_head_action,
        }
    )
    checklist_crypto_route_alignment = _crypto_route_embedding_alignment(
        secondary_focus_area=str(checklist_alignment_focus.get("area") or ""),
        secondary_focus_symbol=str(checklist_alignment_focus.get("symbol") or ""),
        secondary_focus_action=str(checklist_alignment_focus.get("action") or ""),
        quality_brief=str(checklist_research_embedding_quality.get("brief") or ""),
        active_batches=list(checklist_research_embedding_quality.get("active_batches") or []),
    )
    checklist_crypto_route_alignment_recovery = _crypto_route_alignment_recovery_outcome(
        alignment_status=str(checklist_crypto_route_alignment.get("status") or ""),
        quality_status=str(checklist_research_embedding_quality.get("status") or ""),
        source_status=str(crypto_payload.get("status") or ""),
        source_payload=crypto_payload if isinstance(crypto_payload, dict) else {},
    )
    if crypto_route_focus_review_status:
        crypto_route_focus_review_lane = {
            "status": crypto_route_focus_review_status,
            "brief": crypto_route_focus_review_brief or f"{crypto_route_focus_review_status}:{focus_symbol or '-'}",
            "primary_blocker": crypto_route_focus_review_primary_blocker or "review_pending",
            "micro_blocker": crypto_route_focus_review_micro_blocker or "-",
            "blocker_detail": crypto_route_focus_review_blocker_detail
            or crypto_route_focus_execution_blocker_detail
            or "crypto review lane is active.",
            "done_when": crypto_route_focus_review_done_when
            or crypto_route_focus_execution_done_when
            or f"{focus_symbol or 'crypto focus'} leaves review",
        }
    else:
        crypto_route_focus_review_lane = _crypto_route_focus_review_lane(
            symbol=focus_symbol,
            action=focus_action,
            focus_execution_state=crypto_route_focus_execution_state,
            focus_execution_blocker_detail=crypto_route_focus_execution_blocker_detail,
            focus_execution_done_when=crypto_route_focus_execution_done_when,
            focus_execution_micro_veto=crypto_route_focus_execution_micro_veto,
            alignment_status=str(checklist_crypto_route_alignment.get("status") or ""),
            alignment_recovery_status=str(checklist_crypto_route_alignment_recovery.get("status") or ""),
        )
    if crypto_route_focus_review_score_status:
        crypto_route_focus_review_scores = {
            "status": crypto_route_focus_review_score_status,
            "edge_score": crypto_route_focus_review_edge_score,
            "structure_score": crypto_route_focus_review_structure_score,
            "micro_score": crypto_route_focus_review_micro_score,
            "composite_score": crypto_route_focus_review_composite_score,
            "brief": crypto_route_focus_review_score_brief
            or (
                f"scored:{focus_symbol or '-'}:edge={crypto_route_focus_review_edge_score}"
                f"|structure={crypto_route_focus_review_structure_score}"
                f"|micro={crypto_route_focus_review_micro_score}"
                f"|composite={crypto_route_focus_review_composite_score}"
            ),
        }
    else:
        crypto_route_focus_review_scores = _crypto_route_focus_review_scores(
            symbol=focus_symbol,
            action=focus_action,
            focus_execution_state=crypto_route_focus_execution_state,
            focus_execution_micro_classification=crypto_route_focus_execution_micro_classification,
            focus_execution_micro_veto=crypto_route_focus_execution_micro_veto,
            review_primary_blocker=str(crypto_route_focus_review_lane.get("primary_blocker") or ""),
        )
    if crypto_route_focus_review_priority_status:
        crypto_route_focus_review_priority = {
            "status": crypto_route_focus_review_priority_status,
            "score": crypto_route_focus_review_priority_score,
            "tier": crypto_route_focus_review_priority_tier,
            "brief": crypto_route_focus_review_priority_brief
            or (
                f"{crypto_route_focus_review_priority_tier or '-'}:"
                f"{crypto_route_focus_review_priority_score}/100"
            ),
        }
    else:
        crypto_route_focus_review_priority = _crypto_route_focus_review_priority(
            crypto_route_focus_review_scores
        )

    action_queue: list[dict[str, Any]] = []
    operator_action_queue: list[dict[str, Any]] = []
    operator_action_queue_brief = "-"

    commodity_focus_evidence_item_source = "-"
    commodity_focus_evidence_summary: dict[str, Any] = {}
    if next_focus_area in {"commodity_execution_retro", "commodity_execution_close_evidence"}:
        focus_item = _find_execution_item(
            commodity_execution_retro_items,
            execution_id=next_focus_target,
            symbol=next_focus_symbol,
        )
        if focus_item:
            commodity_focus_evidence_item_source = "retro"
            commodity_focus_evidence_summary = _extract_paper_evidence_summary(focus_item)
    elif next_focus_area == "commodity_execution_review":
        focus_item = _find_execution_item(
            commodity_execution_review_items,
            execution_id=next_focus_target,
            symbol=next_focus_symbol,
        )
        if focus_item:
            commodity_focus_evidence_item_source = "review"
            commodity_focus_evidence_summary = _extract_paper_evidence_summary(focus_item)
    commodity_focus_lifecycle = _commodity_focus_lifecycle_gate(
        area=next_focus_area,
        action=next_focus_action,
        symbol=next_focus_symbol,
        focus_evidence_summary=commodity_focus_evidence_summary,
    )
    if (
        next_focus_action == "review_paper_execution_retro"
        and str(commodity_focus_lifecycle.get("status") or "").strip() == "open_position_wait_close_evidence"
    ):
        next_focus_area = "commodity_execution_close_evidence"
        next_focus_action = "wait_for_paper_execution_close_evidence"
        next_focus_reason = "paper_execution_close_evidence_pending"
        commodity_execution_brief = (
            f"close-evidence:{next_focus_symbol}" if next_focus_symbol and next_focus_symbol != "-" else "close-evidence"
        )
        operator_status = "commodity-paper-execution-close-evidence-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-research-queue"
        commodity_focus_lifecycle = _commodity_focus_lifecycle_gate(
            area=next_focus_area,
            action=next_focus_action,
            symbol=next_focus_symbol,
            focus_evidence_summary=commodity_focus_evidence_summary,
        )

    cross_market_review_head_lane = _cross_market_review_head_lane(
        source_payload=cross_market_operator_state_payload,
    )
    cross_market_operator_head_lane = _cross_market_operator_head_lane(
        source_payload=cross_market_operator_state_payload,
    )
    cross_market_operator_repair_head_lane = _cross_market_operator_repair_head_lane(
        source_payload=cross_market_operator_state_payload,
        live_gate_blocker_source_payload=live_gate_blocker_payload,
    )
    source_operator_repair_queue = [
        dict(row)
        for row in (cross_market_operator_state_payload or {}).get("operator_repair_queue", [])
        if isinstance(row, dict)
    ]
    source_operator_action_queue = [
        dict(row)
        for row in (cross_market_operator_state_payload or {}).get("operator_action_queue", [])
        if isinstance(row, dict)
    ]
    source_operator_action_queue_brief = str(
        (cross_market_operator_state_payload or {}).get("operator_action_queue_brief") or ""
    ).strip()
    source_operator_action_checklist = [
        dict(row)
        for row in (cross_market_operator_state_payload or {}).get("operator_action_checklist", [])
        if isinstance(row, dict)
    ]
    source_operator_action_checklist_brief = str(
        (cross_market_operator_state_payload or {}).get("operator_action_checklist_brief") or ""
    ).strip()
    source_operator_focus_slots = [
        dict(row)
        for row in (cross_market_operator_state_payload or {}).get("operator_focus_slots", [])
        if isinstance(row, dict)
    ]
    source_operator_focus_slots_brief = str(
        (cross_market_operator_state_payload or {}).get("operator_focus_slots_brief") or ""
    ).strip()
    source_operator_repair_queue_brief = str(
        (cross_market_operator_state_payload or {}).get("operator_repair_queue_brief") or ""
    ).strip()
    source_operator_repair_checklist = [
        dict(row)
        for row in (cross_market_operator_state_payload or {}).get("operator_repair_checklist", [])
        if isinstance(row, dict)
    ]
    source_operator_repair_checklist_brief = str(
        (cross_market_operator_state_payload or {}).get("operator_repair_checklist_brief") or ""
    ).strip()

    followup_focus_area = "-"
    followup_focus_target = "-"
    followup_focus_symbol = "-"
    followup_focus_action = "-"
    followup_focus_reason = "-"
    repair_queue_payload = dict(
        (live_gate_blocker_payload or {}).get("remote_live_takeover_repair_queue") or {}
    )
    repair_head_area = str(repair_queue_payload.get("head_area") or "").strip()
    repair_head_code = str(repair_queue_payload.get("head_code") or "").strip()
    repair_head_action = str(repair_queue_payload.get("head_action") or "").strip()
    repair_head_clear_when = str(repair_queue_payload.get("head_clear_when") or "").strip()
    repair_done_when = str(repair_queue_payload.get("done_when") or "").strip()
    repair_gate_blocker_detail = str(
        (cross_market_operator_state_payload or {}).get("remote_live_takeover_gate_blocker_detail") or ""
    ).strip()
    repair_clearing_done_when = str(
        (cross_market_operator_state_payload or {}).get("remote_live_takeover_clearing_done_when") or ""
    ).strip()

    if source_operator_action_queue:
        operator_action_queue = [
            {
                **row,
                "rank": idx,
            }
            for idx, row in enumerate(source_operator_action_queue, start=1)
        ]
        operator_action_queue_brief = source_operator_action_queue_brief or _action_queue_brief(
            operator_action_queue
        )
        if source_operator_action_checklist:
            operator_action_checklist = [
                {
                    **row,
                    "rank": idx,
                }
                for idx, row in enumerate(source_operator_action_checklist, start=1)
            ]
        else:
            operator_action_checklist = [
                {
                    **row,
                    "rank": idx,
                    "state": _action_checklist_state(
                        area=str(row.get("area") or ""),
                        action=str(row.get("action") or ""),
                        rank=idx,
                    ),
                    "blocker_detail": str(row.get("blocker_detail") or "").strip(),
                    "done_when": str(row.get("done_when") or "").strip(),
                }
                for idx, row in enumerate(operator_action_queue, start=1)
            ]
        operator_action_checklist_brief = source_operator_action_checklist_brief or _action_checklist_brief(
            operator_action_checklist
        )
    else:
        action_queue = []
        action_queue_detail_overrides: dict[tuple[str, str, str], dict[str, Any]] = {}
        _append_action_queue_item(
            action_queue,
            area=next_focus_area,
            target=next_focus_target,
            symbol=next_focus_symbol,
            action=next_focus_action,
            reason=next_focus_reason,
        )
        _append_action_queue_item(
            action_queue,
            area=commodity_remainder_focus_area,
            target=commodity_remainder_focus_target,
            symbol=commodity_remainder_focus_symbol,
            action=commodity_remainder_focus_action,
            reason=commodity_remainder_focus_reason,
        )
        review_head_area = str(cross_market_review_head_lane.get("head", {}).get("area") or "").strip()
        review_head_symbol = str(cross_market_review_head_lane.get("head", {}).get("symbol") or "").strip().upper()
        review_head_action = str(cross_market_review_head_lane.get("head", {}).get("action") or "").strip()
        review_head_target = str(cross_market_review_head_lane.get("target") or review_head_symbol or "").strip()
        review_head_reason = str(cross_market_review_head_lane.get("reason") or "").strip() or secondary_focus_reason
        if (
            review_head_area
            and review_head_symbol
            and review_head_action
            and review_head_area == str(secondary_focus_area or "").strip()
            and review_head_symbol == str(secondary_focus_symbol or "").strip().upper()
            and review_head_action == str(secondary_focus_action or "").strip()
            and str(secondary_focus_reason or "").strip()
            and str(secondary_focus_reason or "").strip() != "-"
        ):
            review_head_reason = str(secondary_focus_reason or "").strip()
        action_queue_review_area = review_head_area or secondary_focus_area
        action_queue_review_target = review_head_target or secondary_focus_target
        action_queue_review_symbol = review_head_symbol or secondary_focus_symbol
        action_queue_review_action = review_head_action or secondary_focus_action
        action_queue_review_reason = review_head_reason or secondary_focus_reason
        _append_action_queue_item(
            action_queue,
            area=action_queue_review_area,
            target=action_queue_review_target,
            symbol=action_queue_review_symbol,
            action=action_queue_review_action,
            reason=action_queue_review_reason,
        )
        if (
            action_queue_review_area != "-"
            and action_queue_review_target != "-"
            and action_queue_review_action != "-"
            and str(cross_market_review_head_lane.get("status") or "").strip() not in {"", "inactive"}
        ):
            action_queue_detail_overrides[
                (action_queue_review_area, action_queue_review_target, action_queue_review_action)
            ] = {
                "blocker_detail": str(cross_market_review_head_lane.get("blocker_detail") or "").strip(),
                "done_when": str(cross_market_review_head_lane.get("done_when") or "").strip(),
            }
        if secondary_focus_area == "crypto_route" and secondary_focus_target != "-" and secondary_focus_action != "-":
            action_queue_detail_overrides[
                (secondary_focus_area, secondary_focus_target, secondary_focus_action)
            ] = {
                "blocker_detail": crypto_secondary_blocker_override,
                "done_when": crypto_secondary_done_when_override,
            }
        if repair_head_area and repair_head_code and repair_head_action:
            _append_action_queue_item(
                action_queue,
                area=repair_head_area,
                target=repair_head_code,
                symbol=repair_head_code,
                action=repair_head_action,
                reason=repair_head_clear_when or str(repair_queue_payload.get("brief") or "").strip(),
            )
            action_queue_detail_overrides[
                (repair_head_area, repair_head_code, repair_head_action)
            ] = {
                "blocker_detail": repair_gate_blocker_detail or repair_head_clear_when,
                "done_when": repair_done_when or repair_clearing_done_when,
            }
        operator_action_queue = [
            {
                **row,
                "rank": idx,
            }
            for idx, row in enumerate(action_queue, start=1)
        ]
        operator_action_queue_brief = _action_queue_brief(operator_action_queue)

        operator_action_checklist = [
            {
                **row,
                "rank": idx,
                "state": _action_checklist_state(
                    area=str(row.get("area") or ""),
                    action=str(row.get("action") or ""),
                    rank=idx,
                ),
                "blocker_detail": _action_checklist_blocker(
                    area=str(row.get("area") or ""),
                    action=str(row.get("action") or ""),
                    symbol=str(row.get("symbol") or ""),
                    reason=str(row.get("reason") or ""),
                    row_blocker_detail=str(
                        action_queue_detail_overrides.get(
                            (
                                str(row.get("area") or ""),
                                str(row.get("target") or ""),
                                str(row.get("action") or ""),
                            ),
                            {},
                        ).get("blocker_detail")
                        or ""
                    ),
                    stale_signal_dates=commodity_execution_bridge_stale_signal_dates,
                    stale_signal_age_days=commodity_execution_bridge_stale_signal_age_days,
                    focus_evidence_summary=commodity_focus_evidence_summary,
                    focus_area=next_focus_area,
                    focus_symbol=next_focus_symbol,
                    focus_lifecycle_status=str(commodity_focus_lifecycle.get("status") or ""),
                    focus_lifecycle_blocker_detail=str(commodity_focus_lifecycle.get("blocker_detail") or ""),
                    crypto_focus_execution_blocker_detail=crypto_route_focus_execution_blocker_detail,
                    crypto_route_alignment_status=str(checklist_crypto_route_alignment.get("status") or ""),
                    crypto_route_alignment_recovery_status=str(
                        checklist_crypto_route_alignment_recovery.get("status") or ""
                    ),
                    crypto_route_alignment_recovery_zero_trade_batches=list(
                        checklist_crypto_route_alignment_recovery.get("zero_trade_batches") or []
                    ),
                ),
                "done_when": _action_checklist_done_when(
                    action=str(row.get("action") or ""),
                    symbol=str(row.get("symbol") or ""),
                    row_done_when=str(
                        action_queue_detail_overrides.get(
                            (
                                str(row.get("area") or ""),
                                str(row.get("target") or ""),
                                str(row.get("action") or ""),
                            ),
                            {},
                        ).get("done_when")
                        or ""
                    ),
                    focus_lifecycle_status=str(commodity_focus_lifecycle.get("status") or ""),
                    focus_lifecycle_done_when=str(commodity_focus_lifecycle.get("done_when") or ""),
                    crypto_focus_execution_done_when=crypto_route_focus_execution_done_when,
                    crypto_route_alignment_status=str(checklist_crypto_route_alignment.get("status") or ""),
                    crypto_route_alignment_recovery_status=str(
                        checklist_crypto_route_alignment_recovery.get("status") or ""
                    ),
                ),
            }
            for idx, row in enumerate(action_queue, start=1)
        ]
        operator_action_checklist_brief = _action_checklist_brief(operator_action_checklist)

    if source_operator_repair_queue:
        operator_repair_queue = source_operator_repair_queue
        operator_repair_queue_brief = source_operator_repair_queue_brief or _action_queue_brief(
            operator_repair_queue
        )
        if source_operator_repair_checklist:
            operator_repair_checklist = source_operator_repair_checklist
        else:
            operator_repair_checklist = [
                {
                    **row,
                    "rank": idx,
                    "state": _action_checklist_state(
                        area=str(row.get("area") or ""),
                        action=str(row.get("action") or ""),
                        rank=idx,
                    ),
                    "blocker_detail": str(row.get("clear_when") or "").strip()
                    or repair_gate_blocker_detail
                    or repair_head_clear_when,
                    "done_when": str(row.get("clear_when") or "").strip()
                    or repair_done_when
                    or repair_clearing_done_when,
                }
                for idx, row in enumerate(operator_repair_queue, start=1)
            ]
        operator_repair_checklist_brief = source_operator_repair_checklist_brief or _action_checklist_brief(
            operator_repair_checklist
        )
    else:
        repair_queue_items = [
            dict(row)
            for row in repair_queue_payload.get("items", [])
            if isinstance(row, dict)
        ]
        operator_repair_queue = []
        for idx, row in enumerate(repair_queue_items, start=1):
            row_area = str(row.get("area") or "").strip()
            row_code = str(row.get("code") or "").strip()
            row_action = str(row.get("action") or "").strip()
            if not row_area or not row_code or not row_action:
                continue
            operator_repair_queue.append(
                {
                    "rank": idx,
                    "area": row_area,
                    "target": row_code,
                    "symbol": row_code.upper(),
                    "action": row_action,
                    "reason": str(
                        row.get("clear_when")
                        or row.get("goal")
                        or repair_queue_payload.get("brief")
                        or ""
                    ).strip(),
                    "priority_score": int(row.get("priority_score") or 0),
                    "priority_tier": str(row.get("priority_tier") or "").strip(),
                    "command": str(row.get("command") or "").strip(),
                    "clear_when": str(row.get("clear_when") or "").strip(),
                    "goal": str(row.get("goal") or "").strip(),
                    "actions": [
                        str(item).strip()
                        for item in row.get("actions", [])
                        if str(item).strip()
                    ]
                    if isinstance(row.get("actions"), list)
                    else [],
                }
            )
        operator_repair_queue_brief = _action_queue_brief(operator_repair_queue)
        operator_repair_checklist = [
            {
                **row,
                "rank": idx,
                "state": _action_checklist_state(
                    area=str(row.get("area") or ""),
                    action=str(row.get("action") or ""),
                    rank=idx,
                ),
                "blocker_detail": str(row.get("clear_when") or "").strip()
                or repair_gate_blocker_detail
                or repair_head_clear_when,
                "done_when": str(row.get("clear_when") or "").strip()
                or repair_done_when
                or repair_clearing_done_when,
            }
            for idx, row in enumerate(operator_repair_queue, start=1)
        ]
        operator_repair_checklist_brief = _action_checklist_brief(operator_repair_checklist)
    next_focus_state = "-"
    next_focus_blocker_detail = "-"
    next_focus_done_when = "-"
    followup_focus_state = "-"
    followup_focus_blocker_detail = "-"
    followup_focus_done_when = "-"
    secondary_focus_state = "-"
    secondary_focus_blocker_detail = "-"
    secondary_focus_done_when = "-"
    secondary_focus_priority_bound = False
    primary_slot_row: dict[str, Any] = {}
    followup_slot_row: dict[str, Any] = {}
    secondary_slot_row: dict[str, Any] = {}
    source_focus_slot_rows = {
        str(row.get("slot") or "").strip(): dict(row)
        for row in source_operator_focus_slots
        if str(row.get("slot") or "").strip() in {"primary", "followup", "secondary"}
    }
    if source_focus_slot_rows:
        primary_slot_row = source_focus_slot_rows.get("primary", {})
        if primary_slot_row:
            next_focus_area = str(primary_slot_row.get("area") or next_focus_area or "-")
            next_focus_target = str(primary_slot_row.get("target") or next_focus_target or "-")
            next_focus_symbol = str(primary_slot_row.get("symbol") or next_focus_symbol or "-")
            next_focus_action = str(primary_slot_row.get("action") or next_focus_action or "-")
            next_focus_reason = str(primary_slot_row.get("reason") or next_focus_reason or "-")
            next_focus_state = str(primary_slot_row.get("state") or "-")
            next_focus_blocker_detail = str(primary_slot_row.get("blocker_detail") or "-")
            next_focus_done_when = str(primary_slot_row.get("done_when") or "-")
        followup_slot_row = source_focus_slot_rows.get("followup", {})
        if followup_slot_row:
            followup_focus_area = str(followup_slot_row.get("area") or followup_focus_area or "-")
            followup_focus_target = str(followup_slot_row.get("target") or followup_focus_target or "-")
            followup_focus_symbol = str(followup_slot_row.get("symbol") or followup_focus_symbol or "-")
            followup_focus_action = str(followup_slot_row.get("action") or followup_focus_action or "-")
            followup_focus_reason = str(followup_slot_row.get("reason") or followup_focus_reason or "-")
            followup_focus_state = str(followup_slot_row.get("state") or "-")
            followup_focus_blocker_detail = str(followup_slot_row.get("blocker_detail") or "-")
            followup_focus_done_when = str(followup_slot_row.get("done_when") or "-")
        secondary_slot_row = source_focus_slot_rows.get("secondary", {})
        if secondary_slot_row:
            secondary_focus_area = str(secondary_slot_row.get("area") or secondary_focus_area or "-")
            secondary_focus_target = str(secondary_slot_row.get("target") or secondary_focus_target or "-")
            secondary_focus_symbol = str(secondary_slot_row.get("symbol") or secondary_focus_symbol or "-")
            secondary_focus_action = str(secondary_slot_row.get("action") or secondary_focus_action or "-")
            secondary_focus_reason = str(secondary_slot_row.get("reason") or secondary_focus_reason or "-")
            secondary_focus_state = str(secondary_slot_row.get("state") or "-")
            secondary_focus_blocker_detail = str(secondary_slot_row.get("blocker_detail") or "-")
            secondary_focus_done_when = str(secondary_slot_row.get("done_when") or "-")
            secondary_slot_priority_tier = str(secondary_slot_row.get("priority_tier") or "").strip()
            secondary_slot_priority_score = secondary_slot_row.get("priority_score")
            secondary_slot_queue_rank = secondary_slot_row.get("queue_rank")
            if secondary_slot_priority_tier:
                secondary_focus_priority_tier = secondary_slot_priority_tier
                secondary_focus_priority_bound = True
            if secondary_slot_priority_score not in (None, "", "-"):
                secondary_focus_priority_score = int(secondary_slot_priority_score or 0)
                secondary_focus_priority_bound = True
            if secondary_slot_queue_rank not in (None, "", "-"):
                secondary_focus_queue_rank = int(secondary_slot_queue_rank or 0)
                secondary_focus_priority_bound = True
    else:
        if operator_action_checklist:
            active_row = dict(operator_action_checklist[0])
            next_focus_state = str(active_row.get("state") or "-")
            next_focus_blocker_detail = str(active_row.get("blocker_detail") or "-")
            next_focus_done_when = str(active_row.get("done_when") or "-")
        if len(operator_action_checklist) >= 2:
            followup_checklist_row = dict(operator_action_checklist[1])
            followup_focus_area = str(followup_checklist_row.get("area") or followup_focus_area or "-")
            followup_focus_target = str(followup_checklist_row.get("target") or followup_focus_target or "-")
            followup_focus_symbol = str(followup_checklist_row.get("symbol") or followup_focus_symbol or "-")
            followup_focus_action = str(followup_checklist_row.get("action") or followup_focus_action or "-")
            followup_focus_reason = str(followup_checklist_row.get("reason") or followup_focus_reason or "-")
            followup_focus_state = str(followup_checklist_row.get("state") or "-")
            followup_focus_blocker_detail = str(followup_checklist_row.get("blocker_detail") or "-")
            followup_focus_done_when = str(followup_checklist_row.get("done_when") or "-")
        if len(operator_action_checklist) >= 3:
            secondary_checklist_row = dict(operator_action_checklist[2])
            secondary_focus_area = str(secondary_checklist_row.get("area") or secondary_focus_area or "-")
            secondary_focus_target = str(secondary_checklist_row.get("target") or secondary_focus_target or "-")
            secondary_focus_symbol = str(secondary_checklist_row.get("symbol") or secondary_focus_symbol or "-")
            secondary_focus_action = str(secondary_checklist_row.get("action") or secondary_focus_action or "-")
            secondary_focus_reason = str(secondary_checklist_row.get("reason") or secondary_focus_reason or "-")
            secondary_focus_state = str(secondary_checklist_row.get("state") or "-")
            secondary_focus_blocker_detail = str(secondary_checklist_row.get("blocker_detail") or "-")
            secondary_focus_done_when = str(secondary_checklist_row.get("done_when") or "-")
            secondary_checklist_priority_tier = str(
                secondary_checklist_row.get("priority_tier") or ""
            ).strip()
            secondary_checklist_priority_score = secondary_checklist_row.get("priority_score")
            secondary_checklist_queue_rank = secondary_checklist_row.get("queue_rank")
            if secondary_checklist_priority_tier:
                secondary_focus_priority_tier = secondary_checklist_priority_tier
                secondary_focus_priority_bound = True
            if secondary_checklist_priority_score not in (None, "", "-"):
                secondary_focus_priority_score = int(secondary_checklist_priority_score or 0)
                secondary_focus_priority_bound = True
            if secondary_checklist_queue_rank not in (None, "", "-"):
                secondary_focus_queue_rank = int(secondary_checklist_queue_rank or 0)
                secondary_focus_priority_bound = True
    if (
        not secondary_focus_priority_bound
        and (
        secondary_focus_area == str(cross_market_review_head_lane.get("head", {}).get("area") or "").strip()
        and secondary_focus_symbol == str(cross_market_review_head_lane.get("head", {}).get("symbol") or "").strip().upper()
        and secondary_focus_action == str(cross_market_review_head_lane.get("head", {}).get("action") or "").strip()
        )
    ):
        secondary_focus_priority_tier = str(
            (cross_market_review_head_lane.get("head") or {}).get("priority_tier") or secondary_focus_priority_tier or "-"
        )
        secondary_focus_priority_score = int(
            (cross_market_review_head_lane.get("head") or {}).get("priority_score") or secondary_focus_priority_score or 0
        )
        secondary_focus_queue_rank = int(
            cross_market_review_head_lane.get("head_rank") or secondary_focus_queue_rank or 1
        )
        secondary_focus_priority_bound = True
    if (
        not secondary_focus_priority_bound
        and (
        secondary_focus_area != "crypto_route"
        and secondary_focus_area != str(cross_market_review_head_lane.get("head", {}).get("area") or "").strip()
        )
    ):
        secondary_focus_priority_tier = "-"
        secondary_focus_priority_score = "-"
        secondary_focus_queue_rank = "-"
    commodity_execution_close_evidence_status = "not_active"
    commodity_execution_close_evidence_brief = "not_active:-"
    commodity_execution_close_evidence_target = "-"
    commodity_execution_close_evidence_symbol = "-"
    commodity_execution_close_evidence_blocker_detail = (
        "close-evidence lane is not the current operator focus."
    )
    commodity_execution_close_evidence_done_when = (
        "operator focus moves to a close-evidence wait item before reassessing"
    )
    if (
        next_focus_area == "commodity_execution_close_evidence"
        and next_focus_action == "wait_for_paper_execution_close_evidence"
    ):
        commodity_execution_close_evidence_status = "close_evidence_pending"
        commodity_execution_close_evidence_target = next_focus_target or "-"
        commodity_execution_close_evidence_symbol = next_focus_symbol or "-"
        commodity_execution_close_evidence_brief = (
            f"close_evidence_pending:{commodity_execution_close_evidence_symbol}"
            if commodity_execution_close_evidence_symbol and commodity_execution_close_evidence_symbol != "-"
            else "close_evidence_pending"
        )
        commodity_execution_close_evidence_blocker_detail = (
            next_focus_blocker_detail
            or str(commodity_focus_lifecycle.get("blocker_detail") or "").strip()
            or commodity_execution_close_evidence_blocker_detail
        )
        commodity_execution_close_evidence_done_when = (
            next_focus_done_when
            or str(commodity_focus_lifecycle.get("done_when") or "").strip()
            or commodity_execution_close_evidence_done_when
        )
    slot_defaults = {
        "primary": {
            "area": next_focus_area,
            "target": next_focus_target,
            "symbol": next_focus_symbol,
            "action": next_focus_action,
            "reason": next_focus_reason,
            "state": next_focus_state,
            "priority_score": (
                int(primary_slot_row.get("priority_score") or 0)
                if primary_slot_row.get("priority_score") not in (None, "", "-")
                else "-"
            ),
            "priority_tier": str(primary_slot_row.get("priority_tier") or "").strip() or "-",
            "queue_rank": (
                int(primary_slot_row.get("queue_rank") or 0)
                if primary_slot_row.get("queue_rank") not in (None, "", "-")
                else "-"
            ),
            "blocker_detail": next_focus_blocker_detail,
            "done_when": next_focus_done_when,
        },
        "followup": {
            "area": followup_focus_area,
            "target": followup_focus_target,
            "symbol": followup_focus_symbol,
            "action": followup_focus_action,
            "reason": followup_focus_reason,
            "state": followup_focus_state,
            "priority_score": (
                int(followup_slot_row.get("priority_score") or 0)
                if followup_slot_row.get("priority_score") not in (None, "", "-")
                else "-"
            ),
            "priority_tier": str(followup_slot_row.get("priority_tier") or "").strip() or "-",
            "queue_rank": (
                int(followup_slot_row.get("queue_rank") or 0)
                if followup_slot_row.get("queue_rank") not in (None, "", "-")
                else "-"
            ),
            "blocker_detail": followup_focus_blocker_detail,
            "done_when": followup_focus_done_when,
        },
        "secondary": {
            "area": secondary_focus_area,
            "target": secondary_focus_target,
            "symbol": secondary_focus_symbol,
            "action": secondary_focus_action,
            "reason": secondary_focus_reason,
            "state": secondary_focus_state,
            "priority_score": secondary_focus_priority_score,
            "priority_tier": secondary_focus_priority_tier,
            "queue_rank": secondary_focus_queue_rank,
            "blocker_detail": secondary_focus_blocker_detail,
            "done_when": secondary_focus_done_when,
        },
    }
    operator_focus_slots: list[dict[str, Any]] = []
    for slot_name in ("primary", "followup", "secondary"):
        default_row = dict(slot_defaults.get(slot_name) or {})
        source_row = dict(source_focus_slot_rows.get(slot_name) or {})
        if source_row:
            slot_row = dict(source_row)
            slot_row["slot"] = slot_name
            for key, fallback_value in default_row.items():
                if slot_row.get(key) in (None, ""):
                    slot_row[key] = fallback_value
        else:
            slot_row = {"slot": slot_name, **default_row}
        operator_focus_slots.append(slot_row)
    if source_focus_slot_rows:
        operator_focus_slots_brief = source_operator_focus_slots_brief or _focus_slots_brief(
            operator_focus_slots
        )
    else:
        operator_focus_slots_brief = _focus_slots_brief(operator_focus_slots)

    operator_stack_brief = f"commodity:{commodity_execution_brief} | crypto:{crypto_route_short_brief}"

    summary_lines = [
        f"status: {operator_status}",
        f"stack: {operator_stack_brief}",
        f"next-focus: {next_focus_area}:{next_focus_target}",
        f"next-focus-action: {next_focus_action}",
        f"next-focus-state: {next_focus_state}",
        f"next-focus-blocker: {next_focus_blocker_detail}",
        f"next-focus-done-when: {next_focus_done_when}",
        f"followup-focus: {followup_focus_area}:{followup_focus_target}:{followup_focus_action}",
        f"followup-focus-state: {followup_focus_state}",
        f"followup-focus-blocker: {followup_focus_blocker_detail}",
        f"followup-focus-done-when: {followup_focus_done_when}",
        f"secondary-focus: {secondary_focus_area}:{secondary_focus_target}:{secondary_focus_action}",
        f"secondary-focus-state: {secondary_focus_state}",
        f"secondary-focus-blocker: {secondary_focus_blocker_detail}",
        f"secondary-focus-done-when: {secondary_focus_done_when}",
        (
            "secondary-focus-priority: "
            + " | ".join(
                [
                    secondary_focus_priority_tier or "-",
                    f"score={secondary_focus_priority_score}",
                    f"queue_rank={secondary_focus_queue_rank}",
                ]
            )
        ),
        f"focus-slots: {operator_focus_slots_brief}",
        f"action-queue: {operator_action_queue_brief}",
        f"action-checklist: {operator_action_checklist_brief}",
        f"repair-queue: {operator_repair_queue_brief}",
        f"repair-checklist: {operator_repair_checklist_brief}",
        f"primary: {_list_text(focus_primary)}",
        f"regime-filter: {_list_text(focus_regime)}",
        f"research-queue: {_list_text(research_queue)}",
        f"shadow: {_list_text(shadow_only)}",
        f"commodity-status: {commodity_status or '-'}",
        f"commodity-route: {commodity_route_stack or '-'}",
        f"commodity-primary: {_list_text(commodity_primary)}",
        f"commodity-regime-filter: {_list_text(commodity_regime)}",
        f"commodity-shadow: {_list_text(commodity_shadow)}",
        f"commodity-focus: {commodity_focus_batch or '-'}",
        f"commodity-focus-symbols: {_list_text(commodity_focus_symbols)}",
        f"commodity-next-stage: {commodity_next_stage or '-'}",
        f"commodity-ticket-status: {commodity_ticket_status or '-'}",
        f"commodity-ticket-stack: {commodity_ticket_stack or '-'}",
        f"commodity-ticket-focus: {commodity_ticket_focus_batch or '-'}",
        f"commodity-ticket-symbols: {_list_text(commodity_ticket_focus_symbols)}",
        f"commodity-ticket-book-status: {commodity_ticket_book_status or '-'}",
        f"commodity-ticket-book-stack: {commodity_ticket_book_stack or '-'}",
        f"commodity-ticket-book-actionable: {_list_text(commodity_actionable_batches)}",
        f"commodity-ticket-book-shadow: {_list_text(commodity_shadow_batches)}",
        f"commodity-next-ticket-id: {commodity_next_ticket_id or '-'}",
        f"commodity-next-ticket-symbol: {commodity_next_ticket_symbol or '-'}",
        f"commodity-execution-preview-status: {commodity_execution_preview_status or '-'}",
        f"commodity-execution-preview-stack: {commodity_execution_preview_stack or '-'}",
        f"commodity-preview-ready: {_list_text(commodity_preview_ready_batches)}",
        f"commodity-preview-shadow: {_list_text(commodity_preview_shadow_batches)}",
        f"commodity-next-execution-batch: {commodity_next_execution_batch or '-'}",
        f"commodity-next-execution-symbols: {_list_text(commodity_next_execution_symbols)}",
        f"commodity-execution-artifact-status: {commodity_execution_artifact_status or '-'}",
        f"commodity-execution-stack: {commodity_execution_artifact_stack or '-'}",
        f"commodity-execution-batch: {commodity_execution_batch or '-'}",
        f"commodity-execution-symbols: {_list_text(commodity_execution_symbols)}",
        f"commodity-execution-queue-status: {commodity_execution_queue_status or '-'}",
        f"commodity-execution-queue-stack: {commodity_execution_queue_stack or '-'}",
        f"commodity-next-queue-execution-id: {commodity_next_queue_execution_id or '-'}",
        f"commodity-next-queue-execution-symbol: {commodity_next_queue_execution_symbol or '-'}",
        f"commodity-execution-review-status: {commodity_execution_review_status or '-'}",
        f"commodity-execution-retro-status: {commodity_execution_retro_status or '-'}",
        f"commodity-execution-gap-status: {commodity_execution_gap_status or '-'}",
        f"commodity-execution-gap-decision: {commodity_execution_gap_decision or '-'}",
        f"commodity-execution-gap-reasons: {_list_text(commodity_execution_gap_reason_codes)}",
        f"commodity-execution-bridge-status: {commodity_execution_bridge_status or '-'}",
        f"commodity-execution-bridge-signal-missing-count: {commodity_execution_bridge_signal_missing_count}",
        f"commodity-execution-bridge-signal-stale-count: {commodity_execution_bridge_signal_stale_count}",
        f"commodity-execution-bridge-signal-proxy-price-only-count: {commodity_execution_bridge_signal_proxy_price_only_count}",
        f"commodity-execution-bridge-stale-signal-dates: {_mapping_text(commodity_execution_bridge_stale_signal_dates)}",
        f"commodity-execution-bridge-stale-signal-age-days: {_mapping_text(commodity_execution_bridge_stale_signal_age_days)}",
        f"commodity-stale-signal-watch: {commodity_stale_signal_watch_brief}",
        f"commodity-stale-signal-watch-next-id: {commodity_stale_signal_watch_next_execution_id}",
        f"commodity-stale-signal-watch-next-symbol: {commodity_stale_signal_watch_next_symbol}",
        f"commodity-stale-signal-watch-next-signal-date: {commodity_stale_signal_watch_next_signal_date}",
        f"commodity-stale-signal-watch-next-signal-age-days: {commodity_stale_signal_watch_next_signal_age_days}",
        f"commodity-execution-bridge-already-present-count: {commodity_execution_bridge_already_present_count}",
        f"commodity-execution-review-stack: {commodity_execution_review_stack or '-'}",
        f"commodity-execution-retro-stack: {commodity_execution_retro_stack or '-'}",
        f"commodity-next-review-execution-id: {commodity_next_review_execution_id or '-'}",
        f"commodity-next-review-execution-symbol: {commodity_next_review_execution_symbol or '-'}",
        f"commodity-review-pending-symbols: {_list_text(commodity_review_pending_symbols)}",
        f"commodity-review-close-evidence-pending-count: {commodity_review_close_evidence_pending_count}",
        f"commodity-review-close-evidence-pending-symbols: {_list_text(commodity_review_close_evidence_pending_symbols)}",
        f"commodity-next-fill-evidence-execution-id: {commodity_next_fill_evidence_execution_id or '-'}",
        f"commodity-next-fill-evidence-execution-symbol: {commodity_next_fill_evidence_execution_symbol or '-'}",
        f"commodity-fill-evidence-pending-count: {commodity_fill_evidence_pending_count}",
        f"commodity-fill-evidence-pending-symbols: {_list_text(commodity_retro_fill_evidence_pending_symbols or commodity_review_fill_evidence_pending_symbols)}",
        f"commodity-close-evidence-pending-count: {commodity_close_evidence_pending_count}",
        f"commodity-next-close-evidence-execution-id: {commodity_next_close_evidence_execution_id or '-'}",
        f"commodity-next-close-evidence-execution-symbol: {commodity_next_close_evidence_execution_symbol or '-'}",
        f"commodity-close-evidence-pending-symbols: {_list_text(commodity_close_evidence_pending_symbols)}",
        f"commodity-retro-item-count: {commodity_retro_item_count}",
        f"commodity-actionable-retro-item-count: {commodity_actionable_retro_item_count}",
        f"commodity-next-retro-execution-id: {commodity_next_retro_execution_id or '-'}",
        f"commodity-next-retro-execution-symbol: {commodity_next_retro_execution_symbol or '-'}",
        f"commodity-retro-pending-symbols: {_list_text(commodity_retro_pending_symbols)}",
        f"commodity-remainder-focus: {commodity_remainder_focus_area}:{commodity_remainder_focus_target}:{commodity_remainder_focus_action}",
        f"commodity-remainder-focus-signal-date: {commodity_remainder_focus_signal_date}",
        f"commodity-remainder-focus-signal-age-days: {commodity_remainder_focus_signal_age_days}",
        f"crypto-status: {crypto_status or '-'}",
        f"crypto-routes: {route_stack or '-'}",
        f"crypto-focus: {focus_symbol or '-'}",
        f"crypto-action: {focus_action or '-'}",
    ]
    if commodity_focus_evidence_summary:
        summary_lines.append(
            "commodity-focus-paper-evidence: "
            + " ".join(
                [
                    f"source={commodity_focus_evidence_item_source}",
                    f"entry={_fmt_num(commodity_focus_evidence_summary.get('paper_entry_price'))}",
                    f"stop={_fmt_num(commodity_focus_evidence_summary.get('paper_stop_price'))}",
                    f"target={_fmt_num(commodity_focus_evidence_summary.get('paper_target_price'))}",
                    f"quote={_fmt_num(commodity_focus_evidence_summary.get('paper_quote_usdt'))}",
                    f"status={commodity_focus_evidence_summary.get('paper_execution_status') or '-'}",
                    f"ref={commodity_focus_evidence_summary.get('paper_signal_price_reference_source') or '-'}",
                ]
            )
        )
    if str(commodity_focus_lifecycle.get("brief") or "").strip():
        summary_lines.append(
            f"commodity-focus-lifecycle: {str(commodity_focus_lifecycle.get('brief') or '').strip()}"
        )
    if commodity_execution_close_evidence_brief:
        summary_lines.append(f"commodity-close-evidence: {commodity_execution_close_evidence_brief}")
    if commodity_execution_bridge_already_bridged_symbols:
        summary_lines.append(
            f"commodity-execution-bridge-already-bridged-symbols: {_list_text(commodity_execution_bridge_already_bridged_symbols)}"
        )
    if commodity_execution_gap_root_cause_lines:
        summary_lines.append(f"commodity-gap-root-cause: {commodity_execution_gap_root_cause_lines[0]}")
    if commodity_leaders_primary:
        summary_lines.append(f"commodity-leaders-primary: {_list_text(commodity_leaders_primary)}")
    if commodity_leaders_regime:
        summary_lines.append(f"commodity-leaders-regime: {_list_text(commodity_leaders_regime)}")
    if focus_gate:
        summary_lines.append(f"crypto-focus-gate: {focus_gate}")
    if focus_window:
        summary_lines.append(f"crypto-focus-window: {focus_window}")
    if crypto_route_shortline_market_state_brief:
        summary_lines.append(f"crypto-shortline-market-state: {crypto_route_shortline_market_state_brief}")
    if crypto_route_shortline_execution_gate_brief:
        summary_lines.append(f"crypto-shortline-trigger-stack: {crypto_route_shortline_execution_gate_brief}")
    if crypto_route_shortline_no_trade_rule:
        summary_lines.append(f"crypto-shortline-no-trade: {crypto_route_shortline_no_trade_rule}")
    if crypto_route_focus_execution_state:
        summary_lines.append(f"crypto-focus-execution-state: {crypto_route_focus_execution_state}")
    if crypto_route_focus_execution_blocker_detail:
        summary_lines.append(f"crypto-focus-execution-blocker: {crypto_route_focus_execution_blocker_detail}")
    if crypto_route_focus_execution_done_when:
        summary_lines.append(f"crypto-focus-execution-done-when: {crypto_route_focus_execution_done_when}")
    if crypto_route_focus_execution_micro_classification:
        summary_lines.append(
            "crypto-focus-micro: "
            + " | ".join(
                [
                    crypto_route_focus_execution_micro_classification,
                    crypto_route_focus_execution_micro_context or "-",
                    crypto_route_focus_execution_micro_trust_tier or "-",
                    crypto_route_focus_execution_micro_veto or "-",
                ]
            )
        )
    if crypto_route_focus_execution_micro_reasons:
        summary_lines.append(
            "crypto-focus-micro-reasons: " + _list_text(crypto_route_focus_execution_micro_reasons, limit=20)
        )
    if str(crypto_route_focus_review_lane.get("status") or "").strip() not in {"", "not_active"}:
        summary_lines.append(
            "crypto-review-lane: "
            + " | ".join(
                [
                    str(crypto_route_focus_review_lane.get("status") or "-"),
                    str(crypto_route_focus_review_lane.get("primary_blocker") or "-"),
                    str(crypto_route_focus_review_lane.get("micro_blocker") or "-"),
                ]
            )
        )
    if str(crypto_route_focus_review_scores.get("status") or "").strip() == "scored":
        summary_lines.append(
            "crypto-review-scores: "
            + " | ".join(
                [
                    f"edge={int(crypto_route_focus_review_scores.get('edge_score') or 0)}",
                    f"structure={int(crypto_route_focus_review_scores.get('structure_score') or 0)}",
                    f"micro={int(crypto_route_focus_review_scores.get('micro_score') or 0)}",
                    f"composite={int(crypto_route_focus_review_scores.get('composite_score') or 0)}",
                ]
            )
        )
    if str(crypto_route_focus_review_priority.get("status") or "").strip() == "ready":
        summary_lines.append(
            "crypto-review-priority: "
            + " | ".join(
                [
                    str(crypto_route_focus_review_priority.get("tier") or "-"),
                    f"score={int(crypto_route_focus_review_priority.get('score') or 0)}",
                ]
            )
        )
    if crypto_route_review_priority_queue_status:
        summary_lines.append(
            "crypto-review-queue: "
            + " | ".join(
                [
                    crypto_route_review_priority_queue_status or "-",
                    crypto_route_review_priority_queue_brief or "-",
                    f"head={crypto_route_review_priority_head_symbol or '-'}:{crypto_route_review_priority_head_tier or '-'}:{crypto_route_review_priority_head_score}",
                ]
            )
        )
    if crypto_route_shortline_cvd_semantic_status:
        summary_lines.append(
            "crypto-cvd-semantic: "
            + " | ".join(
                [
                    crypto_route_shortline_cvd_semantic_status,
                    crypto_route_shortline_cvd_semantic_takeaway or "-",
                ]
            )
        )
    if crypto_route_shortline_cvd_queue_handoff_status:
        summary_lines.append(
            "crypto-cvd-queue: "
            + " | ".join(
                [
                    crypto_route_shortline_cvd_queue_handoff_status,
                    f"{crypto_route_shortline_cvd_queue_focus_batch or '-'}:{crypto_route_shortline_cvd_queue_focus_action or '-'}",
                    crypto_route_shortline_cvd_queue_stack_brief or "-",
                ]
            )
        )
    if focus_window_floor:
        summary_lines.append(f"crypto-focus-window-floor: {focus_window_floor}")
    if price_state_window_floor:
        summary_lines.append(f"crypto-price-window-floor: {price_state_window_floor}")
    if comparative_window_takeaway:
        summary_lines.append(f"crypto-window-note: {comparative_window_takeaway}")
    if xlong_flow_window_floor:
        summary_lines.append(f"crypto-xlong-flow-floor: {xlong_flow_window_floor}")
    if xlong_comparative_window_takeaway:
        summary_lines.append(f"crypto-xlong-note: {xlong_comparative_window_takeaway}")
    if next_retest_action:
        summary_lines.append(f"crypto-next-retest: {next_retest_action}")
    if focus_brief:
        summary_lines.append(f"crypto-focus-brief: {focus_brief}")

    return {
        "operator_status": operator_status,
        "operator_stack_brief": operator_stack_brief,
        "commodity_route_brief": commodity_route_brief,
        "commodity_ticket_brief": commodity_ticket_brief,
        "commodity_execution_brief": commodity_execution_brief,
        "crypto_route_short_brief": crypto_route_short_brief,
        "next_focus_area": next_focus_area,
        "next_focus_target": next_focus_target,
        "next_focus_symbol": next_focus_symbol,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "next_focus_state": next_focus_state,
        "next_focus_blocker_detail": next_focus_blocker_detail,
        "next_focus_done_when": next_focus_done_when,
        "followup_focus_area": followup_focus_area,
        "followup_focus_target": followup_focus_target,
        "followup_focus_symbol": followup_focus_symbol,
        "followup_focus_action": followup_focus_action,
        "followup_focus_reason": followup_focus_reason,
        "followup_focus_state": followup_focus_state,
        "followup_focus_blocker_detail": followup_focus_blocker_detail,
        "followup_focus_done_when": followup_focus_done_when,
        "operator_focus_slots": operator_focus_slots,
        "operator_focus_slots_brief": operator_focus_slots_brief,
        "operator_action_queue": operator_action_queue,
        "operator_action_queue_brief": operator_action_queue_brief,
        "operator_action_checklist": operator_action_checklist,
        "operator_action_checklist_brief": operator_action_checklist_brief,
        "operator_repair_queue": operator_repair_queue,
        "operator_repair_queue_brief": operator_repair_queue_brief,
        "operator_repair_queue_count": len(operator_repair_queue),
        "operator_repair_checklist": operator_repair_checklist,
        "operator_repair_checklist_brief": operator_repair_checklist_brief,
        "secondary_focus_area": secondary_focus_area,
        "secondary_focus_target": secondary_focus_target,
        "secondary_focus_symbol": secondary_focus_symbol,
        "secondary_focus_action": secondary_focus_action,
        "secondary_focus_reason": secondary_focus_reason,
        "secondary_focus_state": secondary_focus_state,
        "secondary_focus_blocker_detail": secondary_focus_blocker_detail,
        "secondary_focus_done_when": secondary_focus_done_when,
        "secondary_focus_priority_tier": secondary_focus_priority_tier,
        "secondary_focus_priority_score": secondary_focus_priority_score,
        "secondary_focus_queue_rank": secondary_focus_queue_rank,
        "crypto_route_shortline_market_state_brief": crypto_route_shortline_market_state_brief,
        "crypto_route_shortline_execution_gate_brief": crypto_route_shortline_execution_gate_brief,
        "crypto_route_shortline_no_trade_rule": crypto_route_shortline_no_trade_rule,
        "crypto_route_shortline_session_map_brief": crypto_route_shortline_session_map_brief,
        "crypto_route_shortline_cvd_semantic_status": crypto_route_shortline_cvd_semantic_status,
        "crypto_route_shortline_cvd_semantic_takeaway": crypto_route_shortline_cvd_semantic_takeaway,
        "crypto_route_shortline_cvd_queue_handoff_status": crypto_route_shortline_cvd_queue_handoff_status,
        "crypto_route_shortline_cvd_queue_handoff_takeaway": crypto_route_shortline_cvd_queue_handoff_takeaway,
        "crypto_route_shortline_cvd_queue_focus_batch": crypto_route_shortline_cvd_queue_focus_batch,
        "crypto_route_shortline_cvd_queue_focus_action": crypto_route_shortline_cvd_queue_focus_action,
        "crypto_route_shortline_cvd_queue_stack_brief": crypto_route_shortline_cvd_queue_stack_brief,
        "crypto_route_focus_execution_state": crypto_route_focus_execution_state,
        "crypto_route_focus_execution_blocker_detail": crypto_route_focus_execution_blocker_detail,
        "crypto_route_focus_execution_done_when": crypto_route_focus_execution_done_when,
        "crypto_route_focus_execution_micro_classification": crypto_route_focus_execution_micro_classification,
        "crypto_route_focus_execution_micro_context": crypto_route_focus_execution_micro_context,
        "crypto_route_focus_execution_micro_trust_tier": crypto_route_focus_execution_micro_trust_tier,
        "crypto_route_focus_execution_micro_veto": crypto_route_focus_execution_micro_veto,
        "crypto_route_focus_execution_micro_locality_status": crypto_route_focus_execution_micro_locality_status,
        "crypto_route_focus_execution_micro_drift_risk": crypto_route_focus_execution_micro_drift_risk,
        "crypto_route_focus_execution_micro_attack_side": crypto_route_focus_execution_micro_attack_side,
        "crypto_route_focus_execution_micro_attack_presence": crypto_route_focus_execution_micro_attack_presence,
        "crypto_route_focus_execution_micro_reasons": crypto_route_focus_execution_micro_reasons,
        "crypto_route_focus_review_status": str(crypto_route_focus_review_lane.get("status") or ""),
        "crypto_route_focus_review_brief": str(crypto_route_focus_review_lane.get("brief") or ""),
        "crypto_route_focus_review_primary_blocker": str(
            crypto_route_focus_review_lane.get("primary_blocker") or ""
        ),
        "crypto_route_focus_review_micro_blocker": str(
            crypto_route_focus_review_lane.get("micro_blocker") or ""
        ),
        "crypto_route_focus_review_blocker_detail": str(
            crypto_route_focus_review_lane.get("blocker_detail") or ""
        ),
        "crypto_route_focus_review_done_when": str(
            crypto_route_focus_review_lane.get("done_when") or ""
        ),
        "crypto_route_focus_review_score_status": str(crypto_route_focus_review_scores.get("status") or ""),
        "crypto_route_focus_review_edge_score": int(crypto_route_focus_review_scores.get("edge_score") or 0),
        "crypto_route_focus_review_structure_score": int(
            crypto_route_focus_review_scores.get("structure_score") or 0
        ),
        "crypto_route_focus_review_micro_score": int(crypto_route_focus_review_scores.get("micro_score") or 0),
        "crypto_route_focus_review_composite_score": int(
            crypto_route_focus_review_scores.get("composite_score") or 0
        ),
        "crypto_route_focus_review_score_brief": str(crypto_route_focus_review_scores.get("brief") or ""),
        "crypto_route_focus_review_priority_status": str(
            crypto_route_focus_review_priority.get("status") or ""
        ),
        "crypto_route_focus_review_priority_score": int(
            crypto_route_focus_review_priority.get("score") or 0
        ),
        "crypto_route_focus_review_priority_tier": str(
            crypto_route_focus_review_priority.get("tier") or ""
        ),
        "crypto_route_focus_review_priority_brief": str(
            crypto_route_focus_review_priority.get("brief") or ""
        ),
        "crypto_route_review_priority_queue_status": crypto_route_review_priority_queue_status,
        "crypto_route_review_priority_queue_count": crypto_route_review_priority_queue_count,
        "crypto_route_review_priority_queue_brief": crypto_route_review_priority_queue_brief,
        "crypto_route_review_priority_head_symbol": crypto_route_review_priority_head_symbol,
        "crypto_route_review_priority_head_tier": crypto_route_review_priority_head_tier,
        "crypto_route_review_priority_head_score": crypto_route_review_priority_head_score,
        "crypto_route_review_priority_head_action": crypto_route_review_priority_head_action,
        "crypto_route_review_priority_head_reason": crypto_route_review_priority_head_reason,
        "crypto_route_review_priority_head_blocker_detail": crypto_route_review_priority_head_blocker_detail,
        "crypto_route_review_priority_head_done_when": crypto_route_review_priority_head_done_when,
        "crypto_route_review_priority_head_rank": crypto_route_review_priority_head_rank,
        "crypto_route_review_priority_queue": crypto_route_review_priority_queue,
        "commodity_remainder_focus_area": commodity_remainder_focus_area,
        "commodity_remainder_focus_target": commodity_remainder_focus_target,
        "commodity_remainder_focus_symbol": commodity_remainder_focus_symbol,
        "commodity_remainder_focus_action": commodity_remainder_focus_action,
        "commodity_remainder_focus_reason": commodity_remainder_focus_reason,
        "commodity_remainder_focus_signal_date": commodity_remainder_focus_signal_date,
        "commodity_remainder_focus_signal_age_days": commodity_remainder_focus_signal_age_days,
        "commodity_focus_evidence_item_source": commodity_focus_evidence_item_source,
        "commodity_focus_evidence_summary": commodity_focus_evidence_summary,
        "commodity_focus_lifecycle_status": str(commodity_focus_lifecycle.get("status") or ""),
        "commodity_focus_lifecycle_brief": str(commodity_focus_lifecycle.get("brief") or ""),
        "commodity_focus_lifecycle_blocker_detail": str(commodity_focus_lifecycle.get("blocker_detail") or ""),
        "commodity_focus_lifecycle_done_when": str(commodity_focus_lifecycle.get("done_when") or ""),
        "commodity_execution_close_evidence_status": commodity_execution_close_evidence_status,
        "commodity_execution_close_evidence_brief": commodity_execution_close_evidence_brief,
        "commodity_execution_close_evidence_target": commodity_execution_close_evidence_target,
        "commodity_execution_close_evidence_symbol": commodity_execution_close_evidence_symbol,
        "commodity_execution_close_evidence_blocker_detail": commodity_execution_close_evidence_blocker_detail,
        "commodity_execution_close_evidence_done_when": commodity_execution_close_evidence_done_when,
        "focus_primary_batches": focus_primary,
        "focus_with_regime_filter_batches": focus_regime,
        "research_queue_batches": research_queue,
        "shadow_only_batches": shadow_only,
        "avoid_batches": avoid,
        "commodity_route_status": commodity_status,
        "commodity_execution_mode": commodity_execution_mode,
        "commodity_route_stack_brief": commodity_route_stack,
        "commodity_focus_primary_batches": commodity_primary,
        "commodity_focus_with_regime_filter_batches": commodity_regime,
        "commodity_shadow_only_batches": commodity_shadow,
        "commodity_leader_symbols_primary": commodity_leaders_primary,
        "commodity_leader_symbols_regime_filter": commodity_leaders_regime,
        "commodity_focus_batch": commodity_focus_batch,
        "commodity_focus_symbols": commodity_focus_symbols,
        "commodity_next_stage": commodity_next_stage,
        "commodity_ticket_status": commodity_ticket_status,
        "commodity_ticket_stack_brief": commodity_ticket_stack,
        "commodity_paper_ready_batches": commodity_paper_ready_batches,
        "commodity_ticket_focus_batch": commodity_ticket_focus_batch,
        "commodity_ticket_focus_symbols": commodity_ticket_focus_symbols,
        "commodity_ticket_count": commodity_ticket_count,
        "commodity_ticket_book_status": commodity_ticket_book_status,
        "commodity_ticket_book_stack_brief": commodity_ticket_book_stack,
        "commodity_actionable_batches": commodity_actionable_batches,
        "commodity_shadow_batches": commodity_shadow_batches,
        "commodity_next_ticket_id": commodity_next_ticket_id,
        "commodity_next_ticket_symbol": commodity_next_ticket_symbol,
        "commodity_actionable_ticket_count": commodity_actionable_ticket_count,
        "commodity_execution_preview_status": commodity_execution_preview_status,
        "commodity_execution_preview_stack_brief": commodity_execution_preview_stack,
        "commodity_preview_ready_batches": commodity_preview_ready_batches,
        "commodity_preview_shadow_batches": commodity_preview_shadow_batches,
        "commodity_next_execution_batch": commodity_next_execution_batch,
        "commodity_next_execution_symbols": commodity_next_execution_symbols,
        "commodity_next_execution_ticket_ids": commodity_next_execution_ticket_ids,
        "commodity_next_execution_regime_gate": commodity_next_execution_regime_gate,
        "commodity_next_execution_weight_hint_sum": commodity_next_execution_weight_hint_sum,
        "commodity_execution_artifact_status": commodity_execution_artifact_status,
        "commodity_execution_stack_brief": commodity_execution_artifact_stack,
        "commodity_execution_batch": commodity_execution_batch,
        "commodity_execution_symbols": commodity_execution_symbols,
        "commodity_execution_ticket_ids": commodity_execution_ticket_ids,
        "commodity_execution_regime_gate": commodity_execution_regime_gate,
        "commodity_execution_weight_hint_sum": commodity_execution_weight_hint_sum,
        "commodity_execution_item_count": commodity_execution_item_count,
        "commodity_actionable_execution_item_count": commodity_actionable_execution_item_count,
        "commodity_execution_queue_status": commodity_execution_queue_status,
        "commodity_execution_queue_stack_brief": commodity_execution_queue_stack,
        "commodity_queue_depth": commodity_queue_depth,
        "commodity_actionable_queue_depth": commodity_actionable_queue_depth,
        "commodity_next_queue_execution_id": commodity_next_queue_execution_id,
        "commodity_next_queue_execution_symbol": commodity_next_queue_execution_symbol,
        "commodity_execution_review_status": commodity_execution_review_status,
        "commodity_execution_review_stack_brief": commodity_execution_review_stack,
        "commodity_review_item_count": commodity_review_item_count,
        "commodity_actionable_review_item_count": commodity_actionable_review_item_count,
        "commodity_review_pending_symbols": commodity_review_pending_symbols,
        "commodity_review_close_evidence_pending_count": commodity_review_close_evidence_pending_count,
        "commodity_review_close_evidence_pending_symbols": commodity_review_close_evidence_pending_symbols,
        "commodity_review_fill_evidence_pending_count": commodity_review_fill_evidence_pending_count,
        "commodity_review_fill_evidence_pending_symbols": commodity_review_fill_evidence_pending_symbols,
        "commodity_next_review_execution_id": commodity_next_review_execution_id,
        "commodity_next_review_execution_symbol": commodity_next_review_execution_symbol,
        "commodity_next_review_close_evidence_execution_id": commodity_next_review_close_evidence_execution_id,
        "commodity_next_review_close_evidence_execution_symbol": commodity_next_review_close_evidence_execution_symbol,
        "commodity_next_review_fill_evidence_execution_id": commodity_next_review_fill_evidence_execution_id,
        "commodity_next_review_fill_evidence_execution_symbol": commodity_next_review_fill_evidence_execution_symbol,
        "commodity_execution_retro_status": commodity_execution_retro_status,
        "commodity_execution_retro_stack_brief": commodity_execution_retro_stack,
        "commodity_retro_item_count": commodity_retro_item_count,
        "commodity_actionable_retro_item_count": commodity_actionable_retro_item_count,
        "commodity_close_evidence_pending_count": commodity_close_evidence_pending_count,
        "commodity_close_evidence_pending_symbols": commodity_close_evidence_pending_symbols,
        "commodity_next_close_evidence_execution_id": commodity_next_close_evidence_execution_id,
        "commodity_next_close_evidence_execution_symbol": commodity_next_close_evidence_execution_symbol,
        "commodity_retro_pending_symbols": commodity_retro_pending_symbols,
        "commodity_retro_fill_evidence_pending_count": commodity_retro_fill_evidence_pending_count,
        "commodity_retro_fill_evidence_pending_symbols": commodity_retro_fill_evidence_pending_symbols,
        "commodity_next_retro_execution_id": commodity_next_retro_execution_id,
        "commodity_next_retro_execution_symbol": commodity_next_retro_execution_symbol,
        "commodity_next_retro_fill_evidence_execution_id": commodity_next_retro_fill_evidence_execution_id,
        "commodity_next_retro_fill_evidence_execution_symbol": commodity_next_retro_fill_evidence_execution_symbol,
        "commodity_fill_evidence_pending_count": commodity_fill_evidence_pending_count,
        "commodity_next_fill_evidence_execution_id": commodity_next_fill_evidence_execution_id,
        "commodity_next_fill_evidence_execution_symbol": commodity_next_fill_evidence_execution_symbol,
        "commodity_execution_gap_status": commodity_execution_gap_status,
        "commodity_execution_gap_decision": commodity_execution_gap_decision,
        "commodity_execution_gap_reason_codes": commodity_execution_gap_reason_codes,
        "commodity_execution_gap_batch": commodity_execution_gap_batch,
        "commodity_execution_gap_next_execution_id": commodity_execution_gap_next_execution_id,
        "commodity_execution_gap_next_execution_symbol": commodity_execution_gap_next_execution_symbol,
        "commodity_gap_focus_batch": commodity_gap_focus_batch,
        "commodity_gap_focus_execution_id": commodity_gap_focus_execution_id,
        "commodity_gap_focus_symbol": commodity_gap_focus_symbol,
        "commodity_execution_gap_root_cause_lines": commodity_execution_gap_root_cause_lines,
        "commodity_execution_gap_recommended_actions": commodity_execution_gap_recommended_actions,
        "commodity_stale_signal_watch_items": commodity_stale_signal_watch_items,
        "commodity_stale_signal_watch_brief": commodity_stale_signal_watch_brief,
        "commodity_stale_signal_watch_next_execution_id": commodity_stale_signal_watch_next_execution_id,
        "commodity_stale_signal_watch_next_symbol": commodity_stale_signal_watch_next_symbol,
        "commodity_stale_signal_watch_next_signal_date": commodity_stale_signal_watch_next_signal_date,
        "commodity_stale_signal_watch_next_signal_age_days": commodity_stale_signal_watch_next_signal_age_days,
        "commodity_execution_bridge_status": commodity_execution_bridge_status,
        "commodity_execution_bridge_next_ready_id": commodity_execution_bridge_next_ready_id,
        "commodity_execution_bridge_next_ready_symbol": commodity_execution_bridge_next_ready_symbol,
        "commodity_execution_bridge_next_blocked_id": commodity_execution_bridge_next_blocked_id,
        "commodity_execution_bridge_next_blocked_symbol": commodity_execution_bridge_next_blocked_symbol,
        "commodity_execution_bridge_signal_missing_count": commodity_execution_bridge_signal_missing_count,
        "commodity_execution_bridge_signal_stale_count": commodity_execution_bridge_signal_stale_count,
        "commodity_execution_bridge_signal_proxy_price_only_count": commodity_execution_bridge_signal_proxy_price_only_count,
        "commodity_execution_bridge_stale_signal_dates": commodity_execution_bridge_stale_signal_dates,
        "commodity_execution_bridge_stale_signal_age_days": commodity_execution_bridge_stale_signal_age_days,
        "commodity_execution_bridge_already_present_count": commodity_execution_bridge_already_present_count,
        "commodity_execution_bridge_already_bridged_symbols": commodity_execution_bridge_already_bridged_symbols,
        "crypto_route_status": crypto_status,
        "crypto_route_stack_brief": route_stack,
        "crypto_focus_symbol": focus_symbol,
        "crypto_focus_action": focus_action,
        "crypto_focus_reason": focus_reason,
        "crypto_focus_window_gate": focus_gate,
        "crypto_focus_window_verdict": focus_window,
        "crypto_focus_window_floor": focus_window_floor,
        "crypto_price_state_window_floor": price_state_window_floor,
        "crypto_comparative_window_takeaway": comparative_window_takeaway,
        "crypto_xlong_flow_window_floor": xlong_flow_window_floor,
        "crypto_xlong_comparative_window_takeaway": xlong_comparative_window_takeaway,
        "crypto_focus_brief": focus_brief,
        "crypto_next_retest_action": next_retest_action,
        "crypto_next_retest_reason": next_retest_reason,
        "summary_lines": summary_lines,
        "summary_text": " | ".join(summary_lines),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Hot Universe Operator Brief",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_action_artifact: `{payload.get('source_action_artifact') or payload.get('source_artifact') or ''}`",
        f"- source_action_status: `{payload.get('source_action_status') or payload.get('source_status') or ''}`",
        f"- source_commodity_artifact: `{payload.get('source_commodity_artifact') or ''}`",
        f"- source_commodity_status: `{payload.get('source_commodity_status') or ''}`",
        f"- source_commodity_ticket_artifact: `{payload.get('source_commodity_ticket_artifact') or ''}`",
        f"- source_commodity_ticket_status: `{payload.get('source_commodity_ticket_status') or ''}`",
        f"- source_commodity_ticket_book_artifact: `{payload.get('source_commodity_ticket_book_artifact') or ''}`",
        f"- source_commodity_ticket_book_status: `{payload.get('source_commodity_ticket_book_status') or ''}`",
        f"- source_commodity_execution_preview_artifact: `{payload.get('source_commodity_execution_preview_artifact') or ''}`",
        f"- source_commodity_execution_preview_status: `{payload.get('source_commodity_execution_preview_status') or ''}`",
        f"- source_commodity_execution_artifact: `{payload.get('source_commodity_execution_artifact') or ''}`",
        f"- source_commodity_execution_artifact_status: `{payload.get('source_commodity_execution_artifact_status') or ''}`",
        f"- source_commodity_execution_queue_artifact: `{payload.get('source_commodity_execution_queue_artifact') or ''}`",
        f"- source_commodity_execution_queue_status: `{payload.get('source_commodity_execution_queue_status') or ''}`",
        f"- source_commodity_execution_review_artifact: `{payload.get('source_commodity_execution_review_artifact') or ''}`",
        f"- source_commodity_execution_review_status: `{payload.get('source_commodity_execution_review_status') or ''}`",
        f"- source_commodity_execution_retro_artifact: `{payload.get('source_commodity_execution_retro_artifact') or ''}`",
        f"- source_commodity_execution_retro_status: `{payload.get('source_commodity_execution_retro_status') or ''}`",
        f"- source_commodity_execution_gap_artifact: `{payload.get('source_commodity_execution_gap_artifact') or ''}`",
        f"- source_commodity_execution_gap_status: `{payload.get('source_commodity_execution_gap_status') or ''}`",
        f"- source_commodity_execution_bridge_artifact: `{payload.get('source_commodity_execution_bridge_artifact') or ''}`",
        f"- source_commodity_execution_bridge_status: `{payload.get('source_commodity_execution_bridge_status') or ''}`",
        f"- source_crypto_artifact: `{payload.get('source_crypto_artifact') or payload.get('source_artifact') or ''}`",
        f"- source_crypto_status: `{payload.get('source_crypto_status') or payload.get('source_status') or ''}`",
        *_crypto_route_source_debug_lines(payload),
        *_remote_live_source_debug_lines(payload),
        *_brooks_source_debug_lines(payload),
        *_cross_market_source_debug_lines(payload),
        *_cross_market_runtime_debug_lines(payload),
        f"- brooks_structure_review_status: `{payload.get('brooks_structure_review_status') or ''}`",
        f"- brooks_structure_review_brief: `{payload.get('brooks_structure_review_brief') or ''}`",
        f"- brooks_structure_review_queue_status: `{payload.get('brooks_structure_review_queue_status') or ''}`",
        f"- brooks_structure_review_queue_count: `{payload.get('brooks_structure_review_queue_count')}`",
        f"- brooks_structure_review_queue_brief: `{payload.get('brooks_structure_review_queue_brief') or ''}`",
        f"- brooks_structure_review_head_symbol: `{payload.get('brooks_structure_review_head_symbol') or ''}`",
        f"- brooks_structure_review_head_strategy_id: `{payload.get('brooks_structure_review_head_strategy_id') or ''}`",
        f"- brooks_structure_review_head_direction: `{payload.get('brooks_structure_review_head_direction') or ''}`",
        f"- brooks_structure_review_head_tier: `{payload.get('brooks_structure_review_head_tier') or ''}`",
        f"- brooks_structure_review_head_plan_status: `{payload.get('brooks_structure_review_head_plan_status') or ''}`",
        f"- brooks_structure_review_head_action: `{payload.get('brooks_structure_review_head_action') or ''}`",
        f"- brooks_structure_review_head_blocker_detail: `{payload.get('brooks_structure_review_head_blocker_detail') or ''}`",
        f"- brooks_structure_review_head_done_when: `{payload.get('brooks_structure_review_head_done_when') or ''}`",
        f"- brooks_structure_review_blocker_detail: `{payload.get('brooks_structure_review_blocker_detail') or ''}`",
        f"- brooks_structure_review_done_when: `{payload.get('brooks_structure_review_done_when') or ''}`",
        f"- brooks_structure_operator_status: `{payload.get('brooks_structure_operator_status') or ''}`",
        f"- brooks_structure_operator_brief: `{payload.get('brooks_structure_operator_brief') or ''}`",
        f"- brooks_structure_operator_head_symbol: `{payload.get('brooks_structure_operator_head_symbol') or ''}`",
        f"- brooks_structure_operator_head_strategy_id: `{payload.get('brooks_structure_operator_head_strategy_id') or ''}`",
        f"- brooks_structure_operator_head_direction: `{payload.get('brooks_structure_operator_head_direction') or ''}`",
        f"- brooks_structure_operator_head_action: `{payload.get('brooks_structure_operator_head_action') or ''}`",
        f"- brooks_structure_operator_head_plan_status: `{payload.get('brooks_structure_operator_head_plan_status') or ''}`",
        f"- brooks_structure_operator_head_priority_score: `{payload.get('brooks_structure_operator_head_priority_score')}`",
        f"- brooks_structure_operator_head_priority_tier: `{payload.get('brooks_structure_operator_head_priority_tier') or ''}`",
        f"- brooks_structure_operator_backlog_count: `{payload.get('brooks_structure_operator_backlog_count')}`",
        f"- brooks_structure_operator_backlog_brief: `{payload.get('brooks_structure_operator_backlog_brief') or ''}`",
        f"- brooks_structure_operator_blocker_detail: `{payload.get('brooks_structure_operator_blocker_detail') or ''}`",
        f"- brooks_structure_operator_done_when: `{payload.get('brooks_structure_operator_done_when') or ''}`",
        f"- operator_research_embedding_quality_status: `{payload.get('operator_research_embedding_quality_status') or ''}`",
        f"- operator_research_embedding_quality_brief: `{payload.get('operator_research_embedding_quality_brief') or ''}`",
        f"- operator_research_embedding_quality_blocker_detail: `{payload.get('operator_research_embedding_quality_blocker_detail') or ''}`",
        f"- operator_research_embedding_quality_done_when: `{payload.get('operator_research_embedding_quality_done_when') or ''}`",
        f"- operator_research_embedding_active_batches: `{_list_text(payload.get('operator_research_embedding_active_batches', []), limit=20)}`",
        f"- operator_research_embedding_avoid_batches: `{_list_text(payload.get('operator_research_embedding_avoid_batches', []), limit=20)}`",
        f"- operator_research_embedding_zero_trade_deprioritized_batches: `{_list_text(payload.get('operator_research_embedding_zero_trade_deprioritized_batches', []), limit=20)}`",
        f"- operator_crypto_route_alignment_focus_slot: `{payload.get('operator_crypto_route_alignment_focus_slot') or ''}`",
        f"- operator_crypto_route_alignment_status: `{payload.get('operator_crypto_route_alignment_status') or ''}`",
        f"- operator_crypto_route_alignment_brief: `{payload.get('operator_crypto_route_alignment_brief') or ''}`",
        f"- operator_crypto_route_alignment_blocker_detail: `{payload.get('operator_crypto_route_alignment_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_done_when: `{payload.get('operator_crypto_route_alignment_done_when') or ''}`",
        f"- operator_crypto_route_alignment_recovery_status: `{payload.get('operator_crypto_route_alignment_recovery_status') or ''}`",
        f"- operator_crypto_route_alignment_recovery_brief: `{payload.get('operator_crypto_route_alignment_recovery_brief') or ''}`",
        f"- operator_crypto_route_alignment_recovery_blocker_detail: `{payload.get('operator_crypto_route_alignment_recovery_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_recovery_done_when: `{payload.get('operator_crypto_route_alignment_recovery_done_when') or ''}`",
        f"- operator_crypto_route_alignment_recovery_failed_batch_count: `{payload.get('operator_crypto_route_alignment_recovery_failed_batch_count')}`",
        f"- operator_crypto_route_alignment_recovery_timed_out_batch_count: `{payload.get('operator_crypto_route_alignment_recovery_timed_out_batch_count')}`",
        f"- operator_crypto_route_alignment_recovery_zero_trade_batches: `{_list_text(payload.get('operator_crypto_route_alignment_recovery_zero_trade_batches', []), limit=20)}`",
        f"- operator_crypto_route_alignment_cooldown_status: `{payload.get('operator_crypto_route_alignment_cooldown_status') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_brief: `{payload.get('operator_crypto_route_alignment_cooldown_brief') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_blocker_detail: `{payload.get('operator_crypto_route_alignment_cooldown_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_done_when: `{payload.get('operator_crypto_route_alignment_cooldown_done_when') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_last_research_end_date: `{payload.get('operator_crypto_route_alignment_cooldown_last_research_end_date') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_next_eligible_end_date: `{payload.get('operator_crypto_route_alignment_cooldown_next_eligible_end_date') or ''}`",
        f"- operator_crypto_route_alignment_recipe_status: `{payload.get('operator_crypto_route_alignment_recipe_status') or ''}`",
        f"- operator_crypto_route_alignment_recipe_brief: `{payload.get('operator_crypto_route_alignment_recipe_brief') or ''}`",
        f"- operator_crypto_route_alignment_recipe_blocker_detail: `{payload.get('operator_crypto_route_alignment_recipe_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_recipe_done_when: `{payload.get('operator_crypto_route_alignment_recipe_done_when') or ''}`",
        f"- operator_crypto_route_alignment_recipe_ready_on_date: `{payload.get('operator_crypto_route_alignment_recipe_ready_on_date') or ''}`",
        f"- operator_crypto_route_alignment_recipe_script: `{payload.get('operator_crypto_route_alignment_recipe_script') or ''}`",
        f"- operator_crypto_route_alignment_recipe_command_hint: `{payload.get('operator_crypto_route_alignment_recipe_command_hint') or ''}`",
        f"- operator_crypto_route_alignment_recipe_expected_status: `{payload.get('operator_crypto_route_alignment_recipe_expected_status') or ''}`",
        f"- operator_crypto_route_alignment_recipe_note: `{payload.get('operator_crypto_route_alignment_recipe_note') or ''}`",
        f"- operator_crypto_route_alignment_recipe_followup_script: `{payload.get('operator_crypto_route_alignment_recipe_followup_script') or ''}`",
        f"- operator_crypto_route_alignment_recipe_followup_command_hint: `{payload.get('operator_crypto_route_alignment_recipe_followup_command_hint') or ''}`",
        f"- operator_crypto_route_alignment_recipe_verify_hint: `{payload.get('operator_crypto_route_alignment_recipe_verify_hint') or ''}`",
        f"- operator_crypto_route_alignment_recipe_window_days: `{payload.get('operator_crypto_route_alignment_recipe_window_days')}`",
        f"- operator_crypto_route_alignment_recipe_target_batches: `{_list_text(payload.get('operator_crypto_route_alignment_recipe_target_batches', []), limit=20)}`",
        "",
        "## Focus",
        f"- operator_stack_brief: `{payload.get('operator_stack_brief') or ''}`",
        f"- next_focus_area: `{payload.get('next_focus_area') or ''}`",
        f"- next_focus_target: `{payload.get('next_focus_target') or ''}`",
        f"- next_focus_symbol: `{payload.get('next_focus_symbol') or ''}`",
        f"- next_focus_action: `{payload.get('next_focus_action') or ''}`",
        f"- next_focus_reason: `{payload.get('next_focus_reason') or ''}`",
        f"- next_focus_state: `{payload.get('next_focus_state') or ''}`",
        f"- next_focus_blocker_detail: `{payload.get('next_focus_blocker_detail') or ''}`",
        f"- next_focus_done_when: `{payload.get('next_focus_done_when') or ''}`",
        f"- followup_focus_area: `{payload.get('followup_focus_area') or ''}`",
        f"- followup_focus_target: `{payload.get('followup_focus_target') or ''}`",
        f"- followup_focus_symbol: `{payload.get('followup_focus_symbol') or ''}`",
        f"- followup_focus_action: `{payload.get('followup_focus_action') or ''}`",
        f"- followup_focus_reason: `{payload.get('followup_focus_reason') or ''}`",
        f"- followup_focus_state: `{payload.get('followup_focus_state') or ''}`",
        f"- followup_focus_blocker_detail: `{payload.get('followup_focus_blocker_detail') or ''}`",
        f"- followup_focus_done_when: `{payload.get('followup_focus_done_when') or ''}`",
        f"- operator_focus_slots_brief: `{payload.get('operator_focus_slots_brief') or ''}`",
        f"- operator_focus_slot_sources_brief: `{payload.get('operator_focus_slot_sources_brief') or ''}`",
        f"- operator_focus_slot_status_brief: `{payload.get('operator_focus_slot_status_brief') or ''}`",
        f"- operator_focus_slot_recency_brief: `{payload.get('operator_focus_slot_recency_brief') or ''}`",
        f"- operator_focus_slot_health_brief: `{payload.get('operator_focus_slot_health_brief') or ''}`",
        f"- operator_focus_slot_refresh_backlog_brief: `{payload.get('operator_focus_slot_refresh_backlog_brief') or ''}`",
        f"- operator_focus_slot_refresh_backlog_count: `{payload.get('operator_focus_slot_refresh_backlog_count')}`",
        f"- operator_focus_slot_refresh_backlog: `{json.dumps(payload.get('operator_focus_slot_refresh_backlog', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_focus_slot_ready_count: `{payload.get('operator_focus_slot_ready_count')}`",
        f"- operator_focus_slot_total_count: `{payload.get('operator_focus_slot_total_count')}`",
        f"- operator_focus_slot_promotion_gate_brief: `{payload.get('operator_focus_slot_promotion_gate_brief') or ''}`",
        f"- operator_focus_slot_promotion_gate_status: `{payload.get('operator_focus_slot_promotion_gate_status') or ''}`",
        f"- operator_focus_slot_promotion_gate_blocker_detail: `{payload.get('operator_focus_slot_promotion_gate_blocker_detail') or ''}`",
        f"- operator_focus_slot_promotion_gate_done_when: `{payload.get('operator_focus_slot_promotion_gate_done_when') or ''}`",
        f"- operator_focus_slot_actionability_backlog_brief: `{payload.get('operator_focus_slot_actionability_backlog_brief') or ''}`",
        f"- operator_focus_slot_actionability_backlog_count: `{payload.get('operator_focus_slot_actionability_backlog_count')}`",
        f"- operator_focus_slot_actionability_backlog: `{json.dumps(payload.get('operator_focus_slot_actionability_backlog', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_focus_slot_actionable_count: `{payload.get('operator_focus_slot_actionable_count')}`",
        f"- operator_focus_slot_actionability_gate_brief: `{payload.get('operator_focus_slot_actionability_gate_brief') or ''}`",
        f"- operator_focus_slot_actionability_gate_status: `{payload.get('operator_focus_slot_actionability_gate_status') or ''}`",
        f"- operator_focus_slot_actionability_gate_blocker_detail: `{payload.get('operator_focus_slot_actionability_gate_blocker_detail') or ''}`",
        f"- operator_focus_slot_actionability_gate_done_when: `{payload.get('operator_focus_slot_actionability_gate_done_when') or ''}`",
        f"- operator_focus_slot_readiness_gate_ready_count: `{payload.get('operator_focus_slot_readiness_gate_ready_count')}`",
        f"- operator_focus_slot_readiness_gate_brief: `{payload.get('operator_focus_slot_readiness_gate_brief') or ''}`",
        f"- operator_focus_slot_readiness_gate_status: `{payload.get('operator_focus_slot_readiness_gate_status') or ''}`",
        f"- operator_focus_slot_readiness_gate_blocking_gate: `{payload.get('operator_focus_slot_readiness_gate_blocking_gate') or ''}`",
        f"- operator_focus_slot_readiness_gate_blocker_detail: `{payload.get('operator_focus_slot_readiness_gate_blocker_detail') or ''}`",
        f"- operator_focus_slot_readiness_gate_done_when: `{payload.get('operator_focus_slot_readiness_gate_done_when') or ''}`",
        f"- operator_source_refresh_queue_brief: `{payload.get('operator_source_refresh_queue_brief') or ''}`",
        f"- operator_source_refresh_queue_count: `{payload.get('operator_source_refresh_queue_count')}`",
        f"- operator_source_refresh_queue: `{json.dumps(payload.get('operator_source_refresh_queue', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_source_refresh_checklist_brief: `{payload.get('operator_source_refresh_checklist_brief') or ''}`",
        f"- operator_source_refresh_checklist: `{json.dumps(payload.get('operator_source_refresh_checklist', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_source_refresh_pipeline_steps_brief: `{payload.get('operator_source_refresh_pipeline_steps_brief') or ''}`",
        f"- operator_source_refresh_pipeline_step_checkpoint_brief: `{payload.get('operator_source_refresh_pipeline_step_checkpoint_brief') or ''}`",
        f"- operator_source_refresh_pipeline_pending_brief: `{payload.get('operator_source_refresh_pipeline_pending_brief') or ''}`",
        f"- operator_source_refresh_pipeline_pending_count: `{payload.get('operator_source_refresh_pipeline_pending_count')}`",
        f"- operator_source_refresh_pipeline_head_rank: `{payload.get('operator_source_refresh_pipeline_head_rank') or ''}`",
        f"- operator_source_refresh_pipeline_head_name: `{payload.get('operator_source_refresh_pipeline_head_name') or ''}`",
        f"- operator_source_refresh_pipeline_head_checkpoint_state: `{payload.get('operator_source_refresh_pipeline_head_checkpoint_state') or ''}`",
        f"- operator_source_refresh_pipeline_head_expected_artifact_kind: `{payload.get('operator_source_refresh_pipeline_head_expected_artifact_kind') or ''}`",
        f"- operator_source_refresh_pipeline_head_current_artifact: `{payload.get('operator_source_refresh_pipeline_head_current_artifact') or ''}`",
        f"- operator_source_refresh_pipeline_relevance_status: `{payload.get('operator_source_refresh_pipeline_relevance_status') or ''}`",
        f"- operator_source_refresh_pipeline_relevance_brief: `{payload.get('operator_source_refresh_pipeline_relevance_brief') or ''}`",
        f"- operator_source_refresh_pipeline_relevance_blocker_detail: `{payload.get('operator_source_refresh_pipeline_relevance_blocker_detail') or ''}`",
        f"- operator_source_refresh_pipeline_relevance_done_when: `{payload.get('operator_source_refresh_pipeline_relevance_done_when') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_brief: `{payload.get('operator_source_refresh_pipeline_deferred_brief') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_count: `{payload.get('operator_source_refresh_pipeline_deferred_count')}`",
        f"- operator_source_refresh_pipeline_deferred_status: `{payload.get('operator_source_refresh_pipeline_deferred_status') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_until: `{payload.get('operator_source_refresh_pipeline_deferred_until') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_reason: `{payload.get('operator_source_refresh_pipeline_deferred_reason') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_rank: `{payload.get('operator_source_refresh_pipeline_deferred_head_rank') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_name: `{payload.get('operator_source_refresh_pipeline_deferred_head_name') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_checkpoint_state: `{payload.get('operator_source_refresh_pipeline_deferred_head_checkpoint_state') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_expected_artifact_kind: `{payload.get('operator_source_refresh_pipeline_deferred_head_expected_artifact_kind') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_current_artifact: `{payload.get('operator_source_refresh_pipeline_deferred_head_current_artifact') or ''}`",
        f"- operator_focus_slots: `{json.dumps(payload.get('operator_focus_slots', []), ensure_ascii=False, sort_keys=True)}`",
        f"- next_focus_source_kind: `{payload.get('next_focus_source_kind') or ''}`",
        f"- next_focus_source_artifact: `{payload.get('next_focus_source_artifact') or ''}`",
        f"- next_focus_source_status: `{payload.get('next_focus_source_status') or ''}`",
        f"- next_focus_source_as_of: `{payload.get('next_focus_source_as_of') or ''}`",
        f"- next_focus_source_age_minutes: `{payload.get('next_focus_source_age_minutes')}`",
        f"- next_focus_source_recency: `{payload.get('next_focus_source_recency') or ''}`",
        f"- next_focus_source_health: `{payload.get('next_focus_source_health') or ''}`",
        f"- next_focus_source_refresh_action: `{payload.get('next_focus_source_refresh_action') or ''}`",
        f"- followup_focus_source_kind: `{payload.get('followup_focus_source_kind') or ''}`",
        f"- followup_focus_source_artifact: `{payload.get('followup_focus_source_artifact') or ''}`",
        f"- followup_focus_source_status: `{payload.get('followup_focus_source_status') or ''}`",
        f"- followup_focus_source_as_of: `{payload.get('followup_focus_source_as_of') or ''}`",
        f"- followup_focus_source_age_minutes: `{payload.get('followup_focus_source_age_minutes')}`",
        f"- followup_focus_source_recency: `{payload.get('followup_focus_source_recency') or ''}`",
        f"- followup_focus_source_health: `{payload.get('followup_focus_source_health') or ''}`",
        f"- followup_focus_source_refresh_action: `{payload.get('followup_focus_source_refresh_action') or ''}`",
        f"- secondary_focus_source_kind: `{payload.get('secondary_focus_source_kind') or ''}`",
        f"- secondary_focus_source_artifact: `{payload.get('secondary_focus_source_artifact') or ''}`",
        f"- secondary_focus_source_status: `{payload.get('secondary_focus_source_status') or ''}`",
        f"- secondary_focus_source_as_of: `{payload.get('secondary_focus_source_as_of') or ''}`",
        f"- secondary_focus_source_age_minutes: `{payload.get('secondary_focus_source_age_minutes')}`",
        f"- secondary_focus_source_recency: `{payload.get('secondary_focus_source_recency') or ''}`",
        f"- secondary_focus_source_health: `{payload.get('secondary_focus_source_health') or ''}`",
        f"- secondary_focus_source_refresh_action: `{payload.get('secondary_focus_source_refresh_action') or ''}`",
        f"- operator_focus_slot_refresh_head_slot: `{payload.get('operator_focus_slot_refresh_head_slot') or ''}`",
        f"- operator_focus_slot_refresh_head_symbol: `{payload.get('operator_focus_slot_refresh_head_symbol') or ''}`",
        f"- operator_focus_slot_refresh_head_action: `{payload.get('operator_focus_slot_refresh_head_action') or ''}`",
        f"- operator_focus_slot_refresh_head_health: `{payload.get('operator_focus_slot_refresh_head_health') or ''}`",
        f"- operator_source_refresh_next_slot: `{payload.get('operator_source_refresh_next_slot') or ''}`",
        f"- operator_source_refresh_next_symbol: `{payload.get('operator_source_refresh_next_symbol') or ''}`",
        f"- operator_source_refresh_next_action: `{payload.get('operator_source_refresh_next_action') or ''}`",
        f"- operator_source_refresh_next_source_kind: `{payload.get('operator_source_refresh_next_source_kind') or ''}`",
        f"- operator_source_refresh_next_source_health: `{payload.get('operator_source_refresh_next_source_health') or ''}`",
        f"- operator_source_refresh_next_source_artifact: `{payload.get('operator_source_refresh_next_source_artifact') or ''}`",
        f"- operator_source_refresh_next_state: `{payload.get('operator_source_refresh_next_state') or ''}`",
        f"- operator_source_refresh_next_blocker_detail: `{payload.get('operator_source_refresh_next_blocker_detail') or ''}`",
        f"- operator_source_refresh_next_done_when: `{payload.get('operator_source_refresh_next_done_when') or ''}`",
        f"- operator_source_refresh_next_recipe_script: `{payload.get('operator_source_refresh_next_recipe_script') or ''}`",
        f"- operator_source_refresh_next_recipe_command_hint: `{payload.get('operator_source_refresh_next_recipe_command_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_expected_status: `{payload.get('operator_source_refresh_next_recipe_expected_status') or ''}`",
        f"- operator_source_refresh_next_recipe_expected_artifact_kind: `{payload.get('operator_source_refresh_next_recipe_expected_artifact_kind') or ''}`",
        f"- operator_source_refresh_next_recipe_expected_artifact_path_hint: `{payload.get('operator_source_refresh_next_recipe_expected_artifact_path_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_note: `{payload.get('operator_source_refresh_next_recipe_note') or ''}`",
        f"- operator_source_refresh_next_recipe_followup_script: `{payload.get('operator_source_refresh_next_recipe_followup_script') or ''}`",
        f"- operator_source_refresh_next_recipe_followup_command_hint: `{payload.get('operator_source_refresh_next_recipe_followup_command_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_verify_hint: `{payload.get('operator_source_refresh_next_recipe_verify_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_steps_brief: `{payload.get('operator_source_refresh_next_recipe_steps_brief') or ''}`",
        f"- operator_source_refresh_next_recipe_step_checkpoint_brief: `{payload.get('operator_source_refresh_next_recipe_step_checkpoint_brief') or ''}`",
        f"- operator_source_refresh_next_recipe_steps: `{json.dumps(payload.get('operator_source_refresh_next_recipe_steps', []), ensure_ascii=False, sort_keys=True)}`",
        f"- crypto_route_head_source_refresh_status: `{payload.get('crypto_route_head_source_refresh_status') or ''}`",
        f"- crypto_route_head_source_refresh_brief: `{payload.get('crypto_route_head_source_refresh_brief') or ''}`",
        f"- crypto_route_head_source_refresh_slot: `{payload.get('crypto_route_head_source_refresh_slot') or ''}`",
        f"- crypto_route_head_source_refresh_symbol: `{payload.get('crypto_route_head_source_refresh_symbol') or ''}`",
        f"- crypto_route_head_source_refresh_action: `{payload.get('crypto_route_head_source_refresh_action') or ''}`",
        f"- crypto_route_head_source_refresh_source_kind: `{payload.get('crypto_route_head_source_refresh_source_kind') or ''}`",
        f"- crypto_route_head_source_refresh_source_health: `{payload.get('crypto_route_head_source_refresh_source_health') or ''}`",
        f"- crypto_route_head_source_refresh_source_artifact: `{payload.get('crypto_route_head_source_refresh_source_artifact') or ''}`",
        f"- crypto_route_head_source_refresh_blocker_detail: `{payload.get('crypto_route_head_source_refresh_blocker_detail') or ''}`",
        f"- crypto_route_head_source_refresh_done_when: `{payload.get('crypto_route_head_source_refresh_done_when') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_script: `{payload.get('crypto_route_head_source_refresh_recipe_script') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_command_hint: `{payload.get('crypto_route_head_source_refresh_recipe_command_hint') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_expected_status: `{payload.get('crypto_route_head_source_refresh_recipe_expected_status') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_expected_artifact_kind: `{payload.get('crypto_route_head_source_refresh_recipe_expected_artifact_kind') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_expected_artifact_path_hint: `{payload.get('crypto_route_head_source_refresh_recipe_expected_artifact_path_hint') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_note: `{payload.get('crypto_route_head_source_refresh_recipe_note') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_followup_script: `{payload.get('crypto_route_head_source_refresh_recipe_followup_script') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_followup_command_hint: `{payload.get('crypto_route_head_source_refresh_recipe_followup_command_hint') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_verify_hint: `{payload.get('crypto_route_head_source_refresh_recipe_verify_hint') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_steps_brief: `{payload.get('crypto_route_head_source_refresh_recipe_steps_brief') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_step_checkpoint_brief: `{payload.get('crypto_route_head_source_refresh_recipe_step_checkpoint_brief') or ''}`",
        f"- crypto_route_head_source_refresh_recipe_steps: `{json.dumps(payload.get('crypto_route_head_source_refresh_recipe_steps', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_action_queue_brief: `{payload.get('operator_action_queue_brief') or ''}`",
        f"- operator_action_queue: `{json.dumps(payload.get('operator_action_queue', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_action_checklist_brief: `{payload.get('operator_action_checklist_brief') or ''}`",
        f"- operator_action_checklist: `{json.dumps(payload.get('operator_action_checklist', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_repair_queue_brief: `{payload.get('operator_repair_queue_brief') or ''}`",
        f"- operator_repair_queue_count: `{payload.get('operator_repair_queue_count')}`",
        f"- operator_repair_queue: `{json.dumps(payload.get('operator_repair_queue', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_repair_checklist_brief: `{payload.get('operator_repair_checklist_brief') or ''}`",
        f"- operator_repair_checklist: `{json.dumps(payload.get('operator_repair_checklist', []), ensure_ascii=False, sort_keys=True)}`",
        f"- secondary_focus_area: `{payload.get('secondary_focus_area') or ''}`",
        f"- secondary_focus_target: `{payload.get('secondary_focus_target') or ''}`",
        f"- secondary_focus_symbol: `{payload.get('secondary_focus_symbol') or ''}`",
        f"- secondary_focus_action: `{payload.get('secondary_focus_action') or ''}`",
        f"- secondary_focus_reason: `{payload.get('secondary_focus_reason') or ''}`",
        f"- secondary_focus_state: `{payload.get('secondary_focus_state') or ''}`",
        f"- secondary_focus_blocker_detail: `{payload.get('secondary_focus_blocker_detail') or ''}`",
        f"- secondary_focus_done_when: `{payload.get('secondary_focus_done_when') or ''}`",
        f"- secondary_focus_priority_tier: `{payload.get('secondary_focus_priority_tier') or ''}`",
        f"- secondary_focus_priority_score: `{payload.get('secondary_focus_priority_score')}`",
        f"- secondary_focus_queue_rank: `{payload.get('secondary_focus_queue_rank')}`",
        f"- crypto_route_shortline_market_state_brief: `{payload.get('crypto_route_shortline_market_state_brief') or ''}`",
        f"- crypto_route_shortline_execution_gate_brief: `{payload.get('crypto_route_shortline_execution_gate_brief') or ''}`",
        f"- crypto_route_shortline_no_trade_rule: `{payload.get('crypto_route_shortline_no_trade_rule') or ''}`",
        f"- crypto_route_shortline_session_map_brief: `{payload.get('crypto_route_shortline_session_map_brief') or ''}`",
        f"- crypto_route_shortline_cvd_semantic_status: `{payload.get('crypto_route_shortline_cvd_semantic_status') or ''}`",
        f"- crypto_route_shortline_cvd_semantic_takeaway: `{payload.get('crypto_route_shortline_cvd_semantic_takeaway') or ''}`",
        f"- crypto_route_shortline_cvd_queue_handoff_status: `{payload.get('crypto_route_shortline_cvd_queue_handoff_status') or ''}`",
        f"- crypto_route_shortline_cvd_queue_handoff_takeaway: `{payload.get('crypto_route_shortline_cvd_queue_handoff_takeaway') or ''}`",
        f"- crypto_route_shortline_cvd_queue_focus_batch: `{payload.get('crypto_route_shortline_cvd_queue_focus_batch') or ''}`",
        f"- crypto_route_shortline_cvd_queue_focus_action: `{payload.get('crypto_route_shortline_cvd_queue_focus_action') or ''}`",
        f"- crypto_route_shortline_cvd_queue_stack_brief: `{payload.get('crypto_route_shortline_cvd_queue_stack_brief') or ''}`",
        f"- crypto_route_focus_execution_state: `{payload.get('crypto_route_focus_execution_state') or ''}`",
        f"- crypto_route_focus_execution_blocker_detail: `{payload.get('crypto_route_focus_execution_blocker_detail') or ''}`",
        f"- crypto_route_focus_execution_done_when: `{payload.get('crypto_route_focus_execution_done_when') or ''}`",
        f"- crypto_route_focus_execution_micro_classification: `{payload.get('crypto_route_focus_execution_micro_classification') or ''}`",
        f"- crypto_route_focus_execution_micro_context: `{payload.get('crypto_route_focus_execution_micro_context') or ''}`",
        f"- crypto_route_focus_execution_micro_trust_tier: `{payload.get('crypto_route_focus_execution_micro_trust_tier') or ''}`",
        f"- crypto_route_focus_execution_micro_veto: `{payload.get('crypto_route_focus_execution_micro_veto') or ''}`",
        f"- crypto_route_focus_execution_micro_locality_status: `{payload.get('crypto_route_focus_execution_micro_locality_status') or ''}`",
        f"- crypto_route_focus_execution_micro_drift_risk: `{payload.get('crypto_route_focus_execution_micro_drift_risk') or ''}`",
        f"- crypto_route_focus_execution_micro_attack_side: `{payload.get('crypto_route_focus_execution_micro_attack_side') or ''}`",
        f"- crypto_route_focus_execution_micro_attack_presence: `{payload.get('crypto_route_focus_execution_micro_attack_presence') or ''}`",
        f"- crypto_route_focus_execution_micro_reasons: `{_list_text(payload.get('crypto_route_focus_execution_micro_reasons', []), limit=20)}`",
        f"- crypto_route_focus_review_status: `{payload.get('crypto_route_focus_review_status') or ''}`",
        f"- crypto_route_focus_review_brief: `{payload.get('crypto_route_focus_review_brief') or ''}`",
        f"- crypto_route_focus_review_primary_blocker: `{payload.get('crypto_route_focus_review_primary_blocker') or ''}`",
        f"- crypto_route_focus_review_micro_blocker: `{payload.get('crypto_route_focus_review_micro_blocker') or ''}`",
        f"- crypto_route_focus_review_blocker_detail: `{payload.get('crypto_route_focus_review_blocker_detail') or ''}`",
        f"- crypto_route_focus_review_done_when: `{payload.get('crypto_route_focus_review_done_when') or ''}`",
        f"- crypto_route_focus_review_score_status: `{payload.get('crypto_route_focus_review_score_status') or ''}`",
        f"- crypto_route_focus_review_edge_score: `{payload.get('crypto_route_focus_review_edge_score')}`",
        f"- crypto_route_focus_review_structure_score: `{payload.get('crypto_route_focus_review_structure_score')}`",
        f"- crypto_route_focus_review_micro_score: `{payload.get('crypto_route_focus_review_micro_score')}`",
        f"- crypto_route_focus_review_composite_score: `{payload.get('crypto_route_focus_review_composite_score')}`",
        f"- crypto_route_focus_review_score_brief: `{payload.get('crypto_route_focus_review_score_brief') or ''}`",
        f"- crypto_route_focus_review_priority_status: `{payload.get('crypto_route_focus_review_priority_status') or ''}`",
        f"- crypto_route_focus_review_priority_score: `{payload.get('crypto_route_focus_review_priority_score')}`",
        f"- crypto_route_focus_review_priority_tier: `{payload.get('crypto_route_focus_review_priority_tier') or ''}`",
        f"- crypto_route_focus_review_priority_brief: `{payload.get('crypto_route_focus_review_priority_brief') or ''}`",
        f"- crypto_route_review_priority_queue_status: `{payload.get('crypto_route_review_priority_queue_status') or ''}`",
        f"- crypto_route_review_priority_queue_count: `{payload.get('crypto_route_review_priority_queue_count')}`",
        f"- crypto_route_review_priority_queue_brief: `{payload.get('crypto_route_review_priority_queue_brief') or ''}`",
        f"- crypto_route_review_priority_head_symbol: `{payload.get('crypto_route_review_priority_head_symbol') or ''}`",
        f"- crypto_route_review_priority_head_tier: `{payload.get('crypto_route_review_priority_head_tier') or ''}`",
        f"- crypto_route_review_priority_head_score: `{payload.get('crypto_route_review_priority_head_score')}`",
        f"- crypto_route_review_priority_head_action: `{payload.get('crypto_route_review_priority_head_action') or ''}`",
        f"- crypto_route_review_priority_head_reason: `{payload.get('crypto_route_review_priority_head_reason') or ''}`",
        f"- crypto_route_review_priority_head_blocker_detail: `{payload.get('crypto_route_review_priority_head_blocker_detail') or ''}`",
        f"- crypto_route_review_priority_head_done_when: `{payload.get('crypto_route_review_priority_head_done_when') or ''}`",
        f"- crypto_route_review_priority_head_rank: `{payload.get('crypto_route_review_priority_head_rank')}`",
        f"- commodity_remainder_focus_area: `{payload.get('commodity_remainder_focus_area') or ''}`",
        f"- commodity_remainder_focus_target: `{payload.get('commodity_remainder_focus_target') or ''}`",
        f"- commodity_remainder_focus_symbol: `{payload.get('commodity_remainder_focus_symbol') or ''}`",
        f"- commodity_remainder_focus_action: `{payload.get('commodity_remainder_focus_action') or ''}`",
        f"- commodity_remainder_focus_reason: `{payload.get('commodity_remainder_focus_reason') or ''}`",
        f"- commodity_remainder_focus_signal_date: `{payload.get('commodity_remainder_focus_signal_date') or ''}`",
        f"- commodity_remainder_focus_signal_age_days: `{payload.get('commodity_remainder_focus_signal_age_days')}`",
        f"- commodity_focus_evidence_item_source: `{payload.get('commodity_focus_evidence_item_source') or ''}`",
        f"- commodity_focus_evidence_summary: `{json.dumps(payload.get('commodity_focus_evidence_summary', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- commodity_focus_lifecycle_status: `{payload.get('commodity_focus_lifecycle_status') or ''}`",
        f"- commodity_focus_lifecycle_brief: `{payload.get('commodity_focus_lifecycle_brief') or ''}`",
        f"- commodity_focus_lifecycle_blocker_detail: `{payload.get('commodity_focus_lifecycle_blocker_detail') or ''}`",
        f"- commodity_focus_lifecycle_done_when: `{payload.get('commodity_focus_lifecycle_done_when') or ''}`",
        f"- commodity_execution_gap_status: `{payload.get('commodity_execution_gap_status') or ''}`",
        f"- commodity_execution_gap_decision: `{payload.get('commodity_execution_gap_decision') or ''}`",
        f"- commodity_execution_gap_reason_codes: `{_list_text(payload.get('commodity_execution_gap_reason_codes', []), limit=20)}`",
        f"- commodity_execution_bridge_status: `{payload.get('commodity_execution_bridge_status') or ''}`",
        f"- commodity_execution_bridge_next_blocked_id: `{payload.get('commodity_execution_bridge_next_blocked_id') or ''}`",
        f"- commodity_execution_bridge_next_blocked_symbol: `{payload.get('commodity_execution_bridge_next_blocked_symbol') or ''}`",
        f"- commodity_execution_bridge_stale_signal_dates: `{_mapping_text(payload.get('commodity_execution_bridge_stale_signal_dates', {}), limit=20)}`",
        f"- commodity_execution_bridge_stale_signal_age_days: `{_mapping_text(payload.get('commodity_execution_bridge_stale_signal_age_days', {}), limit=20)}`",
        f"- commodity_stale_signal_watch_brief: `{payload.get('commodity_stale_signal_watch_brief') or ''}`",
        f"- commodity_stale_signal_watch_next_execution_id: `{payload.get('commodity_stale_signal_watch_next_execution_id') or ''}`",
        f"- commodity_stale_signal_watch_next_symbol: `{payload.get('commodity_stale_signal_watch_next_symbol') or ''}`",
        f"- commodity_stale_signal_watch_next_signal_date: `{payload.get('commodity_stale_signal_watch_next_signal_date') or ''}`",
        f"- commodity_stale_signal_watch_next_signal_age_days: `{payload.get('commodity_stale_signal_watch_next_signal_age_days')}`",
        f"- commodity_execution_bridge_already_present_count: `{payload.get('commodity_execution_bridge_already_present_count') or 0}`",
        f"- commodity_execution_bridge_already_bridged_symbols: `{_list_text(payload.get('commodity_execution_bridge_already_bridged_symbols', []), limit=20)}`",
        f"- commodity_next_fill_evidence_execution_id: `{payload.get('commodity_next_fill_evidence_execution_id') or ''}`",
        f"- commodity_next_fill_evidence_execution_symbol: `{payload.get('commodity_next_fill_evidence_execution_symbol') or ''}`",
        f"- commodity_fill_evidence_pending_count: `{payload.get('commodity_fill_evidence_pending_count') or 0}`",
        f"- commodity_review_pending_symbols: `{_list_text(payload.get('commodity_review_pending_symbols', []), limit=20)}`",
        f"- commodity_review_close_evidence_pending_symbols: `{_list_text(payload.get('commodity_review_close_evidence_pending_symbols', []), limit=20)}`",
        f"- commodity_retro_pending_symbols: `{_list_text(payload.get('commodity_retro_pending_symbols', []), limit=20)}`",
        f"- commodity_close_evidence_pending_symbols: `{_list_text(payload.get('commodity_close_evidence_pending_symbols', []), limit=20)}`",
        f"- commodity_review_fill_evidence_pending_symbols: `{_list_text(payload.get('commodity_review_fill_evidence_pending_symbols', []), limit=20)}`",
        f"- commodity_retro_fill_evidence_pending_symbols: `{_list_text(payload.get('commodity_retro_fill_evidence_pending_symbols', []), limit=20)}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an operator-focused brief from the latest hot-universe research artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--commodity-execution-queue-json", default="")
    parser.add_argument("--commodity-execution-review-json", default="")
    parser.add_argument("--commodity-execution-retro-json", default="")
    parser.add_argument("--commodity-execution-gap-json", default="")
    parser.add_argument("--commodity-execution-bridge-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    action_source_path = latest_hot_universe_action_source(review_dir, runtime_now)
    crypto_source_path = latest_hot_universe_crypto_source(review_dir, runtime_now)
    crypto_route_source_path = latest_crypto_route_focus_source(review_dir, runtime_now)
    crypto_route_refresh_source_path = latest_crypto_route_refresh_source(review_dir, runtime_now)
    remote_live_history_audit_source_path = latest_remote_live_history_audit_source(review_dir, runtime_now)
    remote_live_handoff_source_path = latest_remote_live_handoff_source(review_dir, runtime_now)
    live_gate_blocker_source_path = latest_live_gate_blocker_report_source(review_dir, runtime_now)
    brooks_route_report_source_path = latest_brooks_price_action_route_report_source(review_dir, runtime_now)
    brooks_execution_plan_source_path = latest_brooks_price_action_execution_plan_source(review_dir, runtime_now)
    brooks_structure_review_queue_source_path = latest_brooks_structure_review_queue_source(review_dir, runtime_now)
    brooks_structure_refresh_source_path = latest_brooks_structure_refresh_source(review_dir, runtime_now)
    cross_market_operator_state_source_path = latest_cross_market_operator_state_source(review_dir, runtime_now)
    system_time_sync_repair_plan_source_path = latest_system_time_sync_repair_plan_source(review_dir, runtime_now)
    system_time_sync_repair_verification_source_path = latest_system_time_sync_repair_verification_source(
        review_dir, runtime_now
    )
    openclaw_orderflow_blueprint_source_path = latest_openclaw_orderflow_blueprint_source(
        review_dir, runtime_now
    )
    commodity_source_path = latest_commodity_execution_lane_source(review_dir, runtime_now)
    commodity_ticket_source_path = latest_commodity_paper_ticket_lane_source(review_dir, runtime_now)
    commodity_ticket_book_source_path = latest_commodity_paper_ticket_book_source(review_dir, runtime_now)
    commodity_execution_preview_source_path = latest_commodity_paper_execution_preview_source(review_dir, runtime_now)
    commodity_execution_artifact_source_path = latest_commodity_paper_execution_artifact_source(review_dir, runtime_now)
    commodity_execution_queue_source_path = resolve_explicit_source(args.commodity_execution_queue_json) or latest_commodity_paper_execution_queue_source(review_dir, runtime_now)
    commodity_execution_review_source_path = resolve_explicit_source(args.commodity_execution_review_json) or latest_commodity_paper_execution_review_source(review_dir, runtime_now)
    commodity_execution_retro_source_path = resolve_explicit_source(args.commodity_execution_retro_json) or latest_commodity_paper_execution_retro_source(review_dir, runtime_now)
    commodity_execution_gap_source_path = resolve_explicit_source(args.commodity_execution_gap_json) or latest_commodity_paper_execution_gap_source(review_dir, runtime_now)
    commodity_execution_bridge_source_path = resolve_explicit_source(args.commodity_execution_bridge_json) or latest_commodity_paper_execution_bridge_source(review_dir, runtime_now)
    action_source_payload = json.loads(action_source_path.read_text(encoding="utf-8"))
    crypto_source_payload = json.loads(crypto_source_path.read_text(encoding="utf-8"))
    crypto_route_source_payload = (
        json.loads(crypto_route_source_path.read_text(encoding="utf-8"))
        if crypto_route_source_path and crypto_route_source_path.exists()
        else None
    )
    crypto_route_refresh_source_payload = (
        json.loads(crypto_route_refresh_source_path.read_text(encoding="utf-8"))
        if crypto_route_refresh_source_path and crypto_route_refresh_source_path.exists()
        else None
    )
    remote_live_history_audit_source_payload = (
        json.loads(remote_live_history_audit_source_path.read_text(encoding="utf-8"))
        if remote_live_history_audit_source_path and remote_live_history_audit_source_path.exists()
        else None
    )
    remote_live_handoff_source_payload = (
        json.loads(remote_live_handoff_source_path.read_text(encoding="utf-8"))
        if remote_live_handoff_source_path and remote_live_handoff_source_path.exists()
        else None
    )
    live_gate_blocker_source_payload = (
        json.loads(live_gate_blocker_source_path.read_text(encoding="utf-8"))
        if live_gate_blocker_source_path and live_gate_blocker_source_path.exists()
        else None
    )
    brooks_route_report_source_payload = (
        json.loads(brooks_route_report_source_path.read_text(encoding="utf-8"))
        if brooks_route_report_source_path and brooks_route_report_source_path.exists()
        else None
    )
    brooks_execution_plan_source_payload = (
        json.loads(brooks_execution_plan_source_path.read_text(encoding="utf-8"))
        if brooks_execution_plan_source_path and brooks_execution_plan_source_path.exists()
        else None
    )
    brooks_structure_review_queue_source_payload = (
        json.loads(brooks_structure_review_queue_source_path.read_text(encoding="utf-8"))
        if brooks_structure_review_queue_source_path and brooks_structure_review_queue_source_path.exists()
        else None
    )
    brooks_structure_refresh_source_payload = (
        json.loads(brooks_structure_refresh_source_path.read_text(encoding="utf-8"))
        if brooks_structure_refresh_source_path and brooks_structure_refresh_source_path.exists()
        else None
    )
    cross_market_operator_state_source_payload = (
        json.loads(cross_market_operator_state_source_path.read_text(encoding="utf-8"))
        if cross_market_operator_state_source_path and cross_market_operator_state_source_path.exists()
        else None
    )
    system_time_sync_repair_plan_source_payload = (
        json.loads(system_time_sync_repair_plan_source_path.read_text(encoding="utf-8"))
        if system_time_sync_repair_plan_source_path and system_time_sync_repair_plan_source_path.exists()
        else None
    )
    system_time_sync_repair_verification_source_payload = (
        json.loads(system_time_sync_repair_verification_source_path.read_text(encoding="utf-8"))
        if system_time_sync_repair_verification_source_path
        and system_time_sync_repair_verification_source_path.exists()
        else None
    )
    openclaw_orderflow_blueprint_source_payload = (
        json.loads(openclaw_orderflow_blueprint_source_path.read_text(encoding="utf-8"))
        if openclaw_orderflow_blueprint_source_path and openclaw_orderflow_blueprint_source_path.exists()
        else None
    )
    remote_live_handoff_operator_payload = _unwrap_remote_live_handoff_operator_payload(
        remote_live_handoff_source_payload
    )
    commodity_source_payload = (
        _normalize_commodity_payload(json.loads(commodity_source_path.read_text(encoding="utf-8")))
        if commodity_source_path
        else None
    )
    commodity_ticket_source_payload = (
        json.loads(commodity_ticket_source_path.read_text(encoding="utf-8"))
        if commodity_ticket_source_path
        else None
    )
    commodity_ticket_book_source_payload = (
        json.loads(commodity_ticket_book_source_path.read_text(encoding="utf-8"))
        if commodity_ticket_book_source_path
        else None
    )
    commodity_execution_preview_source_payload = (
        json.loads(commodity_execution_preview_source_path.read_text(encoding="utf-8"))
        if commodity_execution_preview_source_path
        else None
    )
    commodity_execution_artifact_source_payload = (
        json.loads(commodity_execution_artifact_source_path.read_text(encoding="utf-8"))
        if commodity_execution_artifact_source_path
        else None
    )
    commodity_execution_queue_source_payload = (
        json.loads(commodity_execution_queue_source_path.read_text(encoding="utf-8"))
        if commodity_execution_queue_source_path
        else None
    )
    commodity_execution_review_source_payload = (
        json.loads(commodity_execution_review_source_path.read_text(encoding="utf-8"))
        if commodity_execution_review_source_path
        else None
    )
    commodity_execution_retro_source_payload = (
        json.loads(commodity_execution_retro_source_path.read_text(encoding="utf-8"))
        if commodity_execution_retro_source_path
        else None
    )
    commodity_execution_gap_source_payload = (
        json.loads(commodity_execution_gap_source_path.read_text(encoding="utf-8"))
        if commodity_execution_gap_source_path
        else None
    )
    commodity_execution_bridge_source_payload = (
        json.loads(commodity_execution_bridge_source_path.read_text(encoding="utf-8"))
        if commodity_execution_bridge_source_path
        else None
    )
    research_embedding_quality = _research_embedding_quality(action_source_payload)
    crypto_route_refresh_reuse_audit = _crypto_route_refresh_reuse_audit(crypto_route_refresh_source_payload)
    crypto_route_refresh_reuse_gate = _crypto_route_refresh_reuse_gate(
        crypto_route_refresh_source_payload,
        crypto_route_refresh_reuse_audit,
    )
    remote_live_history_window_map = _remote_live_history_window_map(remote_live_history_audit_source_payload)
    remote_live_history_snapshot = _remote_live_history_snapshot_row(remote_live_history_window_map)
    remote_live_history_longest = _remote_live_history_longest_row(remote_live_history_window_map)
    remote_live_history_window_brief = " | ".join(
        [
            brief_text
            for brief_text in (
                _remote_live_history_window_brief(remote_live_history_window_map.get(24, {})),
                _remote_live_history_window_brief(remote_live_history_window_map.get(168, {})),
                _remote_live_history_window_brief(remote_live_history_window_map.get(720, {})),
            )
            if brief_text != "-"
        ]
    )
    brooks_route_candidates = (
        list((brooks_route_report_source_payload or {}).get("current_candidates") or [])
        if isinstance((brooks_route_report_source_payload or {}).get("current_candidates"), list)
        else []
    )
    brooks_route_head = (
        dict(brooks_route_candidates[0]) if brooks_route_candidates and isinstance(brooks_route_candidates[0], dict) else {}
    )
    brooks_execution_head = (
        dict((brooks_execution_plan_source_payload or {}).get("head_plan_item") or {})
        if isinstance((brooks_execution_plan_source_payload or {}).get("head_plan_item"), dict)
        else {}
    )
    brooks_structure_review_lane = (
        {
            "status": str((brooks_structure_review_queue_source_payload or {}).get("review_status") or ""),
            "brief": str((brooks_structure_review_queue_source_payload or {}).get("review_brief") or ""),
            "queue_status": str((brooks_structure_review_queue_source_payload or {}).get("queue_status") or ""),
            "queue_count": int((brooks_structure_review_queue_source_payload or {}).get("queue_count") or 0),
            "queue": [
                dict(row)
                for row in list((brooks_structure_review_queue_source_payload or {}).get("queue") or [])
                if isinstance(row, dict)
            ],
            "queue_brief": str((brooks_structure_review_queue_source_payload or {}).get("queue_brief") or ""),
            "head": dict((brooks_structure_review_queue_source_payload or {}).get("head") or {})
            if isinstance((brooks_structure_review_queue_source_payload or {}).get("head"), dict)
            else {},
            "priority_status": str((brooks_structure_review_queue_source_payload or {}).get("priority_status") or ""),
            "priority_brief": str((brooks_structure_review_queue_source_payload or {}).get("priority_brief") or ""),
            "blocker_detail": str((brooks_structure_review_queue_source_payload or {}).get("blocker_detail") or ""),
            "done_when": str((brooks_structure_review_queue_source_payload or {}).get("done_when") or ""),
        }
        if brooks_structure_review_queue_source_payload
        else _brooks_structure_review_lane(
            route_report_payload=brooks_route_report_source_payload,
            execution_plan_payload=brooks_execution_plan_source_payload,
        )
    )
    brooks_structure_operator_lane = _brooks_structure_operator_lane(
        queue_payload=brooks_structure_review_queue_source_payload,
        fallback_lane=brooks_structure_review_lane,
    )
    cross_market_review_head_lane = _cross_market_review_head_lane(
        source_payload=cross_market_operator_state_source_payload,
    )
    cross_market_operator_head_lane = _cross_market_operator_head_lane(
        source_payload=cross_market_operator_state_source_payload,
    )
    cross_market_operator_repair_head_lane = _cross_market_operator_repair_head_lane(
        source_payload=cross_market_operator_state_source_payload,
        live_gate_blocker_source_payload=live_gate_blocker_source_payload,
    )
    cross_market_source_snapshot_chunk = _build_cross_market_source_snapshot_chunk(
        cross_market_operator_state_source_payload
    )
    source_cross_market_operator_state_operator_snapshot_brief = str(
        cross_market_source_snapshot_chunk.get("source_cross_market_operator_state_operator_snapshot_brief")
        or ""
    )
    source_cross_market_operator_state_review_snapshot_brief = str(
        cross_market_source_snapshot_chunk.get("source_cross_market_operator_state_review_snapshot_brief")
        or ""
    )
    source_cross_market_operator_state_remote_live_snapshot_brief = str(
        cross_market_source_snapshot_chunk.get(
            "source_cross_market_operator_state_remote_live_snapshot_brief"
        )
        or ""
    )
    source_cross_market_operator_state_snapshot_brief = str(
        cross_market_source_snapshot_chunk.get("source_cross_market_operator_state_snapshot_brief")
        or ""
    )
    brief = build_operator_brief(
        action_source_payload,
        crypto_source_payload,
        crypto_route_source_payload,
        commodity_source_payload,
        commodity_ticket_source_payload,
        commodity_ticket_book_source_payload,
        commodity_execution_preview_source_payload,
        commodity_execution_artifact_source_payload,
        commodity_execution_queue_source_payload,
        commodity_execution_review_source_payload,
        commodity_execution_retro_source_payload,
        commodity_execution_gap_source_payload,
        commodity_execution_bridge_source_payload,
        cross_market_operator_state_source_payload,
        live_gate_blocker_source_payload,
    )

    source_mode = "single-hot-universe-source"
    unique_sources = {str(action_source_path), str(crypto_source_path)}
    if commodity_source_path:
        unique_sources.add(str(commodity_source_path))
    if commodity_ticket_source_path:
        unique_sources.add(str(commodity_ticket_source_path))
    if commodity_ticket_book_source_path:
        unique_sources.add(str(commodity_ticket_book_source_path))
    if commodity_execution_preview_source_path:
        unique_sources.add(str(commodity_execution_preview_source_path))
    if commodity_execution_artifact_source_path:
        unique_sources.add(str(commodity_execution_artifact_source_path))
    if commodity_execution_queue_source_path:
        unique_sources.add(str(commodity_execution_queue_source_path))
    if commodity_execution_review_source_path:
        unique_sources.add(str(commodity_execution_review_source_path))
    if commodity_execution_retro_source_path:
        unique_sources.add(str(commodity_execution_retro_source_path))
    if commodity_execution_gap_source_path:
        unique_sources.add(str(commodity_execution_gap_source_path))
    if commodity_execution_bridge_source_path:
        unique_sources.add(str(commodity_execution_bridge_source_path))
    if brooks_route_report_source_path:
        unique_sources.add(str(brooks_route_report_source_path))
    if brooks_execution_plan_source_path:
        unique_sources.add(str(brooks_execution_plan_source_path))
    if brooks_structure_review_queue_source_path:
        unique_sources.add(str(brooks_structure_review_queue_source_path))
    if brooks_structure_refresh_source_path:
        unique_sources.add(str(brooks_structure_refresh_source_path))
    if cross_market_operator_state_source_path:
        unique_sources.add(str(cross_market_operator_state_source_path))
    if len(unique_sources) > 1:
        if commodity_source_path or commodity_ticket_source_path or commodity_ticket_book_source_path:
            source_mode = "merged-action-commodity-crypto-sources"
        else:
            source_mode = "merged-action-crypto-sources"

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_hot_universe_operator_brief.json"
    md_path = review_dir / f"{stamp}_hot_universe_operator_brief.md"
    checksum_path = review_dir / f"{stamp}_hot_universe_operator_brief_checksum.json"

    crypto_route_source_chunk = _build_crypto_route_source_chunk(
        crypto_route_source_path=crypto_route_source_path,
        crypto_route_source_payload=crypto_route_source_payload,
        crypto_route_refresh_source_path=crypto_route_refresh_source_path,
        crypto_route_refresh_source_payload=crypto_route_refresh_source_payload,
        crypto_route_refresh_reuse_audit=crypto_route_refresh_reuse_audit,
        crypto_route_refresh_reuse_gate=crypto_route_refresh_reuse_gate,
    )
    remote_live_source_chunk = _build_remote_live_source_chunk(
        remote_live_history_audit_source_path=remote_live_history_audit_source_path,
        remote_live_history_audit_source_payload=remote_live_history_audit_source_payload,
        remote_live_history_window_brief=remote_live_history_window_brief,
        remote_live_history_snapshot=remote_live_history_snapshot,
        remote_live_history_window_map=remote_live_history_window_map,
        remote_live_history_longest=remote_live_history_longest,
        remote_live_handoff_source_path=remote_live_handoff_source_path,
        remote_live_handoff_source_payload=remote_live_handoff_source_payload,
        remote_live_handoff_operator_payload=remote_live_handoff_operator_payload,
        live_gate_blocker_source_path=live_gate_blocker_source_path,
        live_gate_blocker_source_payload=live_gate_blocker_source_payload,
        mapping_text_fn=_mapping_text,
    )
    brooks_source_chunk = _build_brooks_source_chunk(
        brooks_route_report_source_path=brooks_route_report_source_path,
        brooks_route_report_source_payload=brooks_route_report_source_payload,
        brooks_route_head=brooks_route_head,
        brooks_execution_plan_source_path=brooks_execution_plan_source_path,
        brooks_execution_plan_source_payload=brooks_execution_plan_source_payload,
        brooks_execution_head=brooks_execution_head,
        brooks_structure_review_queue_source_path=brooks_structure_review_queue_source_path,
        brooks_structure_review_queue_source_payload=brooks_structure_review_queue_source_payload,
        brooks_structure_refresh_source_path=brooks_structure_refresh_source_path,
        brooks_structure_refresh_source_payload=brooks_structure_refresh_source_payload,
    )
    cross_market_runtime_chunk = _build_cross_market_runtime_chunk(
        cross_market_operator_state_source_payload=cross_market_operator_state_source_payload,
        cross_market_operator_head_lane=cross_market_operator_head_lane,
        cross_market_operator_repair_head_lane=cross_market_operator_repair_head_lane,
        cross_market_review_head_lane=cross_market_review_head_lane,
        live_gate_blocker_source_payload=live_gate_blocker_source_payload,
    )
    brooks_runtime_chunk = _build_brooks_runtime_chunk(
        brooks_structure_review_lane=brooks_structure_review_lane,
        brooks_structure_operator_lane=brooks_structure_operator_lane,
    )
    research_embedding_quality_chunk = _build_research_embedding_quality_chunk(
        research_embedding_quality
    )
    openclaw_orderflow_blueprint_current = dict(
        (openclaw_orderflow_blueprint_source_payload or {}).get("current_status") or {}
    )
    openclaw_orderflow_blueprint_backlog = list(
        (openclaw_orderflow_blueprint_source_payload or {}).get("immediate_backlog") or []
    )
    openclaw_orderflow_blueprint_top = (
        dict(openclaw_orderflow_blueprint_backlog[0])
        if openclaw_orderflow_blueprint_backlog
        and isinstance(openclaw_orderflow_blueprint_backlog[0], dict)
        else {}
    )
    openclaw_orderflow_blueprint_brief = " | ".join(
        [
            " -> ".join(
                [
                    str(openclaw_orderflow_blueprint_current.get("current_life_stage") or "").strip(),
                    str(openclaw_orderflow_blueprint_current.get("target_life_stage") or "").strip(),
                ]
            ).strip(" ->"),
            ":".join(
                [
                    f"P{str(openclaw_orderflow_blueprint_top.get('priority') or '').strip()}".strip(":"),
                    str(openclaw_orderflow_blueprint_top.get("target_artifact") or "").strip(),
                    str(openclaw_orderflow_blueprint_top.get("title") or "").strip(),
                ]
            ).strip(":"),
        ]
    ).strip(" | ")

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_mode": source_mode,
        "source_artifact": str(action_source_path if action_source_path == crypto_source_path else crypto_source_path),
        "source_status": str(crypto_source_payload.get("status") or ""),
        "source_action_artifact": str(action_source_path),
        "source_action_status": str(action_source_payload.get("status") or ""),
        "source_commodity_artifact": str(commodity_source_path) if commodity_source_path else "",
        "source_commodity_status": str((commodity_source_payload or {}).get("status") or ""),
        "source_commodity_ticket_artifact": str(commodity_ticket_source_path) if commodity_ticket_source_path else "",
        "source_commodity_ticket_status": str((commodity_ticket_source_payload or {}).get("status") or ""),
        "source_commodity_ticket_book_artifact": str(commodity_ticket_book_source_path) if commodity_ticket_book_source_path else "",
        "source_commodity_ticket_book_status": str((commodity_ticket_book_source_payload or {}).get("status") or ""),
        "source_commodity_execution_preview_artifact": str(commodity_execution_preview_source_path) if commodity_execution_preview_source_path else "",
        "source_commodity_execution_preview_status": str((commodity_execution_preview_source_payload or {}).get("status") or ""),
        "source_commodity_execution_artifact": str(commodity_execution_artifact_source_path) if commodity_execution_artifact_source_path else "",
        "source_commodity_execution_artifact_status": str((commodity_execution_artifact_source_payload or {}).get("status") or ""),
        "source_commodity_execution_queue_artifact": str(commodity_execution_queue_source_path) if commodity_execution_queue_source_path else "",
        "source_commodity_execution_queue_status": str((commodity_execution_queue_source_payload or {}).get("status") or ""),
        "source_commodity_execution_review_artifact": str(commodity_execution_review_source_path) if commodity_execution_review_source_path else "",
        "source_commodity_execution_review_status": str((commodity_execution_review_source_payload or {}).get("status") or ""),
        "source_commodity_execution_retro_artifact": str(commodity_execution_retro_source_path) if commodity_execution_retro_source_path else "",
        "source_commodity_execution_retro_status": str((commodity_execution_retro_source_payload or {}).get("status") or ""),
        "source_commodity_execution_gap_artifact": str(commodity_execution_gap_source_path) if commodity_execution_gap_source_path else "",
        "source_commodity_execution_gap_status": str((commodity_execution_gap_source_payload or {}).get("status") or ""),
        "source_commodity_execution_bridge_artifact": str(commodity_execution_bridge_source_path) if commodity_execution_bridge_source_path else "",
        "source_commodity_execution_bridge_status": str((commodity_execution_bridge_source_payload or {}).get("status") or ""),
        "source_crypto_artifact": str(crypto_source_path),
        "source_crypto_status": str(crypto_source_payload.get("status") or ""),
        **crypto_route_source_chunk,
        **remote_live_source_chunk,
        **brooks_source_chunk,
        "source_cross_market_operator_state_artifact": str(
            cross_market_operator_state_source_path
        )
        if cross_market_operator_state_source_path
        else "",
        "source_system_time_sync_repair_plan_artifact": str(system_time_sync_repair_plan_source_path)
        if system_time_sync_repair_plan_source_path
        else "",
        "source_system_time_sync_repair_plan_status": str(
            (system_time_sync_repair_plan_source_payload or {}).get("status") or ""
        ),
        "source_system_time_sync_repair_plan_brief": str(
            (system_time_sync_repair_plan_source_payload or {}).get("plan_brief") or ""
        ),
        "source_system_time_sync_repair_plan_admin_required": bool(
            (system_time_sync_repair_plan_source_payload or {}).get("admin_required", False)
        ),
        "source_system_time_sync_repair_plan_done_when": str(
            (system_time_sync_repair_plan_source_payload or {}).get("done_when") or ""
        ),
        "source_system_time_sync_repair_verification_artifact": str(
            system_time_sync_repair_verification_source_path
        )
        if system_time_sync_repair_verification_source_path
        else "",
        "source_system_time_sync_repair_verification_status": str(
            (system_time_sync_repair_verification_source_payload or {}).get("status") or ""
        ),
        "source_system_time_sync_repair_verification_brief": str(
            (system_time_sync_repair_verification_source_payload or {}).get("verification_brief") or ""
        ),
        "source_system_time_sync_repair_verification_cleared": bool(
            (system_time_sync_repair_verification_source_payload or {}).get("cleared", False)
        ),
        "source_openclaw_orderflow_blueprint_artifact": str(openclaw_orderflow_blueprint_source_path)
        if openclaw_orderflow_blueprint_source_path
        else "",
        "source_openclaw_orderflow_blueprint_status": str(
            (openclaw_orderflow_blueprint_source_payload or {}).get("status") or ""
        ),
        "source_openclaw_orderflow_blueprint_brief": openclaw_orderflow_blueprint_brief,
        "source_openclaw_orderflow_blueprint_current_life_stage": str(
            openclaw_orderflow_blueprint_current.get("current_life_stage") or ""
        ),
        "source_openclaw_orderflow_blueprint_target_life_stage": str(
            openclaw_orderflow_blueprint_current.get("target_life_stage") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_intent_queue_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_intent_queue_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_intent_queue_status": str(
            openclaw_orderflow_blueprint_current.get("remote_intent_queue_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_intent_queue_recommendation": str(
            openclaw_orderflow_blueprint_current.get("remote_intent_queue_recommendation") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_journal_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_journal_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_journal_status": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_journal_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_journal_append_status": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_journal_append_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_feedback_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_feedback_status": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_feedback_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_feedback_recommendation": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_feedback_recommendation") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_policy_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_policy_status": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_policy_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_policy_decision": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_policy_decision") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_ack_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_ack_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_ack_status": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_ack_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_ack_decision": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_ack_decision") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_actor_canary_gate_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_status": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_actor_canary_gate_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_decision": str(
            openclaw_orderflow_blueprint_current.get("remote_execution_actor_canary_gate_decision") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_quality_report_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_status": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_quality_report_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_recommendation": str(
            openclaw_orderflow_blueprint_current.get("remote_orderflow_quality_report_recommendation") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_score": (
            openclaw_orderflow_blueprint_current.get("remote_orderflow_quality_report_score")
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_quality_shadow_learning_score": (
            openclaw_orderflow_blueprint_current.get("remote_orderflow_quality_shadow_learning_score")
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_quality_execution_readiness_score": (
            openclaw_orderflow_blueprint_current.get("remote_orderflow_quality_execution_readiness_score")
        ),
        "source_openclaw_orderflow_blueprint_remote_orderflow_quality_transport_observability_score": (
            openclaw_orderflow_blueprint_current.get("remote_orderflow_quality_transport_observability_score")
        ),
        "source_openclaw_orderflow_blueprint_remote_live_boundary_hold_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_live_boundary_hold_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_live_boundary_hold_status": str(
            openclaw_orderflow_blueprint_current.get("remote_live_boundary_hold_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_live_boundary_hold_decision": str(
            openclaw_orderflow_blueprint_current.get("remote_live_boundary_hold_decision") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_live_boundary_hold_next_transition": str(
            openclaw_orderflow_blueprint_current.get("remote_live_boundary_hold_next_transition") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_guarded_canary_promotion_gate_brief")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_status": str(
            openclaw_orderflow_blueprint_current.get("remote_guarded_canary_promotion_gate_status")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_decision": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_guarded_canary_promotion_gate_decision"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_title": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_guarded_canary_promotion_gate_blocker_title"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_guarded_canary_promotion_gate_blocker_code"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_target_artifact": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_guarded_canary_promotion_gate_blocker_target_artifact"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_shadow_learning_continuity_brief")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_status": str(
            openclaw_orderflow_blueprint_current.get("remote_shadow_learning_continuity_status")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_decision": str(
            openclaw_orderflow_blueprint_current.get("remote_shadow_learning_continuity_decision")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_blocker_detail": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_shadow_learning_continuity_blocker_detail"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_promotion_unblock_readiness_brief")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_status": str(
            openclaw_orderflow_blueprint_current.get("remote_promotion_unblock_readiness_status")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_decision": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_promotion_unblock_readiness_decision"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_blocker_scope": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_promotion_unblock_primary_blocker_scope"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_title": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_promotion_unblock_primary_local_repair_title"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_target_artifact": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_promotion_unblock_primary_local_repair_target_artifact"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_plan_brief": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_promotion_unblock_primary_local_repair_plan_brief"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_environment_classification": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_promotion_unblock_primary_local_repair_environment_classification"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_environment_blocker_detail": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_promotion_unblock_primary_local_repair_environment_blocker_detail"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_ticket_actionability_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_ticket_actionability_brief")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_ticket_actionability_status": str(
            openclaw_orderflow_blueprint_current.get("remote_ticket_actionability_status")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_ticket_actionability_decision": str(
            openclaw_orderflow_blueprint_current.get("remote_ticket_actionability_decision")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_ticket_actionability_next_action": str(
            openclaw_orderflow_blueprint_current.get("remote_ticket_actionability_next_action")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_ticket_actionability_next_action_target_artifact": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_ticket_actionability_next_action_target_artifact"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_brief": str(
            openclaw_orderflow_blueprint_current.get("crypto_shortline_backtest_slice_brief")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_status": str(
            openclaw_orderflow_blueprint_current.get("crypto_shortline_backtest_slice_status")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_decision": str(
            openclaw_orderflow_blueprint_current.get("crypto_shortline_backtest_slice_decision")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_selected_symbol": str(
            openclaw_orderflow_blueprint_current.get(
                "crypto_shortline_backtest_slice_selected_symbol"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_universe_brief": str(
            openclaw_orderflow_blueprint_current.get(
                "crypto_shortline_backtest_slice_universe_brief"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_brief": str(
            openclaw_orderflow_blueprint_current.get(
                "crypto_shortline_cross_section_backtest_brief"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_status": str(
            openclaw_orderflow_blueprint_current.get(
                "crypto_shortline_cross_section_backtest_status"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_decision": str(
            openclaw_orderflow_blueprint_current.get(
                "crypto_shortline_cross_section_backtest_decision"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_selected_edge_status": str(
            openclaw_orderflow_blueprint_current.get(
                "crypto_shortline_cross_section_backtest_selected_edge_status"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_time_sync_mode": str(
            openclaw_orderflow_blueprint_current.get("remote_time_sync_mode") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_shadow_clock_evidence_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_shadow_clock_evidence_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_shadow_clock_evidence_status": str(
            openclaw_orderflow_blueprint_current.get("remote_shadow_clock_evidence_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_shadow_clock_shadow_learning_allowed": bool(
            openclaw_orderflow_blueprint_current.get(
                "remote_shadow_clock_shadow_learning_allowed", False
            )
        ),
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_brief": str(
            openclaw_orderflow_blueprint_current.get("remote_guardian_blocker_clearance_brief") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_status": str(
            openclaw_orderflow_blueprint_current.get("remote_guardian_blocker_clearance_status") or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_score": (
            openclaw_orderflow_blueprint_current.get("remote_guardian_blocker_clearance_score")
        ),
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code": str(
            openclaw_orderflow_blueprint_current.get("remote_guardian_blocker_clearance_top_blocker_code")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_title": str(
            openclaw_orderflow_blueprint_current.get("remote_guardian_blocker_clearance_top_blocker_title")
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_target_artifact": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_guardian_blocker_clearance_top_blocker_target_artifact"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_next_action": str(
            openclaw_orderflow_blueprint_current.get(
                "remote_guardian_blocker_clearance_top_blocker_next_action"
            )
            or ""
        ),
        "source_openclaw_orderflow_blueprint_top_backlog_title": str(
            openclaw_orderflow_blueprint_top.get("title") or ""
        ),
        "source_openclaw_orderflow_blueprint_top_backlog_target_artifact": str(
            openclaw_orderflow_blueprint_top.get("target_artifact") or ""
        ),
        "source_openclaw_orderflow_blueprint_top_backlog_why": str(
            openclaw_orderflow_blueprint_top.get("why") or ""
        ),
        **_mirror_source_prefixed_fields(
            prefix="source_cross_market_operator_state_",
            source_payload=cross_market_operator_state_source_payload,
            specs=_CROSS_MARKET_SOURCE_MIRROR_SPECS,
        ),
        **cross_market_source_snapshot_chunk,
        **cross_market_runtime_chunk,
        **brooks_runtime_chunk,
        **research_embedding_quality_chunk,
        **brief,
    }
    crypto_route_alignment_focus = _crypto_route_alignment_focus(payload)
    crypto_route_alignment = _crypto_route_embedding_alignment(
        secondary_focus_area=str(crypto_route_alignment_focus.get("area") or ""),
        secondary_focus_symbol=str(crypto_route_alignment_focus.get("symbol") or ""),
        secondary_focus_action=str(crypto_route_alignment_focus.get("action") or ""),
        quality_brief=str(payload.get("operator_research_embedding_quality_brief") or ""),
        active_batches=list(payload.get("operator_research_embedding_active_batches") or []),
    )
    payload["operator_crypto_route_alignment_focus_area"] = str(
        crypto_route_alignment_focus.get("area") or "-"
    )
    payload["operator_crypto_route_alignment_focus_slot"] = str(
        crypto_route_alignment_focus.get("slot") or "-"
    )
    payload["operator_crypto_route_alignment_focus_symbol"] = str(
        crypto_route_alignment_focus.get("symbol") or "-"
    )
    payload["operator_crypto_route_alignment_focus_action"] = str(
        crypto_route_alignment_focus.get("action") or "-"
    )
    payload["operator_crypto_route_alignment_status"] = str(crypto_route_alignment.get("status") or "")
    payload["operator_crypto_route_alignment_brief"] = str(crypto_route_alignment.get("brief") or "")
    payload["operator_crypto_route_alignment_blocker_detail"] = str(
        crypto_route_alignment.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_done_when"] = str(crypto_route_alignment.get("done_when") or "")
    crypto_route_alignment_recipe = _crypto_route_alignment_recovery_recipe(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        source_artifact=str(payload.get("source_crypto_artifact") or ""),
        avoid_batches=list(payload.get("operator_research_embedding_avoid_batches") or []),
        zero_trade_deprioritized_batches=list(
            payload.get("operator_research_embedding_zero_trade_deprioritized_batches") or []
        ),
        reference_now=runtime_now,
    )
    payload["operator_crypto_route_alignment_recipe_script"] = str(
        crypto_route_alignment_recipe.get("script") or ""
    )
    payload["operator_crypto_route_alignment_recipe_command_hint"] = str(
        crypto_route_alignment_recipe.get("command_hint") or ""
    )
    payload["operator_crypto_route_alignment_recipe_expected_status"] = str(
        crypto_route_alignment_recipe.get("expected_status") or ""
    )
    payload["operator_crypto_route_alignment_recipe_note"] = str(
        crypto_route_alignment_recipe.get("note") or ""
    )
    payload["operator_crypto_route_alignment_recipe_followup_script"] = str(
        crypto_route_alignment_recipe.get("followup_script") or ""
    )
    payload["operator_crypto_route_alignment_recipe_followup_command_hint"] = str(
        crypto_route_alignment_recipe.get("followup_command_hint") or ""
    )
    payload["operator_crypto_route_alignment_recipe_verify_hint"] = str(
        crypto_route_alignment_recipe.get("verify_hint") or ""
    )
    payload["operator_crypto_route_alignment_recipe_window_days"] = crypto_route_alignment_recipe.get("window_days")
    payload["operator_crypto_route_alignment_recipe_target_batches"] = list(
        crypto_route_alignment_recipe.get("target_batches") or []
    )
    crypto_route_alignment_recovery_outcome = _crypto_route_alignment_recovery_outcome(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        quality_status=str(payload.get("operator_research_embedding_quality_status") or ""),
        source_status=str(payload.get("source_crypto_status") or ""),
        source_payload=crypto_source_payload if isinstance(crypto_source_payload, dict) else {},
    )
    payload["operator_crypto_route_alignment_recovery_status"] = str(
        crypto_route_alignment_recovery_outcome.get("status") or ""
    )
    payload["operator_crypto_route_alignment_recovery_brief"] = str(
        crypto_route_alignment_recovery_outcome.get("brief") or ""
    )
    payload["operator_crypto_route_alignment_recovery_blocker_detail"] = str(
        crypto_route_alignment_recovery_outcome.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_recovery_done_when"] = str(
        crypto_route_alignment_recovery_outcome.get("done_when") or ""
    )
    payload["operator_crypto_route_alignment_recovery_failed_batch_count"] = (
        crypto_route_alignment_recovery_outcome.get("failed_batch_count")
    )
    payload["operator_crypto_route_alignment_recovery_timed_out_batch_count"] = (
        crypto_route_alignment_recovery_outcome.get("timed_out_batch_count")
    )
    payload["operator_crypto_route_alignment_recovery_zero_trade_batches"] = list(
        crypto_route_alignment_recovery_outcome.get("zero_trade_batches") or []
    )
    crypto_route_alignment_cooldown = _crypto_route_alignment_cooldown(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        recovery_status=str(payload.get("operator_crypto_route_alignment_recovery_status") or ""),
        source_status=str(payload.get("source_crypto_status") or ""),
        source_payload=crypto_source_payload if isinstance(crypto_source_payload, dict) else {},
        reference_now=runtime_now,
    )
    payload["operator_crypto_route_alignment_cooldown_status"] = str(
        crypto_route_alignment_cooldown.get("status") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_brief"] = str(
        crypto_route_alignment_cooldown.get("brief") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_blocker_detail"] = str(
        crypto_route_alignment_cooldown.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_done_when"] = str(
        crypto_route_alignment_cooldown.get("done_when") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_last_research_end_date"] = str(
        crypto_route_alignment_cooldown.get("last_research_end_date") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_next_eligible_end_date"] = str(
        crypto_route_alignment_cooldown.get("next_eligible_end_date") or ""
    )
    crypto_route_alignment_recipe_gate = _crypto_route_alignment_recipe_gate(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        recipe_script=str(payload.get("operator_crypto_route_alignment_recipe_script") or ""),
        cooldown_status=str(payload.get("operator_crypto_route_alignment_cooldown_status") or ""),
        cooldown_brief=str(payload.get("operator_crypto_route_alignment_cooldown_brief") or ""),
        cooldown_blocker_detail=str(payload.get("operator_crypto_route_alignment_cooldown_blocker_detail") or ""),
        cooldown_done_when=str(payload.get("operator_crypto_route_alignment_cooldown_done_when") or ""),
        cooldown_next_eligible_end_date=str(
            payload.get("operator_crypto_route_alignment_cooldown_next_eligible_end_date") or ""
        ),
    )
    payload["operator_crypto_route_alignment_recipe_status"] = str(
        crypto_route_alignment_recipe_gate.get("status") or ""
    )
    payload["operator_crypto_route_alignment_recipe_brief"] = str(
        crypto_route_alignment_recipe_gate.get("brief") or ""
    )
    payload["operator_crypto_route_alignment_recipe_blocker_detail"] = str(
        crypto_route_alignment_recipe_gate.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_recipe_done_when"] = str(
        crypto_route_alignment_recipe_gate.get("done_when") or ""
    )
    payload["operator_crypto_route_alignment_recipe_ready_on_date"] = str(
        crypto_route_alignment_recipe_gate.get("ready_on_date") or ""
    )
    focus_slots = []
    for row in payload.get("operator_focus_slots", []):
        if not isinstance(row, dict):
            continue
        slot = dict(row)
        source_kind = str(slot.get("source_kind") or "").strip()
        source_artifact = str(slot.get("source_artifact") or "").strip()
        if not source_kind or not source_artifact:
            derived_source_kind, derived_source_artifact = _focus_slot_source(
                payload,
                area=str(slot.get("area") or ""),
                action=str(slot.get("action") or ""),
            )
            if not source_kind:
                source_kind = derived_source_kind
            if not source_artifact:
                source_artifact = derived_source_artifact
        slot["source_kind"] = source_kind
        slot["source_artifact"] = source_artifact
        focus_slots.append(slot)
    if focus_slots:
        payload["operator_focus_slots"] = focus_slots
        payload["operator_focus_slots_brief"] = _focus_slots_brief(focus_slots)
        payload["operator_focus_slot_sources_brief"] = _focus_slot_source_brief(focus_slots)
        enriched_slots: list[dict[str, Any]] = []
        for slot in focus_slots:
            enriched = dict(slot)
            if not str(enriched.get("source_status") or "").strip():
                enriched["source_status"] = _focus_slot_source_status(
                    payload,
                    source_kind=str(enriched.get("source_kind") or ""),
                )
            if not str(enriched.get("source_as_of") or "").strip():
                enriched["source_as_of"] = _focus_slot_source_as_of(str(enriched.get("source_artifact") or ""))
            if enriched.get("source_age_minutes") in (None, ""):
                enriched["source_age_minutes"] = _focus_slot_source_age_minutes(
                    reference_now=runtime_now,
                    source_as_of=str(enriched.get("source_as_of") or ""),
                )
            if not str(enriched.get("source_recency") or "").strip():
                enriched["source_recency"] = _focus_slot_source_recency(
                    source_age_minutes=enriched.get("source_age_minutes")
                )
            if not str(enriched.get("source_health") or "").strip():
                enriched["source_health"] = _focus_slot_source_health(
                    source_status=str(enriched.get("source_status") or ""),
                    source_recency=str(enriched.get("source_recency") or ""),
                    source_kind=str(enriched.get("source_kind") or ""),
                    crypto_route_alignment_cooldown_status=str(
                        payload.get("operator_crypto_route_alignment_cooldown_status") or ""
                    ),
                )
            if not str(enriched.get("source_refresh_action") or "").strip():
                enriched["source_refresh_action"] = _focus_slot_source_refresh_action(
                    source_health=str(enriched.get("source_health") or ""),
                    source_kind=str(enriched.get("source_kind") or ""),
                    crypto_route_alignment_cooldown_status=str(
                        payload.get("operator_crypto_route_alignment_cooldown_status") or ""
                    ),
                )
            enriched_slots.append(enriched)
        focus_slots = enriched_slots
        payload["operator_focus_slots"] = focus_slots
        payload["operator_focus_slots_brief"] = _focus_slots_brief(focus_slots)
        payload["operator_focus_slot_sources_brief"] = _focus_slot_source_brief(focus_slots)
        payload["operator_focus_slot_status_brief"] = _focus_slot_status_brief(focus_slots)
        payload["operator_focus_slot_recency_brief"] = _focus_slot_recency_brief(focus_slots)
        payload["operator_focus_slot_health_brief"] = _focus_slot_health_brief(focus_slots)
        crypto_route_alignment_focus = _crypto_route_alignment_focus(payload, focus_slots=focus_slots)
        payload["operator_crypto_route_alignment_focus_area"] = str(
            crypto_route_alignment_focus.get("area") or "-"
        )
        payload["operator_crypto_route_alignment_focus_slot"] = str(
            crypto_route_alignment_focus.get("slot") or "-"
        )
        payload["operator_crypto_route_alignment_focus_symbol"] = str(
            crypto_route_alignment_focus.get("symbol") or "-"
        )
        payload["operator_crypto_route_alignment_focus_action"] = str(
            crypto_route_alignment_focus.get("action") or "-"
        )
        refresh_backlog = _focus_slot_refresh_backlog(focus_slots)
        source_refresh_queue = _source_refresh_queue(refresh_backlog)
        source_refresh_checklist = _source_refresh_checklist(source_refresh_queue, reference_now=runtime_now)
        payload["operator_focus_slot_refresh_backlog"] = refresh_backlog
        payload["operator_focus_slot_refresh_backlog_brief"] = _focus_slot_refresh_backlog_brief(refresh_backlog)
        payload["operator_focus_slot_refresh_backlog_count"] = len(refresh_backlog)
        payload["operator_source_refresh_queue"] = source_refresh_queue
        payload["operator_source_refresh_queue_brief"] = _source_refresh_queue_brief(source_refresh_queue)
        payload["operator_source_refresh_queue_count"] = len(source_refresh_queue)
        payload["operator_source_refresh_checklist"] = source_refresh_checklist
        payload["operator_source_refresh_checklist_brief"] = _source_refresh_checklist_brief(
            source_refresh_checklist
        )
        ready_count = sum(
            1
            for row in focus_slots
            if str(row.get("source_refresh_action") or "") in {"read_current_artifact", "wait_for_next_eligible_end_date"}
        )
        total_count = len(focus_slots)
        promotion_gate_status = _focus_slot_promotion_gate_status(
            total_count=total_count,
            ready_count=ready_count,
        )
        payload["operator_focus_slot_ready_count"] = ready_count
        payload["operator_focus_slot_total_count"] = total_count
        payload["operator_focus_slot_promotion_gate_brief"] = _focus_slot_promotion_gate_brief(
            status=promotion_gate_status,
            ready_count=ready_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_promotion_gate_status"] = promotion_gate_status
        payload["operator_focus_slot_promotion_gate_blocker_detail"] = _focus_slot_promotion_gate_blocker_detail(
            total_count=total_count,
            ready_count=ready_count,
            refresh_backlog=refresh_backlog,
        )
        payload["operator_focus_slot_promotion_gate_done_when"] = _focus_slot_promotion_gate_done_when(
            total_count=total_count,
            ready_count=ready_count,
        )
        actionability_backlog = _focus_slot_actionability_backlog(
            focus_slots,
            alignment_focus_slot=str(payload.get("operator_crypto_route_alignment_focus_slot") or ""),
            alignment_focus_symbol=str(payload.get("operator_crypto_route_alignment_focus_symbol") or ""),
            alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
            alignment_brief=str(payload.get("operator_crypto_route_alignment_brief") or ""),
            alignment_recovery_status=str(payload.get("operator_crypto_route_alignment_recovery_status") or ""),
            alignment_recovery_brief=str(payload.get("operator_crypto_route_alignment_recovery_brief") or ""),
        )
        actionable_count = max(total_count - len(actionability_backlog), 0)
        actionability_gate_status = _focus_slot_actionability_gate_status(
            total_count=total_count,
            actionable_count=actionable_count,
        )
        payload["operator_focus_slot_actionability_backlog"] = actionability_backlog
        payload["operator_focus_slot_actionability_backlog_brief"] = _focus_slot_actionability_backlog_brief(
            actionability_backlog
        )
        payload["operator_focus_slot_actionability_backlog_count"] = len(actionability_backlog)
        payload["operator_focus_slot_actionable_count"] = actionable_count
        payload["operator_focus_slot_actionability_gate_brief"] = _focus_slot_actionability_gate_brief(
            status=actionability_gate_status,
            actionable_count=actionable_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_actionability_gate_status"] = actionability_gate_status
        payload["operator_focus_slot_actionability_gate_blocker_detail"] = (
            _focus_slot_actionability_gate_blocker_detail(
                total_count=total_count,
                actionable_count=actionable_count,
                backlog=actionability_backlog,
            )
        )
        payload["operator_focus_slot_actionability_gate_done_when"] = (
            _focus_slot_actionability_gate_done_when(
                total_count=total_count,
                actionable_count=actionable_count,
                backlog=actionability_backlog,
            )
        )
        readiness_gate_status = _focus_slot_readiness_gate_status(
            total_count=total_count,
            promotion_gate_status=promotion_gate_status,
            actionability_gate_status=actionability_gate_status,
        )
        readiness_ready_count = _focus_slot_readiness_gate_ready_count(
            status=readiness_gate_status,
            ready_count=ready_count,
            actionable_count=actionable_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_readiness_gate_ready_count"] = readiness_ready_count
        payload["operator_focus_slot_readiness_gate_brief"] = _focus_slot_readiness_gate_brief(
            status=readiness_gate_status,
            ready_count=readiness_ready_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_readiness_gate_status"] = readiness_gate_status
        payload["operator_focus_slot_readiness_gate_blocking_gate"] = _focus_slot_readiness_gate_blocking_gate(
            status=readiness_gate_status
        )
        payload["operator_focus_slot_readiness_gate_blocker_detail"] = (
            _focus_slot_readiness_gate_blocker_detail(
                status=readiness_gate_status,
                promotion_gate_blocker_detail=str(
                    payload.get("operator_focus_slot_promotion_gate_blocker_detail") or ""
                ),
                actionability_gate_blocker_detail=str(
                    payload.get("operator_focus_slot_actionability_gate_blocker_detail") or ""
                ),
                total_count=total_count,
            )
        )
        payload["operator_focus_slot_readiness_gate_done_when"] = _focus_slot_readiness_gate_done_when(
            status=readiness_gate_status,
            promotion_gate_done_when=str(payload.get("operator_focus_slot_promotion_gate_done_when") or ""),
            actionability_gate_done_when=str(payload.get("operator_focus_slot_actionability_gate_done_when") or ""),
        )
        primary_slot = dict(focus_slots[0]) if len(focus_slots) >= 1 else {}
        followup_slot = dict(focus_slots[1]) if len(focus_slots) >= 2 else {}
        secondary_slot = dict(focus_slots[2]) if len(focus_slots) >= 3 else {}
        refresh_head = dict(refresh_backlog[0]) if refresh_backlog else {}
        source_refresh_head = dict(source_refresh_queue[0]) if source_refresh_queue else {}
        source_refresh_checklist_head = dict(source_refresh_checklist[0]) if source_refresh_checklist else {}
        crypto_alignment_focus_row = _find_focus_slot_row(
            focus_slots,
            area=str(payload.get("operator_crypto_route_alignment_focus_area") or ""),
            symbol=str(payload.get("operator_crypto_route_alignment_focus_symbol") or ""),
            action=str(payload.get("operator_crypto_route_alignment_focus_action") or ""),
        )
        pipeline_focus_row = (
            dict(crypto_alignment_focus_row)
            if crypto_alignment_focus_row
            else (dict(source_refresh_head) if source_refresh_head else dict(secondary_slot))
        )
        pipeline_source_kind = str(
            pipeline_focus_row.get("source_kind") or source_refresh_head.get("source_kind") or ""
        )
        pipeline_source_artifact = str(
            pipeline_focus_row.get("source_artifact") or source_refresh_head.get("source_artifact") or ""
        )
        pipeline_recipe = _source_refresh_recipe(
            source_kind=pipeline_source_kind,
            source_artifact=pipeline_source_artifact,
            symbol=str(pipeline_focus_row.get("symbol") or source_refresh_head.get("symbol") or ""),
            reference_now=runtime_now,
        )
        pipeline_steps = list(pipeline_recipe.get("steps") or [])
        pipeline_deferred_steps = _recipe_deferred_steps(
            pipeline_steps,
            recipe_gate_status=str(payload.get("operator_crypto_route_alignment_recipe_status") or ""),
        )
        pipeline_pending_steps = [] if pipeline_deferred_steps else _recipe_pending_steps(pipeline_steps)
        pipeline_head = dict(pipeline_pending_steps[0]) if pipeline_pending_steps else {}
        pipeline_deferred_head = dict(pipeline_deferred_steps[0]) if pipeline_deferred_steps else {}
        payload["operator_source_refresh_pipeline_steps_brief"] = str(pipeline_recipe.get("steps_brief") or "-")
        payload["operator_source_refresh_pipeline_step_checkpoint_brief"] = str(
            pipeline_recipe.get("step_checkpoint_brief") or "-"
        )
        payload["operator_source_refresh_pipeline_pending_brief"] = _recipe_pending_brief(pipeline_pending_steps)
        payload["operator_source_refresh_pipeline_pending_count"] = len(pipeline_pending_steps)
        payload["operator_source_refresh_pipeline_head_rank"] = str(pipeline_head.get("rank") or "-")
        payload["operator_source_refresh_pipeline_head_name"] = str(pipeline_head.get("name") or "-")
        payload["operator_source_refresh_pipeline_head_checkpoint_state"] = str(
            pipeline_head.get("checkpoint_state") or "-"
        )
        payload["operator_source_refresh_pipeline_head_expected_artifact_kind"] = str(
            pipeline_head.get("expected_artifact_kind") or "-"
        )
        payload["operator_source_refresh_pipeline_head_current_artifact"] = str(
            pipeline_head.get("current_artifact") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_brief"] = _recipe_pending_brief(pipeline_deferred_steps)
        payload["operator_source_refresh_pipeline_deferred_count"] = len(pipeline_deferred_steps)
        payload["operator_source_refresh_pipeline_deferred_status"] = str(
            payload.get("operator_crypto_route_alignment_recipe_status") or ""
        )
        payload["operator_source_refresh_pipeline_deferred_until"] = str(
            payload.get("operator_crypto_route_alignment_recipe_ready_on_date")
            or payload.get("operator_crypto_route_alignment_cooldown_next_eligible_end_date")
            or ""
        )
        payload["operator_source_refresh_pipeline_deferred_reason"] = str(
            payload.get("operator_crypto_route_alignment_recipe_blocker_detail")
            or payload.get("operator_crypto_route_alignment_cooldown_blocker_detail")
            or ""
        )
        payload["operator_source_refresh_pipeline_deferred_head_rank"] = str(
            pipeline_deferred_head.get("rank") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_name"] = str(
            pipeline_deferred_head.get("name") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_checkpoint_state"] = str(
            pipeline_deferred_head.get("checkpoint_state") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_expected_artifact_kind"] = str(
            pipeline_deferred_head.get("expected_artifact_kind") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_current_artifact"] = str(
            pipeline_deferred_head.get("current_artifact") or "-"
        )
        payload["next_focus_source_kind"] = str(primary_slot.get("source_kind") or "-")
        payload["next_focus_source_artifact"] = str(primary_slot.get("source_artifact") or "")
        payload["next_focus_source_status"] = str(primary_slot.get("source_status") or "-")
        payload["next_focus_source_as_of"] = str(primary_slot.get("source_as_of") or "")
        payload["next_focus_source_age_minutes"] = primary_slot.get("source_age_minutes")
        payload["next_focus_source_recency"] = str(primary_slot.get("source_recency") or "-")
        payload["next_focus_source_health"] = str(primary_slot.get("source_health") or "-")
        payload["next_focus_source_refresh_action"] = str(primary_slot.get("source_refresh_action") or "-")
        payload["followup_focus_source_kind"] = str(followup_slot.get("source_kind") or "-")
        payload["followup_focus_source_artifact"] = str(followup_slot.get("source_artifact") or "")
        payload["followup_focus_source_status"] = str(followup_slot.get("source_status") or "-")
        payload["followup_focus_source_as_of"] = str(followup_slot.get("source_as_of") or "")
        payload["followup_focus_source_age_minutes"] = followup_slot.get("source_age_minutes")
        payload["followup_focus_source_recency"] = str(followup_slot.get("source_recency") or "-")
        payload["followup_focus_source_health"] = str(followup_slot.get("source_health") or "-")
        payload["followup_focus_source_refresh_action"] = str(followup_slot.get("source_refresh_action") or "-")
        payload["secondary_focus_source_kind"] = str(secondary_slot.get("source_kind") or "-")
        payload["secondary_focus_source_artifact"] = str(secondary_slot.get("source_artifact") or "")
        payload["secondary_focus_source_status"] = str(secondary_slot.get("source_status") or "-")
        payload["secondary_focus_source_as_of"] = str(secondary_slot.get("source_as_of") or "")
        payload["secondary_focus_source_age_minutes"] = secondary_slot.get("source_age_minutes")
        payload["secondary_focus_source_recency"] = str(secondary_slot.get("source_recency") or "-")
        payload["secondary_focus_source_health"] = str(secondary_slot.get("source_health") or "-")
        payload["secondary_focus_source_refresh_action"] = str(secondary_slot.get("source_refresh_action") or "-")
        payload["operator_focus_slot_refresh_head_slot"] = str(refresh_head.get("slot") or "-")
        payload["operator_focus_slot_refresh_head_symbol"] = str(refresh_head.get("symbol") or "-")
        payload["operator_focus_slot_refresh_head_action"] = str(refresh_head.get("action") or "-")
        payload["operator_focus_slot_refresh_head_health"] = str(refresh_head.get("source_health") or "-")
        payload["operator_source_refresh_next_slot"] = str(source_refresh_head.get("slot") or "-")
        payload["operator_source_refresh_next_symbol"] = str(source_refresh_head.get("symbol") or "-")
        payload["operator_source_refresh_next_action"] = str(source_refresh_head.get("action") or "-")
        payload["operator_source_refresh_next_source_kind"] = str(source_refresh_head.get("source_kind") or "-")
        payload["operator_source_refresh_next_source_health"] = str(source_refresh_head.get("source_health") or "-")
        payload["operator_source_refresh_next_source_artifact"] = str(source_refresh_head.get("source_artifact") or "")
        payload["operator_source_refresh_next_state"] = str(source_refresh_checklist_head.get("state") or "-")
        payload["operator_source_refresh_next_blocker_detail"] = str(
            source_refresh_checklist_head.get("blocker_detail") or "-"
        )
        payload["operator_source_refresh_next_done_when"] = str(
            source_refresh_checklist_head.get("done_when") or "-"
        )
        payload["operator_source_refresh_next_recipe_script"] = str(
            source_refresh_checklist_head.get("recipe_script") or ""
        )
        payload["operator_source_refresh_next_recipe_command_hint"] = str(
            source_refresh_checklist_head.get("recipe_command_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_expected_status"] = str(
            source_refresh_checklist_head.get("recipe_expected_status") or ""
        )
        payload["operator_source_refresh_next_recipe_expected_artifact_kind"] = str(
            source_refresh_checklist_head.get("recipe_expected_artifact_kind") or ""
        )
        payload["operator_source_refresh_next_recipe_expected_artifact_path_hint"] = str(
            source_refresh_checklist_head.get("recipe_expected_artifact_path_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_note"] = str(
            source_refresh_checklist_head.get("recipe_note") or ""
        )
        payload["operator_source_refresh_next_recipe_followup_script"] = str(
            source_refresh_checklist_head.get("recipe_followup_script") or ""
        )
        payload["operator_source_refresh_next_recipe_followup_command_hint"] = str(
            source_refresh_checklist_head.get("recipe_followup_command_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_verify_hint"] = str(
            source_refresh_checklist_head.get("recipe_verify_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_steps_brief"] = str(
            source_refresh_checklist_head.get("recipe_steps_brief") or ""
        )
        payload["operator_source_refresh_next_recipe_step_checkpoint_brief"] = str(
            source_refresh_checklist_head.get("recipe_step_checkpoint_brief") or ""
        )
        payload["operator_source_refresh_next_recipe_steps"] = list(
            source_refresh_checklist_head.get("recipe_steps") or []
        )
        crypto_route_head_source_refresh_source = dict(
            crypto_route_source_payload
            or crypto_source_payload.get("crypto_route_operator_brief")
            or crypto_source_payload.get("crypto_route_brief")
            or {}
        )
        crypto_route_head_source_refresh = _crypto_route_head_source_refresh_lane(
            row=crypto_alignment_focus_row,
            reference_now=runtime_now,
            cooldown_next_eligible_end_date=str(
                payload.get("operator_crypto_route_alignment_cooldown_next_eligible_end_date") or ""
            ),
            source_payload=crypto_route_head_source_refresh_source,
        )
        crypto_route_head_source_refresh_recipe = dict(
            crypto_route_head_source_refresh.get("recipe") or {}
        )
        payload["crypto_route_head_source_refresh_status"] = str(
            crypto_route_head_source_refresh.get("status") or ""
        )
        payload["crypto_route_head_source_refresh_brief"] = str(
            crypto_route_head_source_refresh.get("brief") or ""
        )
        payload["crypto_route_head_source_refresh_slot"] = str(
            crypto_route_head_source_refresh.get("slot") or "-"
        )
        payload["crypto_route_head_source_refresh_symbol"] = str(
            crypto_route_head_source_refresh.get("symbol") or "-"
        )
        payload["crypto_route_head_source_refresh_action"] = str(
            crypto_route_head_source_refresh.get("action") or "-"
        )
        payload["crypto_route_head_source_refresh_source_kind"] = str(
            crypto_route_head_source_refresh.get("source_kind") or "-"
        )
        payload["crypto_route_head_source_refresh_source_health"] = str(
            crypto_route_head_source_refresh.get("source_health") or "-"
        )
        payload["crypto_route_head_source_refresh_source_artifact"] = str(
            crypto_route_head_source_refresh.get("source_artifact") or ""
        )
        payload["crypto_route_head_source_refresh_blocker_detail"] = str(
            crypto_route_head_source_refresh.get("blocker_detail") or ""
        )
        payload["crypto_route_head_source_refresh_done_when"] = str(
            crypto_route_head_source_refresh.get("done_when") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_script"] = str(
            crypto_route_head_source_refresh_recipe.get("script") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_command_hint"] = str(
            crypto_route_head_source_refresh_recipe.get("command_hint") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_expected_status"] = str(
            crypto_route_head_source_refresh_recipe.get("expected_status") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_expected_artifact_kind"] = str(
            crypto_route_head_source_refresh_recipe.get("expected_artifact_kind") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_expected_artifact_path_hint"] = str(
            crypto_route_head_source_refresh_recipe.get("expected_artifact_path_hint") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_note"] = str(
            crypto_route_head_source_refresh_recipe.get("note") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_followup_script"] = str(
            crypto_route_head_source_refresh_recipe.get("followup_script") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_followup_command_hint"] = str(
            crypto_route_head_source_refresh_recipe.get("followup_command_hint") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_verify_hint"] = str(
            crypto_route_head_source_refresh_recipe.get("verify_hint") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_steps_brief"] = str(
            crypto_route_head_source_refresh_recipe.get("steps_brief") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_step_checkpoint_brief"] = str(
            crypto_route_head_source_refresh_recipe.get("step_checkpoint_brief") or ""
        )
        payload["crypto_route_head_source_refresh_recipe_steps"] = list(
            crypto_route_head_source_refresh_recipe.get("steps") or []
        )
        pipeline_relevance = _source_refresh_pipeline_relevance(
            crypto_head_source_refresh=crypto_route_head_source_refresh,
            pending_steps=pipeline_pending_steps,
            deferred_steps=pipeline_deferred_steps,
        )
        payload["operator_source_refresh_pipeline_relevance_status"] = str(
            pipeline_relevance.get("status") or ""
        )
        payload["operator_source_refresh_pipeline_relevance_brief"] = str(
            pipeline_relevance.get("brief") or ""
        )
        payload["operator_source_refresh_pipeline_relevance_blocker_detail"] = str(
            pipeline_relevance.get("blocker_detail") or ""
        )
        payload["operator_source_refresh_pipeline_relevance_done_when"] = str(
            pipeline_relevance.get("done_when") or ""
        )
        summary_lines = list(payload.get("summary_lines", []))
        summary_lines.append(f"focus-slot-sources: {payload.get('operator_focus_slot_sources_brief') or '-'}")
        summary_lines.append(f"focus-slot-source-status: {payload.get('operator_focus_slot_status_brief') or '-'}")
        summary_lines.append(f"focus-slot-source-recency: {payload.get('operator_focus_slot_recency_brief') or '-'}")
        summary_lines.append(f"focus-slot-source-health: {payload.get('operator_focus_slot_health_brief') or '-'}")
        summary_lines.append(f"focus-slot-refresh-backlog: {payload.get('operator_focus_slot_refresh_backlog_brief') or '-'}")
        summary_lines.append(f"focus-slot-promotion-gate: {payload.get('operator_focus_slot_promotion_gate_brief') or '-'}")
        summary_lines.append(
            f"focus-slot-actionability-gate: {payload.get('operator_focus_slot_actionability_gate_brief') or '-'}"
        )
        summary_lines.append(
            f"focus-slot-readiness-gate: {payload.get('operator_focus_slot_readiness_gate_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-head-source-refresh: {payload.get('crypto_route_head_source_refresh_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-refresh-reuse: {payload.get('source_crypto_route_refresh_reuse_brief') or '-'}"
        )
        summary_lines.append(
            "crypto-route-refresh-reuse-gate: "
            f"{payload.get('source_crypto_route_refresh_reuse_gate_brief') or '-'}"
        )
        summary_lines.append(
            f"remote-live-history: {payload.get('source_remote_live_history_audit_window_brief') or '-'}"
        )
        summary_lines.append(
            "remote-live-risk-guard: "
            f"{payload.get('source_remote_live_history_audit_risk_guard_status') or '-'}:"
            f"{_list_text(payload.get('source_remote_live_history_audit_risk_guard_reasons', []), limit=8)}"
        )
        summary_lines.append(
            "remote-live-scope: "
            f"{payload.get('source_remote_live_handoff_account_scope_alignment_brief') or '-'}"
        )
        summary_lines.append(
            "remote-live-diagnosis: "
            f"{payload.get('source_live_gate_blocker_remote_live_diagnosis_brief') or '-'}"
        )
        summary_lines.append(
            "remote-live-clearing: "
            f"{payload.get('source_live_gate_blocker_remote_live_takeover_clearing_brief') or '-'}"
        )
        summary_lines.append(
            "remote-live-repair-queue: "
            f"{payload.get('remote_live_takeover_repair_queue_brief') or '-'}"
        )
        summary_lines.append(
            "remote-live-operator-alignment: "
            f"{payload.get('source_live_gate_blocker_remote_live_operator_alignment_brief') or '-'}"
        )
        summary_lines.append(
            "cross-market-remote-live-alignment: "
            f"{payload.get('source_cross_market_operator_state_remote_live_operator_alignment_brief') or '-'}"
        )
        summary_lines.append(
            "cross-market-remote-live-takeover-gate: "
            f"{payload.get('source_cross_market_operator_state_remote_live_takeover_gate_brief') or '-'}"
        )
        summary_lines.append(
            "cross-market-remote-live-clearing: "
            f"{payload.get('cross_market_remote_live_takeover_clearing_brief') or '-'}"
        )
        summary_lines.append(
            "cross-market-remote-live-clearing-freshness: "
            f"{payload.get('cross_market_remote_live_takeover_clearing_source_freshness_brief') or '-'}"
        )
        summary_lines.append(
            "cross-market-remote-live-slot-anomaly: "
            f"{payload.get('cross_market_remote_live_takeover_slot_anomaly_breakdown_brief') or '-'}"
        )
        summary_lines.append(
            "brooks-route: "
            f"{payload.get('source_brooks_route_report_selected_routes_brief') or '-'}"
            f" | head={payload.get('source_brooks_route_report_head_symbol') or '-'}"
            f":{payload.get('source_brooks_route_report_head_strategy_id') or '-'}"
            f":{payload.get('source_brooks_route_report_head_bridge_status') or '-'}"
        )
        summary_lines.append(
            "brooks-exec-plan: "
            f"{payload.get('source_brooks_execution_plan_head_symbol') or '-'}"
            f":{payload.get('source_brooks_execution_plan_head_strategy_id') or '-'}"
            f":{payload.get('source_brooks_execution_plan_head_plan_status') or '-'}"
            f":{payload.get('source_brooks_execution_plan_head_execution_action') or '-'}"
        )
        summary_lines.append(
            "brooks-review-queue: "
            + " | ".join(
                [
                    payload.get("brooks_structure_review_queue_status") or "-",
                    payload.get("brooks_structure_review_queue_brief") or "-",
                    f"head={payload.get('brooks_structure_review_head_symbol') or '-'}:{payload.get('brooks_structure_review_head_tier') or '-'}:{payload.get('brooks_structure_review_head_plan_status') or '-'}",
                ]
            )
        )
        summary_lines.append(
            "brooks-review-priority: "
            + " | ".join(
                [
                    payload.get("brooks_structure_review_priority_status") or "-",
                    payload.get("brooks_structure_review_priority_brief") or "-",
                    f"head={payload.get('brooks_structure_review_head_symbol') or '-'}:{payload.get('brooks_structure_review_head_priority_tier') or '-'}:{payload.get('brooks_structure_review_head_priority_score') or 0}",
                ]
            )
        )
        summary_lines.append(
            "brooks-operator-lane: "
            + " | ".join(
                [
                    payload.get("brooks_structure_operator_status") or "-",
                    payload.get("brooks_structure_operator_brief") or "-",
                    f"backlog={payload.get('brooks_structure_operator_backlog_count') or 0}:{payload.get('brooks_structure_operator_backlog_brief') or '-'}",
                ]
            )
        )
        summary_lines.append(
            "brooks-refresh: "
            + " | ".join(
                [
                    payload.get("source_brooks_structure_refresh_status") or "-",
                    payload.get("source_brooks_structure_refresh_brief") or "-",
                    f"head={payload.get('source_brooks_structure_refresh_head_symbol') or '-'}:{payload.get('source_brooks_structure_refresh_head_action') or '-'}:{payload.get('source_brooks_structure_refresh_head_priority_score') or '-'}",
                    f"queue={payload.get('source_brooks_structure_refresh_queue_count') or 0}",
                ]
            )
        )
        summary_lines.append(
            "cross-market-operator: "
            + " | ".join(
                [
                    payload.get("cross_market_operator_head_status") or "-",
                    payload.get("cross_market_operator_head_brief") or "-",
                    f"backlog={payload.get('cross_market_operator_backlog_count') or 0}:{payload.get('cross_market_operator_backlog_brief') or '-'}",
                ]
            )
        )
        summary_lines.append(
            "cross-market-operator-lanes: "
            + " | ".join(
                [
                    f"waiting={payload.get('cross_market_operator_waiting_lane_count') or 0}@{payload.get('cross_market_operator_waiting_lane_priority_total') or 0}",
                    f"review={payload.get('cross_market_operator_review_lane_count') or 0}@{payload.get('cross_market_operator_review_lane_priority_total') or 0}",
                    f"watch={payload.get('cross_market_operator_watch_lane_count') or 0}@{payload.get('cross_market_operator_watch_lane_priority_total') or 0}",
                    f"blocked={payload.get('cross_market_operator_blocked_lane_count') or 0}@{payload.get('cross_market_operator_blocked_lane_priority_total') or 0}",
                    f"repair={payload.get('cross_market_operator_repair_lane_count') or 0}@{payload.get('cross_market_operator_repair_lane_priority_total') or 0}",
                ]
            )
        )
        summary_lines.append(
            "cross-market-operator-lane-heads: "
            + str(payload.get("cross_market_operator_lane_heads_brief") or "-")
        )
        summary_lines.append(
            "cross-market-operator-lane-order: "
            + str(payload.get("cross_market_operator_lane_priority_order_brief") or "-")
        )
        summary_lines.append(
            "cross-market-repair-head: "
            + " | ".join(
                [
                    payload.get("cross_market_operator_repair_head_status") or "-",
                    payload.get("cross_market_operator_repair_head_brief") or "-",
                    f"backlog={payload.get('cross_market_operator_repair_backlog_count') or 0}:{payload.get('cross_market_operator_repair_backlog_brief') or '-'}",
                ]
            )
        )
        summary_lines.append(
            "cross-market-head: "
            + " | ".join(
                [
                    payload.get("cross_market_review_head_status") or "-",
                    payload.get("cross_market_review_head_brief") or "-",
                    f"backlog={payload.get('cross_market_review_backlog_count') or 0}:{payload.get('cross_market_review_backlog_brief') or '-'}",
                ]
            )
        )
        summary_lines.append(
            "cross-market-review: "
            + " | ".join(
                [
                    payload.get("source_cross_market_operator_state_review_backlog_status") or "-",
                    payload.get("source_cross_market_operator_state_review_backlog_brief") or "-",
                    f"head={payload.get('source_cross_market_operator_state_review_head_area') or '-'}:{payload.get('source_cross_market_operator_state_review_head_symbol') or '-'}:{payload.get('source_cross_market_operator_state_review_head_action') or '-'}:{payload.get('source_cross_market_operator_state_review_head_priority_score') or '-'}",
                ]
            )
        )
        summary_lines.append(
            f"research-embedding-quality: {payload.get('operator_research_embedding_quality_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment: {payload.get('operator_crypto_route_alignment_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-slot: {payload.get('operator_crypto_route_alignment_focus_slot') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-recovery-outcome: {payload.get('operator_crypto_route_alignment_recovery_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-cooldown: {payload.get('operator_crypto_route_alignment_cooldown_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-recovery-recipe: {payload.get('operator_crypto_route_alignment_recipe_brief') or '-'}"
        )
        if payload.get("operator_crypto_route_alignment_recipe_target_batches"):
            summary_lines.append(
                "crypto-route-alignment-recovery: "
                + _list_text(payload.get("operator_crypto_route_alignment_recipe_target_batches", []), limit=6)
                + f"@{payload.get('operator_crypto_route_alignment_recipe_window_days') or '-'}d"
            )
        summary_lines.append(f"source-refresh-queue: {payload.get('operator_source_refresh_queue_brief') or '-'}")
        summary_lines.append(f"source-refresh-checklist: {payload.get('operator_source_refresh_checklist_brief') or '-'}")
        summary_lines.append(
            f"source-refresh-pipeline: {payload.get('operator_source_refresh_pipeline_pending_brief') or '-'}"
        )
        summary_lines.append(
            f"source-refresh-pipeline-deferred: {payload.get('operator_source_refresh_pipeline_deferred_brief') or '-'}"
        )
        summary_lines.append(
            f"source-refresh-pipeline-relevance: {payload.get('operator_source_refresh_pipeline_relevance_brief') or '-'}"
        )
        payload["summary_lines"] = summary_lines
        payload["summary_text"] = " | ".join(summary_lines)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": payload["as_of"],
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="hot_universe_operator_brief",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
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
