#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_mod
import re
import sys
from typing import Any


SYSTEM_ROOT = Path(
    str(os.getenv("LIE_SYSTEM_ROOT", "")).strip()
    or str(os.getenv("FENLIE_SYSTEM_ROOT", "")).strip()
    or Path(__file__).resolve().parents[1]
).expanduser().resolve()
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.data.storage import write_json, write_markdown
from lie_engine.research.optimizer import run_research_backtest
from lie_engine.research.strategy_lab import run_strategy_lab


DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_ARTIFACT_TTL_HOURS = 168.0
DEFAULT_KEEP_FILES = 40
DEFAULT_BATCH_TIMEOUT_SECONDS = 60.0
DEFAULT_BATCH_TIMEOUT_SYMBOL_BASE = 4
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


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


def parse_date_text(raw: str) -> dt.date:
    return dt.date.fromisoformat(str(raw).strip())


def write_sha256(path: Path, checksum_path: Path, *, ttl_hours: float, generated_at: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        checksum_path,
        {
            "generated_at_utc": generated_at,
            "artifact_ttl_hours": max(1.0, ttl_hours),
            "files": [{"path": str(path), "sha256": digest}],
        },
    )


def evict_old_artifacts(
    *,
    review_dir: Path,
    protected: set[str],
    now_dt: dt.datetime,
    ttl_hours: float,
    keep_files: int,
) -> int:
    cutoff = now_dt - dt.timedelta(hours=max(1.0, ttl_hours))
    candidates: list[Path] = []
    for pattern in (
        "*_hot_universe_research.json",
        "*_hot_universe_research_checksum.json",
        "*_hot_universe_research.md",
    ):
        candidates.extend(review_dir.glob(pattern))
    deleted = 0
    ordered = sorted(
        [path for path in candidates if path.name not in protected],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep_names = {path.name for path in ordered[: max(0, keep_files - len(protected))]}
    for path in ordered:
        modified = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        if path.name in keep_names and modified >= cutoff:
            continue
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted


def find_latest_universe_file(review_dir: Path) -> Path | None:
    candidates = sorted(review_dir.glob("*_hot_research_universe.json"))
    return candidates[-1] if candidates else None


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


def latest_review_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping in {path}")
    return payload


def load_crypto_symbol_route_handoff(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "binance_indicator_symbol_route_handoff", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "operator_status": str(payload.get("operator_status", "")),
        "route_stack_brief": str(payload.get("route_stack_brief", "")),
        "next_focus_symbol": str(payload.get("next_focus_symbol", "")),
        "next_focus_action": str(payload.get("next_focus_action", "")),
        "next_focus_reason": str(payload.get("next_focus_reason", "")),
        "focus_window_gate": str(payload.get("focus_window_gate", "")),
        "focus_window_gate_reason": str(payload.get("focus_window_gate_reason", "")),
        "focus_window_verdict": str(payload.get("focus_window_verdict", "")),
        "deploy_now_symbols": list(payload.get("deploy_now_symbols", []) or []),
        "watch_priority_symbols": list(payload.get("watch_priority_symbols", []) or []),
        "watch_only_symbols": list(payload.get("watch_only_symbols", []) or []),
        "routes": list(payload.get("routes", []) or []),
        "overall_takeaway": str(payload.get("overall_takeaway", "")),
    }


def load_crypto_route_brief(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "crypto_route_brief", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "operator_status": str(payload.get("operator_status", "")),
        "route_stack_brief": str(payload.get("route_stack_brief", "")),
        "next_focus_symbol": str(payload.get("next_focus_symbol", "")),
        "next_focus_action": str(payload.get("next_focus_action", "")),
        "next_focus_reason": str(payload.get("next_focus_reason", "")),
        "focus_window_gate": str(payload.get("focus_window_gate", "")),
        "focus_window_gate_reason": str(payload.get("focus_window_gate_reason", "")),
        "focus_window_verdict": str(payload.get("focus_window_verdict", "")),
        "focus_brief": str(payload.get("focus_brief", "")),
        "next_retest_action": str(payload.get("next_retest_action", "")),
        "next_retest_reason": str(payload.get("next_retest_reason", "")),
        "deploy_now_symbols": list(payload.get("deploy_now_symbols", []) or []),
        "watch_priority_symbols": list(payload.get("watch_priority_symbols", []) or []),
        "watch_only_symbols": list(payload.get("watch_only_symbols", []) or []),
        "review_symbols": list(payload.get("review_symbols", []) or []),
        "brief_lines": list(payload.get("brief_lines", []) or []),
        "brief_text": str(payload.get("brief_text", "")),
        "overall_takeaway": str(payload.get("overall_takeaway", "")),
    }


def load_crypto_route_operator_brief(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "crypto_route_operator_brief", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "operator_status": str(payload.get("operator_status", "")),
        "route_stack_brief": str(payload.get("route_stack_brief", "")),
        "next_focus_symbol": str(payload.get("next_focus_symbol", "")),
        "next_focus_action": str(payload.get("next_focus_action", "")),
        "next_focus_reason": str(payload.get("next_focus_reason", "")),
        "focus_window_gate": str(payload.get("focus_window_gate", "")),
        "focus_window_verdict": str(payload.get("focus_window_verdict", "")),
        "focus_window_floor": str(payload.get("focus_window_floor", "")),
        "price_state_window_floor": str(payload.get("price_state_window_floor", "")),
        "comparative_window_takeaway": str(payload.get("comparative_window_takeaway", "")),
        "xlong_flow_window_floor": str(payload.get("xlong_flow_window_floor", "")),
        "xlong_comparative_window_takeaway": str(payload.get("xlong_comparative_window_takeaway", "")),
        "focus_brief": str(payload.get("focus_brief", "")),
        "next_retest_action": str(payload.get("next_retest_action", "")),
        "next_retest_reason": str(payload.get("next_retest_reason", "")),
        "operator_lines": list(payload.get("operator_lines", []) or []),
        "operator_text": str(payload.get("operator_text", "")),
    }


def load_bnb_flow_focus(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "binance_indicator_bnb_flow_focus", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "symbol": str(payload.get("symbol", "")),
        "operator_status": str(payload.get("operator_status", "")),
        "promotion_gate": str(payload.get("promotion_gate", "")),
        "promotion_gate_reason": str(payload.get("promotion_gate_reason", "")),
        "action": str(payload.get("action", "")),
        "action_reason": str(payload.get("action_reason", "")),
        "flow_window_verdict": str(payload.get("flow_window_verdict", "")),
        "price_state_window_verdict": str(payload.get("price_state_window_verdict", "")),
        "next_retest_action": str(payload.get("next_retest_action", "")),
        "next_retest_reason": str(payload.get("next_retest_reason", "")),
        "brief": str(payload.get("brief", "")),
    }


def load_commodity_paper_ticket_lane(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "commodity_paper_ticket_lane", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "status": str(payload.get("status", "")),
        "route_status": str(payload.get("route_status", "")),
        "ticket_status": str(payload.get("ticket_status", "")),
        "execution_mode": str(payload.get("execution_mode", "")),
        "paper_ready_batches": [str(x).strip() for x in payload.get("paper_ready_batches", []) if str(x).strip()],
        "shadow_only_batches": [str(x).strip() for x in payload.get("shadow_only_batches", []) if str(x).strip()],
        "missing_batches": [str(x).strip() for x in payload.get("missing_batches", []) if str(x).strip()],
        "next_ticket_batch": str(payload.get("next_ticket_batch", "")),
        "next_ticket_symbols": [str(x).strip().upper() for x in payload.get("next_ticket_symbols", []) if str(x).strip()],
        "ticket_stack_brief": str(payload.get("ticket_stack_brief", "")),
        "tickets": list(payload.get("tickets", []) or []),
    }


def load_commodity_paper_ticket_book(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "commodity_paper_ticket_book", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "status": str(payload.get("status", "")),
        "route_status": str(payload.get("route_status", "")),
        "ticket_book_status": str(payload.get("ticket_book_status", "")),
        "execution_mode": str(payload.get("execution_mode", "")),
        "actionable_batches": [str(x).strip() for x in payload.get("actionable_batches", []) if str(x).strip()],
        "shadow_batches": [str(x).strip() for x in payload.get("shadow_batches", []) if str(x).strip()],
        "next_ticket_id": str(payload.get("next_ticket_id", "")),
        "next_ticket_batch": str(payload.get("next_ticket_batch", "")),
        "next_ticket_symbol": str(payload.get("next_ticket_symbol", "")),
        "next_ticket_regime_gate": str(payload.get("next_ticket_regime_gate", "")),
        "next_ticket_weight_hint": float(payload.get("next_ticket_weight_hint", 0.0) or 0.0),
        "ticket_book_stack_brief": str(payload.get("ticket_book_stack_brief", "")),
        "actionable_ticket_count": int(payload.get("actionable_ticket_count", 0) or 0),
        "tickets": list(payload.get("tickets", []) or []),
    }


def load_commodity_paper_execution_preview(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "commodity_paper_execution_preview", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "status": str(payload.get("status", "")),
        "route_status": str(payload.get("route_status", "")),
        "ticket_book_status": str(payload.get("ticket_book_status", "")),
        "execution_preview_status": str(payload.get("execution_preview_status", "")),
        "execution_mode": str(payload.get("execution_mode", "")),
        "preview_ready_batches": [str(x).strip() for x in payload.get("preview_ready_batches", []) if str(x).strip()],
        "shadow_only_batches": [str(x).strip() for x in payload.get("shadow_only_batches", []) if str(x).strip()],
        "preview_batch_count": int(payload.get("preview_batch_count", 0) or 0),
        "next_execution_batch": str(payload.get("next_execution_batch", "")),
        "next_execution_symbols": [str(x).strip().upper() for x in payload.get("next_execution_symbols", []) if str(x).strip()],
        "next_execution_ticket_ids": [str(x).strip() for x in payload.get("next_execution_ticket_ids", []) if str(x).strip()],
        "next_execution_regime_gate": str(payload.get("next_execution_regime_gate", "")),
        "next_execution_weight_hint_sum": float(payload.get("next_execution_weight_hint_sum", 0.0) or 0.0),
        "preview_stack_brief": str(payload.get("preview_stack_brief", "")),
        "preview_batches": list(payload.get("preview_batches", []) or []),
    }


def load_commodity_paper_execution_artifact(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "commodity_paper_execution_artifact", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "status": str(payload.get("status", "")),
        "route_status": str(payload.get("route_status", "")),
        "ticket_book_status": str(payload.get("ticket_book_status", "")),
        "execution_preview_status": str(payload.get("execution_preview_status", "")),
        "execution_artifact_status": str(payload.get("execution_artifact_status", "")),
        "execution_mode": str(payload.get("execution_mode", "")),
        "execution_batch": str(payload.get("execution_batch", "")),
        "execution_symbols": [str(x).strip().upper() for x in payload.get("execution_symbols", []) if str(x).strip()],
        "execution_ticket_ids": [str(x).strip() for x in payload.get("execution_ticket_ids", []) if str(x).strip()],
        "execution_regime_gate": str(payload.get("execution_regime_gate", "")),
        "execution_weight_hint_sum": float(payload.get("execution_weight_hint_sum", 0.0) or 0.0),
        "execution_item_count": int(payload.get("execution_item_count", 0) or 0),
        "actionable_execution_item_count": int(payload.get("actionable_execution_item_count", 0) or 0),
        "execution_stack_brief": str(payload.get("execution_stack_brief", "")),
        "execution_items": list(payload.get("execution_items", []) or []),
    }


def load_commodity_paper_execution_queue(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "commodity_paper_execution_queue", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "status": str(payload.get("status", "")),
        "route_status": str(payload.get("route_status", "")),
        "ticket_book_status": str(payload.get("ticket_book_status", "")),
        "execution_preview_status": str(payload.get("execution_preview_status", "")),
        "execution_artifact_status": str(payload.get("execution_artifact_status", "")),
        "execution_queue_status": str(payload.get("execution_queue_status", "")),
        "execution_mode": str(payload.get("execution_mode", "")),
        "execution_batch": str(payload.get("execution_batch", "")),
        "execution_symbols": [str(x).strip().upper() for x in payload.get("execution_symbols", []) if str(x).strip()],
        "execution_ticket_ids": [str(x).strip() for x in payload.get("execution_ticket_ids", []) if str(x).strip()],
        "execution_regime_gate": str(payload.get("execution_regime_gate", "")),
        "execution_weight_hint_sum": float(payload.get("execution_weight_hint_sum", 0.0) or 0.0),
        "execution_item_count": int(payload.get("execution_item_count", 0) or 0),
        "actionable_execution_item_count": int(payload.get("actionable_execution_item_count", 0) or 0),
        "queue_depth": int(payload.get("queue_depth", 0) or 0),
        "actionable_queue_depth": int(payload.get("actionable_queue_depth", 0) or 0),
        "next_execution_id": str(payload.get("next_execution_id", "")),
        "next_execution_symbol": str(payload.get("next_execution_symbol", "")).strip().upper(),
        "queue_stack_brief": str(payload.get("queue_stack_brief", "")),
        "queued_items": list(payload.get("queued_items", []) or []),
    }


def load_commodity_paper_execution_review(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "commodity_paper_execution_review", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "status": str(payload.get("status", "")),
        "route_status": str(payload.get("route_status", "")),
        "ticket_book_status": str(payload.get("ticket_book_status", "")),
        "execution_preview_status": str(payload.get("execution_preview_status", "")),
        "execution_artifact_status": str(payload.get("execution_artifact_status", "")),
        "execution_queue_status": str(payload.get("execution_queue_status", "")),
        "execution_review_status": str(payload.get("execution_review_status", "")),
        "execution_mode": str(payload.get("execution_mode", "")),
        "execution_batch": str(payload.get("execution_batch", "")),
        "execution_symbols": [str(x).strip().upper() for x in payload.get("execution_symbols", []) if str(x).strip()],
        "execution_ticket_ids": [str(x).strip() for x in payload.get("execution_ticket_ids", []) if str(x).strip()],
        "execution_regime_gate": str(payload.get("execution_regime_gate", "")),
        "execution_weight_hint_sum": float(payload.get("execution_weight_hint_sum", 0.0) or 0.0),
        "execution_item_count": int(payload.get("execution_item_count", 0) or 0),
        "actionable_execution_item_count": int(payload.get("actionable_execution_item_count", 0) or 0),
        "queue_depth": int(payload.get("queue_depth", 0) or 0),
        "actionable_queue_depth": int(payload.get("actionable_queue_depth", 0) or 0),
        "review_item_count": int(payload.get("review_item_count", 0) or 0),
        "actionable_review_item_count": int(payload.get("actionable_review_item_count", 0) or 0),
        "next_review_execution_id": str(payload.get("next_review_execution_id", "")),
        "next_review_execution_symbol": str(payload.get("next_review_execution_symbol", "")).strip().upper(),
        "review_stack_brief": str(payload.get("review_stack_brief", "")),
        "review_items": list(payload.get("review_items", []) or []),
    }


def load_commodity_paper_execution_retro(review_dir: Path, reference_now: dt.datetime | None = None) -> dict[str, Any] | None:
    path = latest_review_artifact(review_dir, "commodity_paper_execution_retro", reference_now)
    if path is None:
        return None
    payload = load_json(path)
    return {
        "artifact": str(path),
        "status": str(payload.get("status", "")),
        "route_status": str(payload.get("route_status", "")),
        "ticket_book_status": str(payload.get("ticket_book_status", "")),
        "execution_preview_status": str(payload.get("execution_preview_status", "")),
        "execution_artifact_status": str(payload.get("execution_artifact_status", "")),
        "execution_queue_status": str(payload.get("execution_queue_status", "")),
        "execution_review_status": str(payload.get("execution_review_status", "")),
        "execution_retro_status": str(payload.get("execution_retro_status", "")),
        "execution_mode": str(payload.get("execution_mode", "")),
        "execution_batch": str(payload.get("execution_batch", "")),
        "execution_symbols": [str(x).strip().upper() for x in payload.get("execution_symbols", []) if str(x).strip()],
        "execution_ticket_ids": [str(x).strip() for x in payload.get("execution_ticket_ids", []) if str(x).strip()],
        "execution_regime_gate": str(payload.get("execution_regime_gate", "")),
        "execution_weight_hint_sum": float(payload.get("execution_weight_hint_sum", 0.0) or 0.0),
        "review_item_count": int(payload.get("review_item_count", 0) or 0),
        "actionable_review_item_count": int(payload.get("actionable_review_item_count", 0) or 0),
        "retro_item_count": int(payload.get("retro_item_count", 0) or 0),
        "actionable_retro_item_count": int(payload.get("actionable_retro_item_count", 0) or 0),
        "next_retro_execution_id": str(payload.get("next_retro_execution_id", "")),
        "next_retro_execution_symbol": str(payload.get("next_retro_execution_symbol", "")).strip().upper(),
        "retro_stack_brief": str(payload.get("retro_stack_brief", "")),
        "retro_items": list(payload.get("retro_items", []) or []),
    }


def select_batches(payload: dict[str, Any], requested: list[str] | None) -> dict[str, list[str]]:
    batches = payload.get("batches", {})
    if not isinstance(batches, dict):
        raise ValueError("universe_artifact_batches_missing")
    normalized_all: dict[str, list[str]] = {}
    for key, symbols in batches.items():
        if not isinstance(symbols, list):
            continue
        clean = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if clean:
            normalized_all[str(key)] = clean

    normalized: dict[str, list[str]] = {}
    requested_order = [str(x).strip() for x in (requested or []) if str(x).strip()]
    if requested_order:
        for key in requested_order:
            if key in normalized_all and key not in normalized:
                normalized[key] = normalized_all[key]
    else:
        normalized = dict(normalized_all)
    if not normalized:
        raise ValueError("no_batches_selected")
    return normalized


def summarize_research(payload: dict[str, Any]) -> dict[str, Any]:
    mode_summaries = payload.get("mode_summaries", [])
    if not isinstance(mode_summaries, list):
        mode_summaries = []
    ranked = sorted(
        [row for row in mode_summaries if isinstance(row, dict)],
        key=lambda row: float(row.get("best_score", -1e18) or -1e18),
        reverse=True,
    )
    best = ranked[0] if ranked else {}
    return {
        "output_dir": str(payload.get("output_dir", "")),
        "manifest": str(payload.get("manifest", "")),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
        "universe_count": int(payload.get("universe_count", 0) or 0),
        "bars_rows": int(payload.get("bars_rows", 0) or 0),
        "best_mode": str(best.get("mode", "")),
        "best_score": float(best.get("best_score", 0.0) or 0.0),
        "best_metrics": dict(best.get("best_metrics", {}) or {}),
        "best_params": dict(best.get("best_params", {}) or {}),
        "best_by_symbol": dict(best.get("best_by_symbol", {}) or {}),
        "best_by_symbol_regime": dict(best.get("best_by_symbol_regime", {}) or {}),
        "proxy_lookback_applied": int(payload.get("proxy_lookback_applied", 0) or 0),
    }


def summarize_strategy_lab(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    accepted_count = sum(1 for row in candidates if isinstance(row, dict) and bool(row.get("accepted", False)))
    best_candidate = payload.get("best_candidate", {}) if isinstance(payload.get("best_candidate", {}), dict) else {}
    return {
        "output_dir": str(payload.get("output_dir", "")),
        "manifest": str(payload.get("manifest", "")),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
        "candidate_count": int(len(candidates)),
        "accepted_count": int(accepted_count),
        "best_candidate_name": str(best_candidate.get("name", "")),
        "best_candidate_score": float(best_candidate.get("score", 0.0) or 0.0),
        "best_candidate_accepted": bool(best_candidate.get("accepted", False)),
    }


def classify_batch_outcome(row: dict[str, Any]) -> dict[str, Any]:
    rb = row.get("research_backtest", {}) if isinstance(row.get("research_backtest"), dict) else {}
    sl = row.get("strategy_lab", {}) if isinstance(row.get("strategy_lab"), dict) else {}
    status = str(row.get("status", "")).strip().lower()
    if not rb and status in {"planned", "pending"}:
        return {
            "label": "planned",
            "research_annual_return": 0.0,
            "research_score": 0.0,
            "research_trades": 0,
            "accepted_count": 0,
            "takeaway": "Batch is selected but has not been executed yet.",
        }
    score = float(rb.get("best_score", 0.0) or 0.0)
    annual_return = float((rb.get("best_metrics", {}) or {}).get("annual_return", 0.0) or 0.0)
    trades = int((rb.get("best_metrics", {}) or {}).get("trades", 0) or 0)
    accepted_count = int(sl.get("accepted_count", 0) or 0)
    accepted = accepted_count > 0

    if annual_return > 0.0 and accepted:
        label = "validated"
        takeaway = "Research and strategy_lab both support this batch."
    elif annual_return > 0.0 and not accepted:
        label = "research_only"
        takeaway = "Research is positive but strategy_lab still does not validate it."
    elif annual_return <= 0.0 and accepted:
        label = "pilot_only"
        takeaway = "Templates can pass, but research edge is still weak."
    else:
        label = "deprioritize"
        takeaway = "Neither research nor strategy_lab currently supports this batch."

    if trades <= 0:
        takeaway += " No research trades were observed."
    elif trades < 8:
        takeaway += " Trade count is thin; treat as fragile."

    return {
        "label": label,
        "research_annual_return": annual_return,
        "research_score": score,
        "research_trades": trades,
        "accepted_count": accepted_count,
        "takeaway": takeaway,
    }


def summarize_batch_results(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        [row for row in batch_results if isinstance(row, dict)],
        key=lambda row: float((row.get("research_backtest", {}) or {}).get("best_score", -1e18) or -1e18),
        reverse=True,
    )
    labels: dict[str, list[str]] = {
        "planned": [],
        "validated": [],
        "research_only": [],
        "pilot_only": [],
        "deprioritize": [],
    }
    rankings: list[dict[str, Any]] = []
    for row in ranked:
        outcome = classify_batch_outcome(row)
        batch_name = str(row.get("batch", "")).strip()
        if batch_name:
            labels.setdefault(str(outcome["label"]), []).append(batch_name)
        rankings.append(
            {
                "batch": batch_name,
                "status_label": outcome["label"],
                "research_score": float(outcome["research_score"]),
                "research_annual_return": float(outcome["research_annual_return"]),
                "research_trades": int(outcome["research_trades"]),
                "accepted_count": int(outcome["accepted_count"]),
                "takeaway": str(outcome["takeaway"]),
            }
        )
    return {
        "ranked_batches": rankings,
        "planned_batches": labels["planned"],
        "validated_batches": labels["validated"],
        "research_only_batches": labels["research_only"],
        "pilot_only_batches": labels["pilot_only"],
        "deprioritized_batches": labels["deprioritize"],
    }


def derive_batch_playbook(summary: dict[str, Any]) -> dict[str, Any]:
    ranked = [row for row in summary.get("ranked_batches", []) if isinstance(row, dict)]
    preferred: list[dict[str, Any]] = []
    fragile: list[dict[str, Any]] = []
    research_queue: list[dict[str, Any]] = []
    avoid: list[dict[str, Any]] = []

    by_batch: dict[str, dict[str, Any]] = {}
    for row in ranked:
        batch = str(row.get("batch", "")).strip()
        if not batch:
            continue
        by_batch[batch] = row
        label = str(row.get("status_label", "")).strip()
        trades = int(row.get("research_trades", 0) or 0)
        accepted_count = int(row.get("accepted_count", 0) or 0)
        annual_return = float(row.get("research_annual_return", 0.0) or 0.0)
        item = {
            "batch": batch,
            "status_label": label,
            "research_trades": trades,
            "accepted_count": accepted_count,
            "research_annual_return": annual_return,
            "takeaway": str(row.get("takeaway", "")),
        }
        fragile_batch = trades < 8 or accepted_count < 2
        if label == "validated" and not fragile_batch:
            preferred.append(item)
        elif label == "validated":
            item["fragility_reason"] = "validated_but_low_confirmation"
            fragile.append(item)
        elif label in {"research_only", "pilot_only"}:
            research_queue.append(item)
        elif label == "deprioritize":
            avoid.append(item)

    regime_takeaways: list[dict[str, str]] = []
    energy_liquids = by_batch.get("energy_liquids")
    energy_gas = by_batch.get("energy_gas")
    if energy_liquids and energy_gas:
        if str(energy_liquids.get("status_label", "")) == "validated" and str(energy_gas.get("status_label", "")) == "deprioritize":
            regime_takeaways.append(
                {
                    "theme": "energy",
                    "signal": "paired_liquids_outperform_single_gas",
                    "takeaway": "WTI+BRENT works as a sleeve; NATGAS alone currently has no support.",
                }
            )
    metals_all = by_batch.get("metals_all")
    precious_metals = by_batch.get("precious_metals")
    base_metals = by_batch.get("base_metals")
    if metals_all and (precious_metals or base_metals):
        metals_all_validated = str(metals_all.get("status_label", "")) == "validated"
        precious_weak = str((precious_metals or {}).get("status_label", "")) in {"research_only", "deprioritize"}
        base_weak = str((base_metals or {}).get("status_label", "")) == "deprioritize"
        if metals_all_validated and (precious_weak or base_weak):
            regime_takeaways.append(
                {
                    "theme": "metals",
                    "signal": "mixed_metals_outperform_single_sleeves",
                    "takeaway": "XAU+XAG+COPPER works better than narrow precious/base-metal sleeves in the current window.",
                }
            )
    benchmark = by_batch.get("commodities_benchmark")
    if benchmark and str(benchmark.get("status_label", "")) == "validated":
        benchmark_trades = int(benchmark.get("research_trades", 0) or 0)
        if benchmark_trades < 8:
            regime_takeaways.append(
                {
                    "theme": "basket_width",
                    "signal": "broad_basket_secondary_only",
                    "takeaway": "The broad commodity benchmark validates, but low trade count makes it a secondary sleeve rather than the primary one.",
                }
            )
    if not regime_takeaways:
        if preferred:
            regime_takeaways.append(
                {
                    "theme": "selection",
                    "signal": "prefer_validated_batches",
                    "takeaway": "Prefer validated batches with enough trades and accepted candidates; keep the rest in research queue.",
                }
            )
        else:
            regime_takeaways.append(
                {
                    "theme": "selection",
                    "signal": "prefer_validated_batches",
                    "takeaway": "No batch has validated yet; finish execution first, then prefer validated batches with enough trades and accepted candidates.",
                }
            )
    return {
        "preferred_batches": preferred,
        "fragile_batches": fragile,
        "research_queue_batches": research_queue,
        "avoid_batches": avoid,
        "market_regime_takeaways": regime_takeaways,
    }


def derive_symbol_attribution(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    batch_rankings: list[dict[str, Any]] = []
    top_symbols: dict[str, dict[str, Any]] = {}
    for row in batch_results:
        if not isinstance(row, dict):
            continue
        batch = str(row.get("batch", "")).strip()
        rb = row.get("research_backtest", {}) if isinstance(row.get("research_backtest"), dict) else {}
        by_symbol = rb.get("best_by_symbol", {}) if isinstance(rb.get("best_by_symbol"), dict) else {}
        ranked_symbols: list[dict[str, Any]] = []
        for symbol, stats in by_symbol.items():
            if not isinstance(stats, dict):
                continue
            ranked_symbols.append(
                {
                    "symbol": str(symbol),
                    "total_pnl": float(stats.get("total_pnl", 0.0) or 0.0),
                    "avg_pnl": float(stats.get("avg_pnl", 0.0) or 0.0),
                    "trade_count": int(stats.get("trade_count", 0) or 0),
                    "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
                }
            )
        ranked_symbols.sort(key=lambda item: float(item.get("total_pnl", 0.0)), reverse=True)
        if batch:
            batch_rankings.append({"batch": batch, "ranked_symbols": ranked_symbols})
            if ranked_symbols:
                top = dict(ranked_symbols[0])
                top["batch"] = batch
                top_symbols[batch] = top
    return {
        "batch_symbol_rankings": batch_rankings,
        "top_symbol_by_batch": top_symbols,
    }


def derive_regime_attribution(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    batch_regime_rankings: list[dict[str, Any]] = []
    top_symbol_by_batch_regime: dict[str, dict[str, dict[str, Any]]] = {}
    dominant_regime_by_batch: dict[str, dict[str, Any]] = {}
    for row in batch_results:
        if not isinstance(row, dict):
            continue
        batch = str(row.get("batch", "")).strip()
        rb = row.get("research_backtest", {}) if isinstance(row.get("research_backtest"), dict) else {}
        by_symbol_regime = rb.get("best_by_symbol_regime", {}) if isinstance(rb.get("best_by_symbol_regime"), dict) else {}
        regime_rows: list[dict[str, Any]] = []
        regime_totals: list[dict[str, Any]] = []
        top_by_regime: dict[str, dict[str, Any]] = {}
        for symbol, regime_map in by_symbol_regime.items():
            if not isinstance(regime_map, dict):
                continue
            for regime_name, stats in regime_map.items():
                if not isinstance(stats, dict):
                    continue
                regime_name = str(regime_name).strip()
                if not regime_name:
                    continue
                row_payload = {
                    "symbol": str(symbol),
                    "regime": regime_name,
                    "total_pnl": float(stats.get("total_pnl", 0.0) or 0.0),
                    "avg_pnl": float(stats.get("avg_pnl", 0.0) or 0.0),
                    "trade_count": int(stats.get("trade_count", 0) or 0),
                    "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
                }
                regime_rows.append(row_payload)
        regime_names = sorted({str(item.get("regime", "")).strip() for item in regime_rows if str(item.get("regime", "")).strip()})
        for regime_name in regime_names:
            ranked_symbols = [item for item in regime_rows if str(item.get("regime", "")) == regime_name]
            ranked_symbols.sort(key=lambda item: float(item.get("total_pnl", 0.0)), reverse=True)
            regime_total_pnl = float(sum(float(item.get("total_pnl", 0.0)) for item in ranked_symbols))
            regime_trade_count = int(sum(int(item.get("trade_count", 0)) for item in ranked_symbols))
            regime_payload = {
                "regime": regime_name,
                "total_pnl": regime_total_pnl,
                "trade_count": regime_trade_count,
                "ranked_symbols": ranked_symbols,
            }
            regime_totals.append(regime_payload)
            if ranked_symbols:
                top_by_regime[regime_name] = {
                    "batch": batch,
                    "regime": regime_name,
                    **dict(ranked_symbols[0]),
                }
        regime_totals.sort(key=lambda item: float(item.get("total_pnl", 0.0)), reverse=True)
        if batch:
            batch_regime_rankings.append({"batch": batch, "ranked_regimes": regime_totals})
            top_symbol_by_batch_regime[batch] = top_by_regime
            if regime_totals:
                dominant = dict(regime_totals[0])
                dominant_regime_by_batch[batch] = {
                    "batch": batch,
                    "regime": str(dominant.get("regime", "")),
                    "total_pnl": float(dominant.get("total_pnl", 0.0)),
                    "trade_count": int(dominant.get("trade_count", 0)),
                    "top_symbol": str(((dominant.get("ranked_symbols", []) or [{}])[0]).get("symbol", "")),
                }
    return {
        "batch_regime_rankings": batch_regime_rankings,
        "top_symbol_by_batch_regime": top_symbol_by_batch_regime,
        "dominant_regime_by_batch": dominant_regime_by_batch,
    }


def derive_regime_playbook(
    batch_results: list[dict[str, Any]],
    regime_attribution: dict[str, Any],
) -> dict[str, Any]:
    batch_rules: list[dict[str, Any]] = []
    preferred_focus: list[str] = []
    avoid_focus: list[str] = []
    takeaways: list[dict[str, str]] = []

    batch_map = {
        str(row.get("batch", "")).strip(): row
        for row in batch_results
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    for item in regime_attribution.get("batch_regime_rankings", []):
        if not isinstance(item, dict):
            continue
        batch = str(item.get("batch", "")).strip()
        ranked_regimes = item.get("ranked_regimes", [])
        if not batch:
            continue
        if not ranked_regimes:
            batch_rules.append(
                {
                    "batch": batch,
                    "status": "inactive",
                    "dominant_regime": "",
                    "leader_symbols": [],
                    "avoid_regimes": [],
                    "execution_profile": "inactive",
                    "takeaway": "No regime-level edge is currently available.",
                }
            )
            avoid_focus.append(batch)
            continue

        dominant = ranked_regimes[0] if isinstance(ranked_regimes[0], dict) else {}
        dominant_regime = str(dominant.get("regime", "")).strip()
        dominant_total = float(dominant.get("total_pnl", 0.0) or 0.0)
        dominant_symbols = dominant.get("ranked_symbols", []) if isinstance(dominant.get("ranked_symbols"), list) else []
        top_total = float((dominant_symbols[0] or {}).get("total_pnl", 0.0)) if dominant_symbols else 0.0
        leader_symbols = [
            str(row.get("symbol", "")).strip()
            for row in dominant_symbols
            if str(row.get("symbol", "")).strip()
            and (
                abs(float(row.get("total_pnl", 0.0) or 0.0) - top_total) <= 1e-9
                or (abs(top_total) > 1e-9 and float(row.get("total_pnl", 0.0) or 0.0) / top_total >= 0.95)
            )
        ]
        second = ranked_regimes[1] if len(ranked_regimes) > 1 and isinstance(ranked_regimes[1], dict) else {}
        avoid_regimes: list[str] = []
        execution_profile = "single_regime"
        takeaway = f"Bias the batch toward {dominant_regime} with leader {'/'.join(leader_symbols) or 'n/a'}."
        if dominant_regime == "强趋势":
            execution_profile = "trend_only"
            takeaway = f"Use {batch} as a trend sleeve; leader {'/'.join(leader_symbols) or 'n/a'} is strongest in 强趋势."
        second_regime = str(second.get("regime", "")).strip()
        second_total = float(second.get("total_pnl", 0.0) or 0.0)
        if second_regime == "震荡" and second_total < 0.0 < dominant_total:
            avoid_regimes.append("震荡")
            execution_profile = "range_avoid"
            takeaway = (
                f"Use {batch} only in {dominant_regime}; avoid 震荡 because regime PnL turns negative while "
                f"{'/'.join(leader_symbols) or 'the leader'} only works in trend."
            )

        row = batch_map.get(batch, {})
        outcome = row.get("outcome", {}) if isinstance(row.get("outcome"), dict) else {}
        if str(outcome.get("label", "")) == "validated":
            preferred_focus.append(batch)
        elif str(outcome.get("label", "")) == "deprioritize":
            avoid_focus.append(batch)

        batch_rules.append(
            {
                "batch": batch,
                "status": str(outcome.get("label", "")).strip() or "unknown",
                "dominant_regime": dominant_regime,
                "leader_symbols": leader_symbols,
                "avoid_regimes": avoid_regimes,
                "execution_profile": execution_profile,
                "takeaway": takeaway,
            }
        )
        takeaways.append(
            {
                "batch": batch,
                "signal": execution_profile,
                "takeaway": takeaway,
            }
        )

    return {
        "batch_rules": batch_rules,
        "preferred_focus_batches": preferred_focus,
        "avoid_focus_batches": avoid_focus,
        "takeaways": takeaways,
    }


def classify_leader_structure(ranked_symbols: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [
        {
            "symbol": str(row.get("symbol", "")).strip(),
            "total_pnl": float(row.get("total_pnl", 0.0) or 0.0),
            "trade_count": int(row.get("trade_count", 0) or 0),
        }
        for row in ranked_symbols
        if isinstance(row, dict)
        and str(row.get("symbol", "")).strip()
        and float(row.get("total_pnl", 0.0) or 0.0) > 0.0
    ]
    if not positive:
        return {
            "leader_structure": "inactive",
            "leader_symbols": [],
            "positive_symbol_count": 0,
            "positive_total_pnl": 0.0,
            "top_share": 0.0,
            "second_share": 0.0,
            "takeaway": "No positive symbol contribution was observed.",
        }
    total_positive = float(sum(row["total_pnl"] for row in positive))
    top = positive[0]
    second = positive[1] if len(positive) > 1 else None
    top_share = float(top["total_pnl"] / total_positive) if total_positive > 0 else 0.0
    second_share = float(second["total_pnl"] / total_positive) if second and total_positive > 0 else 0.0
    coequal = [
        row["symbol"]
        for row in positive
        if top["total_pnl"] > 0.0 and float(row["total_pnl"] / top["total_pnl"]) >= 0.95
    ]
    if len(coequal) >= 2:
        structure = "paired_symmetric" if len(coequal) == 2 else "coequal_cluster"
        takeaway = f"Top contribution is shared almost evenly by {'/'.join(coequal)}."
        leaders = coequal
    elif top_share >= 0.65:
        structure = "single_leader"
        takeaway = f"{top['symbol']} dominates the sleeve and should be treated as the lead leg."
        leaders = [top["symbol"]]
    elif second is not None and (top_share + second_share) >= 0.70:
        structure = "dual_leader"
        takeaway = f"{top['symbol']} and {second['symbol']} jointly drive most of the sleeve outcome."
        leaders = [top["symbol"], second["symbol"]]
    else:
        structure = "distributed"
        takeaway = "Positive contribution is spread across multiple legs without a single dominant leader."
        leaders = [row["symbol"] for row in positive[:3]]
    return {
        "leader_structure": structure,
        "leader_symbols": leaders,
        "positive_symbol_count": int(len(positive)),
        "positive_total_pnl": total_positive,
        "top_share": top_share,
        "second_share": second_share,
        "takeaway": takeaway,
    }


def derive_leader_profiles(
    symbol_attribution: dict[str, Any],
    regime_attribution: dict[str, Any],
) -> dict[str, Any]:
    overall_map = {
        str(row.get("batch", "")).strip(): row
        for row in symbol_attribution.get("batch_symbol_rankings", [])
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    regime_map = {
        str(row.get("batch", "")).strip(): row
        for row in regime_attribution.get("batch_regime_rankings", [])
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    batch_profiles: list[dict[str, Any]] = []
    by_batch: dict[str, Any] = {}
    for batch in sorted(set(overall_map) | set(regime_map)):
        overall_ranked = overall_map.get(batch, {}).get("ranked_symbols", [])
        overall_profile = classify_leader_structure(overall_ranked if isinstance(overall_ranked, list) else [])
        ranked_regimes = regime_map.get(batch, {}).get("ranked_regimes", [])
        dominant_regime_row = ranked_regimes[0] if isinstance(ranked_regimes, list) and ranked_regimes else {}
        dominant_regime = str(dominant_regime_row.get("regime", "")).strip()
        dominant_ranked = dominant_regime_row.get("ranked_symbols", []) if isinstance(dominant_regime_row, dict) else []
        dominant_profile = classify_leader_structure(dominant_ranked if isinstance(dominant_ranked, list) else [])
        payload = {
            "batch": batch,
            "dominant_regime": dominant_regime,
            "overall_profile": overall_profile,
            "dominant_regime_profile": dominant_profile,
        }
        batch_profiles.append(payload)
        by_batch[batch] = payload
    return {
        "batch_profiles": batch_profiles,
        "by_batch": by_batch,
    }


def derive_batch_relationships(
    batch_results: list[dict[str, Any]],
    playbook: dict[str, Any],
    symbol_attribution: dict[str, Any],
) -> dict[str, Any]:
    batch_map = {
        str(row.get("batch", "")).strip(): row
        for row in batch_results
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    preferred_batches = [str(row.get("batch", "")).strip() for row in playbook.get("preferred_batches", []) if str(row.get("batch", "")).strip()]
    top_symbol_by_batch = symbol_attribution.get("top_symbol_by_batch", {}) if isinstance(symbol_attribution.get("top_symbol_by_batch"), dict) else {}
    shadow_pairs: list[dict[str, Any]] = []
    shadowed = set()
    primary = set(preferred_batches)

    for candidate in preferred_batches:
        cand_row = batch_map.get(candidate, {})
        cand_symbols = {str(s).strip().upper() for s in cand_row.get("symbols", []) if str(s).strip()}
        cand_rb = cand_row.get("research_backtest", {}) if isinstance(cand_row.get("research_backtest"), dict) else {}
        cand_metrics = cand_rb.get("best_metrics", {}) if isinstance(cand_rb.get("best_metrics"), dict) else {}
        cand_score = float(cand_rb.get("best_score", 0.0) or 0.0)
        cand_annual = float(cand_metrics.get("annual_return", 0.0) or 0.0)
        cand_trades = int(cand_metrics.get("trades", 0) or 0)
        cand_top_symbol = str((top_symbol_by_batch.get(candidate, {}) or {}).get("symbol", "")).strip()
        for baseline in preferred_batches:
            if baseline == candidate:
                continue
            base_row = batch_map.get(baseline, {})
            base_symbols = {str(s).strip().upper() for s in base_row.get("symbols", []) if str(s).strip()}
            if not base_symbols or not cand_symbols or not base_symbols < cand_symbols:
                continue
            base_rb = base_row.get("research_backtest", {}) if isinstance(base_row.get("research_backtest"), dict) else {}
            base_metrics = base_rb.get("best_metrics", {}) if isinstance(base_rb.get("best_metrics"), dict) else {}
            base_score = float(base_rb.get("best_score", 0.0) or 0.0)
            base_annual = float(base_metrics.get("annual_return", 0.0) or 0.0)
            base_trades = int(base_metrics.get("trades", 0) or 0)
            base_top_symbol = str((top_symbol_by_batch.get(baseline, {}) or {}).get("symbol", "")).strip()
            overlap_ratio = float(len(base_symbols & cand_symbols) / max(1, len(cand_symbols)))
            if (
                abs(cand_score - base_score) <= 1e-9
                and abs(cand_annual - base_annual) <= 1e-9
                and cand_trades == base_trades
                and bool(cand_top_symbol)
                and cand_top_symbol == base_top_symbol
            ):
                shadow_pairs.append(
                    {
                        "primary_batch": baseline,
                        "shadowed_batch": candidate,
                        "relation": "subset_dominance_same_metrics",
                        "same_top_symbol": True,
                        "top_symbol": cand_top_symbol,
                        "overlap_ratio": overlap_ratio,
                        "score_gap": abs(cand_score - base_score),
                        "annual_return_gap": abs(cand_annual - base_annual),
                        "trade_gap": abs(cand_trades - base_trades),
                    }
                )
                shadowed.add(candidate)
                primary.add(baseline)
                break

    primary_batches = [batch for batch in preferred_batches if batch not in shadowed]
    secondary_batches = [batch for batch in preferred_batches if batch in shadowed]
    takeaways: list[dict[str, str]] = []
    for item in shadow_pairs:
        takeaways.append(
            {
                "theme": "batch_relationship",
                "signal": "shadowed_validated_batch",
                "takeaway": f"{item['shadowed_batch']} is likely a secondary sleeve; {item['primary_batch']} already explains the validated edge with the same top symbol {item['top_symbol']}.",
            }
        )
    return {
        "primary_batches": primary_batches,
        "secondary_batches": secondary_batches,
        "shadow_pairs": shadow_pairs,
        "relationship_takeaways": takeaways,
    }


def derive_research_action_ladder(
    summary: dict[str, Any],
    playbook: dict[str, Any],
    regime_playbook: dict[str, Any],
    relationships: dict[str, Any],
    leader_profiles: dict[str, Any],
) -> dict[str, Any]:
    ranked = [row for row in summary.get("ranked_batches", []) if isinstance(row, dict)]
    preferred = {
        str(row.get("batch", "")).strip()
        for row in playbook.get("preferred_batches", [])
        if str(row.get("batch", "")).strip()
    }
    fragile = {
        str(row.get("batch", "")).strip()
        for row in playbook.get("fragile_batches", [])
        if str(row.get("batch", "")).strip()
    }
    research_queue = {
        str(row.get("batch", "")).strip()
        for row in playbook.get("research_queue_batches", [])
        if str(row.get("batch", "")).strip()
    }
    avoid = {
        str(row.get("batch", "")).strip()
        for row in playbook.get("avoid_batches", [])
        if str(row.get("batch", "")).strip()
    }
    primary_batches = {
        str(batch).strip()
        for batch in relationships.get("primary_batches", [])
        if str(batch).strip()
    }
    secondary_batches = {
        str(batch).strip()
        for batch in relationships.get("secondary_batches", [])
        if str(batch).strip()
    }
    regime_rules = {
        str(row.get("batch", "")).strip(): row
        for row in regime_playbook.get("batch_rules", [])
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    leader_map = {
        str(batch).strip(): row
        for batch, row in (leader_profiles.get("by_batch", {}) or {}).items()
        if str(batch).strip() and isinstance(row, dict)
    }

    priority_rank = {
        "focus_primary": 0,
        "focus_with_regime_filter": 1,
        "shadow_only": 2,
        "watch_fragile": 3,
        "research_queue": 4,
        "avoid": 5,
        "observe": 6,
    }
    queue: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = {key: [] for key in priority_rank}
    paired_focus_batches: list[str] = []
    single_leader_batches: list[str] = []
    distributed_batches: list[str] = []
    for row in ranked:
        batch = str(row.get("batch", "")).strip()
        if not batch:
            continue
        rule = regime_rules.get(batch, {})
        leader_profile = leader_map.get(batch, {})
        dominant_profile = leader_profile.get("dominant_regime_profile", {}) if isinstance(leader_profile, dict) else {}
        leader_structure = str(dominant_profile.get("leader_structure", "")).strip()
        leader_note = str(dominant_profile.get("takeaway", "")).strip()
        execution_profile = str(rule.get("execution_profile", "")).strip() or "unknown"
        dominant_regime = str(rule.get("dominant_regime", "")).strip()
        leader_symbols = [
            str(symbol).strip()
            for symbol in rule.get("leader_symbols", [])
            if str(symbol).strip()
        ]
        avoid_regimes = [
            str(regime).strip()
            for regime in rule.get("avoid_regimes", [])
            if str(regime).strip()
        ]
        action = "observe"
        reason = "Batch does not yet have a clear research action."
        if batch in avoid or execution_profile == "inactive":
            action = "avoid"
            reason = "No regime-level edge is currently available."
        elif batch in secondary_batches:
            action = "shadow_only"
            reason = "Validated metrics are shadowed by a primary sleeve with the same dominant edge."
        elif batch in fragile:
            action = "watch_fragile"
            reason = "Validated, but confirmation depth is still thin."
        elif batch in research_queue:
            action = "research_queue"
            reason = "Promising, but not yet validated enough to prioritize."
        elif batch in preferred and execution_profile == "range_avoid":
            action = "focus_with_regime_filter"
            reason = "Use only in the dominant regime and actively avoid the weak regime."
        elif batch in preferred and execution_profile in {"trend_only", "single_regime"}:
            action = "focus_primary"
            reason = "Validated primary sleeve with a clear dominant regime."
        queue_item = {
            "batch": batch,
            "action": action,
            "priority": int(priority_rank[action]),
            "status_label": str(row.get("status_label", "")).strip(),
            "execution_profile": execution_profile,
            "dominant_regime": dominant_regime,
            "leader_symbols": leader_symbols,
            "leader_structure": leader_structure,
            "leader_note": leader_note,
            "avoid_regimes": avoid_regimes,
            "reason": reason,
        }
        queue.append(queue_item)
        grouped[action].append(batch)
        if action in {"focus_primary", "focus_with_regime_filter"} and leader_structure in {"paired_symmetric", "dual_leader"}:
            paired_focus_batches.append(batch)
        elif leader_structure == "single_leader":
            single_leader_batches.append(batch)
        elif leader_structure in {"distributed", "coequal_cluster"}:
            distributed_batches.append(batch)
    queue.sort(key=lambda item: (int(item.get("priority", 99)), -float(next((row.get("research_score", 0.0) for row in ranked if str(row.get("batch", "")).strip() == item["batch"]), 0.0))))
    return {
        "ranked_actions": queue,
        "focus_primary_batches": grouped["focus_primary"],
        "focus_with_regime_filter_batches": grouped["focus_with_regime_filter"],
        "shadow_only_batches": grouped["shadow_only"],
        "watch_fragile_batches": grouped["watch_fragile"],
        "research_queue_batches": grouped["research_queue"],
        "avoid_batches": grouped["avoid"],
        "focus_now_batches": grouped["focus_primary"] + grouped["focus_with_regime_filter"],
        "paired_focus_batches": paired_focus_batches,
        "single_leader_batches": single_leader_batches,
        "distributed_batches": distributed_batches,
    }


def derive_microstructure_playbook(
    universe_payload: dict[str, Any],
    batch_results: list[dict[str, Any]],
    action_ladder: dict[str, Any],
) -> dict[str, Any]:
    crypto_selected = {
        str(symbol).strip().upper()
        for symbol in (((universe_payload.get("crypto") or {}).get("selected")) or [])
        if str(symbol).strip()
    }
    focus_now = {
        str(batch).strip()
        for batch in action_ladder.get("focus_now_batches", [])
        if str(batch).strip()
    }
    research_queue = {
        str(batch).strip()
        for batch in action_ladder.get("research_queue_batches", [])
        if str(batch).strip()
    }

    batch_profiles: list[dict[str, Any]] = []
    cvd_full: list[str] = []
    cvd_partial: list[str] = []
    cvd_none: list[str] = []
    cvd_priority: list[str] = []
    focus_macro_only: list[str] = []
    mixed_bridge_batches: list[str] = []

    for row in batch_results:
        if not isinstance(row, dict):
            continue
        batch = str(row.get("batch", "")).strip()
        symbols = [str(symbol).strip().upper() for symbol in row.get("symbols", []) if str(symbol).strip()]
        if not batch or not symbols:
            continue
        supported = [symbol for symbol in symbols if symbol in crypto_selected or symbol.endswith("USDT")]
        unsupported = [symbol for symbol in symbols if symbol not in supported]
        coverage_ratio = float(len(supported) / max(1, len(symbols)))
        if len(supported) == len(symbols):
            coverage_label = "cvd_lite_full"
            recommendation = "cvd_lite_primary_confirm"
            cvd_full.append(batch)
        elif not supported:
            coverage_label = "cvd_lite_none"
            recommendation = "macro_structure_only"
            cvd_none.append(batch)
        else:
            coverage_label = "cvd_lite_partial"
            recommendation = "split_macro_and_micro"
            cvd_partial.append(batch)
            mixed_bridge_batches.append(batch)

        if batch in focus_now and coverage_label == "cvd_lite_none":
            focus_macro_only.append(batch)
        if batch in research_queue and coverage_label in {"cvd_lite_full", "cvd_lite_partial"}:
            cvd_priority.append(batch)

        batch_profiles.append(
            {
                "batch": batch,
                "coverage_label": coverage_label,
                "coverage_ratio": coverage_ratio,
                "cvd_eligible_symbols": supported,
                "proxy_only_symbols": unsupported,
                "recommendation": recommendation,
            }
        )

    if focus_macro_only:
        overall_takeaway = (
            "Current focus batches are mostly commodity sleeves, so ICT+CVD-lite should remain a secondary "
            "research path rather than the primary driver of batch selection."
        )
    elif cvd_priority:
        overall_takeaway = (
            "CVD-lite is best used on crypto research_queue batches where venue trade flow exists and can act as a filter."
        )
    else:
        overall_takeaway = "CVD-lite coverage is available, but it does not yet align with the current preferred sleeves."

    return {
        "batch_profiles": batch_profiles,
        "cvd_lite_supported_batches": cvd_full,
        "cvd_lite_partial_batches": cvd_partial,
        "cvd_lite_unsupported_batches": cvd_none,
        "cvd_priority_batches": cvd_priority,
        "focus_macro_only_batches": focus_macro_only,
        "mixed_bridge_batches": mixed_bridge_batches,
        "overall_takeaway": overall_takeaway,
    }


def derive_crypto_cvd_queue_profile(
    batch_results: list[dict[str, Any]],
    action_ladder: dict[str, Any],
    regime_playbook: dict[str, Any],
    leader_profiles: dict[str, Any],
    microstructure_playbook: dict[str, Any],
) -> dict[str, Any]:
    batch_map = {
        str(row.get("batch", "")).strip(): row
        for row in batch_results
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    regime_rules = {
        str(row.get("batch", "")).strip(): row
        for row in regime_playbook.get("batch_rules", [])
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    leader_map = {
        str(batch).strip(): row
        for batch, row in (leader_profiles.get("by_batch", {}) or {}).items()
        if str(batch).strip() and isinstance(row, dict)
    }
    micro_profiles = {
        str(row.get("batch", "")).strip(): row
        for row in microstructure_playbook.get("batch_profiles", [])
        if isinstance(row, dict) and str(row.get("batch", "")).strip()
    }
    research_queue = [
        str(batch).strip()
        for batch in action_ladder.get("research_queue_batches", [])
        if str(batch).strip()
    ]
    cvd_priority = {
        str(batch).strip()
        for batch in microstructure_playbook.get("cvd_priority_batches", [])
        if str(batch).strip()
    }

    batch_profiles: list[dict[str, Any]] = []
    priority_batches: list[str] = []
    trend_confirmation_batches: list[str] = []
    reversal_watch_batches: list[str] = []
    mixed_bridge_filter_batches: list[str] = []
    basket_consensus_batches: list[str] = []
    leader_alignment_batches: list[str] = []

    for batch in research_queue:
        micro_row = micro_profiles.get(batch, {})
        coverage_label = str(micro_row.get("coverage_label", "")).strip()
        if batch not in cvd_priority and coverage_label not in {"cvd_lite_full", "cvd_lite_partial"}:
            continue
        regime_row = regime_rules.get(batch, {})
        leader_row = leader_map.get(batch, {})
        dominant_profile = (
            leader_row.get("dominant_regime_profile", {})
            if isinstance(leader_row.get("dominant_regime_profile"), dict)
            else {}
        )
        dominant_regime = str(regime_row.get("dominant_regime", "")).strip()
        execution_profile = str(regime_row.get("execution_profile", "")).strip() or "unknown"
        leader_structure = str(dominant_profile.get("leader_structure", "")).strip() or "unknown"
        leader_symbols = [
            str(symbol).strip()
            for symbol in regime_row.get("leader_symbols", [])
            if str(symbol).strip()
        ]
        status_label = str((batch_map.get(batch, {}).get("outcome", {}) or {}).get("label", "")).strip()

        if coverage_label == "cvd_lite_partial":
            queue_mode = "mixed_bridge_filter"
            preferred_contexts = ["unclear"]
            veto_biases = ["cross_asset_mix", "proxy_leg_dominance"]
            mixed_bridge_filter_batches.append(batch)
        elif dominant_regime == "震荡":
            queue_mode = "reversal_absorption_watch"
            preferred_contexts = ["reversal", "absorption", "failed_auction"]
            veto_biases = ["trend_chase", "one_leg_breakout"]
            reversal_watch_batches.append(batch)
        else:
            queue_mode = "trend_confirmation"
            preferred_contexts = ["continuation", "failed_auction"]
            veto_biases = ["sweep_without_delta_confirmation", "effort_result_divergence"]
            trend_confirmation_batches.append(batch)

        if leader_structure in {"distributed", "coequal_cluster"}:
            trust_requirement = "basket_consensus"
            basket_consensus_batches.append(batch)
        elif leader_structure in {"paired_symmetric", "dual_leader"}:
            trust_requirement = "dual_leader_alignment"
            leader_alignment_batches.append(batch)
        else:
            trust_requirement = "leader_plus_index_alignment"
            leader_alignment_batches.append(batch)

        if batch in cvd_priority:
            priority_batches.append(batch)

        takeaway = (
            f"Use {queue_mode} with {trust_requirement}; dominant regime={dominant_regime or 'unknown'} "
            f"and leaders={', '.join(leader_symbols) or 'n/a'}."
        )
        batch_profiles.append(
            {
                "batch": batch,
                "status_label": status_label,
                "coverage_label": coverage_label,
                "coverage_ratio": float(micro_row.get("coverage_ratio", 0.0) or 0.0),
                "cvd_eligible_symbols": list(micro_row.get("cvd_eligible_symbols", []) or []),
                "proxy_only_symbols": list(micro_row.get("proxy_only_symbols", []) or []),
                "dominant_regime": dominant_regime,
                "execution_profile": execution_profile,
                "leader_structure": leader_structure,
                "leader_symbols": leader_symbols,
                "queue_mode": queue_mode,
                "trust_requirement": trust_requirement,
                "preferred_contexts": preferred_contexts,
                "veto_biases": veto_biases,
                "takeaway": takeaway,
            }
        )

    if priority_batches:
        overall_takeaway = (
            "Crypto research_queue batches should use ICT+CVD-lite as a regime-aware filter: "
            "trend batches want continuation confirmation, while range batches want reversal/absorption validation."
        )
    else:
        overall_takeaway = (
            "No research_queue batch currently combines enough CVD-lite coverage and research quality to justify a dedicated queue profile."
        )

    return {
        "batch_profiles": batch_profiles,
        "priority_batches": priority_batches,
        "trend_confirmation_batches": trend_confirmation_batches,
        "reversal_watch_batches": reversal_watch_batches,
        "mixed_bridge_filter_batches": mixed_bridge_filter_batches,
        "basket_consensus_batches": basket_consensus_batches,
        "leader_alignment_batches": leader_alignment_batches,
        "overall_takeaway": overall_takeaway,
    }


def _batch_execution_context_name() -> str:
    methods = set(mp.get_all_start_methods())
    if "spawn" in methods:
        return "spawn"
    if "forkserver" in methods:
        return "forkserver"
    return "fork" if "fork" in methods else mp.get_start_method()


def _run_single_batch_worker(
    *,
    batch_name: str,
    symbols: list[str],
    output_root: Path,
    start: dt.date,
    end: dt.date,
    hours_budget: float,
    workers: int,
    max_trials_per_mode: int,
    review_days: int,
    run_strategy_lab_enabled: bool,
    strategy_lab_candidate_count: int,
    seed: int,
    result_queue: Any,
) -> None:
    started = now_utc()
    batch_result: dict[str, Any] = {
        "batch": batch_name,
        "symbols": list(symbols),
        "status": "pending",
        "timed_out": False,
    }
    try:
        research_payload = run_research_backtest(
            output_root=output_root,
            core_symbols=symbols,
            start=start,
            end=end,
            hours_budget=hours_budget,
            max_symbols=len(symbols),
            report_symbol_cap=5,
            workers=workers,
            max_trials_per_mode=max_trials_per_mode,
            seed=seed,
            review_days=review_days,
        ).to_dict()
        batch_result["research_backtest"] = summarize_research(research_payload)
        batch_result["status"] = "research_ok"
        if run_strategy_lab_enabled:
            strategy_payload = run_strategy_lab(
                output_root=output_root,
                core_symbols=symbols,
                start=start,
                end=end,
                max_symbols=len(symbols),
                report_symbol_cap=5,
                workers=workers,
                review_days=review_days,
                candidate_count=strategy_lab_candidate_count,
            ).to_dict()
            batch_result["strategy_lab"] = summarize_strategy_lab(strategy_payload)
            batch_result["status"] = "research_and_lab_ok"
        batch_result["outcome"] = classify_batch_outcome(batch_result)
    except Exception as exc:  # noqa: BLE001
        batch_result["status"] = "failed"
        batch_result["error"] = f"{type(exc).__name__}:{exc}"
    batch_result["elapsed_seconds"] = max(0.0, (now_utc() - started).total_seconds())
    result_queue.put(batch_result)


def effective_batch_timeout_seconds(*, batch_timeout_seconds: float, symbols: list[str]) -> float:
    base_timeout = max(1.0, float(batch_timeout_seconds))
    symbol_count = max(1, len([str(symbol).strip() for symbol in symbols if str(symbol).strip()]))
    symbol_multiplier = max(1.0, float(symbol_count) / float(DEFAULT_BATCH_TIMEOUT_SYMBOL_BASE))
    return base_timeout * symbol_multiplier


def _run_single_batch_with_timeout(
    *,
    batch_name: str,
    symbols: list[str],
    output_root: Path,
    start: dt.date,
    end: dt.date,
    hours_budget: float,
    workers: int,
    max_trials_per_mode: int,
    review_days: int,
    run_strategy_lab_enabled: bool,
    strategy_lab_candidate_count: int,
    seed: int,
    batch_timeout_seconds: float,
) -> dict[str, Any]:
    timeout_seconds = effective_batch_timeout_seconds(
        batch_timeout_seconds=batch_timeout_seconds,
        symbols=symbols,
    )
    started = now_utc()
    ctx = mp.get_context(_batch_execution_context_name())
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_single_batch_worker,
        kwargs={
            "batch_name": batch_name,
            "symbols": list(symbols),
            "output_root": output_root,
            "start": start,
            "end": end,
            "hours_budget": hours_budget,
            "workers": workers,
            "max_trials_per_mode": max_trials_per_mode,
            "review_days": review_days,
            "run_strategy_lab_enabled": run_strategy_lab_enabled,
            "strategy_lab_candidate_count": strategy_lab_candidate_count,
            "seed": seed,
            "result_queue": result_queue,
        },
    )
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        result_queue.close()
        result_queue.join_thread()
        return {
            "batch": batch_name,
            "symbols": list(symbols),
            "status": "failed",
            "timed_out": True,
            "error": f"TimeoutError:batch_timeout_seconds_exceeded:{timeout_seconds:g}",
            "effective_timeout_seconds": timeout_seconds,
            "elapsed_seconds": max(0.0, (now_utc() - started).total_seconds()),
        }
    try:
        batch_result = result_queue.get(timeout=1.0)
    except queue_mod.Empty:
        batch_result = {
            "batch": batch_name,
            "symbols": list(symbols),
            "status": "failed",
            "timed_out": False,
            "error": f"RuntimeError:no_batch_result_emitted(exitcode={proc.exitcode})",
            "effective_timeout_seconds": timeout_seconds,
            "elapsed_seconds": max(0.0, (now_utc() - started).total_seconds()),
        }
    finally:
        result_queue.close()
        result_queue.join_thread()
    if "elapsed_seconds" not in batch_result:
        batch_result["elapsed_seconds"] = max(0.0, (now_utc() - started).total_seconds())
    if "timed_out" not in batch_result:
        batch_result["timed_out"] = False
    batch_result["effective_timeout_seconds"] = float(batch_result.get("effective_timeout_seconds", timeout_seconds) or timeout_seconds)
    return batch_result


def run_batches(
    *,
    universe_payload: dict[str, Any],
    selected_batches: dict[str, list[str]],
    output_root: Path,
    start: dt.date,
    end: dt.date,
    hours_budget: float,
    workers: int,
    max_trials_per_mode: int,
    review_days: int,
    run_strategy_lab_enabled: bool,
    strategy_lab_candidate_count: int,
    seed: int,
    dry_run: bool,
    batch_timeout_seconds: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for batch_name, symbols in selected_batches.items():
        batch_result: dict[str, Any] = {
            "batch": batch_name,
            "symbols": symbols,
            "status": "planned" if dry_run else "pending",
        }
        if dry_run:
            results.append(batch_result)
            continue
        batch_result = _run_single_batch_with_timeout(
            batch_name=batch_name,
            symbols=symbols,
            output_root=output_root,
            start=start,
            end=end,
            hours_budget=hours_budget,
            workers=workers,
            max_trials_per_mode=max_trials_per_mode,
            review_days=review_days,
            run_strategy_lab_enabled=run_strategy_lab_enabled,
            strategy_lab_candidate_count=strategy_lab_candidate_count,
            seed=seed,
            batch_timeout_seconds=batch_timeout_seconds,
        )
        results.append(batch_result)
    return results


def overall_run_status(*, batch_results: list[dict[str, Any]], dry_run: bool) -> str:
    if dry_run:
        return "dry_run"
    if any(str(row.get("status") or "").strip() == "failed" for row in batch_results if isinstance(row, dict)):
        return "partial_failure"
    return "ok"


def build_report(
    *,
    universe_payload: dict[str, Any],
    batch_results: list[dict[str, Any]],
    dry_run: bool,
    start: dt.date,
    end: dt.date,
    commodity_paper_ticket_lane: dict[str, Any] | None = None,
    commodity_paper_ticket_book: dict[str, Any] | None = None,
    commodity_paper_execution_preview: dict[str, Any] | None = None,
    commodity_paper_execution_artifact: dict[str, Any] | None = None,
    commodity_paper_execution_queue: dict[str, Any] | None = None,
    commodity_paper_execution_review: dict[str, Any] | None = None,
    commodity_paper_execution_retro: dict[str, Any] | None = None,
    crypto_symbol_route_handoff: dict[str, Any] | None = None,
    crypto_route_brief: dict[str, Any] | None = None,
    crypto_route_operator_brief: dict[str, Any] | None = None,
    bnb_flow_focus: dict[str, Any] | None = None,
) -> str:
    summary = summarize_batch_results(batch_results)
    playbook = derive_batch_playbook(summary)
    symbol_attribution = derive_symbol_attribution(batch_results)
    regime_attribution = derive_regime_attribution(batch_results)
    regime_playbook = derive_regime_playbook(batch_results, regime_attribution)
    leader_profiles = derive_leader_profiles(symbol_attribution, regime_attribution)
    relationships = derive_batch_relationships(batch_results, playbook, symbol_attribution)
    action_ladder = derive_research_action_ladder(summary, playbook, regime_playbook, relationships, leader_profiles)
    microstructure_playbook = derive_microstructure_playbook(universe_payload, batch_results, action_ladder)
    crypto_cvd_queue_profile = derive_crypto_cvd_queue_profile(
        batch_results,
        action_ladder,
        regime_playbook,
        leader_profiles,
        microstructure_playbook,
    )
    lines = [
        "# Hot Universe Research",
        f"- universe_source_tier: `{universe_payload.get('source_tier', '')}`",
        f"- start: `{start.isoformat()}`",
        f"- end: `{end.isoformat()}`",
        f"- dry_run: `{str(dry_run).lower()}`",
        "",
        "## Universe",
        f"- crypto: `{', '.join(universe_payload.get('crypto', {}).get('selected', []))}`",
        f"- commodities: `{', '.join(universe_payload.get('commodities', {}).get('selected', []))}`",
        "",
        "## Batches",
    ]
    for row in batch_results:
        lines.append(f"- `{row.get('batch', '')}`: status=`{row.get('status', '')}` symbols=`{', '.join(row.get('symbols', []))}`")
        if isinstance(row.get("research_backtest"), dict):
            rb = row["research_backtest"]
            lines.append(
                f"  research=`{rb.get('best_mode', '')}` score=`{rb.get('best_score', 0.0):.4f}` output=`{rb.get('output_dir', '')}`"
            )
        if isinstance(row.get("strategy_lab"), dict):
            sl = row["strategy_lab"]
            lines.append(
                f"  strategy_lab=`{sl.get('best_candidate_name', '')}` accepted=`{str(sl.get('best_candidate_accepted', False)).lower()}`"
            )
        if isinstance(row.get("outcome"), dict):
            outcome = row["outcome"]
            lines.append(
                f"  outcome=`{outcome.get('label', '')}` trades=`{outcome.get('research_trades', 0)}` accepted_count=`{outcome.get('accepted_count', 0)}`"
            )
            lines.append(f"  takeaway=`{outcome.get('takeaway', '')}`")
        if row.get("error"):
            lines.append(f"  error=`{row.get('error', '')}`")
    lines.extend(
        [
            "",
            "## Ranking",
            f"- planned: `{', '.join(summary.get('planned_batches', []))}`",
            f"- validated: `{', '.join(summary.get('validated_batches', []))}`",
            f"- research_only: `{', '.join(summary.get('research_only_batches', []))}`",
            f"- pilot_only: `{', '.join(summary.get('pilot_only_batches', []))}`",
            f"- deprioritized: `{', '.join(summary.get('deprioritized_batches', []))}`",
            "",
            "## Research Playbook",
            f"- preferred: `{', '.join(item.get('batch', '') for item in playbook.get('preferred_batches', []))}`",
            f"- primary: `{', '.join(relationships.get('primary_batches', []))}`",
            f"- secondary: `{', '.join(relationships.get('secondary_batches', []))}`",
            f"- fragile: `{', '.join(item.get('batch', '') for item in playbook.get('fragile_batches', []))}`",
            f"- research_queue: `{', '.join(item.get('batch', '') for item in playbook.get('research_queue_batches', []))}`",
            f"- avoid: `{', '.join(item.get('batch', '') for item in playbook.get('avoid_batches', []))}`",
            "",
            "## Market Regime Takeaways",
        ]
    )
    for item in playbook.get("market_regime_takeaways", []):
        lines.append(
            f"- `{item.get('theme', '')}` / `{item.get('signal', '')}`: `{item.get('takeaway', '')}`"
        )
    lines.extend(
        [
            "",
            "## Batch Relationships",
        ]
    )
    if not relationships.get("shadow_pairs"):
        lines.append("- `none`")
    else:
        for item in relationships.get("shadow_pairs", []):
            lines.append(
                f"- `{item.get('shadowed_batch', '')}` shadowed_by `{item.get('primary_batch', '')}` top_symbol=`{item.get('top_symbol', '')}` overlap=`{float(item.get('overlap_ratio', 0.0)):.4f}`"
            )
    for item in relationships.get("relationship_takeaways", []):
        lines.append(
            f"  takeaway=`{item.get('takeaway', '')}`"
        )
    lines.extend(
        [
            "",
            "## Symbol Attribution",
        ]
    )
    for item in symbol_attribution.get("batch_symbol_rankings", []):
        lines.append(f"- `{item.get('batch', '')}`:")
        ranked_symbols = item.get("ranked_symbols", [])
        if not ranked_symbols:
            lines.append("  no-symbol-attribution=`none`")
            continue
        for symbol_row in ranked_symbols[:3]:
            lines.append(
                f"  `{symbol_row.get('symbol', '')}` total_pnl=`{float(symbol_row.get('total_pnl', 0.0)):.6f}` trades=`{int(symbol_row.get('trade_count', 0))}` win_rate=`{float(symbol_row.get('win_rate', 0.0)):.4f}`"
            )
    lines.extend(
        [
            "",
            "## Regime Attribution",
        ]
    )
    for item in regime_attribution.get("batch_regime_rankings", []):
        lines.append(f"- `{item.get('batch', '')}`:")
        ranked_regimes = item.get("ranked_regimes", [])
        if not ranked_regimes:
            lines.append("  no-regime-attribution=`none`")
            continue
        for regime_row in ranked_regimes[:2]:
            lines.append(
                f"  regime=`{regime_row.get('regime', '')}` total_pnl=`{float(regime_row.get('total_pnl', 0.0)):.6f}` trades=`{int(regime_row.get('trade_count', 0))}`"
            )
            ranked_symbols = regime_row.get("ranked_symbols", [])
            for symbol_row in ranked_symbols[:2]:
                lines.append(
                    f"    `{symbol_row.get('symbol', '')}` total_pnl=`{float(symbol_row.get('total_pnl', 0.0)):.6f}` trades=`{int(symbol_row.get('trade_count', 0))}` win_rate=`{float(symbol_row.get('win_rate', 0.0)):.4f}`"
                )
    lines.extend(
        [
            "",
            "## Regime Playbook",
            f"- preferred-focus: `{', '.join(regime_playbook.get('preferred_focus_batches', []))}`",
            f"- avoid-focus: `{', '.join(regime_playbook.get('avoid_focus_batches', []))}`",
        ]
    )
    for item in regime_playbook.get("batch_rules", []):
        leader_profile = leader_profiles.get("by_batch", {}).get(str(item.get("batch", "")).strip(), {})
        dominant_profile = leader_profile.get("dominant_regime_profile", {}) if isinstance(leader_profile, dict) else {}
        lines.append(
            f"- `{item.get('batch', '')}`: profile=`{item.get('execution_profile', '')}` dominant_regime=`{item.get('dominant_regime', '')}` leaders=`{', '.join(item.get('leader_symbols', []))}` avoid_regimes=`{', '.join(item.get('avoid_regimes', []))}`"
        )
        lines.append(f"  takeaway=`{item.get('takeaway', '')}`")
        if dominant_profile:
            lines.append(
                f"  leader-structure=`{dominant_profile.get('leader_structure', '')}` top_share=`{float(dominant_profile.get('top_share', 0.0)):.4f}` second_share=`{float(dominant_profile.get('second_share', 0.0)):.4f}`"
            )
            lines.append(f"  leader-note=`{dominant_profile.get('takeaway', '')}`")
    lines.extend(
        [
            "",
            "## Leader Profiles",
        ]
    )
    for item in leader_profiles.get("batch_profiles", []):
        overall = item.get("overall_profile", {}) if isinstance(item.get("overall_profile"), dict) else {}
        dominant = item.get("dominant_regime_profile", {}) if isinstance(item.get("dominant_regime_profile"), dict) else {}
        lines.append(
            f"- `{item.get('batch', '')}`: overall=`{overall.get('leader_structure', '')}` dominant_regime=`{item.get('dominant_regime', '')}` dominant_profile=`{dominant.get('leader_structure', '')}`"
        )
        lines.append(
            f"  overall-leaders=`{', '.join(overall.get('leader_symbols', []))}` dominant-leaders=`{', '.join(dominant.get('leader_symbols', []))}`"
        )
        lines.append(
            f"  overall-note=`{overall.get('takeaway', '')}` dominant-note=`{dominant.get('takeaway', '')}`"
        )
    lines.extend(
        [
            "",
            "## Research Action Ladder",
            f"- focus-now: `{', '.join(action_ladder.get('focus_now_batches', []))}`",
            f"- paired-focus: `{', '.join(action_ladder.get('paired_focus_batches', []))}`",
            f"- single-leader: `{', '.join(action_ladder.get('single_leader_batches', []))}`",
            f"- distributed: `{', '.join(action_ladder.get('distributed_batches', []))}`",
            f"- shadow-only: `{', '.join(action_ladder.get('shadow_only_batches', []))}`",
            f"- watch-fragile: `{', '.join(action_ladder.get('watch_fragile_batches', []))}`",
            f"- research-queue: `{', '.join(action_ladder.get('research_queue_batches', []))}`",
            f"- avoid: `{', '.join(action_ladder.get('avoid_batches', []))}`",
        ]
    )
    for item in action_ladder.get("ranked_actions", []):
        lines.append(
            f"- `{item.get('batch', '')}`: action=`{item.get('action', '')}` profile=`{item.get('execution_profile', '')}` dominant_regime=`{item.get('dominant_regime', '')}` leaders=`{', '.join(item.get('leader_symbols', []))}` leader_structure=`{item.get('leader_structure', '')}` avoid_regimes=`{', '.join(item.get('avoid_regimes', []))}`"
        )
        lines.append(f"  reason=`{item.get('reason', '')}`")
        if item.get("leader_note"):
            lines.append(f"  leader-note=`{item.get('leader_note', '')}`")
    lines.extend(
        [
            "",
            "## Microstructure Playbook",
            f"- cvd-lite-full: `{', '.join(microstructure_playbook.get('cvd_lite_supported_batches', []))}`",
            f"- cvd-lite-partial: `{', '.join(microstructure_playbook.get('cvd_lite_partial_batches', []))}`",
            f"- cvd-lite-none: `{', '.join(microstructure_playbook.get('cvd_lite_unsupported_batches', []))}`",
            f"- cvd-priority: `{', '.join(microstructure_playbook.get('cvd_priority_batches', []))}`",
            f"- focus-macro-only: `{', '.join(microstructure_playbook.get('focus_macro_only_batches', []))}`",
            f"- mixed-bridge: `{', '.join(microstructure_playbook.get('mixed_bridge_batches', []))}`",
            f"- takeaway: `{microstructure_playbook.get('overall_takeaway', '')}`",
        ]
    )
    for item in microstructure_playbook.get("batch_profiles", []):
        lines.append(
            f"- `{item.get('batch', '')}`: coverage=`{item.get('coverage_label', '')}` ratio=`{float(item.get('coverage_ratio', 0.0)):.4f}` cvd=`{', '.join(item.get('cvd_eligible_symbols', []))}` proxy=`{', '.join(item.get('proxy_only_symbols', []))}`"
        )
        lines.append(f"  recommendation=`{item.get('recommendation', '')}`")
    lines.extend(
        [
            "",
            "## Crypto CVD Queue Profile",
            f"- priority-batches: `{', '.join(crypto_cvd_queue_profile.get('priority_batches', []))}`",
            f"- trend-confirmation: `{', '.join(crypto_cvd_queue_profile.get('trend_confirmation_batches', []))}`",
            f"- reversal-watch: `{', '.join(crypto_cvd_queue_profile.get('reversal_watch_batches', []))}`",
            f"- mixed-bridge-filter: `{', '.join(crypto_cvd_queue_profile.get('mixed_bridge_filter_batches', []))}`",
            f"- basket-consensus: `{', '.join(crypto_cvd_queue_profile.get('basket_consensus_batches', []))}`",
            f"- leader-alignment: `{', '.join(crypto_cvd_queue_profile.get('leader_alignment_batches', []))}`",
            f"- takeaway: `{crypto_cvd_queue_profile.get('overall_takeaway', '')}`",
        ]
    )
    for item in crypto_cvd_queue_profile.get("batch_profiles", []):
        lines.append(
            f"- `{item.get('batch', '')}`: queue=`{item.get('queue_mode', '')}` trust=`{item.get('trust_requirement', '')}` regime=`{item.get('dominant_regime', '')}` structure=`{item.get('leader_structure', '')}`"
        )
        lines.append(
            f"  contexts=`{', '.join(item.get('preferred_contexts', []))}` veto=`{', '.join(item.get('veto_biases', []))}` leaders=`{', '.join(item.get('leader_symbols', []))}`"
        )
        lines.append(f"  takeaway=`{item.get('takeaway', '')}`")
    lines.extend(
        [
            "",
            "## Commodity Paper Ticket Lane",
        ]
    )
    if not commodity_paper_ticket_lane:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- route_status: `{commodity_paper_ticket_lane.get('route_status', '')}`",
                f"- ticket_status: `{commodity_paper_ticket_lane.get('ticket_status', '')}`",
                f"- execution_mode: `{commodity_paper_ticket_lane.get('execution_mode', '')}`",
                f"- paper_ready: `{', '.join(commodity_paper_ticket_lane.get('paper_ready_batches', []))}`",
                f"- shadow_only: `{', '.join(commodity_paper_ticket_lane.get('shadow_only_batches', []))}`",
                f"- next_ticket_batch: `{commodity_paper_ticket_lane.get('next_ticket_batch', '')}`",
                f"- next_ticket_symbols: `{', '.join(commodity_paper_ticket_lane.get('next_ticket_symbols', []))}`",
                f"- ticket_stack: `{commodity_paper_ticket_lane.get('ticket_stack_brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Commodity Paper Ticket Book",
        ]
    )
    if not commodity_paper_ticket_book:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- route_status: `{commodity_paper_ticket_book.get('route_status', '')}`",
                f"- ticket_book_status: `{commodity_paper_ticket_book.get('ticket_book_status', '')}`",
                f"- execution_mode: `{commodity_paper_ticket_book.get('execution_mode', '')}`",
                f"- actionable_batches: `{', '.join(commodity_paper_ticket_book.get('actionable_batches', []))}`",
                f"- shadow_batches: `{', '.join(commodity_paper_ticket_book.get('shadow_batches', []))}`",
                f"- actionable_ticket_count: `{commodity_paper_ticket_book.get('actionable_ticket_count', 0)}`",
                f"- next_ticket_id: `{commodity_paper_ticket_book.get('next_ticket_id', '')}`",
                f"- next_ticket_batch: `{commodity_paper_ticket_book.get('next_ticket_batch', '')}`",
                f"- next_ticket_symbol: `{commodity_paper_ticket_book.get('next_ticket_symbol', '')}`",
                f"- next_ticket_regime_gate: `{commodity_paper_ticket_book.get('next_ticket_regime_gate', '')}`",
                f"- ticket_book_stack: `{commodity_paper_ticket_book.get('ticket_book_stack_brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Commodity Paper Execution Preview",
        ]
    )
    if not commodity_paper_execution_preview:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- route_status: `{commodity_paper_execution_preview.get('route_status', '')}`",
                f"- ticket_book_status: `{commodity_paper_execution_preview.get('ticket_book_status', '')}`",
                f"- execution_preview_status: `{commodity_paper_execution_preview.get('execution_preview_status', '')}`",
                f"- execution_mode: `{commodity_paper_execution_preview.get('execution_mode', '')}`",
                f"- preview_ready_batches: `{', '.join(commodity_paper_execution_preview.get('preview_ready_batches', []))}`",
                f"- shadow_only_batches: `{', '.join(commodity_paper_execution_preview.get('shadow_only_batches', []))}`",
                f"- preview_batch_count: `{commodity_paper_execution_preview.get('preview_batch_count', 0)}`",
                f"- next_execution_batch: `{commodity_paper_execution_preview.get('next_execution_batch', '')}`",
                f"- next_execution_symbols: `{', '.join(commodity_paper_execution_preview.get('next_execution_symbols', []))}`",
                f"- next_execution_regime_gate: `{commodity_paper_execution_preview.get('next_execution_regime_gate', '')}`",
                f"- next_execution_weight_hint_sum: `{commodity_paper_execution_preview.get('next_execution_weight_hint_sum', 0.0)}`",
                f"- preview_stack: `{commodity_paper_execution_preview.get('preview_stack_brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Commodity Paper Execution Artifact",
        ]
    )
    if not commodity_paper_execution_artifact:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- route_status: `{commodity_paper_execution_artifact.get('route_status', '')}`",
                f"- ticket_book_status: `{commodity_paper_execution_artifact.get('ticket_book_status', '')}`",
                f"- execution_preview_status: `{commodity_paper_execution_artifact.get('execution_preview_status', '')}`",
                f"- execution_artifact_status: `{commodity_paper_execution_artifact.get('execution_artifact_status', '')}`",
                f"- execution_mode: `{commodity_paper_execution_artifact.get('execution_mode', '')}`",
                f"- execution_batch: `{commodity_paper_execution_artifact.get('execution_batch', '')}`",
                f"- execution_symbols: `{', '.join(commodity_paper_execution_artifact.get('execution_symbols', []))}`",
                f"- execution_regime_gate: `{commodity_paper_execution_artifact.get('execution_regime_gate', '')}`",
                f"- execution_weight_hint_sum: `{commodity_paper_execution_artifact.get('execution_weight_hint_sum', 0.0)}`",
                f"- execution_item_count: `{commodity_paper_execution_artifact.get('execution_item_count', 0)}`",
                f"- actionable_execution_item_count: `{commodity_paper_execution_artifact.get('actionable_execution_item_count', 0)}`",
                f"- execution_stack: `{commodity_paper_execution_artifact.get('execution_stack_brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Commodity Paper Execution Queue",
        ]
    )
    if not commodity_paper_execution_queue:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- route_status: `{commodity_paper_execution_queue.get('route_status', '')}`",
                f"- ticket_book_status: `{commodity_paper_execution_queue.get('ticket_book_status', '')}`",
                f"- execution_preview_status: `{commodity_paper_execution_queue.get('execution_preview_status', '')}`",
                f"- execution_artifact_status: `{commodity_paper_execution_queue.get('execution_artifact_status', '')}`",
                f"- execution_queue_status: `{commodity_paper_execution_queue.get('execution_queue_status', '')}`",
                f"- execution_mode: `{commodity_paper_execution_queue.get('execution_mode', '')}`",
                f"- execution_batch: `{commodity_paper_execution_queue.get('execution_batch', '')}`",
                f"- execution_symbols: `{', '.join(commodity_paper_execution_queue.get('execution_symbols', []))}`",
                f"- execution_regime_gate: `{commodity_paper_execution_queue.get('execution_regime_gate', '')}`",
                f"- queue_depth: `{commodity_paper_execution_queue.get('queue_depth', 0)}`",
                f"- actionable_queue_depth: `{commodity_paper_execution_queue.get('actionable_queue_depth', 0)}`",
                f"- next_execution_id: `{commodity_paper_execution_queue.get('next_execution_id', '')}`",
                f"- next_execution_symbol: `{commodity_paper_execution_queue.get('next_execution_symbol', '')}`",
                f"- execution_queue_stack: `{commodity_paper_execution_queue.get('queue_stack_brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Commodity Paper Execution Review",
        ]
    )
    if not commodity_paper_execution_review:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- route_status: `{commodity_paper_execution_review.get('route_status', '')}`",
                f"- ticket_book_status: `{commodity_paper_execution_review.get('ticket_book_status', '')}`",
                f"- execution_preview_status: `{commodity_paper_execution_review.get('execution_preview_status', '')}`",
                f"- execution_artifact_status: `{commodity_paper_execution_review.get('execution_artifact_status', '')}`",
                f"- execution_queue_status: `{commodity_paper_execution_review.get('execution_queue_status', '')}`",
                f"- execution_review_status: `{commodity_paper_execution_review.get('execution_review_status', '')}`",
                f"- execution_mode: `{commodity_paper_execution_review.get('execution_mode', '')}`",
                f"- execution_batch: `{commodity_paper_execution_review.get('execution_batch', '')}`",
                f"- execution_symbols: `{', '.join(commodity_paper_execution_review.get('execution_symbols', []))}`",
                f"- execution_regime_gate: `{commodity_paper_execution_review.get('execution_regime_gate', '')}`",
                f"- review_item_count: `{commodity_paper_execution_review.get('review_item_count', 0)}`",
                f"- actionable_review_item_count: `{commodity_paper_execution_review.get('actionable_review_item_count', 0)}`",
                f"- next_review_execution_id: `{commodity_paper_execution_review.get('next_review_execution_id', '')}`",
                f"- next_review_execution_symbol: `{commodity_paper_execution_review.get('next_review_execution_symbol', '')}`",
                f"- execution_review_stack: `{commodity_paper_execution_review.get('review_stack_brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Commodity Paper Execution Retro",
        ]
    )
    if not commodity_paper_execution_retro:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- route_status: `{commodity_paper_execution_retro.get('route_status', '')}`",
                f"- ticket_book_status: `{commodity_paper_execution_retro.get('ticket_book_status', '')}`",
                f"- execution_preview_status: `{commodity_paper_execution_retro.get('execution_preview_status', '')}`",
                f"- execution_artifact_status: `{commodity_paper_execution_retro.get('execution_artifact_status', '')}`",
                f"- execution_queue_status: `{commodity_paper_execution_retro.get('execution_queue_status', '')}`",
                f"- execution_review_status: `{commodity_paper_execution_retro.get('execution_review_status', '')}`",
                f"- execution_retro_status: `{commodity_paper_execution_retro.get('execution_retro_status', '')}`",
                f"- execution_mode: `{commodity_paper_execution_retro.get('execution_mode', '')}`",
                f"- execution_batch: `{commodity_paper_execution_retro.get('execution_batch', '')}`",
                f"- execution_symbols: `{', '.join(commodity_paper_execution_retro.get('execution_symbols', []))}`",
                f"- execution_regime_gate: `{commodity_paper_execution_retro.get('execution_regime_gate', '')}`",
                f"- retro_item_count: `{commodity_paper_execution_retro.get('retro_item_count', 0)}`",
                f"- actionable_retro_item_count: `{commodity_paper_execution_retro.get('actionable_retro_item_count', 0)}`",
                f"- next_retro_execution_id: `{commodity_paper_execution_retro.get('next_retro_execution_id', '')}`",
                f"- next_retro_execution_symbol: `{commodity_paper_execution_retro.get('next_retro_execution_symbol', '')}`",
                f"- execution_retro_stack: `{commodity_paper_execution_retro.get('retro_stack_brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Crypto Route Brief",
        ]
    )
    if not crypto_route_brief:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- operator_status: `{crypto_route_brief.get('operator_status', '')}`",
                f"- route_stack: `{crypto_route_brief.get('route_stack_brief', '')}`",
                f"- next_focus_symbol: `{crypto_route_brief.get('next_focus_symbol', '')}`",
                f"- next_focus_action: `{crypto_route_brief.get('next_focus_action', '')}`",
                f"- focus_window_gate: `{crypto_route_brief.get('focus_window_gate', '')}`",
                f"- focus_window_verdict: `{crypto_route_brief.get('focus_window_verdict', '')}`",
                f"- next_retest_action: `{crypto_route_brief.get('next_retest_action', '')}`",
                f"- next_retest_reason: `{crypto_route_brief.get('next_retest_reason', '')}`",
                f"- brief: `{crypto_route_brief.get('brief_text', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Crypto Route Operator Brief",
        ]
    )
    if not crypto_route_operator_brief:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- operator_status: `{crypto_route_operator_brief.get('operator_status', '')}`",
                f"- route_stack: `{crypto_route_operator_brief.get('route_stack_brief', '')}`",
                f"- next_focus_symbol: `{crypto_route_operator_brief.get('next_focus_symbol', '')}`",
                f"- next_focus_action: `{crypto_route_operator_brief.get('next_focus_action', '')}`",
                f"- focus_window_gate: `{crypto_route_operator_brief.get('focus_window_gate', '')}`",
                f"- focus_window_verdict: `{crypto_route_operator_brief.get('focus_window_verdict', '')}`",
                f"- focus_window_floor: `{crypto_route_operator_brief.get('focus_window_floor', '')}`",
                f"- price_state_window_floor: `{crypto_route_operator_brief.get('price_state_window_floor', '')}`",
                f"- comparative_window_takeaway: `{crypto_route_operator_brief.get('comparative_window_takeaway', '')}`",
                f"- xlong_flow_window_floor: `{crypto_route_operator_brief.get('xlong_flow_window_floor', '')}`",
                f"- xlong_comparative_window_takeaway: `{crypto_route_operator_brief.get('xlong_comparative_window_takeaway', '')}`",
                f"- next_retest_action: `{crypto_route_operator_brief.get('next_retest_action', '')}`",
                f"- next_retest_reason: `{crypto_route_operator_brief.get('next_retest_reason', '')}`",
                f"- brief: `{crypto_route_operator_brief.get('operator_text', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## BNB Flow Focus",
        ]
    )
    if not bnb_flow_focus:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- symbol: `{bnb_flow_focus.get('symbol', '')}`",
                f"- operator_status: `{bnb_flow_focus.get('operator_status', '')}`",
                f"- promotion_gate: `{bnb_flow_focus.get('promotion_gate', '')}`",
                f"- promotion_gate_reason: `{bnb_flow_focus.get('promotion_gate_reason', '')}`",
                f"- action: `{bnb_flow_focus.get('action', '')}`",
                f"- action_reason: `{bnb_flow_focus.get('action_reason', '')}`",
                f"- flow_window_verdict: `{bnb_flow_focus.get('flow_window_verdict', '')}`",
                f"- price_state_window_verdict: `{bnb_flow_focus.get('price_state_window_verdict', '')}`",
                f"- next_retest_action: `{bnb_flow_focus.get('next_retest_action', '')}`",
                f"- next_retest_reason: `{bnb_flow_focus.get('next_retest_reason', '')}`",
                f"- brief: `{bnb_flow_focus.get('brief', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Crypto Symbol Routes",
        ]
    )
    if not crypto_symbol_route_handoff:
        lines.append("- `none`")
    else:
        lines.extend(
            [
                f"- operator_status: `{crypto_symbol_route_handoff.get('operator_status', '')}`",
                f"- route_stack: `{crypto_symbol_route_handoff.get('route_stack_brief', '')}`",
                f"- deploy_now: `{', '.join(crypto_symbol_route_handoff.get('deploy_now_symbols', []))}`",
                f"- watch_priority: `{', '.join(crypto_symbol_route_handoff.get('watch_priority_symbols', []))}`",
                f"- watch_only: `{', '.join(crypto_symbol_route_handoff.get('watch_only_symbols', []))}`",
                f"- next_focus_symbol: `{crypto_symbol_route_handoff.get('next_focus_symbol', '')}`",
                f"- next_focus_action: `{crypto_symbol_route_handoff.get('next_focus_action', '')}`",
                f"- focus_window_gate: `{crypto_symbol_route_handoff.get('focus_window_gate', '')}`",
                f"- focus_window_verdict: `{crypto_symbol_route_handoff.get('focus_window_verdict', '')}`",
                f"- takeaway: `{crypto_symbol_route_handoff.get('overall_takeaway', '')}`",
            ]
        )
        for row in crypto_symbol_route_handoff.get("routes", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('symbol', '')}` lane=`{row.get('lane', '')}` deployment=`{row.get('deployment', '')}` action=`{row.get('action', '')}` status=`{row.get('status_label', '')}`"
            )
            lines.append(f"  reason=`{row.get('reason', '')}`")
    lines.extend(
        [
            "",
            "## Ranked Batches",
        ]
    )
    for item in summary.get("ranked_batches", []):
        lines.append(
            f"- `{item.get('batch', '')}`: label=`{item.get('status_label', '')}` score=`{float(item.get('research_score', 0.0)):.4f}` accepted=`{item.get('accepted_count', 0)}` trades=`{item.get('research_trades', 0)}`"
        )
        lines.append(f"  takeaway=`{item.get('takeaway', '')}`")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch Fenlie research on a hot crypto + commodities universe.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--universe-file", default="")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--hours-budget", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-trials-per-mode", type=int, default=3)
    parser.add_argument("--review-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-strategy-lab", action="store_true")
    parser.add_argument("--strategy-lab-candidate-count", type=int, default=4)
    parser.add_argument("--batch", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-timeout-seconds", type=float, default=DEFAULT_BATCH_TIMEOUT_SECONDS)
    parser.add_argument("--artifact-ttl-hours", type=float, default=DEFAULT_ARTIFACT_TTL_HOURS)
    parser.add_argument("--keep-files", type=int, default=DEFAULT_KEEP_FILES)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(str(args.output_root)).expanduser().resolve()
    review_dir = Path(str(args.review_dir)).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    universe_path = Path(str(args.universe_file)).expanduser().resolve() if str(args.universe_file).strip() else find_latest_universe_file(review_dir)
    if universe_path is None or not universe_path.exists():
        raise FileNotFoundError("hot_research_universe_artifact_missing")
    universe_payload = load_json(universe_path)
    selected_batches = select_batches(universe_payload, args.batch)
    start = parse_date_text(args.start)
    end = parse_date_text(args.end)
    now_dt = parse_now(str(args.now))
    batch_results = run_batches(
        universe_payload=universe_payload,
        selected_batches=selected_batches,
        output_root=output_root,
        start=start,
        end=end,
        hours_budget=max(0.0001, float(args.hours_budget)),
        workers=max(1, int(args.workers)),
        max_trials_per_mode=max(1, int(args.max_trials_per_mode)),
        review_days=max(0, int(args.review_days)),
        run_strategy_lab_enabled=bool(args.run_strategy_lab),
        strategy_lab_candidate_count=max(1, int(args.strategy_lab_candidate_count)),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        batch_timeout_seconds=max(1.0, float(args.batch_timeout_seconds)),
    )
    run_status = overall_run_status(batch_results=batch_results, dry_run=bool(args.dry_run))
    failed_batch_count = sum(
        1 for row in batch_results if isinstance(row, dict) and str(row.get("status") or "").strip() == "failed"
    )
    timed_out_batch_count = sum(
        1 for row in batch_results if isinstance(row, dict) and bool(row.get("timed_out", False))
    )
    stamp = now_dt.strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_hot_universe_research.json"
    checksum_path = review_dir / f"{stamp}_hot_universe_research_checksum.json"
    report_path = review_dir / f"{stamp}_hot_universe_research.md"
    generated_at = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict[str, Any] = {
        "action": "run_hot_universe_research",
        "ok": run_status in {"ok", "dry_run"},
        "status": run_status,
        "generated_at_utc": generated_at,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "universe_file": str(universe_path),
        "source_tier": str(universe_payload.get("source_tier", "")),
        "batch_results": batch_results,
        "batch_timeout_seconds": max(1.0, float(args.batch_timeout_seconds)),
        "failed_batch_count": failed_batch_count,
        "timed_out_batch_count": timed_out_batch_count,
        "run_strategy_lab_enabled": bool(args.run_strategy_lab),
        "batch_summary": summarize_batch_results(batch_results),
        "artifact": str(artifact_path),
        "checksum": str(checksum_path),
        "report": str(report_path),
        "artifact_label": "hot-universe-research",
        "artifact_status_label": (
            "hot-universe-research-dry-run"
            if bool(args.dry_run)
            else ("hot-universe-research-partial-failure" if run_status == "partial_failure" else "hot-universe-research-ok")
        ),
    }
    payload["commodity_paper_ticket_lane"] = load_commodity_paper_ticket_lane(review_dir, now_dt)
    payload["commodity_paper_ticket_book"] = load_commodity_paper_ticket_book(review_dir, now_dt)
    payload["commodity_paper_execution_preview"] = load_commodity_paper_execution_preview(review_dir, now_dt)
    payload["commodity_paper_execution_artifact"] = load_commodity_paper_execution_artifact(review_dir, now_dt)
    payload["commodity_paper_execution_queue"] = load_commodity_paper_execution_queue(review_dir, now_dt)
    payload["commodity_paper_execution_review"] = load_commodity_paper_execution_review(review_dir, now_dt)
    payload["commodity_paper_execution_retro"] = load_commodity_paper_execution_retro(review_dir, now_dt)
    payload["crypto_symbol_route_handoff"] = load_crypto_symbol_route_handoff(review_dir, now_dt)
    payload["crypto_route_brief"] = load_crypto_route_brief(review_dir, now_dt)
    payload["crypto_route_operator_brief"] = load_crypto_route_operator_brief(review_dir, now_dt)
    payload["bnb_flow_focus"] = load_bnb_flow_focus(review_dir, now_dt)
    payload["batch_playbook"] = derive_batch_playbook(payload["batch_summary"])
    payload["symbol_attribution"] = derive_symbol_attribution(batch_results)
    payload["regime_attribution"] = derive_regime_attribution(batch_results)
    payload["regime_playbook"] = derive_regime_playbook(batch_results, payload["regime_attribution"])
    payload["leader_profiles"] = derive_leader_profiles(payload["symbol_attribution"], payload["regime_attribution"])
    payload["batch_relationships"] = derive_batch_relationships(batch_results, payload["batch_playbook"], payload["symbol_attribution"])
    payload["research_action_ladder"] = derive_research_action_ladder(
        payload["batch_summary"],
        payload["batch_playbook"],
        payload["regime_playbook"],
        payload["batch_relationships"],
        payload["leader_profiles"],
    )
    payload["microstructure_playbook"] = derive_microstructure_playbook(
        universe_payload,
        batch_results,
        payload["research_action_ladder"],
    )
    payload["crypto_cvd_queue_profile"] = derive_crypto_cvd_queue_profile(
        batch_results,
        payload["research_action_ladder"],
        payload["regime_playbook"],
        payload["leader_profiles"],
        payload["microstructure_playbook"],
    )
    write_markdown(
        report_path,
        build_report(
            universe_payload=universe_payload,
            batch_results=batch_results,
            dry_run=bool(args.dry_run),
            start=start,
            end=end,
            commodity_paper_ticket_lane=payload.get("commodity_paper_ticket_lane"),
            commodity_paper_ticket_book=payload.get("commodity_paper_ticket_book"),
            commodity_paper_execution_preview=payload.get("commodity_paper_execution_preview"),
            commodity_paper_execution_artifact=payload.get("commodity_paper_execution_artifact"),
            commodity_paper_execution_queue=payload.get("commodity_paper_execution_queue"),
            commodity_paper_execution_review=payload.get("commodity_paper_execution_review"),
            commodity_paper_execution_retro=payload.get("commodity_paper_execution_retro"),
            crypto_symbol_route_handoff=payload.get("crypto_symbol_route_handoff"),
            crypto_route_brief=payload.get("crypto_route_brief"),
            crypto_route_operator_brief=payload.get("crypto_route_operator_brief"),
            bnb_flow_focus=payload.get("bnb_flow_focus"),
        ),
    )
    payload["retention"] = {
        "evicted_files": evict_old_artifacts(
            review_dir=review_dir,
            protected={artifact_path.name, checksum_path.name, report_path.name},
            now_dt=now_dt,
            ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
            keep_files=max(3, int(args.keep_files)),
        )
    }
    write_json(artifact_path, payload)
    write_sha256(artifact_path, checksum_path, ttl_hours=max(1.0, float(args.artifact_ttl_hours)), generated_at=generated_at)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
