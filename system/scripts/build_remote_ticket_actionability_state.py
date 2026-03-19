#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import yaml


DEFAULT_TICKET_FRESHNESS_SECONDS = 900


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"
DEFAULT_SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text_value = str(raw or "").strip()
    if not text_value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_utc(raw: Any) -> dt.datetime | None:
    text_value = str(raw or "").strip()
    if not text_value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dedupe_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_unique(parts: list[Any], *, sep: str = " | ") -> str:
    return sep.join(dedupe_text(parts))


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    try:
        return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (0 if is_future else 1, artifact_stamp(path), mtime, path.name)


def find_latest(
    review_dir: Path,
    pattern: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    return files[0] if files else None


def payload_scope_symbol(payload: dict[str, Any]) -> str:
    for key in ("route_symbol", "symbol", "focus_symbol"):
        value = text(payload.get(key)).upper()
        if value:
            return value
    return ""


def payload_matches_symbol(payload: dict[str, Any], symbol: str) -> bool:
    normalized = text(symbol).upper()
    if not normalized:
        return True
    scoped = payload_scope_symbol(payload)
    return not scoped or scoped == normalized


def find_latest_scoped(
    review_dir: Path,
    pattern: str,
    symbol: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    for path in files:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        if payload_matches_symbol(payload, symbol):
            return path
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_remote_ticket_actionability_state.json",
        "*_remote_ticket_actionability_state.md",
        "*_remote_ticket_actionability_state_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
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
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def load_shortline_policy(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    shortline = payload.get("shortline", {}) if isinstance(payload, dict) else {}
    return {
        "supported_symbols": dedupe_text(as_list(shortline.get("supported_symbols")))
        or list(DEFAULT_SUPPORTED_SYMBOLS),
        "no_trade_rule": text(shortline.get("no_trade_rule")) or "no_sweep_no_mss_no_cvd_no_trade",
    }


def select_ticket_surface_path(
    *,
    review_dir: Path,
    route_symbol: str,
    supported_symbols: list[str],
    reference_now: dt.datetime,
) -> Path | None:
    candidates = sorted(
        review_dir.glob("*_signal_to_order_tickets.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    if not candidates:
        return None

    normalized_route = text(route_symbol).upper()
    supported = {text(symbol).upper() for symbol in supported_symbols if text(symbol)}
    latest_candidate = candidates[0]
    first_supported_surface: Path | None = None

    for path in candidates:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        rows = [as_dict(row) for row in as_list(payload.get("tickets")) if isinstance(row, dict)]
        surface_symbols = {text(row.get("symbol")).upper() for row in rows if text(row.get("symbol"))}
        if normalized_route and normalized_route in surface_symbols:
            return path
        if first_supported_surface is None and surface_symbols.intersection(supported):
            first_supported_surface = path

    return first_supported_surface or latest_candidate


def route_symbol(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()


def resolve_symbol(
    *,
    explicit_symbol: str,
    intent_payload: dict[str, Any],
    operator_payload: dict[str, Any],
) -> str:
    return text(explicit_symbol).upper() or route_symbol(intent_payload, operator_payload)


def route_action(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )


def load_ticket_surface(
    *,
    tickets_path: Path | None,
    route_symbol: str,
    reference_now: dt.datetime,
    freshness_seconds: int,
    supported_symbols: list[str],
) -> dict[str, Any]:
    if tickets_path is None:
        return {
            "artifact": "",
            "artifact_status": "missing_artifact",
            "artifact_age_seconds": None,
            "route_row_status": "row_missing",
            "route_row_allowed": False,
            "route_row_reasons": [],
            "route_row": {},
            "ticket_match_brief": "ticket_artifact_missing",
            "surface_symbols": [],
            "surface_coverage_status": "missing_surface",
            "surface_contains_route_symbol": False,
        }

    payload = load_json_mapping(tickets_path)
    signal_source = as_dict(payload.get("signal_source"))
    generated_at = parse_utc(payload.get("generated_at_utc"))
    if generated_at is None:
        generated_at = dt.datetime.fromtimestamp(tickets_path.stat().st_mtime, tz=dt.timezone.utc)
    age_seconds = max(0.0, (reference_now - generated_at).total_seconds())
    artifact_status = (
        "stale_artifact"
        if age_seconds > float(max(1, int(freshness_seconds)))
        else "fresh_artifact"
    )
    rows = [as_dict(row) for row in as_list(payload.get("tickets")) if isinstance(row, dict)]
    surface_symbols = [text(row.get("symbol")).upper() for row in rows if text(row.get("symbol"))]
    route_row = {}
    for row in rows:
        if text(row.get("symbol")).upper() == route_symbol.upper():
            route_row = row
            break
    route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))
    route_row_allowed = bool(route_row.get("allowed", False))
    if route_row:
        route_row_status = "row_ready" if route_row_allowed else "row_blocked"
        ticket_match_brief = (
            f"{artifact_status}:ticket_row_ready:{route_symbol}"
            if route_row_allowed
            else join_unique(
                [
                    f"{artifact_status}:ticket_row_blocked:{route_symbol}",
                    ",".join(route_row_reasons),
                ]
            )
        )
    else:
        route_row_status = "row_missing"
        ticket_match_brief = f"{artifact_status}:ticket_row_missing:{route_symbol}"

    supported = {text(symbol).upper() for symbol in supported_symbols}
    supported_surface_symbols = [symbol for symbol in surface_symbols if symbol in supported]
    if not surface_symbols:
        surface_coverage_status = "missing_surface"
    elif route_symbol.upper() in surface_symbols:
        surface_coverage_status = "route_symbol_present"
    elif supported_surface_symbols:
        surface_coverage_status = "crypto_present_missing_route"
    else:
        surface_coverage_status = "commodity_only"
    return {
        "artifact": str(tickets_path),
        "artifact_status": artifact_status,
        "artifact_age_seconds": age_seconds,
        "signal_source_path": text(signal_source.get("path")),
        "signal_source_kind": text(signal_source.get("kind")),
        "signal_source_selection_reason": text(signal_source.get("selection_reason")),
        "signal_source_artifact_date": text(signal_source.get("artifact_date")),
        "route_row_status": route_row_status,
        "route_row_allowed": route_row_allowed,
        "route_row_reasons": route_row_reasons,
        "route_row": route_row,
        "ticket_match_brief": ticket_match_brief,
        "surface_symbols": surface_symbols,
        "surface_coverage_status": surface_coverage_status,
        "surface_contains_route_symbol": route_symbol.upper() in surface_symbols,
    }


def parse_iso_date(raw: Any) -> dt.date | None:
    text_value = text(raw)
    if not text_value:
        return None
    try:
        return dt.date.fromisoformat(text_value[:10])
    except ValueError:
        return None


def find_gate_row(gate_payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(gate_payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def find_batch_row(queue_payload: dict[str, Any], batch_name: str, symbol: str) -> dict[str, Any]:
    for key in ("runtime_queue", "batch_runtime_profiles"):
        for row in as_list(queue_payload.get(key)):
            item = as_dict(row)
            if batch_name and text(item.get("batch")) == batch_name:
                return item
            if symbol.upper() in {text(x).upper() for x in as_list(item.get("eligible_symbols"))}:
                return item
    return {}


def find_matching_symbol(batch_row: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(batch_row.get("matching_symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def build_payload(
    *,
    resolved_symbol: str,
    intent_queue_path: Path | None,
    intent_queue_payload: dict[str, Any],
    tickets_path: Path | None,
    signal_source_freshness_path: Path | None,
    signal_source_freshness_payload: dict[str, Any],
    signal_source_refresh_readiness_path: Path | None,
    signal_source_refresh_readiness_payload: dict[str, Any],
    crypto_route_operator_path: Path,
    crypto_route_operator_payload: dict[str, Any],
    shortline_gate_path: Path,
    shortline_gate_payload: dict[str, Any],
    cvd_queue_path: Path,
    cvd_queue_payload: dict[str, Any],
    backtest_slice_path: Path | None,
    backtest_slice_payload: dict[str, Any],
    cross_section_backtest_path: Path | None,
    cross_section_backtest_payload: dict[str, Any],
    material_change_trigger_path: Path | None,
    material_change_trigger_payload: dict[str, Any],
    pattern_router_path: Path | None,
    pattern_router_payload: dict[str, Any],
    profile_location_watch_path: Path | None,
    profile_location_watch_payload: dict[str, Any],
    mss_watch_path: Path | None,
    mss_watch_payload: dict[str, Any],
    cvd_confirmation_watch_path: Path | None,
    cvd_confirmation_watch_payload: dict[str, Any],
    liquidity_sweep_watch_path: Path | None,
    liquidity_sweep_watch_payload: dict[str, Any],
    price_reference_watch_path: Path | None,
    price_reference_watch_payload: dict[str, Any],
    execution_quality_watch_path: Path | None,
    execution_quality_watch_payload: dict[str, Any],
    slippage_snapshot_path: Path | None,
    slippage_snapshot_payload: dict[str, Any],
    fill_capacity_watch_path: Path | None,
    fill_capacity_watch_payload: dict[str, Any],
    signal_quality_watch_path: Path | None,
    signal_quality_watch_payload: dict[str, Any],
    sizing_watch_path: Path | None,
    sizing_watch_payload: dict[str, Any],
    gate_stack_progress_path: Path | None,
    gate_stack_progress_payload: dict[str, Any],
    ticket_constraint_diagnosis_path: Path | None,
    ticket_constraint_diagnosis_payload: dict[str, Any],
    setup_transition_watch_path: Path | None,
    setup_transition_watch_payload: dict[str, Any],
    policy: dict[str, Any],
    reference_now: dt.datetime,
    freshness_seconds: int,
) -> dict[str, Any]:
    route_focus_symbol = route_symbol(intent_queue_payload, crypto_route_operator_payload)
    symbol = text(resolved_symbol).upper() or route_focus_symbol
    action = route_action(intent_queue_payload, crypto_route_operator_payload)
    focus_review_matches_symbol = not route_focus_symbol or route_focus_symbol == symbol
    focus_review_status = (
        text(crypto_route_operator_payload.get("focus_review_status"))
        if focus_review_matches_symbol
        else ""
    )
    focus_review_brief = (
        text(crypto_route_operator_payload.get("focus_review_brief"))
        if focus_review_matches_symbol
        else ""
    )
    focus_review_blocker_detail = (
        text(crypto_route_operator_payload.get("focus_review_blocker_detail"))
        if focus_review_matches_symbol
        else ""
    )
    remote_market = text(intent_queue_payload.get("remote_market")) or "portfolio_margin_um"
    ticket_surface = load_ticket_surface(
        tickets_path=tickets_path,
        route_symbol=symbol,
        reference_now=reference_now,
        freshness_seconds=freshness_seconds,
        supported_symbols=list(policy.get("supported_symbols", [])),
    )
    gate_row = find_gate_row(shortline_gate_payload, symbol)
    queue_batch_name = text(cvd_queue_payload.get("next_focus_batch"))
    queue_batch = find_batch_row(cvd_queue_payload, queue_batch_name, symbol)
    batch_match = find_matching_symbol(queue_batch, symbol)
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    effective_missing_gates = dedupe_text(as_list(gate_row.get("effective_missing_gates")))
    pattern_family_hint = text(gate_row.get("pattern_family_hint"))
    pattern_stage_hint = text(gate_row.get("pattern_stage_hint"))
    pattern_hint_brief = text(gate_row.get("pattern_hint_brief"))
    execution_state = text(gate_row.get("execution_state")) or text(batch_match.get("classification"))
    queue_focus_action = text(cvd_queue_payload.get("next_focus_action"))
    backtest_slice_brief = text(backtest_slice_payload.get("slice_brief"))
    cross_section_backtest_brief = text(cross_section_backtest_payload.get("backtest_brief"))
    cross_section_backtest_status = text(cross_section_backtest_payload.get("backtest_status"))
    cross_section_backtest_decision = text(cross_section_backtest_payload.get("research_decision"))
    cross_section_selected_edge_status = text(
        cross_section_backtest_payload.get("selected_edge_status")
    )
    material_change_trigger_brief = text(material_change_trigger_payload.get("trigger_brief"))
    material_change_trigger_status = text(material_change_trigger_payload.get("trigger_status"))
    material_change_trigger_decision = text(material_change_trigger_payload.get("trigger_decision"))
    material_change_trigger_rerun_recommended = bool(
        material_change_trigger_payload.get("rerun_recommended", False)
    )
    pattern_router_brief = text(pattern_router_payload.get("pattern_brief"))
    pattern_router_status = text(pattern_router_payload.get("pattern_status"))
    pattern_router_decision = text(pattern_router_payload.get("pattern_decision"))
    pattern_router_family = text(pattern_router_payload.get("pattern_family"))
    pattern_router_stage = text(pattern_router_payload.get("pattern_stage"))
    gate_stack_progress_brief = text(gate_stack_progress_payload.get("gate_stack_brief"))
    gate_stack_progress_status = text(gate_stack_progress_payload.get("gate_stack_status"))
    gate_stack_progress_decision = text(gate_stack_progress_payload.get("gate_stack_decision"))
    gate_stack_progress_primary_stage = text(gate_stack_progress_payload.get("primary_stage"))
    profile_location_watch_brief = text(profile_location_watch_payload.get("watch_brief"))
    profile_location_watch_status = text(profile_location_watch_payload.get("watch_status"))
    profile_location_watch_decision = text(profile_location_watch_payload.get("watch_decision"))
    mss_watch_brief = text(mss_watch_payload.get("watch_brief"))
    mss_watch_status = text(mss_watch_payload.get("watch_status"))
    mss_watch_decision = text(mss_watch_payload.get("watch_decision"))
    cvd_confirmation_watch_brief = text(cvd_confirmation_watch_payload.get("watch_brief"))
    cvd_confirmation_watch_status = text(cvd_confirmation_watch_payload.get("watch_status"))
    cvd_confirmation_watch_decision = text(
        cvd_confirmation_watch_payload.get("watch_decision")
    )
    liquidity_sweep_watch_brief = text(liquidity_sweep_watch_payload.get("watch_brief"))
    liquidity_sweep_watch_status = text(liquidity_sweep_watch_payload.get("watch_status"))
    liquidity_sweep_watch_decision = text(liquidity_sweep_watch_payload.get("watch_decision"))
    price_reference_watch_brief = text(price_reference_watch_payload.get("watch_brief"))
    price_reference_watch_status = text(price_reference_watch_payload.get("watch_status"))
    price_reference_watch_decision = text(price_reference_watch_payload.get("watch_decision"))
    execution_quality_watch_brief = text(execution_quality_watch_payload.get("watch_brief"))
    execution_quality_watch_status = text(execution_quality_watch_payload.get("watch_status"))
    execution_quality_watch_decision = text(execution_quality_watch_payload.get("watch_decision"))
    slippage_snapshot_brief = text(slippage_snapshot_payload.get("snapshot_brief"))
    slippage_snapshot_status = text(slippage_snapshot_payload.get("snapshot_status"))
    slippage_snapshot_decision = text(slippage_snapshot_payload.get("snapshot_decision"))
    fill_capacity_watch_brief = text(fill_capacity_watch_payload.get("watch_brief"))
    fill_capacity_watch_status = text(fill_capacity_watch_payload.get("watch_status"))
    fill_capacity_watch_decision = text(fill_capacity_watch_payload.get("watch_decision"))
    signal_quality_watch_brief = text(signal_quality_watch_payload.get("watch_brief"))
    signal_quality_watch_status = text(signal_quality_watch_payload.get("watch_status"))
    signal_quality_watch_decision = text(signal_quality_watch_payload.get("watch_decision"))
    sizing_watch_brief = text(sizing_watch_payload.get("watch_brief"))
    sizing_watch_status = text(sizing_watch_payload.get("watch_status"))
    sizing_watch_decision = text(sizing_watch_payload.get("watch_decision"))
    signal_source_freshness_brief = text(signal_source_freshness_payload.get("freshness_brief"))
    signal_source_freshness_status = text(signal_source_freshness_payload.get("freshness_status"))
    signal_source_freshness_decision = text(signal_source_freshness_payload.get("freshness_decision"))
    signal_source_freshness_refresh_recommended = bool(
        signal_source_freshness_payload.get("refresh_recommended", False)
    )
    signal_source_refresh_readiness_brief = text(
        signal_source_refresh_readiness_payload.get("readiness_brief")
    )
    signal_source_refresh_readiness_status = text(
        signal_source_refresh_readiness_payload.get("readiness_status")
    )
    signal_source_refresh_readiness_decision = text(
        signal_source_refresh_readiness_payload.get("readiness_decision")
    )
    signal_source_refresh_needed = bool(
        signal_source_refresh_readiness_payload.get("refresh_needed", False)
    )
    signal_source_refresh_target_artifact = (
        "crypto_signal_source_refresh_readiness"
        if signal_source_refresh_readiness_path is not None
        and signal_source_refresh_readiness_path.exists()
        else "crypto_signal_source_freshness"
    )
    ticket_constraint_diagnosis_brief = text(
        ticket_constraint_diagnosis_payload.get("diagnosis_brief")
        or ticket_constraint_diagnosis_payload.get("ticket_actionability_brief")
    )
    ticket_constraint_diagnosis_status = text(
        ticket_constraint_diagnosis_payload.get("diagnosis_status")
        or ticket_constraint_diagnosis_payload.get("ticket_actionability_status")
    )
    ticket_constraint_diagnosis_decision = text(
        ticket_constraint_diagnosis_payload.get("diagnosis_decision")
        or ticket_constraint_diagnosis_payload.get("ticket_actionability_decision")
    )
    ticket_constraint_primary_code = text(
        ticket_constraint_diagnosis_payload.get("primary_constraint_code")
    )
    setup_transition_brief = text(setup_transition_watch_payload.get("transition_brief"))
    setup_transition_status = text(setup_transition_watch_payload.get("transition_status"))
    setup_transition_decision = text(setup_transition_watch_payload.get("transition_decision"))
    setup_transition_primary_missing_gate = text(
        setup_transition_watch_payload.get("primary_missing_gate")
    )
    ticket_signal_source_path = text(ticket_surface.get("signal_source_path"))
    ticket_signal_source_kind = text(ticket_surface.get("signal_source_kind"))
    ticket_signal_source_selection_reason = text(
        ticket_surface.get("signal_source_selection_reason")
    )
    ticket_signal_source_artifact_date = text(ticket_surface.get("signal_source_artifact_date"))
    ticket_signal_source_age_days: int | None = None
    ticket_signal_source_date = parse_iso_date(ticket_signal_source_artifact_date)
    if ticket_signal_source_date is not None:
        ticket_signal_source_age_days = max(0, (reference_now.date() - ticket_signal_source_date).days)
    route_row_reasons = dedupe_text(as_list(ticket_surface.get("route_row_reasons")))

    blocker_target_artifact = "remote_ticket_actionability_state"
    if ticket_surface.get("route_row_status") == "row_ready" and ticket_surface.get("artifact_status") != "stale_artifact":
        ticket_actionability_status = "ticket_actionable_ready"
        ticket_actionability_decision = "route_ticket_ready_for_guardian_review"
        actionable_ready = True
        blocker_title = "Ticket actionability clear"
        next_action = "review_guarded_canary_promotion"
        next_action_target_artifact = "remote_guarded_canary_promotion_gate"
        done_when = "fresh actionable ticket row remains ready and guardian review clears for promotion"
    elif (
        ticket_surface.get("surface_coverage_status") == "commodity_only"
        and symbol
        and execution_state == "Bias_Only"
    ):
        actionable_ready = False
        blocker_title = "Resolve crypto ticket actionability before guarded canary review"
        done_when = f"{symbol} reaches Setup_Ready and a fresh crypto ticket row exists for {symbol}"
        if cross_section_backtest_status == "watch_only_cross_section_positive":
            ticket_actionability_status = "crypto_shortline_backtest_positive_not_ticketed"
            ticket_actionability_decision = "continue_shadow_learning_wait_for_setup_ready"
            next_action = "refresh_shortline_execution_gate_until_setup_ready"
            next_action_target_artifact = "crypto_shortline_execution_gate"
        elif cross_section_backtest_status in {
            "watch_only_cross_section_mixed",
            "watch_only_cross_section_low_sample",
        }:
            ticket_actionability_status = "crypto_shortline_backtest_mixed_not_ticketed"
            ticket_actionability_decision = "collect_more_orderflow_and_wait_for_setup_ready"
            next_action = "rerun_shortline_cross_section_backtest_on_material_change"
            next_action_target_artifact = "crypto_shortline_cross_section_backtest"
        elif cross_section_backtest_status == "watch_only_cross_section_no_edge":
            if material_change_trigger_rerun_recommended:
                ticket_actionability_status = (
                    "crypto_shortline_material_change_pending_not_ticketed"
                )
                ticket_actionability_decision = (
                    "rerun_shortline_execution_gate_and_recheck_ticket_actionability"
                )
                next_action = "rerun_shortline_execution_gate_and_recheck_ticket_actionability"
                next_action_target_artifact = "crypto_shortline_execution_gate"
                done_when = (
                    f"{symbol} reruns the shortline execution gate on the detected material change, "
                    f"reaches Setup_Ready, and a fresh crypto ticket row exists for {symbol}"
                )
            else:
                ticket_actionability_status = "crypto_shortline_gate_cold_no_edge_not_ticketed"
                ticket_actionability_decision = (
                    "wait_for_material_orderflow_change_then_refresh_execution_gate"
                )
                next_action = "monitor_material_orderflow_change_trigger"
                next_action_target_artifact = "crypto_shortline_material_change_trigger"
                done_when = (
                    f"{symbol} leaves Bias_Only, reaches Setup_Ready, and a fresh crypto ticket row exists for {symbol}"
                )
        else:
            ticket_actionability_status = "crypto_shortline_bias_only_not_ticketed"
            ticket_actionability_decision = "run_shortline_slice_backtest_and_wait_for_setup_ready"
            next_action = "run_shortline_slice_backtest_then_wait_for_setup_ready"
            next_action_target_artifact = "crypto_shortline_backtest_slice"
    elif ticket_surface.get("artifact_status") == "stale_artifact":
        ticket_actionability_status = "ticket_surface_stale"
        ticket_actionability_decision = "refresh_ticket_surface_then_recheck_route_symbol"
        actionable_ready = False
        blocker_title = "Refresh ticket surface before guarded canary review"
        next_action = "refresh_signal_to_order_tickets"
        next_action_target_artifact = "signal_to_order_tickets"
        done_when = f"latest signal_to_order_tickets row for {symbol} is fresh and actionable"
    elif ticket_surface.get("route_row_status") == "row_blocked":
        actionable_ready = False
        if "stale_signal" in route_row_reasons:
            ticket_actionability_status = "ticket_row_blocked_stale_signal"
            ticket_actionability_decision = (
                signal_source_refresh_readiness_decision
                or signal_source_freshness_decision
                or "refresh_crypto_signal_source_then_rebuild_tickets"
            )
            if (
                signal_source_refresh_readiness_status
                == "no_newer_crypto_signal_candidate_available"
            ):
                blocker_title = (
                    "Generate fresh crypto signal source before guarded canary review"
                )
            elif signal_source_refresh_readiness_status == "newer_signal_candidate_available":
                blocker_title = (
                    "Rebuild tickets from newer crypto signal candidate before guarded canary review"
                )
            else:
                blocker_title = "Refresh crypto signal source before guarded canary review"
            blocker_target_artifact = signal_source_refresh_target_artifact
            next_action = ticket_actionability_decision
            next_action_target_artifact = signal_source_refresh_target_artifact
            done_when = (
                f"{symbol} gets a fresh crypto signal source, rebuilt tickets drop stale_signal, "
                "and the route ticket becomes allowed"
            )
        else:
            if ticket_constraint_diagnosis_brief:
                if setup_transition_brief and (
                    ticket_constraint_primary_code in {
                        "route_not_setup_ready",
                        "route_missing_gates",
                    }
                    or ticket_constraint_primary_code.startswith("pattern_router:")
                ):
                    if (
                        pattern_router_brief
                        and pattern_router_family
                        in {
                            "value_rotation_scalp",
                            "sweep_reversal",
                            "imbalance_continuation",
                        }
                    ):
                        ticket_actionability_status = (
                            pattern_router_status
                            or gate_stack_progress_status
                            or setup_transition_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            pattern_router_decision
                            or gate_stack_progress_decision
                            or setup_transition_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "repair_ticket_constraints_before_promotion"
                            )
                        )
                        blocker_title = (
                            text(pattern_router_payload.get("blocker_title"))
                            or text(gate_stack_progress_payload.get("blocker_title"))
                            or text(setup_transition_watch_payload.get("blocker_title"))
                            or text(ticket_constraint_diagnosis_payload.get("blocker_title"))
                            or "Route shortline pattern before guarded canary review"
                        )
                        blocker_target_artifact = (
                            text(pattern_router_payload.get("blocker_target_artifact"))
                            or "crypto_shortline_pattern_router"
                        )
                        next_action = (
                            text(pattern_router_payload.get("next_action"))
                            or ticket_actionability_decision
                        )
                        next_action_target_artifact = (
                            text(pattern_router_payload.get("next_action_target_artifact"))
                            or "crypto_shortline_pattern_router"
                        )
                        done_when = (
                            text(pattern_router_payload.get("done_when"))
                            or (
                                f"{symbol} completes the active shortline pattern preconditions and a fresh allowed ticket row exists"
                            )
                        )
                    elif (
                        gate_stack_progress_primary_stage == "profile_location"
                        and profile_location_watch_brief
                    ):
                        ticket_actionability_status = (
                            profile_location_watch_status
                            or gate_stack_progress_status
                            or setup_transition_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            profile_location_watch_decision
                            or gate_stack_progress_decision
                            or setup_transition_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "repair_ticket_constraints_before_promotion"
                            )
                        )
                        blocker_title = (
                            text(profile_location_watch_payload.get("blocker_title"))
                            or text(gate_stack_progress_payload.get("blocker_title"))
                            or text(setup_transition_watch_payload.get("blocker_title"))
                            or text(ticket_constraint_diagnosis_payload.get("blocker_title"))
                            or "Track profile-location alignment before guarded canary review"
                        )
                        blocker_target_artifact = (
                            text(profile_location_watch_payload.get("blocker_target_artifact"))
                            or text(gate_stack_progress_payload.get("blocker_target_artifact"))
                            or "crypto_shortline_profile_location_watch"
                        )
                        next_action = (
                            text(profile_location_watch_payload.get("next_action"))
                            or text(gate_stack_progress_payload.get("next_action"))
                            or text(setup_transition_watch_payload.get("next_action"))
                            or ticket_actionability_decision
                        )
                        next_action_target_artifact = (
                            text(profile_location_watch_payload.get("next_action_target_artifact"))
                            or text(gate_stack_progress_payload.get("next_action_target_artifact"))
                            or "crypto_shortline_profile_location_watch"
                        )
                        done_when = (
                            text(profile_location_watch_payload.get("done_when"))
                            or text(gate_stack_progress_payload.get("done_when"))
                            or text(setup_transition_watch_payload.get("done_when"))
                            or (
                                f"{symbol} reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists"
                            )
                        )
                    elif gate_stack_progress_primary_stage == "mss" and mss_watch_brief:
                        ticket_actionability_status = (
                            mss_watch_status
                            or gate_stack_progress_status
                            or setup_transition_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            mss_watch_decision
                            or gate_stack_progress_decision
                            or setup_transition_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "repair_ticket_constraints_before_promotion"
                            )
                        )
                        blocker_title = (
                            text(mss_watch_payload.get("blocker_title"))
                            or text(gate_stack_progress_payload.get("blocker_title"))
                            or text(setup_transition_watch_payload.get("blocker_title"))
                            or text(ticket_constraint_diagnosis_payload.get("blocker_title"))
                            or "Track market-structure shift before guarded canary review"
                        )
                        blocker_target_artifact = (
                            text(mss_watch_payload.get("blocker_target_artifact"))
                            or text(gate_stack_progress_payload.get("blocker_target_artifact"))
                            or "crypto_shortline_mss_watch"
                        )
                        next_action = (
                            text(mss_watch_payload.get("next_action"))
                            or text(gate_stack_progress_payload.get("next_action"))
                            or text(setup_transition_watch_payload.get("next_action"))
                            or ticket_actionability_decision
                        )
                        next_action_target_artifact = (
                            text(mss_watch_payload.get("next_action_target_artifact"))
                            or text(gate_stack_progress_payload.get("next_action_target_artifact"))
                            or "crypto_shortline_mss_watch"
                        )
                        done_when = (
                            text(mss_watch_payload.get("done_when"))
                            or text(gate_stack_progress_payload.get("done_when"))
                            or text(setup_transition_watch_payload.get("done_when"))
                            or (
                                f"{symbol} reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists"
                            )
                        )
                    elif (
                        gate_stack_progress_primary_stage == "cvd_confirmation"
                        and cvd_confirmation_watch_brief
                    ):
                        ticket_actionability_status = (
                            cvd_confirmation_watch_status
                            or gate_stack_progress_status
                            or setup_transition_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            cvd_confirmation_watch_decision
                            or gate_stack_progress_decision
                            or setup_transition_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "repair_ticket_constraints_before_promotion"
                            )
                        )
                        blocker_title = (
                            text(cvd_confirmation_watch_payload.get("blocker_title"))
                            or text(gate_stack_progress_payload.get("blocker_title"))
                            or text(setup_transition_watch_payload.get("blocker_title"))
                            or text(ticket_constraint_diagnosis_payload.get("blocker_title"))
                            or "Track CVD confirmation before guarded canary review"
                        )
                        blocker_target_artifact = (
                            text(cvd_confirmation_watch_payload.get("blocker_target_artifact"))
                            or text(gate_stack_progress_payload.get("blocker_target_artifact"))
                            or "crypto_shortline_cvd_confirmation_watch"
                        )
                        next_action = (
                            text(cvd_confirmation_watch_payload.get("next_action"))
                            or text(gate_stack_progress_payload.get("next_action"))
                            or text(setup_transition_watch_payload.get("next_action"))
                            or ticket_actionability_decision
                        )
                        next_action_target_artifact = (
                            text(cvd_confirmation_watch_payload.get("next_action_target_artifact"))
                            or text(gate_stack_progress_payload.get("next_action_target_artifact"))
                            or "crypto_shortline_cvd_confirmation_watch"
                        )
                        done_when = (
                            text(cvd_confirmation_watch_payload.get("done_when"))
                            or text(gate_stack_progress_payload.get("done_when"))
                            or text(setup_transition_watch_payload.get("done_when"))
                            or (
                                f"{symbol} reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists"
                            )
                        )
                    elif (
                        gate_stack_progress_primary_stage == "liquidity_sweep"
                        and liquidity_sweep_watch_brief
                    ):
                        ticket_actionability_status = (
                            liquidity_sweep_watch_status
                            or gate_stack_progress_status
                            or setup_transition_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            liquidity_sweep_watch_decision
                            or gate_stack_progress_decision
                            or setup_transition_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "repair_ticket_constraints_before_promotion"
                            )
                        )
                        blocker_title = (
                            text(liquidity_sweep_watch_payload.get("blocker_title"))
                            or text(gate_stack_progress_payload.get("blocker_title"))
                            or text(setup_transition_watch_payload.get("blocker_title"))
                            or text(ticket_constraint_diagnosis_payload.get("blocker_title"))
                            or "Track liquidity sweep before guarded canary review"
                        )
                        blocker_target_artifact = (
                            text(liquidity_sweep_watch_payload.get("blocker_target_artifact"))
                            or text(gate_stack_progress_payload.get("blocker_target_artifact"))
                            or "crypto_shortline_liquidity_sweep_watch"
                        )
                        next_action = (
                            text(liquidity_sweep_watch_payload.get("next_action"))
                            or text(gate_stack_progress_payload.get("next_action"))
                            or text(setup_transition_watch_payload.get("next_action"))
                            or ticket_actionability_decision
                        )
                        next_action_target_artifact = (
                            text(liquidity_sweep_watch_payload.get("next_action_target_artifact"))
                            or text(gate_stack_progress_payload.get("next_action_target_artifact"))
                            or "crypto_shortline_liquidity_sweep_watch"
                        )
                        done_when = (
                            text(liquidity_sweep_watch_payload.get("done_when"))
                            or text(gate_stack_progress_payload.get("done_when"))
                            or text(setup_transition_watch_payload.get("done_when"))
                            or (
                                f"{symbol} reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists"
                            )
                        )
                    else:
                        ticket_actionability_status = gate_stack_progress_status or setup_transition_status or (
                            ticket_constraint_diagnosis_status or "ticket_row_blocked"
                        )
                        ticket_actionability_decision = (
                            gate_stack_progress_decision
                            or setup_transition_decision
                            or (
                            ticket_constraint_diagnosis_decision
                            or "repair_ticket_constraints_before_promotion"
                            )
                        )
                        blocker_title = (
                            text(gate_stack_progress_payload.get("blocker_title"))
                            or text(setup_transition_watch_payload.get("blocker_title"))
                            or text(ticket_constraint_diagnosis_payload.get("blocker_title"))
                            or "Track shortline trigger stack before guarded canary review"
                        )
                        blocker_target_artifact = (
                            text(gate_stack_progress_payload.get("blocker_target_artifact"))
                            or text(setup_transition_watch_payload.get("blocker_target_artifact"))
                            or "crypto_shortline_gate_stack_progress"
                        )
                        next_action = (
                            text(gate_stack_progress_payload.get("next_action"))
                            or text(setup_transition_watch_payload.get("next_action"))
                            or ticket_actionability_decision
                        )
                        next_action_target_artifact = (
                            text(gate_stack_progress_payload.get("next_action_target_artifact"))
                            or text(setup_transition_watch_payload.get("next_action_target_artifact"))
                            or "crypto_shortline_execution_gate"
                        )
                        done_when = (
                            text(gate_stack_progress_payload.get("done_when"))
                            or text(setup_transition_watch_payload.get("done_when"))
                            or (
                            f"{symbol} reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists"
                        )
                        )
                else:
                    if (
                        ticket_constraint_primary_code == "proxy_price_reference_only"
                        and price_reference_watch_brief
                    ):
                        ticket_actionability_status = (
                            price_reference_watch_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            price_reference_watch_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "repair_price_reference_then_recheck_execution_gate"
                            )
                        )
                        blocker_title = text(
                            price_reference_watch_payload.get("blocker_title")
                        ) or "Build executable price reference before guarded canary review"
                        blocker_target_artifact = text(
                            price_reference_watch_payload.get("blocker_target_artifact")
                        ) or "crypto_shortline_price_reference_watch"
                        next_action = text(price_reference_watch_payload.get("next_action")) or (
                            ticket_actionability_decision
                        )
                        next_action_target_artifact = text(
                            price_reference_watch_payload.get("next_action_target_artifact")
                        ) or "crypto_shortline_price_reference_watch"
                        done_when = text(price_reference_watch_payload.get("done_when")) or (
                            f"{symbol} keeps execution_price_ready=true, entry/stop/target stay non-zero, and the route ticket drops proxy_price_reference_only"
                        )
                    elif (
                        ticket_constraint_primary_code == "size_below_min_notional"
                        and sizing_watch_brief
                    ):
                        ticket_actionability_status = (
                            sizing_watch_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            sizing_watch_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "raise_effective_shortline_size_then_recheck_execution_gate"
                            )
                        )
                        blocker_title = text(
                            sizing_watch_payload.get("blocker_title")
                        ) or "Raise effective shortline size before guarded canary review"
                        blocker_target_artifact = text(
                            sizing_watch_payload.get("blocker_target_artifact")
                        ) or "crypto_shortline_sizing_watch"
                        next_action = text(sizing_watch_payload.get("next_action")) or (
                            ticket_actionability_decision
                        )
                        next_action_target_artifact = text(
                            sizing_watch_payload.get("next_action_target_artifact")
                        ) or "crypto_shortline_sizing_watch"
                        done_when = text(sizing_watch_payload.get("done_when")) or (
                            f"{symbol} clears size_below_min_notional on the route ticket"
                        )
                    elif (
                        ticket_constraint_primary_code
                        in {
                            "confidence_below_threshold",
                            "convexity_below_threshold",
                        }
                        and signal_quality_watch_brief
                    ):
                        ticket_actionability_status = (
                            signal_quality_watch_status
                            or (ticket_constraint_diagnosis_status or "ticket_row_blocked")
                        )
                        ticket_actionability_decision = (
                            signal_quality_watch_decision
                            or (
                                ticket_constraint_diagnosis_decision
                                or "repair_signal_quality_then_recheck_execution_gate"
                            )
                        )
                        blocker_title = text(
                            signal_quality_watch_payload.get("blocker_title")
                        ) or "Improve shortline signal quality before guarded canary review"
                        blocker_target_artifact = text(
                            signal_quality_watch_payload.get("blocker_target_artifact")
                        ) or "crypto_shortline_signal_quality_watch"
                        next_action = text(signal_quality_watch_payload.get("next_action")) or (
                            ticket_actionability_decision
                        )
                        next_action_target_artifact = text(
                            signal_quality_watch_payload.get("next_action_target_artifact")
                        ) or "crypto_shortline_signal_quality_watch"
                        done_when = text(signal_quality_watch_payload.get("done_when")) or (
                            f"{symbol} clears confidence/convexity/size blockers on the route ticket"
                        )
                    else:
                        ticket_actionability_status = (
                            ticket_constraint_diagnosis_status or "ticket_row_blocked"
                        )
                        ticket_actionability_decision = (
                            ticket_constraint_diagnosis_decision
                            or "repair_ticket_constraints_before_promotion"
                        )
                        blocker_title = text(
                            ticket_constraint_diagnosis_payload.get("blocker_title")
                        ) or "Repair route ticket constraints before guarded canary review"
                        blocker_target_artifact = text(
                            ticket_constraint_diagnosis_payload.get("blocker_target_artifact")
                        ) or "crypto_shortline_ticket_constraint_diagnosis"
                        next_action = text(ticket_constraint_diagnosis_payload.get("next_action")) or (
                            ticket_actionability_decision
                        )
                        next_action_target_artifact = text(
                            ticket_constraint_diagnosis_payload.get("next_action_target_artifact")
                        ) or "crypto_shortline_execution_gate"
                        done_when = text(ticket_constraint_diagnosis_payload.get("done_when")) or (
                            f"ticket row for {symbol} becomes allowed and remains fresh"
                        )
            else:
                ticket_actionability_status = "ticket_row_blocked"
                ticket_actionability_decision = "repair_ticket_constraints_before_promotion"
                blocker_title = "Repair route ticket constraints before guarded canary review"
                next_action = "repair_ticket_constraints_before_promotion"
                next_action_target_artifact = "signal_to_order_tickets"
                done_when = f"ticket row for {symbol} becomes allowed and remains fresh"
    else:
        ticket_actionability_status = "ticket_row_missing"
        ticket_actionability_decision = "refresh_ticket_surface_for_route_symbol"
        actionable_ready = False
        blocker_title = "Generate route ticket coverage before guarded canary review"
        next_action = "refresh_signal_to_order_tickets_for_route_symbol"
        next_action_target_artifact = "signal_to_order_tickets"
        done_when = f"latest ticket surface includes {symbol} with a fresh actionable row"

    blocker_detail = join_unique(
        [
            text(ticket_surface.get("ticket_match_brief")),
            (
                "ticket_surface="
                + ":".join(
                    [
                        text(ticket_surface.get("surface_coverage_status")),
                        ",".join(as_list(ticket_surface.get("surface_symbols"))) or "-",
                    ]
                )
            ),
            focus_review_brief,
            focus_review_blocker_detail,
            (
                "; ".join(
                    part
                    for part in [
                        f"shortline={execution_state}" if execution_state else "",
                        (
                            f"pattern_hint={pattern_hint_brief}"
                            if pattern_hint_brief
                            else (
                                f"pattern_hint={pattern_family_hint}:{pattern_stage_hint}"
                                if pattern_family_hint or pattern_stage_hint
                                else ""
                            )
                        ),
                        (
                            f"effective_missing_gates={','.join(effective_missing_gates)}"
                            if effective_missing_gates
                            else ""
                        ),
                        (
                            f"raw_missing_gates={','.join(missing_gates)}"
                            if effective_missing_gates and missing_gates
                            else (
                                f"missing_gates={','.join(missing_gates)}"
                                if missing_gates
                                else ""
                            )
                        ),
                    ]
                    if part
                )
                if execution_state
                or missing_gates
                or effective_missing_gates
                or pattern_hint_brief
                or pattern_family_hint
                or pattern_stage_hint
                else ""
            ),
            (
                f"cvd_queue={text(cvd_queue_payload.get('queue_status'))}:{queue_batch_name}:{queue_focus_action}"
                if text(cvd_queue_payload.get("queue_status")) or queue_batch_name or queue_focus_action
                else ""
            ),
            (f"backtest_slice={backtest_slice_brief}" if backtest_slice_brief else ""),
            (
                f"cross_section_backtest={cross_section_backtest_brief}:{cross_section_backtest_decision}"
                if cross_section_backtest_brief or cross_section_backtest_decision
                else ""
            ),
            (
                f"material_change_trigger={material_change_trigger_brief}:{material_change_trigger_decision}"
                if material_change_trigger_brief or material_change_trigger_decision
                else ""
            ),
            (
                f"pattern_router={pattern_router_brief}:{pattern_router_family}:{pattern_router_stage}:{pattern_router_decision}"
                if pattern_router_brief
                or pattern_router_family
                or pattern_router_stage
                or pattern_router_decision
                else ""
            ),
            (
                f"gate_stack_progress={gate_stack_progress_brief}:{gate_stack_progress_primary_stage}"
                if gate_stack_progress_brief or gate_stack_progress_primary_stage
                else ""
            ),
            (
                f"profile_location_watch={profile_location_watch_brief}:{profile_location_watch_decision}"
                if profile_location_watch_brief or profile_location_watch_decision
                else ""
            ),
            (
                f"mss_watch={mss_watch_brief}:{mss_watch_decision}"
                if mss_watch_brief or mss_watch_decision
                else ""
            ),
            (
                f"liquidity_sweep_watch={liquidity_sweep_watch_brief}:{liquidity_sweep_watch_decision}"
                if liquidity_sweep_watch_brief or liquidity_sweep_watch_decision
                else ""
            ),
            (
                f"price_reference_watch={price_reference_watch_brief}:{price_reference_watch_decision}"
                if price_reference_watch_brief or price_reference_watch_decision
                else ""
            ),
            (
                f"execution_quality_watch={execution_quality_watch_brief}:{execution_quality_watch_decision}"
                if execution_quality_watch_brief or execution_quality_watch_decision
                else ""
            ),
            (
                f"slippage_snapshot={slippage_snapshot_brief}:{slippage_snapshot_decision}"
                if slippage_snapshot_brief or slippage_snapshot_decision
                else ""
            ),
            (
                f"fill_capacity_watch={fill_capacity_watch_brief}:{fill_capacity_watch_decision}"
                if fill_capacity_watch_brief or fill_capacity_watch_decision
                else ""
            ),
            (
                f"signal_quality_watch={signal_quality_watch_brief}:{signal_quality_watch_decision}"
                if signal_quality_watch_brief or signal_quality_watch_decision
                else ""
            ),
            (
                f"sizing_watch={sizing_watch_brief}:{sizing_watch_decision}"
                if sizing_watch_brief or sizing_watch_decision
                else ""
            ),
            (
                f"ticket_constraint_diagnosis={ticket_constraint_diagnosis_brief}:{ticket_constraint_primary_code}"
                if ticket_constraint_diagnosis_brief or ticket_constraint_primary_code
                else ""
            ),
            (
                f"setup_transition_watch={setup_transition_brief}:{setup_transition_primary_missing_gate}"
                if setup_transition_brief or setup_transition_primary_missing_gate
                else ""
            ),
            (
                f"signal_source_freshness={signal_source_freshness_brief}:{signal_source_freshness_decision}"
                if signal_source_freshness_brief or signal_source_freshness_decision
                else ""
            ),
            (
                "signal_source_refresh_readiness="
                f"{signal_source_refresh_readiness_brief}:{signal_source_refresh_readiness_decision}"
                if signal_source_refresh_readiness_brief
                or signal_source_refresh_readiness_decision
                else ""
            ),
            (
                "ticket_signal_source="
                + ":".join(
                    [
                        ticket_signal_source_kind or "-",
                        Path(ticket_signal_source_path).name if ticket_signal_source_path else "-",
                        ticket_signal_source_artifact_date or "-",
                        (
                            f"age_days={ticket_signal_source_age_days}"
                            if ticket_signal_source_age_days is not None
                            else "age_days=-"
                        ),
                        ticket_signal_source_selection_reason or "-",
                    ]
                )
                if ticket_signal_source_path
                or ticket_signal_source_kind
                or ticket_signal_source_artifact_date
                or ticket_signal_source_selection_reason
                else ""
            ),
        ]
    )
    ticket_actionability_brief = ":".join(
        [ticket_actionability_status, symbol or "-", ticket_actionability_decision, remote_market or "-"]
    )
    return {
        "action": "build_remote_ticket_actionability_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_focus_symbol": route_focus_symbol,
        "route_action": action,
        "remote_market": remote_market,
        "actionable_ready": actionable_ready,
        "ticket_actionability_status": ticket_actionability_status,
        "ticket_actionability_brief": ticket_actionability_brief,
        "ticket_actionability_decision": ticket_actionability_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": blocker_target_artifact,
        "blocker_detail": blocker_detail,
        "next_action": next_action,
        "next_action_target_artifact": next_action_target_artifact,
        "done_when": done_when,
        "ticket_artifact": text(ticket_surface.get("artifact")),
        "ticket_artifact_status": text(ticket_surface.get("artifact_status")),
        "ticket_artifact_age_seconds": ticket_surface.get("artifact_age_seconds"),
        "ticket_match_brief": text(ticket_surface.get("ticket_match_brief")),
        "ticket_signal_source_path": ticket_signal_source_path,
        "ticket_signal_source_kind": ticket_signal_source_kind,
        "ticket_signal_source_selection_reason": ticket_signal_source_selection_reason,
        "ticket_signal_source_artifact_date": ticket_signal_source_artifact_date,
        "ticket_signal_source_age_days": ticket_signal_source_age_days,
        "signal_source_freshness_brief": signal_source_freshness_brief,
        "signal_source_freshness_status": signal_source_freshness_status,
        "signal_source_freshness_decision": signal_source_freshness_decision,
        "signal_source_freshness_refresh_recommended": signal_source_freshness_refresh_recommended,
        "signal_source_freshness_artifact": str(signal_source_freshness_path)
        if signal_source_freshness_path
        else "",
        "signal_source_refresh_readiness_brief": signal_source_refresh_readiness_brief,
        "signal_source_refresh_readiness_status": signal_source_refresh_readiness_status,
        "signal_source_refresh_readiness_decision": signal_source_refresh_readiness_decision,
        "signal_source_refresh_needed": signal_source_refresh_needed,
        "signal_source_refresh_readiness_artifact": str(signal_source_refresh_readiness_path)
        if signal_source_refresh_readiness_path
        else "",
        "pattern_router_brief": pattern_router_brief,
        "pattern_router_status": pattern_router_status,
        "pattern_router_decision": pattern_router_decision,
        "pattern_router_family": pattern_router_family,
        "pattern_router_stage": pattern_router_stage,
        "pattern_router_artifact": str(pattern_router_path) if pattern_router_path else "",
        "ticket_constraint_diagnosis_brief": ticket_constraint_diagnosis_brief,
        "ticket_constraint_diagnosis_status": ticket_constraint_diagnosis_status,
        "ticket_constraint_diagnosis_decision": ticket_constraint_diagnosis_decision,
        "ticket_constraint_primary_code": ticket_constraint_primary_code,
        "ticket_constraint_diagnosis_artifact": str(ticket_constraint_diagnosis_path)
        if ticket_constraint_diagnosis_path
        else "",
        "setup_transition_watch_brief": setup_transition_brief,
        "setup_transition_watch_status": setup_transition_status,
        "setup_transition_watch_decision": setup_transition_decision,
        "setup_transition_watch_primary_missing_gate": setup_transition_primary_missing_gate,
        "setup_transition_watch_artifact": str(setup_transition_watch_path)
        if setup_transition_watch_path
        else "",
        "gate_stack_progress_brief": gate_stack_progress_brief,
        "gate_stack_progress_status": gate_stack_progress_status,
        "gate_stack_progress_decision": gate_stack_progress_decision,
        "gate_stack_progress_primary_stage": gate_stack_progress_primary_stage,
        "gate_stack_progress_artifact": str(gate_stack_progress_path)
        if gate_stack_progress_path
        else "",
        "profile_location_watch_brief": profile_location_watch_brief,
        "profile_location_watch_status": profile_location_watch_status,
        "profile_location_watch_decision": profile_location_watch_decision,
        "profile_location_watch_artifact": str(profile_location_watch_path)
        if profile_location_watch_path
        else "",
        "mss_watch_brief": mss_watch_brief,
        "mss_watch_status": mss_watch_status,
        "mss_watch_decision": mss_watch_decision,
        "mss_watch_artifact": str(mss_watch_path) if mss_watch_path else "",
        "cvd_confirmation_watch_brief": cvd_confirmation_watch_brief,
        "cvd_confirmation_watch_status": cvd_confirmation_watch_status,
        "cvd_confirmation_watch_decision": cvd_confirmation_watch_decision,
        "cvd_confirmation_watch_artifact": str(cvd_confirmation_watch_path)
        if cvd_confirmation_watch_path
        else "",
        "liquidity_sweep_watch_brief": liquidity_sweep_watch_brief,
        "liquidity_sweep_watch_status": liquidity_sweep_watch_status,
        "liquidity_sweep_watch_decision": liquidity_sweep_watch_decision,
        "liquidity_sweep_watch_artifact": str(liquidity_sweep_watch_path)
        if liquidity_sweep_watch_path
        else "",
        "price_reference_watch_brief": price_reference_watch_brief,
        "price_reference_watch_status": price_reference_watch_status,
        "price_reference_watch_decision": price_reference_watch_decision,
        "price_reference_watch_artifact": str(price_reference_watch_path)
        if price_reference_watch_path
        else "",
        "execution_quality_watch_brief": execution_quality_watch_brief,
        "execution_quality_watch_status": execution_quality_watch_status,
        "execution_quality_watch_decision": execution_quality_watch_decision,
        "execution_quality_watch_artifact": str(execution_quality_watch_path)
        if execution_quality_watch_path
        else "",
        "slippage_snapshot_brief": slippage_snapshot_brief,
        "slippage_snapshot_status": slippage_snapshot_status,
        "slippage_snapshot_decision": slippage_snapshot_decision,
        "slippage_snapshot_artifact": str(slippage_snapshot_path)
        if slippage_snapshot_path
        else "",
        "fill_capacity_watch_brief": fill_capacity_watch_brief,
        "fill_capacity_watch_status": fill_capacity_watch_status,
        "fill_capacity_watch_decision": fill_capacity_watch_decision,
        "fill_capacity_watch_artifact": str(fill_capacity_watch_path)
        if fill_capacity_watch_path
        else "",
        "signal_quality_watch_brief": signal_quality_watch_brief,
        "signal_quality_watch_status": signal_quality_watch_status,
        "signal_quality_watch_decision": signal_quality_watch_decision,
        "signal_quality_watch_artifact": str(signal_quality_watch_path)
        if signal_quality_watch_path
        else "",
        "sizing_watch_brief": sizing_watch_brief,
        "sizing_watch_status": sizing_watch_status,
        "sizing_watch_decision": sizing_watch_decision,
        "sizing_watch_artifact": str(sizing_watch_path) if sizing_watch_path else "",
        "ticket_surface_symbols": as_list(ticket_surface.get("surface_symbols")),
        "ticket_surface_coverage_status": text(ticket_surface.get("surface_coverage_status")),
        "ticket_surface_contains_route_symbol": bool(
            ticket_surface.get("surface_contains_route_symbol", False)
        ),
        "ticket_row_status": text(ticket_surface.get("route_row_status")),
        "ticket_row_allowed": bool(ticket_surface.get("route_row_allowed", False)),
        "ticket_row_reasons": as_list(ticket_surface.get("route_row_reasons")),
        "focus_review_status": focus_review_status,
        "focus_review_brief": focus_review_brief,
        "focus_review_blocker_detail": focus_review_blocker_detail,
        "shortline_execution_state": execution_state,
        "shortline_missing_gates": missing_gates,
        "shortline_queue_focus_batch": queue_batch_name,
        "shortline_queue_focus_action": queue_focus_action,
        "shortline_queue_stack_brief": text(cvd_queue_payload.get("queue_stack_brief")),
        "shortline_cvd_queue_status": text(cvd_queue_payload.get("queue_status")),
        "shortline_cvd_semantic_status": text(cvd_queue_payload.get("semantic_status")),
        "backtest_slice_recommended": next_action_target_artifact == "crypto_shortline_backtest_slice",
        "backtest_slice_brief": backtest_slice_brief,
        "backtest_slice_artifact": str(backtest_slice_path) if backtest_slice_path else "",
        "cross_section_backtest_brief": cross_section_backtest_brief,
        "cross_section_backtest_status": cross_section_backtest_status,
        "cross_section_backtest_decision": cross_section_backtest_decision,
        "cross_section_selected_edge_status": cross_section_selected_edge_status,
        "cross_section_backtest_artifact": str(cross_section_backtest_path)
        if cross_section_backtest_path
        else "",
        "material_change_trigger_brief": material_change_trigger_brief,
        "material_change_trigger_status": material_change_trigger_status,
        "material_change_trigger_decision": material_change_trigger_decision,
        "material_change_trigger_rerun_recommended": material_change_trigger_rerun_recommended,
        "material_change_trigger_artifact": str(material_change_trigger_path)
        if material_change_trigger_path
        else "",
        "artifacts": {
            "remote_intent_queue": str(intent_queue_path) if intent_queue_path else "",
            "signal_to_order_tickets": text(ticket_surface.get("artifact")),
            "crypto_signal_source_freshness": str(signal_source_freshness_path)
            if signal_source_freshness_path
            else "",
            "crypto_signal_source_refresh_readiness": str(signal_source_refresh_readiness_path)
            if signal_source_refresh_readiness_path
            else "",
            "crypto_shortline_ticket_constraint_diagnosis": str(ticket_constraint_diagnosis_path)
            if ticket_constraint_diagnosis_path
            else "",
            "crypto_shortline_setup_transition_watch": str(setup_transition_watch_path)
            if setup_transition_watch_path
            else "",
            "crypto_shortline_gate_stack_progress": str(gate_stack_progress_path)
            if gate_stack_progress_path
            else "",
            "crypto_shortline_profile_location_watch": str(profile_location_watch_path)
            if profile_location_watch_path
            else "",
            "crypto_shortline_mss_watch": str(mss_watch_path) if mss_watch_path else "",
            "crypto_shortline_cvd_confirmation_watch": str(cvd_confirmation_watch_path)
            if cvd_confirmation_watch_path
            else "",
            "crypto_shortline_liquidity_sweep_watch": str(liquidity_sweep_watch_path)
            if liquidity_sweep_watch_path
            else "",
            "crypto_shortline_price_reference_watch": str(price_reference_watch_path)
            if price_reference_watch_path
            else "",
            "crypto_shortline_execution_quality_watch": str(execution_quality_watch_path)
            if execution_quality_watch_path
            else "",
            "crypto_shortline_slippage_snapshot": str(slippage_snapshot_path)
            if slippage_snapshot_path
            else "",
            "crypto_shortline_fill_capacity_watch": str(fill_capacity_watch_path)
            if fill_capacity_watch_path
            else "",
            "crypto_shortline_signal_quality_watch": str(signal_quality_watch_path)
            if signal_quality_watch_path
            else "",
            "crypto_shortline_sizing_watch": str(sizing_watch_path) if sizing_watch_path else "",
            "crypto_route_operator_brief": str(crypto_route_operator_path),
            "crypto_shortline_execution_gate": str(shortline_gate_path),
            "crypto_cvd_queue_handoff": str(cvd_queue_path),
            "crypto_shortline_backtest_slice": str(backtest_slice_path) if backtest_slice_path else "",
            "crypto_shortline_cross_section_backtest": str(cross_section_backtest_path)
            if cross_section_backtest_path
            else "",
            "crypto_shortline_material_change_trigger": str(material_change_trigger_path)
            if material_change_trigger_path
            else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Ticket Actionability State",
            "",
            f"- brief: `{text(payload.get('ticket_actionability_brief'))}`",
            f"- actionable_ready: `{payload.get('actionable_ready')}`",
            f"- next_action: `{text(payload.get('next_action'))}`",
            f"- next_action_target_artifact: `{text(payload.get('next_action_target_artifact'))}`",
            f"- ticket_surface_coverage_status: `{text(payload.get('ticket_surface_coverage_status'))}`",
            f"- signal_source_freshness_brief: `{text(payload.get('signal_source_freshness_brief')) or '-'}`",
            f"- signal_source_refresh_readiness_brief: `{text(payload.get('signal_source_refresh_readiness_brief')) or '-'}`",
            f"- pattern_router_brief: `{text(payload.get('pattern_router_brief')) or '-'}`",
            f"- ticket_constraint_diagnosis_brief: `{text(payload.get('ticket_constraint_diagnosis_brief')) or '-'}`",
            f"- setup_transition_watch_brief: `{text(payload.get('setup_transition_watch_brief')) or '-'}`",
            f"- gate_stack_progress_brief: `{text(payload.get('gate_stack_progress_brief')) or '-'}`",
            f"- profile_location_watch_brief: `{text(payload.get('profile_location_watch_brief')) or '-'}`",
            f"- mss_watch_brief: `{text(payload.get('mss_watch_brief')) or '-'}`",
            f"- cvd_confirmation_watch_brief: `{text(payload.get('cvd_confirmation_watch_brief')) or '-'}`",
            f"- liquidity_sweep_watch_brief: `{text(payload.get('liquidity_sweep_watch_brief')) or '-'}`",
            f"- price_reference_watch_brief: `{text(payload.get('price_reference_watch_brief')) or '-'}`",
            f"- execution_quality_watch_brief: `{text(payload.get('execution_quality_watch_brief')) or '-'}`",
            f"- slippage_snapshot_brief: `{text(payload.get('slippage_snapshot_brief')) or '-'}`",
            f"- fill_capacity_watch_brief: `{text(payload.get('fill_capacity_watch_brief')) or '-'}`",
            f"- signal_quality_watch_brief: `{text(payload.get('signal_quality_watch_brief')) or '-'}`",
            f"- sizing_watch_brief: `{text(payload.get('sizing_watch_brief')) or '-'}`",
            f"- shortline_execution_state: `{text(payload.get('shortline_execution_state'))}`",
            f"- backtest_slice_brief: `{text(payload.get('backtest_slice_brief')) or '-'}`",
            f"- cross_section_backtest_brief: `{text(payload.get('cross_section_backtest_brief')) or '-'}`",
            f"- material_change_trigger_brief: `{text(payload.get('material_change_trigger_brief')) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote ticket actionability state artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--ticket-freshness-seconds", type=int, default=DEFAULT_TICKET_FRESHNESS_SECONDS)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    parser.add_argument("--symbol", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()
    reference_now = parse_now(args.now)

    policy = load_shortline_policy(config_path)
    intent_queue_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    crypto_route_operator_path = find_latest(
        review_dir, "*_crypto_route_operator_brief.json", reference_now
    )
    missing = [
        name
        for name, path in (
            ("crypto_route_operator_brief", crypto_route_operator_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")
    intent_queue_payload = (
        load_json_mapping(intent_queue_path)
        if intent_queue_path is not None and intent_queue_path.exists()
        else {}
    )
    crypto_route_operator_payload = load_json_mapping(crypto_route_operator_path)
    symbol = resolve_symbol(
        explicit_symbol=args.symbol,
        intent_payload=intent_queue_payload,
        operator_payload=crypto_route_operator_payload,
    )
    shortline_gate_path = find_latest(
        review_dir, "*_crypto_shortline_execution_gate.json", reference_now
    )
    cvd_queue_path = find_latest(review_dir, "*_crypto_cvd_queue_handoff.json", reference_now)
    backtest_slice_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_backtest_slice.json", symbol, reference_now
    )
    cross_section_backtest_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_cross_section_backtest.json", symbol, reference_now
    )
    material_change_trigger_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_material_change_trigger.json", symbol, reference_now
    )
    pattern_router_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_pattern_router.json", symbol, reference_now
    )
    ticket_constraint_diagnosis_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_ticket_constraint_diagnosis.json", symbol, reference_now
    )
    setup_transition_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_setup_transition_watch.json", symbol, reference_now
    )
    gate_stack_progress_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_gate_stack_progress.json", symbol, reference_now
    )
    profile_location_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_profile_location_watch.json", symbol, reference_now
    )
    mss_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_mss_watch.json", symbol, reference_now
    )
    cvd_confirmation_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_cvd_confirmation_watch.json", symbol, reference_now
    )
    liquidity_sweep_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_liquidity_sweep_watch.json", symbol, reference_now
    )
    price_reference_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_price_reference_watch.json", symbol, reference_now
    )
    execution_quality_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_execution_quality_watch.json", symbol, reference_now
    )
    slippage_snapshot_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_slippage_snapshot.json", symbol, reference_now
    )
    fill_capacity_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_fill_capacity_watch.json", symbol, reference_now
    )
    signal_quality_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_signal_quality_watch.json", symbol, reference_now
    )
    sizing_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_sizing_watch.json", symbol, reference_now
    )
    signal_source_freshness_path = find_latest_scoped(
        review_dir, "*_crypto_signal_source_freshness.json", symbol, reference_now
    )
    signal_source_refresh_readiness_path = find_latest_scoped(
        review_dir, "*_crypto_signal_source_refresh_readiness.json", symbol, reference_now
    )
    missing = [
        name
        for name, path in (
            ("crypto_shortline_execution_gate", shortline_gate_path),
            ("crypto_cvd_queue_handoff", cvd_queue_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")
    tickets_path = select_ticket_surface_path(
        review_dir=review_dir,
        route_symbol=symbol,
        supported_symbols=list(policy.get("supported_symbols", [])),
        reference_now=reference_now,
    )

    payload = build_payload(
        resolved_symbol=symbol,
        intent_queue_path=intent_queue_path,
        intent_queue_payload=intent_queue_payload,
        tickets_path=tickets_path,
        signal_source_freshness_path=signal_source_freshness_path,
        signal_source_freshness_payload=load_json_mapping(signal_source_freshness_path)
        if signal_source_freshness_path is not None and signal_source_freshness_path.exists()
        else {},
        signal_source_refresh_readiness_path=signal_source_refresh_readiness_path,
        signal_source_refresh_readiness_payload=load_json_mapping(signal_source_refresh_readiness_path)
        if signal_source_refresh_readiness_path is not None
        and signal_source_refresh_readiness_path.exists()
        else {},
        crypto_route_operator_path=crypto_route_operator_path,
        crypto_route_operator_payload=crypto_route_operator_payload,
        shortline_gate_path=shortline_gate_path,
        shortline_gate_payload=load_json_mapping(shortline_gate_path),
        cvd_queue_path=cvd_queue_path,
        cvd_queue_payload=load_json_mapping(cvd_queue_path),
        backtest_slice_path=backtest_slice_path,
        backtest_slice_payload=load_json_mapping(backtest_slice_path)
        if backtest_slice_path is not None and backtest_slice_path.exists()
        else {},
        cross_section_backtest_path=cross_section_backtest_path,
        cross_section_backtest_payload=load_json_mapping(cross_section_backtest_path)
        if cross_section_backtest_path is not None and cross_section_backtest_path.exists()
        else {},
        material_change_trigger_path=material_change_trigger_path,
        material_change_trigger_payload=load_json_mapping(material_change_trigger_path)
        if material_change_trigger_path is not None and material_change_trigger_path.exists()
        else {},
        pattern_router_path=pattern_router_path,
        pattern_router_payload=load_json_mapping(pattern_router_path)
        if pattern_router_path is not None and pattern_router_path.exists()
        else {},
        profile_location_watch_path=profile_location_watch_path,
        profile_location_watch_payload=load_json_mapping(profile_location_watch_path)
        if profile_location_watch_path is not None and profile_location_watch_path.exists()
        else {},
        mss_watch_path=mss_watch_path,
        mss_watch_payload=load_json_mapping(mss_watch_path)
        if mss_watch_path is not None and mss_watch_path.exists()
        else {},
        cvd_confirmation_watch_path=cvd_confirmation_watch_path,
        cvd_confirmation_watch_payload=load_json_mapping(cvd_confirmation_watch_path)
        if cvd_confirmation_watch_path is not None
        and cvd_confirmation_watch_path.exists()
        else {},
        liquidity_sweep_watch_path=liquidity_sweep_watch_path,
        liquidity_sweep_watch_payload=load_json_mapping(liquidity_sweep_watch_path)
        if liquidity_sweep_watch_path is not None and liquidity_sweep_watch_path.exists()
        else {},
        price_reference_watch_path=price_reference_watch_path,
        price_reference_watch_payload=load_json_mapping(price_reference_watch_path)
        if price_reference_watch_path is not None and price_reference_watch_path.exists()
        else {},
        execution_quality_watch_path=execution_quality_watch_path,
        execution_quality_watch_payload=load_json_mapping(execution_quality_watch_path)
        if execution_quality_watch_path is not None and execution_quality_watch_path.exists()
        else {},
        slippage_snapshot_path=slippage_snapshot_path,
        slippage_snapshot_payload=load_json_mapping(slippage_snapshot_path)
        if slippage_snapshot_path is not None and slippage_snapshot_path.exists()
        else {},
        fill_capacity_watch_path=fill_capacity_watch_path,
        fill_capacity_watch_payload=load_json_mapping(fill_capacity_watch_path)
        if fill_capacity_watch_path is not None and fill_capacity_watch_path.exists()
        else {},
        signal_quality_watch_path=signal_quality_watch_path,
        signal_quality_watch_payload=load_json_mapping(signal_quality_watch_path)
        if signal_quality_watch_path is not None and signal_quality_watch_path.exists()
        else {},
        sizing_watch_path=sizing_watch_path,
        sizing_watch_payload=load_json_mapping(sizing_watch_path)
        if sizing_watch_path is not None and sizing_watch_path.exists()
        else {},
        gate_stack_progress_path=gate_stack_progress_path,
        gate_stack_progress_payload=load_json_mapping(gate_stack_progress_path)
        if gate_stack_progress_path is not None and gate_stack_progress_path.exists()
        else {},
        ticket_constraint_diagnosis_path=ticket_constraint_diagnosis_path,
        ticket_constraint_diagnosis_payload=load_json_mapping(ticket_constraint_diagnosis_path)
        if ticket_constraint_diagnosis_path is not None
        and ticket_constraint_diagnosis_path.exists()
        else {},
        setup_transition_watch_path=setup_transition_watch_path,
        setup_transition_watch_payload=load_json_mapping(setup_transition_watch_path)
        if setup_transition_watch_path is not None and setup_transition_watch_path.exists()
        else {},
        policy=policy,
        reference_now=reference_now,
        freshness_seconds=args.ticket_freshness_seconds,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_ticket_actionability_state.json"
    markdown = review_dir / f"{stamp}_remote_ticket_actionability_state.md"
    checksum = review_dir / f"{stamp}_remote_ticket_actionability_state_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
                "generated_at_utc": fmt_utc(reference_now),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
