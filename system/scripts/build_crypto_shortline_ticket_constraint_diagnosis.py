#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


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


def text(value: Any) -> str:
    return str(value or "").strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def select_ticket_surface_path(review_dir: Path, route_symbol: str) -> Path | None:
    candidates = sorted(review_dir.glob("*_signal_to_order_tickets.json"), reverse=True)
    if not candidates:
        return None
    normalized_route = text(route_symbol).upper()
    for path in candidates:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        for row in as_list(payload.get("tickets")):
            item = as_dict(row)
            if text(item.get("symbol")).upper() == normalized_route:
                return path
    return candidates[0]


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
        "*_crypto_shortline_ticket_constraint_diagnosis.json",
        "*_crypto_shortline_ticket_constraint_diagnosis.md",
        "*_crypto_shortline_ticket_constraint_diagnosis_checksum.json",
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


def route_symbol(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()


def route_action(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )


def payload_route_symbol(payload: dict[str, Any]) -> str:
    return text(payload.get("route_symbol")).upper()


def payload_matches_symbol(payload: dict[str, Any], symbol: str) -> bool:
    scoped_symbol = payload_route_symbol(payload)
    return not scoped_symbol or scoped_symbol == text(symbol).upper()


def scoped_payload(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    return payload if payload_matches_symbol(payload, symbol) else {}


def find_route_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("tickets")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def find_gate_stack_progress_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    normalized = text(symbol).upper()
    if text(payload.get("route_symbol")).upper() == normalized:
        return payload
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == normalized:
            return item
    return {}


def find_latest_gate_stack_progress(review_dir: Path, symbol: str) -> Path | None:
    files = sorted(review_dir.glob("*_crypto_shortline_gate_stack_progress.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    normalized = text(symbol).upper()
    for path in files:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        if find_gate_stack_progress_row(payload, normalized):
            return path
    return files[0] if files else None


def classify_constraint(
    *,
    execution_state: str,
    missing_gates: list[str],
    route_row_reasons: list[str],
) -> tuple[str, str, str, str]:
    codes: list[str] = []
    if execution_state and execution_state != "Setup_Ready":
        codes.append("route_not_setup_ready")
    if missing_gates:
        codes.append("route_missing_gates")
    codes.extend(route_row_reasons)
    primary = (
        "route_not_setup_ready"
        if "route_not_setup_ready" in codes
        else "proxy_price_reference_only"
        if "proxy_price_reference_only" in codes
        else "confidence_below_threshold"
        if "confidence_below_threshold" in codes
        else "convexity_below_threshold"
        if "convexity_below_threshold" in codes
        else "size_below_min_notional"
        if "size_below_min_notional" in codes
        else codes[0]
        if codes
        else "ticket_constraints_clear"
    )
    if "route_not_setup_ready" in codes or "route_missing_gates" in codes:
        if "proxy_price_reference_only" in codes:
            return (
                "shortline_route_not_setup_ready_proxy_price_blocked",
                "wait_for_setup_ready_and_executable_price_reference",
                "Wait for setup-ready route and executable price reference before guarded canary review",
                primary,
            )
        return (
            "shortline_route_not_setup_ready",
            "wait_for_setup_ready_then_recheck_execution_gate",
            "Wait for setup-ready route before guarded canary review",
            primary,
        )
    if "proxy_price_reference_only" in codes:
        return (
            "shortline_price_reference_blocked",
            "build_price_template_then_recheck_execution_gate",
            "Build executable price reference before guarded canary review",
            primary,
        )
    if "confidence_below_threshold" in codes or "convexity_below_threshold" in codes:
        return (
            "shortline_signal_quality_blocked",
            "improve_shortline_signal_quality_then_rebuild_ticket",
            "Improve shortline signal quality before guarded canary review",
            primary,
        )
    if "size_below_min_notional" in codes:
        return (
            "ticket_size_inputs_blocked",
            "repair_ticket_size_inputs_then_rebuild_ticket",
            "Repair ticket sizing inputs before guarded canary review",
            primary,
        )
    return (
        "ticket_constraints_clear",
        "ticket_constraints_clear",
        "Ticket constraints clear for guarded canary review",
        primary,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    gaps = as_dict(payload.get("constraint_gaps"))
    return "\n".join(
        [
            "# Crypto Shortline Ticket Constraint Diagnosis",
            "",
            f"- brief: `{text(payload.get('diagnosis_brief'))}`",
            f"- decision: `{text(payload.get('ticket_actionability_decision'))}`",
            f"- primary_constraint_code: `{text(payload.get('primary_constraint_code'))}`",
            f"- pattern_router_brief: `{text(payload.get('pattern_router_brief')) or '-'}`",
            f"- confidence_gap: `{gaps.get('confidence_gap')}`",
            f"- convexity_gap: `{gaps.get('convexity_gap')}`",
            f"- quote_gap_usdt: `{gaps.get('quote_gap_usdt')}`",
            f"- required_equity_usdt_current_signal: `{gaps.get('required_equity_usdt_current_signal')}`",
            f"- required_base_risk_pct_current_signal: `{gaps.get('required_base_risk_pct_current_signal')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_constraint_gaps(route_row: dict[str, Any], tickets_payload: dict[str, Any]) -> dict[str, Any]:
    signal = as_dict(route_row.get("signal"))
    sizing = as_dict(route_row.get("sizing"))
    thresholds = as_dict(tickets_payload.get("thresholds"))
    confidence = safe_float(signal.get("confidence"), 0.0)
    convexity = safe_float(signal.get("convexity_ratio"), 0.0)
    min_confidence = safe_float(thresholds.get("min_confidence"), 0.0)
    min_convexity = safe_float(thresholds.get("min_convexity"), 0.0)
    quote_usdt = safe_float(sizing.get("quote_usdt"), 0.0)
    min_notional_usdt = safe_float(
        sizing.get("min_notional_usdt"),
        safe_float(thresholds.get("min_notional_usdt"), 0.0),
    )
    equity_usdt = safe_float(sizing.get("equity_usdt"), 0.0)
    base_risk_pct = safe_float(thresholds.get("base_risk_pct"), 0.0)

    quote_scale = (min_notional_usdt / quote_usdt) if quote_usdt > 0.0 else None
    required_equity = equity_usdt * quote_scale if quote_scale is not None else None
    required_base_risk_pct = base_risk_pct * quote_scale if quote_scale is not None else None

    return {
        "confidence": confidence,
        "min_confidence": min_confidence,
        "confidence_gap": round(max(0.0, min_confidence - confidence), 6),
        "convexity_ratio": convexity,
        "min_convexity": min_convexity,
        "convexity_gap": round(max(0.0, min_convexity - convexity), 6),
        "quote_usdt": round(quote_usdt, 6),
        "min_notional_usdt": round(min_notional_usdt, 6),
        "quote_gap_usdt": round(max(0.0, min_notional_usdt - quote_usdt), 6),
        "quote_scale_to_min_notional": round(quote_scale, 6) if quote_scale is not None else None,
        "equity_usdt_current_signal": round(equity_usdt, 6),
        "required_equity_usdt_current_signal": round(required_equity, 6)
        if required_equity is not None
        else None,
        "base_risk_pct_current": round(base_risk_pct, 6),
        "required_base_risk_pct_current_signal": round(required_base_risk_pct, 6)
        if required_base_risk_pct is not None
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build crypto shortline ticket constraint diagnosis artifact."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="")
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    execution_gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
    freshness_path = find_latest(review_dir, "*_crypto_signal_source_freshness.json")
    readiness_path = find_latest(review_dir, "*_crypto_signal_source_refresh_readiness.json")
    pattern_router_path = find_latest(review_dir, "*_crypto_shortline_pattern_router.json")
    price_reference_watch_path = find_latest(
        review_dir, "*_crypto_shortline_price_reference_watch.json"
    )
    if operator_path is None or execution_gate_path is None:
        missing = [
            name
            for name, path in (
                ("crypto_route_operator_brief", operator_path),
                ("crypto_shortline_execution_gate", execution_gate_path),
            )
            if path is None
        ]
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    execution_gate_payload = load_json_mapping(execution_gate_path)
    symbol = text(args.symbol).upper() or route_symbol(intent_payload, operator_payload)
    gate_stack_progress_path = find_latest_gate_stack_progress(review_dir, symbol)
    freshness_payload = scoped_payload(
        load_json_mapping(freshness_path)
        if freshness_path is not None and freshness_path.exists()
        else {},
        symbol,
    )
    readiness_payload = scoped_payload(
        load_json_mapping(readiness_path)
        if readiness_path is not None and readiness_path.exists()
        else {},
        symbol,
    )
    pattern_router_payload = scoped_payload(
        load_json_mapping(pattern_router_path)
        if pattern_router_path is not None and pattern_router_path.exists()
        else {},
        symbol,
    )
    gate_stack_progress_payload = find_gate_stack_progress_row(
        load_json_mapping(gate_stack_progress_path)
        if gate_stack_progress_path is not None and gate_stack_progress_path.exists()
        else {},
        symbol,
    )
    price_reference_watch_payload = scoped_payload(
        load_json_mapping(price_reference_watch_path)
        if price_reference_watch_path is not None and price_reference_watch_path.exists()
        else {},
        symbol,
    )
    route_action_value = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    tickets_path = select_ticket_surface_path(review_dir, symbol)
    if tickets_path is None:
        raise SystemExit("missing_required_artifacts:signal_to_order_tickets")
    tickets_payload = load_json_mapping(tickets_path)
    route_row = find_route_row(tickets_payload, symbol)
    raw_route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))
    price_reference_watch_status = text(price_reference_watch_payload.get("watch_status"))
    if price_reference_watch_payload:
        if "price_reference_blocked" in price_reference_watch_payload:
            price_reference_blocked = bool(
                price_reference_watch_payload.get("price_reference_blocked", False)
            )
        else:
            price_reference_blocked = price_reference_watch_status not in {
                "",
                "price_reference_ready",
            }
    else:
        price_reference_blocked = "proxy_price_reference_only" in raw_route_row_reasons
    route_row_reasons = list(raw_route_row_reasons)
    if not price_reference_blocked:
        route_row_reasons = [
            reason for reason in route_row_reasons if reason != "proxy_price_reference_only"
        ]
    signal = as_dict(route_row.get("signal"))
    execution_state = text(find_gate_row(execution_gate_payload, symbol).get("execution_state"))
    missing_gates = dedupe_text(
        as_list(find_gate_row(execution_gate_payload, symbol).get("missing_gates"))
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
    constraint_gaps = build_constraint_gaps(route_row, tickets_payload)

    diagnosis_status, diagnosis_decision, blocker_title, primary_constraint_code = classify_constraint(
        execution_state=execution_state,
        missing_gates=missing_gates,
        route_row_reasons=route_row_reasons,
    )
    blocker_target_artifact = "crypto_shortline_ticket_constraint_diagnosis"
    next_action_target_artifact = (
        "crypto_shortline_execution_gate"
        if diagnosis_status != "ticket_size_inputs_blocked"
        else "signal_to_order_tickets"
    )
    done_when = (
        f"{symbol} reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists"
        if diagnosis_status
        in {
            "shortline_route_not_setup_ready_proxy_price_blocked",
            "shortline_route_not_setup_ready",
        }
        else f"{symbol} clears ticket constraints and a fresh allowed ticket row exists"
    )
    if (
        execution_state
        and execution_state != "Setup_Ready"
        and gate_stack_progress_status
        and not pattern_router_status
    ):
        diagnosis_status = gate_stack_progress_status
        diagnosis_decision = gate_stack_progress_decision or diagnosis_decision
        blocker_title = (
            text(gate_stack_progress_payload.get("blocker_title"))
            or blocker_title
        )
        blocker_target_artifact = (
            text(gate_stack_progress_payload.get("blocker_target_artifact"))
            or "crypto_shortline_gate_stack_progress"
        )
        next_action_target_artifact = (
            text(gate_stack_progress_payload.get("next_action_target_artifact"))
            or blocker_target_artifact
        )
        primary_constraint_code = (
            f"gate_stack:{gate_stack_progress_primary_stage or primary_constraint_code}"
        )
        done_when = text(gate_stack_progress_payload.get("done_when")) or done_when
    if (
        execution_state
        and execution_state != "Setup_Ready"
        and pattern_router_family in {"value_rotation_scalp", "sweep_reversal", "imbalance_continuation"}
        and pattern_router_status
    ):
        diagnosis_status = pattern_router_status
        diagnosis_decision = pattern_router_decision or diagnosis_decision
        blocker_title = (
            text(pattern_router_payload.get("blocker_title"))
            or blocker_title
        )
        blocker_target_artifact = (
            text(pattern_router_payload.get("blocker_target_artifact"))
            or "crypto_shortline_pattern_router"
        )
        next_action_target_artifact = (
            text(pattern_router_payload.get("next_action_target_artifact"))
            or text(pattern_router_payload.get("underlying_target_artifact"))
            or blocker_target_artifact
        )
        primary_constraint_code = (
            f"pattern_router:{pattern_router_family}:{pattern_router_stage or '-'}"
        )
        done_when = (
            text(pattern_router_payload.get("done_when"))
            or done_when
        )
    diagnosis_brief = ":".join(
        [diagnosis_status, symbol or "-", diagnosis_decision, remote_market or "-"]
    )
    blocker_detail = join_unique(
        [
            f"ticket_row_reasons={','.join(route_row_reasons) or '-'}",
            f"execution_state={execution_state or '-'}",
            (
                f"missing_gates={','.join(missing_gates)}"
                if missing_gates
                else ""
            ),
            (
                f"ticket_signal={text(tickets_payload.get('signal_source', {}).get('kind'))}:"
                f"{text(tickets_payload.get('signal_source', {}).get('artifact_date'))}"
            ),
            (
                f"ticket_signal_row={text(route_row.get('date'))}:"
                f"conf={text(signal.get('confidence'))}:conv={text(signal.get('convexity_ratio'))}:"
                f"price_ready={signal.get('execution_price_ready')}"
                if route_row
                else ""
            ),
            (
                f"ticket_gap=conf_gap:{constraint_gaps.get('confidence_gap')}:"
                f"conv_gap:{constraint_gaps.get('convexity_gap')}:"
                f"quote_gap_usdt:{constraint_gaps.get('quote_gap_usdt')}:"
                f"required_equity:{constraint_gaps.get('required_equity_usdt_current_signal')}:"
                f"required_base_risk_pct:{constraint_gaps.get('required_base_risk_pct_current_signal')}"
                if route_row
                else ""
            ),
            (
                f"gate_stack_progress={gate_stack_progress_brief}:{gate_stack_progress_primary_stage}:"
                f"{gate_stack_progress_decision}"
                if gate_stack_progress_brief
                or gate_stack_progress_primary_stage
                or gate_stack_progress_decision
                else ""
            ),
            (
                f"pattern_router={pattern_router_brief}:{pattern_router_family}:"
                f"{pattern_router_stage}:{pattern_router_decision}"
                if pattern_router_brief
                or pattern_router_family
                or pattern_router_stage
                or pattern_router_decision
                else ""
            ),
            (
                f"price_reference_watch={text(price_reference_watch_payload.get('watch_brief'))}:"
                f"{text(price_reference_watch_payload.get('watch_decision'))}"
                if price_reference_watch_payload
                else ""
            ),
            (
                text(operator_payload.get("focus_review_blocker_detail"))
                if text(operator_payload.get("review_priority_head_symbol")).upper() == symbol
                or text(operator_payload.get("next_focus_symbol")).upper() == symbol
                else ""
            ),
            (
                f"signal_source_freshness={text(freshness_payload.get('freshness_brief'))}:"
                f"{text(freshness_payload.get('freshness_decision'))}"
                if freshness_payload
                else ""
            ),
            (
                f"signal_source_refresh_readiness={text(readiness_payload.get('readiness_brief'))}:"
                f"{text(readiness_payload.get('readiness_decision'))}"
                if readiness_payload
                else ""
            ),
        ]
    )

    payload = {
        "action": "build_crypto_shortline_ticket_constraint_diagnosis",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": route_action_value,
        "remote_market": remote_market,
        "ticket_actionability_status": diagnosis_status,
        "ticket_actionability_brief": diagnosis_brief,
        "ticket_actionability_decision": diagnosis_decision,
        "diagnosis_status": diagnosis_status,
        "diagnosis_brief": diagnosis_brief,
        "diagnosis_decision": diagnosis_decision,
        "primary_constraint_code": primary_constraint_code,
        "constraint_codes": dedupe_text(
            [
                "route_not_setup_ready" if execution_state and execution_state != "Setup_Ready" else "",
                "route_missing_gates" if missing_gates else "",
                *route_row_reasons,
            ]
        ),
        "blocker_title": blocker_title,
        "blocker_target_artifact": blocker_target_artifact,
        "blocker_detail": blocker_detail,
        "next_action": diagnosis_decision,
        "next_action_target_artifact": next_action_target_artifact,
        "done_when": done_when,
        "ticket_row_reasons": route_row_reasons,
        "price_reference_blocked": price_reference_blocked,
        "price_reference_watch_status": price_reference_watch_status,
        "pattern_router_brief": pattern_router_brief,
        "pattern_router_status": pattern_router_status,
        "pattern_router_decision": pattern_router_decision,
        "pattern_router_family": pattern_router_family,
        "pattern_router_stage": pattern_router_stage,
        "gate_stack_progress_brief": gate_stack_progress_brief,
        "gate_stack_progress_status": gate_stack_progress_status,
        "gate_stack_progress_decision": gate_stack_progress_decision,
        "gate_stack_progress_primary_stage": gate_stack_progress_primary_stage,
        "shortline_execution_state": execution_state,
        "shortline_missing_gates": missing_gates,
        "ticket_surface_artifact": str(tickets_path),
        "ticket_signal_source_kind": text(as_dict(tickets_payload.get("signal_source")).get("kind")),
        "ticket_signal_source_artifact_date": text(
            as_dict(tickets_payload.get("signal_source")).get("artifact_date")
        ),
        "constraint_gaps": constraint_gaps,
        "artifacts": {
            "signal_to_order_tickets": str(tickets_path),
            "crypto_shortline_execution_gate": str(execution_gate_path),
            "crypto_route_operator_brief": str(operator_path),
            "crypto_signal_source_freshness": str(freshness_path)
            if freshness_path and freshness_payload
            else "",
            "crypto_signal_source_refresh_readiness": str(readiness_path)
            if readiness_path and readiness_payload
            else "",
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path and pattern_router_payload
            else "",
            "crypto_shortline_gate_stack_progress": str(gate_stack_progress_path)
            if gate_stack_progress_path and gate_stack_progress_payload
            else "",
            "crypto_shortline_price_reference_watch": str(price_reference_watch_path)
            if price_reference_watch_path and price_reference_watch_payload
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_ticket_constraint_diagnosis.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_ticket_constraint_diagnosis.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_ticket_constraint_diagnosis_checksum.json"
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
