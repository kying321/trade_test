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
DEFAULT_TRIGGER_STACK = (
    "liquidity_sweep",
    "mss",
    "fvg_ob_breaker_retest",
    "cvd_confirmation",
)
TRIGGER_STAGE_DEFINITIONS = (
    ("4h_profile_location", "profile_location", ("profile_location=", "cvd_key_level_context")),
    ("liquidity_sweep", "liquidity_sweep", ("liquidity_sweep",)),
    ("1m_5m_mss_or_choch", "mss", ("mss",)),
    (
        "15m_cvd_divergence_or_confirmation",
        "cvd_confirmation",
        ("cvd_local_window", "cvd_drift_guard", "cvd_attack_confirmation", "cvd_confirmation"),
    ),
    ("fvg_ob_breaker_retest", "fvg_ob_breaker_retest", ("fvg_ob_breaker_retest",)),
    (
        "15m_reversal_or_breakout_candle",
        "reversal_or_breakout_candle",
        ("reversal_or_breakout_candle",),
    ),
    ("route_state", "route_state", ("route_state=",)),
)


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


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
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


def find_latest_scoped(review_dir: Path, pattern: str, symbol: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
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
        "*_crypto_shortline_setup_transition_watch.json",
        "*_crypto_shortline_setup_transition_watch.md",
        "*_crypto_shortline_setup_transition_watch_checksum.json",
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


def normalize_trigger_stack(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        values = [text(item) for item in raw]
    else:
        values = [text(item) for item in str(raw).split(",")]
    cleaned = [item for item in values if item]
    return cleaned or list(DEFAULT_TRIGGER_STACK)


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def ordered_missing_gates(missing_gates: list[str], trigger_stack: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for trigger_gate in trigger_stack:
        matched: list[str] = []
        for alias, canonical, prefixes in TRIGGER_STAGE_DEFINITIONS:
            if trigger_gate not in {alias, canonical}:
                continue
            matched = [
                gate
                for gate in missing_gates
                if any(gate.startswith(prefix) for prefix in prefixes)
            ]
            break
        if not matched and trigger_gate in missing_gates:
            matched = [trigger_gate]
        for gate in matched:
            if gate in seen:
                continue
            ordered.append(gate)
            seen.add(gate)
    for gate in missing_gates:
        if gate not in seen:
            ordered.append(gate)
            seen.add(gate)
    return ordered


def resolve_route_symbol(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()


def resolve_route_action(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )


def resolve_symbol(
    *,
    explicit_symbol: str,
    intent_payload: dict[str, Any],
    operator_payload: dict[str, Any],
) -> str:
    return text(explicit_symbol).upper() or resolve_route_symbol(intent_payload, operator_payload)


def classify_transition(
    *,
    primary_missing_gate: str,
    execution_state: str,
    route_state: str,
    price_reference_blocked: bool,
) -> tuple[str, str, str]:
    gate = text(primary_missing_gate)
    if execution_state == "Setup_Ready" and not gate:
        return (
            "shortline_setup_transition_ready",
            "review_guarded_canary_promotion",
            "Shortline setup transition ready for guarded canary review",
        )
    if gate == "liquidity_sweep":
        status = "shortline_setup_transition_wait_liquidity_sweep"
        decision = "wait_for_liquidity_sweep_then_recheck_execution_gate"
        title = "Wait for liquidity sweep before guarded canary review"
    elif gate == "mss":
        status = "shortline_setup_transition_wait_mss"
        decision = "wait_for_mss_then_recheck_execution_gate"
        title = "Wait for market-structure shift before guarded canary review"
    elif gate == "fvg_ob_breaker_retest":
        status = "shortline_setup_transition_wait_fvg_ob_breaker_retest"
        decision = "wait_for_fvg_ob_breaker_retest_then_recheck_execution_gate"
        title = "Wait for FVG/OB retest before guarded canary review"
    elif gate == "cvd_confirmation":
        status = "shortline_setup_transition_wait_cvd_confirmation"
        decision = "wait_for_cvd_confirmation_then_recheck_execution_gate"
        title = "Wait for CVD confirmation before guarded canary review"
    elif gate.startswith("route_state="):
        status = "shortline_setup_transition_wait_route_state"
        decision = "wait_for_route_state_promotion_then_recheck_execution_gate"
        title = "Wait for route-state promotion before guarded canary review"
    else:
        status = "shortline_setup_transition_wait_gate_clearance"
        decision = "wait_for_shortline_gate_clearance_then_recheck_execution_gate"
        title = "Wait for shortline gate clearance before guarded canary review"

    if price_reference_blocked and status.startswith("shortline_setup_transition_wait_"):
        status = f"{status}_proxy_price_blocked"
    return status, decision, title


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Setup Transition Watch",
            "",
            f"- brief: `{text(payload.get('transition_brief'))}`",
            f"- decision: `{text(payload.get('transition_decision'))}`",
            f"- primary_missing_gate: `{text(payload.get('primary_missing_gate')) or '-'}`",
            f"- raw_primary_missing_gate: `{text(payload.get('raw_primary_missing_gate')) or '-'}`",
            f"- profile_watch_status: `{text(payload.get('profile_watch_status')) or '-'}`",
            f"- mss_watch_status: `{text(payload.get('mss_watch_status')) or '-'}`",
            f"- cvd_confirmation_watch_status: `{text(payload.get('cvd_confirmation_watch_status')) or '-'}`",
            f"- retest_watch_status: `{text(payload.get('retest_watch_status')) or '-'}`",
            f"- remaining_trigger_stack: `{', '.join(as_list(payload.get('remaining_trigger_stack'))) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build crypto shortline setup transition watch artifact."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    parser.add_argument("--symbol", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
    if operator_path is None or gate_path is None:
        missing = [
            name
            for name, path in (
                ("crypto_route_operator_brief", operator_path),
                ("crypto_shortline_execution_gate", gate_path),
            )
            if path is None
        ]
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    gate_payload = load_json_mapping(gate_path)
    route_focus_symbol = resolve_route_symbol(intent_payload, operator_payload)
    selected_symbol = resolve_symbol(
        explicit_symbol=args.symbol,
        intent_payload=intent_payload,
        operator_payload=operator_payload,
    )
    material_change_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_material_change_trigger.json", selected_symbol
    )
    diagnosis_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_ticket_constraint_diagnosis.json", selected_symbol
    )
    pattern_router_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_pattern_router.json", selected_symbol
    )
    profile_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_profile_location_watch.json", selected_symbol
    )
    mss_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_mss_watch.json", selected_symbol
    )
    cvd_confirmation_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_cvd_confirmation_watch.json", selected_symbol
    )
    retest_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_retest_watch.json", selected_symbol
    )
    price_reference_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_price_reference_watch.json", selected_symbol
    )
    material_change_payload = (
        load_json_mapping(material_change_path)
        if material_change_path is not None and material_change_path.exists()
        else {}
    )
    diagnosis_payload = (
        load_json_mapping(diagnosis_path)
        if diagnosis_path is not None and diagnosis_path.exists()
        else {}
    )
    pattern_router_payload = (
        load_json_mapping(pattern_router_path)
        if pattern_router_path is not None and pattern_router_path.exists()
        else {}
    )
    profile_watch_payload = (
        load_json_mapping(profile_watch_path)
        if profile_watch_path is not None and profile_watch_path.exists()
        else {}
    )
    mss_watch_payload = (
        load_json_mapping(mss_watch_path)
        if mss_watch_path is not None and mss_watch_path.exists()
        else {}
    )
    cvd_confirmation_watch_payload = (
        load_json_mapping(cvd_confirmation_watch_path)
        if cvd_confirmation_watch_path is not None
        and cvd_confirmation_watch_path.exists()
        else {}
    )
    retest_watch_payload = (
        load_json_mapping(retest_watch_path)
        if retest_watch_path is not None and retest_watch_path.exists()
        else {}
    )
    price_reference_watch_payload = (
        load_json_mapping(price_reference_watch_path)
        if price_reference_watch_path is not None and price_reference_watch_path.exists()
        else {}
    )

    route_symbol = selected_symbol
    route_action = resolve_route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    gate_row = find_gate_row(gate_payload, route_symbol)
    shortline_policy = as_dict(gate_payload.get("shortline_policy"))
    trigger_stack = normalize_trigger_stack(shortline_policy.get("trigger_stack"))
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    ordered_missing = ordered_missing_gates(missing_gates, trigger_stack)
    raw_primary_missing_gate = ordered_missing[0] if ordered_missing else ""
    execution_state = text(gate_row.get("execution_state")) or "Bias_Only"
    route_state = text(gate_row.get("route_state"))
    raw_ticket_row_reasons = dedupe_text(as_list(diagnosis_payload.get("ticket_row_reasons")))
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
        price_reference_blocked = "proxy_price_reference_only" in raw_ticket_row_reasons
    ticket_row_reasons = list(raw_ticket_row_reasons)
    if not price_reference_blocked:
        ticket_row_reasons = [
            reason for reason in ticket_row_reasons if reason != "proxy_price_reference_only"
        ]

    profile_watch_status = text(profile_watch_payload.get("watch_status"))
    profile_watch_decision = text(profile_watch_payload.get("watch_decision"))
    pattern_router_status = text(pattern_router_payload.get("pattern_status")) or text(
        pattern_router_payload.get("status")
    )
    pattern_router_family = text(pattern_router_payload.get("pattern_family"))
    pattern_router_stage = text(pattern_router_payload.get("pattern_stage"))
    profile_context_missing = any(gate.startswith("profile_location=") for gate in missing_gates) or (
        "cvd_key_level_context" in missing_gates
    )
    primary_missing_gate = raw_primary_missing_gate
    if (
        execution_state != "Setup_Ready"
        and profile_context_missing
        and profile_watch_status.startswith("profile_location_lvn_key_level_ready_rotation_")
    ):
        primary_missing_gate = "profile_location"
    if (
        execution_state != "Setup_Ready"
        and pattern_router_family == "imbalance_continuation"
        and pattern_router_status.startswith("imbalance_continuation_wait_retest")
    ):
        primary_missing_gate = "fvg_ob_breaker_retest"

    effective_ordered_missing = ordered_missing
    if primary_missing_gate:
        effective_ordered_missing = [primary_missing_gate] + [
            gate for gate in ordered_missing if gate != primary_missing_gate
        ]

    transition_status, transition_decision, blocker_title = classify_transition(
        primary_missing_gate=primary_missing_gate,
        execution_state=execution_state,
        route_state=route_state,
        price_reference_blocked=price_reference_blocked,
    )
    transition_brief = ":".join(
        [transition_status, route_symbol or "-", transition_decision, remote_market or "-"]
    )
    blocker_target_artifact = "crypto_shortline_setup_transition_watch"
    next_action = transition_decision
    next_action_target_artifact = "crypto_shortline_execution_gate"
    done_when = (
        text(gate_row.get("done_when"))
        or f"{route_symbol} reaches Setup_Ready and keeps an executable price reference"
    )
    mss_watch_status = text(mss_watch_payload.get("watch_status"))
    mss_watch_decision = text(mss_watch_payload.get("watch_decision"))
    cvd_confirmation_watch_status = text(cvd_confirmation_watch_payload.get("watch_status"))
    cvd_confirmation_watch_decision = text(cvd_confirmation_watch_payload.get("watch_decision"))
    retest_watch_status = text(retest_watch_payload.get("watch_status"))
    retest_watch_decision = text(retest_watch_payload.get("watch_decision"))
    if primary_missing_gate == "profile_location" and profile_watch_status:
        transition_status = profile_watch_status
        transition_decision = profile_watch_decision or transition_decision
        blocker_title = text(profile_watch_payload.get("blocker_title")) or blocker_title
        blocker_target_artifact = (
            text(profile_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        next_action = text(profile_watch_payload.get("next_action")) or transition_decision
        next_action_target_artifact = (
            text(profile_watch_payload.get("next_action_target_artifact"))
            or blocker_target_artifact
        )
        done_when = text(profile_watch_payload.get("done_when")) or done_when
    elif primary_missing_gate == "mss" and mss_watch_status:
        transition_status = mss_watch_status
        transition_decision = mss_watch_decision or transition_decision
        blocker_title = text(mss_watch_payload.get("blocker_title")) or blocker_title
        blocker_target_artifact = (
            text(mss_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_mss_watch"
        )
        next_action = text(mss_watch_payload.get("next_action")) or transition_decision
        next_action_target_artifact = (
            text(mss_watch_payload.get("next_action_target_artifact"))
            or blocker_target_artifact
        )
        done_when = text(mss_watch_payload.get("done_when")) or done_when
    elif primary_missing_gate == "cvd_confirmation" and cvd_confirmation_watch_status:
        transition_status = cvd_confirmation_watch_status
        transition_decision = cvd_confirmation_watch_decision or transition_decision
        blocker_title = (
            text(cvd_confirmation_watch_payload.get("blocker_title")) or blocker_title
        )
        blocker_target_artifact = (
            text(cvd_confirmation_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_cvd_confirmation_watch"
        )
        next_action = text(cvd_confirmation_watch_payload.get("next_action")) or transition_decision
        next_action_target_artifact = (
            text(cvd_confirmation_watch_payload.get("next_action_target_artifact"))
            or blocker_target_artifact
        )
        done_when = text(cvd_confirmation_watch_payload.get("done_when")) or done_when
    elif primary_missing_gate == "fvg_ob_breaker_retest" and retest_watch_status:
        transition_status = retest_watch_status
        transition_decision = retest_watch_decision or transition_decision
        blocker_title = text(retest_watch_payload.get("blocker_title")) or blocker_title
        blocker_target_artifact = (
            text(retest_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_retest_watch"
        )
        next_action = text(retest_watch_payload.get("next_action")) or transition_decision
        next_action_target_artifact = (
            text(retest_watch_payload.get("next_action_target_artifact"))
            or blocker_target_artifact
        )
        done_when = text(retest_watch_payload.get("done_when")) or done_when
    transition_brief = ":".join(
        [transition_status, route_symbol or "-", transition_decision, remote_market or "-"]
    )

    blocker_detail = join_unique(
        [
            f"execution_state={execution_state}",
            f"route_state={route_state or '-'}",
            f"raw_primary_missing_gate={raw_primary_missing_gate or '-'}",
            f"primary_missing_gate={primary_missing_gate or '-'}",
            (
                f"remaining_trigger_stack={','.join(effective_ordered_missing)}"
                if effective_ordered_missing
                else "remaining_trigger_stack=-"
            ),
            (
                f"pattern_router={pattern_router_status}:{pattern_router_family}:{pattern_router_stage}"
                if pattern_router_status
                else ""
            ),
            (
                f"profile_watch={text(profile_watch_payload.get('watch_brief'))}:"
                f"{text(profile_watch_payload.get('watch_decision'))}"
                if profile_watch_payload and primary_missing_gate == "profile_location"
                else ""
            ),
            (
                f"ticket_row_reasons={','.join(ticket_row_reasons)}"
                if ticket_row_reasons
                else ""
            ),
            (
                f"mss_watch={text(mss_watch_payload.get('watch_brief'))}:"
                f"{text(mss_watch_payload.get('watch_decision'))}"
                if mss_watch_payload and primary_missing_gate == "mss"
                else ""
            ),
            (
                f"cvd_confirmation_watch={text(cvd_confirmation_watch_payload.get('watch_brief'))}:"
                f"{text(cvd_confirmation_watch_payload.get('watch_decision'))}"
                if cvd_confirmation_watch_payload and primary_missing_gate == "cvd_confirmation"
                else ""
            ),
            (
                f"retest_watch={text(retest_watch_payload.get('watch_brief'))}:"
                f"{text(retest_watch_payload.get('watch_decision'))}"
                if retest_watch_payload and primary_missing_gate == "fvg_ob_breaker_retest"
                else ""
            ),
            (
                f"price_reference_watch={text(price_reference_watch_payload.get('watch_brief'))}:"
                f"{text(price_reference_watch_payload.get('watch_decision'))}"
                if price_reference_watch_payload
                else ""
            ),
            text(gate_row.get("blocker_detail")),
            (
                f"material_change_trigger={text(material_change_payload.get('trigger_brief'))}:"
                f"{text(material_change_payload.get('trigger_decision'))}"
                if material_change_payload
                else ""
            ),
        ]
    )

    payload = {
        "action": "build_crypto_shortline_setup_transition_watch",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "route_focus_symbol": route_focus_symbol,
        "route_action": route_action,
        "remote_market": remote_market,
        "transition_status": transition_status,
        "transition_brief": transition_brief,
        "transition_decision": transition_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": blocker_target_artifact,
        "blocker_detail": blocker_detail,
        "next_action": next_action,
        "next_action_target_artifact": next_action_target_artifact,
        "done_when": done_when,
        "shortline_execution_state": execution_state,
        "shortline_route_state": route_state,
        "primary_missing_gate": primary_missing_gate,
        "raw_primary_missing_gate": raw_primary_missing_gate,
        "pattern_router_status": pattern_router_status,
        "pattern_router_family": pattern_router_family,
        "pattern_router_stage": pattern_router_stage,
        "remaining_trigger_stack": effective_ordered_missing,
        "trigger_stack": trigger_stack,
        "ticket_row_reasons": ticket_row_reasons,
        "price_reference_blocked": price_reference_blocked,
        "price_reference_watch_status": price_reference_watch_status,
        "profile_watch_status": profile_watch_status,
        "profile_watch_brief": text(profile_watch_payload.get("watch_brief")),
        "profile_watch_decision": profile_watch_decision,
        "mss_watch_status": mss_watch_status,
        "mss_watch_brief": text(mss_watch_payload.get("watch_brief")),
        "mss_watch_decision": mss_watch_decision,
        "cvd_confirmation_watch_status": cvd_confirmation_watch_status,
        "cvd_confirmation_watch_brief": text(cvd_confirmation_watch_payload.get("watch_brief")),
        "cvd_confirmation_watch_decision": cvd_confirmation_watch_decision,
        "retest_watch_status": retest_watch_status,
        "retest_watch_brief": text(retest_watch_payload.get("watch_brief")),
        "retest_watch_decision": retest_watch_decision,
        "material_change_trigger_status": text(material_change_payload.get("trigger_status")),
        "material_change_trigger_decision": text(material_change_payload.get("trigger_decision")),
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_execution_gate": str(gate_path),
            "crypto_shortline_material_change_trigger": str(material_change_path)
            if material_change_path
            else "",
            "crypto_shortline_ticket_constraint_diagnosis": str(diagnosis_path)
            if diagnosis_path
            else "",
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path
            else "",
            "crypto_shortline_profile_location_watch": str(profile_watch_path)
            if profile_watch_path
            else "",
            "crypto_shortline_mss_watch": str(mss_watch_path)
            if mss_watch_path
            else "",
            "crypto_shortline_cvd_confirmation_watch": str(cvd_confirmation_watch_path)
            if cvd_confirmation_watch_path
            else "",
            "crypto_shortline_retest_watch": str(retest_watch_path)
            if retest_watch_path
            else "",
            "crypto_shortline_price_reference_watch": str(price_reference_watch_path)
            if price_reference_watch_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_setup_transition_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_setup_transition_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_setup_transition_watch_checksum.json"
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
