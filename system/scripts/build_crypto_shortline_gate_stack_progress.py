#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_TRIGGER_STACK = (
    "4h_profile_location",
    "liquidity_sweep",
    "1m_5m_mss_or_choch",
    "15m_cvd_divergence_or_confirmation",
    "fvg_ob_breaker_retest",
    "15m_reversal_or_breakout_candle",
)
DEFAULT_REVIEW_DIR = Path(__file__).resolve().parents[1] / "output" / "review"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    value = str(raw or "").strip()
    if not value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
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
    unique = dedupe_text(parts)
    return sep.join(unique)


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
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(review_dir: Path, pattern: str, reference_now: dt.datetime | None = None) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    if not files:
        return None
    future_cutoff = (reference_now or now_utc()) + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    for path in files:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is None or stamp_dt <= future_cutoff:
            return path
    return files[0]


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
    if not files:
        return None
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
        "*_crypto_shortline_gate_stack_progress.json",
        "*_crypto_shortline_gate_stack_progress.md",
        "*_crypto_shortline_gate_stack_progress_checksum.json",
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


def classify_route_symbol(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def stage_definitions() -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        ("4h_profile_location", "profile_location", ("profile_location=", "cvd_key_level_context")),
        ("liquidity_sweep", "liquidity_sweep", ("liquidity_sweep",)),
        ("1m_5m_mss_or_choch", "mss", ("mss",)),
        (
            "15m_cvd_divergence_or_confirmation",
            "cvd_confirmation",
            ("cvd_local_window", "cvd_drift_guard", "cvd_attack_confirmation", "cvd_confirmation"),
        ),
        ("fvg_ob_breaker_retest", "fvg_ob_breaker_retest", ("fvg_ob_breaker_retest",)),
        ("15m_reversal_or_breakout_candle", "reversal_or_breakout_candle", ("reversal_or_breakout_candle",)),
        ("route_state", "route_state", ("route_state=",)),
    ]


def stage_missing_codes(missing_gates: list[str], prefixes: tuple[str, ...]) -> list[str]:
    matched: list[str] = []
    for gate in missing_gates:
        if any(gate.startswith(prefix) for prefix in prefixes):
            matched.append(gate)
    return matched


def decision_for_stage(stage_code: str) -> str:
    if stage_code == "liquidity_sweep":
        return "wait_for_liquidity_sweep_then_refresh_gate_stack"
    if stage_code == "mss":
        return "wait_for_mss_then_refresh_gate_stack"
    if stage_code == "fvg_ob_breaker_retest":
        return "wait_for_fvg_ob_breaker_retest_then_refresh_gate_stack"
    if stage_code == "cvd_confirmation":
        return "wait_for_cvd_confirmation_then_refresh_gate_stack"
    if stage_code == "profile_location":
        return "wait_for_profile_location_alignment_then_refresh_gate_stack"
    if stage_code == "route_state":
        return "wait_for_route_state_promotion_then_refresh_gate_stack"
    if stage_code == "reversal_or_breakout_candle":
        return "wait_for_reversal_or_breakout_candle_then_refresh_gate_stack"
    return "refresh_gate_stack_after_material_change"


def title_for_stage(stage_code: str) -> str:
    if stage_code == "liquidity_sweep":
        return "Track liquidity sweep before shortline setup promotion"
    if stage_code == "mss":
        return "Track market-structure shift before shortline setup promotion"
    if stage_code == "fvg_ob_breaker_retest":
        return "Track FVG/OB retest before shortline setup promotion"
    if stage_code == "cvd_confirmation":
        return "Track CVD confirmation before shortline setup promotion"
    if stage_code == "profile_location":
        return "Track profile-location alignment before shortline setup promotion"
    if stage_code == "route_state":
        return "Track route-state promotion before shortline setup promotion"
    if stage_code == "reversal_or_breakout_candle":
        return "Track reversal/breakout candle before shortline setup promotion"
    return "Track shortline trigger stack before setup promotion"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Shortline Gate Stack Progress",
        "",
        f"- brief: `{text(payload.get('gate_stack_brief'))}`",
        f"- decision: `{text(payload.get('gate_stack_decision'))}`",
        f"- primary_stage: `{text(payload.get('primary_stage')) or '-'}`",
        f"- profile_location_watch_status: `{text(payload.get('profile_location_watch_status')) or '-'}`",
        f"- mss_watch_status: `{text(payload.get('mss_watch_status')) or '-'}`",
        f"- cvd_confirmation_watch_status: `{text(payload.get('cvd_confirmation_watch_status')) or '-'}`",
        f"- retest_watch_status: `{text(payload.get('retest_watch_status')) or '-'}`",
        f"- blocker: `{text(payload.get('blocker_detail'))}`",
        "",
        "## Stage Progress",
    ]
    for row in as_list(payload.get("gate_stack_rows")):
        item = as_dict(row)
        lines.append(
            f"- `{text(item.get('stage_code'))}` status=`{text(item.get('status'))}` missing=`{', '.join(as_list(item.get('missing_codes'))) or '-'}`"
        )
    symbol_rows = [as_dict(row) for row in as_list(payload.get("symbols")) if as_dict(row)]
    if len(symbol_rows) > 1:
        lines.extend(["", "## Symbol Progress"])
        for row in symbol_rows:
            lines.append(
                f"- `{text(row.get('symbol'))}` status=`{text(row.get('gate_stack_status'))}` primary_stage=`{text(row.get('primary_stage')) or '-'}` execution_state=`{text(row.get('execution_state')) or '-'}` route_state=`{text(row.get('route_state')) or '-'}`"
            )
    lines.append("")
    return "\n".join(lines)


def ordered_gate_symbols(route_focus_symbol: str, gate_payload: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    focus = text(route_focus_symbol).upper()
    if focus:
        ordered.append(focus)
    for raw in as_list(gate_payload.get("symbols")):
        symbol = text(as_dict(raw).get("symbol")).upper()
        if symbol and symbol not in ordered:
            ordered.append(symbol)
    return ordered


def build_progress_payload(
    *,
    review_dir: Path,
    reference_now: dt.datetime,
    intent_path: Path | None,
    intent_payload: dict[str, Any],
    operator_path: Path,
    operator_payload: dict[str, Any],
    gate_path: Path,
    gate_payload: dict[str, Any],
    route_focus_symbol: str,
    route_symbol: str,
) -> dict[str, Any]:
    material_change_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_material_change_trigger.json", route_symbol, reference_now
    )
    diagnosis_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_ticket_constraint_diagnosis.json", route_symbol, reference_now
    )
    pattern_router_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_pattern_router.json", route_symbol, reference_now
    )
    profile_location_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_profile_location_watch.json", route_symbol, reference_now
    )
    mss_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_mss_watch.json", route_symbol, reference_now
    )
    cvd_confirmation_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_cvd_confirmation_watch.json", route_symbol, reference_now
    )
    retest_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_retest_watch.json", route_symbol, reference_now
    )
    liquidity_sweep_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_liquidity_sweep_watch.json", route_symbol, reference_now
    )
    price_reference_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_price_reference_watch.json", route_symbol, reference_now
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
    profile_location_watch_payload = (
        load_json_mapping(profile_location_watch_path)
        if profile_location_watch_path is not None and profile_location_watch_path.exists()
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
    liquidity_sweep_watch_payload = (
        load_json_mapping(liquidity_sweep_watch_path)
        if liquidity_sweep_watch_path is not None and liquidity_sweep_watch_path.exists()
        else {}
    )
    price_reference_watch_payload = (
        load_json_mapping(price_reference_watch_path)
        if price_reference_watch_path is not None and price_reference_watch_path.exists()
        else {}
    )

    route_action = text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    gate_row = find_gate_row(gate_payload, route_symbol)
    shortline_policy = as_dict(gate_payload.get("shortline_policy"))
    trigger_stack = normalize_trigger_stack(shortline_policy.get("trigger_stack"))
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
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
    execution_state = text(gate_row.get("execution_state")) or "Bias_Only"
    route_state = text(gate_row.get("route_state"))
    pattern_router_status = text(pattern_router_payload.get("pattern_status")) or text(
        pattern_router_payload.get("status")
    )
    pattern_router_family = text(pattern_router_payload.get("pattern_family"))
    pattern_router_stage = text(pattern_router_payload.get("pattern_stage"))
    pattern_router_retest_override = (
        execution_state != "Setup_Ready"
        and pattern_router_family == "imbalance_continuation"
        and pattern_router_status.startswith("imbalance_continuation_wait_retest")
    )

    rows: list[dict[str, Any]] = []
    primary_stage = ""
    primary_seen = False
    for stage_label, stage_code, prefixes in stage_definitions():
        missing_codes = stage_missing_codes(missing_gates, prefixes)
        if execution_state == "Setup_Ready":
            status = "cleared"
        elif missing_codes and not primary_seen:
            status = "blocking"
            primary_stage = primary_stage or stage_code
            primary_seen = True
        elif primary_seen:
            status = "pending"
        else:
            status = "cleared"
        rows.append(
            {
                "stage_label": stage_label,
                "stage_code": stage_code,
                "status": status,
                "missing_codes": missing_codes,
            }
        )

    if pattern_router_retest_override:
        primary_stage = "fvg_ob_breaker_retest"
        before_primary = True
        for row in rows:
            stage_code = text(row.get("stage_code"))
            if stage_code == primary_stage:
                row["status"] = "blocking"
                before_primary = False
            elif before_primary:
                row["status"] = "cleared"
            else:
                row["status"] = "pending" if as_list(row.get("missing_codes")) else "cleared"

    if execution_state == "Setup_Ready":
        gate_stack_status = "shortline_gate_stack_ready"
        gate_stack_decision = "review_setup_transition_readiness"
        blocker_title = "Shortline trigger stack ready for guarded canary review"
    else:
        gate_stack_status = f"shortline_gate_stack_blocked_at_{primary_stage or 'unknown'}"
        gate_stack_decision = decision_for_stage(primary_stage)
        blocker_title = title_for_stage(primary_stage)
    if price_reference_blocked and gate_stack_status.startswith("shortline_gate_stack_blocked_at_"):
        gate_stack_status = f"{gate_stack_status}_proxy_price_blocked"

    blocker_target_artifact = "crypto_shortline_gate_stack_progress"
    next_action = gate_stack_decision
    next_action_target_artifact = "crypto_shortline_execution_gate"
    done_when = (
        text(gate_row.get("done_when"))
        or f"{route_symbol} clears the remaining shortline trigger stack and keeps an executable price reference"
    )
    liquidity_sweep_watch_status = text(liquidity_sweep_watch_payload.get("watch_status"))
    liquidity_sweep_watch_decision = text(liquidity_sweep_watch_payload.get("watch_decision"))
    profile_location_watch_status = text(profile_location_watch_payload.get("watch_status"))
    profile_location_watch_decision = text(profile_location_watch_payload.get("watch_decision"))
    mss_watch_status = text(mss_watch_payload.get("watch_status"))
    mss_watch_decision = text(mss_watch_payload.get("watch_decision"))
    cvd_confirmation_watch_status = text(cvd_confirmation_watch_payload.get("watch_status"))
    cvd_confirmation_watch_decision = text(
        cvd_confirmation_watch_payload.get("watch_decision")
    )
    retest_watch_status = text(retest_watch_payload.get("watch_status"))
    retest_watch_decision = text(retest_watch_payload.get("watch_decision"))
    if primary_stage == "profile_location" and profile_location_watch_status:
        gate_stack_decision = profile_location_watch_decision or gate_stack_decision
        blocker_title = (
            text(profile_location_watch_payload.get("blocker_title")) or blocker_title
        )
        blocker_target_artifact = (
            text(profile_location_watch_payload.get("blocker_target_artifact"))
            or blocker_target_artifact
        )
        next_action = text(profile_location_watch_payload.get("next_action")) or gate_stack_decision
        next_action_target_artifact = (
            text(profile_location_watch_payload.get("next_action_target_artifact"))
            or next_action_target_artifact
        )
        done_when = text(profile_location_watch_payload.get("done_when")) or done_when
    if primary_stage == "liquidity_sweep" and liquidity_sweep_watch_status:
        gate_stack_decision = liquidity_sweep_watch_decision or gate_stack_decision
        blocker_title = (
            text(liquidity_sweep_watch_payload.get("blocker_title")) or blocker_title
        )
        blocker_target_artifact = (
            text(liquidity_sweep_watch_payload.get("blocker_target_artifact"))
            or blocker_target_artifact
        )
        next_action = text(liquidity_sweep_watch_payload.get("next_action")) or gate_stack_decision
        next_action_target_artifact = (
            text(liquidity_sweep_watch_payload.get("next_action_target_artifact"))
            or next_action_target_artifact
        )
        done_when = text(liquidity_sweep_watch_payload.get("done_when")) or done_when
    if primary_stage == "mss" and mss_watch_status:
        gate_stack_decision = mss_watch_decision or gate_stack_decision
        blocker_title = text(mss_watch_payload.get("blocker_title")) or blocker_title
        blocker_target_artifact = (
            text(mss_watch_payload.get("blocker_target_artifact"))
            or blocker_target_artifact
        )
        next_action = text(mss_watch_payload.get("next_action")) or gate_stack_decision
        next_action_target_artifact = (
            text(mss_watch_payload.get("next_action_target_artifact"))
            or next_action_target_artifact
        )
        done_when = text(mss_watch_payload.get("done_when")) or done_when
    if primary_stage == "cvd_confirmation" and cvd_confirmation_watch_status:
        gate_stack_decision = cvd_confirmation_watch_decision or gate_stack_decision
        blocker_title = (
            text(cvd_confirmation_watch_payload.get("blocker_title")) or blocker_title
        )
        blocker_target_artifact = (
            text(cvd_confirmation_watch_payload.get("blocker_target_artifact"))
            or blocker_target_artifact
        )
        next_action = (
            text(cvd_confirmation_watch_payload.get("next_action")) or gate_stack_decision
        )
        next_action_target_artifact = (
            text(cvd_confirmation_watch_payload.get("next_action_target_artifact"))
            or next_action_target_artifact
        )
        done_when = text(cvd_confirmation_watch_payload.get("done_when")) or done_when
    if primary_stage == "fvg_ob_breaker_retest" and retest_watch_status:
        gate_stack_decision = retest_watch_decision or gate_stack_decision
        blocker_title = text(retest_watch_payload.get("blocker_title")) or blocker_title
        blocker_target_artifact = (
            text(retest_watch_payload.get("blocker_target_artifact"))
            or blocker_target_artifact
        )
        next_action = text(retest_watch_payload.get("next_action")) or gate_stack_decision
        next_action_target_artifact = (
            text(retest_watch_payload.get("next_action_target_artifact"))
            or next_action_target_artifact
        )
        done_when = text(retest_watch_payload.get("done_when")) or done_when

    remaining_blocked = [
        text(row.get("stage_code"))
        for row in rows
        if text(row.get("status")) in {"blocking", "pending"}
    ]
    gate_stack_brief = ":".join(
        [gate_stack_status, route_symbol or "-", gate_stack_decision, remote_market or "-"]
    )
    blocker_detail = join_unique(
        [
            f"execution_state={execution_state}",
            f"route_state={route_state or '-'}",
            f"primary_stage={primary_stage or '-'}",
            (
                f"pattern_router={pattern_router_status}:{pattern_router_family}:{pattern_router_stage}"
                if pattern_router_status
                else ""
            ),
            f"remaining_blocked_stages={','.join(remaining_blocked) if remaining_blocked else '-'}",
            f"ticket_row_reasons={','.join(ticket_row_reasons)}" if ticket_row_reasons else "",
            (
                f"profile_location_watch={text(profile_location_watch_payload.get('watch_brief'))}:"
                f"{text(profile_location_watch_payload.get('watch_decision'))}"
                if profile_location_watch_payload
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
            (
                f"mss_watch={text(mss_watch_payload.get('watch_brief'))}:"
                f"{text(mss_watch_payload.get('watch_decision'))}"
                if mss_watch_payload and primary_stage == "mss"
                else ""
            ),
            (
                f"cvd_confirmation_watch={text(cvd_confirmation_watch_payload.get('watch_brief'))}:"
                f"{text(cvd_confirmation_watch_payload.get('watch_decision'))}"
                if cvd_confirmation_watch_payload and primary_stage == "cvd_confirmation"
                else ""
            ),
            (
                f"retest_watch={text(retest_watch_payload.get('watch_brief'))}:"
                f"{text(retest_watch_payload.get('watch_decision'))}"
                if retest_watch_payload and primary_stage == "fvg_ob_breaker_retest"
                else ""
            ),
            (
                f"liquidity_sweep_watch={text(liquidity_sweep_watch_payload.get('watch_brief'))}:"
                f"{text(liquidity_sweep_watch_payload.get('watch_decision'))}"
                if liquidity_sweep_watch_payload and primary_stage == "liquidity_sweep"
                else ""
            ),
        ]
    )
    return {
        "symbol": route_symbol,
        "route_symbol": route_symbol,
        "route_focus_symbol": route_focus_symbol,
        "route_action": route_action,
        "remote_market": remote_market,
        "gate_stack_status": gate_stack_status,
        "gate_stack_brief": gate_stack_brief,
        "gate_stack_decision": gate_stack_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": blocker_target_artifact,
        "blocker_detail": blocker_detail,
        "next_action": next_action,
        "next_action_target_artifact": next_action_target_artifact,
        "done_when": done_when,
        "execution_state": execution_state,
        "route_state": route_state,
        "primary_stage": primary_stage,
        "pattern_router_status": pattern_router_status,
        "pattern_router_family": pattern_router_family,
        "pattern_router_stage": pattern_router_stage,
        "remaining_blocked_stages": remaining_blocked,
        "gate_stack_rows": rows,
        "ticket_row_reasons": ticket_row_reasons,
        "price_reference_blocked": price_reference_blocked,
        "profile_location_watch_status": profile_location_watch_status,
        "profile_location_watch_decision": profile_location_watch_decision,
        "mss_watch_status": mss_watch_status,
        "mss_watch_brief": text(mss_watch_payload.get("watch_brief")),
        "mss_watch_decision": mss_watch_decision,
        "cvd_confirmation_watch_status": cvd_confirmation_watch_status,
        "cvd_confirmation_watch_brief": text(cvd_confirmation_watch_payload.get("watch_brief")),
        "cvd_confirmation_watch_decision": cvd_confirmation_watch_decision,
        "retest_watch_status": retest_watch_status,
        "retest_watch_brief": text(retest_watch_payload.get("watch_brief")),
        "retest_watch_decision": retest_watch_decision,
        "price_reference_watch_status": price_reference_watch_status,
        "material_change_trigger_status": text(material_change_payload.get("trigger_status")),
        "material_change_trigger_decision": text(material_change_payload.get("trigger_decision")),
        "liquidity_sweep_watch_status": liquidity_sweep_watch_status,
        "liquidity_sweep_watch_decision": liquidity_sweep_watch_decision,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_execution_gate": str(gate_path),
            "crypto_shortline_material_change_trigger": str(material_change_path)
            if material_change_path
            else "",
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path
            else "",
            "crypto_shortline_profile_location_watch": str(profile_location_watch_path)
            if profile_location_watch_path
            else "",
            "crypto_shortline_mss_watch": str(mss_watch_path) if mss_watch_path else "",
            "crypto_shortline_cvd_confirmation_watch": str(cvd_confirmation_watch_path)
            if cvd_confirmation_watch_path
            else "",
            "crypto_shortline_retest_watch": str(retest_watch_path)
            if retest_watch_path
            else "",
            "crypto_shortline_ticket_constraint_diagnosis": str(diagnosis_path)
            if diagnosis_path
            else "",
            "crypto_shortline_liquidity_sweep_watch": str(liquidity_sweep_watch_path)
            if liquidity_sweep_watch_path
            else "",
            "crypto_shortline_price_reference_watch": str(price_reference_watch_path)
            if price_reference_watch_path
            else "",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crypto shortline gate-stack progress artifact.")
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

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json", reference_now)
    gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json", reference_now)
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

    intent_payload = load_json_mapping(intent_path) if intent_path and intent_path.exists() else {}
    operator_payload = load_json_mapping(operator_path)
    gate_payload = load_json_mapping(gate_path)
    route_focus_symbol = classify_route_symbol(intent_payload, operator_payload)
    route_symbol = text(args.symbol).upper() or route_focus_symbol
    payload = {
        "action": "build_crypto_shortline_gate_stack_progress",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        **build_progress_payload(
            review_dir=review_dir,
            reference_now=reference_now,
            intent_path=intent_path,
            intent_payload=intent_payload,
            operator_path=operator_path,
            operator_payload=operator_payload,
            gate_path=gate_path,
            gate_payload=gate_payload,
            route_focus_symbol=route_focus_symbol,
            route_symbol=route_symbol,
        ),
    }
    symbol_rows: list[dict[str, Any]] = []
    for symbol in (
        [route_symbol]
        if text(args.symbol)
        else ordered_gate_symbols(route_focus_symbol, gate_payload)
    ):
        symbol_payload = payload if symbol == route_symbol else build_progress_payload(
            review_dir=review_dir,
            reference_now=reference_now,
            intent_path=intent_path,
            intent_payload=intent_payload,
            operator_path=operator_path,
            operator_payload=operator_payload,
            gate_path=gate_path,
            gate_payload=gate_payload,
            route_focus_symbol=route_focus_symbol,
            route_symbol=symbol,
        )
        symbol_rows.append(
            {
                key: value
                for key, value in symbol_payload.items()
                if key not in {"action", "ok", "status", "generated_at_utc", "artifact", "markdown", "checksum"}
            }
        )
    payload["symbols"] = symbol_rows
    payload["symbol_count"] = len(symbol_rows)
    payload["ready_symbol_count"] = sum(
        1 for row in symbol_rows if text(row.get("execution_state")) == "Setup_Ready"
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_gate_stack_progress.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_gate_stack_progress.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_gate_stack_progress_checksum.json"
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
