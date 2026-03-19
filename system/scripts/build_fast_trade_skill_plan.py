#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XAUUSD")
SKILL_ID = "fast_order_risk_skill_v1"
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_MIN_CONFIDENCE = 60.0
DEFAULT_MIN_CONVEXITY = 3.0
DEFAULT_SHORTLINE_SUPPORTED = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DEFAULT_SHORTLINE_CAUTION = ("PAXGUSDT",)
DEFAULT_SHORTLINE_UNSUPPORTED = ("XAGUSDT", "XAUTUSDT")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def pick_latest_tickets(review_dir: Path) -> Path:
    files = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    if not files:
        raise FileNotFoundError(f"no signal_to_order_tickets artifact under: {review_dir}")
    return files[-1]


def normalize_symbols(raw: str) -> tuple[str, ...]:
    parts = [str(x).strip().upper() for x in str(raw).split(",")]
    out = [x for x in parts if x]
    return tuple(out) if out else DEFAULT_SYMBOLS


def normalize_symbol_list(raw: Any, *, default: tuple[str, ...]) -> list[str]:
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip().upper() for x in raw]
    else:
        items = [str(x).strip().upper() for x in str(raw).split(",")]
    out = [item for item in items if item]
    return out if out else list(default)


def load_policy_thresholds(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    source = "builtin_default"
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = str(config_path)
        except Exception:
            payload = {}
            source = f"invalid_config:{config_path}"
    thresholds = payload.get("thresholds", {}) if isinstance(payload, dict) else {}
    return {
        "signal_confidence_min": float(to_float(thresholds.get("signal_confidence_min", DEFAULT_MIN_CONFIDENCE), DEFAULT_MIN_CONFIDENCE)),
        "convexity_min": float(to_float(thresholds.get("convexity_min", DEFAULT_MIN_CONVEXITY), DEFAULT_MIN_CONVEXITY)),
        "source": source,
    }


def load_shortline_policy(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    source = "builtin_default"
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = str(config_path)
        except Exception:
            payload = {}
            source = f"invalid_config:{config_path}"
    shortline = payload.get("shortline", {}) if isinstance(payload, dict) else {}
    return {
        "status": str(shortline.get("status", "builtin_default")),
        "structure_timeframe": str(shortline.get("structure_timeframe", "4h")),
        "execution_timeframe": str(shortline.get("execution_timeframe", "15m")),
        "structure_engine": str(shortline.get("structure_engine", "fixed_range_volume_profile_proxy")),
        "structure_reference": str(shortline.get("structure_reference", "vpvr_concepts_deterministic_proxy")),
        "market_state_engine": str(shortline.get("market_state_engine", "bias_only_vs_setup_ready")),
        "default_market_state": str(shortline.get("default_market_state", "Bias_Only")),
        "setup_ready_state": str(shortline.get("setup_ready_state", "Setup_Ready")),
        "profile_lookback_bars": int(to_int(shortline.get("profile_lookback_bars", 120), 120)),
        "profile_value_area_pct": float(to_float(shortline.get("profile_value_area_pct", 0.70), 0.70)),
        "location_priority": normalize_symbol_list(shortline.get("location_priority", ("HVN", "POC")), default=("HVN", "POC")),
        "avoid_location": str(shortline.get("avoid_location", "LVN")),
        "no_trade_zone": str(shortline.get("no_trade_zone", "range_mid")),
        "no_trade_rule": str(shortline.get("no_trade_rule", "no_sweep_no_mss_no_cvd_no_trade")),
        "flow_confirmation_engine": str(shortline.get("flow_confirmation_engine", "cvd_lite")),
        "flow_confirmation_role": str(shortline.get("flow_confirmation_role", "confirm_and_veto_only")),
        "flow_confirmation_half_life_seconds": int(
            to_int(shortline.get("flow_confirmation_half_life_seconds", 180), 180)
        ),
        "micro_structure_engine": str(shortline.get("micro_structure_engine", "ict_sweep_mss")),
        "micro_structure_timeframes": [str(x).strip() for x in shortline.get("micro_structure_timeframes", []) if str(x).strip()]
        or ["1m", "5m"],
        "liquidity_sweep_required": bool(shortline.get("liquidity_sweep_required", True)),
        "mss_required": bool(shortline.get("mss_required", True)),
        "entry_retest_priority": [str(x).strip() for x in shortline.get("entry_retest_priority", []) if str(x).strip()]
        or ["FVG", "OB", "Breaker"],
        "session_liquidity_map": [str(x).strip() for x in shortline.get("session_liquidity_map", []) if str(x).strip()]
        or ["asia_high_low", "london_high_low", "prior_day_high_low", "equal_highs_lows"],
        "execution_style": str(shortline.get("execution_style", "right_side_only")),
        "holding_window_minutes": {
            "min": int(to_int((shortline.get("holding_window_minutes", {}) or {}).get("min", 15), 15)),
            "max": int(to_int((shortline.get("holding_window_minutes", {}) or {}).get("max", 180), 180)),
        },
        "trigger_stack": [str(x).strip() for x in shortline.get("trigger_stack", []) if str(x).strip()]
        or [
            "4h_profile_location",
            "liquidity_sweep",
            "1m_5m_mss_or_choch",
            "15m_cvd_divergence_or_confirmation",
            "fvg_ob_breaker_retest",
            "15m_reversal_or_breakout_candle",
        ],
        "supported_symbols": normalize_symbol_list(
            shortline.get("supported_symbols", DEFAULT_SHORTLINE_SUPPORTED),
            default=DEFAULT_SHORTLINE_SUPPORTED,
        ),
        "caution_symbols": normalize_symbol_list(
            shortline.get("caution_symbols", DEFAULT_SHORTLINE_CAUTION),
            default=DEFAULT_SHORTLINE_CAUTION,
        ),
        "unsupported_binance_spot_symbols": normalize_symbol_list(
            shortline.get("unsupported_binance_spot_symbols", DEFAULT_SHORTLINE_UNSUPPORTED),
            default=DEFAULT_SHORTLINE_UNSUPPORTED,
        ),
        "metals_note": str(
            shortline.get(
                "metals_note",
                "Binance spot metals support is proxy-grade: PAXGUSDT is supported; XAGUSDT and XAUTUSDT are not listed.",
            )
        ),
        "source": source,
    }


def derive_quote_usdt(
    *,
    equity_usdt: float,
    reserve_pct: float,
    alloc_pct: float,
    min_notional_usdt: float,
) -> float:
    reserve = max(0.0, min(0.95, float(reserve_pct)))
    alloc = max(0.01, min(0.95, float(alloc_pct)))
    reserve_quote = max(0.0, float(equity_usdt) * reserve)
    usable_quote = max(0.0, float(equity_usdt) - reserve_quote)
    alloc_cap = max(0.0, float(equity_usdt) * alloc)
    quote = min(usable_quote, alloc_cap)
    if quote > 0.0:
        quote = max(float(min_notional_usdt), quote)
    return round(float(quote), 4)


def side_to_order_side(side: str) -> str:
    side_u = str(side).strip().upper()
    if side_u in {"LONG", "BUY", "B"}:
        return "BUY"
    if side_u in {"SHORT", "SELL", "S"}:
        return "SELL"
    return "BUY"


def calc_risk_metrics(*, side: str, entry_price: float, stop_price: float, target_price: float) -> dict[str, float | None]:
    entry = max(0.0, float(entry_price))
    stop = max(0.0, float(stop_price))
    target = max(0.0, float(target_price))
    if entry <= 0.0 or stop <= 0.0 or target <= 0.0:
        return {
            "risk_per_unit": None,
            "reward_per_unit": None,
            "rr_ratio": None,
            "stop_gap_pct": None,
            "target_gap_pct": None,
        }

    side_u = str(side).strip().upper()
    if side_u in {"LONG", "BUY", "B"}:
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0.0:
        return {
            "risk_per_unit": None,
            "reward_per_unit": None,
            "rr_ratio": None,
            "stop_gap_pct": None,
            "target_gap_pct": None,
        }
    rr = reward / risk if reward > 0.0 else 0.0
    return {
        "risk_per_unit": float(risk),
        "reward_per_unit": float(reward),
        "rr_ratio": float(rr),
        "stop_gap_pct": float(abs(risk) / entry * 100.0),
        "target_gap_pct": float(abs(reward) / entry * 100.0),
    }


def build_fast_plan(
    *,
    tickets_payload: dict[str, Any],
    symbols: tuple[str, ...],
    equity_usdt_override: float | None,
    reserve_pct: float,
    alloc_pct: float,
    min_notional_usdt: float,
    max_age_days: int,
    min_confidence: float,
    min_convexity: float,
    decision_ttl_seconds: int,
) -> dict[str, Any]:
    sizing_context = tickets_payload.get("sizing_context", {}) if isinstance(tickets_payload, dict) else {}
    equity_usdt = to_float(sizing_context.get("equity_usdt", 0.0), 0.0)
    equity_source = str(sizing_context.get("equity_source", "unknown"))
    if equity_usdt_override is not None and equity_usdt_override > 0.0:
        equity_usdt = float(equity_usdt_override)
        equity_source = "cli_override"

    quote_usdt = derive_quote_usdt(
        equity_usdt=equity_usdt,
        reserve_pct=reserve_pct,
        alloc_pct=alloc_pct,
        min_notional_usdt=min_notional_usdt,
    )
    as_of = str(tickets_payload.get("as_of", ""))

    rows: list[dict[str, Any]] = []
    for raw in (tickets_payload.get("tickets", []) if isinstance(tickets_payload, dict) else []):
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol", "")).upper()
        if symbol not in symbols:
            continue

        signal = raw.get("signal", {}) if isinstance(raw.get("signal", {}), dict) else {}
        levels = raw.get("levels", {}) if isinstance(raw.get("levels", {}), dict) else {}
        execution = raw.get("execution", {}) if isinstance(raw.get("execution", {}), dict) else {}

        reasons = [str(x) for x in raw.get("reasons", []) if str(x)]
        side = str(signal.get("side", "")).strip().upper()
        confidence = to_float(signal.get("confidence", 0.0), 0.0)
        convexity = to_float(signal.get("convexity_ratio", 0.0), 0.0)
        age_days = to_int(raw.get("age_days", -1), -1)
        base_allowed = bool(raw.get("allowed", False))
        stale = ("stale_signal" in reasons) or (age_days >= 0 and age_days > int(max_age_days))
        not_found = "signal_not_found" in reasons
        only_size_blocked = bool(reasons) and all(x == "size_below_min_notional" for x in reasons)

        action = "SKIP"
        action_reason = "gate_not_met"
        executable = False
        order_side = side_to_order_side(side)

        if not not_found and not stale:
            if side in {"LONG", "BUY", "B"} and confidence >= min_confidence and convexity >= min_convexity:
                if base_allowed or only_size_blocked:
                    action = "CANARY_BUY"
                    action_reason = "micro_override_size_floor" if only_size_blocked else "base_allowed"
                    executable = quote_usdt >= float(min_notional_usdt)
                else:
                    action = "SKIP"
                    action_reason = "base_allowed_false"
            elif side in {"SHORT", "SELL", "S"}:
                action = "HEDGE_ONLY"
                action_reason = "spot_account_no_direct_short"
            else:
                action = "SKIP"
                action_reason = "side_or_threshold_not_met"
        elif not_found:
            action_reason = "signal_not_found"
        elif stale:
            action_reason = "stale_signal"

        risk = calc_risk_metrics(
            side=side,
            entry_price=to_float(levels.get("entry_price", 0.0), 0.0),
            stop_price=to_float(levels.get("stop_price", 0.0), 0.0),
            target_price=to_float(levels.get("target_price", 0.0), 0.0),
        )
        score = float(max(0.0, confidence) * max(0.0, convexity))

        rows.append(
            {
                "symbol": symbol,
                "signal_date": str(raw.get("date", "")),
                "age_days": age_days,
                "side": side,
                "regime": str(signal.get("regime", "")),
                "confidence": float(confidence),
                "convexity_ratio": float(convexity),
                "score": score,
                "base_allowed": bool(base_allowed),
                "reasons": reasons,
                "stale": bool(stale),
                "not_found": bool(not_found),
                "action": action,
                "action_reason": action_reason,
                "executable": bool(executable),
                "order_side": order_side,
                "quote_usdt": float(quote_usdt if executable else 0.0),
                "entry_price": to_float(levels.get("entry_price", 0.0), 0.0),
                "stop_price": to_float(levels.get("stop_price", 0.0), 0.0),
                "target_price": to_float(levels.get("target_price", 0.0), 0.0),
                "risk": risk,
                "execution_hint": {
                    "order_type_hint": str(execution.get("order_type_hint", "LIMIT") or "LIMIT"),
                    "max_slippage_bps": float(to_float(execution.get("max_slippage_bps", 6.0), 6.0)),
                },
            }
        )

    rows_sorted = sorted(rows, key=lambda x: (0 if bool(x.get("executable", False)) else 1, -to_float(x.get("score", 0.0), 0.0), str(x.get("symbol", ""))))
    selected_row = next((r for r in rows_sorted if bool(r.get("executable", False))), None)

    selected: dict[str, Any]
    if selected_row is None:
        selected = {
            "executable": False,
            "reason": "no_actionable_candidate",
        }
    else:
        selected_symbol = str(selected_row.get("symbol", ""))
        selected_side = str(selected_row.get("order_side", "BUY"))
        selected_quote = float(to_float(selected_row.get("quote_usdt", 0.0), 0.0))
        seed = f"{as_of}:{selected_symbol}:{selected_side}:{selected_quote:.4f}:{SKILL_ID}"
        selected = {
            "executable": True,
            "skill_id": SKILL_ID,
            "symbol": selected_symbol,
            "side": selected_side,
            "quote_usdt": selected_quote,
            "order_type_hint": str((selected_row.get("execution_hint", {}) or {}).get("order_type_hint", "LIMIT")),
            "max_slippage_bps": float(to_float((selected_row.get("execution_hint", {}) or {}).get("max_slippage_bps", 6.0), 6.0)),
            "entry_price": float(to_float(selected_row.get("entry_price", 0.0), 0.0)),
            "stop_price": float(to_float(selected_row.get("stop_price", 0.0), 0.0)),
            "target_price": float(to_float(selected_row.get("target_price", 0.0), 0.0)),
            "risk": dict(selected_row.get("risk", {}) if isinstance(selected_row.get("risk", {}), dict) else {}),
            "signal_date": str(selected_row.get("signal_date", "")),
            "decision_ttl_seconds": int(max(30, decision_ttl_seconds)),
            "idempotency_key": hashlib.sha256(seed.encode("utf-8")).hexdigest()[:28],
            "score": float(to_float(selected_row.get("score", 0.0), 0.0)),
        }

    return {
        "generated_at_utc": now_utc_iso(),
        "as_of": as_of,
        "skill_id": SKILL_ID,
        "input_artifact": str(tickets_payload.get("_input_artifact", "")),
        "symbols": list(symbols),
        "sizing_context": {
            "equity_usdt": float(equity_usdt),
            "equity_source": equity_source,
            "quote_usdt": float(quote_usdt),
        },
        "policy": {
            "reserve_pct": float(reserve_pct),
            "alloc_pct": float(alloc_pct),
            "min_notional_usdt": float(min_notional_usdt),
            "max_age_days": int(max_age_days),
            "min_confidence": float(min_confidence),
            "min_convexity": float(min_convexity),
            "decision_ttl_seconds": int(max(30, decision_ttl_seconds)),
        },
        "rows": rows_sorted,
        "selected": selected,
        "summary": {
            "candidate_count": int(len(rows_sorted)),
            "executable_count": int(sum(1 for r in rows_sorted if bool(r.get("executable", False)))),
            "hedge_only_count": int(sum(1 for r in rows_sorted if str(r.get("action", "")) == "HEDGE_ONLY")),
            "skip_count": int(sum(1 for r in rows_sorted if str(r.get("action", "")) == "SKIP")),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Fast Trade Skill Plan")
    lines.append("")
    lines.append(f"- generated_at_utc: `{payload.get('generated_at_utc', '')}`")
    lines.append(f"- as_of: `{payload.get('as_of', '')}`")
    lines.append(f"- skill_id: `{payload.get('skill_id', '')}`")
    lines.append(f"- input_artifact: `{payload.get('input_artifact', '')}`")
    lines.append("")
    selected = payload.get("selected", {}) if isinstance(payload.get("selected", {}), dict) else {}
    lines.append("## Selected")
    lines.append(f"- executable: `{bool(selected.get('executable', False))}`")
    if bool(selected.get("executable", False)):
        lines.append(f"- symbol: `{selected.get('symbol', '')}`")
        lines.append(f"- side: `{selected.get('side', '')}`")
        lines.append(f"- quote_usdt: `{to_float(selected.get('quote_usdt', 0.0), 0.0):.4f}`")
        lines.append(f"- idempotency_key: `{selected.get('idempotency_key', '')}`")
        risk = selected.get("risk", {}) if isinstance(selected.get("risk", {}), dict) else {}
        lines.append(f"- rr_ratio: `{to_float(risk.get('rr_ratio', 0.0), 0.0):.4f}`")
        lines.append(f"- stop_gap_pct: `{to_float(risk.get('stop_gap_pct', 0.0), 0.0):.4f}`")
        lines.append(f"- target_gap_pct: `{to_float(risk.get('target_gap_pct', 0.0), 0.0):.4f}`")
    else:
        lines.append(f"- reason: `{selected.get('reason', 'unknown')}`")
    lines.append("")
    policy = payload.get("policy", {}) if isinstance(payload.get("policy", {}), dict) else {}
    shortline = policy.get("shortline", {}) if isinstance(policy.get("shortline", {}), dict) else {}
    lines.append("## Policy")
    lines.append(f"- min_confidence: `{to_float(policy.get('min_confidence', 0.0), 0.0):.2f}`")
    lines.append(f"- min_convexity: `{to_float(policy.get('min_convexity', 0.0), 0.0):.2f}`")
    lines.append(f"- max_age_days: `{to_int(policy.get('max_age_days', 0), 0)}`")
    if shortline:
        lines.append(
            "- shortline: `{structure}` -> `{execution}` | profile=`{profile}` | flow=`{flow}`".format(
                structure=str(shortline.get("structure_timeframe", "")),
                execution=str(shortline.get("execution_timeframe", "")),
                profile=str(shortline.get("structure_engine", "")),
                flow=str(shortline.get("flow_confirmation_engine", "")),
            )
        )
        lines.append(
            f"- shortline_location_priority: `{', '.join(str(x) for x in shortline.get('location_priority', [])) or '-'}`"
        )
        lines.append(
            "- shortline_market_state: `{bias}` -> `{setup}` | no_trade=`{rule}`".format(
                bias=str(shortline.get("default_market_state", "")),
                setup=str(shortline.get("setup_ready_state", "")),
                rule=str(shortline.get("no_trade_rule", "")),
            )
        )
        lines.append(
            "- shortline_micro_trigger: `{engine}` | tfs=`{tfs}` | sweep=`{sweep}` | mss=`{mss}` | retest=`{retest}`".format(
                engine=str(shortline.get("micro_structure_engine", "")),
                tfs=", ".join(str(x) for x in shortline.get("micro_structure_timeframes", [])) or "-",
                sweep=str(bool(shortline.get("liquidity_sweep_required", False))).lower(),
                mss=str(bool(shortline.get("mss_required", False))).lower(),
                retest=", ".join(str(x) for x in shortline.get("entry_retest_priority", [])) or "-",
            )
        )
        lines.append(
            "- shortline_session_map: `{sessions}` | hold=`{hold_min}-{hold_max}m` | style=`{style}`".format(
                sessions=", ".join(str(x) for x in shortline.get("session_liquidity_map", [])) or "-",
                hold_min=str(((shortline.get("holding_window_minutes", {}) or {}).get("min", "-"))),
                hold_max=str(((shortline.get("holding_window_minutes", {}) or {}).get("max", "-"))),
                style=str(shortline.get("execution_style", "")),
            )
        )
        lines.append(
            f"- shortline_supported_symbols: `{', '.join(str(x) for x in shortline.get('supported_symbols', [])) or '-'}`"
        )
        lines.append(
            f"- shortline_caution_symbols: `{', '.join(str(x) for x in shortline.get('caution_symbols', [])) or '-'}`"
        )
    lines.append("")
    lines.append("## Candidates")
    lines.append("| symbol | signal_date | side | conf | convexity | action | executable | quote | rr | reason |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---|")
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        risk = row.get("risk", {}) if isinstance(row.get("risk", {}), dict) else {}
        lines.append(
            "| {symbol} | {signal_date} | {side} | {conf:.2f} | {conv:.2f} | {action} | {exe} | {quote:.4f} | {rr} | {reason} |".format(
                symbol=str(row.get("symbol", "")),
                signal_date=str(row.get("signal_date", "")),
                side=str(row.get("side", "")),
                conf=to_float(row.get("confidence", 0.0), 0.0),
                conv=to_float(row.get("convexity_ratio", 0.0), 0.0),
                action=str(row.get("action", "")),
                exe="yes" if bool(row.get("executable", False)) else "no",
                quote=to_float(row.get("quote_usdt", 0.0), 0.0),
                rr=("-" if risk.get("rr_ratio") is None else f"{to_float(risk.get('rr_ratio', 0.0), 0.0):.2f}"),
                reason=str(row.get("action_reason", "")),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    system_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build fast order+risk plan from latest signal tickets.")
    parser.add_argument("--review-dir", default="output/review")
    parser.add_argument("--tickets-json", default="")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--equity-usdt", type=float, default=0.0)
    parser.add_argument("--reserve-pct", type=float, default=0.45)
    parser.add_argument("--alloc-pct", type=float, default=0.30)
    parser.add_argument("--min-notional-usdt", type=float, default=5.0)
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-convexity", type=float, default=None)
    parser.add_argument("--decision-ttl-seconds", type=int, default=300)
    parser.add_argument("--output-dir", default="output/review")
    args = parser.parse_args()

    review_dir = resolve_path(args.review_dir, anchor=system_root)
    config_path = resolve_path(args.config, anchor=system_root)
    tickets_path = resolve_path(args.tickets_json, anchor=system_root) if str(args.tickets_json).strip() else pick_latest_tickets(review_dir)
    payload_raw = read_json(tickets_path)
    if not isinstance(payload_raw, dict):
        raise ValueError("tickets payload must be a JSON object")
    payload_raw["_input_artifact"] = str(tickets_path)
    policy_thresholds = load_policy_thresholds(config_path)
    shortline_policy = load_shortline_policy(config_path)
    min_confidence = (
        float(args.min_confidence)
        if args.min_confidence is not None
        else float(policy_thresholds.get("signal_confidence_min", DEFAULT_MIN_CONFIDENCE))
    )
    min_convexity = (
        float(args.min_convexity)
        if args.min_convexity is not None
        else float(policy_thresholds.get("convexity_min", DEFAULT_MIN_CONVEXITY))
    )

    plan = build_fast_plan(
        tickets_payload=payload_raw,
        symbols=normalize_symbols(args.symbols),
        equity_usdt_override=(float(args.equity_usdt) if float(args.equity_usdt) > 0.0 else None),
        reserve_pct=float(args.reserve_pct),
        alloc_pct=float(args.alloc_pct),
        min_notional_usdt=max(0.1, float(args.min_notional_usdt)),
        max_age_days=max(1, int(args.max_age_days)),
        min_confidence=min_confidence,
        min_convexity=min_convexity,
        decision_ttl_seconds=max(30, int(args.decision_ttl_seconds)),
    )
    policy = plan.get("policy", {}) if isinstance(plan.get("policy", {}), dict) else {}
    policy["config_path"] = str(config_path)
    policy["threshold_source"] = str(policy_thresholds.get("source", "builtin_default"))
    policy["min_confidence_source"] = "cli_override" if args.min_confidence is not None else str(policy_thresholds.get("source", "builtin_default"))
    policy["min_convexity_source"] = "cli_override" if args.min_convexity is not None else str(policy_thresholds.get("source", "builtin_default"))
    policy["shortline"] = shortline_policy
    plan["policy"] = policy

    out_dir = resolve_path(args.output_dir, anchor=system_root)
    stamp = now_utc_compact()
    out_json = out_dir / f"{stamp}_fast_trade_skill_plan.json"
    out_md = out_dir / f"{stamp}_fast_trade_skill_plan.md"
    write_json(out_json, plan)
    write_text(out_md, render_markdown(plan))
    print(
        json.dumps(
            {
                "json": str(out_json),
                "md": str(out_md),
                "selected": plan.get("selected", {}),
                "summary": plan.get("summary", {}),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
