#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XAUUSD")
DEFAULT_CONFIG_PATH = "config.yaml"
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


def normalize_symbol_list(raw: Any, *, default: tuple[str, ...]) -> list[str]:
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip().upper() for x in raw]
    else:
        items = [str(x).strip().upper() for x in str(raw).split(",")]
    out = [item for item in items if item]
    return out if out else list(default)


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
        "profile_lookback_bars": int(to_int(shortline.get("profile_lookback_bars", 120), 120)),
        "profile_value_area_pct": float(to_float(shortline.get("profile_value_area_pct", 0.70), 0.70)),
        "location_priority": normalize_symbol_list(shortline.get("location_priority", ("HVN", "POC")), default=("HVN", "POC")),
        "avoid_location": str(shortline.get("avoid_location", "LVN")),
        "flow_confirmation_engine": str(shortline.get("flow_confirmation_engine", "cvd_lite")),
        "flow_confirmation_role": str(shortline.get("flow_confirmation_role", "confirm_and_veto_only")),
        "flow_confirmation_half_life_seconds": int(
            to_int(shortline.get("flow_confirmation_half_life_seconds", 180), 180)
        ),
        "trigger_stack": [str(x).strip() for x in shortline.get("trigger_stack", []) if str(x).strip()]
        or ["4h_profile_location", "15m_cvd_divergence_or_confirmation", "15m_reversal_or_breakout_candle"],
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


def build_brief(
    *,
    tickets_payload: dict[str, Any],
    equity_usdt_override: float | None,
    reserve_pct: float,
    alloc_pct: float,
    min_notional_usdt: float,
    max_age_days: int,
) -> dict[str, Any]:
    sizing_context = tickets_payload.get("sizing_context", {}) if isinstance(tickets_payload, dict) else {}
    equity_usdt = to_float(sizing_context.get("equity_usdt", 0.0), 0.0)
    equity_source = str(sizing_context.get("equity_source", "unknown"))
    if equity_usdt_override is not None and equity_usdt_override > 0.0:
        equity_usdt = float(equity_usdt_override)
        equity_source = "cli_override"

    reserve_quote = max(0.0, equity_usdt * max(0.0, min(0.95, reserve_pct)))
    usable_quote = max(0.0, equity_usdt - reserve_quote)
    cap_by_alloc = max(0.0, equity_usdt * max(0.01, min(0.95, alloc_pct)))
    canary_quote = min(usable_quote, cap_by_alloc)
    if canary_quote > 0.0:
        canary_quote = max(min_notional_usdt, canary_quote)
    canary_quote = round(canary_quote, 4)

    rows: list[dict[str, Any]] = []
    for raw in (tickets_payload.get("tickets", []) if isinstance(tickets_payload, dict) else []):
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol", "")).upper()
        if symbol not in DEFAULT_SYMBOLS:
            continue
        reasons = [str(x) for x in raw.get("reasons", []) if str(x)]
        signal = raw.get("signal", {}) if isinstance(raw.get("signal", {}), dict) else {}
        side = str(signal.get("side", ""))
        confidence = to_float(signal.get("confidence", 0.0), 0.0)
        convexity = to_float(signal.get("convexity_ratio", 0.0), 0.0)
        age_days = to_int(raw.get("age_days", -1), -1)
        stale = ("stale_signal" in reasons) or (age_days >= 0 and age_days > max_age_days)
        not_found = "signal_not_found" in reasons
        only_size_blocked = bool(reasons) and all(x == "size_below_min_notional" for x in reasons)

        micro_tradable = False
        action = "SKIP"
        action_reason = ",".join(reasons) if reasons else "no_action_rule"
        if not not_found and not stale:
            if side == "LONG" and confidence >= 14.0 and convexity >= 1.2 and canary_quote >= min_notional_usdt:
                # For tiny accounts, allow micro-canary when the only blocker is min notional sizing.
                if bool(raw.get("allowed", False)) or only_size_blocked:
                    micro_tradable = True
                    action = "CANARY_BUY"
                    action_reason = "micro_override_size_floor" if only_size_blocked else "base_allowed"
            elif side == "SHORT":
                action = "HEDGE_ONLY"
                action_reason = "spot_account_no_direct_short"

        rows.append(
            {
                "symbol": symbol,
                "signal_date": str(raw.get("date", "")),
                "age_days": age_days,
                "side": side,
                "regime": str(signal.get("regime", "")),
                "confidence": confidence,
                "convexity_ratio": convexity,
                "base_allowed": bool(raw.get("allowed", False)),
                "reasons": reasons,
                "micro_tradable": micro_tradable,
                "action": action,
                "action_reason": action_reason,
                "recommended_quote_usdt": canary_quote if micro_tradable else 0.0,
                "entry_price": to_float((raw.get("levels", {}) if isinstance(raw.get("levels", {}), dict) else {}).get("entry_price", 0.0), 0.0),
                "stop_price": to_float((raw.get("levels", {}) if isinstance(raw.get("levels", {}), dict) else {}).get("stop_price", 0.0), 0.0),
                "target_price": to_float((raw.get("levels", {}) if isinstance(raw.get("levels", {}), dict) else {}).get("target_price", 0.0), 0.0),
            }
        )

    rows_sorted = sorted(rows, key=lambda x: (0 if x["micro_tradable"] else 1, x["symbol"]))
    return {
        "generated_at_utc": now_utc_iso(),
        "as_of": str(tickets_payload.get("as_of", "")),
        "symbols": list(DEFAULT_SYMBOLS),
        "input_artifact": str(tickets_payload.get("_input_artifact", "")),
        "equity_usdt": float(equity_usdt),
        "equity_source": equity_source,
        "policy": {
            "reserve_pct": float(reserve_pct),
            "alloc_pct": float(alloc_pct),
            "min_notional_usdt": float(min_notional_usdt),
            "max_age_days": int(max_age_days),
            "derived": {
                "reserve_quote_usdt": float(round(reserve_quote, 4)),
                "usable_quote_usdt": float(round(usable_quote, 4)),
                "canary_quote_usdt": float(canary_quote),
            },
        },
        "rows": rows_sorted,
        "summary": {
            "micro_tradable_count": int(sum(1 for r in rows_sorted if bool(r.get("micro_tradable", False)))),
            "hedge_only_count": int(sum(1 for r in rows_sorted if str(r.get("action", "")) == "HEDGE_ONLY")),
            "skip_count": int(sum(1 for r in rows_sorted if str(r.get("action", "")) == "SKIP")),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Micro Signal Brief")
    lines.append("")
    lines.append(f"- generated_at_utc: `{payload.get('generated_at_utc', '')}`")
    lines.append(f"- as_of: `{payload.get('as_of', '')}`")
    lines.append(f"- input_artifact: `{payload.get('input_artifact', '')}`")
    lines.append(f"- equity_usdt: `{to_float(payload.get('equity_usdt', 0.0), 0.0):.4f}`")
    lines.append("")
    pol = payload.get("policy", {}) if isinstance(payload.get("policy", {}), dict) else {}
    der = pol.get("derived", {}) if isinstance(pol.get("derived", {}), dict) else {}
    lines.append("## Policy")
    lines.append(f"- reserve_pct: `{to_float(pol.get('reserve_pct', 0.0), 0.0):.2%}`")
    lines.append(f"- alloc_pct: `{to_float(pol.get('alloc_pct', 0.0), 0.0):.2%}`")
    lines.append(f"- min_notional_usdt: `{to_float(pol.get('min_notional_usdt', 0.0), 0.0):.2f}`")
    lines.append(f"- max_age_days: `{to_int(pol.get('max_age_days', 0), 0)}`")
    lines.append(f"- canary_quote_usdt: `{to_float(der.get('canary_quote_usdt', 0.0), 0.0):.4f}`")
    shortline = pol.get("shortline", {}) if isinstance(pol.get("shortline", {}), dict) else {}
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
            f"- shortline_supported_symbols: `{', '.join(str(x) for x in shortline.get('supported_symbols', [])) or '-'}`"
        )
    lines.append("")
    lines.append("## Rows")
    lines.append("| symbol | side | age_days | confidence | convexity | action | quote_usdt | reason |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---|")
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {symbol} | {side} | {age} | {conf:.2f} | {conv:.2f} | {action} | {quote:.4f} | {reason} |".format(
                symbol=str(row.get("symbol", "")),
                side=str(row.get("side", "")),
                age=to_int(row.get("age_days", -1), -1),
                conf=to_float(row.get("confidence", 0.0), 0.0),
                conv=to_float(row.get("convexity_ratio", 0.0), 0.0),
                action=str(row.get("action", "")),
                quote=to_float(row.get("recommended_quote_usdt", 0.0), 0.0),
                reason=str(row.get("action_reason", "")),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    system_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build micro-account signal execution brief from latest signal tickets.")
    parser.add_argument("--review-dir", default="output/review", help="Review artifact directory.")
    parser.add_argument("--tickets-json", default="", help="Explicit signal_to_order_tickets json path.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Optional system config yaml.")
    parser.add_argument("--equity-usdt", type=float, default=0.0, help="Optional equity override.")
    parser.add_argument("--reserve-pct", type=float, default=0.45, help="Cash reserve ratio for micro account.")
    parser.add_argument("--alloc-pct", type=float, default=0.30, help="Max single-trade allocation ratio.")
    parser.add_argument("--min-notional-usdt", type=float, default=5.0, help="Exchange minimum notional.")
    parser.add_argument("--max-age-days", type=int, default=14, help="Signal staleness threshold.")
    parser.add_argument("--output-dir", default="output/review", help="Output artifact directory.")
    args = parser.parse_args()

    review_dir = resolve_path(args.review_dir, anchor=system_root)
    config_path = resolve_path(args.config, anchor=system_root)
    tickets_path = resolve_path(args.tickets_json, anchor=system_root) if str(args.tickets_json).strip() else pick_latest_tickets(review_dir)
    payload_raw = read_json(tickets_path)
    if not isinstance(payload_raw, dict):
        raise ValueError("tickets payload must be a JSON object")
    payload_raw["_input_artifact"] = str(tickets_path)

    brief = build_brief(
        tickets_payload=payload_raw,
        equity_usdt_override=(float(args.equity_usdt) if float(args.equity_usdt) > 0.0 else None),
        reserve_pct=float(args.reserve_pct),
        alloc_pct=float(args.alloc_pct),
        min_notional_usdt=max(0.1, float(args.min_notional_usdt)),
        max_age_days=max(1, int(args.max_age_days)),
    )
    policy = brief.get("policy", {}) if isinstance(brief.get("policy", {}), dict) else {}
    policy["config_path"] = str(config_path)
    policy["shortline"] = load_shortline_policy(config_path)
    brief["policy"] = policy

    out_dir = resolve_path(args.output_dir, anchor=system_root)
    stamp = now_utc_compact()
    out_json = out_dir / f"{stamp}_micro_signal_brief.json"
    out_md = out_dir / f"{stamp}_micro_signal_brief.md"
    write_json(out_json, brief)
    write_text(out_md, render_markdown(brief))
    print(json.dumps({"json": str(out_json), "md": str(out_md), "summary": brief.get("summary", {})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
