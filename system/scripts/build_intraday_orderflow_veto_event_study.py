#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_MICRO_CAPTURE_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "micro_capture"
PRICE_ACTION_SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "build_price_action_breakout_pullback_sim_only.py"
DEFAULT_SCENARIO = {
    "scenario_id": "moderate_costs",
    "label": "中性成本",
    "fee_bps_per_side": 5.0,
    "slippage_bps_per_side": 3.0,
}


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def parse_utc(raw: str) -> dt.datetime | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        parsed = dt.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def latest_artifact(review_dir: Path, pattern: str) -> Path:
    candidates = sorted(review_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"missing_required_artifact:{pattern}")
    return candidates[-1]


def load_price_action_module() -> Any:
    spec = importlib.util.spec_from_file_location("fenlie_price_action", PRICE_ACTION_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_hours_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in text(raw).split(","):
        if not part.strip():
            continue
        value = to_float(part, 0.0)
        if value > 0:
            values.append(value)
    return sorted(set(values))


def apply_costs_full(price_action_module: Any, trades: list[dict[str, Any]], scenario: dict[str, Any]) -> list[dict[str, Any]]:
    effective: list[dict[str, Any]] = []
    for trade in trades:
        result = price_action_module.apply_cost_scenario({"trades": [trade], "metrics": {}}, scenario)
        sample = list(result.get("trade_sample") or [])
        if sample:
            effective.append(sample[0])
    return effective


def load_micro_rows(micro_capture_dir: Path, symbol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = symbol.upper()
    for path in sorted(micro_capture_dir.glob("*_micro_capture.json")):
        payload = load_json_mapping(path)
        captured_at = parse_utc(payload.get("captured_at_utc") or payload.get("generated_at_utc"))
        for raw in payload.get("selected_micro") or []:
            if not isinstance(raw, dict):
                continue
            if text(raw.get("symbol")).upper() != target:
                continue
            row = dict(raw)
            row["_captured_at_utc"] = fmt_utc(captured_at)
            row["_artifact_path"] = str(path)
            rows.append(row)
    return rows


def align_trades_to_micro(
    *,
    trades: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
    coverage_windows_hours: list[float],
) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    indexed_micro = []
    for row in micro_rows:
        captured_at = parse_utc(row.get("_captured_at_utc"))
        if captured_at is None:
            continue
        indexed_micro.append((captured_at, row))
    for trade in trades:
        entry_ts = parse_utc(trade.get("entry_ts_utc"))
        nearest_ts: dt.datetime | None = None
        nearest_row: dict[str, Any] | None = None
        delta_hours: float | None = None
        if entry_ts is not None and indexed_micro:
            nearest_ts, nearest_row = min(indexed_micro, key=lambda item: abs((item[0] - entry_ts).total_seconds()))
            delta_hours = abs((nearest_ts - entry_ts).total_seconds()) / 3600.0
        aligned.append(
            {
                "signal_ts_utc": text(trade.get("signal_ts_utc")),
                "entry_ts_utc": text(trade.get("entry_ts_utc")),
                "exit_ts_utc": text(trade.get("exit_ts_utc")),
                "exit_reason": text(trade.get("exit_reason")),
                "bars_held": to_int(trade.get("bars_held"), 0),
                "net_r_multiple": to_float(trade.get("net_r_multiple"), to_float(trade.get("r_multiple"), 0.0)),
                "net_pnl_pct": to_float(trade.get("net_pnl_pct"), to_float(trade.get("pnl_pct"), 0.0)),
                "nearest_capture_ts_utc": fmt_utc(nearest_ts),
                "nearest_capture_delta_hours": float(round(delta_hours, 6)) if delta_hours is not None else None,
                "within_window": {
                    f"{int(window) if float(window).is_integer() else window}h": bool(
                        delta_hours is not None and delta_hours <= window
                    )
                    for window in coverage_windows_hours
                },
                "matched_context_mode": text((nearest_row or {}).get("cvd_context_mode")),
                "matched_context_note": text((nearest_row or {}).get("cvd_context_note")),
                "matched_trust_tier": text((nearest_row or {}).get("cvd_trust_tier_hint")),
                "matched_veto_hint": text((nearest_row or {}).get("cvd_veto_hint")),
                "matched_queue_imbalance": to_float((nearest_row or {}).get("queue_imbalance"), 0.0),
                "matched_ofi_norm": to_float((nearest_row or {}).get("ofi_norm"), 0.0),
                "matched_micro_alignment": to_float((nearest_row or {}).get("micro_alignment"), 0.0),
                "matched_trade_count": to_int((nearest_row or {}).get("trade_count"), 0),
                "matched_artifact_path": text((nearest_row or {}).get("_artifact_path")),
            }
        )
    return aligned


def summarize_replay(
    *,
    symbol: str,
    role: str,
    aligned_trades: list[dict[str, Any]],
    coverage_windows_hours: list[float],
) -> dict[str, Any]:
    trade_count = len(aligned_trades)
    exit_reason_counts = Counter(text(row.get("exit_reason")) for row in aligned_trades if text(row.get("exit_reason")))
    coverage_by_window = {
        f"within_{int(window) if float(window).is_integer() else window}h": {
            "count": int(
                sum(1 for row in aligned_trades if bool((row.get("within_window") or {}).get(f"{int(window) if float(window).is_integer() else window}h")))
            ),
            "ratio": float(
                safe_ratio(
                    sum(
                        1
                        for row in aligned_trades
                        if bool((row.get("within_window") or {}).get(f"{int(window) if float(window).is_integer() else window}h"))
                    ),
                    trade_count,
                )
            ),
        }
        for window in coverage_windows_hours
    }
    context_available_48h = [
        row
        for row in aligned_trades
        if bool((row.get("within_window") or {}).get("48h")) and text(row.get("matched_context_mode"))
    ]
    cumulative_return = 1.0
    for row in aligned_trades:
        cumulative_return *= 1.0 + to_float(row.get("net_pnl_pct"), 0.0)
    if symbol == "ETHUSDT":
        transfer_status = "mainline_template_active"
    elif trade_count <= 0:
        transfer_status = "no_trade_under_eth_template"
    else:
        transfer_status = "trade_generated_under_eth_template"
    return {
        "symbol": symbol,
        "role": role,
        "trade_count": trade_count,
        "cumulative_return": float(cumulative_return - 1.0) if trade_count else 0.0,
        "avg_net_r_multiple": float(
            sum(to_float(row.get("net_r_multiple"), 0.0) for row in aligned_trades) / trade_count
        )
        if trade_count
        else 0.0,
        "exit_reason_counts": dict(exit_reason_counts),
        "coverage_by_window": coverage_by_window,
        "context_available_within_48h_count": int(len(context_available_48h)),
        "transfer_status": transfer_status,
    }


def classify_rule(*, veto: list[dict[str, Any]], keep: list[dict[str, Any]]) -> str:
    if not veto and not keep:
        return "no_usable_coverage"
    if keep and not veto:
        return "no_effect_on_current_sample"
    if veto and not keep:
        return "overblocking_current_sample"
    veto_positive = sum(1 for row in veto if to_float(row.get("net_r_multiple"), 0.0) > 0.0)
    veto_negative = sum(1 for row in veto if to_float(row.get("net_r_multiple"), 0.0) < 0.0)
    keep_positive = sum(1 for row in keep if to_float(row.get("net_r_multiple"), 0.0) > 0.0)
    keep_negative = sum(1 for row in keep if to_float(row.get("net_r_multiple"), 0.0) < 0.0)
    keep_avg = sum(to_float(row.get("net_r_multiple"), 0.0) for row in keep) / len(keep) if keep else 0.0
    veto_avg = sum(to_float(row.get("net_r_multiple"), 0.0) for row in veto) / len(veto) if veto else 0.0
    if veto_negative > 0 and veto_positive == 0 and keep_avg >= 0.0:
        return "tentatively_helpful_small_sample"
    if veto_positive >= veto_negative or veto_avg > keep_avg:
        return "harmful_on_current_sample"
    if keep_negative > 0 and veto_negative == 0:
        return "not_helpful_on_current_sample"
    return "mixed_small_sample"


def evaluate_rule(
    *,
    name: str,
    description: str,
    covered_events: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    veto = [row for row in covered_events if predicate(row)]
    keep = [row for row in covered_events if not predicate(row)]
    veto_positive = sum(1 for row in veto if to_float(row.get("net_r_multiple"), 0.0) > 0.0)
    veto_negative = sum(1 for row in veto if to_float(row.get("net_r_multiple"), 0.0) < 0.0)
    keep_positive = sum(1 for row in keep if to_float(row.get("net_r_multiple"), 0.0) > 0.0)
    keep_negative = sum(1 for row in keep if to_float(row.get("net_r_multiple"), 0.0) < 0.0)
    return {
        "rule_name": name,
        "description": description,
        "covered_trade_count": len(covered_events),
        "veto_trade_count": len(veto),
        "keep_trade_count": len(keep),
        "veto_positive_count": veto_positive,
        "veto_negative_count": veto_negative,
        "keep_positive_count": keep_positive,
        "keep_negative_count": keep_negative,
        "veto_avg_net_r_multiple": float(
            sum(to_float(row.get("net_r_multiple"), 0.0) for row in veto) / len(veto)
        )
        if veto
        else 0.0,
        "keep_avg_net_r_multiple": float(
            sum(to_float(row.get("net_r_multiple"), 0.0) for row in keep) / len(keep)
        )
        if keep
        else 0.0,
        "classification": classify_rule(veto=veto, keep=keep),
        "veto_entries": [
            {
                "entry_ts_utc": text(row.get("entry_ts_utc")),
                "net_r_multiple": to_float(row.get("net_r_multiple"), 0.0),
                "exit_reason": text(row.get("exit_reason")),
                "matched_context_mode": text(row.get("matched_context_mode")),
                "matched_trust_tier": text(row.get("matched_trust_tier")),
                "matched_veto_hint": text(row.get("matched_veto_hint")),
            }
            for row in veto
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Intraday Orderflow Veto Event Study",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Fixed ETH Template Replay",
        "",
    ]
    for row in payload.get("symbol_replay_rows", []):
        lines.append(
            f"- `{row['symbol']}` | role=`{row['role']}` | transfer_status=`{row['transfer_status']}` | "
            f"trades=`{row['trade_count']}` | cum_return=`{float(row['cumulative_return']):.2%}` | "
            f"coverage24=`{float(((row.get('coverage_by_window') or {}).get('within_24h') or {}).get('ratio', 0.0)):.2%}` | "
            f"coverage48=`{float(((row.get('coverage_by_window') or {}).get('within_48h') or {}).get('ratio', 0.0)):.2%}`"
        )
    lines.extend(["", "## ETH 48h Context-Covered Event Rules", ""])
    for row in payload.get("eth_veto_rules_48h", []):
        lines.append(
            f"- `{row['rule_name']}` | classification=`{row['classification']}` | veto=`{row['veto_trade_count']}` | "
            f"keep=`{row['keep_trade_count']}` | veto_avg_r=`{float(row['veto_avg_net_r_multiple']):.3f}` | "
            f"keep_avg_r=`{float(row['keep_avg_net_r_multiple']):.3f}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- `{text(payload.get('research_note'))}`",
            f"- `{text(payload.get('limitation_note'))}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a RESEARCH_ONLY fixed-template replay and orderflow veto event study before any parameter fitting."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--micro-capture-dir", default=str(DEFAULT_MICRO_CAPTURE_DIR))
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--price-action-path", default="")
    parser.add_argument("--blueprint-path", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--coverage-hours", default="24,48")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    micro_capture_dir = Path(args.micro_capture_dir).expanduser().resolve()
    price_action_path = Path(args.price_action_path).expanduser().resolve() if text(args.price_action_path) else latest_artifact(review_dir, "*_price_action_breakout_pullback_sim_only.json")
    blueprint_path = Path(args.blueprint_path).expanduser().resolve() if text(args.blueprint_path) else latest_artifact(review_dir, "*_intraday_orderflow_strategy_blueprint.json")

    price_action = load_json_mapping(price_action_path)
    blueprint = load_json_mapping(blueprint_path)
    price_action_module = load_price_action_module()

    dataset_path = Path(args.dataset_path).expanduser().resolve() if text(args.dataset_path) else Path(text(price_action.get("dataset_path"))).expanduser().resolve()
    frame = price_action_module.add_features(price_action_module.load_frame(dataset_path))

    symbol_input = [item.strip().upper() for item in text(args.symbols).split(",") if item.strip()]
    if symbol_input:
        target_symbols = symbol_input
    else:
        target_symbols = [
            text(row.get("symbol")).upper()
            for row in blueprint.get("symbol_depth_ladder", [])
            if isinstance(row, dict) and text(row.get("symbol"))
        ]
    ladder_map = {
        text(row.get("symbol")).upper(): row
        for row in blueprint.get("symbol_depth_ladder", [])
        if isinstance(row, dict) and text(row.get("symbol"))
    }

    selected_params = dict((price_action.get("focus_symbol_result") or {}).get("selected_params") or price_action.get("selected_params") or {})
    coverage_hours = parse_hours_list(args.coverage_hours) or [24.0, 48.0]
    if 48.0 not in coverage_hours:
        coverage_hours = sorted(set([*coverage_hours, 48.0]))
    scenario = dict(DEFAULT_SCENARIO)

    symbol_replay_rows: list[dict[str, Any]] = []
    eth_aligned_trades: list[dict[str, Any]] = []
    for symbol in target_symbols:
        symbol_frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
        role = text((ladder_map.get(symbol) or {}).get("role"))
        if symbol_frame.empty:
            symbol_replay_rows.append(
                {
                    "symbol": symbol,
                    "role": role,
                    "trade_count": 0,
                    "cumulative_return": 0.0,
                    "avg_net_r_multiple": 0.0,
                    "exit_reason_counts": {},
                    "coverage_by_window": {},
                    "context_available_within_48h_count": 0,
                    "transfer_status": "missing_symbol_data",
                }
            )
            continue
        replay = price_action_module.simulate_symbol(symbol_frame, selected_params)
        full_trades = apply_costs_full(price_action_module, list(replay.get("trades") or []), scenario)
        aligned = align_trades_to_micro(
            trades=full_trades,
            micro_rows=load_micro_rows(micro_capture_dir, symbol),
            coverage_windows_hours=coverage_hours,
        )
        symbol_replay_rows.append(
            summarize_replay(
                symbol=symbol,
                role=role,
                aligned_trades=aligned,
                coverage_windows_hours=coverage_hours,
            )
        )
        if symbol == "ETHUSDT":
            eth_aligned_trades = aligned

    eth_context_covered_48h = [
        row
        for row in eth_aligned_trades
        if bool((row.get("within_window") or {}).get("48h")) and text(row.get("matched_context_mode"))
    ]
    eth_veto_rules_48h = [
        evaluate_rule(
            name="any_veto_hint",
            description="只要最近 48h 内出现非空 cvd_veto_hint 就 veto。",
            covered_events=eth_context_covered_48h,
            predicate=lambda row: bool(text(row.get("matched_veto_hint"))),
        ),
        evaluate_rule(
            name="non_continuation_context",
            description="reversal / absorption / failed_auction 一律视作 hard veto。",
            covered_events=eth_context_covered_48h,
            predicate=lambda row: text(row.get("matched_context_mode")) in {"reversal", "absorption", "failed_auction"},
        ),
        evaluate_rule(
            name="effort_result_only",
            description="只把 absorption / failed_auction 视作 hard veto。",
            covered_events=eth_context_covered_48h,
            predicate=lambda row: text(row.get("matched_context_mode")) in {"absorption", "failed_auction"},
        ),
        evaluate_rule(
            name="low_trust_only",
            description="只要 trust 仍是 single_exchange_low / cross_exchange_conflicted 就 veto。",
            covered_events=eth_context_covered_48h,
            predicate=lambda row: text(row.get("matched_trust_tier")) in {"single_exchange_low", "cross_exchange_conflicted"},
        ),
    ]

    harmful_rules = [row["rule_name"] for row in eth_veto_rules_48h if row["classification"] in {"harmful_on_current_sample", "overblocking_current_sample"}]
    non_eth_trade_count = sum(row.get("trade_count", 0) for row in symbol_replay_rows if row["symbol"] != "ETHUSDT")
    research_decision = (
        "no_hard_veto_prototype_survives_current_eth_event_sample_keep_sampling"
        if harmful_rules
        else "event_sample_supports_next_eth_veto_replay"
    )
    recommended_brief = (
        "orderflow_veto_event_study:"
        f"eth_trades={next((row['trade_count'] for row in symbol_replay_rows if row['symbol']=='ETHUSDT'), 0)},"
        f"eth_ctx48={len(eth_context_covered_48h)},"
        f"non_eth_trades={non_eth_trade_count},"
        f"harmful_rules={','.join(harmful_rules) if harmful_rules else '-'},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_intraday_orderflow_veto_event_study",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "dataset_path": str(dataset_path),
            "price_action_path": str(price_action_path),
            "blueprint_path": str(blueprint_path),
            "micro_capture_dir": str(micro_capture_dir),
        },
        "fixed_template": {
            "template_symbol": text(price_action.get("focus_symbol")) or "ETHUSDT",
            "family": "price_action_breakout_pullback",
            "selected_params": selected_params,
            "scenario": scenario,
        },
        "symbol_replay_rows": symbol_replay_rows,
        "eth_aligned_events": eth_aligned_trades,
        "eth_context_covered_48h_count": int(len(eth_context_covered_48h)),
        "eth_veto_rules_48h": eth_veto_rules_48h,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份事件研究把 ETH 当前固定主线模板直接回放到 symbol-depth ladder，"
            "并用本地 micro_capture 做最近邻对齐，先验证最朴素的 hard veto 原型是否成立。"
        ),
        "limitation_note": (
            "当前研究仍受限于 micro_capture 的稀疏覆盖和 single_exchange_low trust；"
            "因此结论只能用于排除坏原型，不能直接作为回测或拟合输入。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_veto_event_study.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_veto_event_study.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_veto_event_study.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "latest_json_path": str(latest_json_path),
                "research_decision": research_decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
