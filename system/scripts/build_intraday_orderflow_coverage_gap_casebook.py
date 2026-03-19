#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


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
    for path in sorted(micro_capture_dir.glob("*_micro_capture.json")):
        payload = load_json_mapping(path)
        captured_at = parse_utc(payload.get("captured_at_utc") or payload.get("generated_at_utc"))
        for raw in payload.get("selected_micro") or []:
            if not isinstance(raw, dict):
                continue
            if text(raw.get("symbol")).upper() != symbol.upper():
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
                "matched_artifact_path": text((nearest_row or {}).get("_artifact_path")),
            }
        )
    return aligned


def classify_gap_priority(role: str, coverage_24h: float, coverage_48h: float) -> str:
    if role in {"mainline_primary", "majors_anchor"}:
        if coverage_24h < 0.6 or coverage_48h < 0.8:
            return "urgent_major_capture_gap"
        return "major_gap_watch"
    if coverage_48h < 0.4:
        return "beta_watch_low_priority_gap"
    return "beta_gap_watch"


def build_symbol_gap_row(symbol: str, role: str, aligned_events: list[dict[str, Any]]) -> dict[str, Any]:
    trade_count = len(aligned_events)
    coverage_24h_count = sum(1 for row in aligned_events if bool((row.get("within_window") or {}).get("24h")))
    coverage_48h_count = sum(1 for row in aligned_events if bool((row.get("within_window") or {}).get("48h")))
    context_48h = [
        row
        for row in aligned_events
        if bool((row.get("within_window") or {}).get("48h")) and text(row.get("matched_context_mode"))
    ]
    missing_context_48h = [
        row
        for row in aligned_events
        if bool((row.get("within_window") or {}).get("48h")) and not text(row.get("matched_context_mode"))
    ]
    uncovered_24h = [row for row in aligned_events if not bool((row.get("within_window") or {}).get("24h"))]
    uncovered_48h = [row for row in aligned_events if not bool((row.get("within_window") or {}).get("48h"))]
    hour_counts = Counter()
    for row in uncovered_48h if uncovered_48h else uncovered_24h:
        entry_ts = parse_utc(row.get("entry_ts_utc"))
        if entry_ts is not None:
            hour_counts[f"{entry_ts.hour:02d}:00"] += 1
    coverage_24h = safe_ratio(coverage_24h_count, trade_count)
    coverage_48h = safe_ratio(coverage_48h_count, trade_count)
    return {
        "symbol": symbol,
        "role": role,
        "trade_count": trade_count,
        "coverage_24h_count": int(coverage_24h_count),
        "coverage_48h_count": int(coverage_48h_count),
        "coverage_24h_ratio": float(coverage_24h),
        "coverage_48h_ratio": float(coverage_48h),
        "context_available_48h_count": int(len(context_48h)),
        "context_available_48h_ratio": float(safe_ratio(len(context_48h), trade_count)),
        "missing_context_48h_count": int(len(missing_context_48h)),
        "uncovered_24h_count": int(len(uncovered_24h)),
        "uncovered_48h_count": int(len(uncovered_48h)),
        "gap_priority": classify_gap_priority(role, coverage_24h, coverage_48h),
        "top_capture_hours_utc": [
            {"hour_utc": hour, "count": count} for hour, count in hour_counts.most_common(4)
        ],
        "gap_examples": [
            {
                "entry_ts_utc": text(row.get("entry_ts_utc")),
                "exit_reason": text(row.get("exit_reason")),
                "net_r_multiple": to_float(row.get("net_r_multiple"), 0.0),
                "nearest_capture_ts_utc": text(row.get("nearest_capture_ts_utc")),
                "nearest_capture_delta_hours": to_float(row.get("nearest_capture_delta_hours"), 0.0),
                "matched_context_mode": text(row.get("matched_context_mode")),
            }
            for row in (uncovered_48h[:3] if uncovered_48h else uncovered_24h[:3])
        ],
    }


def classify_case(row: dict[str, Any]) -> tuple[str, str]:
    exit_reason = text(row.get("exit_reason"))
    net_r = to_float(row.get("net_r_multiple"), 0.0)
    within_48h = bool((row.get("within_window") or {}).get("48h"))
    context = text(row.get("matched_context_mode"))
    trust = text(row.get("matched_trust_tier"))
    if net_r <= 0.0:
        severity = "hard_negative"
    else:
        severity = "soft_time_exit"
    if not within_48h:
        diagnosis = "capture_gap_blocks_inference"
    elif not context:
        diagnosis = "capture_present_but_context_missing"
    elif context == "continuation" and trust == "single_exchange_low":
        diagnosis = "covered_but_low_trust_continuation"
    elif context in {"reversal", "absorption", "failed_auction"} and trust == "single_exchange_low":
        diagnosis = "covered_non_continuation_but_low_trust"
    else:
        diagnosis = "covered_context_requires_manual_review"
    return severity, diagnosis


def build_eth_bad_trade_casebook(aligned_events: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    diagnosis_counts = Counter()
    severity_counts = Counter()
    for row in aligned_events:
        exit_reason = text(row.get("exit_reason"))
        net_r = to_float(row.get("net_r_multiple"), 0.0)
        if not (net_r <= 0.0 or (exit_reason == "time_exit" and net_r < 0.2)):
            continue
        severity, diagnosis = classify_case(row)
        severity_counts[severity] += 1
        diagnosis_counts[diagnosis] += 1
        cases.append(
            {
                "entry_ts_utc": text(row.get("entry_ts_utc")),
                "exit_ts_utc": text(row.get("exit_ts_utc")),
                "exit_reason": exit_reason,
                "net_r_multiple": net_r,
                "severity": severity,
                "diagnosis": diagnosis,
                "nearest_capture_ts_utc": text(row.get("nearest_capture_ts_utc")),
                "nearest_capture_delta_hours": to_float(row.get("nearest_capture_delta_hours"), 0.0),
                "matched_context_mode": text(row.get("matched_context_mode")),
                "matched_context_note": text(row.get("matched_context_note")),
                "matched_trust_tier": text(row.get("matched_trust_tier")),
                "matched_veto_hint": text(row.get("matched_veto_hint")),
                "matched_queue_imbalance": to_float(row.get("matched_queue_imbalance"), 0.0),
                "matched_ofi_norm": to_float(row.get("matched_ofi_norm"), 0.0),
                "matched_micro_alignment": to_float(row.get("matched_micro_alignment"), 0.0),
            }
        )
    return {
        "rule": "bad trade = net_r_multiple <= 0 OR (time_exit AND net_r_multiple < 0.2)",
        "case_count": int(len(cases)),
        "severity_counts": dict(severity_counts),
        "diagnosis_counts": dict(diagnosis_counts),
        "cases": cases,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Intraday Orderflow Coverage Gap Casebook",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Symbol Gap Summary",
        "",
    ]
    for row in payload.get("symbol_gap_rows", []):
        lines.append(
            f"- `{row['symbol']}` | role=`{row['role']}` | gap_priority=`{row['gap_priority']}` | "
            f"trades=`{row['trade_count']}` | cov24=`{float(row['coverage_24h_ratio']):.2%}` | "
            f"cov48=`{float(row['coverage_48h_ratio']):.2%}` | ctx48=`{float(row['context_available_48h_ratio']):.2%}`"
        )
    lines.extend(["", "## ETH Bad Trade Casebook", ""])
    casebook = payload.get("eth_bad_trade_casebook") or {}
    lines.append(f"- rule: `{text(casebook.get('rule'))}`")
    lines.append(f"- case_count: `{to_int(casebook.get('case_count'), 0)}`")
    lines.append(f"- diagnosis_counts: `{json.dumps(casebook.get('diagnosis_counts') or {}, ensure_ascii=False)}`")
    lines.append("")
    for row in casebook.get("cases", []):
        lines.append(
            f"- `{row['entry_ts_utc']}` | severity=`{row['severity']}` | diagnosis=`{row['diagnosis']}` | "
            f"net_r=`{float(row['net_r_multiple']):.3f}` | ctx=`{row['matched_context_mode']}` | trust=`{row['matched_trust_tier']}`"
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
        description="Build a RESEARCH_ONLY orderflow coverage gap summary and ETH bad-trade casebook from the fixed ETH template replay."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--micro-capture-dir", default=str(DEFAULT_MICRO_CAPTURE_DIR))
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--price-action-path", default="")
    parser.add_argument("--blueprint-path", default="")
    parser.add_argument("--veto-event-study-path", default="")
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
    veto_event_study_path = Path(args.veto_event_study_path).expanduser().resolve() if text(args.veto_event_study_path) else latest_artifact(review_dir, "*_intraday_orderflow_veto_event_study.json")

    price_action = load_json_mapping(price_action_path)
    blueprint = load_json_mapping(blueprint_path)
    veto_event_study = load_json_mapping(veto_event_study_path)
    price_action_module = load_price_action_module()

    dataset_path = Path(args.dataset_path).expanduser().resolve() if text(args.dataset_path) else Path(text(price_action.get("dataset_path"))).expanduser().resolve()
    frame = price_action_module.add_features(price_action_module.load_frame(dataset_path))
    selected_params = dict((price_action.get("focus_symbol_result") or {}).get("selected_params") or price_action.get("selected_params") or {})
    scenario = dict(DEFAULT_SCENARIO)

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
    coverage_hours = parse_hours_list(args.coverage_hours) or [24.0, 48.0]
    if 48.0 not in coverage_hours:
        coverage_hours = sorted(set([*coverage_hours, 48.0]))

    symbol_gap_rows: list[dict[str, Any]] = []
    eth_aligned_events: list[dict[str, Any]] = []
    for symbol in target_symbols:
        symbol_frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
        role = text((ladder_map.get(symbol) or {}).get("role"))
        replay = price_action_module.simulate_symbol(symbol_frame, selected_params) if not symbol_frame.empty else {"trades": []}
        full_trades = apply_costs_full(price_action_module, list(replay.get("trades") or []), scenario)
        aligned = align_trades_to_micro(
            trades=full_trades,
            micro_rows=load_micro_rows(micro_capture_dir, symbol),
            coverage_windows_hours=coverage_hours,
        )
        symbol_gap_rows.append(build_symbol_gap_row(symbol, role, aligned))
        if symbol == "ETHUSDT":
            eth_aligned_events = aligned

    symbol_gap_rows.sort(key=lambda row: (row["gap_priority"], row["symbol"]))
    eth_bad_trade_casebook = build_eth_bad_trade_casebook(eth_aligned_events)
    majors = [row["symbol"] for row in symbol_gap_rows if row["role"] in {"mainline_primary", "majors_anchor"}]
    urgent_majors = [row["symbol"] for row in symbol_gap_rows if row["gap_priority"] == "urgent_major_capture_gap"]
    harmful_rules = list(
        veto_event_study.get("recommended_brief", "")
    )
    research_decision = "capture_majors_gap_and_build_eth_bad_trade_casebook_before_new_replay"
    recommended_brief = (
        "orderflow_gap_casebook:"
        f"urgent_majors={','.join(urgent_majors) if urgent_majors else '-'},"
        f"majors={','.join(majors) if majors else '-'},"
        f"eth_bad_cases={to_int(eth_bad_trade_casebook.get('case_count'), 0)},"
        f"diagnosis={','.join(sorted((eth_bad_trade_casebook.get('diagnosis_counts') or {}).keys())) if (eth_bad_trade_casebook.get('diagnosis_counts') or {}) else '-'},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_intraday_orderflow_coverage_gap_casebook",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "dataset_path": str(dataset_path),
            "price_action_path": str(price_action_path),
            "blueprint_path": str(blueprint_path),
            "veto_event_study_path": str(veto_event_study_path),
            "micro_capture_dir": str(micro_capture_dir),
        },
        "fixed_template": {
            "template_symbol": text(price_action.get("focus_symbol")) or "ETHUSDT",
            "family": "price_action_breakout_pullback",
            "selected_params": selected_params,
            "scenario": scenario,
        },
        "symbol_gap_rows": symbol_gap_rows,
        "eth_bad_trade_casebook": eth_bad_trade_casebook,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份工件把 ETH 固定模板 replay 的覆盖缺口与坏交易 casebook 放到同一 source-owned 视图里；"
            "目标是先决定采样和事件研究优先级，而不是继续扩参数。"
        ),
        "limitation_note": (
            "当前 casebook 仍受 micro_capture 稀疏覆盖和 single_exchange_low trust 限制；"
            "它只能告诉我们哪里该补采样、哪些坏交易值得盯，不代表已找到可交易 veto 规则。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_coverage_gap_casebook.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_coverage_gap_casebook.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_coverage_gap_casebook.json"
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
