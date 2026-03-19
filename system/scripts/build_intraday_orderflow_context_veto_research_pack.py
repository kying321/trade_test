#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_MICRO_CAPTURE_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "micro_capture"


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


def parse_hours_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in text(raw).split(","):
        if not part.strip():
            continue
        value = to_float(part, 0.0)
        if value > 0:
            values.append(value)
    return sorted(set(values))


def load_micro_rows(micro_capture_dir: Path, target_symbols: set[str]) -> dict[str, list[dict[str, Any]]]:
    symbol_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(micro_capture_dir.glob("*_micro_capture.json")):
        payload = load_json_mapping(path)
        captured_at = parse_utc(payload.get("captured_at_utc") or payload.get("generated_at_utc"))
        selected_micro = payload.get("selected_micro") or []
        if not isinstance(selected_micro, list):
            continue
        for raw in selected_micro:
            if not isinstance(raw, dict):
                continue
            symbol = text(raw.get("symbol")).upper()
            if symbol not in target_symbols:
                continue
            row = dict(raw)
            row["_captured_at_utc"] = fmt_utc(captured_at)
            row["_artifact_path"] = str(path)
            symbol_rows[symbol].append(row)
    return symbol_rows


def derive_logic_hypothesis(context_mode: str, role: str) -> str:
    if context_mode == "reversal":
        return (
            "先把 reversal 视作 breakout / pullback 延续交易的 veto 候选；"
            "在 trust 没升级前，不把它当确认信号。"
        )
    if context_mode in {"absorption", "failed_auction"}:
        return (
            "先把 effort-result divergence 视作风险上下文；"
            "优先用于拒绝或延后交易，而不是放行。"
        )
    if context_mode == "continuation":
        if role in {"mainline_primary", "majors_anchor"}:
            return (
                "continuation 只保留为未来可能的 confirmation 候选；"
                "当前因 trust 仍低，只允许观察，不允许直接提升 entry 置信度。"
            )
        return "continuation 先保留在 beta flow watch；在 cross-source 质量改善前不升格。"
    return "上下文未稳定，先保留为 research-only 描述字段。"


def derive_route_status(row: dict[str, Any]) -> str:
    role = text(row.get("role"))
    trust = text(row.get("dominant_trust_tier"))
    cross_source_pass_ratio = to_float(row.get("cross_source_pass_ratio"), 0.0)
    gate = text(row.get("research_logic_gate"))
    if gate != "ready_context_veto_only":
        return "collect_more_sidecar_samples"
    if role in {"mainline_primary", "majors_anchor"}:
        if cross_source_pass_ratio >= 0.65 and trust == "single_exchange_low":
            return "majors_context_veto_only_pending_trust_upgrade"
        if cross_source_pass_ratio >= 0.65:
            return "majors_context_veto_candidate"
        return "majors_collect_cross_source_quality"
    if cross_source_pass_ratio >= 0.25:
        return "beta_watch_context_only"
    return "beta_watch_only_low_cross_source_quality"


def build_symbol_routes(blueprint: dict[str, Any], sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    sidecar_rows = {text(row.get("symbol")).upper(): row for row in sidecar.get("symbol_rows", []) if isinstance(row, dict)}
    routes: list[dict[str, Any]] = []
    for ladder_row in blueprint.get("symbol_depth_ladder", []):
        if not isinstance(ladder_row, dict):
            continue
        symbol = text(ladder_row.get("symbol")).upper()
        sidecar_row = sidecar_rows.get(symbol)
        if sidecar_row is None:
            routes.append(
                {
                    "symbol": symbol,
                    "priority": to_int(ladder_row.get("priority"), 0),
                    "role": text(ladder_row.get("role")),
                    "route_status": "missing_sidecar_row",
                    "recommended_use": "collect_more_sidecar_samples",
                    "logic_hypothesis": "缺少 sidecar 样本，当前不能进入逻辑验证。",
                }
            )
            continue
        routes.append(
            {
                "symbol": symbol,
                "priority": to_int(ladder_row.get("priority"), 0),
                "role": text(sidecar_row.get("role") or ladder_row.get("role")),
                "route_status": derive_route_status(sidecar_row),
                "recommended_use": text(sidecar_row.get("recommended_use")),
                "logic_hypothesis": derive_logic_hypothesis(
                    text(sidecar_row.get("dominant_context_mode")),
                    text(sidecar_row.get("role") or ladder_row.get("role")),
                ),
                "evidence_snapshot": {
                    "sample_count": to_int(sidecar_row.get("sample_count"), 0),
                    "schema_ok_ratio": to_float(sidecar_row.get("schema_ok_ratio"), 0.0),
                    "cross_source_pass_ratio": to_float(sidecar_row.get("cross_source_pass_ratio"), 0.0),
                    "dominant_context_mode": text(sidecar_row.get("dominant_context_mode")),
                    "dominant_trust_tier": text(sidecar_row.get("dominant_trust_tier")),
                    "dominant_veto_hint": text(sidecar_row.get("dominant_veto_hint")),
                    "time_sync_ok_ratio": to_float(sidecar_row.get("time_sync_ok_ratio"), 0.0),
                },
            }
        )
    routes.sort(key=lambda row: (to_int(row.get("priority"), 999), text(row.get("symbol"))))
    return routes


def build_eth_trade_alignment(
    *,
    symbol: str,
    trades: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
    coverage_windows_hours: list[float],
) -> dict[str, Any]:
    max_window = max(coverage_windows_hours) if coverage_windows_hours else 24.0
    alignments: list[dict[str, Any]] = []
    coverage_counts = {window: 0 for window in coverage_windows_hours}
    covered_context_counts = Counter()
    covered_exit_reason_counts = Counter()

    indexed_micro = []
    for row in micro_rows:
        captured_at = parse_utc(row.get("_captured_at_utc"))
        if captured_at is None:
            continue
        indexed_micro.append((captured_at, row))

    for trade in trades:
        entry_ts = parse_utc(trade.get("entry_ts_utc"))
        exit_ts = parse_utc(trade.get("exit_ts_utc"))
        nearest_ts: dt.datetime | None = None
        nearest_row: dict[str, Any] | None = None
        nearest_delta_hours = 0.0
        if entry_ts is not None and indexed_micro:
            nearest_ts, nearest_row = min(indexed_micro, key=lambda item: abs((item[0] - entry_ts).total_seconds()))
            nearest_delta_hours = abs((nearest_ts - entry_ts).total_seconds()) / 3600.0
            for window in coverage_windows_hours:
                if nearest_delta_hours <= window:
                    coverage_counts[window] += 1
            if nearest_delta_hours <= max_window:
                covered_context_counts[text((nearest_row or {}).get("cvd_context_mode"))] += 1
                covered_exit_reason_counts[text(trade.get("exit_reason"))] += 1

        alignments.append(
            {
                "entry_ts_utc": text(trade.get("entry_ts_utc")),
                "exit_ts_utc": text(trade.get("exit_ts_utc")),
                "exit_reason": text(trade.get("exit_reason")),
                "bars_held": to_int(trade.get("bars_held"), 0),
                "net_r_multiple": to_float(trade.get("net_r_multiple"), to_float(trade.get("r_multiple"), 0.0)),
                "nearest_capture_ts_utc": fmt_utc(nearest_ts),
                "nearest_capture_delta_hours": float(round(nearest_delta_hours, 6)) if nearest_row is not None else None,
                "within_window": {
                    f"{int(window) if float(window).is_integer() else window}h": bool(
                        nearest_row is not None and nearest_delta_hours <= window
                    )
                    for window in coverage_windows_hours
                },
                "matched_context_mode": text((nearest_row or {}).get("cvd_context_mode")),
                "matched_trust_tier": text((nearest_row or {}).get("cvd_trust_tier_hint")),
                "matched_veto_hint": text((nearest_row or {}).get("cvd_veto_hint")),
                "matched_queue_imbalance": to_float((nearest_row or {}).get("queue_imbalance"), 0.0),
                "matched_ofi_norm": to_float((nearest_row or {}).get("ofi_norm"), 0.0),
                "matched_micro_alignment": to_float((nearest_row or {}).get("micro_alignment"), 0.0),
                "matched_artifact_path": text((nearest_row or {}).get("_artifact_path")),
            }
        )

    trade_count = len(alignments)
    coverage_by_window = {
        f"within_{int(window) if float(window).is_integer() else window}h": {
            "count": int(coverage_counts[window]),
            "ratio": float(safe_ratio(coverage_counts[window], trade_count)),
        }
        for window in coverage_windows_hours
    }

    coverage_12h = coverage_by_window.get("within_12h", {}).get("ratio", 0.0)
    coverage_24h = coverage_by_window.get("within_24h", {}).get("ratio", 0.0)
    if coverage_24h >= 0.8 and coverage_12h >= 0.6:
        alignment_gate = "ready_for_signal_level_event_study"
    elif coverage_24h >= 0.4:
        alignment_gate = "partial_alignment_only_not_backtest_ready"
    else:
        alignment_gate = "alignment_gap_blocks_signal_level_use"

    return {
        "symbol": symbol,
        "trade_count": trade_count,
        "coverage_windows_hours": coverage_windows_hours,
        "coverage_by_window": coverage_by_window,
        "covered_context_counts_within_max_window": dict(covered_context_counts),
        "covered_exit_reason_counts_within_max_window": dict(covered_exit_reason_counts),
        "alignment_gate": alignment_gate,
        "trade_alignments": alignments,
    }


def build_pre_backtest_gates(
    *,
    sidecar: dict[str, Any],
    symbol_routes: list[dict[str, Any]],
    eth_alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    sidecar_rows = {text(row.get("symbol")).upper(): row for row in sidecar.get("symbol_rows", []) if isinstance(row, dict)}
    eth_row = sidecar_rows.get("ETHUSDT", {})
    btc_row = sidecar_rows.get("BTCUSDT", {})
    bnb_row = sidecar_rows.get("BNBUSDT", {})
    sol_row = sidecar_rows.get("SOLUSDT", {})
    within_12h = ((eth_alignment.get("coverage_by_window") or {}).get("within_12h") or {}).get("ratio", 0.0)
    within_24h = ((eth_alignment.get("coverage_by_window") or {}).get("within_24h") or {}).get("ratio", 0.0)
    majors_routes = {
        row["symbol"]: row["route_status"]
        for row in symbol_routes
        if text(row.get("role")) in {"mainline_primary", "majors_anchor"}
    }
    gates = [
        {
            "gate": "unified_orderflow_sidecar_schema",
            "status": "pass" if len(sidecar.get("ready_symbols", [])) >= 2 else "fail",
            "detail": "ETH/BTC/BNB/SOL 已有统一 sidecar schema 与 source-owned 字段。",
        },
        {
            "gate": "majors_context_veto_route",
            "status": "pass" if set(majors_routes.values()) <= {"majors_context_veto_only_pending_trust_upgrade", "majors_context_veto_candidate"} and len(majors_routes) >= 2 else "fail",
            "detail": f"majors_routes={majors_routes}",
        },
        {
            "gate": "eth_trade_alignment_within_24h",
            "status": "pass" if within_24h >= 0.8 else "fail",
            "detail": f"ETH validation trade coverage within 24h={within_24h:.2%}",
        },
        {
            "gate": "eth_trade_alignment_within_12h",
            "status": "pass" if within_12h >= 0.6 else "fail",
            "detail": f"ETH validation trade coverage within 12h={within_12h:.2%}",
        },
        {
            "gate": "majors_trust_tier_upgrade",
            "status": "pass"
            if text(eth_row.get("dominant_trust_tier")) in {"single_exchange_ok", "cross_exchange_confirmed"}
            and text(btc_row.get("dominant_trust_tier")) in {"single_exchange_ok", "cross_exchange_confirmed"}
            else "fail",
            "detail": (
                f"ETH trust={text(eth_row.get('dominant_trust_tier'))}, "
                f"BTC trust={text(btc_row.get('dominant_trust_tier'))}"
            ),
        },
        {
            "gate": "beta_cross_source_quality_watch_only",
            "status": "pass"
            if to_float(bnb_row.get("cross_source_pass_ratio"), 0.0) < 0.25
            and to_float(sol_row.get("cross_source_pass_ratio"), 0.0) < 0.25
            else "fail",
            "detail": (
                f"BNB cross_source_pass_ratio={to_float(bnb_row.get('cross_source_pass_ratio'), 0.0):.2%}, "
                f"SOL cross_source_pass_ratio={to_float(sol_row.get('cross_source_pass_ratio'), 0.0):.2%}"
            ),
        },
    ]
    return gates


def build_research_queue(
    *,
    eth_alignment: dict[str, Any],
    symbol_routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    within_12h = ((eth_alignment.get("coverage_by_window") or {}).get("within_12h") or {}).get("ratio", 0.0)
    within_24h = ((eth_alignment.get("coverage_by_window") or {}).get("within_24h") or {}).get("ratio", 0.0)
    majors = [row["symbol"] for row in symbol_routes if text(row.get("role")) in {"mainline_primary", "majors_anchor"}]
    beta_watch = [row["symbol"] for row in symbol_routes if text(row.get("role")).startswith("beta_flow")]
    return [
        {
            "priority": 1,
            "title": "补齐 ETH/BTC orderflow 对齐覆盖率",
            "why": f"当前 ETH validation trade coverage 仅 12h={within_12h:.2%}, 24h={within_24h:.2%}，不足以做 signal-level backtest gating。",
            "done_when": "ETH validation / replay trade windows 的 orderflow 覆盖率达到 24h>=80% 且 12h>=60%。",
            "scope": majors,
        },
        {
            "priority": 2,
            "title": "只做硬逻辑 veto 事件研究",
            "why": "在 trust 仍低时，不应直接参数化 orderflow；先验证 quality veto / reversal / absorption 是否能解释坏交易。",
            "done_when": "形成 ETH 主线交易样本的 keep/veto 事件分层结果，仍保持无参数拟合。",
            "scope": ["ETHUSDT"],
        },
        {
            "priority": 3,
            "title": "用 BTC 做 majors anchor 对照",
            "why": "BTC dominant_context_mode=absorption，可作为 majors effort-result divergence 的对照锚。",
            "done_when": "明确 ETH/BTC 在相同 veto 合同下的差异，只输出 research-only 结论。",
            "scope": ["BTCUSDT", "ETHUSDT"],
        },
        {
            "priority": 4,
            "title": "保持 BNB/SOL 为 beta flow watch",
            "why": "BNB/SOL cross-source 质量仍弱，当前只能积累 flow case，不应混入统一回测家族。",
            "done_when": "beta watch 先形成更稳定的 cross-source / time-sync 画像，再决定是否升格。",
            "scope": beta_watch,
        },
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    current_mainline = payload.get("current_mainline") or {}
    eth_alignment = payload.get("eth_signal_alignment") or {}
    lines: list[str] = [
        "# Intraday Orderflow Context Veto Research Pack",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Current Mainline",
        "",
        f"- symbol: `{text(current_mainline.get('symbol'))}`",
        f"- validation_status: `{text(current_mainline.get('validation_status'))}`",
        f"- selected_params: `{json.dumps(current_mainline.get('selected_params') or {}, ensure_ascii=False)}`",
        f"- validation_metrics: `{json.dumps(current_mainline.get('validation_metrics') or {}, ensure_ascii=False)}`",
        f"- exit_hold_forward_decision: `{text(current_mainline.get('exit_hold_forward_decision'))}`",
        "",
        "## Pre-Backtest Gates",
        "",
    ]
    for gate in payload.get("pre_backtest_gates", []):
        lines.append(f"- `{gate['gate']}` | status=`{gate['status']}` | detail=`{gate['detail']}`")
    lines.extend(["", "## Symbol Routes", ""])
    for row in payload.get("symbol_routes", []):
        lines.append(
            f"- `{row['symbol']}` | role=`{row['role']}` | route_status=`{row['route_status']}` | "
            f"recommended_use=`{row['recommended_use']}` | logic_hypothesis=`{row['logic_hypothesis']}`"
        )
    lines.extend(
        [
            "",
            "## ETH Signal Alignment",
            "",
            f"- alignment_gate: `{text(eth_alignment.get('alignment_gate'))}`",
            f"- coverage_by_window: `{json.dumps(eth_alignment.get('coverage_by_window') or {}, ensure_ascii=False)}`",
            "",
            "## Engine Logic Contract",
            "",
        ]
    )
    for item in payload.get("engine_logic_contract", []):
        lines.append(
            f"- `{item['rule']}` | current_action=`{item['current_action']}` | note=`{item['note']}`"
        )
    lines.extend(["", "## Research Queue", ""])
    for row in payload.get("research_queue", []):
        lines.append(
            f"- priority=`{row['priority']}` | title=`{row['title']}` | why=`{row['why']}` | done_when=`{row['done_when']}`"
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
        description="Build a RESEARCH_ONLY intraday orderflow context/veto research pack before any new backtest or parameter fitting."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--micro-capture-dir", default=str(DEFAULT_MICRO_CAPTURE_DIR))
    parser.add_argument("--blueprint-path", default="")
    parser.add_argument("--sidecar-path", default="")
    parser.add_argument("--price-action-path", default="")
    parser.add_argument("--exit-forward-path", default="")
    parser.add_argument("--coverage-hours", default="1,4,12,24")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    micro_capture_dir = Path(args.micro_capture_dir).expanduser().resolve()

    blueprint_path = Path(args.blueprint_path).expanduser().resolve() if text(args.blueprint_path) else latest_artifact(review_dir, "*_intraday_orderflow_strategy_blueprint.json")
    sidecar_path = Path(args.sidecar_path).expanduser().resolve() if text(args.sidecar_path) else latest_artifact(review_dir, "*_intraday_orderflow_sidecar.json")
    price_action_path = Path(args.price_action_path).expanduser().resolve() if text(args.price_action_path) else latest_artifact(review_dir, "*_price_action_breakout_pullback_sim_only.json")
    exit_forward_path = Path(args.exit_forward_path).expanduser().resolve() if text(args.exit_forward_path) else latest_artifact(review_dir, "*_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json")

    blueprint = load_json_mapping(blueprint_path)
    sidecar = load_json_mapping(sidecar_path)
    price_action = load_json_mapping(price_action_path)
    exit_forward = load_json_mapping(exit_forward_path)

    target_symbols = {text(row.get("symbol")).upper() for row in blueprint.get("symbol_depth_ladder", []) if isinstance(row, dict) and text(row.get("symbol"))}
    micro_rows_by_symbol = load_micro_rows(micro_capture_dir, target_symbols)

    focus_symbol = text(price_action.get("focus_symbol")) or text((price_action.get("focus_symbol_result") or {}).get("symbol")) or "ETHUSDT"
    focus_result = dict(price_action.get("focus_symbol_result") or {})
    trade_sample = list(focus_result.get("validation_trade_sample") or [])
    coverage_hours = parse_hours_list(args.coverage_hours) or [1.0, 4.0, 12.0, 24.0]

    symbol_routes = build_symbol_routes(blueprint, sidecar)
    eth_alignment = build_eth_trade_alignment(
        symbol=focus_symbol,
        trades=trade_sample,
        micro_rows=micro_rows_by_symbol.get(focus_symbol, []),
        coverage_windows_hours=coverage_hours,
    )
    pre_backtest_gates = build_pre_backtest_gates(
        sidecar=sidecar,
        symbol_routes=symbol_routes,
        eth_alignment=eth_alignment,
    )
    majors = [row["symbol"] for row in symbol_routes if text(row.get("role")) in {"mainline_primary", "majors_anchor"}]
    beta_watch = [row["symbol"] for row in symbol_routes if text(row.get("role")).startswith("beta_flow")]

    engine_logic_contract = [
        {
            "rule": "quality_veto_bias",
            "current_action": "只要 schema/sync/gap/time_sync/low_sample 有风险，就先保留 veto 倾向，不做 confirmation。",
            "note": "直接锚定 engine.py 里的 micro risk penalty 逻辑。",
        },
        {
            "rule": "trust_tier_upgrade_required_for_confirmation",
            "current_action": "只有 single_exchange_ok / cross_exchange_confirmed 才允许 future confirmation 假设；当前 dominant_trust_tier 仍低，禁止提权。",
            "note": "当前 sidecar 的 majors/beta 仍以 single_exchange_low 为主。",
        },
        {
            "rule": "reversal_absorption_failed_auction_default_to_veto",
            "current_action": "reversal / absorption / failed_auction 先视作风险上下文，用来拒绝或延后 price-state 触发。",
            "note": "对应 engine.py 中的 sweep_without_delta_confirmation 与 effort_result_divergence 风险。 ",
        },
        {
            "rule": "continuation_is_observe_only_until_trust_improves",
            "current_action": "continuation 当前只能做观察标签；在 trust 与对齐覆盖率改善前，不作为 entry boost。",
            "note": "避免用低质量单交易所 flow 反向污染当前 ETH 主线。",
        },
    ]

    research_queue = build_research_queue(eth_alignment=eth_alignment, symbol_routes=symbol_routes)
    coverage_12h = ((eth_alignment.get("coverage_by_window") or {}).get("within_12h") or {}).get("ratio", 0.0)
    coverage_24h = ((eth_alignment.get("coverage_by_window") or {}).get("within_24h") or {}).get("ratio", 0.0)
    research_decision = (
        "keep_price_state_primary_event_alignment_gap_blocks_backtest"
        if eth_alignment.get("alignment_gate") != "ready_for_signal_level_event_study"
        else "ready_for_eth_event_study_veto_only"
    )
    recommended_brief = (
        "orderflow_context_veto_pack:"
        f"eth_12h_cov={coverage_12h:.2%},"
        f"eth_24h_cov={coverage_24h:.2%},"
        f"majors={','.join(majors) if majors else '-'},"
        f"beta_watch={','.join(beta_watch) if beta_watch else '-'},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_intraday_orderflow_context_veto_research_pack",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "blueprint_path": str(blueprint_path),
            "sidecar_path": str(sidecar_path),
            "price_action_path": str(price_action_path),
            "exit_forward_path": str(exit_forward_path),
            "micro_capture_dir": str(micro_capture_dir),
        },
        "current_mainline": {
            "symbol": focus_symbol,
            "family": "price_action_breakout_pullback",
            "validation_status": text(focus_result.get("validation_status") or price_action.get("validation_status")),
            "selected_params": dict(focus_result.get("selected_params") or price_action.get("selected_params") or {}),
            "validation_metrics": dict(focus_result.get("validation_metrics") or price_action.get("validation_metrics") or {}),
            "exit_hold_forward_decision": text(exit_forward.get("research_decision")),
            "exit_hold_forward_summary": dict(exit_forward.get("comparison_summary") or {}),
        },
        "engine_logic_contract": engine_logic_contract,
        "symbol_routes": symbol_routes,
        "eth_signal_alignment": eth_alignment,
        "pre_backtest_gates": pre_backtest_gates,
        "research_queue": research_queue,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份 pack 把 price-state 主线、orderflow sidecar、底层 veto 逻辑和 ETH 样本对齐覆盖率收束到一份 source-owned 工件里；"
            "目的不是直接回测，而是先固定逻辑合同与可行性门槛。"
        ),
        "limitation_note": (
            "当前 ETH 交易样本与 micro_capture 的时间对齐覆盖率仍偏低，且 dominant_trust_tier 仍以 single_exchange_low 为主；"
            "因此 orderflow 还不能进入参数拟合或回测真值，只能继续做 research-only 事件研究与采样补齐。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_context_veto_research_pack.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_context_veto_research_pack.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_context_veto_research_pack.json"
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
