#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
MICROSTRUCTURE_PATH = SYSTEM_ROOT / "src" / "lie_engine" / "signal" / "microstructure.py"


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or dt.datetime.now(dt.timezone.utc)
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


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


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Intraday Orderflow Strategy Blueprint",
        "",
        f"- symbol: `{text((payload.get('current_mainline') or {}).get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Current Mainline",
        "",
        f"- family: `{text((payload.get('current_mainline') or {}).get('family'))}`",
        f"- mainline_status: `{text((payload.get('current_mainline') or {}).get('validation_status'))}`",
        f"- exit_forward_decision: `{text((((payload.get('current_mainline') or {}).get('exit_hold_forward_compare')) or {}).get('research_decision'))}`",
        "",
        "## Orderflow Takeaways",
        "",
    ]
    for row in payload.get("orderflow_foundation", {}).get("repo_takeaways", []):
        lines.append(f"- `{row}`")
    lines.extend(["", "## Symbol Depth Ladder", ""])
    for row in payload.get("symbol_depth_ladder", []):
        lines.append(
            f"- `{row['symbol']}` | role=`{row['role']}` | priority=`{row['priority']}` | reason=`{row['reason']}`"
        )
    lines.extend(["", "## Logic Stack", ""])
    for row in payload.get("strategy_logic_stack", []):
        inputs = ",".join(row.get("inputs", []))
        lines.append(
            f"- `{row['layer']}` | purpose=`{row['purpose']}`"
            + (f" | inputs=`{inputs}`" if inputs else "")
        )
    lines.extend(["", "## Feasibility Gates Before Backtest", ""])
    for gate_name, items in (payload.get("feasibility_gates_before_backtest") or {}).items():
        lines.append(f"### {gate_name}")
        for item in items:
            lines.append(f"- `{item}`")
        lines.append("")
    plan = payload.get("autoresearch_plan", {})
    lines.extend(
        [
            "## Autoresearch Launch Block",
            "",
            "$codex-autoresearch",
            "Mode: plan",
            f"Goal: {text(plan.get('goal'))}",
            f"Scope: {text(plan.get('scope'))}",
            f"Metric: {text(plan.get('metric'))}",
            f"Direction: {text(plan.get('direction'))}",
            f"Verify: {text(plan.get('verify'))}",
            f"Guard: {text(plan.get('guard'))}",
            f"Iterations: {text(plan.get('iterations'))}",
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
        description="Build a RESEARCH_ONLY blueprint for expanding the ETH intraday mainline with orderflow and symbol-depth logic before backtest/fitting."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    price_action_path = latest_artifact(review_dir, "*_price_action_breakout_pullback_sim_only.json")
    exit_forward_compare_path = latest_artifact(review_dir, "*_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json")
    symbol_route_path = latest_artifact(review_dir, "*_binance_indicator_symbol_route_handoff.json")
    lane_playbook_path = latest_artifact(review_dir, "*_binance_indicator_native_lane_playbook.json")
    lane_stability_path = latest_artifact(review_dir, "*_binance_indicator_native_lane_stability_report.json")
    public_quant_path = latest_artifact(review_dir, "*_public_quant_strategy_cross_section_research.json")

    price_action = load_json_mapping(price_action_path)
    exit_forward = load_json_mapping(exit_forward_compare_path)
    symbol_route = load_json_mapping(symbol_route_path)
    lane_playbook = load_json_mapping(lane_playbook_path)
    lane_stability = load_json_mapping(lane_stability_path)
    public_quant = load_json_mapping(public_quant_path)

    micro_spec = importlib.util.spec_from_file_location("fenlie_microstructure", MICROSTRUCTURE_PATH)
    micro = importlib.util.module_from_spec(micro_spec)
    assert micro_spec is not None and micro_spec.loader is not None
    micro_spec.loader.exec_module(micro)

    focus_symbol = text(price_action.get("focus_symbol")) or "ETHUSDT"
    focus_result = dict(price_action.get("focus_symbol_result") or {})
    deploy_now_symbols = list(symbol_route.get("deploy_now_symbols") or [])
    watch_priority_symbols = list(symbol_route.get("watch_priority_symbols") or [])
    watch_only_symbols = list(symbol_route.get("watch_only_symbols") or [])
    overall_takeaway = text(symbol_route.get("overall_takeaway"))
    lane_takeaway = text(lane_playbook.get("overall_takeaway"))
    stability_takeaway = text(lane_stability.get("overall_takeaway"))
    public_quant_brief = text(public_quant.get("recommended_brief"))

    orderflow_repo_takeaways = [
        overall_takeaway or "BTC/ETH 仍是 majors 价格状态主线，beta flow 先看 BNB 再看 SOL。",
        lane_takeaway or "Native lane 仍未确认 beta flow 可升级，保持 research-only。",
        stability_takeaway or "Long-window stability 仍不足，不能把 flow 直接升成主触发。",
        "geometry_delta_breakout 的公开 cross-section 结果仍未转正，只可保留为 intraday/true-flow 再验证候选。",
        "对 majors 更合理的角色是：price-state 做主触发，orderflow 先做 context/veto，再决定是否升级到确认层。",
    ]

    current_mainline = {
        "symbol": focus_symbol,
        "family": "price_action_breakout_pullback",
        "validation_status": text(focus_result.get("validation_status")),
        "selected_params": dict(focus_result.get("selected_params") or price_action.get("selected_params") or {}),
        "validation_metrics": dict(focus_result.get("validation_metrics") or price_action.get("validation_metrics") or {}),
        "validation_gross_metrics": dict(
            focus_result.get("validation_gross_metrics") or price_action.get("validation_gross_metrics") or {}
        ),
        "exit_hold_forward_compare": {
            "artifact": str(exit_forward_compare_path),
            "research_decision": text(exit_forward.get("research_decision")),
            "comparison_summary": dict(exit_forward.get("comparison_summary") or {}),
        },
    }

    orderflow_foundation = {
        "microstructure_required_fields": {
            "l2": list(getattr(micro, "L2_REQUIRED_FIELDS", ())),
            "trades": list(getattr(micro, "TRADES_REQUIRED_FIELDS", ())),
        },
        "microstructure_primitives": [
            "queue_imbalance",
            "ofi_norm",
            "micro_alignment",
            "cvd_delta_ratio",
            "cvd_price_move_pct",
            "cvd_range_pct",
            "cvd_effort_result_ratio",
            "cvd_context_mode",
            "cvd_context_note",
            "cvd_trust_tier_hint",
            "trade_count",
            "max_trade_gap_ms",
            "sync_skew_ms",
            "sync_ok",
            "gap_ok",
            "evidence_score",
        ],
        "repo_takeaways": orderflow_repo_takeaways,
        "public_orderflow_proxy_brief": public_quant_brief,
    }

    symbol_depth_ladder = [
        {
            "priority": 1,
            "symbol": "ETHUSDT",
            "role": "mainline_primary",
            "reason": "当前最佳 source-owned 单币主线；先在 ETH 上验证 orderflow 只做 context/veto 是否成立。",
        },
        {
            "priority": 2,
            "symbol": "BTCUSDT",
            "role": "majors_anchor",
            "reason": "symbol route 已给出 BTC/ETH deploy-now on price-state only；BTC 适合作为 majors 对照锚。",
        },
        {
            "priority": 3,
            "symbol": "BNBUSDT",
            "role": "beta_flow_first_watch_leg",
            "reason": "现有 route handoff 明确 BNB 是 first beta flow watch leg，适合先做 flow 深挖。",
        },
        {
            "priority": 4,
            "symbol": "SOLUSDT",
            "role": "beta_flow_second_watch_leg",
            "reason": "保持在 BNB 后面；只有在 BNB 的 flow 逻辑更清晰后再扩到 SOL。",
        },
    ]
    if not deploy_now_symbols:
        deploy_now_symbols = ["BTCUSDT", "ETHUSDT"]
    if not watch_priority_symbols:
        watch_priority_symbols = ["BNBUSDT"]
    if not watch_only_symbols:
        watch_only_symbols = ["SOLUSDT"]

    strategy_logic_stack = [
        {
            "layer": "price_state_trigger",
            "purpose": "保持 ETH/BTC 的 breakout-pullback / price-state 作为当前主触发，不让 flow 直接替代 entry backbone。",
            "inputs": ["breakout_context", "pullback_depth_atr", "ema_trend_state", "trigger_bar"],
        },
        {
            "layer": "orderflow_context",
            "purpose": "把 orderflow 先降级为 context/veto，验证它是否改善主触发的样本质量、方向一致性和回撤控制。",
            "inputs": ["taker_oi_bias", "cvd_lite_proxy", "micro_alignment", "queue_imbalance", "ofi_norm"],
        },
        {
            "layer": "effort_result_interpreter",
            "purpose": "对 continuation/reversal/absorption/failed_auction 做模式分类，先建立逻辑可解释性，再决定是否纳入回测。",
            "inputs": ["cvd_context_mode", "cvd_context_note", "cvd_effort_result_ratio", "cvd_price_move_pct"],
        },
        {
            "layer": "symbol_depth_router",
            "purpose": "按 ETH -> BTC -> BNB -> SOL 的梯度扩深，先 majors 后 beta，先 price-state 主线后 flow watch leg。",
            "inputs": ["deploy_now_symbols", "watch_priority_symbols", "watch_only_symbols"],
        },
    ]

    feasibility_gates_before_backtest = {
        "data_gate": [
            "研究侧必须能为 ETH/BTC/BNB/SOL 产出统一 schema 的 orderflow sidecar。",
            "每个 symbol 至少要有 schema_ok、evidence_score、cvd_context_mode 这些 source-owned 字段。",
            "任何 orderflow 外部数据补充都只能进 research-only sidecar，不能进 live path。",
        ],
        "logic_gate": [
            "先证明 orderflow 对主线的角色是 context/veto/confirmation 中的哪一种，而不是直接上参数搜索。",
            "先看 continuation / reversal / absorption / failed_auction 是否能稳定解释 price-state 结果差异。",
            "只有当 majors 与 beta 的 role boundary 清晰后，才允许做跨标的统一回测。",
        ],
        "sample_gate": [
            "ETH 主线先证明 signal density 不会因 orderflow 过滤而退化到不可研究样本。",
            "BTC 只做 majors 对照；BNB/SOL 只做 beta flow watch，不直接混成一个统一 entry family。",
            "在 logic gate 通过前，不做大规模参数因子拟合。",
        ],
    }

    autoresearch_plan = {
        "goal": "构建 research-only 的 intraday orderflow sidecar，并验证它对 ETH/BTC/BNB/SOL 15m price-state 主线的 context/veto 价值，再决定是否进入回测和参数拟合。",
        "scope": "system/scripts/build_intraday_orderflow_strategy_blueprint.py,system/scripts/build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py,system/src/lie_engine/signal/microstructure.py,system/output/review/*_intraday_orderflow_*",
        "metric": "usable_symbol_count_with_orderflow_logic_gate",
        "direction": "higher",
        "verify": "python -m py_compile system/scripts/build_intraday_orderflow_strategy_blueprint.py && test -f system/output/review/latest_intraday_orderflow_blueprint.json",
        "guard": "python -m py_compile system/src/lie_engine/signal/microstructure.py system/scripts/build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py",
        "iterations": "3",
    }

    research_decision = "continue_price_state_primary_build_orderflow_sidecar_before_backtest"
    recommended_brief = (
        f"{focus_symbol}:orderflow_blueprint:decision={research_decision},"
        f"majors={','.join(deploy_now_symbols)},beta_watch={','.join(watch_priority_symbols + watch_only_symbols)},"
        f"exit_forward={text(exit_forward.get('research_decision'))}"
    )

    payload = {
        "action": "build_intraday_orderflow_strategy_blueprint",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "price_action_mainline": str(price_action_path),
            "exit_hold_forward_compare": str(exit_forward_compare_path),
            "symbol_route_handoff": str(symbol_route_path),
            "native_lane_playbook": str(lane_playbook_path),
            "native_lane_stability_report": str(lane_stability_path),
            "public_quant_cross_section_research": str(public_quant_path),
        },
        "current_mainline": current_mainline,
        "orderflow_foundation": orderflow_foundation,
        "symbol_depth_ladder": symbol_depth_ladder,
        "deploy_now_symbols": deploy_now_symbols,
        "watch_priority_symbols": watch_priority_symbols,
        "watch_only_symbols": watch_only_symbols,
        "strategy_logic_stack": strategy_logic_stack,
        "feasibility_gates_before_backtest": feasibility_gates_before_backtest,
        "autoresearch_plan": autoresearch_plan,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "当前主线不再继续堆 entry filter；先把 orderflow 从底层拆成 schema、context、解释层、symbol depth 四段，"
            "确认逻辑可行后再进入回测与参数拟合。"
        ),
        "limitation_note": (
            "这份蓝图只做 research-only 策略架构，不代表 orderflow 数据已经补齐；"
            "在真实 lower-timeframe / taker/OI / microstructure sidecar 建好之前，不能把 flow 结论升级为 live 或大规模拟合前提。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_strategy_blueprint.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_strategy_blueprint.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_blueprint.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "latest_json_path": str(latest_json_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
