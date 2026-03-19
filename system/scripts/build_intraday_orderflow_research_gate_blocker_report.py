#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def build_evidence_rows(
    *,
    base_artifact: dict[str, Any],
    hold_compare_artifact: dict[str, Any],
    context_pack: dict[str, Any],
    veto_event_study: dict[str, Any],
    casebook: dict[str, Any],
    execution_checklist: dict[str, Any],
) -> list[dict[str, Any]]:
    harmful_rules = [
        text(row.get("rule_name"))
        for row in veto_event_study.get("eth_veto_rules_48h", [])
        if text(row.get("classification")) in {"harmful_on_current_sample", "overblocking_current_sample"}
    ]
    symbol_gap_rows = {text(row.get("symbol")): row for row in casebook.get("symbol_gap_rows", [])}
    eth_gap = symbol_gap_rows.get("ETHUSDT", {})
    btc_gap = symbol_gap_rows.get("BTCUSDT", {})
    return [
        {
            "evidence_id": "eth_price_state_backbone_retained",
            "status": "pass",
            "detail": (
                f"ETH 基础 price-state artifact 仍是 `{text(base_artifact.get('validation_status'))}`；"
                f"focus_symbol={text(base_artifact.get('focus_symbol'))}。"
            ),
        },
        {
            "evidence_id": "exit_hold_forward_still_mixed",
            "status": "warn",
            "detail": (
                "exit/risk 前推仍是 mixed："
                f"{text(hold_compare_artifact.get('research_decision'))}。"
            ),
        },
        {
            "evidence_id": "orderflow_alignment_not_backtest_ready",
            "status": "fail",
            "detail": (
                f"ETH 12h 覆盖={to_float((((context_pack.get('eth_signal_alignment') or {}).get('coverage_by_window') or {}).get('within_12h') or {}).get('ratio'), 0.0):.2%}，"
                f"24h 覆盖={to_float((((context_pack.get('eth_signal_alignment') or {}).get('coverage_by_window') or {}).get('within_24h') or {}).get('ratio'), 0.0):.2%}；"
                f"结论={text(context_pack.get('research_decision'))}。"
            ),
        },
        {
            "evidence_id": "hard_veto_rules_not_surviving_sample",
            "status": "fail",
            "detail": (
                f"ETH 48h covered sample={int(veto_event_study.get('eth_context_covered_48h_count') or 0)}；"
                f"有害/过度阻断规则={','.join(harmful_rules) or '-'}。"
            ),
        },
        {
            "evidence_id": "majors_capture_gap_persists",
            "status": "fail",
            "detail": (
                f"ETH gap={text(eth_gap.get('gap_priority'))} "
                f"(24h={to_float(eth_gap.get('coverage_24h_ratio'), 0.0):.2%},48h={to_float(eth_gap.get('coverage_48h_ratio'), 0.0):.2%}); "
                f"BTC gap={text(btc_gap.get('gap_priority'))} "
                f"(24h={to_float(btc_gap.get('coverage_24h_ratio'), 0.0):.2%},48h={to_float(btc_gap.get('coverage_48h_ratio'), 0.0):.2%})。"
            ),
        },
        {
            "evidence_id": "historical_micro_backfill_blocked",
            "status": "fail",
            "detail": (
                f"execution checklist 已固化 `{text(execution_checklist.get('research_decision'))}`；"
                "当前 orderflow 缺口不可用现有 CLI/engine 做历史回填。"
            ),
        },
    ]


def build_release_conditions(execution_checklist: dict[str, Any]) -> list[dict[str, Any]]:
    policy = execution_checklist.get("prospective_capture_policy") or {}
    thresholds = policy.get("done_when_thresholds") or {}
    min_runs = int(thresholds.get("execution_micro_capture_min_runs") or 4)
    lookback_days = int(thresholds.get("execution_micro_capture_lookback_days") or 7)
    return [
        {
            "condition_id": "future_capture_evidence_upgrade",
            "status": "pending",
            "detail": (
                f"未来 {lookback_days} 天内，ETH/BTC 围绕目标时段至少各获得 {min_runs} 次 prospective capture，"
                "且 rolling stats 达到 config 阈值。"
            ),
        },
        {
            "condition_id": "majors_trust_upgrade",
            "status": "pending",
            "detail": "ETH/BTC 不再停留在 `single_exchange_low`，能够给出更高 trust 的 context 字段。",
        },
        {
            "condition_id": "hard_veto_retest_with_larger_sample",
            "status": "pending",
            "detail": "在更大、覆盖更好的 ETH sample 上重新验证 veto 原型，至少不能继续表现为 harmful/overblocking。",
        },
        {
            "condition_id": "alternative_research_only_historical_source",
            "status": "optional_path",
            "detail": "若引入新的纯 research-only 历史 orderflow 数据源，也可替代现有 micro-capture 缺口路径。",
        },
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Intraday Orderflow Research Gate Blocker Report",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Gate State",
        "",
    ]
    for row in payload.get("gate_state", []):
        lines.append(f"- `{row['gate']}` | status=`{row['status']}` | detail=`{row['detail']}`")
    lines.extend(["", "## Evidence", ""])
    for row in payload.get("evidence_rows", []):
        lines.append(f"- `{row['evidence_id']}` | status=`{row['status']}` | detail=`{row['detail']}`")
    lines.extend(["", "## Allowed Next Steps", ""])
    for row in payload.get("allowed_next_steps", []):
        lines.append(f"- `{row}`")
    lines.extend(["", "## Forbidden Until Gate Upgrade", ""])
    for row in payload.get("forbidden_now", []):
        lines.append(f"- `{row}`")
    lines.extend(["", "## Release Conditions", ""])
    for row in payload.get("release_conditions", []):
        lines.append(f"- `{row['condition_id']}` | status=`{row['status']}` | detail=`{row['detail']}`")
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
        description="Build a RESEARCH_ONLY blocker report that freezes current orderflow gate state and clarifies what can continue next."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--base-artifact-path", default="")
    parser.add_argument("--hold-compare-path", default="")
    parser.add_argument("--context-pack-path", default="")
    parser.add_argument("--veto-event-study-path", default="")
    parser.add_argument("--coverage-gap-casebook-path", default="")
    parser.add_argument("--execution-checklist-path", default="")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def resolve_path(review_dir: Path, explicit: str, pattern: str) -> Path:
    return Path(explicit).expanduser().resolve() if text(explicit) else latest_artifact(review_dir, pattern)


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    base_path = resolve_path(review_dir, args.base_artifact_path, "*_price_action_breakout_pullback_sim_only.json")
    hold_path = resolve_path(review_dir, args.hold_compare_path, "*_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json")
    context_pack_path = resolve_path(review_dir, args.context_pack_path, "*_intraday_orderflow_context_veto_research_pack.json")
    veto_path = resolve_path(review_dir, args.veto_event_study_path, "*_intraday_orderflow_veto_event_study.json")
    casebook_path = resolve_path(review_dir, args.coverage_gap_casebook_path, "*_intraday_orderflow_coverage_gap_casebook.json")
    execution_checklist_path = resolve_path(review_dir, args.execution_checklist_path, "*_intraday_orderflow_majors_capture_execution_checklist.json")

    base_artifact = load_json_mapping(base_path)
    hold_compare_artifact = load_json_mapping(hold_path)
    context_pack = load_json_mapping(context_pack_path)
    veto_event_study = load_json_mapping(veto_path)
    casebook = load_json_mapping(casebook_path)
    execution_checklist = load_json_mapping(execution_checklist_path)

    evidence_rows = build_evidence_rows(
        base_artifact=base_artifact,
        hold_compare_artifact=hold_compare_artifact,
        context_pack=context_pack,
        veto_event_study=veto_event_study,
        casebook=casebook,
        execution_checklist=execution_checklist,
    )
    gate_state = [
        {
            "gate": "price_state_only_replay",
            "status": "allowed_with_existing_scope",
            "detail": "可以继续 ETH price-state 主线的 source-owned replay/exit-risk 比较，但不能混入新的 orderflow 衍生过滤器。",
        },
        {
            "gate": "orderflow_context_to_backtest",
            "status": "blocked",
            "detail": "当前 orderflow 只能停留在 context/veto research field，不能进入 replay/backtest 输入。",
        },
        {
            "gate": "orderflow_to_parameter_fitting",
            "status": "blocked",
            "detail": "当前禁止把 orderflow 特征拉进参数拟合，因为 alignment/trust/sample/backfill 四个条件都没过。",
        },
        {
            "gate": "historical_micro_backfill",
            "status": "blocked",
            "detail": "现有 CLI/engine/config 组合下，micro-capture 只能做 prospective reference，不能回填旧事件。",
        },
        {
            "gate": "beta_symbol_promotion",
            "status": "blocked",
            "detail": "BNB/SOL 仍停留在 beta watch，不允许升级成统一主线模板消费者。",
        },
    ]
    allowed_next_steps = [
        "继续维护 ETHUSDT 15m price_action_breakout_pullback 作为唯一主 backbone。",
        "若要继续 price-state 研究，只做 orderflow-free 的 exit/risk 或 forward compare。",
        "把 ETH/BTC orderflow 缺口继续保留为 source-owned future capture planning，不执行 micro-capture。",
        "如果后续出现新的纯 research-only 历史 orderflow 数据源，再单独接入 sidecar 评估链。",
    ]
    forbidden_now = [
        "启动任何基于当前 orderflow sidecar 的新 replay/backtest。",
        "对 orderflow 特征做参数因子拟合。",
        "把当前 hard veto 原型提升为交易规则。",
        "把历史缺口当成可由当前 micro-capture CLI/daemon 回填的任务。",
        "把 BNBUSDT 或 SOLUSDT 提升为当前主线。",
    ]
    release_conditions = build_release_conditions(execution_checklist)
    research_decision = "block_orderflow_replay_and_fitting_keep_eth_price_state_only_until_evidence_upgrade"
    recommended_brief = (
        "orderflow_research_gate_blocker:"
        "price_state_only=allowed,"
        "orderflow_replay=blocked,"
        "orderflow_fitting=blocked,"
        f"release={','.join(row['condition_id'] for row in release_conditions[:3])},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_intraday_orderflow_research_gate_blocker_report",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "base_artifact_path": str(base_path),
            "hold_compare_path": str(hold_path),
            "context_pack_path": str(context_pack_path),
            "veto_event_study_path": str(veto_path),
            "coverage_gap_casebook_path": str(casebook_path),
            "execution_checklist_path": str(execution_checklist_path),
        },
        "current_mainline": {
            "symbol": text(base_artifact.get("focus_symbol")),
            "family": "price_action_breakout_pullback",
            "validation_status": text(base_artifact.get("validation_status")),
            "selected_params": dict(base_artifact.get("selected_params") or {}),
            "exit_hold_forward_decision": text(hold_compare_artifact.get("research_decision")),
        },
        "gate_state": gate_state,
        "evidence_rows": evidence_rows,
        "allowed_next_steps": allowed_next_steps,
        "forbidden_now": forbidden_now,
        "release_conditions": release_conditions,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份 blocker report 把当前双轨结论固定下来：price-state 主线还能继续，"
            "但 orderflow-derived replay/fitting 现在必须明确阻断。"
        ),
        "limitation_note": (
            "它不会提升样本覆盖或 trust，只负责把现阶段的 no-go 边界写成 source-owned 约束。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_research_gate_blocker_report.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_research_gate_blocker_report.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_research_gate_blocker_report.json"
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
