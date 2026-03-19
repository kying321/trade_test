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
BASE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_sim_only.py")


BASE_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_base", BASE_SCRIPT_PATH)
BASE_MODULE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE_MODULE)


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


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def optional_artifact(review_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(review_dir.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def objective_terms(metrics: dict[str, Any]) -> dict[str, Any]:
    trade_count = int(metrics.get("trade_count", 0) or 0)
    return {
        "ret_term": float(metrics.get("cumulative_return", 0.0) or 0.0) * 120.0,
        "sharpe_term": float(metrics.get("sharpe_per_trade", 0.0) or 0.0) * 8.0,
        "pf_term": min(float(metrics.get("profit_factor", 0.0) or 0.0), 5.0) * 4.0,
        "exp_term": float(metrics.get("expectancy_r", 0.0) or 0.0) * 10.0,
        "dd_term": -float(metrics.get("max_drawdown", 0.0) or 0.0) * 120.0,
        "trade_term": min(trade_count, 12) * 0.4,
        "objective": float(BASE_MODULE.objective(metrics)),
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_average_term_rows(hold_family_payload: dict[str, Any], config_ids: list[str]) -> dict[str, dict[str, Any]]:
    by_config: dict[str, dict[str, list[float]]] = {
        config_id: {
            "ret_term": [],
            "sharpe_term": [],
            "pf_term": [],
            "exp_term": [],
            "dd_term": [],
            "trade_term": [],
            "objective": [],
        }
        for config_id in config_ids
    }
    for grid_row in hold_family_payload.get("grid_rows", []):
        metrics_by_config = dict(grid_row.get("aggregate_selected_metrics_by_config") or {})
        for config_id in config_ids:
            metrics = dict(metrics_by_config.get(config_id) or {})
            terms = objective_terms(metrics)
            for key, value in terms.items():
                by_config[config_id][key].append(float(value))
    return {
        config_id: {key: round(mean(values), 6) for key, values in values_by_term.items()}
        for config_id, values_by_term in by_config.items()
    }


def build_frontier_rows(
    *,
    hold_family_payload: dict[str, Any],
    hold_robustness_payload: dict[str, Any],
    average_terms: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    overall = dict(hold_family_payload.get("overall_summary") or {})
    unique_ret = dict(overall.get("unique_slice_return_wins_total") or {})
    unique_obj = dict(overall.get("unique_slice_objective_wins_total") or {})
    agg_ret = dict(overall.get("aggregate_return_scheme_wins") or {})
    agg_obj = dict(overall.get("aggregate_objective_scheme_wins") or {})
    robust_summary = dict(hold_robustness_payload.get("overall_summary") or {})
    return [
        {
            "config_id": "hold16_zero",
            "role": "baseline_anchor",
            "why": (
                "旧 source artifact 仍以 hold16 为 baseline；"
                f"unique_slice_return_wins={to_int(unique_ret.get('hold16_zero'), 0)},"
                f"unique_slice_objective_wins={to_int(unique_obj.get('hold16_zero'), 0)}，"
                "说明它仍保留局部一致性锚点。"
            ),
            "evidence": {
                "aggregate_return_scheme_wins": to_int(agg_ret.get("hold16_zero"), 0),
                "aggregate_objective_scheme_wins": to_int(agg_obj.get("hold16_zero"), 0),
                "unique_slice_return_wins_total": to_int(unique_ret.get("hold16_zero"), 0),
                "unique_slice_objective_wins_total": to_int(unique_obj.get("hold16_zero"), 0),
                "legacy_hold_robustness_decision": text(hold_robustness_payload.get("research_decision")),
            },
            "average_objective_terms": dict(average_terms.get("hold16_zero") or {}),
        },
        {
            "config_id": "hold8_zero",
            "role": "objective_leader_candidate",
            "why": (
                f"aggregate_objective_scheme_wins={to_int(agg_obj.get('hold8_zero'), 0)} / {to_int(overall.get('grid_count'), 0)}，"
                f"且 unique_slice_wins(return/objective)=({to_int(unique_ret.get('hold8_zero'), 0)}/{to_int(unique_obj.get('hold8_zero'), 0)})；"
                "它是当前最强的 objective 候选。"
            ),
            "evidence": {
                "aggregate_return_scheme_wins": to_int(agg_ret.get("hold8_zero"), 0),
                "aggregate_objective_scheme_wins": to_int(agg_obj.get("hold8_zero"), 0),
                "unique_slice_return_wins_total": to_int(unique_ret.get("hold8_zero"), 0),
                "unique_slice_objective_wins_total": to_int(unique_obj.get("hold8_zero"), 0),
                "legacy_hold_robustness_agg8": to_int((robust_summary.get("aggregate_return_scheme_wins") or {}).get("hold_8_zero_risk"), 0),
            },
            "average_objective_terms": dict(average_terms.get("hold8_zero") or {}),
        },
        {
            "config_id": "hold24_zero",
            "role": "return_leader_candidate",
            "why": (
                f"aggregate_return_scheme_wins={to_int(agg_ret.get('hold24_zero'), 0)} / {to_int(overall.get('grid_count'), 0)}，"
                f"且 unique_slice_wins(return/objective)=({to_int(unique_ret.get('hold24_zero'), 0)}/{to_int(unique_obj.get('hold24_zero'), 0)})；"
                "它是当前最强的收益聚合候选。"
            ),
            "evidence": {
                "aggregate_return_scheme_wins": to_int(agg_ret.get("hold24_zero"), 0),
                "aggregate_objective_scheme_wins": to_int(agg_obj.get("hold24_zero"), 0),
                "unique_slice_return_wins_total": to_int(unique_ret.get("hold24_zero"), 0),
                "unique_slice_objective_wins_total": to_int(unique_obj.get("hold24_zero"), 0),
            },
            "average_objective_terms": dict(average_terms.get("hold24_zero") or {}),
        },
    ]


def build_dropped_rows(
    *,
    hold_family_payload: dict[str, Any],
    rider_triage_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    overall = dict(hold_family_payload.get("overall_summary") or {})
    unique_ret = dict(overall.get("unique_slice_return_wins_total") or {})
    unique_obj = dict(overall.get("unique_slice_objective_wins_total") or {})
    rider_rows = {text(row.get("candidate_id")): row for row in (rider_triage_payload.get("triage_summary") or {}).get("candidate_vs_baseline", [])}
    return [
        {
            "config_id": "hold12_zero",
            "reason": (
                f"hold12 在纯 hold 家族里 unique_slice_return_wins={to_int(unique_ret.get('hold12_zero'), 0)}，"
                f"unique_slice_objective_wins={to_int(unique_obj.get('hold12_zero'), 0)}；当前没有保留价值。"
            ),
        },
        {
            "config_id": "hold16_be075",
            "reason": (
                f"break-even rider 相对 baseline worse_objective_grids="
                f"{to_int((rider_rows.get('hold16_be075') or {}).get('worse_objective_grids'), 0)}。"
            ),
        },
        {
            "config_id": "hold16_trail15",
            "reason": (
                f"trailing rider 相对 baseline worse_objective_grids="
                f"{to_int((rider_rows.get('hold16_trail15') or {}).get('worse_objective_grids'), 0)}。"
            ),
        },
        {
            "config_id": "hold16_cd2x16",
            "reason": (
                f"cooldown rider 与 baseline exact_match="
                f"{bool((rider_rows.get('hold16_cd2x16') or {}).get('exact_metric_match_to_baseline_all_grids'))}。"
            ),
        },
    ]


def build_frontier_explanation(frontier_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_map = {text(row.get("config_id")): row for row in frontier_rows}
    hold8 = dict((row_map.get("hold8_zero") or {}).get("average_objective_terms") or {})
    hold16 = dict((row_map.get("hold16_zero") or {}).get("average_objective_terms") or {})
    hold24 = dict((row_map.get("hold24_zero") or {}).get("average_objective_terms") or {})
    return [
        {
            "point": "hold8_wins_objective",
            "detail": (
                "hold8 的 objective 优势主要来自更高的 pf/sharpe 与极低 dd；"
                f"avg_pf_term={to_float(hold8.get('pf_term'), 0.0):.3f}, "
                f"avg_sharpe_term={to_float(hold8.get('sharpe_term'), 0.0):.3f}, "
                f"avg_dd_term={to_float(hold8.get('dd_term'), 0.0):.3f}。"
            ),
        },
        {
            "point": "hold24_wins_return",
            "detail": (
                "hold24 的收益聚合领先主要来自更高 ret_term / exp_term；"
                f"avg_ret_term={to_float(hold24.get('ret_term'), 0.0):.3f}, "
                f"avg_exp_term={to_float(hold24.get('exp_term'), 0.0):.3f}。"
            ),
        },
        {
            "point": "hold24_fails_objective",
            "detail": (
                "hold24 没赢 objective，是因为更高回撤和较弱的 sharpe/pf 抵消了收益项；"
                f"avg_dd_term={to_float(hold24.get('dd_term'), 0.0):.3f}, "
                f"vs hold8 dd_term={to_float(hold8.get('dd_term'), 0.0):.3f}, "
                f"avg_pf_term={to_float(hold24.get('pf_term'), 0.0):.3f}。"
            ),
        },
        {
            "point": "hold16_role",
            "detail": (
                "hold16 当前更像 baseline anchor，而不是 aggregate leader；"
                f"avg_objective={to_float(hold16.get('objective'), 0.0):.3f}，"
                "但它仍保留较多 unique slice wins。"
            ),
        },
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Price Action Breakout Pullback Hold Frontier Report SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        f"- source_head_status: `{text(payload.get('source_head_status'))}`",
        f"- canonical_source_head: `{text(payload.get('canonical_source_head'))}`",
        "",
        "## Frontier",
        "",
    ]
    for row in payload.get("frontier_rows", []):
        lines.append(
            f"- `{row['config_id']}` | role=`{row['role']}` | why=`{row['why']}` | "
            f"avg_objective=`{float((row.get('average_objective_terms') or {}).get('objective', 0.0)):.3f}`"
        )
    lines.extend(["", "## Dropped", ""])
    for row in payload.get("dropped_rows", []):
        lines.append(f"- `{row['config_id']}` | reason=`{row['reason']}`")
    lines.extend(["", "## Frontier Explanation", ""])
    for row in payload.get("frontier_explanation", []):
        lines.append(f"- `{row['point']}` | detail=`{row['detail']}`")
    lines.extend(["", "## Next Steps", ""])
    for row in payload.get("next_steps", []):
        lines.append(f"- `{row}`")
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
        description="Build a SIM_ONLY frontier report for the ETH price-state hold family after dropping simple riders."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--base-artifact-path", default="")
    parser.add_argument("--hold-robustness-path", default="")
    parser.add_argument("--hold-family-triage-path", default="")
    parser.add_argument("--rider-triage-path", default="")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def resolve_path(review_dir: Path, explicit: str, pattern: str) -> Path:
    return Path(explicit).expanduser().resolve() if text(explicit) else latest_artifact(review_dir, pattern)


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    base_artifact_path = resolve_path(review_dir, args.base_artifact_path, "*_price_action_breakout_pullback_sim_only.json")
    hold_robustness_path = resolve_path(review_dir, args.hold_robustness_path, "*_price_action_breakout_pullback_exit_hold_robustness_sim_only.json")
    hold_family_triage_path = resolve_path(review_dir, args.hold_family_triage_path, "*_price_action_breakout_pullback_hold_family_triage_sim_only.json")
    rider_triage_path = resolve_path(review_dir, args.rider_triage_path, "*_price_action_breakout_pullback_exit_rider_triage_sim_only.json")

    base_artifact = load_json_mapping(base_artifact_path)
    hold_robustness_payload = load_json_mapping(hold_robustness_path)
    hold_family_payload = load_json_mapping(hold_family_triage_path)
    rider_triage_payload = load_json_mapping(rider_triage_path)
    handoff_path = optional_artifact(review_dir, "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json")
    handoff_payload = load_json_mapping(handoff_path) if handoff_path else {}

    config_ids = ["hold8_zero", "hold16_zero", "hold24_zero"]
    average_terms = build_average_term_rows(hold_family_payload, config_ids=config_ids)
    frontier_rows = build_frontier_rows(
        hold_family_payload=hold_family_payload,
        hold_robustness_payload=hold_robustness_payload,
        average_terms=average_terms,
    )
    dropped_rows = build_dropped_rows(
        hold_family_payload=hold_family_payload,
        rider_triage_payload=rider_triage_payload,
    )
    frontier_explanation = build_frontier_explanation(frontier_rows)
    research_decision = "freeze_hold16_baseline_with_dual_candidates_hold8_objective_hold24_return"
    source_head_status = "frontier_head_active"
    canonical_source_head = str(review_dir / "latest_price_action_breakout_pullback_hold_frontier_report_sim_only.json")
    consumer_rule = "可直接读取 frontier report 作为当前 hold frontier head。"
    if text(handoff_payload.get("research_decision")) == "use_hold_selection_gate_as_canonical_head":
        source_head_status = "superseded_by_hold_selection_handoff"
        canonical_source_head = text(handoff_payload.get("canonical_source_head")) or str(handoff_path)
        consumer_rule = text(handoff_payload.get("consumer_rule")) or consumer_rule
    recommended_brief = (
        f"{text(base_artifact.get('focus_symbol'))}:hold_frontier:{BASE_MODULE.SELECTION_SCENARIO_ID}:"
        "baseline=hold16_zero,"
        "objective_candidate=hold8_zero,"
        "return_candidate=hold24_zero,"
        "drop=hold12_zero,hold16_be075,hold16_trail15,hold16_cd2x16,"
        f"head_status={source_head_status},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_price_action_breakout_pullback_hold_frontier_report_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "base_artifact_path": str(base_artifact_path),
            "hold_robustness_path": str(hold_robustness_path),
            "hold_family_triage_path": str(hold_family_triage_path),
            "rider_triage_path": str(rider_triage_path),
        },
        "current_mainline": {
            "symbol": text(base_artifact.get("focus_symbol")),
            "family": "price_action_breakout_pullback",
            "selected_params": dict(base_artifact.get("selected_params") or {}),
        },
        "source_head_status": source_head_status,
        "canonical_source_head": canonical_source_head,
        "consumer_rule": consumer_rule,
        "superseded_by_handoff_path": str(handoff_path) if handoff_path else "",
        "frontier_rows": frontier_rows,
        "dropped_rows": dropped_rows,
        "frontier_explanation": frontier_explanation,
        "next_steps": [
            "继续保留 hold16 作为 baseline source-of-truth，不直接替换。",
            "后续 price-state-only 研究只围绕 hold8 与 hold24 做候选对比，不再回到 hold12 或 simple riders。",
            "如需再前推，只做 baseline vs hold8 vs hold24 的 source-owned frontier compare。",
        ],
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份 frontier report 把当前 hold 家族正式收敛成 baseline + 双候选结构，"
            "避免后续研究继续在已淘汰配置上打转。"
            + (" 但当 hold_selection_handoff 已存在时，这份工件只保留为历史 frontier 证据，不再是 canonical head。"
               if source_head_status == "superseded_by_hold_selection_handoff" else "")
        ),
        "limitation_note": (
            "它只是在现有 60d ETH 15m SIM_ONLY 证据上固化 frontier role；"
            "并不等于已经选出最终替代 baseline 的唯一赢家。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_frontier_report_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_frontier_report_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_frontier_report_sim_only.json"
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
