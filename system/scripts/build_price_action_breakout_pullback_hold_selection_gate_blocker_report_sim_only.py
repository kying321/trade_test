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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY blocker report for ETH hold-family selection after transfer-drift checks."
    )
    parser.add_argument("--frontier-report-path", required=True)
    parser.add_argument("--frontier-cost-path", required=True)
    parser.add_argument("--router-hypothesis-path", required=True)
    parser.add_argument("--router-transfer-path", required=True)
    parser.add_argument("--family-transfer-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def build_frontier_role_map(frontier_report: dict[str, Any]) -> dict[str, str]:
    rows = frontier_report.get("frontier_rows")
    result: dict[str, str] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            config_id = text(row.get("config_id"))
            role = text(row.get("role"))
            if config_id and role:
                result[config_id] = role
    if result:
        return result

    brief = text(frontier_report.get("recommended_brief"))
    brief_map: dict[str, str] = {}
    for token in brief.split(","):
        key, _, value = token.partition("=")
        if key and value:
            brief_map[key.rsplit(":", 1)[-1].strip()] = value.strip()
    if brief_map.get("objective_candidate") == "hold8_zero":
        result["hold8_zero"] = "objective_leader_candidate"
    elif brief_map.get("objective_watch") == "hold8_zero":
        result["hold8_zero"] = "objective_watch_candidate"
    if brief_map.get("return_candidate") == "hold24_zero":
        result["hold24_zero"] = "return_leader_candidate"
    elif brief_map.get("return_watch") == "hold24_zero":
        result["hold24_zero"] = "return_watch_candidate"
    if brief_map.get("transfer_watch") == "hold12_zero":
        result["hold12_zero"] = "transfer_watch_candidate"
    return result


def singleton_list(enabled: bool, value: str) -> list[str]:
    return [value] if enabled and value else []


def classify_hold24_promotion(*, hold24_role: str, family_transfer_decision: str) -> str:
    if hold24_role not in {"return_leader_candidate", "return_watch_candidate"}:
        return "blocked_frontier_no_active_return_candidate"
    if "demotes_hold24" in family_transfer_decision:
        return "blocked_transfer_demoted"
    if hold24_role == "return_leader_candidate":
        return "blocked_return_candidate_until_longer_forward_oos"
    return "blocked_return_watch_only"


def classify_hold12_drop(*, hold12_active: bool, family_transfer_decision: str) -> str:
    if "revives_hold12" in family_transfer_decision or "revived_in_transfer_watch_only" in family_transfer_decision:
        return "blocked_transfer_revived_watch_only"
    if hold12_active:
        return "blocked_transfer_watch_only"
    return "allowed_no_transfer_watch_remaining"


def classify_router_promotion(*, router_transfer_decision: str, router_hypothesis_decision: str) -> str:
    if "positive_on_historical_transfer_but_future_tail_insufficient" in router_transfer_decision:
        return "blocked_future_tail_insufficient_after_positive_historical_transfer"
    if "does_not_beat_hold8" in router_transfer_decision:
        return "blocked_transfer_does_not_beat_hold8"
    if "historical_transfer_unavailable" in router_transfer_decision:
        return "blocked_transfer_unavailable"
    if "same_sample_only" in router_hypothesis_decision or "requires_new_forward_challenge" in router_hypothesis_decision:
        return "blocked_same_sample_only_until_transfer_challenge"
    return "blocked_transfer_failed"


def render_markdown(payload: dict[str, Any]) -> str:
    gates = dict(payload.get("gate_state") or {})
    lines = [
        "# Price Action Breakout Pullback Hold Selection Gate Blocker Report SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Gate State",
        "",
    ]
    for key, value in gates.items():
        lines.append(f"- {key}: `{text(value)}`")
    lines.extend(
        [
            "",
            "## Current Source-Owned Roles",
            "",
            f"- active_baseline: `{text(payload.get('active_baseline'))}`",
            f"- local_candidate: `{text(payload.get('local_candidate'))}`",
            f"- transfer_watch: `{json.dumps(payload.get('transfer_watch') or [], ensure_ascii=False)}`",
            f"- return_candidate: `{json.dumps(payload.get('return_candidate') or [], ensure_ascii=False)}`",
            f"- return_watch: `{json.dumps(payload.get('return_watch') or [], ensure_ascii=False)}`",
            f"- demoted_candidate: `{json.dumps(payload.get('demoted_candidate') or [], ensure_ascii=False)}`",
            "",
            "## Release Conditions",
            "",
        ]
    )
    for row in payload.get("release_conditions", []):
        lines.append(f"- `{text(row)}`")
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


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    frontier_report_path = Path(args.frontier_report_path).expanduser().resolve()
    frontier_cost_path = Path(args.frontier_cost_path).expanduser().resolve()
    router_hypothesis_path = Path(args.router_hypothesis_path).expanduser().resolve()
    router_transfer_path = Path(args.router_transfer_path).expanduser().resolve()
    family_transfer_path = Path(args.family_transfer_path).expanduser().resolve()

    frontier_report = load_json_mapping(frontier_report_path)
    frontier_cost = load_json_mapping(frontier_cost_path)
    router_hypothesis = load_json_mapping(router_hypothesis_path)
    router_transfer = load_json_mapping(router_transfer_path)
    family_transfer = load_json_mapping(family_transfer_path)

    frontier_roles = build_frontier_role_map(frontier_report)
    hold8_role = text(frontier_roles.get("hold8_zero"))
    hold12_role = text(frontier_roles.get("hold12_zero"))
    hold24_role = text(frontier_roles.get("hold24_zero"))
    router_hypothesis_decision = text(router_hypothesis.get("research_decision"))
    router_transfer_decision = text(router_transfer.get("research_decision"))
    family_transfer_decision = text(family_transfer.get("research_decision"))

    transfer_watch = singleton_list(
        hold12_role == "transfer_watch_candidate" or "hold12" in family_transfer_decision,
        "hold12_zero",
    )
    return_candidate = singleton_list(hold24_role == "return_leader_candidate", "hold24_zero")
    return_watch = singleton_list(hold24_role == "return_watch_candidate", "hold24_zero")
    hold24_promotion = classify_hold24_promotion(
        hold24_role=hold24_role,
        family_transfer_decision=family_transfer_decision,
    )
    hold12_drop = classify_hold12_drop(
        hold12_active=bool(transfer_watch),
        family_transfer_decision=family_transfer_decision,
    )
    router_promotion = classify_router_promotion(
        router_transfer_decision=router_transfer_decision,
        router_hypothesis_decision=router_hypothesis_decision,
    )

    gate_state = {
        "hold16_baseline_anchor": "allowed",
        "hold8_promotion": "blocked_until_longer_forward_oos" if hold8_role else "blocked_frontier_no_local_candidate",
        "hold24_promotion": hold24_promotion,
        "hold12_global_drop": hold12_drop,
        "dynamic_router_promotion": router_promotion,
        "new_hold_param_fitting": "blocked_until_frontier_reconverges",
        "source_head_override_required": "yes_transfer_evidence_overrides_old_frontier_head",
    }

    research_decision = (
        "block_hold_candidate_promotion_keep_hold16_anchor_reopen_hold12_watch_demote_hold24_and_router"
    )

    payload = {
        "action": "build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "frontier_report_path": str(frontier_report_path),
        "frontier_cost_path": str(frontier_cost_path),
        "router_hypothesis_path": str(router_hypothesis_path),
        "router_transfer_path": str(router_transfer_path),
        "family_transfer_path": str(family_transfer_path),
        "source_evidence": {
            "frontier_report_research_decision": text(frontier_report.get("research_decision")),
            "frontier_cost_research_decision": text(frontier_cost.get("research_decision")),
            "router_hypothesis_research_decision": router_hypothesis_decision,
            "router_transfer_research_decision": router_transfer_decision,
            "family_transfer_research_decision": family_transfer_decision,
        },
        "gate_state": gate_state,
        "active_baseline": "hold16_zero",
        "local_candidate": "hold8_zero" if hold8_role else "",
        "transfer_watch": transfer_watch,
        "return_candidate": return_candidate,
        "return_watch": return_watch,
        "demoted_candidate": [
            *singleton_list(hold24_promotion.startswith("blocked_"), "hold24_zero"),
            *singleton_list(router_promotion.startswith("blocked_"), "pullback_depth_atr_router"),
        ],
        "blocked_now": [
            "promote_hold8_as_new_baseline",
            "promote_hold24_as_return_candidate_beyond_local_window",
            "drop_hold12_globally",
            "promote_pullback_depth_atr_router",
            "fit_new_hold_router_or_new_hold_grid_before_longer_forward_oos",
        ],
        "allowed_now": [
            "keep_hold16_as_current_baseline_anchor",
            *singleton_list(bool(hold8_role), "keep_hold8_as_local_window_candidate_only"),
            *singleton_list(bool(transfer_watch), "treat_hold12_as_transfer_watch_only"),
            "treat_hold24_as_demoted_until_new_forward_evidence",
            "collect_longer_forward_tail_and_re-run_transfer_or_forward_challenge",
        ],
        "release_conditions": [
            "future tail 增长到足够形成新的 non-overlap forward challenge，而不是仅 36 根 15m bars",
            "hold8 或 hold24 在新的 forward OOS 上稳定胜过 hold16，而不是只在 derivation 或 historical transfer 的单侧窗口成立",
            "hold12 不再在 transfer 窗口复活，或其复活被更长 OOS 证据否定",
            "router frozen-threshold challenge 在新的非重叠 forward OOS 上实证打赢 pure hold8",
        ],
        "research_decision": research_decision,
        "recommended_brief": (
            "ETHUSDT:hold_selection_gate:"
            f"frontier={text(frontier_report.get('research_decision'))},"
            f"router_transfer={router_transfer_decision},"
            f"family_transfer={family_transfer_decision},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 blocker 把旧 frontier 头与最新 transfer 证据做 source-owned 合并；"
            "目标不是再选新 winner，而是先阻断错误 promotion。"
        ),
        "limitation_note": (
            "当前 drift 证据主要来自 historical transfer，因为 derivation 后只有 36 根 15m future bars；"
            "所以 blocker 的重点是冻结 promotion，而不是宣告最终新 frontier。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.json"
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
