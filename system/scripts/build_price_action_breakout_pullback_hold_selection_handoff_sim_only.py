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
        description="Build a SIM_ONLY canonical handoff for ETH hold selection after transfer-driven gate override."
    )
    parser.add_argument("--gate-blocker-path", required=True)
    parser.add_argument("--frontier-report-path", required=True)
    parser.add_argument("--family-transfer-path", required=True)
    parser.add_argument("--router-transfer-path", required=True)
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


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Hold Selection Handoff SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Canonical Source Head",
        "",
        f"- canonical_source_head: `{text(payload.get('canonical_source_head'))}`",
        f"- source_head_status: `{text(payload.get('source_head_status'))}`",
        f"- superseded_head: `{text(payload.get('superseded_head'))}`",
        "",
        "## Active Selection State",
        "",
        f"- active_baseline: `{text(payload.get('active_baseline'))}`",
        f"- local_candidate: `{text(payload.get('local_candidate'))}`",
        f"- transfer_watch: `{json.dumps(payload.get('transfer_watch') or [], ensure_ascii=False)}`",
        f"- demoted_candidate: `{json.dumps(payload.get('demoted_candidate') or [], ensure_ascii=False)}`",
        "",
        "## Consumer Rule",
        "",
        f"- `{text(payload.get('consumer_rule'))}`",
        "",
        "## Blocked Now",
        "",
    ]
    for row in payload.get("blocked_now", []):
        lines.append(f"- `{text(row)}`")
    lines.extend(["", "## Allowed Now", ""])
    for row in payload.get("allowed_now", []):
        lines.append(f"- `{text(row)}`")
    lines.extend(["", "## Release Conditions", ""])
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

    gate_blocker_path = Path(args.gate_blocker_path).expanduser().resolve()
    frontier_report_path = Path(args.frontier_report_path).expanduser().resolve()
    family_transfer_path = Path(args.family_transfer_path).expanduser().resolve()
    router_transfer_path = Path(args.router_transfer_path).expanduser().resolve()

    gate = load_json_mapping(gate_blocker_path)
    frontier = load_json_mapping(frontier_report_path)
    family_transfer = load_json_mapping(family_transfer_path)
    router_transfer = load_json_mapping(router_transfer_path)

    gate_state = dict(gate.get("gate_state") or {})
    override_required = text(gate_state.get("source_head_override_required")) == "yes_transfer_evidence_overrides_old_frontier_head"
    research_decision = "hold_selection_handoff_inconclusive"
    source_head_status = "inconclusive"
    if override_required:
        research_decision = "use_hold_selection_gate_as_canonical_head"
        source_head_status = "gate_override_active"

    payload = {
        "action": "build_price_action_breakout_pullback_hold_selection_handoff_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "gate_blocker_path": str(gate_blocker_path),
        "frontier_report_path": str(frontier_report_path),
        "family_transfer_path": str(family_transfer_path),
        "router_transfer_path": str(router_transfer_path),
        "canonical_source_head": str(gate_blocker_path),
        "source_head_status": source_head_status,
        "superseded_head": str(frontier_report_path),
        "source_evidence": {
            "gate_research_decision": text(gate.get("research_decision")),
            "frontier_research_decision": text(frontier.get("research_decision")),
            "family_transfer_research_decision": text(family_transfer.get("research_decision")),
            "router_transfer_research_decision": text(router_transfer.get("research_decision")),
        },
        "active_baseline": text(gate.get("active_baseline")),
        "local_candidate": text(gate.get("local_candidate")),
        "transfer_watch": list(gate.get("transfer_watch") or []),
        "demoted_candidate": list(gate.get("demoted_candidate") or []),
        "blocked_now": list(gate.get("blocked_now") or []),
        "allowed_now": list(gate.get("allowed_now") or []),
        "release_conditions": list(gate.get("release_conditions") or []),
        "consumer_rule": (
            "后续所有 ETH hold selection brief / review / consumer 必须先读取 "
            "`latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json`；"
            "不得再把旧 frontier report 单独当成当前 head。"
        ),
        "handoff_state": {
            "selection_mode": "conservative_anchor_with_local_candidate_and_transfer_watch",
            "baseline_scope": "current_mainline_anchor_only",
            "candidate_scope": "local_window_only",
            "transfer_scope": "watch_only",
            "promotion_state": "blocked",
        },
        "research_decision": research_decision,
        "recommended_brief": (
            "ETHUSDT:hold_selection_handoff:"
            f"head={text(gate.get('research_decision'))},"
            f"frontier={text(frontier.get('research_decision'))},"
            f"family_transfer={text(family_transfer.get('research_decision'))},"
            f"router_transfer={text(router_transfer.get('research_decision'))},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 handoff 的目的不是产生新策略结论，而是消除 consumer drift；"
            "后续只需读一个 canonical head。"
        ),
        "limitation_note": (
            "handoff 依赖上游 gate / transfer 工件；它本身不新增 OOS 证据，"
            "只是把最新 source-owned 结论收敛成单一读取入口。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_selection_handoff_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
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
