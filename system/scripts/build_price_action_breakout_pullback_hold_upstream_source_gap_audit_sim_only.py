#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


ARTIFACT_SCRIPT_PAIRS: list[tuple[str, str, str]] = [
    (
        "hold_family_transfer",
        "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json",
        "build_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.py",
    ),
    (
        "hold_frontier_cost_sensitivity",
        "latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json",
        "build_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.py",
    ),
    (
        "hold_frontier_report",
        "latest_price_action_breakout_pullback_hold_frontier_report_sim_only.json",
        "build_price_action_breakout_pullback_hold_frontier_report_sim_only.py",
    ),
    (
        "hold_router_hypothesis",
        "latest_price_action_breakout_pullback_hold_router_hypothesis_sim_only.json",
        "build_price_action_breakout_pullback_hold_router_hypothesis_sim_only.py",
    ),
    (
        "hold_router_transfer",
        "latest_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json",
        "build_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.py",
    ),
    (
        "exit_hold_window_consensus",
        "latest_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.json",
        "build_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.py",
    ),
    (
        "exit_hold_forward_window_capacity",
        "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json",
        "build_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.py",
    ),
    (
        "exit_hold_overlap_sidecar",
        "latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json",
        "build_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.py",
    ),
    (
        "hold_selection_handoff",
        "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py",
    ),
    (
        "exit_hold_forward_stop_condition",
        "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json",
        "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit hold upstream source gaps between current latest artifacts and builder source files."
    )
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--review-dir", default="", help="Optional explicit review directory override.")
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


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot_resolve_system_root:{workspace}")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Hold Upstream Source Gap Audit SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- finding_count: `{int(payload.get('finding_count') or 0)}`",
        f"- missing_builder_labels: `{json.dumps(payload.get('missing_builder_labels') or [], ensure_ascii=False)}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload.get("findings", []):
        lines.extend(
            [
                f"### {text(row.get('label'))}",
                f"- latest_artifact: `{text(row.get('latest_artifact'))}`",
                f"- builder_script: `{text(row.get('builder_script'))}`",
                f"- issue: `{text(row.get('issue'))}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    scripts_dir = system_root / "scripts"
    review_dir = Path(args.review_dir).expanduser().resolve() if text(args.review_dir) else system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for label, latest_name, script_name in ARTIFACT_SCRIPT_PAIRS:
        latest_artifact = review_dir / latest_name
        builder_script = scripts_dir / script_name
        latest_exists = latest_artifact.exists()
        script_exists = builder_script.exists()
        coverage_rows.append(
            {
                "label": label,
                "latest_artifact": str(latest_artifact),
                "latest_exists": latest_exists,
                "builder_script": str(builder_script),
                "builder_script_exists": script_exists,
            }
        )
        if latest_exists and not script_exists:
            findings.append(
                {
                    "label": label,
                    "latest_artifact": str(latest_artifact),
                    "builder_script": str(builder_script),
                    "issue": "latest_artifact_exists_but_builder_script_missing",
                }
            )

    findings.sort(key=lambda row: text(row.get("label")))
    missing_builder_labels = [text(row.get("label")) for row in findings]
    research_decision = (
        "hold_upstream_source_gap_detected_missing_builder_sources"
        if findings
        else "hold_upstream_builder_sources_present_for_current_latest_artifacts"
    )

    payload = {
        "action": "build_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "workspace": str(workspace),
        "review_dir": str(review_dir),
        "research_decision": research_decision,
        "finding_count": len(findings),
        "missing_builder_labels": missing_builder_labels,
        "findings": findings,
        "coverage_rows": coverage_rows,
        "recommended_brief": (
            f"hold_upstream_source_gap:findings={len(findings)},"
            f"missing={','.join(missing_builder_labels) or 'none'},"
            f"decision={research_decision}"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "latest_json_path": str(latest_json_path),
                "latest_md_path": str(latest_md_path),
                "research_decision": research_decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
