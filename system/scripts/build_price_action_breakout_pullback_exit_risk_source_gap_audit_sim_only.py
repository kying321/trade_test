#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


ARTIFACT_SCRIPT_PAIRS: list[tuple[str, str, str]] = [
    (
        "exit_risk_selected",
        "latest_price_action_breakout_pullback_exit_risk_sim_only.json",
        "build_price_action_breakout_pullback_exit_risk_sim_only.py",
    ),
    (
        "forward_blocker",
        "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py",
    ),
    (
        "forward_consensus",
        "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json",
        "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py",
    ),
    (
        "break_even_sidecar",
        "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json",
        "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py",
    ),
    (
        "forward_tail_capacity",
        "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json",
        "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py",
    ),
    (
        "canonical_handoff",
        "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json",
        "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py",
    ),
]

OPTIONAL_ARTIFACT_SCRIPT_PAIRS: list[tuple[str, str, str]] = [
    (
        "hold_selection_handoff",
        "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py",
    ),
    (
        "hold_selection_aligned_review_lane",
        "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json",
        "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exit/risk source gaps and consumer drift risk against the canonical handoff."
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


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def normalize_exit_params(payload: Any) -> dict[str, Any]:
    data = dict(payload or {})
    return {
        "max_hold_bars": int(data.get("max_hold_bars") or 0),
        "break_even_trigger_r": float(data.get("break_even_trigger_r") or 0.0),
        "trailing_stop_atr": float(data.get("trailing_stop_atr") or 0.0),
        "cooldown_after_losses": int(data.get("cooldown_after_losses") or 0),
        "cooldown_bars": int(data.get("cooldown_bars") or 0),
    }


def scaled_decimal(value: float, *, scale: int, width: int = 0) -> str:
    scaled = int(round(float(value or 0.0) * scale))
    if scaled == 0:
        return "0"
    return f"{scaled:0{width}d}" if width else str(scaled)


def format_exit_risk_anchor_slug(params: dict[str, Any]) -> str:
    max_hold_bars = int(params.get("max_hold_bars") or 0)
    trailing = scaled_decimal(float(params.get("trailing_stop_atr") or 0.0), scale=10)
    break_even = float(params.get("break_even_trigger_r") or 0.0)
    break_even_slug = "no_be" if break_even <= 0 else f"be{scaled_decimal(break_even, scale=100, width=3)}"
    return f"hold{max_hold_bars}_trail{trailing}_{break_even_slug}"


def extract_hold_bars(slug: str) -> int:
    match = re.match(r"^hold(\d+)(?:_|$)", text(slug))
    if not match:
        return 0
    return int(match.group(1))


def canonical_handoff_baseline_retained(*, handoff: dict[str, Any], blocker: dict[str, Any], consumer_rule_ok: bool) -> bool:
    return (
        consumer_rule_ok
        and text(handoff.get("research_decision")) == "use_exit_risk_handoff_as_canonical_anchor"
        and text(handoff.get("source_head_status")) == "baseline_anchor_active"
        and text(blocker.get("research_decision")).startswith("block_exit_risk_promotion_keep_baseline_anchor_pair_")
    )


def hold_selection_aligned_review_lane_ready(*, handoff: dict[str, Any], aligned_lane: dict[str, Any]) -> bool:
    return (
        text(handoff.get("research_decision")) == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        and text(handoff.get("source_head_status")) == "upstream_hold_selection_conflict"
        and text(aligned_lane.get("research_decision"))
        == "hold_selection_aligned_break_even_review_lane_ready_but_canonical_handoff_conflict_remains"
        and text(aligned_lane.get("review_conclusion_research_decision"))
        == "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
        and text(aligned_lane.get("review_conclusion_arbitration_state")) == "review_only"
        and text(aligned_lane.get("primary_anchor_review_research_decision"))
        == "break_even_primary_anchor_review_complete_keep_baseline_anchor"
        and bool(text(aligned_lane.get("active_baseline")))
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Risk Source Gap Audit SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- canonical_anchor: `{text(payload.get('canonical_anchor')) or 'none'}`",
        f"- canonical_anchor_hold_bars: `{int(payload.get('canonical_anchor_hold_bars') or 0)}`",
        f"- hold_selection_active_baseline: `{text(payload.get('hold_selection_active_baseline')) or 'none'}`",
        f"- hold_selection_active_hold_bars: `{int(payload.get('hold_selection_active_hold_bars') or 0)}`",
        f"- consumer_rule_ok: `{bool(payload.get('consumer_rule_ok'))}`",
        f"- finding_count: `{int(payload.get('finding_count') or 0)}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload.get("findings", []):
        lines.extend(
            [
                f"### {text(row.get('label'))}",
                f"- issue: `{text(row.get('issue'))}`",
                f"- observed_slug: `{text(row.get('observed_slug')) or 'none'}`",
                f"- canonical_anchor: `{text(row.get('canonical_anchor')) or 'none'}`",
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
    artifacts: dict[str, dict[str, Any]] = {}
    latest_paths: dict[str, Path] = {}

    for label, latest_name, script_name in ARTIFACT_SCRIPT_PAIRS:
        latest_artifact = review_dir / latest_name
        builder_script = scripts_dir / script_name
        latest_exists = latest_artifact.exists()
        script_exists = builder_script.exists()
        latest_paths[label] = latest_artifact
        coverage_rows.append(
            {
                "label": label,
                "latest_artifact": str(latest_artifact),
                "latest_exists": latest_exists,
                "builder_script": str(builder_script),
                "builder_script_exists": script_exists,
            }
        )
        if latest_exists:
            artifacts[label] = load_json_mapping(latest_artifact)
        if not latest_exists:
            findings.append(
                {
                    "label": label,
                    "issue": "required_latest_artifact_missing",
                    "latest_artifact": str(latest_artifact),
                    "builder_script": str(builder_script),
                }
            )
        elif not script_exists:
            findings.append(
                {
                    "label": label,
                    "issue": "latest_artifact_exists_but_builder_script_missing",
                    "latest_artifact": str(latest_artifact),
                    "builder_script": str(builder_script),
                }
            )

    for label, latest_name, script_name in OPTIONAL_ARTIFACT_SCRIPT_PAIRS:
        latest_artifact = review_dir / latest_name
        builder_script = scripts_dir / script_name
        latest_exists = latest_artifact.exists()
        script_exists = builder_script.exists()
        latest_paths[label] = latest_artifact
        coverage_rows.append(
            {
                "label": label,
                "latest_artifact": str(latest_artifact),
                "latest_exists": latest_exists,
                "builder_script": str(builder_script),
                "builder_script_exists": script_exists,
                "required": False,
            }
        )
        if latest_exists:
            artifacts[label] = load_json_mapping(latest_artifact)
            if not script_exists:
                findings.append(
                    {
                        "label": label,
                        "issue": "latest_artifact_exists_but_builder_script_missing",
                        "latest_artifact": str(latest_artifact),
                        "builder_script": str(builder_script),
                    }
                )

    canonical_handoff_alias = str(latest_paths["canonical_handoff"])
    handoff = artifacts.get("canonical_handoff", {})
    sidecar = artifacts.get("break_even_sidecar", {})
    exit_risk = artifacts.get("exit_risk_selected", {})
    blocker = artifacts.get("forward_blocker", {})
    hold_selection_handoff = artifacts.get("hold_selection_handoff", {})
    aligned_review_lane = artifacts.get("hold_selection_aligned_review_lane", {})

    canonical_anchor = text(handoff.get("active_baseline"))
    canonical_anchor_hold_bars = extract_hold_bars(canonical_anchor)
    consumer_rule = text(handoff.get("consumer_rule"))
    consumer_rule_ok = canonical_handoff_alias.split("/")[-1] in consumer_rule
    baseline_retained_canonical_handoff = canonical_handoff_baseline_retained(
        handoff=handoff,
        blocker=blocker,
        consumer_rule_ok=consumer_rule_ok,
    )
    aligned_review_lane_ready = hold_selection_aligned_review_lane_ready(
        handoff=handoff,
        aligned_lane=aligned_review_lane,
    )
    aligned_review_lane_active_baseline = text(aligned_review_lane.get("active_baseline"))
    aligned_review_lane_preferred_watch_candidate = text(aligned_review_lane.get("preferred_watch_candidate"))

    if handoff and not consumer_rule_ok:
        findings.append(
            {
                "label": "canonical_handoff_consumer_rule",
                "issue": "canonical_handoff_consumer_rule_missing_latest_alias",
                "canonical_anchor": canonical_anchor,
            }
        )

    selected_exit_slug = ""
    if exit_risk:
        selected_exit_slug = format_exit_risk_anchor_slug(normalize_exit_params(exit_risk.get("selected_exit_params")))
        if canonical_anchor and selected_exit_slug and selected_exit_slug != canonical_anchor:
            issue = "selected_exit_params_differs_from_canonical_handoff_anchor"
            if baseline_retained_canonical_handoff:
                issue = "selected_exit_params_superseded_by_canonical_handoff_anchor"
            elif aligned_review_lane_ready and selected_exit_slug == aligned_review_lane_active_baseline:
                issue = "selected_exit_params_aligned_review_lane_watch_only"
            findings.append(
                {
                    "label": "exit_risk_selected_exit_params",
                    "issue": issue,
                    "observed_slug": selected_exit_slug,
                    "canonical_anchor": canonical_anchor,
                }
            )

    blocker_baseline_slug = ""
    challenge_pair = dict(blocker.get("challenge_pair") or {})
    if challenge_pair:
        blocker_baseline_slug = format_exit_risk_anchor_slug(
            normalize_exit_params(challenge_pair.get("baseline_exit_params"))
        )
        if canonical_anchor and blocker_baseline_slug and blocker_baseline_slug != canonical_anchor:
            issue = "challenge_pair_baseline_differs_from_canonical_handoff_anchor"
            watch_candidate = text(handoff.get("watch_candidate")) or text(sidecar.get("watch_candidate"))
            if baseline_retained_canonical_handoff and blocker_baseline_slug == watch_candidate:
                issue = "challenge_pair_baseline_superseded_by_canonical_handoff_anchor"
            elif aligned_review_lane_ready and blocker_baseline_slug == watch_candidate:
                issue = "challenge_pair_baseline_aligned_review_lane_watch_only"
            findings.append(
                {
                    "label": "forward_blocker_challenge_pair_baseline",
                    "issue": issue,
                    "observed_slug": blocker_baseline_slug,
                    "canonical_anchor": canonical_anchor,
                }
            )

    sidecar_active_baseline = text(sidecar.get("active_baseline"))
    if canonical_anchor and sidecar_active_baseline and sidecar_active_baseline != canonical_anchor:
        findings.append(
            {
                "label": "break_even_sidecar_active_baseline",
                "issue": "sidecar_active_baseline_differs_from_canonical_handoff_anchor",
                "observed_slug": sidecar_active_baseline,
                "canonical_anchor": canonical_anchor,
            }
        )

    hold_selection_active_baseline = text(hold_selection_handoff.get("active_baseline"))
    hold_selection_active_hold_bars = extract_hold_bars(hold_selection_active_baseline)
    if (
        canonical_anchor_hold_bars > 0
        and hold_selection_active_hold_bars > 0
        and hold_selection_active_hold_bars != canonical_anchor_hold_bars
    ):
        issue = "hold_selection_active_hold_bars_differs_from_exit_risk_canonical_anchor"
        if (
            aligned_review_lane_ready
            and hold_selection_active_baseline
            and hold_selection_active_baseline == text(aligned_review_lane.get("hold_selection_active_baseline"))
        ):
            issue = "hold_selection_active_hold_bars_aligned_review_lane_watch_only"
        findings.append(
            {
                "label": "hold_selection_handoff_active_baseline",
                "issue": issue,
                "observed_slug": hold_selection_active_baseline,
                "canonical_anchor": canonical_anchor,
                "observed_hold_bars": hold_selection_active_hold_bars,
                "canonical_hold_bars": canonical_anchor_hold_bars,
            }
        )

    findings.sort(key=lambda row: (text(row.get("label")), text(row.get("issue"))))
    missing_builder_labels = [
        text(row.get("label"))
        for row in findings
        if text(row.get("issue")) == "latest_artifact_exists_but_builder_script_missing"
    ]
    missing_latest_labels = [
        text(row.get("label"))
        for row in findings
        if text(row.get("issue")) == "required_latest_artifact_missing"
    ]
    drift_issue_types = {
        "canonical_handoff_consumer_rule_missing_latest_alias",
        "selected_exit_params_differs_from_canonical_handoff_anchor",
        "challenge_pair_baseline_differs_from_canonical_handoff_anchor",
        "sidecar_active_baseline_differs_from_canonical_handoff_anchor",
        "hold_selection_active_hold_bars_differs_from_exit_risk_canonical_anchor",
    }
    watch_only_issue_types = {
        "selected_exit_params_superseded_by_canonical_handoff_anchor",
        "challenge_pair_baseline_superseded_by_canonical_handoff_anchor",
        "selected_exit_params_aligned_review_lane_watch_only",
        "challenge_pair_baseline_aligned_review_lane_watch_only",
        "hold_selection_active_hold_bars_aligned_review_lane_watch_only",
    }
    has_drift_risk = any(text(row.get("issue")) in drift_issue_types for row in findings)
    has_watch_only = any(text(row.get("issue")) in watch_only_issue_types for row in findings)
    if has_drift_risk:
        research_decision = "exit_risk_source_gap_detected_consumer_drift_risk"
    elif has_watch_only:
        research_decision = "exit_risk_source_gap_detected_superseded_upstream_divergence_watch_only"
    elif findings:
        research_decision = "exit_risk_source_gap_detected_builder_or_latest_artifact_gap"
    else:
        research_decision = "exit_risk_source_gap_audit_pass_canonical_handoff_contract_intact"

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "workspace": str(workspace),
        "review_dir": str(review_dir),
        "canonical_handoff_alias": canonical_handoff_alias,
        "canonical_anchor": canonical_anchor,
        "canonical_anchor_hold_bars": canonical_anchor_hold_bars,
        "consumer_rule_ok": consumer_rule_ok,
        "hold_selection_aligned_review_lane_ready": aligned_review_lane_ready,
        "hold_selection_aligned_review_lane_research_decision": text(aligned_review_lane.get("research_decision")),
        "hold_selection_aligned_review_lane_active_baseline": aligned_review_lane_active_baseline,
        "hold_selection_aligned_review_lane_preferred_watch_candidate": aligned_review_lane_preferred_watch_candidate,
        "selected_exit_slug": selected_exit_slug,
        "blocker_baseline_slug": blocker_baseline_slug,
        "sidecar_active_baseline": sidecar_active_baseline,
        "sidecar_watch_candidate": text(sidecar.get("watch_candidate")),
        "hold_selection_active_baseline": hold_selection_active_baseline,
        "hold_selection_active_hold_bars": hold_selection_active_hold_bars,
        "research_decision": research_decision,
        "finding_count": len(findings),
        "missing_builder_labels": missing_builder_labels,
        "missing_latest_labels": missing_latest_labels,
        "findings": findings,
        "coverage_rows": coverage_rows,
        "recommended_brief": (
            f"exit_risk_source_gap:findings={len(findings)},"
            f"canonical={canonical_anchor or 'none'},"
            f"decision={research_decision}"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.md"
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
                "finding_count": len(findings),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
