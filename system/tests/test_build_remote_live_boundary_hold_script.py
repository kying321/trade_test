from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_remote_live_boundary_hold.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_live_boundary_hold_keeps_shadow_only_until_guardian_and_review_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T112000Z_remote_execution_actor_canary_gate.json",
        {
            "status": "ok",
            "canary_gate_status": "shadow_canary_gate_blocked",
            "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um",
            "canary_gate_decision": "deny_canary_until_guardian_clear",
            "blocker_detail": "guardian blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T112030Z_remote_orderflow_quality_report.json",
        {
            "status": "ok",
            "quality_status": "quality_degraded_guardian_blocked_shadow_only",
            "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_0:portfolio_margin_um",
            "quality_recommendation": "keep_downranked_shadow_until_guardian_clear",
            "blocker_detail": "quality degraded",
            "done_when": "quality improves",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T103500Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_blocked",
            "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "reject_until_guardian_clear",
            "blocker_detail": "policy blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked"},
                {"name": "risk_guard", "status": "blocked"},
            ]
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
            "review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only | time-sync=threshold_breach:scope=clock_skew_only"
            ),
            "review_head_done_when": "clear review blocker and time sync blocker",
        },
    )
    _write_json(
        review_dir / "20260315T112040Z_system_time_sync_repair_verification_report.json",
        {
            "status": "blocked",
            "verification_brief": (
                "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked"
            ),
            "cleared": False,
        },
    )
    _write_json(
        review_dir / "20260315T112045Z_remote_shadow_clock_evidence.json",
        {
            "status": "ok",
            "evidence_status": "shadow_clock_evidence_present",
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "shadow_learning_allowed": True,
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:21:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["hold_status"] == "live_boundary_hold_active"
    assert (
        payload["hold_brief"]
        == "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um"
    )
    assert payload["hold_decision"] == "keep_shadow_transport_only"
    assert payload["next_transition"] == "guardian_blocker_clearance"
    assert payload["guardian_blocked"] is True
    assert payload["review_blocked"] is True
    assert payload["time_sync_blocked"] is True
    assert payload["time_sync_mode"] == "promotion_blocked_shadow_learning_allowed"
    assert payload["remote_shadow_clock_shadow_learning_allowed"] is True
    assert payload["remote_shadow_clock_evidence_status"] == "shadow_clock_evidence_present"
    assert "time_sync_blocked" in payload["hold_reason_codes"]


def test_build_remote_live_boundary_hold_prefers_latest_time_sync_verification_by_stamp_not_mtime(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T001450Z_remote_execution_actor_canary_gate.json",
        {
            "status": "ok",
            "canary_gate_status": "shadow_canary_gate_blocked",
            "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T001455Z_remote_orderflow_quality_report.json",
        {
            "status": "ok",
            "quality_status": "quality_degraded_guardian_blocked_shadow_only",
            "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_44:portfolio_margin_um",
            "quality_recommendation": "keep_downranked_shadow_until_guardian_clear",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T001500Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_learning_only",
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "accept_shadow_learning_only",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T001505Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "ops_live_gate", "status": "blocked"}]},
    )
    _write_json(
        review_dir / "20260316T001510Z_cross_market_operator_state.json",
        {
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
            "review_head_blocker_detail": "SOLUSDT remains Bias_Only",
            "review_head_done_when": "clear review blocker",
        },
    )
    stale_verification_path = review_dir / "20260316T001200Z_system_time_sync_repair_verification_report.json"
    _write_json(
        stale_verification_path,
        {
            "status": "blocked",
            "verification_brief": "blocked:SOLUSDT:probe_blocked",
            "cleared": False,
        },
    )
    fresh_verification_path = review_dir / "20260316T001520Z_system_time_sync_repair_verification_report.json"
    _write_json(
        fresh_verification_path,
        {
            "status": "cleared",
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    stale_mtime = fresh_verification_path.stat().st_mtime + 60
    os.utime(stale_verification_path, (stale_mtime, stale_mtime))
    _write_json(
        review_dir / "20260316T001525Z_remote_shadow_clock_evidence.json",
        {
            "status": "ok",
            "evidence_status": "shadow_clock_evidence_present",
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "shadow_learning_allowed": True,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T00:16:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["time_sync_blocked"] is False
    assert payload["time_sync_mode"] == "time_sync_clear_shadow_learning_allowed"
    assert payload["time_sync_verification_status"] == "cleared"
    assert payload["time_sync_verification_brief"] == "cleared:SOLUSDT:time_sync_ok"
