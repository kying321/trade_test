from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "openclaw_orderflow_policy.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_openclaw_orderflow_policy_keeps_shadow_learning_alive_when_feedback_and_guards_disagree(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T103300Z_remote_intent_queue.json",
        {
            "queue_status": "queued_wait_trade_readiness",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "guard_alignment_brief": "risk_guard_candidate_mismatch:BNBUSDT->SOLUSDT",
        },
    )
    _write_json(
        review_dir / "20260315T103310Z_remote_orderflow_feedback.json",
        {
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
            "route_symbol": "SOLUSDT",
            "route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "dominant_guard_reason": "ticket_missing:no_actionable_ticket",
        },
    )
    _write_json(
        review_dir / "20260315T103320Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked", "reason_codes": ["rollback_hard"]},
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": [
                        "ticket_missing:no_actionable_ticket",
                        "panic_cooldown_active",
                    ],
                },
            ]
        },
    )
    _write_json(
        review_dir / "20260315T103325Z_remote_scope_router_state.json",
        {
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:34:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["policy_status"] == "shadow_policy_learning_only"
    assert payload["policy_decision"] == "accept_shadow_learning_only"
    assert payload["policy_recommendation"] == "continue_shadow_learning_until_guardian_clear"
    assert payload["feedback_status"] == "downrank_guardian_blocked_route"
    assert payload["blocked_gate_names"] == ["ops_live_gate", "risk_guard"]
    assert payload["policy_brief"].startswith("shadow_policy_learning_only:SOLUSDT")
    assert payload["shadow_learning_allowed"] is True
    assert payload["live_transport_allowed"] is False


def test_openclaw_orderflow_policy_blocks_on_non_executable_contract_even_when_queue_is_ready(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T103300Z_remote_intent_queue.json",
        {
            "queue_status": "queued_execution_contract_blocked",
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:portfolio_margin_um",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "portfolio_margin_um",
            "ticket_match_brief": "fresh_artifact:ticket_row_ready:SOLUSDT",
            "guard_alignment_brief": "risk_guard_not_ticket_blocked",
            "execution_contract_status": "non_executable_contract",
            "execution_contract_brief": "non_executable_contract:portfolio_margin_um:portfolio_margin_um:spot_remote_lane_missing,portfolio_margin_um_read_only_mode,shadow_executor_only_mode",
            "execution_contract_executor_mode": "shadow_guarded",
            "execution_contract_executor_mode_source": "contract_state",
            "execution_contract_reason_codes": [
                "spot_remote_lane_missing",
                "portfolio_margin_um_read_only_mode",
                "shadow_executor_only_mode",
            ],
            "execution_contract_live_orders_allowed": False,
        },
    )
    _write_json(
        review_dir / "20260315T103310Z_remote_orderflow_feedback.json",
        {
            "feedback_status": "route_quality_ok",
            "feedback_brief": "route_quality_ok:SOLUSDT:ticket_ready",
            "feedback_recommendation": "promote_when_transport_contract_is_live",
            "route_symbol": "SOLUSDT",
            "route_action": "seed_ticket",
            "remote_market": "portfolio_margin_um",
            "dominant_guard_reason": "",
        },
    )
    _write_json(
        review_dir / "20260315T103320Z_live_gate_blocker_report.json",
        {"blockers": []},
    )
    _write_json(
        review_dir / "20260315T103325Z_remote_scope_router_state.json",
        {
            "scope_router_status": "review_candidate_inside_remote_scope",
            "scope_router_brief": "review_candidate_inside_remote_scope:SOLUSDT:seed_ticket:portfolio_margin_um",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:34:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["policy_status"] == "shadow_policy_execution_contract_blocked"
    assert payload["policy_decision"] == "accept_shadow_learning_only"
    assert payload["policy_recommendation"] == "keep_shadow_learning_until_execution_contract_promoted"
    assert payload["execution_contract_status"] == "non_executable_contract"
    assert payload["execution_contract_executor_mode"] == "shadow_guarded"
    assert payload["execution_contract_executor_mode_source"] == "contract_state"
    assert payload["execution_contract_reason_codes"] == [
        "spot_remote_lane_missing",
        "portfolio_margin_um_read_only_mode",
        "shadow_executor_only_mode",
    ]
    assert payload["shadow_learning_allowed"] is True
    assert payload["live_transport_allowed"] is False


def test_openclaw_orderflow_policy_accepts_guarded_probe_candidate_before_live_promotion(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T103300Z_remote_intent_queue.json",
        {
            "queue_status": "queued_guarded_probe_ready",
            "queue_brief": "queued_guarded_probe_ready:SOLUSDT:seed_ticket:spot",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "spot",
            "ticket_match_brief": "fresh_artifact:ticket_row_ready:SOLUSDT",
            "guard_alignment_brief": "risk_guard_not_ticket_blocked",
            "execution_contract_status": "probe_only_contract",
            "execution_contract_brief": "probe_only_contract:spot:spot:guarded_probe_only_mode",
            "execution_contract_mode": "guarded_probe_only",
            "execution_contract_guarded_probe_allowed": True,
            "execution_contract_executor_mode": "spot_live_guarded",
            "execution_contract_executor_mode_source": "contract_state",
            "execution_contract_reason_codes": ["guarded_probe_only_mode"],
            "execution_contract_live_orders_allowed": False,
        },
    )
    _write_json(
        review_dir / "20260315T103310Z_remote_orderflow_feedback.json",
        {
            "feedback_status": "route_quality_ok",
            "feedback_brief": "route_quality_ok:SOLUSDT:ticket_ready",
            "feedback_recommendation": "promote_when_transport_contract_is_live",
            "route_symbol": "SOLUSDT",
            "route_action": "seed_ticket",
            "remote_market": "spot",
            "dominant_guard_reason": "",
        },
    )
    _write_json(
        review_dir / "20260315T103320Z_live_gate_blocker_report.json",
        {"blockers": []},
    )
    _write_json(
        review_dir / "20260315T103325Z_remote_scope_router_state.json",
        {
            "scope_router_status": "review_candidate_inside_remote_scope",
            "scope_router_brief": "review_candidate_inside_remote_scope:SOLUSDT:seed_ticket:spot",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:34:01Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["policy_status"] == "shadow_policy_guarded_probe_candidate"
    assert payload["policy_decision"] == "accept_guarded_probe_candidate"
    assert payload["policy_recommendation"] == "run_guarded_probe_before_live_promotion"
    assert payload["execution_contract_guarded_probe_allowed"] is True
    assert payload["guarded_probe_allowed"] is True
    assert payload["shadow_learning_allowed"] is False
    assert payload["live_transport_allowed"] is False
