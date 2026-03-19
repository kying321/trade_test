from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_openclaw_orderflow_blueprint.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_openclaw_orderflow_blueprint_reads_remote_live_stack(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"

    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "next_focus_reason": "ops_live_gate_blocked",
                "secondary_focus_reason": "risk_guard_blocked",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                    "blocker_detail": "ops_live_gate and risk_guard still block automation",
                    "profitability_window": "30d",
                    "profitability_pnl": 18.79,
                    "profitability_trade_count": 38,
                },
                "remote_live_history": {
                    "window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "live_decision": {
                "current_decision": "do_not_start_formal_live",
                "summary": "Clear ops_live_gate, then risk_guard, then scope mismatch.",
            },
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked", "reason_codes": ["rollback_hard", "slot_anomaly"]},
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket", "panic_cooldown_active"],
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(
        review_dir / "20260314T190514Z_hot_universe_operator_brief.json",
        {
            "operator_action_queue": [
                {"symbol": "XAUUSD", "action": "wait_for_paper_execution_close_evidence"},
                {"symbol": "SOLUSDT", "action": "deprioritize_flow"},
            ],
        },
    )
    _write_json(
        review_dir / "latest_remote_live_history_audit.json",
        {
            "status": "ok",
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    current = payload["current_status"]
    assert current["current_life_stage"] == "guarded_remote_guardian"
    assert current["remote_project_dir"] == "/home/ubuntu/openclaw-system"
    assert current["remote_live_diagnosis_brief"] == (
        "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
    )
    assert current["cross_market_review_head_brief"] == "review:crypto_route:SOLUSDT:deprioritize_flow:58"
    assert current["hot_operator_action_queue"] == "XAUUSD:wait_for_paper_execution_close_evidence -> SOLUSDT:deprioritize_flow"
    assert payload["digital_layers"][0]["layer"] == "perception"
    assert [row["stage"] for row in payload["control_chain"]] == [
        "research",
        "signal",
        "risk",
        "execution",
        "reconcile",
        "post_trade",
    ]
    assert payload["control_chain"][0]["interface_fields"] == [
        "crypto_shortline_cross_section_backtest_status",
        "crypto_shortline_cross_section_backtest_decision",
        "crypto_shortline_backtest_slice_status",
        "crypto_shortline_backtest_slice_decision",
    ]
    assert payload["continuous_optimization_backlog"][0]["stage"] == "execution"
    assert payload["continuous_optimization_backlog"][0]["title"] == "持续优化执行传导链"
    assert payload["refactor_phases"][0]["name"] == "identity_and_scope"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_execution_identity_state"
    assert Path(str(payload["artifact"])).name == "20260314T191000Z_openclaw_orderflow_blueprint.json"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_openclaw_orderflow_blueprint_advances_backlog_after_phase1_artifacts_exist(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"

    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
                "remote_live_history": {"window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"},
            }
        },
    )
    _write_json(review_dir / "20260314T140104Z_live_gate_blocker_report.json", {"blockers": [{"name": "ops_live_gate"}]})
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
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
            "2026-03-15T03:20:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "identity_scoped_remote_guardian"
    assert current["remote_execution_identity_brief"] == "43.153.148.242:portfolio_margin_um:split_scope:blocked"
    assert (
        current["remote_scope_router_brief"]
        == "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um"
    )
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_intent_queue"


def test_build_openclaw_orderflow_blueprint_advances_to_intent_queued_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"

    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
                "remote_live_history": {"window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"},
            }
        },
    )
    _write_json(review_dir / "20260314T140104Z_live_gate_blocker_report.json", {"blockers": [{"name": "ops_live_gate"}]})
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:30:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "intent_queued_remote_guardian"
    assert current["remote_intent_queue_brief"] == (
        "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um"
    )
    assert current["remote_intent_queue_recommendation"] == "hold_remote_idle_until_ticket_ready"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_execution_journal"


def test_build_openclaw_orderflow_blueprint_advances_to_journaled_stage(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"

    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
                "remote_live_history": {"window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"},
            }
        },
    )
    _write_json(review_dir / "20260314T140104Z_live_gate_blocker_report.json", {"blockers": [{"name": "ops_live_gate"}]})
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T094800Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:ticket_missing:no_actionable_ticket | not_attempted_wait_trade_readiness",
            "append_status": "appended",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:40:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "journaled_remote_guardian"
    assert current["remote_execution_journal_status"] == "intent_logged_guardian_blocked"
    assert current["remote_execution_journal_append_status"] == "appended"
    assert current["remote_execution_journal_brief"].startswith(
        "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um"
    )
    assert payload["immediate_backlog"][0]["target_artifact"] == "openclaw_orderflow_executor.service"


def test_build_openclaw_orderflow_blueprint_advances_to_executor_scaffolded_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"

    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
                "remote_live_history": {"window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"},
            }
        },
    )
    _write_json(review_dir / "20260314T140104Z_live_gate_blocker_report.json", {"blockers": [{"name": "ops_live_gate"}]})
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T094800Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json",
        {
            "status": "ok",
            "executor_status": "shadow_guarded_executor_ready",
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:50:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "executor_scaffolded_remote_guardian"
    assert current["openclaw_orderflow_executor_status"] == "shadow_guarded_executor_ready"
    assert current["openclaw_orderflow_executor_service_name"] == "openclaw-orderflow-executor.service"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_orderflow_feedback"


def test_build_openclaw_orderflow_blueprint_advances_to_feedback_stage(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "next_focus_reason": "ops_live_gate_blocked",
                "secondary_focus_reason": "risk_guard_blocked",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                    "blocker_detail": "ops_live_gate and risk_guard still block automation",
                    "profitability_window": "30d",
                    "profitability_pnl": 18.79,
                    "profitability_trade_count": 38,
                },
                "remote_live_history": {
                    "window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "live_decision": {
                "current_decision": "do_not_start_formal_live",
                "summary": "Clear ops_live_gate, then risk_guard, then scope mismatch.",
            },
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked", "reason_codes": ["rollback_hard", "slot_anomaly"]},
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket", "panic_cooldown_active"],
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(
        review_dir / "20260314T190514Z_hot_universe_operator_brief.json",
        {
            "operator_action_queue": [
                {"symbol": "XAUUSD", "action": "wait_for_paper_execution_close_evidence"},
                {"symbol": "SOLUSDT", "action": "deprioritize_flow"},
            ],
        },
    )
    _write_json(
        review_dir / "latest_remote_live_history_audit.json",
        {
            "status": "ok",
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open",
        },
    )
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T094800Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json",
        {
            "status": "ok",
            "executor_status": "shadow_guarded_executor_ready",
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )
    _write_json(
        review_dir / "20260315T095000Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_aging_high:ticket_missing:no_actionable_ticket",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:55:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "feedback_learning_remote_guardian"
    assert current["remote_orderflow_feedback_status"] == "downrank_guardian_blocked_route"
    assert current["remote_orderflow_feedback_recommendation"] == (
        "downrank_until_ticket_fresh_and_guardian_clear"
    )
    assert payload["immediate_backlog"][0]["target_artifact"] == "openclaw_orderflow_policy.py"


def test_build_openclaw_orderflow_blueprint_advances_to_policy_stage(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "next_focus_reason": "ops_live_gate_blocked",
                "secondary_focus_reason": "risk_guard_blocked",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                    "blocker_detail": "ops_live_gate and risk_guard still block automation",
                    "profitability_window": "30d",
                    "profitability_pnl": 18.79,
                    "profitability_trade_count": 38,
                },
            }
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "live_decision": {
                "current_decision": "do_not_start_formal_live",
                "summary": "Clear ops_live_gate, then risk_guard, then scope mismatch.",
            },
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked", "reason_codes": ["rollback_hard", "slot_anomaly"]},
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket", "panic_cooldown_active"],
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(
        review_dir / "20260314T190514Z_hot_universe_operator_brief.json",
        {
            "operator_action_queue": [
                {"symbol": "XAUUSD", "action": "wait_for_paper_execution_close_evidence"},
                {"symbol": "SOLUSDT", "action": "deprioritize_flow"},
            ],
        },
    )
    _write_json(
        review_dir / "latest_remote_live_history_audit.json",
        {
            "status": "ok",
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open",
        },
    )
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T094800Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json",
        {
            "status": "ok",
            "executor_status": "shadow_guarded_executor_ready",
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )
    _write_json(
        review_dir / "20260315T095000Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_aging_high:ticket_missing:no_actionable_ticket",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T095100Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_blocked",
            "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "reject_until_guardian_clear",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:56:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "policy_scoped_remote_guardian"
    assert current["remote_orderflow_policy_status"] == "shadow_policy_blocked"
    assert current["remote_orderflow_policy_decision"] == "reject_until_guardian_clear"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_execution_ack_state"


def test_build_openclaw_orderflow_blueprint_advances_to_ack_stage(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "ops_live_gate", "status": "blocked"}]},
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T094800Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json",
        {
            "status": "ok",
            "executor_status": "shadow_guarded_executor_ready",
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )
    _write_json(
        review_dir / "20260315T095000Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T095100Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_blocked",
            "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "reject_until_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T095110Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_no_send_ack_recorded",
            "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted",
            "ack_decision": "record_reject_without_transport",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:57:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "ack_traced_remote_guardian"
    assert current["remote_execution_ack_status"] == "shadow_no_send_ack_recorded"
    assert current["remote_execution_ack_decision"] == "record_reject_without_transport"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_execution_actor.service"


def test_build_openclaw_orderflow_blueprint_advances_to_actor_stage(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "ops_live_gate", "status": "blocked"}]},
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T094800Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json",
        {
            "status": "ok",
            "executor_status": "shadow_guarded_executor_ready",
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )
    _write_json(
        review_dir / "20260315T095000Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T095100Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_blocked",
            "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "reject_until_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T095110Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_no_send_ack_recorded",
            "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted",
            "ack_decision": "record_reject_without_transport",
        },
    )
    _write_json(
        review_dir / "20260315T095120Z_remote_execution_actor_state.json",
        {
            "status": "ok",
            "actor_status": "shadow_actor_ready_policy_blocked",
            "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um",
            "actor_service_name": "remote_execution_actor.service",
            "transport_phase": "shadow_only_no_transport",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:58:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "actor_scaffolded_remote_guardian"
    assert current["remote_execution_actor_status"] == "shadow_actor_ready_policy_blocked"
    assert current["remote_execution_actor_service_name"] == "remote_execution_actor.service"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_execution_actor_guarded_transport"


def test_build_openclaw_orderflow_blueprint_advances_to_guarded_transport_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "ops_live_gate", "status": "blocked"}]},
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T094600Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T094700Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T094800Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json",
        {
            "status": "ok",
            "executor_status": "shadow_guarded_executor_ready",
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )
    _write_json(
        review_dir / "20260315T095000Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T095100Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_blocked",
            "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "reject_until_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T095110Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_no_send_ack_recorded",
            "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted",
            "ack_decision": "record_reject_without_transport",
        },
    )
    _write_json(
        review_dir / "20260315T095120Z_remote_execution_actor_state.json",
        {
            "status": "ok",
            "actor_status": "shadow_actor_ready_policy_blocked",
            "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um",
            "actor_service_name": "remote_execution_actor.service",
            "transport_phase": "shadow_only_no_transport",
        },
    )
    _write_json(
        review_dir / "20260315T095130Z_remote_execution_actor_guarded_transport.json",
        {
            "status": "ok",
            "guarded_transport_status": "guarded_transport_preview_blocked",
            "guarded_transport_brief": "guarded_transport_preview_blocked:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um",
            "guarded_transport_decision": "do_not_arm_transport_policy_blocked",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T03:59:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "guarded_transport_preview_remote_guardian"
    assert current["remote_execution_guarded_transport_status"] == "guarded_transport_preview_blocked"
    assert current["remote_execution_guarded_transport_decision"] == (
        "do_not_arm_transport_policy_blocked"
    )
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_execution_transport_sla"


def test_build_openclaw_orderflow_blueprint_advances_to_transport_sla_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "ops_live_gate", "status": "blocked"}]},
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(review_dir / "20260315T094500Z_remote_execution_identity_state.json", {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"})
    _write_json(review_dir / "20260315T094600Z_remote_scope_router_state.json", {"status": "ok", "scope_router_status": "review_candidate_inside_scope_not_trade_ready", "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um"})
    _write_json(review_dir / "20260315T094700Z_remote_intent_queue.json", {"status": "ok", "queue_status": "queued_wait_trade_readiness", "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um", "queue_recommendation": "hold_remote_idle_until_ticket_ready"})
    _write_json(review_dir / "20260315T094800Z_remote_execution_journal.json", {"status": "ok", "journal_status": "intent_logged_guardian_blocked", "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness", "append_status": "appended"})
    _write_json(review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json", {"status": "ok", "executor_status": "shadow_guarded_executor_ready", "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um", "service_name": "openclaw-orderflow-executor.service"})
    _write_json(review_dir / "20260315T095000Z_remote_orderflow_feedback.json", {"status": "ok", "feedback_status": "downrank_guardian_blocked_route", "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket", "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear"})
    _write_json(review_dir / "20260315T095100Z_remote_orderflow_policy_state.json", {"status": "ok", "policy_status": "shadow_policy_blocked", "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route", "policy_decision": "reject_until_guardian_clear"})
    _write_json(review_dir / "20260315T095110Z_remote_execution_ack_state.json", {"status": "ok", "ack_status": "shadow_no_send_ack_recorded", "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted", "ack_decision": "record_reject_without_transport"})
    _write_json(review_dir / "20260315T095120Z_remote_execution_actor_state.json", {"status": "ok", "actor_status": "shadow_actor_ready_policy_blocked", "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um", "actor_service_name": "remote_execution_actor.service", "transport_phase": "shadow_only_no_transport"})
    _write_json(review_dir / "20260315T095130Z_remote_execution_actor_guarded_transport.json", {"status": "ok", "guarded_transport_status": "guarded_transport_preview_blocked", "guarded_transport_brief": "guarded_transport_preview_blocked:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "guarded_transport_decision": "do_not_arm_transport_policy_blocked"})
    _write_json(review_dir / "20260315T095140Z_remote_execution_transport_sla.json", {"status": "ok", "transport_sla_status": "shadow_transport_sla_blocked_no_send", "transport_sla_brief": "shadow_transport_sla_blocked_no_send:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "transport_sla_decision": "define_sla_before_canary"})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T04:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "transport_sla_shadow_remote_guardian"
    assert current["remote_execution_transport_sla_status"] == "shadow_transport_sla_blocked_no_send"
    assert current["remote_execution_transport_sla_decision"] == "define_sla_before_canary"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_execution_actor_canary_gate"


def test_build_openclaw_orderflow_blueprint_advances_to_canary_gate_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {"operator_handoff": {"remote_host": "43.153.148.242", "remote_user": "ubuntu", "remote_project_dir": "/home/ubuntu/openclaw-system", "handoff_state": "ops_live_gate_blocked", "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled", "focus_stack_brief": "gate -> risk_guard", "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um", "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"}, "remote_live_diagnosis": {"brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"}}},
    )
    _write_json(review_dir / "20260314T140104Z_live_gate_blocker_report.json", {"live_decision": {"current_decision": "do_not_start_formal_live"}, "blockers": [{"name": "ops_live_gate", "status": "blocked"}]})
    _write_json(review_dir / "20260314T190200Z_cross_market_operator_state.json", {"remote_live_operator_alignment_status": "local_operator_head_outside_remote_live_scope", "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um", "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um", "review_head": {"status": "review", "area": "crypto_route", "symbol": "SOLUSDT", "action": "deprioritize_flow", "priority_score": 58}})
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(review_dir / "20260315T094500Z_remote_execution_identity_state.json", {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"})
    _write_json(review_dir / "20260315T094600Z_remote_scope_router_state.json", {"status": "ok", "scope_router_status": "review_candidate_inside_scope_not_trade_ready", "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um"})
    _write_json(review_dir / "20260315T094700Z_remote_intent_queue.json", {"status": "ok", "queue_status": "queued_wait_trade_readiness", "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um", "queue_recommendation": "hold_remote_idle_until_ticket_ready"})
    _write_json(review_dir / "20260315T094800Z_remote_execution_journal.json", {"status": "ok", "journal_status": "intent_logged_guardian_blocked", "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness", "append_status": "appended"})
    _write_json(review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json", {"status": "ok", "executor_status": "shadow_guarded_executor_ready", "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um", "service_name": "openclaw-orderflow-executor.service"})
    _write_json(review_dir / "20260315T095000Z_remote_orderflow_feedback.json", {"status": "ok", "feedback_status": "downrank_guardian_blocked_route", "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket", "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear"})
    _write_json(review_dir / "20260315T095100Z_remote_orderflow_policy_state.json", {"status": "ok", "policy_status": "shadow_policy_blocked", "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route", "policy_decision": "reject_until_guardian_clear", "queue_status": "queued_wait_trade_readiness", "scope_router_status": "review_candidate_inside_scope_not_trade_ready", "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT", "risk_reason_codes": ["ticket_missing:no_actionable_ticket"]})
    _write_json(review_dir / "20260315T095110Z_remote_execution_ack_state.json", {"status": "ok", "ack_status": "shadow_no_send_ack_recorded", "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted", "ack_decision": "record_reject_without_transport"})
    _write_json(review_dir / "20260315T095120Z_remote_execution_actor_state.json", {"status": "ok", "actor_status": "shadow_actor_ready_policy_blocked", "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um", "actor_service_name": "remote_execution_actor.service", "transport_phase": "shadow_only_no_transport", "route_symbol": "SOLUSDT", "remote_market": "portfolio_margin_um"})
    _write_json(review_dir / "20260315T095130Z_remote_execution_actor_guarded_transport.json", {"status": "ok", "guarded_transport_status": "guarded_transport_preview_blocked", "guarded_transport_brief": "guarded_transport_preview_blocked:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "guarded_transport_decision": "do_not_arm_transport_policy_blocked"})
    _write_json(review_dir / "20260315T095140Z_remote_execution_transport_sla.json", {"status": "ok", "transport_sla_status": "shadow_transport_sla_blocked_no_send", "transport_sla_brief": "shadow_transport_sla_blocked_no_send:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "transport_sla_decision": "define_sla_before_canary"})
    _write_json(review_dir / "20260315T095150Z_remote_execution_actor_canary_gate.json", {"status": "ok", "canary_gate_status": "shadow_canary_gate_blocked", "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um", "canary_gate_decision": "deny_canary_until_guardian_clear"})

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir), "--now", "2026-03-15T04:01:00+08:00"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "canary_gated_remote_guardian"
    assert current["remote_execution_actor_canary_gate_status"] == "shadow_canary_gate_blocked"
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_orderflow_quality_report"


def test_build_openclaw_orderflow_blueprint_advances_to_quality_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {"operator_handoff": {"remote_host": "43.153.148.242", "remote_user": "ubuntu", "remote_project_dir": "/home/ubuntu/openclaw-system", "handoff_state": "ops_live_gate_blocked", "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled", "focus_stack_brief": "gate -> risk_guard", "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um", "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"}, "remote_live_diagnosis": {"brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"}}},
    )
    _write_json(review_dir / "20260314T140104Z_live_gate_blocker_report.json", {"live_decision": {"current_decision": "do_not_start_formal_live"}, "blockers": [{"name": "ops_live_gate", "status": "blocked"}]})
    _write_json(review_dir / "20260314T190200Z_cross_market_operator_state.json", {"remote_live_operator_alignment_status": "local_operator_head_outside_remote_live_scope", "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um", "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um", "review_head": {"status": "review", "area": "crypto_route", "symbol": "SOLUSDT", "action": "deprioritize_flow", "priority_score": 58}})
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(review_dir / "20260315T094500Z_remote_execution_identity_state.json", {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"})
    _write_json(review_dir / "20260315T094600Z_remote_scope_router_state.json", {"status": "ok", "scope_router_status": "review_candidate_inside_scope_not_trade_ready", "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um"})
    _write_json(review_dir / "20260315T094700Z_remote_intent_queue.json", {"status": "ok", "queue_status": "queued_wait_trade_readiness", "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um", "queue_recommendation": "hold_remote_idle_until_ticket_ready"})
    _write_json(review_dir / "20260315T094800Z_remote_execution_journal.json", {"status": "ok", "journal_status": "intent_logged_guardian_blocked", "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness", "append_status": "appended", "blocker_detail": "guardian blocked"})
    _write_json(review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json", {"status": "ok", "executor_status": "shadow_guarded_executor_ready", "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um", "service_name": "openclaw-orderflow-executor.service"})
    _write_json(review_dir / "20260315T095000Z_remote_orderflow_feedback.json", {"status": "ok", "feedback_status": "downrank_guardian_blocked_route", "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket", "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear", "queue_age_status": "queue_warming", "ticket_artifact_status": "stale_artifact", "guardian_blocked_count": 1, "no_fill_count": 1, "recent_outcomes": ["not_attempted_wait_trade_readiness"], "blocker_detail": "guardian blocked"})
    _write_json(review_dir / "20260315T095100Z_remote_orderflow_policy_state.json", {"status": "ok", "policy_status": "shadow_policy_blocked", "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route", "policy_decision": "reject_until_guardian_clear"})
    _write_json(review_dir / "20260315T095110Z_remote_execution_ack_state.json", {"status": "ok", "ack_status": "shadow_no_send_ack_recorded", "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted", "ack_decision": "record_reject_without_transport", "blocker_detail": "guardian blocked"})
    _write_json(review_dir / "20260315T095120Z_remote_execution_actor_state.json", {"status": "ok", "actor_status": "shadow_actor_ready_policy_blocked", "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um", "actor_service_name": "remote_execution_actor.service", "transport_phase": "shadow_only_no_transport"})
    _write_json(review_dir / "20260315T095130Z_remote_execution_actor_guarded_transport.json", {"status": "ok", "guarded_transport_status": "guarded_transport_preview_blocked", "guarded_transport_brief": "guarded_transport_preview_blocked:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "guarded_transport_decision": "do_not_arm_transport_policy_blocked"})
    _write_json(review_dir / "20260315T095140Z_remote_execution_transport_sla.json", {"status": "ok", "transport_sla_status": "shadow_transport_sla_blocked_no_send", "transport_sla_brief": "shadow_transport_sla_blocked_no_send:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "transport_sla_decision": "define_sla_before_canary"})
    _write_json(review_dir / "20260315T095150Z_remote_execution_actor_canary_gate.json", {"status": "ok", "canary_gate_status": "shadow_canary_gate_blocked", "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um", "canary_gate_decision": "deny_canary_until_guardian_clear"})
    _write_json(review_dir / "20260315T095160Z_remote_orderflow_quality_report.json", {"status": "ok", "quality_status": "quality_degraded_guardian_blocked_shadow_only", "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_0:portfolio_margin_um", "quality_recommendation": "keep_downranked_shadow_until_guardian_clear", "quality_score": 0})

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir), "--now", "2026-03-15T04:02:00+08:00"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "quality_scored_remote_guardian"
    assert current["remote_orderflow_quality_report_status"] == "quality_degraded_guardian_blocked_shadow_only"
    assert payload["immediate_backlog"][0]["target_artifact"] == "live_boundary_hold"


def test_build_openclaw_orderflow_blueprint_advances_to_live_boundary_hold_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260314T183100Z_remote_live_handoff.json",
        {"operator_handoff": {"remote_host": "43.153.148.242", "remote_user": "ubuntu", "remote_project_dir": "/home/ubuntu/openclaw-system", "handoff_state": "ops_live_gate_blocked", "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled", "focus_stack_brief": "gate -> risk_guard", "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um", "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"}, "remote_live_diagnosis": {"brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"}}},
    )
    _write_json(review_dir / "20260314T140104Z_live_gate_blocker_report.json", {"live_decision": {"current_decision": "do_not_start_formal_live"}, "blockers": [{"name": "ops_live_gate", "status": "blocked"}]})
    _write_json(review_dir / "20260314T190200Z_cross_market_operator_state.json", {"remote_live_operator_alignment_status": "local_operator_head_outside_remote_live_scope", "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um", "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um", "review_head": {"status": "review", "area": "crypto_route", "symbol": "SOLUSDT", "action": "deprioritize_flow", "priority_score": 58}, "review_head_blocker_detail": "SOLUSDT remains Bias_Only | time-sync=threshold_breach:scope=clock_skew_only", "review_head_done_when": "clear review blocker and time sync blocker"})
    _write_json(review_dir / "20260314T190514Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(review_dir / "20260315T094500Z_remote_execution_identity_state.json", {"status": "ok", "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"})
    _write_json(review_dir / "20260315T094600Z_remote_scope_router_state.json", {"status": "ok", "scope_router_status": "review_candidate_inside_scope_not_trade_ready", "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um"})
    _write_json(review_dir / "20260315T094700Z_remote_intent_queue.json", {"status": "ok", "queue_status": "queued_wait_trade_readiness", "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um", "queue_recommendation": "hold_remote_idle_until_ticket_ready"})
    _write_json(review_dir / "20260315T094800Z_remote_execution_journal.json", {"status": "ok", "journal_status": "intent_logged_guardian_blocked", "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness", "append_status": "appended", "blocker_detail": "guardian blocked"})
    _write_json(review_dir / "20260315T094900Z_openclaw_orderflow_executor_state.json", {"status": "ok", "executor_status": "shadow_guarded_executor_ready", "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um", "service_name": "openclaw-orderflow-executor.service"})
    _write_json(review_dir / "20260315T095000Z_remote_orderflow_feedback.json", {"status": "ok", "feedback_status": "downrank_guardian_blocked_route", "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket", "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear", "queue_age_status": "queue_warming", "ticket_artifact_status": "stale_artifact", "guardian_blocked_count": 1, "no_fill_count": 1, "recent_outcomes": ["not_attempted_wait_trade_readiness"], "blocker_detail": "guardian blocked"})
    _write_json(review_dir / "20260315T095100Z_remote_orderflow_policy_state.json", {"status": "ok", "policy_status": "shadow_policy_blocked", "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route", "policy_decision": "reject_until_guardian_clear", "blocker_detail": "policy blocked"})
    _write_json(review_dir / "20260315T095110Z_remote_execution_ack_state.json", {"status": "ok", "ack_status": "shadow_no_send_ack_recorded", "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted", "ack_decision": "record_reject_without_transport", "blocker_detail": "guardian blocked"})
    _write_json(review_dir / "20260315T095120Z_remote_execution_actor_state.json", {"status": "ok", "actor_status": "shadow_actor_ready_policy_blocked", "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um", "actor_service_name": "remote_execution_actor.service", "transport_phase": "shadow_only_no_transport"})
    _write_json(review_dir / "20260315T095130Z_remote_execution_actor_guarded_transport.json", {"status": "ok", "guarded_transport_status": "guarded_transport_preview_blocked", "guarded_transport_brief": "guarded_transport_preview_blocked:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "guarded_transport_decision": "do_not_arm_transport_policy_blocked"})
    _write_json(review_dir / "20260315T095140Z_remote_execution_transport_sla.json", {"status": "ok", "transport_sla_status": "shadow_transport_sla_blocked_no_send", "transport_sla_brief": "shadow_transport_sla_blocked_no_send:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um", "transport_sla_decision": "define_sla_before_canary"})
    _write_json(review_dir / "20260315T095150Z_remote_execution_actor_canary_gate.json", {"status": "ok", "canary_gate_status": "shadow_canary_gate_blocked", "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um", "canary_gate_decision": "deny_canary_until_guardian_clear", "blocker_detail": "guardian blocked"})
    _write_json(review_dir / "20260315T095160Z_remote_orderflow_quality_report.json", {"status": "ok", "quality_status": "quality_degraded_guardian_blocked_shadow_only", "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_0:portfolio_margin_um", "quality_recommendation": "keep_downranked_shadow_until_guardian_clear", "quality_score": 0, "blocker_detail": "quality degraded"})
    _write_json(review_dir / "20260315T095170Z_system_time_sync_repair_verification_report.json", {"status": "blocked", "verification_brief": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked", "cleared": False})
    _write_json(review_dir / "20260315T095175Z_remote_shadow_clock_evidence.json", {"status": "ok", "evidence_status": "shadow_clock_evidence_present", "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um", "shadow_learning_allowed": True})
    _write_json(review_dir / "20260315T095180Z_remote_live_boundary_hold.json", {"status": "ok", "hold_status": "live_boundary_hold_active", "hold_brief": "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um", "hold_decision": "keep_shadow_transport_only", "next_transition": "guardian_blocker_clearance", "guardian_blocked": True, "review_blocked": True, "time_sync_blocked": True, "time_sync_mode": "promotion_blocked_shadow_learning_allowed", "remote_shadow_clock_evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um", "remote_shadow_clock_evidence_status": "shadow_clock_evidence_present", "remote_shadow_clock_shadow_learning_allowed": True})

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir), "--now", "2026-03-15T04:03:00+08:00"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "boundary_hardened_remote_guardian"
    assert current["remote_live_boundary_hold_status"] == "live_boundary_hold_active"
    assert current["remote_live_boundary_hold_next_transition"] == "guardian_blocker_clearance"
    assert current["remote_time_sync_mode"] == "promotion_blocked_shadow_learning_allowed"
    assert current["remote_shadow_clock_evidence_status"] == "shadow_clock_evidence_present"
    assert current["remote_shadow_clock_shadow_learning_allowed"] is True
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_guarded_canary_promotion_gate"


def test_build_openclaw_orderflow_blueprint_uses_guardian_clearance_top_blocker_when_present(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T113100Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260315T113105Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "ops_live_gate", "status": "blocked"}]},
    )
    _write_json(
        review_dir / "20260315T113110Z_cross_market_operator_state.json",
        {
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            }
        },
    )
    _write_json(review_dir / "20260315T113115Z_hot_universe_operator_brief.json", {"operator_action_queue": []})
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T113115Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_scope_router_state.json",
        {
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_status": "queued_wait_trade_readiness",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "journal_status": "intent_logged_guardian_blocked",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_openclaw_orderflow_executor_state.json",
        {
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "executor_status": "shadow_guarded_executor_ready",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_orderflow_feedback.json",
        {
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_status": "shadow_policy_blocked",
            "policy_decision": "reject_until_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted",
            "ack_status": "shadow_no_send_ack_recorded",
            "ack_decision": "record_reject_without_transport",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um",
            "actor_status": "shadow_actor_ready_policy_blocked",
            "actor_service_name": "remote_execution_actor.service",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_execution_actor_guarded_transport.json",
        {
            "guarded_transport_brief": "guarded_transport_preview_blocked:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um",
            "guarded_transport_status": "guarded_transport_preview_blocked",
            "guarded_transport_decision": "do_not_arm_transport_policy_blocked",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_execution_transport_sla.json",
        {
            "transport_sla_brief": "shadow_transport_sla_blocked_no_send:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um",
            "transport_sla_status": "shadow_transport_sla_blocked_no_send",
            "transport_sla_decision": "define_sla_before_canary",
        },
    )
    _write_json(
        review_dir / "20260315T113115Z_remote_execution_actor_canary_gate.json",
        {
            "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um",
            "canary_gate_status": "shadow_canary_gate_blocked",
            "canary_gate_decision": "deny_canary_until_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T113116Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_46:portfolio_margin_um",
            "quality_status": "quality_degraded_guardian_blocked_shadow_only",
            "quality_recommendation": "keep_downranked_shadow_until_guardian_clear",
            "quality_score": 46,
        },
    )
    _write_json(
        review_dir / "20260315T113117Z_remote_live_boundary_hold.json",
        {
            "hold_status": "live_boundary_hold_active",
            "hold_brief": "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um",
            "hold_decision": "keep_shadow_transport_only",
            "next_transition": "guardian_blocker_clearance",
            "guardian_blocked": True,
            "review_blocked": True,
            "time_sync_blocked": True,
            "time_sync_mode": "promotion_blocked_shadow_learning_allowed",
            "remote_shadow_clock_evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "remote_shadow_clock_evidence_status": "shadow_clock_evidence_present",
            "remote_shadow_clock_shadow_learning_allowed": True,
        },
    )
    _write_json(
        review_dir / "20260315T113117Z_remote_shadow_clock_evidence.json",
        {
            "evidence_status": "shadow_clock_evidence_present",
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "shadow_learning_allowed": True,
        },
    )
    _write_json(
        review_dir / "20260315T113118Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:5_blocked:portfolio_margin_um",
            "clearance_status": "guardian_blocker_clearance_blocked",
            "clearance_score": 0,
            "top_blocker_code": "timed_ntp_via_fake_ip_clearance",
            "top_blocker_title": "Repair fake-ip NTP path before any orderflow promotion",
            "top_blocker_target_artifact": "system_time_sync_repair_verification_report",
            "top_blocker_next_action": "repair_fake_ip_ntp_path_then_verify",
            "top_blocker_detail": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked | manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip",
            "time_sync_mode": "promotion_blocked_shadow_learning_allowed",
            "remote_shadow_clock_evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "remote_shadow_clock_evidence_status": "shadow_clock_evidence_present",
            "remote_shadow_clock_shadow_learning_allowed": True,
        },
    )
    _write_json(
        review_dir / "20260315T113119Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_shadow_learning_allowed",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_shadow_learning_allowed:SOLUSDT:block_promotion_continue_shadow_learning:portfolio_margin_um",
            "promotion_gate_decision": "block_promotion_continue_shadow_learning",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "promotion_ready": False,
            "promotion_blocker_code": "timed_ntp_via_fake_ip_clearance",
            "promotion_blocker_title": "Repair fake-ip NTP path before any orderflow promotion",
            "promotion_blocker_target_artifact": "system_time_sync_repair_verification_report",
            "promotion_blocker_detail": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked | manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip",
            "time_sync_mode": "promotion_blocked_shadow_learning_allowed",
        },
    )
    _write_json(
        review_dir / "20260315T113120Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "blocker_detail": "",
        },
    )
    _write_json(
        review_dir / "20260315T113121Z_remote_promotion_unblock_readiness.json",
        {
            "readiness_status": "local_time_sync_primary_blocker_shadow_ready",
            "readiness_brief": "local_time_sync_primary_blocker_shadow_ready:SOLUSDT:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um",
            "readiness_decision": "repair_local_fake_ip_ntp_path_then_review_guarded_canary",
            "remote_preconditions_status": "shadow_ready_remote_preconditions_viable",
            "primary_blocker_scope": "timed_ntp_via_fake_ip",
            "primary_local_repair_required": True,
            "primary_local_repair_title": "Repair local fake-ip NTP path to unlock guarded canary review",
            "primary_local_repair_target_artifact": "system_time_sync_repair_verification_report",
            "primary_local_repair_detail": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked | repair_plan=manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip | time_sync_env=timed_ntp_via_fake_ip:timed_source=NTP; ntp_ip=198.18.0.17; delay_ms=118.702; clash_dns_mode=fake-ip; tun_stack=gvisor | time_sync_fix_hint=exclude macOS timed / UDP 123 from Clash TUN fake-ip handling or provide a direct NTP path, then rerun time_sync_probe",
            "primary_local_repair_plan_brief": "manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip",
            "primary_local_repair_environment_classification": "timed_ntp_via_fake_ip",
            "primary_local_repair_environment_blocker_detail": "timed_source=NTP; ntp_ip=198.18.0.17; delay_ms=118.702; clash_dns_mode=fake-ip; tun_stack=gvisor",
        },
    )

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir), "--now", "2026-03-15T04:05:00+08:00"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "local_repair_gated_continuity_hardened_remote_guardian"
    assert current["remote_guardian_blocker_clearance_brief"].startswith(
        "guardian_blocker_clearance_blocked:SOLUSDT"
    )
    assert current["remote_guardian_blocker_clearance_top_blocker_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert current["remote_guarded_canary_promotion_gate_status"] == (
        "guarded_canary_promotion_blocked_shadow_learning_allowed"
    )
    assert current["remote_guarded_canary_promotion_gate_blocker_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert current["remote_shadow_learning_continuity_status"] == "shadow_learning_continuity_stable"
    assert current["remote_shadow_learning_continuity_decision"] == (
        "continue_shadow_learning_collect_feedback"
    )
    assert current["remote_promotion_unblock_readiness_status"] == (
        "local_time_sync_primary_blocker_shadow_ready"
    )
    assert current["remote_promotion_unblock_readiness_decision"] == (
        "repair_local_fake_ip_ntp_path_then_review_guarded_canary"
    )
    assert current["remote_promotion_unblock_primary_blocker_scope"] == "timed_ntp_via_fake_ip"
    assert current["remote_promotion_unblock_primary_local_repair_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert current["remote_promotion_unblock_primary_local_repair_plan_brief"] == (
        "manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip"
    )
    assert current["remote_promotion_unblock_primary_local_repair_environment_classification"] == (
        "timed_ntp_via_fake_ip"
    )
    assert current["remote_time_sync_mode"] == "promotion_blocked_shadow_learning_allowed"
    assert current["remote_shadow_clock_evidence_status"] == "shadow_clock_evidence_present"
    assert payload["immediate_backlog"][0]["title"] == (
        "Repair local fake-ip NTP path to unlock guarded canary review"
    )
    assert "manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip" in payload["immediate_backlog"][0]["why"]
    assert payload["immediate_backlog"][0]["why"].count("ntp_ip=") == 1
    assert payload["immediate_backlog"][0]["target_artifact"] == "system_time_sync_repair_verification_report"


def test_build_openclaw_orderflow_blueprint_promotes_ticket_actionability_and_shortline_slice_to_first_class_sources(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"

    _write_json(
        review_dir / "20260315T100000Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "focus_stack_brief": "gate -> risk_guard",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {"brief": "split_scope_spot_vs_portfolio_margin_um"},
                "remote_live_diagnosis": {
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                },
                "remote_live_history": {"window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"},
            }
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "ops_live_gate"}, {"name": "risk_guard"}]},
    )
    _write_json(
        review_dir / "20260315T100020Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_inside_scope_but_not_trade_ready:SOLUSDT:portfolio_margin_um",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_hot_universe_operator_brief.json",
        {"operator_action_queue": [{"symbol": "SOLUSDT", "action": "deprioritize_flow"}]},
    )
    _write_json(review_dir / "latest_remote_live_history_audit.json", {"status": "ok"})
    _write_json(
        review_dir / "20260315T100040Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_status": "queued_wait_trade_readiness",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T100050Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:ticket_missing:no_actionable_ticket | not_attempted_wait_trade_readiness",
            "journal_status": "intent_logged_guardian_blocked",
            "append_status": "appended",
        },
    )
    _write_json(
        review_dir / "20260315T100060Z_openclaw_orderflow_executor_state.json",
        {
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "executor_status": "shadow_guarded_executor_ready",
            "service_name": "openclaw-orderflow-executor.service",
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_remote_orderflow_feedback.json",
        {
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
        },
    )
    _write_json(
        review_dir / "20260315T100080Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_status": "shadow_policy_learning_only",
            "policy_decision": "accept_shadow_learning_only",
        },
    )
    _write_json(
        review_dir / "20260315T100090Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_learning_ack_recorded:SOLUSDT:not_sent_learning_only:no_fill_execution_not_attempted",
            "ack_status": "shadow_learning_ack_recorded",
            "ack_decision": "record_learning_without_transport",
        },
    )
    _write_json(
        review_dir / "20260315T100100Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_status": "quality_learning_only_shadow_viable",
            "quality_score": 49,
            "shadow_learning_score": 65,
            "execution_readiness_score": 5,
            "transport_observability_score": 100,
        },
    )
    _write_json(
        review_dir / "20260315T100110Z_remote_shadow_clock_evidence.json",
        {
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "evidence_status": "shadow_clock_evidence_present",
            "shadow_learning_allowed": True,
        },
    )
    _write_json(
        review_dir / "20260315T100120Z_remote_live_boundary_hold.json",
        {
            "hold_brief": "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um",
            "hold_status": "live_boundary_hold_active",
            "hold_decision": "keep_shadow_transport_only",
            "next_transition": "guardian_blocker_clearance",
            "guardian_blocked": True,
            "review_blocked": True,
            "time_sync_blocked": False,
        },
    )
    _write_json(
        review_dir / "20260315T100130Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "clearance_status": "guardian_blocker_clearance_blocked",
            "clearance_score": 35,
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Resolve crypto ticket actionability before guarded canary review",
            "top_blocker_target_artifact": "remote_ticket_actionability_state",
            "top_blocker_next_action": "run_shortline_slice_backtest_then_wait_for_setup_ready",
            "top_blocker_done_when": "SOLUSDT reaches Setup_Ready and a fresh crypto ticket row exists for SOLUSDT",
            "top_blocker_detail": "fresh_artifact:ticket_row_missing:SOLUSDT | ticket_surface=commodity_only:XAUUSD,XAGUSD,COPPER",
        },
    )
    _write_json(
        review_dir / "20260315T100140Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "promotion_gate_decision": "clear_guardian_review_blockers_before_promotion",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "promotion_ready": False,
            "promotion_blocker_code": "guardian_ticket_actionability",
            "promotion_blocker_title": "Resolve crypto ticket actionability before guarded canary review",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_detail": "fresh_artifact:ticket_row_missing:SOLUSDT | ticket_surface=commodity_only:XAUUSD,XAGUSD,COPPER",
        },
    )
    _write_json(
        review_dir / "20260315T100150Z_remote_shadow_learning_continuity.json",
        {
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "blocker_detail": "",
        },
    )
    _write_json(
        review_dir / "20260315T100160Z_remote_promotion_unblock_readiness.json",
        {
            "readiness_brief": "shadow_ready_ticket_actionability_blocked:SOLUSDT:resolve_ticket_actionability_then_review_guarded_canary:portfolio_margin_um",
            "readiness_status": "shadow_ready_ticket_actionability_blocked",
            "readiness_decision": "resolve_ticket_actionability_then_review_guarded_canary",
            "remote_preconditions_status": "shadow_ready_remote_preconditions_viable",
            "primary_blocker_scope": "guardian_ticket_actionability",
            "primary_local_repair_required": False,
            "primary_local_repair_title": "Resolve crypto ticket actionability before guarded canary review",
            "primary_local_repair_target_artifact": "remote_ticket_actionability_state",
            "primary_local_repair_detail": "fresh_artifact:ticket_row_missing:SOLUSDT | ticket_surface=commodity_only:XAUUSD,XAGUSD,COPPER",
        },
    )
    _write_json(
        review_dir / "20260315T100170Z_remote_ticket_actionability_state.json",
        {
            "ticket_actionability_brief": "crypto_shortline_bias_only_not_ticketed:SOLUSDT:run_shortline_slice_backtest_and_wait_for_setup_ready:portfolio_margin_um",
            "ticket_actionability_status": "crypto_shortline_bias_only_not_ticketed",
            "ticket_actionability_decision": "run_shortline_slice_backtest_and_wait_for_setup_ready",
            "next_action": "run_shortline_slice_backtest_then_wait_for_setup_ready",
            "next_action_target_artifact": "crypto_shortline_backtest_slice",
        },
    )
    _write_json(
        review_dir / "20260315T100172Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review",
            "freshness_status": "route_signal_row_stale",
            "freshness_decision": "refresh_crypto_signal_source_then_rebuild_tickets",
            "refresh_recommended": True,
        },
    )
    _write_json(
        review_dir / "20260315T100173Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "no_newer_crypto_signal_candidate_available:SOLUSDT:2026-03-06:route_signal_date=2026-02-05",
            "readiness_status": "no_newer_crypto_signal_candidate_available",
            "readiness_decision": "generate_fresh_crypto_signal_source_before_rebuild_tickets",
            "refresh_needed": True,
        },
    )
    _write_json(
        review_dir / "20260315T100175Z_crypto_shortline_material_change_trigger.json",
        {
            "trigger_brief": "no_material_orderflow_change_since_cross_section_anchor:SOLUSDT:wait_for_material_orderflow_change_before_rerun:deprioritize_flow",
            "trigger_status": "no_material_orderflow_change_since_cross_section_anchor",
            "trigger_decision": "wait_for_material_orderflow_change_before_rerun",
            "rerun_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260315T100180Z_crypto_shortline_backtest_slice.json",
        {
            "slice_brief": "selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m",
            "slice_status": "selected_watch_only_orderflow_slice",
            "research_decision": "run_watch_only_orderflow_cross_section_backtest",
            "selected_symbol": "SOLUSDT",
            "slice_universe_brief": "SOLUSDT,BTCUSDT,ETHUSDT,BNBUSDT",
        },
    )
    _write_json(
        review_dir / "20260315T100190Z_crypto_shortline_cross_section_backtest.json",
        {
            "backtest_brief": "watch_only_cross_section_positive:SOLUSDT:positive:h60m",
            "backtest_status": "watch_only_cross_section_positive",
            "research_decision": "continue_shadow_learning_wait_for_setup_ready",
            "selected_edge_status": "positive",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_status"]
    assert current["current_life_stage"] == "ticket_actionability_gated_remote_guardian"
    assert current["remote_ticket_actionability_brief"] == (
        "crypto_shortline_bias_only_not_ticketed:SOLUSDT:run_shortline_slice_backtest_and_wait_for_setup_ready:portfolio_margin_um"
    )
    assert current["remote_ticket_actionability_status"] == "crypto_shortline_bias_only_not_ticketed"
    assert current["remote_ticket_actionability_next_action_target_artifact"] == (
        "crypto_shortline_backtest_slice"
    )
    assert current["crypto_signal_source_refresh_readiness_status"] == (
        "no_newer_crypto_signal_candidate_available"
    )
    assert current["crypto_signal_source_refresh_readiness_decision"] == (
        "generate_fresh_crypto_signal_source_before_rebuild_tickets"
    )
    assert current["crypto_signal_source_refresh_needed"] is True
    assert current["crypto_signal_source_freshness_status"] == "route_signal_row_stale"
    assert current["crypto_signal_source_freshness_decision"] == (
        "refresh_crypto_signal_source_then_rebuild_tickets"
    )
    assert current["crypto_shortline_material_change_trigger_brief"] == (
        "no_material_orderflow_change_since_cross_section_anchor:SOLUSDT:wait_for_material_orderflow_change_before_rerun:deprioritize_flow"
    )
    assert current["crypto_shortline_material_change_trigger_status"] == (
        "no_material_orderflow_change_since_cross_section_anchor"
    )
    assert current["crypto_shortline_material_change_trigger_rerun_recommended"] is False
    assert current["crypto_shortline_backtest_slice_brief"] == (
        "selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m"
    )
    assert current["crypto_shortline_backtest_slice_status"] == "selected_watch_only_orderflow_slice"
    assert current["crypto_shortline_cross_section_backtest_brief"] == (
        "watch_only_cross_section_positive:SOLUSDT:positive:h60m"
    )
    assert current["crypto_shortline_cross_section_backtest_status"] == (
        "watch_only_cross_section_positive"
    )
    assert current["crypto_shortline_cross_section_backtest_selected_edge_status"] == "positive"
    assert payload["immediate_backlog"][0]["title"] == (
        "Resolve crypto ticket actionability before guarded canary review"
    )
    assert payload["immediate_backlog"][0]["target_artifact"] == "remote_ticket_actionability_state"
    assert "fresh_artifact:ticket_row_missing:SOLUSDT" in payload["immediate_backlog"][0]["why"]
    assert payload["current_status"]["artifacts"]["crypto_signal_source_refresh_readiness"].endswith(
        "20260315T100173Z_crypto_signal_source_refresh_readiness.json"
    )
