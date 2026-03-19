from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_daily_ops_skill_checklist.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("daily_ops_skill_checklist_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_daily_ops_skill_checklist(tmp_path: Path, capsys) -> None:
    module = _load_module()
    review_dir = tmp_path / "review"
    docs_dir = tmp_path / "docs"
    review_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    playbook_path = docs_dir / "FENLIE_SKILL_USAGE_PLAYBOOK.md"
    playbook_path.write_text("# playbook\n", encoding="utf-8")

    (review_dir / "20260314T084209Z_hot_universe_operator_brief.json").write_text(
        json.dumps(
            {
                "cross_market_operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
                "cross_market_review_head_brief": "refresh_required:brooks_structure:SC2603:consider_refresh_before_promotion:96",
                "cross_market_review_head_symbol": "SC2603",
                "cross_market_review_head_blocker_detail": "SC2603 requires refreshed structure source. | time-sync=threshold_breach:scope=clock_skew_and_latency; est_offset_ms=74.730; est_rtt_ms=151.402",
                "cross_market_review_head_done_when": "synchronize system clock and reduce network latency/jitter until time-sync offset and RTT stay within configured limits, then rerun time_sync_probe",
                "cross_market_operator_repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
                "cross_market_remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
                "cross_market_operator_backlog_state_brief": "waiting=2 | review=3 | watch=0 | blocked=0 | repair=7",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260314T084200Z_cross_market_operator_state.json").write_text(
        json.dumps(
            {
                "review_head_lane_brief": "refresh_required:brooks_structure:SC2603:consider_refresh_before_promotion:96",
                "operator_action_queue_brief": "1:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence | 2:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence | 3:brooks_structure:SC2603:consider_refresh_before_promotion | 4:ops_live_gate:rollback_hard:clear_ops_live_gate_condition",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260314T090800Z_operator_task_visual_panel.json").write_text(
        json.dumps(
            {
                "summary": {
                    "operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
                    "review_head_brief": "refresh_required:brooks_structure:SC2603:consider_refresh_before_promotion:96",
                    "repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
                    "remote_live_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
                    "lane_state_brief": "waiting=2 | review=3 | watch=0 | blocked=0 | repair=7",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260314T090900Z_system_time_sync_repair_plan.json").write_text(
        json.dumps(
            {
                "plan_brief": "manual_time_repair_required:SC2603:clock_skew_and_latency",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260314T091000Z_system_time_sync_repair_verification_report.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "verification_brief": (
                    "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260314T091005Z_remote_promotion_unblock_readiness.json").write_text(
        json.dumps(
            {
                "readiness_status": "local_time_sync_primary_blocker_shadow_ready",
                "readiness_brief": "local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um",
                "readiness_decision": "repair_local_fake_ip_ntp_path_then_review_guarded_canary",
                "route_symbol": "SC2603",
                "primary_blocker_scope": "timed_ntp_via_fake_ip",
                "primary_local_repair_target_artifact": "system_time_sync_repair_verification_report",
                "primary_local_repair_detail": (
                    "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
                ),
                "done_when": (
                    "shadow learning continuity remains stable, quality stays viable, and the local time-sync verification clears so guarded canary review can move from local repair gate into promotion review"
                ),
                "artifact": str(review_dir / "20260314T091005Z_remote_promotion_unblock_readiness.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    module.main.__wrapped__ if hasattr(module.main, "__wrapped__") else None
    argv = [
        "build_daily_ops_skill_checklist.py",
        "--review-dir",
        str(review_dir),
        "--playbook-path",
        str(playbook_path),
        "--now",
        "2026-03-14T09:10:00Z",
    ]
    import sys
    old_argv = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["operator_head_brief"] == "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99"
    assert payload["review_head_brief"] == "refresh_required:brooks_structure:SC2603:consider_refresh_before_promotion:96"
    assert payload["remote_live_gate_brief"] == "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um"
    assert payload["priority_repair_status"] == "run_now"
    assert payload["priority_repair_brief"] == (
        "repair_local_fake_ip_ntp_path_then_review_guarded_canary:SC2603:timed_ntp_via_fake_ip"
    )
    assert payload["priority_repair_blocker_detail"] == (
        "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert payload["priority_repair_target_artifact"] == "system_time_sync_repair_verification_report"
    assert payload["priority_repair_source_artifact"].endswith(
        "_remote_promotion_unblock_readiness.json"
    )
    assert payload["priority_repair_source_brief"] == (
        "local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um"
    )
    assert payload["priority_repair_plan_brief"] == "manual_time_repair_required:SC2603:clock_skew_and_latency"
    assert payload["priority_repair_plan_artifact"].endswith("_system_time_sync_repair_plan.json")
    assert payload["priority_repair_verification_status"] == "blocked"
    assert payload["priority_repair_verification_brief"] == (
        "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert payload["priority_repair_verification_artifact"].endswith(
        "_system_time_sync_repair_verification_report.json"
    )
    assert payload["checklist_brief"] == (
        "1:source_ownership_review:run_now | 2:cross_market_refresh_audit:run_now | "
        "3:remote_live_guard_diagnostics:run_now | 4:operator_panel_refresh:run_after_source_checks"
    )
    assert len(payload["skills"]) == 4
    assert payload["skills"][0]["skill_id"] == "source_ownership_review"
    assert payload["skills"][0]["status"] == "run_now"
    assert payload["skills"][3]["skill_id"] == "operator_panel_refresh"
    assert payload["skills"][3]["status"] == "run_after_source_checks"
    assert Path(payload["artifact"]).exists()
    assert Path(payload["markdown"]).exists()
    assert Path(payload["checksum"]).exists()
    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "# Daily Ops Skill Checklist" in markdown
    assert "## Priority Repair" in markdown
    assert "repair_local_fake_ip_ntp_path_then_review_guarded_canary:SC2603:timed_ntp_via_fake_ip" in markdown
    assert "_remote_promotion_unblock_readiness.json" in markdown
    assert "_system_time_sync_repair_plan.json" in markdown
    assert "_system_time_sync_repair_verification_report.json" in markdown
    assert "Source Ownership Review" in markdown
    assert "Cross-Market Refresh Audit" in markdown


def test_latest_review_json_artifact_skips_missing_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    live_path = review_dir / "20260315T093110Z_system_time_sync_repair_verification_report.json"
    live_path.write_text('{"status":"blocked"}\n', encoding="utf-8")
    ghost_path = review_dir / "20260315T093120Z_system_time_sync_repair_verification_report.json"

    original_glob = module.Path.glob

    def fake_glob(self: Path, pattern: str):
        if self == review_dir and pattern == "*_system_time_sync_repair_verification_report.json":
            return [ghost_path, live_path]
        return original_glob(self, pattern)

    monkeypatch.setattr(module.Path, "glob", fake_glob)

    selected = module.latest_review_json_artifact(
        review_dir,
        "system_time_sync_repair_verification_report",
        module.parse_now("2026-03-15T09:31:20Z"),
    )

    assert selected == live_path


def test_build_daily_ops_skill_checklist_surfaces_ticket_actionability_as_priority_action(
    tmp_path: Path, capsys
) -> None:
    module = _load_module()
    review_dir = tmp_path / "review"
    docs_dir = tmp_path / "docs"
    review_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    playbook_path = docs_dir / "FENLIE_SKILL_USAGE_PLAYBOOK.md"
    playbook_path.write_text("# playbook\n", encoding="utf-8")

    (review_dir / "20260315T104134Z_hot_universe_operator_brief.json").write_text(
        json.dumps(
            {
                "cross_market_operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
                "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
                "cross_market_review_head_symbol": "SOLUSDT",
                "cross_market_review_head_blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation.",
                "cross_market_review_head_done_when": "liquidity_sweep + mss + cvd_confirmation clear",
                "cross_market_operator_repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
                "cross_market_remote_live_takeover_gate_brief": "current_head_inside_scope_but_not_trade_ready:SOLUSDT:portfolio_margin_um",
                "cross_market_operator_backlog_state_brief": "waiting=2 | review=2 | watch=0 | blocked=0 | repair=1",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260315T104050Z_cross_market_operator_state.json").write_text(
        json.dumps(
            {
                "review_head_lane_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
                "operator_action_queue_brief": "1:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence | 2:crypto_route:SOLUSDT:deprioritize_flow",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260315T104110Z_operator_task_visual_panel.json").write_text(
        json.dumps(
            {
                "summary": {
                    "operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
                    "review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
                    "repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
                    "remote_live_gate_brief": "current_head_inside_scope_but_not_trade_ready:SOLUSDT:portfolio_margin_um",
                    "lane_state_brief": "waiting=2 | review=2 | watch=0 | blocked=0 | repair=1",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readiness_path = review_dir / "20260315T104040Z_remote_promotion_unblock_readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "readiness_status": "shadow_ready_ticket_actionability_blocked",
                "readiness_brief": "shadow_ready_ticket_actionability_blocked:SOLUSDT:wait_for_liquidity_sweep_then_review_guarded_canary:portfolio_margin_um",
                "readiness_decision": "wait_for_liquidity_sweep_then_review_guarded_canary",
                "route_symbol": "SOLUSDT",
                "primary_blocker_scope": "guardian_ticket_actionability",
                "primary_local_repair_target_artifact": "crypto_shortline_gate_stack_progress",
                "primary_local_repair_detail": "fresh_artifact:ticket_row_blocked:SOLUSDT | proxy_price_reference_only",
                "done_when": "SOLUSDT reaches Setup_Ready and a fresh crypto ticket row exists for SOLUSDT",
                "artifact": str(readiness_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    import sys

    old_argv = sys.argv
    sys.argv = [
        "build_daily_ops_skill_checklist.py",
        "--review-dir",
        str(review_dir),
        "--playbook-path",
        str(playbook_path),
        "--now",
        "2026-03-15T10:41:20Z",
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(capsys.readouterr().out)
    assert payload["priority_repair_status"] == "run_now"
    assert payload["priority_repair_brief"] == (
        "wait_for_liquidity_sweep_then_review_guarded_canary:SOLUSDT:guardian_ticket_actionability"
    )
    assert payload["priority_repair_target_artifact"] == "crypto_shortline_gate_stack_progress"
    assert payload["priority_repair_source_artifact"].endswith(
        "_remote_promotion_unblock_readiness.json"
    )
    assert payload["priority_repair_plan_brief"] == ""
    assert payload["priority_repair_verification_brief"] == ""


def test_build_daily_ops_skill_checklist_surfaces_persistent_liquidity_pressure_as_priority_action(
    tmp_path: Path, capsys
) -> None:
    module = _load_module()
    review_dir = tmp_path / "review"
    docs_dir = tmp_path / "docs"
    review_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    playbook_path = docs_dir / "FENLIE_SKILL_USAGE_PLAYBOOK.md"
    playbook_path.write_text("# playbook\n", encoding="utf-8")

    (review_dir / "20260316T042700Z_hot_universe_operator_brief.json").write_text(
        json.dumps(
            {
                "cross_market_operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
                "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
                "cross_market_review_head_symbol": "SOLUSDT",
                "cross_market_review_head_blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation.",
                "cross_market_review_head_done_when": "liquidity_sweep + mss + cvd_confirmation clear",
                "cross_market_operator_repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
                "cross_market_remote_live_takeover_gate_brief": "current_head_inside_scope_but_not_trade_ready:SOLUSDT:portfolio_margin_um",
                "cross_market_operator_backlog_state_brief": "waiting=2 | review=2 | watch=0 | blocked=0 | repair=1",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260316T042650Z_cross_market_operator_state.json").write_text(
        json.dumps(
            {
                "review_head_lane_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
                "operator_action_queue_brief": "1:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence | 2:crypto_route:SOLUSDT:deprioritize_flow",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260316T042730Z_operator_task_visual_panel.json").write_text(
        json.dumps(
            {
                "summary": {
                    "operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
                    "review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
                    "repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
                    "remote_live_gate_brief": "current_head_inside_scope_but_not_trade_ready:SOLUSDT:portfolio_margin_um",
                    "lane_state_brief": "waiting=2 | review=2 | watch=0 | blocked=0 | repair=1",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readiness_path = review_dir / "20260316T042700Z_remote_promotion_unblock_readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "readiness_status": "shadow_ready_ticket_actionability_blocked",
                "readiness_brief": "shadow_ready_ticket_actionability_blocked:SOLUSDT:monitor_persistent_orderflow_pressure_for_liquidity_sweep_then_review_guarded_canary:portfolio_margin_um",
                "readiness_decision": "monitor_persistent_orderflow_pressure_for_liquidity_sweep_then_review_guarded_canary",
                "route_symbol": "SOLUSDT",
                "primary_blocker_scope": "guardian_ticket_actionability",
                "primary_local_repair_target_artifact": "crypto_shortline_liquidity_event_trigger",
                "primary_local_repair_detail": "liquidity_event_trigger=liquidity_sweep_pressure_persisting:SOLUSDT:monitor_persistent_orderflow_pressure_for_liquidity_sweep:portfolio_margin_um",
                "done_when": "persistent orderflow pressure resolves into liquidity sweep and setup-ready confirmation",
                "artifact": str(readiness_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    import sys

    old_argv = sys.argv
    sys.argv = [
        "build_daily_ops_skill_checklist.py",
        "--review-dir",
        str(review_dir),
        "--playbook-path",
        str(playbook_path),
        "--now",
        "2026-03-16T04:27:40Z",
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(capsys.readouterr().out)
    assert payload["priority_repair_status"] == "run_now"
    assert payload["priority_repair_brief"] == (
        "monitor_persistent_orderflow_pressure_for_liquidity_sweep_then_review_guarded_canary:SOLUSDT:guardian_ticket_actionability"
    )
    assert payload["priority_repair_target_artifact"] == "crypto_shortline_liquidity_event_trigger"
    assert payload["priority_repair_source_artifact"].endswith(
        "_remote_promotion_unblock_readiness.json"
    )
    assert payload["priority_repair_plan_brief"] == ""
    assert payload["priority_repair_verification_brief"] == ""
