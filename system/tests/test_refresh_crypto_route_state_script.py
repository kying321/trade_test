from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/refresh_crypto_route_state.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("crypto_route_refresh_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_now_is_monotonic() -> None:
    mod = _load_module()
    base = mod.parse_now("2026-03-12T05:40:00Z")
    assert mod.step_now(base, 0).isoformat() == "2026-03-12T05:40:00+00:00"
    assert mod.step_now(base, 4).isoformat() == "2026-03-12T05:40:04+00:00"
    assert mod.step_now(base, 5) > mod.step_now(base, 4)


def test_derive_runtime_now_ignores_far_future_stamped_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260316T075723Z_crypto_route_refresh.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mod, "now_utc", lambda: mod.parse_now("2026-03-16T05:19:50Z"))

    derived = mod.derive_runtime_now(review_dir, None)

    assert derived.isoformat() == "2026-03-16T05:19:50+00:00"


def test_normalize_output_root_accepts_system_root_when_review_dir_is_under_output(tmp_path: Path) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    normalized = mod.normalize_output_root(system_root, review_dir)

    assert normalized == system_root / "output"


def test_main_runs_guarded_refresh_chain_and_writes_refresh_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    micro_capture_dir = output_root / "artifacts" / "micro_capture"
    micro_capture_dir.mkdir(parents=True, exist_ok=True)
    micro_capture_path = micro_capture_dir / "20260312T054508Z_micro_capture.json"
    micro_capture_path.write_text(json.dumps({"status": "degraded", "selected_micro": []}), encoding="utf-8")
    (
        review_dir / "20260312T053000Z_hot_universe_research.json"
    ).write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    calls: list[tuple[str, list[str]]] = []
    scripted_wall_clock = {
        "build_source_control_report": "2026-03-12T05:45:00Z",
        "build_native_group_report": "2026-03-12T05:45:01Z",
        "build_native_lane_stability_report": "2026-03-12T05:45:02Z",
        "build_native_beta_leg_report": "2026-03-12T05:45:03Z",
        "build_native_lane_playbook": "2026-03-12T05:45:04Z",
        "build_native_beta_leg_window_report": "2026-03-12T05:45:05Z",
        "build_combo_playbook": "2026-03-12T05:45:06Z",
        "build_symbol_route_handoff": "2026-03-12T05:45:07Z",
        "build_system_time_sync_environment_report": "2026-03-12T05:45:08Z",
        "build_crypto_cvd_semantic_snapshot": "2026-03-12T05:45:09Z",
        "build_crypto_cvd_queue_handoff": "2026-03-12T05:45:10Z",
        "build_crypto_shortline_live_bars_snapshot": "2026-03-12T05:45:12Z",
        "build_crypto_shortline_execution_gate": "2026-03-12T05:45:13Z",
        "build_crypto_shortline_profile_location_watch": "2026-03-12T05:45:14Z",
        "build_crypto_shortline_mss_watch": "2026-03-12T05:45:15Z",
        "build_crypto_shortline_cvd_confirmation_watch": "2026-03-12T05:45:16Z",
        "build_crypto_shortline_retest_watch": "2026-03-12T05:45:17Z",
        "build_crypto_shortline_pattern_router": "2026-03-12T05:45:18Z",
        "build_crypto_route_brief": "2026-03-12T05:45:19Z",
        "build_crypto_route_operator_brief": "2026-03-12T05:45:20Z",
        "build_crypto_shortline_material_change_trigger": "2026-03-12T05:45:21Z",
        "build_crypto_shortline_liquidity_event_trigger": "2026-03-12T05:45:22Z",
        "build_crypto_shortline_live_orderflow_snapshot": "2026-03-12T05:45:23Z",
        "build_crypto_shortline_execution_quality_watch": "2026-03-12T05:45:24Z",
        "build_signal_to_order_tickets": "2026-03-12T05:45:25Z",
        "build_crypto_shortline_slippage_snapshot": "2026-03-12T05:45:26Z",
        "build_crypto_shortline_fill_capacity_watch": "2026-03-12T05:45:27Z",
        "build_crypto_shortline_sizing_watch": "2026-03-12T05:45:28Z",
        "build_crypto_shortline_signal_quality_watch": "2026-03-12T05:45:29Z",
    }

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append((step_name, list(cmd)))
        artifact_name = f"{step_name}.json"
        if step_name == "build_crypto_route_brief":
            artifact_name = "crypto_route_brief.json"
        if step_name == "build_crypto_cvd_semantic_snapshot":
            artifact_name = "crypto_cvd_semantic_snapshot.json"
        if step_name == "build_crypto_cvd_queue_handoff":
            artifact_name = "crypto_cvd_queue_handoff.json"
        if step_name == "build_crypto_shortline_execution_gate":
            artifact_name = "crypto_shortline_execution_gate.json"
        if step_name == "build_crypto_shortline_profile_location_watch":
            artifact_name = "crypto_shortline_profile_location_watch.json"
        if step_name == "build_crypto_shortline_mss_watch":
            artifact_name = "crypto_shortline_mss_watch.json"
        if step_name == "build_crypto_shortline_cvd_confirmation_watch":
            artifact_name = "crypto_shortline_cvd_confirmation_watch.json"
        if step_name == "build_crypto_shortline_retest_watch":
            artifact_name = "crypto_shortline_retest_watch.json"
        if step_name == "build_crypto_shortline_pattern_router":
            artifact_name = "crypto_shortline_pattern_router.json"
        if step_name == "build_crypto_shortline_live_bars_snapshot":
            artifact_name = "crypto_shortline_live_bars_snapshot.json"
        if step_name == "build_crypto_shortline_material_change_trigger":
            artifact_name = "crypto_shortline_material_change_trigger.json"
        if step_name == "build_crypto_shortline_liquidity_event_trigger":
            artifact_name = "crypto_shortline_liquidity_event_trigger.json"
        if step_name == "build_crypto_shortline_live_orderflow_snapshot":
            artifact_name = "crypto_shortline_live_orderflow_snapshot.json"
        if step_name == "build_crypto_shortline_execution_quality_watch":
            artifact_name = "crypto_shortline_execution_quality_watch.json"
        if step_name == "build_crypto_shortline_slippage_snapshot":
            artifact_name = "crypto_shortline_slippage_snapshot.json"
        if step_name == "build_crypto_shortline_fill_capacity_watch":
            artifact_name = "crypto_shortline_fill_capacity_watch.json"
        if step_name == "build_crypto_shortline_sizing_watch":
            artifact_name = "crypto_shortline_sizing_watch.json"
        if step_name == "build_crypto_shortline_signal_quality_watch":
            artifact_name = "crypto_shortline_signal_quality_watch.json"
        if step_name == "build_crypto_route_operator_brief":
            artifact_name = "crypto_route_operator_brief.json"
        if "--now" in cmd:
            as_of = cmd[cmd.index("--now") + 1]
        else:
            as_of = scripted_wall_clock[step_name]
        payload = {
            "ok": True,
            "status": "ok",
            "as_of": as_of,
            "artifact": str(review_dir / artifact_name),
        }
        if step_name == "build_crypto_cvd_semantic_snapshot":
            payload.update(
                {
                    "environment_status": "ok",
                    "environment_classification": "none",
                    "environment_blocker_detail": "micro public data environment looks normal.",
                    "environment_remediation_hint": "keep current network path",
                    "time_sync_status": "degraded",
                    "time_sync_classification": "fake_ip_dns_intercept",
                    "time_sync_intercept_scope": "environment_wide",
                    "time_sync_blocker_detail": "intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com",
                    "time_sync_remediation_hint": "disable fake-ip DNS interception or bypass proxy DNS for SNTP/NTP hosts; switching source hostnames is unlikely to help until environment routing is fixed, then rerun time_sync_probe",
                    "time_sync_environment_classification": "timed_ntp_via_fake_ip",
                    "time_sync_environment_blocker_detail": "timed_source=NTP; ntp_ip=198.18.0.64",
                }
            )
        if step_name == "build_system_time_sync_environment_report":
            payload.update(
                {
                    "classification": "timed_ntp_via_fake_ip",
                    "blocker_detail": "timed_source=NTP; ntp_ip=198.18.0.64",
                    "remediation_hint": "exclude macOS timed / UDP 123 from Clash TUN fake-ip handling or provide a direct NTP path, then rerun time_sync_probe",
                }
            )
        if step_name == "build_crypto_route_brief":
            payload.update(
                {
                    "next_focus_symbol": "SOLUSDT",
                    "next_focus_action": "deprioritize_flow",
                    "focus_review_status": "review_no_edge_bias_only_micro_veto",
                    "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
                    "focus_review_primary_blocker": "no_edge",
                    "focus_review_micro_blocker": "low_sample_or_gap_risk",
                    "focus_review_edge_score": 5,
                    "focus_review_structure_score": 25,
                    "focus_review_micro_score": 20,
                    "focus_review_composite_score": 17,
                    "focus_review_priority_status": "ready",
                    "focus_review_priority_score": 17,
                    "focus_review_priority_tier": "deprioritized_review",
                    "focus_review_priority_brief": "deprioritized_review:17/100",
                    "review_priority_queue_status": "ready",
                    "review_priority_queue_count": 2,
                    "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25",
                    "review_priority_head_symbol": "SOLUSDT",
                    "review_priority_head_tier": "review_queue_now",
                    "review_priority_head_score": 73,
                    "focus_review_done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
                }
            )
        if step_name == "build_crypto_route_operator_brief":
            payload.update(
                {
                    "next_focus_symbol": "SOLUSDT",
                    "next_focus_action": "deprioritize_flow",
                    "focus_review_status": "review_no_edge_bias_only_micro_veto",
                    "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
                    "focus_review_primary_blocker": "no_edge",
                    "focus_review_micro_blocker": "low_sample_or_gap_risk",
                    "focus_review_edge_score": 5,
                    "focus_review_structure_score": 25,
                    "focus_review_micro_score": 20,
                    "focus_review_composite_score": 17,
                    "focus_review_priority_status": "ready",
                    "focus_review_priority_score": 17,
                    "focus_review_priority_tier": "deprioritized_review",
                    "focus_review_priority_brief": "deprioritized_review:17/100",
                    "review_priority_queue_status": "ready",
                    "review_priority_queue_count": 2,
                    "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25",
                    "review_priority_head_symbol": "SOLUSDT",
                    "review_priority_head_tier": "review_queue_now",
                    "review_priority_head_score": 73,
                    "focus_review_done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
                }
            )
        if step_name == "build_crypto_shortline_material_change_trigger":
            payload.update(
                {
                    "trigger_status": "no_material_orderflow_change_since_cross_section_anchor",
                    "trigger_brief": "no_material_orderflow_change_since_cross_section_anchor:SOLUSDT:wait_for_material_orderflow_change_before_rerun:deprioritize_flow",
                    "trigger_decision": "wait_for_material_orderflow_change_before_rerun",
                    "rerun_recommended": False,
                }
            )
        if step_name == "build_crypto_shortline_sizing_watch":
            payload.update(
                {
                    "watch_status": "value_rotation_scalp_ticket_size_below_min_notional",
                    "watch_brief": "value_rotation_scalp_ticket_size_below_min_notional:SOLUSDT:raise_value_rotation_effective_size_then_recheck_execution_gate:portfolio_margin_um",
                    "watch_decision": "raise_value_rotation_effective_size_then_recheck_execution_gate",
                    "pattern_family": "value_rotation_scalp",
                    "pattern_stage": "profile_alignment",
                }
            )
        if step_name == "build_crypto_shortline_signal_quality_watch":
            payload.update(
                {
                    "watch_status": "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold",
                    "watch_brief": "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold:SOLUSDT:improve_value_rotation_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
                    "watch_decision": "improve_value_rotation_signal_quality_then_recheck_execution_gate",
                    "pattern_family": "value_rotation_scalp",
                    "pattern_stage": "profile_alignment",
                }
            )
        if step_name == "build_crypto_shortline_cvd_confirmation_watch":
            payload.update(
                {
                    "watch_status": "cvd_confirmation_waiting_after_mss",
                    "watch_brief": "cvd_confirmation_waiting_after_mss:SOLUSDT:wait_for_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
                    "watch_decision": "wait_for_cvd_confirmation_then_recheck_execution_gate",
                }
            )
        if step_name == "build_crypto_shortline_retest_watch":
            payload.update(
                {
                    "watch_status": "value_rotation_scalp_retest_precondition_cvd_pending",
                    "watch_brief": "value_rotation_scalp_retest_precondition_cvd_pending:SOLUSDT:wait_for_value_rotation_cvd_confirmation_then_recheck_retest:portfolio_margin_um",
                    "watch_decision": "wait_for_value_rotation_cvd_confirmation_then_recheck_retest",
                }
            )
        if step_name == "build_crypto_shortline_liquidity_event_trigger":
            payload.update(
                {
                    "trigger_status": "no_liquidity_sweep_event_observed",
                    "trigger_brief": "no_liquidity_sweep_event_observed:SOLUSDT:wait_for_liquidity_sweep_event_then_recheck_execution_gate:portfolio_margin_um",
                    "trigger_decision": "wait_for_liquidity_sweep_event_then_recheck_execution_gate",
                    "current_signal_source": "bars_live_snapshot",
                    "current_signal_source_artifact": str(
                        output_root / "research" / "20260312_054500" / "bars_used.csv"
                    ),
                    "previous_signal_source": "previous_liquidity_event_trigger",
                    "previous_signal_source_artifact": str(
                        review_dir / "20260312T054459Z_crypto_shortline_liquidity_event_trigger.json"
                    ),
                }
            )
        if step_name == "build_crypto_shortline_live_orderflow_snapshot":
            payload.update(
                {
                    "snapshot_status": "live_orderflow_snapshot_degraded",
                    "snapshot_brief": "live_orderflow_snapshot_degraded:SOLUSDT:repair_live_orderflow_snapshot_quality_then_recheck_signal_quality:portfolio_margin_um",
                    "snapshot_decision": "repair_live_orderflow_snapshot_quality_then_recheck_signal_quality",
                }
            )
        if step_name == "build_crypto_shortline_execution_quality_watch":
            payload.update(
                {
                    "watch_status": "value_rotation_scalp_execution_quality_degraded",
                    "watch_brief": "value_rotation_scalp_execution_quality_degraded:SOLUSDT:repair_value_rotation_execution_quality_then_recheck_execution_gate:portfolio_margin_um",
                    "watch_decision": "repair_value_rotation_execution_quality_then_recheck_execution_gate",
                    "pattern_family": "value_rotation_scalp",
                    "pattern_stage": "profile_alignment",
                }
            )
        if step_name == "build_crypto_shortline_slippage_snapshot":
            payload.update(
                {
                    "snapshot_status": "value_rotation_scalp_post_cost_degraded",
                    "snapshot_brief": "value_rotation_scalp_post_cost_degraded:SOLUSDT:repair_value_rotation_post_cost_then_recheck_execution_gate:portfolio_margin_um",
                    "snapshot_decision": "repair_value_rotation_post_cost_then_recheck_execution_gate",
                    "pattern_family": "value_rotation_scalp",
                    "pattern_stage": "profile_alignment",
                    "post_cost_viable": False,
                    "estimated_roundtrip_cost_bps": 18.5,
                }
            )
        if step_name == "build_crypto_shortline_fill_capacity_watch":
            payload.update(
                {
                    "watch_status": "value_rotation_scalp_fill_capacity_constrained",
                    "watch_brief": "value_rotation_scalp_fill_capacity_constrained:SOLUSDT:repair_value_rotation_fill_capacity_then_recheck_execution_gate:portfolio_margin_um",
                    "watch_decision": "repair_value_rotation_fill_capacity_then_recheck_execution_gate",
                    "pattern_family": "value_rotation_scalp",
                    "pattern_stage": "profile_alignment",
                    "fill_capacity_viable": False,
                    "entry_headroom_bps": 0.75,
                }
            )
        if step_name == "build_crypto_shortline_live_bars_snapshot":
            payload.update(
                {
                    "snapshot_status": "bars_live_snapshot_ready",
                    "snapshot_brief": "bars_live_snapshot_ready:SOLUSDT:use_live_bars_snapshot_for_shortline_event_detection:portfolio_margin_um",
                    "snapshot_decision": "use_live_bars_snapshot_for_shortline_event_detection",
                }
            )
        if step_name == "build_crypto_shortline_profile_location_watch":
            payload.update(
                {
                    "watch_status": "profile_location_lvn_key_level_ready_rotation_approaching",
                    "watch_brief": "profile_location_lvn_key_level_ready_rotation_approaching:SOLUSDT:monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
                    "watch_decision": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
                    "blocker_title": "Track final LVN-to-HVN/POC rotation before shortline setup promotion",
                    "rotation_proximity_state": "approaching",
                    "profile_rotation_alignment_band": "approaching",
                    "profile_rotation_next_milestone": "into_final_band",
                    "profile_rotation_confidence": 0.53,
                    "active_rotation_targets": ["HVN", "POC"],
                    "profile_rotation_target_tag": "HVN",
                    "profile_rotation_target_bin_distance": 2,
                    "profile_rotation_target_distance_bps": 219.298246,
                }
            )
        if step_name == "build_crypto_shortline_mss_watch":
            payload.update(
                {
                    "watch_status": "mss_waiting_after_liquidity_sweep",
                    "watch_brief": "mss_waiting_after_liquidity_sweep:SOLUSDT:wait_for_mss_then_recheck_execution_gate:portfolio_margin_um",
                    "watch_decision": "wait_for_mss_then_recheck_execution_gate",
                    "blocker_title": "Track market-structure shift after liquidity sweep before shortline setup promotion",
                }
            )
        if step_name == "build_crypto_shortline_pattern_router":
            payload.update(
                {
                    "pattern_status": "value_rotation_scalp_wait_profile_alignment_final",
                    "pattern_brief": "value_rotation_scalp_wait_profile_alignment_final:SOLUSDT:monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um:value_rotation_scalp",
                    "pattern_decision": "monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate",
                    "pattern_family": "value_rotation_scalp",
                    "pattern_stage": "profile_alignment",
                }
            )
        if step_name == "build_signal_to_order_tickets":
            payload = {
                "json": str(review_dir / "signal_to_order_tickets.json"),
                "md": str(review_dir / "signal_to_order_tickets.md"),
                "checksum": str(review_dir / "signal_to_order_tickets_checksum.json"),
                "signal_source": {
                    "kind": "crypto_shortline_signal_source",
                    "symbol_scope_mode": "source_narrowed_default_symbols",
                },
                "summary": {
                    "proxy_price_only_count": 0,
                },
            }
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-12T05:40:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["failure_step_count"] == 0
    assert payload["failure_step_brief"] == ""
    assert payload["cvd_semantic_snapshot_artifact"] == str(review_dir / "crypto_cvd_semantic_snapshot.json")
    assert payload["cvd_semantic_time_sync_status"] == "degraded"
    assert payload["cvd_semantic_time_sync_classification"] == "fake_ip_dns_intercept"
    assert payload["cvd_semantic_time_sync_blocker_detail"] == (
        "intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com"
    )
    assert payload["cvd_queue_handoff_artifact"] == str(review_dir / "crypto_cvd_queue_handoff.json")
    assert payload["shortline_execution_gate_artifact"] == str(review_dir / "crypto_shortline_execution_gate.json")
    assert payload["shortline_pattern_router_artifact"] == str(
        review_dir / "crypto_shortline_pattern_router.json"
    )
    assert payload["shortline_profile_location_watch_artifact"] == str(
        review_dir / "crypto_shortline_profile_location_watch.json"
    )
    assert payload["shortline_mss_watch_artifact"] == str(
        review_dir / "crypto_shortline_mss_watch.json"
    )
    assert payload["shortline_cvd_confirmation_watch_artifact"] == str(
        review_dir / "crypto_shortline_cvd_confirmation_watch.json"
    )
    assert payload["shortline_retest_watch_artifact"] == str(
        review_dir / "crypto_shortline_retest_watch.json"
    )
    assert payload["shortline_live_bars_snapshot_artifact"] == str(
        review_dir / "crypto_shortline_live_bars_snapshot.json"
    )
    assert payload["shortline_material_change_trigger_artifact"] == str(
        review_dir / "crypto_shortline_material_change_trigger.json"
    )
    assert payload["shortline_liquidity_event_trigger_artifact"] == str(
        review_dir / "crypto_shortline_liquidity_event_trigger.json"
    )
    assert payload["shortline_live_orderflow_snapshot_artifact"] == str(
        review_dir / "crypto_shortline_live_orderflow_snapshot.json"
    )
    assert payload["shortline_execution_quality_watch_artifact"] == str(
        review_dir / "crypto_shortline_execution_quality_watch.json"
    )
    assert payload["shortline_fill_capacity_watch_artifact"] == str(
        review_dir / "crypto_shortline_fill_capacity_watch.json"
    )
    assert payload["shortline_sizing_watch_artifact"] == str(
        review_dir / "crypto_shortline_sizing_watch.json"
    )
    assert payload["shortline_signal_quality_watch_artifact"] == str(
        review_dir / "crypto_shortline_signal_quality_watch.json"
    )
    assert payload["signal_to_order_tickets_artifact"] == str(
        review_dir / "signal_to_order_tickets.json"
    )
    assert payload["signal_to_order_tickets_signal_source_kind"] == "crypto_shortline_signal_source"
    assert payload["signal_to_order_tickets_symbol_scope_mode"] == "source_narrowed_default_symbols"
    assert payload["signal_to_order_tickets_proxy_price_only_count"] == 0
    assert payload["crypto_route_brief_artifact"] == str(review_dir / "crypto_route_brief.json")
    assert payload["crypto_route_operator_brief_artifact"] == str(review_dir / "crypto_route_operator_brief.json")
    assert payload["source_control_artifact"] == str(review_dir / "build_source_control_report.json")
    assert payload["route_handoff_artifact"] == str(review_dir / "build_symbol_route_handoff.json")
    assert payload["system_time_sync_environment_report_artifact"] == str(
        review_dir / "build_system_time_sync_environment_report.json"
    )
    assert payload["system_time_sync_environment_report_classification"] == "timed_ntp_via_fake_ip"
    assert payload["system_time_sync_environment_report_blocker_detail"] == (
        "timed_source=NTP; ntp_ip=198.18.0.64"
    )
    assert payload["crypto_route_refresh_reuse_level"] == "informational"
    assert payload["crypto_route_refresh_reuse_gate_status"] == "fresh_non_blocking"
    assert payload["crypto_route_refresh_reuse_gate_brief"] == "fresh_non_blocking:full_guarded_refresh:0/9"
    assert payload["crypto_route_refresh_reuse_gate_blocking"] is False
    assert payload["focus_review_status"] == "review_no_edge_bias_only_micro_veto"
    assert payload["focus_review_primary_blocker"] == "no_edge"
    assert payload["focus_review_micro_blocker"] == "low_sample_or_gap_risk"
    assert payload["shortline_material_change_trigger_status"] == (
        "no_material_orderflow_change_since_cross_section_anchor"
    )
    assert payload["shortline_material_change_trigger_decision"] == (
        "wait_for_material_orderflow_change_before_rerun"
    )
    assert payload["shortline_material_change_trigger_rerun_recommended"] is False
    assert payload["shortline_liquidity_event_trigger_status"] == "no_liquidity_sweep_event_observed"
    assert payload["shortline_liquidity_event_trigger_decision"] == (
        "wait_for_liquidity_sweep_event_then_recheck_execution_gate"
    )
    assert payload["shortline_liquidity_event_trigger_current_signal_source"] == (
        "bars_live_snapshot"
    )
    assert payload["shortline_liquidity_event_trigger_current_signal_source_artifact"] == str(
        output_root / "research" / "20260312_054500" / "bars_used.csv"
    )
    assert payload["shortline_liquidity_event_trigger_previous_signal_source"] == (
        "previous_liquidity_event_trigger"
    )
    assert payload["shortline_liquidity_event_trigger_previous_signal_source_artifact"] == str(
        review_dir / "20260312T054459Z_crypto_shortline_liquidity_event_trigger.json"
    )
    assert payload["shortline_live_bars_snapshot_status"] == "bars_live_snapshot_ready"
    assert payload["shortline_live_bars_snapshot_decision"] == (
        "use_live_bars_snapshot_for_shortline_event_detection"
    )
    assert payload["shortline_profile_location_watch_status"] == (
        "profile_location_lvn_key_level_ready_rotation_approaching"
    )
    assert payload["shortline_profile_location_watch_decision"] == (
        "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["shortline_profile_location_watch_rotation_proximity_state"] == "approaching"
    assert (
        payload["shortline_profile_location_watch_profile_rotation_alignment_band"]
        == "approaching"
    )
    assert (
        payload["shortline_profile_location_watch_profile_rotation_next_milestone"]
        == "into_final_band"
    )
    assert payload["shortline_profile_location_watch_profile_rotation_confidence"] == 0.53
    assert payload["shortline_profile_location_watch_active_rotation_targets"] == ["HVN", "POC"]
    assert payload["shortline_profile_location_watch_profile_rotation_target_tag"] == "HVN"
    assert payload["shortline_profile_location_watch_profile_rotation_target_bin_distance"] == 2
    assert payload["shortline_profile_location_watch_profile_rotation_target_distance_bps"] == 219.298246
    assert payload["shortline_pattern_router_status"] == (
        "value_rotation_scalp_wait_profile_alignment_final"
    )
    assert payload["shortline_pattern_router_decision"] == (
        "monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["shortline_pattern_router_family"] == "value_rotation_scalp"
    assert payload["shortline_pattern_router_stage"] == "profile_alignment"
    assert payload["shortline_mss_watch_status"] == "mss_waiting_after_liquidity_sweep"
    assert payload["shortline_mss_watch_decision"] == (
        "wait_for_mss_then_recheck_execution_gate"
    )
    assert payload["shortline_cvd_confirmation_watch_status"] == (
        "cvd_confirmation_waiting_after_mss"
    )
    assert payload["shortline_cvd_confirmation_watch_decision"] == (
        "wait_for_cvd_confirmation_then_recheck_execution_gate"
    )
    assert payload["shortline_retest_watch_status"] == (
        "value_rotation_scalp_retest_precondition_cvd_pending"
    )
    assert payload["shortline_retest_watch_decision"] == (
        "wait_for_value_rotation_cvd_confirmation_then_recheck_retest"
    )
    assert payload["shortline_live_orderflow_snapshot_status"] == (
        "live_orderflow_snapshot_degraded"
    )
    assert payload["shortline_live_orderflow_snapshot_decision"] == (
        "repair_live_orderflow_snapshot_quality_then_recheck_signal_quality"
    )
    assert payload["shortline_execution_quality_watch_status"] == (
        "value_rotation_scalp_execution_quality_degraded"
    )
    assert payload["shortline_execution_quality_watch_decision"] == (
        "repair_value_rotation_execution_quality_then_recheck_execution_gate"
    )
    assert payload["shortline_execution_quality_watch_pattern_family"] == "value_rotation_scalp"
    assert payload["shortline_execution_quality_watch_pattern_stage"] == "profile_alignment"
    assert payload["shortline_slippage_snapshot_status"] == (
        "value_rotation_scalp_post_cost_degraded"
    )
    assert payload["shortline_slippage_snapshot_decision"] == (
        "repair_value_rotation_post_cost_then_recheck_execution_gate"
    )
    assert payload["shortline_slippage_snapshot_pattern_family"] == "value_rotation_scalp"
    assert payload["shortline_slippage_snapshot_pattern_stage"] == "profile_alignment"
    assert payload["shortline_slippage_snapshot_post_cost_viable"] is False
    assert payload["shortline_slippage_snapshot_estimated_roundtrip_cost_bps"] == 18.5
    assert payload["shortline_fill_capacity_watch_status"] == (
        "value_rotation_scalp_fill_capacity_constrained"
    )
    assert payload["shortline_fill_capacity_watch_decision"] == (
        "repair_value_rotation_fill_capacity_then_recheck_execution_gate"
    )
    assert payload["shortline_fill_capacity_watch_pattern_family"] == "value_rotation_scalp"
    assert payload["shortline_fill_capacity_watch_pattern_stage"] == "profile_alignment"
    assert payload["shortline_fill_capacity_watch_fill_capacity_viable"] is False
    assert payload["shortline_fill_capacity_watch_entry_headroom_bps"] == 0.75
    assert payload["shortline_sizing_watch_status"] == "value_rotation_scalp_ticket_size_below_min_notional"
    assert payload["shortline_sizing_watch_decision"] == (
        "raise_value_rotation_effective_size_then_recheck_execution_gate"
    )
    assert payload["shortline_sizing_watch_pattern_family"] == "value_rotation_scalp"
    assert payload["shortline_sizing_watch_pattern_stage"] == "profile_alignment"
    assert payload["shortline_signal_quality_watch_status"] == "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold"
    assert payload["shortline_signal_quality_watch_decision"] == (
        "improve_value_rotation_signal_quality_then_recheck_execution_gate"
    )
    assert payload["shortline_signal_quality_watch_pattern_family"] == "value_rotation_scalp"
    assert payload["shortline_signal_quality_watch_pattern_stage"] == "profile_alignment"
    assert payload["focus_review_edge_score"] == 5
    assert payload["focus_review_composite_score"] == 17
    assert payload["focus_review_priority_status"] == "ready"
    assert payload["focus_review_priority_score"] == 17
    assert payload["focus_review_priority_tier"] == "deprioritized_review"
    assert payload["review_priority_queue_status"] == "ready"
    assert payload["review_priority_queue_count"] == 2
    assert payload["review_priority_head_symbol"] == "SOLUSDT"
    assert payload["review_priority_head_tier"] == "review_queue_now"
    assert payload["review_priority_head_score"] == 73
    assert payload["crypto_route_head_source_refresh_status"] == "ready"
    assert payload["crypto_route_head_source_refresh_brief"] == "ready:SOLUSDT:read_current_artifact"
    assert payload["crypto_route_head_source_refresh_symbol"] == "SOLUSDT"
    assert payload["crypto_route_head_source_refresh_action"] == "read_current_artifact"
    assert payload["crypto_route_head_source_refresh_source_kind"] == "crypto_route"
    assert payload["crypto_route_head_source_refresh_source_health"] == "ready"
    assert payload["crypto_route_head_source_refresh_source_artifact"] == str(review_dir / "crypto_route_operator_brief.json")
    assert payload["crypto_route_head_source_refresh_priority_tier"] == "review_queue_now"
    assert payload["crypto_route_head_source_refresh_route_action"] == "deprioritize_flow"
    assert payload["crypto_route_head_source_refresh_blocker_detail"].startswith(
        "SOLUSDT currently uses the freshly refreshed crypto_route artifact"
    )
    assert payload["crypto_route_head_source_refresh_done_when"].startswith(
        "keep SOLUSDT on the current crypto_route artifact"
    )
    assert payload["crypto_route_head_downstream_embedding_status"] == "carry_over_non_blocking"
    assert payload["crypto_route_head_downstream_embedding_brief"] == "carry_over_non_blocking:SOLUSDT"
    assert payload["crypto_route_head_downstream_embedding_artifact"] == str(
        review_dir / "20260312T053000Z_hot_universe_research.json"
    )
    assert payload["crypto_route_head_downstream_embedding_as_of"] == "2026-03-12T05:30:00Z"
    assert "broader carry-over outside route-refresh scope" in payload[
        "crypto_route_head_downstream_embedding_blocker_detail"
    ]
    assert payload["artifact"].endswith("_crypto_route_refresh.json")
    assert Path(payload["artifact"]).exists()
    step_now_map = {row["name"]: row["now"] for row in payload["steps"]}
    assert step_now_map["build_crypto_cvd_queue_handoff"].endswith("Z")
    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "- status: `ok`" in markdown
    assert "- failure_step_count: `0`" in markdown
    assert "## Refresh Reuse Gate" in markdown
    assert "- brief: `fresh_non_blocking:full_guarded_refresh:0/9`" in markdown
    assert "## Head Source Refresh" in markdown
    assert "- brief: `ready:SOLUSDT:read_current_artifact`" in markdown
    assert "- route_action: `deprioritize_flow`" in markdown
    assert "## Downstream Embedding" in markdown
    assert "- brief: `carry_over_non_blocking:SOLUSDT`" in markdown

    step_names = [row["name"] for row in payload["steps"]]
    assert step_names[:3] == ["native_custom", "native_majors", "native_beta"]
    assert step_names[-18:] == [
        "build_crypto_shortline_live_bars_snapshot",
        "build_crypto_shortline_execution_gate",
        "build_crypto_shortline_profile_location_watch",
        "build_crypto_shortline_mss_watch",
        "build_crypto_shortline_cvd_confirmation_watch",
        "build_crypto_shortline_retest_watch",
        "build_crypto_shortline_pattern_router",
        "build_crypto_route_brief",
        "build_crypto_route_operator_brief",
        "build_crypto_shortline_material_change_trigger",
        "build_crypto_shortline_liquidity_event_trigger",
        "build_crypto_shortline_live_orderflow_snapshot",
        "build_crypto_shortline_execution_quality_watch",
        "build_signal_to_order_tickets",
        "build_crypto_shortline_slippage_snapshot",
        "build_crypto_shortline_fill_capacity_watch",
        "build_crypto_shortline_sizing_watch",
        "build_crypto_shortline_signal_quality_watch",
    ]

    first_name, first_cmd = calls[0]
    assert first_name == "native_custom"
    assert first_cmd[1].endswith("run_backtest_binance_indicator_combo_native_crypto_guarded.py")
    assert "--symbol-group" in first_cmd and "custom" in first_cmd

    call_map = {name: cmd for name, cmd in calls}

    semantic_cmd = call_map["build_crypto_cvd_semantic_snapshot"]
    assert semantic_cmd[1].endswith("build_crypto_cvd_semantic_snapshot.py")
    assert "--artifact-dir" in semantic_cmd and str(output_root / "artifacts" / "micro_capture") in semantic_cmd
    assert "--micro-capture-file" in semantic_cmd and str(micro_capture_path) in semantic_cmd

    queue_cmd = call_map["build_crypto_cvd_queue_handoff"]
    assert queue_cmd[1].endswith("build_crypto_cvd_queue_handoff.py")
    assert "--now" in queue_cmd

    gate_cmd = call_map["build_crypto_shortline_execution_gate"]
    assert gate_cmd[1].endswith("build_crypto_shortline_execution_gate.py")
    assert "--output-root" in gate_cmd and str(output_root) in gate_cmd
    assert "--artifact-dir" in gate_cmd and str(output_root / "artifacts" / "micro_capture") in gate_cmd

    profile_watch_cmd = call_map["build_crypto_shortline_profile_location_watch"]
    assert profile_watch_cmd[1].endswith("build_crypto_shortline_profile_location_watch.py")
    assert "--now" in profile_watch_cmd

    mss_watch_cmd = call_map["build_crypto_shortline_mss_watch"]
    assert mss_watch_cmd[1].endswith("build_crypto_shortline_mss_watch.py")
    assert "--now" in mss_watch_cmd

    cvd_watch_cmd = call_map["build_crypto_shortline_cvd_confirmation_watch"]
    assert cvd_watch_cmd[1].endswith("build_crypto_shortline_cvd_confirmation_watch.py")
    assert "--now" in cvd_watch_cmd

    retest_watch_cmd = call_map["build_crypto_shortline_retest_watch"]
    assert retest_watch_cmd[1].endswith("build_crypto_shortline_retest_watch.py")
    assert "--now" in retest_watch_cmd

    live_bars_cmd = call_map["build_crypto_shortline_live_bars_snapshot"]
    assert live_bars_cmd[1].endswith("build_crypto_shortline_live_bars_snapshot.py")
    assert "--output-root" in live_bars_cmd and str(output_root) in live_bars_cmd

    brief_cmd = call_map["build_crypto_route_brief"]
    assert brief_cmd[1].endswith("build_crypto_route_brief.py")
    assert "--now" in brief_cmd

    operator_cmd = call_map["build_crypto_route_operator_brief"]
    assert operator_cmd[1].endswith("build_crypto_route_operator_brief.py")
    assert "--now" in operator_cmd

    material_change_cmd = call_map["build_crypto_shortline_material_change_trigger"]
    assert material_change_cmd[1].endswith("build_crypto_shortline_material_change_trigger.py")
    assert "--now" in material_change_cmd

    liquidity_event_cmd = call_map["build_crypto_shortline_liquidity_event_trigger"]
    assert liquidity_event_cmd[1].endswith("build_crypto_shortline_liquidity_event_trigger.py")
    assert "--now" in liquidity_event_cmd

    live_orderflow_cmd = call_map["build_crypto_shortline_live_orderflow_snapshot"]
    assert live_orderflow_cmd[1].endswith("build_crypto_shortline_live_orderflow_snapshot.py")
    assert "--artifact-dir" in live_orderflow_cmd and str(output_root / "artifacts" / "micro_capture") in live_orderflow_cmd
    assert "--micro-capture-file" in live_orderflow_cmd and str(micro_capture_path) in live_orderflow_cmd
    assert "--config" in live_orderflow_cmd

    execution_quality_cmd = call_map["build_crypto_shortline_execution_quality_watch"]
    assert execution_quality_cmd[1].endswith("build_crypto_shortline_execution_quality_watch.py")
    assert "--now" in execution_quality_cmd

    slippage_snapshot_cmd = call_map["build_crypto_shortline_slippage_snapshot"]
    assert slippage_snapshot_cmd[1].endswith("build_crypto_shortline_slippage_snapshot.py")
    assert "--now" in slippage_snapshot_cmd

    fill_capacity_cmd = call_map["build_crypto_shortline_fill_capacity_watch"]
    assert fill_capacity_cmd[1].endswith("build_crypto_shortline_fill_capacity_watch.py")
    assert "--now" in fill_capacity_cmd

    sizing_cmd = call_map["build_crypto_shortline_sizing_watch"]
    assert sizing_cmd[1].endswith("build_crypto_shortline_sizing_watch.py")
    assert "--now" in sizing_cmd

    signal_quality_cmd = call_map["build_crypto_shortline_signal_quality_watch"]
    assert signal_quality_cmd[1].endswith("build_crypto_shortline_signal_quality_watch.py")
    assert "--now" in signal_quality_cmd

    ticket_cmd = call_map["build_signal_to_order_tickets"]
    assert ticket_cmd[1].endswith("build_order_ticket.py")
    assert "--output-root" in ticket_cmd and str(output_root) in ticket_cmd
    assert "--review-dir" in ticket_cmd and str(review_dir) in ticket_cmd
    assert "--output-dir" in ticket_cmd and str(review_dir) in ticket_cmd
    assert "--config" in ticket_cmd
    assert "--now" in ticket_cmd
    scripted_now_steps = {
        "build_source_control_report",
        "build_native_group_report",
        "build_native_lane_stability_report",
        "build_native_beta_leg_report",
        "build_native_lane_playbook",
        "build_native_beta_leg_window_report",
        "build_bnb_flow_focus",
        "build_combo_playbook",
        "build_symbol_route_handoff",
    }
    for step_name, step_cmd in calls:
        if step_name in scripted_now_steps:
            assert "--now" in step_cmd


def test_main_normalizes_system_root_output_path_for_micro_capture_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    review_dir = system_root / "output" / "review"
    output_root = system_root / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    micro_capture_dir = output_root / "artifacts" / "micro_capture"
    micro_capture_dir.mkdir(parents=True, exist_ok=True)
    micro_capture_path = micro_capture_dir / "20260312T054508Z_micro_capture.json"
    micro_capture_path.write_text(json.dumps({"status": "degraded", "selected_micro": []}), encoding="utf-8")

    calls: list[tuple[str, list[str]]] = []

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append((step_name, list(cmd)))
        return {
            "ok": True,
            "status": "ok",
            "as_of": cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-12T05:45:00Z",
            "artifact": str(review_dir / f"{step_name}.json"),
        }

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(system_root),
            "--now",
            "2026-03-12T05:40:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_root"] == str(output_root)
    call_map = {name: cmd for name, cmd in calls}
    semantic_cmd = call_map["build_crypto_cvd_semantic_snapshot"]
    assert "--artifact-dir" in semantic_cmd and str(output_root / "artifacts" / "micro_capture") in semantic_cmd
    assert "--micro-capture-file" in semantic_cmd and str(micro_capture_path) in semantic_cmd
    gate_cmd = call_map["build_crypto_shortline_execution_gate"]
    assert "--output-root" in gate_cmd and str(output_root) in gate_cmd


def test_main_can_skip_native_refresh_and_reuse_latest_native_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    (review_dir / "20260312T053000Z_hot_universe_research.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052000Z_custom_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052001Z_majors_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052002Z_beta_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052003Z_sol_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052004Z_bnb_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052005Z_majors_long_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052006Z_beta_long_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052007Z_sol_long_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (review_dir / "20260312T052008Z_bnb_long_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )

    calls: list[tuple[str, list[str]]] = []

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append((step_name, list(cmd)))
        payload = {
            "ok": True,
            "status": "ok",
            "as_of": cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-12T05:45:00Z",
            "artifact": str(review_dir / f"{step_name}.json"),
        }
        if step_name == "build_crypto_route_brief":
            payload.update(
                {
                    "next_focus_symbol": "SOLUSDT",
                    "next_focus_action": "deprioritize_flow",
                    "focus_review_status": "review_no_edge_bias_only_micro_veto",
                    "focus_review_primary_blocker": "no_edge",
                    "focus_review_micro_blocker": "low_sample_or_gap_risk",
                    "focus_review_edge_score": 5,
                    "focus_review_structure_score": 25,
                    "focus_review_micro_score": 20,
                    "focus_review_composite_score": 17,
                    "focus_review_priority_status": "ready",
                    "focus_review_priority_score": 17,
                    "focus_review_priority_tier": "deprioritized_review",
                    "focus_review_priority_brief": "deprioritized_review:17/100",
                    "review_priority_queue_status": "ready",
                    "review_priority_queue_count": 1,
                    "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73",
                    "review_priority_head_symbol": "SOLUSDT",
                    "review_priority_head_tier": "review_queue_now",
                    "review_priority_head_score": 73,
                }
            )
        if step_name == "build_crypto_route_operator_brief":
            payload.update(
                {
                    "next_focus_symbol": "SOLUSDT",
                    "next_focus_action": "deprioritize_flow",
                    "focus_review_status": "review_no_edge_bias_only_micro_veto",
                    "focus_review_primary_blocker": "no_edge",
                    "focus_review_micro_blocker": "low_sample_or_gap_risk",
                    "focus_review_edge_score": 5,
                    "focus_review_structure_score": 25,
                    "focus_review_micro_score": 20,
                    "focus_review_composite_score": 17,
                    "focus_review_priority_status": "ready",
                    "focus_review_priority_score": 17,
                    "focus_review_priority_tier": "deprioritized_review",
                    "focus_review_priority_brief": "deprioritized_review:17/100",
                    "review_priority_queue_status": "ready",
                    "review_priority_queue_count": 1,
                    "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73",
                    "review_priority_head_symbol": "SOLUSDT",
                    "review_priority_head_tier": "review_queue_now",
                    "review_priority_head_score": 73,
                }
            )
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-12T05:40:00Z",
            "--skip-native-refresh",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["native_refresh_mode"] == "skip_native_refresh"
    assert payload["crypto_route_refresh_reuse_level"] == "informational"
    assert payload["crypto_route_refresh_reuse_gate_status"] == "reuse_non_blocking"
    assert payload["crypto_route_refresh_reuse_gate_brief"] == "reuse_non_blocking:skip_native_refresh:9/9"
    assert payload["crypto_route_refresh_reuse_gate_blocking"] is False
    assert payload["steps"][0]["name"] == "native_custom"
    assert payload["steps"][0]["status"] == "reused_previous_artifact"
    assert payload["steps"][0]["artifact"] == str(
        review_dir / "20260312T052000Z_custom_binance_indicator_combo_native_crypto.json"
    )
    assert payload["steps"][5]["name"] == "native_majors_long"
    assert payload["steps"][5]["status"] == "reused_previous_artifact"
    assert payload["steps"][5]["now"] == "2026-03-12T05:20:05Z"
    assert all(not name.startswith("native_") for name, _ in calls)
    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "- native_refresh_mode: `skip_native_refresh`" in markdown
    assert "## Refresh Reuse Gate" in markdown
    assert "- brief: `reuse_non_blocking:skip_native_refresh:9/9`" in markdown


def test_main_marks_partial_failure_when_native_guarded_steps_fail(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    micro_capture_dir = output_root / "artifacts" / "micro_capture"
    micro_capture_dir.mkdir(parents=True, exist_ok=True)
    micro_capture_path = micro_capture_dir / "20260312T054508Z_micro_capture.json"
    micro_capture_path.write_text(json.dumps({"status": "ok", "selected_micro": []}), encoding="utf-8")

    native_names = {
        "native_custom",
        "native_majors",
        "native_beta",
        "native_sol",
        "native_bnb",
        "native_majors_long",
        "native_beta_long",
        "native_sol_long",
        "native_bnb_long",
    }

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        artifact = str(review_dir / f"{step_name}.json")
        as_of = cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-12T05:45:00Z"
        if step_name in native_names:
            return {
                "ok": False,
                "status": "partial_failure",
                "as_of": as_of,
                "artifact": artifact,
            }
        payload = {
            "ok": True,
            "status": "ok",
            "as_of": as_of,
            "artifact": artifact,
        }
        if step_name == "build_crypto_cvd_semantic_snapshot":
            payload.update(
                {
                    "environment_status": "ok",
                    "environment_classification": "none",
                    "environment_blocker_detail": "",
                    "environment_remediation_hint": "",
                    "time_sync_status": "ok",
                    "time_sync_classification": "none",
                    "time_sync_blocker_detail": "",
                    "time_sync_remediation_hint": "",
                }
            )
        if step_name in {"build_crypto_route_brief", "build_crypto_route_operator_brief"}:
            payload.update(
                {
                    "next_focus_symbol": "SOLUSDT",
                    "next_focus_action": "deprioritize_flow",
                    "focus_review_status": "review_no_edge_bias_only_micro_veto",
                    "focus_review_primary_blocker": "no_edge",
                    "focus_review_micro_blocker": "low_sample_or_gap_risk",
                    "focus_review_edge_score": 5,
                    "focus_review_structure_score": 25,
                    "focus_review_micro_score": 20,
                    "focus_review_composite_score": 17,
                    "focus_review_priority_status": "ready",
                    "focus_review_priority_score": 17,
                    "focus_review_priority_tier": "deprioritized_review",
                    "focus_review_priority_brief": "deprioritized_review:17/100",
                    "review_priority_queue_status": "ready",
                    "review_priority_queue_count": 1,
                    "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73",
                    "review_priority_head_symbol": "SOLUSDT",
                    "review_priority_head_tier": "review_queue_now",
                    "review_priority_head_score": 73,
                }
            )
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-12T05:40:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["status"] == "partial_failure"
    assert payload["failure_step_count"] == 9
    assert "native_custom:partial_failure" in payload["failure_step_brief"]
    assert payload["crypto_route_refresh_reuse_level"] == "blocking"
    assert payload["crypto_route_refresh_reuse_gate_status"] == "native_refresh_partial_failure"
    assert payload["crypto_route_refresh_reuse_gate_brief"] == "native_refresh_partial_failure:full_guarded_refresh:9/9"
    assert payload["crypto_route_refresh_reuse_gate_blocking"] is True
    assert "native_* steps returned partial_failure" in payload["crypto_route_refresh_reuse_gate_blocker_detail"]
    assert payload["artifact_label"] == "crypto-route-refresh:partial_failure"
    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "- status: `partial_failure`" in markdown
    assert "- failure_step_count: `9`" in markdown
    assert "- brief: `native_refresh_partial_failure:full_guarded_refresh:9/9`" in markdown


def test_main_reuses_previous_shortline_gate_when_local_bars_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    (review_dir / "20260312T053000Z_hot_universe_research.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    for index, group_slug in enumerate(
        ("custom", "majors", "beta", "sol", "bnb", "majors_long", "beta_long", "sol_long", "bnb_long"),
        start=0,
    ):
        stamp = f"20260312T05200{index}Z"
        (review_dir / f"{stamp}_{group_slug}_binance_indicator_combo_native_crypto.json").write_text(
            json.dumps({"status": "ok"}), encoding="utf-8"
        )

    reused_gate_path = review_dir / "20260313T055016Z_crypto_shortline_execution_gate.json"
    reused_gate_path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "ok",
                "as_of": "2026-03-13T05:50:16Z",
                "artifact": str(reused_gate_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append(step_name)
        if step_name == "build_crypto_shortline_execution_gate":
            raise RuntimeError("build_crypto_shortline_execution_gate_failed: no_local_bars_artifact")
        payload = {
            "ok": True,
            "status": "ok",
            "as_of": cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-12T05:45:00Z",
            "artifact": str(review_dir / f"{step_name}.json"),
        }
        if step_name in {"build_crypto_route_brief", "build_crypto_route_operator_brief"}:
            payload.update(
                {
                    "next_focus_symbol": "SOLUSDT",
                    "next_focus_action": "deprioritize_flow",
                    "focus_review_status": "review_no_edge_bias_only_micro_veto",
                    "focus_review_primary_blocker": "no_edge",
                    "focus_review_micro_blocker": "low_sample_or_gap_risk",
                    "focus_review_edge_score": 5,
                    "focus_review_structure_score": 25,
                    "focus_review_micro_score": 20,
                    "focus_review_composite_score": 17,
                    "focus_review_priority_status": "ready",
                    "focus_review_priority_score": 17,
                    "focus_review_priority_tier": "deprioritized_review",
                    "focus_review_priority_brief": "deprioritized_review:17/100",
                    "review_priority_queue_status": "ready",
                    "review_priority_queue_count": 1,
                    "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73",
                    "review_priority_head_symbol": "SOLUSDT",
                    "review_priority_head_tier": "review_queue_now",
                    "review_priority_head_score": 73,
                }
            )
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-14T09:27:00Z",
            "--skip-native-refresh",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    step_map = {row["name"]: row for row in payload["steps"]}
    assert step_map["build_crypto_shortline_execution_gate"]["status"] == "carry_over_previous_artifact"
    assert step_map["build_crypto_shortline_execution_gate"]["artifact"] == str(reused_gate_path)
    assert payload["shortline_execution_gate_artifact"] == str(reused_gate_path)
    assert "build_crypto_shortline_material_change_trigger" in calls
    assert "build_crypto_route_brief" in calls
    assert "build_crypto_route_operator_brief" in calls
