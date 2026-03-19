from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_live_gate_blocker_report.py")


def test_build_live_gate_blocker_report(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)

    handoff = {
        "operator_handoff": {
            "ready": False,
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
            "next_focus_area": "gate",
            "next_focus_reason": "ops_live_gate_blocked",
            "secondary_focus_area": "risk_guard",
            "secondary_focus_reason": "risk_guard_blocked",
            "next_focus_command": "cmd-gate",
            "secondary_focus_command": "cmd-risk",
            "ready_check_scope_market": "portfolio_margin_um",
            "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
            "account_scope_alignment": {
                "status": "split_scope_spot_vs_portfolio_margin_um",
                "brief": "split_scope_spot_vs_portfolio_margin_um",
                "blocking": False,
                "blocker_detail": "spot ready-check and portfolio margin history refer to different scopes.",
            },
            "execution_contract": {
                "status": "non_executable_contract",
                "brief": "non_executable_contract:portfolio_margin_um:portfolio_margin_um:spot_remote_lane_missing,portfolio_margin_um_read_only_mode,shadow_executor_only_mode",
                "mode": "shadow_only",
                "live_orders_allowed": False,
                "reason_codes": [
                    "spot_remote_lane_missing",
                    "portfolio_margin_um_read_only_mode",
                    "shadow_executor_only_mode",
                ],
                "blocker_detail": "remote execution contract remains non-executable; target=portfolio_margin_um; lane=portfolio_margin_um; scope=split_scope_spot_vs_portfolio_margin_um",
                "done_when": "promote an explicit non-shadow remote send/ack/fill contract on the target market",
            },
            "remote_live_history": {
                "artifact": "/tmp/remote_live_history_audit.json",
                "status": "ok",
                "market": "portfolio_margin_um",
                "generated_at": "2026-03-13T01:20:00Z",
                "window_brief": "24h:14.82pnl/20tr/1open | 30d:18.79pnl/38tr/1open",
                "quote_available": -0.87,
                "open_positions": 1,
                "risk_guard_status": "blocked",
                "risk_guard_reasons": [
                    "ticket_missing:no_actionable_ticket",
                    "panic_cooldown_active",
                    "open_exposure_above_cap",
                ],
                "blocked_candidate_symbol": "BNBUSDT",
                "pnl_24h": 14.82144002,
                "pnl_30d": 18.79356,
                "trade_count_24h": 20,
                "trade_count_30d": 38,
                "symbol_pnl_brief": "BTCUSDT:15.17, ETHUSDT:10.18",
                "day_pnl_brief": "2026-03-12:14.82",
            },
            "remote_live_diagnosis": {
                "status": "profitability_confirmed_but_auto_live_blocked",
                "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                "market": "portfolio_margin_um",
                "profitability_confirmed": True,
                "profitability_window": "30d",
                "profitability_pnl": 18.79356,
                "profitability_trade_count": 38,
                "blocking_layers": ["ops_live_gate", "risk_guard"],
                "blocker_detail": "sentinel:reuse_existing_remote_live_diagnosis",
                "done_when": "sentinel:done_when",
            },
        },
        "ready_check": {
            "ops_reconcile": {
                "artifact_path": "output/review/2026-03-10_ops_report.json",
                "artifact_age_hours": 0.25,
                "artifact_mtime_utc": "2026-03-10T00:15:00Z",
                "max_age_hours": 48,
            },
            "ops_live_gate": {
                "ok": False,
                "blocking_reason_codes": ["rollback_hard", "risk_violations", "max_drawdown", "slot_anomaly"],
                "rollback_level": "hard",
                "rollback_action": "rollback_now",
                "rollback_reason_codes": ["risk_violations", "max_drawdown", "slot_anomaly"],
                "gate_failed_checks": ["risk_violations_ok", "max_drawdown_ok", "slot_anomaly_ok"],
            },
            "risk_guard": {
                "reasons": ["ticket_missing:no_actionable_ticket"],
                "generated_at_utc": "2026-03-10T00:19:10Z",
                "fuse_age_seconds": 42.0,
                "fuse_artifact": "/tmp/20260310T001910Z_live_risk_guard.json",
            },
            "guarded_exec": {
                "takeover": {
                    "payload": {
                        "steps": {
                            "signal_selection": {
                                "blocked_candidate": {
                                    "symbol": "BNBUSDT",
                                    "ticket_reasons": ["confidence_below_threshold", "size_below_min_notional"],
                                }
                            }
                        }
                    }
                }
            },
        },
    }
    research = {
        "research_action_ladder": {
            "focus_primary_batches": ["metals_all", "precious_metals"],
            "focus_with_regime_filter_batches": ["energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "avoid_batches": ["energy_gas"],
            "focus_now_batches": ["metals_all", "precious_metals", "energy_liquids"],
        },
        "regime_playbook": {
            "batch_rules": [
                {"batch": "metals_all", "leader_symbols": ["XAGUSD", "COPPER"]},
                {"batch": "precious_metals", "leader_symbols": ["XAGUSD", "XAUUSD"]},
                {"batch": "energy_liquids", "leader_symbols": ["BRENTUSD", "WTIUSD"]},
            ]
        },
    }
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    research_path = review_dir / "20260310T000000Z_hot_universe_research.json"
    cross_market_path = review_dir / "20260310T000000Z_cross_market_operator_state.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    research_path.write_text(json.dumps(research), encoding="utf-8")
    cross_market_path.write_text(
        json.dumps(
            {
                "operator_head": {
                    "area": "commodity_execution_close_evidence",
                    "symbol": "XAUUSD",
                    "action": "wait_for_paper_execution_close_evidence",
                    "state": "waiting",
                    "priority_score": 99,
                    "priority_tier": "waiting_now",
                    "blocker_detail": "paper execution evidence is present, but position is still OPEN",
                    "done_when": "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available",
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-dir",
            str(review_dir),
            "--handoff-json",
            str(handoff_path),
            "--research-json",
            str(research_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["live_decision"]["current_decision"] == "do_not_start_formal_live"
    assert payload["blockers"][0]["name"] == "ops_live_gate"
    assert payload["blockers"][1]["name"] == "risk_guard"
    assert payload["blockers"][2]["status"] == "active"
    assert payload["blockers"][1]["reason_codes"] == [
        "ticket_missing:no_actionable_ticket",
    ]
    assert payload["blockers"][3]["name"] == "account_scope_alignment"
    assert payload["blockers"][3]["status"] == "blocked"
    assert payload["blockers"][3]["reason_codes"] == [
        "spot_remote_lane_missing",
        "portfolio_margin_um_read_only_mode",
    ]
    assert payload["blockers"][4]["name"] == "remote_execution_contract"
    assert payload["blockers"][4]["status"] == "blocked"
    assert payload["blockers"][4]["reason_codes"] == [
        "spot_remote_lane_missing",
        "portfolio_margin_um_read_only_mode",
        "shadow_executor_only_mode",
    ]
    assert payload["remote_live_context"]["window_brief"] == "24h:14.82pnl/20tr/1open | 30d:18.79pnl/38tr/1open"
    assert payload["remote_live_context"]["ready_check_scope_brief"] == "portfolio_margin_um:portfolio_margin_um"
    assert payload["remote_live_context"]["account_scope_alignment_brief"] == "split_scope_spot_vs_portfolio_margin_um"
    assert payload["remote_live_context"]["execution_contract_brief"] == (
        "non_executable_contract:portfolio_margin_um:portfolio_margin_um:spot_remote_lane_missing,portfolio_margin_um_read_only_mode,shadow_executor_only_mode"
    )
    assert payload["remote_live_context"]["symbol_pnl_brief"] == "BTCUSDT:15.17, ETHUSDT:10.18"
    assert payload["source_freshness"]["ops_reconcile_status"] == "fresh"
    assert payload["source_freshness"]["ops_reconcile_artifact_age_hours"] == 0.25
    assert payload["source_freshness"]["risk_guard_status"] == "fresh"
    assert payload["source_freshness"]["risk_guard_age_seconds"] == 42.0
    assert payload["remote_live_diagnosis"]["status"] == "profitability_confirmed_but_auto_live_blocked"
    assert payload["remote_live_diagnosis"]["brief"] == (
        "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
    )
    assert payload["remote_live_diagnosis"]["profitability_confirmed"] is True
    assert payload["remote_live_diagnosis"]["profitability_window"] == "30d"
    assert payload["remote_live_diagnosis"]["blocking_layers"] == ["ops_live_gate", "risk_guard"]
    assert payload["remote_live_diagnosis"]["blocker_detail"] == "sentinel:reuse_existing_remote_live_diagnosis"
    assert payload["remote_live_diagnosis"]["done_when"] == "sentinel:done_when"
    assert payload["remote_live_operator_alignment"]["status"] == "local_operator_head_outside_remote_live_scope"
    assert payload["remote_live_operator_alignment"]["brief"] == (
        "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um"
    )
    assert payload["remote_live_operator_alignment"]["head_symbol"] == "XAUUSD"
    assert payload["remote_live_operator_alignment"]["head_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["remote_live_operator_alignment"]["remote_status"] == "profitability_confirmed_but_auto_live_blocked"
    assert payload["remote_live_operator_alignment"]["remote_market"] == "portfolio_margin_um"
    assert payload["remote_live_operator_alignment"]["eligible_for_remote_live"] is False
    assert "outside remote-live executable scope" in payload["remote_live_operator_alignment"]["blocker_detail"]
    assert payload["ops_live_gate_clearing"]["status"] == "clearing_required"
    assert payload["ops_live_gate_clearing"]["conditions_brief"] == (
        "rollback_hard, risk_violations, max_drawdown, slot_anomaly"
    )
    assert payload["risk_guard_clearing"]["status"] == "clearing_required"
    assert payload["risk_guard_clearing"]["conditions_brief"] == (
        "ticket_missing:no_actionable_ticket"
    )
    assert payload["account_scope_alignment_clearing"]["status"] == "clearing_required"
    assert payload["account_scope_alignment_clearing"]["conditions_brief"] == (
        "spot_remote_lane_missing, portfolio_margin_um_read_only_mode"
    )
    assert payload["remote_execution_contract_clearing"]["status"] == "clearing_required"
    assert payload["remote_execution_contract_clearing"]["conditions_brief"] == (
        "spot_remote_lane_missing, portfolio_margin_um_read_only_mode, shadow_executor_only_mode"
    )
    assert payload["remote_live_takeover_clearing"]["status"] == "clearing_required"
    assert payload["remote_live_takeover_clearing"]["brief"] == (
        "clearing_required:ops_live_gate+risk_guard"
    )
    assert payload["remote_live_takeover_repair_queue"]["status"] == "ready"
    assert payload["remote_live_takeover_repair_queue"]["brief"] == (
        "ready:ops_live_gate:rollback_hard:99"
    )
    assert payload["remote_live_takeover_repair_queue"]["count"] == 5
    assert payload["remote_live_takeover_repair_queue"]["head_area"] == "ops_live_gate"
    assert payload["remote_live_takeover_repair_queue"]["head_code"] == "rollback_hard"
    assert payload["remote_live_takeover_repair_queue"]["head_action"] == "clear_ops_live_gate_condition"
    assert payload["remote_live_takeover_repair_queue"]["head_priority_score"] == 99
    assert payload["remote_live_takeover_repair_queue"]["head_priority_tier"] == "repair_queue_now"
    assert payload["remote_live_takeover_repair_queue"]["head_command"] == "cmd-gate"
    assert payload["commodity_execution_path"]["focus_primary_batches"] == ["metals_all", "precious_metals"]
    assert payload["repair_sequence"][0]["reason_codes"] == ["risk_violations", "max_drawdown", "slot_anomaly"]
    assert Path(payload["artifact"]).exists()
    assert Path(payload["markdown"]).exists()
    assert Path(payload["checksum"]).exists()


def test_build_live_gate_blocker_report_prefers_contract_state_when_present(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    research_path = review_dir / "20260310T000000Z_hot_universe_research.json"
    contract_path = review_dir / "20260310T000000Z_remote_execution_contract_state.json"
    handoff_path.write_text(
        json.dumps(
            {
                "operator_handoff": {
                    "ready": False,
                    "ready_check_scope_market": "portfolio_margin_um",
                    "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                    "account_scope_alignment": {
                        "status": "split_scope_spot_vs_portfolio_margin_um",
                        "brief": "split_scope_spot_vs_portfolio_margin_um",
                        "blocker_detail": "scope split",
                    },
                    "execution_contract": {
                        "status": "live_executable_contract",
                        "brief": "stale_handoff_contract",
                        "mode": "live_executable",
                        "live_orders_allowed": True,
                        "reason_codes": [],
                    },
                    "remote_live_history": {
                        "artifact": "/tmp/history.json",
                        "status": "ok",
                        "market": "portfolio_margin_um",
                        "window_brief": "30d:18.79pnl/38tr/1open",
                    },
                },
                "ready_check": {
                    "ops_live_gate": {"ok": True},
                    "risk_guard": {"reasons": []},
                },
            }
        ),
        encoding="utf-8",
    )
    contract_path.write_text(
        json.dumps(
            {
                "contract_status": "non_executable_contract",
                "contract_brief": "non_executable_contract:portfolio_margin_um:portfolio_margin_um:shadow_executor_only_mode",
                "contract_mode": "shadow_only",
                "guarded_probe_allowed": True,
                "live_orders_allowed": False,
                "reason_codes": ["shadow_executor_only_mode"],
                "blocker_detail": "contract_state_is_source_owned",
                "done_when": "promote contract state",
            }
        ),
        encoding="utf-8",
    )
    research_path.write_text(
        json.dumps(
            {
                "research_action_ladder": {
                    "focus_primary_batches": [],
                    "focus_with_regime_filter_batches": [],
                    "shadow_only_batches": [],
                    "avoid_batches": [],
                    "focus_now_batches": [],
                },
                "regime_playbook": {"batch_rules": []},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-dir",
            str(review_dir),
            "--handoff-json",
            str(handoff_path),
            "--research-json",
            str(research_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["remote_execution_contract"]["contract_status"] == "non_executable_contract"
    assert payload["remote_execution_contract"]["contract_brief"] == (
        "non_executable_contract:portfolio_margin_um:portfolio_margin_um:shadow_executor_only_mode"
    )
    assert payload["remote_execution_contract_source"].endswith(
        "_remote_execution_contract_state.json"
    )
    assert payload["remote_live_context"]["execution_contract_source"].endswith(
        "_remote_execution_contract_state.json"
    )
    assert payload["remote_execution_contract"]["guarded_probe_allowed"] is True
    assert payload["blockers"][4]["guarded_probe_allowed"] is True
    assert payload["blockers"][4]["reason_codes"] == ["shadow_executor_only_mode"]


def test_build_live_gate_blocker_report_clears_scope_blockers_for_spot_target_lane(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    research_path = review_dir / "20260310T000000Z_hot_universe_research.json"
    handoff_path.write_text(
        json.dumps(
            {
                "operator_handoff": {
                    "ready": False,
                    "ready_check_scope_market": "spot",
                    "ready_check_scope_brief": "spot:spot",
                    "account_scope_alignment": {
                        "status": "split_scope_spot_vs_portfolio_margin_um",
                        "brief": "split_scope_spot_vs_portfolio_margin_um",
                        "blocker_detail": "scope split",
                    },
                    "execution_contract": {
                        "status": "non_executable_contract",
                        "brief": "non_executable_contract:spot:spot:shadow_executor_only_mode",
                        "mode": "shadow_only",
                        "live_orders_allowed": False,
                        "executor_mode": "shadow_guarded",
                        "executor_mode_source": "bridge_context",
                        "reason_codes": ["shadow_executor_only_mode"],
                        "blocker_detail": "shadow only",
                        "done_when": "promote non-shadow executor",
                    },
                    "remote_live_history": {
                        "artifact": "/tmp/history.json",
                        "status": "ok",
                        "market": "portfolio_margin_um",
                        "window_brief": "30d:18.79pnl/38tr/1open",
                    },
                },
                "ready_check": {
                    "ops_live_gate": {"ok": True},
                    "risk_guard": {"reasons": []},
                },
            }
        ),
        encoding="utf-8",
    )
    research_path.write_text(
        json.dumps(
            {
                "research_action_ladder": {
                    "focus_primary_batches": [],
                    "focus_with_regime_filter_batches": [],
                    "shadow_only_batches": [],
                    "avoid_batches": [],
                    "focus_now_batches": [],
                },
                "regime_playbook": {"batch_rules": []},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-dir",
            str(review_dir),
            "--handoff-json",
            str(handoff_path),
            "--research-json",
            str(research_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["remote_live_context"]["ready_check_scope_brief"] == "spot:spot"
    assert payload["blockers"][3]["name"] == "account_scope_alignment"
    assert payload["blockers"][3]["reason_codes"] == []
    assert payload["account_scope_alignment_clearing"]["status"] == "clear"
    assert payload["blockers"][4]["reason_codes"] == ["shadow_executor_only_mode"]
    assert payload["blockers"][4]["executor_mode"] == "shadow_guarded"
    assert payload["remote_live_context"]["execution_contract_executor_mode"] == "shadow_guarded"


def test_build_live_gate_blocker_report_prefers_aligned_ops_breakdown_for_repair_head(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff = {
        "operator_handoff": {
            "ready": False,
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
            "next_focus_area": "gate",
            "next_focus_reason": "ops_live_gate_blocked",
            "secondary_focus_area": "risk_guard",
            "secondary_focus_reason": "risk_guard_blocked",
            "next_focus_command": "cmd-gate",
            "secondary_focus_command": "cmd-risk",
        },
        "ready_check": {
            "ops_live_gate": {
                "ok": False,
                "blocking_reason_codes": [
                    "rollback_hard",
                    "mode_drift",
                    "slot_anomaly",
                    "backtest_snapshot",
                    "ops_status_red",
                ],
                "rollback_reason_codes": ["mode_drift", "slot_anomaly"],
                "gate_failed_checks": [
                    "backtest_snapshot_ok",
                    "health_ok",
                    "mode_drift_ok",
                    "review_pass_gate",
                    "slot_anomaly_ok",
                    "stable_replay_ok",
                    "stress_autorun_reason_drift_ok",
                    "unresolved_conflict_ok",
                ],
            },
            "risk_guard": {
                "reasons": [
                    "ticket_missing:no_actionable_ticket",
                    "open_exposure_above_cap",
                    "panic_cooldown_active",
                ]
            },
        },
    }
    research = {
        "research_action_ladder": {
            "focus_primary_batches": [],
            "focus_with_regime_filter_batches": [],
            "shadow_only_batches": [],
            "avoid_batches": [],
            "focus_now_batches": [],
        }
    }
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    research_path = review_dir / "20260310T000000Z_hot_universe_research.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    research_path.write_text(json.dumps(research), encoding="utf-8")
    (review_dir / "20260310T000001Z_ops_live_gate_breakdown.json").write_text(
        json.dumps(
            {
                "handoff_source": str(handoff_path),
                "root_causes": [
                    {"code": "slot_anomaly", "priority": 3, "fix_action": "repair slot anomaly first"},
                    {"code": "mode_drift", "priority": 4, "fix_action": "repair mode drift next"},
                    {"code": "backtest_snapshot", "priority": 5, "fix_action": "refresh snapshot after gate state stabilizes"},
                ],
                "primary_root_cause_code": "slot_anomaly",
                "primary_root_cause_fix_action": "repair slot anomaly first",
                "derived_wrappers": [
                    {"code": "rollback_hard"},
                    {"code": "ops_status_red"},
                ],
                "secondary_checks": [
                    {"code": "review_gate", "action": "repair review gate"},
                    {"code": "stable_replay", "action": "repair stable replay"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-dir",
            str(review_dir),
            "--handoff-json",
            str(handoff_path),
            "--research-json",
            str(research_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ops_live_gate_breakdown"]["status"] == "root_causes_identified"
    assert payload["ops_live_gate_breakdown"]["primary_root_cause_code"] == "slot_anomaly"
    assert payload["remote_live_takeover_repair_queue"]["brief"] == "ready:ops_live_gate:slot_anomaly:100"
    assert payload["remote_live_takeover_repair_queue"]["head_code"] == "slot_anomaly"
    assert payload["remote_live_takeover_repair_queue"]["head_action"] == "clear_ops_live_gate_root_cause"
    assert payload["remote_live_takeover_repair_queue"]["head_clear_when"] == "repair slot anomaly first"
    assert payload["remote_live_takeover_repair_queue"]["count"] == 8


def test_build_live_gate_blocker_report_loads_aligned_slot_anomaly_breakdown(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff = {
        "operator_handoff": {
            "ready": False,
            "next_focus_command": "cmd-gate",
            "secondary_focus_command": "cmd-risk",
        },
        "ready_check": {
            "ops_reconcile": {
                "artifact_path": "output/review/2026-03-16_ops_report.json",
                "artifact_age_hours": 0.125,
                "artifact_date": "2026-03-16",
            },
            "ops_live_gate": {
                "ok": False,
                "blocking_reason_codes": ["rollback_hard", "slot_anomaly", "ops_status_red"],
                "rollback_reason_codes": ["slot_anomaly"],
                "gate_failed_checks": ["slot_anomaly_ok"],
            },
            "risk_guard": {"reasons": []},
        },
    }
    research = {"research_action_ladder": {"focus_primary_batches": [], "focus_with_regime_filter_batches": [], "shadow_only_batches": [], "avoid_batches": [], "focus_now_batches": []}}
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    research_path = review_dir / "20260310T000000Z_hot_universe_research.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    research_path.write_text(json.dumps(research), encoding="utf-8")
    (review_dir / "20260310T000001Z_slot_anomaly_breakdown.json").write_text(
        json.dumps(
            {
                "handoff_source": str(handoff_path),
                "status": "slot_anomaly_active_root_cause",
                "brief": "slot_anomaly_active_root_cause:2026-03-16",
                "repair_focus": "优先修复 slot_anomaly 缺陷并重跑 lie ops-report --date 2026-03-16 --window-days 7",
                "payload_gap": "slot payload missing in handoff",
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-dir",
            str(review_dir),
            "--handoff-json",
            str(handoff_path),
            "--research-json",
            str(research_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["slot_anomaly_breakdown"]["status"] == "slot_anomaly_active_root_cause"
    assert payload["slot_anomaly_breakdown"]["brief"] == "slot_anomaly_active_root_cause:2026-03-16"
    assert payload["slot_anomaly_breakdown"]["repair_focus"].endswith("2026-03-16 --window-days 7")


def test_build_live_gate_blocker_report_selects_latest_valid_research(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff = {
        "operator_handoff": {
            "ready": False,
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
            "next_focus_area": "gate",
            "next_focus_reason": "ops_live_gate_blocked",
            "secondary_focus_area": "risk_guard",
            "secondary_focus_reason": "risk_guard_blocked",
            "next_focus_command": "cmd-gate",
            "secondary_focus_command": "cmd-risk",
        },
        "ready_check": {"ops_live_gate": {"ok": False}, "risk_guard": {"reasons": ["ticket_missing:no_actionable_ticket"]}},
    }
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    older_research = {
        "research_action_ladder": {
            "focus_primary_batches": ["metals_all"],
            "focus_with_regime_filter_batches": ["energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "avoid_batches": [],
            "focus_now_batches": ["metals_all", "energy_liquids"],
        },
        "regime_playbook": {
            "batch_rules": [
                {"batch": "metals_all", "leader_symbols": ["XAGUSD", "COPPER"]},
                {"batch": "energy_liquids", "leader_symbols": ["BRENTUSD", "WTIUSD"]},
            ]
        },
    }
    newer_research = {
        "research_action_ladder": {
            "focus_primary_batches": [],
            "focus_with_regime_filter_batches": [],
            "shadow_only_batches": [],
            "avoid_batches": ["mixed_macro"],
            "focus_now_batches": [],
        }
    }
    (review_dir / "20260310T000001Z_hot_universe_research.json").write_text(json.dumps(older_research), encoding="utf-8")
    (review_dir / "20260310T000002Z_hot_universe_research.json").write_text(json.dumps(newer_research), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-dir",
            str(review_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["research_source"].endswith("20260310T000002Z_hot_universe_research.json")
    assert payload["commodity_execution_path"]["focus_primary_batches"] == []
