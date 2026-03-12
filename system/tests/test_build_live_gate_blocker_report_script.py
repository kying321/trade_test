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
        },
        "ready_check": {
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
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    research_path.write_text(json.dumps(research), encoding="utf-8")

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
    assert payload["commodity_execution_path"]["focus_primary_batches"] == ["metals_all", "precious_metals"]
    assert payload["repair_sequence"][0]["reason_codes"] == ["risk_violations", "max_drawdown", "slot_anomaly"]
    assert Path(payload["artifact"]).exists()
    assert Path(payload["markdown"]).exists()
    assert Path(payload["checksum"]).exists()


def test_build_live_gate_blocker_report_selects_latest_nonempty_research(tmp_path: Path) -> None:
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
    assert payload["research_source"].endswith("20260310T000001Z_hot_universe_research.json")
    assert payload["commodity_execution_path"]["focus_primary_batches"] == ["metals_all"]
