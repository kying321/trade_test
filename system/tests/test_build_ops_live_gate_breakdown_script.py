from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_ops_live_gate_breakdown.py")


def test_build_ops_live_gate_breakdown(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff = {
        "ready_check": {
            "ops_live_gate": {
                "ok": False,
                "blocking_reason_codes": [
                    "rollback_hard",
                    "risk_violations",
                    "max_drawdown",
                    "slot_anomaly",
                    "mode_drift",
                    "backtest_snapshot",
                    "ops_status_red",
                ],
                "rollback_reason_codes": ["risk_violations", "max_drawdown", "slot_anomaly", "mode_drift"],
                "gate_failed_checks": [
                    "backtest_snapshot_ok",
                    "health_ok",
                    "max_drawdown_ok",
                    "mode_drift_ok",
                    "positive_window_ratio_ok",
                    "review_pass_gate",
                    "risk_violations_ok",
                    "slot_anomaly_ok",
                    "stable_replay_ok",
                    "stress_autorun_reason_drift_ok",
                    "unresolved_conflict_ok",
                ],
            }
        }
    }
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--review-dir", str(review_dir), "--handoff-json", str(handoff_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert [row["code"] for row in payload["root_causes"]] == [
        "risk_violations",
        "max_drawdown",
        "slot_anomaly",
        "mode_drift",
        "backtest_snapshot",
    ]
    assert payload["primary_root_cause_code"] == "risk_violations"
    assert [row["code"] for row in payload["derived_wrappers"]] == ["rollback_hard", "ops_status_red"]
    assert any(row["code"] == "review_gate" for row in payload["secondary_checks"])
    assert payload["repair_order"][0]["group"] == "root_causes"
    assert Path(payload["artifact"]).exists()


def test_build_ops_live_gate_breakdown_includes_runtime_root_causes(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff = {
        "ready_check": {
            "ops_live_gate": {
                "ok": False,
                "blocking_reason_codes": [
                    "runtime_health",
                    "runtime_replay",
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
                    "state_stability_ok",
                    "stress_autorun_reason_drift_ok",
                ],
            }
        }
    }
    handoff_path = review_dir / "20260310T000000Z_remote_live_handoff.json"
    ops_status_path = review_dir / "20260310T000001Z_remote_live_ops_reconcile_status.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    ops_status_path.write_text(
        json.dumps(
            {
                "action": "capture_remote_live_handoff_input",
                "capture_kind": "remote_live_ops_reconcile_status",
                "captured_at_utc": "2026-03-10T00:00:01Z",
                "returncode": 0,
                "payload": {
                    "action": "live-ops-reconcile-status",
                    "ok": True,
                    "status": "passed",
                    "live_runtime_health": {
                        "status": "degraded",
                        "checks": {
                            "daily_briefing": False,
                            "daily_signals": True,
                            "daily_positions": True,
                            "sqlite": True,
                        },
                        "missing": ["daily_briefing"],
                    },
                    "live_runtime_replay": {
                        "passed": False,
                        "replay_days": 3,
                        "mode": "cached_runtime_health_only",
                        "checks": [
                            {
                                "date": "2026-03-10",
                                "ok": False,
                                "health": {
                                    "status": "degraded",
                                    "missing": ["daily_briefing"],
                                },
                            },
                            {
                                "date": "2026-03-09",
                                "ok": True,
                                "health": {
                                    "status": "healthy",
                                    "missing": [],
                                },
                            },
                        ],
                    },
                },
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
            "--ops-status-json",
            str(ops_status_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert [row["code"] for row in payload["root_causes"]] == [
        "runtime_health",
        "runtime_replay",
        "slot_anomaly",
        "mode_drift",
        "backtest_snapshot",
    ]
    assert payload["primary_root_cause_code"] == "runtime_health"
    assert [row["code"] for row in payload["derived_wrappers"]] == ["rollback_hard", "ops_status_red"]
    assert [row["code"] for row in payload["secondary_checks"]] == [
        "review_gate",
        "stress_autorun_reason_drift",
    ]
    assert payload["runtime_health_detail"]["status"] == "degraded"
    assert payload["runtime_health_detail"]["missing"] == ["daily_briefing"]
    assert payload["runtime_replay_detail"]["status"] == "failed"
    assert payload["runtime_replay_detail"]["failed_dates"] == ["2026-03-10"]
    assert payload["runtime_replay_detail"]["first_failed_missing"] == ["daily_briefing"]
