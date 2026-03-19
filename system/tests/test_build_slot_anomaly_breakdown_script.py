from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_slot_anomaly_breakdown.py")


def test_build_slot_anomaly_breakdown_detects_active_root_cause(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff = {
        "ready_check": {
            "ops_reconcile": {
                "artifact_path": "output/review/2026-03-16_ops_report.json",
                "artifact_age_hours": 0.5,
                "artifact_date": "2026-03-16",
            },
            "ops_live_gate": {
                "blocking_reason_codes": [
                    "rollback_hard",
                    "mode_drift",
                    "slot_anomaly",
                    "backtest_snapshot",
                    "ops_status_red",
                ],
                "rollback_reason_codes": ["mode_drift", "slot_anomaly"],
                "gate_failed_checks": ["slot_anomaly_ok", "mode_drift_ok", "backtest_snapshot_ok"],
                "rollback_level": "hard",
                "rollback_action": "rollback_now_and_lock_parameter_updates",
            },
        }
    }
    handoff_path = review_dir / "20260316T000000Z_remote_live_handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--review-dir", str(review_dir), "--handoff-json", str(handoff_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "slot_anomaly_active_root_cause"
    assert payload["blocker_active"] is True
    assert payload["slot_anomaly_reason_present"] is True
    assert payload["slot_anomaly_check_failed"] is True
    assert payload["rollback_wrapper_present"] is True
    assert payload["ops_status_red_present"] is True
    assert payload["repair_focus"].endswith("2026-03-16 --window-days 7")
    assert Path(payload["artifact"]).exists()


def test_build_slot_anomaly_breakdown_reads_persisted_ops_status_detail(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    handoff = {
        "ready_check": {
            "ops_reconcile": {
                "artifact_path": "output/review/2026-03-17_ops_report.json",
                "artifact_age_hours": 0.2,
                "artifact_date": "2026-03-17",
            },
            "ops_live_gate": {
                "blocking_reason_codes": ["rollback_hard", "slot_anomaly", "ops_status_red"],
                "rollback_reason_codes": ["slot_anomaly"],
                "gate_failed_checks": ["slot_anomaly_ok"],
                "rollback_level": "hard",
                "rollback_action": "rollback_now_and_lock_parameter_updates",
            },
        }
    }
    handoff_path = review_dir / "20260317T000000Z_remote_live_handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    ops_status = {
        "action": "capture_remote_live_handoff_input",
        "payload": {
            "live_slot_anomaly": {
                "active": True,
                "window_days": 7,
                "samples": 7,
                "min_samples": 3,
                "checks": {
                    "missing_ratio_ok": True,
                    "premarket_anomaly_ok": False,
                    "intraday_anomaly_ok": False,
                    "eod_quality_anomaly_ok": False,
                },
                "alerts": [
                    "slot_premarket_anomaly_high",
                    "slot_intraday_anomaly_high",
                    "slot_eod_quality_anomaly_high",
                ],
                "metrics": {
                    "missing_ratio": 0.0,
                    "premarket_anomaly_ratio": 1.0,
                    "intraday_anomaly_ratio": 1.0,
                    "eod_quality_anomaly_ratio": 1.0,
                },
                "slots": {
                    "premarket": {"anomaly_ratio": 1.0},
                    "intraday": {"anomaly_ratio": 1.0},
                    "eod": {"anomaly_ratio": 1.0},
                },
                "series": [
                    {"date": "2026-03-17", "anomalies": 4, "alerts": ["premarket:quality_failed"]},
                ],
            }
        },
    }
    ops_status_path = review_dir / "20260317T000001Z_remote_live_ops_reconcile_status.json"
    ops_status_path.write_text(json.dumps(ops_status), encoding="utf-8")
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
    assert payload["payload_gap"] == ""
    assert payload["slot_detail"]["status"] == "active"
    assert payload["slot_detail"]["failed_checks"] == [
        "premarket_anomaly_ok",
        "intraday_anomaly_ok",
        "eod_quality_anomaly_ok",
    ]
    assert payload["slot_detail"]["alerts"][:2] == [
        "slot_premarket_anomaly_high",
        "slot_intraday_anomaly_high",
    ]
    assert payload["slot_detail"]["metrics"]["premarket_anomaly_ratio"] == 1.0
    assert payload["brief"] == "slot_anomaly_active_root_cause:2026-03-17:premarket_anomaly_ok,intraday_anomaly_ok,eod_quality_anomaly_ok"
    assert payload["repair_focus"].startswith("优先修复 premarket/intraday/eod quality 与 source_confidence 异常")
