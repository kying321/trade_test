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
                    "ops_status_red",
                ],
                "rollback_reason_codes": ["risk_violations", "max_drawdown", "slot_anomaly"],
                "gate_failed_checks": [
                    "health_ok",
                    "max_drawdown_ok",
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
    assert [row["code"] for row in payload["root_causes"]] == ["risk_violations", "max_drawdown", "slot_anomaly"]
    assert [row["code"] for row in payload["derived_wrappers"]] == ["rollback_hard", "ops_status_red"]
    assert any(row["code"] == "review_gate" for row in payload["secondary_checks"])
    assert payload["repair_order"][0]["group"] == "root_causes"
    assert Path(payload["artifact"]).exists()
