from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


SCRIPT = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_backtest_snapshot_breakdown.py")


def test_build_backtest_snapshot_breakdown_detects_stale_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    review_dir = output_dir / "review"
    artifacts_dir = output_dir / "artifacts"
    review_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "required_stable_replay_days": 3,
                    "review_backtest_start_date": "2015-01-01",
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (review_dir / "2026-03-10_ops_report.json").write_text(
        json.dumps(
            {
                "date": "2026-03-10",
                "gate_checks": {
                    "backtest_snapshot_ok": False,
                    "max_drawdown_ok": True,
                    "risk_violations_ok": True,
                },
                "gate_failed_checks": ["backtest_snapshot_ok"],
                "live_gate": {
                    "blocking_reason_codes": ["rollback_hard", "slot_anomaly", "backtest_snapshot", "ops_status_red"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "backtest_2015-01-01_2026-03-05.json").write_text(
        json.dumps({"annual_return": 0.1}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["blocker_active"] is True
    assert payload["snapshot_state"]["status"] == "snapshot_stale"
    assert payload["snapshot_state"]["latest_snapshot_end"] == "2026-03-05"
    assert payload["snapshot_state"]["latest_snapshot_age_days"] == 5
    assert payload["repair_sequence"][0]["command"].endswith("--start 2015-01-01 --end 2026-03-10")
    assert payload["repair_sequence"][2]["command"].endswith(
        "scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh && scripts/openclaw_cloud_bridge.sh remote-live-handoff"
    )
