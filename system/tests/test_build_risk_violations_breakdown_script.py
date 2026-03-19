from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


SCRIPT = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_risk_violations_breakdown.py")


def test_build_risk_violations_breakdown_detects_duplicate_drawdown(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "max_drawdown_max": 0.18,
                    "mode_health_max_drawdown_max": 0.30,
                    "mode_health_max_violations": 1,
                    "positive_window_ratio_min": 0.70,
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (review_dir / "20260310T105015Z_live_gate_blocker_report.json").write_text(
        json.dumps({"current_decision": "do_not_start_formal_live"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260310T111017Z_ops_live_gate_breakdown.json").write_text(
        json.dumps(
            {
                "blocking_reason_codes": ["rollback_hard", "risk_violations", "max_drawdown", "slot_anomaly"],
                "root_causes": [
                    {"code": "risk_violations"},
                    {"code": "max_drawdown"},
                    {"code": "slot_anomaly"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
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
    assert payload["duplicate_drawdown_blocker_active"] is True
    assert payload["config_thresholds"]["validation.max_drawdown_max"] == 0.18
    assert payload["config_thresholds"]["validation.mode_health_max_violations"] == 1
    assert "same drawdown breach" in payload["finding"]["detail"]
    assert payload["patch_options"]["minimal_patch"]["summary"]

