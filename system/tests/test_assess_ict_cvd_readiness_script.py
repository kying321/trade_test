from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "assess_ict_cvd_readiness.py"
)
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = SYSTEM_ROOT / "config" / "ict_cvd_factor_spec.yaml"


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_assess_ict_cvd_readiness_current_like_public_dual(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "data": {"provider_profile": "dual_binance_bybit_public"},
            "validation": {
                "micro_cross_source_audit_enabled": True,
                "micro_min_trade_count": 20,
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--spec",
            str(DEFAULT_SPEC),
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(SYSTEM_ROOT),
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_profile_assessment"]
    assert current["profile"] == "dual_binance_bybit_public"
    assert current["cvd_lite_ready"] is True
    assert current["strict_cvd_ready"] is False
    assert current["readiness_label"] == "cvd-lite-ready"
    assert payload["data_sufficiency_conclusion"]["strict_cvd_supported_now"] is False


def test_assess_ict_cvd_readiness_opensource_profile_is_unavailable(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "data": {"provider_profile": "opensource_dual"},
            "validation": {
                "micro_cross_source_audit_enabled": False,
                "micro_min_trade_count": 10,
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--spec",
            str(DEFAULT_SPEC),
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(SYSTEM_ROOT),
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    current = payload["current_profile_assessment"]
    assert current["cvd_lite_ready"] is False
    assert current["readiness_label"] == "cvd-unavailable"
    assert "profile_not_supported:opensource_dual" in current["reasons"]
