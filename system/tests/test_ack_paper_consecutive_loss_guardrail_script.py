from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ack_paper_consecutive_loss_guardrail.py"
)


def test_ack_guardrail_dry_run_blocks_when_last_loss_ts_missing(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "cash_usdt": 0.0,
                "eth_qty": 0.0,
                "avg_cost": 0.0,
                "equity_peak": 0.0,
                "daily_realized_pnl": 0.0,
                "consecutive_losses": 8,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--state-path",
            str(state_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["eligible"] is False
    assert payload["state_fingerprint"]
    assert "last_loss_ts_missing" in payload["reasons"]
    assert payload["write_performed"] is False


def test_ack_guardrail_write_emits_artifact_and_checksum(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ack_path = tmp_path / "paper_consecutive_loss_ack.json"
    checksum_path = tmp_path / "paper_consecutive_loss_ack_checksum.json"
    pulse_lock_path = tmp_path / "run-halfhour-pulse.lock"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "cash_usdt": 0.0,
                "eth_qty": 0.0,
                "avg_cost": 0.0,
                "equity_peak": 0.0,
                "daily_realized_pnl": -1.5,
                "consecutive_losses": 4,
                "last_loss_ts": "2026-03-07T20:00:00+00:00",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--state-path",
            str(state_path),
            "--ack-path",
            str(ack_path),
            "--checksum-path",
            str(checksum_path),
            "--pulse-lock-path",
            str(pulse_lock_path),
            "--cooldown-hours",
            "12",
            "--ttl-hours",
            "24",
            "--write",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["write_performed"] is True
    assert ack_path.exists()
    assert checksum_path.exists()

    ack = json.loads(ack_path.read_text(encoding="utf-8"))
    assert ack["guardrail"] == "consecutive_loss_stop"
    assert ack["streak_snapshot"] == 4
    assert payload["state_fingerprint_match"] is None

    checksum = json.loads(checksum_path.read_text(encoding="utf-8"))
    assert checksum["artifact"] == str(ack_path)
    assert checksum["sha256"]


def test_ack_guardrail_write_blocks_when_expected_fingerprint_mismatches(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ack_path = tmp_path / "paper_consecutive_loss_ack.json"
    checksum_path = tmp_path / "paper_consecutive_loss_ack_checksum.json"
    pulse_lock_path = tmp_path / "run-halfhour-pulse.lock"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "daily_realized_pnl": -1.5,
                "consecutive_losses": 4,
                "last_loss_ts": "2026-03-07T20:00:00+00:00",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--state-path",
            str(state_path),
            "--ack-path",
            str(ack_path),
            "--checksum-path",
            str(checksum_path),
            "--pulse-lock-path",
            str(pulse_lock_path),
            "--cooldown-hours",
            "12",
            "--ttl-hours",
            "24",
            "--expected-state-fingerprint",
            "deadbeef",
            "--write",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 5, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["write_performed"] is False
    assert payload["state_fingerprint_match"] is False
    assert payload["reasons"] == ["state_fingerprint_mismatch"]
