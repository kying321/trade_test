from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backfill_paper_last_loss_ts.py"
)


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_backfill_dry_run_requires_trailing_streak_match(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ledger_path = tmp_path / "paper_execution_ledger.jsonl"
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 4, "last_loss_ts": None}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_ledger(
        ledger_path,
        [
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:00:00+00:00", "realized_pnl_change": 1.0},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:30:00+00:00", "realized_pnl_change": -0.2},
        ],
    )

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--state-path", str(state_path), "--ledger-path", str(ledger_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["eligible"] is False
    assert payload["state_fingerprint"]
    assert payload["selected_last_loss_ts"] is None
    assert "trailing_negative_streak_mismatch(expected=4,actual=1)" in payload["reasons"]


def test_backfill_write_sets_last_loss_ts_on_exact_streak_match(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ledger_path = tmp_path / "paper_execution_ledger.jsonl"
    pulse_lock_path = tmp_path / "run_halfhour_pulse.lock"
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 2, "last_loss_ts": None}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_ledger(
        ledger_path,
        [
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:00:00+00:00", "realized_pnl_change": 0.4},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:30:00+00:00", "realized_pnl_change": -0.1},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T09:00:00+00:00", "realized_pnl_change": -0.3},
        ],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--state-path",
            str(state_path),
            "--ledger-path",
            str(ledger_path),
            "--pulse-lock-path",
            str(pulse_lock_path),
            "--write",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["write_performed"] is True
    assert payload["selected_method"] == "ledger_trailing_streak_match"
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["last_loss_ts"] == "2026-03-08T09:00:00+00:00"
    assert payload["backup_path"]
    assert payload["state_fingerprint_match"] is None


def test_backfill_fallback_can_select_latest_negative_without_match(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ledger_path = tmp_path / "paper_execution_ledger.jsonl"
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 5, "last_loss_ts": None}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_ledger(
        ledger_path,
        [
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:00:00+00:00", "realized_pnl_change": 0.4},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:30:00+00:00", "realized_pnl_change": -0.1},
        ],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--state-path",
            str(state_path),
            "--ledger-path",
            str(ledger_path),
            "--allow-latest-loss-fallback",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["eligible"] is True
    assert payload["selected_method"] == "ledger_latest_negative_fallback"
    assert payload["selected_last_loss_ts"] == "2026-03-08T08:30:00+00:00"


def test_backfill_write_blocks_when_expected_fingerprint_mismatches(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ledger_path = tmp_path / "paper_execution_ledger.jsonl"
    pulse_lock_path = tmp_path / "run_halfhour_pulse.lock"
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 2, "last_loss_ts": None}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_ledger(
        ledger_path,
        [
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:00:00+00:00", "realized_pnl_change": 0.4},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T08:30:00+00:00", "realized_pnl_change": -0.1},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-03-08T09:00:00+00:00", "realized_pnl_change": -0.3},
        ],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--state-path",
            str(state_path),
            "--ledger-path",
            str(ledger_path),
            "--pulse-lock-path",
            str(pulse_lock_path),
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
