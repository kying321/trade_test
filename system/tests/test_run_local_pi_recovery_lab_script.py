from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_local_pi_recovery_lab.py"
)
RUNTIME_CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "pi"
    / "scripts"
    / "lie_spot_halfhour_core.py"
)


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_recovery_lab_runs_isolated_fallback_ack_and_archive(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    state_path = workspace_root / "output" / "state" / "spot_paper_state.json"
    ledger_path = workspace_root / "output" / "logs" / "paper_execution_ledger.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "consecutive_losses": 8,
                "daily_realized_pnl": -1.2,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_ledger(
        ledger_path,
        [
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T07:00:00+00:00", "realized_pnl_change": 0.3},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T08:00:00+00:00", "realized_pnl_change": -0.1},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T09:00:00+00:00", "realized_pnl_change": -0.2},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T10:00:00+00:00", "realized_pnl_change": -0.4},
        ],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace-system-root",
            str(workspace_root),
            "--lab-parent-dir",
            str(workspace_root / "output" / "review"),
            "--runtime-core-path",
            str(RUNTIME_CORE_PATH),
            "--allow-fallback-write",
            "--now",
            "2026-03-08T12:00:00+00:00",
            "--cycle-ts",
            "2026-03-08T12:05:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["copied_inputs"]["state"] is True
    assert payload["copied_inputs"]["ledger"] is True
    assert payload["backfill_result"]["write_performed"] is True
    assert payload["backfill_result"]["selected_method"] == "ledger_latest_negative_fallback"
    assert payload["ack_result"]["write_performed"] is True
    assert payload["runtime_consume"]["consume_ok"] is True
    assert payload["runtime_consume"]["archive_ok"] is True
    assert payload["final_archive_status"]["archive"]["file_count"] == 1
    assert payload["final_archive_status"]["archive"]["latest_archive"]["payload"]["archive_reason"] == "consumed"
    assert payload["final_status"]["ack_archive"]["archive"]["file_count"] == 1
    assert payload["artifact_status_label"] == "lab-ok"
    assert payload["artifact_label"] == "recovery-lab:ok"
    assert payload["artifact_tags"] == ["local-pi", "recovery-lab", "ok", "lab-ok"]
    assert payload["projection_validation"]["prefix_match"] is True
    assert payload["projection_validation"]["full_sequence_match"] is False
    assert payload["projection_validation"]["actual_step_count"] == 3
    assert payload["projection_validation"]["actual_steps"][0]["simulated_step"] == "fallback_backfill_write"
    assert payload["projection_validation"]["actual_steps"][1]["simulated_step"] == "ack_write"
    assert payload["projection_validation"]["actual_steps"][2]["simulated_step"] == "ack_consumed_archive_ok"
    assert payload["projection_validation"]["compared_steps"][0]["match"] is True
    assert payload["projection_validation"]["compared_steps"][1]["match"] is True
    assert payload["projection_validation"]["compared_steps"][2]["match"] is False
    assert payload["operator_note"]["projection_prefix_match"] is True
    assert payload["operator_note"]["terminal_reason"] == "lab_validated_backfill_ack_and_archive_but_not_full_cycle"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_recovery_lab_reports_missing_required_inputs(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace-system-root",
            str(workspace_root),
            "--lab-parent-dir",
            str(workspace_root / "output" / "review"),
            "--runtime-core-path",
            str(RUNTIME_CORE_PATH),
            "--now",
            "2026-03-08T12:00:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 4, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "missing_required_inputs"
    assert payload["artifact_status_label"] == "missing-inputs"
    assert payload["artifact_label"] == "recovery-lab:missing_required_inputs"
    assert payload["artifact_tags"] == ["local-pi", "recovery-lab", "missing_required_inputs", "missing-inputs"]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()
