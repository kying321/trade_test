from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_local_pi_recovery_handoff.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_local_pi_recovery_handoff_reports_consumed_archive(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    review_dir = workspace_root / "output" / "review"
    checkpoint_dir = review_dir / "local_pi_recovery_checkpoints"
    archive_dir = review_dir / "paper_consecutive_loss_ack_archive"

    _write_json(
        workspace_root / "output" / "state" / "spot_paper_state.json",
        {
            "date": "2026-03-09",
            "cash_usdt": 0.0,
            "eth_qty": 0.0,
            "avg_cost": 0.0,
            "equity_peak": 0.0,
            "daily_realized_pnl": 0.0,
            "consecutive_losses": 8,
            "last_loss_ts": "2026-02-27T08:00:00+00:00",
        },
    )
    _write_jsonl(
        workspace_root / "output" / "logs" / "paper_execution_ledger.jsonl",
        [
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T07:00:00+00:00", "realized_pnl_change": 0.3},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T08:00:00+00:00", "realized_pnl_change": -0.1},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T09:00:00+00:00", "realized_pnl_change": -0.2},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T10:00:00+00:00", "realized_pnl_change": -0.4},
        ],
    )

    archive_payload = {
        "generated_at": "2026-03-08T17:36:56.712044+00:00",
        "expires_at": "2026-03-09T17:36:56.712044+00:00",
        "active": False,
        "uses_remaining": 0,
        "use_limit": 1,
        "streak_snapshot": 8,
        "last_loss_ts": "2026-02-27T08:00:00+00:00",
        "consumed_at": "2026-03-09T01:38:14.648796+08:00",
        "consume_reason": "single_use_consumed",
        "archived_at": "2026-03-08T17:38:38.205814+00:00",
        "archive_reason": "consumed",
        "archive_cycle_ts": "2026-03-09T01:38:14.648796+08:00",
        "note": "",
    }
    archive_path = archive_dir / "paper_consecutive_loss_ack_20260308T173838_205814Z.json"
    _write_json(archive_path, archive_payload)
    checksum_path = archive_path.with_suffix(".json.sha256")
    checksum_path.write_text(
        f"{hashlib.sha256(archive_path.read_bytes()).hexdigest()}  {archive_path.name}\n",
        encoding="utf-8",
    )
    _write_jsonl(
        archive_dir / "manifest.jsonl",
        [
            {
                "ts_utc": "2026-03-08T17:38:38.205814+00:00",
                "event": "consumed_archive",
                "path": str(archive_path),
                "checksum_path": str(checksum_path),
                "cycle_ts": "2026-03-09T01:38:14.648796+08:00",
            }
        ],
    )

    checkpoint_root = checkpoint_dir / "20260308T173656Z_local_pi_recovery_checkpoint"
    _write_json(
        checkpoint_root / "checkpoint.json",
        {
            "generated_at": "2026-03-08T17:36:56.679014+00:00",
            "note": "recovery_step:write_manual_ack",
            "state_fingerprint": "6f2f91b10f22e8aef4d10d86013b739fecb8adf93dc9434312e516541a5a4ae0",
            "files": [],
            "lock_acquired": True,
        },
    )
    _write_json(
        checkpoint_root / "checkpoint_checksum.json",
        {"sha256": "abc"},
    )

    _write_json(
        review_dir / "20260308T173846Z_pi_launchd_auto_retro.json",
        {
            "ts": "2026-03-08T17:38:46+00:00",
            "status": "degraded",
            "core_execution_status": "ok",
            "core_execution_reason": "no_trade_after_execution_check",
            "core_execution_decision": "no-trade",
            "ops_next_action": "none",
            "ops_next_action_reason": "stable",
            "duration_sec": 132.0,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace-system-root",
            str(workspace_root),
            "--review-dir",
            str(review_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["artifact_status_label"] == "handoff-ok"
    assert payload["operator_handoff"]["handoff_state"] == "ack_consumed_archived"
    assert payload["operator_handoff"]["latest_retro_reason"] == "no_trade_after_execution_check"
    assert payload["latest_checkpoint"]["present"] is True
    assert "rollback-local-pi-recovery-state" in payload["latest_checkpoint"]["rollback_guidance"]["dry_run_command"]
    assert payload["latest_ack_archive_status"]["latest_archive"]["payload"]["archive_reason"] == "consumed"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_local_pi_recovery_handoff_extracts_retro_summary_from_launchd_log(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    review_dir = workspace_root / "output" / "review"
    checkpoint_dir = review_dir / "local_pi_recovery_checkpoints"
    launchd_log = tmp_path / "logs" / "pi_cycle_launchd.log"

    _write_json(
        workspace_root / "output" / "state" / "spot_paper_state.json",
        {
            "date": "2026-03-09",
            "cash_usdt": 0.0,
            "eth_qty": 0.0,
            "avg_cost": 0.0,
            "equity_peak": 0.0,
            "daily_realized_pnl": 0.0,
            "consecutive_losses": 8,
            "last_loss_ts": "2026-02-27T08:00:00+00:00",
        },
    )
    _write_jsonl(
        workspace_root / "output" / "logs" / "paper_execution_ledger.jsonl",
        [
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T08:00:00+00:00", "realized_pnl_change": -0.1},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T09:00:00+00:00", "realized_pnl_change": -0.2},
            {"domain": "paper_execution", "side": "SELL", "ts": "2026-02-27T10:00:00+00:00", "realized_pnl_change": -0.4},
        ],
    )
    launchd_log.parent.mkdir(parents=True, exist_ok=True)
    launchd_log.write_text(
        "\n".join(
            [
                "[2026-03-08T17:30:44Z] pi_cycle_launchd done rc=0",
                json.dumps(
                    {
                        "envelope_version": "1.0",
                        "domain": "pi_cycle",
                        "ts": "2026-03-08T17:38:46+00:00",
                        "status": "degraded",
                        "duration_sec": 131.1,
                        "core_execution_status": "ok",
                        "core_execution_reason": "no_trade_after_execution_check",
                        "core_execution_decision": "no-trade",
                        "ops_next_action": "none",
                        "ops_next_action_reason": "stable",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260308T173846Z_pi_launchd_auto_retro.json",
        {
            "generated_at_utc": "2026-03-08T17:38:46Z",
            "launchd_log": str(launchd_log),
            "launchd": {"done_rows": [{"ts_utc": "2026-03-08T17:30:44Z", "rc": 0}]},
            "samples": {"total_success_rate": 1.0},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace-system-root",
            str(workspace_root),
            "--review-dir",
            str(review_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["latest_retro"]["summary_source"] == "launchd_log_tail"
    assert payload["latest_retro"]["core_execution_reason"] == "no_trade_after_execution_check"
    assert payload["operator_handoff"]["latest_retro_reason"] == "no_trade_after_execution_check"


def test_build_local_pi_recovery_handoff_errors_when_workspace_missing(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace-system-root",
            str(tmp_path / "missing"),
            "--review-dir",
            str(tmp_path / "review"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 4, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "workspace_missing"
