from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "paper_consecutive_loss_guardrail_status.py"
)


def test_status_reports_manual_ack_blocked_without_last_loss_ts(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ledger_path = tmp_path / "paper_execution_ledger.jsonl"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "consecutive_losses": 8,
                "daily_realized_pnl": 0.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "domain": "paper_execution",
                        "side": "SELL",
                        "ts": "2026-02-27T07:00:00+00:00",
                        "realized_pnl_change": 3.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "domain": "paper_execution",
                        "side": "SELL",
                        "ts": "2026-02-27T08:00:00+00:00",
                        "realized_pnl_change": -2.5,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "domain": "paper_execution",
                        "side": "SELL",
                        "ts": "2026-02-27T09:00:00+00:00",
                        "realized_pnl_change": -1.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "domain": "paper_execution",
                        "side": "SELL",
                        "ts": "2026-02-27T10:00:00+00:00",
                        "realized_pnl_change": -0.5,
                    },
                    ensure_ascii=False,
                ),
            ]
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
            "--ledger-path",
            str(ledger_path),
            "--now",
            "2026-03-08T12:00:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["guardrail_hit"] is True
    assert payload["manual_ack_eligible"] is False
    assert "last_loss_ts_missing" in payload["manual_ack_reasons"]
    assert payload["ack"]["present"] is False
    assert payload["ack"]["eligible_for_apply_now"] is False
    assert "ack_missing" in payload["ack"]["apply_reasons"]
    assert payload["backfill_preview"]["trailing_negative_streak"] == 3
    assert payload["backfill_preview"]["strict_candidate"]["eligible"] is False
    assert "trailing_negative_streak_mismatch(expected=8,actual=3)" in payload["backfill_preview"]["strict_candidate"][
        "reasons"
    ]
    assert payload["backfill_preview"]["fallback_candidate"]["eligible"] is True
    assert payload["backfill_preview"]["fallback_candidate"]["selected_method"] == "ledger_latest_negative_fallback"
    assert (
        payload["backfill_preview"]["fallback_candidate"]["selected_last_loss_ts"]
        == "2026-02-27T10:00:00+00:00"
    )
    assert payload["recovery_plan"]["next_action"] == "review_fallback_last_loss_ts_backfill"
    assert payload["recovery_plan"]["action_level"] == "review"
    assert len(payload["recovery_plan"]["commands"]) == 3
    assert payload["write_projection"]["write_chain_possible"] is True
    assert payload["write_projection"]["would_require_fallback_write"] is True
    assert payload["write_projection"]["projected_step_count"] == 3
    assert payload["write_projection"]["projected_steps"][0]["simulated_step"] == "fallback_backfill_write"
    assert payload["write_projection"]["projected_steps"][0]["projected_next_action"] == "write_manual_ack"
    assert payload["write_projection"]["projected_steps"][1]["simulated_step"] == "ack_write"
    assert payload["write_projection"]["projected_steps"][1]["projected_next_action"] == "run_full_cycle_with_existing_ack"
    assert payload["write_projection"]["terminal_action"] == "full_cycle_gate_ready"
    assert payload["ack_archive"]["live_ack"]["present"] is False
    assert payload["ack_archive"]["archive"]["file_count"] == 0
    assert payload["ack_archive"]["archive"]["manifest_present"] is False


def test_status_reports_existing_ack_eligible_when_snapshot_matches(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ack_path = tmp_path / "paper_consecutive_loss_ack.json"
    checksum_path = tmp_path / "paper_consecutive_loss_ack_checksum.json"

    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "consecutive_losses": 4,
                "last_loss_ts": "2026-03-07T20:00:00+00:00",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ack_payload = {
        "generated_at": "2026-03-08T10:00:00+00:00",
        "expires_at": "2026-03-09T10:00:00+00:00",
        "guardrail": "consecutive_loss_stop",
        "use_limit": 1,
        "uses_remaining": 1,
        "active": True,
        "streak_snapshot": 4,
        "cooldown_hours_required": 12.0,
        "last_loss_ts": "2026-03-07T20:00:00+00:00",
        "allow_missing_last_loss_ts": False,
        "note": "manual-check",
    }
    ack_path.write_text(json.dumps(ack_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sha_proc = subprocess.run(
        [
            "python3",
            "-c",
            "import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); h=hashlib.sha256(p.read_bytes()).hexdigest(); print(h)",
            str(ack_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-08T10:00:00+00:00",
                "artifact": str(ack_path),
                "sha256": sha_proc.stdout.strip(),
            },
            ensure_ascii=False,
            indent=2,
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
            "--now",
            "2026-03-08T12:00:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["manual_ack_eligible"] is True
    assert payload["ack"]["present"] is True
    assert payload["ack"]["checksum_valid"] is True
    assert payload["ack"]["streak_matches_current"] is True
    assert payload["ack"]["eligible_for_apply_now"] is True
    assert payload["ack"]["apply_reasons"] == []
    assert payload["recovery_plan"]["next_action"] == "run_full_cycle_with_existing_ack"
    assert payload["recovery_plan"]["action_level"] == "execute"
    assert payload["write_projection"]["write_chain_possible"] is True
    assert payload["write_projection"]["terminal_action"] == "full_cycle_gate_ready"
    assert payload["write_projection"]["projected_step_count"] == 1
    assert payload["write_projection"]["projected_steps"][0]["simulated_step"] == "full_cycle_gate_ready"
    assert payload["ack_archive"]["live_ack"]["present"] is True
    assert payload["ack_archive"]["live_ack"]["checksum_valid"] is True
    assert payload["ack_archive"]["archive"]["latest_archive"] is None


def test_status_backfill_preview_reports_strict_match_candidate(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    ledger_path = tmp_path / "paper_execution_ledger.jsonl"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "consecutive_losses": 2,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "domain": "paper_execution",
                        "side": "SELL",
                        "ts": "2026-03-07T07:00:00+00:00",
                        "realized_pnl_change": 1.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "domain": "paper_execution",
                        "side": "SELL",
                        "ts": "2026-03-07T08:00:00+00:00",
                        "realized_pnl_change": -1.2,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "domain": "paper_execution",
                        "side": "SELL",
                        "ts": "2026-03-07T09:00:00+00:00",
                        "realized_pnl_change": -0.4,
                    },
                    ensure_ascii=False,
                ),
            ]
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
            "--ledger-path",
            str(ledger_path),
            "--now",
            "2026-03-08T12:00:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["backfill_preview"]["strict_candidate"]["eligible"] is True
    assert payload["backfill_preview"]["strict_candidate"]["selected_method"] == "ledger_trailing_streak_match"
    assert payload["backfill_preview"]["strict_candidate"]["selected_last_loss_ts"] == "2026-03-07T09:00:00+00:00"
    assert payload["backfill_preview"]["strict_candidate"]["reasons"] == []


def test_status_reports_ack_archive_summary_when_archive_exists(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    archive_dir = tmp_path / "archive"
    archive_manifest_path = archive_dir / "manifest.jsonl"
    archive_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "consecutive_losses": 8,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    archive_path = archive_dir / "paper_consecutive_loss_ack_20260308T120000_000000Z.json"
    archive_payload = {
        "generated_at": "2026-03-08T00:00:00+00:00",
        "expires_at": "2026-03-09T00:00:00+00:00",
        "active": False,
        "uses_remaining": 0,
        "use_limit": 1,
        "streak_snapshot": 4,
        "last_loss_ts": "2026-03-07T00:00:00+00:00",
        "consumed_at": "2026-03-08T12:00:00+00:00",
        "consume_reason": "single_use_consumed",
        "archived_at": "2026-03-08T12:00:01+00:00",
        "archive_reason": "consumed",
    }
    archive_path.write_text(
        json.dumps(archive_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sha_proc = subprocess.run(
        [
            "python3",
            "-c",
            "import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); h=hashlib.sha256(p.read_bytes()).hexdigest(); print(h)",
            str(archive_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    archive_path.with_suffix(".json.sha256").write_text(
        f"{sha_proc.stdout.strip()}  {archive_path.name}\n",
        encoding="utf-8",
    )
    archive_manifest_path.write_text(
        json.dumps(
            {
                "ts_utc": "2026-03-08T12:00:01+00:00",
                "event": "consumed_archive",
                "path": str(archive_path),
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
            "--archive-dir",
            str(archive_dir),
            "--archive-manifest-path",
            str(archive_manifest_path),
            "--now",
            "2026-03-08T12:00:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ack_archive"]["archive"]["present"] is True
    assert payload["ack_archive"]["archive"]["file_count"] == 1
    assert payload["ack_archive"]["archive"]["consumed_archive_events_in_tail"] == 1
    assert payload["ack_archive"]["archive"]["latest_archive"]["checksum_valid"] is True
    assert payload["ack_archive"]["archive"]["latest_archive"]["payload"]["archive_reason"] == "consumed"


def test_status_recovery_plan_prefers_manual_ack_when_last_loss_ts_present(tmp_path: Path) -> None:
    state_path = tmp_path / "spot_paper_state.json"
    state_path.write_text(
        json.dumps(
            {
                "date": "2026-03-08",
                "consecutive_losses": 4,
                "last_loss_ts": "2026-03-07T00:00:00+00:00",
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
            "--now",
            "2026-03-08T12:00:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["manual_ack_eligible"] is True
    assert payload["recovery_plan"]["next_action"] == "write_manual_ack"
    assert payload["recovery_plan"]["action_level"] == "write"
    assert len(payload["recovery_plan"]["commands"]) == 2
