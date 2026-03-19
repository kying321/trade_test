from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reconstruct_pi_halfhour_matrix.py"
)


def _run_script(tmp_path: Path, state_md: Path, *, now_utc: str, window_hours: int) -> subprocess.CompletedProcess[str]:
    output_dir = tmp_path / "review"
    return subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--state-md",
            str(state_md),
            "--output-dir",
            str(output_dir),
            "--now-utc",
            now_utc,
            "--window-hours",
            str(window_hours),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _run_script_with_launchd_log(
    tmp_path: Path,
    state_md: Path,
    *,
    now_utc: str,
    window_hours: int,
    launchd_log: Path,
    warmup_window_hours: int | None = None,
) -> subprocess.CompletedProcess[str]:
    output_dir = tmp_path / "review"
    cmd = [
        "python3",
        str(SCRIPT_PATH),
        "--state-md",
        str(state_md),
        "--output-dir",
        str(output_dir),
        "--now-utc",
        now_utc,
        "--window-hours",
        str(window_hours),
        "--launchd-log",
        str(launchd_log),
    ]
    if warmup_window_hours is not None:
        cmd.extend(["--warmup-window-hours", str(warmup_window_hours)])
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_reconstruct_pi_halfhour_matrix_reports_ready_when_all_buckets_present(tmp_path: Path) -> None:
    state_md = tmp_path / "STATE.md"
    state_md.write_text(
        "\n".join(
            [
                'LIE_DRYRUN_EVENT={"ts":"2026-03-08T08:05:00+00:00","decision":"no-trade","action":"FLAT_USDT","bucket":"premarket","guardrails":{"hit":false,"reasons":[]},"lie":{"daemon":{"would_run_pulse":true,"pulse_reason":"run_slot_due","mode":"compat_scheduler","command":"run-daemon","due_slots":["premarket"]},"pulse":{"ran":true,"reason":null}},"paper":{"equity":100.0,"cash_usdt":100.0,"eth_qty":0.0,"consecutive_losses":0}}',
                'LIE_DRYRUN_EVENT={"ts":"2026-03-08T08:35:00+00:00","decision":"no-trade","action":"FLAT_USDT","bucket":"intraday","guardrails":{"hit":false,"reasons":[]},"lie":{"daemon":{"would_run_pulse":true,"pulse_reason":"run_slot_due","mode":"compat_scheduler","command":"run-daemon","due_slots":["intraday:08:30"]},"pulse":{"ran":true,"reason":null}},"paper":{"equity":100.0,"cash_usdt":100.0,"eth_qty":0.0,"consecutive_losses":0}}',
                'LIE_DRYRUN_EVENT={"ts":"2026-03-08T09:05:00+00:00","decision":"no-trade","action":"FLAT_USDT","bucket":"intraday","guardrails":{"hit":false,"reasons":[]},"lie":{"daemon":{"would_run_pulse":true,"pulse_reason":"run_slot_due","mode":"compat_scheduler","command":"run-daemon","due_slots":["intraday:09:00"]},"pulse":{"ran":true,"reason":null}},"paper":{"equity":100.0,"cash_usdt":100.0,"eth_qty":0.0,"consecutive_losses":0}}',
                'PI_CYCLE_EVENT={"ts":"2026-03-08T09:35:00+00:00","status":"ok","core_execution_status":"ok","core_execution_reason":"no_trade_guardrail","core_execution_action":"FLAT_USDT","core_execution_decision":"no-trade","core_execution_guardrail_hit":false,"core_execution_guardrail_reasons":[]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = _run_script(
        tmp_path,
        state_md,
        now_utc="2026-03-08T10:00:00Z",
        window_hours=2,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ready"
    assert payload["ready_for_paper_mode"] is True
    assert payload["coverage"] == 1.0
    assert payload["missing_buckets"] == 0
    assert payload["fail_reasons"] == []
    assert Path(payload["paper_readiness_json"]).exists()
    assert Path(payload["checksum_json"]).exists()

    checksum = json.loads(Path(payload["checksum_json"]).read_text(encoding="utf-8"))
    assert "paper_readiness_json" in checksum["artifacts"]


def test_reconstruct_pi_halfhour_matrix_includes_warmup_gate_from_launchd_log(tmp_path: Path) -> None:
    state_md = tmp_path / "STATE.md"
    state_md.write_text(
        'LIE_DRYRUN_EVENT={"ts":"2026-03-08T08:05:00+00:00","decision":"no-trade","action":"FLAT_USDT","bucket":"premarket","guardrails":{"hit":true,"reasons":["paper_mode_readiness_gate(paper_mode_readiness_blocked)"]},"lie":{"daemon":{"would_run_pulse":true,"pulse_reason":"run_slot_due","mode":"compat_scheduler","command":"run-daemon","due_slots":["premarket"]},"pulse":{"ran":true,"reason":null}},"paper":{"equity":0.0,"cash_usdt":0.0,"eth_qty":0.0,"consecutive_losses":8}}\n',
        encoding="utf-8",
    )
    launchd_log = tmp_path / "pi_cycle_launchd.log"
    launchd_log.write_text(
        "\n".join(
            [
                "[2026-03-08T08:00:00Z] pi_cycle_launchd start",
                "[2026-03-08T08:30:00Z] pi_cycle_launchd start",
                "[2026-03-08T09:00:00Z] pi_cycle_launchd start",
                "[2026-03-08T09:30:00Z] pi_cycle_launchd start",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = _run_script_with_launchd_log(
        tmp_path,
        state_md,
        now_utc="2026-03-08T10:00:00Z",
        window_hours=2,
        launchd_log=launchd_log,
        warmup_window_hours=2,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    warmup_gate = payload["warmup_gate"]
    assert warmup_gate["ok"] is True
    assert warmup_gate["coverage"] == 1.0
    assert warmup_gate["missing_buckets"] == 0
    assert warmup_gate["largest_missing_block_hours"] == 0.0


def test_reconstruct_pi_halfhour_matrix_reports_blocked_when_buckets_are_missing(tmp_path: Path) -> None:
    state_md = tmp_path / "STATE.md"
    state_md.write_text(
        'LIE_DRYRUN_EVENT={"ts":"2026-03-08T08:05:00+00:00","decision":"no-trade","action":"FLAT_USDT","bucket":"premarket","guardrails":{"hit":true,"reasons":["paper_mode_readiness_gate(paper_mode_readiness_blocked)"]},"lie":{"daemon":{"would_run_pulse":true,"pulse_reason":"run_slot_due","mode":"compat_scheduler","command":"run-daemon","due_slots":["premarket"]},"pulse":{"ran":true,"reason":null}},"paper":{"equity":0.0,"cash_usdt":0.0,"eth_qty":0.0,"consecutive_losses":8}}\n',
        encoding="utf-8",
    )

    proc = _run_script(
        tmp_path,
        state_md,
        now_utc="2026-03-08T10:00:00Z",
        window_hours=2,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "blocked"
    assert payload["ready_for_paper_mode"] is False
    assert payload["missing_buckets"] == 3
    assert "coverage_min" in payload["fail_reasons"]
    assert "after_last_allowlisted_buckets_max" in payload["fail_reasons"]
