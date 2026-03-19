from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_openclaw_orderflow_executor_state.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_openclaw_orderflow_executor_state_marks_ready_when_unit_and_heartbeat_exist(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked", "ready_check_scope_market": "portfolio_margin_um"},
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_status": "queued_wait_trade_readiness",
            "preferred_route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-1",
            "blocker_detail": "no edge",
        },
    )
    _write_json(
        review_dir / "20260315T102003Z_openclaw_orderflow_executor_heartbeat.json",
        {
            "executor_status": "shadow_guarded_idle",
            "executor_brief": "shadow_guarded_idle:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "idempotency_key": "entry-1",
        },
    )
    unit_preview = review_dir / "20260315T102004Z_openclaw_orderflow_executor.service"
    unit_preview.write_text("[Unit]\nDescription=Fenlie OpenClaw Orderflow Executor\n", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:20:40Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_status"] == "shadow_guarded_executor_ready"
    assert payload["service_name"] == "openclaw-orderflow-executor.service"
    assert payload["service_mode"] == "shadow_guarded"
    assert payload["service_mode_source"] == "heartbeat"
    assert payload["runtime_boundary_status"] == ""
    assert payload["heartbeat_status"] == "shadow_guarded_idle"
    assert payload["unit_preview_path"].endswith("_openclaw_orderflow_executor.service")


def test_build_openclaw_orderflow_executor_state_surfaces_guarded_probe_only_runtime(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:spot:split_scope:blocked", "ready_check_scope_market": "spot"},
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot",
            "queue_status": "queued_execution_contract_blocked",
            "preferred_route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot | blocked:remote_execution_contract | not_attempted_execution_contract_blocked",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-2",
            "blocker_detail": "runtime promotion requested",
        },
    )
    _write_json(
        review_dir / "20260315T102003Z_openclaw_orderflow_executor_heartbeat.json",
        {
            "executor_mode": "spot_live_guarded",
            "executor_mode_source": "contract_state",
            "executor_status": "spot_live_guarded_probe_capable",
            "executor_brief": "spot_live_guarded_probe_capable:SOLUSDT:queued_execution_contract_blocked:spot",
            "executor_runtime_boundary_status": "guarded_probe_only_runtime",
            "executor_runtime_boundary_reason_codes": ["guarded_probe_only_mode"],
            "idempotency_key": "entry-2",
        },
    )
    unit_preview = review_dir / "20260315T102004Z_openclaw_orderflow_executor.service"
    unit_preview.write_text("[Unit]\nDescription=Fenlie OpenClaw Orderflow Executor\n", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:20:41Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_status"] == "spot_live_guarded_probe_capable"
    assert payload["service_mode"] == "spot_live_guarded"
    assert payload["service_mode_source"] == "contract_state"
    assert payload["runtime_boundary_status"] == "guarded_probe_only_runtime"
    assert payload["runtime_boundary_reason_codes"] == ["guarded_probe_only_mode"]


def test_build_openclaw_orderflow_executor_state_surfaces_guarded_probe_completion(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:spot:spot_lane:promotion_requested", "ready_check_scope_market": "spot"},
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot",
            "queue_status": "queued_execution_contract_blocked",
            "preferred_route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot | blocked:remote_execution_contract | not_attempted_execution_contract_blocked",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-3",
            "blocker_detail": "runtime promotion requested",
        },
    )
    _write_json(
        review_dir / "20260315T102003Z_openclaw_orderflow_executor_heartbeat.json",
        {
            "executor_mode": "spot_live_guarded",
            "executor_mode_source": "contract_state",
            "executor_status": "spot_live_guarded_probe_completed",
            "executor_brief": "spot_live_guarded_probe_completed:SOLUSDT:queued_execution_contract_blocked:spot",
            "executor_runtime_boundary_status": "guarded_probe_only_runtime",
            "executor_runtime_boundary_reason_codes": ["guarded_probe_only_mode"],
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
            "idempotency_key": "entry-3",
        },
    )
    unit_preview = review_dir / "20260315T102004Z_openclaw_orderflow_executor.service"
    unit_preview.write_text("[Unit]\nDescription=Fenlie OpenClaw Orderflow Executor\n", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:20:42Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_status"] == "spot_live_guarded_probe_completed"
    assert payload["service_mode"] == "spot_live_guarded"
    assert payload["service_mode_source"] == "contract_state"
    assert payload["guarded_exec_probe_status"] == "probe_completed"
    assert payload["guarded_exec_probe_artifact"] == "/tmp/probe.json"
