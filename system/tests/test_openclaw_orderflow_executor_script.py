from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "openclaw_orderflow_executor.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_openclaw_orderflow_executor_writes_shadow_guarded_heartbeat(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_intent_queue.json",
        {
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "blocker_detail": "no edge",
            "done_when": "edge appears",
        },
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "journal_status": "intent_logged_guardian_blocked",
            "risk_verdict_brief": "blocked:risk_guard",
            "fill_status": "no_fill_execution_not_attempted",
            "last_entry_key": "entry-1",
            "blocker_detail": "no edge",
            "done_when": "edge appears",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {"name": "risk_guard", "status": "blocked", "reason_codes": ["ticket_missing:no_actionable_ticket"]}
            ]
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--once",
            "--now",
            "2026-03-15T10:20:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_status"] == "shadow_guarded_idle"
    assert payload["executor_mode"] == "shadow_guarded"
    assert payload["executor_mode_source"] == "default"
    assert payload["executor_runtime_boundary_status"] == "shadow_runtime_only"
    assert payload["executor_runtime_boundary_reason_codes"] == ["shadow_executor_only_mode"]
    assert payload["idempotency_status"] == "fresh_intent_observed"
    assert payload["executor_action"] == "observe_only_no_transport"
    assert Path(str(payload["artifact"])).name == "20260315T102030Z_openclaw_orderflow_executor_heartbeat.json"


def test_openclaw_orderflow_executor_marks_duplicate_intent(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_intent_queue.json",
        {
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-1",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_live_gate_blocker_report.json",
        {"blockers": [{"name": "risk_guard", "status": "blocked", "reason_codes": []}]},
    )
    _write_json(
        review_dir / "20260315T102010Z_openclaw_orderflow_executor_heartbeat.json",
        {"last_intent_key": "entry-1"},
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--once",
            "--now",
            "2026-03-15T10:20:31Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["idempotency_status"] == "duplicate_intent_seen"


def test_openclaw_orderflow_executor_marks_execution_contract_blocked(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_intent_queue.json",
        {
            "queue_status": "queued_execution_contract_blocked",
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:portfolio_margin_um",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "portfolio_margin_um",
            "blocker_detail": "execution contract is shadow only",
            "done_when": "promote non-shadow execution contract",
        },
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:portfolio_margin_um | blocked:remote_execution_contract | not_attempted_execution_contract_blocked",
            "journal_status": "intent_logged_guardian_blocked",
            "risk_verdict_brief": "blocked:remote_execution_contract",
            "fill_status": "no_fill_execution_not_attempted",
            "last_entry_key": "entry-contract-1",
            "blocker_detail": "execution contract is shadow only",
            "done_when": "promote non-shadow execution contract",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "remote_execution_contract",
                    "status": "blocked",
                    "reason_codes": ["shadow_executor_only_mode"],
                }
            ]
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--once",
            "--now",
            "2026-03-15T10:20:32Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_status"] == "shadow_guarded_contract_blocked"
    assert payload["executor_action"] == "idle_execution_contract_blocked"


def test_openclaw_orderflow_executor_surfaces_guarded_probe_only_runtime_before_probe(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_intent_queue.json",
        {
            "queue_status": "queued_execution_contract_blocked",
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "spot",
            "blocker_detail": "promotion requested but runtime not implemented",
            "done_when": "implement runtime",
        },
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot | blocked:remote_execution_contract | not_attempted_execution_contract_blocked",
            "journal_status": "intent_logged_guardian_blocked",
            "risk_verdict_brief": "blocked:remote_execution_contract",
            "fill_status": "no_fill_execution_not_attempted",
            "last_entry_key": "entry-contract-2",
            "blocker_detail": "promotion requested but runtime not implemented",
            "done_when": "implement runtime",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "remote_execution_contract",
                    "status": "blocked",
                    "reason_codes": ["requested_executor_mode_not_implemented"],
                }
            ]
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--mode",
            "spot_live_guarded",
            "--once",
            "--now",
            "2026-03-15T10:20:33Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_mode"] == "spot_live_guarded"
    assert payload["executor_status"] == "spot_live_guarded_probe_capable"
    assert payload["executor_runtime_boundary_status"] == "guarded_probe_only_runtime"
    assert payload["executor_runtime_boundary_reason_codes"] == ["guarded_probe_only_mode"]
    assert payload["executor_action"] == "idle_guarded_probe_only_runtime"


def test_openclaw_orderflow_executor_runs_guarded_probe_for_spot_live_guarded_ticket(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_intent_queue.json",
        {
            "queue_status": "queued_guarded_probe_ready",
            "queue_brief": "queued_guarded_probe_ready:SOLUSDT:seed_ticket:spot",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "spot",
            "execution_contract_mode": "guarded_probe_only",
            "execution_contract_guarded_probe_allowed": True,
            "execution_contract_reason_codes": ["guarded_probe_only_mode"],
            "ticket_selected_row": {
                "symbol": "SOLUSDT",
                "date": "2026-03-15",
                "allowed": True,
                "signal": {"side": "LONG"},
                "execution": {"mode": "SPOT_LONG_OR_SELL"},
                "sizing": {"quote_usdt": 4.5},
            },
            "blocker_detail": "guarded probe only",
            "done_when": "review probe evidence before live runtime",
        },
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot | blocked:remote_execution_contract | not_attempted_execution_contract_blocked",
            "journal_status": "intent_logged_guardian_blocked",
            "risk_verdict_brief": "blocked:remote_execution_contract",
            "fill_status": "no_fill_execution_not_attempted",
            "last_entry_key": "entry-contract-3",
            "blocker_detail": "promotion requested but runtime not implemented",
            "done_when": "implement runtime",
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "remote_execution_contract",
                    "status": "blocked",
                    "reason_codes": ["requested_executor_mode_not_implemented"],
                }
            ]
        },
    )
    stub_script = tmp_path / "guarded_exec_stub.py"
    stub_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "payload = {",
                '  "status": "probe_completed",',
                '  "artifact": "/tmp/probe.json",',
                '  "takeover": {"payload": {"steps": {"canary_order": {"executed": False}}}},',
                '  "argv": sys.argv[1:],',
                "}",
                "print(json.dumps(payload, ensure_ascii=False))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["OPENCLAW_GUARDED_EXEC_SCRIPT"] = str(stub_script)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--mode",
            "spot_live_guarded",
            "--once",
            "--now",
            "2026-03-15T10:20:34Z",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_mode"] == "spot_live_guarded"
    assert payload["executor_status"] == "spot_live_guarded_probe_completed"
    assert payload["executor_action"] == "guarded_probe_completed"
    assert payload["guarded_exec_probe_status"] == "probe_completed"
    assert payload["guarded_exec_probe_artifact"] == "/tmp/probe.json"
    request = payload["guarded_exec_probe_request"]
    assert request["symbol"] == "SOLUSDT"
    assert request["order_side"] == "BUY"
    assert request["quote_usdt"] == 4.5


def test_openclaw_orderflow_executor_prefers_contract_state_mode_when_arg_missing(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T102000Z_remote_execution_contract_state.json",
        {
            "contract_status": "probe_only_contract",
            "contract_mode": "guarded_probe_only",
            "guarded_probe_allowed": True,
            "executor_mode": "spot_live_guarded",
            "executor_mode_source": "contract_state",
            "reason_codes": ["guarded_probe_only_mode"],
        },
    )
    _write_json(
        review_dir / "20260315T102001Z_remote_intent_queue.json",
        {
            "queue_status": "queued_ticket_blocked",
            "queue_brief": "queued_ticket_blocked:BNBUSDT:seed_ticket:spot",
            "preferred_route_symbol": "BNBUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "spot",
            "execution_contract_mode": "guarded_probe_only",
            "execution_contract_guarded_probe_allowed": True,
            "execution_contract_reason_codes": ["guarded_probe_only_mode"],
        },
    )
    _write_json(
        review_dir / "20260315T102002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_ticket_blocked:BNBUSDT:seed_ticket:spot | blocked:risk_guard | no_actionable_ticket",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-contract-source-1",
            "blocker_detail": "ticket blocked",
        },
    )
    _write_json(
        review_dir / "20260315T102003Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "remote_execution_contract",
                    "status": "blocked",
                    "reason_codes": ["guarded_probe_only_mode"],
                }
            ]
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--once",
            "--now",
            "2026-03-15T10:20:35Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["executor_mode"] == "spot_live_guarded"
    assert payload["executor_mode_source"] == "contract_state"
    assert payload["executor_status"] == "spot_live_guarded_probe_capable"
    assert payload["guarded_exec_probe_status"] == "ticket_selected_row_missing"
