from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_execution_identity_state.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_execution_identity_state_reads_remote_scope_and_diagnosis(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T094000Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "remote_user": "ubuntu",
                "remote_project_dir": "/home/ubuntu/openclaw-system",
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_quad": "runtime-ok / gate-blocked / risk-guard-blocked / notify-disabled",
                "ready_check_scope_market": "portfolio_margin_um",
                "ready_check_scope_source": "portfolio_margin_um",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {
                    "status": "split_scope_spot_vs_portfolio_margin_um",
                    "brief": "split_scope_spot_vs_portfolio_margin_um",
                    "blocking": False,
                    "blocker_detail": "spot ready-check and unified-account history refer to different execution scopes.",
                    "done_when": "promote market-specific live routing policy",
                },
                "execution_contract": {
                    "status": "non_executable_contract",
                    "brief": "non_executable_contract:portfolio_margin_um:portfolio_margin_um:portfolio_margin_um_read_only_mode,shadow_executor_only_mode",
                    "mode": "shadow_only",
                    "live_orders_allowed": False,
                    "executor_mode": "shadow_guarded",
                    "executor_mode_source": "bridge_context",
                    "reason_codes": [
                        "portfolio_margin_um_read_only_mode",
                        "shadow_executor_only_mode",
                    ],
                    "blocker_detail": "remote execution contract remains non-executable",
                    "done_when": "promote a non-shadow remote send/ack/fill contract",
                },
                "remote_live_diagnosis": {
                    "status": "profitability_confirmed_but_auto_live_blocked",
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                    "blocker_detail": "ops_live_gate and risk_guard still block automation",
                    "done_when": "clear ops_live_gate and risk_guard blockers",
                    "profitability_window": "30d",
                    "profitability_pnl": 18.79,
                    "profitability_trade_count": 38,
                },
                "remote_live_history": {
                    "window_brief": "24h:14.8pnl/20tr/1open | 30d:18.79pnl/38tr/1open"
                },
            }
        },
    )
    _write_json(
        review_dir / "20260315T094001Z_live_gate_blocker_report.json",
        {
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked"},
                {"name": "risk_guard", "status": "blocked"},
            ],
            "live_decision": {
                "current_decision": "do_not_start_formal_live",
                "summary": "clear ops_live_gate first",
            },
        },
    )
    _write_json(
        review_dir / "latest_remote_live_history_audit.json",
        {"market": "portfolio_margin_um", "window_brief": "30d:18.79pnl/38tr/1open"},
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T09:45:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["ready_check_scope_brief"] == "portfolio_margin_um:portfolio_margin_um"
    assert payload["account_scope_alignment_brief"] == "split_scope_spot_vs_portfolio_margin_um"
    assert payload["execution_contract_status"] == "non_executable_contract"
    assert payload["execution_contract_mode"] == "shadow_only"
    assert payload["execution_contract_executor_mode"] == "shadow_guarded"
    assert payload["execution_contract_executor_mode_source"] == "bridge_context"
    assert payload["execution_contract_live_orders_allowed"] is False
    assert payload["execution_contract_reason_codes"] == [
        "portfolio_margin_um_read_only_mode",
        "shadow_executor_only_mode",
    ]
    assert payload["remote_live_diagnosis_status"] == "profitability_confirmed_but_auto_live_blocked"
    assert payload["identity_brief"].startswith("43.153.148.242:portfolio_margin_um:")
    assert payload["blocking_layers"] == ["ops_live_gate", "risk_guard"]
    assert Path(str(payload["artifact"])).name == "20260315T094500Z_remote_execution_identity_state.json"


def test_build_remote_execution_identity_state_prefers_contract_state_when_present(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T094000Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "remote_host": "43.153.148.242",
                "ready_check_scope_market": "portfolio_margin_um",
                "ready_check_scope_source": "portfolio_margin_um",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {
                    "status": "split_scope_spot_vs_portfolio_margin_um",
                    "brief": "split_scope_spot_vs_portfolio_margin_um",
                },
                "execution_contract": {
                    "status": "live_executable_contract",
                    "brief": "stale_handoff_contract",
                    "mode": "live_executable",
                    "live_orders_allowed": True,
                    "reason_codes": [],
                },
                "remote_live_diagnosis": {
                    "status": "profitability_confirmed_but_auto_live_blocked",
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                },
            }
        },
    )
    _write_json(
        review_dir / "20260315T094000Z_remote_execution_contract_state.json",
        {
            "contract_status": "non_executable_contract",
            "contract_brief": "non_executable_contract:portfolio_margin_um:portfolio_margin_um:shadow_executor_only_mode",
            "contract_mode": "shadow_only",
            "guarded_probe_allowed": True,
            "live_orders_allowed": False,
            "executor_mode": "shadow_guarded",
            "executor_mode_source": "contract_state",
            "reason_codes": ["shadow_executor_only_mode"],
            "blocker_detail": "contract_state_is_source_owned",
            "done_when": "promote contract state",
        },
    )
    _write_json(
        review_dir / "20260315T094001Z_live_gate_blocker_report.json",
        {
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "blockers": [],
            "live_decision": {
                "current_decision": "do_not_start_formal_live",
                "summary": "clear ops_live_gate first",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T09:45:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_contract_status"] == "non_executable_contract"
    assert payload["execution_contract_brief"] == (
        "non_executable_contract:portfolio_margin_um:portfolio_margin_um:shadow_executor_only_mode"
    )
    assert payload["execution_contract_mode"] == "shadow_only"
    assert payload["execution_contract_executor_mode"] == "shadow_guarded"
    assert payload["execution_contract_executor_mode_source"] == "contract_state"
    assert payload["execution_contract_guarded_probe_allowed"] is True
    assert payload["execution_contract_live_orders_allowed"] is False
    assert payload["execution_contract_reason_codes"] == ["shadow_executor_only_mode"]
    assert payload["artifacts"]["remote_execution_contract_state"].endswith(
        "_remote_execution_contract_state.json"
    )
