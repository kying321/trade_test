from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_execution_contract_state.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_execution_contract_state_extracts_contract_from_handoff(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260317T050000Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "ready_check_scope_market": "portfolio_margin_um",
                "ready_check_scope_source": "portfolio_margin_um",
                "account_scope_alignment": {
                    "status": "split_scope_spot_vs_portfolio_margin_um",
                    "brief": "split_scope_spot_vs_portfolio_margin_um",
                },
                "execution_contract": {
                    "status": "non_executable_contract",
                    "brief": "non_executable_contract:portfolio_margin_um:portfolio_margin_um:spot_remote_lane_missing,portfolio_margin_um_read_only_mode,shadow_executor_only_mode",
                    "mode": "shadow_only",
                    "live_orders_allowed": False,
                    "executor_mode": "shadow_guarded",
                    "executor_mode_source": "bridge_context",
                    "reason_codes": [
                        "spot_remote_lane_missing",
                        "portfolio_margin_um_read_only_mode",
                        "shadow_executor_only_mode",
                    ],
                    "target_market": "portfolio_margin_um",
                    "target_source": "portfolio_margin_um",
                    "executable_lane_market": "portfolio_margin_um",
                    "account_scope_alignment_status": "split_scope_spot_vs_portfolio_margin_um",
                    "blocker_detail": "remote execution contract remains non-executable",
                    "done_when": "promote an explicit non-shadow remote send/ack/fill contract",
                },
            }
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-17T05:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["contract_status"] == "non_executable_contract"
    assert payload["contract_mode"] == "shadow_only"
    assert payload["executor_mode"] == "shadow_guarded"
    assert payload["executor_mode_source"] == "bridge_context"
    assert payload["live_orders_allowed"] is False
    assert payload["reason_codes"] == [
        "spot_remote_lane_missing",
        "portfolio_margin_um_read_only_mode",
        "shadow_executor_only_mode",
    ]
    assert payload["target_market"] == "portfolio_margin_um"
    assert payload["executable_lane_market"] == "portfolio_margin_um"
    assert Path(str(payload["artifact"])).name == "20260317T050100Z_remote_execution_contract_state.json"


def test_build_remote_execution_contract_state_surfaces_guarded_probe_allowed(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260317T050000Z_remote_live_handoff.json",
        {
            "operator_handoff": {
                "ready_check_scope_market": "spot",
                "ready_check_scope_source": "spot",
                "execution_contract": {
                    "status": "probe_only_contract",
                    "brief": "probe_only_contract:spot:spot:guarded_probe_only_mode",
                    "mode": "guarded_probe_only",
                    "guarded_probe_allowed": True,
                    "live_orders_allowed": False,
                    "executor_mode": "spot_live_guarded",
                    "executor_mode_source": "bridge_context",
                    "reason_codes": ["guarded_probe_only_mode"],
                    "target_market": "spot",
                    "target_source": "spot",
                    "executable_lane_market": "spot",
                    "account_scope_alignment_status": "scope_aligned",
                    "blocker_detail": "guarded probe only",
                    "done_when": "implement live send runtime",
                },
            }
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-17T05:01:01Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["contract_status"] == "probe_only_contract"
    assert payload["contract_mode"] == "guarded_probe_only"
    assert payload["guarded_probe_allowed"] is True
    assert payload["reason_codes"] == ["guarded_probe_only_mode"]
