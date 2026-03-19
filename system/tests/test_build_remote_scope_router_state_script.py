from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_scope_router_state.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_scope_router_state_prefers_crypto_review_candidate(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "ready_check_scope_market": "portfolio_margin_um",
            "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope_spot_vs_portfolio_margin_um:profitability_confirmed_but_auto_live_blocked",
        },
    )
    _write_json(
        review_dir / "20260315T094510Z_cross_market_operator_state.json",
        {
            "operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
            "review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "operator_head": {
                "area": "commodity_execution_close_evidence",
                "symbol": "XAUUSD",
                "action": "wait_for_paper_execution_close_evidence",
            },
            "review_head": {
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
                "blocker_detail": "no edge",
                "done_when": "micro confirms",
            },
            "review_backlog": [
                {
                    "rank": 1,
                    "area": "crypto_route",
                    "symbol": "SOLUSDT",
                    "action": "deprioritize_flow",
                    "priority_score": 58,
                    "blocker_detail": "no edge",
                    "done_when": "micro confirms",
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T09:46:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["scope_router_status"] == "review_candidate_inside_scope_not_trade_ready"
    assert payload["preferred_route_symbol"] == "SOLUSDT"
    assert payload["preferred_route_action"] == "deprioritize_flow"
    assert payload["route_recommendation"] == "hold_remote_idle_until_ticket_ready"
    assert payload["route_candidates_count"] == 1
    assert Path(str(payload["artifact"])).name == "20260315T094600Z_remote_scope_router_state.json"


def test_build_remote_scope_router_state_accepts_spot_market_candidate(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T094500Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "ready_check_scope_market": "spot",
            "identity_brief": "43.153.148.242:spot:aligned:shadow-only",
        },
    )
    _write_json(
        review_dir / "20260315T094510Z_cross_market_operator_state.json",
        {
            "operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
            "review_head_brief": "review:crypto_route:SOLUSDT:seed_ticket:77",
            "remote_live_operator_alignment_brief": "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:spot",
            "remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:spot",
            "review_head": {
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "seed_ticket",
                "priority_score": 77,
                "blocker_detail": "",
                "done_when": "journal is ready",
            },
            "review_backlog": [
                {
                    "rank": 1,
                    "area": "crypto_route",
                    "symbol": "SOLUSDT",
                    "action": "seed_ticket",
                    "priority_score": 77,
                    "blocker_detail": "",
                    "done_when": "journal is ready",
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T09:46:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["remote_market"] == "spot"
    assert payload["scope_router_status"] == "review_candidate_inside_remote_scope"
    assert payload["preferred_route_symbol"] == "SOLUSDT"
    assert payload["route_candidates_count"] == 1
