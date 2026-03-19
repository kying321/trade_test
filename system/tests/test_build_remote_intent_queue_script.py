from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_intent_queue.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_intent_queue_blocks_on_not_trade_ready_scope_candidate(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T095500Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "ready_check_scope_market": "portfolio_margin_um",
            "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked",
        },
    )
    _write_json(
        review_dir / "20260315T095530Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "scope_router_brief": "review_candidate_inside_scope_not_trade_ready:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "preferred_route_priority_score": 58,
            "route_recommendation": "hold_remote_idle_until_ticket_ready",
            "blocker_detail": "no edge",
            "done_when": "micro confirms",
        },
    )
    _write_json(
        review_dir / "20260315T095540Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket", "panic_cooldown_active"],
                    "blocked_candidate": {"symbol": "BNBUSDT", "side": "BUY"},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T095520Z_cross_market_operator_state.json",
        {
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
                "priority_tier": "review_queue_next",
                "reason": "no edge",
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
                    "priority_tier": "review_queue_next",
                    "reason": "no edge",
                    "blocker_detail": "no edge",
                    "done_when": "micro confirms",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T023857Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T02:38:57Z",
            "tickets": [
                {
                    "symbol": "XAUUSD",
                    "allowed": False,
                    "reasons": ["unsupported_symbol"],
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
            "2026-03-15T09:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["queue_status"] == "queued_wait_trade_readiness"
    assert payload["queue_recommendation"] == "hold_remote_idle_until_ticket_ready"
    assert payload["preferred_route_symbol"] == "SOLUSDT"
    assert payload["ticket_match_status"] == "row_missing"
    assert payload["guard_alignment_status"] == "risk_guard_candidate_mismatch"
    assert Path(str(payload["artifact"])).name == "20260315T095600Z_remote_intent_queue.json"


def test_build_remote_intent_queue_marks_ticket_ready_candidate(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T101000Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "ready_check_scope_market": "portfolio_margin_um",
            "identity_brief": "43.153.148.242:portfolio_margin_um:aligned:blocked",
        },
    )
    _write_json(
        review_dir / "20260315T101010Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_remote_scope",
            "scope_router_brief": "review_candidate_inside_remote_scope:ETHUSDT:seed_ticket:portfolio_margin_um",
            "preferred_route_symbol": "ETHUSDT",
            "preferred_route_action": "seed_ticket",
            "preferred_route_priority_score": 77,
            "route_recommendation": "seed_remote_intent_queue_from_review_head",
            "blocker_detail": "",
            "done_when": "journal is ready",
        },
    )
    _write_json(
        review_dir / "20260315T101020Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket"],
                    "blocked_candidate": {"symbol": "ETHUSDT", "side": "BUY"},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T101005Z_cross_market_operator_state.json",
        {
            "review_backlog": [
                {
                    "rank": 1,
                    "area": "crypto_route",
                    "symbol": "ETHUSDT",
                    "action": "seed_ticket",
                    "priority_score": 77,
                    "priority_tier": "review_queue_now",
                    "reason": "ticketable edge",
                    "blocker_detail": "",
                    "done_when": "journal is ready",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101030Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [
                {
                    "symbol": "ETHUSDT",
                    "allowed": True,
                    "reasons": [],
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
            "2026-03-15T10:11:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["queue_status"] == "queued_ticket_ready"
    assert payload["queue_recommendation"] == "seed_execution_journal_from_ticket_ready_intent"
    assert payload["intent_ready"] is True
    assert payload["ticket_match_status"] == "row_ready"
    assert payload["guard_alignment_status"] == "risk_guard_candidate_matches_route_symbol"
    assert payload["queue_rows_count"] == 1


def test_build_remote_intent_queue_blocks_on_non_executable_contract(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T101000Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "ready_check_scope_market": "portfolio_margin_um",
            "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked",
            "execution_contract_status": "non_executable_contract",
            "execution_contract_brief": "non_executable_contract:portfolio_margin_um:portfolio_margin_um:spot_remote_lane_missing,portfolio_margin_um_read_only_mode,shadow_executor_only_mode",
            "execution_contract_mode": "shadow_only",
            "execution_contract_live_orders_allowed": False,
            "execution_contract_reason_codes": [
                "spot_remote_lane_missing",
                "portfolio_margin_um_read_only_mode",
                "shadow_executor_only_mode",
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101010Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_remote_scope",
            "scope_router_brief": "review_candidate_inside_remote_scope:ETHUSDT:seed_ticket:portfolio_margin_um",
            "preferred_route_symbol": "ETHUSDT",
            "preferred_route_action": "seed_ticket",
            "preferred_route_priority_score": 77,
            "route_recommendation": "seed_remote_intent_queue_from_review_head",
            "blocker_detail": "",
            "done_when": "journal is ready",
        },
    )
    _write_json(
        review_dir / "20260315T101020Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "risk_guard",
                    "status": "clear",
                    "reason_codes": [],
                    "blocked_candidate": {},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T101005Z_cross_market_operator_state.json",
        {
            "review_backlog": [
                {
                    "rank": 1,
                    "area": "crypto_route",
                    "symbol": "ETHUSDT",
                    "action": "seed_ticket",
                    "priority_score": 77,
                    "priority_tier": "review_queue_now",
                    "reason": "ticketable edge",
                    "blocker_detail": "",
                    "done_when": "journal is ready",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101030Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [
                {
                    "symbol": "ETHUSDT",
                    "allowed": True,
                    "reasons": [],
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
            "2026-03-15T10:11:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["queue_status"] == "queued_execution_contract_blocked"
    assert payload["queue_recommendation"] == "hold_remote_idle_until_execution_contract_promoted"
    assert payload["intent_ready"] is False
    assert payload["execution_contract_status"] == "non_executable_contract"
    assert payload["execution_contract_mode"] == "shadow_only"
    assert payload["execution_contract_reason_codes"] == [
        "spot_remote_lane_missing",
        "portfolio_margin_um_read_only_mode",
        "shadow_executor_only_mode",
    ]


def test_build_remote_intent_queue_accepts_spot_market_but_keeps_shadow_contract_block(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T101000Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "ready_check_scope_market": "spot",
            "identity_brief": "43.153.148.242:spot:aligned:shadow-only",
            "execution_contract_status": "non_executable_contract",
            "execution_contract_brief": "non_executable_contract:spot:spot:shadow_executor_only_mode",
            "execution_contract_mode": "shadow_only",
            "execution_contract_live_orders_allowed": False,
            "execution_contract_executor_mode": "shadow_guarded",
            "execution_contract_executor_mode_source": "bridge_context",
            "execution_contract_reason_codes": ["shadow_executor_only_mode"],
        },
    )
    _write_json(
        review_dir / "20260315T101010Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_remote_scope",
            "scope_router_brief": "review_candidate_inside_remote_scope:ETHUSDT:seed_ticket:spot",
            "preferred_route_symbol": "ETHUSDT",
            "preferred_route_action": "seed_ticket",
            "preferred_route_priority_score": 77,
            "route_recommendation": "seed_remote_intent_queue_from_review_head",
            "blocker_detail": "",
            "done_when": "journal is ready",
        },
    )
    _write_json(review_dir / "20260315T101020Z_live_gate_blocker_report.json", {"blockers": []})
    _write_json(
        review_dir / "20260315T101005Z_cross_market_operator_state.json",
        {
            "review_backlog": [
                {
                    "rank": 1,
                    "area": "crypto_route",
                    "symbol": "ETHUSDT",
                    "action": "seed_ticket",
                    "priority_score": 77,
                    "priority_tier": "review_queue_now",
                    "reason": "ticketable edge",
                    "blocker_detail": "",
                    "done_when": "journal is ready",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101030Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [{"symbol": "ETHUSDT", "allowed": True, "reasons": []}],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:11:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["remote_market"] == "spot"
    assert payload["queue_status"] == "queued_execution_contract_blocked"
    assert payload["execution_contract_executor_mode"] == "shadow_guarded"
    assert payload["execution_contract_executor_mode_source"] == "bridge_context"
    assert payload["execution_contract_reason_codes"] == ["shadow_executor_only_mode"]
    assert payload["preferred_route_symbol"] == "ETHUSDT"


def test_build_remote_intent_queue_marks_guarded_probe_ready_for_spot_probe_only_contract(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T101000Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "ready_check_scope_market": "spot",
            "identity_brief": "43.153.148.242:spot:aligned:probe-only",
            "execution_contract_status": "probe_only_contract",
            "execution_contract_brief": "probe_only_contract:spot:spot:guarded_probe_only_mode",
            "execution_contract_mode": "guarded_probe_only",
            "execution_contract_guarded_probe_allowed": True,
            "execution_contract_live_orders_allowed": False,
            "execution_contract_executor_mode": "spot_live_guarded",
            "execution_contract_executor_mode_source": "bridge_context",
            "execution_contract_reason_codes": ["guarded_probe_only_mode"],
        },
    )
    _write_json(
        review_dir / "20260315T101010Z_remote_scope_router_state.json",
        {
            "status": "ok",
            "scope_router_status": "review_candidate_inside_remote_scope",
            "scope_router_brief": "review_candidate_inside_remote_scope:ETHUSDT:seed_ticket:spot",
            "preferred_route_symbol": "ETHUSDT",
            "preferred_route_action": "seed_ticket",
            "preferred_route_priority_score": 77,
            "route_recommendation": "seed_remote_intent_queue_from_review_head",
            "blocker_detail": "",
            "done_when": "journal is ready",
        },
    )
    _write_json(review_dir / "20260315T101020Z_live_gate_blocker_report.json", {"blockers": []})
    _write_json(
        review_dir / "20260315T101005Z_cross_market_operator_state.json",
        {
            "review_backlog": [
                {
                    "rank": 1,
                    "area": "crypto_route",
                    "symbol": "ETHUSDT",
                    "action": "seed_ticket",
                    "priority_score": 77,
                    "priority_tier": "review_queue_now",
                    "reason": "ticketable edge",
                    "blocker_detail": "",
                    "done_when": "journal is ready",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101030Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [{"symbol": "ETHUSDT", "allowed": True, "reasons": []}],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:11:01Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["queue_status"] == "queued_guarded_probe_ready"
    assert payload["queue_recommendation"] == "run_guarded_probe_from_ticket_ready_intent"
    assert payload["intent_ready"] is True
    assert payload["execution_contract_guarded_probe_allowed"] is True
    assert payload["execution_contract_reason_codes"] == ["guarded_probe_only_mode"]
