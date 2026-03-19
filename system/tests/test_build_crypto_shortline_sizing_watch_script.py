from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_sizing_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_sizing_watch_tracks_min_notional_shortfall(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T005000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T005005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T005010Z_signal_to_order_tickets.json",
        {
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "size_below_min_notional",
                    ],
                    "sizing": {
                        "conviction": 0.1711533,
                        "quote_usdt": 0.06548524035594559,
                        "min_notional_usdt": 5.0,
                        "risk_budget_usdt": 0.0042361921021219,
                        "equity_usdt": 10.30527809,
                        "max_alloc_pct": 0.3,
                    },
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
            "--now",
            "2026-03-16T00:50:20Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "ticket_size_below_min_notional"
    assert (
        payload["watch_decision"]
        == "raise_effective_shortline_size_then_recheck_execution_gate"
    )
    assert payload["size_below_min_notional"] is True
    assert payload["quote_usdt"] == 0.06548524035594559
    assert payload["min_notional_usdt"] == 5.0
    assert payload["size_shortfall_usdt"] > 4.9
    assert payload["size_shortfall_ratio"] > 0.98
    assert payload["blocker_target_artifact"] == "crypto_shortline_sizing_watch"


def test_build_crypto_shortline_sizing_watch_becomes_pattern_aware_for_value_rotation(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T005000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T005005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T005009Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
        },
    )
    _write_json(
        review_dir / "20260316T005010Z_signal_to_order_tickets.json",
        {
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": ["size_below_min_notional"],
                    "sizing": {
                        "conviction": 0.1711533,
                        "quote_usdt": 0.06548524035594559,
                        "min_notional_usdt": 5.0,
                        "risk_budget_usdt": 0.0042361921021219,
                        "equity_usdt": 10.30527809,
                        "max_alloc_pct": 0.3,
                    },
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
            "--now",
            "2026-03-16T00:50:20Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_ticket_size_below_min_notional"
    assert (
        payload["watch_decision"]
        == "raise_value_rotation_effective_size_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "profile_alignment"
    assert payload["artifacts"]["crypto_shortline_pattern_router"] == str(
        review_dir / "20260316T005009Z_crypto_shortline_pattern_router.json"
    )
