from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_price_reference_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_price_reference_watch_tracks_missing_template_proxy_only(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T004000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T004005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T004010Z_signal_to_order_tickets.json",
        {
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "proxy_price_reference_only",
                    ],
                    "levels": {
                        "entry_price": 0.0,
                        "stop_price": 0.0,
                        "target_price": 0.0,
                    },
                    "signal": {
                        "price_reference_kind": "shortline_missing_price_template",
                        "price_reference_source": "crypto_shortline_execution_gate",
                        "execution_price_ready": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T004015Z_crypto_shortline_signal_source.json",
        {
            "signals": {
                "SOLUSDT": [
                    {
                        "symbol": "SOLUSDT",
                        "entry_price": 0.0,
                        "stop_price": 0.0,
                        "target_price": 0.0,
                        "price_reference_kind": "shortline_missing_price_template",
                        "price_reference_source": "crypto_shortline_execution_gate",
                        "execution_price_ready": False,
                    }
                ]
            }
        },
    )
    _write_json(
        review_dir / "20260316T004020Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_route_not_setup_ready_proxy_price_blocked:SOLUSDT:wait_for_setup_ready_and_executable_price_reference:portfolio_margin_um",
            "diagnosis_decision": "wait_for_setup_ready_and_executable_price_reference",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T00:40:37Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "price_reference_missing_template_proxy_only"
    assert payload["watch_decision"] == "build_price_template_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_price_reference_watch"
    assert payload["execution_price_ready"] is False
    assert payload["missing_level_fields"] == ["entry_price", "stop_price", "target_price"]
    assert payload["price_reference_kind"] == "shortline_missing_price_template"
