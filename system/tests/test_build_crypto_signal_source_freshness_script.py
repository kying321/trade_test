from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_signal_source_freshness.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_signal_source_freshness_marks_route_row_stale(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100100Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T100200Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:02:00Z",
            "signal_source": {
                "kind": "recent5_review",
                "path": "/tmp/20260306_strategy_recent5_signals.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-06",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-02-05",
                    "age_days": 38,
                    "allowed": False,
                    "reasons": ["stale_signal", "confidence_below_threshold"],
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
            "2026-03-15T10:03:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["freshness_status"] == "route_signal_row_stale"
    assert payload["freshness_decision"] == "refresh_crypto_signal_source_then_rebuild_tickets"
    assert payload["refresh_recommended"] is True
    assert payload["route_signal_date"] == "2026-02-05"
    assert payload["route_signal_age_days"] == 38
    assert payload["ticket_signal_source_kind"] == "recent5_review"
    assert payload["ticket_signal_source_artifact_date"] == "2026-03-06"
    assert payload["ticket_signal_source_artifact_age_days"] == 9


def test_build_crypto_signal_source_freshness_marks_route_row_fresh(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "attack",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100100Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "attack",
        },
    )
    _write_json(
        review_dir / "20260315T100200Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:02:00Z",
            "signal_source": {
                "kind": "recent5_review",
                "path": "/tmp/20260315_strategy_recent5_signals.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-14",
                    "age_days": 1,
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
            "2026-03-15T10:03:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["freshness_status"] == "route_signal_row_fresh"
    assert payload["freshness_decision"] == "signal_source_fresh_no_refresh_needed"
    assert payload["refresh_recommended"] is False
