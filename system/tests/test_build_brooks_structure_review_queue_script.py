from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_brooks_structure_review_queue.py"
)


def test_builds_priority_queue_from_execution_plan_and_route_report(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    route_payload = {
        "action": "build_brooks_price_action_route_report",
        "ok": True,
        "status": "ok",
        "current_candidates": [
            {
                "symbol": "SC2603",
                "asset_class": "future",
                "strategy_id": "second_entry_trend_continuation",
                "direction": "LONG",
                "signal_age_bars": 2,
                "route_selection_score": 81.68,
                "signal_score": 80,
                "route_bridge_status": "manual_structure_route",
                "route_bridge_blocker_detail": "manual only",
            },
            {
                "symbol": "AU2406",
                "asset_class": "future",
                "strategy_id": "breakout_pullback_resume",
                "direction": "SHORT",
                "signal_age_bars": 4,
                "route_selection_score": 64.125,
                "signal_score": 61,
                "route_bridge_status": "blocked_shortline_gate",
                "route_bridge_blocker_detail": "lower timeframe confirmation still missing",
            },
            {
                "symbol": "SPY",
                "asset_class": "etf",
                "strategy_id": "three_push_climax_reversal",
                "direction": "LONG",
                "signal_age_bars": 1,
                "route_selection_score": 55.0,
                "signal_score": 58,
                "route_bridge_status": "informational_only",
                "route_bridge_blocker_detail": "await execution plan promotion",
            },
        ],
    }
    execution_payload = {
        "action": "build_brooks_price_action_execution_plan",
        "ok": True,
        "status": "ok",
        "plan_items": [
            {
                "symbol": "SC2603",
                "asset_class": "future",
                "direction": "LONG",
                "strategy_id": "second_entry_trend_continuation",
                "plan_status": "manual_structure_review_now",
                "execution_action": "review_manual_stop_entry",
                "route_selection_score": 81.68,
                "signal_score": 80,
                "signal_age_bars": 2,
                "plan_blocker_detail": "manual execution bridge only",
                "plan_done_when": "manual trader confirms venue and sizing",
            },
            {
                "symbol": "AU2406",
                "asset_class": "future",
                "direction": "SHORT",
                "strategy_id": "breakout_pullback_resume",
                "plan_status": "blocked_shortline_gate",
                "execution_action": "wait_for_shortline_setup_ready",
                "route_selection_score": 64.125,
                "signal_score": 61,
                "signal_age_bars": 4,
                "plan_blocker_detail": "lower timeframe confirmation still missing",
                "plan_done_when": "shortline gate stack completes",
            },
        ],
    }
    (review_dir / "20260313T050000Z_brooks_price_action_route_report.json").write_text(
        json.dumps(route_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260313T050100Z_brooks_price_action_execution_plan.json").write_text(
        json.dumps(execution_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-13T05:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["review_status"] == "ready"
    assert payload["queue_status"] == "ready"
    assert payload["queue_count"] == 3
    assert payload["priority_status"] == "ready"
    assert payload["priority_brief"].startswith("ready:SC2603:96:")
    assert payload["head"]["symbol"] == "SC2603"
    assert payload["head"]["priority_score"] == 96
    assert payload["head"]["priority_tier"] == "review_queue_now"
    assert payload["queue"][1]["symbol"] == "AU2406"
    assert payload["queue"][1]["priority_score"] == 53
    assert payload["queue"][1]["priority_tier"] == "blocked_review"
    assert payload["queue"][2]["symbol"] == "SPY"
    assert payload["queue"][2]["plan_status"] == "route_candidate_only"
    assert payload["queue"][2]["priority_tier"] == "route_candidate_only"
    assert "artifact" in payload and Path(str(payload["artifact"])).exists()
    assert "markdown" in payload and Path(str(payload["markdown"])).exists()
    assert "checksum" in payload and Path(str(payload["checksum"])).exists()
