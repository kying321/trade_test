from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_brooks_price_action_execution_plan.py"
)


def test_builds_execution_plan_from_route_report(tmp_path: Path) -> None:
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
                "strategy_label": "Second-Entry Trend Continuation",
                "direction": "LONG",
                "signal_ts": "2026-02-11T00:00:00Z",
                "signal_age_bars": 2,
                "route_selection_score": 81.68,
                "signal_score": 80,
                "entry_price": 462.0,
                "stop_price": 455.0,
                "target_price": 477.4,
                "risk_per_unit": 7.0,
                "rr_ratio": 2.2,
                "route_bridge_status": "manual_structure_route",
                "route_bridge_blocker_detail": "manual only",
                "route_bridge_done_when": "review manually",
                "note": "strong trend",
            },
            {
                "symbol": "BTCUSDT",
                "asset_class": "crypto",
                "strategy_id": "trend_pullback_continuation",
                "strategy_label": "Trend Pullback Continuation",
                "direction": "LONG",
                "signal_ts": "2026-02-11T00:00:00Z",
                "signal_age_bars": 1,
                "route_selection_score": 60.0,
                "signal_score": 75,
                "entry_price": 70000.0,
                "stop_price": 69000.0,
                "target_price": 72000.0,
                "risk_per_unit": 1000.0,
                "rr_ratio": 2.0,
                "route_bridge_status": "blocked_shortline_gate",
                "route_bridge_blocker_detail": "Bias_Only",
                "route_bridge_done_when": "complete shortline gate stack",
                "note": "waiting micro gate",
            },
        ],
    }
    (review_dir / "20260313T050000Z_brooks_price_action_route_report.json").write_text(
        json.dumps(route_payload, ensure_ascii=False, indent=2) + "\n",
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
    assert payload["actionable_count"] == 1
    assert payload["blocked_count"] == 1
    assert payload["head_plan_item"]["symbol"] == "SC2603"
    assert payload["head_plan_item"]["plan_status"] == "manual_structure_review_now"
    items = {row["symbol"]: row for row in payload["plan_items"]}
    assert items["BTCUSDT"]["plan_status"] == "blocked_shortline_gate"
    assert "artifact" in payload and Path(str(payload["artifact"])).exists()
    assert "markdown" in payload and Path(str(payload["markdown"])).exists()
    assert "checksum" in payload and Path(str(payload["checksum"])).exists()
