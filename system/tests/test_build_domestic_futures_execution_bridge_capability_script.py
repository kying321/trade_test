from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_domestic_futures_execution_bridge_capability.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_domestic_futures_execution_bridge_capability_creates_manual_only_artifact(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260328T010000Z_brooks_price_action_execution_plan.json",
        {
            "status": "ok",
            "as_of": "2026-03-28T01:00:00Z",
            "plan_items": [
                {
                    "symbol": "SC2603",
                    "asset_class": "future",
                    "direction": "LONG",
                    "strategy_id": "brooks_structure",
                    "signal_ts": "2026-03-28T00:30:00Z",
                    "signal_age_bars": 1,
                    "route_selection_score": 96.0,
                    "signal_score": 92,
                    "execution_action": "review_manual_stop_entry",
                    "plan_status": "manual_structure_review_now",
                    "plan_blocker_detail": (
                        "Structure route is valid, but this asset class has no automated execution bridge in-system."
                    ),
                    "plan_done_when": (
                        "manual trader confirms venue, sizing, and lower-timeframe trigger before placing a discretionary order"
                    ),
                    "route_bridge_status": "manual_structure_route",
                },
                {
                    "symbol": "BTCUSDT",
                    "asset_class": "crypto",
                    "execution_action": "watch_only",
                    "plan_status": "informational_only",
                },
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
            "2026-03-28T09:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["capability_count"] == 1
    assert payload["bridge_stage_counts"]["manual_only"] == 1
    assert payload["head_symbol"] == "SC2603"
    assert payload["source_execution_plan_artifact"].endswith("_brooks_price_action_execution_plan.json")

    row = payload["capabilities"][0]
    assert row["symbol"] == "SC2603"
    assert row["asset_class"] == "future"
    assert row["bridge_stage"] == "manual_only"
    assert row["account_scope"] == "manual"
    assert row["adapter_kind"] == "manual"
    assert row["allowed_actions"] == ["review_manual_stop_entry", "queue_review", "source_refresh"]
    assert row["blocked_actions"] == ["auto_route", "auto_send", "live_promotion"]
    assert row["blocker_code"] == "no_automated_execution_bridge"
    assert row["execution_truth_source"] == "manual_confirmation"
    assert row["consumer_mappings"]["plan_status"] == "manual_structure_review_now"
    assert row["consumer_mappings"]["execution_action"] == "review_manual_stop_entry"
    assert row["consumer_mappings"]["route_bridge_status"] == "manual_structure_route"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
