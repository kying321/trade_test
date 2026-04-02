from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_hot_universe_operator_brief.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_hot_universe_operator_brief_prefers_non_dry_run(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T100000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all", "precious_metals"],
                "research_queue_batches": ["crypto_hot"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "next_focus_symbol": "BNBUSDT",
                "next_retest_action": "rerun_bnb_native_long_window",
            },
        },
    )
    _write_json(
        review_dir / "20260310T110000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {"focus_primary_batches": []},
            "crypto_route_brief": {"operator_status": "watch-all"},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "ok"
    assert payload["source_mode"] == "single-hot-universe-source"
    assert payload["focus_primary_batches"] == ["metals_all", "precious_metals"]
    assert payload["research_queue_batches"] == ["crypto_hot"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"
    assert payload["crypto_next_retest_action"] == "rerun_bnb_native_long_window"
    assert "primary: metals_all, precious_metals" in payload["summary_text"]


def test_build_hot_universe_operator_brief_falls_back_to_rich_dry_run(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T100000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"focus_primary_batches": []},
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260310T110000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "research_queue_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "next_focus_symbol": "BNBUSDT",
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "next_focus_symbol": "BNBUSDT",
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
            "2026-03-10T12:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "dry_run"
    assert payload["focus_primary_batches"] == ["metals_all"]
    assert payload["research_queue_batches"] == ["crypto_majors"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"


def test_build_hot_universe_operator_brief_surfaces_domestic_futures_bridge_capability(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260328T010000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"focus_primary_batches": ["metals_all"]},
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260328T011000Z_domestic_futures_execution_bridge_capability.json",
        {
            "status": "ok",
            "as_of": "2026-03-28T01:10:00Z",
            "head_symbol": "SC2603",
            "bridge_stage_counts": {
                "research_only": 0,
                "paper_only": 0,
                "manual_only": 1,
                "guarded_canary": 0,
                "executable": 0,
            },
            "capabilities": [
                {
                    "symbol": "SC2603",
                    "bridge_stage": "manual_only",
                    "blocker_code": "no_automated_execution_bridge",
                    "blocker_detail": "Structure route is valid, but this asset class has no automated execution bridge in-system.",
                    "execution_truth_source": "manual_confirmation",
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
            "2026-03-28T01:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_domestic_futures_execution_bridge_capability_artifact"].endswith(
        "_domestic_futures_execution_bridge_capability.json"
    )
    assert payload["source_domestic_futures_execution_bridge_capability_status"] == "ok"
    assert payload["source_domestic_futures_execution_bridge_capability_head_symbol"] == "SC2603"
    assert payload["source_domestic_futures_execution_bridge_capability_head_stage"] == "manual_only"
    assert (
        payload["source_domestic_futures_execution_bridge_capability_head_blocker_code"]
        == "no_automated_execution_bridge"
    )
    assert payload["source_domestic_futures_execution_bridge_capability_head_truth_source"] == "manual_confirmation"


def test_build_hot_universe_operator_brief_applies_capability_gate_to_brooks_lane(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260328T010000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"focus_primary_batches": ["metals_all"]},
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260328T011100Z_brooks_price_action_execution_plan.json",
        {
            "status": "ok",
            "as_of": "2026-03-28T01:11:00Z",
            "plan_items": [
                {
                    "symbol": "SC2603",
                    "asset_class": "future",
                    "strategy_id": "brooks_structure",
                    "direction": "LONG",
                    "execution_action": "review_manual_stop_entry",
                    "plan_status": "manual_structure_review_now",
                    "route_selection_score": 96.0,
                    "signal_score": 92,
                    "signal_age_bars": 1,
                    "plan_blocker_detail": "manual bridge missing",
                    "plan_done_when": "manual trader confirms trigger",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260328T011200Z_brooks_structure_review_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-28T01:12:00Z",
            "review_status": "ready",
            "review_brief": "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now",
            "queue_status": "ready",
            "queue_count": 1,
            "priority_status": "ready",
            "priority_brief": "ready:SC2603:96:review_queue_now",
            "blocker_detail": "manual bridge missing",
            "done_when": "manual trader confirms trigger",
            "head": {
                "symbol": "SC2603",
                "strategy_id": "brooks_structure",
                "direction": "LONG",
                "execution_action": "review_manual_stop_entry",
                "plan_status": "manual_structure_review_now",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "blocker_detail": "manual bridge missing",
                "done_when": "manual trader confirms trigger",
            },
            "queue": [
                {
                    "rank": 1,
                    "symbol": "SC2603",
                    "strategy_id": "brooks_structure",
                    "direction": "LONG",
                    "execution_action": "review_manual_stop_entry",
                    "plan_status": "manual_structure_review_now",
                    "priority_score": 96,
                    "priority_tier": "review_queue_now",
                    "blocker_detail": "manual bridge missing",
                    "done_when": "manual trader confirms trigger",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260328T011300Z_domestic_futures_execution_bridge_capability.json",
        {
            "status": "ok",
            "as_of": "2026-03-28T01:13:00Z",
            "head_symbol": "SC2603",
            "capabilities": [
                {
                    "symbol": "SC2603",
                    "bridge_stage": "manual_only",
                    "blocker_code": "no_automated_execution_bridge",
                    "blocker_detail": "Structure route is valid, but this asset class has no automated execution bridge in-system.",
                    "execution_truth_source": "manual_confirmation",
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
            "2026-03-28T01:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "no_automated_execution_bridge" in payload["brooks_structure_review_head_blocker_detail"]
    assert "manual_only" in payload["brooks_structure_review_head_blocker_detail"]
    assert payload["brooks_structure_review_head_done_when"] == (
        "automated execution bridge becomes explicit before promotion"
    )
    assert "no_automated_execution_bridge" in payload["brooks_structure_operator_blocker_detail"]
