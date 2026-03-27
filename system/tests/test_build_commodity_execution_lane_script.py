from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_commodity_execution_lane.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_execution_lane_falls_back_to_blocker_report(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T105015Z_live_gate_blocker_report.json",
        {
            "status": "ok",
            "commodity_execution_path": {
                "design_status": "proposed",
                "execution_mode": "paper_first",
                "focus_primary_batches": ["metals_all", "precious_metals"],
                "focus_with_regime_filter_batches": ["energy_liquids"],
                "shadow_only_batches": ["commodities_benchmark"],
                "leader_symbols_primary": ["XAGUSD", "COPPER", "XAUUSD"],
                "leader_symbols_regime_filter": ["BRENTUSD", "WTIUSD"],
                "stages": [
                    {"stage": "paper_ticket_lane", "batches": ["metals_all", "precious_metals", "energy_liquids"]},
                ],
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
            "2026-03-10T18:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_mode"] == "blocker-report-commodity-path"
    assert payload["route_status"] == "paper-first"
    assert payload["focus_primary_batches"] == ["metals_all", "precious_metals"]
    assert payload["focus_with_regime_filter_batches"] == ["energy_liquids"]
    assert payload["next_focus_batch"] == "metals_all"
    assert payload["next_focus_symbols"] == ["XAGUSD", "COPPER", "XAUUSD"]


def test_build_commodity_execution_lane_accepts_inactive_hot_universe_research(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T125700Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["metals_all"],
            },
            "regime_playbook": {
                "batch_rules": [
                    {
                        "batch": "metals_all",
                        "leader_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
                    }
                ]
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
            "2026-03-10T18:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_mode"] == "hot-universe-action-ladder"
    assert payload["design_status"] == "inactive"
    assert payload["route_status"] == "paper-first"
    assert payload["focus_primary_batches"] == []
    assert payload["focus_with_regime_filter_batches"] == []
    assert payload["shadow_only_batches"] == ["metals_all"]
    assert payload["avoid_batches"] == ["metals_all"]
    assert payload["next_focus_batch"] == "metals_all"
    assert payload["next_focus_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
