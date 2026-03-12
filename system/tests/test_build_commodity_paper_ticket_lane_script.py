from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_ticket_lane.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_paper_ticket_lane_maps_batches_to_symbols(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260311T020000Z_commodity_execution_lane.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "route_stack_brief": "paper-primary:metals_all,precious_metals | regime-filter:energy_liquids | shadow:commodities_benchmark",
            "focus_primary_batches": ["metals_all", "precious_metals"],
            "focus_with_regime_filter_batches": ["energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "leader_symbols_primary": ["XAGUSD", "COPPER", "XAUUSD"],
            "leader_symbols_regime_filter": ["BRENTUSD", "WTIUSD"],
            "next_focus_batch": "metals_all",
            "next_focus_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
        },
    )
    _write_json(
        review_dir / "20260310T144800Z_hot_research_universe.json",
        {
            "batches": {
                "metals_all": ["XAUUSD", "XAGUSD", "COPPER"],
                "precious_metals": ["XAUUSD", "XAGUSD"],
                "energy_liquids": ["WTIUSD", "BRENTUSD"],
                "commodities_benchmark": ["XAUUSD", "XAGUSD", "WTIUSD", "BRENTUSD", "NATGAS", "COPPER"],
            }
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-11T02:10:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_status"] == "paper-ready"
    assert payload["paper_ready_batches"] == ["metals_all", "precious_metals", "energy_liquids"]
    assert payload["shadow_only_batches"] == ["commodities_benchmark"]
    assert payload["next_ticket_batch"] == "metals_all"
    assert payload["next_ticket_symbols"] == ["XAGUSD", "COPPER", "XAUUSD"]
    assert len(payload["tickets"]) == 4
    assert payload["tickets"][0]["ticket_id"] == "commodity-paper:metals_all"
    assert payload["tickets"][0]["symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["tickets"][2]["regime_gate"] == "strong_trend_only"
    assert payload["tickets"][3]["allow_paper_ticket"] is False
