from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_execution_preview.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_paper_execution_preview_from_ticket_book(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260311T023000Z_commodity_paper_ticket_book.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_mode": "paper_only",
            "actionable_batches": ["metals_all"],
            "shadow_batches": ["commodities_benchmark"],
            "tickets": [
                {
                    "ticket_id": "commodity-paper-ticket:energy_liquids:WTIUSD",
                    "batch": "energy_liquids",
                    "symbol": "WTIUSD",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "regime_filter",
                    "regime_gate": "strong_trend_only",
                    "weight_hint": 0.7,
                },
                {
                    "ticket_id": "commodity-paper-ticket:energy_liquids:BRENTUSD",
                    "batch": "energy_liquids",
                    "symbol": "BRENTUSD",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "regime_filter",
                    "regime_gate": "strong_trend_only",
                    "weight_hint": 0.5,
                },
                {
                    "ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "batch": "metals_all",
                    "symbol": "XAUUSD",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "focus_primary",
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                },
                {
                    "ticket_id": "commodity-paper-ticket:metals_all:XAGUSD",
                    "batch": "metals_all",
                    "symbol": "XAGUSD",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "focus_primary",
                    "regime_gate": "paper_only",
                    "weight_hint": 0.8,
                },
                {
                    "ticket_id": "commodity-paper-ticket:commodities_benchmark:XAUUSD",
                    "batch": "commodities_benchmark",
                    "symbol": "XAUUSD",
                    "ticket_role": "shadow",
                    "allow_paper_ticket": False,
                    "route_class": "shadow_only",
                    "regime_gate": "shadow_only_no_allocation",
                    "weight_hint": 0.4,
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
            "2026-03-11T02:40:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_preview_status"] == "paper-execution-ready"
    assert payload["preview_ready_batches"] == ["metals_all", "energy_liquids"]
    assert payload["shadow_only_batches"] == ["commodities_benchmark"]
    assert payload["next_execution_batch"] == "metals_all"
    assert payload["next_execution_symbols"] == ["XAUUSD", "XAGUSD"]
    assert payload["next_execution_ticket_ids"] == [
        "commodity-paper-ticket:metals_all:XAUUSD",
        "commodity-paper-ticket:metals_all:XAGUSD",
    ]
    assert payload["next_execution_regime_gate"] == "paper_only"
    assert payload["preview_stack_brief"] == "paper-execution-ready:metals_all,energy_liquids | shadow:commodities_benchmark"
    assert [row["batch"] for row in payload["preview_batches"]] == [
        "metals_all",
        "energy_liquids",
        "commodities_benchmark",
    ]
