from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_ticket_book.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_paper_ticket_book_from_lane(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260311T022000Z_commodity_paper_ticket_lane.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_status": "paper-ready",
            "ticket_stack_brief": "paper-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
            "paper_ready_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "next_ticket_batch": "metals_all",
            "next_ticket_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
            "tickets": [
                {
                    "ticket_id": "commodity-paper:metals_all",
                    "batch": "metals_all",
                    "route_class": "focus_primary",
                    "priority_rank": 1,
                    "allow_paper_ticket": True,
                    "regime_gate": "paper_only",
                    "ticket_note": "Primary commodity sleeve.",
                    "symbols": ["XAUUSD", "XAGUSD", "COPPER"],
                    "leader_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
                },
                {
                    "ticket_id": "commodity-paper:commodities_benchmark",
                    "batch": "commodities_benchmark",
                    "route_class": "shadow_only",
                    "priority_rank": 4,
                    "allow_paper_ticket": False,
                    "regime_gate": "shadow_only_no_allocation",
                    "ticket_note": "Shadow only.",
                    "symbols": ["XAUUSD", "XAGUSD"],
                    "leader_symbols": [],
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
            "2026-03-11T02:30:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_book_status"] == "paper-ready"
    assert payload["actionable_batches"] == ["metals_all"]
    assert payload["shadow_batches"] == ["commodities_benchmark"]
    assert payload["next_ticket_id"] == "commodity-paper-ticket:metals_all:XAUUSD"
    assert payload["next_ticket_symbol"] == "XAUUSD"
    assert payload["actionable_ticket_count"] == 3
    assert payload["shadow_ticket_count"] == 2
    assert payload["ticket_book_stack_brief"] == "paper-ready:metals_all | shadow:commodities_benchmark"
