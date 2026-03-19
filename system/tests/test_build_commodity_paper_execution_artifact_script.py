from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_execution_artifact.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_paper_execution_artifact_from_preview_and_ticket_book(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260311T030000Z_commodity_paper_ticket_book.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_mode": "paper_only",
            "actionable_batches": ["metals_all", "precious_metals"],
            "shadow_batches": ["commodities_benchmark"],
            "tickets": [
                {
                    "ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "batch": "metals_all",
                    "symbol": "XAUUSD",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "focus_primary",
                    "batch_priority_rank": 1,
                    "symbol_rank": 1,
                    "leader_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "ticket_note": "gold leader",
                },
                {
                    "ticket_id": "commodity-paper-ticket:metals_all:XAGUSD",
                    "batch": "metals_all",
                    "symbol": "XAGUSD",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "focus_primary",
                    "batch_priority_rank": 1,
                    "symbol_rank": 2,
                    "leader_rank": 2,
                    "regime_gate": "paper_only",
                    "weight_hint": 0.8,
                    "ticket_note": "silver leader",
                },
                {
                    "ticket_id": "commodity-paper-ticket:metals_all:COPPER",
                    "batch": "metals_all",
                    "symbol": "COPPER",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "focus_primary",
                    "batch_priority_rank": 1,
                    "symbol_rank": 3,
                    "leader_rank": 3,
                    "regime_gate": "paper_only",
                    "weight_hint": 0.5,
                    "ticket_note": "copper leader",
                },
                {
                    "ticket_id": "commodity-paper-ticket:precious_metals:XAUUSD",
                    "batch": "precious_metals",
                    "symbol": "XAUUSD",
                    "ticket_role": "leader",
                    "allow_paper_ticket": True,
                    "route_class": "focus_primary",
                    "batch_priority_rank": 2,
                    "symbol_rank": 1,
                    "leader_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 0.7,
                    "ticket_note": "secondary batch",
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260311T030020Z_commodity_paper_execution_preview.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_mode": "paper_only",
            "preview_ready_batches": ["metals_all", "precious_metals"],
            "shadow_only_batches": ["commodities_benchmark"],
            "preview_batch_count": 3,
            "next_execution_batch": "metals_all",
            "next_execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "next_execution_regime_gate": "paper_only",
            "next_execution_weight_hint_sum": 2.3,
            "preview_stack_brief": "paper-execution-ready:metals_all,precious_metals | shadow:commodities_benchmark",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-11T03:05:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_artifact_status"] == "paper-execution-artifact-ready"
    assert payload["execution_batch"] == "metals_all"
    assert payload["execution_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["execution_ticket_ids"] == [
        "commodity-paper-ticket:metals_all:XAUUSD",
        "commodity-paper-ticket:metals_all:XAGUSD",
        "commodity-paper-ticket:metals_all:COPPER",
    ]
    assert payload["execution_regime_gate"] == "paper_only"
    assert payload["execution_weight_hint_sum"] == 2.3
    assert payload["execution_item_count"] == 3
    assert payload["actionable_execution_item_count"] == 3
    assert payload["execution_stack_brief"] == "paper-execution-artifact:metals_all:XAUUSD, XAGUSD, COPPER"
    assert [row["symbol"] for row in payload["execution_items"]] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["execution_items"][0]["execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["execution_items"][0]["allow_proxy_price_reference_execution"] is True
    assert payload["execution_items"][0]["execution_price_normalization_mode"] == "paper_proxy_reference"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
