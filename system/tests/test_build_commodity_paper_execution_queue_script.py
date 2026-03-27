from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_commodity_paper_execution_queue.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_paper_execution_queue_creates_queue_artifact(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T192000Z_commodity_paper_execution_artifact.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "execution_item_count": 3,
            "actionable_execution_item_count": 3,
            "execution_stack_brief": "paper-execution-artifact:metals_all:XAUUSD, XAGUSD, COPPER",
            "execution_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "execution_status": "planned",
                    "allow_paper_execution": True,
                    "allow_proxy_price_reference_execution": True,
                    "execution_price_normalization_mode": "paper_proxy_reference",
                    "weight_hint": 1.0,
                    "regime_gate": "paper_only",
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "symbol": "XAGUSD",
                    "execution_status": "planned",
                    "allow_paper_execution": True,
                    "allow_proxy_price_reference_execution": True,
                    "execution_price_normalization_mode": "paper_proxy_reference",
                    "weight_hint": 0.8,
                    "regime_gate": "paper_only",
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:COPPER",
                    "symbol": "COPPER",
                    "execution_status": "planned",
                    "allow_paper_execution": True,
                    "allow_proxy_price_reference_execution": True,
                    "execution_price_normalization_mode": "paper_proxy_reference",
                    "weight_hint": 0.5,
                    "regime_gate": "paper_only",
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
            "2026-03-10T20:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_queue_status"] == "paper-execution-queued"
    assert payload["execution_batch"] == "metals_all"
    assert payload["next_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_execution_symbol"] == "XAUUSD"
    assert payload["queue_depth"] == 3
    assert payload["actionable_queue_depth"] == 3
    assert payload["queue_stack_brief"] == "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER"
    assert payload["queued_items"][0]["allow_proxy_price_reference_execution"] is True
    assert payload["queued_items"][0]["execution_price_normalization_mode"] == "paper_proxy_reference"
    assert Path(str(payload["artifact"])).exists()
