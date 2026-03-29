from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_commodity_paper_ticket_book.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_paper_ticket_book_builds_bu2606_symbol_ticket(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260328T024000Z_commodity_paper_ticket_lane.json",
        {
            "status": "ok",
            "as_of": "2026-03-28T02:40:00Z",
            "route_status": "paper-first",
            "ticket_status": "paper-ready",
            "ticket_stack_brief": "paper-ready:asphalt_cn",
            "next_ticket_batch": "asphalt_cn",
            "next_ticket_symbols": ["BU2606"],
            "tickets": [
                {
                    "ticket_id": "commodity-paper:asphalt_cn",
                    "batch": "asphalt_cn",
                    "route_class": "focus_primary",
                    "priority_rank": 1,
                    "symbols": ["BU2606"],
                    "leader_symbols": ["BU2606"],
                    "allow_paper_ticket": True,
                    "allow_live_ticket": False,
                    "paper_execution_mode": "paper_only",
                    "regime_gate": "paper_only",
                    "stage": "paper_ticket_lane",
                    "ticket_status": "paper_ready",
                    "ticket_note": "Primary domestic futures sleeve. Route into paper tickets first; keep live disabled.",
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
            "2026-03-28T10:45:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_book_status"] == "paper-ready"
    assert payload["execution_mode"] == "paper_only"
    assert payload["ticket_count"] == 1
    assert payload["actionable_ticket_count"] == 1
    assert payload["next_ticket_id"] == "commodity-paper-ticket:asphalt_cn:BU2606"
    assert payload["next_ticket_batch"] == "asphalt_cn"
    assert payload["next_ticket_symbol"] == "BU2606"
    assert payload["next_ticket_regime_gate"] == "paper_only"
    assert payload["next_ticket_weight_hint"] == 1.0
    ticket = payload["tickets"][0]
    assert ticket["ticket_id"] == "commodity-paper-ticket:asphalt_cn:BU2606"
    assert ticket["symbol"] == "BU2606"
    assert ticket["ticket_role"] == "leader"
    assert ticket["allow_paper_ticket"] is True
    assert ticket["allow_live_ticket"] is False
    assert ticket["ticket_status"] == "paper_ready"
