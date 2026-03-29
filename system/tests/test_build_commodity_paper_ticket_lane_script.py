from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_commodity_paper_ticket_lane.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_commodity_paper_ticket_lane_builds_bu2606_paper_ticket(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260328T022000Z_commodity_execution_lane.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "focus_primary_batches": ["asphalt_cn"],
            "focus_with_regime_filter_batches": [],
            "shadow_only_batches": [],
            "leader_symbols_primary": ["BU2606"],
            "leader_symbols_regime_filter": [],
            "next_focus_batch": "asphalt_cn",
            "next_focus_symbols": ["BU2606"],
            "route_stack_brief": "paper-primary:asphalt_cn",
        },
    )
    _write_json(
        review_dir / "20260328T021000Z_hot_research_universe.json",
        {
            "status": "ok",
            "domestic_futures": {
                "selected": ["BU2606"],
                "count": 1,
                "batches": ["asphalt_cn"],
            },
            "batches": {
                "asphalt_cn": ["BU2606"],
                "domestic_futures_cn": ["BU2606"],
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
            "2026-03-28T10:40:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_status"] == "paper-ready"
    assert payload["execution_mode"] == "paper_only"
    assert payload["commodity_focus_batch"] == "asphalt_cn"
    assert payload["commodity_focus_symbols"] == ["BU2606"]
    assert payload["next_ticket_batch"] == "asphalt_cn"
    assert payload["next_ticket_symbols"] == ["BU2606"]
    assert payload["paper_ready_batches"] == ["asphalt_cn"]
    assert payload["shadow_only_batches"] == []
    ticket = payload["tickets"][0]
    assert ticket["ticket_id"] == "commodity-paper:asphalt_cn"
    assert ticket["batch"] == "asphalt_cn"
    assert ticket["symbols"] == ["BU2606"]
    assert ticket["leader_symbols"] == ["BU2606"]
    assert ticket["allow_paper_ticket"] is True
    assert ticket["allow_live_ticket"] is False
    assert ticket["ticket_status"] == "paper_ready"
    assert ticket["regime_gate"] == "paper_only"
