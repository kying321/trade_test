from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bridge_commodity_paper_execution_queue.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_bridge_commodity_queue_allows_empty_queue_without_signal_tickets(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "",
            "execution_symbols": [],
            "queued_items": [],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-10T12:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bridge_status"] == "bridge_empty"
    assert payload["ready_count"] == 0
    assert payload["blocked_count"] == 0
    assert payload["signal_missing_count"] == 0
    assert payload["bridge_items"] == []
