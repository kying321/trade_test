from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bridge_commodity_paper_execution_queue.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_module():
    spec = importlib.util.spec_from_file_location("bridge_commodity_paper_execution_queue_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_bridge_commodity_queue_degrades_when_signal_builder_is_missing(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "asphalt_cn",
            "execution_symbols": ["BU2606"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
                    "source_ticket_id": "commodity-paper-ticket:asphalt_cn:BU2606",
                    "batch": "asphalt_cn",
                    "symbol": "BU2606",
                    "execution_status": "planned",
                    "allow_paper_execution": True,
                    "allow_live_execution": False,
                    "allow_proxy_price_reference_execution": False,
                    "execution_price_normalization_mode": "",
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "execution_note": "paper only",
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
    assert payload["bridge_status"] == "blocked_missing_directional_signal"
    assert payload["ready_count"] == 0
    assert payload["blocked_count"] == 1
    assert payload["signal_missing_count"] == 1
    assert payload["source_commodity_signal_artifact"] == ""
    assert Path(str(payload["source_signal_tickets_artifact"])).exists()
    assert payload["bridge_items"][0]["symbol"] == "BU2606"
    assert payload["bridge_items"][0]["bridge_status"] == "blocked_missing_directional_signal"


def test_build_signal_tickets_prefers_builder_direct_signal_tickets_output(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    signal_json = review_dir / "commodity_directional_signals.json"
    signal_json.write_text("{}", encoding="utf-8")
    tickets_json = review_dir / "signal_to_order_tickets.json"
    tickets_json.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-10T12:00:00Z",
                "as_of": "2026-03-10",
                "symbols": ["BU2606"],
                "tickets": [{"symbol": "BU2606"}],
                "summary": {"ticket_count": 1},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mod,
        "build_commodity_directional_signals",
        lambda **kwargs: (signal_json, tickets_json),
    )

    signal_tickets_path, commodity_signal_path = mod.build_signal_tickets(
        review_dir=review_dir,
        output_root=output_root,
        as_of=dt.date(2026, 3, 10),
        symbols=["BU2606"],
    )

    assert signal_tickets_path == tickets_json.resolve()
    assert commodity_signal_path == signal_json.resolve()
