from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_execution_review.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in payloads) + "\n",
        encoding="utf-8",
    )


def _create_plan_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE trade_plans(
                symbol TEXT,
                side TEXT,
                size_pct REAL,
                risk_pct REAL,
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                status TEXT,
                date TEXT,
                bridge_execution_id TEXT,
                bridge_idempotency_key TEXT,
                source_ticket_id TEXT,
                runtime_mode TEXT,
                execution_price_normalization_mode TEXT,
                paper_proxy_price_normalized REAL,
                signal_price_reference_source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE executed_plans(
                date TEXT,
                open_date TEXT,
                symbol TEXT,
                side TEXT,
                direction TEXT,
                runtime_mode TEXT,
                mode TEXT,
                size_pct REAL,
                risk_pct REAL,
                entry_price REAL,
                status TEXT,
                bridge_execution_id TEXT,
                bridge_idempotency_key TEXT,
                source_ticket_id TEXT,
                execution_price_normalization_mode TEXT,
                paper_proxy_price_normalized REAL,
                signal_price_reference_source TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO trade_plans VALUES
            ('XAUUSD','LONG',1.54,0.10,5198.1,4847.8,5758.58,'OPEN','2026-03-11',
             'commodity-paper-execution:metals_all:XAUUSD',
             'bridge-key-xau',
             'commodity-paper-ticket:metals_all:XAUUSD',
             'commodity_queue_bridge',
             'paper_proxy_reference',
             1.0,
             'yfinance:GC=F')
            """
        )
        conn.execute(
            """
            INSERT INTO executed_plans VALUES
            ('2026-03-11','2026-03-11','XAUUSD','LONG','LONG','commodity_queue_bridge','paper',
             1.54,0.10,5198.1,'OPEN',
             'commodity-paper-execution:metals_all:XAUUSD',
             'bridge-key-xau',
             'commodity-paper-ticket:metals_all:XAUUSD',
             'paper_proxy_reference',
             1.0,
             'yfinance:GC=F')
            """
        )
        conn.commit()


def test_build_commodity_paper_execution_review_creates_review_artifact(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
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
            "queue_depth": 3,
            "actionable_queue_depth": 3,
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                    "weight_hint": 1.0,
                    "regime_gate": "paper_only",
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "symbol": "XAGUSD",
                    "queue_rank": 2,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                    "weight_hint": 0.8,
                    "regime_gate": "paper_only",
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:COPPER",
                    "symbol": "COPPER",
                    "queue_rank": 3,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                    "weight_hint": 0.5,
                    "regime_gate": "paper_only",
                },
            ],
        },
    )
    _write_jsonl(
        tmp_path / "logs" / "paper_execution_ledger.jsonl",
        [
            {"domain": "paper_execution", "symbol": "XAUUSD"},
            {"domain": "paper_execution", "symbol": "XAGUSD"},
            {"domain": "paper_execution", "symbol": "COPPER"},
        ],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_review_status"] == "paper-execution-review-pending"
    assert payload["execution_batch"] == "metals_all"
    assert payload["next_review_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_review_execution_symbol"] == "XAUUSD"
    assert payload["review_item_count"] == 3
    assert payload["actionable_review_item_count"] == 3
    assert payload["review_stack_brief"] == "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
    assert payload["fill_evidence_pending_count"] == 0
    assert payload["review_pending_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["fill_evidence_pending_symbols"] == []
    assert all(item["paper_execution_evidence_present"] for item in payload["review_items"])


def test_build_commodity_paper_execution_review_marks_missing_fill_evidence(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD"],
            "execution_ticket_ids": ["commodity-paper-ticket:metals_all:XAUUSD"],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 1.0,
            "execution_item_count": 1,
            "actionable_execution_item_count": 1,
            "queue_depth": 1,
            "actionable_queue_depth": 1,
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD",
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                    "weight_hint": 1.0,
                    "regime_gate": "paper_only",
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
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_review_status"] == "paper-execution-awaiting-fill-evidence"
    assert payload["actionable_review_item_count"] == 0
    assert payload["fill_evidence_pending_count"] == 1
    assert payload["review_pending_symbols"] == []
    assert payload["fill_evidence_pending_symbols"] == ["XAUUSD"]
    assert payload["next_review_execution_id"] == ""
    assert payload["next_review_execution_symbol"] == ""
    assert payload["next_fill_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_fill_evidence_execution_symbol"] == "XAUUSD"
    assert payload["review_items"][0]["review_status"] == "awaiting_paper_execution_fill"
    assert payload["review_items"][0]["paper_execution_evidence_present"] is False


def test_build_commodity_paper_execution_review_uses_explicit_queue_json(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    old_queue = review_dir / "20260310T155905Z_commodity_paper_execution_queue.json"
    new_queue = review_dir / "20260311T084601Z_commodity_paper_execution_queue.json"
    _write_json(
        old_queue,
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                }
            ],
        },
    )
    _write_json(
        new_queue,
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["COPPER"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:COPPER",
                    "symbol": "COPPER",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
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
            "--execution-queue-json",
            str(new_queue),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_execution_queue_artifact"] == str(new_queue.resolve())
    assert payload["execution_symbols"] == ["COPPER"]


def test_build_commodity_paper_execution_review_marks_partial_fill_remainder(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "symbol": "XAGUSD",
                    "queue_rank": 2,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                },
            ],
        },
    )
    _write_jsonl(
        tmp_path / "logs" / "paper_execution_ledger.jsonl",
        [{"domain": "paper_execution", "symbol": "XAUUSD"}],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_review_status"] == "paper-execution-review-pending-fill-remainder"
    assert payload["actionable_review_item_count"] == 1
    assert payload["fill_evidence_pending_count"] == 1
    assert payload["review_pending_symbols"] == ["XAUUSD"]
    assert payload["fill_evidence_pending_symbols"] == ["XAGUSD"]
    assert payload["next_review_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_review_execution_symbol"] == "XAUUSD"
    assert payload["next_fill_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["next_fill_evidence_execution_symbol"] == "XAGUSD"


def test_build_commodity_paper_execution_review_marks_close_evidence_pending_fill_remainder(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    queue_path = review_dir / "20260310T155905Z_commodity_paper_execution_queue.json"
    positions_path = tmp_path / "artifacts" / "paper_positions_open.json"
    _write_json(
        queue_path,
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "symbol": "XAGUSD",
                    "queue_rank": 2,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                },
            ],
        },
    )
    _write_json(
        positions_path,
        {
            "as_of": "2026-03-11T08:40:01+00:00",
            "positions": [
                {
                    "open_date": "2026-03-11",
                    "symbol": "XAUUSD",
                    "status": "OPEN",
                    "source_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
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
            "--execution-queue-json",
            str(queue_path),
            "--paper-positions-path",
            str(positions_path),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_review_status"] == "paper-execution-close-evidence-pending-fill-remainder"
    assert payload["actionable_review_item_count"] == 0
    assert payload["review_pending_symbols"] == []
    assert payload["close_evidence_pending_count"] == 1
    assert payload["close_evidence_pending_symbols"] == ["XAUUSD"]
    assert payload["next_close_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_close_evidence_execution_symbol"] == "XAUUSD"
    assert payload["fill_evidence_pending_symbols"] == ["XAGUSD"]
    assert payload["review_items"][0]["review_status"] == "awaiting_paper_execution_close_evidence"


def test_build_commodity_paper_execution_review_hydrates_evidence_snapshots(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    queue_path = review_dir / "20260310T155905Z_commodity_paper_execution_queue.json"
    positions_path = tmp_path / "artifacts" / "paper_positions_open.json"
    ledger_path = tmp_path / "logs" / "paper_execution_ledger.jsonl"
    db_path = tmp_path / "artifacts" / "lie_engine.db"
    _write_json(
        queue_path,
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                    "execution_price_normalization_mode": "paper_proxy_reference",
                }
            ],
        },
    )
    _write_json(
        positions_path,
        {
            "as_of": "2026-03-11T08:40:01+00:00",
            "positions": [
                {
                    "open_date": "2026-03-11",
                    "symbol": "XAUUSD",
                    "side": "LONG",
                    "size_pct": 1.5425170540530218,
                    "risk_pct": 0.10395032142372562,
                    "entry_price": 5198.10009765625,
                    "stop_price": 4847.7998046875,
                    "target_price": 5758.58056640625,
                    "runtime_mode": "commodity_queue_bridge",
                    "status": "OPEN",
                    "source_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "bridge_idempotency_key": "bridge-key-xau",
                    "quote_usdt": 0.15896067200583952,
                    "signal_date": "2026-03-11",
                    "regime_gate": "paper_only",
                    "execution_price_normalization_mode": "paper_proxy_reference",
                    "paper_proxy_price_normalized": True,
                    "signal_price_reference_kind": "commodity_proxy_market",
                    "signal_price_reference_source": "yfinance:GC=F",
                    "signal_price_reference_provider": "yfinance",
                    "signal_price_reference_symbol": "GC=F",
                }
            ],
        },
    )
    _write_jsonl(
        ledger_path,
        [
            {
                "domain": "paper_execution",
                "ts": "2026-03-11T08:40:01+00:00",
                "event_source": "commodity_queue_bridge",
                "symbol": "XAUUSD",
                "action": "OPEN",
                "decision": "bridge_apply",
                "route": "commodity_queue_bridge",
                "side": "LONG",
                "qty": 3.058e-05,
                "mark_px": 5198.10009766,
                "fill_px": 5198.10009766,
                "notional_usdt": 0.15896067,
                "order_mode": "paper_bridge_proxy_reference",
                "bridge_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                "bridge_idempotency_key": "bridge-key-xau",
                "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                "execution_price_normalization_mode": "paper_proxy_reference",
                "paper_proxy_price_normalized": True,
                "signal_price_reference_kind": "commodity_proxy_market",
                "signal_price_reference_source": "yfinance:GC=F",
                "signal_price_reference_provider": "yfinance",
                "signal_price_reference_symbol": "GC=F",
            }
        ],
    )
    _create_plan_db(db_path)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--execution-queue-json",
            str(queue_path),
            "--paper-ledger-path",
            str(ledger_path),
            "--paper-positions-path",
            str(positions_path),
            "--paper-db-path",
            str(db_path),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_review_status"] == "paper-execution-close-evidence-pending"
    assert payload["actionable_review_item_count"] == 0
    assert payload["review_pending_symbols"] == []
    assert payload["close_evidence_pending_count"] == 1
    assert payload["close_evidence_pending_symbols"] == ["XAUUSD"]
    assert payload["next_close_evidence_execution_symbol"] == "XAUUSD"
    item = payload["review_items"][0]
    assert item["review_status"] == "awaiting_paper_execution_close_evidence"
    assert item["paper_execution_evidence_present"] is True
    assert item["paper_entry_price"] == 5198.10009765625
    assert item["paper_stop_price"] == 4847.7998046875
    assert item["paper_target_price"] == 5758.58056640625
    assert item["paper_quote_usdt"] == 0.15896067200583952
    assert item["paper_execution_side"] == "LONG"
    assert item["paper_execution_status"] == "OPEN"
    assert item["paper_runtime_mode"] == "commodity_queue_bridge"
    assert item["paper_order_mode"] == "paper_bridge_proxy_reference"
    assert item["paper_signal_price_reference_source"] == "yfinance:GC=F"
    assert item["paper_execution_evidence_snapshot"]["position"]["entry_price"] == 5198.10009765625
    assert item["paper_execution_evidence_snapshot"]["ledger"]["order_mode"] == "paper_bridge_proxy_reference"
    assert item["paper_execution_evidence_snapshot"]["trade_plan"]["target_price"] == 5758.58
    assert item["paper_execution_evidence_snapshot"]["executed_plan"]["status"] == "OPEN"
