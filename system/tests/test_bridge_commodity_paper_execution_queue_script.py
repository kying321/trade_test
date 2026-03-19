from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/bridge_commodity_paper_execution_queue.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_bridge_commodity_queue_blocks_without_directional_signal(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "batch": "metals_all",
                    "symbol": "XAUUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAGUSD",
                    "batch": "metals_all",
                    "symbol": "XAGUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 2,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:COPPER",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:COPPER",
                    "batch": "metals_all",
                    "symbol": "COPPER",
                    "route_class": "focus_primary",
                    "queue_rank": 3,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260310T120100Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-10T12:01:00Z",
            "as_of": "2026-03-10",
            "symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "tickets": [
                {
                    "symbol": "XAUUSD",
                    "date": "",
                    "allowed": False,
                    "reasons": ["signal_not_found"],
                    "signal": {"side": ""},
                    "levels": {"entry_price": 0.0, "stop_price": 0.0, "target_price": 0.0},
                    "sizing": {"equity_usdt": 100.0, "quote_usdt": 0.0, "risk_budget_usdt": 0.0},
                },
                {
                    "symbol": "XAGUSD",
                    "date": "",
                    "allowed": False,
                    "reasons": ["signal_not_found"],
                    "signal": {"side": ""},
                    "levels": {"entry_price": 0.0, "stop_price": 0.0, "target_price": 0.0},
                    "sizing": {"equity_usdt": 100.0, "quote_usdt": 0.0, "risk_budget_usdt": 0.0},
                },
                {
                    "symbol": "COPPER",
                    "date": "",
                    "allowed": False,
                    "reasons": ["signal_not_found"],
                    "signal": {"side": ""},
                    "levels": {"entry_price": 0.0, "stop_price": 0.0, "target_price": 0.0},
                    "sizing": {"equity_usdt": 100.0, "quote_usdt": 0.0, "risk_budget_usdt": 0.0},
                },
            ],
            "summary": {"ticket_count": 3, "allowed_count": 0, "missing_count": 3},
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
            "--signal-tickets-json",
            str(review_dir / "20260310T120100Z_signal_to_order_tickets.json"),
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
    assert payload["blocked_count"] == 3
    assert payload["already_present_count"] == 0
    assert payload["signal_missing_count"] == 3
    assert payload["signal_stale_count"] == 0
    assert payload["positions_written"] == 0
    assert payload["ledger_rows_written"] == 0
    assert payload["trade_plan_rows_written"] == 0
    assert payload["next_blocked_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_blocked_symbol"] == "XAUUSD"


def test_bridge_commodity_queue_marks_stale_directional_signal(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    signal_path = review_dir / "20260310T120100Z_signal_to_order_tickets.json"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "batch": "metals_all",
                    "symbol": "XAUUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                    "allow_live_execution": False,
                }
            ],
        },
    )
    _write_json(
        signal_path,
        {
            "generated_at_utc": "2026-03-10T12:01:00Z",
            "as_of": "2026-03-10",
            "symbols": ["XAUUSD"],
            "tickets": [
                {
                    "symbol": "XAUUSD",
                    "date": "2026-01-28",
                    "allowed": False,
                    "reasons": ["stale_signal", "size_below_min_notional"],
                    "signal": {"side": "BUY"},
                    "levels": {
                        "entry_price": 2900.0,
                        "stop_price": 2860.0,
                        "target_price": 2980.0,
                    },
                    "sizing": {
                        "equity_usdt": 100.0,
                        "quote_usdt": 2.0,
                        "risk_budget_usdt": 0.6,
                        "max_alloc_pct": 0.3,
                    },
                }
            ],
            "summary": {"ticket_count": 1, "allowed_count": 0, "missing_count": 0, "stale_count": 1},
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
            "--signal-tickets-json",
            str(signal_path),
            "--now",
            "2026-03-10T12:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bridge_status"] == "blocked_stale_directional_signal"
    assert payload["ready_count"] == 0
    assert payload["blocked_count"] == 1
    assert payload["signal_missing_count"] == 0
    assert payload["signal_stale_count"] == 1
    assert payload["next_blocked_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    row = payload["bridge_items"][0]
    assert row["bridge_status"] == "blocked_stale_directional_signal"
    assert row["bridge_reasons"] == ["stale_signal"]
    assert row["signal_date"] == "2026-01-28"
    assert row["signal_age_days"] == 41
    assert row["paper_only_ignored_ticket_reasons"] == ["size_below_min_notional"]
    assert payload["signal_stale_age_days"] == {"XAUUSD": 41}
    assert "signal-stale-age-days: XAUUSD:41" in payload["summary_text"]


def test_bridge_commodity_queue_blocks_proxy_price_reference_only(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    signal_path = review_dir / "20260310T120100Z_signal_to_order_tickets.json"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAGUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAGUSD",
                    "batch": "metals_all",
                    "symbol": "XAGUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                    "allow_live_execution": False,
                }
            ],
        },
    )
    _write_json(
        signal_path,
        {
            "generated_at_utc": "2026-03-10T12:01:00Z",
            "as_of": "2026-03-10",
            "symbols": ["XAGUSD"],
            "tickets": [
                {
                    "symbol": "XAGUSD",
                    "date": "2026-03-10",
                    "allowed": False,
                    "reasons": ["proxy_price_reference_only", "size_below_min_notional"],
                    "signal": {
                        "side": "BUY",
                        "price_reference_kind": "commodity_proxy_market",
                        "price_reference_source": "yfinance:SI=F",
                        "execution_price_ready": False,
                    },
                    "levels": {"entry_price": 115.08, "stop_price": 69.856, "target_price": 187.4384},
                    "sizing": {"equity_usdt": 100.0, "quote_usdt": 2.0, "risk_budget_usdt": 0.6},
                }
            ],
            "summary": {"ticket_count": 1, "allowed_count": 0, "missing_count": 0, "proxy_price_only_count": 1},
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
            "--signal-tickets-json",
            str(signal_path),
            "--now",
            "2026-03-10T12:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bridge_status"] == "blocked_proxy_price_reference_only"
    assert payload["ready_count"] == 0
    assert payload["blocked_count"] == 1
    assert payload["signal_missing_count"] == 0
    assert payload["signal_stale_count"] == 0
    assert payload["signal_proxy_price_only_count"] == 1
    row = payload["bridge_items"][0]
    assert row["bridge_status"] == "blocked_proxy_price_reference_only"
    assert row["bridge_reasons"] == ["proxy_price_reference_only"]
    assert row["paper_only_ignored_ticket_reasons"] == ["size_below_min_notional"]
    assert row["signal_execution_price_ready"] is False
    assert row["signal_price_reference_kind"] == "commodity_proxy_market"
    assert row["signal_price_reference_source"] == "yfinance:SI=F"


def test_bridge_commodity_queue_applies_proxy_price_normalized_paper_signal(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    signal_path = review_dir / "20260310T120100Z_signal_to_order_tickets.json"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAGUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAGUSD",
                    "batch": "metals_all",
                    "symbol": "XAGUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                    "allow_live_execution": False,
                    "allow_proxy_price_reference_execution": True,
                    "execution_price_normalization_mode": "paper_proxy_reference",
                }
            ],
        },
    )
    _write_json(
        signal_path,
        {
            "generated_at_utc": "2026-03-10T12:01:00Z",
            "as_of": "2026-03-10",
            "symbols": ["XAGUSD"],
            "tickets": [
                {
                    "symbol": "XAGUSD",
                    "date": "2026-03-10",
                    "allowed": False,
                    "reasons": ["proxy_price_reference_only", "size_below_min_notional"],
                    "signal": {
                        "side": "BUY",
                        "price_reference_kind": "commodity_proxy_market",
                        "price_reference_source": "yfinance:SI=F",
                        "execution_price_ready": False,
                    },
                    "levels": {"entry_price": 31.25, "stop_price": 30.5, "target_price": 32.45},
                    "sizing": {"equity_usdt": 100.0, "quote_usdt": 2.0, "risk_budget_usdt": 0.6},
                }
            ],
            "summary": {"ticket_count": 1, "allowed_count": 0, "missing_count": 0, "proxy_price_only_count": 1},
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
            "--signal-tickets-json",
            str(signal_path),
            "--now",
            "2026-03-10T12:05:00Z",
            "--apply",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bridge_status"] == "paper_execution_bridged"
    assert payload["ready_count"] == 1
    assert payload["positions_written"] == 1
    row = payload["bridge_items"][0]
    assert row["bridge_status"] == "bridge_ready"
    assert row["bridge_reasons"] == []
    assert row["paper_proxy_price_normalized"] is True
    assert row["paper_only_ignored_ticket_reasons"] == ["proxy_price_reference_only", "size_below_min_notional"]
    assert row["signal_price_reference_provider"] == "yfinance"
    assert row["signal_price_reference_symbol"] == "SI=F"

    positions_payload = json.loads((output_root / "artifacts" / "paper_positions_open.json").read_text(encoding="utf-8"))
    assert positions_payload["positions"][0]["symbol"] == "XAGUSD"
    assert positions_payload["positions"][0]["execution_price_normalization_mode"] == "paper_proxy_reference"
    assert positions_payload["positions"][0]["paper_proxy_price_normalized"] is True
    assert positions_payload["positions"][0]["signal_price_reference_source"] == "yfinance:SI=F"

    ledger_lines = (output_root / "logs" / "paper_execution_ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 1
    ledger_payload = json.loads(ledger_lines[0])
    assert ledger_payload["symbol"] == "XAGUSD"
    assert ledger_payload["order_mode"] == "paper_bridge_proxy_reference"
    assert ledger_payload["signal_price_reference_symbol"] == "SI=F"

    with sqlite3.connect(output_root / "artifacts" / "lie_engine.db") as conn:
        trade_rows = conn.execute(
            "SELECT symbol, execution_price_normalization_mode, paper_proxy_price_normalized, signal_price_reference_source FROM trade_plans"
        ).fetchall()
        exec_rows = conn.execute(
            "SELECT symbol, execution_price_normalization_mode, paper_proxy_price_normalized, signal_price_reference_source FROM executed_plans"
        ).fetchall()
    assert trade_rows == [("XAGUSD", "paper_proxy_reference", 1, "yfinance:SI=F")]
    assert exec_rows == [("XAGUSD", "paper_proxy_reference", 1, "yfinance:SI=F")]


def test_bridge_commodity_queue_applies_and_is_idempotent(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    queue_path = review_dir / "20260310T120000Z_commodity_paper_execution_queue.json"
    signal_path = review_dir / "20260310T120100Z_signal_to_order_tickets.json"
    _write_json(
        queue_path,
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "batch": "metals_all",
                    "symbol": "XAUUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                }
            ],
        },
    )
    _write_json(
        signal_path,
        {
            "generated_at_utc": "2026-03-10T12:01:00Z",
            "as_of": "2026-03-10",
            "symbols": ["XAUUSD"],
            "tickets": [
                {
                    "symbol": "XAUUSD",
                    "date": "2026-03-10",
                    "allowed": True,
                    "reasons": [],
                    "signal": {"side": "BUY"},
                    "levels": {
                        "entry_price": 2900.0,
                        "stop_price": 2860.0,
                        "target_price": 2980.0,
                    },
                    "sizing": {
                        "equity_usdt": 100.0,
                        "quote_usdt": 15.0,
                        "risk_budget_usdt": 0.6,
                        "max_alloc_pct": 0.3,
                    },
                }
            ],
            "summary": {"ticket_count": 1, "allowed_count": 1, "missing_count": 0},
        },
    )

    base_cmd = [
        "python3",
        str(SCRIPT_PATH),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--signal-tickets-json",
        str(signal_path),
        "--now",
        "2026-03-10T12:05:00Z",
        "--apply",
    ]
    proc = subprocess.run(
        base_cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bridge_status"] == "paper_execution_bridged"
    assert payload["ready_count"] == 1
    assert payload["positions_written"] == 1
    assert payload["ledger_rows_written"] == 1
    assert payload["trade_plan_rows_written"] == 1
    assert payload["executed_plan_rows_written"] == 1
    assert payload["applied_execution_ids"] == ["commodity-paper-execution:metals_all:XAUUSD"]

    positions_payload = json.loads((output_root / "artifacts" / "paper_positions_open.json").read_text(encoding="utf-8"))
    assert positions_payload["positions"][0]["symbol"] == "XAUUSD"
    assert positions_payload["positions"][0]["side"] == "LONG"
    assert positions_payload["positions"][0]["source_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"

    ledger_lines = (output_root / "logs" / "paper_execution_ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 1
    ledger_payload = json.loads(ledger_lines[0])
    assert ledger_payload["symbol"] == "XAUUSD"
    assert ledger_payload["bridge_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"

    with sqlite3.connect(output_root / "artifacts" / "lie_engine.db") as conn:
        trade_rows = conn.execute("SELECT symbol, side, bridge_execution_id FROM trade_plans").fetchall()
        exec_rows = conn.execute("SELECT symbol, side, bridge_execution_id, status FROM executed_plans").fetchall()
    assert trade_rows == [("XAUUSD", "LONG", "commodity-paper-execution:metals_all:XAUUSD")]
    assert exec_rows == [("XAUUSD", "LONG", "commodity-paper-execution:metals_all:XAUUSD", "OPEN")]

    proc_again = subprocess.run(
        base_cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc_again.returncode == 0, proc_again.stderr
    payload_again = json.loads(proc_again.stdout)
    assert payload_again["bridge_status"] == "bridge_noop_already_bridged"
    assert payload_again["positions_written"] == 0
    assert payload_again["ledger_rows_written"] == 0
    assert payload_again["trade_plan_rows_written"] == 0
    assert payload_again["executed_plan_rows_written"] == 0


def test_bridge_commodity_queue_ignores_live_only_ticket_reasons_for_paper_mode(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    signal_path = review_dir / "20260310T120100Z_signal_to_order_tickets.json"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "batch": "metals_all",
                    "symbol": "XAUUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                    "allow_live_execution": False,
                }
            ],
        },
    )
    _write_json(
        signal_path,
        {
            "generated_at_utc": "2026-03-10T12:01:00Z",
            "as_of": "2026-03-10",
            "symbols": ["XAUUSD"],
            "tickets": [
                {
                    "symbol": "XAUUSD",
                    "date": "2026-03-10",
                    "allowed": False,
                    "reasons": ["unsupported_symbol", "size_below_min_notional"],
                    "signal": {"side": "BUY"},
                    "levels": {
                        "entry_price": 2900.0,
                        "stop_price": 2860.0,
                        "target_price": 2980.0,
                    },
                    "sizing": {
                        "equity_usdt": 100.0,
                        "quote_usdt": 2.0,
                        "risk_budget_usdt": 0.6,
                        "max_alloc_pct": 0.3,
                    },
                }
            ],
            "summary": {"ticket_count": 1, "allowed_count": 0, "missing_count": 0},
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
            "--signal-tickets-json",
            str(signal_path),
            "--now",
            "2026-03-10T12:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bridge_status"] == "bridge_ready"
    assert payload["ready_count"] == 1
    assert payload["blocked_count"] == 0
    row = payload["bridge_items"][0]
    assert row["paper_only_ignored_ticket_reasons"] == ["unsupported_symbol", "size_below_min_notional"]
    assert row["bridge_reasons"] == []


def test_bridge_commodity_queue_marks_partial_stale_remainder_when_one_symbol_already_bridged(tmp_path: Path) -> None:
    review_dir = tmp_path / "output" / "review"
    output_root = tmp_path / "output"
    signal_path = review_dir / "20260310T120100Z_signal_to_order_tickets.json"
    _write_json(
        review_dir / "20260310T120000Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00Z",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
                    "batch": "metals_all",
                    "symbol": "XAUUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 1,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "source_ticket_id": "commodity-paper-ticket:metals_all:XAGUSD",
                    "batch": "metals_all",
                    "symbol": "XAGUSD",
                    "route_class": "focus_primary",
                    "queue_rank": 2,
                    "regime_gate": "paper_only",
                    "weight_hint": 1.0,
                    "allow_paper_execution": True,
                },
            ],
        },
    )
    _write_json(
        output_root / "artifacts" / "paper_positions_open.json",
        {
            "as_of": "2026-03-10",
            "positions": [
                {
                    "symbol": "XAUUSD",
                    "source_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                }
            ],
        },
    )
    _write_json(
        signal_path,
        {
            "generated_at_utc": "2026-03-10T12:01:00Z",
            "as_of": "2026-03-10",
            "symbols": ["XAUUSD", "XAGUSD"],
            "tickets": [
                {
                    "symbol": "XAUUSD",
                    "date": "2026-03-10",
                    "allowed": True,
                    "reasons": [],
                    "signal": {"side": "BUY"},
                    "levels": {"entry_price": 2900.0, "stop_price": 2860.0, "target_price": 2980.0},
                    "sizing": {"equity_usdt": 100.0, "quote_usdt": 15.0, "risk_budget_usdt": 0.6},
                },
                {
                    "symbol": "XAGUSD",
                    "date": "2026-01-28",
                    "allowed": False,
                    "reasons": ["stale_signal", "size_below_min_notional"],
                    "signal": {"side": "BUY"},
                    "levels": {"entry_price": 31.25, "stop_price": 30.5, "target_price": 32.45},
                    "sizing": {"equity_usdt": 100.0, "quote_usdt": 2.0, "risk_budget_usdt": 0.6},
                },
            ],
            "summary": {"ticket_count": 2, "allowed_count": 1, "missing_count": 0, "stale_count": 1},
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
            "--signal-tickets-json",
            str(signal_path),
            "--now",
            "2026-03-10T12:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bridge_status"] == "bridge_partially_bridged_stale_remainder"
    assert payload["ready_count"] == 0
    assert payload["blocked_count"] == 1
    assert payload["already_present_count"] == 1
    assert payload["signal_stale_count"] == 1
    assert payload["already_bridged_symbols"] == ["XAUUSD"]
    assert payload["next_blocked_execution_id"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["next_blocked_symbol"] == "XAGUSD"
    assert payload["bridge_items"][0]["bridge_status"] == "already_bridged"
    assert payload["bridge_items"][1]["bridge_status"] == "blocked_stale_directional_signal"
    assert payload["bridge_items"][1]["signal_age_days"] == 41
    assert payload["signal_stale_age_days"] == {"XAGUSD": 41}
