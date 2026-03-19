from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_execution_gap_report.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in payloads) + "\n",
        encoding="utf-8",
    )


def _write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _create_db(path: Path, *, trade_symbols: list[str], executed_symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE trade_plans(symbol TEXT)")
        conn.execute("CREATE TABLE executed_plans(symbol TEXT)")
        for symbol in trade_symbols:
            conn.execute("INSERT INTO trade_plans(symbol) VALUES (?)", (symbol,))
        for symbol in executed_symbols:
            conn.execute("INSERT INTO executed_plans(symbol) VALUES (?)", (symbol,))
        conn.commit()


def test_build_commodity_paper_execution_gap_report_flags_unwired_commodity_queue(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {"status": "ok", "execution_review_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {"status": "ok", "execution_retro_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "BTCUSDT", asset_class: "crypto"}
    - {symbol: "ETHUSDT", asset_class: "crypto"}
    - {symbol: "BNBUSDT", asset_class: "crypto"}
""",
    )
    _write_json(tmp_path / "artifacts" / "paper_positions_open.json", {"as_of": "2026-03-10", "positions": []})
    _write_jsonl(tmp_path / "logs" / "paper_execution_ledger.jsonl", [{"domain": "paper_execution", "symbol": "ETHUSDT"}])
    _create_db(
        tmp_path / "artifacts" / "lie_engine.db",
        trade_symbols=["ETHUSDT"],
        executed_symbols=["ETHUSDT"],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["gap_status"] == "blocking_gap_active"
    assert payload["current_decision"] == "do_not_assume_commodity_paper_execution_active"
    assert payload["queue_symbols_missing_from_core_universe"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["queue_symbols_without_any_evidence"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert "queue_symbols_missing_from_core_universe" in payload["gap_reason_codes"]
    assert "core_universe_crypto_only" in payload["gap_reason_codes"]
    assert "queue_symbols_missing_from_trade_plans" in payload["gap_reason_codes"]
    assert "queue_symbols_missing_from_paper_execution_evidence" in payload["gap_reason_codes"]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_commodity_paper_execution_gap_report_lists_partial_evidence_gaps(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {"status": "ok", "execution_review_status": "paper-execution-review-pending"},
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {"status": "ok", "execution_retro_status": "paper-execution-retro-pending"},
    )
    _write_json(
        review_dir / "20260310T155911Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_partially_bridged_stale_remainder",
            "already_present_count": 1,
            "already_bridged_symbols": ["XAUUSD"],
            "bridge_items": [
                {"symbol": "XAUUSD", "bridge_status": "already_bridged"},
                {
                    "symbol": "XAGUSD",
                    "bridge_status": "blocked_stale_directional_signal",
                    "bridge_reasons": ["stale_signal"],
                },
                {
                    "symbol": "COPPER",
                    "bridge_status": "blocked_stale_directional_signal",
                    "bridge_reasons": ["stale_signal"],
                },
            ],
        },
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "BTCUSDT", asset_class: "crypto"}
""",
    )
    _write_json(
        tmp_path / "artifacts" / "paper_positions_open.json",
        {"as_of": "2026-03-10", "positions": [{"symbol": "XAUUSD"}]},
    )
    _write_jsonl(
        tmp_path / "logs" / "paper_execution_ledger.jsonl",
        [{"domain": "paper_execution", "symbol": "XAUUSD"}],
    )
    _create_db(
        tmp_path / "artifacts" / "lie_engine.db",
        trade_symbols=["XAUUSD"],
        executed_symbols=["XAUUSD"],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_bridge_status"] == "bridge_partially_bridged_stale_remainder"
    assert payload["queue_symbols_already_bridged"] == ["XAUUSD"]
    assert payload["queue_symbols_with_trade_plans"] == ["XAUUSD"]
    assert payload["queue_symbols_without_trade_plans"] == ["XAGUSD", "COPPER"]
    assert payload["queue_symbols_with_any_evidence"] == ["XAUUSD"]
    assert payload["queue_symbols_without_any_evidence"] == ["XAGUSD", "COPPER"]
    assert "queue_symbols_missing_from_trade_plans" in payload["gap_reason_codes"]
    assert "queue_symbols_missing_from_paper_execution_evidence" in payload["gap_reason_codes"]
    assert (
        "Continue paper review/retro for already bridged symbols while the remaining queue symbols stay blocked."
        == payload["recommended_actions"][0]
    )
    assert "already-bridged-symbols: XAUUSD" in payload["summary_lines"]
    assert "Queue symbols still missing from sqlite trade_plans: XAGUSD, COPPER." in payload["root_cause_lines"]
    assert (
        "Queue symbols still missing from paper execution ledger, open paper positions, or executed_plans: XAGUSD, COPPER."
        in payload["root_cause_lines"]
    )


def test_build_commodity_paper_execution_gap_report_clears_when_bridge_evidence_exists(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {"status": "ok", "execution_review_status": "paper-execution-review-pending"},
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {"status": "ok", "execution_retro_status": "paper-execution-retro-pending"},
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "XAUUSD", asset_class: "commodity"}
    - {symbol: "ETHUSDT", asset_class: "crypto"}
""",
    )
    _write_json(
        tmp_path / "artifacts" / "paper_positions_open.json",
        {"as_of": "2026-03-10", "positions": [{"symbol": "XAUUSD"}]},
    )
    _write_jsonl(
        tmp_path / "logs" / "paper_execution_ledger.jsonl",
        [{"domain": "paper_execution", "symbol": "XAUUSD"}],
    )
    _create_db(
        tmp_path / "artifacts" / "lie_engine.db",
        trade_symbols=["XAUUSD"],
        executed_symbols=["XAUUSD"],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["gap_status"] == "gap_clear"
    assert payload["current_decision"] == "commodity_paper_execution_path_present"
    assert payload["queue_symbols_missing_from_core_universe"] == []
    assert payload["queue_symbols_with_any_evidence"] == ["XAUUSD"]
    assert payload["queue_symbols_without_any_evidence"] == []
    assert payload["gap_reason_codes"] == []


def test_build_commodity_paper_execution_gap_report_flags_missing_directional_signal(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {"status": "ok", "execution_review_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {"status": "ok", "execution_retro_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155911Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_missing_directional_signal",
            "bridge_items": [
                {"symbol": "XAUUSD", "bridge_status": "blocked_missing_directional_signal"},
                {"symbol": "XAGUSD", "bridge_status": "blocked_missing_directional_signal"},
            ],
        },
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "BTCUSDT", asset_class: "crypto"}
""",
    )
    _write_json(tmp_path / "artifacts" / "paper_positions_open.json", {"as_of": "2026-03-10", "positions": []})
    _write_jsonl(tmp_path / "logs" / "paper_execution_ledger.jsonl", [])
    _create_db(
        tmp_path / "artifacts" / "lie_engine.db",
        trade_symbols=[],
        executed_symbols=[],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_bridge_status"] == "blocked_missing_directional_signal"
    assert payload["queue_symbols_missing_directional_signal"] == ["XAUUSD", "XAGUSD"]
    assert "queue_symbols_missing_directional_signal" in payload["gap_reason_codes"]
    assert payload["recommended_actions"][0] == (
        "Keep commodity queue in paper-only bridge review mode until config/runtime coverage and execution evidence are both present."
    )


def test_build_commodity_paper_execution_gap_report_flags_stale_directional_signal(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {"status": "ok", "execution_review_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {"status": "ok", "execution_retro_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155911Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_stale_directional_signal",
            "bridge_items": [
                {
                    "symbol": "XAUUSD",
                    "bridge_status": "blocked_stale_directional_signal",
                    "bridge_reasons": ["stale_signal"],
                    "signal_date": "2026-01-28",
                    "signal_age_days": 41,
                },
                {
                    "symbol": "XAGUSD",
                    "bridge_status": "blocked_stale_directional_signal",
                    "bridge_reasons": ["stale_signal"],
                    "signal_date": "2026-01-27",
                    "signal_age_days": 42,
                },
            ],
        },
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "BTCUSDT", asset_class: "crypto"}
""",
    )
    _write_json(tmp_path / "artifacts" / "paper_positions_open.json", {"as_of": "2026-03-10", "positions": []})
    _write_jsonl(tmp_path / "logs" / "paper_execution_ledger.jsonl", [])
    _create_db(
        tmp_path / "artifacts" / "lie_engine.db",
        trade_symbols=[],
        executed_symbols=[],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_bridge_status"] == "blocked_stale_directional_signal"
    assert payload["queue_symbols_with_stale_directional_signal"] == ["XAUUSD", "XAGUSD"]
    assert payload["queue_symbols_with_stale_directional_signal_dates"] == {
        "XAGUSD": "2026-01-27",
        "XAUUSD": "2026-01-28",
    }
    assert payload["queue_symbols_with_stale_directional_signal_age_days"] == {
        "XAGUSD": 42,
        "XAUUSD": 41,
    }
    assert payload["stale_directional_signal_watch_items"] == [
        {
            "execution_id": "",
            "symbol": "XAGUSD",
            "signal_date": "2026-01-27",
            "signal_age_days": 42,
        },
        {
            "execution_id": "",
            "symbol": "XAUUSD",
            "signal_date": "2026-01-28",
            "signal_age_days": 41,
        },
    ]
    assert payload["queue_symbols_missing_directional_signal"] == []
    assert "queue_symbols_with_stale_directional_signal" in payload["gap_reason_codes"]
    assert "stale-directional-signal-age-days: XAGUSD:42, XAUUSD:41" in payload["summary_lines"]
    assert "stale-directional-watch: XAGUSD:42d@2026-01-27, XAUUSD:41d@2026-01-28" in payload["summary_lines"]
    assert "Latest stale directional trigger ages in days by symbol: XAGUSD:42, XAUUSD:41." in payload["root_cause_lines"]
    assert payload["recommended_actions"][0] == (
        "Keep commodity queue in paper-only bridge review mode until config/runtime coverage and execution evidence are both present."
    )
    assert (
        "Add a commodity price-normalization path or retarget the queue to proxy instruments before allowing bridge apply."
        not in payload["recommended_actions"]
    )


def test_build_commodity_paper_execution_gap_report_flags_proxy_price_reference_only(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {"status": "ok", "execution_review_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {"status": "ok", "execution_retro_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155911Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_proxy_price_reference_only",
            "bridge_items": [
                {
                    "symbol": "XAUUSD",
                    "bridge_status": "blocked_proxy_price_reference_only",
                    "bridge_reasons": ["proxy_price_reference_only", "stale_signal"],
                },
                {
                    "symbol": "XAGUSD",
                    "bridge_status": "blocked_proxy_price_reference_only",
                    "bridge_reasons": ["proxy_price_reference_only"],
                },
            ],
        },
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "BTCUSDT", asset_class: "crypto"}
""",
    )
    _write_json(tmp_path / "artifacts" / "paper_positions_open.json", {"as_of": "2026-03-10", "positions": []})
    _write_jsonl(tmp_path / "logs" / "paper_execution_ledger.jsonl", [])
    _create_db(
        tmp_path / "artifacts" / "lie_engine.db",
        trade_symbols=[],
        executed_symbols=[],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_bridge_status"] == "blocked_proxy_price_reference_only"
    assert payload["queue_symbols_with_proxy_price_reference_only"] == ["XAUUSD", "XAGUSD"]
    assert payload["queue_symbols_with_stale_directional_signal"] == ["XAUUSD"]
    assert "queue_symbols_with_proxy_price_reference_only" in payload["gap_reason_codes"]
    assert payload["recommended_actions"][1] == (
        "Add a commodity price-normalization path or retarget the queue to proxy instruments before allowing bridge apply."
    )


def test_build_commodity_paper_execution_gap_report_prefers_apply_when_bridge_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {"status": "ok", "execution_review_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {"status": "ok", "execution_retro_status": "paper-execution-awaiting-fill-evidence"},
    )
    _write_json(
        review_dir / "20260310T155911Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_ready",
            "bridge_items": [
                {"symbol": "XAUUSD", "bridge_status": "bridge_ready", "bridge_reasons": []},
                {"symbol": "XAGUSD", "bridge_status": "blocked_stale_directional_signal", "bridge_reasons": ["stale_signal"]},
            ],
        },
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "BTCUSDT", asset_class: "crypto"}
""",
    )
    _write_json(tmp_path / "artifacts" / "paper_positions_open.json", {"as_of": "2026-03-10", "positions": []})
    _write_jsonl(tmp_path / "logs" / "paper_execution_ledger.jsonl", [])
    _create_db(
        tmp_path / "artifacts" / "lie_engine.db",
        trade_symbols=[],
        executed_symbols=[],
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
            "--now",
            "2026-03-10T16:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_bridge_status"] == "bridge_ready"
    assert payload["queue_symbols_with_stale_directional_signal"] == ["XAGUSD"]
    assert payload["recommended_actions"][0] == (
        "Apply the ready commodity bridge items into paper execution evidence, then reassess the remaining stale queue symbols."
    )
    assert payload["recommended_actions"][-1] == (
        "Keep stale queue symbols in watch until they generate a fresh commodity breakout/reclaim trigger."
    )


def test_build_commodity_paper_execution_gap_report_uses_explicit_artifacts(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    old_queue = review_dir / "20260310T155905Z_commodity_paper_execution_queue.json"
    new_queue = review_dir / "20260311T084601Z_commodity_paper_execution_queue.json"
    old_review = review_dir / "20260310T155907Z_commodity_paper_execution_review.json"
    new_review = review_dir / "20260311T084602Z_commodity_paper_execution_review.json"
    old_retro = review_dir / "20260310T155909Z_commodity_paper_execution_retro.json"
    new_retro = review_dir / "20260311T084603Z_commodity_paper_execution_retro.json"
    old_bridge = review_dir / "20260310T155911Z_commodity_paper_execution_bridge.json"
    new_bridge = review_dir / "20260311T084600Z_commodity_paper_execution_bridge.json"
    _write_json(old_queue, {"status": "ok", "execution_queue_status": "paper-execution-queued", "execution_symbols": ["XAUUSD"]})
    _write_json(new_queue, {"status": "ok", "execution_queue_status": "paper-execution-queued", "execution_symbols": ["COPPER"]})
    _write_json(old_review, {"status": "ok", "execution_review_status": "paper-execution-awaiting-fill-evidence"})
    _write_json(new_review, {"status": "ok", "execution_review_status": "paper-execution-review-pending"})
    _write_json(old_retro, {"status": "ok", "execution_retro_status": "paper-execution-awaiting-fill-evidence"})
    _write_json(new_retro, {"status": "ok", "execution_retro_status": "paper-execution-retro-pending"})
    _write_json(old_bridge, {"status": "ok", "bridge_status": "blocked_missing_directional_signal", "bridge_items": []})
    _write_json(
        new_bridge,
        {
            "status": "ok",
            "bridge_status": "blocked_stale_directional_signal",
            "bridge_items": [{"symbol": "COPPER", "bridge_status": "blocked_stale_directional_signal", "bridge_reasons": ["stale_signal"]}],
        },
    )
    _write_config(
        tmp_path / "config.yaml",
        """
timezone: Asia/Shanghai
universe:
  core:
    - {symbol: "BTCUSDT", asset_class: "crypto"}
""",
    )
    _write_json(tmp_path / "artifacts" / "paper_positions_open.json", {"as_of": "2026-03-10", "positions": []})
    _write_jsonl(tmp_path / "logs" / "paper_execution_ledger.jsonl", [])
    _create_db(tmp_path / "artifacts" / "lie_engine.db", trade_symbols=[], executed_symbols=[])

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--execution-queue-json",
            str(new_queue),
            "--execution-review-json",
            str(new_review),
            "--execution-retro-json",
            str(new_retro),
            "--execution-bridge-json",
            str(new_bridge),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--paper-ledger-path",
            str(tmp_path / "logs" / "paper_execution_ledger.jsonl"),
            "--paper-positions-path",
            str(tmp_path / "artifacts" / "paper_positions_open.json"),
            "--paper-db-path",
            str(tmp_path / "artifacts" / "lie_engine.db"),
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
    assert payload["source_execution_review_artifact"] == str(new_review.resolve())
    assert payload["source_execution_retro_artifact"] == str(new_retro.resolve())
    assert payload["source_execution_bridge_artifact"] == str(new_bridge.resolve())
    assert payload["execution_symbols"] == ["COPPER"]
