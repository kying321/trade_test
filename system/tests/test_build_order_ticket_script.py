from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "build_order_ticket.py"
    spec = importlib.util.spec_from_file_location("build_order_ticket", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_main_builds_ticket_artifacts_from_recent5_snapshot(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306_strategy_recent5_signals.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-06T00:00:00Z",
                "start": "2025-12-01",
                "end": "2026-03-06",
                "signals": {
                    "BTCUSDT": [
                        {
                            "date": "2026-03-05",
                            "symbol": "BTCUSDT",
                            "side": "LONG",
                            "regime": "强趋势",
                            "confidence": 72.0,
                            "convexity_ratio": 3.6,
                            "entry_price": 70000.0,
                            "stop_price": 69965.0,
                            "target_price": 70350.0,
                            "factor_flags": ["trend_alignment"],
                            "notes": "test-long",
                        }
                    ],
                    "ETHUSDT": [
                        {
                            "date": "2026-02-01",
                            "symbol": "ETHUSDT",
                            "side": "SHORT",
                            "regime": "下跌趋势",
                            "confidence": 75.0,
                            "convexity_ratio": 3.2,
                            "entry_price": 2000.0,
                            "stop_price": 2100.0,
                            "target_price": 1750.0,
                            "factor_flags": [],
                            "notes": "test-short",
                        }
                    ],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps({"steps": {"account_overview": {"quote_available": 20.0}}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    mode_dir = output_root / "state" / "output" / "daily"
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "2026-03-06_mode_feedback.json").write_text(
        json.dumps({"risk_control": {"risk_multiplier": 1.0}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-06",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "BTCUSDT,ETHUSDT",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("as_of", "")) == "2026-03-06"
    assert str(parsed.get("signal_source", {}).get("kind", "")) == "recent5_review"
    tickets = parsed.get("tickets", [])
    assert isinstance(tickets, list) and len(tickets) == 2
    btc = next(row for row in tickets if str(row.get("symbol", "")) == "BTCUSDT")
    eth = next(row for row in tickets if str(row.get("symbol", "")) == "ETHUSDT")
    assert bool(btc.get("allowed", False)) is True
    assert str((btc.get("execution", {}) or {}).get("mode", "")) == "SPOT_LONG_OR_SELL"
    assert bool(eth.get("allowed", True)) is False
    assert "stale_signal" in list(eth.get("reasons", []))
    checksum = sorted(review_dir.glob("*_signal_to_order_tickets_checksum.json"))
    assert checksum
    checksum_payload = json.loads(checksum[-1].read_text(encoding="utf-8"))
    assert len(checksum_payload.get("files", [])) == 2


def test_script_main_writes_missing_signal_tickets_when_source_absent(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-06",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "BTCUSDT,XAUUSD",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    tickets = parsed.get("tickets", [])
    assert isinstance(tickets, list) and len(tickets) == 2
    for row in tickets:
        assert list(row.get("reasons", [])) == ["signal_not_found"]
        assert bool(row.get("allowed", True)) is False


def test_script_main_falls_back_to_latest_daily_with_matching_symbol_rows(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    daily_dir = output_root / "daily"
    review_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / "2026-03-06_signals.json").write_text("[]\n", encoding="utf-8")
    (daily_dir / "2026-03-05_signals.json").write_text(
        json.dumps(
            [
                {
                    "date": "2026-03-05",
                    "symbol": "ADAUSDT",
                    "side": "LONG",
                    "confidence": 88.0,
                    "convexity_ratio": 3.8,
                    "entry_price": 1.0,
                    "stop_price": 0.95,
                    "target_price": 1.15,
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (daily_dir / "2026-03-04_signals.json").write_text(
        json.dumps(
            [
                {
                    "date": "2026-03-04",
                    "symbol": "BNBUSDT",
                    "side": "LONG",
                    "confidence": 82.0,
                    "convexity_ratio": 3.5,
                    "entry_price": 600.0,
                    "stop_price": 588.0,
                    "target_price": 640.0,
                    "factor_flags": ["trend_alignment"],
                    "notes": "fallback-hit",
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps({"steps": {"account_overview": {"quote_available": 50.0}}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-06",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "BNBUSDT",
            "--max-age-days",
            "14",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    source = parsed.get("signal_source", {})
    assert str(source.get("kind", "")) == "daily_signals"
    assert str(source.get("path", "")).endswith("2026-03-04_signals.json")
    assert bool(source.get("fallback_used", False)) is True
    assert str(source.get("selection_reason", "")) == "matched_symbol_rows"
    tickets = parsed.get("tickets", [])
    assert len(tickets) == 1
    row = tickets[0]
    assert str(row.get("symbol", "")) == "BNBUSDT"
    assert bool(row.get("allowed", False)) is True
    assert list(row.get("reasons", [])) == []


def test_script_main_blocks_proxy_price_reference_only_signal(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    signals_path = review_dir / "20260311T070313Z_commodity_directional_signals.json"
    signals_path.write_text(
        json.dumps(
            {
                "signals": {
                    "XAGUSD": [
                        {
                            "date": "2026-03-10",
                            "symbol": "XAGUSD",
                            "side": "LONG",
                            "regime": "commodity_combo_breakout",
                            "confidence": 72.0,
                            "convexity_ratio": 1.6,
                            "combo_id": "ad_rsi_breakout",
                            "combo_id_canonical": "cvd_rsi_breakout",
                            "confirmation_indicator": "cvd_lite_proxy_plus_rsi",
                            "entry_price": 115.08,
                            "stop_price": 69.856,
                            "target_price": 187.4384,
                            "price_reference_kind": "commodity_proxy_market",
                            "price_reference_source": "yfinance:SI=F",
                            "execution_price_ready": False,
                            "factor_flags": ["combo:ad_rsi_breakout", "combo_canonical:cvd_rsi_breakout"],
                            "notes": "proxy-price-test",
                        }
                    ]
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps({"steps": {"account_overview": {"quote_available": 50.0}}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 14.0, "convexity_min": 1.2}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-11",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--signals-json",
            str(signals_path),
            "--symbols",
            "XAGUSD",
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("signal_source", {}).get("kind", "")) == "explicit"
    assert parsed.get("summary", {}).get("proxy_price_only_count") == 1
    tickets = parsed.get("tickets", [])
    assert len(tickets) == 1
    row = tickets[0]
    assert row["symbol"] == "XAGUSD"
    assert row["allowed"] is False
    assert "proxy_price_reference_only" in row["reasons"]
    assert row["signal"]["price_reference_kind"] == "commodity_proxy_market"
    assert row["signal"]["price_reference_source"] == "yfinance:SI=F"
    assert row["signal"]["execution_price_ready"] is False
    assert row["signal"]["combo_id"] == "ad_rsi_breakout"
    assert row["signal"]["combo_id_canonical"] == "cvd_rsi_breakout"
    assert row["signal"]["confirmation_indicator"] == "cvd_lite_proxy_plus_rsi"


def test_script_main_keeps_fail_closed_when_no_candidate_matches_symbols(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    daily_dir = output_root / "daily"
    review_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / "2026-03-06_signals.json").write_text("[]\n", encoding="utf-8")
    (daily_dir / "2026-03-05_signals.json").write_text(
        json.dumps(
            [
                {
                    "date": "2026-03-05",
                    "symbol": "ADAUSDT",
                    "side": "LONG",
                    "confidence": 88.0,
                    "convexity_ratio": 3.8,
                    "entry_price": 1.0,
                    "stop_price": 0.95,
                    "target_price": 1.15,
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-06",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "BTCUSDT",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    source = parsed.get("signal_source", {})
    assert str(source.get("path", "")).endswith("2026-03-06_signals.json")
    assert str(source.get("selection_reason", "")) == "no_matching_symbol_rows"
    assert int(source.get("candidate_count", 0)) == 2
    tickets = parsed.get("tickets", [])
    assert len(tickets) == 1
    assert list(tickets[0].get("reasons", [])) == ["signal_not_found"]
    assert bool(tickets[0].get("allowed", True)) is False


def test_script_main_uses_artifact_date_when_daily_row_date_missing(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    daily_dir = output_root / "daily"
    review_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / "2026-03-04_signals.json").write_text(
        json.dumps(
            [
                {
                    "symbol": "BNBUSDT",
                    "side": "LONG",
                    "confidence": 82.0,
                    "convexity_ratio": 3.5,
                    "entry_price": 600.0,
                    "stop_price": 588.0,
                    "target_price": 640.0,
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps({"steps": {"account_overview": {"quote_available": 50.0}}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-06",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "BNBUSDT",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    source = parsed.get("signal_source", {})
    assert str(source.get("artifact_date", "")) == "2026-03-04"
    row = parsed.get("tickets", [])[0]
    assert str(row.get("date", "")) == "2026-03-04"
    assert int(row.get("age_days", -1)) == 2
