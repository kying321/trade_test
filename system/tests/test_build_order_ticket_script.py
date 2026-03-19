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


def test_script_main_prefers_params_live_thresholds_over_mode_feedback_and_config(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    signals_path = tmp_path / "signals.json"
    signals_path.write_text(
        json.dumps(
            [
                {
                    "date": "2026-03-17",
                    "symbol": "SOLUSDT",
                    "side": "LONG",
                    "confidence": 65.0,
                    "convexity_ratio": 2.95,
                    "entry_price": 100.0,
                    "stop_price": 99.0,
                    "target_price": 103.0,
                    "execution_price_ready": True,
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_spot.json").write_text(
        json.dumps(
            {"market": "spot", "steps": {"account_overview": {"quote_available": 50.0}}},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_root / "artifacts" / "params_live.yaml").write_text(
        yaml.safe_dump({"signal_confidence_min": 68.0, "convexity_min": 2.9}, sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "daily").mkdir(parents=True, exist_ok=True)
    (output_root / "daily" / "2026-03-17_mode_feedback.json").write_text(
        json.dumps(
            {
                "runtime_mode": "ultra_short",
                "runtime_params": {"signal_confidence_min": 55.0, "convexity_min": 2.5},
                "risk_control": {"risk_multiplier": 1.0},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0},
                "validation": {"binance_live_takeover_market": "spot"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-17",
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
            "SOLUSDT",
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    thresholds = parsed.get("thresholds", {})
    assert thresholds["min_confidence"] == 68.0
    assert thresholds["min_convexity"] == 2.9
    assert thresholds["threshold_source_kind"] == "params_live"
    assert str(thresholds["threshold_source"]).endswith("params_live.yaml")
    row = parsed.get("tickets", [])[0]
    assert row["allowed"] is False
    assert "confidence_below_threshold" in row["reasons"]
    assert "convexity_below_threshold" not in row["reasons"]


def test_script_main_falls_back_to_mode_feedback_runtime_params_for_thresholds(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    signals_path = tmp_path / "signals.json"
    signals_path.write_text(
        json.dumps(
            [
                {
                    "date": "2026-03-17",
                    "symbol": "SOLUSDT",
                    "side": "LONG",
                    "confidence": 65.5,
                    "convexity_ratio": 2.8,
                    "entry_price": 100.0,
                    "stop_price": 99.0,
                    "target_price": 103.0,
                    "execution_price_ready": True,
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_spot.json").write_text(
        json.dumps(
            {"market": "spot", "steps": {"account_overview": {"quote_available": 50.0}}},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "daily").mkdir(parents=True, exist_ok=True)
    (output_root / "daily" / "2026-03-17_mode_feedback.json").write_text(
        json.dumps(
            {
                "runtime_mode": "swing",
                "runtime_params": {"signal_confidence_min": 66.0, "convexity_min": 2.6},
                "risk_control": {"risk_multiplier": 1.0},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0},
                "validation": {"binance_live_takeover_market": "spot"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-17",
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
            "SOLUSDT",
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    thresholds = parsed.get("thresholds", {})
    assert thresholds["min_confidence"] == 66.0
    assert thresholds["min_convexity"] == 2.6
    assert thresholds["threshold_source_kind"] == "mode_feedback_runtime_params"
    assert thresholds["runtime_mode"] == "swing"
    row = parsed.get("tickets", [])[0]
    assert row["allowed"] is False
    assert "confidence_below_threshold" in row["reasons"]
    assert "convexity_below_threshold" not in row["reasons"]


def test_script_main_prefers_runtime_params_live_thresholds_over_params_live_and_mode_feedback(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    signals_path = tmp_path / "signals.json"
    signals_path.write_text(
        json.dumps(
            [
                {
                    "date": "2026-03-17",
                    "symbol": "SOLUSDT",
                    "side": "LONG",
                    "confidence": 66.0,
                    "convexity_ratio": 2.8,
                    "entry_price": 100.0,
                    "stop_price": 99.0,
                    "target_price": 103.0,
                    "execution_price_ready": True,
                }
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_spot.json").write_text(
        json.dumps(
            {"market": "spot", "steps": {"account_overview": {"quote_available": 50.0}}},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_root / "artifacts" / "params_live.yaml").write_text(
        yaml.safe_dump({"signal_confidence_min": 68.0, "convexity_min": 2.9}, sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "artifacts" / "runtime_params_live.json").write_text(
        json.dumps(
            {
                "runtime_mode": "swing",
                "signal_confidence_min": 65.0,
                "convexity_min": 2.7,
                "source_kind": "runtime_params_live",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "daily").mkdir(parents=True, exist_ok=True)
    (output_root / "daily" / "2026-03-17_mode_feedback.json").write_text(
        json.dumps(
            {
                "runtime_mode": "ultra_short",
                "runtime_params": {"signal_confidence_min": 55.0, "convexity_min": 2.5},
                "risk_control": {"risk_multiplier": 1.0},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0},
                "validation": {"binance_live_takeover_market": "spot"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-17",
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
            "SOLUSDT",
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    thresholds = parsed.get("thresholds", {})
    assert thresholds["min_confidence"] == 65.0
    assert thresholds["min_convexity"] == 2.7
    assert thresholds["threshold_source_kind"] == "runtime_params_live"
    assert str(thresholds["threshold_source"]).endswith("runtime_params_live.json")
    row = parsed.get("tickets", [])[0]
    assert row["allowed"] is True
    assert list(row["reasons"]) == []


def test_derive_equity_usdt_prefers_market_specific_takeover_snapshot(tmp_path: Path) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps(
            {
                "market": "portfolio_margin_um",
                "steps": {"account_overview": {"market": "portfolio_margin_um", "quote_available": 67.0}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_spot.json").write_text(
        json.dumps(
            {
                "market": "spot",
                "steps": {"account_overview": {"market": "spot", "quote_available": 20.0}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    equity, source = mod.derive_equity_usdt(output_root=output_root, override_equity_usdt=None, market="spot")
    assert abs(float(equity) - 20.0) < 1e-12
    assert str(source) == "latest_binance_live_takeover_spot.json.account_overview.quote_available"


def test_derive_equity_usdt_ignores_generic_takeover_from_other_market(tmp_path: Path) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps(
            {
                "market": "portfolio_margin_um",
                "steps": {"account_overview": {"market": "portfolio_margin_um", "quote_available": 67.0}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    equity, source = mod.derive_equity_usdt(output_root=output_root, override_equity_usdt=None, market="spot")
    assert abs(float(equity)) < 1e-12
    assert str(source) == "default_zero"


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


def test_script_main_prefers_crypto_shortline_signal_source_over_stale_recent5(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306_strategy_recent5_signals.json").write_text(
        json.dumps(
            {
                "signals": {
                    "SOLUSDT": [
                        {
                            "date": "2026-02-05",
                            "symbol": "SOLUSDT",
                            "side": "LONG",
                            "confidence": 72.0,
                            "convexity_ratio": 3.6,
                            "entry_price": 150.0,
                            "stop_price": 147.0,
                            "target_price": 156.0,
                        }
                    ]
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260315T113000Z_crypto_shortline_signal_source.json").write_text(
        json.dumps(
            {
                "signals": {
                    "SOLUSDT": [
                        {
                            "date": "2026-03-15",
                            "symbol": "SOLUSDT",
                            "side": "SHORT",
                            "regime": "短线观察",
                            "confidence": 24.0,
                            "convexity_ratio": 1.4,
                            "entry_price": 0.0,
                            "stop_price": 0.0,
                            "target_price": 0.0,
                            "price_reference_kind": "shortline_missing_price_template",
                            "price_reference_source": "crypto_shortline_execution_gate",
                            "execution_price_ready": False,
                            "factor_flags": ["generated_from_crypto_shortline_execution_gate"],
                            "notes": "fresh-shortline-source",
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
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 20.0, "convexity_min": 1.2}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-15",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "SOLUSDT",
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    source = parsed.get("signal_source", {})
    assert str(source.get("kind", "")) == "crypto_shortline_signal_source"
    assert str(source.get("path", "")).endswith("20260315T113000Z_crypto_shortline_signal_source.json")
    assert str(source.get("artifact_date", "")) == "2026-03-15"
    row = parsed.get("tickets", [])[0]
    assert row["symbol"] == "SOLUSDT"
    assert row["date"] == "2026-03-15"
    assert "stale_signal" not in row["reasons"]
    assert "proxy_price_reference_only" in row["reasons"]


def test_script_main_marks_market_scope_mismatch_for_crypto_shortline_source(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260315T113000Z_crypto_shortline_signal_source.json").write_text(
        json.dumps(
            {
                "signals": {
                    "SOLUSDT": [
                        {
                            "date": "2026-03-15",
                            "symbol": "SOLUSDT",
                            "side": "LONG",
                            "regime": "短线观察",
                            "confidence": 84.0,
                            "convexity_ratio": 4.2,
                            "entry_price": 100.0,
                            "stop_price": 99.0,
                            "target_price": 103.0,
                            "execution_price_ready": True,
                        }
                    ]
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260315T113500Z_remote_ticket_actionability_state.json").write_text(
        json.dumps(
            {
                "remote_market": "portfolio_margin_um",
                "ticket_actionability_status": "watch",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_spot.json").write_text(
        json.dumps(
            {
                "market": "spot",
                "steps": {"account_overview": {"market": "spot", "quote_available": 50.0}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0},
                "validation": {"binance_live_takeover_market": "spot"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-15",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "SOLUSDT",
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    source = parsed.get("signal_source", {})
    assert str(source.get("kind", "")) == "crypto_shortline_signal_source"
    assert str(source.get("takeover_market", "")) == "spot"
    assert str(source.get("remote_market", "")) == "portfolio_margin_um"
    assert bool(source.get("market_scope_mismatch", False)) is True
    assert str(source.get("remote_market_source_kind", "")) == "remote_ticket_actionability_state"
    assert str(source.get("remote_market_artifact", "")).endswith("20260315T113500Z_remote_ticket_actionability_state.json")
    assert int(parsed.get("summary", {}).get("market_scope_mismatch_count", 0)) == 1
    row = parsed.get("tickets", [])[0]
    assert row["symbol"] == "SOLUSDT"
    assert row["allowed"] is False
    assert "signal_market_scope_mismatch" in row["reasons"]
    assert row.get("market_scope", {}).get("takeover_market") == "spot"
    assert row.get("market_scope", {}).get("signal_remote_market") == "portfolio_margin_um"
    assert row.get("market_scope", {}).get("mismatch") is True


def test_script_main_marks_portfolio_margin_target_as_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260315T113000Z_crypto_shortline_signal_source.json").write_text(
        json.dumps(
            {
                "signals": {
                    "SOLUSDT": [
                        {
                            "date": "2026-03-15",
                            "symbol": "SOLUSDT",
                            "side": "LONG",
                            "regime": "短线观察",
                            "confidence": 84.0,
                            "convexity_ratio": 4.2,
                            "entry_price": 100.0,
                            "stop_price": 99.0,
                            "target_price": 103.0,
                            "execution_price_ready": True,
                        }
                    ]
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "20260315T113500Z_remote_ticket_actionability_state.json").write_text(
        json.dumps(
            {
                "remote_market": "portfolio_margin_um",
                "ticket_actionability_status": "watch",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_portfolio_margin_um.json").write_text(
        json.dumps(
            {
                "market": "portfolio_margin_um",
                "steps": {"account_overview": {"market": "portfolio_margin_um", "quote_available": 67.0}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0},
                "validation": {"binance_live_takeover_market": "portfolio_margin_um"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-15",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "SOLUSDT",
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    source = parsed.get("signal_source", {})
    assert str(source.get("takeover_market", "")) == "portfolio_margin_um"
    assert bool(source.get("target_market_read_only", False)) is True
    assert int(parsed.get("summary", {}).get("target_market_read_only_count", 0)) == 1
    row = parsed.get("tickets", [])[0]
    assert row["symbol"] == "SOLUSDT"
    assert row["allowed"] is False
    assert row.get("market_scope", {}).get("takeover_market") == "portfolio_margin_um"
    assert row.get("market_scope", {}).get("read_only") is True
    assert "target_market_read_only" in row["reasons"]


def test_script_main_marks_route_not_setup_ready_for_crypto_shortline_source(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260315T113000Z_crypto_shortline_signal_source.json").write_text(
        json.dumps(
            {
                "signals": {
                    "BNBUSDT": [
                        {
                            "date": "2026-03-15",
                            "symbol": "BNBUSDT",
                            "side": "LONG",
                            "regime": "短线观察",
                            "confidence": 84.0,
                            "convexity_ratio": 4.2,
                            "entry_price": 600.0,
                            "stop_price": 597.0,
                            "target_price": 609.0,
                            "price_reference_kind": "shortline_live_bars_template",
                            "price_reference_source": "live-template",
                            "execution_price_ready": True,
                            "execution_state": "Bias_Only",
                            "route_action": "watch_priority_until_long_window_confirms",
                            "route_state": "review",
                            "missing_gates": ["mss", "cvd_confirmation"],
                            "pattern_hint_brief": "imbalance_continuation:imbalance_retest:mss,cvd_confirmation",
                        }
                    ]
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_spot.json").write_text(
        json.dumps(
            {
                "market": "spot",
                "steps": {"account_overview": {"market": "spot", "quote_available": 50.0}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "thresholds": {"signal_confidence_min": 60.0, "convexity_min": 3.0},
                "validation": {"binance_live_takeover_market": "spot"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-15",
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
            "--min-notional-usdt",
            "0.01",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    row = parsed.get("tickets", [])[0]
    assert "route_not_setup_ready" in row["reasons"]
    assert row["signal"]["execution_state"] == "Bias_Only"
    assert row["signal"]["route_action"] == "watch_priority_until_long_window_confirms"
    assert row["signal"]["route_state"] == "review"
    assert row["signal"]["missing_gates"] == ["mss", "cvd_confirmation"]


def test_script_main_auto_narrows_default_symbols_for_crypto_shortline_source(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260315T113000Z_crypto_shortline_signal_source.json").write_text(
        json.dumps(
            {
                "signals": {
                    "SOLUSDT": [
                        {
                            "date": "2026-03-15",
                            "symbol": "SOLUSDT",
                            "side": "LONG",
                            "confidence": 24.0,
                            "convexity_ratio": 1.4,
                            "entry_price": 0.0,
                            "stop_price": 0.0,
                            "target_price": 0.0,
                            "price_reference_kind": "shortline_missing_price_template",
                            "price_reference_source": "crypto_shortline_execution_gate",
                            "execution_price_ready": False,
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
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 20.0, "convexity_min": 1.2}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-15",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    source = parsed.get("signal_source", {})
    assert str(source.get("kind", "")) == "crypto_shortline_signal_source"
    assert str(source.get("symbol_scope_mode", "")) == "source_narrowed_default_symbols"
    tickets = parsed.get("tickets", [])
    assert len(tickets) == 1
    assert tickets[0]["symbol"] == "SOLUSDT"


def test_script_main_uses_explicit_now_for_generated_at_and_artifact_stamp(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260315T113000Z_crypto_shortline_signal_source.json").write_text(
        json.dumps(
            {
                "signals": {
                    "SOLUSDT": [
                        {
                            "date": "2026-03-15",
                            "symbol": "SOLUSDT",
                            "side": "LONG",
                            "confidence": 24.0,
                            "convexity_ratio": 1.4,
                            "entry_price": 0.0,
                            "stop_price": 0.0,
                            "target_price": 0.0,
                            "price_reference_kind": "shortline_missing_price_template",
                            "price_reference_source": "crypto_shortline_execution_gate",
                            "execution_price_ready": False,
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
        yaml.safe_dump({"thresholds": {"signal_confidence_min": 20.0, "convexity_min": 1.2}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_ticket.py",
            "--date",
            "2026-03-16",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(review_dir),
            "--symbols",
            "SOLUSDT",
            "--now",
            "2026-03-16T00:16:00Z",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_signal_to_order_tickets.json"))
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert generated[-1].name.startswith("20260316T001600Z_")
    assert parsed["generated_at_utc"] == "2026-03-16T00:16:00Z"


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
