from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "build_fast_trade_skill_plan.py"
    spec = importlib.util.spec_from_file_location("build_fast_trade_skill_plan", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_fast_plan_selects_btc_canary() -> None:
    mod = _load_module()
    payload = {
        "_input_artifact": "x.json",
        "as_of": "2026-03-06",
        "sizing_context": {"equity_usdt": 20.0, "equity_source": "test"},
        "tickets": [
            {
                "symbol": "BTCUSDT",
                "date": "2026-03-05",
                "age_days": 1,
                "allowed": False,
                "reasons": ["size_below_min_notional"],
                "signal": {"side": "LONG", "regime": "弱趋势", "confidence": 18.5, "convexity_ratio": 1.6},
                "levels": {"entry_price": 70000.0, "stop_price": 68000.0, "target_price": 73000.0},
                "execution": {"order_type_hint": "LIMIT", "max_slippage_bps": 6.0},
            },
            {
                "symbol": "ETHUSDT",
                "date": "2026-03-05",
                "age_days": 1,
                "allowed": False,
                "reasons": ["stale_signal"],
                "signal": {"side": "SHORT", "regime": "下跌趋势", "confidence": 40.0, "convexity_ratio": 2.0},
            },
        ],
    }
    out = mod.build_fast_plan(
        tickets_payload=payload,
        symbols=("BTCUSDT", "ETHUSDT"),
        equity_usdt_override=None,
        reserve_pct=0.45,
        alloc_pct=0.30,
        min_notional_usdt=5.0,
        max_age_days=14,
        min_confidence=14.0,
        min_convexity=1.2,
        decision_ttl_seconds=300,
    )
    selected = out.get("selected", {})
    assert isinstance(selected, dict)
    assert bool(selected.get("executable", False)) is True
    assert str(selected.get("symbol", "")) == "BTCUSDT"
    assert str(selected.get("side", "")) == "BUY"
    assert float(selected.get("quote_usdt", 0.0)) >= 5.0
    assert len(str(selected.get("idempotency_key", ""))) == 28


def test_script_main_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    ticket_path = review_dir / "20260306T000000Z_signal_to_order_tickets.json"
    ticket_path.write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "sizing_context": {"equity_usdt": 20.0, "equity_source": "test"},
                "tickets": [
                    {
                        "symbol": "XAUUSD",
                        "allowed": False,
                        "reasons": ["signal_not_found"],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_fast_trade_skill_plan.py",
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(out_dir),
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(out_dir.glob("*_fast_trade_skill_plan.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("as_of", "")) == "2026-03-06"
    selected = parsed.get("selected", {})
    assert isinstance(selected, dict)
    assert bool(selected.get("executable", True)) is False


def test_script_main_inherits_core_thresholds_from_config(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    ticket_path = review_dir / "20260306T000000Z_signal_to_order_tickets.json"
    ticket_path.write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "sizing_context": {"equity_usdt": 20.0, "equity_source": "test"},
                "tickets": [
                    {
                        "symbol": "BTCUSDT",
                        "date": "2026-03-05",
                        "age_days": 1,
                        "allowed": False,
                        "reasons": ["size_below_min_notional"],
                        "signal": {"side": "LONG", "confidence": 18.5, "convexity_ratio": 1.6},
                        "levels": {"entry_price": 70000.0, "stop_price": 68000.0, "target_price": 73000.0},
                    }
                ],
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
                "shortline": {
                    "structure_timeframe": "4h",
                    "execution_timeframe": "15m",
                    "structure_engine": "fixed_range_volume_profile_proxy",
                    "flow_confirmation_engine": "cvd_lite",
                    "default_market_state": "Bias_Only",
                    "setup_ready_state": "Setup_Ready",
                    "no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
                    "micro_structure_engine": "ict_sweep_mss",
                    "micro_structure_timeframes": ["1m", "5m"],
                    "liquidity_sweep_required": True,
                    "mss_required": True,
                    "entry_retest_priority": ["FVG", "OB", "Breaker"],
                    "session_liquidity_map": ["asia_high_low", "london_high_low"],
                    "holding_window_minutes": {"min": 15, "max": 180},
                    "location_priority": ["HVN", "POC"],
                    "supported_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_fast_trade_skill_plan.py",
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(out_dir.glob("*_fast_trade_skill_plan.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    policy = parsed.get("policy", {})
    assert isinstance(policy, dict)
    assert float(policy.get("min_confidence", 0.0)) == 60.0
    assert float(policy.get("min_convexity", 0.0)) == 3.0
    assert str(policy.get("min_confidence_source", "")) == str(config_path)
    shortline = policy.get("shortline", {})
    assert isinstance(shortline, dict)
    assert str(shortline.get("structure_timeframe", "")) == "4h"
    assert str(shortline.get("execution_timeframe", "")) == "15m"
    assert str(shortline.get("flow_confirmation_engine", "")) == "cvd_lite"
    assert str(shortline.get("default_market_state", "")) == "Bias_Only"
    assert str(shortline.get("setup_ready_state", "")) == "Setup_Ready"
    assert str(shortline.get("micro_structure_engine", "")) == "ict_sweep_mss"
    assert shortline.get("micro_structure_timeframes") == ["1m", "5m"]
    assert bool(shortline.get("liquidity_sweep_required", False)) is True
    assert bool(shortline.get("mss_required", False)) is True
    assert shortline.get("entry_retest_priority") == ["FVG", "OB", "Breaker"]
    assert shortline.get("session_liquidity_map") == ["asia_high_low", "london_high_low"]
    assert shortline.get("holding_window_minutes") == {"min": 15, "max": 180}
    assert shortline.get("location_priority") == ["HVN", "POC"]
    assert "BNBUSDT" in list(shortline.get("supported_symbols", []))
    selected = parsed.get("selected", {})
    assert isinstance(selected, dict)
    assert bool(selected.get("executable", True)) is False
