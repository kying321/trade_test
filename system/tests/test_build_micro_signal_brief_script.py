from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "build_micro_signal_brief.py"
    spec = importlib.util.spec_from_file_location("build_micro_signal_brief", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_brief_marks_long_size_floor_as_micro_tradable() -> None:
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
                "signal": {"side": "LONG", "regime": "弱趋势", "confidence": 18.0, "convexity_ratio": 1.6},
                "levels": {"entry_price": 70000.0, "stop_price": 68000.0, "target_price": 73000.0},
            }
        ],
    }
    out = mod.build_brief(
        tickets_payload=payload,
        equity_usdt_override=None,
        reserve_pct=0.45,
        alloc_pct=0.30,
        min_notional_usdt=5.0,
        max_age_days=14,
    )
    rows = out.get("rows", [])
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert bool(row.get("micro_tradable", False)) is True
    assert str(row.get("action", "")) == "CANARY_BUY"
    assert float(row.get("recommended_quote_usdt", 0.0)) >= 5.0


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
                        "symbol": "ETHUSDT",
                        "date": "2026-03-05",
                        "age_days": 1,
                        "allowed": False,
                        "reasons": ["stale_signal"],
                        "signal": {"side": "SHORT", "regime": "下跌趋势", "confidence": 40.0, "convexity_ratio": 2.0},
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
            "build_micro_signal_brief.py",
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(out_dir),
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(out_dir.glob("*_micro_signal_brief.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("as_of", "")) == "2026-03-06"


def test_script_main_loads_shortline_policy_from_config(tmp_path: Path, monkeypatch) -> None:
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
                        "signal": {"side": "LONG", "confidence": 18.0, "convexity_ratio": 1.6},
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
                    "caution_symbols": ["PAXGUSDT"],
                }
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
            "build_micro_signal_brief.py",
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
    generated = sorted(out_dir.glob("*_micro_signal_brief.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    policy = parsed.get("policy", {})
    assert isinstance(policy, dict)
    assert str(policy.get("config_path", "")) == str(config_path)
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
    assert shortline.get("caution_symbols") == ["PAXGUSDT"]
