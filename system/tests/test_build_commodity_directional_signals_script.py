from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_directional_signals.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_commodity_directional_signals", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _combo_payload(symbol: str) -> dict:
    return {
        "commodity_family": {
            "ranked_combos": [
                {
                    "combo_id": "ad_breakout",
                    "confirmation_indicator": "cvd_lite_proxy",
                    "mode": "breakout",
                    "discard_reason": None,
                    "per_asset": [
                        {
                            "symbol": symbol,
                            "score": 21.0,
                            "win_rate": 0.58,
                            "consistency": 0.66,
                            "profit_factor": 1.4,
                            "lag_metrics": {"timely_hit_rate": 0.85},
                        }
                    ],
                }
            ]
        }
    }


def _frame_with_breakout(*, start: str, periods: int, breakout_index: int) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="D", tz="UTC")
    rows: list[dict] = []
    for idx, ts_value in enumerate(ts):
        if idx == breakout_index:
            rows.append(
                {
                    "ts": ts_value,
                    "symbol": "XAUUSD",
                    "open": 100.0,
                    "high": 112.0,
                    "low": 99.0,
                    "close": 111.5,
                    "volume": 2000.0,
                }
            )
        elif idx > breakout_index:
            rows.append(
                {
                    "ts": ts_value,
                    "symbol": "XAUUSD",
                    "open": 108.0,
                    "high": 109.0,
                    "low": 107.0,
                    "close": 108.5,
                    "volume": 900.0,
                }
            )
        else:
            rows.append(
                {
                    "ts": ts_value,
                    "symbol": "XAUUSD",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.8,
                    "volume": 1000.0 + idx,
                }
            )
    return pd.DataFrame(rows)


def test_build_symbol_snapshot_returns_recent_combo_trigger(monkeypatch) -> None:
    mod = _load_module()
    combo_module = mod.load_combo_module()
    frame = _frame_with_breakout(start="2026-02-11", periods=30, breakout_index=29)

    monkeypatch.setattr(
        mod,
        "fetch_symbol_frame",
        lambda **_: (
            frame,
            {
                "bars_source": "fresh",
                "bars_end_date": "2026-03-12",
                "fresh_error": "",
                "cache_meta_path": "",
                "cache_bars_path": "",
            },
        ),
    )

    rows, diagnostics = mod.build_symbol_snapshot(
        symbol="XAUUSD",
        combo_module=combo_module,
        combo_payload=_combo_payload("XAUUSD"),
        output_root=Path("/tmp"),
        as_of=dt.date(2026, 3, 12),
        lookback_days=60,
        max_age_days=14,
        enable_state_carry=False,
        state_carry_max_age_days=5,
    )

    assert diagnostics["signal_status"] == "recent_combo_trigger"
    assert diagnostics["bars_source"] == "fresh"
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "XAUUSD"
    assert row["side"] == "LONG"
    assert row["date"] == "2026-03-12"
    assert row["entry_price"] > row["stop_price"] > 0.0
    assert row["target_price"] > row["entry_price"]
    assert row["convexity_ratio"] == 1.6
    assert row["price_reference_kind"] == "commodity_proxy_market"
    assert row["price_reference_source"] == ""
    assert row["execution_price_ready"] is False
    assert row["combo_id"] == "ad_breakout"
    assert row["combo_id_canonical"] == "cvd_breakout"
    assert row["confirmation_indicator"] == "cvd_lite_proxy"
    assert "combo_canonical:cvd_breakout" in row["factor_flags"]
    assert "confirmation_indicator:cvd_lite_proxy" in row["factor_flags"]
    assert diagnostics["price_reference_kind"] == "commodity_proxy_market"
    assert diagnostics["execution_price_ready"] is False
    assert diagnostics["selected_combo_id_canonical"] == "cvd_breakout"
    assert diagnostics["selected_combo_confirmation_indicator"] == "cvd_lite_proxy"


def test_build_symbol_snapshot_returns_stale_combo_trigger(monkeypatch) -> None:
    mod = _load_module()
    combo_module = mod.load_combo_module()
    frame = _frame_with_breakout(start="2026-01-01", periods=45, breakout_index=21)

    monkeypatch.setattr(
        mod,
        "fetch_symbol_frame",
        lambda **_: (
            frame,
            {
                "bars_source": "cache",
                "bars_end_date": "2026-02-18",
                "fresh_error": "fresh_download_empty",
                "cache_meta_path": "/tmp/cache_meta.json",
                "cache_bars_path": "/tmp/cache_bars.parquet",
            },
        ),
    )

    rows, diagnostics = mod.build_symbol_snapshot(
        symbol="XAUUSD",
        combo_module=combo_module,
        combo_payload=_combo_payload("XAUUSD"),
        output_root=Path("/tmp"),
        as_of=dt.date(2026, 2, 14),
        lookback_days=60,
        max_age_days=14,
        enable_state_carry=False,
        state_carry_max_age_days=5,
    )

    assert diagnostics["signal_status"] == "stale_combo_trigger"
    assert diagnostics["bars_source"] == "cache"
    assert diagnostics["fresh_error"] == "fresh_download_empty"
    assert len(rows) == 1
    row = rows[0]
    assert row["date"] == "2026-01-22"
    assert row["side"] == "LONG"
    assert row["price_reference_kind"] == "commodity_proxy_market"
    assert row["execution_price_ready"] is False
    assert diagnostics["price_reference_kind"] == "commodity_proxy_market"
    assert diagnostics["execution_price_ready"] is False


def test_build_symbol_snapshot_returns_recent_state_carry_when_enabled(monkeypatch) -> None:
    mod = _load_module()
    combo_module = mod.load_combo_module()
    frame = _frame_with_breakout(start="2026-01-01", periods=45, breakout_index=21)

    monkeypatch.setattr(
        mod,
        "fetch_symbol_frame",
        lambda **_: (
            frame,
            {
                "bars_source": "fresh",
                "bars_end_date": "2026-02-14",
                "fresh_error": "",
                "cache_meta_path": "",
                "cache_bars_path": "",
            },
        ),
    )

    rows, diagnostics = mod.build_symbol_snapshot(
        symbol="XAUUSD",
        combo_module=combo_module,
        combo_payload=_combo_payload("XAUUSD"),
        output_root=Path("/tmp"),
        as_of=dt.date(2026, 2, 14),
        lookback_days=60,
        max_age_days=14,
        enable_state_carry=True,
        state_carry_max_age_days=5,
    )

    assert diagnostics["signal_status"] == "recent_combo_state_carry"
    assert diagnostics["signal_kind"] == "state_carry"
    assert diagnostics["trigger_date"] == "2026-02-14"
    assert diagnostics["anchor_trigger_date"] == "2026-01-22"
    assert len(rows) == 1
    row = rows[0]
    assert row["date"] == "2026-02-14"
    assert row["signal_kind"] == "state_carry"
    assert row["anchor_trigger_date"] == "2026-01-22"
    assert row["combo_id_canonical"] == "cvd_breakout"
    assert "signal_kind:state_carry" in row["factor_flags"]
    assert "combo_canonical:cvd_breakout" in row["factor_flags"]
