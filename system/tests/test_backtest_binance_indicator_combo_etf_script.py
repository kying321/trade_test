from __future__ import annotations

import importlib.util
from pathlib import Path
import urllib.error

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backtest_binance_indicator_combo_etf.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_combo_etf_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_cvd_lite_proxy_and_rsi_shapes() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        {
            "open": [9.2, 10.2, 11.0, 11.6, 12.2, 13.1, 13.6, 14.1, 15.3, 16.0, 17.1, 17.4, 18.1, 18.9, 19.4, 20.2],
            "high": [10, 11, 12, 12, 13, 14, 14, 15, 16, 17, 18, 18, 19, 20, 20, 21],
            "low": [9, 10, 10.5, 11, 12, 13, 13, 14, 15, 15.5, 16, 17, 18, 18.5, 19, 20],
            "close": [9.5, 10.8, 11.8, 11.5, 12.8, 13.7, 13.9, 14.8, 15.7, 16.8, 17.5, 17.8, 18.7, 19.2, 19.8, 20.7],
            "volume": [100, 110, 120, 130, 140, 150, 160, 150, 145, 155, 160, 165, 170, 175, 180, 190],
        }
    )
    cvd_lite = module.cumulative_volume_delta_proxy_line(frame)
    rsi = module.rsi_wilder(frame["close"], 14)
    assert len(cvd_lite) == len(frame)
    assert len(rsi) == len(frame)
    assert cvd_lite.notna().all()
    assert rsi.between(0, 100).all()


def test_legacy_accumulation_distribution_alias_uses_cvd_lite_proxy() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.2, 10.1],
            "high": [10.5, 10.7, 10.6],
            "low": [9.8, 10.0, 9.9],
            "close": [10.3, 10.1, 10.5],
            "volume": [100, 120, 140],
        }
    )
    assert module.accumulation_distribution_line(frame).equals(module.cumulative_volume_delta_proxy_line(frame))


def test_rank_combo_results_discards_laggy_combo() -> None:
    module = _load_module()
    kept, dropped = module.rank_combo_results(
        [
            {
                "combo_id": "taker_oi_ad_breakout",
                "mode": "breakout",
                "trade_count": 8,
                "win_rate": 0.62,
                "total_return": 0.08,
                "profit_factor": 1.8,
                "lag_metrics": {"avg_lag_bars": 0.7, "timely_hit_rate": 0.7},
                "score": 18.0,
                "discard_reason": None,
            },
            {
                "combo_id": "rsi_breakout",
                "mode": "breakout",
                "trade_count": 10,
                "win_rate": 0.54,
                "total_return": 0.03,
                "profit_factor": 1.1,
                "lag_metrics": {"avg_lag_bars": 2.6, "timely_hit_rate": 0.2},
                "score": 4.0,
                "discard_reason": "laggy_support_resistance_timing",
            },
        ]
    )
    assert [row["combo_id"] for row in kept] == ["taker_oi_ad_breakout"]
    assert [row["combo_id"] for row in dropped] == ["rsi_breakout"]


def test_request_json_retries_then_succeeds(monkeypatch) -> None:
    module = _load_module()

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{\"ok\": true}'

    calls = {"count": 0}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.URLError("timeout")
        return _FakeResponse()

    monkeypatch.setattr(module.urllib.request, "urlopen", _fake_urlopen)
    bucket = module.TokenBucket(rate_per_minute=120, capacity=5)
    payload = module.request_json(url="https://example.com", bucket=bucket, timeout_ms=5000, retries=3)
    assert payload == {"ok": True}
    assert calls["count"] == 3


def test_evaluate_lag_metrics_zero_delay_is_not_treated_as_missing() -> None:
    module = _load_module()
    frame = pd.DataFrame({"x": range(4)})
    long_state = pd.Series([False, True, False, False])
    short_state = pd.Series([False, False, False, False])
    long_event = pd.Series([False, True, False, False])
    short_event = pd.Series([False, False, False, False])
    lag = module.evaluate_lag_metrics(frame, long_state, short_state, long_event, short_event, max_bars=3)
    assert lag["avg_lag_bars"] == 0.0
    assert lag["timely_hit_rate"] == 1.0
    assert lag["laggy"] is False


def test_build_family_takeaway_keeps_measured_and_practitioner_layers_separate() -> None:
    module = _load_module()
    measured, practitioner = module.build_family_takeaway(
        "crypto",
        {
            "ranked_combos": [
                {
                    "combo_id": "rsi_breakout",
                    "avg_total_return": 0.03,
                    "avg_timely_hit_rate": 1.0,
                    "trade_count": 23,
                }
            ],
            "discarded_combos": [{"combo_id": "taker_oi_breakout"}],
        },
    )
    assert "rsi_breakout" in measured
    assert "context/veto filters" in measured
    assert "re-tested on native 24/7 crypto bars" in practitioner


def test_build_family_takeaway_marks_negative_commodity_sample_as_unvalidated() -> None:
    module = _load_module()
    measured, practitioner = module.build_family_takeaway(
        "commodity",
        {
            "ranked_combos": [
                {
                    "combo_id": "ad_rsi_breakout",
                    "avg_total_return": -0.02,
                    "avg_timely_hit_rate": 0.84,
                    "trade_count": 62,
                }
            ],
            "discarded_combos": [],
        },
    )
    assert "No commodity combo produced positive aggregate return" in measured
    assert "CVD-lite proxy plus RSI" in practitioner
    assert "research prioritization" in practitioner


def test_build_family_takeaway_commodity_prefers_positive_combo_in_summary() -> None:
    module = _load_module()
    measured, _ = module.build_family_takeaway(
        "commodity",
        {
            "ranked_combos": [
                {
                    "combo_id": "rsi_breakout",
                    "avg_total_return": -0.02,
                    "avg_timely_hit_rate": 0.84,
                    "trade_count": 62,
                },
                {
                    "combo_id": "ad_breakout",
                    "avg_total_return": 0.01,
                    "avg_timely_hit_rate": 0.81,
                    "trade_count": 18,
                },
            ],
            "discarded_combos": [],
        },
    )
    assert "ad_breakout" in measured
    assert "positive-return commodity setup" in measured
    assert "return=0.0100" in measured
