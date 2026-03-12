from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backtest_binance_indicator_combo_native_crypto.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_combo_native_crypto_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_fetch_futures_klines_parses_binance_rows(monkeypatch) -> None:
    module = _load_module()

    def _fake_request_json(*, url, bucket, timeout_ms):  # noqa: ANN001
        return [
            [1700000000000, "100", "110", "95", "108", "1234", 0, 0, 0, 0, 0, 0],
            [1700003600000, "108", "112", "107", "111", "1567", 0, 0, 0, 0, 0, 0],
        ]

    monkeypatch.setattr(module.combo, "request_json", _fake_request_json)
    frame = module.fetch_futures_klines(
        "BTCUSDT",
        interval="1h",
        limit=100,
        timeout_ms=5000,
        bucket=module.combo.TokenBucket(rate_per_minute=120, capacity=5),
    )
    assert list(frame.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert len(frame) == 2
    assert frame["close"].iloc[-1] == 111.0


def test_render_markdown_mentions_control_note() -> None:
    module = _load_module()
    md = module.render_markdown(
        {
            "as_of": "2026-03-10T00:00:00+00:00",
            "symbol_group": "majors",
            "interval": "1h",
            "lookback_bars": 300,
            "sample_windows": 3,
            "hold_bars": 4,
            "coverage": [{"symbol": "BTCUSDT", "rows": 300}],
            "native_crypto_family": {
                "ranked_combos": [
                    {
                        "combo_id": "rsi_breakout",
                        "combo_id_canonical": "rsi_breakout",
                        "avg_total_return": 0.03,
                        "avg_win_rate": 0.56,
                        "avg_lag_bars": 0.0,
                        "avg_timely_hit_rate": 1.0,
                        "trade_count": 20,
                    }
                ],
                "discarded_combos": [],
            },
            "native_crypto_takeaway": "measured",
            "native_crypto_practitioner_note": "practitioner",
            "control_note": "same combos, different price source",
        }
    )
    assert "same combos, different price source" in md
    assert "rsi_breakout" in md
    assert "canonical=`rsi_breakout`" in md
    assert "symbol group" in md


def test_build_native_takeaway_marks_taker_stack_as_recovered_but_not_leading() -> None:
    module = _load_module()
    measured, practitioner = module.build_native_takeaway(
        {
            "ranked_combos": [
                {
                    "combo_id": "rsi_breakout",
                    "avg_total_return": -0.01,
                    "avg_timely_hit_rate": 1.0,
                    "trade_count": 65,
                },
                {
                    "combo_id": "taker_oi_breakout",
                    "avg_total_return": -0.003,
                    "avg_timely_hit_rate": 0.58,
                    "trade_count": 37,
                },
            ],
            "discarded_combos": [],
        }
    )
    assert "survive the lag filter and enter the ranked set" in measured
    assert "secondary confirmation" in practitioner
    assert "CVD-lite" in practitioner


def test_build_native_takeaway_uses_canonical_name_for_legacy_price_state_combo() -> None:
    module = _load_module()
    measured, practitioner = module.build_native_takeaway(
        {
            "ranked_combos": [
                {
                    "combo_id": "ad_breakout",
                    "avg_total_return": 0.02,
                    "avg_timely_hit_rate": 0.98,
                    "trade_count": 48,
                }
            ],
            "discarded_combos": [
                {
                    "combo_id": "taker_oi_ad_rsi_breakout",
                    "avg_total_return": -0.004,
                    "avg_timely_hit_rate": 0.39,
                    "trade_count": 7,
                }
            ],
        }
    )
    assert "did not rescue taker/OI timing" in measured
    assert "`cvd_breakout` still led" in measured
    assert "CVD-lite" in practitioner


def test_with_canonical_combo_ids_adds_alias_field() -> None:
    module = _load_module()
    enriched = module.with_canonical_combo_ids(
        {
            "ranked_combos": [{"combo_id": "ad_breakout"}],
            "discarded_combos": [{"combo_id": "taker_oi_ad_rsi_breakout"}],
        }
    )
    assert enriched["ranked_combos"][0]["combo_id_canonical"] == "cvd_breakout"
    assert enriched["discarded_combos"][0]["combo_id_canonical"] == "taker_oi_cvd_rsi_breakout"


def test_safe_group_slug_normalizes_for_artifact_names() -> None:
    module = _load_module()
    assert module.safe_group_slug("Majors / BTC-ETH") == "majors_btc_eth"


def test_artifact_stem_for_group_scopes_pruning_by_symbol_group() -> None:
    module = _load_module()
    assert module.artifact_stem_for_group("majors") == "majors_binance_indicator_combo_native_crypto"


def test_build_symbol_frame_raises_deadline_exceeded_on_timeout(monkeypatch) -> None:
    module = _load_module()

    def _slow_fetch(*args, **kwargs):  # noqa: ANN002, ANN003
        import time

        time.sleep(0.05)
        return pd.DataFrame()

    monkeypatch.setattr(module, "fetch_futures_klines", _slow_fetch)

    try:
        module.build_symbol_frame(
            "BTCUSDT",
            interval="1h",
            lookback_bars=100,
            binance_period="1h",
            binance_limit=100,
            timeout_ms=5000,
            bucket=module.combo.TokenBucket(rate_per_minute=120, capacity=5),
            symbol_timeout_seconds=0.01,
        )
    except module.DeadlineExceeded as exc:
        assert "symbol_timeout:BTCUSDT" in str(exc)
    else:
        raise AssertionError("expected DeadlineExceeded")


def test_main_writes_partial_failure_artifact_when_symbol_times_out(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()

    def _fake_build_symbol_frame(symbol, **kwargs):  # noqa: ANN001
        if symbol == "ETHUSDT":
            raise module.DeadlineExceeded("symbol_timeout:ETHUSDT:timeout_seconds_exceeded:1")
        return pd.DataFrame(
            [
                {"ts": pd.Timestamp("2026-03-10T00:00:00Z"), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
            ]
        )

    def _fake_summarize_family(family, asset_frames, hold_bars, sample_windows, window_bars):  # noqa: ANN001
        assert family == "crypto"
        assert list(asset_frames.keys()) == ["BTCUSDT"]
        return {
            "ranked_combos": [{"combo_id": "rsi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 1.0, "trade_count": 5}],
            "discarded_combos": [],
        }

    monkeypatch.setattr(module, "build_symbol_frame", _fake_build_symbol_frame)
    monkeypatch.setattr(module.combo, "summarize_family", _fake_summarize_family)

    rc = module.main(
        [
            "--review-dir",
            str(tmp_path),
            "--symbols",
            "BTCUSDT,ETHUSDT",
            "--symbol-group",
            "custom",
            "--now",
            "2026-03-12T05:30:00+00:00",
        ]
    )
    assert rc == 0
    artifacts = sorted(tmp_path.glob("*_custom_binance_indicator_combo_native_crypto.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["status"] == "partial_failure"
    assert payload["ok"] is False
    assert payload["completed_symbols"] == ["BTCUSDT"]
    assert payload["timed_out_symbols"] == ["ETHUSDT"]
    assert payload["error_count"] == 1
    assert payload["artifact_label"].endswith(":partial_failure")
