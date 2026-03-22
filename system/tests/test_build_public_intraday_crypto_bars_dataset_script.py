from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_public_intraday_crypto_bars_dataset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("public_intraday_crypto_bars_dataset_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_render_markdown_includes_symbol_coverage(tmp_path: Path) -> None:
    module = _load_module()
    dataset_csv = tmp_path / "fixture.csv"
    pd.DataFrame(
        [
            {"ts": "2026-03-19T00:00:00Z", "symbol": "BTCUSDT", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"ts": "2026-03-19T00:15:00Z", "symbol": "BTCUSDT", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ]
    ).to_csv(dataset_csv, index=False)
    payload = {
        "interval": "15m",
        "symbol_count": 1,
        "symbols": ["BTCUSDT"],
        "coverage_start_utc": "2026-03-19T00:00:00Z",
        "coverage_end_utc": "2026-03-19T00:15:00Z",
        "row_count": 2,
        "cadence_minutes": 15,
        "dataset_status": "complete",
        "symbol_coverage": [
            {
                "symbol": "BTCUSDT",
                "rows": 2,
                "start_utc": "2026-03-19T00:00:00Z",
                "end_utc": "2026-03-19T00:15:00Z",
            }
        ],
        "research_note": "note",
        "limitation_note": "limit",
    }
    markdown = module.render_markdown(payload)
    assert "BTCUSDT" in markdown
    assert "15m" in markdown
    assert "complete" in markdown


def test_fetch_binance_futures_bars_pages_backwards_and_trims(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    calls: list[str] = []
    monkeypatch.setattr(module, "MAX_KLINES_PER_REQUEST", 3)
    first_page = [
        [1741393800000, "11", "12", "10.8", "11.7", "140"],
        [1741394700000, "11.7", "12.2", "11.5", "12.0", "160"],
        [1741395600000, "12.0", "12.3", "11.8", "12.1", "170"],
    ]
    second_page = [
        [1741391100000, "9.8", "10.2", "9.6", "10", "90"],
        [1741392000000, "10", "11", "9", "10.5", "100"],
        [1741392900000, "10.5", "11.5", "10", "11", "120"],
    ]
    payloads = [first_page, second_page]

    def fake_http_get_json(*, url: str, timeout_ms: int, bucket):  # type: ignore[no-untyped-def]
        calls.append(url)
        return payloads.pop(0)

    monkeypatch.setattr(module, "http_get_json", fake_http_get_json)
    frame = module.fetch_binance_futures_bars(
        symbol="BTCUSDT",
        interval="15m",
        limit=5,
        timeout_ms=5000,
        bucket=module.TokenBucket(60),
    )
    assert len(calls) == 2
    assert len(frame) == 5
    assert frame["ts"].is_monotonic_increasing
    assert frame["symbol"].eq("BTCUSDT").all()
    assert frame.iloc[0]["close"] == 10.5
    assert frame.iloc[-1]["close"] == 12.1
