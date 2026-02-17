from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd


def make_bars(symbol: str, n: int = 220, start: str = "2025-01-01", trend: float = 0.05, seed: int = 7, asset_class: str = "equity") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)
    base = 50 + np.linspace(0, trend * n, n) + rng.normal(0, 1.2, n).cumsum()
    close = np.maximum(0.5, base)
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.5, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.5, 0.2, n))
    vol = np.maximum(1e5, rng.normal(8e6, 2e6, n))
    return pd.DataFrame(
        {
            "ts": idx,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "source": "test",
            "asset_class": asset_class,
        }
    )


def make_multi_symbol_bars() -> pd.DataFrame:
    frames = [
        make_bars("300750", seed=1, trend=0.06, asset_class="equity"),
        make_bars("002050", seed=2, trend=0.04, asset_class="equity"),
        make_bars("513130", seed=3, trend=0.03, asset_class="etf"),
        make_bars("LC2603", seed=4, trend=-0.02, asset_class="future"),
    ]
    return pd.concat(frames, ignore_index=True)
