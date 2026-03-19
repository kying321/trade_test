#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - fallback for minimal environments
    certifi = None


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
sys.path.insert(0, str(SYSTEM_ROOT / "src"))

from lie_engine.research.real_data import _yfinance_download  # noqa: E402


DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"


class TokenBucket:
    def __init__(self, rate_per_minute: int, capacity: int = 5) -> None:
        self.capacity = max(1, float(capacity))
        self.tokens = self.capacity
        self.rate_per_second = max(1.0, float(rate_per_minute)) / 60.0
        self.last_refill = time.monotonic()

    def take(self) -> None:
        while True:
            now_mono = time.monotonic()
            elapsed = max(0.0, now_mono - self.last_refill)
            if elapsed > 0.0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
                self.last_refill = now_mono
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            wait_seconds = max(0.01, (1.0 - self.tokens) / max(1e-9, self.rate_per_second))
            time.sleep(min(wait_seconds, 0.25))


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _public_https_opener(*, ctx: ssl.SSLContext):
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(context=ctx),
    )


def request_json(*, url: str, bucket: TokenBucket, timeout_ms: int, retries: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "fenlie-indicator-combo-backtest/1.0",
            "Accept": "application/json",
        },
    )
    opener = _public_https_opener(ctx=_public_ssl_context())
    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        bucket.take()
        try:
            with opener.open(request, timeout=max(0.1, timeout_ms / 1000.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt >= max(1, int(retries)) - 1:
                break
            time.sleep(min(0.6 * (attempt + 1), 1.5))
    if last_err is not None:
        raise RuntimeError(f"request_json_failed:{type(last_err).__name__}:{last_err}") from last_err
    raise RuntimeError("request_json_failed:unknown")


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    review_dir.mkdir(parents=True, exist_ok=True)
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda item: item.stat().st_mtime, reverse=True)

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


ETF_PROXY_MAP: dict[str, list[str]] = {
    "BTCUSDT": ["IBIT"],
    "ETHUSDT": ["ETHA"],
    "XAUUSD": ["GLD"],
    "XAGUSD": ["SLV"],
    "COPPER": ["CPER"],
    "BRENTUSD": ["BNO"],
    "WTIUSD": ["DBO", "USL", "USO"],
    "NATGAS": ["UNG"],
}

BINANCE_SERIES_ENDPOINTS: dict[str, str] = {
    "open_interest": "/futures/data/openInterestHist",
    "taker_ratio": "/futures/data/takerlongshortRatio",
    "top_position_ratio": "/futures/data/topLongShortPositionRatio",
    "top_account_ratio": "/futures/data/topLongShortAccountRatio",
    "global_account_ratio": "/futures/data/globalLongShortAccountRatio",
}

CRYPTO_SYMBOLS = ("BTCUSDT", "ETHUSDT")
COMMODITY_SYMBOLS = ("XAUUSD", "XAGUSD", "COPPER", "BRENTUSD", "WTIUSD", "NATGAS")

SOURCE_NOTES: list[dict[str, str]] = [
    {
        "kind": "official",
        "label": "Binance Open Interest History",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics",
        "takeaway": "Open interest is best treated as leverage participation context, not standalone direction.",
    },
    {
        "kind": "official",
        "label": "Binance Taker Buy/Sell Volume",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume",
        "takeaway": "Taker imbalance is the closest public Binance metric to immediate aggressive force.",
    },
    {
        "kind": "official",
        "label": "Binance Top Trader Position Ratio",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Trader-Long-Short-Ratio",
        "takeaway": "Top trader position ratio is crowding context, not a fast trigger.",
    },
    {
        "kind": "official",
        "label": "Binance Top Trader Account Ratio",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Long-Short-Account-Ratio",
        "takeaway": "Account ratio is slower and less useful for support/resistance timing than taker flow.",
    },
    {
        "kind": "official",
        "label": "Binance Global Long/Short Ratio",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Long-Short-Ratio",
        "takeaway": "Global long/short is positioning context and tends to lag break/reclaim decisions.",
    },
    {
        "kind": "internal",
        "label": "Fenlie ICT CVD-lite Factor Spec",
        "url": str((SYSTEM_ROOT / "config" / "ict_cvd_factor_spec.yaml").resolve()),
        "takeaway": "Current stack supports CVD-lite confirmation using public flow proxies, not strict institutional session CVD.",
    },
    {
        "kind": "educational",
        "label": "Investopedia RSI",
        "url": "https://www.investopedia.com/terms/r/rsi.asp",
        "takeaway": "RSI is a lagging oscillator; it helps with state and regime, but not precise support/resistance timing on its own.",
    },
    {
        "kind": "practitioner",
        "label": "CryptoQuant Taker Buy Sell Volume/Ratio Guide",
        "url": "https://userguide.cryptoquant.com/cryptoquant-metrics/market/taker-buy-sell-volume-ratio",
        "takeaway": "Practitioner consensus treats taker imbalance as immediate participation pressure and pairs it with OI for confirmation.",
    },
    {
        "kind": "practitioner",
        "label": "CryptoQuant Open Interest Guide",
        "url": "https://userguide.cryptoquant.com/cryptoquant-metrics/market/open-interest",
        "takeaway": "OI expansion confirms leverage participation; OI decline around reversals often reflects squeeze/cover dynamics instead of fresh trend conviction.",
    },
    {
        "kind": "research",
        "label": "Reconciling Open Interest with Traded Volume in Perpetual Swaps",
        "url": "https://arxiv.org/abs/2310.14973",
        "takeaway": "Open interest quality differs across venues, so OI should remain a context variable rather than a sole trigger.",
    },
]


def load_yf_intraday(symbol: str, *, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for candidate in ETF_PROXY_MAP.get(symbol, []):
        try:
            raw = _yfinance_download(
                candidate,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}:{type(exc).__name__}")
            continue
        if raw is None or raw.empty:
            errors.append(f"{candidate}:empty")
            continue
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(col[0]) for col in frame.columns]
        frame = frame.rename(columns=lambda col: str(col).strip().lower().replace(" ", "_"))
        if "adj_close" not in frame.columns and "adj close" in frame.columns:
            frame = frame.rename(columns={"adj close": "adj_close"})
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.rename_axis("ts").reset_index()
        expected = {"open", "high", "low", "close", "volume"}
        if not expected.issubset(frame.columns):
            errors.append(f"{candidate}:missing_columns")
            continue
        frame = frame[["ts", "open", "high", "low", "close", "volume"]].copy()
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna().sort_values("ts").reset_index(drop=True)
        if not frame.empty:
            return frame, candidate
        errors.append(f"{candidate}:normalized_empty")
    raise RuntimeError(f"No ETF proxy data for {symbol}: {', '.join(errors) or 'unknown'}")


def fetch_binance_indicator_series(
    symbol: str,
    *,
    period: str,
    limit: int,
    timeout_ms: int,
    bucket: TokenBucket,
) -> pd.DataFrame:
    base = "https://fapi.binance.com"
    merged: pd.DataFrame | None = None
    for feature_name, path in BINANCE_SERIES_ENDPOINTS.items():
        url = base + path + "?" + urllib.parse.urlencode({"symbol": symbol, "period": period, "limit": int(limit)})
        payload = request_json(url=url, bucket=bucket, timeout_ms=timeout_ms)
        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"Unexpected payload for {feature_name}:{symbol}")
        frame = pd.DataFrame(payload)
        frame["ts"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
        frame = frame.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        if feature_name == "open_interest":
            frame["open_interest_value"] = pd.to_numeric(frame.get("sumOpenInterestValue"), errors="coerce")
            frame["open_interest_contracts"] = pd.to_numeric(frame.get("sumOpenInterest"), errors="coerce")
            keep_cols = ["ts", "open_interest_value", "open_interest_contracts"]
        elif feature_name == "taker_ratio":
            frame["taker_buy_sell_ratio"] = pd.to_numeric(frame.get("buySellRatio"), errors="coerce")
            frame["taker_buy_vol"] = pd.to_numeric(frame.get("buyVol"), errors="coerce")
            frame["taker_sell_vol"] = pd.to_numeric(frame.get("sellVol"), errors="coerce")
            keep_cols = ["ts", "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"]
        elif feature_name == "top_position_ratio":
            frame["top_position_long_short_ratio"] = pd.to_numeric(frame.get("longShortRatio"), errors="coerce")
            keep_cols = ["ts", "top_position_long_short_ratio"]
        elif feature_name == "top_account_ratio":
            frame["top_account_long_short_ratio"] = pd.to_numeric(frame.get("longShortRatio"), errors="coerce")
            keep_cols = ["ts", "top_account_long_short_ratio"]
        else:
            frame["global_account_long_short_ratio"] = pd.to_numeric(frame.get("longShortRatio"), errors="coerce")
            keep_cols = ["ts", "global_account_long_short_ratio"]
        frame = frame[keep_cols].dropna().reset_index(drop=True)
        merged = frame if merged is None else pd.merge(merged, frame, on="ts", how="outer")
    if merged is None or merged.empty:
        raise RuntimeError(f"No indicator series for {symbol}")
    return merged.sort_values("ts").reset_index(drop=True)


def cvd_lite_bar_delta(frame: pd.DataFrame) -> pd.Series:
    spread = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    body_ratio = ((frame["close"] - frame["open"]) / spread).clip(-1.0, 1.0)
    close_location_bias = (
        ((frame["close"] - frame["low"]) - (frame["high"] - frame["close"])) / spread
    ).clip(-1.0, 1.0)
    delta_proxy = frame["volume"] * ((0.7 * body_ratio.fillna(0.0)) + (0.3 * close_location_bias.fillna(0.0)))
    return delta_proxy.fillna(0.0)


def cumulative_volume_delta_proxy_line(frame: pd.DataFrame) -> pd.Series:
    return cvd_lite_bar_delta(frame).cumsum()


def accumulation_distribution_line(frame: pd.DataFrame) -> pd.Series:
    # Legacy compatibility alias. The active stack now uses a deterministic CVD-lite proxy.
    spread = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    if spread.empty:
        return pd.Series(dtype=float)
    return cumulative_volume_delta_proxy_line(frame)


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / float(window), min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(window), min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_market_frame(
    symbol: str,
    etf: pd.DataFrame,
    etf_proxy: str,
    binance_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = etf.copy()
    frame["symbol"] = symbol
    frame["etf_proxy"] = etf_proxy
    frame["cvd_lite_delta"] = cvd_lite_bar_delta(frame)
    frame["cvd_lite"] = cumulative_volume_delta_proxy_line(frame)
    frame["cvd_lite_slope_3"] = frame["cvd_lite"].diff(3)
    frame["rsi14"] = rsi_wilder(frame["close"], 14)
    frame["volume_z20"] = zscore(frame["volume"], 20)
    frame["ret_1"] = frame["close"].pct_change()
    frame["prev_high_20"] = frame["high"].rolling(20).max().shift(1)
    frame["prev_low_20"] = frame["low"].rolling(20).min().shift(1)
    frame["breakout_up"] = (frame["close"] > frame["prev_high_20"]).fillna(False)
    frame["breakout_down"] = (frame["close"] < frame["prev_low_20"]).fillna(False)
    frame["reclaim_up"] = ((frame["low"] < frame["prev_low_20"]) & (frame["close"] > frame["prev_low_20"])).fillna(False)
    frame["reclaim_down"] = ((frame["high"] > frame["prev_high_20"]) & (frame["close"] < frame["prev_high_20"])).fillna(False)
    frame["price_break_dir"] = np.select(
        [frame["breakout_up"], frame["breakout_down"], frame["reclaim_up"], frame["reclaim_down"]],
        [1, -1, 1, -1],
        default=0,
    )
    frame["rsi_up"] = frame["rsi14"] > 55.0
    frame["rsi_down"] = frame["rsi14"] < 45.0
    frame["cvd_lite_up"] = frame["cvd_lite_slope_3"] > 0.0
    frame["cvd_lite_down"] = frame["cvd_lite_slope_3"] < 0.0
    frame["rsi_reversal_up"] = (frame["rsi14"] < 45.0) & (frame["rsi14"].diff() > 0.0)
    frame["rsi_reversal_down"] = (frame["rsi14"] > 55.0) & (frame["rsi14"].diff() < 0.0)

    if binance_metrics is None or binance_metrics.empty:
        frame["has_binance_metrics"] = False
        frame["taker_buy_sell_ratio"] = np.nan
        frame["open_interest_value"] = np.nan
        frame["open_interest_pct_change"] = np.nan
        frame["top_position_z20"] = np.nan
        frame["top_account_z20"] = np.nan
        frame["global_account_z20"] = np.nan
        frame["crowded_long"] = False
        frame["crowded_short"] = False
        frame["taker_up"] = False
        frame["taker_down"] = False
        frame["oi_up"] = False
        frame["oi_down"] = False
        return frame

    metrics = binance_metrics.copy()
    metrics = metrics.sort_values("ts").reset_index(drop=True)
    metrics["open_interest_pct_change"] = (
        metrics["open_interest_value"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    metrics["top_position_z20"] = zscore(np.log(metrics["top_position_long_short_ratio"].clip(lower=1e-9)), 20)
    metrics["top_account_z20"] = zscore(np.log(metrics["top_account_long_short_ratio"].clip(lower=1e-9)), 20)
    metrics["global_account_z20"] = zscore(np.log(metrics["global_account_long_short_ratio"].clip(lower=1e-9)), 20)
    merged = pd.merge_asof(
        frame.sort_values("ts"),
        metrics.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(hours=2),
    )
    merged["has_binance_metrics"] = merged["taker_buy_sell_ratio"].notna()
    merged["taker_up"] = merged["taker_buy_sell_ratio"] > 1.02
    merged["taker_down"] = merged["taker_buy_sell_ratio"] < 0.98
    merged["oi_up"] = merged["open_interest_pct_change"] > 0.0
    merged["oi_down"] = merged["open_interest_pct_change"] <= 0.0
    merged["crowded_long"] = (
        (merged["top_position_z20"] > 1.0)
        & (merged["top_account_z20"] > 1.0)
        & (merged["global_account_z20"] > 1.0)
    ).fillna(False)
    merged["crowded_short"] = (
        (merged["top_position_z20"] < -1.0)
        & (merged["top_account_z20"] < -1.0)
        & (merged["global_account_z20"] < -1.0)
    ).fillna(False)
    return merged


def _state_breakout_rsi(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return frame["rsi_up"], frame["rsi_down"], frame["breakout_up"], frame["breakout_down"]


def _state_breakout_ad(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return frame["cvd_lite_up"], frame["cvd_lite_down"], frame["breakout_up"], frame["breakout_down"]


def _state_breakout_ad_rsi(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return (
        frame["cvd_lite_up"] & frame["rsi_up"],
        frame["cvd_lite_down"] & frame["rsi_down"],
        frame["breakout_up"],
        frame["breakout_down"],
    )


def _state_breakout_ad_rsi_vol(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    vol_ok = frame["volume_z20"] > 0.0
    return (
        frame["cvd_lite_up"] & frame["rsi_up"] & vol_ok,
        frame["cvd_lite_down"] & frame["rsi_down"] & vol_ok,
        frame["breakout_up"],
        frame["breakout_down"],
    )


def _state_breakout_taker_oi(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return frame["taker_up"] & frame["oi_up"], frame["taker_down"] & frame["oi_up"], frame["breakout_up"], frame["breakout_down"]


def _state_breakout_taker_oi_ad(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return (
        frame["taker_up"] & frame["oi_up"] & frame["cvd_lite_up"],
        frame["taker_down"] & frame["oi_up"] & frame["cvd_lite_down"],
        frame["breakout_up"],
        frame["breakout_down"],
    )


def _state_breakout_taker_oi_ad_rsi(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return (
        frame["taker_up"] & frame["oi_up"] & frame["cvd_lite_up"] & frame["rsi_up"],
        frame["taker_down"] & frame["oi_up"] & frame["cvd_lite_down"] & frame["rsi_down"],
        frame["breakout_up"],
        frame["breakout_down"],
    )


def _state_breakout_crowding_filter(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    long_state = frame["taker_up"] & frame["oi_up"] & frame["cvd_lite_up"] & frame["rsi_up"] & (~frame["crowded_long"])
    short_state = frame["taker_down"] & frame["oi_up"] & frame["cvd_lite_down"] & frame["rsi_down"] & (~frame["crowded_short"])
    return long_state, short_state, frame["breakout_up"], frame["breakout_down"]


def _state_reclaim_ad_rsi(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return (
        frame["cvd_lite_up"] & frame["rsi_reversal_up"],
        frame["cvd_lite_down"] & frame["rsi_reversal_down"],
        frame["reclaim_up"],
        frame["reclaim_down"],
    )


def _state_reclaim_taker(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return (
        frame["cvd_lite_up"] & frame["taker_up"] & frame["oi_down"],
        frame["cvd_lite_down"] & frame["taker_down"] & frame["oi_down"],
        frame["reclaim_up"],
        frame["reclaim_down"],
    )


COMBO_DEFS: list[dict[str, Any]] = [
    {"combo_id": "rsi_breakout", "family": "all", "mode": "breakout", "state_fn": _state_breakout_rsi, "confirmation_indicator": "rsi"},
    {
        "combo_id": "ad_breakout",
        "family": "all",
        "mode": "breakout",
        "state_fn": _state_breakout_ad,
        "confirmation_indicator": "cvd_lite_proxy",
        "legacy_combo_id": "ad_breakout",
    },
    {
        "combo_id": "ad_rsi_breakout",
        "family": "all",
        "mode": "breakout",
        "state_fn": _state_breakout_ad_rsi,
        "confirmation_indicator": "cvd_lite_proxy_plus_rsi",
        "legacy_combo_id": "ad_rsi_breakout",
    },
    {
        "combo_id": "ad_rsi_vol_breakout",
        "family": "commodity",
        "mode": "breakout",
        "state_fn": _state_breakout_ad_rsi_vol,
        "confirmation_indicator": "cvd_lite_proxy_plus_rsi_plus_volume",
        "legacy_combo_id": "ad_rsi_vol_breakout",
    },
    {"combo_id": "taker_oi_breakout", "family": "crypto", "mode": "breakout", "state_fn": _state_breakout_taker_oi, "confirmation_indicator": "taker_oi"},
    {
        "combo_id": "taker_oi_ad_breakout",
        "family": "crypto",
        "mode": "breakout",
        "state_fn": _state_breakout_taker_oi_ad,
        "confirmation_indicator": "taker_oi_plus_cvd_lite_proxy",
        "legacy_combo_id": "taker_oi_ad_breakout",
    },
    {
        "combo_id": "taker_oi_ad_rsi_breakout",
        "family": "crypto",
        "mode": "breakout",
        "state_fn": _state_breakout_taker_oi_ad_rsi,
        "confirmation_indicator": "taker_oi_plus_cvd_lite_proxy_plus_rsi",
        "legacy_combo_id": "taker_oi_ad_rsi_breakout",
    },
    {"combo_id": "crowding_filtered_breakout", "family": "crypto", "mode": "breakout", "state_fn": _state_breakout_crowding_filter, "confirmation_indicator": "taker_oi_plus_cvd_lite_proxy_plus_rsi_plus_crowding_filter"},
    {
        "combo_id": "ad_rsi_reclaim",
        "family": "all",
        "mode": "reclaim",
        "state_fn": _state_reclaim_ad_rsi,
        "confirmation_indicator": "cvd_lite_proxy_plus_rsi",
        "legacy_combo_id": "ad_rsi_reclaim",
    },
    {"combo_id": "taker_reclaim", "family": "crypto", "mode": "reclaim", "state_fn": _state_reclaim_taker, "confirmation_indicator": "taker_oi_plus_cvd_lite_proxy"},
]


def evaluate_lag_metrics(
    frame: pd.DataFrame,
    long_state: pd.Series,
    short_state: pd.Series,
    long_event: pd.Series,
    short_event: pd.Series,
    *,
    max_bars: int = 3,
) -> dict[str, Any]:
    delays: list[int] = []
    total_events = 0
    timely_hits = 0
    for state, event in ((long_state, long_event), (short_state, short_event)):
        event_idx = np.flatnonzero(event.fillna(False).to_numpy())
        state_arr = state.fillna(False).to_numpy()
        for idx in event_idx:
            total_events += 1
            found_delay: int | None = None
            upper = min(len(state_arr), idx + max_bars + 1)
            for probe in range(idx, upper):
                if bool(state_arr[probe]):
                    found_delay = probe - idx
                    break
            if found_delay is None:
                continue
            delays.append(found_delay)
            if found_delay <= 1:
                timely_hits += 1
    avg_delay = float(np.mean(delays)) if delays else None
    timely_hit_rate = float(timely_hits / total_events) if total_events else 0.0
    lag_reference = avg_delay if avg_delay is not None else 99.0
    return {
        "event_count": int(total_events),
        "timely_hits": int(timely_hits),
        "timely_hit_rate": timely_hit_rate,
        "avg_lag_bars": avg_delay,
        "laggy": bool(total_events and (lag_reference > 2.0 or timely_hit_rate < 0.35)),
    }


def generate_trades(
    frame: pd.DataFrame,
    long_signal: pd.Series,
    short_signal: pd.Series,
    *,
    hold_bars: int,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    next_available_idx = 0
    signal_arr_long = long_signal.fillna(False).to_numpy()
    signal_arr_short = short_signal.fillna(False).to_numpy()
    open_arr = frame["open"].to_numpy()
    close_arr = frame["close"].to_numpy()
    ts_arr = frame["ts"].tolist()
    for idx in range(len(frame)):
        if idx < next_available_idx:
            continue
        direction = 0
        if signal_arr_long[idx]:
            direction = 1
        elif signal_arr_short[idx]:
            direction = -1
        if direction == 0:
            continue
        entry_idx = idx + 1
        exit_idx = min(len(frame) - 1, idx + hold_bars)
        if entry_idx >= len(frame) or exit_idx <= entry_idx:
            continue
        entry_px = float(open_arr[entry_idx])
        exit_px = float(close_arr[exit_idx])
        raw_return = (exit_px / entry_px) - 1.0
        signed_return = raw_return if direction > 0 else -raw_return
        trades.append(
            {
                "signal_ts": str(ts_arr[idx]),
                "entry_ts": str(ts_arr[entry_idx]),
                "exit_ts": str(ts_arr[exit_idx]),
                "direction": "long" if direction > 0 else "short",
                "signed_return": float(signed_return),
            }
        )
        next_available_idx = exit_idx + 1
    return trades


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }
    returns = np.array([float(row["signed_return"]) for row in trades], dtype=float)
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf") if gross_profit > 0.0 else 0.0
    return {
        "trade_count": int(len(trades)),
        "win_rate": float((returns > 0.0).mean()),
        "avg_return": float(returns.mean()),
        "total_return": float(returns.sum()),
        "profit_factor": float(profit_factor if math.isfinite(profit_factor) else 9.99),
        "expectancy": float(returns.mean()),
    }


def build_windows(frame: pd.DataFrame, *, sample_windows: int, window_bars: int, warmup_bars: int = 30) -> list[pd.DataFrame]:
    if len(frame) <= warmup_bars + 5:
        return []
    trimmed = frame.iloc[warmup_bars:].reset_index(drop=True)
    windows: list[pd.DataFrame] = []
    for offset in range(sample_windows):
        end = len(trimmed) - (offset * window_bars)
        start = max(0, end - window_bars)
        if end - start < max(12, window_bars // 2):
            continue
        windows.append(trimmed.iloc[start:end].reset_index(drop=True))
    return list(reversed(windows))


def evaluate_combo_on_frame(
    frame: pd.DataFrame,
    combo: dict[str, Any],
    *,
    hold_bars: int,
    sample_windows: int,
    window_bars: int,
) -> dict[str, Any]:
    long_state, short_state, long_event, short_event = combo["state_fn"](frame)
    long_signal = long_state & long_event
    short_signal = short_state & short_event
    lag_metrics = evaluate_lag_metrics(frame, long_state, short_state, long_event, short_event)
    overall_trades = generate_trades(frame, long_signal, short_signal, hold_bars=hold_bars)
    summary = summarize_trades(overall_trades)

    windows = build_windows(frame, sample_windows=sample_windows, window_bars=window_bars)
    window_results: list[dict[str, Any]] = []
    for idx, window in enumerate(windows, start=1):
        w_long_state, w_short_state, w_long_event, w_short_event = combo["state_fn"](window)
        w_long_signal = w_long_state & w_long_event
        w_short_signal = w_short_state & w_short_event
        w_trades = generate_trades(window, w_long_signal, w_short_signal, hold_bars=hold_bars)
        w_summary = summarize_trades(w_trades)
        w_summary["window_id"] = idx
        w_summary["start_ts"] = str(window["ts"].iloc[0])
        w_summary["end_ts"] = str(window["ts"].iloc[-1])
        window_results.append(w_summary)

    window_returns = [float(row["total_return"]) for row in window_results if int(row["trade_count"]) > 0]
    consistency = float(sum(1 for val in window_returns if val > 0.0) / len(window_returns)) if window_returns else 0.0
    score = (
        float(summary["total_return"]) * 100.0
        + float(summary["win_rate"]) * 15.0
        + float(lag_metrics["timely_hit_rate"]) * 10.0
        + consistency * 8.0
        - (float(lag_metrics["avg_lag_bars"]) if lag_metrics["avg_lag_bars"] is not None else 3.0) * 4.0
    )
    discard_reason = None
    if lag_metrics["laggy"]:
        discard_reason = "laggy_support_resistance_timing"
    elif int(summary["trade_count"]) < 3:
        discard_reason = "low_sample_count"
    return {
        "combo_id": str(combo["combo_id"]),
        "confirmation_indicator": str(combo.get("confirmation_indicator") or ""),
        "mode": str(combo["mode"]),
        "trade_count": int(summary["trade_count"]),
        "win_rate": float(summary["win_rate"]),
        "avg_return": float(summary["avg_return"]),
        "total_return": float(summary["total_return"]),
        "profit_factor": float(summary["profit_factor"]),
        "expectancy": float(summary["expectancy"]),
        "lag_metrics": lag_metrics,
        "window_results": window_results,
        "consistency": consistency,
        "score": score,
        "discard_reason": discard_reason,
    }


def rank_combo_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = [row for row in results if not row.get("discard_reason")]
    dropped = [row for row in results if row.get("discard_reason")]
    kept_sorted = sorted(kept, key=lambda row: (float(row["score"]), float(row["total_return"])), reverse=True)
    dropped_sorted = sorted(dropped, key=lambda row: (str(row["discard_reason"]), str(row["combo_id"])))
    return kept_sorted, dropped_sorted


def summarize_family(
    family_name: str,
    asset_frames: dict[str, pd.DataFrame],
    *,
    hold_bars: int,
    sample_windows: int,
    window_bars: int,
) -> dict[str, Any]:
    family_results: list[dict[str, Any]] = []
    combos = [
        combo
        for combo in COMBO_DEFS
        if combo["family"] in {"all", family_name}
    ]
    for combo in combos:
        merged_results: list[dict[str, Any]] = []
        per_asset: list[dict[str, Any]] = []
        for symbol, frame in asset_frames.items():
            result = evaluate_combo_on_frame(
                frame,
                combo,
                hold_bars=hold_bars,
                sample_windows=sample_windows,
                window_bars=window_bars,
            )
            result["symbol"] = symbol
            per_asset.append(result)
            merged_results.append(result)
        if not per_asset:
            continue
        agg_trade_count = int(sum(int(row["trade_count"]) for row in per_asset))
        avg_win_rate = float(np.mean([float(row["win_rate"]) for row in per_asset])) if per_asset else 0.0
        avg_total_return = float(np.mean([float(row["total_return"]) for row in per_asset])) if per_asset else 0.0
        avg_profit_factor = float(np.mean([float(row["profit_factor"]) for row in per_asset])) if per_asset else 0.0
        lag_values = [row["lag_metrics"]["avg_lag_bars"] for row in per_asset if row["lag_metrics"]["avg_lag_bars"] is not None]
        timely_values = [float(row["lag_metrics"]["timely_hit_rate"]) for row in per_asset]
        score = (
            avg_total_return * 100.0
            + avg_win_rate * 15.0
            + (float(np.mean(timely_values)) if timely_values else 0.0) * 10.0
            - (float(np.mean(lag_values)) if lag_values else 3.0) * 4.0
        )
        discard_reason = None
        if any(str(row.get("discard_reason")) == "laggy_support_resistance_timing" for row in per_asset):
            discard_reason = "laggy_support_resistance_timing"
        elif agg_trade_count < max(4, len(per_asset)):
            discard_reason = "low_sample_count"
        family_results.append(
            {
                "combo_id": str(combo["combo_id"]),
                "confirmation_indicator": str(combo.get("confirmation_indicator") or ""),
                "mode": str(combo["mode"]),
                "asset_count": len(per_asset),
                "trade_count": agg_trade_count,
                "avg_win_rate": avg_win_rate,
                "avg_total_return": avg_total_return,
                "avg_profit_factor": avg_profit_factor,
                "avg_lag_bars": float(np.mean(lag_values)) if lag_values else None,
                "avg_timely_hit_rate": float(np.mean(timely_values)) if timely_values else 0.0,
                "score": score,
                "discard_reason": discard_reason,
                "per_asset": per_asset,
            }
        )
    kept, dropped = rank_combo_results(
        [
            {
                "combo_id": row["combo_id"],
                "mode": row["mode"],
                "trade_count": row["trade_count"],
                "win_rate": row["avg_win_rate"],
                "total_return": row["avg_total_return"],
                "profit_factor": row["avg_profit_factor"],
                "lag_metrics": {
                    "avg_lag_bars": row["avg_lag_bars"],
                    "timely_hit_rate": row["avg_timely_hit_rate"],
                },
                "score": row["score"],
                "discard_reason": row["discard_reason"],
                "per_asset": row["per_asset"],
            }
            for row in family_results
        ]
    )
    id_to_full = {row["combo_id"]: row for row in family_results}
    return {
        "family": family_name,
        "ranked_combos": [id_to_full[row["combo_id"]] for row in kept],
        "discarded_combos": [id_to_full[row["combo_id"]] for row in dropped],
    }


def build_family_takeaway(
    family_name: str,
    family_payload: dict[str, Any],
) -> tuple[str, str]:
    ranked = list(family_payload.get("ranked_combos", []))
    discarded = list(family_payload.get("discarded_combos", []))
    if not ranked:
        return (
            f"No {family_name} combo produced enough timely trades.",
            f"Keep {family_name} in observation mode until a timing-qualified combo survives the lag filter.",
        )

    top = ranked[0]
    combo_id = str(top.get("combo_id") or "")
    avg_total_return = float(top.get("avg_total_return") or 0.0)
    timely = float(top.get("avg_timely_hit_rate") or 0.0)
    trades = int(top.get("trade_count") or 0)

    if family_name == "crypto":
        taker_sparse = any(str(row.get("combo_id") or "").startswith("taker_oi") for row in discarded)
        measured = (
            f"In this ETF-proxy sample, `{combo_id}` ranked first with return={avg_total_return:.4f}, "
            f"timely-hit={timely:.2%}, trades={trades}. "
            + (
                "Binance taker/OI stacks stayed too sparse or too delayed for ETF support/resistance timing, so they fit better as context/veto filters here."
                if taker_sparse
                else "ETF proxy timing still favored price-state filters over venue-flow overlays."
            )
        )
        practitioner = (
            "Practitioner prior still supports pairing taker imbalance with open interest, but that hypothesis should be re-tested on native 24/7 crypto bars rather than ETF proxies, because venue/session mismatch is likely suppressing signal density."
        )
        return measured, practitioner

    positive_ranked = [row for row in ranked if float(row.get("avg_total_return") or 0.0) > 0.0]
    if positive_ranked:
        best_positive = positive_ranked[0]
        positive_combo_id = str(best_positive.get("combo_id") or "")
        positive_return = float(best_positive.get("avg_total_return") or 0.0)
        positive_timely = float(best_positive.get("avg_timely_hit_rate") or 0.0)
        positive_trades = int(best_positive.get("trade_count") or 0)
        measured = (
            f"`{positive_combo_id}` was the best positive-return {family_name} setup with return={positive_return:.4f}, "
            f"timely-hit={positive_timely:.2%}, trades={positive_trades}. "
            "On this sample, price-volume confirmation outperformed Binance sentiment overlays."
        )
    else:
        measured = (
            f"No {family_name} combo produced positive aggregate return in this ETF sample. `{combo_id}` was only the least-bad timing-qualified setup "
            f"with return={avg_total_return:.4f}, timely-hit={timely:.2%}, trades={trades}."
        )
    practitioner = (
        "CVD-lite proxy plus RSI remains a reasonable confirmation stack for ETF-style commodity bars, but current results only justify research prioritization, not validation for live deployment."
    )
    return measured, practitioner


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Indicator Combo ETF Backtest",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- interval: `{payload.get('interval')}`",
        f"- period: `{payload.get('period')}`",
        f"- sample windows: `{payload.get('sample_windows')}`",
        f"- hold bars: `{payload.get('hold_bars')}`",
        "",
        "## Source Notes",
    ]
    for row in payload.get("source_notes", []):
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('kind')}` [{row.get('label')}]({row.get('url')})")
        lines.append(f"  - takeaway: {row.get('takeaway')}")

    lines.extend(["", "## ETF Proxy Coverage"])
    for family in payload.get("coverage", []):
        if not isinstance(family, dict):
            continue
        lines.append(f"- `{family.get('symbol')}` -> `{family.get('proxy')}` rows=`{family.get('rows')}`")

    lines.extend(["", "## Recommended Crypto Combos"])
    for row in payload.get("crypto_family", {}).get("ranked_combos", [])[:5]:
        lines.append(
            f"- `{row.get('combo_id')}` return=`{row.get('avg_total_return'):.4f}` "
            f"win=`{row.get('avg_win_rate'):.2%}` lag=`{row.get('avg_lag_bars')}` "
            f"timely=`{row.get('avg_timely_hit_rate'):.2%}`"
        )
    lines.extend(["", "## Recommended Commodity Combos"])
    for row in payload.get("commodity_family", {}).get("ranked_combos", [])[:5]:
        lines.append(
            f"- `{row.get('combo_id')}` return=`{row.get('avg_total_return'):.4f}` "
            f"win=`{row.get('avg_win_rate'):.2%}` lag=`{row.get('avg_lag_bars')}` "
            f"timely=`{row.get('avg_timely_hit_rate'):.2%}`"
        )

    lines.extend(["", "## Discarded As Too Laggy"])
    seen_ids: set[str] = set()
    for family_key in ("crypto_family", "commodity_family"):
        for row in payload.get(family_key, {}).get("discarded_combos", []):
            combo_id = str(row.get("combo_id"))
            if combo_id in seen_ids:
                continue
            seen_ids.add(combo_id)
            lines.append(
                f"- `{combo_id}` reason=`{row.get('discard_reason')}` "
                f"lag=`{row.get('avg_lag_bars')}` timely=`{row.get('avg_timely_hit_rate'):.2%}`"
            )

    lines.extend(
        [
            "",
            "## Conclusions",
            f"- crypto measured takeaway: {payload.get('crypto_takeaway')}",
            f"- crypto practitioner note: {payload.get('crypto_practitioner_note')}",
            f"- commodity measured takeaway: {payload.get('commodity_takeaway')}",
            f"- commodity practitioner note: {payload.get('commodity_practitioner_note')}",
            f"- discard rule: {payload.get('discard_rule')}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest Binance sentiment/flow indicators against ETF intraday proxies.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--period", default="30d")
    parser.add_argument("--interval", default="60m")
    parser.add_argument("--sample-windows", type=int, default=3)
    parser.add_argument("--window-bars", type=int, default=40)
    parser.add_argument("--hold-bars", type=int, default=4)
    parser.add_argument("--binance-limit", type=int, default=300)
    parser.add_argument("--binance-period", default="1h")
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    bucket = TokenBucket(rate_per_minute=max(1, int(args.rpm)), capacity=5)
    coverage: list[dict[str, Any]] = []

    crypto_frames: dict[str, pd.DataFrame] = {}
    commodity_frames: dict[str, pd.DataFrame] = {}

    binance_series: dict[str, pd.DataFrame] = {}
    for symbol in CRYPTO_SYMBOLS:
        binance_series[symbol] = fetch_binance_indicator_series(
            symbol,
            period=str(args.binance_period),
            limit=max(50, int(args.binance_limit)),
            timeout_ms=min(5000, max(100, int(args.timeout_ms))),
            bucket=bucket,
        )

    for symbol in CRYPTO_SYMBOLS + COMMODITY_SYMBOLS:
        etf_frame, proxy = load_yf_intraday(symbol, period=str(args.period), interval=str(args.interval))
        coverage.append({"symbol": symbol, "proxy": proxy, "rows": int(len(etf_frame))})
        market_frame = build_market_frame(symbol, etf_frame, proxy, binance_series.get(symbol))
        if symbol in CRYPTO_SYMBOLS:
            crypto_frames[symbol] = market_frame
        else:
            commodity_frames[symbol] = market_frame

    crypto_family = summarize_family(
        "crypto",
        crypto_frames,
        hold_bars=max(2, int(args.hold_bars)),
        sample_windows=max(1, int(args.sample_windows)),
        window_bars=max(20, int(args.window_bars)),
    )
    commodity_family = summarize_family(
        "commodity",
        commodity_frames,
        hold_bars=max(2, int(args.hold_bars)),
        sample_windows=max(1, int(args.sample_windows)),
        window_bars=max(20, int(args.window_bars)),
    )

    crypto_takeaway, crypto_practitioner_note = build_family_takeaway("crypto", crypto_family)
    commodity_takeaway, commodity_practitioner_note = build_family_takeaway("commodity", commodity_family)

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "interval": str(args.interval),
        "period": str(args.period),
        "sample_windows": int(args.sample_windows),
        "window_bars": int(args.window_bars),
        "hold_bars": int(args.hold_bars),
        "coverage": coverage,
        "source_notes": SOURCE_NOTES,
        "crypto_family": crypto_family,
        "commodity_family": commodity_family,
        "crypto_takeaway": crypto_takeaway,
        "crypto_practitioner_note": crypto_practitioner_note,
        "commodity_takeaway": commodity_takeaway,
        "commodity_practitioner_note": commodity_practitioner_note,
        "discard_rule": "Drop combinations with avg lag > 2 bars or timely-hit rate < 35% when judging breakout/reclaim timing.",
        "artifact_label": "binance-indicator-combo-etf:ok",
    }

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_combo_etf.json"
    md_path = review_dir / f"{stamp}_binance_indicator_combo_etf.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_combo_etf_checksum.json"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(json_path),
        "artifact_sha256": sha256_file(json_path),
        "markdown": str(md_path),
        "markdown_sha256": sha256_file(md_path),
        "generated_at": fmt_utc(runtime_now),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="binance_indicator_combo_etf",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )

    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["artifact_sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
