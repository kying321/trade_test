from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd


FRAMEWORK_FIXUPS = [
    "remove_fixed_probabilities",
    "remove_fixed_price_ranges",
    "split_evidence_from_scenarios",
    "surface_missing_coverage_explicitly",
]

EXPECTED_FIELDS = (
    "fuel_oil_inventory",
    "fuel_oil_inventory_delta",
    "lfu_inventory",
    "lfu_inventory_delta",
    "bcti_index",
    "bdti_index",
    "cargo_volume_yoy",
    "coastal_port_throughput_yoy",
    "last_price",
    "rsi14",
    "macd_hist",
    "calendar_spread",
    "participant_net_top2",
    "refinery_run_rate",
    "refinery_margin",
    "environmental_policy_signal",
)


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def _normalize_time_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    ts_col = "ts" if "ts" in out.columns else "date" if "date" in out.columns else None
    if ts_col is None:
        return out.reset_index(drop=True)
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return out


def _latest_row(df: pd.DataFrame) -> Mapping[str, Any]:
    out = _normalize_time_frame(df)
    if out.empty:
        return {}
    return out.iloc[-1].to_dict()


def _previous_row(df: pd.DataFrame) -> Mapping[str, Any]:
    out = _normalize_time_frame(df)
    if len(out) < 2:
        return {}
    return out.iloc[-2].to_dict()


def _latest_non_null(df: pd.DataFrame, column: str) -> float:
    out = _normalize_time_frame(df)
    if out.empty or column not in out.columns:
        return float("nan")
    series = pd.to_numeric(out[column], errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return _safe_float(series.iloc[-1])


def _previous_non_null(df: pd.DataFrame, column: str) -> float:
    out = _normalize_time_frame(df)
    if out.empty or column not in out.columns:
        return float("nan")
    series = pd.to_numeric(out[column], errors="coerce").dropna()
    if len(series) < 2:
        return float("nan")
    return _safe_float(series.iloc[-2])


def _ma(series: pd.Series, window: int) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return _safe_float(numeric.rolling(window, min_periods=max(3, window // 3)).mean().iloc[-1])


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def _rsi14(series: pd.Series) -> float:
    close = pd.to_numeric(series, errors="coerce").dropna()
    if len(close) < 15:
        return float("nan")
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean().iloc[-1]
    avg_loss = loss.rolling(14, min_periods=14).mean().iloc[-1]
    if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
        return float("nan")
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return _safe_float(100.0 - 100.0 / (1.0 + rs))


def _macd(series: pd.Series) -> tuple[float, float, float]:
    close = pd.to_numeric(series, errors="coerce").dropna()
    if len(close) < 26:
        return float("nan"), float("nan"), float("nan")
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return _safe_float(macd_line.iloc[-1]), _safe_float(signal.iloc[-1]), _safe_float(hist.iloc[-1])


def _atr14(df: pd.DataFrame) -> float:
    out = _normalize_time_frame(df)
    if out.empty:
        return float("nan")
    high = pd.to_numeric(out.get("high"), errors="coerce")
    low = pd.to_numeric(out.get("low"), errors="coerce")
    close = pd.to_numeric(out.get("close"), errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    val = tr.rolling(14, min_periods=5).mean().iloc[-1]
    return _safe_float(val)


def _daily_trend(last_price: float, ma20: float, ma60: float) -> str:
    if np.isfinite(last_price) and np.isfinite(ma20) and np.isfinite(ma60):
        if last_price > ma20 > ma60:
            return "up"
        if last_price < ma20 < ma60:
            return "down"
    return "mixed"


def _weekly_trend(df: pd.DataFrame) -> str:
    out = _normalize_time_frame(df)
    if out.empty:
        return "mixed"
    ts_col = "ts" if "ts" in out.columns else "date"
    close = pd.to_numeric(out.get("close"), errors="coerce")
    weekly = (
        pd.DataFrame({"ts": pd.to_datetime(out[ts_col], errors="coerce"), "close": close})
        .dropna(subset=["ts", "close"])
        .set_index("ts")
        .resample("W-FRI")
        .last()
        .dropna()
    )
    if len(weekly) < 4:
        return "mixed"
    ma4 = _ma(weekly["close"], 4)
    ma13 = _ma(weekly["close"], 13)
    last_price = _safe_float(weekly["close"].iloc[-1])
    return _daily_trend(last_price, ma4, ma13)


def _level_candidates(df: pd.DataFrame, kind: str, reference_price: float) -> list[float]:
    out = _normalize_time_frame(df)
    if out.empty:
        return []
    price_col = "low" if kind == "support" else "high"
    series = pd.to_numeric(out.get(price_col), errors="coerce").dropna()
    close = pd.to_numeric(out.get("close"), errors="coerce").dropna()
    if series.empty or close.empty:
        return []
    ma20 = _ma(close, 20)
    ma60 = _ma(close, 60)
    last20 = _safe_float(series.tail(20).min() if kind == "support" else series.tail(20).max())
    last60 = _safe_float(series.tail(60).min() if kind == "support" else series.tail(60).max())
    levels = [x for x in [ma20, ma60, last20, last60] if np.isfinite(x)]
    if kind == "support":
        levels = [x for x in levels if x <= reference_price * 1.002] if np.isfinite(reference_price) else levels
        levels = sorted(set(round(x, 2) for x in levels), reverse=True)
    else:
        levels = [x for x in levels if x >= reference_price * 0.998] if np.isfinite(reference_price) else levels
        levels = sorted(set(round(x, 2) for x in levels))
    return levels[:2]


def _pct_change(latest: float, previous: float) -> float:
    if not np.isfinite(latest) or not np.isfinite(previous) or previous == 0:
        return float("nan")
    return float(latest / previous - 1.0)


def _relative_strength(contract_close: pd.Series, benchmark_close: pd.Series) -> float:
    contract = pd.to_numeric(contract_close, errors="coerce").dropna().reset_index(drop=True)
    benchmark = pd.to_numeric(benchmark_close, errors="coerce").dropna().reset_index(drop=True)
    length = min(len(contract), len(benchmark))
    if length < 25:
        return float("nan")
    contract = contract.tail(length).reset_index(drop=True)
    benchmark = benchmark.tail(length).reset_index(drop=True)
    contract_ret = _pct_change(_safe_float(contract.iloc[-1]), _safe_float(contract.iloc[-21]))
    benchmark_ret = _pct_change(_safe_float(benchmark.iloc[-1]), _safe_float(benchmark.iloc[-21]))
    if not np.isfinite(contract_ret) or not np.isfinite(benchmark_ret):
        return float("nan")
    return float(contract_ret - benchmark_ret)


def _signal_from_two(value_a: float, value_b: float, *, positive_threshold: float = 0.0) -> str:
    if np.isfinite(value_a) and np.isfinite(value_b):
        if value_a > positive_threshold and value_b > positive_threshold:
            return "firm"
        if value_a < positive_threshold and value_b < positive_threshold:
            return "soft"
    return "mixed"


def _member_position_sum(df: pd.DataFrame, value_col: str) -> tuple[float, float]:
    out = _as_frame(df)
    if out.empty or value_col not in out.columns:
        return float("nan"), float("nan")
    vals = pd.to_numeric(out.get(value_col), errors="coerce").dropna()
    delta = pd.to_numeric(out.get("比上交易增减"), errors="coerce").dropna()
    return _safe_float(vals.head(2).sum()), _safe_float(delta.head(2).sum())


def build_fuel_oil_2607_input_packet(
    *,
    contract_focus: str,
    benchmark_contract: str,
    deferred_contract: str,
    macro_frame: pd.DataFrame,
    contract_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    deferred_daily: pd.DataFrame,
    spot_snapshot: Mapping[str, Any] | None,
    benchmark_spot_snapshot: Mapping[str, Any] | None,
    member_rank_payload: Mapping[str, pd.DataFrame] | None,
    report_text: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    macro = _normalize_time_frame(_as_frame(macro_frame))
    contract = _normalize_time_frame(_as_frame(contract_daily))
    benchmark = _normalize_time_frame(_as_frame(benchmark_daily))
    deferred = _normalize_time_frame(_as_frame(deferred_daily))
    macro_last = _latest_row(macro)
    macro_prev = _previous_row(macro)
    contract_last = _latest_row(contract)
    contract_prev = _previous_row(contract)
    benchmark_last = _latest_row(benchmark)
    benchmark_prev = _previous_row(benchmark)
    deferred_last = _latest_row(deferred)
    spot = dict(spot_snapshot or {})
    benchmark_spot = dict(benchmark_spot_snapshot or {})
    member_rank_payload = dict(member_rank_payload or {})

    contract_close = pd.to_numeric(contract.get("close"), errors="coerce")
    benchmark_close = pd.to_numeric(benchmark.get("close"), errors="coerce")
    last_close = _safe_float(spot.get("current_price"), _safe_float(contract_last.get("close")))
    prev_settle = _safe_float(spot.get("last_settle_price"), _safe_float(contract_prev.get("close")))
    benchmark_last_price = _safe_float(
        benchmark_spot.get("current_price"),
        _safe_float(benchmark_last.get("close")),
    )
    benchmark_prev_settle = _safe_float(
        benchmark_spot.get("last_settle_price"),
        _safe_float(benchmark_prev.get("close")),
    )
    deferred_last_price = _safe_float(deferred_last.get("close"))
    ma5 = _ma(contract_close, 5)
    ma20 = _ma(contract_close, 20)
    ma60 = _ma(contract_close, 60)
    macd_line, macd_signal, macd_hist = _macd(contract_close)
    rsi14 = _rsi14(contract_close)
    atr14 = _atr14(contract)
    support_levels = _level_candidates(contract, "support", last_close)
    resistance_levels = _level_candidates(contract, "resistance", last_close)
    rel20 = _relative_strength(contract_close, benchmark_close)
    relative_strength_score = _clamp01((0.08 + (rel20 if np.isfinite(rel20) else 0.0)) / 0.16)

    inventory = _latest_non_null(macro, "fuel_oil_inventory")
    inventory_delta = _latest_non_null(macro, "fuel_oil_inventory_delta")
    lfu_inventory = _latest_non_null(macro, "lfu_inventory")
    lfu_inventory_delta = _latest_non_null(macro, "lfu_inventory_delta")
    bcti = _latest_non_null(macro, "bcti_index")
    bdti = _latest_non_null(macro, "bdti_index")
    bdi = _latest_non_null(macro, "bdi_index")
    bcti_prev = _previous_non_null(macro, "bcti_index")
    bdti_prev = _previous_non_null(macro, "bdti_index")
    bdi_prev = _previous_non_null(macro, "bdi_index")
    cargo_yoy = _latest_non_null(macro, "cargo_volume_yoy")
    port_yoy = _latest_non_null(macro, "coastal_port_throughput_yoy")

    inventory_signal = "mixed"
    if np.isfinite(inventory_delta) and np.isfinite(lfu_inventory_delta):
        if inventory_delta < 0 and lfu_inventory_delta <= 0:
            inventory_signal = "tightening"
        elif inventory_delta > 0 and lfu_inventory_delta >= 0:
            inventory_signal = "loosening"
    freight_signal = _signal_from_two(
        _pct_change(bcti, bcti_prev) if np.isfinite(bcti_prev) else (1.0 if bcti >= 700 else -1.0),
        _pct_change(bdti, bdti_prev) if np.isfinite(bdti_prev) else (1.0 if bdti >= 900 else -1.0),
    )
    demand_signal = _signal_from_two(cargo_yoy, port_yoy)
    demand_signal_source = "direct_transport"
    bdi_change = _pct_change(bdi, bdi_prev)
    if demand_signal == "mixed" and not (np.isfinite(cargo_yoy) and np.isfinite(port_yoy)):
        demand_signal_source = "proxy_bdi"
        if np.isfinite(bdi_change):
            if bdi_change >= 0.05:
                demand_signal = "firm"
            elif bdi_change <= -0.05:
                demand_signal = "soft"
            else:
                demand_signal = "mixed"
        elif np.isfinite(bdi):
            demand_signal = "firm" if bdi >= 1800 else "soft" if bdi <= 1200 else "mixed"
        else:
            demand_signal = "mixed"

    long_top2, long_change_top2 = _member_position_sum(_as_frame(member_rank_payload.get("多单持仓")), "多单持仓")
    short_top2, short_change_top2 = _member_position_sum(_as_frame(member_rank_payload.get("空单持仓")), "空单持仓")
    volume_top2, volume_change_top2 = _member_position_sum(_as_frame(member_rank_payload.get("成交量")), "成交量")
    participant_net = long_top2 - short_top2 if np.isfinite(long_top2) and np.isfinite(short_top2) else float("nan")
    participant_bias = (
        "net_long"
        if np.isfinite(participant_net) and participant_net > 0
        else "net_short"
        if np.isfinite(participant_net) and participant_net < 0
        else "balanced"
    )

    covered = {
        "fuel_oil_inventory": np.isfinite(inventory),
        "fuel_oil_inventory_delta": np.isfinite(inventory_delta),
        "lfu_inventory": np.isfinite(lfu_inventory),
        "lfu_inventory_delta": np.isfinite(lfu_inventory_delta),
        "bcti_index": np.isfinite(bcti),
        "bdti_index": np.isfinite(bdti),
        "cargo_volume_yoy": np.isfinite(cargo_yoy),
        "coastal_port_throughput_yoy": np.isfinite(port_yoy),
        "last_price": np.isfinite(last_close),
        "rsi14": np.isfinite(rsi14),
        "macd_hist": np.isfinite(macd_hist),
        "calendar_spread": np.isfinite(last_close) and np.isfinite(deferred_last_price),
        "participant_net_top2": np.isfinite(participant_net),
        "refinery_run_rate": False,
        "refinery_margin": False,
        "environmental_policy_signal": False,
    }
    missing_fields = [field for field in EXPECTED_FIELDS if not covered.get(field, False)]

    coverage_ratio = float(sum(1 for value in covered.values() if value) / max(1, len(covered)))
    generated_at_utc = _build_iso_utc_timestamp(generated_at)
    as_of_date = ""
    for candidate in [contract_last.get("date"), contract_last.get("ts"), generated_at_utc]:
        ts = pd.to_datetime(candidate, errors="coerce")
        if pd.notna(ts):
            as_of_date = ts.date().isoformat()
            break

    return {
        "generated_at_utc": generated_at_utc,
        "as_of_date": as_of_date,
        "contract_focus": str(contract_focus or "").upper(),
        "benchmark_contract": str(benchmark_contract or "").upper(),
        "deferred_contract": str(deferred_contract or "").upper(),
        "coverage": {
            "coverage_ratio": coverage_ratio,
            "covered_fields": [field for field, ok in covered.items() if ok],
            "missing_fields": missing_fields,
        },
        "framework_fixups": list(FRAMEWORK_FIXUPS),
        "report_digest": str(report_text or "").strip()[:240],
        "price_snapshot": {
            "last_price": last_close,
            "prev_settle": prev_settle,
            "contract_pct_change_1d": _pct_change(last_close, prev_settle),
            "benchmark_last_price": benchmark_last_price,
            "benchmark_prev_settle": benchmark_prev_settle,
            "benchmark_pct_change_1d": _pct_change(benchmark_last_price, benchmark_prev_settle),
            "calendar_spread": last_close - deferred_last_price if np.isfinite(last_close) and np.isfinite(deferred_last_price) else float("nan"),
            "fuel_sc_ratio": last_close / benchmark_last_price if np.isfinite(last_close) and np.isfinite(benchmark_last_price) and benchmark_last_price != 0 else float("nan"),
        },
        "fundamental_snapshot": {
            "fuel_oil_inventory": inventory,
            "fuel_oil_inventory_delta": inventory_delta,
            "lfu_inventory": lfu_inventory,
            "lfu_inventory_delta": lfu_inventory_delta,
            "bcti_index": bcti,
            "bdti_index": bdti,
            "bdi_index": bdi,
            "bdi_change_1step": bdi_change,
            "cargo_volume_yoy": cargo_yoy,
            "coastal_port_throughput_yoy": port_yoy,
            "inventory_signal": inventory_signal,
            "freight_signal": freight_signal,
            "demand_signal": demand_signal,
            "demand_signal_source": demand_signal_source,
            "relative_strength_score": relative_strength_score,
        },
        "technical_snapshot": {
            "last_price": last_close,
            "last_volume": _safe_float(spot.get("volume"), _safe_float(contract_last.get("volume"))),
            "last_hold": _safe_float(spot.get("hold"), _safe_float(contract_last.get("hold"))),
            "volume_change_1d": _pct_change(_safe_float(contract_last.get("volume")), _safe_float(contract_prev.get("volume"))),
            "hold_change_1d": _pct_change(_safe_float(contract_last.get("hold")), _safe_float(contract_prev.get("hold"))),
            "moving_averages": {"ma5": ma5, "ma20": ma20, "ma60": ma60},
            "daily_trend": _daily_trend(last_close, ma20, ma60),
            "weekly_trend": _weekly_trend(contract),
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "atr14": atr14,
            "rsi14": rsi14,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "benchmark_relative_strength_20d": rel20,
            "calendar_spread": last_close - deferred_last_price if np.isfinite(last_close) and np.isfinite(deferred_last_price) else float("nan"),
        },
        "participant_snapshot": {
            "long_top2": long_top2,
            "short_top2": short_top2,
            "net_top2": participant_net,
            "long_change_top2": long_change_top2,
            "short_change_top2": short_change_top2,
            "volume_top2": volume_top2,
            "volume_change_top2": volume_change_top2,
            "long_short_bias": participant_bias,
            "leaders": {
                "volume": str(_as_frame(member_rank_payload.get("成交量")).head(1).get("会员简称", pd.Series([""])).iloc[0] or ""),
                "long": str(_as_frame(member_rank_payload.get("多单持仓")).head(1).get("会员简称", pd.Series([""])).iloc[0] or ""),
                "short": str(_as_frame(member_rank_payload.get("空单持仓")).head(1).get("会员简称", pd.Series([""])).iloc[0] or ""),
            },
        },
    }
