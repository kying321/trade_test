from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from lie_engine.models import BacktestResult


FACTOR_COLS = ["mom_5", "mom_20", "mom_60", "trend_ma20_60", "vol_20", "range_10", "volume_z20", "rsi_14"]


@dataclass(slots=True)
class CryptoFactorBacktestSummary:
    result: BacktestResult
    objective: float
    dd_ok: bool
    params: dict[str, Any]
    factor_weights: dict[str, float]
    factor_ic_top: list[dict[str, Any]]


@dataclass(slots=True)
class _ScoreRun:
    result: BacktestResult
    sharpe: float
    objective: float


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _zscore_by_date(panel: pd.DataFrame, col: str) -> pd.Series:
    grp = panel.groupby("date")[col]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    z = (panel[col] - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(abs(dd.min()))


def _build_factor_panel(bars: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, grp in bars.groupby("symbol"):
        g = grp.sort_values("ts").copy()
        g["date"] = pd.to_datetime(g["ts"], errors="coerce").dt.date
        g["ret_1d"] = pd.to_numeric(g["close"], errors="coerce").pct_change()
        g["next_ret_1d"] = g["ret_1d"].shift(-1)
        g["mom_5"] = pd.to_numeric(g["close"], errors="coerce").pct_change(5)
        g["mom_20"] = pd.to_numeric(g["close"], errors="coerce").pct_change(20)
        g["mom_60"] = pd.to_numeric(g["close"], errors="coerce").pct_change(60)
        ma20 = pd.to_numeric(g["close"], errors="coerce").rolling(20, min_periods=20).mean()
        ma60 = pd.to_numeric(g["close"], errors="coerce").rolling(60, min_periods=60).mean()
        g["trend_ma20_60"] = ma20 / ma60 - 1.0
        g["vol_20"] = g["ret_1d"].rolling(20, min_periods=20).std(ddof=0)
        close_safe = pd.to_numeric(g["close"], errors="coerce").replace(0.0, np.nan)
        g["range_10"] = ((pd.to_numeric(g["high"], errors="coerce") - pd.to_numeric(g["low"], errors="coerce")) / close_safe).rolling(
            10, min_periods=10
        ).mean()
        lv = np.log(pd.to_numeric(g["volume"], errors="coerce").clip(lower=1.0))
        g["volume_z20"] = (lv - lv.rolling(20, min_periods=20).mean()) / lv.rolling(20, min_periods=20).std(ddof=0)
        g["rsi_14"] = _compute_rsi(pd.to_numeric(g["close"], errors="coerce"), 14) / 100.0 - 0.5
        rows.append(g[["date", "symbol", "ret_1d", "next_ret_1d"] + FACTOR_COLS])
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["date", "symbol", "ret_1d", "next_ret_1d"]).reset_index(drop=True)
    return panel


def _summarize_factor_ic(panel: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in FACTOR_COLS:
        ic_vals: list[float] = []
        valid_days = 0
        for _, g in panel[["date", f, "next_ret_1d"]].dropna().groupby("date"):
            if len(g) < 4:
                continue
            xr = g[f].rank(method="average")
            yr = g["next_ret_1d"].rank(method="average")
            v = xr.corr(yr)
            if v is None or not np.isfinite(v):
                continue
            ic_vals.append(float(v))
            valid_days += 1
        if not ic_vals:
            out.append({"factor": f, "sample_days": 0, "mean_ic": 0.0, "std_ic": 0.0, "ir": 0.0})
            continue
        mean_ic = float(np.mean(ic_vals))
        std_ic = float(np.std(ic_vals))
        ir = float(mean_ic / std_ic * np.sqrt(len(ic_vals))) if std_ic > 1e-9 else 0.0
        out.append({"factor": f, "sample_days": int(valid_days), "mean_ic": mean_ic, "std_ic": std_ic, "ir": ir})
    out.sort(key=lambda x: abs(float(x["mean_ic"])), reverse=True)
    return out


def _select_factor_weights(ic_summary: list[dict[str, Any]], max_factors: int) -> dict[str, float]:
    picked = [x for x in ic_summary if int(x.get("sample_days", 0)) >= 40 and abs(float(x.get("mean_ic", 0.0))) >= 0.01][
        : max(1, int(max_factors))
    ]
    if not picked:
        return {"mom_20": 0.5, "trend_ma20_60": 0.5}
    raw: dict[str, float] = {}
    for row in picked:
        mean_ic = float(row["mean_ic"])
        raw[str(row["factor"])] = float(np.sign(mean_ic) * max(1e-6, abs(mean_ic)))
    norm = sum(abs(v) for v in raw.values())
    return {k: float(v / norm) for k, v in raw.items()}


def _build_score_panel(panel: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = panel[["date", "symbol", "ret_1d"]].copy()
    score = pd.Series(0.0, index=panel.index, dtype=float)
    for f, w in weights.items():
        score += float(w) * _zscore_by_date(panel, f)
    out["score"] = score
    return out


def _run_score_backtest(score_panel: pd.DataFrame, top_k: int, hold_days: int, fee_bps: float, risk_scale: float) -> _ScoreRun:
    ret = score_panel.pivot(index="date", columns="symbol", values="ret_1d").sort_index()
    score = score_panel.pivot(index="date", columns="symbol", values="score").sort_index()
    if ret.empty or score.empty:
        empty = BacktestResult(
            start=date(2000, 1, 1),
            end=date(2000, 1, 1),
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            trades=0,
            violations=0,
            positive_window_ratio=0.0,
            equity_curve=[],
            by_asset={},
        )
        return _ScoreRun(result=empty, sharpe=0.0, objective=-999.0)

    dates = list(ret.index)
    w_prev = pd.Series(0.0, index=ret.columns, dtype=float)
    equity = 1.0
    eq_rows: list[float] = []
    daily_ret: list[float] = []
    trades = 0

    for i, d in enumerate(dates):
        r_today = ret.loc[d].fillna(0.0)
        day_r = float((w_prev * r_today).sum())
        need_rebalance = i > 0 and (i % max(1, int(hold_days)) == 0)
        if need_rebalance:
            s_prev = score.loc[dates[i - 1]].dropna().sort_values(ascending=False)
            selected = list(s_prev.index[: max(1, int(top_k))])
            w_new = pd.Series(0.0, index=ret.columns, dtype=float)
            if selected:
                w_new[selected] = float(max(0.0, min(1.0, risk_scale))) / float(len(selected))
            turnover = float((w_new - w_prev).abs().sum())
            day_r -= turnover * float(fee_bps) / 10000.0
            trades += int(len(selected))
            w_prev = w_new
        equity *= max(0.0, 1.0 + day_r)
        eq_rows.append(float(equity))
        daily_ret.append(float(day_r))

    eq_series = pd.Series(eq_rows, index=dates)
    daily_series = pd.Series(daily_ret, index=dates)
    total_return = float(eq_series.iloc[-1] / max(1e-9, eq_series.iloc[0]) - 1.0)
    n_days = max(1, len(eq_series))
    annual_return = float((1.0 + total_return) ** (365.0 / n_days) - 1.0) if total_return > -1 else -1.0
    dd = _max_drawdown(eq_series)
    mu = float(daily_series.mean()) if not daily_series.empty else 0.0
    sd = float(daily_series.std(ddof=0)) if not daily_series.empty else 0.0
    sharpe = float(mu / sd * np.sqrt(365.0)) if sd > 1e-9 else 0.0
    win_rate = float((daily_series > 0.0).mean()) if not daily_series.empty else 0.0
    gross_profit = float(daily_series[daily_series > 0.0].sum())
    gross_loss = abs(float(daily_series[daily_series < 0.0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else (10.0 if gross_profit > 0 else 0.0)
    positive_window_ratio = float((daily_series >= -0.005).mean()) if not daily_series.empty else 1.0
    expectancy = float(daily_series.mean()) if not daily_series.empty else 0.0
    objective = float(annual_return - 1.5 * dd + 0.2 * sharpe)

    result = BacktestResult(
        start=min(dates),
        end=max(dates),
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=dd,
        win_rate=win_rate,
        profit_factor=float(profit_factor),
        expectancy=expectancy,
        trades=int(trades),
        violations=1 if dd > 0.18 else 0,
        positive_window_ratio=float(positive_window_ratio),
        equity_curve=[{"date": str(d), "equity": float(v)} for d, v in eq_series.items()],
        by_asset={"cash": float(expectancy)},
    )
    return _ScoreRun(result=result, sharpe=sharpe, objective=objective)


def run_crypto_factor_backtest(
    *,
    bars: pd.DataFrame,
    start: date,
    end: date,
    dd_limit: float = 0.05,
    max_factors: int = 4,
) -> CryptoFactorBacktestSummary:
    scoped = bars.copy()
    scoped["ts"] = pd.to_datetime(scoped["ts"], errors="coerce")
    scoped = scoped.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
    scoped = scoped[(scoped["ts"].dt.date >= start) & (scoped["ts"].dt.date <= end)]
    panel = _build_factor_panel(scoped)
    if panel.empty:
        empty = BacktestResult(
            start=start,
            end=end,
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            trades=0,
            violations=0,
            positive_window_ratio=0.0,
            equity_curve=[],
            by_asset={},
        )
        return CryptoFactorBacktestSummary(
            result=empty,
            objective=-999.0,
            dd_ok=False,
            params={},
            factor_weights={},
            factor_ic_top=[],
        )

    ic = _summarize_factor_ic(panel)
    weights = _select_factor_weights(ic, max_factors=max_factors)
    score_panel = _build_score_panel(panel, weights)

    candidates: list[dict[str, Any]] = []
    for top_k in (2, 3, 4):
        for hold_days in (3, 5):
            for fee_bps in (6.0,):
                for risk_scale in (0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25):
                    run = _run_score_backtest(
                        score_panel=score_panel,
                        top_k=top_k,
                        hold_days=hold_days,
                        fee_bps=fee_bps,
                        risk_scale=risk_scale,
                    )
                    candidates.append(
                        {
                            "params": {"top_k": top_k, "hold_days": hold_days, "fee_bps": fee_bps, "risk_scale": risk_scale},
                            "run": run,
                            "dd_ok": bool(run.result.max_drawdown <= dd_limit),
                        }
                    )

    feasible = [x for x in candidates if bool(x["dd_ok"])]
    ranked = sorted(feasible if feasible else candidates, key=lambda x: float(x["run"].objective), reverse=True)
    best = ranked[0]
    run = best["run"]
    result = run.result
    result.start = start
    result.end = end
    return CryptoFactorBacktestSummary(
        result=result,
        objective=float(run.objective),
        dd_ok=bool(best["dd_ok"]),
        params=dict(best["params"]),
        factor_weights=weights,
        factor_ic_top=ic[:10],
    )

