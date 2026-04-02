from __future__ import annotations

from datetime import date

import pandas as pd

from lie_engine.backtest.engine import BacktestConfig, run_event_backtest
from lie_engine.models import BacktestResult


def _add_months(d: date, months: int) -> date:
    p = pd.Timestamp(d) + pd.DateOffset(months=months)
    return p.date()


def run_walk_forward_backtest(
    bars: pd.DataFrame,
    start: date,
    end: date,
    trend_thr: float,
    mean_thr: float,
    atr_extreme: float,
    cfg_template: BacktestConfig | None = None,
    train_years: int = 3,
    valid_years: int = 1,
    step_months: int = 3,
) -> BacktestResult:
    windows: list[BacktestResult] = []
    cursor = start

    while True:
        train_end = _add_months(cursor, train_years * 12)
        valid_end = _add_months(train_end, valid_years * 12)
        if valid_end > end:
            break

        result = run_event_backtest(
            bars=bars,
            start=train_end,
            end=valid_end,
            cfg=cfg_template or BacktestConfig(),
            trend_thr=trend_thr,
            mean_thr=mean_thr,
            atr_extreme=atr_extreme,
        )
        windows.append(result)
        cursor = _add_months(cursor, step_months)

    if not windows:
        return run_event_backtest(
            bars=bars,
            start=start,
            end=end,
            cfg=cfg_template or BacktestConfig(),
            trend_thr=trend_thr,
            mean_thr=mean_thr,
            atr_extreme=atr_extreme,
        )

    agg_total = float(sum(r.total_return for r in windows) / len(windows))
    agg_annual = float(sum(r.annual_return for r in windows) / len(windows))
    agg_dd = float(max(r.max_drawdown for r in windows))
    agg_wr = float(sum(r.win_rate for r in windows) / len(windows))
    agg_pf = float(sum(r.profit_factor for r in windows) / len(windows))
    agg_exp = float(sum(r.expectancy for r in windows) / len(windows))
    agg_trades = int(sum(r.trades for r in windows))
    agg_pos = float(sum(r.positive_window_ratio for r in windows) / len(windows))
    agg_viol = int(sum(r.violations for r in windows))

    merged_curve = []
    merged_daily_symbol_exposure = []
    merged_trade_journal = []
    merged_holding_daily_symbol_exposure = []
    for r in windows:
        merged_curve.extend(r.equity_curve)
        merged_daily_symbol_exposure.extend(getattr(r, "daily_symbol_exposure", []) or [])
        merged_trade_journal.extend(getattr(r, "trade_journal", []) or [])
        merged_holding_daily_symbol_exposure.extend(getattr(r, "holding_daily_symbol_exposure", []) or [])

    by_asset: dict[str, float] = {}
    for r in windows:
        for k, v in r.by_asset.items():
            by_asset.setdefault(k, 0.0)
            by_asset[k] += v / len(windows)

    return BacktestResult(
        start=start,
        end=end,
        total_return=agg_total,
        annual_return=agg_annual,
        max_drawdown=agg_dd,
        win_rate=agg_wr,
        profit_factor=agg_pf,
        expectancy=agg_exp,
        trades=agg_trades,
        violations=agg_viol,
        positive_window_ratio=agg_pos,
        equity_curve=merged_curve,
        by_asset=by_asset,
        daily_symbol_exposure=merged_daily_symbol_exposure,
        trade_journal=merged_trade_journal,
        holding_daily_symbol_exposure=merged_holding_daily_symbol_exposure,
    )
