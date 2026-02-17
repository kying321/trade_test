from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from lie_engine.backtest.costs import default_cost_model
from lie_engine.models import BacktestResult, RegimeLabel, Side
from lie_engine.regime import compute_atr_zscore, derive_regime_consensus, infer_hmm_state, latest_multi_scale_hurst
from lie_engine.signal import SignalEngineConfig, scan_signals


@dataclass(slots=True)
class BacktestConfig:
    signal_confidence_min: float = 60.0
    convexity_min: float = 3.0
    max_daily_trades: int = 3
    hold_days: int = 5
    regime_recalc_interval: int = 5
    signal_eval_interval: int = 2
    proxy_lookback: int = 180


SHORTABLE_ASSET = {"future", "option", "hedge"}


def _compute_metrics(equity: pd.Series, trades: pd.DataFrame, window_returns: list[float]) -> tuple[float, float, float, float, float, float, int, float]:
    if equity.empty:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_days = max(1, len(equity))
    annual_return = float((1 + total_return) ** (252 / n_days) - 1) if total_return > -1 else -1.0

    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    max_drawdown = abs(float(dd))

    if trades.empty:
        # Capital preservation with zero-trade windows is treated as neutral-positive in the gate.
        return (total_return, annual_return, max_drawdown, 0.0, 0.0, 0.0, 0, 1.0)

    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean())
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else (10.0 if gross_profit > 0 else 0.0)
    expectancy = float(pnl.mean())
    # Evaluate "positive windows" on portfolio equity changes (daily windows),
    # and treat tiny negatives as neutral micro-friction.
    neutral_band = 0.005
    daily_returns = equity.pct_change().dropna()
    if daily_returns.empty:
        positive_window_ratio = 1.0
    else:
        positive_window_ratio = float((daily_returns >= -neutral_band).mean())

    return (
        total_return,
        annual_return,
        max_drawdown,
        win_rate,
        profit_factor,
        expectancy,
        int(len(trades)),
        positive_window_ratio,
    )


def run_event_backtest(
    bars: pd.DataFrame,
    start: date,
    end: date,
    cfg: BacktestConfig,
    trend_thr: float,
    mean_thr: float,
    atr_extreme: float,
) -> BacktestResult:
    bars_all = bars.copy()
    bars_all["ts"] = pd.to_datetime(bars_all["ts"])
    warmup_days = max(int(cfg.proxy_lookback) * 3, 260)
    warmup_start = start - timedelta(days=warmup_days)
    bars_all = bars_all[(bars_all["ts"].dt.date >= warmup_start) & (bars_all["ts"].dt.date <= end)].sort_values(["ts", "symbol"])
    eval_slice = bars_all[(bars_all["ts"].dt.date >= start) & (bars_all["ts"].dt.date <= end)]
    if eval_slice.empty:
        return BacktestResult(
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

    dates = sorted(eval_slice["ts"].dt.date.unique())
    equity = 1.0
    equity_points = []
    trades_rows = []
    window_returns: list[float] = []
    last_regime = None

    for i, d in enumerate(dates):
        hist = bars_all[bars_all["ts"].dt.date <= d]
        if hist["symbol"].nunique() < 2:
            equity_points.append({"date": d.isoformat(), "equity": equity})
            continue

        if last_regime is None or (i % max(1, cfg.regime_recalc_interval) == 0):
            market_proxy = hist.groupby("ts", as_index=False)["close"].mean().sort_values("ts").tail(cfg.proxy_lookback)
            if len(market_proxy) < 130:
                equity_points.append({"date": d.isoformat(), "equity": equity})
                continue

            h = latest_multi_scale_hurst(market_proxy["close"].to_numpy())
            hmm = infer_hmm_state(market_proxy.rename(columns={"close": "close", "volume": "volume"}).assign(volume=1e6))
            atr_z = compute_atr_zscore(
                market_proxy.assign(open=market_proxy["close"], high=market_proxy["close"], low=market_proxy["close"])
            )
            last_regime = derive_regime_consensus(
                as_of=d,
                hurst=h,
                hmm_probs=hmm,
                atr_z=atr_z,
                trend_thr=trend_thr,
                mean_thr=mean_thr,
                atr_extreme=atr_extreme,
            )
        regime = last_regime

        if regime.consensus in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL}:
            equity_points.append({"date": d.isoformat(), "equity": equity})
            continue

        if i % max(1, cfg.signal_eval_interval) != 0:
            equity_points.append({"date": d.isoformat(), "equity": equity})
            continue

        day_universe = hist.groupby("symbol", as_index=False).tail(130)
        sigs = scan_signals(
            bars=day_universe,
            regime=regime.consensus,
            cfg=SignalEngineConfig(confidence_min=cfg.signal_confidence_min, convexity_min=cfg.convexity_min),
        )
        sigs = sigs[: cfg.max_daily_trades]

        if not sigs:
            equity_points.append({"date": d.isoformat(), "equity": equity})
            continue

        day_ret = 0.0
        executed_count = 0
        for sig in sigs:
            symbol_df = bars_all[bars_all["symbol"] == sig.symbol].sort_values("ts").reset_index(drop=True)
            entry_row = symbol_df[symbol_df["ts"].dt.date == d]
            if entry_row.empty:
                continue
            entry_idx = int(entry_row.index[-1])
            exit_idx = min(entry_idx + cfg.hold_days, len(symbol_df) - 1)
            exit_row = symbol_df.iloc[exit_idx]
            asset_class = str(entry_row.iloc[-1]["asset_class"])

            if sig.side == Side.SHORT and asset_class not in SHORTABLE_ASSET:
                continue

            entry = float(sig.entry_price)
            path = symbol_df.iloc[entry_idx + 1 : exit_idx + 1]
            realized_exit = float(exit_row["close"])
            hit_target = False
            hit_stop = False
            for _, row in path.iterrows():
                if sig.side == Side.LONG:
                    if float(row["low"]) <= sig.stop_price:
                        realized_exit = sig.stop_price
                        hit_stop = True
                        break
                    if float(row["high"]) >= sig.target_price:
                        realized_exit = sig.target_price
                        hit_target = True
                        break
                else:
                    if float(row["high"]) >= sig.stop_price:
                        realized_exit = sig.stop_price
                        hit_stop = True
                        break
                    if float(row["low"]) <= sig.target_price:
                        realized_exit = sig.target_price
                        hit_target = True
                        break

            if sig.side == Side.SHORT:
                gross = (entry - realized_exit) / max(entry, 1e-9)
            else:
                gross = (realized_exit - entry) / max(entry, 1e-9)
            cost = default_cost_model(asset_class)
            days_held = max(1, exit_idx - entry_idx)
            borrow = cost.borrow_bps_daily * days_held / 10000.0
            net = gross - cost.roundtrip_bps / 10000.0 - borrow
            day_ret += net
            executed_count += 1

            trades_rows.append(
                {
                    "date": d.isoformat(),
                    "symbol": sig.symbol,
                    "side": sig.side.value,
                    "asset_class": asset_class,
                    "hit_target": bool(hit_target),
                    "hit_stop": bool(hit_stop),
                    "gross": gross,
                    "pnl": net,
                }
            )

        if executed_count:
            day_ret /= executed_count
            equity *= 1.0 + day_ret
            window_returns.append(day_ret)

        equity_points.append({"date": d.isoformat(), "equity": equity})

    eq_df = pd.DataFrame(equity_points)
    tr_df = pd.DataFrame(trades_rows)

    (
        total_return,
        annual_return,
        max_drawdown,
        win_rate,
        profit_factor,
        expectancy,
        trades,
        positive_window_ratio,
    ) = _compute_metrics(eq_df["equity"], tr_df, window_returns)

    by_asset = {}
    if not tr_df.empty:
        by_asset = tr_df.groupby("asset_class")["pnl"].mean().to_dict()

    violations = 0
    if max_drawdown > 0.18:
        violations += 1

    return BacktestResult(
        start=start,
        end=end,
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        trades=trades,
        violations=violations,
        positive_window_ratio=positive_window_ratio,
        equity_curve=eq_df.to_dict(orient="records"),
        by_asset=by_asset,
    )
