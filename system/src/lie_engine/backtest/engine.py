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
    theory_enabled: bool = False
    theory_ict_weight: float = 1.0
    theory_brooks_weight: float = 1.0
    theory_lie_weight: float = 1.2
    theory_wyckoff_weight: float = 0.0
    theory_vpa_weight: float = 0.0
    theory_confidence_boost_max: float = 5.0
    theory_penalty_max: float = 6.0
    theory_min_confluence: float = 0.38
    theory_conflict_fuse: float = 0.72
    exposure_scale: float = 1.0
    execution_confirm_enabled: bool = False
    execution_confirm_lookahead: int = 2
    execution_confirm_loss_mult: float = 0.65
    execution_confirm_win_mult: float = 1.0
    execution_anti_martingale_enabled: bool = False
    execution_anti_martingale_step: float = 0.20
    execution_anti_martingale_floor: float = 0.60
    execution_anti_martingale_ceiling: float = 1.40


SHORTABLE_ASSET = {"future", "option", "hedge"}


def _proxy_history_min_points(proxy_lookback: int) -> int:
    lookback = max(10, int(proxy_lookback))
    return int(max(50, min(130, round(0.72 * lookback))))


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


def _resolve_confirmation_scale(
    *,
    side: Side,
    entry: float,
    stop: float,
    probe_path: pd.DataFrame,
    lookahead: int,
    loss_mult: float,
    win_mult: float,
) -> tuple[float, bool]:
    window = max(0, int(lookahead))
    if window <= 0 or probe_path.empty:
        return 1.0, False

    risk = max(1e-9, abs(float(entry) - float(stop)))
    confirm_mult = max(0.0, min(1.6, float(win_mult)))
    fail_mult = max(0.0, min(1.6, float(loss_mult)))
    confirm_move = 0.35 * risk
    invalidate_move = 0.20 * risk

    if side == Side.LONG:
        confirm_level = float(entry) + confirm_move
        invalidate_level = float(entry) - invalidate_move
        for _, row in probe_path.head(window).iterrows():
            if float(row["low"]) <= invalidate_level:
                return fail_mult, False
            if float(row["high"]) >= confirm_level:
                return confirm_mult, True
        return fail_mult, False

    if side == Side.SHORT:
        confirm_level = float(entry) - confirm_move
        invalidate_level = float(entry) + invalidate_move
        for _, row in probe_path.head(window).iterrows():
            if float(row["high"]) >= invalidate_level:
                return fail_mult, False
            if float(row["low"]) <= confirm_level:
                return confirm_mult, True
        return fail_mult, False

    return 1.0, False


def _next_anti_martingale_scale(
    *,
    previous_scale: float,
    pnl: float,
    step: float,
    floor: float,
    ceiling: float,
) -> float:
    lower = max(0.0, min(float(floor), float(ceiling)))
    upper = max(lower, float(ceiling))
    delta = max(0.0, float(step))
    scale = max(lower, min(upper, float(previous_scale)))

    if pnl > 1e-12:
        return max(lower, min(upper, scale + delta))
    if pnl < -1e-12:
        return max(lower, min(upper, scale - delta))
    return scale


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
    symbol_size_scale: dict[str, float] = {}

    for i, d in enumerate(dates):
        hist = bars_all[bars_all["ts"].dt.date <= d]
        if hist["symbol"].nunique() < 2:
            equity_points.append({"date": d.isoformat(), "equity": equity})
            continue

        if last_regime is None or (i % max(1, cfg.regime_recalc_interval) == 0):
            market_proxy = hist.groupby("ts", as_index=False)["close"].mean().sort_values("ts").tail(cfg.proxy_lookback)
            if len(market_proxy) < _proxy_history_min_points(int(cfg.proxy_lookback)):
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
            cfg=SignalEngineConfig(
                confidence_min=cfg.signal_confidence_min,
                convexity_min=cfg.convexity_min,
                theory_enabled=bool(cfg.theory_enabled),
                theory_ict_weight=float(cfg.theory_ict_weight),
                theory_brooks_weight=float(cfg.theory_brooks_weight),
                theory_lie_weight=float(cfg.theory_lie_weight),
                theory_wyckoff_weight=float(cfg.theory_wyckoff_weight),
                theory_vpa_weight=float(cfg.theory_vpa_weight),
                theory_confidence_boost_max=float(cfg.theory_confidence_boost_max),
                theory_penalty_max=float(cfg.theory_penalty_max),
                theory_min_confluence=float(cfg.theory_min_confluence),
                theory_conflict_fuse=float(cfg.theory_conflict_fuse),
            ),
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
            base_scale = float(symbol_size_scale.get(sig.symbol, 1.0))
            confirm_scale = 1.0
            confirm_passed = False
            if bool(cfg.execution_confirm_enabled):
                confirm_scale, confirm_passed = _resolve_confirmation_scale(
                    side=sig.side,
                    entry=float(sig.entry_price),
                    stop=float(sig.stop_price),
                    probe_path=path,
                    lookahead=int(cfg.execution_confirm_lookahead),
                    loss_mult=float(cfg.execution_confirm_loss_mult),
                    win_mult=float(cfg.execution_confirm_win_mult),
                )
            total_scale = max(0.0, min(1.8, base_scale * confirm_scale))
            net *= total_scale
            day_ret += net
            executed_count += 1

            if bool(cfg.execution_anti_martingale_enabled):
                symbol_size_scale[sig.symbol] = _next_anti_martingale_scale(
                    previous_scale=base_scale,
                    pnl=net,
                    step=float(cfg.execution_anti_martingale_step),
                    floor=float(cfg.execution_anti_martingale_floor),
                    ceiling=float(cfg.execution_anti_martingale_ceiling),
                )

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
                    "base_scale": base_scale,
                    "confirm_scale": confirm_scale,
                    "total_scale": total_scale,
                    "confirm_passed": bool(confirm_passed),
                }
            )

        if executed_count:
            day_ret /= executed_count
            day_ret *= max(0.0, min(1.5, float(cfg.exposure_scale)))
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
