from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import time as time_module

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
    timing_path: str | Path | None = None,
) -> BacktestResult:
    started_mono = time_module.monotonic()
    timing_file = Path(timing_path) if timing_path is not None else None
    timing: dict[str, object] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "status": "running",
        "elapsed_sec": 0.0,
        "current_stage": "window_loop",
        "windows": [],
    }

    def _persist_timing() -> None:
        if timing_file is None:
            return
        timing["elapsed_sec"] = round(time_module.monotonic() - started_mono, 3)
        timing_file.parent.mkdir(parents=True, exist_ok=True)
        timing_file.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")

    windows: list[BacktestResult] = []
    cursor = start
    window_idx = 0
    _persist_timing()

    while True:
        train_end = _add_months(cursor, train_years * 12)
        valid_end = _add_months(train_end, valid_years * 12)
        if valid_end > end:
            break

        window_started_mono = time_module.monotonic()
        timing["current_stage"] = "run_event_backtest"
        timing["current_window"] = {
            "index": window_idx,
            "train_start": cursor.isoformat(),
            "train_end": train_end.isoformat(),
            "valid_start": train_end.isoformat(),
            "valid_end": valid_end.isoformat(),
        }
        window_event_timing: dict[str, object] = {}

        def _window_timing_callback(payload: dict[str, object]) -> None:
            window_event_timing.clear()
            window_event_timing.update(payload)
            timing["current_window_timing"] = dict(window_event_timing)
            _persist_timing()

        _persist_timing()
        result = run_event_backtest(
            bars=bars,
            start=train_end,
            end=valid_end,
            cfg=cfg_template or BacktestConfig(),
            trend_thr=trend_thr,
            mean_thr=mean_thr,
            atr_extreme=atr_extreme,
            timing_callback=_window_timing_callback,
        )
        windows.append(result)
        timing.setdefault("windows", []).append(
            {
                "index": window_idx,
                "train_start": cursor.isoformat(),
                "train_end": train_end.isoformat(),
                "valid_start": train_end.isoformat(),
                "valid_end": valid_end.isoformat(),
                "elapsed_sec": round(time_module.monotonic() - window_started_mono, 3),
                "trades": int(result.trades),
                "violations": int(result.violations),
                "positive_window_ratio": float(result.positive_window_ratio),
                "event_backtest_timing": dict(window_event_timing),
            }
        )
        timing["current_stage"] = "window_loop"
        timing.pop("current_window", None)
        timing.pop("current_window_timing", None)
        _persist_timing()
        cursor = _add_months(cursor, step_months)
        window_idx += 1

    if not windows:
        fallback_started_mono = time_module.monotonic()
        timing["current_stage"] = "run_event_backtest_fallback"
        timing["current_window"] = {
            "index": 0,
            "train_start": start.isoformat(),
            "train_end": start.isoformat(),
            "valid_start": start.isoformat(),
            "valid_end": end.isoformat(),
        }
        window_event_timing: dict[str, object] = {}

        def _window_timing_callback(payload: dict[str, object]) -> None:
            window_event_timing.clear()
            window_event_timing.update(payload)
            timing["current_window_timing"] = dict(window_event_timing)
            _persist_timing()

        _persist_timing()
        result = run_event_backtest(
            bars=bars,
            start=start,
            end=end,
            cfg=cfg_template or BacktestConfig(),
            trend_thr=trend_thr,
            mean_thr=mean_thr,
            atr_extreme=atr_extreme,
            timing_callback=_window_timing_callback,
        )
        timing["windows"] = [
            {
                "index": 0,
                "train_start": start.isoformat(),
                "train_end": start.isoformat(),
                "valid_start": start.isoformat(),
                "valid_end": end.isoformat(),
                "elapsed_sec": round(time_module.monotonic() - fallback_started_mono, 3),
                "trades": int(result.trades),
                "violations": int(result.violations),
                "positive_window_ratio": float(result.positive_window_ratio),
                "fallback_full_range": True,
                "event_backtest_timing": dict(window_event_timing),
            }
        ]
        timing["status"] = "completed"
        timing["current_stage"] = ""
        timing.pop("current_window", None)
        timing.pop("current_window_timing", None)
        _persist_timing()
        return result

    timing["current_stage"] = "aggregate_results"
    _persist_timing()
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
    for r in windows:
        merged_curve.extend(r.equity_curve)

    by_asset: dict[str, float] = {}
    for r in windows:
        for k, v in r.by_asset.items():
            by_asset.setdefault(k, 0.0)
            by_asset[k] += v / len(windows)

    result = BacktestResult(
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
    )
    timing["status"] = "completed"
    timing["current_stage"] = ""
    timing.pop("current_window", None)
    timing.pop("current_window_timing", None)
    timing["summary"] = {
        "window_count": int(len(windows)),
        "trades": int(result.trades),
        "violations": int(result.violations),
        "positive_window_ratio": float(result.positive_window_ratio),
    }
    _persist_timing()
    return result
