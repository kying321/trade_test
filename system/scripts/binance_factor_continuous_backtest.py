#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lie_engine.config import load_settings
from lie_engine.data.providers import TushareBinanceHybridProvider


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT"]
FACTOR_COLS = ["mom_5", "mom_20", "mom_60", "trend_ma20_60", "vol_20", "range_10", "volume_z20", "rsi_14"]


@dataclass(slots=True)
class StrategyMetrics:
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    trades: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return": float(self.total_return),
            "annual_return": float(self.annual_return),
            "max_drawdown": float(self.max_drawdown),
            "sharpe": float(self.sharpe),
            "win_rate": float(self.win_rate),
            "trades": int(self.trades),
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance continuous factor backtest")
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--symbols", default="", help="comma-separated binance symbols")
    p.add_argument("--window-days", default="270")
    p.add_argument("--step-days", default="30")
    p.add_argument("--dd-limit", default="0.05")
    p.add_argument("--max-factors", default="4")
    p.add_argument("--apply-live-params", action="store_true", help="Write suggested runtime params to params_live.yaml")
    p.add_argument("--live-params-path", default="", help="Override params_live.yaml target path")
    p.add_argument("--lock-path", default="", help="File lock path for mutual exclusion")
    p.add_argument("--out-dir", default="", help="Override output/review dir")
    return p.parse_args()


def _date(s: str) -> date:
    return date.fromisoformat(str(s).strip())


def _resolve_root(config_path: str | None) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve().parent
    return Path(__file__).resolve().parents[1]


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = lock_path.open("w", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            f.write(str(os.getpid()))
            f.flush()
        except BlockingIOError:
            acquired = False
        yield {"path": str(lock_path), "acquired": bool(acquired)}
    finally:
        if acquired:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        f.close()


def _derive_runtime_param_suggestion(recommended: dict[str, Any], ic_top: list[dict[str, Any]], dd_limit: float) -> dict[str, float]:
    params = recommended.get("params", {}) if isinstance(recommended.get("params", {}), dict) else {}
    metrics = recommended.get("metrics", {}) if isinstance(recommended.get("metrics", {}), dict) else {}
    top_k = int(params.get("top_k", 2))
    hold_days = int(params.get("hold_days", 3))
    risk_scale = float(params.get("risk_scale", 0.2))
    sharpe = float(metrics.get("sharpe", 0.0))
    dd = float(metrics.get("max_drawdown", 1.0))
    ic_strength = 0.0
    if ic_top:
        ic_strength = float(np.mean([abs(float(x.get("mean_ic", 0.0))) for x in ic_top[:4]]))

    base_conf = 45.0 + min(12.0, ic_strength * 350.0) + min(8.0, max(0.0, sharpe) * 2.0)
    if dd > dd_limit:
        base_conf += min(6.0, (dd - dd_limit) * 100.0)

    return {
        "signal_confidence_min": float(max(35.0, min(68.0, base_conf))),
        "convexity_min": float(max(0.8, min(2.4, 1.1 + max(0.0, 0.25 - risk_scale) * 2.0))),
        "hold_days": float(max(1, min(12, hold_days))),
        "max_daily_trades": float(max(1, min(5, top_k))),
    }


def _apply_live_params(path: Path, updates: dict[str, float]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            existing = dict(raw)
    merged = dict(existing)
    merged.update({k: float(v) for k, v in updates.items()})
    merged["factor_params_updated_at"] = datetime.now().isoformat()
    path.write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"path": str(path), "updated_keys": sorted(list(updates.keys()))}


def _load_symbols(args: argparse.Namespace) -> list[str]:
    if str(args.symbols).strip():
        raw = [x.strip().upper() for x in str(args.symbols).split(",")]
        return [x for x in raw if x]

    settings = load_settings(args.config)
    core = settings.universe.get("core", [])
    out: list[str] = []
    for item in core:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if symbol:
            out.append(symbol)
    if out:
        return list(dict.fromkeys(out))
    return list(DEFAULT_SYMBOLS)


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


def _metrics_from_equity(equity: pd.Series, daily_ret: pd.Series, trades: int) -> StrategyMetrics:
    if equity.empty:
        return StrategyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    total_return = float(equity.iloc[-1] / max(1e-9, equity.iloc[0]) - 1.0)
    n_days = max(1, len(equity))
    annual_return = float((1.0 + total_return) ** (365.0 / n_days) - 1.0) if total_return > -1 else -1.0
    dd = _max_drawdown(equity)
    mu = float(daily_ret.mean()) if not daily_ret.empty else 0.0
    sd = float(daily_ret.std(ddof=0)) if not daily_ret.empty else 0.0
    sharpe = float(mu / sd * np.sqrt(365.0)) if sd > 1e-9 else 0.0
    win_rate = float((daily_ret > 0.0).mean()) if not daily_ret.empty else 0.0
    return StrategyMetrics(total_return, annual_return, dd, sharpe, win_rate, int(trades))


def fetch_binance_bars(symbols: list[str], start: date, end: date) -> tuple[pd.DataFrame, dict[str, int]]:
    provider = TushareBinanceHybridProvider()
    frames: list[pd.DataFrame] = []
    rows_by_symbol: dict[str, int] = {}
    for symbol in symbols:
        df = provider.fetch_ohlcv(symbol=symbol, start=start, end=end, freq="1d")
        if df.empty:
            rows_by_symbol[symbol] = 0
            continue
        df = df.copy()
        df["symbol"] = str(symbol).upper()
        frames.append(df)
        rows_by_symbol[symbol] = int(len(df))
    if not frames:
        return pd.DataFrame(), rows_by_symbol
    bars = pd.concat(frames, ignore_index=True)
    bars["ts"] = pd.to_datetime(bars["ts"], errors="coerce")
    bars = bars.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
    return bars, rows_by_symbol


def build_factor_panel(bars: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, grp in bars.groupby("symbol"):
        g = grp.sort_values("ts").copy()
        g["date"] = g["ts"].dt.date
        g["ret_1d"] = g["close"].pct_change()
        g["next_ret_1d"] = g["ret_1d"].shift(-1)
        g["mom_5"] = g["close"].pct_change(5)
        g["mom_20"] = g["close"].pct_change(20)
        g["mom_60"] = g["close"].pct_change(60)
        ma20 = g["close"].rolling(20, min_periods=20).mean()
        ma60 = g["close"].rolling(60, min_periods=60).mean()
        g["trend_ma20_60"] = ma20 / ma60 - 1.0
        g["vol_20"] = g["ret_1d"].rolling(20, min_periods=20).std(ddof=0)
        g["range_10"] = ((g["high"] - g["low"]) / g["close"].replace(0.0, np.nan)).rolling(10, min_periods=10).mean()
        lv = np.log(g["volume"].clip(lower=1.0))
        g["volume_z20"] = (lv - lv.rolling(20, min_periods=20).mean()) / lv.rolling(20, min_periods=20).std(ddof=0)
        g["rsi_14"] = _compute_rsi(g["close"], 14) / 100.0 - 0.5
        rows.append(g[["date", "symbol", "ret_1d", "next_ret_1d"] + FACTOR_COLS])
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def summarize_factor_ic(panel: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in FACTOR_COLS:
        ic_vals: list[float] = []
        valid_days = 0
        for _, g in panel[["date", f, "next_ret_1d"]].dropna().groupby("date"):
            if len(g) < 4:
                continue
            # Spearman proxy without scipy dependency: corr(rank(x), rank(y)).
            xr = g[f].rank(method="average")
            yr = g["next_ret_1d"].rank(method="average")
            v = xr.corr(yr)
            if v is None or not np.isfinite(v):
                continue
            ic_vals.append(float(v))
            valid_days += 1
        if not ic_vals:
            out.append(
                {
                    "factor": f,
                    "sample_days": 0,
                    "mean_ic": 0.0,
                    "std_ic": 0.0,
                    "ir": 0.0,
                    "positive_ic_ratio": 0.0,
                }
            )
            continue
        mean_ic = float(np.mean(ic_vals))
        std_ic = float(np.std(ic_vals))
        ir = float(mean_ic / std_ic * np.sqrt(len(ic_vals))) if std_ic > 1e-9 else 0.0
        out.append(
            {
                "factor": f,
                "sample_days": int(valid_days),
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ir": ir,
                "positive_ic_ratio": float(np.mean(np.array(ic_vals) > 0.0)),
            }
        )
    out.sort(key=lambda x: abs(float(x["mean_ic"])), reverse=True)
    return out


def select_factor_weights(ic_summary: list[dict[str, Any]], max_factors: int) -> dict[str, float]:
    picked = [x for x in ic_summary if int(x["sample_days"]) >= 40 and abs(float(x["mean_ic"])) >= 0.01][: max(1, int(max_factors))]
    if not picked:
        return {"mom_20": 0.5, "trend_ma20_60": 0.5}
    raw: dict[str, float] = {}
    for row in picked:
        mean_ic = float(row["mean_ic"])
        raw[str(row["factor"])] = float(np.sign(mean_ic) * max(1e-6, abs(mean_ic)))
    norm = sum(abs(v) for v in raw.values())
    return {k: float(v / norm) for k, v in raw.items()}


def build_score_panel(panel: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = panel[["date", "symbol", "ret_1d"]].copy()
    score = pd.Series(0.0, index=panel.index, dtype=float)
    for f, w in weights.items():
        z = _zscore_by_date(panel, f)
        score += float(w) * z
    out["score"] = score
    return out


def run_score_backtest(
    score_panel: pd.DataFrame,
    top_k: int,
    hold_days: int,
    fee_bps: float,
    risk_scale: float,
) -> StrategyMetrics:
    ret = score_panel.pivot(index="date", columns="symbol", values="ret_1d").sort_index()
    score = score_panel.pivot(index="date", columns="symbol", values="score").sort_index()
    if ret.empty or score.empty:
        return StrategyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)

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

    return _metrics_from_equity(pd.Series(eq_rows, index=dates), pd.Series(daily_ret, index=dates), trades=trades)


def optimize_params(score_panel: pd.DataFrame, dd_limit: float) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for top_k in (2, 3, 4):
        for hold_days in (1, 3, 5):
            for fee_bps in (6.0, 10.0):
                for risk_scale in (0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0):
                    m = run_score_backtest(
                        score_panel,
                        top_k=top_k,
                        hold_days=hold_days,
                        fee_bps=fee_bps,
                        risk_scale=risk_scale,
                    )
                    objective = float(m.annual_return - 1.5 * m.max_drawdown + 0.2 * m.sharpe)
                    candidates.append(
                        {
                            "params": {
                                "top_k": top_k,
                                "hold_days": hold_days,
                                "fee_bps": fee_bps,
                                "risk_scale": risk_scale,
                            },
                            "metrics": m.to_dict(),
                            "objective": objective,
                            "dd_ok": bool(m.max_drawdown <= dd_limit),
                        }
                    )
    feasible = [x for x in candidates if bool(x["dd_ok"])]
    ranked = sorted(feasible if feasible else candidates, key=lambda x: float(x["objective"]), reverse=True)
    return {"best": ranked[0] if ranked else {}, "all": ranked}


def rolling_windows(start: date, end: date, window_days: int, step_days: int) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur_end = start + timedelta(days=max(30, int(window_days)))
    while cur_end <= end:
        out.append((cur_end - timedelta(days=max(30, int(window_days))), cur_end))
        cur_end += timedelta(days=max(7, int(step_days)))
    if not out:
        out.append((start, end))
    return out


def _render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Binance 连续回测与因子提炼 | {payload.get('as_of', '')}")
    lines.append("")
    lines.append("## 数据覆盖")
    lines.append(f"- symbols: `{payload.get('symbols', [])}`")
    lines.append(f"- bars_rows: `{payload.get('bars_rows', 0)}`")
    lines.append(f"- window_count: `{payload.get('window_count', 0)}`")
    lines.append("")
    lines.append("## 因子 IC Top")
    for row in payload.get("factor_ic_top", [])[:6]:
        lines.append(
            f"- `{row.get('factor')}`: mean_ic={float(row.get('mean_ic', 0.0)):.4f}, "
            f"ir={float(row.get('ir', 0.0)):.4f}, sample_days={int(row.get('sample_days', 0))}"
        )
    lines.append("")
    rec = payload.get("recommended_strategy", {})
    lines.append("## 推荐策略")
    lines.append(f"- params: `{rec.get('params', {})}`")
    lines.append(f"- factor_weights: `{rec.get('factor_weights', {})}`")
    lines.append(f"- metrics: `{rec.get('metrics', {})}`")
    lines.append(f"- dd_limit: `{payload.get('dd_limit', 0.05)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    start = _date(args.start)
    end = _date(args.end)
    dd_limit = float(args.dd_limit)
    window_days = int(args.window_days)
    step_days = int(args.step_days)
    max_factors = int(args.max_factors)
    root = _resolve_root(args.config)
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (root / "output" / "review")
    logs_dir = root / "output" / "logs"
    state_dir = root / "output" / "state"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(args.lock_path).expanduser().resolve() if str(args.lock_path).strip() else (state_dir / "binance_factor_continuous_backtest.lock")

    with _file_lock(lock_path) as lock:
        if not bool(lock.get("acquired", False)):
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "lock_busy",
                        "lock": lock,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        symbols = _load_symbols(args)
        bars, rows_by_symbol = fetch_binance_bars(symbols=symbols, start=start, end=end)
        if bars.empty:
            raise RuntimeError("No bars fetched from Binance provider")

        panel = build_factor_panel(bars)
        panel = panel.dropna(subset=["date", "symbol", "ret_1d", "next_ret_1d"]).reset_index(drop=True)
        if panel.empty:
            raise RuntimeError("No valid factor panel rows")

        windows = rolling_windows(start=start, end=end, window_days=window_days, step_days=step_days)
        window_rows: list[dict[str, Any]] = []
        for ws, we in windows:
            p = panel[(panel["date"] >= ws) & (panel["date"] <= we)].copy()
            if len(p) < 200:
                continue
            ic = summarize_factor_ic(p)
            weights = select_factor_weights(ic, max_factors=max_factors)
            score_panel = build_score_panel(p, weights=weights)
            opt = optimize_params(score_panel, dd_limit=dd_limit)
            best = opt.get("best", {})
            window_rows.append(
                {
                    "start": ws.isoformat(),
                    "end": we.isoformat(),
                    "rows": int(len(p)),
                    "factor_weights": weights,
                    "best": best,
                }
            )

        ic_full = summarize_factor_ic(panel)
        weights_full = select_factor_weights(ic_full, max_factors=max_factors)
        score_full = build_score_panel(panel, weights_full)
        opt_full = optimize_params(score_full, dd_limit=dd_limit)
        best_full = opt_full.get("best", {})

        recommended = {
            "params": best_full.get("params", {}),
            "metrics": best_full.get("metrics", {}),
            "objective": best_full.get("objective", 0.0),
            "dd_ok": best_full.get("dd_ok", False),
            "factor_weights": weights_full,
        }
        runtime_param_suggestion = _derive_runtime_param_suggestion(recommended=recommended, ic_top=ic_full[:10], dd_limit=dd_limit)

        payload = {
            "as_of": datetime.now().isoformat(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": symbols,
            "rows_by_symbol": rows_by_symbol,
            "bars_rows": int(len(bars)),
            "panel_rows": int(len(panel)),
            "dd_limit": dd_limit,
            "window_count": int(len(window_rows)),
            "windows": window_rows,
            "factor_ic_top": ic_full[:10],
            "recommended_strategy": recommended,
            "runtime_param_suggestion": runtime_param_suggestion,
            "lock": lock,
        }

        applied: dict[str, Any] = {"applied": False}
        if bool(args.apply_live_params):
            live_params_path = (
                Path(args.live_params_path).expanduser().resolve()
                if str(args.live_params_path).strip()
                else (root / "output" / "artifacts" / "params_live.yaml")
            )
            applied = {"applied": True} | _apply_live_params(path=live_params_path, updates=runtime_param_suggestion)
            payload["live_params_apply"] = applied

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"{date.today().isoformat()}_binance_factor_continuous_backtest_{stamp}.json"
        md_path = out_dir / f"{date.today().isoformat()}_binance_factor_continuous_backtest_{stamp}.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(_render_md(payload), encoding="utf-8")

        test_log = logs_dir / f"tests_binance_factor_continuous_backtest_{stamp}.json"
        test_log.write_text(
            json.dumps(
                {
                    "status": "PASSED",
                    "json": str(json_path),
                    "md": str(md_path),
                    "dd_ok": bool(recommended.get("dd_ok", False)),
                    "lock_acquired": bool(lock.get("acquired", False)),
                    "live_params_applied": bool(applied.get("applied", False)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                {
                    "status": "PASSED",
                    "json": str(json_path),
                    "md": str(md_path),
                    "test_log": str(test_log),
                    "recommended_strategy": payload["recommended_strategy"],
                    "runtime_param_suggestion": runtime_param_suggestion,
                    "live_params_apply": applied,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
