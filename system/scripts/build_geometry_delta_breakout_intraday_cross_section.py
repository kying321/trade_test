#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"


SOURCE_CATALOG: list[dict[str, str]] = [
    {
        "source_id": "tradingview_volumebreakout",
        "source_type": "open_source_community_script",
        "title": "VOLUMEBREAKOUT",
        "url": "https://tr.tradingview.com/scripts/volumebreakout/",
        "summary": "用 K 线几何近似买卖量，做 intraday breakout confirmation。",
    },
    {
        "source_id": "tradingview_volume_profile_basics",
        "source_type": "platform_support_doc",
        "title": "Volume profile indicators: basic concepts",
        "url": "https://www.tradingview.com/support/solutions/43000502040-volume-profile-indicators-basic-concepts/",
        "summary": "本轮不重建真实 VPVR，只把它保留成背景解释来源。",
    },
]


PARAM_GRID: list[dict[str, Any]] = [
    {
        "breakout_lookback": breakout_lookback,
        "delta_z_threshold": delta_z_threshold,
        "top_k": top_k,
        "rebalance_every": rebalance_every,
        "trend_filter": trend_filter,
    }
    for breakout_lookback, delta_z_threshold, top_k, rebalance_every, trend_filter in itertools.product(
        [20, 40, 80],
        [0.0, 0.5, 1.0],
        [1, 2],
        [1, 2, 4],
        [False, True],
    )
]


EXECUTION_COST_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario_id": "gross",
        "label": "毛收益",
        "fee_bps_per_side": 0.0,
        "slippage_bps_per_side": 0.0,
    },
    {
        "scenario_id": "moderate_costs",
        "label": "中性成本",
        "fee_bps_per_side": 5.0,
        "slippage_bps_per_side": 3.0,
    },
    {
        "scenario_id": "stress_costs",
        "label": "压力成本",
        "fee_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
    },
]
SELECTION_SCENARIO_ID = "moderate_costs"


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | pd.Timestamp | None) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset_missing_columns:{','.join(sorted(missing))}")
    work = frame.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["symbol"] = work["symbol"].astype(str).str.upper().str.strip()
    work["asset_class"] = work.get("asset_class", "crypto")
    work["asset_class"] = work["asset_class"].astype(str).str.lower().str.strip()
    work = work.dropna(subset=["ts", "symbol", "open", "high", "low", "close", "volume"]).copy()
    work = work.sort_values(["symbol", "ts"]).drop_duplicates(subset=["symbol", "ts"]).reset_index(drop=True)
    return work


def select_latest_intraday_dataset(review_dir: Path) -> Path:
    candidates = sorted(review_dir.glob("*_public_intraday_crypto_bars_dataset.csv"))
    if not candidates:
        raise FileNotFoundError("no_public_intraday_crypto_bars_dataset_found")
    return candidates[-1]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    grouped = work.groupby("symbol")
    work["ret_fwd_1bar"] = grouped["close"].shift(-1) / work["close"] - 1.0
    work["ema_20"] = grouped["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    work["ema_50"] = grouped["close"].transform(lambda s: s.ewm(span=50, adjust=False).mean())
    work["volume_mean_20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    work["volume_std_20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=10).std(ddof=0))
    work["volume_z_20"] = (
        (work["volume"] - work["volume_mean_20"]) / work["volume_std_20"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    work["range"] = (work["high"] - work["low"]).clip(lower=1e-9)
    work["delta_geometry"] = ((2.0 * work["close"] - work["high"] - work["low"]) / work["range"]) * work["volume"]
    work["delta_geom_mean_20"] = grouped["delta_geometry"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    work["delta_geom_std_20"] = grouped["delta_geometry"].transform(lambda s: s.rolling(20, min_periods=10).std(ddof=0))
    work["delta_geom_z_20"] = (
        (work["delta_geometry"] - work["delta_geom_mean_20"]) / work["delta_geom_std_20"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    for lookback in (20, 40, 80):
        work[f"roll_high_{lookback}_prev"] = grouped["high"].transform(
            lambda s, lb=lookback: s.rolling(lb, min_periods=lb).max().shift(1)
        )
    return work


def split_dates(frame: pd.DataFrame, train_ratio: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Series(sorted(frame["ts"].dropna().unique())).astype("datetime64[ns]")
    if len(unique_dates) < 160:
        raise ValueError("dataset_too_small_for_intraday_cross_section_split")
    split_idx = max(96, min(len(unique_dates) - 48, int(math.floor(len(unique_dates) * train_ratio))))
    return pd.Timestamp(unique_dates.iloc[split_idx - 1]), pd.Timestamp(unique_dates.iloc[-1])


def score_geometry_delta_breakout(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    breakout_lb = int(params["breakout_lookback"])
    delta_z_threshold = float(params["delta_z_threshold"])
    trend_filter = bool(params["trend_filter"])
    signal = (
        (frame["close"] > frame[f"roll_high_{breakout_lb}_prev"])
        & (frame["delta_geom_z_20"] >= delta_z_threshold)
        & (frame["volume_z_20"] >= 0.0)
    )
    if trend_filter:
        signal = signal & (frame["close"] > frame["ema_20"]) & (frame["ema_20"] > frame["ema_50"])
    strength = (
        (frame["close"] / frame[f"roll_high_{breakout_lb}_prev"] - 1.0).replace([np.inf, -np.inf], np.nan)
        + 0.1 * frame["delta_geom_z_20"].clip(lower=0.0)
    )
    return strength.where(signal, 0.0).fillna(0.0)


def choose_weights(selected: pd.DataFrame) -> dict[str, float]:
    if selected.empty:
        return {}
    equal_weight = 1.0 / float(len(selected))
    return {str(symbol): equal_weight for symbol in selected["symbol"]}


def average_rank_ic(chunk: pd.DataFrame) -> float:
    ranked = chunk[["score", "ret_fwd_1bar"]].dropna()
    if len(ranked) < 3:
        return float("nan")
    if ranked["score"].nunique() < 2 or ranked["ret_fwd_1bar"].nunique() < 2:
        return float("nan")
    return float(ranked["score"].rank().corr(ranked["ret_fwd_1bar"].rank(), method="pearson"))


def backtest_long_only(frame: pd.DataFrame, *, score_col: str, top_k: int, rebalance_every: int) -> dict[str, Any]:
    work = frame.sort_values(["ts", "symbol"]).copy()
    dates = list(pd.Series(sorted(work["ts"].dropna().unique())).astype("datetime64[ns]"))
    positions: dict[str, float] = {}
    daily_rows: list[dict[str, Any]] = []
    holdings_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    trade_count = 0
    last_selection: set[str] = set()
    last_weights: dict[str, float] = {}
    for idx, current_ts in enumerate(dates[:-1]):
        chunk = work[work["ts"] == current_ts].copy()
        next_ts = dates[idx + 1]
        rebalance = idx % max(1, rebalance_every) == 0
        turnover = 0.0
        if rebalance:
            selected = chunk[chunk[score_col] > 0.0].sort_values(score_col, ascending=False).head(max(1, int(top_k)))
            positions = choose_weights(selected)
            current_selection = set(positions)
            trade_count += len(current_selection.symmetric_difference(last_selection))
            universe = current_selection | set(last_weights)
            turnover = 0.5 * sum(abs(float(positions.get(sym, 0.0)) - float(last_weights.get(sym, 0.0))) for sym in universe)
            last_selection = current_selection
            last_weights = dict(positions)
        portfolio_return = 0.0
        if positions:
            returns = chunk.set_index("symbol")["ret_fwd_1bar"].to_dict()
            portfolio_return = float(sum(float(returns.get(sym, 0.0) or 0.0) * weight for sym, weight in positions.items()))
        score_ic = average_rank_ic(chunk.rename(columns={score_col: "score"}))
        daily_rows.append(
            {
                "ts": current_ts,
                "next_ts": next_ts,
                "portfolio_return": portfolio_return,
                "turnover": turnover,
                "active_positions": int(len(positions)),
                "selected_symbols": sorted(positions.keys()),
                "rank_ic": score_ic,
            }
        )
        turnover_rows.append({"ts": current_ts, "turnover": turnover})
        for symbol, weight in positions.items():
            holdings_rows.append({"ts": current_ts, "symbol": symbol, "weight": float(weight)})
    daily = pd.DataFrame(daily_rows)
    metrics = summarize_metrics(daily if not daily.empty else pd.DataFrame(columns=["portfolio_return", "active_positions", "rank_ic"]), trade_count)
    return {
        "daily": daily,
        "holdings": holdings_rows,
        "turnover_rows": turnover_rows,
        "metrics": metrics,
    }


def summarize_metrics(daily: pd.DataFrame, trade_count: int) -> dict[str, Any]:
    if daily.empty:
        return {
            "active_bars": 0,
            "trade_count": 0,
            "win_rate": 0.0,
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_bar_return": 0.0,
            "avg_cross_section_ic": 0.0,
        }
    equity = (1.0 + daily["portfolio_return"]).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    avg_ret = float(daily["portfolio_return"].mean())
    std_ret = float(daily["portfolio_return"].std(ddof=0))
    active_slice = daily[daily["active_positions"] > 0].copy()
    active_win_rate = float((active_slice["portfolio_return"] > 0.0).mean()) if not active_slice.empty else 0.0
    annual_factor = math.sqrt(365.0 * 24.0 * 4.0)
    sharpe = float((avg_ret / std_ret) * annual_factor) if std_ret > 0.0 else 0.0
    avg_ic = float(pd.Series(daily["rank_ic"]).dropna().mean() or 0.0)
    if not math.isfinite(avg_ic):
        avg_ic = 0.0
    return {
        "active_bars": int((daily["active_positions"] > 0).sum()),
        "trade_count": int(trade_count),
        "win_rate": active_win_rate,
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(abs(drawdown.min())),
        "avg_bar_return": avg_ret,
        "avg_cross_section_ic": avg_ic,
    }


def apply_cost_scenario(base_result: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    daily = base_result["daily"].copy()
    if daily.empty:
        return {
            "scenario_id": str(scenario["scenario_id"]),
            "label": str(scenario["label"]),
            "fee_bps_per_side": float(scenario["fee_bps_per_side"]),
            "slippage_bps_per_side": float(scenario["slippage_bps_per_side"]),
            "metrics": summarize_metrics(daily, int(base_result["metrics"]["trade_count"])),
            "daily_sample": [],
        }
    one_way_cost = (float(scenario["fee_bps_per_side"]) + float(scenario["slippage_bps_per_side"])) / 10000.0
    daily["net_return"] = daily["portfolio_return"] - daily["turnover"] * one_way_cost
    net_daily = pd.DataFrame(
        {
            "ts": daily["ts"],
            "next_ts": daily["next_ts"],
            "portfolio_return": daily["net_return"],
            "active_positions": daily["active_positions"],
            "selected_symbols": daily["selected_symbols"],
            "rank_ic": daily["rank_ic"],
        }
    )
    metrics = summarize_metrics(net_daily, int(base_result["metrics"]["trade_count"]))
    return {
        "scenario_id": str(scenario["scenario_id"]),
        "label": str(scenario["label"]),
        "fee_bps_per_side": float(scenario["fee_bps_per_side"]),
        "slippage_bps_per_side": float(scenario["slippage_bps_per_side"]),
        "metrics": metrics,
        "daily_sample": [
            {
                "ts_utc": fmt_utc(pd.Timestamp(row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "next_ts_utc": fmt_utc(pd.Timestamp(row["next_ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "net_return": round(float(row["net_return"]), 6),
                "turnover": round(float(row["turnover"]), 6),
                "selected_symbols": list(row["selected_symbols"]),
            }
            for row in daily.head(12).to_dict("records")
        ],
    }


def objective(metrics: dict[str, Any]) -> float:
    active_bars = int(metrics.get("active_bars", 0) or 0)
    trade_count = int(metrics.get("trade_count", 0) or 0)
    avg_ic = float(metrics.get("avg_cross_section_ic", 0.0) or 0.0)
    if not math.isfinite(avg_ic):
        avg_ic = 0.0
    penalty = 0.0
    if active_bars <= 0 or trade_count <= 0:
        penalty -= 40.0
    elif active_bars < 10 or trade_count < 8:
        penalty -= 10.0
    return float(
        float(metrics.get("cumulative_return", 0.0)) * 100.0
        + float(metrics.get("sharpe", 0.0)) * 6.0
        + avg_ic * 25.0
        - float(metrics.get("max_drawdown", 0.0)) * 80.0
        + float(metrics.get("win_rate", 0.0)) * 4.0
        + penalty
    )


def classify_validation(metrics: dict[str, Any]) -> str:
    active_bars = int(metrics.get("active_bars", 0) or 0)
    trade_count = int(metrics.get("trade_count", 0) or 0)
    cumulative_return = float(metrics.get("cumulative_return", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    if active_bars <= 0 or trade_count <= 0:
        return "inactive_oos"
    if cumulative_return > 0.03 and sharpe > 0.6 and max_drawdown < 0.08 and trade_count >= 8:
        return "promising_small_sample"
    if cumulative_return > 0.0 and max_drawdown < 0.12:
        return "mixed_positive"
    return "not_promising"


def param_sort_key(row: dict[str, Any]) -> tuple[float, int, float, int]:
    valid = dict(row.get("validation_metrics") or {})
    trade_count = int(valid.get("trade_count", 0) or 0)
    robust_trade_flag = 1 if trade_count >= 8 else 0
    return (
        float(row["train_objective"]),
        robust_trade_flag,
        float(row["validation_objective"]),
        trade_count,
    )


def build_param_result(frame: pd.DataFrame, params: dict[str, Any], train_end: pd.Timestamp) -> dict[str, Any]:
    train_frame = frame[frame["ts"] <= train_end].copy()
    valid_frame = frame[frame["ts"] > train_end].copy()
    train_scored = train_frame.copy()
    valid_scored = valid_frame.copy()
    train_scored["strategy_score"] = score_geometry_delta_breakout(train_scored, params)
    valid_scored["strategy_score"] = score_geometry_delta_breakout(valid_scored, params)
    train_gross = backtest_long_only(train_scored, score_col="strategy_score", top_k=int(params["top_k"]), rebalance_every=int(params["rebalance_every"]))
    valid_gross = backtest_long_only(valid_scored, score_col="strategy_score", top_k=int(params["top_k"]), rebalance_every=int(params["rebalance_every"]))
    train_scenarios = [apply_cost_scenario(train_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
    valid_scenarios = [apply_cost_scenario(valid_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
    train_selected = next(row for row in train_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
    valid_selected = next(row for row in valid_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
    return {
        "params": params,
        "train_metrics": train_selected["metrics"],
        "validation_metrics": valid_selected["metrics"],
        "train_gross_metrics": train_gross["metrics"],
        "validation_gross_metrics": valid_gross["metrics"],
        "train_objective": float(objective(train_selected["metrics"])),
        "validation_objective": float(objective(valid_selected["metrics"])),
        "validation_status": classify_validation(valid_selected["metrics"]),
        "train_scenarios": train_scenarios,
        "validation_scenarios": valid_scenarios,
        "validation_daily_sample": valid_selected["daily_sample"][:10],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Geometry Delta Breakout Intraday Cross Section",
        "",
        f"- dataset_path: `{text(payload.get('dataset_path'))}`",
        f"- symbol_count: `{payload.get('symbol_count')}`",
        f"- symbols: `{text(payload.get('universe_brief'))}`",
        f"- cadence_minutes: `{payload.get('cadence_minutes')}`",
        f"- coverage: `{text(payload.get('coverage_start_utc'))} -> {text(payload.get('coverage_end_utc'))}`",
        f"- train_end_utc: `{text(payload.get('train_end_utc'))}`",
        f"- selection_scenario: `{text(payload.get('selection_scenario_id'))}`",
        f"- selected_params: `{json.dumps(payload.get('selected_params', {}), ensure_ascii=False)}`",
        f"- validation_status: `{text(payload.get('validation_status'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Validation Metrics",
        "",
    ]
    valid = dict(payload.get("validation_metrics") or {})
    for key in ("active_bars", "trade_count", "win_rate", "cumulative_return", "sharpe", "max_drawdown", "avg_bar_return", "avg_cross_section_ic"):
        value = valid.get(key)
        if isinstance(value, float):
            if "rate" in key or "return" in key or key == "max_drawdown":
                lines.append(f"- {key}: `{value:.4%}`")
            else:
                lines.append(f"- {key}: `{value:.4f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Gross vs Cost Scenarios", ""])
    gross = dict(payload.get("validation_gross_metrics") or {})
    lines.append(
        f"- gross: `return={float(gross.get('cumulative_return', 0.0)):.4%}, sharpe={float(gross.get('sharpe', 0.0)):.2f}, ic={float(gross.get('avg_cross_section_ic', 0.0)):.3f}, trades={int(gross.get('trade_count', 0) or 0)}`"
    )
    for row in payload.get("validation_scenarios", []):
        if text(row.get("scenario_id")) == "gross":
            continue
        metrics = dict(row.get("metrics") or {})
        lines.append(
            f"- {row['scenario_id']}: `fee={float(row['fee_bps_per_side']):.1f}bps/side, slip={float(row['slippage_bps_per_side']):.1f}bps/side, return={float(metrics.get('cumulative_return', 0.0)):.4%}, sharpe={float(metrics.get('sharpe', 0.0)):.2f}, ic={float(metrics.get('avg_cross_section_ic', 0.0)):.3f}`"
        )
    lines.extend(["", "## Validation Daily Sample", ""])
    for row in payload.get("validation_daily_sample", []):
        lines.append(
            f"- `{row['ts_utc']}` -> `{row['next_ts_utc']}` | net_return=`{row['net_return']:.4%}` | turnover=`{row['turnover']:.3f}` | picks=`{','.join(row['selected_symbols']) or '-'}`"
        )
    lines.extend(["", "## Source Catalog", ""])
    for row in payload.get("source_catalog", []):
        lines.append(f"- `{row['source_id']}` | [{row['title']}]({row['url']}) | {row['summary']}")
    lines.extend(["", "## Notes", "", f"- `{text(payload.get('research_note'))}`", f"- `{text(payload.get('limitation_note'))}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a geometry-delta intraday cross-section research backtest on local multi-symbol bars.")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--min-symbols", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser().resolve() if text(args.dataset_path) else select_latest_intraday_dataset(review_dir)
    frame = load_frame(dataset_path)
    if int(frame["symbol"].nunique()) < int(args.min_symbols):
        raise SystemExit(f"dataset_symbol_count_below_min:{int(frame['symbol'].nunique())}<{int(args.min_symbols)}")
    featured = add_features(frame)
    train_end, valid_end = split_dates(featured, float(args.train_ratio))
    param_results = [build_param_result(featured, params, train_end) for params in PARAM_GRID]
    ranked = sorted(param_results, key=param_sort_key, reverse=True)
    selected = ranked[0]
    selected_params = dict(selected["params"])
    final = build_param_result(featured, selected_params, train_end)
    sample_symbol = str(sorted(frame["symbol"].unique())[0])
    sample_ts = pd.to_datetime(
        frame.loc[frame["symbol"] == sample_symbol, "ts"],
        utc=True,
    ).sort_values()
    cadence = sample_ts.diff().dropna().mode()
    cadence_minutes = int(cadence.iloc[0] / pd.Timedelta(minutes=1)) if not cadence.empty else None
    final_valid = next(row for row in final["validation_scenarios"] if row["scenario_id"] == SELECTION_SCENARIO_ID)
    final_train = next(row for row in final["train_scenarios"] if row["scenario_id"] == SELECTION_SCENARIO_ID)
    payload = {
        "action": "build_geometry_delta_breakout_intraday_cross_section",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "coverage_start_utc": fmt_utc(pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "coverage_end_utc": fmt_utc(pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "train_end_utc": fmt_utc(train_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "validation_end_utc": fmt_utc(valid_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "symbol_count": int(frame["symbol"].nunique()),
        "universe_brief": ",".join(sorted(frame["symbol"].unique())),
        "bar_rows": int(len(frame)),
        "cadence_minutes": cadence_minutes,
        "strategy_family": "geometry_delta_breakout",
        "selection_scenario_id": SELECTION_SCENARIO_ID,
        "execution_cost_scenarios": EXECUTION_COST_SCENARIOS,
        "selected_params": selected_params,
        "train_metrics": final_train["metrics"],
        "validation_metrics": final_valid["metrics"],
        "train_gross_metrics": final["train_gross_metrics"],
        "validation_gross_metrics": final["validation_gross_metrics"],
        "validation_scenarios": final["validation_scenarios"],
        "validation_status": classify_validation(final_valid["metrics"]),
        "train_objective": float(objective(final_train["metrics"])),
        "validation_objective": float(objective(final_valid["metrics"])),
        "validation_daily_sample": final["validation_daily_sample"],
        "param_results": [
            {
                "params": row["params"],
                "train_objective": round(float(row["train_objective"]), 6),
                "validation_objective": round(float(row["validation_objective"]), 6),
                "validation_status": row["validation_status"],
                "validation_trade_count": int(row["validation_metrics"]["trade_count"]),
                "validation_active_bars": int(row["validation_metrics"]["active_bars"]),
                "validation_cumulative_return": round(float(row["validation_metrics"]["cumulative_return"]), 6),
                "validation_sharpe": round(float(row["validation_metrics"]["sharpe"]), 6),
                "validation_avg_cross_section_ic": round(float(row["validation_metrics"]["avg_cross_section_ic"]), 6),
                "validation_max_drawdown": round(float(row["validation_metrics"]["max_drawdown"]), 6),
            }
            for row in ranked[:12]
        ],
        "recommended_brief": (
            f"geometry_delta_breakout:{SELECTION_SCENARIO_ID}:status={classify_validation(final_valid['metrics'])},"
            f"valid_return={float(final_valid['metrics']['cumulative_return']):.2%},"
            f"gross_return={float(final['validation_gross_metrics']['cumulative_return']):.2%},"
            f"sharpe={float(final_valid['metrics']['sharpe']):.2f},"
            f"ic={float(final_valid['metrics']['avg_cross_section_ic']):.3f},"
            f"trades={int(final_valid['metrics']['trade_count'])}"
        ),
        "source_catalog": SOURCE_CATALOG,
        "research_note": "这轮用新拉下来的 15m 多 symbol 基座，把 geometry-delta-breakout 推进到轻量 intraday 截面回测。",
        "limitation_note": "当前只覆盖 4 个币、约 10 天 15m bars，且仍用几何 delta proxy；结果只能说明是否值得继续扩大样本，不应用于直接放行 live。",
    }
    json_path = review_dir / f"{args.stamp}_geometry_delta_breakout_intraday_cross_section.json"
    md_path = review_dir / f"{args.stamp}_geometry_delta_breakout_intraday_cross_section.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "validation_status": payload["validation_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
