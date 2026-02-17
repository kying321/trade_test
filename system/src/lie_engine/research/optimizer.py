from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lie_engine.backtest.engine import BacktestConfig, run_event_backtest
from lie_engine.data.storage import write_json, write_markdown
from lie_engine.research.modes import MODE_SPACES, ModeSpace
from lie_engine.research.real_data import RealDataBundle, load_real_data_bundle


@dataclass(slots=True)
class TrialResult:
    mode: str
    trial: int
    params: dict[str, float | int]
    score: float
    annual_return: float
    max_drawdown: float
    positive_window_ratio: float
    trades: int
    factor_alignment: float
    started_at: str
    elapsed_sec: float


@dataclass(slots=True)
class ModeOptimizationSummary:
    mode: str
    trials: int
    best_score: float
    best_params: dict[str, float | int]
    best_metrics: dict[str, float | int]
    budget_seconds: float
    elapsed_seconds: float
    trial_log_path: str


@dataclass(slots=True)
class ResearchRunSummary:
    started_at: str
    ended_at: str
    elapsed_seconds: float
    start_date: str
    end_date: str
    cutoff_date: str
    cutoff_ts: str
    bar_max_ts: str
    news_max_ts: str
    report_max_ts: str
    review_days: int
    hours_budget: float
    universe_count: int
    bars_rows: int
    news_records: int
    report_records: int
    review_bars_rows: int
    review_bar_max_ts: str
    review_news_max_ts: str
    review_report_max_ts: str
    review_news_records: int
    review_report_records: int
    data_fetch_stats: dict[str, Any]
    mode_summaries: list[ModeOptimizationSummary] = field(default_factory=list)
    output_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode_summaries"] = [asdict(x) for x in self.mode_summaries]
        return d


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _sample_params(rng: np.random.Generator, space: ModeSpace, best: dict[str, float | int] | None) -> dict[str, float | int]:
    if not best:
        hold = int(rng.integers(space.hold_days_min, space.hold_days_max + 1))
        trades = int(rng.integers(space.trades_min, space.trades_max + 1))
        # Bias early exploration toward tradable regions (lower confidence/convexity) to avoid zero-trade cold start.
        conf_upper = space.confidence_min + 0.55 * (space.confidence_max - space.confidence_min)
        conv_upper = space.convexity_min + 0.60 * (space.convexity_max - space.convexity_min)
        conf = float(rng.uniform(space.confidence_min, conf_upper))
        conv = float(rng.uniform(space.convexity_min, conv_upper))
        nw = float(rng.uniform(0.0, 1.0))
        rw = float(rng.uniform(0.0, 1.0))
    else:
        hold = int(round(_clip(float(rng.normal(float(best["hold_days"]), 2.0)), space.hold_days_min, space.hold_days_max)))
        trades = int(round(_clip(float(rng.normal(float(best["max_daily_trades"]), 1.0)), space.trades_min, space.trades_max)))
        conf = _clip(float(rng.normal(float(best["signal_confidence_min"]), 4.0)), space.confidence_min, space.confidence_max)
        conv = _clip(float(rng.normal(float(best["convexity_min"]), 0.4)), space.convexity_min, space.convexity_max)
        nw = _clip(float(rng.normal(float(best.get("news_weight", 0.5)), 0.15)), 0.0, 1.0)
        rw = _clip(float(rng.normal(float(best.get("report_weight", 0.5)), 0.15)), 0.0, 1.0)

    return {
        "hold_days": hold,
        "max_daily_trades": trades,
        "signal_confidence_min": conf,
        "convexity_min": conv,
        "news_weight": nw,
        "report_weight": rw,
    }


def _factor_alignment(result_curve: list[dict[str, Any]], news_daily: pd.Series, report_daily: pd.Series, news_weight: float, report_weight: float) -> float:
    if not result_curve:
        return 0.0
    eq = pd.DataFrame(result_curve)
    if eq.empty or "date" not in eq.columns or "equity" not in eq.columns:
        return 0.0
    eq["date"] = pd.to_datetime(eq["date"]).dt.date
    eq = eq.sort_values("date")
    eq["ret"] = eq["equity"].pct_change()
    eq = eq.dropna(subset=["ret"])
    if eq.empty:
        return 0.0

    idx = eq["date"]
    ns = news_daily.reindex(idx).fillna(0.0) if not news_daily.empty else pd.Series(0.0, index=idx)
    rs = report_daily.reindex(idx).fillna(0.0) if not report_daily.empty else pd.Series(0.0, index=idx)
    factor = news_weight * ns + report_weight * rs
    aligned = pd.DataFrame({"ret": eq["ret"].to_numpy(), "factor": factor.to_numpy()})
    if aligned["factor"].std(ddof=0) < 1e-9 or aligned["ret"].std(ddof=0) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(aligned["factor"].shift(1).fillna(0.0), aligned["ret"])[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _objective(
    annual_return: float,
    max_drawdown: float,
    positive_ratio: float,
    factor_alignment: float,
    violations: int,
    trades: int,
    min_total_trades: int,
) -> float:
    penalty = 0.35 * float(max(0, violations))
    drawdown_penalty = 2.3 * max_drawdown + 8.0 * max(0.0, max_drawdown - 0.18)
    # Avoid selecting "no-trade" solutions that look good only by capital preservation.
    if trades <= 0:
        trade_penalty = 2.0
    elif trades < max(1, min_total_trades):
        trade_penalty = 0.06 * float(min_total_trades - trades)
    else:
        trade_penalty = 0.0
    return (
        1.2 * annual_return
        - drawdown_penalty
        + 0.8 * positive_ratio
        + 0.3 * factor_alignment
        - penalty
        - trade_penalty
    )


def _recent_factor_pressure(
    *,
    series: pd.Series,
    begin_d: date,
    end_d: date,
) -> float:
    if series.empty:
        return 0.0
    window = series[(series.index >= begin_d) & (series.index <= end_d)]
    if window.empty:
        return 0.0
    recent_mean = float(window.mean())
    base_mean = float(series.mean())
    base_std = float(series.std(ddof=0))
    if not np.isfinite(base_std) or base_std < 1e-9:
        return float(np.tanh(recent_mean))
    return float(np.clip((recent_mean - base_mean) / base_std, -4.0, 4.0))


def _optimize_one_mode(
    *,
    mode: str,
    space: ModeSpace,
    bars: pd.DataFrame,
    start: date,
    end: date,
    news_daily: pd.Series,
    report_daily: pd.Series,
    deadline_ts: float,
    max_trials: int,
    rng: np.random.Generator,
    out_dir: Path,
) -> ModeOptimizationSummary:
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    trials: list[TrialResult] = []
    best_score = -1e18
    best_params: dict[str, float | int] | None = None
    best_metrics: dict[str, float | int] = {}

    trial_idx = 0
    start_ts = time.time()
    while trial_idx < max_trials and time.time() < deadline_ts:
        trial_idx += 1
        t0 = time.time()
        params = _sample_params(rng=rng, space=space, best=best_params)

        # Use news/report pressure to dynamically calibrate confidence gate.
        recent_days = 20
        end_d = end
        begin_d = end_d - timedelta(days=recent_days)
        news_pressure = _recent_factor_pressure(series=news_daily, begin_d=begin_d, end_d=end_d)
        report_pressure = _recent_factor_pressure(series=report_daily, begin_d=begin_d, end_d=end_d)
        pressure = float(params["news_weight"]) * news_pressure + float(params["report_weight"]) * report_pressure
        conf_adj = float(params["signal_confidence_min"]) - 6.0 * float(np.tanh(pressure))
        conf_adj = float(np.clip(conf_adj, space.confidence_min, space.confidence_max))

        cfg = BacktestConfig(
            signal_confidence_min=conf_adj,
            convexity_min=float(params["convexity_min"]),
            max_daily_trades=int(params["max_daily_trades"]),
            hold_days=int(params["hold_days"]),
            regime_recalc_interval=5,
            signal_eval_interval=1,
            proxy_lookback=180,
        )
        bt = run_event_backtest(
            bars=bars,
            start=start,
            end=end,
            cfg=cfg,
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        if int(bt.trades) <= 0:
            # One fast retry with lower confidence gate to avoid wasting a full trial on no-trade regions.
            conf_retry = float(np.clip(conf_adj - 10.0, space.confidence_min, space.confidence_max))
            if conf_retry < conf_adj:
                cfg_retry = BacktestConfig(
                    signal_confidence_min=conf_retry,
                    convexity_min=float(params["convexity_min"]),
                    max_daily_trades=int(params["max_daily_trades"]),
                    hold_days=int(params["hold_days"]),
                    regime_recalc_interval=5,
                    signal_eval_interval=1,
                    proxy_lookback=180,
                )
                bt_retry = run_event_backtest(
                    bars=bars,
                    start=start,
                    end=end,
                    cfg=cfg_retry,
                    trend_thr=0.6,
                    mean_thr=0.4,
                    atr_extreme=2.0,
                )
                if int(bt_retry.trades) > int(bt.trades):
                    bt = bt_retry
                    conf_adj = conf_retry

        fa = _factor_alignment(
            result_curve=bt.equity_curve,
            news_daily=news_daily,
            report_daily=report_daily,
            news_weight=float(params["news_weight"]),
            report_weight=float(params["report_weight"]),
        )
        score = _objective(
            annual_return=float(bt.annual_return),
            max_drawdown=float(bt.max_drawdown),
            positive_ratio=float(bt.positive_window_ratio),
            factor_alignment=fa,
            violations=int(bt.violations),
            trades=int(bt.trades),
            min_total_trades=int(space.min_total_trades),
        )
        elapsed = time.time() - t0

        rec = TrialResult(
            mode=mode,
            trial=trial_idx,
            params={**params, "signal_confidence_effective": conf_adj},
            score=score,
            annual_return=float(bt.annual_return),
            max_drawdown=float(bt.max_drawdown),
            positive_window_ratio=float(bt.positive_window_ratio),
            trades=int(bt.trades),
            factor_alignment=fa,
            started_at=datetime.now().isoformat(),
            elapsed_sec=elapsed,
        )
        trials.append(rec)

        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_metrics = {
                "annual_return": float(bt.annual_return),
                "max_drawdown": float(bt.max_drawdown),
                "positive_window_ratio": float(bt.positive_window_ratio),
                "trades": int(bt.trades),
                "factor_alignment": fa,
                "score": score,
            }

    trials_df = pd.DataFrame(
        [
            {
                "mode": t.mode,
                "trial": t.trial,
                "params": json.dumps(t.params, ensure_ascii=False),
                "score": t.score,
                "annual_return": t.annual_return,
                "max_drawdown": t.max_drawdown,
                "positive_window_ratio": t.positive_window_ratio,
                "trades": t.trades,
                "factor_alignment": t.factor_alignment,
                "started_at": t.started_at,
                "elapsed_sec": t.elapsed_sec,
            }
            for t in trials
        ]
    )
    trial_log = mode_dir / "trials.csv"
    trials_df.to_csv(trial_log, index=False)

    return ModeOptimizationSummary(
        mode=mode,
        trials=int(len(trials)),
        best_score=float(best_score if best_score > -1e17 else 0.0),
        best_params=best_params or {},
        best_metrics=best_metrics,
        budget_seconds=float(max(0.0, deadline_ts - start_ts)),
        elapsed_seconds=float(time.time() - start_ts),
        trial_log_path=str(trial_log),
    )


def _render_report(summary: ResearchRunSummary) -> str:
    lines: list[str] = []
    lines.append(f"# 实盘多模式回测与优化报告 | {summary.ended_at[:10]}")
    lines.append("")
    lines.append("## 运行概览")
    lines.append(f"- 时间预算: `{summary.hours_budget:.2f}` 小时")
    lines.append(f"- 实际耗时: `{summary.elapsed_seconds/3600:.2f}` 小时")
    lines.append(f"- 区间: `{summary.start_date} ~ {summary.end_date}`")
    lines.append(f"- 回测截止日(严格隔离): `{summary.cutoff_date}`")
    lines.append(f"- 截止时间戳: `{summary.cutoff_ts}`")
    lines.append(f"- 回测行情最大时间戳(<=截止日): `{summary.bar_max_ts}`")
    lines.append(f"- 回测新闻最大时间戳(<=截止日): `{summary.news_max_ts}`")
    lines.append(f"- 回测研报最大时间戳(<=截止日): `{summary.report_max_ts}`")
    lines.append(f"- 复盘窗口天数: `{summary.review_days}`")
    lines.append(f"- 覆盖标的数: `{summary.universe_count}`")
    lines.append(f"- 行情记录数: `{summary.bars_rows}`")
    lines.append(f"- 复盘行情记录数: `{summary.review_bars_rows}`")
    lines.append(f"- 复盘行情最大时间戳(>截止日): `{summary.review_bar_max_ts}`")
    lines.append(f"- 回测新闻记录数(<=截止日): `{summary.news_records}`")
    lines.append(f"- 回测研报记录数(<=截止日): `{summary.report_records}`")
    lines.append(f"- 复盘新闻记录数(>截止日): `{summary.review_news_records}`")
    lines.append(f"- 复盘研报记录数(>截止日): `{summary.review_report_records}`")
    lines.append(f"- 复盘新闻最大时间戳(>截止日): `{summary.review_news_max_ts}`")
    lines.append(f"- 复盘研报最大时间戳(>截止日): `{summary.review_report_max_ts}`")
    lines.append("")
    lines.append("## 模式结果")
    for item in summary.mode_summaries:
        lines.append(f"### {item.mode}")
        lines.append(f"- trials: `{item.trials}`")
        lines.append(f"- best_score: `{item.best_score:.6f}`")
        lines.append(f"- best_params: `{item.best_params}`")
        lines.append(f"- best_metrics: `{item.best_metrics}`")
        lines.append(f"- trial_log: `{item.trial_log_path}`")
        lines.append("")
    lines.append("## 说明")
    lines.append("- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。")
    lines.append("- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。")
    lines.append("- 回测优化只使用截止日及之前新闻/研报；截止日之后数据仅用于复盘，不参与回测评分。")
    lines.append("- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。")
    return "\n".join(lines) + "\n"


def run_research_backtest(
    *,
    output_root: Path,
    core_symbols: list[str],
    start: date,
    end: date,
    hours_budget: float = 10.0,
    max_symbols: int = 120,
    report_symbol_cap: int = 40,
    workers: int = 8,
    max_trials_per_mode: int = 500,
    seed: int = 42,
    modes: list[str] | None = None,
    review_days: int = 5,
) -> ResearchRunSummary:
    t_start = time.time()
    started_at = datetime.now().isoformat()
    run_dir = output_root / "research" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "artifacts" / "research_cache"

    bundle = load_real_data_bundle(
        core_symbols=core_symbols,
        start=start,
        end=end,
        max_symbols=max_symbols,
        report_symbol_cap=report_symbol_cap,
        workers=workers,
        cache_dir=cache_dir,
        cache_ttl_hours=8.0,
        strict_cutoff=end,
        review_days=int(review_days),
        include_post_review=True,
    )
    cutoff_iso = (bundle.cutoff_date or end).isoformat()
    cutoff_ts = str(bundle.cutoff_ts or f"{cutoff_iso}T23:59:59")
    bars = bundle.bars
    bars_path = run_dir / "bars_used.parquet"
    if not bars.empty:
        try:
            bars.to_parquet(bars_path, index=False)
        except Exception:
            bars.to_csv(run_dir / "bars_used.csv", index=False)

    news_path = run_dir / "news_daily_pre_cutoff.csv"
    report_path = run_dir / "report_daily_pre_cutoff.csv"
    review_news_path = run_dir / "news_daily_post_cutoff_review.csv"
    review_report_path = run_dir / "report_daily_post_cutoff_review.csv"
    bundle.news_daily.rename("news_score").to_csv(news_path, header=True)
    bundle.report_daily.rename("report_score").to_csv(report_path, header=True)
    bundle.review_news_daily.rename("news_score").to_csv(review_news_path, header=True)
    bundle.review_report_daily.rename("report_score").to_csv(review_report_path, header=True)
    write_markdown(
        run_dir / "temporal_policy.md",
        (
            "# 时间隔离策略\n\n"
            f"- 回测截止日: `{(bundle.cutoff_date or end).isoformat()}`\n"
            f"- 复盘窗口天数: `{int(bundle.review_days)}`\n"
            "- `pre_cutoff` 文件仅用于回测与参数优化。\n"
            "- `post_cutoff_review` 文件仅用于盘后复盘与研报吸收，不参与回测评分。\n"
        ),
    )

    # Budget is reserved for optimization loops; data fetch is cached and excluded from budget accounting.
    deadline_ts = time.time() + max(0.05, float(hours_budget)) * 3600.0
    rng = np.random.default_rng(seed)

    mode_summaries: list[ModeOptimizationSummary] = []
    mode_names = [m.strip() for m in (modes or ["ultra_short", "swing", "long"]) if str(m).strip()]
    mode_names = [m for m in mode_names if m in MODE_SPACES]
    if not mode_names:
        mode_names = ["ultra_short", "swing", "long"]
    for i, mode in enumerate(mode_names):
        remain = deadline_ts - time.time()
        if remain <= 3:
            break
        mode_budget_end = time.time() + remain / max(1, len(mode_names) - i)
        summary = _optimize_one_mode(
            mode=mode,
            space=MODE_SPACES[mode],
            bars=bars,
            start=start,
            end=end,
            news_daily=bundle.news_daily,
            report_daily=bundle.report_daily,
            deadline_ts=min(deadline_ts, mode_budget_end),
            max_trials=max_trials_per_mode,
            rng=rng,
            out_dir=run_dir,
        )
        mode_summaries.append(summary)

    ended_at = datetime.now().isoformat()
    run_summary = ResearchRunSummary(
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=float(time.time() - t_start),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        cutoff_date=cutoff_iso,
        cutoff_ts=cutoff_ts,
        bar_max_ts=str(bundle.bar_max_ts),
        news_max_ts=str(bundle.news_max_ts),
        report_max_ts=str(bundle.report_max_ts),
        review_days=int(bundle.review_days),
        hours_budget=float(hours_budget),
        universe_count=int(len(bundle.universe)),
        bars_rows=int(len(bundle.bars)),
        news_records=int(bundle.news_records),
        report_records=int(bundle.report_records),
        review_bars_rows=int(len(bundle.review_bars)),
        review_bar_max_ts=str(bundle.review_bar_max_ts),
        review_news_max_ts=str(bundle.review_news_max_ts),
        review_report_max_ts=str(bundle.review_report_max_ts),
        review_news_records=int(bundle.review_news_records),
        review_report_records=int(bundle.review_report_records),
        data_fetch_stats=bundle.fetch_stats,
        mode_summaries=mode_summaries,
        output_dir=str(run_dir),
    )

    write_json(run_dir / "summary.json", run_summary.to_dict())
    write_markdown(run_dir / "report.md", _render_report(run_summary))
    write_markdown(run_dir / "best_params.yaml", yaml.safe_dump({m.mode: {"best_params": m.best_params, "best_metrics": m.best_metrics} for m in mode_summaries}, allow_unicode=True, sort_keys=False))
    return run_summary
