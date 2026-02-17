from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lie_engine.backtest.engine import BacktestConfig, run_event_backtest
from lie_engine.data.storage import write_json, write_markdown
from lie_engine.research.real_data import load_real_data_bundle


@dataclass(slots=True)
class StrategyCandidateResult:
    name: str
    rationale: str
    params: dict[str, float | int]
    train_metrics: dict[str, float | int]
    validation_metrics: dict[str, float | int]
    factor_alignment_train: float
    factor_alignment_validation: float
    score: float
    review_metrics: dict[str, float | int] = field(default_factory=dict)
    factor_alignment_review: float = 0.0
    robustness_score: float = 0.0
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StrategyLabSummary:
    started_at: str
    ended_at: str
    elapsed_seconds: float
    start_date: str
    end_date: str
    train_end_date: str
    validation_start_date: str
    cutoff_date: str
    cutoff_ts: str
    bar_max_ts: str
    news_max_ts: str
    report_max_ts: str
    review_days: int
    universe_count: int
    bars_rows: int
    review_bars_rows: int
    news_records: int
    report_records: int
    market_insights: dict[str, float]
    report_insights: dict[str, float]
    review_start_date: str = ""
    review_end_date: str = ""
    review_bar_max_ts: str = ""
    review_news_max_ts: str = ""
    review_report_max_ts: str = ""
    review_news_records: int = 0
    review_report_records: int = 0
    candidates: list[StrategyCandidateResult] = field(default_factory=list)
    best_candidate: dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""
    data_fetch_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [c.to_dict() for c in self.candidates]
        return data


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _safe_z(v: float, mean: float, std: float) -> float:
    if not np.isfinite(std) or std < 1e-9:
        return 0.0
    return float(np.clip((v - mean) / std, -4.0, 4.0))


def _daily_proxy_returns(bars: pd.DataFrame) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype=float)
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame["ts"])
    px = frame.groupby("ts", as_index=False)["close"].mean().sort_values("ts")
    px["ret"] = px["close"].pct_change()
    px = px.dropna(subset=["ret"])
    if px.empty:
        return pd.Series(dtype=float)
    return pd.Series(px["ret"].to_numpy(), index=px["ts"].dt.date.tolist()).sort_index()


def _market_insights(bars: pd.DataFrame) -> dict[str, float]:
    ret = _daily_proxy_returns(bars)
    if ret.empty:
        return {
            "trend_strength_z": 0.0,
            "volatility_z": 0.0,
            "tail_risk_z": 0.0,
            "mean_return": 0.0,
            "volatility": 0.0,
        }
    trend_raw = float(ret.rolling(20).mean().dropna().mean()) if len(ret) >= 20 else float(ret.mean())
    vol_raw = float(ret.rolling(20).std(ddof=0).dropna().mean()) if len(ret) >= 20 else float(ret.std(ddof=0))
    tail_raw = float(ret.quantile(0.05))
    base_mean = float(ret.mean())
    base_std = float(ret.std(ddof=0))
    return {
        "trend_strength_z": _safe_z(trend_raw, base_mean, base_std),
        "volatility_z": _safe_z(vol_raw, float(ret.std(ddof=0)), float(ret.std(ddof=0))),
        "tail_risk_z": _safe_z(tail_raw, base_mean, base_std),
        "mean_return": base_mean,
        "volatility": float(ret.std(ddof=0)),
    }


def _report_insights(news_daily: pd.Series, report_daily: pd.Series) -> dict[str, float]:
    ns = news_daily.astype(float).sort_index() if not news_daily.empty else pd.Series(dtype=float)
    rs = report_daily.astype(float).sort_index() if not report_daily.empty else pd.Series(dtype=float)
    idx = sorted(set(ns.index.tolist()) | set(rs.index.tolist()))
    if not idx:
        return {
            "news_bias_z": 0.0,
            "report_bias_z": 0.0,
            "news_report_agreement": 0.0,
            "news_bias": 0.0,
            "report_bias": 0.0,
        }
    ns = ns.reindex(idx).fillna(0.0)
    rs = rs.reindex(idx).fillna(0.0)

    news_bias = float(ns.mean())
    report_bias = float(rs.mean())
    news_std = float(ns.std(ddof=0))
    report_std = float(rs.std(ddof=0))
    if news_std < 1e-9 or report_std < 1e-9:
        agreement = 0.0
    else:
        agreement = float(np.clip(np.corrcoef(ns.to_numpy(), rs.to_numpy())[0, 1], -1.0, 1.0))
        if not np.isfinite(agreement):
            agreement = 0.0

    return {
        "news_bias_z": _safe_z(news_bias, 0.0, max(1e-6, news_std)),
        "report_bias_z": _safe_z(report_bias, 0.0, max(1e-6, report_std)),
        "news_report_agreement": agreement,
        "news_bias": news_bias,
        "report_bias": report_bias,
    }


def _factor_alignment(result_curve: list[dict[str, Any]], news_daily: pd.Series, report_daily: pd.Series) -> float:
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

    idx = eq["date"].tolist()
    ns = news_daily.reindex(idx).fillna(0.0) if not news_daily.empty else pd.Series(0.0, index=idx)
    rs = report_daily.reindex(idx).fillna(0.0) if not report_daily.empty else pd.Series(0.0, index=idx)
    factor = 0.5 * ns + 0.5 * rs
    aligned = pd.DataFrame({"ret": eq["ret"].to_numpy(), "factor": factor.to_numpy()})
    if aligned["ret"].std(ddof=0) < 1e-9 or aligned["factor"].std(ddof=0) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(aligned["factor"].shift(1).fillna(0.0), aligned["ret"])[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _split_dates(bars: pd.DataFrame) -> tuple[date, date]:
    dates = sorted(pd.to_datetime(bars["ts"]).dt.date.unique().tolist())
    if len(dates) <= 2:
        return dates[0], dates[-1]
    if len(dates) < 80:
        split = max(1, min(len(dates) - 1, len(dates) // 2))
        return dates[split - 1], dates[split]
    split = int(len(dates) * 0.72)
    split = max(60, min(len(dates) - 20, split))
    train_end = dates[split - 1]
    valid_start = dates[split]
    return train_end, valid_start


def _generate_candidates(
    market: dict[str, float],
    report: dict[str, float],
    candidate_count: int,
) -> list[dict[str, Any]]:
    base_templates: list[dict[str, Any]] = [
        {"name": "trend_convex", "hold_days": 9, "max_daily_trades": 2, "signal_confidence_min": 50.0, "convexity_min": 2.2, "rationale": "趋势驱动+凸性约束"},
        {"name": "report_momentum", "hold_days": 12, "max_daily_trades": 2, "signal_confidence_min": 48.0, "convexity_min": 2.0, "rationale": "研报偏向驱动的动量延续"},
        {"name": "news_reversion", "hold_days": 5, "max_daily_trades": 3, "signal_confidence_min": 45.0, "convexity_min": 1.8, "rationale": "新闻冲击后的均值回归"},
        {"name": "balanced_flow", "hold_days": 7, "max_daily_trades": 2, "signal_confidence_min": 47.0, "convexity_min": 2.0, "rationale": "新闻与研报一致性中庸策略"},
        {"name": "defensive_tail", "hold_days": 6, "max_daily_trades": 1, "signal_confidence_min": 52.0, "convexity_min": 2.6, "rationale": "尾部风险防御优先"},
        {"name": "aggressive_breakout", "hold_days": 4, "max_daily_trades": 4, "signal_confidence_min": 43.0, "convexity_min": 1.6, "rationale": "高波动突破捕捉"},
    ]

    trend = float(market.get("trend_strength_z", 0.0))
    vol = float(market.get("volatility_z", 0.0))
    tail = float(market.get("tail_risk_z", 0.0))
    news_bias = float(report.get("news_bias_z", 0.0))
    report_bias = float(report.get("report_bias_z", 0.0))
    agree = float(report.get("news_report_agreement", 0.0))

    out: list[dict[str, Any]] = []
    for i in range(max(1, int(candidate_count))):
        tpl = base_templates[i % len(base_templates)].copy()
        regime_push = float(np.tanh(trend - 0.6 * tail))
        info_push = float(np.tanh(0.6 * report_bias + 0.4 * news_bias))
        conf = float(tpl["signal_confidence_min"]) - 5.0 * info_push + 2.0 * max(0.0, -agree)
        convex = float(tpl["convexity_min"]) + 0.35 * max(0.0, vol) + 0.20 * max(0.0, -tail)
        hold = int(round(float(tpl["hold_days"]) + 3.0 * regime_push))
        trades = int(round(float(tpl["max_daily_trades"]) + (1.0 if abs(info_push) > 0.8 else 0.0)))

        tpl["signal_confidence_min"] = _clip(conf, 28.0, 75.0)
        tpl["convexity_min"] = _clip(convex, 1.0, 3.5)
        tpl["hold_days"] = int(max(1, min(20, hold)))
        tpl["max_daily_trades"] = int(max(1, min(5, trades)))
        tpl["name"] = f"{tpl['name']}_{i+1:02d}"
        out.append(tpl)
    return out


def _score_candidate(
    *,
    train_bt,
    valid_bt,
    align_train: float,
    align_valid: float,
    review_bt=None,
    align_review: float = 0.0,
    robustness_score: float = 0.0,
) -> float:
    trades_valid = int(getattr(valid_bt, "trades", 0))
    trade_penalty = 0.0 if trades_valid >= 5 else 0.08 * float(5 - trades_valid)
    base = float(
        0.25 * float(getattr(train_bt, "annual_return", 0.0))
        + 0.55 * float(getattr(valid_bt, "annual_return", 0.0))
        + 0.65 * float(getattr(valid_bt, "positive_window_ratio", 0.0))
        + 0.20 * float(align_valid)
        + 0.08 * float(align_train)
        - 2.40 * float(getattr(valid_bt, "max_drawdown", 0.0))
        - 0.35 * float(getattr(valid_bt, "violations", 0))
        - trade_penalty
    )
    if review_bt is None:
        return base
    review_trades = int(getattr(review_bt, "trades", 0))
    review_penalty = 0.0 if review_trades >= 2 else 0.04 * float(2 - review_trades)
    return float(
        base
        + 0.18 * float(getattr(review_bt, "annual_return", 0.0))
        + 0.22 * float(getattr(review_bt, "positive_window_ratio", 0.0))
        + 0.10 * float(align_review)
        + 0.20 * float(robustness_score)
        - 1.20 * float(getattr(review_bt, "max_drawdown", 0.0))
        - 0.20 * float(getattr(review_bt, "violations", 0))
        - review_penalty
    )


def _robustness_score(*, valid_bt, review_bt, align_valid: float, align_review: float) -> float:
    valid_s = (
        0.50 * float(getattr(valid_bt, "positive_window_ratio", 0.0))
        + 0.20 * max(0.0, float(align_valid))
        + 0.30 * max(0.0, 1.0 - float(getattr(valid_bt, "max_drawdown", 1.0)))
    )
    review_s = (
        0.55 * float(getattr(review_bt, "positive_window_ratio", 0.0))
        + 0.15 * max(0.0, float(align_review))
        + 0.30 * max(0.0, 1.0 - float(getattr(review_bt, "max_drawdown", 1.0)))
    )
    return float(max(0.0, min(1.0, 0.55 * valid_s + 0.45 * review_s)))


def _render_report(summary: StrategyLabSummary) -> str:
    lines: list[str] = []
    lines.append(f"# 策略学习实验室报告 | {summary.ended_at[:10]}")
    lines.append("")
    lines.append("## 概览")
    lines.append(f"- 区间: `{summary.start_date} ~ {summary.end_date}`")
    lines.append(f"- 训练截止: `{summary.train_end_date}`")
    lines.append(f"- 验证起点: `{summary.validation_start_date}`")
    lines.append(f"- 截止时间戳: `{summary.cutoff_ts}`")
    lines.append(f"- 训练行情最大时间戳: `{summary.bar_max_ts}`")
    lines.append(f"- 训练新闻最大时间戳: `{summary.news_max_ts}`")
    lines.append(f"- 训练研报最大时间戳: `{summary.report_max_ts}`")
    if summary.review_start_date and summary.review_end_date:
        lines.append(f"- 复盘检验窗口: `{summary.review_start_date} ~ {summary.review_end_date}`")
        lines.append(f"- 复盘行情最大时间戳: `{summary.review_bar_max_ts}`")
        lines.append(f"- 复盘新闻最大时间戳: `{summary.review_news_max_ts}`")
        lines.append(f"- 复盘研报最大时间戳: `{summary.review_report_max_ts}`")
    lines.append(f"- 覆盖标的: `{summary.universe_count}`")
    lines.append(f"- 行情记录: `{summary.bars_rows}`")
    lines.append(f"- 复盘行情记录: `{summary.review_bars_rows}`")
    lines.append(f"- 新闻记录: `{summary.news_records}`")
    lines.append(f"- 研报记录: `{summary.report_records}`")
    lines.append(f"- 复盘新闻记录: `{summary.review_news_records}`")
    lines.append(f"- 复盘研报记录: `{summary.review_report_records}`")
    lines.append("")
    lines.append("## 市场学习信号")
    for k, v in summary.market_insights.items():
        lines.append(f"- `{k}`: `{float(v):.4f}`")
    lines.append("")
    lines.append("## 报告学习信号")
    for k, v in summary.report_insights.items():
        lines.append(f"- `{k}`: `{float(v):.4f}`")
    lines.append("")
    lines.append("## 候选策略评分")
    for c in summary.candidates:
        lines.append(f"### {c.name}")
        lines.append(f"- rationale: {c.rationale}")
        lines.append(f"- params: `{c.params}`")
        lines.append(f"- train: `{c.train_metrics}`")
        lines.append(f"- validation: `{c.validation_metrics}`")
        lines.append(f"- review: `{c.review_metrics}`")
        lines.append(
            f"- align(train/valid/review): `{c.factor_alignment_train:.4f}/{c.factor_alignment_validation:.4f}/{c.factor_alignment_review:.4f}`"
        )
        lines.append(f"- robustness: `{c.robustness_score:.4f}` | accepted: `{c.accepted}`")
        lines.append(f"- score: `{c.score:.6f}`")
        lines.append("")
    lines.append("## 结论")
    if summary.best_candidate:
        lines.append(f"- 最优策略: `{summary.best_candidate.get('name', '')}`")
        lines.append(f"- 最优参数: `{summary.best_candidate.get('params', {})}`")
    else:
        lines.append("- 无有效候选策略")
    return "\n".join(lines) + "\n"


def run_strategy_lab(
    *,
    output_root: Path,
    core_symbols: list[str],
    start: date,
    end: date,
    max_symbols: int = 120,
    report_symbol_cap: int = 40,
    workers: int = 8,
    review_days: int = 5,
    candidate_count: int = 10,
) -> StrategyLabSummary:
    t0 = datetime.now()
    run_dir = output_root / "research" / f"strategy_lab_{t0:%Y%m%d_%H%M%S}"
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

    bars = bundle.bars.copy()
    review_bars = bundle.review_bars.copy()
    if bars.empty:
        summary = StrategyLabSummary(
            started_at=t0.isoformat(),
            ended_at=datetime.now().isoformat(),
            elapsed_seconds=0.0,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            train_end_date=end.isoformat(),
            validation_start_date=end.isoformat(),
            cutoff_date=cutoff_iso,
            cutoff_ts=cutoff_ts,
            bar_max_ts=str(bundle.bar_max_ts),
            news_max_ts=str(bundle.news_max_ts),
            report_max_ts=str(bundle.report_max_ts),
            review_days=int(bundle.review_days),
            review_start_date="",
            review_end_date="",
            review_bar_max_ts=str(bundle.review_bar_max_ts),
            review_news_max_ts=str(bundle.review_news_max_ts),
            review_report_max_ts=str(bundle.review_report_max_ts),
            universe_count=int(len(bundle.universe)),
            bars_rows=0,
            review_bars_rows=int(len(review_bars)),
            news_records=int(bundle.news_records),
            report_records=int(bundle.report_records),
            review_news_records=int(bundle.review_news_records),
            review_report_records=int(bundle.review_report_records),
            market_insights={},
            report_insights={},
            candidates=[],
            best_candidate={},
            output_dir=str(run_dir),
            data_fetch_stats=bundle.fetch_stats,
        )
        write_json(run_dir / "summary.json", summary.to_dict())
        write_markdown(run_dir / "report.md", _render_report(summary))
        write_markdown(run_dir / "best_strategy.yaml", yaml.safe_dump({}, allow_unicode=True, sort_keys=False))
        return summary

    bars["ts"] = pd.to_datetime(bars["ts"])
    review_start: date | None = None
    review_end: date | None = None
    if not review_bars.empty:
        review_bars["ts"] = pd.to_datetime(review_bars["ts"])
        review_start = pd.to_datetime(review_bars["ts"]).dt.date.min()
        review_end = pd.to_datetime(review_bars["ts"]).dt.date.max()

    bars_all = pd.concat([bars, review_bars], ignore_index=True) if not review_bars.empty else bars.copy()
    bars_all = bars_all.sort_values(["ts", "symbol"]).reset_index(drop=True)
    train_end, valid_start = _split_dates(bars)

    market = _market_insights(bars)
    report = _report_insights(bundle.news_daily, bundle.report_daily)
    candidates_cfg = _generate_candidates(market=market, report=report, candidate_count=candidate_count)

    out_candidates: list[StrategyCandidateResult] = []
    for spec in candidates_cfg:
        cfg = BacktestConfig(
            signal_confidence_min=float(spec["signal_confidence_min"]),
            convexity_min=float(spec["convexity_min"]),
            max_daily_trades=int(spec["max_daily_trades"]),
            hold_days=int(spec["hold_days"]),
            regime_recalc_interval=5,
            signal_eval_interval=1,
            proxy_lookback=180,
        )
        train_bt = run_event_backtest(
            bars=bars,
            start=start,
            end=train_end,
            cfg=cfg,
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        valid_bt = run_event_backtest(
            bars=bars,
            start=valid_start,
            end=end,
            cfg=cfg,
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        align_train = _factor_alignment(train_bt.equity_curve, bundle.news_daily, bundle.report_daily)
        align_valid = _factor_alignment(valid_bt.equity_curve, bundle.news_daily, bundle.report_daily)
        review_bt = None
        align_review = 0.0
        review_metrics: dict[str, float | int] = {}
        robustness = float(
            max(
                0.0,
                min(
                    1.0,
                    0.65 * float(valid_bt.positive_window_ratio)
                    + 0.20 * max(0.0, float(align_valid))
                    + 0.15 * max(0.0, 1.0 - float(valid_bt.max_drawdown)),
                ),
            )
        )
        if review_start is not None and review_end is not None and review_start <= review_end:
            review_bt = run_event_backtest(
                bars=bars_all,
                start=review_start,
                end=review_end,
                cfg=cfg,
                trend_thr=0.6,
                mean_thr=0.4,
                atr_extreme=2.0,
            )
            align_review = _factor_alignment(review_bt.equity_curve, bundle.review_news_daily, bundle.review_report_daily)
            robustness = _robustness_score(
                valid_bt=valid_bt,
                review_bt=review_bt,
                align_valid=align_valid,
                align_review=align_review,
            )
            review_metrics = {
                "annual_return": float(review_bt.annual_return),
                "max_drawdown": float(review_bt.max_drawdown),
                "positive_window_ratio": float(review_bt.positive_window_ratio),
                "trades": int(review_bt.trades),
                "violations": int(review_bt.violations),
            }
        score = _score_candidate(
            train_bt=train_bt,
            valid_bt=valid_bt,
            align_train=align_train,
            align_valid=align_valid,
            review_bt=review_bt,
            align_review=align_review,
            robustness_score=robustness,
        )
        accepted = (
            float(valid_bt.positive_window_ratio) >= 0.70
            and float(valid_bt.max_drawdown) <= 0.18
            and int(valid_bt.violations) == 0
            and (review_bt is None or (float(review_bt.positive_window_ratio) >= 0.60 and float(review_bt.max_drawdown) <= 0.22))
        )
        out_candidates.append(
            StrategyCandidateResult(
                name=str(spec["name"]),
                rationale=str(spec["rationale"]),
                params={
                    "signal_confidence_min": float(spec["signal_confidence_min"]),
                    "convexity_min": float(spec["convexity_min"]),
                    "hold_days": int(spec["hold_days"]),
                    "max_daily_trades": int(spec["max_daily_trades"]),
                },
                train_metrics={
                    "annual_return": float(train_bt.annual_return),
                    "max_drawdown": float(train_bt.max_drawdown),
                    "positive_window_ratio": float(train_bt.positive_window_ratio),
                    "trades": int(train_bt.trades),
                },
                validation_metrics={
                    "annual_return": float(valid_bt.annual_return),
                    "max_drawdown": float(valid_bt.max_drawdown),
                    "positive_window_ratio": float(valid_bt.positive_window_ratio),
                    "trades": int(valid_bt.trades),
                    "violations": int(valid_bt.violations),
                },
                factor_alignment_train=align_train,
                factor_alignment_validation=align_valid,
                review_metrics=review_metrics,
                factor_alignment_review=align_review,
                robustness_score=robustness,
                accepted=accepted,
                score=score,
            )
        )

    out_candidates.sort(key=lambda x: (1 if x.accepted else 0, x.score), reverse=True)
    best = out_candidates[0].to_dict() if out_candidates else {}

    t1 = datetime.now()
    summary = StrategyLabSummary(
        started_at=t0.isoformat(),
        ended_at=t1.isoformat(),
        elapsed_seconds=float((t1 - t0).total_seconds()),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        train_end_date=train_end.isoformat(),
        validation_start_date=valid_start.isoformat(),
        cutoff_date=cutoff_iso,
        cutoff_ts=cutoff_ts,
        bar_max_ts=str(bundle.bar_max_ts),
        news_max_ts=str(bundle.news_max_ts),
        report_max_ts=str(bundle.report_max_ts),
        review_days=int(bundle.review_days),
        review_start_date=review_start.isoformat() if review_start else "",
        review_end_date=review_end.isoformat() if review_end else "",
        review_bar_max_ts=str(bundle.review_bar_max_ts),
        review_news_max_ts=str(bundle.review_news_max_ts),
        review_report_max_ts=str(bundle.review_report_max_ts),
        universe_count=int(len(bundle.universe)),
        bars_rows=int(len(bars)),
        review_bars_rows=int(len(review_bars)),
        news_records=int(bundle.news_records),
        report_records=int(bundle.report_records),
        review_news_records=int(bundle.review_news_records),
        review_report_records=int(bundle.review_report_records),
        market_insights=market,
        report_insights=report,
        candidates=out_candidates,
        best_candidate=best,
        output_dir=str(run_dir),
        data_fetch_stats=bundle.fetch_stats,
    )

    write_json(run_dir / "summary.json", summary.to_dict())
    write_markdown(run_dir / "report.md", _render_report(summary))
    write_markdown(run_dir / "best_strategy.yaml", yaml.safe_dump(best, allow_unicode=True, sort_keys=False))
    if out_candidates:
        pd.DataFrame([c.to_dict() for c in out_candidates]).to_csv(run_dir / "candidates.csv", index=False)
    return summary
