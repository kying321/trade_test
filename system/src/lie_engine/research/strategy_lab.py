from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import hashlib
import json
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
    max_drawdown_target: float
    review_max_drawdown_target: float
    drawdown_soft_band: float
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
    term_registry: dict[str, Any] = field(default_factory=dict)
    exposure_cap_applied: float = 0.0
    exposure_cap_components: dict[str, Any] = field(default_factory=dict)
    proxy_lookback_applied: int = 180

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [c.to_dict() for c in self.candidates]
        return data


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _term_registry_snapshot() -> dict[str, Any]:
    system_root = Path(__file__).resolve().parents[3]
    path = system_root / "config" / "term_atoms.yaml"
    out: dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists()),
        "version": "",
        "checksum_sha256": "",
        "atoms_total": 0,
        "l2_atoms": 0,
        "families": {},
    }
    if not path.exists():
        return out

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return out
    if not isinstance(payload, dict):
        return out

    atoms_raw = payload.get("atoms", [])
    atoms = atoms_raw if isinstance(atoms_raw, list) else []
    families: dict[str, int] = {}
    l2_atoms = 0
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        family = str(atom.get("family", "")).strip()
        if family:
            families[family] = families.get(family, 0) + 1
        if str(atom.get("formula_type", "")).strip() == "l2_native":
            l2_atoms += 1

    checksum = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    out["version"] = str(payload.get("version", ""))
    out["checksum_sha256"] = checksum
    out["atoms_total"] = int(len(atoms))
    out["l2_atoms"] = int(l2_atoms)
    out["families"] = families
    return out


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


def _brooks_market_proxies(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {
            "brooks_trend_bar_z": 0.0,
            "brooks_micro_channel_bias": 0.0,
            "brooks_two_legged_bias": 0.0,
            "brooks_exhaustion_z": 0.0,
        }

    work = bars.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    daily = (
        work.groupby("ts", as_index=False)
        .agg(
            open=("open", "mean"),
            high=("high", "mean"),
            low=("low", "mean"),
            close=("close", "mean"),
            volume=("volume", "sum"),
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )
    if len(daily) < 8:
        return {
            "brooks_trend_bar_z": 0.0,
            "brooks_micro_channel_bias": 0.0,
            "brooks_two_legged_bias": 0.0,
            "brooks_exhaustion_z": 0.0,
        }

    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            (daily["high"] - daily["low"]).abs(),
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean().bfill().ffill()
    bar_range = (daily["high"] - daily["low"]).replace(0.0, np.nan).abs().fillna(1e-9)
    body = (daily["close"] - daily["open"]).abs()
    upper_wick = daily["high"] - daily[["close", "open"]].max(axis=1)
    lower_wick = daily[["close", "open"]].min(axis=1) - daily["low"]
    close_loc = ((daily["close"] - daily["low"]) / bar_range).clip(0.0, 1.0)
    body_share = (body / bar_range).clip(0.0, 1.0)
    body_vs_atr = (body / atr14.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    vol_ma20 = daily["volume"].rolling(20, min_periods=5).mean().replace(0.0, np.nan).bfill().ffill()
    vol_support = ((daily["volume"] / vol_ma20 - 0.6) / 1.0).clip(0.0, 1.0).fillna(0.0)

    trend_bar_long = (
        (daily["close"] >= daily["open"]).astype(float)
        * close_loc
        * body_share
        * body_vs_atr
        * (0.55 + 0.45 * vol_support)
    )
    trend_bar_short = (
        (daily["close"] <= daily["open"]).astype(float)
        * (1.0 - close_loc)
        * body_share
        * body_vs_atr
        * (0.55 + 0.45 * vol_support)
    )
    trend_bar_quality = pd.Series(np.maximum(trend_bar_long.to_numpy(), trend_bar_short.to_numpy()), index=daily.index)

    high_step = daily["high"].diff()
    low_step = daily["low"].diff()
    micro_long = 0.5 * ((high_step >= -1e-9).astype(float) + (low_step >= -1e-9).astype(float))
    micro_short = 0.5 * ((high_step <= 1e-9).astype(float) + (low_step <= 1e-9).astype(float))
    micro_bias = (micro_long - micro_short).fillna(0.0)

    ma20 = daily["close"].rolling(20, min_periods=5).mean()
    ma60 = daily["close"].rolling(60, min_periods=10).mean()
    ret_px = daily["close"].diff().fillna(0.0)
    leg_threshold = (0.12 * atr14).fillna(0.0)
    down_leg = ((ret_px <= -leg_threshold) & (ma20 >= ma60)).astype(float)
    up_leg = ((ret_px >= leg_threshold) & (ma20 <= ma60)).astype(float)
    down_legs5 = down_leg.rolling(5, min_periods=3).sum()
    up_legs5 = up_leg.rolling(5, min_periods=3).sum()
    reclaim_long = (daily["close"] >= daily["high"].shift(1) * 0.999) & (ma20 >= ma60)
    reclaim_short = (daily["close"] <= daily["low"].shift(1) * 1.001) & (ma20 <= ma60)
    two_leg_long = ((down_legs5 >= 2.0) & reclaim_long).astype(float)
    two_leg_short = ((up_legs5 >= 2.0) & reclaim_short).astype(float)
    two_leg_bias = (two_leg_long - two_leg_short).fillna(0.0)

    roll_high20_prev = daily["high"].shift(1).rolling(20, min_periods=5).max()
    roll_low20_prev = daily["low"].shift(1).rolling(20, min_periods=5).min()
    upside_exhaust = ((upper_wick / body.replace(0.0, np.nan) - 0.45) / 1.6).clip(0.0, 1.0).fillna(0.0)
    downside_exhaust = ((lower_wick / body.replace(0.0, np.nan) - 0.45) / 1.6).clip(0.0, 1.0).fillna(0.0)
    impulse = ((body / (1.2 * atr14.replace(0.0, np.nan)) - 0.4)).clip(0.0, 1.0).fillna(0.0)
    exhaust_long = (
        (daily["close"] >= roll_high20_prev * 0.998).astype(float) * (0.5 * upside_exhaust + 0.5 * impulse * vol_support)
    )
    exhaust_short = (
        (daily["close"] <= roll_low20_prev * 1.002).astype(float) * (0.5 * downside_exhaust + 0.5 * impulse * vol_support)
    )
    exhaust = pd.Series(np.maximum(exhaust_long.to_numpy(), exhaust_short.to_numpy()), index=daily.index)

    def _recent_z(series: pd.Series, lookback: int = 20) -> float:
        clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        recent = float(clean.tail(lookback).mean())
        mean = float(clean.mean())
        std = float(clean.std(ddof=0))
        return _safe_z(recent, mean, std)

    micro_bias_recent = float(np.clip(micro_bias.tail(30).mean(), -1.0, 1.0)) if not micro_bias.empty else 0.0
    two_leg_recent = float(np.clip(two_leg_bias.tail(40).mean(), -1.0, 1.0)) if not two_leg_bias.empty else 0.0
    return {
        "brooks_trend_bar_z": float(np.clip(_recent_z(trend_bar_quality, lookback=20), -4.0, 4.0)),
        "brooks_micro_channel_bias": micro_bias_recent,
        "brooks_two_legged_bias": two_leg_recent,
        "brooks_exhaustion_z": float(np.clip(_recent_z(exhaust, lookback=20), -4.0, 4.0)),
    }


def _wyckoff_vpa_market_proxies(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {
            "wyckoff_accumulation_bias": 0.0,
            "wyckoff_distribution_bias": 0.0,
            "vpa_effort_result_bias": 0.0,
            "vpa_climax_z": 0.0,
        }

    work = bars.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    daily = (
        work.groupby("ts", as_index=False)
        .agg(
            open=("open", "mean"),
            high=("high", "mean"),
            low=("low", "mean"),
            close=("close", "mean"),
            volume=("volume", "sum"),
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )
    if len(daily) < 8:
        return {
            "wyckoff_accumulation_bias": 0.0,
            "wyckoff_distribution_bias": 0.0,
            "vpa_effort_result_bias": 0.0,
            "vpa_climax_z": 0.0,
        }

    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            (daily["high"] - daily["low"]).abs(),
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean().bfill().ffill().replace(0.0, np.nan)
    roll_high20_prev = daily["high"].shift(1).rolling(20, min_periods=5).max()
    roll_low20_prev = daily["low"].shift(1).rolling(20, min_periods=5).min()
    ma20 = daily["close"].rolling(20, min_periods=5).mean().bfill().ffill()
    ma60 = daily["close"].rolling(60, min_periods=10).mean().bfill().ffill()
    bar_range = (daily["high"] - daily["low"]).replace(0.0, np.nan).abs().fillna(1e-9)
    close_loc = ((daily["close"] - daily["low"]) / bar_range).clip(0.0, 1.0)
    upper_wick = (daily["high"] - daily[["close", "open"]].max(axis=1)).clip(lower=0.0)
    lower_wick = (daily[["close", "open"]].min(axis=1) - daily["low"]).clip(lower=0.0)

    vol_ma20 = daily["volume"].rolling(20, min_periods=5).mean().replace(0.0, np.nan).bfill().ffill()
    vol_ratio = (daily["volume"] / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    spring = (
        (daily["low"] < roll_low20_prev)
        & (daily["close"] > roll_low20_prev)
        & (close_loc > 0.55)
        & (vol_ratio >= 1.0)
    ).astype(float)
    upthrust = (
        (daily["high"] > roll_high20_prev)
        & (daily["close"] < roll_high20_prev)
        & (close_loc < 0.45)
        & (vol_ratio >= 1.0)
    ).astype(float)

    accumulation = ((ma20 >= ma60) & (daily["close"] >= ma20) & (daily["close"] >= prev_close)).astype(float)
    distribution = ((ma20 <= ma60) & (daily["close"] <= ma20) & (daily["close"] <= prev_close)).astype(float)

    demand_absorb = (((lower_wick - upper_wick) / bar_range + 0.5) * ((vol_ratio - 0.9) / 1.4).clip(0.0, 1.0)).clip(0.0, 1.0)
    supply_absorb = (((upper_wick - lower_wick) / bar_range + 0.5) * ((vol_ratio - 0.9) / 1.4).clip(0.0, 1.0)).clip(0.0, 1.0)

    wyckoff_long = 0.32 * spring + 0.24 * accumulation + 0.24 * demand_absorb
    wyckoff_short = 0.32 * upthrust + 0.24 * distribution + 0.24 * supply_absorb
    wyckoff_long = (wyckoff_long * (1.0 - 0.35 * upthrust)).clip(0.0, 1.0)
    wyckoff_short = (wyckoff_short * (1.0 - 0.35 * spring)).clip(0.0, 1.0)

    effort = ((daily["close"] - prev_close) / atr14).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    effort = effort.clip(-1.0, 1.0)
    effort_result = (effort * ((vol_ratio - 0.8) / 1.6).clip(0.0, 1.0)).clip(-1.0, 1.0)
    climax_raw = (vol_ratio * (bar_range / atr14).replace([np.inf, -np.inf], np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _recent(series: pd.Series, lookback: int) -> float:
        clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        return float(clean.tail(lookback).mean())

    def _recent_z(series: pd.Series, lookback: int = 20) -> float:
        clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        recent = float(clean.tail(lookback).mean())
        mean = float(clean.mean())
        std = float(clean.std(ddof=0))
        return _safe_z(recent, mean, std)

    accumulation_bias = float(np.clip(_recent(wyckoff_long - wyckoff_short, 30), -1.0, 1.0))
    distribution_bias = float(np.clip(_recent(wyckoff_short - wyckoff_long, 30), -1.0, 1.0))
    effort_bias = float(np.clip(_recent(effort_result, 30), -1.0, 1.0))
    climax_z = float(np.clip(_recent_z(climax_raw, lookback=20), -4.0, 4.0))

    return {
        "wyckoff_accumulation_bias": accumulation_bias,
        "wyckoff_distribution_bias": distribution_bias,
        "vpa_effort_result_bias": effort_bias,
        "vpa_climax_z": climax_z,
    }


def _market_insights(bars: pd.DataFrame) -> dict[str, float]:
    ret = _daily_proxy_returns(bars)
    if ret.empty:
        return {
            "trend_strength_z": 0.0,
            "volatility_z": 0.0,
            "tail_risk_z": 0.0,
            "mean_return": 0.0,
            "volatility": 0.0,
            **_brooks_market_proxies(bars),
            **_wyckoff_vpa_market_proxies(bars),
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
        **_brooks_market_proxies(bars),
        **_wyckoff_vpa_market_proxies(bars),
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


def _latest_feasible_ablation_exposure(*, output_root: Path, start: date, end: date) -> dict[str, Any]:
    research_dir = output_root / "research"
    out: dict[str, Any] = {
        "found": False,
        "path": "",
        "exposure_scale": 0.0,
        "case_name": "",
        "risk_feasible": False,
    }
    if not research_dir.exists():
        return out

    runs = sorted(
        [p for p in research_dir.iterdir() if p.is_dir() and p.name.startswith("theory_ablation_")],
        key=lambda p: p.name,
        reverse=True,
    )
    best_match: tuple[int, int, dict[str, Any]] | None = None
    for run in runs:
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        start_raw = str(payload.get("start", ""))
        end_raw = str(payload.get("end", ""))
        try:
            run_start = date.fromisoformat(start_raw)
            run_end = date.fromisoformat(end_raw)
        except ValueError:
            continue
        risk_feasible = bool(payload.get("risk_feasible", False))
        feasible = payload.get("best_feasible_case", {})
        if not isinstance(feasible, dict):
            feasible = {}
        exposure = _clip(float(feasible.get("exposure_scale", 0.0)), 0.0, 1.0)
        if not risk_feasible or exposure <= 0.0:
            continue

        priority = 9
        gap_days = 999999
        if run_start == start and run_end == end:
            priority = 0
            gap_days = 0
        elif run_end == end and run_start <= start:
            priority = 1
            gap_days = int((start - run_start).days)
        elif run_start <= start and run_end >= end:
            priority = 2
            gap_days = int((start - run_start).days + (run_end - end).days)
        else:
            continue

        candidate = {
            "found": True,
            "path": str(summary_path),
            "exposure_scale": float(exposure),
            "case_name": str(feasible.get("name", "")),
            "risk_feasible": bool(risk_feasible),
            "match_priority": int(priority),
            "match_gap_days": int(gap_days),
            "match_window_start": run_start.isoformat(),
            "match_window_end": run_end.isoformat(),
        }
        key = (priority, gap_days)
        if best_match is None or key < (best_match[0], best_match[1]):
            best_match = (priority, gap_days, candidate)
            if priority == 0:
                break

    if best_match is not None:
        out.update(best_match[2])
        return out
    return out


def _resolve_exposure_cap(
    *,
    output_root: Path,
    start: date,
    end: date,
    max_drawdown_target: float,
    market: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    base_cap = _clip(float(max_drawdown_target) / 0.22, 0.10, 0.35)
    ablation = _latest_feasible_ablation_exposure(output_root=output_root, start=start, end=end)
    ablation_cap = float(base_cap)
    if bool(ablation.get("found", False)):
        ablation_cap = _clip(float(ablation.get("exposure_scale", 0.0)) * 1.15, 0.08, float(base_cap))

    brooks_hazard = max(0.0, float(market.get("brooks_exhaustion_z", 0.0)))
    brooks_support = max(0.0, float(market.get("brooks_trend_bar_z", 0.0))) + 0.5 * abs(float(market.get("brooks_two_legged_bias", 0.0)))
    hazard_mult = _clip(1.0 - 0.10 * brooks_hazard + 0.03 * brooks_support, 0.75, 1.05)

    raw_cap = min(float(base_cap), float(ablation_cap))
    final_cap = _clip(raw_cap * hazard_mult, 0.08, float(base_cap))
    components: dict[str, Any] = {
        "base_cap": float(base_cap),
        "ablation_cap": float(ablation_cap),
        "hazard_multiplier": float(hazard_mult),
        "raw_cap": float(raw_cap),
        "ablation_found": bool(ablation.get("found", False)),
        "ablation_case_name": str(ablation.get("case_name", "")),
        "ablation_summary_path": str(ablation.get("path", "")),
        "ablation_match_priority": int(ablation.get("match_priority", -1)),
        "ablation_match_gap_days": int(ablation.get("match_gap_days", -1)),
        "ablation_match_window_start": str(ablation.get("match_window_start", "")),
        "ablation_match_window_end": str(ablation.get("match_window_end", "")),
        "brooks_hazard": float(brooks_hazard),
        "brooks_support": float(brooks_support),
    }
    return float(final_cap), components


def _generate_candidates(
    market: dict[str, float],
    report: dict[str, float],
    candidate_count: int,
    exposure_cap: float = 0.35,
    low_activity_mode: bool = False,
    crypto_mode: bool = False,
) -> list[dict[str, Any]]:
    base_templates: list[dict[str, Any]] = [
        {
            "name": "trend_convex",
            "hold_days": 9,
            "max_daily_trades": 2,
            "signal_confidence_min": 32.0,
            "convexity_min": 2.2,
            "exposure_scale": 0.28,
            "theory_ict_weight": 1.2,
            "theory_brooks_weight": 0.9,
            "theory_lie_weight": 1.4,
            "theory_wyckoff_weight": 1.0,
            "theory_vpa_weight": 0.9,
            "theory_confidence_boost_max": 5.8,
            "theory_penalty_max": 5.4,
            "theory_min_confluence": 0.34,
            "theory_conflict_fuse": 0.76,
            "execution_confirm_enabled": True,
            "execution_confirm_lookahead": 2,
            "execution_confirm_loss_mult": 0.62,
            "execution_anti_martingale_enabled": True,
            "execution_anti_martingale_step": 0.18,
            "execution_anti_martingale_floor": 0.65,
            "execution_anti_martingale_ceiling": 1.35,
            "rationale": "趋势驱动+凸性约束",
        },
        {
            "name": "report_momentum",
            "hold_days": 12,
            "max_daily_trades": 2,
            "signal_confidence_min": 30.0,
            "convexity_min": 2.0,
            "exposure_scale": 0.24,
            "theory_ict_weight": 1.0,
            "theory_brooks_weight": 1.0,
            "theory_lie_weight": 1.3,
            "theory_wyckoff_weight": 0.9,
            "theory_vpa_weight": 1.1,
            "theory_confidence_boost_max": 5.2,
            "theory_penalty_max": 6.0,
            "theory_min_confluence": 0.36,
            "theory_conflict_fuse": 0.72,
            "execution_confirm_enabled": True,
            "execution_confirm_lookahead": 2,
            "execution_confirm_loss_mult": 0.65,
            "execution_anti_martingale_enabled": True,
            "execution_anti_martingale_step": 0.16,
            "execution_anti_martingale_floor": 0.70,
            "execution_anti_martingale_ceiling": 1.30,
            "rationale": "研报偏向驱动的动量延续",
        },
        {
            "name": "news_reversion",
            "hold_days": 5,
            "max_daily_trades": 3,
            "signal_confidence_min": 26.0,
            "convexity_min": 1.8,
            "exposure_scale": 0.22,
            "theory_ict_weight": 1.2,
            "theory_brooks_weight": 1.3,
            "theory_lie_weight": 0.9,
            "theory_wyckoff_weight": 1.2,
            "theory_vpa_weight": 1.0,
            "theory_confidence_boost_max": 4.6,
            "theory_penalty_max": 6.8,
            "theory_min_confluence": 0.40,
            "theory_conflict_fuse": 0.68,
            "execution_confirm_enabled": True,
            "execution_confirm_lookahead": 3,
            "execution_confirm_loss_mult": 0.68,
            "execution_anti_martingale_enabled": True,
            "execution_anti_martingale_step": 0.14,
            "execution_anti_martingale_floor": 0.72,
            "execution_anti_martingale_ceiling": 1.24,
            "rationale": "新闻冲击后的均值回归",
        },
        {
            "name": "balanced_flow",
            "hold_days": 7,
            "max_daily_trades": 2,
            "signal_confidence_min": 28.0,
            "convexity_min": 2.0,
            "exposure_scale": 0.23,
            "theory_ict_weight": 1.0,
            "theory_brooks_weight": 1.0,
            "theory_lie_weight": 1.1,
            "theory_wyckoff_weight": 0.9,
            "theory_vpa_weight": 0.9,
            "theory_confidence_boost_max": 5.0,
            "theory_penalty_max": 6.2,
            "theory_min_confluence": 0.38,
            "theory_conflict_fuse": 0.72,
            "execution_confirm_enabled": True,
            "execution_confirm_lookahead": 2,
            "execution_confirm_loss_mult": 0.66,
            "execution_anti_martingale_enabled": True,
            "execution_anti_martingale_step": 0.16,
            "execution_anti_martingale_floor": 0.68,
            "execution_anti_martingale_ceiling": 1.30,
            "rationale": "新闻与研报一致性中庸策略",
        },
        {
            "name": "defensive_tail",
            "hold_days": 6,
            "max_daily_trades": 1,
            "signal_confidence_min": 34.0,
            "convexity_min": 2.6,
            "exposure_scale": 0.18,
            "theory_ict_weight": 0.9,
            "theory_brooks_weight": 1.3,
            "theory_lie_weight": 1.2,
            "theory_wyckoff_weight": 1.1,
            "theory_vpa_weight": 1.2,
            "theory_confidence_boost_max": 4.2,
            "theory_penalty_max": 7.4,
            "theory_min_confluence": 0.44,
            "theory_conflict_fuse": 0.64,
            "execution_confirm_enabled": True,
            "execution_confirm_lookahead": 3,
            "execution_confirm_loss_mult": 0.72,
            "execution_anti_martingale_enabled": True,
            "execution_anti_martingale_step": 0.12,
            "execution_anti_martingale_floor": 0.74,
            "execution_anti_martingale_ceiling": 1.20,
            "rationale": "尾部风险防御优先",
        },
        {
            "name": "aggressive_breakout",
            "hold_days": 4,
            "max_daily_trades": 4,
            "signal_confidence_min": 24.0,
            "convexity_min": 1.6,
            "exposure_scale": 0.26,
            "theory_ict_weight": 1.4,
            "theory_brooks_weight": 0.8,
            "theory_lie_weight": 1.1,
            "theory_wyckoff_weight": 0.8,
            "theory_vpa_weight": 1.0,
            "theory_confidence_boost_max": 6.6,
            "theory_penalty_max": 4.8,
            "theory_min_confluence": 0.30,
            "theory_conflict_fuse": 0.82,
            "execution_confirm_enabled": True,
            "execution_confirm_lookahead": 1,
            "execution_confirm_loss_mult": 0.58,
            "execution_anti_martingale_enabled": True,
            "execution_anti_martingale_step": 0.22,
            "execution_anti_martingale_floor": 0.62,
            "execution_anti_martingale_ceiling": 1.42,
            "rationale": "高波动突破捕捉",
        },
    ]
    if bool(crypto_mode):
        crypto_templates: list[dict[str, Any]] = [
            {
                "name": "crypto_tactical",
                "hold_days": 2,
                "max_daily_trades": 5,
                "signal_confidence_min": 18.0,
                "convexity_min": 1.4,
                "exposure_scale": 0.20,
                "theory_ict_weight": 1.15,
                "theory_brooks_weight": 1.25,
                "theory_lie_weight": 0.95,
                "theory_wyckoff_weight": 1.15,
                "theory_vpa_weight": 1.00,
                "theory_confidence_boost_max": 4.8,
                "theory_penalty_max": 6.4,
                "theory_min_confluence": 0.30,
                "theory_conflict_fuse": 0.74,
                "execution_confirm_enabled": True,
                "execution_confirm_lookahead": 1,
                "execution_confirm_loss_mult": 0.70,
                "execution_anti_martingale_enabled": True,
                "execution_anti_martingale_step": 0.14,
                "execution_anti_martingale_floor": 0.66,
                "execution_anti_martingale_ceiling": 1.34,
                "rationale": "crypto微观结构驱动的快切",
            },
            {
                "name": "crypto_swing_flow",
                "hold_days": 3,
                "max_daily_trades": 4,
                "signal_confidence_min": 20.0,
                "convexity_min": 1.5,
                "exposure_scale": 0.22,
                "theory_ict_weight": 1.25,
                "theory_brooks_weight": 1.15,
                "theory_lie_weight": 1.00,
                "theory_wyckoff_weight": 1.10,
                "theory_vpa_weight": 1.05,
                "theory_confidence_boost_max": 5.1,
                "theory_penalty_max": 6.0,
                "theory_min_confluence": 0.32,
                "theory_conflict_fuse": 0.76,
                "execution_confirm_enabled": True,
                "execution_confirm_lookahead": 1,
                "execution_confirm_loss_mult": 0.68,
                "execution_anti_martingale_enabled": True,
                "execution_anti_martingale_step": 0.16,
                "execution_anti_martingale_floor": 0.64,
                "execution_anti_martingale_ceiling": 1.38,
                "rationale": "crypto波段顺势+订单流过滤",
            },
        ]
        base_templates = crypto_templates + base_templates

    trend = float(market.get("trend_strength_z", 0.0))
    vol = float(market.get("volatility_z", 0.0))
    tail = float(market.get("tail_risk_z", 0.0))
    brooks_trend_bar = float(market.get("brooks_trend_bar_z", 0.0))
    brooks_micro_bias = float(market.get("brooks_micro_channel_bias", 0.0))
    brooks_two_legged = float(market.get("brooks_two_legged_bias", 0.0))
    brooks_exhaust = float(market.get("brooks_exhaustion_z", 0.0))
    wyckoff_acc = float(market.get("wyckoff_accumulation_bias", 0.0))
    wyckoff_dist = float(market.get("wyckoff_distribution_bias", 0.0))
    vpa_effort = float(market.get("vpa_effort_result_bias", 0.0))
    vpa_climax = float(market.get("vpa_climax_z", 0.0))
    news_bias = float(report.get("news_bias_z", 0.0))
    report_bias = float(report.get("report_bias_z", 0.0))
    agree = float(report.get("news_report_agreement", 0.0))

    out: list[dict[str, Any]] = []
    for i in range(max(1, int(candidate_count))):
        tpl = base_templates[i % len(base_templates)].copy()
        regime_push = float(np.tanh(trend - 0.6 * tail))
        info_push = float(np.tanh(0.6 * report_bias + 0.4 * news_bias))
        brooks_support = float(
            np.tanh(
                0.50 * max(0.0, brooks_trend_bar)
                + 0.30 * abs(brooks_micro_bias)
                + 0.30 * abs(brooks_two_legged)
                - 0.55 * max(0.0, brooks_exhaust)
            )
        )
        brooks_hazard = float(np.tanh(0.65 * max(0.0, brooks_exhaust) + 0.20 * max(0.0, -brooks_micro_bias)))
        wyckoff_support = float(
            np.tanh(0.70 * max(0.0, wyckoff_acc) - 0.55 * max(0.0, wyckoff_dist) + 0.25 * max(0.0, vpa_effort))
        )
        wyckoff_hazard = float(
            np.tanh(0.70 * max(0.0, wyckoff_dist) + 0.30 * max(0.0, vpa_climax) - 0.25 * max(0.0, wyckoff_acc))
        )
        vpa_support = float(np.tanh(0.60 * max(0.0, vpa_effort) + 0.20 * max(0.0, wyckoff_acc) - 0.28 * max(0.0, vpa_climax)))
        vpa_hazard = float(
            np.tanh(0.60 * max(0.0, -vpa_effort) + 0.34 * max(0.0, vpa_climax) + 0.20 * max(0.0, wyckoff_dist))
        )

        conf = float(tpl["signal_confidence_min"]) - 5.0 * info_push + 2.0 * max(0.0, -agree)
        conf += 1.4 * max(0.0, brooks_hazard) - 1.1 * max(0.0, brooks_support)
        conf += 1.0 * max(0.0, wyckoff_hazard) - 0.85 * max(0.0, wyckoff_support)
        conf += 0.70 * max(0.0, vpa_hazard) - 0.65 * max(0.0, vpa_support)

        convex = float(tpl["convexity_min"]) + 0.35 * max(0.0, vol) + 0.20 * max(0.0, -tail)
        convex += 0.22 * max(0.0, brooks_hazard) - 0.10 * max(0.0, brooks_support)
        convex += 0.16 * max(0.0, wyckoff_hazard) - 0.07 * max(0.0, wyckoff_support)
        convex += 0.10 * max(0.0, vpa_hazard) - 0.06 * max(0.0, vpa_support)

        exposure = (
            float(tpl.get("exposure_scale", 0.60))
            + 0.10 * max(0.0, trend)
            - 0.22 * max(0.0, vol)
            - 0.25 * max(0.0, tail)
            - 0.15 * max(0.0, -agree)
            + 0.06 * max(0.0, brooks_support)
            - 0.10 * max(0.0, brooks_hazard)
            + 0.05 * max(0.0, wyckoff_support)
            - 0.07 * max(0.0, wyckoff_hazard)
            + 0.04 * max(0.0, vpa_support)
            - 0.06 * max(0.0, vpa_hazard)
        )

        hold = int(
            round(
                float(tpl["hold_days"])
                + 3.0 * regime_push
                + 1.0 * max(0.0, brooks_support)
                + 0.7 * max(0.0, wyckoff_support)
                - 2.0 * max(0.0, brooks_hazard)
                - 1.1 * max(0.0, wyckoff_hazard)
            )
        )
        trades = int(
            round(
                float(tpl["max_daily_trades"])
                + (1.0 if abs(info_push) > 0.8 else 0.0)
                + 0.8 * max(0.0, brooks_support)
                + 0.6 * max(0.0, vpa_support)
                - 1.2 * max(0.0, brooks_hazard)
                - 0.8 * max(0.0, vpa_hazard)
            )
        )

        ict_w = (
            float(tpl["theory_ict_weight"])
            + 0.25 * max(0.0, info_push)
            + 0.10 * max(0.0, trend)
            + 0.05 * max(0.0, wyckoff_support)
            - 0.08 * max(0.0, wyckoff_hazard)
        )
        brooks_w = (
            float(tpl["theory_brooks_weight"])
            + 0.20 * max(0.0, vol)
            + 0.20 * max(0.0, -agree)
            + 0.25 * max(0.0, brooks_support)
            - 0.20 * max(0.0, brooks_hazard)
            + 0.04 * max(0.0, vpa_support)
        )
        lie_w = (
            float(tpl["theory_lie_weight"])
            + 0.25 * max(0.0, trend)
            - 0.12 * max(0.0, vol)
            + 0.08 * max(0.0, brooks_support)
            + 0.10 * max(0.0, wyckoff_support)
            - 0.10 * max(0.0, brooks_hazard)
            - 0.08 * max(0.0, wyckoff_hazard)
        )
        wyckoff_w = (
            float(tpl["theory_wyckoff_weight"])
            + 0.26 * max(0.0, wyckoff_support)
            + 0.10 * max(0.0, vpa_support)
            + 0.08 * max(0.0, -agree)
            - 0.20 * max(0.0, wyckoff_hazard)
            - 0.06 * max(0.0, vpa_hazard)
        )
        vpa_w = (
            float(tpl["theory_vpa_weight"])
            + 0.24 * max(0.0, vpa_support)
            + 0.08 * max(0.0, info_push)
            - 0.16 * max(0.0, vpa_hazard)
            - 0.06 * max(0.0, wyckoff_hazard)
        )

        boost = (
            float(tpl["theory_confidence_boost_max"])
            + 0.9 * max(0.0, trend)
            - 0.6 * max(0.0, vol)
            + 0.30 * max(0.0, brooks_support)
            + 0.20 * max(0.0, wyckoff_support)
            + 0.14 * max(0.0, vpa_support)
            - 0.55 * max(0.0, brooks_hazard)
            - 0.20 * max(0.0, wyckoff_hazard)
            - 0.15 * max(0.0, vpa_hazard)
        )
        penalty = (
            float(tpl["theory_penalty_max"])
            + 0.8 * max(0.0, vol)
            + 0.7 * max(0.0, -agree)
            + 0.65 * max(0.0, brooks_hazard)
            + 0.32 * max(0.0, wyckoff_hazard)
            + 0.24 * max(0.0, vpa_hazard)
            - 0.18 * max(0.0, brooks_support)
            - 0.08 * max(0.0, wyckoff_support)
        )
        min_conf = (
            float(tpl["theory_min_confluence"])
            + 0.05 * max(0.0, vol)
            + 0.03 * max(0.0, -tail)
            + 0.03 * max(0.0, brooks_hazard)
            + 0.02 * max(0.0, wyckoff_hazard)
            + 0.02 * max(0.0, vpa_hazard)
            - 0.02 * max(0.0, brooks_support)
            - 0.01 * max(0.0, wyckoff_support)
        )
        conflict_fuse = (
            float(tpl["theory_conflict_fuse"])
            - 0.04 * max(0.0, vol)
            + 0.03 * max(0.0, trend)
            - 0.03 * max(0.0, brooks_hazard)
            - 0.02 * max(0.0, wyckoff_hazard)
            - 0.01 * max(0.0, vpa_hazard)
            + 0.01 * max(0.0, brooks_support)
            + 0.01 * max(0.0, wyckoff_support)
        )

        confirm_loss_mult = (
            float(tpl["execution_confirm_loss_mult"])
            + 0.10 * max(0.0, brooks_hazard)
            + 0.08 * max(0.0, wyckoff_hazard)
            - 0.06 * max(0.0, brooks_support)
            - 0.05 * max(0.0, wyckoff_support)
        )
        confirm_lookahead = int(
            round(
                float(tpl["execution_confirm_lookahead"])
                + 1.1 * max(0.0, brooks_hazard)
                + 0.8 * max(0.0, wyckoff_hazard)
                - 0.7 * max(0.0, brooks_support)
            )
        )
        anti_step = (
            float(tpl["execution_anti_martingale_step"])
            + 0.05 * max(0.0, brooks_support)
            + 0.04 * max(0.0, wyckoff_support)
            - 0.08 * max(0.0, brooks_hazard)
            - 0.05 * max(0.0, wyckoff_hazard)
        )
        anti_floor = (
            float(tpl["execution_anti_martingale_floor"])
            + 0.10 * max(0.0, brooks_hazard)
            + 0.08 * max(0.0, wyckoff_hazard)
            - 0.05 * max(0.0, brooks_support)
        )
        anti_ceiling = (
            float(tpl["execution_anti_martingale_ceiling"])
            + 0.12 * max(0.0, brooks_support)
            + 0.08 * max(0.0, wyckoff_support)
            - 0.16 * max(0.0, brooks_hazard)
            - 0.10 * max(0.0, wyckoff_hazard)
        )

        tpl["signal_confidence_min"] = _clip(conf, 15.0, 72.0)
        tpl["convexity_min"] = _clip(convex, 1.0, 3.5)
        tpl["exposure_scale"] = _clip(exposure, 0.08, max(0.08, float(exposure_cap)))
        tpl["hold_days"] = int(max(1, min(20, hold)))
        tpl["max_daily_trades"] = int(max(1, min(5, trades)))
        tpl["theory_ict_weight"] = _clip(ict_w, 0.6, 1.8)
        tpl["theory_brooks_weight"] = _clip(brooks_w, 0.6, 1.8)
        tpl["theory_lie_weight"] = _clip(lie_w, 0.6, 1.8)
        tpl["theory_wyckoff_weight"] = _clip(wyckoff_w, 0.0, 1.8)
        tpl["theory_vpa_weight"] = _clip(vpa_w, 0.0, 1.8)
        tpl["theory_confidence_boost_max"] = _clip(boost, 2.5, 10.0)
        tpl["theory_penalty_max"] = _clip(penalty, 3.0, 10.0)
        tpl["theory_min_confluence"] = _clip(min_conf, 0.20, 0.60)
        tpl["theory_conflict_fuse"] = _clip(conflict_fuse, 0.50, 0.90)
        tpl["execution_confirm_loss_mult"] = _clip(confirm_loss_mult, 0.45, 0.95)
        tpl["execution_confirm_lookahead"] = int(max(1, min(5, confirm_lookahead)))
        tpl["execution_anti_martingale_step"] = _clip(anti_step, 0.05, 0.35)
        tpl["execution_anti_martingale_floor"] = _clip(anti_floor, 0.40, 1.00)
        tpl["execution_anti_martingale_ceiling"] = _clip(anti_ceiling, 1.05, 1.80)
        if float(tpl["execution_anti_martingale_floor"]) > float(tpl["execution_anti_martingale_ceiling"]):
            tpl["execution_anti_martingale_floor"] = float(tpl["execution_anti_martingale_ceiling"])
        tpl["brooks_support_score"] = _clip(brooks_support, -1.0, 1.0)
        tpl["brooks_hazard_score"] = _clip(brooks_hazard, 0.0, 1.0)
        tpl["wyckoff_support_score"] = _clip(wyckoff_support, -1.0, 1.0)
        tpl["wyckoff_hazard_score"] = _clip(wyckoff_hazard, 0.0, 1.0)
        tpl["vpa_support_score"] = _clip(vpa_support, -1.0, 1.0)
        tpl["vpa_hazard_score"] = _clip(vpa_hazard, 0.0, 1.0)

        if float(tpl["brooks_support_score"]) >= 0.25:
            tpl["rationale"] = f"{tpl['rationale']} | Brooks顺势结构强化"
        if float(tpl["brooks_hazard_score"]) >= 0.25:
            tpl["rationale"] = f"{tpl['rationale']} | Brooks高潮风险抑制"
        if float(tpl["wyckoff_support_score"]) >= 0.20:
            tpl["rationale"] = f"{tpl['rationale']} | Wyckoff吸筹偏多"
        if float(tpl["wyckoff_hazard_score"]) >= 0.20:
            tpl["rationale"] = f"{tpl['rationale']} | Wyckoff派发风控"
        if float(tpl["vpa_support_score"]) >= 0.20:
            tpl["rationale"] = f"{tpl['rationale']} | VPA量价共振"
        if float(tpl["vpa_hazard_score"]) >= 0.20:
            tpl["rationale"] = f"{tpl['rationale']} | VPA量价背离"

        if bool(low_activity_mode):
            tpl["signal_confidence_min"] = _clip(float(tpl["signal_confidence_min"]) - 4.0, 12.0, 72.0)
            tpl["convexity_min"] = _clip(float(tpl["convexity_min"]) - 0.30, 0.9, 3.5)
            tpl["hold_days"] = int(max(1, min(20, int(tpl["hold_days"]) - 2)))
            tpl["max_daily_trades"] = int(max(1, min(5, int(tpl["max_daily_trades"]) + 1)))
            tpl["execution_confirm_loss_mult"] = _clip(float(tpl["execution_confirm_loss_mult"]) + 0.04, 0.45, 0.95)
            tpl["execution_confirm_lookahead"] = int(max(1, min(5, int(tpl["execution_confirm_lookahead"]) - 1)))
            tpl["rationale"] = f"{tpl['rationale']} | 低活跃窗口松绑"

        if bool(low_activity_mode) and bool(crypto_mode):
            tpl["signal_confidence_min"] = _clip(float(tpl["signal_confidence_min"]) - 5.0, 8.0, 72.0)
            tpl["convexity_min"] = _clip(float(tpl["convexity_min"]) - 0.35, 0.7, 3.5)
            tpl["hold_days"] = int(max(1, min(20, int(tpl["hold_days"]) - 1)))
            tpl["max_daily_trades"] = int(max(1, min(5, int(tpl["max_daily_trades"]) + 1)))
            tpl["theory_min_confluence"] = _clip(float(tpl["theory_min_confluence"]) - 0.05, 0.15, 0.60)
            tpl["execution_confirm_loss_mult"] = _clip(float(tpl["execution_confirm_loss_mult"]) + 0.06, 0.45, 0.95)
            tpl["execution_confirm_lookahead"] = int(max(1, min(5, int(tpl["execution_confirm_lookahead"]) - 1)))
            tpl["rationale"] = f"{tpl['rationale']} | crypto低活跃战术松绑"

        if bool(low_activity_mode) and bool(crypto_mode) and i == 0:
            tpl["name"] = "crypto_tactical"
            tpl["signal_confidence_min"] = 14.0
            tpl["convexity_min"] = 1.2
            tpl["hold_days"] = 1
            tpl["max_daily_trades"] = 5
            tpl["exposure_scale"] = _clip(min(float(tpl.get("exposure_scale", 0.20)), 0.20), 0.08, max(0.08, float(exposure_cap)))
            tpl["theory_ict_weight"] = 1.2
            tpl["theory_brooks_weight"] = 1.3
            tpl["theory_lie_weight"] = 0.9
            tpl["theory_wyckoff_weight"] = 1.2
            tpl["theory_vpa_weight"] = 1.0
            tpl["theory_confidence_boost_max"] = 4.6
            tpl["theory_penalty_max"] = 6.8
            tpl["theory_min_confluence"] = 0.30
            tpl["theory_conflict_fuse"] = 0.74
            tpl["execution_confirm_lookahead"] = 1
            tpl["execution_confirm_loss_mult"] = 0.70
            tpl["execution_anti_martingale_step"] = 0.14
            tpl["rationale"] = f"{tpl['rationale']} | crypto战术快切(1D)"

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
    max_drawdown_target: float = 0.05,
    review_max_drawdown_target: float = 0.07,
    drawdown_soft_band: float = 0.03,
) -> float:
    dd_target = max(0.0, float(max_drawdown_target))
    dd_review_target = max(dd_target, float(review_max_drawdown_target))
    dd_soft = max(0.0, float(drawdown_soft_band))

    def _dd_target_penalty(mdd: float, target: float) -> float:
        excess = max(0.0, float(mdd) - float(target))
        return float(2.2 * excess + 5.0 * max(0.0, excess - dd_soft))

    trades_valid = int(getattr(valid_bt, "trades", 0))
    pwr_valid = float(getattr(valid_bt, "positive_window_ratio", 0.0))
    if trades_valid <= 0:
        pwr_valid = 0.0
    elif trades_valid < 5:
        pwr_valid *= float(trades_valid) / 5.0

    trade_penalty = 0.0 if trades_valid >= 5 else 0.12 * float(5 - trades_valid)
    if trades_valid == 0:
        trade_penalty += 0.25
    dd_penalty_valid = _dd_target_penalty(float(getattr(valid_bt, "max_drawdown", 0.0)), dd_target)
    base = float(
        0.25 * float(getattr(train_bt, "annual_return", 0.0))
        + 0.55 * float(getattr(valid_bt, "annual_return", 0.0))
        + 0.65 * pwr_valid
        + 0.20 * float(align_valid)
        + 0.08 * float(align_train)
        - 2.40 * float(getattr(valid_bt, "max_drawdown", 0.0))
        - dd_penalty_valid
        - 0.35 * float(getattr(valid_bt, "violations", 0))
        - trade_penalty
    )
    if review_bt is None:
        return base
    review_trades = int(getattr(review_bt, "trades", 0))
    pwr_review = float(getattr(review_bt, "positive_window_ratio", 0.0))
    if review_trades <= 0:
        pwr_review = 0.0
    elif review_trades < 2:
        pwr_review *= float(review_trades) / 2.0

    review_penalty = 0.0 if review_trades >= 2 else 0.08 * float(2 - review_trades)
    if review_trades == 0:
        review_penalty += 0.10
    dd_penalty_review = _dd_target_penalty(float(getattr(review_bt, "max_drawdown", 0.0)), dd_review_target)
    return float(
        base
        + 0.18 * float(getattr(review_bt, "annual_return", 0.0))
        + 0.22 * pwr_review
        + 0.10 * float(align_review)
        + 0.20 * float(robustness_score)
        - 1.20 * float(getattr(review_bt, "max_drawdown", 0.0))
        - 0.60 * dd_penalty_review
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
    lines.append(f"- 训练回撤目标: `{summary.max_drawdown_target:.2%}`")
    lines.append(f"- 复盘回撤目标: `{summary.review_max_drawdown_target:.2%}`")
    lines.append(f"- 回撤软带宽: `{summary.drawdown_soft_band:.2%}`")
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
    lines.append(f"- 暴露上限(应用): `{float(summary.exposure_cap_applied):.4f}`")
    lines.append(f"- proxy_lookback(应用): `{int(summary.proxy_lookback_applied)}`")
    if summary.exposure_cap_components:
        lines.append(f"- 暴露上限(基线): `{float(summary.exposure_cap_components.get('base_cap', 0.0)):.4f}`")
        lines.append(f"- 暴露上限(ablation): `{float(summary.exposure_cap_components.get('ablation_cap', 0.0)):.4f}`")
        lines.append(f"- 暴露乘数(hazard): `{float(summary.exposure_cap_components.get('hazard_multiplier', 1.0)):.4f}`")
        lines.append(f"- ablation_found: `{bool(summary.exposure_cap_components.get('ablation_found', False))}`")
        if str(summary.exposure_cap_components.get("ablation_case_name", "")):
            lines.append(f"- ablation_case: `{summary.exposure_cap_components.get('ablation_case_name', '')}`")
    lines.append("")
    lines.append("## 市场学习信号")
    for k, v in summary.market_insights.items():
        lines.append(f"- `{k}`: `{float(v):.4f}`")
    lines.append("")
    lines.append("## 报告学习信号")
    for k, v in summary.report_insights.items():
        lines.append(f"- `{k}`: `{float(v):.4f}`")
    lines.append("")
    lines.append("## 术语原子注册")
    term_registry = summary.term_registry if isinstance(summary.term_registry, dict) else {}
    lines.append(f"- registry_exists: `{bool(term_registry.get('exists', False))}`")
    lines.append(f"- registry_version: `{term_registry.get('version', '')}`")
    lines.append(f"- registry_checksum_sha256: `{term_registry.get('checksum_sha256', '')}`")
    lines.append(f"- atoms_total: `{int(term_registry.get('atoms_total', 0))}`")
    lines.append(f"- l2_atoms: `{int(term_registry.get('l2_atoms', 0))}`")
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
    max_drawdown_target: float = 0.05,
    review_max_drawdown_target: float = 0.07,
    drawdown_soft_band: float = 0.03,
) -> StrategyLabSummary:
    t0 = datetime.now()
    run_dir = output_root / "research" / f"strategy_lab_{t0:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "artifacts" / "research_cache"
    term_registry = _term_registry_snapshot()
    valid_dd_target = _clip(float(max_drawdown_target), 0.01, 0.30)
    review_dd_target = _clip(float(review_max_drawdown_target), valid_dd_target, 0.35)
    dd_soft_band = _clip(float(drawdown_soft_band), 0.0, 0.12)

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
            max_drawdown_target=float(valid_dd_target),
            review_max_drawdown_target=float(review_dd_target),
            drawdown_soft_band=float(dd_soft_band),
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
            term_registry=term_registry,
            exposure_cap_applied=0.0,
            exposure_cap_components={},
            proxy_lookback_applied=180,
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
    valid_days = int(
        len({d for d in pd.to_datetime(bars["ts"]).dt.date.tolist() if d >= valid_start})
    )
    low_activity_mode = bool(valid_days < 45)
    asset_classes = {str(x).strip().lower() for x in bars["asset_class"].dropna().unique().tolist()}
    crypto_mode = bool(asset_classes) and asset_classes.issubset({"crypto", "perp", "perpetual", "spot"})
    total_days = int(len(pd.to_datetime(bars["ts"]).dt.date.unique()))
    proxy_lookback = int(_clip(float(total_days) * 0.70, 60.0, 180.0))
    if low_activity_mode:
        proxy_lookback = int(min(proxy_lookback, max(50, int(valid_days * 1.6))))
    if low_activity_mode and crypto_mode:
        proxy_lookback = int(max(proxy_lookback, 84))

    market = _market_insights(bars)
    report = _report_insights(bundle.news_daily, bundle.report_daily)
    exposure_cap, exposure_cap_components = _resolve_exposure_cap(
        output_root=output_root,
        start=start,
        end=end,
        max_drawdown_target=float(valid_dd_target),
        market=market,
    )
    candidates_cfg = _generate_candidates(
        market=market,
        report=report,
        candidate_count=candidate_count,
        exposure_cap=exposure_cap,
        low_activity_mode=low_activity_mode,
        crypto_mode=crypto_mode,
    )

    out_candidates: list[StrategyCandidateResult] = []
    for spec in candidates_cfg:
        cfg = BacktestConfig(
            signal_confidence_min=float(spec["signal_confidence_min"]),
            convexity_min=float(spec["convexity_min"]),
            max_daily_trades=int(spec["max_daily_trades"]),
            hold_days=int(spec["hold_days"]),
            exposure_scale=float(spec.get("exposure_scale", 1.0)),
            regime_recalc_interval=5,
            signal_eval_interval=1,
            proxy_lookback=int(proxy_lookback),
            theory_enabled=True,
            theory_ict_weight=float(spec.get("theory_ict_weight", 1.0)),
            theory_brooks_weight=float(spec.get("theory_brooks_weight", 1.0)),
            theory_lie_weight=float(spec.get("theory_lie_weight", 1.2)),
            theory_wyckoff_weight=float(spec.get("theory_wyckoff_weight", 0.0)),
            theory_vpa_weight=float(spec.get("theory_vpa_weight", 0.0)),
            theory_confidence_boost_max=float(spec.get("theory_confidence_boost_max", 5.0)),
            theory_penalty_max=float(spec.get("theory_penalty_max", 6.0)),
            theory_min_confluence=float(spec.get("theory_min_confluence", 0.38)),
            theory_conflict_fuse=float(spec.get("theory_conflict_fuse", 0.72)),
            execution_confirm_enabled=bool(spec.get("execution_confirm_enabled", False)),
            execution_confirm_lookahead=int(spec.get("execution_confirm_lookahead", 2)),
            execution_confirm_loss_mult=float(spec.get("execution_confirm_loss_mult", 0.65)),
            execution_confirm_win_mult=float(spec.get("execution_confirm_win_mult", 1.0)),
            execution_anti_martingale_enabled=bool(spec.get("execution_anti_martingale_enabled", False)),
            execution_anti_martingale_step=float(spec.get("execution_anti_martingale_step", 0.2)),
            execution_anti_martingale_floor=float(spec.get("execution_anti_martingale_floor", 0.6)),
            execution_anti_martingale_ceiling=float(spec.get("execution_anti_martingale_ceiling", 1.4)),
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
            max_drawdown_target=float(valid_dd_target),
            review_max_drawdown_target=float(review_dd_target),
            drawdown_soft_band=float(dd_soft_band),
        )
        valid_trade_gate = 5
        review_trade_gate = 2
        valid_pwr_gate = 0.70
        review_pwr_gate = 0.60
        if bool(crypto_mode) and not bool(low_activity_mode):
            valid_trade_gate = 2
            review_trade_gate = 0
            valid_pwr_gate = 0.62
            review_pwr_gate = 0.50
        if bool(low_activity_mode):
            valid_trade_gate = 3
            review_trade_gate = 1
            valid_pwr_gate = 0.65
            review_pwr_gate = 0.55
        if bool(low_activity_mode) and bool(crypto_mode):
            valid_trade_gate = 1
            review_trade_gate = 0
            valid_pwr_gate = 0.60
            review_pwr_gate = 0.50

        review_ok = True
        if review_bt is not None:
            review_ok = (
                int(review_bt.trades) >= int(review_trade_gate)
                and float(review_bt.annual_return) >= 0.0
                and float(review_bt.positive_window_ratio) >= float(review_pwr_gate)
                and float(review_bt.max_drawdown) <= float(review_dd_target)
                and int(review_bt.violations) == 0
            )

        accepted = (
            int(valid_bt.trades) >= int(valid_trade_gate)
            and float(valid_bt.annual_return) > 0.0
            and float(valid_bt.positive_window_ratio) >= float(valid_pwr_gate)
            and float(valid_bt.max_drawdown) <= float(valid_dd_target)
            and int(valid_bt.violations) == 0
            and bool(review_ok)
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
                    "exposure_scale": float(spec.get("exposure_scale", 1.0)),
                    "theory_ict_weight": float(spec.get("theory_ict_weight", 1.0)),
                    "theory_brooks_weight": float(spec.get("theory_brooks_weight", 1.0)),
                    "theory_lie_weight": float(spec.get("theory_lie_weight", 1.2)),
                    "theory_wyckoff_weight": float(spec.get("theory_wyckoff_weight", 0.0)),
                    "theory_vpa_weight": float(spec.get("theory_vpa_weight", 0.0)),
                    "theory_confidence_boost_max": float(spec.get("theory_confidence_boost_max", 5.0)),
                    "theory_penalty_max": float(spec.get("theory_penalty_max", 6.0)),
                    "theory_min_confluence": float(spec.get("theory_min_confluence", 0.38)),
                    "theory_conflict_fuse": float(spec.get("theory_conflict_fuse", 0.72)),
                    "brooks_support_score": float(spec.get("brooks_support_score", 0.0)),
                    "brooks_hazard_score": float(spec.get("brooks_hazard_score", 0.0)),
                    "wyckoff_support_score": float(spec.get("wyckoff_support_score", 0.0)),
                    "wyckoff_hazard_score": float(spec.get("wyckoff_hazard_score", 0.0)),
                    "vpa_support_score": float(spec.get("vpa_support_score", 0.0)),
                    "vpa_hazard_score": float(spec.get("vpa_hazard_score", 0.0)),
                    "execution_confirm_enabled": bool(spec.get("execution_confirm_enabled", False)),
                    "execution_confirm_lookahead": int(spec.get("execution_confirm_lookahead", 2)),
                    "execution_confirm_loss_mult": float(spec.get("execution_confirm_loss_mult", 0.65)),
                    "execution_confirm_win_mult": float(spec.get("execution_confirm_win_mult", 1.0)),
                    "execution_anti_martingale_enabled": bool(spec.get("execution_anti_martingale_enabled", False)),
                    "execution_anti_martingale_step": float(spec.get("execution_anti_martingale_step", 0.2)),
                    "execution_anti_martingale_floor": float(spec.get("execution_anti_martingale_floor", 0.6)),
                    "execution_anti_martingale_ceiling": float(spec.get("execution_anti_martingale_ceiling", 1.4)),
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
        max_drawdown_target=float(valid_dd_target),
        review_max_drawdown_target=float(review_dd_target),
        drawdown_soft_band=float(dd_soft_band),
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
        term_registry=term_registry,
        exposure_cap_applied=float(exposure_cap),
        exposure_cap_components=exposure_cap_components,
        proxy_lookback_applied=int(proxy_lookback),
    )

    write_json(run_dir / "summary.json", summary.to_dict())
    write_markdown(run_dir / "report.md", _render_report(summary))
    write_markdown(run_dir / "best_strategy.yaml", yaml.safe_dump(best, allow_unicode=True, sort_keys=False))
    if out_candidates:
        pd.DataFrame([c.to_dict() for c in out_candidates]).to_csv(run_dir / "candidates.csv", index=False)
    return summary
