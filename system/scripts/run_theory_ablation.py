#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from lie_engine.backtest.engine import BacktestConfig, run_event_backtest
from lie_engine.research.real_data import load_real_data_bundle


@dataclass(slots=True)
class AblationCase:
    name: str
    theory_enabled: bool
    exposure_scale: float
    theory_ict_weight: float
    theory_brooks_weight: float
    theory_lie_weight: float
    theory_confidence_boost_max: float
    theory_penalty_max: float
    theory_min_confluence: float
    theory_conflict_fuse: float
    annual_return: float
    max_drawdown: float
    drawdown_excess: float
    target_breached: bool
    positive_window_ratio: float
    profit_factor: float
    win_rate: float
    trades: int
    violations: int
    sharpe_like: float
    objective: float


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _parse_symbols(raw: str) -> list[str]:
    out = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not out:
        return ["300750", "002050", "603026", "000830", "513130", "LC2603", "SC2603", "RB2605"]
    return out


def _parse_weight_grid(raw: str) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for block in [x.strip() for x in str(raw).split(";") if x.strip()]:
        parts = [p.strip() for p in block.split(":")]
        if len(parts) != 3:
            continue
        try:
            out.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not out:
        return [(1.0, 1.0, 1.2), (1.2, 1.0, 1.3), (1.2, 1.1, 1.3), (1.4, 0.9, 1.2)]
    return out


def _parse_gate_grid(raw: str) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for block in [x.strip() for x in str(raw).split(";") if x.strip()]:
        parts = [p.strip() for p in block.split(":")]
        if len(parts) != 4:
            continue
        try:
            out.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if not out:
        return [
            (5.0, 6.0, 0.38, 0.72),
            (6.6, 4.8, 0.30, 0.82),
            (4.2, 7.4, 0.44, 0.64),
            (5.8, 5.4, 0.34, 0.76),
        ]
    return out


def _parse_exposure_grid(raw: str) -> list[float]:
    out: list[float] = []
    for block in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            out.append(float(block))
        except ValueError:
            continue
    return out


def _safe_div(x: float, y: float) -> float:
    if abs(y) <= 1e-12:
        return 0.0
    return float(x / y)


def _sharpe_like(equity_curve: list[dict[str, object]]) -> float:
    if not equity_curve:
        return 0.0
    eq = np.array([float(x.get("equity", 0.0)) for x in equity_curve], dtype=float)
    if eq.size < 3:
        return 0.0
    ret = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if ret.size < 2:
        return 0.0
    mu = float(np.mean(ret))
    sd = float(np.std(ret, ddof=0))
    return float(np.sqrt(252.0) * _safe_div(mu, sd))


def _objective(
    annual_return: float,
    max_drawdown: float,
    positive_window_ratio: float,
    violations: int,
    *,
    max_drawdown_target: float,
    drawdown_soft_band: float,
) -> float:
    target = max(0.0, float(max_drawdown_target))
    soft_band = max(0.0, float(drawdown_soft_band))
    excess = max(0.0, float(max_drawdown) - target)
    # Penalize drawdown target breach more aggressively than baseline noise.
    target_penalty = 4.0 * excess + 8.0 * max(0.0, excess - soft_band)
    breach_penalty = 0.35 if excess > 0.0 else 0.0
    return float(
        annual_return
        - 0.90 * max_drawdown
        + 0.20 * positive_window_ratio
        - 0.15 * float(violations)
        - target_penalty
        - breach_penalty
    )


def _run_case(
    *,
    name: str,
    bars,
    start: date,
    end: date,
    signal_confidence_min: float,
    convexity_min: float,
    max_daily_trades: int,
    hold_days: int,
    proxy_lookback: int,
    theory_enabled: bool,
    exposure_scale: float,
    theory_ict_weight: float,
    theory_brooks_weight: float,
    theory_lie_weight: float,
    theory_confidence_boost_max: float,
    theory_penalty_max: float,
    theory_min_confluence: float,
    theory_conflict_fuse: float,
    max_drawdown_target: float,
    drawdown_soft_band: float,
) -> AblationCase:
    bt = run_event_backtest(
        bars=bars,
        start=start,
        end=end,
        cfg=BacktestConfig(
            signal_confidence_min=float(signal_confidence_min),
            convexity_min=float(convexity_min),
            max_daily_trades=int(max_daily_trades),
            hold_days=int(hold_days),
            exposure_scale=float(exposure_scale),
            regime_recalc_interval=5,
            signal_eval_interval=1,
            proxy_lookback=int(proxy_lookback),
            theory_enabled=bool(theory_enabled),
            theory_ict_weight=float(theory_ict_weight),
            theory_brooks_weight=float(theory_brooks_weight),
            theory_lie_weight=float(theory_lie_weight),
            theory_confidence_boost_max=float(theory_confidence_boost_max),
            theory_penalty_max=float(theory_penalty_max),
            theory_min_confluence=float(theory_min_confluence),
            theory_conflict_fuse=float(theory_conflict_fuse),
        ),
        trend_thr=0.6,
        mean_thr=0.4,
        atr_extreme=2.0,
    )
    sharpe_like = _sharpe_like(bt.equity_curve)
    objective = _objective(
        annual_return=float(bt.annual_return),
        max_drawdown=float(bt.max_drawdown),
        positive_window_ratio=float(bt.positive_window_ratio),
        violations=int(bt.violations),
        max_drawdown_target=float(max_drawdown_target),
        drawdown_soft_band=float(drawdown_soft_band),
    )
    dd_excess = max(0.0, float(bt.max_drawdown) - float(max_drawdown_target))
    return AblationCase(
        name=name,
        theory_enabled=bool(theory_enabled),
        exposure_scale=float(exposure_scale),
        theory_ict_weight=float(theory_ict_weight),
        theory_brooks_weight=float(theory_brooks_weight),
        theory_lie_weight=float(theory_lie_weight),
        theory_confidence_boost_max=float(theory_confidence_boost_max),
        theory_penalty_max=float(theory_penalty_max),
        theory_min_confluence=float(theory_min_confluence),
        theory_conflict_fuse=float(theory_conflict_fuse),
        annual_return=float(bt.annual_return),
        max_drawdown=float(bt.max_drawdown),
        drawdown_excess=float(dd_excess),
        target_breached=bool(dd_excess > 0.0),
        positive_window_ratio=float(bt.positive_window_ratio),
        profit_factor=float(bt.profit_factor),
        win_rate=float(bt.win_rate),
        trades=int(bt.trades),
        violations=int(bt.violations),
        sharpe_like=float(sharpe_like),
        objective=float(objective),
    )


def _load_local_fallback_bars(
    *,
    output_root: Path,
    start: date,
    end: date,
    symbols: list[str],
) -> tuple[pd.DataFrame, str]:
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    symbol_set = set(symbols)

    def _finalize(frame: pd.DataFrame, path_label: str) -> tuple[pd.DataFrame, str]:
        frame = frame.copy()
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        frame = frame.dropna(subset=["ts"])
        frame = frame[(frame["ts"].dt.date >= start) & (frame["ts"].dt.date <= end)]
        if frame.empty:
            return pd.DataFrame(), ""
        if symbols:
            filtered = frame[frame["symbol"].astype(str).isin(symbol_set)].copy()
        else:
            filtered = frame.copy()
        if "asset_class" not in frame.columns:
            frame["asset_class"] = "equity"
        if "source" not in frame.columns:
            frame["source"] = f"fallback:{Path(path_label).name}"
        if not filtered.empty:
            if "asset_class" not in filtered.columns:
                filtered["asset_class"] = "equity"
            if "source" not in filtered.columns:
                filtered["source"] = f"fallback:{Path(path_label).name}"
            return filtered.sort_values(["ts", "symbol"]).reset_index(drop=True), path_label
        return frame.sort_values(["ts", "symbol"]).reset_index(drop=True), f"{path_label}#all_symbols"

    candidates = sorted(
        list((output_root / "artifacts" / "research_cache").glob("*_bars.parquet"))
        + list((output_root / "research").glob("*/bars_used.parquet"))
        + list((output_root / "artifacts" / "raw").glob("*_bars_raw.csv"))
        + list((output_root / "artifacts" / "normalized").glob("*_bars_normalized.csv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    scored: list[tuple[tuple[int, int, int, int, float], pd.DataFrame, str]] = []
    for path in candidates:
        try:
            bars = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        if bars.empty or not required.issubset(set(bars.columns)):
            continue
        out, label = _finalize(bars, str(path))
        if out.empty:
            continue
        ts_min = pd.to_datetime(out["ts"], errors="coerce").min()
        ts_max = pd.to_datetime(out["ts"], errors="coerce").max()
        span_days = int(max(0, (ts_max - ts_min).days)) if pd.notna(ts_min) and pd.notna(ts_max) else 0
        matched_symbols = 0 if label.endswith("#all_symbols") else 1
        score = (
            matched_symbols,
            int(len(out)),
            int(out["symbol"].astype(str).nunique()),
            span_days,
            float(path.stat().st_mtime),
        )
        scored.append((score, out, label))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        _, best_frame, best_label = scored[0]
        return best_frame, best_label
    return pd.DataFrame(), ""


def _render_markdown(
    *,
    started_at: str,
    ended_at: str,
    start: date,
    end: date,
    symbols: list[str],
    bars_rows: int,
    cases: list[AblationCase],
    data_source: str,
    max_drawdown_target: float,
    drawdown_soft_band: float,
    auto_exposure_search: dict[str, object] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Theory Ablation Report | {ended_at[:19]}")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- started_at: `{started_at}`")
    lines.append(f"- ended_at: `{ended_at}`")
    lines.append(f"- window: `{start.isoformat()} ~ {end.isoformat()}`")
    lines.append(f"- symbols: `{','.join(symbols)}`")
    lines.append(f"- bars_rows: `{bars_rows}`")
    lines.append(f"- data_source: `{data_source}`")
    lines.append(f"- max_drawdown_target: `{max_drawdown_target:.4f}`")
    lines.append(f"- drawdown_soft_band: `{drawdown_soft_band:.4f}`")
    auto_cfg = auto_exposure_search if isinstance(auto_exposure_search, dict) else {}
    if bool(auto_cfg.get("enabled", False)):
        lines.append("- auto_exposure_search: `on`")
        lines.append(
            "- "
            + f"auto_exposure_range: `{float(auto_cfg.get('low', 0.0)):.4f} ~ {float(auto_cfg.get('high', 0.0)):.4f}`"
        )
        lines.append(f"- auto_exposure_iters: `{int(auto_cfg.get('iterations', 0))}`")
        lines.append(f"- auto_exposure_theory: `{int(bool(auto_cfg.get('theory_enabled', False)))}`")
        best_exp = auto_cfg.get("best_exposure", None)
        if best_exp is None:
            lines.append("- auto_exposure_best: `none`")
        else:
            lines.append(f"- auto_exposure_best: `{float(best_exp):.4f}`")
    lines.append("")
    lines.append("## Cases")
    lines.append(
        "| name | theory | exp | ict | brooks | lie | boost | penalty | min_conf | cfuse | ann_ret | mdd | mdd_excess | breach | pwr | pf | wr | trades | viol | sharpe_like | obj |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        lines.append(
            "| "
            + f"{c.name} | {int(c.theory_enabled)} | {c.exposure_scale:.2f} | {c.theory_ict_weight:.2f} | {c.theory_brooks_weight:.2f} | {c.theory_lie_weight:.2f} | "
            + f"{c.theory_confidence_boost_max:.2f} | {c.theory_penalty_max:.2f} | {c.theory_min_confluence:.2f} | {c.theory_conflict_fuse:.2f} | "
            + f"{c.annual_return:.4f} | {c.max_drawdown:.4f} | {c.drawdown_excess:.4f} | {int(c.target_breached)} | {c.positive_window_ratio:.4f} | {c.profit_factor:.4f} | "
            + f"{c.win_rate:.4f} | {c.trades} | {c.violations} | {c.sharpe_like:.4f} | {c.objective:.4f} |"
        )
    lines.append("")

    base = next((x for x in cases if x.name == "baseline_off"), None)
    if base is not None:
        lines.append("## Delta Vs Baseline")
        for c in cases:
            if c.name == "baseline_off":
                continue
            lines.append(
                "- "
                + f"`{c.name}`: Δann_ret={c.annual_return - base.annual_return:+.4f}, "
                + f"Δmdd={c.max_drawdown - base.max_drawdown:+.4f}, "
                + f"Δmdd_excess={c.drawdown_excess - base.drawdown_excess:+.4f}, "
                + f"Δpf={c.profit_factor - base.profit_factor:+.4f}, "
                + f"Δwr={c.win_rate - base.win_rate:+.4f}, "
                + f"Δtrades={c.trades - base.trades:+d}, "
                + f"Δobj={c.objective - base.objective:+.4f}"
            )
        lines.append("")

    best = max(cases, key=lambda x: x.objective) if cases else None
    feasible = [x for x in cases if not bool(x.target_breached)]
    best_feasible = max(feasible, key=lambda x: x.objective) if feasible else None
    lines.append("## Conclusion")
    if best is None:
        lines.append("- no case executed")
    else:
        lines.append(
            "- "
            + f"best_by_objective: `{best.name}` "
            + f"(obj={best.objective:.4f}, ann_ret={best.annual_return:.4f}, mdd={best.max_drawdown:.4f}, trades={best.trades})"
        )
    lines.append(f"- feasible_cases_under_target: `{len(feasible)}`")
    if best_feasible is not None:
        lines.append(
            "- "
            + f"best_feasible_case: `{best_feasible.name}` "
            + f"(obj={best_feasible.objective:.4f}, ann_ret={best_feasible.annual_return:.4f}, "
            + f"mdd={best_feasible.max_drawdown:.4f}, exp={best_feasible.exposure_scale:.2f})"
        )
    else:
        lines.append("- best_feasible_case: `none`")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run theory on/off with weight-grid and theory-gate ablation using a consistent backtest setup.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        default="300750,002050,603026,000830,513130,LC2603,SC2603,RB2605",
        help="comma-separated symbols",
    )
    parser.add_argument("--max-symbols", type=int, default=8)
    parser.add_argument("--report-symbol-cap", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--signal-confidence-min", type=float, default=5.0)
    parser.add_argument("--convexity-min", type=float, default=0.3)
    parser.add_argument("--max-daily-trades", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--exposure-scale", type=float, default=1.0)
    parser.add_argument("--proxy-lookback", type=int, default=180)
    parser.add_argument("--max-drawdown-target", type=float, default=0.05)
    parser.add_argument("--drawdown-soft-band", type=float, default=0.03)
    parser.add_argument("--exposure-grid", default="", help="comma-separated exposure scan values, e.g. 1.0,0.6,0.3,0.2")
    parser.add_argument("--auto-exposure-search", action="store_true")
    parser.add_argument("--auto-exposure-low", type=float, default=0.08)
    parser.add_argument("--auto-exposure-high", type=float, default=1.0)
    parser.add_argument("--auto-exposure-iters", type=int, default=8)
    parser.add_argument("--auto-exposure-theory", action="store_true")
    parser.add_argument(
        "--weight-grid",
        default="1.0:1.0:1.2;1.2:1.0:1.3;1.2:1.1:1.3;1.4:0.9:1.2",
        help="semicolon-separated ict:brooks:lie triples",
    )
    parser.add_argument("--theory-confidence-boost-max", type=float, default=5.0)
    parser.add_argument("--theory-penalty-max", type=float, default=6.0)
    parser.add_argument("--theory-min-confluence", type=float, default=0.38)
    parser.add_argument("--theory-conflict-fuse", type=float, default=0.72)
    parser.add_argument(
        "--gate-grid",
        default="5.0:6.0:0.38:0.72;6.6:4.8:0.30:0.82;4.2:7.4:0.44:0.64;5.8:5.4:0.34:0.76",
        help="semicolon-separated boost:penalty:min_conflict:conflict_fuse quadruples",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    system_root = Path(__file__).resolve().parents[1]
    output_root = system_root / "output"
    cache_dir = output_root / "artifacts" / "research_cache"
    run_dir = output_root / "research" / f"theory_ablation_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    symbols = _parse_symbols(args.symbols)
    weight_grid = _parse_weight_grid(args.weight_grid)
    gate_grid = _parse_gate_grid(args.gate_grid)
    exposure_grid = _parse_exposure_grid(args.exposure_grid)
    auto_exposure_cfg: dict[str, object] = {"enabled": bool(args.auto_exposure_search)}

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    t0 = datetime.now()
    bundle = load_real_data_bundle(
        core_symbols=symbols,
        start=start,
        end=end,
        max_symbols=int(args.max_symbols),
        report_symbol_cap=int(args.report_symbol_cap),
        workers=int(args.workers),
        cache_dir=cache_dir,
        cache_ttl_hours=8.0,
        strict_cutoff=end,
        review_days=0,
        include_post_review=False,
    )
    bars = bundle.bars.copy()
    data_source = "remote_bundle"
    fallback_path = ""
    if bars.empty:
        bars, fallback_path = _load_local_fallback_bars(
            output_root=output_root,
            start=start,
            end=end,
            symbols=symbols,
        )
        if not bars.empty:
            data_source = f"local_fallback:{Path(fallback_path).name}"
        else:
            data_source = "remote_bundle_empty"

    if bars.empty:
        t1 = datetime.now()
        payload = {
            "started_at": t0.isoformat(),
            "ended_at": t1.isoformat(),
            "elapsed_seconds": float((t1 - t0).total_seconds()),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": symbols,
            "bars_rows": 0,
            "data_source": data_source,
            "fallback_path": fallback_path,
            "bundle_fetch_stats": bundle.fetch_stats,
            "cases": [],
            "risk_feasible": False,
            "feasible_count": 0,
            "best_feasible_case": {},
            "auto_exposure_search": auto_exposure_cfg,
            "error": "no_bars_available_after_remote_and_local_fallback",
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = _render_markdown(
            started_at=str(payload["started_at"]),
            ended_at=str(payload["ended_at"]),
            start=start,
            end=end,
            symbols=symbols,
            bars_rows=0,
            cases=[],
            data_source=data_source,
            max_drawdown_target=float(args.max_drawdown_target),
            drawdown_soft_band=float(args.drawdown_soft_band),
            auto_exposure_search=auto_exposure_cfg,
        )
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "bars_rows": 0, "error": payload["error"]}, ensure_ascii=False, indent=2))
        return 2

    cases: list[AblationCase] = []
    cases.append(
        _run_case(
            name="baseline_off",
            bars=bars,
            start=start,
            end=end,
            signal_confidence_min=float(args.signal_confidence_min),
            convexity_min=float(args.convexity_min),
            max_daily_trades=int(args.max_daily_trades),
            hold_days=int(args.hold_days),
            exposure_scale=float(args.exposure_scale),
            proxy_lookback=int(args.proxy_lookback),
            theory_enabled=False,
            theory_ict_weight=1.0,
            theory_brooks_weight=1.0,
            theory_lie_weight=1.2,
            theory_confidence_boost_max=float(args.theory_confidence_boost_max),
            theory_penalty_max=float(args.theory_penalty_max),
            theory_min_confluence=float(args.theory_min_confluence),
            theory_conflict_fuse=float(args.theory_conflict_fuse),
            max_drawdown_target=float(args.max_drawdown_target),
            drawdown_soft_band=float(args.drawdown_soft_band),
        )
    )
    for idx, (w_ict, w_brooks, w_lie) in enumerate(weight_grid, start=1):
        cases.append(
            _run_case(
                name=f"weight_on_{idx:02d}",
                bars=bars,
                start=start,
                end=end,
                signal_confidence_min=float(args.signal_confidence_min),
                convexity_min=float(args.convexity_min),
                max_daily_trades=int(args.max_daily_trades),
                hold_days=int(args.hold_days),
                exposure_scale=float(args.exposure_scale),
                proxy_lookback=int(args.proxy_lookback),
                theory_enabled=True,
                theory_ict_weight=float(w_ict),
                theory_brooks_weight=float(w_brooks),
                theory_lie_weight=float(w_lie),
                theory_confidence_boost_max=float(args.theory_confidence_boost_max),
                theory_penalty_max=float(args.theory_penalty_max),
                theory_min_confluence=float(args.theory_min_confluence),
                theory_conflict_fuse=float(args.theory_conflict_fuse),
                max_drawdown_target=float(args.max_drawdown_target),
                drawdown_soft_band=float(args.drawdown_soft_band),
            )
        )
    base_weights = weight_grid[0] if weight_grid else (1.0, 1.0, 1.2)
    for idx, (boost, penalty, min_conf, cfuse) in enumerate(gate_grid, start=1):
        cases.append(
            _run_case(
                name=f"gate_on_{idx:02d}",
                bars=bars,
                start=start,
                end=end,
                signal_confidence_min=float(args.signal_confidence_min),
                convexity_min=float(args.convexity_min),
                max_daily_trades=int(args.max_daily_trades),
                hold_days=int(args.hold_days),
                exposure_scale=float(args.exposure_scale),
                proxy_lookback=int(args.proxy_lookback),
                theory_enabled=True,
                theory_ict_weight=float(base_weights[0]),
                theory_brooks_weight=float(base_weights[1]),
                theory_lie_weight=float(base_weights[2]),
                theory_confidence_boost_max=float(boost),
                theory_penalty_max=float(penalty),
                theory_min_confluence=float(min_conf),
                theory_conflict_fuse=float(cfuse),
                max_drawdown_target=float(args.max_drawdown_target),
                drawdown_soft_band=float(args.drawdown_soft_band),
            )
        )
    for idx, exp in enumerate(exposure_grid, start=1):
        cases.append(
            _run_case(
                name=f"exp_scan_{idx:02d}",
                bars=bars,
                start=start,
                end=end,
                signal_confidence_min=float(args.signal_confidence_min),
                convexity_min=float(args.convexity_min),
                max_daily_trades=int(args.max_daily_trades),
                hold_days=int(args.hold_days),
                exposure_scale=float(exp),
                proxy_lookback=int(args.proxy_lookback),
                theory_enabled=False,
                theory_ict_weight=1.0,
                theory_brooks_weight=1.0,
                theory_lie_weight=1.2,
                theory_confidence_boost_max=float(args.theory_confidence_boost_max),
                theory_penalty_max=float(args.theory_penalty_max),
                theory_min_confluence=float(args.theory_min_confluence),
                theory_conflict_fuse=float(args.theory_conflict_fuse),
                max_drawdown_target=float(args.max_drawdown_target),
                drawdown_soft_band=float(args.drawdown_soft_band),
            )
        )
    if bool(args.auto_exposure_search):
        auto_low = float(min(args.auto_exposure_low, args.auto_exposure_high))
        auto_high = float(max(args.auto_exposure_low, args.auto_exposure_high))
        auto_low = max(0.05, auto_low)
        auto_high = min(1.50, max(auto_low, auto_high))
        auto_iters = max(1, int(args.auto_exposure_iters))
        auto_theory = bool(args.auto_exposure_theory)

        auto_weights = weight_grid[0] if weight_grid else (1.0, 1.0, 1.2)
        probe_lo = float(auto_low)
        probe_hi = float(auto_high)
        probe_trace: list[dict[str, object]] = []
        best_feasible: AblationCase | None = None
        for i in range(auto_iters):
            probe = 0.5 * (probe_lo + probe_hi)
            probe_case = _run_case(
                name=f"exp_auto_probe_{i + 1:02d}",
                bars=bars,
                start=start,
                end=end,
                signal_confidence_min=float(args.signal_confidence_min),
                convexity_min=float(args.convexity_min),
                max_daily_trades=int(args.max_daily_trades),
                hold_days=int(args.hold_days),
                exposure_scale=float(probe),
                proxy_lookback=int(args.proxy_lookback),
                theory_enabled=auto_theory,
                theory_ict_weight=float(auto_weights[0]),
                theory_brooks_weight=float(auto_weights[1]),
                theory_lie_weight=float(auto_weights[2]),
                theory_confidence_boost_max=float(args.theory_confidence_boost_max),
                theory_penalty_max=float(args.theory_penalty_max),
                theory_min_confluence=float(args.theory_min_confluence),
                theory_conflict_fuse=float(args.theory_conflict_fuse),
                max_drawdown_target=float(args.max_drawdown_target),
                drawdown_soft_band=float(args.drawdown_soft_band),
            )
            probe_trace.append(
                {
                    "iter": int(i + 1),
                    "probe_exposure": float(probe),
                    "max_drawdown": float(probe_case.max_drawdown),
                    "drawdown_excess": float(probe_case.drawdown_excess),
                    "target_breached": bool(probe_case.target_breached),
                    "annual_return": float(probe_case.annual_return),
                    "objective": float(probe_case.objective),
                }
            )
            if probe_case.target_breached:
                probe_hi = float(probe)
            else:
                probe_lo = float(probe)
                if best_feasible is None or float(probe_case.objective) > float(best_feasible.objective):
                    best_feasible = probe_case

        refine_exposures = sorted(
            {
                float(round(probe_lo, 4)),
                float(round(max(auto_low, probe_lo * 0.92), 4)),
                float(round(max(auto_low, probe_lo * 0.84), 4)),
                float(round(max(auto_low, probe_lo - 0.02), 4)),
            }
        )
        for idx, exp in enumerate(refine_exposures, start=1):
            refine_case = _run_case(
                name=f"exp_auto_refine_{idx:02d}",
                bars=bars,
                start=start,
                end=end,
                signal_confidence_min=float(args.signal_confidence_min),
                convexity_min=float(args.convexity_min),
                max_daily_trades=int(args.max_daily_trades),
                hold_days=int(args.hold_days),
                exposure_scale=float(exp),
                proxy_lookback=int(args.proxy_lookback),
                theory_enabled=auto_theory,
                theory_ict_weight=float(auto_weights[0]),
                theory_brooks_weight=float(auto_weights[1]),
                theory_lie_weight=float(auto_weights[2]),
                theory_confidence_boost_max=float(args.theory_confidence_boost_max),
                theory_penalty_max=float(args.theory_penalty_max),
                theory_min_confluence=float(args.theory_min_confluence),
                theory_conflict_fuse=float(args.theory_conflict_fuse),
                max_drawdown_target=float(args.max_drawdown_target),
                drawdown_soft_band=float(args.drawdown_soft_band),
            )
            probe_trace.append(
                {
                    "iter": int(auto_iters + idx),
                    "probe_exposure": float(exp),
                    "max_drawdown": float(refine_case.max_drawdown),
                    "drawdown_excess": float(refine_case.drawdown_excess),
                    "target_breached": bool(refine_case.target_breached),
                    "annual_return": float(refine_case.annual_return),
                    "objective": float(refine_case.objective),
                }
            )
            if not bool(refine_case.target_breached):
                if best_feasible is None or float(refine_case.objective) > float(best_feasible.objective):
                    best_feasible = refine_case

        if best_feasible is not None:
            best_payload = asdict(best_feasible)
            best_payload["name"] = "exp_auto_best"
            cases.append(AblationCase(**best_payload))
        auto_exposure_cfg = {
            "enabled": True,
            "low": float(auto_low),
            "high": float(auto_high),
            "iterations": int(auto_iters),
            "theory_enabled": bool(auto_theory),
            "best_exposure": None if best_feasible is None else float(best_feasible.exposure_scale),
            "best_case": {} if best_feasible is None else asdict(best_feasible),
            "trace": probe_trace,
        }

    t1 = datetime.now()
    sorted_cases = sorted(cases, key=lambda x: x.objective, reverse=True)
    feasible_cases = [x for x in sorted_cases if not bool(x.target_breached)]
    payload = {
        "started_at": t0.isoformat(),
        "ended_at": t1.isoformat(),
        "elapsed_seconds": float((t1 - t0).total_seconds()),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": symbols,
        "bars_rows": int(len(bars)),
        "data_source": data_source,
        "exposure_scale": float(args.exposure_scale),
        "max_drawdown_target": float(args.max_drawdown_target),
        "drawdown_soft_band": float(args.drawdown_soft_band),
        "fallback_path": fallback_path,
        "bundle_fetch_stats": bundle.fetch_stats,
        "cases": [asdict(c) for c in sorted_cases],
        "risk_feasible": bool(feasible_cases),
        "feasible_count": int(len(feasible_cases)),
        "best_feasible_case": asdict(feasible_cases[0]) if feasible_cases else {},
        "auto_exposure_search": auto_exposure_cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = _render_markdown(
        started_at=str(payload["started_at"]),
        ended_at=str(payload["ended_at"]),
        start=start,
        end=end,
        symbols=symbols,
        bars_rows=int(len(bars)),
        cases=sorted_cases,
        data_source=data_source,
        max_drawdown_target=float(args.max_drawdown_target),
        drawdown_soft_band=float(args.drawdown_soft_band),
        auto_exposure_search=auto_exposure_cfg,
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "bars_rows": int(len(bars)),
                "best_case": asdict(max(cases, key=lambda x: x.objective)) if cases else {},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
