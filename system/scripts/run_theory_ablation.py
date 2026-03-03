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
    theory_ict_weight: float
    theory_brooks_weight: float
    theory_lie_weight: float
    annual_return: float
    max_drawdown: float
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


def _objective(annual_return: float, max_drawdown: float, positive_window_ratio: float, violations: int) -> float:
    return float(annual_return - 0.90 * max_drawdown + 0.20 * positive_window_ratio - 0.15 * float(violations))


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
    theory_ict_weight: float,
    theory_brooks_weight: float,
    theory_lie_weight: float,
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
            regime_recalc_interval=5,
            signal_eval_interval=1,
            proxy_lookback=int(proxy_lookback),
            theory_enabled=bool(theory_enabled),
            theory_ict_weight=float(theory_ict_weight),
            theory_brooks_weight=float(theory_brooks_weight),
            theory_lie_weight=float(theory_lie_weight),
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
    )
    return AblationCase(
        name=name,
        theory_enabled=bool(theory_enabled),
        theory_ict_weight=float(theory_ict_weight),
        theory_brooks_weight=float(theory_brooks_weight),
        theory_lie_weight=float(theory_lie_weight),
        annual_return=float(bt.annual_return),
        max_drawdown=float(bt.max_drawdown),
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

    def _finalize(frame: pd.DataFrame, path_label: str) -> tuple[pd.DataFrame, str]:
        frame = frame.copy()
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        frame = frame.dropna(subset=["ts"])
        frame = frame[(frame["ts"].dt.date >= start) & (frame["ts"].dt.date <= end)]
        if frame.empty:
            return pd.DataFrame(), ""
        filtered = frame[frame["symbol"].astype(str).isin(set(symbols))] if symbols else frame.copy()
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
        + list((output_root / "research").glob("*/bars_used.parquet")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            bars = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        if bars.empty or not required.issubset(set(bars.columns)):
            continue
        out, label = _finalize(bars, str(path))
        if not out.empty:
            return out, label

    csv_candidates = sorted(
        list((output_root / "artifacts" / "raw").glob("*_bars_raw.csv"))
        + list((output_root / "artifacts" / "normalized").glob("*_bars_normalized.csv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in csv_candidates:
        try:
            bars = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        if bars.empty or not required.issubset(set(bars.columns)):
            continue
        out, label = _finalize(bars, str(path))
        if not out.empty:
            return out, label
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
    lines.append("")
    lines.append("## Cases")
    lines.append("| name | theory | ict | brooks | lie | ann_ret | mdd | pwr | pf | wr | trades | viol | sharpe_like | obj |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        lines.append(
            "| "
            + f"{c.name} | {int(c.theory_enabled)} | {c.theory_ict_weight:.2f} | {c.theory_brooks_weight:.2f} | {c.theory_lie_weight:.2f} | "
            + f"{c.annual_return:.4f} | {c.max_drawdown:.4f} | {c.positive_window_ratio:.4f} | {c.profit_factor:.4f} | "
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
                + f"Δpf={c.profit_factor - base.profit_factor:+.4f}, "
                + f"Δwr={c.win_rate - base.win_rate:+.4f}, "
                + f"Δtrades={c.trades - base.trades:+d}, "
                + f"Δobj={c.objective - base.objective:+.4f}"
            )
        lines.append("")

    best = max(cases, key=lambda x: x.objective) if cases else None
    lines.append("## Conclusion")
    if best is None:
        lines.append("- no case executed")
    else:
        lines.append(
            "- "
            + f"best_by_objective: `{best.name}` "
            + f"(obj={best.objective:.4f}, ann_ret={best.annual_return:.4f}, mdd={best.max_drawdown:.4f}, trades={best.trades})"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run theory on/off and weight-grid ablation with consistent backtest setup.")
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
    parser.add_argument("--proxy-lookback", type=int, default=180)
    parser.add_argument(
        "--weight-grid",
        default="1.0:1.0:1.2;1.2:1.0:1.3;1.2:1.1:1.3;1.4:0.9:1.2",
        help="semicolon-separated ict:brooks:lie triples",
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
            proxy_lookback=int(args.proxy_lookback),
            theory_enabled=False,
            theory_ict_weight=1.0,
            theory_brooks_weight=1.0,
            theory_lie_weight=1.2,
        )
    )
    for idx, (w_ict, w_brooks, w_lie) in enumerate(weight_grid, start=1):
        cases.append(
            _run_case(
                name=f"theory_on_{idx:02d}",
                bars=bars,
                start=start,
                end=end,
                signal_confidence_min=float(args.signal_confidence_min),
                convexity_min=float(args.convexity_min),
                max_daily_trades=int(args.max_daily_trades),
                hold_days=int(args.hold_days),
                proxy_lookback=int(args.proxy_lookback),
                theory_enabled=True,
                theory_ict_weight=float(w_ict),
                theory_brooks_weight=float(w_brooks),
                theory_lie_weight=float(w_lie),
            )
        )

    t1 = datetime.now()
    payload = {
        "started_at": t0.isoformat(),
        "ended_at": t1.isoformat(),
        "elapsed_seconds": float((t1 - t0).total_seconds()),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": symbols,
        "bars_rows": int(len(bars)),
        "data_source": data_source,
        "fallback_path": fallback_path,
        "bundle_fetch_stats": bundle.fetch_stats,
        "cases": [asdict(c) for c in sorted(cases, key=lambda x: x.objective, reverse=True)],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = _render_markdown(
        started_at=str(payload["started_at"]),
        ended_at=str(payload["ended_at"]),
        start=start,
        end=end,
        symbols=symbols,
        bars_rows=int(len(bars)),
        cases=sorted(cases, key=lambda x: x.objective, reverse=True),
        data_source=data_source,
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
