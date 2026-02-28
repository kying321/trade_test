#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from datetime import date, datetime
from statistics import median
from typing import Any, Iterable

import pandas as pd

import lie_engine.backtest.engine as bt_engine
from lie_engine.backtest.costs import AssetCost
from lie_engine.backtest.engine import BacktestConfig
from lie_engine.backtest.walk_forward import run_walk_forward_backtest
from lie_engine.engine import LieEngine
from lie_engine.models import BacktestResult
from lie_engine.signal.engine import expand_universe
from lie_engine.data.storage import write_json, write_markdown


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _parse_csv_floats(raw: str) -> list[float]:
    out: list[float] = []
    for x in str(raw).split(","):
        s = x.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("empty friction list")
    return out


def _default_windows() -> list[dict[str, str]]:
    return [
        {"name": "2015_crash", "start": "2015-01-01", "end": "2015-12-31"},
        {"name": "2020_pandemic", "start": "2020-01-01", "end": "2020-12-31"},
        {"name": "2022_geopolitical", "start": "2022-01-01", "end": "2022-12-31"},
        {"name": "extreme_gap", "start": "2024-01-01", "end": "2025-06-30"},
    ]


@contextmanager
def _scaled_cost_model(multiplier: float):
    orig = bt_engine.default_cost_model

    def _scaled(asset_class: str) -> AssetCost:
        c = orig(asset_class)
        m = float(multiplier)
        return AssetCost(
            fee_bps=float(c.fee_bps) * m,
            slippage_bps=float(c.slippage_bps) * m,
            impact_bps=float(c.impact_bps) * m,
            borrow_bps_daily=float(c.borrow_bps_daily) * m,
        )

    bt_engine.default_cost_model = _scaled
    try:
        yield
    finally:
        bt_engine.default_cost_model = orig


def _zero_result(start: date, end: date) -> BacktestResult:
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


def _robustness_score(row: dict[str, Any]) -> float:
    return float(
        float(row.get("avg_annual_return", 0.0))
        - float(row.get("worst_drawdown", 0.0))
        + 0.10 * (float(row.get("avg_profit_factor", 0.0)) - 1.0)
        + 0.05 * (float(row.get("avg_positive_window_ratio", 0.0)) - 0.5)
        - 0.02 * float(row.get("total_violations", 0))
    )


def _summarize(mode: str, friction: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [x for x in rows if str(x.get("status", "")) == "ok"]
    if not ok_rows:
        out = {
            "mode": mode,
            "friction_multiplier": float(friction),
            "windows": 0,
            "avg_annual_return": 0.0,
            "worst_drawdown": 0.0,
            "avg_win_rate": 0.0,
            "avg_profit_factor": 0.0,
            "avg_positive_window_ratio": 0.0,
            "total_violations": 0,
        }
        out["robustness_score"] = _robustness_score(out)
        return out
    n = float(len(ok_rows))
    out = {
        "mode": mode,
        "friction_multiplier": float(friction),
        "windows": int(len(ok_rows)),
        "avg_annual_return": float(sum(float(r.get("annual_return", 0.0)) for r in ok_rows) / n),
        "worst_drawdown": float(max(float(r.get("max_drawdown", 0.0)) for r in ok_rows)),
        "avg_win_rate": float(sum(float(r.get("win_rate", 0.0)) for r in ok_rows) / n),
        "avg_profit_factor": float(sum(float(r.get("profit_factor", 0.0)) for r in ok_rows) / n),
        "avg_positive_window_ratio": float(sum(float(r.get("positive_window_ratio", 0.0)) for r in ok_rows) / n),
        "total_violations": int(sum(int(r.get("violations", 0)) for r in ok_rows)),
    }
    out["robustness_score"] = _robustness_score(out)
    return out


def _nearest(values: Iterable[float], target: float) -> float:
    xs = [float(x) for x in values]
    if not xs:
        return float(target)
    return min(xs, key=lambda x: abs(x - float(target)))


def _render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Paper Stress + Friction Matrix ({payload.get('as_of')})")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- modes: `{', '.join(payload.get('modes', []))}`")
    lines.append(f"- friction_multipliers: `{payload.get('friction_multipliers', [])}`")
    lines.append(f"- windows: `{len(payload.get('windows', []))}`")
    lines.append("")
    lines.append("## Summary")
    for row in payload.get("summary", []):
        lines.append(
            "- "
            + f"mode={row.get('mode')} friction={row.get('friction_multiplier'):.2f} "
            + f"robustness={row.get('robustness_score'):.4f} "
            + f"ann={row.get('avg_annual_return'):.4f} dd={row.get('worst_drawdown'):.4f} "
            + f"pf={row.get('avg_profit_factor'):.4f} pwr={row.get('avg_positive_window_ratio'):.4f}"
        )
    lines.append("")
    lines.append("## Adaptive Recommendation (paper-only)")
    rec = payload.get("adaptive_recommendation", {})
    lines.append(f"- calibration_friction: `{rec.get('calibration_friction')}`")
    lines.append(f"- target_max_drawdown: `{rec.get('target_max_drawdown')}`")
    lines.append(f"- observed_worst_drawdown: `{rec.get('observed_worst_drawdown')}`")
    lines.append(f"- drawdown_pressure_ratio: `{rec.get('drawdown_pressure_ratio')}`")
    lines.append(
        "- "
        + "execution_min_risk_multiplier: "
        + f"`{rec.get('current_execution_min_risk_multiplier')} -> {rec.get('recommended_execution_min_risk_multiplier')}`"
    )
    lines.append(
        "- "
        + "mode_adaptive_update_step: "
        + f"`{rec.get('current_mode_adaptive_update_step')} -> {rec.get('recommended_mode_adaptive_update_step')}`"
    )
    lines.append(
        "- "
        + "review_loop_stress_matrix_autorun_max_runs: "
        + f"`{rec.get('current_review_loop_stress_matrix_autorun_max_runs')} -> {rec.get('recommended_review_loop_stress_matrix_autorun_max_runs')}`"
    )
    for reason in rec.get("reasons", []):
        lines.append(f"- reason: {reason}")
    lines.append("")
    lines.append("## Artifacts")
    paths = payload.get("paths", {})
    lines.append(f"- json: `{paths.get('json')}`")
    lines.append(f"- md: `{paths.get('md')}`")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run paper stress+friction matrix and output adaptive recommendations.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", required=True, help="as-of date, format YYYY-MM-DD")
    ap.add_argument("--modes", default="ultra_short,swing,long")
    ap.add_argument("--frictions", default="1.0,1.25,1.5,2.0")
    ap.add_argument("--calibration-friction", default="1.5")
    args = ap.parse_args()

    as_of = _parse_date(str(args.date))
    mode_raw = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    frictions = sorted(set(_parse_csv_floats(str(args.frictions))))
    calibration_friction = float(args.calibration_friction)

    eng = LieEngine(config_path=str(args.config))
    profiles = eng._resolved_mode_profiles()
    modes = [m for m in mode_raw if m in profiles]
    if not modes:
        modes = ["swing"] if "swing" in profiles else [next(iter(profiles.keys()))]

    windows = []
    for row in _default_windows():
        start = _parse_date(str(row["start"]))
        end = min(_parse_date(str(row["end"])), as_of)
        if end < start:
            continue
        windows.append({"name": str(row["name"]), "start": start, "end": end})

    symbols = eng._core_symbols()
    symbols = expand_universe(symbols, pd.DataFrame(), int(eng.settings.universe.get("max_dynamic_additions", 5)))
    trend_thr = float(eng.settings.thresholds.get("hurst_trend", 0.6))
    mean_thr = float(eng.settings.thresholds.get("hurst_mean_revert", 0.4))
    atr_extreme = float(eng.settings.thresholds.get("atr_extreme", 2.0))

    rows: list[dict[str, Any]] = []
    for w in windows:
        bars, _ = eng._run_ingestion_range(start=w["start"], end=w["end"], symbols=symbols)
        bars_empty = bool(bars.empty)
        for mode in modes:
            profile = profiles.get(mode, {})
            cfg = BacktestConfig(
                signal_confidence_min=float(profile.get("signal_confidence_min", 60.0)),
                convexity_min=float(profile.get("convexity_min", 2.0)),
                max_daily_trades=max(1, int(profile.get("max_daily_trades", 2))),
                hold_days=max(1, int(profile.get("hold_days", 5))),
            )
            for friction in frictions:
                if bars_empty:
                    result = _zero_result(start=w["start"], end=w["end"])
                    status = "no_data"
                else:
                    with _scaled_cost_model(friction):
                        result = run_walk_forward_backtest(
                            bars=bars,
                            start=w["start"],
                            end=w["end"],
                            trend_thr=trend_thr,
                            mean_thr=mean_thr,
                            atr_extreme=atr_extreme,
                            cfg_template=cfg,
                            train_years=3,
                            valid_years=1,
                            step_months=3,
                        )
                    status = "ok"
                rows.append(
                    {
                        "window": str(w["name"]),
                        "start": str(w["start"].isoformat()),
                        "end": str(w["end"].isoformat()),
                        "mode": mode,
                        "friction_multiplier": float(friction),
                        "status": status,
                        "total_return": float(result.total_return),
                        "annual_return": float(result.annual_return),
                        "max_drawdown": float(result.max_drawdown),
                        "win_rate": float(result.win_rate),
                        "profit_factor": float(result.profit_factor),
                        "expectancy": float(result.expectancy),
                        "trades": int(result.trades),
                        "violations": int(result.violations),
                        "positive_window_ratio": float(result.positive_window_ratio),
                    }
                )

    summary: list[dict[str, Any]] = []
    for mode in modes:
        for friction in frictions:
            group = [
                x
                for x in rows
                if str(x.get("mode", "")) == mode and float(x.get("friction_multiplier", 1.0)) == float(friction)
            ]
            summary.append(_summarize(mode, friction, group))

    summary.sort(key=lambda x: (str(x.get("mode", "")), float(x.get("friction_multiplier", 0.0))))

    val = eng.settings.validation if isinstance(eng.settings.validation, dict) else {}
    target_dd = float(val.get("max_drawdown_max", 0.05))
    current_exec_floor = float(val.get("execution_min_risk_multiplier", 0.20))
    current_mode_step = float(val.get("mode_adaptive_update_step", 0.08))
    current_auto_max_runs = int(val.get("review_loop_stress_matrix_autorun_max_runs", 1))
    cal_friction = _nearest(frictions, calibration_friction)
    cal_rows = [x for x in summary if float(x.get("friction_multiplier", 1.0)) == float(cal_friction)]
    observed_worst_dd = float(max((float(x.get("worst_drawdown", 0.0)) for x in cal_rows), default=0.0))
    observed_pf = float(median([float(x.get("avg_profit_factor", 0.0)) for x in cal_rows])) if cal_rows else 0.0
    observed_pwr = (
        float(median([float(x.get("avg_positive_window_ratio", 0.0)) for x in cal_rows])) if cal_rows else 0.0
    )
    dd_pressure = observed_worst_dd / max(target_dd, 1e-9)

    recommended_exec_floor = current_exec_floor
    if dd_pressure > 1.0:
        recommended_exec_floor = _clamp_float(current_exec_floor / dd_pressure, 0.10, current_exec_floor)
    recommended_mode_step = current_mode_step
    if observed_pf < 1.0 or observed_pwr < float(val.get("positive_window_ratio_min", 0.70)):
        recommended_mode_step = _clamp_float(current_mode_step * 0.75, 0.02, 0.50)
    recommended_auto_max_runs = current_auto_max_runs
    if dd_pressure > 1.20 and current_auto_max_runs < 3:
        recommended_auto_max_runs = current_auto_max_runs + 1

    reasons: list[str] = []
    reasons.append(
        "dd_pressure={:.4f}, observed_worst_dd={:.4f}, target_dd={:.4f}".format(
            dd_pressure, observed_worst_dd, target_dd
        )
    )
    reasons.append("observed_pf={:.4f}, observed_pwr={:.4f}".format(observed_pf, observed_pwr))
    if recommended_exec_floor < current_exec_floor:
        reasons.append("drawdown pressure > 1, reduce execution_min_risk_multiplier in paper.")
    if recommended_mode_step < current_mode_step:
        reasons.append("friction-degraded quality detected, reduce mode_adaptive_update_step to avoid thrash.")
    if recommended_auto_max_runs > current_auto_max_runs:
        reasons.append("drawdown pressure high, increase stress autorun max runs for faster convergence.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = eng.ctx.output_dir / "review" / f"{as_of.isoformat()}_paper_stress_friction_matrix_{ts}.json"
    out_md = eng.ctx.output_dir / "review" / f"{as_of.isoformat()}_paper_stress_friction_matrix_{ts}.md"
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "as_of": as_of.isoformat(),
        "modes": modes,
        "friction_multipliers": frictions,
        "windows": [
            {
                "name": str(w["name"]),
                "start": str(w["start"].isoformat()),
                "end": str(w["end"].isoformat()),
            }
            for w in windows
        ],
        "matrix": rows,
        "summary": summary,
        "adaptive_recommendation": {
            "calibration_friction": float(cal_friction),
            "target_max_drawdown": float(target_dd),
            "observed_worst_drawdown": float(observed_worst_dd),
            "drawdown_pressure_ratio": float(dd_pressure),
            "observed_profit_factor_median": float(observed_pf),
            "observed_positive_window_ratio_median": float(observed_pwr),
            "current_execution_min_risk_multiplier": float(current_exec_floor),
            "recommended_execution_min_risk_multiplier": float(recommended_exec_floor),
            "current_mode_adaptive_update_step": float(current_mode_step),
            "recommended_mode_adaptive_update_step": float(recommended_mode_step),
            "current_review_loop_stress_matrix_autorun_max_runs": int(current_auto_max_runs),
            "recommended_review_loop_stress_matrix_autorun_max_runs": int(recommended_auto_max_runs),
            "reasons": reasons,
        },
        "paths": {"json": str(out_json), "md": str(out_md)},
    }
    write_json(out_json, payload)
    write_markdown(out_md, _render_md(payload))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
