#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml

from lie_engine.data.storage import write_json, write_markdown
from lie_engine.research.strategy_lab import StrategyLabSummary, run_strategy_lab


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _parse_windows(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        days = int(p)
        if days < 30:
            raise ValueError("window days must be >= 30")
        out.append(days)
    uniq = sorted(set(out), reverse=True)
    if not uniq:
        raise ValueError("at least one window is required")
    return uniq


def _parse_symbols(raw: str) -> list[str]:
    out = [s.strip() for s in str(raw).split(",") if s.strip()]
    return out


def _candidate_family(name: str) -> str:
    base = str(name or "").strip()
    if not base:
        return ""
    if "_" not in base:
        return base
    head, tail = base.rsplit("_", 1)
    if tail.isdigit():
        return head
    return base


def _param_drift_score(base_params: dict[str, Any], test_params: dict[str, Any], keys: list[str]) -> float:
    drifts: list[float] = []
    for key in keys:
        if key not in base_params or key not in test_params:
            continue
        try:
            a = float(base_params[key])
            b = float(test_params[key])
        except (TypeError, ValueError):
            continue
        den = max(abs(a), abs(b), 1e-9)
        drifts.append(min(4.0, abs(a - b) / den))
    if not drifts:
        return 1.0
    return float(mean(drifts))


def _stability_score(rows: list[dict[str, Any]], mdd_target: float, min_trades: int = 5) -> dict[str, float]:
    if not rows:
        return {
            "score": 0.0,
            "accept_ratio": 0.0,
            "dd_ok_ratio": 0.0,
            "return_pos_ratio": 0.0,
            "trade_activity_ratio": 0.0,
            "param_drift_penalty": 1.0,
            "exposure_dispersion": 0.0,
        }

    accepts = [1.0 if bool(r.get("best_accepted", False)) else 0.0 for r in rows]
    active = [1.0 if int(r.get("validation_trades", 0)) >= int(min_trades) else 0.0 for r in rows]
    dd_ok = [
        1.0
        if int(r.get("validation_trades", 0)) >= int(min_trades) and float(r.get("validation_max_drawdown", 1.0)) <= float(mdd_target)
        else 0.0
        for r in rows
    ]
    ret_pos = [
        1.0
        if int(r.get("validation_trades", 0)) >= int(min_trades) and float(r.get("validation_annual_return", 0.0)) > 0.0
        else 0.0
        for r in rows
    ]

    base = rows[0]
    base_params = base.get("best_params", {}) if isinstance(base.get("best_params", {}), dict) else {}
    drift_keys = [
        "signal_confidence_min",
        "convexity_min",
        "hold_days",
        "max_daily_trades",
        "exposure_scale",
        "theory_ict_weight",
        "theory_brooks_weight",
        "theory_lie_weight",
        "theory_confidence_boost_max",
        "theory_penalty_max",
        "theory_min_confluence",
        "theory_conflict_fuse",
    ]
    drift_scores: list[float] = []
    for row in rows[1:]:
        params = row.get("best_params", {}) if isinstance(row.get("best_params", {}), dict) else {}
        drift_scores.append(_param_drift_score(base_params, params, drift_keys))
    drift_penalty = float(mean(drift_scores)) if drift_scores else 0.0

    caps = [float(r.get("exposure_cap_applied", 0.0)) for r in rows]
    cap_mean = max(1e-9, float(mean(caps)))
    cap_disp = float(pstdev(caps) / cap_mean) if len(caps) > 1 else 0.0
    cap_penalty = min(1.0, cap_disp)

    score = (
        0.30 * float(mean(accepts))
        + 0.20 * float(mean(dd_ok))
        + 0.20 * float(mean(ret_pos))
        + 0.15 * float(mean(active))
        + 0.15 * float(max(0.0, 1.0 - drift_penalty))
        + 0.00 * float(max(0.0, 1.0 - cap_penalty))
    )
    return {
        "score": float(max(0.0, min(1.0, score))),
        "accept_ratio": float(mean(accepts)),
        "dd_ok_ratio": float(mean(dd_ok)),
        "return_pos_ratio": float(mean(ret_pos)),
        "trade_activity_ratio": float(mean(active)),
        "param_drift_penalty": float(drift_penalty),
        "exposure_dispersion": float(cap_disp),
    }


@dataclass(slots=True)
class StabilityWindowResult:
    window_days: int
    start: str
    end: str
    output_dir: str
    best_name: str
    best_family: str
    best_accepted: bool
    validation_annual_return: float
    validation_max_drawdown: float
    validation_trades: int
    exposure_cap_applied: float
    exposure_cap_components: dict[str, Any]
    best_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_window_result(window_days: int, end: date, summary: StrategyLabSummary) -> StabilityWindowResult:
    best = summary.best_candidate if isinstance(summary.best_candidate, dict) else {}
    valid = best.get("validation_metrics", {}) if isinstance(best.get("validation_metrics", {}), dict) else {}
    params = best.get("params", {}) if isinstance(best.get("params", {}), dict) else {}
    start = date.fromisoformat(summary.start_date)
    return StabilityWindowResult(
        window_days=int(window_days),
        start=start.isoformat(),
        end=end.isoformat(),
        output_dir=str(summary.output_dir),
        best_name=str(best.get("name", "")),
        best_family=_candidate_family(str(best.get("name", ""))),
        best_accepted=bool(best.get("accepted", False)),
        validation_annual_return=float(valid.get("annual_return", 0.0)),
        validation_max_drawdown=float(valid.get("max_drawdown", 0.0)),
        validation_trades=int(valid.get("trades", 0)),
        exposure_cap_applied=float(summary.exposure_cap_applied),
        exposure_cap_components=dict(summary.exposure_cap_components),
        best_params=dict(params),
    )


def _render_report(payload: dict[str, Any]) -> str:
    rows = payload.get("windows", [])
    stats = payload.get("stability", {})
    lines: list[str] = []
    lines.append(f"# Strategy Stability Report | {payload.get('ended_at', '')[:10]}")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- symbols: `{payload.get('symbols', [])}`")
    lines.append(f"- windows_days: `{payload.get('window_days', [])}`")
    lines.append(f"- end: `{payload.get('end', '')}`")
    lines.append(f"- max_drawdown_target: `{float(payload.get('max_drawdown_target', 0.0)):.2%}`")
    lines.append("")
    lines.append("## Stability")
    lines.append(f"- score: `{float(stats.get('score', 0.0)):.4f}`")
    lines.append(f"- accept_ratio: `{float(stats.get('accept_ratio', 0.0)):.4f}`")
    lines.append(f"- dd_ok_ratio: `{float(stats.get('dd_ok_ratio', 0.0)):.4f}`")
    lines.append(f"- return_pos_ratio: `{float(stats.get('return_pos_ratio', 0.0)):.4f}`")
    lines.append(f"- trade_activity_ratio: `{float(stats.get('trade_activity_ratio', 0.0)):.4f}`")
    lines.append(f"- param_drift_penalty: `{float(stats.get('param_drift_penalty', 0.0)):.4f}`")
    lines.append(f"- exposure_dispersion: `{float(stats.get('exposure_dispersion', 0.0)):.4f}`")
    lines.append("")
    lines.append("## Windows")
    lines.append("| window | best | family | accepted | ann_ret | mdd | trades | exposure_cap |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + f"{int(row.get('window_days', 0))}d | "
            + f"{row.get('best_name', '')} | "
            + f"{row.get('best_family', '')} | "
            + f"{int(bool(row.get('best_accepted', False)))} | "
            + f"{float(row.get('validation_annual_return', 0.0)):.4f} | "
            + f"{float(row.get('validation_max_drawdown', 0.0)):.4f} | "
            + f"{int(row.get('validation_trades', 0))} | "
            + f"{float(row.get('exposure_cap_applied', 0.0)):.4f} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    for row in rows:
        lines.append(f"- `{int(row.get('window_days', 0))}d`: `{row.get('output_dir', '')}`")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling strategy_lab windows and summarize cross-window stability.")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--window-days", default="365,180", help="comma-separated rolling windows, e.g. 365,180")
    parser.add_argument("--symbols", default="300750,002050,513130,LC2603", help="comma-separated symbols")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--max-symbols", type=int, default=8)
    parser.add_argument("--report-symbol-cap", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--review-days", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--max-drawdown-target", type=float, default=0.05)
    parser.add_argument("--review-max-drawdown-target", type=float, default=0.07)
    parser.add_argument("--drawdown-soft-band", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ended_at = datetime.now().isoformat()
    end = _parse_date(args.end)
    windows = _parse_windows(args.window_days)
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("symbols cannot be empty")

    output_root = Path(args.output_root)
    run_dir = output_root / "research" / f"strategy_stability_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[StabilityWindowResult] = []
    for days in windows:
        start = end - timedelta(days=int(days) - 1)
        summary = run_strategy_lab(
            output_root=output_root,
            core_symbols=list(symbols),
            start=start,
            end=end,
            max_symbols=int(args.max_symbols),
            report_symbol_cap=int(args.report_symbol_cap),
            workers=int(args.workers),
            review_days=int(args.review_days),
            candidate_count=int(args.candidate_count),
            max_drawdown_target=float(args.max_drawdown_target),
            review_max_drawdown_target=float(args.review_max_drawdown_target),
            drawdown_soft_band=float(args.drawdown_soft_band),
        )
        rows.append(_extract_window_result(days, end, summary))

    rows = sorted(rows, key=lambda r: int(r.window_days), reverse=True)
    row_dicts = [r.to_dict() for r in rows]
    stats = _stability_score(row_dicts, float(args.max_drawdown_target))
    payload = {
        "started_at": ended_at,
        "ended_at": datetime.now().isoformat(),
        "end": end.isoformat(),
        "window_days": list(windows),
        "symbols": list(symbols),
        "max_drawdown_target": float(args.max_drawdown_target),
        "windows": row_dicts,
        "stability": stats,
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir / "report.md", _render_report(payload))
    write_markdown(run_dir / "best.yaml", yaml.safe_dump(payload.get("windows", [{}])[0], allow_unicode=True, sort_keys=False))
    print(json.dumps({"run_dir": str(run_dir), "stability_score": float(stats.get("score", 0.0))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
