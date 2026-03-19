#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
EXIT_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_exit_risk_sim_only.py")
FAMILY_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_hold_family_triage_sim_only.py")
ROUTER_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_hold_router_hypothesis_sim_only.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXIT_MODULE = load_module("fenlie_price_action_breakout_pullback_exit_risk", EXIT_SCRIPT_PATH)
FAMILY_MODULE = load_module("fenlie_hold_family_triage", FAMILY_SCRIPT_PATH)
ROUTER_MODULE = load_module("fenlie_hold_router_hypothesis", ROUTER_SCRIPT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY transfer challenge report for the hold-family frontier using a frozen derivation window."
    )
    parser.add_argument("--long-dataset-path", required=True)
    parser.add_argument("--derivation-dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--current-hold-family-artifact-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def frame_window_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"row_count": 0, "coverage_start_utc": "", "coverage_end_utc": ""}
    return {
        "row_count": int(len(frame)),
        "coverage_start_utc": FAMILY_MODULE.COMPARE_MODULE.BASE_MODULE.fmt_utc(pd.Timestamp(frame["ts"].min()).to_pydatetime()),
        "coverage_end_utc": FAMILY_MODULE.COMPARE_MODULE.BASE_MODULE.fmt_utc(pd.Timestamp(frame["ts"].max()).to_pydatetime()),
    }


def evaluate_window(frame: pd.DataFrame, base_entry_params: dict[str, Any]) -> dict[str, Any]:
    grid_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for grid in FAMILY_MODULE.ROBUSTNESS_GRIDS:
        try:
            row = FAMILY_MODULE.evaluate_grid(frame=frame, base_entry_params=base_entry_params, grid=grid)
            grid_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            blocked_rows.append(
                {
                    "grid_id": text(grid.get("grid_id")),
                    "status": "blocked",
                    "blocked_reason": text(exc),
                }
            )
    overall_summary = FAMILY_MODULE.build_overall_summary(grid_rows) if grid_rows else {}
    window_decision = FAMILY_MODULE.classify_research_decision(overall_summary) if grid_rows else "window_unavailable"
    return {
        "window_summary": frame_window_summary(frame),
        "eligible_grid_count": int(len(grid_rows)),
        "grid_rows": grid_rows,
        "blocked_rows": blocked_rows,
        "overall_summary": overall_summary,
        "window_decision": window_decision,
    }


def classify_transfer_decision(
    *,
    current_family_payload: dict[str, Any],
    historical_transfer: dict[str, Any],
    future_tail_probe: dict[str, Any],
) -> str:
    hist_overall = dict(historical_transfer.get("overall_summary") or {})
    hist_agg_ret = dict(hist_overall.get("aggregate_return_scheme_wins") or {})
    hist_agg_obj = dict(hist_overall.get("aggregate_objective_scheme_wins") or {})
    future_eligible = int(future_tail_probe.get("eligible_grid_count", 0) or 0)
    if int(historical_transfer.get("eligible_grid_count", 0) or 0) <= 0:
        return "hold_family_transfer_unavailable"
    hold12_revives = int(hist_agg_ret.get("hold12_zero", 0) or 0) > 0 or int(hist_agg_obj.get("hold12_zero", 0) or 0) > 0
    hold24_fades = int(hist_agg_ret.get("hold24_zero", 0) or 0) == 0 and int(hist_agg_obj.get("hold24_zero", 0) or 0) == 0
    current_decision = text(current_family_payload.get("research_decision"))
    if hold12_revives and hold24_fades and future_eligible == 0:
        return "historical_transfer_revives_hold12_and_demotes_hold24_future_tail_insufficient"
    if hold12_revives and hold24_fades:
        return "historical_transfer_revives_hold12_and_demotes_hold24"
    if current_decision and text(historical_transfer.get("window_decision")) != current_decision:
        return "hold_family_transfer_regime_shift_detected"
    return "hold_family_transfer_consistent_or_inconclusive"


def render_markdown(payload: dict[str, Any]) -> str:
    hist = dict(payload.get("historical_transfer_challenge") or {})
    fut = dict(payload.get("future_tail_probe") or {})
    current = dict(payload.get("current_hold_family_head") or {})
    lines = [
        "# Price Action Breakout Pullback Hold Family Transfer Challenge SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Current Derivation Head",
        "",
        f"- current_research_decision: `{text(current.get('research_decision'))}`",
        f"- current_overall_summary: `{json.dumps(current.get('overall_summary') or {}, ensure_ascii=False)}`",
        "",
        "## Historical Transfer Challenge",
        "",
        f"- coverage: `{text(hist.get('window_summary', {}).get('coverage_start_utc'))} -> {text(hist.get('window_summary', {}).get('coverage_end_utc'))}`",
        f"- eligible_grid_count: `{hist.get('eligible_grid_count')}`",
        f"- window_decision: `{text(hist.get('window_decision'))}`",
        f"- overall_summary: `{json.dumps(hist.get('overall_summary') or {}, ensure_ascii=False)}`",
        "",
        "## Future Tail Probe",
        "",
        f"- coverage: `{text(fut.get('window_summary', {}).get('coverage_start_utc'))} -> {text(fut.get('window_summary', {}).get('coverage_end_utc'))}`",
        f"- eligible_grid_count: `{fut.get('eligible_grid_count')}`",
        "",
        "## Notes",
        "",
        f"- `{text(payload.get('research_note'))}`",
        f"- `{text(payload.get('limitation_note'))}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    stamp_dt = FAMILY_MODULE.COMPARE_MODULE.BASE_MODULE.parse_stamp(args.stamp)
    long_dataset_path = Path(args.long_dataset_path).expanduser().resolve()
    derivation_dataset_path = Path(args.derivation_dataset_path).expanduser().resolve()
    base_artifact_path = Path(args.base_artifact_path).expanduser().resolve()
    current_hold_family_artifact_path = Path(args.current_hold_family_artifact_path).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    symbol = text(args.symbol).upper()

    current_family_payload = json.loads(current_hold_family_artifact_path.read_text(encoding="utf-8"))
    base_entry_params, base_payload = EXIT_MODULE.load_base_entry_params(base_artifact_path, symbol)
    long_frame = ROUTER_MODULE.load_signal_frame(long_dataset_path, symbol, int(base_entry_params["breakout_lookback"]))
    derivation_frame = ROUTER_MODULE.load_signal_frame(derivation_dataset_path, symbol, int(base_entry_params["breakout_lookback"]))
    derivation_start = pd.Timestamp(derivation_frame["ts"].min())
    derivation_end = pd.Timestamp(derivation_frame["ts"].max())

    historical_transfer_frame = long_frame[long_frame["ts"] < derivation_start].copy().reset_index(drop=True)
    future_tail_frame = long_frame[long_frame["ts"] > derivation_end].copy().reset_index(drop=True)

    historical_transfer = evaluate_window(historical_transfer_frame, base_entry_params)
    future_tail_probe = evaluate_window(future_tail_frame, base_entry_params)
    research_decision = classify_transfer_decision(
        current_family_payload=current_family_payload,
        historical_transfer=historical_transfer,
        future_tail_probe=future_tail_probe,
    )

    payload = {
        "action": "build_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": FAMILY_MODULE.COMPARE_MODULE.BASE_MODULE.fmt_utc(stamp_dt),
        "symbol": symbol,
        "long_dataset_path": str(long_dataset_path),
        "derivation_dataset_path": str(derivation_dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "current_hold_family_artifact_path": str(current_hold_family_artifact_path),
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "long_dataset_window": frame_window_summary(long_frame),
        "derivation_window": frame_window_summary(derivation_frame),
        "current_hold_family_head": {
            "research_decision": text(current_family_payload.get("research_decision")),
            "overall_summary": current_family_payload.get("overall_summary") or {},
        },
        "historical_transfer_challenge": historical_transfer,
        "future_tail_probe": future_tail_probe,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:hold_family_transfer:"
            f"hist_grids={int(historical_transfer.get('eligible_grid_count', 0) or 0)},"
            f"hist_decision={text(historical_transfer.get('window_decision'))},"
            f"future_grids={int(future_tail_probe.get('eligible_grid_count', 0) or 0)},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份工件冻结当前 derivation window 的 hold-family head，"
            "然后用完全不重叠的历史窗口检查候选集是否发生 regime drift。"
        ),
        "limitation_note": (
            "derivation 之后新增 future tail 只有 36 根 15m bars，"
            "不足以提供新的 forward challenge；因此当前 drift 证据主要来自 historical transfer。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "latest_json_path": str(latest_json_path),
                "research_decision": research_decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
