from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_hold_frontier_report_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_frontier(
    tmp_path: Path,
    *,
    hold_family_overall_summary: dict,
    hold_robustness_decision: str,
) -> dict:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    base_artifact_path = review_dir / "20260321T085500Z_price_action_breakout_pullback_sim_only.json"
    hold_robustness_path = review_dir / "20260321T091200Z_price_action_breakout_pullback_exit_hold_robustness_sim_only.json"
    hold_family_triage_path = review_dir / "20260321T091400Z_price_action_breakout_pullback_hold_family_triage_sim_only.json"
    rider_triage_path = review_dir / "20260321T091600Z_price_action_breakout_pullback_exit_rider_triage_sim_only.json"

    _write_json(
        base_artifact_path,
        {
            "focus_symbol": "ETHUSDT",
            "selected_params": {
                "breakout_lookback": 40,
                "breakout_memory_bars": 16,
                "pullback_max_atr": 1.2,
                "stop_buffer_atr": 0.1,
                "target_r": 2.0,
                "max_hold_bars": 16,
            },
        },
    )
    _write_json(
        hold_robustness_path,
        {
            "research_decision": hold_robustness_decision,
            "overall_summary": {
                "aggregate_return_scheme_wins": {"hold_8_zero_risk": 0, "hold_16_zero_risk": 4, "tie": 0},
                "aggregate_objective_scheme_wins": {"hold_8_zero_risk": 0, "hold_16_zero_risk": 4, "tie": 0},
                "slice_majority_return_scheme_wins": {"hold_8_zero_risk": 0, "hold_16_zero_risk": 4, "tie": 0},
                "slice_majority_objective_scheme_wins": {"hold_8_zero_risk": 0, "hold_16_zero_risk": 4, "tie": 0},
                "scheme_decision_counts": {"hold_8_scheme_leader": 0, "hold_16_scheme_leader": 4, "mixed_profile": 0},
            },
        },
    )
    _write_json(
        hold_family_triage_path,
        {
            "research_decision": "hold_family_triage_fixture",
            "overall_summary": hold_family_overall_summary,
            "grid_rows": [
                {
                    "aggregate_selected_metrics_by_config": {
                        "hold8_zero": {
                            "trade_count": 6,
                            "cumulative_return": -0.008,
                            "sharpe_per_trade": -0.4,
                            "profit_factor": 1.05,
                            "expectancy_r": 0.01,
                            "max_drawdown": 0.02,
                        },
                        "hold12_zero": {
                            "trade_count": 6,
                            "cumulative_return": 0.010,
                            "sharpe_per_trade": 0.2,
                            "profit_factor": 1.4,
                            "expectancy_r": 0.12,
                            "max_drawdown": 0.015,
                        },
                        "hold16_zero": {
                            "trade_count": 6,
                            "cumulative_return": 0.033,
                            "sharpe_per_trade": 1.3,
                            "profit_factor": 3.8,
                            "expectancy_r": 0.66,
                            "max_drawdown": 0.011,
                        },
                        "hold24_zero": {
                            "trade_count": 6,
                            "cumulative_return": 0.016,
                            "sharpe_per_trade": 0.7,
                            "profit_factor": 2.0,
                            "expectancy_r": 0.21,
                            "max_drawdown": 0.019,
                        },
                    }
                }
            ],
        },
    )
    _write_json(
        rider_triage_path,
        {
            "triage_summary": {
                "candidate_vs_baseline": [
                    {
                        "candidate_id": "hold16_be075",
                        "worse_objective_grids": 4,
                        "exact_metric_match_to_baseline_all_grids": False,
                    },
                    {
                        "candidate_id": "hold16_trail15",
                        "worse_objective_grids": 4,
                        "exact_metric_match_to_baseline_all_grids": False,
                    },
                    {
                        "candidate_id": "hold16_cd2x16",
                        "worse_objective_grids": 0,
                        "exact_metric_match_to_baseline_all_grids": True,
                    },
                ]
            }
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--base-artifact-path",
            str(base_artifact_path),
            "--hold-robustness-path",
            str(hold_robustness_path),
            "--hold-family-triage-path",
            str(hold_family_triage_path),
            "--rider-triage-path",
            str(rider_triage_path),
            "--stamp",
            "20260321T092500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    return json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))


def test_frontier_report_does_not_overstate_dual_candidates_when_aggregate_wins_are_zero(tmp_path: Path) -> None:
    payload = _run_frontier(
        tmp_path,
        hold_family_overall_summary={
            "grid_count": 4,
            "aggregate_return_scheme_wins": {"hold8_zero": 0, "hold12_zero": 0, "hold16_zero": 4, "hold24_zero": 0},
            "aggregate_objective_scheme_wins": {"hold8_zero": 0, "hold12_zero": 1, "hold16_zero": 3, "hold24_zero": 0},
            "unique_slice_return_wins_total": {"hold8_zero": 5, "hold12_zero": 4, "hold16_zero": 4, "hold24_zero": 0},
            "unique_slice_objective_wins_total": {"hold8_zero": 5, "hold12_zero": 4, "hold16_zero": 4, "hold24_zero": 0},
        },
        hold_robustness_decision="hold_16_consistency_reinforced_keep_baseline",
    )

    rows = {row["config_id"]: row for row in payload["frontier_rows"]}
    assert rows["hold16_zero"]["role"] == "baseline_anchor"
    assert rows["hold8_zero"]["role"] != "objective_leader_candidate"
    assert rows["hold12_zero"]["role"] == "transfer_watch_candidate"
    assert "hold24_zero" not in rows
    dropped_ids = {row["config_id"] for row in payload["dropped_rows"]}
    assert "hold12_zero" not in dropped_ids
    assert "objective_candidate=hold8_zero" not in payload["recommended_brief"]
    assert "return_candidate=hold24_zero" not in payload["recommended_brief"]


def test_frontier_report_preserves_dual_candidate_structure_when_aggregate_leaders_exist(tmp_path: Path) -> None:
    payload = _run_frontier(
        tmp_path,
        hold_family_overall_summary={
            "grid_count": 4,
            "aggregate_return_scheme_wins": {"hold8_zero": 0, "hold12_zero": 0, "hold16_zero": 0, "hold24_zero": 4},
            "aggregate_objective_scheme_wins": {"hold8_zero": 4, "hold12_zero": 0, "hold16_zero": 0, "hold24_zero": 0},
            "unique_slice_return_wins_total": {"hold8_zero": 2, "hold12_zero": 0, "hold16_zero": 3, "hold24_zero": 5},
            "unique_slice_objective_wins_total": {"hold8_zero": 5, "hold12_zero": 0, "hold16_zero": 3, "hold24_zero": 1},
        },
        hold_robustness_decision="mixed_robustness_keep_hold16_baseline_hold8_candidate",
    )

    rows = {row["config_id"]: row for row in payload["frontier_rows"]}
    assert rows["hold16_zero"]["role"] == "baseline_anchor"
    assert rows["hold8_zero"]["role"] == "objective_leader_candidate"
    assert rows["hold24_zero"]["role"] == "return_leader_candidate"
    dropped_ids = {row["config_id"] for row in payload["dropped_rows"]}
    assert "hold12_zero" in dropped_ids
    assert "objective_candidate=hold8_zero" in payload["recommended_brief"]
    assert "return_candidate=hold24_zero" in payload["recommended_brief"]
