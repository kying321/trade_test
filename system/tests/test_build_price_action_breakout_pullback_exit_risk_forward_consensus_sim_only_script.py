from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fenlie_exit_risk_forward_consensus_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_classify_forward_consensus_keeps_baseline_when_30d_wins_and_40d_ties() -> None:
    module = _load_module()

    assert module.classify_forward_consensus(
        [
            {
                "train_days": 30,
                "research_decision": "baseline_forward_oos_pair_keeps_anchor",
                "winner_by_aggregate_return": "baseline_pair",
                "winner_by_aggregate_objective": "baseline_pair",
            },
            {
                "train_days": 40,
                "research_decision": "mixed_forward_oos_pair_agg_ret_tie_agg_obj_tie_slice_ret_tie_slice_obj_tie",
                "winner_by_aggregate_return": "tie",
                "winner_by_aggregate_objective": "tie",
            },
        ]
    ) == "baseline_pair_keeps_anchor_challenger_not_promoted_across_30d_40d_forward_oos"


def test_classify_forward_consensus_marks_challenger_promotable_when_no_baseline_windows_remain() -> None:
    module = _load_module()

    assert module.classify_forward_consensus(
        [
            {
                "train_days": 30,
                "research_decision": "challenger_forward_oos_pair_wins",
                "winner_by_aggregate_return": "challenger_pair",
                "winner_by_aggregate_objective": "challenger_pair",
            },
            {
                "train_days": 45,
                "research_decision": "challenger_forward_oos_pair_wins",
                "winner_by_aggregate_return": "challenger_pair",
                "winner_by_aggregate_objective": "challenger_pair",
            },
            {
                "train_days": 55,
                "research_decision": "mixed_forward_oos_pair_agg_ret_tie_agg_obj_tie_slice_ret_tie_slice_obj_tie",
                "winner_by_aggregate_return": "tie",
                "winner_by_aggregate_objective": "tie",
            },
        ]
    ) == "challenger_pair_promotable_across_current_forward_oos"


def test_builder_writes_forward_consensus_and_latest_alias(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_30 = review_dir / "20260323T033500Z_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    compare_40 = review_dir / "20260323T032500Z_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    _write_json(
        compare_30,
        {
            "symbol": "ETHUSDT",
            "train_days": 30,
            "validation_days": 10,
            "step_days": 10,
            "challenge_pair_source_decision": "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos",
            "research_decision": "baseline_forward_oos_pair_keeps_anchor",
            "comparison_summary": {
                "winner_by_aggregate_return": "baseline_pair",
                "winner_by_aggregate_objective": "baseline_pair",
            },
        },
    )
    _write_json(
        compare_40,
        {
            "symbol": "ETHUSDT",
            "train_days": 40,
            "validation_days": 10,
            "step_days": 10,
            "challenge_pair_source_decision": "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos",
            "research_decision": "mixed_forward_oos_pair_agg_ret_tie_agg_obj_tie_slice_ret_tie_slice_obj_tie",
            "comparison_summary": {
                "winner_by_aggregate_return": "tie",
                "winner_by_aggregate_objective": "tie",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T034500Z",
            "--compare-path",
            str(compare_30),
            "--compare-path",
            str(compare_40),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    json_path = Path(output["json_path"])
    latest_json_path = Path(output["latest_json_path"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "baseline_pair_keeps_anchor_challenger_not_promoted_across_30d_40d_forward_oos"
    assert payload["baseline_windows"] == 1
    assert payload["tie_windows"] == 1
    assert payload["challenger_windows"] == 0
    assert payload["blocked_now"] == ["promote_challenger_pair_as_new_exit_risk_anchor"]
    assert payload["allowed_now"] == [
        "keep_baseline_pair_as_current_exit_risk_anchor",
        "treat_40d_tie_window_as_watch_only",
    ]
    assert payload["next_research_priority"] == "collect_more_tail_or_test_break_even_sidecar_separately"
    assert payload["window_summary"] == [
        {
            "train_days": 30,
            "validation_days": 10,
            "step_days": 10,
            "research_decision": "baseline_forward_oos_pair_keeps_anchor",
            "winner_by_aggregate_return": "baseline_pair",
            "winner_by_aggregate_objective": "baseline_pair",
        },
        {
            "train_days": 40,
            "validation_days": 10,
            "step_days": 10,
            "research_decision": "mixed_forward_oos_pair_agg_ret_tie_agg_obj_tie_slice_ret_tie_slice_obj_tie",
            "winner_by_aggregate_return": "tie",
            "winner_by_aggregate_objective": "tie",
        },
    ]
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_marks_allowed_now_when_challenger_is_promotable(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_30 = review_dir / "20260323T200130Z_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    compare_55 = review_dir / "20260323T200155Z_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    _write_json(
        compare_30,
        {
            "symbol": "ETHUSDT",
            "train_days": 30,
            "validation_days": 10,
            "step_days": 10,
            "challenge_pair_source_decision": "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos",
            "research_decision": "challenger_forward_oos_pair_wins",
            "comparison_summary": {
                "winner_by_aggregate_return": "challenger_pair",
                "winner_by_aggregate_objective": "challenger_pair",
            },
        },
    )
    _write_json(
        compare_55,
        {
            "symbol": "ETHUSDT",
            "train_days": 55,
            "validation_days": 10,
            "step_days": 10,
            "challenge_pair_source_decision": "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos",
            "research_decision": "mixed_forward_oos_pair_agg_ret_tie_agg_obj_tie_slice_ret_tie_slice_obj_tie",
            "comparison_summary": {
                "winner_by_aggregate_return": "tie",
                "winner_by_aggregate_objective": "tie",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T200300Z",
            "--compare-path",
            str(compare_30),
            "--compare-path",
            str(compare_55),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "challenger_pair_promotable_across_current_forward_oos"
    assert payload["blocked_now"] == []
    assert payload["allowed_now"] == [
        "promote_challenger_pair_as_new_exit_risk_anchor",
        "treat_55d_plus_tie_windows_as_watch_only",
    ]
    assert payload["next_research_priority"] == "refresh_exit_risk_anchor_after_forward_oos_promotion"


def test_builder_marks_baseline_only_consensus_as_defer_to_canonical_follow_up(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_30 = review_dir / "20260324T050002Z_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    compare_40 = review_dir / "20260324T050003Z_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    _write_json(
        compare_30,
        {
            "symbol": "ETHUSDT",
            "train_days": 30,
            "validation_days": 10,
            "step_days": 10,
            "challenge_pair_source_decision": "block_exit_risk_promotion_require_hold16_vs_hold24_forward_oos",
            "research_decision": "baseline_forward_oos_pair_keeps_anchor",
            "comparison_summary": {
                "winner_by_aggregate_return": "baseline_pair",
                "winner_by_aggregate_objective": "baseline_pair",
            },
        },
    )
    _write_json(
        compare_40,
        {
            "symbol": "ETHUSDT",
            "train_days": 40,
            "validation_days": 10,
            "step_days": 10,
            "challenge_pair_source_decision": "block_exit_risk_promotion_require_hold16_vs_hold24_forward_oos",
            "research_decision": "baseline_forward_oos_pair_keeps_anchor",
            "comparison_summary": {
                "winner_by_aggregate_return": "baseline_pair",
                "winner_by_aggregate_objective": "baseline_pair",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T050008Z",
            "--compare-path",
            str(compare_30),
            "--compare-path",
            str(compare_40),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "baseline_pair_keeps_anchor_across_current_forward_oos"
    assert payload["blocked_now"] == ["promote_challenger_pair_as_new_exit_risk_anchor"]
    assert payload["allowed_now"] == ["keep_baseline_pair_as_current_exit_risk_anchor"]
    assert payload["next_research_priority"] == "defer_to_canonical_exit_risk_blocker_or_handoff"
