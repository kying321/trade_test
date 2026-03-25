from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fenlie_exit_risk_break_even_sidecar_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_compare(
    path: Path,
    *,
    train_days: int,
    validation_days: int,
    step_days: int,
    validation_window_mode: str,
    research_decision: str,
    winner_by_aggregate_return: str,
    winner_by_aggregate_objective: str,
    winner_by_slice_majority_return: str,
    winner_by_slice_majority_objective: str,
) -> None:
    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "symbol": "ETHUSDT",
        "train_days": train_days,
        "validation_days": validation_days,
        "step_days": step_days,
        "validation_window_mode": validation_window_mode,
        "slice_count": 2,
        "research_decision": research_decision,
        "comparison_configs": [
            {
                "config_id": "anchor_no_be",
                "label": "anchor hold=16, be=0.0, trail=0.0",
                "exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
            },
            {
                "config_id": "anchor_with_be",
                "label": "anchor hold=16, be=0.75, trail=0.0",
                "exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.75,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
            },
        ],
        "comparison_summary": {
            "winner_by_aggregate_return": winner_by_aggregate_return,
            "winner_by_aggregate_objective": winner_by_aggregate_objective,
            "winner_by_slice_majority_return": winner_by_slice_majority_return,
            "winner_by_slice_majority_objective": winner_by_slice_majority_objective,
        },
        "aggregate_validation_metrics_by_config": {
            "anchor_no_be": {
                "cumulative_return": -0.002 if winner_by_aggregate_return == "anchor_no_be" else -0.01,
                "max_drawdown": 0.02,
                "trade_count": 8,
                "avg_hold_bars": 5.0,
            },
            "anchor_with_be": {
                "cumulative_return": -0.01 if winner_by_aggregate_return == "anchor_no_be" else -0.002,
                "max_drawdown": 0.02,
                "trade_count": 8,
                "avg_hold_bars": 5.0,
            },
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_classify_sidecar_decision_keeps_anchor_when_break_even_does_not_win() -> None:
    module = _load_module()
    windows = [
        {
            "winner_by_aggregate_return": "anchor_no_be",
            "winner_by_aggregate_objective": "anchor_no_be",
        },
        {
            "winner_by_aggregate_return": "tie",
            "winner_by_aggregate_objective": "tie",
        },
    ]

    assert module.classify_sidecar_decision(windows) == "break_even_sidecar_not_promising_keep_anchor"


def test_classify_sidecar_decision_marks_full_tie_as_no_delta() -> None:
    module = _load_module()
    windows = [
        {
            "winner_by_aggregate_return": "tie",
            "winner_by_aggregate_objective": "tie",
        },
        {
            "winner_by_aggregate_return": "tie",
            "winner_by_aggregate_objective": "tie",
        },
    ]

    assert module.classify_sidecar_decision(windows) == "break_even_sidecar_no_observed_delta_keep_anchor"


def test_builder_writes_break_even_sidecar_and_latest_alias(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_30 = review_dir / "20260323T041500Z_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.json"
    compare_40 = review_dir / "20260323T042500Z_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.json"

    _write_compare(
        compare_30,
        train_days=30,
        validation_days=10,
        step_days=10,
        validation_window_mode="non_overlapping",
        research_decision="break_even_anchor_no_be_keeps_edge",
        winner_by_aggregate_return="anchor_no_be",
        winner_by_aggregate_objective="anchor_no_be",
        winner_by_slice_majority_return="anchor_no_be",
        winner_by_slice_majority_objective="anchor_no_be",
    )
    _write_compare(
        compare_40,
        train_days=40,
        validation_days=10,
        step_days=10,
        validation_window_mode="non_overlapping",
        research_decision="break_even_forward_profile_mixed_watch_only",
        winner_by_aggregate_return="tie",
        winner_by_aggregate_objective="tie",
        winner_by_slice_majority_return="tie",
        winner_by_slice_majority_objective="tie",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T043000Z",
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

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["evidence_scope"] == "non_overlapping_forward_compare_windows_only"
    assert payload["window_count"] == 2
    assert payload["research_decision"] == "break_even_sidecar_not_promising_keep_anchor"
    assert payload["active_baseline"] == "hold16_trail0_no_be"
    assert payload["watch_candidate"] == "hold16_trail0_be075"
    assert payload["anchor_no_be_windows"] == 1
    assert payload["anchor_with_be_windows"] == 0
    assert payload["tie_windows"] == 1
    assert payload["limitation_note"] == (
        "这是 sidecar 证据，不替代 canonical exit/risk 共识；"
        "即便后续结果继续偏正向，也只能作为 watch sidecar 与主前推证据合并审读，不能直接晋级 anchor。"
    )
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_positive_watch_sidecar_does_not_force_more_tail_extension(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_30 = review_dir / "20260324T044010Z_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.json"
    compare_40 = review_dir / "20260324T044011Z_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.json"

    _write_compare(
        compare_30,
        train_days=30,
        validation_days=10,
        step_days=10,
        validation_window_mode="non_overlapping",
        research_decision="break_even_forward_profile_anchor_with_be_watch_only",
        winner_by_aggregate_return="anchor_with_be",
        winner_by_aggregate_objective="anchor_with_be",
        winner_by_slice_majority_return="tie",
        winner_by_slice_majority_objective="tie",
    )
    _write_compare(
        compare_40,
        train_days=40,
        validation_days=10,
        step_days=10,
        validation_window_mode="non_overlapping",
        research_decision="break_even_forward_profile_anchor_with_be_watch_only",
        winner_by_aggregate_return="anchor_with_be",
        winner_by_aggregate_objective="anchor_with_be",
        winner_by_slice_majority_return="tie",
        winner_by_slice_majority_objective="tie",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T044012Z",
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
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_sidecar_positive_watch_only"
    assert payload["anchor_no_be_windows"] == 0
    assert payload["anchor_with_be_windows"] == 2
    assert payload["tie_windows"] == 0
    assert payload["limitation_note"] == (
        "这是 sidecar 证据，不替代 canonical exit/risk 共识；"
        "即便后续结果继续偏正向，也只能作为 watch sidecar 与主前推证据合并审读，不能直接晋级 anchor。"
    )
    assert "补更多 non-overlapping forward 尾部窗口" not in payload["limitation_note"]


def test_builder_marks_full_overlapping_anchor_with_be_consensus_as_guarded_review_ready(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_paths: list[Path] = []
    for stamp, train_days in [
        ("20260324T153000Z", 30),
        ("20260324T153001Z", 40),
        ("20260324T153002Z", 50),
    ]:
        path = review_dir / f"{stamp}_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.json"
        _write_compare(
            path,
            train_days=train_days,
            validation_days=10,
            step_days=5,
            validation_window_mode="overlapping",
            research_decision="break_even_forward_profile_anchor_with_be_watch_only",
            winner_by_aggregate_return="anchor_with_be",
            winner_by_aggregate_objective="anchor_with_be",
            winner_by_slice_majority_return="anchor_with_be",
            winner_by_slice_majority_objective="anchor_with_be",
        )
        compare_paths.append(path)

    cmd = [
        "python3",
        str(SCRIPT_PATH),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260324T153059Z",
    ]
    for path in compare_paths:
        cmd.extend(["--compare-path", str(path)])

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_sidecar_positive_watch_only"
    assert payload["evidence_scope"] == "overlapping_forward_compare_windows_only"
    assert payload["confidence_tier"] == "guarded_review_ready"
    assert payload["promotion_review_ready"] is True
    assert payload["next_research_priority"] == "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_derives_active_baseline_and_watch_candidate_from_compare_configs(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_30 = review_dir / "20260324T050010Z_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.json"
    _write_compare(
        compare_30,
        train_days=30,
        validation_days=10,
        step_days=10,
        validation_window_mode="non_overlapping",
        research_decision="break_even_forward_profile_anchor_with_be_watch_only",
        winner_by_aggregate_return="anchor_with_be",
        winner_by_aggregate_objective="anchor_with_be",
        winner_by_slice_majority_return="tie",
        winner_by_slice_majority_objective="tie",
    )

    payload = json.loads(compare_30.read_text(encoding="utf-8"))
    payload["comparison_configs"][0]["exit_params"]["trailing_stop_atr"] = 2.0
    payload["comparison_configs"][1]["exit_params"]["trailing_stop_atr"] = 2.0
    payload["comparison_configs"][1]["exit_params"]["break_even_trigger_r"] = 1.0
    compare_30.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T050012Z",
            "--compare-path",
            str(compare_30),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    built = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))

    assert built["active_baseline"] == "hold16_trail20_no_be"
    assert built["watch_candidate"] == "hold16_trail20_be100"
