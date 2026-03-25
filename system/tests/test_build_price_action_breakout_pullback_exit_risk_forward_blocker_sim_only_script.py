from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fenlie_exit_risk_forward_blocker_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_classify_forward_blocker_requires_hold8_vs_hold16_oos() -> None:
    module = _load_module()

    assert module.classify_forward_blocker_decision(
        exit_risk_decision="selected_exit_risk_improves_but_train_first_validation_diverges",
        hold_forward_stop_decision="stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
        selected_exit_params={
            "max_hold_bars": 8,
            "break_even_trigger_r": 0.75,
            "trailing_stop_atr": 1.5,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
        validation_leader_exit_params={
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 1.5,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    ) == "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos"


def test_derive_baseline_follow_up_priority_upgrades_guarded_review_ready_sidecar() -> None:
    module = _load_module()

    assert module.derive_baseline_follow_up_priority(
        break_even_sidecar_decision="break_even_sidecar_positive_watch_only",
        break_even_sidecar_confidence_tier="guarded_review_ready",
        tail_capacity_decision="exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        fallback="forward_oos_follow_up_pending",
    ) == "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_writes_exit_risk_forward_blocker_and_latest_alias(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260321T130100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "20260323T022500Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "selected_exit_risk_improves_but_train_first_validation_diverges",
            "selected_exit_params": {
                "max_hold_bars": 8,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 1.5,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 1.5,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "baseline_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "recommended_brief": "ETHUSDT:exit_risk:moderate_costs:selected_return=-0.63%,baseline_return=-1.90%,status=not_promising,improves=true,decision=selected_exit_risk_improves_but_train_first_validation_diverges",
        },
    )
    _write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            "next_research_priority": "review_long_window_regime_split_before_baseline_promotion",
            "recommended_brief": "ETHUSDT:exit_hold_forward_stop:capacity_limited=blocked,overlap_35d_40d=watch_only,window_consensus=hold16_consensus_bias_keep_baseline_watch_long_window_regime_split,next=review_long_window_regime_split_before_baseline_promotion,decision=stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T024000Z",
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

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "block_exit_risk_promotion_require_forward_oos_pair_hold8_vs_hold16"
    assert payload["blocked_now"] == [
        "promote_selected_exit_risk_config_without_forward_oos",
        "promote_validation_leader_exit_risk_config_without_forward_oos",
    ]
    assert payload["allowed_now"] == [
        "run_exit_risk_forward_oos_hold8_vs_hold16_under_trailing_1_5",
        "keep_break_even_delta_as_watch_sidecar_until_forward_oos_resolves",
    ]
    assert payload["next_research_priority"] == "exit_risk_forward_oos_hold8_vs_hold16"
    assert payload["challenge_pair"] == {
        "baseline_exit_params": {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 1.5,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
        "challenger_exit_params": {
            "max_hold_bars": 8,
            "break_even_trigger_r": 0.75,
            "trailing_stop_atr": 1.5,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
        "primary_axis": "max_hold_bars",
        "baseline_hold_bars": 16,
        "challenger_hold_bars": 8,
        "shared_trailing_stop_atr": 1.5,
    }
    assert payload["recommended_brief"] == (
        "ETHUSDT:exit_risk_forward_blocker:pair=hold8_vs_hold16,trail=1.5,"
        "next=exit_risk_forward_oos_hold8_vs_hold16,"
        "decision=block_exit_risk_promotion_require_forward_oos_pair_hold8_vs_hold16"
    )
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_clears_blocker_when_forward_consensus_promotes_challenger(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260321T130100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "20260323T022500Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    forward_consensus_path = review_dir / "20260323T200300Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "selected_exit_risk_improves_but_train_first_validation_diverges",
            "selected_exit_params": {
                "max_hold_bars": 8,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 1.5,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 1.5,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            "next_research_priority": "review_long_window_regime_split_before_baseline_promotion",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "challenger_pair_promotable_across_current_forward_oos",
            "allowed_now": [
                "promote_challenger_pair_as_new_exit_risk_anchor",
                "treat_55d_plus_tie_windows_as_watch_only",
            ],
            "next_research_priority": "refresh_exit_risk_anchor_after_forward_oos_promotion",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T201000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "exit_risk_forward_blocker_cleared_promote_challenger_pair"
    assert payload["blocked_now"] == []
    assert payload["allowed_now"] == [
        "promote_challenger_pair_as_new_exit_risk_anchor",
        "treat_55d_plus_tie_windows_as_watch_only",
    ]
    assert payload["next_research_priority"] == "refresh_exit_risk_anchor_after_forward_oos_promotion"
    assert payload["challenge_pair"]["challenger_hold_bars"] == 8


def test_builder_keeps_baseline_anchor_when_forward_consensus_confirms_current_pair(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T041100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "20260324T041022Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    forward_consensus_path = review_dir / "20260324T041008Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "validation_leader_improves_train_first_selected_not_promoted",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 24,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            "next_research_priority": "review_long_window_regime_split_before_baseline_promotion",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "baseline_windows": 6,
            "tie_windows": 0,
            "challenger_windows": 0,
            "next_research_priority": "forward_oos_follow_up_pending",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T041009Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24"
    assert payload["blocked_now"] == ["promote_challenger_pair_as_new_exit_risk_anchor"]
    assert payload["allowed_now"] == ["keep_baseline_pair_as_current_exit_risk_anchor"]
    assert payload["next_research_priority"] == "forward_oos_follow_up_pending"


def test_builder_derives_specific_follow_up_when_baseline_retained_and_sidecar_positive(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T045100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "20260324T045122Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    forward_consensus_path = review_dir / "20260324T045108Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    break_even_sidecar_path = review_dir / "20260324T045112Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260324T045113Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "validation_leader_improves_train_first_selected_not_promoted",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 24,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 0.0,
            },
        },
    )
    _write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            "next_research_priority": "review_long_window_regime_split_before_baseline_promotion",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "baseline_windows": 6,
            "tie_windows": 0,
            "challenger_windows": 0,
            "next_research_priority": "forward_oos_follow_up_pending",
        },
    )
    _write_json(
        break_even_sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
        },
    )
    _write_json(
        tail_capacity_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--break-even-sidecar-path",
            str(break_even_sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T045114Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24"
    assert payload["next_research_priority"] == "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"
    assert payload["recommended_brief"] == (
        "ETHUSDT:exit_risk_forward_blocker:pair=hold16_vs_hold24,trail=0.0,"
        "next=watch_break_even_candidate_keep_baseline_anchor_defer_more_tail,"
        "decision=block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24"
    )


def test_builder_formats_recommended_brief_from_actual_challenge_pair_when_blocker_is_inconclusive(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T021100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "20260324T021218Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    forward_consensus_path = review_dir / "20260324T021208Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "validation_leader_improves_train_first_selected_not_promoted",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 24,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "hold_forward_evidence_stop_condition_inconclusive",
            "next_research_priority": "hold_forward_compare_review_pending",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "next_research_priority": "forward_oos_follow_up_pending",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T021209Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "exit_risk_forward_blocker_inconclusive"
    assert payload["challenge_pair"]["baseline_hold_bars"] == 24
    assert payload["challenge_pair"]["challenger_hold_bars"] == 16
    assert payload["recommended_brief"] == (
        "ETHUSDT:exit_risk_forward_blocker:pair=hold16_vs_hold24,trail=0.0,"
        "next=hold_forward_compare_review_pending,"
        "decision=exit_risk_forward_blocker_inconclusive"
    )


def test_builder_upgrades_baseline_follow_up_to_guarded_review_when_sidecar_is_review_ready(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T161100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "20260324T161218Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    forward_consensus_path = review_dir / "20260324T161208Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    break_even_sidecar_path = review_dir / "20260324T161210Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260324T161211Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "validation_leader_improves_train_first_selected_not_promoted",
            "selected_exit_params": {
                "max_hold_bars": 12,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            "next_research_priority": "review_long_window_regime_split_before_baseline_promotion",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "baseline_windows": 6,
            "tie_windows": 0,
            "challenger_windows": 0,
            "next_research_priority": "forward_oos_follow_up_pending",
        },
    )
    _write_json(
        break_even_sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )
    _write_json(
        tail_capacity_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--break-even-sidecar-path",
            str(break_even_sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T161219Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "block_exit_risk_promotion_keep_baseline_anchor_pair_hold12_vs_hold16"
    assert payload["next_research_priority"] == "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_formats_dynamic_forward_oos_action_and_priority_from_actual_blocked_pair(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T031100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "20260324T031218Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "validation_leader_improves_train_first_selected_not_promoted",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 24,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            "next_research_priority": "review_long_window_regime_split_before_baseline_promotion",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T031219Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "block_exit_risk_promotion_require_forward_oos_pair_hold16_vs_hold24"
    assert payload["allowed_now"] == [
        "run_exit_risk_forward_oos_hold16_vs_hold24_under_trailing_0_0",
        "keep_break_even_delta_as_watch_sidecar_until_forward_oos_resolves",
    ]
    assert payload["next_research_priority"] == "exit_risk_forward_oos_hold16_vs_hold24"
