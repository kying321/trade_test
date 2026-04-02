from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fenlie_exit_risk_handoff_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_format_exit_risk_anchor_slug() -> None:
    module = _load_module()

    assert module.format_exit_risk_anchor_slug(
        {
            "max_hold_bars": 8,
            "break_even_trigger_r": 0.75,
            "trailing_stop_atr": 1.5,
        }
    ) == "hold8_trail15_be075"
    assert module.format_exit_risk_anchor_slug(
        {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 1.5,
        }
    ) == "hold16_trail15_no_be"


def test_derive_baseline_follow_up_priority_accepts_fully_covered_tail_capacity() -> None:
    module = _load_module()

    assert module.derive_baseline_follow_up_priority(
        break_even_sidecar_decision="break_even_sidecar_positive_watch_only",
        break_even_sidecar_confidence_tier="watch_only",
        tail_capacity_decision="exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        fallback="forward_oos_follow_up_pending",
    ) == "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_derive_baseline_follow_up_priority_upgrades_guarded_review_ready_sidecar() -> None:
    module = _load_module()

    assert module.derive_baseline_follow_up_priority(
        break_even_sidecar_decision="break_even_sidecar_positive_watch_only",
        break_even_sidecar_confidence_tier="guarded_review_ready",
        tail_capacity_decision="exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        fallback="forward_oos_follow_up_pending",
    ) == "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_promotes_challenger_pair_as_canonical_anchor(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260321T130100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker_path = review_dir / "20260323T201000Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus_path = review_dir / "20260323T200300Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar_path = review_dir / "20260323T040500Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260323T050500Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

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
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_blocker_cleared_promote_challenger_pair",
            "allowed_now": [
                "promote_challenger_pair_as_new_exit_risk_anchor",
                "treat_55d_plus_tie_windows_as_watch_only",
            ],
            "blocked_now": [],
            "next_research_priority": "refresh_exit_risk_anchor_after_forward_oos_promotion",
            "challenge_pair": {
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
            },
        },
    )
    _write_json(
        consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "challenger_pair_promotable_across_current_forward_oos",
            "baseline_windows": 0,
            "tie_windows": 2,
            "challenger_windows": 4,
            "allowed_now": [
                "promote_challenger_pair_as_new_exit_risk_anchor",
                "treat_55d_plus_tie_windows_as_watch_only",
            ],
            "blocked_now": [],
            "next_research_priority": "refresh_exit_risk_anchor_after_forward_oos_promotion",
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_no_observed_delta_keep_anchor",
            "active_baseline": "hold16_trail15_no_be",
        },
    )
    _write_json(
        tail_capacity_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset",
            "max_credible_train_days": 40,
            "first_insufficient_train_days": 45,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--forward-consensus-path",
            str(consensus_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T202500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    json_path = Path(output["json_path"])
    latest_json_path = Path(output["latest_json_path"])
    md_path = Path(output["md_path"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_handoff_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "use_exit_risk_handoff_as_canonical_anchor"
    assert payload["source_head_status"] == "challenger_anchor_active"
    assert payload["active_baseline"] == "hold8_trail15_be075"
    assert payload["superseded_anchor"] == "hold16_trail15_no_be"
    assert payload["transfer_watch"] == ["55d_plus_tie_windows"]
    assert payload["baseline_windows"] == 0
    assert payload["tie_windows"] == 2
    assert payload["challenger_windows"] == 4
    assert payload["blocked_now"] == []
    assert payload["allowed_now"] == [
        "promote_challenger_pair_as_new_exit_risk_anchor",
        "treat_55d_plus_tie_windows_as_watch_only",
    ]
    assert payload["next_research_priority"] == "refresh_exit_risk_anchor_after_forward_oos_promotion"
    assert payload["consumer_rule"].startswith("后续所有 ETH exit/risk brief / review / consumer 必须先读取")
    assert payload["source_evidence"] == {
        "exit_risk_research_decision": "selected_exit_risk_improves_but_train_first_validation_diverges",
        "forward_blocker_research_decision": "exit_risk_forward_blocker_cleared_promote_challenger_pair",
        "forward_consensus_research_decision": "challenger_pair_promotable_across_current_forward_oos",
        "break_even_sidecar_research_decision": "break_even_sidecar_no_observed_delta_keep_anchor",
        "break_even_sidecar_confidence_tier": "",
        "tail_capacity_research_decision": "exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset",
        "hold_selection_research_decision": "",
        "upstream_hold_alignment_state": "hold_selection_unavailable",
    }
    assert latest_payload["research_decision"] == payload["research_decision"]
    assert "challenger_anchor_active" in md_path.read_text(encoding="utf-8")


def test_builder_marks_inconclusive_when_promotion_not_cleared(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260321T130100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker_path = review_dir / "20260323T024000Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus_path = review_dir / "20260323T034500Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar_path = review_dir / "20260323T040500Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260323T050500Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 8,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 1.5,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 1.5,
            },
        },
    )
    _write_json(
        blocker_path,
        {
            "research_decision": "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos",
            "allowed_now": ["run_exit_risk_forward_oos_hold8_vs_hold16_under_trailing_1_5"],
            "blocked_now": ["promote_selected_exit_risk_config_without_forward_oos"],
            "challenge_pair": {
                "baseline_exit_params": {"max_hold_bars": 16, "break_even_trigger_r": 0.0, "trailing_stop_atr": 1.5},
                "challenger_exit_params": {"max_hold_bars": 8, "break_even_trigger_r": 0.75, "trailing_stop_atr": 1.5},
            },
        },
    )
    _write_json(
        consensus_path,
        {
            "research_decision": "baseline_pair_keeps_anchor_challenger_not_promoted_across_30d_40d_forward_oos",
            "baseline_windows": 1,
            "tie_windows": 1,
            "challenger_windows": 0,
            "blocked_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
            "allowed_now": ["keep_baseline_pair_as_current_exit_risk_anchor"],
        },
    )
    _write_json(sidecar_path, {"active_baseline": "hold16_trail15_no_be"})
    _write_json(tail_capacity_path, {"research_decision": "exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset"})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--forward-consensus-path",
            str(consensus_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T203000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "exit_risk_handoff_inconclusive"
    assert payload["source_head_status"] == "inconclusive"
    assert payload["active_baseline"] == "hold16_trail15_no_be"
    assert payload["superseded_anchor"] == ""
    assert payload["transfer_watch"] == []
    assert payload["next_research_priority"] == "exit_risk_anchor_pending_refresh"


def test_builder_keeps_baseline_pair_as_canonical_anchor_when_forward_oos_confirms_current_head(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T041100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker_path = review_dir / "20260324T041009Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus_path = review_dir / "20260324T041008Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar_path = review_dir / "20260324T041012Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260324T041013Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

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
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24",
            "allowed_now": ["keep_baseline_pair_as_current_exit_risk_anchor"],
            "blocked_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
            "next_research_priority": "forward_oos_follow_up_pending",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 24,
                    "break_even_trigger_r": 0.75,
                    "trailing_stop_atr": 0.0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                },
            },
        },
    )
    _write_json(
        consensus_path,
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
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )
    _write_json(
        tail_capacity_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_limited_watch_only",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--forward-consensus-path",
            str(consensus_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T041014Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "use_exit_risk_handoff_as_canonical_anchor"
    assert payload["source_head_status"] == "baseline_anchor_active"
    assert payload["active_baseline"] == "hold16_trail0_no_be"
    assert payload["watch_candidate"] == "hold16_trail0_be075"
    assert payload["superseded_anchor"] == ""
    assert payload["blocked_now"] == ["promote_challenger_pair_as_new_exit_risk_anchor"]
    assert payload["allowed_now"] == ["keep_baseline_pair_as_current_exit_risk_anchor"]
    assert payload["next_research_priority"] == "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"
    assert payload["handoff_state"]["anchor_state"] == "baseline_retained"
    assert payload["recommended_brief"] == (
        "ETHUSDT:exit_risk_handoff:anchor=hold16_trail0_no_be,"
        "superseded=none,"
        "watch=hold16_trail0_be075,"
        "decision=use_exit_risk_handoff_as_canonical_anchor"
    )


def test_builder_baseline_retained_falls_back_to_blocker_baseline_when_sidecar_anchor_missing(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T151100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker_path = review_dir / "20260324T151109Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus_path = review_dir / "20260324T151108Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar_path = review_dir / "20260324T151112Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260324T151113Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "validation_leader_improves_train_first_selected_not_promoted",
            "selected_exit_params": {
                "max_hold_bars": 12,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 0.0,
            },
        },
    )
    _write_json(
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold12_vs_hold16",
            "allowed_now": ["keep_baseline_pair_as_current_exit_risk_anchor"],
            "blocked_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
            "next_research_priority": "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.75,
                    "trailing_stop_atr": 0.0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 12,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                },
            },
        },
    )
    _write_json(
        consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "baseline_windows": 4,
            "tie_windows": 0,
            "challenger_windows": 0,
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "watch_candidate": "hold16_trail0_be075",
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
            "--forward-blocker-path",
            str(blocker_path),
            "--forward-consensus-path",
            str(consensus_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T151114Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "use_exit_risk_handoff_as_canonical_anchor"
    assert payload["source_head_status"] == "baseline_anchor_active"
    assert payload["active_baseline"] == "hold16_trail0_be075"
    assert payload["watch_candidate"] == "hold16_trail0_be075"


def test_builder_baseline_retained_upgrades_next_priority_when_sidecar_is_guarded_review_ready(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T162100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker_path = review_dir / "20260324T162109Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus_path = review_dir / "20260324T162108Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar_path = review_dir / "20260324T162112Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260324T162113Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

    _write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "validation_leader_improves_train_first_selected_not_promoted",
            "selected_exit_params": {
                "max_hold_bars": 12,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 0.0,
            },
        },
    )
    _write_json(
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold12_vs_hold16",
            "allowed_now": ["keep_baseline_pair_as_current_exit_risk_anchor"],
            "blocked_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
            "next_research_priority": "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.75,
                    "trailing_stop_atr": 0.0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 12,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                },
            },
        },
    )
    _write_json(
        consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "baseline_windows": 4,
            "tie_windows": 0,
            "challenger_windows": 0,
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "active_baseline": "hold16_trail0_be075",
            "watch_candidate": "hold16_trail0_be075",
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
            "--forward-blocker-path",
            str(blocker_path),
            "--forward-consensus-path",
            str(consensus_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T162114Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "use_exit_risk_handoff_as_canonical_anchor"
    assert payload["source_head_status"] == "baseline_anchor_active"
    assert payload["next_research_priority"] == "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_blocks_canonical_exit_risk_handoff_when_hold_selection_upstream_anchor_conflicts(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T163100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker_path = review_dir / "20260324T163109Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus_path = review_dir / "20260324T163108Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar_path = review_dir / "20260324T163112Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260324T163113Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

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
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24",
            "allowed_now": ["keep_baseline_pair_as_current_exit_risk_anchor"],
            "blocked_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 24,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                },
            },
        },
    )
    _write_json(
        consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "baseline_windows": 6,
            "tie_windows": 0,
            "challenger_windows": 0,
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )
    _write_json(
        tail_capacity_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_hold_selection_gate_as_canonical_head",
            "active_baseline": "hold16_zero",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--forward-consensus-path",
            str(consensus_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T163114Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
    assert payload["source_head_status"] == "upstream_hold_selection_conflict"
    assert payload["active_baseline"] == "hold24_trail0_no_be"
    assert payload["watch_candidate"] == "hold24_trail0_be075"
    assert payload["hold_selection_active_baseline"] == "hold16_zero"
    assert payload["hold_selection_active_hold_bars"] == 16
    assert payload["active_baseline_hold_bars"] == 24
    assert "keep_hold_selection_anchor_as_upstream_mainline_gate" in payload["allowed_now"]
    assert "promote_exit_risk_anchor_that_conflicts_with_hold_selection_baseline" in payload["blocked_now"]
    assert payload["next_research_priority"] == "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"
    assert payload["handoff_state"]["anchor_state"] == "blocked_by_upstream_hold_selection"


def test_builder_exposes_aligned_review_lane_bridge_when_upstream_conflict_has_review_only_path(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = review_dir / "20260324T170100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker_path = review_dir / "20260324T170109Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus_path = review_dir / "20260324T170108Z_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar_path = review_dir / "20260324T170112Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity_path = review_dir / "20260324T170113Z_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

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
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24",
            "allowed_now": ["keep_baseline_pair_as_current_exit_risk_anchor"],
            "blocked_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 24,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                },
            },
        },
    )
    _write_json(
        consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "baseline_windows": 6,
            "tie_windows": 0,
            "challenger_windows": 0,
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )
    _write_json(
        tail_capacity_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_hold_selection_gate_as_canonical_head",
            "active_baseline": "hold16_zero",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "hold_selection_aligned_break_even_review_lane_ready_but_canonical_handoff_conflict_remains",
            "canonical_handoff_research_decision": "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict",
            "canonical_handoff_source_head_status": "upstream_hold_selection_conflict",
            "active_baseline": "hold16_trail0_no_be",
            "preferred_watch_candidate": "hold16_trail0_be075",
            "hold_selection_active_baseline": "hold16_zero",
            "review_conclusion_research_decision": "break_even_review_conclusion_ready_keep_baseline_anchor_review_only",
            "review_conclusion_arbitration_state": "review_only",
            "primary_anchor_review_research_decision": "break_even_primary_anchor_review_complete_keep_baseline_anchor",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--exit-risk-path",
            str(exit_risk_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--forward-consensus-path",
            str(consensus_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T170114Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
    assert payload["source_head_status"] == "upstream_hold_selection_conflict"
    assert payload["upstream_conflict_review_only_state"] == "ready"
    assert payload["aligned_review_lane_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json"
    )
    assert payload["aligned_review_lane_research_decision"] == (
        "hold_selection_aligned_break_even_review_lane_ready_but_canonical_handoff_conflict_remains"
    )
    assert payload["aligned_review_lane_active_baseline"] == "hold16_trail0_no_be"
    assert payload["aligned_review_lane_preferred_watch_candidate"] == "hold16_trail0_be075"
    assert payload["aligned_review_lane_review_conclusion_research_decision"] == (
        "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
    )
    assert payload["aligned_review_lane_primary_anchor_review_research_decision"] == (
        "break_even_primary_anchor_review_complete_keep_baseline_anchor"
    )
    assert "consume_hold_selection_aligned_break_even_review_lane_as_review_only_evidence" in payload["allowed_now"]
    assert payload["handoff_state"]["upstream_conflict_review_only_state"] == "ready"
