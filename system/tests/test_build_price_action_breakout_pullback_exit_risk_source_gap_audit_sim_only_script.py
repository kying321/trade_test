from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_builder_detects_consumer_drift_risk_when_handoff_has_not_yet_become_canonical_source_head(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    scripts_dir = system_root / "scripts"
    review_dir = system_root / "output" / "review"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    for script_name in [
        "build_price_action_breakout_pullback_exit_risk_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py",
    ]:
        (scripts_dir / script_name).write_text("# stub\n", encoding="utf-8")

    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 12,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.75,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 12,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_handoff_inconclusive",
            "source_head_status": "pending",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "consumer_rule": (
                "后续所有 ETH exit/risk brief / review / consumer 必须先读取 "
                "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json`；"
                "不得再手工拼 exit_risk + forward_blocker + forward_consensus + sidecar + tail_capacity。"
            ),
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T141500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))

    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "exit_risk_source_gap_detected_consumer_drift_risk"
    assert payload["canonical_handoff_alias"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"
    )
    assert payload["canonical_anchor"] == "hold16_trail0_no_be"
    assert payload["consumer_rule_ok"] is True
    assert payload["finding_count"] == 2
    assert payload["findings"] == [
        {
            "label": "exit_risk_selected_exit_params",
            "issue": "selected_exit_params_differs_from_canonical_handoff_anchor",
            "observed_slug": "hold12_trail0_no_be",
            "canonical_anchor": "hold16_trail0_no_be",
        },
        {
            "label": "forward_blocker_challenge_pair_baseline",
            "issue": "challenge_pair_baseline_differs_from_canonical_handoff_anchor",
            "observed_slug": "hold16_trail0_be075",
            "canonical_anchor": "hold16_trail0_no_be",
        },
    ]



def test_builder_downgrades_superseded_upstream_divergence_when_handoff_is_canonical_source_head(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    scripts_dir = system_root / "scripts"
    review_dir = system_root / "output" / "review"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    for script_name in [
        "build_price_action_breakout_pullback_exit_risk_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py",
        "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py",
    ]:
        (scripts_dir / script_name).write_text("# stub\n", encoding="utf-8")

    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 12,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold12_vs_hold16",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.75,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 12,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "baseline_anchor_active",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "consumer_rule": (
                "后续所有 ETH exit/risk brief / review / consumer 必须先读取 "
                "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json`；"
                "不得再手工拼 exit_risk + forward_blocker + forward_consensus + sidecar + tail_capacity。"
            ),
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T150500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "exit_risk_source_gap_detected_superseded_upstream_divergence_watch_only"
    assert payload["finding_count"] == 2
    assert payload["consumer_rule_ok"] is True
    assert payload["findings"] == [
        {
            "label": "exit_risk_selected_exit_params",
            "issue": "selected_exit_params_superseded_by_canonical_handoff_anchor",
            "observed_slug": "hold12_trail0_no_be",
            "canonical_anchor": "hold16_trail0_no_be",
        },
        {
            "label": "forward_blocker_challenge_pair_baseline",
            "issue": "challenge_pair_baseline_superseded_by_canonical_handoff_anchor",
            "observed_slug": "hold16_trail0_be075",
            "canonical_anchor": "hold16_trail0_no_be",
        },
    ]

def test_builder_marks_chain_consistent_when_all_latest_aliases_match_canonical_handoff(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    scripts_dir = system_root / "scripts"
    review_dir = system_root / "output" / "review"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("latest_price_action_breakout_pullback_exit_risk_sim_only.json", "build_price_action_breakout_pullback_exit_risk_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json", "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json", "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
    ]
    for latest_name, script_name in pairs:
        (scripts_dir / script_name).write_text("# stub\n", encoding="utf-8")

    canonical_anchor = "hold16_trail0_no_be"
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
            },
        },
    )
    _write_json(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json", {})
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json",
        {"active_baseline": canonical_anchor},
    )
    _write_json(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json", {})
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "active_baseline": canonical_anchor,
            "consumer_rule": (
                "后续所有 ETH exit/risk brief / review / consumer 必须先读取 "
                "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json`；"
                "不得再手工拼 exit_risk + forward_blocker + forward_consensus + sidecar + tail_capacity。"
            ),
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T141700Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "exit_risk_source_gap_audit_pass_canonical_handoff_contract_intact"
    assert payload["finding_count"] == 0
    assert payload["consumer_rule_ok"] is True


def test_builder_flags_cross_chain_hold_baseline_mismatch_against_exit_risk_canonical_anchor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    scripts_dir = system_root / "scripts"
    review_dir = system_root / "output" / "review"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("latest_price_action_breakout_pullback_exit_risk_sim_only.json", "build_price_action_breakout_pullback_exit_risk_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json", "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json", "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
        ("latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json", "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py"),
    ]
    for _, script_name in pairs:
        (scripts_dir / script_name).write_text("# stub\n", encoding="utf-8")

    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 24,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 24,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "baseline_anchor_active",
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
            "consumer_rule": (
                "后续所有 ETH exit/risk brief / review / consumer 必须先读取 "
                "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json`；"
                "不得再手工拼 exit_risk + forward_blocker + forward_consensus + sidecar + tail_capacity。"
            ),
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_hold_selection_gate_as_canonical_head",
            "active_baseline": "hold16_zero",
            "consumer_rule": (
                "后续所有 ETH hold selection brief / review / consumer 必须先读取 "
                "`latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json`。"
            ),
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T160100Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "exit_risk_source_gap_detected_consumer_drift_risk"
    assert payload["canonical_anchor"] == "hold24_trail0_no_be"
    assert payload["hold_selection_active_baseline"] == "hold16_zero"
    assert payload["hold_selection_active_hold_bars"] == 16
    assert payload["canonical_anchor_hold_bars"] == 24
    assert payload["findings"] == [
        {
            "label": "hold_selection_handoff_active_baseline",
            "issue": "hold_selection_active_hold_bars_differs_from_exit_risk_canonical_anchor",
            "observed_slug": "hold16_zero",
            "canonical_anchor": "hold24_trail0_no_be",
            "observed_hold_bars": 16,
            "canonical_hold_bars": 24,
        }
    ]


def test_builder_downgrades_hold_selection_conflict_to_watch_only_when_aligned_review_lane_is_ready(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    scripts_dir = system_root / "scripts"
    review_dir = system_root / "output" / "review"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("latest_price_action_breakout_pullback_exit_risk_sim_only.json", "build_price_action_breakout_pullback_exit_risk_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json", "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json", "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json", "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
        ("latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json", "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py"),
        (
            "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json",
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
        ),
    ]
    for _, script_name in pairs:
        (scripts_dir / script_name).write_text("# stub\n", encoding="utf-8")

    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24",
            "challenge_pair": {
                "baseline_exit_params": {
                    "max_hold_bars": 24,
                    "break_even_trigger_r": 0.75,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
                "challenger_exit_params": {
                    "max_hold_bars": 16,
                    "break_even_trigger_r": 0.0,
                    "trailing_stop_atr": 0.0,
                    "cooldown_after_losses": 0,
                    "cooldown_bars": 0,
                },
            },
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict",
            "source_head_status": "upstream_hold_selection_conflict",
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
            "hold_selection_active_baseline": "hold16_zero",
            "hold_selection_active_hold_bars": 16,
            "consumer_rule": (
                "后续所有 ETH exit/risk brief / review / consumer 必须先读取 "
                "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json`；"
                "不得再手工拼 exit_risk + forward_blocker + forward_consensus + sidecar + tail_capacity。"
            ),
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_hold_selection_gate_as_canonical_head",
            "source_head_status": "gate_override_active",
            "active_baseline": "hold16_zero",
        },
    )
    _write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "hold_selection_aligned_break_even_review_lane_ready_but_canonical_handoff_conflict_remains",
            "active_baseline": "hold16_trail0_no_be",
            "preferred_watch_candidate": "hold16_trail0_be075",
            "hold_selection_active_baseline": "hold16_zero",
            "hold_selection_active_hold_bars": 16,
            "review_conclusion_research_decision": "break_even_review_conclusion_ready_keep_baseline_anchor_review_only",
            "review_conclusion_arbitration_state": "review_only",
            "primary_anchor_review_research_decision": "break_even_primary_anchor_review_complete_keep_baseline_anchor",
            "primary_anchor_review_state": "completed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T161500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "exit_risk_source_gap_detected_superseded_upstream_divergence_watch_only"
    assert payload["canonical_anchor"] == "hold24_trail0_no_be"
    assert payload["hold_selection_active_baseline"] == "hold16_zero"
    assert payload["hold_selection_active_hold_bars"] == 16
    assert payload["findings"] == [
        {
            "label": "exit_risk_selected_exit_params",
            "issue": "selected_exit_params_aligned_review_lane_watch_only",
            "observed_slug": "hold16_trail0_no_be",
            "canonical_anchor": "hold24_trail0_no_be",
        },
        {
            "label": "forward_blocker_challenge_pair_baseline",
            "issue": "challenge_pair_baseline_aligned_review_lane_watch_only",
            "observed_slug": "hold24_trail0_be075",
            "canonical_anchor": "hold24_trail0_no_be",
        },
        {
            "label": "hold_selection_handoff_active_baseline",
            "issue": "hold_selection_active_hold_bars_aligned_review_lane_watch_only",
            "observed_slug": "hold16_zero",
            "canonical_anchor": "hold24_trail0_no_be",
            "observed_hold_bars": 16,
            "canonical_hold_bars": 24,
        },
    ]
