from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fenlie_exit_hold_forward_evidence_stop_condition_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_classify_stop_condition_locks_45d_plus_and_shifts_priority() -> None:
    module = _load_module()

    assert module.classify_stop_condition_decision(
        forward_capacity_decision="non_overlapping_forward_capacity_supports_40d_watch_45d_plus_insufficient",
        overlap_sidecar_decision="overlapping_long_window_split_keep_non_overlapping_baseline_watch_hold8",
        handoff_decision="use_hold_selection_gate_as_canonical_head",
        window_consensus_decision="hold16_consensus_bias_keep_baseline",
    ) == "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos"


def test_classify_stop_condition_prefers_long_window_regime_split_review_when_consensus_already_exists() -> None:
    module = _load_module()

    assert module.classify_stop_condition_decision(
        forward_capacity_decision="non_overlapping_forward_capacity_supports_40d_watch_45d_plus_insufficient",
        overlap_sidecar_decision="overlapping_long_window_split_keep_non_overlapping_baseline_watch_hold8",
        handoff_decision="use_hold_selection_gate_as_canonical_head",
        window_consensus_decision="hold16_consensus_bias_keep_baseline_watch_long_window_regime_split",
    ) == "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split"


def test_classify_stop_condition_prefers_long_window_review_for_current_latest_source_owned_decisions() -> None:
    module = _load_module()

    assert module.classify_stop_condition_decision(
        forward_capacity_decision="non_overlapping_forward_capacity_limited_but_usable",
        overlap_sidecar_decision="overlapping_long_window_hold16_bias_keep_non_overlapping_baseline",
        handoff_decision="use_hold_selection_gate_as_canonical_head",
        window_consensus_decision="hold16_consensus_bias_keep_baseline_watch_long_window_regime_split",
    ) == "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split"


def test_builder_writes_stop_condition_artifact_and_latest_alias(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    forward_capacity_path = review_dir / "20260323T014500Z_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json"
    overlap_sidecar_path = review_dir / "20260323T013800Z_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json"
    handoff_path = review_dir / "20260323T014300Z_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
    window_consensus_path = review_dir / "20260323T014100Z_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.json"

    _write_json(
        forward_capacity_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "non_overlapping_forward_capacity_limited_but_usable",
            "recommended_brief": "ETHUSDT:exit_hold_forward_capacity:non_overlapping:max_credible=60d,max_robust=60d,first_insufficient=0d,decision=non_overlapping_forward_capacity_limited_but_usable",
        },
    )
    _write_json(
        overlap_sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "overlapping_long_window_hold16_bias_keep_non_overlapping_baseline",
            "consumer_rule": "do_not_override_non_overlapping_consensus_without_fresh_non_overlapping_confirmation",
        },
    )
    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_hold_selection_gate_as_canonical_head",
            "consumer_rule": "后续 consumer 必须先读 hold_selection_handoff，并同时尊重 40d credible ceiling 与 overlap watch-only 边界。",
        },
    )
    _write_json(
        window_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "hold16_consensus_bias_keep_baseline_watch_long_window_regime_split",
            "recommended_brief": "ETHUSDT:exit_hold_window_consensus:non_overlapping:vote_hold8=6,vote_hold16=10,decision=hold16_consensus_bias_keep_baseline_watch_long_window_regime_split",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--forward-capacity-path",
            str(forward_capacity_path),
            "--overlap-sidecar-path",
            str(overlap_sidecar_path),
            "--handoff-path",
            str(handoff_path),
            "--window-consensus-path",
            str(window_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T020500Z",
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

    assert payload["action"] == "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["research_decision"] == (
        "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split"
    )
    assert payload["source_evidence"] == {
        "forward_capacity_research_decision": "non_overlapping_forward_capacity_limited_but_usable",
        "overlap_sidecar_research_decision": "overlapping_long_window_hold16_bias_keep_non_overlapping_baseline",
        "handoff_research_decision": "use_hold_selection_gate_as_canonical_head",
        "window_consensus_research_decision": "hold16_consensus_bias_keep_baseline_watch_long_window_regime_split",
    }
    assert payload["blocked_now"] == [
        "use_capacity_limited_non_overlap_forward_compare_as_main_evidence",
        "extend_hold_forward_train_window_beyond_40d_without_fresh_non_overlap_capacity",
        "promote_hold8_over_hold16_without_long_window_regime_split_review",
    ]
    assert payload["watch_only"] == [
        "treat_35d_40d_overlapping_compare_as_watch_sidecar_only",
        "treat_long_window_regime_split_as_review_gate_only",
    ]
    assert payload["next_research_priority"] == "review_long_window_regime_split_before_baseline_promotion"
    assert payload["recommended_brief"] == (
        "ETHUSDT:exit_hold_forward_stop:capacity_limited=blocked,overlap_35d_40d=watch_only,"
        "window_consensus=hold16_consensus_bias_keep_baseline_watch_long_window_regime_split,"
        "next=review_long_window_regime_split_before_baseline_promotion,"
        "decision=stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split"
    )
    assert "45d_plus" not in payload["recommended_brief"]
    assert "45d_plus" not in payload["consumer_rule"]
    assert "45d_plus" not in payload["research_note"]
    assert latest_payload["research_decision"] == payload["research_decision"]
