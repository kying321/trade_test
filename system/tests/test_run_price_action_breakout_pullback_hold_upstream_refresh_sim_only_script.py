from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_price_action_breakout_pullback_hold_upstream_refresh_sim_only.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_hold_upstream_refresh_script",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_main_runs_full_hold_upstream_refresh_chain(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    derivation_dataset_path = review_dir / "20260323T180000Z_public_intraday_crypto_bars_dataset.csv"
    long_dataset_path = review_dir / "20260321T114900Z_public_intraday_crypto_bars_dataset.csv"
    base_artifact_path = review_dir / "20260323T181000Z_price_action_breakout_pullback_sim_only.json"
    hold_robustness_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json"
    rider_triage_path = review_dir / "latest_price_action_breakout_pullback_exit_rider_triage_sim_only.json"
    latest_family_transfer_head = review_dir / "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json"

    write_text(derivation_dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_text(long_dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_json(
        base_artifact_path,
        {
            "focus_symbol": "ETHUSDT",
            "dataset_path": str(derivation_dataset_path),
            "selected_params": {"breakout_lookback": 40},
        },
    )
    write_json(hold_robustness_path, {"research_decision": "mixed_robustness_keep_hold16_baseline_hold8_candidate"})
    write_json(rider_triage_path, {"research_decision": "simple_rider_triage_inconclusive"})
    write_json(
        latest_family_transfer_head,
        {
            "research_decision": "historical_transfer_revives_hold12_and_demotes_hold24_future_tail_insufficient",
            "long_dataset_path": str(long_dataset_path),
        },
    )

    seen_calls: list[tuple[str, list[str]]] = []

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_calls.append((name, list(cmd)))
        stamp = cmd[cmd.index("--stamp") + 1] if "--stamp" in cmd else "from_now"
        if name == "build_hold_family_triage":
            return {
                "json_path": str(review_dir / "hold_family_triage.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_family_triage_sim_only.json"),
                "research_decision": "hold_family_triage_inconclusive",
            }
        if name == "build_hold_frontier_report":
            return {
                "json_path": str(review_dir / "hold_frontier_report.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_frontier_report_sim_only.json"),
                "research_decision": "freeze_hold16_baseline_with_dual_candidates_hold8_objective_hold24_return",
            }
        if name == "build_hold_frontier_cost_sensitivity":
            return {
                "json_path": str(review_dir / "hold_frontier_cost.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json"),
                "research_decision": "frontier_cost_sensitivity_mixed",
            }
        if name == "build_hold_router_hypothesis":
            return {
                "json_path": str(review_dir / "hold_router_hypothesis.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_router_hypothesis_sim_only.json"),
                "research_decision": "no_clean_price_state_router_hypothesis",
            }
        if name == "build_hold_router_transfer":
            return {
                "json_path": str(review_dir / "hold_router_transfer.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json"),
                "research_decision": "router_transfer_failed_keep_pure_hold_selection",
            }
        if name == "build_hold_family_transfer":
            return {
                "json_path": str(review_dir / "hold_family_transfer.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json"),
                "research_decision": "historical_transfer_revives_hold12_and_demotes_hold24_future_tail_insufficient",
            }
        if name == "build_exit_hold_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            step_days = cmd[cmd.index("--step-days") + 1]
            return {
                "json_path": str(review_dir / f"{stamp}_hold_compare_t{train_days}_s{step_days}.json"),
                "research_decision": f"hold_compare_{train_days}_{step_days}",
            }
        if name == "build_exit_hold_window_consensus":
            return {
                "json_path": str(review_dir / "hold_window_consensus.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.json"),
                "research_decision": "hold16_consensus_bias_keep_baseline_watch_long_window_regime_split",
            }
        if name == "build_exit_hold_overlap_sidecar":
            return {
                "json_path": str(review_dir / "hold_overlap_sidecar.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json"),
                "research_decision": "overlapping_long_window_split_keep_non_overlapping_baseline_watch_hold8",
            }
        if name == "build_exit_hold_forward_window_capacity":
            return {
                "json_path": str(review_dir / "hold_forward_capacity.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json"),
                "research_decision": "non_overlapping_forward_capacity_supports_40d_watch_45d_plus_insufficient",
            }
        if name == "build_hold_selection_gate_blocker":
            return {
                "json_path": str(review_dir / "hold_gate_blocker.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.json"),
                "research_decision": "block_hold_candidate_promotion_keep_hold16_anchor_reopen_hold12_watch_demote_hold24_and_router",
            }
        if name == "build_hold_selection_handoff":
            return {
                "json_path": str(review_dir / "hold_selection_handoff.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"),
                "research_decision": "use_hold_selection_gate_as_canonical_head",
                "source_head_status": "gate_override_active",
            }
        if name == "build_exit_hold_forward_stop":
            return {
                "json_path": str(review_dir / "hold_forward_stop.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"),
                "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            }
        if name == "build_hold_upstream_source_gap_audit":
            return {
                "json_path": str(review_dir / "hold_upstream_source_gap_audit.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only.json"),
                "research_decision": "hold_upstream_builder_sources_present_for_current_latest_artifacts",
            }
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_hold_upstream_refresh_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-24T00:10:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["mode"] == "hold_upstream_refresh_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["stamp"] == "20260324T001000Z"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["derivation_dataset_path"] == str(derivation_dataset_path)
    assert payload["long_dataset_path"] == str(long_dataset_path)
    assert payload["base_artifact_path"] == str(base_artifact_path)
    assert payload["hold_robustness_path"] == str(hold_robustness_path)
    assert payload["rider_triage_path"] == str(rider_triage_path)
    assert payload["hold_family_triage_path"] == str(review_dir / "hold_family_triage.json")
    assert payload["frontier_report_path"] == str(review_dir / "hold_frontier_report.json")
    assert payload["frontier_cost_path"] == str(review_dir / "hold_frontier_cost.json")
    assert payload["router_hypothesis_path"] == str(review_dir / "hold_router_hypothesis.json")
    assert payload["router_transfer_path"] == str(review_dir / "hold_router_transfer.json")
    assert payload["family_transfer_path"] == str(review_dir / "hold_family_transfer.json")
    assert payload["window_consensus_path"] == str(review_dir / "hold_window_consensus.json")
    assert payload["forward_capacity_path"] == str(review_dir / "hold_forward_capacity.json")
    assert payload["overlap_sidecar_path"] == str(review_dir / "hold_overlap_sidecar.json")
    assert payload["gate_blocker_path"] == str(review_dir / "hold_gate_blocker.json")
    assert payload["handoff_path"] == str(review_dir / "hold_selection_handoff.json")
    assert payload["stop_condition_path"] == str(review_dir / "hold_forward_stop.json")
    assert payload["source_gap_audit_path"] == str(review_dir / "hold_upstream_source_gap_audit.json")
    assert payload["research_decision"] == "use_hold_selection_gate_as_canonical_head"
    assert payload["stop_condition_research_decision"] == (
        "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split"
    )
    assert payload["source_gap_audit_research_decision"] == (
        "hold_upstream_builder_sources_present_for_current_latest_artifacts"
    )
    assert payload["non_overlap_train_days"] == [20, 25, 30, 35, 40, 45, 50, 55, 60]
    assert payload["overlap_train_days"] == [35, 40]

    call_names = [name for name, _ in seen_calls]
    assert call_names == [
        "build_hold_family_triage",
        "build_hold_frontier_report",
        "build_hold_frontier_cost_sensitivity",
        "build_hold_router_hypothesis",
        "build_hold_router_transfer",
        "build_hold_family_transfer",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_window_consensus",
        "build_exit_hold_forward_compare",
        "build_exit_hold_forward_compare",
        "build_exit_hold_overlap_sidecar",
        "build_exit_hold_forward_window_capacity",
        "build_hold_selection_gate_blocker",
        "build_hold_selection_handoff",
        "build_exit_hold_forward_stop",
        "build_hold_upstream_source_gap_audit",
    ]

    first_cmd = seen_calls[0][1]
    assert first_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_family_triage_sim_only.py"),
        "--dataset-path",
        str(derivation_dataset_path),
        "--base-artifact-path",
        str(base_artifact_path),
        "--review-dir",
        str(review_dir),
        "--symbol",
        "ETHUSDT",
        "--stamp",
        "20260324T001000Z",
    ]

    gate_cmd = seen_calls[20][1]
    assert gate_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.py"),
        "--frontier-report-path",
        str(review_dir / "hold_frontier_report.json"),
        "--frontier-cost-path",
        str(review_dir / "hold_frontier_cost.json"),
        "--router-hypothesis-path",
        str(review_dir / "hold_router_hypothesis.json"),
        "--router-transfer-path",
        str(review_dir / "hold_router_transfer.json"),
        "--family-transfer-path",
        str(review_dir / "hold_family_transfer.json"),
        "--window-consensus-path",
        str(review_dir / "hold_window_consensus.json"),
        "--forward-capacity-path",
        str(review_dir / "hold_forward_capacity.json"),
        "--overlap-sidecar-path",
        str(review_dir / "hold_overlap_sidecar.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260324T001020Z",
    ]

    stop_cmd = seen_calls[22][1]
    assert stop_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py"),
        "--forward-capacity-path",
        str(review_dir / "hold_forward_capacity.json"),
        "--overlap-sidecar-path",
        str(review_dir / "hold_overlap_sidecar.json"),
        "--handoff-path",
        str(review_dir / "hold_selection_handoff.json"),
        "--window-consensus-path",
        str(review_dir / "hold_window_consensus.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260324T001022Z",
    ]


def test_main_fails_when_required_hold_robustness_latest_is_missing(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    derivation_dataset_path = review_dir / "20260323T180000Z_public_intraday_crypto_bars_dataset.csv"
    base_artifact_path = review_dir / "20260323T181000Z_price_action_breakout_pullback_sim_only.json"
    rider_triage_path = review_dir / "latest_price_action_breakout_pullback_exit_rider_triage_sim_only.json"
    latest_family_transfer_head = review_dir / "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json"

    write_text(derivation_dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_json(
        base_artifact_path,
        {
            "focus_symbol": "ETHUSDT",
            "dataset_path": str(derivation_dataset_path),
            "selected_params": {"breakout_lookback": 40},
        },
    )
    write_json(rider_triage_path, {"research_decision": "simple_rider_triage_inconclusive"})
    write_json(latest_family_transfer_head, {"long_dataset_path": str(derivation_dataset_path)})

    with pytest.raises(FileNotFoundError, match="missing_exit_hold_robustness_latest"):
        mod.latest_review_artifact(
            review_dir,
            "latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json",
            "missing_exit_hold_robustness_latest",
        )


def test_resolve_derivation_dataset_path_prefers_latest_dataset_alias_over_timestamp_glob(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    latest_alias = review_dir / "latest_public_intraday_crypto_bars_dataset.csv"
    newer_timestamp_dataset = review_dir / "99999999T999999Z_public_intraday_crypto_bars_dataset.csv"
    write_text(latest_alias, "ts,symbol,open,high,low,close,volume\n")
    write_text(newer_timestamp_dataset, "ts,symbol,open,high,low,close,volume\n")

    selected = mod.resolve_derivation_dataset_path(
        explicit_dataset_path="",
        base_payload={},
        review_dir=review_dir,
    )

    assert selected == latest_alias
