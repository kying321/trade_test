from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_exit_risk_research_chain_script",
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


def test_main_runs_full_exit_risk_research_chain_and_optional_panel_refresh(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = review_dir / "20260323T010000Z_public_intraday_crypto_bars_dataset.csv"
    base_artifact_path = review_dir / "20260323T011000Z_price_action_breakout_pullback_sim_only.json"
    older_exit_risk_path = review_dir / "20260323T012000Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    latest_exit_risk_path = review_dir / "20260323T013000Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"

    write_text(dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_json(base_artifact_path, {"focus_symbol": "ETHUSDT", "selected_params": {"breakout_lookback": 20}})
    write_json(older_exit_risk_path, {"generated_at_utc": "2026-03-23T01:20:00Z"})
    write_json(
        latest_exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 8,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 1.5,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos",
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {"research_decision": "use_hold_selection_gate_as_canonical_head"},
    )

    seen_calls: list[tuple[str, list[str]]] = []

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_calls.append((name, list(cmd)))
        if name == "run_hold_upstream_refresh":
            return {
                "handoff_path": str(review_dir / "hold_selection_handoff.json"),
                "handoff_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
                ),
                "stop_condition_path": str(review_dir / "hold_forward_stop.json"),
                "stop_condition_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
                ),
                "research_decision": "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos",
            }
        if name == "build_exit_risk_forward_blocker":
            if "--forward-consensus-path" not in cmd:
                return {
                    "json_path": str(review_dir / "seed_forward_blocker.json"),
                    "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"),
                    "research_decision": "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos",
                }
            return {
                "json_path": str(review_dir / "forward_blocker.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"),
                "research_decision": "exit_risk_forward_blocker_cleared_promote_challenger_pair",
            }
        if name == "build_exit_risk_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"compare_{train_days}.json"), "research_decision": f"compare_{train_days}_ok"}
        if name == "build_exit_risk_forward_consensus":
            return {
                "json_path": str(review_dir / "forward_consensus.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"),
                "research_decision": "challenger_pair_promotable_across_current_forward_oos",
            }
        if name == "build_exit_risk_break_even_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"break_even_compare_{train_days}.json")}
        if name == "build_exit_risk_break_even_sidecar":
            return {
                "json_path": str(review_dir / "break_even_sidecar.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"),
                "research_decision": "break_even_sidecar_no_observed_delta_keep_anchor",
            }
        if name == "build_exit_risk_forward_tail_capacity":
            return {
                "json_path": str(review_dir / "tail_capacity.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"),
                "research_decision": "exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset",
            }
        if name == "build_exit_risk_handoff":
            return {
                "json_path": str(review_dir / "handoff.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"),
                "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
                "source_head_status": "challenger_anchor_active",
            }
        if name == "build_exit_risk_guarded_review":
            return {
                "json_path": str(review_dir / "guarded_review.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json"),
                "research_decision": "break_even_guarded_review_ready_keep_baseline_anchor",
            }
        if name == "build_exit_risk_review_packet":
            return {
                "json_path": str(review_dir / "review_packet.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json"),
                "research_decision": "break_even_review_packet_ready_for_primary_anchor_review",
            }
        if name == "build_exit_risk_review_conclusion":
            return {
                "json_path": str(review_dir / "review_conclusion.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json"),
                "research_decision": "break_even_review_conclusion_ready_keep_baseline_anchor_review_only",
                "arbitration_state": "review_only",
            }
        if name == "build_exit_risk_primary_anchor_review":
            return {
                "json_path": str(review_dir / "primary_anchor_review.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json"),
                "research_decision": "break_even_primary_anchor_review_complete_keep_baseline_anchor",
                "review_state": "completed",
            }
        if name == "build_exit_risk_source_gap_audit":
            return {
                "json_path": str(review_dir / "source_gap_audit.json"),
                "latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.json"
                ),
                "research_decision": "exit_risk_source_gap_detected_consumer_drift_risk",
                "finding_count": 2,
            }
        if name == "run_operator_panel_refresh":
            return {"ok": True, "snapshot_public": str(system_root / "dashboard" / "web" / "public" / "data" / "fenlie_dashboard_snapshot.json")}
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-23T22:10:00Z",
            "--refresh-panel",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["mode"] == "exit_risk_research_chain_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["stamp"] == "20260323T221000Z"
    assert payload["dataset_path"] == str(dataset_path)
    assert payload["base_artifact_path"] == str(base_artifact_path)
    assert payload["exit_risk_path"] == str(latest_exit_risk_path)
    assert payload["seed_blocker_path"] == str(review_dir / "seed_forward_blocker.json")
    assert payload["hold_forward_stop_path"] == str(review_dir / "hold_forward_stop.json")
    assert payload["hold_forward_stop_latest_json_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    )
    assert payload["hold_upstream_handoff_path"] == str(review_dir / "hold_selection_handoff.json")
    assert payload["forward_compare_train_days"] == [30, 40, 45, 50, 55, 60]
    assert payload["break_even_train_days"] == [30, 40, 45, 50, 55, 60]
    assert payload["break_even_step_days"] == 5
    assert payload["forward_compare_paths"] == [
        str(review_dir / "compare_30.json"),
        str(review_dir / "compare_40.json"),
        str(review_dir / "compare_45.json"),
        str(review_dir / "compare_50.json"),
        str(review_dir / "compare_55.json"),
        str(review_dir / "compare_60.json"),
    ]
    assert payload["break_even_compare_paths"] == [
        str(review_dir / "break_even_compare_30.json"),
        str(review_dir / "break_even_compare_40.json"),
        str(review_dir / "break_even_compare_45.json"),
        str(review_dir / "break_even_compare_50.json"),
        str(review_dir / "break_even_compare_55.json"),
        str(review_dir / "break_even_compare_60.json"),
    ]
    assert payload["forward_consensus_path"] == str(review_dir / "forward_consensus.json")
    assert payload["forward_blocker_path"] == str(review_dir / "forward_blocker.json")
    assert payload["break_even_sidecar_path"] == str(review_dir / "break_even_sidecar.json")
    assert payload["tail_capacity_path"] == str(review_dir / "tail_capacity.json")
    assert payload["json_path"] == str(review_dir / "handoff.json")
    assert payload["latest_json_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"
    )
    assert payload["guarded_review_path"] == str(review_dir / "guarded_review.json")
    assert payload["guarded_review_latest_json_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json"
    )
    assert payload["guarded_review_research_decision"] == "break_even_guarded_review_ready_keep_baseline_anchor"
    assert payload["review_packet_path"] == str(review_dir / "review_packet.json")
    assert payload["review_packet_latest_json_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json"
    )
    assert payload["review_packet_research_decision"] == "break_even_review_packet_ready_for_primary_anchor_review"
    assert payload["review_conclusion_path"] == str(review_dir / "review_conclusion.json")
    assert payload["review_conclusion_latest_json_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json"
    )
    assert payload["review_conclusion_research_decision"] == "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
    assert payload["review_conclusion_arbitration_state"] == "review_only"
    assert payload["primary_anchor_review_path"] == str(review_dir / "primary_anchor_review.json")
    assert payload["primary_anchor_review_latest_json_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json"
    )
    assert payload["primary_anchor_review_research_decision"] == "break_even_primary_anchor_review_complete_keep_baseline_anchor"
    assert payload["primary_anchor_review_state"] == "completed"
    assert payload["source_gap_audit_path"] == str(review_dir / "source_gap_audit.json")
    assert payload["source_gap_audit_latest_json_path"] == str(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.json"
    )
    assert payload["source_gap_audit_research_decision"] == "exit_risk_source_gap_detected_consumer_drift_risk"
    assert payload["source_gap_audit_finding_count"] == 2
    assert payload["research_decision"] == "use_exit_risk_handoff_as_canonical_anchor"
    assert payload["source_head_status"] == "challenger_anchor_active"
    assert payload["panel_refresh"]["ok"] is True

    call_names = [name for name, _ in seen_calls]
    assert call_names == [
        "run_hold_upstream_refresh",
        "build_exit_risk_forward_blocker",
        "build_exit_risk_forward_compare",
        "build_exit_risk_forward_compare",
        "build_exit_risk_forward_compare",
        "build_exit_risk_forward_compare",
        "build_exit_risk_forward_compare",
        "build_exit_risk_forward_compare",
        "build_exit_risk_forward_consensus",
        "build_exit_risk_break_even_forward_compare",
        "build_exit_risk_break_even_forward_compare",
        "build_exit_risk_break_even_forward_compare",
        "build_exit_risk_break_even_forward_compare",
        "build_exit_risk_break_even_forward_compare",
        "build_exit_risk_break_even_forward_compare",
        "build_exit_risk_break_even_sidecar",
        "build_exit_risk_forward_tail_capacity",
        "build_exit_risk_forward_blocker",
        "build_exit_risk_handoff",
        "build_exit_risk_guarded_review",
        "build_exit_risk_review_packet",
        "build_exit_risk_review_conclusion",
        "build_exit_risk_primary_anchor_review",
        "build_exit_risk_source_gap_audit",
        "run_operator_panel_refresh",
    ]

    first_cmd = seen_calls[0][1]
    assert first_cmd == [
        sys.executable,
        str(system_root / "scripts" / "run_price_action_breakout_pullback_hold_upstream_refresh_sim_only.py"),
        "--workspace",
        str(workspace),
        "--review-dir",
        str(review_dir),
        "--symbol",
        "ETHUSDT",
        "--base-artifact-path",
        str(base_artifact_path),
        "--now",
        "2026-03-23T22:10:00Z",
    ]

    second_cmd = seen_calls[1][1]
    assert second_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
        "--exit-risk-path",
        str(latest_exit_risk_path),
        "--hold-forward-stop-path",
        str(review_dir / "hold_forward_stop.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221001Z",
    ]

    first_compare_cmd = seen_calls[2][1]
    assert first_compare_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.py"),
        "--dataset-path",
        str(dataset_path),
        "--base-artifact-path",
        str(base_artifact_path),
        "--challenge-pair-path",
        str(review_dir / "seed_forward_blocker.json"),
        "--symbol",
        "ETHUSDT",
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221002Z",
        "--train-days",
        "30",
        "--validation-days",
        "10",
        "--step-days",
        "10",
    ]

    first_break_even_compare_cmd = seen_calls[9][1]
    assert first_break_even_compare_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.py"),
        "--dataset-path",
        str(dataset_path),
        "--base-artifact-path",
        str(base_artifact_path),
        "--exit-risk-path",
        str(latest_exit_risk_path),
        "--symbol",
        "ETHUSDT",
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221009Z",
        "--train-days",
        "30",
        "--validation-days",
        "10",
        "--step-days",
        "5",
    ]

    refreshed_blocker_cmd = seen_calls[17][1]
    assert refreshed_blocker_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
        "--exit-risk-path",
        str(latest_exit_risk_path),
        "--hold-forward-stop-path",
        str(review_dir / "hold_forward_stop.json"),
        "--forward-consensus-path",
        str(review_dir / "forward_consensus.json"),
        "--break-even-sidecar-path",
        str(review_dir / "break_even_sidecar.json"),
        "--tail-capacity-path",
        str(review_dir / "tail_capacity.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221017Z",
    ]

    guarded_review_cmd = seen_calls[19][1]
    assert guarded_review_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.py"),
        "--handoff-path",
        str(review_dir / "handoff.json"),
        "--forward-blocker-path",
        str(review_dir / "forward_blocker.json"),
        "--break-even-sidecar-path",
        str(review_dir / "break_even_sidecar.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221019Z",
    ]

    review_packet_cmd = seen_calls[20][1]
    assert review_packet_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.py"),
        "--guarded-review-path",
        str(review_dir / "guarded_review.json"),
        "--handoff-path",
        str(review_dir / "handoff.json"),
        "--break-even-sidecar-path",
        str(review_dir / "break_even_sidecar.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221020Z",
    ]

    review_conclusion_cmd = seen_calls[21][1]
    assert review_conclusion_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.py"),
        "--review-packet-path",
        str(review_dir / "review_packet.json"),
        "--guarded-review-path",
        str(review_dir / "guarded_review.json"),
        "--handoff-path",
        str(review_dir / "handoff.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221021Z",
    ]

    primary_anchor_review_cmd = seen_calls[22][1]
    assert primary_anchor_review_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.py"),
        "--review-conclusion-path",
        str(review_dir / "review_conclusion.json"),
        "--review-packet-path",
        str(review_dir / "review_packet.json"),
        "--handoff-path",
        str(review_dir / "handoff.json"),
        "--forward-consensus-path",
        str(review_dir / "forward_consensus.json"),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221022Z",
    ]

    source_gap_audit_cmd = seen_calls[23][1]
    assert source_gap_audit_cmd == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.py"),
        "--workspace",
        str(workspace),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T221023Z",
    ]


def test_latest_review_artifact_raises_for_missing_dataset(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="no_public_intraday_crypto_bars_dataset_found"):
        mod.latest_review_artifact(
            review_dir,
            "*_public_intraday_crypto_bars_dataset.csv",
            "no_public_intraday_crypto_bars_dataset_found",
        )


def test_preferred_review_artifact_prefers_latest_alias_when_present(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    newer_timestamped = review_dir / "20260324T051100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    latest_alias = review_dir / "latest_price_action_breakout_pullback_exit_risk_sim_only.json"
    write_json(newer_timestamped, {"generated_at_utc": "2026-03-24T05:11:00Z"})
    write_json(latest_alias, {"generated_at_utc": "2026-03-23T18:49:04Z"})

    selected = mod.preferred_review_artifact(
        review_dir,
        latest_name="latest_price_action_breakout_pullback_exit_risk_sim_only.json",
        pattern="*_price_action_breakout_pullback_exit_risk_sim_only.json",
        error_code="no_exit_risk_sim_only_artifact_found",
    )

    assert selected == latest_alias


def test_main_fails_when_hold_selection_handoff_latest_is_missing(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    write_text(review_dir / "20260323T010000Z_public_intraday_crypto_bars_dataset.csv", "ts,symbol,open,high,low,close,volume\n")
    write_json(review_dir / "20260323T011000Z_price_action_breakout_pullback_sim_only.json", {"focus_symbol": "ETHUSDT"})
    write_json(review_dir / "20260323T013000Z_price_action_breakout_pullback_exit_risk_sim_only.json", {"symbol": "ETHUSDT"})
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json",
        {"research_decision": "non_overlapping_forward_capacity_supports_40d_watch_45d_plus_insufficient"},
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json",
        {"research_decision": "overlapping_long_window_split_keep_non_overlapping_baseline_watch_hold8"},
    )

    with pytest.raises(FileNotFoundError, match="missing_hold_selection_handoff_latest"):
        mod.require_path(
            review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
            "missing_hold_selection_handoff_latest",
        )


def test_main_uses_unique_stamps_for_repeated_compare_builders(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    write_text(review_dir / "20260323T010000Z_public_intraday_crypto_bars_dataset.csv", "ts,symbol,open,high,low,close,volume\n")
    write_json(review_dir / "20260323T011000Z_price_action_breakout_pullback_sim_only.json", {"focus_symbol": "ETHUSDT"})
    write_json(review_dir / "20260323T013000Z_price_action_breakout_pullback_exit_risk_sim_only.json", {"symbol": "ETHUSDT"})
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        {
            "challenge_pair": {
                "baseline_exit_params": {"max_hold_bars": 16},
                "challenger_exit_params": {"max_hold_bars": 8},
            }
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json",
        {"research_decision": "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos"},
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {"research_decision": "use_hold_selection_gate_as_canonical_head"},
    )

    seen_calls: list[tuple[str, list[str]]] = []

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_calls.append((name, list(cmd)))
        stamp = cmd[cmd.index("--stamp") + 1] if "--stamp" in cmd else "from_now"
        if name == "build_exit_risk_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_compare_{train_days}.json")}
        if name == "build_exit_risk_forward_consensus":
            return {"json_path": str(review_dir / f"{stamp}_forward_consensus.json")}
        if name == "run_hold_upstream_refresh":
            return {
                "handoff_path": str(review_dir / "from_refresh_hold_selection_handoff.json"),
                "handoff_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
                ),
                "stop_condition_path": str(review_dir / "from_refresh_hold_forward_stop.json"),
                "stop_condition_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
                ),
            }
        if name == "build_exit_risk_forward_blocker":
            return {"json_path": str(review_dir / f"{stamp}_forward_blocker.json")}
        if name == "build_exit_risk_break_even_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_break_even_{train_days}.json")}
        if name == "build_exit_risk_break_even_sidecar":
            return {"json_path": str(review_dir / f"{stamp}_sidecar.json")}
        if name == "build_exit_risk_forward_tail_capacity":
            return {"json_path": str(review_dir / f"{stamp}_tail.json")}
        if name == "build_exit_risk_handoff":
            return {"json_path": str(review_dir / f"{stamp}_handoff.json"), "latest_json_path": str(review_dir / "latest_handoff.json")}
        if name == "build_exit_risk_guarded_review":
            return {
                "json_path": str(review_dir / f"{stamp}_guarded_review.json"),
                "latest_json_path": str(review_dir / "latest_guarded_review.json"),
            }
        if name == "build_exit_risk_review_packet":
            return {
                "json_path": str(review_dir / f"{stamp}_review_packet.json"),
                "latest_json_path": str(review_dir / "latest_review_packet.json"),
            }
        if name == "build_exit_risk_review_conclusion":
            return {
                "json_path": str(review_dir / f"{stamp}_review_conclusion.json"),
                "latest_json_path": str(review_dir / "latest_review_conclusion.json"),
            }
        if name == "build_exit_risk_primary_anchor_review":
            return {
                "json_path": str(review_dir / f"{stamp}_primary_anchor_review.json"),
                "latest_json_path": str(review_dir / "latest_primary_anchor_review.json"),
            }
        if name == "build_exit_risk_source_gap_audit":
            return {"json_path": str(review_dir / f"{stamp}_source_gap_audit.json"), "latest_json_path": str(review_dir / "latest_source_gap_audit.json")}
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-23T22:10:00Z",
        ],
    )

    mod.main()
    _ = capsys.readouterr()

    forward_compare_stamps = [
        cmd[cmd.index("--stamp") + 1]
        for name, cmd in seen_calls
        if name == "build_exit_risk_forward_compare"
    ]
    break_even_compare_stamps = [
        cmd[cmd.index("--stamp") + 1]
        for name, cmd in seen_calls
        if name == "build_exit_risk_break_even_forward_compare"
    ]

    assert len(set(forward_compare_stamps)) == len(forward_compare_stamps)
    assert len(set(break_even_compare_stamps)) == len(break_even_compare_stamps)


def test_main_reads_handoff_head_fields_from_written_artifact_when_stdout_payload_is_minimal(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    write_text(review_dir / "20260323T010000Z_public_intraday_crypto_bars_dataset.csv", "ts,symbol,open,high,low,close,volume\n")
    write_json(review_dir / "20260323T011000Z_price_action_breakout_pullback_sim_only.json", {"focus_symbol": "ETHUSDT"})
    write_json(review_dir / "20260323T013000Z_price_action_breakout_pullback_exit_risk_sim_only.json", {"symbol": "ETHUSDT"})
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        {
            "challenge_pair": {
                "baseline_exit_params": {"max_hold_bars": 16},
                "challenger_exit_params": {"max_hold_bars": 8},
            }
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json",
        {"research_decision": "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos"},
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {"research_decision": "use_hold_selection_gate_as_canonical_head"},
    )

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        stamp = cmd[cmd.index("--stamp") + 1] if "--stamp" in cmd else "from_now"
        if name == "build_exit_risk_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_compare_{train_days}.json")}
        if name == "build_exit_risk_forward_consensus":
            return {"json_path": str(review_dir / f"{stamp}_forward_consensus.json")}
        if name == "run_hold_upstream_refresh":
            return {
                "handoff_path": str(review_dir / "from_refresh_hold_selection_handoff.json"),
                "handoff_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
                ),
                "stop_condition_path": str(review_dir / "from_refresh_hold_forward_stop.json"),
                "stop_condition_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
                ),
            }
        if name == "build_exit_risk_forward_blocker":
            return {"json_path": str(review_dir / f"{stamp}_forward_blocker.json")}
        if name == "build_exit_risk_break_even_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_break_even_{train_days}.json")}
        if name == "build_exit_risk_break_even_sidecar":
            return {"json_path": str(review_dir / f"{stamp}_sidecar.json")}
        if name == "build_exit_risk_forward_tail_capacity":
            return {"json_path": str(review_dir / f"{stamp}_tail.json")}
        if name == "build_exit_risk_handoff":
            json_path = review_dir / f"{stamp}_handoff.json"
            write_json(
                json_path,
                {
                    "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
                    "source_head_status": "challenger_anchor_active",
                    "active_baseline": "hold8_trail15_be075",
                    "superseded_anchor": "hold16_trail15_no_be",
                    "transfer_watch": ["55d_plus_tie_windows"],
                },
            )
            return {"json_path": str(json_path), "latest_json_path": str(review_dir / "latest_handoff.json")}
        if name == "build_exit_risk_guarded_review":
            return {
                "json_path": str(review_dir / f"{stamp}_guarded_review.json"),
                "latest_json_path": str(review_dir / "latest_guarded_review.json"),
            }
        if name == "build_exit_risk_review_packet":
            return {
                "json_path": str(review_dir / f"{stamp}_review_packet.json"),
                "latest_json_path": str(review_dir / "latest_review_packet.json"),
            }
        if name == "build_exit_risk_review_conclusion":
            return {
                "json_path": str(review_dir / f"{stamp}_review_conclusion.json"),
                "latest_json_path": str(review_dir / "latest_review_conclusion.json"),
            }
        if name == "build_exit_risk_primary_anchor_review":
            return {
                "json_path": str(review_dir / f"{stamp}_primary_anchor_review.json"),
                "latest_json_path": str(review_dir / "latest_primary_anchor_review.json"),
            }
        if name == "build_exit_risk_source_gap_audit":
            json_path = review_dir / f"{stamp}_source_gap_audit.json"
            write_json(
                json_path,
                {
                    "research_decision": "exit_risk_source_gap_detected_consumer_drift_risk",
                    "finding_count": 2,
                },
            )
            return {"json_path": str(json_path), "latest_json_path": str(review_dir / "latest_source_gap_audit.json")}
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-23T22:10:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["research_decision"] == "use_exit_risk_handoff_as_canonical_anchor"
    assert payload["source_head_status"] == "challenger_anchor_active"
    assert payload["active_baseline"] == "hold8_trail15_be075"
    assert payload["superseded_anchor"] == "hold16_trail15_no_be"
    assert payload["transfer_watch"] == ["55d_plus_tie_windows"]


def test_main_prefers_exit_risk_dataset_contract_over_latest_review_dataset_when_cli_dataset_absent(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    latest_dataset_path = review_dir / "20260323T194500Z_public_intraday_crypto_bars_dataset.csv"
    exit_risk_dataset_path = review_dir / "20260321T114900Z_public_intraday_crypto_bars_dataset.csv"
    base_artifact_path = review_dir / "20260323T011000Z_price_action_breakout_pullback_sim_only.json"
    exit_risk_path = review_dir / "20260323T013000Z_price_action_breakout_pullback_exit_risk_sim_only.json"

    write_text(latest_dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_text(exit_risk_dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_json(base_artifact_path, {"focus_symbol": "ETHUSDT"})
    write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "dataset_path": str(exit_risk_dataset_path),
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {"research_decision": "use_hold_selection_gate_as_canonical_head"},
    )

    seen_calls: list[tuple[str, list[str]]] = []

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_calls.append((name, list(cmd)))
        stamp = cmd[cmd.index("--stamp") + 1] if "--stamp" in cmd else "from_now"
        if name == "run_hold_upstream_refresh":
            return {
                "handoff_path": str(review_dir / "from_refresh_hold_selection_handoff.json"),
                "handoff_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
                ),
                "stop_condition_path": str(review_dir / "from_refresh_hold_forward_stop.json"),
                "stop_condition_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
                ),
            }
        if name == "build_exit_risk_forward_blocker":
            return {"json_path": str(review_dir / f"{stamp}_forward_blocker.json")}
        if name == "build_exit_risk_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_compare_{train_days}.json")}
        if name == "build_exit_risk_forward_consensus":
            return {"json_path": str(review_dir / f"{stamp}_forward_consensus.json")}
        if name == "build_exit_risk_break_even_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_break_even_{train_days}.json")}
        if name == "build_exit_risk_break_even_sidecar":
            return {"json_path": str(review_dir / f"{stamp}_sidecar.json")}
        if name == "build_exit_risk_forward_tail_capacity":
            return {"json_path": str(review_dir / f"{stamp}_tail.json")}
        if name == "build_exit_risk_handoff":
            json_path = review_dir / f"{stamp}_handoff.json"
            write_json(json_path, {"active_baseline": "hold16_trail15_no_be"})
            return {"json_path": str(json_path), "latest_json_path": str(review_dir / "latest_handoff.json")}
        if name == "build_exit_risk_guarded_review":
            return {
                "json_path": str(review_dir / f"{stamp}_guarded_review.json"),
                "latest_json_path": str(review_dir / "latest_guarded_review.json"),
            }
        if name == "build_exit_risk_review_packet":
            return {
                "json_path": str(review_dir / f"{stamp}_review_packet.json"),
                "latest_json_path": str(review_dir / "latest_review_packet.json"),
            }
        if name == "build_exit_risk_review_conclusion":
            return {
                "json_path": str(review_dir / f"{stamp}_review_conclusion.json"),
                "latest_json_path": str(review_dir / "latest_review_conclusion.json"),
            }
        if name == "build_exit_risk_primary_anchor_review":
            return {
                "json_path": str(review_dir / f"{stamp}_primary_anchor_review.json"),
                "latest_json_path": str(review_dir / "latest_primary_anchor_review.json"),
            }
        if name == "build_exit_risk_source_gap_audit":
            return {"json_path": str(review_dir / f"{stamp}_source_gap_audit.json"), "latest_json_path": str(review_dir / "latest_source_gap_audit.json")}
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-23T22:10:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["dataset_path"] == str(exit_risk_dataset_path)
    first_compare_cmd = next(cmd for name, cmd in seen_calls if name == "build_exit_risk_forward_compare")
    assert first_compare_cmd[first_compare_cmd.index("--dataset-path") + 1] == str(exit_risk_dataset_path)


def test_main_falls_back_to_base_artifact_dataset_contract_before_latest_review_dataset(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    latest_dataset_path = review_dir / "20260323T194500Z_public_intraday_crypto_bars_dataset.csv"
    base_dataset_path = review_dir / "20260321T115755Z_public_intraday_crypto_bars_dataset.csv"
    base_artifact_path = review_dir / "20260323T011000Z_price_action_breakout_pullback_sim_only.json"
    exit_risk_path = review_dir / "20260323T013000Z_price_action_breakout_pullback_exit_risk_sim_only.json"

    write_text(latest_dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_text(base_dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_json(
        base_artifact_path,
        {
            "focus_symbol": "ETHUSDT",
            "dataset_path": str(base_dataset_path),
        },
    )
    write_json(exit_risk_path, {"symbol": "ETHUSDT"})
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {"research_decision": "use_hold_selection_gate_as_canonical_head"},
    )

    seen_calls: list[tuple[str, list[str]]] = []

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_calls.append((name, list(cmd)))
        stamp = cmd[cmd.index("--stamp") + 1] if "--stamp" in cmd else "from_now"
        if name == "run_hold_upstream_refresh":
            return {
                "handoff_path": str(review_dir / "from_refresh_hold_selection_handoff.json"),
                "handoff_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
                ),
                "stop_condition_path": str(review_dir / "from_refresh_hold_forward_stop.json"),
                "stop_condition_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
                ),
            }
        if name == "build_exit_risk_forward_blocker":
            return {"json_path": str(review_dir / f"{stamp}_forward_blocker.json")}
        if name == "build_exit_risk_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_compare_{train_days}.json")}
        if name == "build_exit_risk_forward_consensus":
            return {"json_path": str(review_dir / f"{stamp}_forward_consensus.json")}
        if name == "build_exit_risk_break_even_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_break_even_{train_days}.json")}
        if name == "build_exit_risk_break_even_sidecar":
            return {"json_path": str(review_dir / f"{stamp}_sidecar.json")}
        if name == "build_exit_risk_forward_tail_capacity":
            return {"json_path": str(review_dir / f"{stamp}_tail.json")}
        if name == "build_exit_risk_handoff":
            json_path = review_dir / f"{stamp}_handoff.json"
            write_json(json_path, {"active_baseline": "hold16_trail15_no_be"})
            return {"json_path": str(json_path), "latest_json_path": str(review_dir / "latest_handoff.json")}
        if name == "build_exit_risk_guarded_review":
            return {
                "json_path": str(review_dir / f"{stamp}_guarded_review.json"),
                "latest_json_path": str(review_dir / "latest_guarded_review.json"),
            }
        if name == "build_exit_risk_review_packet":
            return {
                "json_path": str(review_dir / f"{stamp}_review_packet.json"),
                "latest_json_path": str(review_dir / "latest_review_packet.json"),
            }
        if name == "build_exit_risk_review_conclusion":
            return {
                "json_path": str(review_dir / f"{stamp}_review_conclusion.json"),
                "latest_json_path": str(review_dir / "latest_review_conclusion.json"),
            }
        if name == "build_exit_risk_primary_anchor_review":
            return {
                "json_path": str(review_dir / f"{stamp}_primary_anchor_review.json"),
                "latest_json_path": str(review_dir / "latest_primary_anchor_review.json"),
            }
        if name == "build_exit_risk_source_gap_audit":
            return {"json_path": str(review_dir / f"{stamp}_source_gap_audit.json"), "latest_json_path": str(review_dir / "latest_source_gap_audit.json")}
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-23T22:10:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["dataset_path"] == str(base_dataset_path)
    first_compare_cmd = next(cmd for name, cmd in seen_calls if name == "build_exit_risk_forward_compare")
    assert first_compare_cmd[first_compare_cmd.index("--dataset-path") + 1] == str(base_dataset_path)


def test_resolve_dataset_path_prefers_latest_dataset_alias_over_timestamp_glob(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    latest_alias = review_dir / "latest_public_intraday_crypto_bars_dataset.csv"
    newer_timestamp_dataset = review_dir / "99999999T999999Z_public_intraday_crypto_bars_dataset.csv"
    write_text(latest_alias, "ts,symbol,open,high,low,close,volume\n")
    write_text(newer_timestamp_dataset, "ts,symbol,open,high,low,close,volume\n")

    selected = mod.resolve_dataset_path(
        explicit_dataset_path="",
        exit_risk_payload={},
        base_payload={},
        review_dir=review_dir,
    )

    assert selected == latest_alias


def test_main_builds_hold_selection_aligned_break_even_review_lane_when_canonical_handoff_conflicts(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = review_dir / "20260323T194500Z_public_intraday_crypto_bars_dataset.csv"
    base_artifact_path = review_dir / "20260323T011000Z_price_action_breakout_pullback_sim_only.json"
    exit_risk_path = review_dir / "20260323T013000Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"

    write_text(dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_json(base_artifact_path, {"focus_symbol": "ETHUSDT"})
    write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 24,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_hold_selection_gate_as_canonical_head",
            "active_baseline": "hold16_zero",
            "active_baseline_hold_bars": 16,
        },
    )

    seen_calls: list[tuple[str, list[str]]] = []
    handoff_call_count = 0

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        nonlocal handoff_call_count
        seen_calls.append((name, list(cmd)))
        stamp = cmd[cmd.index("--stamp") + 1] if "--stamp" in cmd else "from_now"
        aligned_review_dir = review_dir / "hold_selection_aligned_break_even_review"
        if name == "run_hold_upstream_refresh":
            return {
                "handoff_path": str(review_dir / "hold_selection_handoff.json"),
                "handoff_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
                ),
                "stop_condition_path": str(review_dir / "hold_forward_stop.json"),
                "stop_condition_latest_json_path": str(hold_forward_stop_path),
                "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
            }
        if name == "build_exit_risk_forward_blocker":
            if "--forward-consensus-path" not in cmd:
                return {
                    "json_path": str(review_dir / "seed_forward_blocker.json"),
                    "research_decision": "block_exit_risk_promotion_require_forward_oos_pair_hold16_vs_hold24",
                }
            return {
                "json_path": str(review_dir / "forward_blocker.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"),
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
            }
        if name == "build_exit_risk_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"compare_{train_days}.json")}
        if name == "build_exit_risk_forward_consensus":
            return {
                "json_path": str(review_dir / "forward_consensus.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"),
                "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            }
        if name == "build_exit_risk_break_even_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"break_even_compare_{train_days}.json")}
        if name == "build_exit_risk_break_even_sidecar":
            return {
                "json_path": str(review_dir / "break_even_sidecar.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"),
                "research_decision": "break_even_sidecar_positive_watch_only",
                "confidence_tier": "guarded_review_ready",
                "promotion_review_ready": True,
                "active_baseline": "hold24_trail0_no_be",
                "watch_candidate": "hold24_trail0_be075",
            }
        if name == "build_exit_risk_forward_tail_capacity":
            return {
                "json_path": str(review_dir / "tail_capacity.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"),
                "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
            }
        if name == "build_exit_risk_handoff":
            handoff_call_count += 1
            if handoff_call_count == 2:
                return {
                    "json_path": str(review_dir / "handoff_bridge_refresh.json"),
                    "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"),
                    "research_decision": "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict",
                    "source_head_status": "upstream_hold_selection_conflict",
                    "active_baseline": "hold24_trail0_no_be",
                    "watch_candidate": "hold24_trail0_be075",
                    "upstream_conflict_review_only_state": "ready",
                }
            return {
                "json_path": str(review_dir / "handoff.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"),
                "research_decision": "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict",
                "source_head_status": "upstream_hold_selection_conflict",
                "active_baseline": "hold24_trail0_no_be",
                "watch_candidate": "hold24_trail0_be075",
            }
        if name == "build_exit_risk_guarded_review":
            return {
                "json_path": str(review_dir / "guarded_review.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json"),
                "research_decision": "break_even_guarded_review_blocked_by_upstream_hold_selection_conflict",
            }
        if name == "build_exit_risk_review_packet":
            return {
                "json_path": str(review_dir / "review_packet.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json"),
                "research_decision": "break_even_review_packet_blocked_by_upstream_hold_selection_conflict",
            }
        if name == "build_exit_risk_review_conclusion":
            return {
                "json_path": str(review_dir / "review_conclusion.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json"),
                "research_decision": "break_even_review_conclusion_blocked_by_upstream_hold_selection_conflict",
            }
        if name == "build_exit_risk_primary_anchor_review":
            return {
                "json_path": str(review_dir / "primary_anchor_review.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json"),
                "research_decision": "break_even_primary_anchor_review_blocked_by_upstream_hold_selection_conflict",
            }
        if name == "build_exit_risk_break_even_forward_compare_aligned":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {
                "json_path": str(aligned_review_dir / f"break_even_compare_{train_days}.json"),
            }
        if name == "build_exit_risk_break_even_sidecar_aligned":
            return {
                "json_path": str(aligned_review_dir / "break_even_sidecar.json"),
                "latest_json_path": str(aligned_review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"),
                "research_decision": "break_even_sidecar_positive_watch_only",
                "confidence_tier": "guarded_review_ready",
                "promotion_review_ready": True,
                "active_baseline": "hold16_trail0_no_be",
                "watch_candidate": "hold16_trail0_be075",
                "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
            }
        if name == "build_exit_risk_guarded_review_aligned":
            return {
                "json_path": str(aligned_review_dir / "guarded_review.json"),
                "latest_json_path": str(aligned_review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json"),
                "research_decision": "break_even_guarded_review_ready_keep_baseline_anchor",
                "review_state": "ready",
            }
        if name == "build_exit_risk_review_packet_aligned":
            return {
                "json_path": str(aligned_review_dir / "review_packet.json"),
                "latest_json_path": str(aligned_review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json"),
                "research_decision": "break_even_review_packet_ready_for_primary_anchor_review",
            }
        if name == "build_exit_risk_review_conclusion_aligned":
            return {
                "json_path": str(aligned_review_dir / "review_conclusion.json"),
                "latest_json_path": str(aligned_review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json"),
                "research_decision": "break_even_review_conclusion_ready_keep_baseline_anchor_review_only",
                "arbitration_state": "review_only",
            }
        if name == "build_exit_risk_primary_anchor_review_aligned":
            return {
                "json_path": str(aligned_review_dir / "primary_anchor_review.json"),
                "latest_json_path": str(aligned_review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json"),
                "research_decision": "break_even_primary_anchor_review_complete_keep_baseline_anchor",
                "review_state": "completed",
            }
        if name == "build_exit_risk_source_gap_audit":
            return {
                "json_path": str(review_dir / "source_gap_audit.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.json"),
                "research_decision": "exit_risk_source_gap_detected_superseded_upstream_divergence_watch_only",
                "finding_count": 1,
            }
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-24T05:00:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["json_path"] == str(review_dir / "handoff_bridge_refresh.json")
    aligned_lane = payload["hold_selection_aligned_break_even_review_lane"]
    assert aligned_lane["enabled"] is True
    assert aligned_lane["review_dir"] == str(review_dir / "hold_selection_aligned_break_even_review")
    assert aligned_lane["active_baseline"] == "hold16_trail0_no_be"
    assert aligned_lane["preferred_watch_candidate"] == "hold16_trail0_be075"
    assert aligned_lane["research_decision"] == (
        "hold_selection_aligned_break_even_review_lane_ready_but_canonical_handoff_conflict_remains"
    )
    assert aligned_lane["break_even_sidecar_research_decision"] == "break_even_sidecar_positive_watch_only"
    assert aligned_lane["review_conclusion_research_decision"] == (
        "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
    )
    assert aligned_lane["primary_anchor_review_research_decision"] == (
        "break_even_primary_anchor_review_complete_keep_baseline_anchor"
    )
    assert aligned_lane["primary_anchor_review_state"] == "completed"

    aligned_call_names = [name for name, _ in seen_calls if name.endswith("_aligned")]
    assert aligned_call_names == [
        "build_exit_risk_break_even_forward_compare_aligned",
        "build_exit_risk_break_even_forward_compare_aligned",
        "build_exit_risk_break_even_forward_compare_aligned",
        "build_exit_risk_break_even_forward_compare_aligned",
        "build_exit_risk_break_even_forward_compare_aligned",
        "build_exit_risk_break_even_forward_compare_aligned",
        "build_exit_risk_break_even_sidecar_aligned",
        "build_exit_risk_guarded_review_aligned",
        "build_exit_risk_review_packet_aligned",
        "build_exit_risk_review_conclusion_aligned",
        "build_exit_risk_primary_anchor_review_aligned",
    ]
    assert [name for name, _ in seen_calls].count("build_exit_risk_handoff") == 2


def test_main_refreshes_canonical_anchor_consumers_after_challenger_promotion(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = review_dir / "20260325T103332Z_public_intraday_crypto_bars_dataset.csv"
    base_artifact_path = review_dir / "20260325T103332Z_price_action_breakout_pullback_sim_only.json"
    exit_risk_path = review_dir / "20260325T103332Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    hold_forward_stop_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"

    write_text(dataset_path, "ts,symbol,open,high,low,close,volume\n")
    write_json(base_artifact_path, {"focus_symbol": "ETHUSDT"})
    write_json(
        exit_risk_path,
        {
            "symbol": "ETHUSDT",
            "selected_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "validation_leader_exit_params": {
                "max_hold_bars": 24,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 0.0,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
        },
    )
    write_json(
        hold_forward_stop_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos",
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {"research_decision": "use_hold_selection_gate_as_canonical_head"},
    )

    seen_calls: list[tuple[str, list[str]]] = []
    sidecar_call_count = 0
    blocker_call_count = 0
    handoff_call_count = 0

    hold16_params = {
        "max_hold_bars": 16,
        "break_even_trigger_r": 0.0,
        "trailing_stop_atr": 0.0,
        "cooldown_after_losses": 0,
        "cooldown_bars": 0,
    }
    hold24_params = {
        "max_hold_bars": 24,
        "break_even_trigger_r": 0.0,
        "trailing_stop_atr": 0.0,
        "cooldown_after_losses": 0,
        "cooldown_bars": 0,
    }

    def blocker_payload(*, baseline_params: dict, challenger_params: dict, json_name: str) -> dict:
        latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
        json_path = review_dir / json_name
        body = {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_forward_blocker_cleared_promote_challenger_pair",
            "challenge_pair": {
                "baseline_exit_params": baseline_params,
                "challenger_exit_params": challenger_params,
                "baseline_hold_bars": int(baseline_params["max_hold_bars"]),
                "challenger_hold_bars": int(challenger_params["max_hold_bars"]),
                "shared_trailing_stop_atr": 0.0,
            },
        }
        write_json(json_path, body)
        write_json(latest_path, body)
        return {
            "json_path": str(json_path),
            "latest_json_path": str(latest_path),
            "research_decision": body["research_decision"],
            "challenge_pair": body["challenge_pair"],
        }

    def sidecar_payload(*, active_baseline: str, watch_candidate: str, json_name: str) -> dict:
        latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
        json_path = review_dir / json_name
        body = {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "active_baseline": active_baseline,
            "watch_candidate": watch_candidate,
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        }
        write_json(json_path, body)
        write_json(latest_path, body)
        return {
            "json_path": str(json_path),
            "latest_json_path": str(latest_path),
            "research_decision": body["research_decision"],
            "active_baseline": active_baseline,
            "watch_candidate": watch_candidate,
        }

    def handoff_payload(*, json_name: str) -> dict:
        latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"
        json_path = review_dir / json_name
        body = {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "challenger_anchor_active",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "superseded_anchor": "hold24_trail0_no_be",
            "consumer_rule": (
                "后续 consumer 必须先读取 "
                "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json`。"
            ),
        }
        write_json(json_path, body)
        write_json(latest_path, body)
        return {
            "json_path": str(json_path),
            "latest_json_path": str(latest_path),
            "research_decision": body["research_decision"],
            "source_head_status": body["source_head_status"],
        }

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        nonlocal sidecar_call_count, blocker_call_count, handoff_call_count
        seen_calls.append((name, list(cmd)))
        stamp = cmd[cmd.index("--stamp") + 1] if "--stamp" in cmd else "from_now"
        if name == "run_hold_upstream_refresh":
            return {
                "handoff_path": str(review_dir / "hold_selection_handoff.json"),
                "handoff_latest_json_path": str(
                    review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
                ),
                "stop_condition_path": str(review_dir / "hold_forward_stop.json"),
                "stop_condition_latest_json_path": str(hold_forward_stop_path),
                "research_decision": "stop_45d_plus_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos",
            }
        if name == "build_exit_risk_forward_blocker":
            if "--forward-consensus-path" not in cmd:
                return {
                    "json_path": str(review_dir / "seed_forward_blocker.json"),
                    "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"),
                    "research_decision": "block_exit_risk_promotion_require_hold16_vs_hold24_forward_oos",
                }
            blocker_call_count += 1
            if blocker_call_count == 1:
                return blocker_payload(
                    baseline_params=hold24_params,
                    challenger_params=hold16_params,
                    json_name="forward_blocker.json",
                )
            return blocker_payload(
                baseline_params=hold16_params,
                challenger_params=hold16_params,
                json_name="forward_blocker_refresh.json",
            )
        if name == "build_exit_risk_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            return {"json_path": str(review_dir / f"{stamp}_compare_{train_days}.json")}
        if name == "build_exit_risk_forward_consensus":
            json_path = review_dir / "forward_consensus.json"
            latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
            body = {
                "symbol": "ETHUSDT",
                "research_decision": "challenger_pair_promotable_across_current_forward_oos",
                "allowed_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
                "next_research_priority": "refresh_exit_risk_anchor_after_forward_oos_promotion",
            }
            write_json(json_path, body)
            write_json(latest_path, body)
            return {
                "json_path": str(json_path),
                "latest_json_path": str(latest_path),
                "research_decision": body["research_decision"],
            }
        if name == "build_exit_risk_break_even_forward_compare":
            train_days = cmd[cmd.index("--train-days") + 1]
            exit_risk_source = cmd[cmd.index("--exit-risk-path") + 1]
            json_path = review_dir / f"{stamp}_break_even_{train_days}.json"
            write_json(json_path, {"exit_risk_path": exit_risk_source})
            return {"json_path": str(json_path)}
        if name == "build_exit_risk_break_even_sidecar":
            sidecar_call_count += 1
            if sidecar_call_count == 1:
                return sidecar_payload(
                    active_baseline="hold24_trail0_no_be",
                    watch_candidate="hold24_trail0_be075",
                    json_name="break_even_sidecar.json",
                )
            return sidecar_payload(
                active_baseline="hold16_trail0_no_be",
                watch_candidate="hold16_trail0_be075",
                json_name="break_even_sidecar_refresh.json",
            )
        if name == "build_exit_risk_forward_tail_capacity":
            json_path = review_dir / "tail_capacity.json"
            latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"
            body = {
                "symbol": "ETHUSDT",
                "research_decision": "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
            }
            write_json(json_path, body)
            write_json(latest_path, body)
            return {
                "json_path": str(json_path),
                "latest_json_path": str(latest_path),
                "research_decision": body["research_decision"],
            }
        if name == "build_exit_risk_handoff":
            handoff_call_count += 1
            if handoff_call_count == 1:
                return handoff_payload(json_name="handoff.json")
            return handoff_payload(json_name="handoff_refresh.json")
        if name == "build_exit_risk_guarded_review":
            json_path = review_dir / f"{stamp}_guarded_review.json"
            latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json"
            body = {
                "symbol": "ETHUSDT",
                "research_decision": "break_even_guarded_review_prerequisites_missing_keep_baseline_anchor",
                "review_state": "blocked",
            }
            write_json(json_path, body)
            write_json(latest_path, body)
            return {
                "json_path": str(json_path),
                "latest_json_path": str(latest_path),
                "research_decision": body["research_decision"],
                "review_state": body["review_state"],
            }
        if name == "build_exit_risk_review_packet":
            return {
                "json_path": str(review_dir / f"{stamp}_review_packet.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json"),
                "research_decision": "break_even_review_packet_blocked_by_guarded_review",
            }
        if name == "build_exit_risk_review_conclusion":
            return {
                "json_path": str(review_dir / f"{stamp}_review_conclusion.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json"),
                "research_decision": "break_even_review_conclusion_blocked_by_guarded_review",
                "arbitration_state": "blocked",
            }
        if name == "build_exit_risk_primary_anchor_review":
            return {
                "json_path": str(review_dir / f"{stamp}_primary_anchor_review.json"),
                "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json"),
                "research_decision": "break_even_primary_anchor_review_blocked_by_guarded_review",
                "review_state": "blocked",
            }
        if name == "build_exit_risk_source_gap_audit":
            handoff = json.loads(
                (review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json").read_text(
                    encoding="utf-8"
                )
            )
            sidecar = json.loads(
                (
                    review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
                ).read_text(encoding="utf-8")
            )
            blocker = json.loads(
                (review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json").read_text(
                    encoding="utf-8"
                )
            )
            canonical_anchor = handoff["active_baseline"]
            blocker_baseline_hold = int(blocker["challenge_pair"]["baseline_exit_params"]["max_hold_bars"])
            pass_ready = sidecar["active_baseline"] == canonical_anchor and blocker_baseline_hold == 16
            json_path = review_dir / f"{stamp}_source_gap_audit.json"
            latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.json"
            body = {
                "research_decision": (
                    "exit_risk_source_gap_audit_pass_canonical_handoff_contract_intact"
                    if pass_ready
                    else "exit_risk_source_gap_detected_consumer_drift_risk"
                ),
                "finding_count": 0 if pass_ready else 2,
            }
            write_json(json_path, body)
            write_json(latest_path, body)
            return {
                "json_path": str(json_path),
                "latest_json_path": str(latest_path),
                "research_decision": body["research_decision"],
                "finding_count": body["finding_count"],
            }
        raise AssertionError(f"unexpected_run_json_call:{name}")

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-25T10:33:32Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["source_gap_audit_research_decision"] == "exit_risk_source_gap_audit_pass_canonical_handoff_contract_intact"
    assert payload["source_gap_audit_finding_count"] == 0
    assert payload["break_even_sidecar_path"] == str(review_dir / "break_even_sidecar_refresh.json")
    assert payload["forward_blocker_path"] == str(review_dir / "forward_blocker_refresh.json")
    assert payload["json_path"] == str(review_dir / "handoff_refresh.json")

    canonical_seed_calls = [
        cmd
        for name, cmd in seen_calls
        if name == "build_exit_risk_break_even_forward_compare"
        and "canonical_anchor_seed" in cmd[cmd.index("--exit-risk-path") + 1]
    ]
    assert len(canonical_seed_calls) == 6

    canonical_seed_path = Path(
        canonical_seed_calls[0][canonical_seed_calls[0].index("--exit-risk-path") + 1]
    )
    canonical_seed = json.loads(canonical_seed_path.read_text(encoding="utf-8"))
    assert canonical_seed["selected_exit_params"]["max_hold_bars"] == 16
    assert canonical_seed["validation_leader_exit_params"]["max_hold_bars"] == 16

    real_blocker_calls = [
        cmd for name, cmd in seen_calls if name == "build_exit_risk_forward_blocker" and "--forward-consensus-path" in cmd
    ]
    assert len(real_blocker_calls) == 2
    assert [name for name, _ in seen_calls].count("build_exit_risk_break_even_sidecar") == 2
    assert [name for name, _ in seen_calls].count("build_exit_risk_handoff") == 2
