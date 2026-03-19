from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_liquidity_sweep_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_liquidity_sweep_watch_tracks_first_stage_from_execution_gate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "cvd_confirmation",
                    ],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, cvd_confirmation.",
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_material_change_trigger.json",
        {
            "trigger_brief": "no_material_orderflow_change_since_cross_section_anchor:SOLUSDT:wait_for_material_orderflow_change_before_rerun:deprioritize_flow",
            "trigger_status": "no_material_orderflow_change_since_cross_section_anchor",
            "trigger_decision": "wait_for_material_orderflow_change_before_rerun",
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "ticket_row_reasons": ["proxy_price_reference_only"],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:06:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "liquidity_sweep_waiting_proxy_price_blocked"
    assert payload["watch_decision"] == "wait_for_liquidity_sweep_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_liquidity_sweep_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_liquidity_sweep_watch"
    assert payload["liquidity_sweep_missing"] is True
    assert payload["liquidity_sweep_long"] is False
    assert payload["liquidity_sweep_short"] is False
    assert payload["mss_long"] is False
    assert payload["mss_short"] is False


def test_build_crypto_shortline_liquidity_sweep_watch_delegates_to_price_reference_watch_once_sweep_clears(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Setup_Ready",
                    "route_state": "watch",
                    "missing_gates": [],
                    "blocker_detail": "SOLUSDT keeps setup but still lacks executable price reference.",
                    "structure_signals": {
                        "sweep_long": True,
                        "sweep_short": False,
                        "mss_long": True,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "ticket_row_reasons": ["proxy_price_reference_only"],
        },
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_shortline_price_reference_watch.json",
        {
            "watch_brief": "price_reference_missing_template_proxy_only:SOLUSDT:build_price_template_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "price_reference_missing_template_proxy_only",
            "watch_decision": "build_price_template_then_recheck_execution_gate",
            "blocker_title": "Build executable price reference before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_price_reference_watch",
            "next_action": "build_price_template_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_price_reference_watch",
            "done_when": "SOLUSDT keeps execution_price_ready=true, entry/stop/target stay non-zero, and the route ticket drops proxy_price_reference_only",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:06:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "price_reference_missing_template_proxy_only"
    assert payload["watch_decision"] == "build_price_template_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_price_reference_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_price_reference_watch"
    assert payload["price_reference_watch_status"] == "price_reference_missing_template_proxy_only"


def test_build_crypto_shortline_liquidity_sweep_watch_prefers_event_trigger_as_top_target_when_sweep_missing(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss.",
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "ticket_row_reasons": ["proxy_price_reference_only"],
        },
    )
    _write_json(
        review_dir / "20260315T100045Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_status": "no_liquidity_sweep_event_observed",
            "trigger_decision": "wait_for_liquidity_sweep_event_then_recheck_execution_gate",
            "trigger_brief": "no_liquidity_sweep_event_observed:SOLUSDT:wait_for_liquidity_sweep_event_then_recheck_execution_gate:portfolio_margin_um",
            "blocker_title": "Track new liquidity sweep event before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "next_action_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "done_when": "SOLUSDT records a new liquidity sweep event and the next gate refresh confirms the updated stage",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:06:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "no_liquidity_sweep_event_observed_proxy_price_blocked"
    assert payload["watch_decision"] == "wait_for_liquidity_sweep_event_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_liquidity_event_trigger"
    assert payload["next_action_target_artifact"] == "crypto_shortline_liquidity_event_trigger"
    assert payload["liquidity_event_trigger_status"] == "no_liquidity_sweep_event_observed"


def test_build_crypto_shortline_liquidity_sweep_watch_prefers_ready_price_reference_over_stale_proxy_reason(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss.",
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "ticket_row_reasons": ["confidence_below_threshold", "proxy_price_reference_only"],
        },
    )
    _write_json(
        review_dir / "20260315T100045Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_status": "liquidity_sweep_pending_orderflow_pressure",
            "trigger_decision": "wait_for_liquidity_sweep_event_then_recheck_execution_gate",
            "trigger_brief": "liquidity_sweep_pending_orderflow_pressure:SOLUSDT:wait_for_liquidity_sweep_event_then_recheck_execution_gate:portfolio_margin_um",
            "blocker_title": "Track liquidity sweep confirmation while orderflow pressure builds",
            "blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "next_action_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "done_when": "SOLUSDT records a new liquidity sweep event and the next gate refresh confirms the updated stage",
        },
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_shortline_price_reference_watch.json",
        {
            "watch_status": "price_reference_ready",
            "watch_decision": "recheck_shortline_execution_gate_after_price_reference_ready",
            "price_reference_blocked": False,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:06:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "liquidity_sweep_pending_orderflow_pressure"
    assert payload["price_reference_blocked"] is False
    assert payload["ticket_row_reasons"] == ["confidence_below_threshold"]


def test_build_crypto_shortline_liquidity_sweep_watch_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T124500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T124501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260318T124502Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "promoted",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "blocker_detail": "eth_only_liquidity_gap",
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260318T124503Z_crypto_shortline_material_change_trigger.json",
        {
            "route_symbol": "SOLUSDT",
            "trigger_brief": "sol_only_material_change",
            "trigger_status": "sol_only_material",
            "trigger_decision": "sol_only_action",
        },
    )
    _write_json(
        review_dir / "20260318T124504Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "route_symbol": "SOLUSDT",
            "ticket_row_reasons": ["proxy_price_reference_only"],
        },
    )
    _write_json(
        review_dir / "20260318T124505Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "route_symbol": "SOLUSDT",
            "trigger_status": "sol_only_liquidity_event",
            "trigger_decision": "sol_only_liquidity_action",
            "trigger_brief": "sol_only_liquidity_event:SOLUSDT:sol_only_liquidity_action:spot",
        },
    )
    _write_json(
        review_dir / "20260318T124506Z_crypto_shortline_price_reference_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_price_reference",
            "watch_brief": "sol_only_price_reference:SOLUSDT:recheck:spot",
            "price_reference_blocked": True,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--symbol",
            "ETHUSDT",
            "--now",
            "2026-03-18T12:45:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["watch_status"] == "liquidity_sweep_waiting"
    assert payload["watch_decision"] == "wait_for_liquidity_sweep_then_recheck_execution_gate"
    assert payload["liquidity_event_trigger_status"] == ""
    assert payload["price_reference_watch_status"] == ""
    assert payload["ticket_row_reasons"] == []
    assert payload["artifacts"]["crypto_shortline_material_change_trigger"] == ""
    assert payload["artifacts"]["crypto_shortline_ticket_constraint_diagnosis"] == ""
    assert payload["artifacts"]["crypto_shortline_liquidity_event_trigger"] == ""
    assert payload["artifacts"]["crypto_shortline_price_reference_watch"] == ""
    assert "sol_only" not in payload["blocker_detail"]
    assert "eth_only_liquidity_gap" in payload["blocker_detail"]
