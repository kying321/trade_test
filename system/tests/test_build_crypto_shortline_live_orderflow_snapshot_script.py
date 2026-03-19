from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_live_orderflow_snapshot.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_live_orderflow_snapshot_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T010000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T010010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        artifact_dir / "20260316T010020Z_micro_capture.json",
        {
            "status": "ok",
            "summary": "selected_micro_ready",
            "selected_micro": [
                {
                    "symbol": "SOLUSDT",
                    "schema_ok": True,
                    "gap_ok": True,
                    "time_sync_ok": True,
                    "trade_count": 48,
                    "evidence_score": 0.91,
                    "queue_imbalance": 0.42,
                    "ofi_norm": 0.21,
                    "micro_alignment": 0.33,
                    "cvd_delta_ratio": 0.18,
                    "cvd_context_mode": "continuation",
                    "cvd_veto_hint": "",
                    "cvd_locality_status": "local_window_ok",
                    "cvd_attack_presence": "buyers_attacking",
                    "cvd_attack_side": "buyers",
                    "cvd_trust_tier_hint": "single_exchange_ok",
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--now",
            "2026-03-16T01:02:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "live_orderflow_snapshot_ready"
    assert payload["snapshot_decision"] == "use_live_orderflow_snapshot_for_shortline_signal_quality"
    assert payload["signal_source"] == "crypto_shortline_live_orderflow_snapshot"
    assert payload["micro_quality_ok"] is True
    assert payload["raw_time_sync_ok"] is True
    assert payload["time_sync_ok"] is True
    assert payload["time_sync_resolution_source"] == "micro_capture"
    assert payload["cvd_ready"] is True
    assert payload["queue_imbalance"] == 0.42
    assert payload["artifacts"]["micro_capture"].endswith("_micro_capture.json")
    assert payload["blocker_target_artifact"] == "crypto_shortline_live_orderflow_snapshot"


def test_build_crypto_shortline_live_orderflow_snapshot_degraded_when_selected_micro_missing(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T010000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T010010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        artifact_dir / "20260316T010020Z_micro_capture.json",
        {
            "status": "degraded",
            "summary": "selected_micro_missing",
            "selected_micro": [],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--now",
            "2026-03-16T01:02:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "live_orderflow_snapshot_selected_micro_missing"
    assert payload["snapshot_decision"] == (
        "refresh_micro_capture_before_shortline_signal_quality_check"
    )
    assert payload["selected_micro_count"] == 0
    assert payload["blocker_target_artifact"] == "crypto_shortline_live_orderflow_snapshot"


def test_build_crypto_shortline_live_orderflow_snapshot_overrides_stale_micro_time_sync_with_newer_verification(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T010000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T010010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        artifact_dir / "20260314T145512Z_micro_capture.json",
        {
            "status": "degraded",
            "summary": "selected_micro_ready_but_time_sync_stale",
            "selected_micro": [
                {
                    "symbol": "SOLUSDT",
                    "schema_ok": True,
                    "gap_ok": True,
                    "time_sync_ok": False,
                    "trade_count": 500,
                    "evidence_score": 1.0,
                    "queue_imbalance": 0.036055889295320075,
                    "ofi_norm": 0.2696056801743102,
                    "micro_alignment": 0.14115329519086564,
                    "cvd_delta_ratio": 0.2696056801743102,
                    "cvd_context_mode": "continuation",
                    "cvd_context_note": "delta_confirms_directional_price_result|time_sync_risk",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "cvd_locality_status": "proxy_from_current_snapshot",
                    "cvd_attack_presence": "buyers_attacking",
                    "cvd_attack_side": "buyers",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101030Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--now",
            "2026-03-16T01:02:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "live_orderflow_snapshot_ready"
    assert payload["raw_time_sync_ok"] is False
    assert payload["time_sync_ok"] is True
    assert payload["time_sync_resolution_source"] == "repair_verification_override"
    assert payload["time_sync_verification_brief"] == "cleared:SOLUSDT:time_sync_ok"
    assert payload["micro_quality_ok"] is True
    assert payload["trust_ok"] is False
    assert payload["artifacts"]["system_time_sync_repair_verification_report"].endswith(
        "_system_time_sync_repair_verification_report.json"
    )
