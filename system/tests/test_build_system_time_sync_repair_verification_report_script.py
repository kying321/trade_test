from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_system_time_sync_repair_verification_report.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_system_time_sync_repair_verification_report_prefers_current_env_classification_over_stale_plan(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = output_root / "artifacts" / "time_sync"

    hot_path = review_dir / "20260314T151249Z_hot_universe_operator_brief.json"
    env_path = review_dir / "20260314T151200Z_system_time_sync_environment_report.json"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"
    plan_path = review_dir / "20260314T153000Z_system_time_sync_repair_plan.json"

    _write_json(
        hot_path,
        {
            "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "cross_market_review_head_symbol": "SOLUSDT",
            "cross_market_review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only. | time-sync=threshold_breach:scope=clock_skew_only; est_offset_ms=81.261"
            ),
        },
    )
    _write_json(
        env_path,
        {
            "classification": "timed_apns_fallback_ntp_reachable",
            "blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source",
        },
    )
    _write_json(
        probe_path,
        {
            "classification": "threshold_breach",
            "threshold_breach_scope": "clock_skew_only",
            "threshold_breach_estimated_offset_ms": 81.261,
            "threshold_breach_estimated_rtt_ms": 117.276,
        },
    )
    _write_json(
        plan_path,
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "review_head_symbol": "SOLUSDT",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--hot-brief-file",
            str(hot_path),
            "--environment-report-file",
            str(env_path),
            "--time-sync-probe-file",
            str(probe_path),
            "--repair-plan-file",
            str(plan_path),
            "--now",
            "2026-03-14T15:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "blocked"
    assert payload["verification_brief"] == (
        "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert payload["repair_plan_brief"] == "manual_time_repair_required:SOLUSDT:timed_apns_fallback_ntp_reachable"
    assert payload["time_sync_probe_classification"] == "threshold_breach"
    assert payload["environment_classification"] == "timed_apns_fallback_ntp_reachable"
    assert payload["check_rows"][0]["status"] == "failed"
    assert payload["check_rows"][1]["status"] == "failed"
    assert payload["check_rows"][2]["status"] == "failed"
    assert payload["next_actions"][1] == "manual_time_repair_required:SOLUSDT:timed_apns_fallback_ntp_reachable"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
    markdown = Path(str(payload["markdown"])).read_text(encoding="utf-8")
    assert "# System Time Sync Repair Verification Report" in markdown
    assert "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked" in markdown


def test_build_system_time_sync_repair_verification_report_prefers_cross_market_review_head_over_stale_hot_brief(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = output_root / "artifacts" / "time_sync"

    cross_market_path = review_dir / "20260314T182015Z_cross_market_operator_state.json"
    hot_path = review_dir / "20260314T181757Z_hot_universe_operator_brief.json"
    env_path = review_dir / "20260314T181753Z_system_time_sync_environment_report.json"
    probe_path = artifact_dir / "20260314T161857Z_time_sync_probe.json"
    plan_path = review_dir / "20260314T181753Z_system_time_sync_repair_plan.json"

    _write_json(
        cross_market_path,
        {
            "review_head_symbol": "SOLUSDT",
            "review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only. | time-sync=threshold_breach:scope=clock_skew_only; est_offset_ms=81.261 "
                "| env=timed_apns_fallback_ntp_reachable_flapping:recent_ntp_direct_hits_present; recent_ntp_ips=17.253.116.125,17.253.114.35"
            ),
            "review_head_done_when": "synchronize system clock and keep macOS timed on NTP",
        },
    )
    _write_json(
        hot_path,
        {
            "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "cross_market_review_head_symbol": "SOLUSDT",
            "cross_market_review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only. | time-sync=threshold_breach:scope=clock_skew_only; est_offset_ms=81.261 "
                "| env=timed_apns_fallback_ntp_reachable"
            ),
            "cross_market_review_head_done_when": "stale hot brief done_when",
        },
    )
    _write_json(
        env_path,
        {
            "classification": "timed_apns_fallback_ntp_reachable_flapping",
            "blocker_detail": "timed_source=APNS; recent_ntp_direct_hits_present; recent_ntp_ips=17.253.116.125,17.253.114.35",
        },
    )
    _write_json(
        probe_path,
        {
            "classification": "threshold_breach",
            "threshold_breach_scope": "clock_skew_only",
            "threshold_breach_estimated_offset_ms": 85.219,
        },
    )
    _write_json(
        plan_path,
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback_ntp_reachable_flapping",
            "review_head_symbol": "SOLUSDT",
            "done_when": "stale plan done_when",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--cross-market-file",
            str(cross_market_path),
            "--hot-brief-file",
            str(hot_path),
            "--environment-report-file",
            str(env_path),
            "--time-sync-probe-file",
            str(probe_path),
            "--repair-plan-file",
            str(plan_path),
            "--now",
            "2026-03-14T18:20:15Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["cross_market_artifact"] == str(cross_market_path)
    assert "recent_ntp_direct_hits_present" in payload["review_head_blocker_detail"]
    assert payload["next_actions"][2] == "synchronize system clock and keep macOS timed on NTP"


def test_build_system_time_sync_repair_verification_report_clears_apns_fallback_when_probe_is_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = output_root / "artifacts" / "time_sync"

    cross_market_path = review_dir / "20260315T100810Z_cross_market_operator_state.json"
    env_path = review_dir / "20260315T100840Z_system_time_sync_environment_report.json"
    probe_path = artifact_dir / "20260315T100159Z_time_sync_probe.json"
    plan_path = review_dir / "20260315T100900Z_system_time_sync_repair_plan.json"

    _write_json(
        cross_market_path,
        {
            "review_head_symbol": "SOLUSDT",
            "review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "review_head_blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation.",
            "review_head_done_when": "wait for structure confirmation only",
        },
    )
    _write_json(
        env_path,
        {
            "classification": "timed_apns_fallback",
            "blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source",
        },
    )
    _write_json(
        probe_path,
        {
            "classification": "none",
            "status": "ok",
            "pass": True,
        },
    )
    _write_json(
        plan_path,
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "review_head_symbol": "SOLUSDT",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--cross-market-file",
            str(cross_market_path),
            "--environment-report-file",
            str(env_path),
            "--time-sync-probe-file",
            str(probe_path),
            "--repair-plan-file",
            str(plan_path),
            "--now",
            "2026-03-15T10:10:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "cleared"
    assert payload["verification_brief"] == "cleared:SOLUSDT:time_sync_ok"
    assert payload["check_rows"][1]["status"] == "passed"
