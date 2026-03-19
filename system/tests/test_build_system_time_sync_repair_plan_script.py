from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_system_time_sync_repair_plan.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_system_time_sync_repair_plan_writes_manual_repair_artifact(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = output_root / "artifacts" / "time_sync"

    hot_path = review_dir / "20260314T151249Z_hot_universe_operator_brief.json"
    env_path = review_dir / "20260314T151200Z_system_time_sync_environment_report.json"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"
    verification_path = review_dir / "20260314T151500Z_system_time_sync_repair_verification_report.json"

    _write_json(
        hot_path,
        {
            "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "cross_market_review_head_symbol": "SOLUSDT",
            "cross_market_review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only. | time-sync=threshold_breach:scope=clock_skew_only; est_offset_ms=81.261"
            ),
            "cross_market_review_head_done_when": (
                "synchronize system clock until time-sync offset stays within configured limits, then rerun time_sync_probe"
            ),
        },
    )
    _write_json(
        env_path,
        {
            "classification": "timed_apns_fallback",
            "blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source; dns_ntp_hosts_resolve_cleanly",
            "remediation_hint": (
                "manual clock resync or administrator-level network time repair may be required; "
                "restore a direct NTP path for macOS timed before relying on APNS fallback, then rerun time_sync_probe"
            ),
        },
    )
    _write_json(
        probe_path,
        {
            "classification": "threshold_breach",
            "threshold_breach_scope": "clock_skew_only",
        },
    )
    _write_json(
        verification_path,
        {
            "status": "blocked",
            "verification_brief": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked",
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
            "--now",
            "2026-03-14T15:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "run_now"
    assert payload["plan_brief"] == "manual_time_repair_required:SOLUSDT:timed_apns_fallback"
    assert payload["admin_required"] is True
    assert payload["environment_classification"] == "timed_apns_fallback"
    assert payload["environment_blocker_detail"] == (
        "timed_source=APNS; os_time_service is not using NTP as the active source; dns_ntp_hosts_resolve_cleanly"
    )
    assert payload["blocker_detail"].endswith(
        "| env=timed_source=APNS; os_time_service is not using NTP as the active source; dns_ntp_hosts_resolve_cleanly"
    )
    assert payload["time_sync_probe_classification"] == "threshold_breach"
    assert payload["verification_status"] == "blocked"
    assert payload["verification_brief"] == (
        "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert len(payload["steps"]) == 5
    assert payload["steps"][1]["type"] == "manual_ui"
    assert "support.apple.com" in payload["steps"][1]["references"][0]
    assert "sudo /usr/sbin/systemsetup -setusingnetworktime on" in payload["steps"][2]["commands"]
    assert "sudo /bin/launchctl kickstart -k system/com.apple.timed" in payload["steps"][2]["commands"]
    assert "sudo /usr/bin/sntp -sS ntp.aliyun.com" in payload["steps"][3]["commands"]
    assert "build_system_time_sync_repair_verification_report.py" in payload["steps"][4]["commands"][-1]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
    markdown = Path(str(payload["markdown"])).read_text(encoding="utf-8")
    assert "# System Time Sync Repair Plan" in markdown
    assert "manual_time_repair_required:SOLUSDT:timed_apns_fallback" in markdown
    assert "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked" in markdown
    assert "sudo /usr/sbin/systemsetup -setusingnetworktime on" in markdown


def test_build_system_time_sync_repair_plan_preserves_fake_ip_residual_classification(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = output_root / "artifacts" / "time_sync"

    hot_path = review_dir / "20260314T151249Z_hot_universe_operator_brief.json"
    env_path = review_dir / "20260314T151200Z_system_time_sync_environment_report.json"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"

    _write_json(
        hot_path,
        {
            "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "cross_market_review_head_symbol": "SOLUSDT",
            "cross_market_review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only. | time-sync=threshold_breach:scope=clock_skew_only; est_offset_ms=81.261"
            ),
            "cross_market_review_head_done_when": "synchronize system clock and rerun time_sync_probe",
        },
    )
    _write_json(
        env_path,
        {
            "classification": "timed_apns_fallback_ntp_reachable_fake_ip_residual",
            "blocker_detail": (
                "timed_source=APNS; os_time_service is not using NTP as the active source; "
                "dns_ntp_hosts_resolve_cleanly; recent_fake_ip_attempt=198.18.0.144; "
                "recent_fake_ip_result=6; recent_fake_ip_delay_ms=unreliable; probe_scope=clock_skew_only; probe_offset_ms=81.261"
            ),
            "remediation_hint": "direct NTP sources are reachable, but macOS timed is still using APNS fallback and recently attempted a fake-IP/proxy path",
            "residual_path_hint": "fake_ip_filter_present_stale_cache_or_tun_intercept",
            "clash_config_path": "/tmp/clash-verge.yaml",
            "clash_merge_path": "/tmp/Merge.yaml",
            "clash_dns_enhanced_mode": "fake-ip",
            "clash_tun_stack": "gvisor",
        },
    )
    _write_json(probe_path, {"classification": "threshold_breach", "threshold_breach_scope": "clock_skew_only"})

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
            "--now",
            "2026-03-14T15:25:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["plan_brief"] == "manual_time_repair_required:SOLUSDT:timed_apns_fallback_ntp_reachable_fake_ip_residual"
    assert payload["admin_required"] is True
    assert "recent_fake_ip_attempt=198.18.0.144" in payload["blocker_detail"]
    assert "fake-ip filter already covers the probe sources" in payload["steps"][2]["details"]
    assert "dns_mode=fake-ip, tun_stack=gvisor" in payload["steps"][2]["details"]
    assert any("/tmp/clash-verge.yaml" in command for command in payload["steps"][2]["commands"])


def test_build_system_time_sync_repair_plan_prioritizes_os_time_recovery_when_clash_exemptions_already_present(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = output_root / "artifacts" / "time_sync"

    hot_path = review_dir / "20260315T092040Z_hot_universe_operator_brief.json"
    env_path = review_dir / "20260315T092000Z_system_time_sync_environment_report.json"
    probe_path = artifact_dir / "20260314T161857Z_time_sync_probe.json"

    _write_json(
        hot_path,
        {
            "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "cross_market_review_head_symbol": "SOLUSDT",
            "cross_market_review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only. | time-sync=threshold_breach:scope=clock_skew_only; est_offset_ms=85.219"
            ),
            "cross_market_review_head_done_when": "synchronize system clock until offset returns within configured limits",
        },
    )
    _write_json(
        env_path,
        {
            "classification": "timed_apns_fallback_ntp_reachable_flapping",
            "blocker_detail": (
                "timed_source=APNS; os_time_service is not using NTP as the active source; dns_ntp_hosts_resolve_cleanly; "
                "probe_scope=clock_skew_only; probe_offset_ms=85.219; clash_time_exemptions_present; "
                "residual_path_hint=clash_exemptions_present_os_time_service_not_recovered"
            ),
            "remediation_hint": (
                "direct NTP sources are reachable and Clash already exempts timed/sntp plus the probe hosts; prioritize "
                "re-enabling macOS network time, toggling automatic time off/on, forcing a one-time admin resync, and "
                "restarting the OS time service path before editing Clash rules again, then rerun time_sync_probe"
            ),
            "residual_path_hint": "clash_exemptions_present_os_time_service_not_recovered",
            "timed_recent_ntp_success_detected": True,
            "timed_recent_ntp_fallback_seconds": 2.788,
            "timed_recent_direct_ntp_ips": ["17.253.114.35"],
            "timed_recent_direct_ntp_delay_ms_min": 79.8,
            "timed_recent_direct_ntp_delay_ms_max": 79.8,
            "clash_config_path": "/tmp/clash-verge.yaml",
            "clash_merge_path": "/tmp/Merge.yaml",
            "clash_dns_enhanced_mode": "fake-ip",
            "clash_tun_stack": "gvisor",
        },
    )
    _write_json(
        probe_path,
        {
            "classification": "threshold_breach",
            "threshold_breach_scope": "clock_skew_only",
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
            "--now",
            "2026-03-15T09:25:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["plan_brief"] == "manual_time_repair_required:SOLUSDT:timed_apns_fallback_ntp_reachable_flapping"
    assert "falling back to APNS again instead of staying on NTP" in payload["steps"][2]["details"]
    assert "Clash already exempts timed/sntp" in payload["steps"][2]["details"]
    assert "system/com.apple.timed" in payload["steps"][2]["details"]
    assert "briefly reached NTP and then fell back to APNS again" in payload["steps"][2]["details"]
    assert "after about 2.788s" in payload["steps"][2]["details"]
    assert "before editing Clash again" in payload["steps"][2]["details"]
    assert "direct NTP fetches from timed already succeeded" in payload["steps"][4]["details"]
    assert "17.253.114.35" in payload["steps"][4]["details"]
    assert "before falling back to APNS after about 2.788s" in payload["steps"][4]["details"]
    assert "sudo /usr/sbin/systemsetup -setusingnetworktime on" in payload["steps"][2]["commands"]
    assert "sudo /bin/launchctl print system/com.apple.timed" in payload["steps"][2]["commands"]
    assert "sudo /bin/launchctl kickstart -k system/com.apple.timed" in payload["steps"][2]["commands"]


def test_build_system_time_sync_repair_plan_prefers_cross_market_review_head_over_stale_hot_brief(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = output_root / "artifacts" / "time_sync"

    hot_path = review_dir / "20260315T092040Z_hot_universe_operator_brief.json"
    cross_market_path = review_dir / "20260315T092045Z_cross_market_operator_state.json"
    env_path = review_dir / "20260315T092000Z_system_time_sync_environment_report.json"
    probe_path = artifact_dir / "20260314T161857Z_time_sync_probe.json"

    _write_json(
        hot_path,
        {
            "cross_market_review_head_brief": "review:crypto_route:OLDUSDT:stale_action:10",
            "cross_market_review_head_symbol": "OLDUSDT",
            "cross_market_review_head_blocker_detail": "stale hot brief blocker",
            "cross_market_review_head_done_when": "stale hot brief done_when",
        },
    )
    _write_json(
        cross_market_path,
        {
            "review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "review_head_symbol": "SOLUSDT",
            "review_head_blocker_detail": "fresh cross market blocker",
            "review_head_done_when": "fresh cross market done_when",
        },
    )
    _write_json(
        env_path,
        {
            "classification": "timed_apns_fallback_ntp_reachable_flapping",
            "blocker_detail": "timed_source=APNS; probe_scope=clock_skew_only",
            "remediation_hint": "repair timed",
        },
    )
    _write_json(
        probe_path,
        {
            "classification": "threshold_breach",
            "threshold_breach_scope": "clock_skew_only",
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
            "--now",
            "2026-03-15T09:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["review_head_symbol"] == "SOLUSDT"
    assert payload["review_head_brief"] == "review:crypto_route:SOLUSDT:deprioritize_flow:58"
    assert payload["blocker_detail"] == "fresh cross market blocker | env=timed_source=APNS; probe_scope=clock_skew_only"
    assert payload["done_when"] == "fresh cross market done_when"
    assert payload["cross_market_artifact"] == str(cross_market_path)
    assert "Fenlie done_when: fresh cross market done_when" in payload["steps"][4]["details"]
