from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_system_time_sync_environment_report.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    lines.append(f"  {sub_key}:")
                    for item in sub_value:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {sub_key}: {json.dumps(sub_value)}")
        else:
            lines.append(f"{key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_system_time_sync_environment_report", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prune_artifacts_ignores_deleted_candidates(tmp_path: Path) -> None:
    module = _load_script_module()

    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    live_json = review_dir / "20260314T190100Z_system_time_sync_environment_report.json"
    stale_json = review_dir / "20260314T182730Z_system_time_sync_environment_report.json"
    live_json.write_text("{}", encoding="utf-8")
    stale_json.write_text("{}", encoding="utf-8")
    stale_json.unlink()

    class _FakeReviewDir:
        def glob(self, pattern: str):
            if pattern.endswith("_system_time_sync_environment_report.json"):
                return [live_json, stale_json]
            return []

    pruned_keep, pruned_age = module.prune_artifacts(
        _FakeReviewDir(),
        current_paths=[live_json],
        keep=1,
        ttl_hours=24.0,
    )

    assert pruned_keep == []
    assert pruned_age == []
    assert live_json.exists()


def test_build_system_time_sync_environment_report_classifies_timed_ntp_fake_ip(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "time_sync"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"
    timed_log_path = tmp_path / "timed.log"
    clash_config_path = tmp_path / "clash-verge.yaml"
    clash_merge_path = tmp_path / "Merge.yaml"

    _write_json(
        probe_path,
        {
            "as_of": "2026-03-14T13:29:56Z",
            "dns_diagnostics": [
                {"source": "time.google.com", "resolved_addrs": ["216.239.35.0"], "fake_ip_detected": False},
                {"source": "time.cloudflare.com", "resolved_addrs": ["162.159.200.1"], "fake_ip_detected": False},
                {"source": "time.apple.com", "resolved_addrs": ["198.18.0.24"], "fake_ip_detected": True},
                {"source": "time.windows.com", "resolved_addrs": ["198.18.0.5"], "fake_ip_detected": True},
            ],
        },
    )
    timed_log_path.write_text(
        "\n".join(
            [
                '2026-03-14 21:43:13.499 Df timed[142:1def4d] [com.apple.timed:text] Received time Sat Mar 14 21:43:12 2026±0.23 from "NTP"',
                "2026-03-14 21:43:13.499 Df timed[142:1def4d] [com.apple.timed:data] cmd,fetchTime,num,2,result,0,mach,1067129346728,tv_sec,1773495793,tv_usec,445720,delay,0.237656,dispersion,0.000107,more,1,ip,198.18.0.64,port,123,slept,0,use_service_port,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_yaml(
        clash_config_path,
        {
            "tun": {"enable": True, "stack": "gvisor", "auto-route": True},
            "dns": {
                "enhanced-mode": "fake-ip",
                "fake-ip-filter": ["time.apple.com", "time.windows.com"],
            },
        },
    )
    _write_yaml(clash_merge_path, {})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--time-sync-probe-file",
            str(probe_path),
            "--timed-log-file",
            str(timed_log_path),
            "--clash-config-file",
            str(clash_config_path),
            "--clash-merge-file",
            str(clash_merge_path),
            "--now",
            "2026-03-14T21:45:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["classification"] == "timed_ntp_via_fake_ip"
    assert payload["timed_recent_source"] == "NTP"
    assert payload["timed_recent_ntp_fake_ip"] == "198.18.0.64"
    assert payload["timed_recent_ntp_fake_ip_delay_ms"] == 237.656
    assert "fake_ip_hosts=time.apple.com,time.windows.com" in payload["blocker_detail"]
    assert payload["clash_dns_enhanced_mode"] == "fake-ip"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_system_time_sync_environment_report_marks_recent_fake_ip_residual_under_apns_fallback(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "time_sync"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"
    timed_log_path = tmp_path / "timed.log"
    clash_config_path = tmp_path / "clash-verge.yaml"
    clash_merge_path = tmp_path / "Merge.yaml"

    _write_json(
        probe_path,
        {
            "as_of": "2026-03-14T13:29:56Z",
            "classification": "threshold_breach",
            "available_sources": 2,
            "threshold_breach_scope": "clock_skew_only",
            "threshold_breach_estimated_offset_ms": 81.261,
            "dns_diagnostics": [
                {"source": "time.apple.com", "resolved_addrs": ["17.253.114.35"], "fake_ip_detected": False},
                {"source": "time.windows.com", "resolved_addrs": ["52.231.114.183"], "fake_ip_detected": False},
            ],
        },
    )
    timed_log_path.write_text(
        "\n".join(
            [
                '2026-03-14 21:43:13.499 Df timed[142:1def4d] [com.apple.timed:text] Received time Sat Mar 14 21:43:12 2026±0.23 from "NTP"',
                "2026-03-14 21:43:13.499 Df timed[142:1def4d] [com.apple.timed:data] cmd,fetchTime,num,2,result,0,mach,1067129346728,tv_sec,1773495793,tv_usec,445720,delay,0.237656,dispersion,0.000107,more,1,ip,198.18.0.64,port,123,slept,0,use_service_port,1",
                '2026-03-14 21:44:13.499 Df timed[142:1def4d] [com.apple.timed:text] Received time Sat Mar 14 21:44:13 2026±35.00 from "APNS"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_yaml(
        clash_config_path,
        {
            "tun": {"enable": True, "stack": "gvisor", "auto-route": True},
            "dns": {
                "enhanced-mode": "fake-ip",
                "fake-ip-filter": ["time.apple.com", "time.windows.com"],
            },
        },
    )
    _write_yaml(clash_merge_path, {})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--time-sync-probe-file",
            str(probe_path),
            "--timed-log-file",
            str(timed_log_path),
            "--clash-config-file",
            str(clash_config_path),
            "--clash-merge-file",
            str(clash_merge_path),
            "--now",
            "2026-03-14T21:45:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["classification"] == "timed_apns_fallback_ntp_reachable_fake_ip_residual"
    assert payload["timed_recent_source"] == "APNS"
    assert payload["timed_recent_ntp_fake_ip"] == "198.18.0.64"
    assert payload["timed_recent_fake_ip_fetch_rows"][-1]["result"] == "0"
    assert payload["residual_path_hint"] == "fake_ip_filter_present_stale_cache_or_tun_intercept"
    assert "time.apple.com" in payload["covered_probe_sources"]
    assert "recent_fake_ip_attempt=198.18.0.64" in payload["blocker_detail"]
    assert "residual_path_hint=fake_ip_filter_present_stale_cache_or_tun_intercept" in payload["blocker_detail"]
    assert "dns_ntp_hosts_resolve_cleanly" in payload["blocker_detail"]
    assert "restart Clash/TUN or clear stale fake-IP state" in payload["remediation_hint"]


def test_build_system_time_sync_environment_report_marks_apns_fallback_ntp_reachable_without_fake_ip(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "time_sync"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"
    timed_log_path = tmp_path / "timed.log"
    clash_config_path = tmp_path / "clash-verge.yaml"
    clash_merge_path = tmp_path / "Merge.yaml"

    _write_json(
        probe_path,
        {
            "as_of": "2026-03-14T13:29:56Z",
            "classification": "threshold_breach",
            "available_sources": 2,
            "threshold_breach_scope": "clock_skew_only",
            "threshold_breach_estimated_offset_ms": 81.261,
            "dns_diagnostics": [
                {"source": "time.apple.com", "resolved_addrs": ["17.253.114.35"], "fake_ip_detected": False},
                {"source": "time.windows.com", "resolved_addrs": ["52.231.114.183"], "fake_ip_detected": False},
            ],
        },
    )
    timed_log_path.write_text(
        '2026-03-14 21:44:13.499 Df timed[142:1def4d] [com.apple.timed:text] Received time Sat Mar 14 21:44:13 2026±35.00 from "APNS"\n',
        encoding="utf-8",
    )
    _write_yaml(
        clash_config_path,
        {
            "tun": {"enable": True, "stack": "gvisor", "auto-route": True},
            "dns": {
                "enhanced-mode": "fake-ip",
                "fake-ip-filter": ["time.apple.com", "time.windows.com"],
            },
        },
    )
    _write_yaml(clash_merge_path, {})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--time-sync-probe-file",
            str(probe_path),
            "--timed-log-file",
            str(timed_log_path),
            "--clash-config-file",
            str(clash_config_path),
            "--clash-merge-file",
            str(clash_merge_path),
            "--now",
            "2026-03-14T21:45:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["classification"] == "timed_apns_fallback_ntp_reachable"
    assert payload["timed_recent_ntp_fake_ip"] == ""
    assert payload["residual_path_hint"] == ""
    assert "recent_fake_ip_attempt=" not in payload["blocker_detail"]


def test_build_system_time_sync_environment_report_marks_apns_fallback_with_clash_exemptions_present(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "time_sync"
    probe_path = artifact_dir / "20260315T001857Z_time_sync_probe.json"
    timed_log_path = tmp_path / "timed.log"
    clash_config_path = tmp_path / "clash-verge.yaml"
    clash_merge_path = tmp_path / "Merge.yaml"

    _write_json(
        probe_path,
        {
            "as_of": "2026-03-15T00:18:57Z",
            "classification": "threshold_breach",
            "available_sources": 2,
            "threshold_breach_scope": "clock_skew_only",
            "threshold_breach_estimated_offset_ms": 85.219,
            "dns_diagnostics": [
                {"source": "time.apple.com", "resolved_addrs": ["17.253.114.35"], "fake_ip_detected": False},
                {"source": "pool.ntp.org", "resolved_addrs": ["84.16.67.12"], "fake_ip_detected": False},
            ],
        },
    )
    timed_log_path.write_text(
        "\n".join(
            [
                '2026-03-15 09:18:10.711 Df timed[142:1def4d] [com.apple.timed:text] Received time Sun Mar 15 09:18:10 2026±0.09 from "NTP"',
                "2026-03-15 09:18:10.711 Df timed[142:1def4d] [com.apple.timed:data] cmd,fetchTime,num,2,result,0,mach,1067129346728,tv_sec,1773495790,tv_usec,711000,delay,0.079800,dispersion,0.000107,more,1,ip,17.253.114.35,port,123,slept,0,use_service_port,1",
                '2026-03-15 09:18:13.499 Df timed[142:1def4d] [com.apple.timed:text] Received time Sun Mar 15 09:18:13 2026±35.00 from "APNS"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_yaml(
        clash_config_path,
        {
            "rules": [
                "PROCESS-NAME,timed,DIRECT",
                "PROCESS-NAME,sntp,DIRECT",
            ],
            "tun": {"enable": True, "stack": "gvisor", "auto-route": True},
            "dns": {
                "enhanced-mode": "fake-ip",
                "fake-ip-filter": ["time.apple.com", "pool.ntp.org"],
            },
        },
    )
    _write_yaml(clash_merge_path, {})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--time-sync-probe-file",
            str(probe_path),
            "--timed-log-file",
            str(timed_log_path),
            "--clash-config-file",
            str(clash_config_path),
            "--clash-merge-file",
            str(clash_merge_path),
            "--now",
            "2026-03-15T09:20:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["classification"] == "timed_apns_fallback_ntp_reachable_flapping"
    assert payload["residual_path_hint"] == "clash_exemptions_present_os_time_service_not_recovered"
    assert payload["clash_timed_direct_rule_present"] is True
    assert payload["clash_sntp_direct_rule_present"] is True
    assert payload["timed_recent_ntp_success_detected"] is True
    assert payload["timed_recent_source_sequence"] == ["NTP", "APNS"]
    assert payload["timed_recent_source_transition_count"] == 1
    assert payload["timed_recent_ntp_fallback_seconds"] == 2.788
    assert payload["timed_recent_direct_ntp_ips"] == ["17.253.114.35"]
    assert payload["timed_recent_direct_ntp_delay_ms_min"] == 79.8
    assert payload["timed_recent_direct_ntp_delay_ms_max"] == 79.8
    assert "clash_time_exemptions_present" in payload["blocker_detail"]
    assert "recent_ntp_success_observed" in payload["blocker_detail"]
    assert "recent_source_sequence=NTP>APNS" in payload["blocker_detail"]
    assert "recent_ntp_fallback_seconds=2.788" in payload["blocker_detail"]
    assert "recent_ntp_direct_hits_present" in payload["blocker_detail"]
    assert "recent_ntp_ips=17.253.114.35" in payload["blocker_detail"]
    assert "residual_path_hint=clash_exemptions_present_os_time_service_not_recovered" in payload["blocker_detail"]
    assert "after about 2.788s" in payload["remediation_hint"]
    assert "direct NTP fetches from timed already succeeded to 17.253.114.35" in payload["remediation_hint"]
    assert "fell back to APNS again" in payload["remediation_hint"]


def test_build_system_time_sync_environment_report_prefers_latest_direct_ntp_over_earlier_fake_ip(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "time_sync"
    probe_path = artifact_dir / "20260315T014857Z_time_sync_probe.json"
    timed_log_path = tmp_path / "timed.log"
    clash_config_path = tmp_path / "clash-verge.yaml"
    clash_merge_path = tmp_path / "Merge.yaml"

    _write_json(
        probe_path,
        {
            "as_of": "2026-03-15T09:48:57Z",
            "classification": "threshold_breach",
            "available_sources": 2,
            "threshold_breach_scope": "clock_skew_only",
            "threshold_breach_estimated_offset_ms": 81.261,
            "dns_diagnostics": [
                {"source": "ntp.aliyun.com", "resolved_addrs": ["203.107.6.88"], "fake_ip_detected": False},
                {"source": "time.apple.com", "resolved_addrs": ["17.253.114.35"], "fake_ip_detected": False},
            ],
        },
    )
    timed_log_path.write_text(
        "\n".join(
            [
                '2026-03-15 17:48:44.396 Df timed[142:1def4d] [com.apple.timed:text] Received time Sun Mar 15 17:48:44 2026±0.08 from "NTP"',
                "2026-03-15 17:48:44.396 Df timed[142:1def4d] [com.apple.timed:data] cmd,fetchTime,num,2,result,0,mach,1067129346728,tv_sec,1773568124,tv_usec,403389,delay,0.079885,dispersion,0.000107,more,1,ip,198.18.0.46,port,123,slept,0,use_service_port,1",
                '2026-03-15 17:48:52.192 Df timed[142:1def4d] [com.apple.timed:text] Received time Sun Mar 15 17:48:52 2026±0.03 from "NTP"',
                "2026-03-15 17:48:52.192 Df timed[142:1def4d] [com.apple.timed:data] cmd,fetchTime,num,2,result,0,mach,1067129346729,tv_sec,1773568132,tv_usec,231939,delay,0.025536,dispersion,0.001053,more,1,ip,203.107.6.88,port,123,slept,0,use_service_port,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_yaml(
        clash_config_path,
        {
            "rules": [
                "PROCESS-NAME,timed,DIRECT",
                "PROCESS-NAME,sntp,DIRECT",
            ],
            "tun": {"enable": True, "stack": "gvisor", "auto-route": True},
            "dns": {
                "enhanced-mode": "fake-ip",
                "fake-ip-filter": ["ntp.aliyun.com", "time.apple.com"],
            },
        },
    )
    _write_yaml(clash_merge_path, {})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--time-sync-probe-file",
            str(probe_path),
            "--timed-log-file",
            str(timed_log_path),
            "--clash-config-file",
            str(clash_config_path),
            "--clash-merge-file",
            str(clash_merge_path),
            "--now",
            "2026-03-15T09:50:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["classification"] == "none"
    assert payload["timed_recent_source"] == "NTP"
    assert payload["timed_recent_ntp_fake_ip"] == "198.18.0.46"
    assert payload["timed_recent_direct_ntp_ips"] == ["203.107.6.88"]
    assert payload["latest_success_ip"] == "203.107.6.88"
    assert payload["latest_success_is_direct"] is True
    assert payload["latest_success_delay_ms"] == 25.536
    assert payload["direct_ntp_recovered"] is True
    assert payload["blocker_detail"] == ""
    assert payload["remediation_hint"] == ""
    assert payload["residual_path_hint"] == ""


def test_build_system_time_sync_environment_report_ignores_implausible_fetch_delay(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "time_sync"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"
    timed_log_path = tmp_path / "timed.log"
    clash_config_path = tmp_path / "clash-verge.yaml"
    clash_merge_path = tmp_path / "Merge.yaml"

    _write_json(probe_path, {"as_of": "2026-03-14T13:29:56Z", "dns_diagnostics": []})
    timed_log_path.write_text(
        "\n".join(
            [
                '2026-03-14 21:43:13.499 Df timed[142:1def4d] [com.apple.timed:text] Received time Sat Mar 14 21:43:12 2026±0.23 from "NTP"',
                "2026-03-14 21:43:13.499 Df timed[142:1def4d] [com.apple.timed:data] cmd,fetchTime,num,2,result,0,mach,1067129346728,tv_sec,1773495793,tv_usec,445720,delay,1803825229.566716,dispersion,0.000107,more,1,ip,198.18.0.64,port,123,slept,0,use_service_port,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_yaml(clash_config_path, {"dns": {"enhanced-mode": "fake-ip"}})
    _write_yaml(clash_merge_path, {})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--time-sync-probe-file",
            str(probe_path),
            "--timed-log-file",
            str(timed_log_path),
            "--clash-config-file",
            str(clash_config_path),
            "--clash-merge-file",
            str(clash_merge_path),
            "--now",
            "2026-03-14T21:45:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["timed_recent_fetch_rows"][-1]["ip"] == "198.18.0.64"
    assert payload["timed_recent_fetch_rows"][-1]["delay_ms"] is None
    assert payload["timed_recent_fetch_rows"][-1]["delay_unreliable"] is True
    assert payload["timed_recent_ntp_fake_ip"] == "198.18.0.64"


def test_build_system_time_sync_environment_report_marks_failed_fetch_delay_unreliable(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "time_sync"
    probe_path = artifact_dir / "20260314T132956Z_time_sync_probe.json"
    timed_log_path = tmp_path / "timed.log"
    clash_config_path = tmp_path / "clash-verge.yaml"
    clash_merge_path = tmp_path / "Merge.yaml"

    _write_json(
        probe_path,
        {
            "as_of": "2026-03-14T13:29:56Z",
            "classification": "threshold_breach",
            "available_sources": 2,
            "threshold_breach_scope": "clock_skew_only",
            "threshold_breach_estimated_offset_ms": 81.261,
            "dns_diagnostics": [
                {"source": "time.apple.com", "resolved_addrs": ["17.253.114.43"], "fake_ip_detected": False},
            ],
        },
    )
    timed_log_path.write_text(
        "\n".join(
            [
                '2026-03-14 23:57:36.999 Df timed[142:1def4d] [com.apple.timed:text] Received time Sat Mar 14 23:57:36 2026±35.00 from "APNS"',
                "2026-03-14 23:57:39.338 Df timed[142:1def4d] [com.apple.timed:data] cmd,fetchTime,num,2,result,6,mach,1067129346728,tv_sec,-2208988800,tv_usec,0,delay,-36744974.397407,dispersion,0.000107,more,1,ip,17.253.114.43,port,123,slept,0,use_service_port,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_yaml(clash_config_path, {"dns": {"enhanced-mode": "fake-ip"}})
    _write_yaml(clash_merge_path, {})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--time-sync-probe-file",
            str(probe_path),
            "--timed-log-file",
            str(timed_log_path),
            "--clash-config-file",
            str(clash_config_path),
            "--clash-merge-file",
            str(clash_merge_path),
            "--now",
            "2026-03-14T23:58:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    row = payload["timed_recent_fetch_rows"][-1]
    assert row["ip"] == "17.253.114.43"
    assert row["result"] == "6"
    assert row["delay_ms"] is None
    assert row["delay_unreliable"] is True
