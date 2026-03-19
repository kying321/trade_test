from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_cvd_semantic_snapshot.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_cvd_semantic_snapshot_reads_micro_capture(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    source_path = artifact_dir / "20260310T094850Z_micro_capture.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-10",
            "captured_at_utc": "2026-03-10T09:48:50Z",
            "status": "degraded",
            "pass": False,
            "symbols_requested": 3,
            "symbols_selected": 3,
            "cvd_semantics": {
                "context_counts": {"absorption": 1, "reversal": 1, "failed_auction": 1},
                "trust_counts": {"single_exchange_low": 3},
                "veto_hint_counts": {"low_sample_or_gap_risk": 3},
            },
            "system_time_sync": {
                "status": "degraded",
                "pass": False,
                "classification": "fake_ip_dns_intercept",
                "fake_ip_intercept_scope": "environment_wide",
                "remediation_hint": "disable fake-ip DNS interception or bypass proxy DNS for SNTP/NTP hosts; switching source hostnames is unlikely to help until environment routing is fixed, then rerun time_sync_probe",
                "fake_ip_sources": ["time.google.com", "time.cloudflare.com"],
                "ok_sources": 0,
                "available_sources": 2,
            },
            "selected_micro": [
                {
                    "symbol": "BTCUSDT",
                    "cvd_context_mode": "absorption",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 95,
                    "evidence_score": 0.61,
                    "cvd_context_note": "aggressive_flow_without_clean_price_result|time_sync_risk",
                },
                {
                    "symbol": "ETHUSDT",
                    "cvd_context_mode": "reversal",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 88,
                    "evidence_score": 0.58,
                    "cvd_context_note": "delta_and_price_move_disagree|time_sync_risk",
                },
                {
                    "symbol": "SOLUSDT",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 71,
                    "evidence_score": 0.55,
                    "cvd_context_note": "range_expanded_but_close_failed_to_hold|time_sync_risk",
                },
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
            "2026-03-10T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["as_of"] == "2026-03-10T09:48:50Z"
    assert payload["trust_counts"]["single_exchange_low"] == 3
    assert payload["watch_only_symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert payload["trend_confirmation_watch"] == []
    assert payload["reversal_absorption_watch"] == []
    assert payload["artifact_label"] == "crypto-cvd-semantic-snapshot:degraded"
    assert payload["locality_counts"]["proxy_from_current_snapshot"] == 3
    assert payload["attack_side_counts"]["balanced"] == 3
    assert payload["time_sync_status"] == "degraded"
    assert payload["time_sync_classification"] == "fake_ip_dns_intercept"
    assert payload["time_sync_intercept_scope"] == "environment_wide"
    assert payload["time_sync_blocker_detail"] == (
        "intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com"
    )
    assert "switching source hostnames is unlikely to help" in payload["time_sync_remediation_hint"]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_crypto_cvd_semantic_snapshot_reports_missing_selected_micro(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    source_path = artifact_dir / "20260310T094850Z_micro_capture.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-10",
            "captured_at_utc": "2026-03-10T09:48:50Z",
            "status": "degraded",
            "pass": False,
            "symbols_requested": 0,
            "symbols_selected": 0,
            "environment_diagnostics": {
                "status": "environment_blocked",
                "classification": "proxy_tls_and_http_access_block",
                "blocker_detail": "proxy_env_present; ssl_verify_failed=binance_spot_public:BTCUSDT",
                "remediation_hint": "fix proxy certificate trust or bypass proxy for exchange hosts, then rerun micro_capture",
            },
            "cvd_semantics": {},
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
            "2026-03-10T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "selected_micro_missing"
    assert payload["symbols"] == []
    assert payload["environment_status"] == "environment_blocked"


def test_build_crypto_cvd_semantic_snapshot_formats_threshold_breach(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    source_path = artifact_dir / "20260310T094850Z_micro_capture.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-10",
            "captured_at_utc": "2026-03-10T09:48:50Z",
            "status": "degraded",
            "pass": False,
            "symbols_requested": 1,
            "symbols_selected": 1,
            "cvd_semantics": {
                "context_counts": {"failed_auction": 1},
                "trust_counts": {"single_exchange_low": 1},
                "veto_hint_counts": {"low_sample_or_gap_risk": 1},
            },
            "system_time_sync": {
                "status": "degraded",
                "pass": False,
                "classification": "threshold_breach",
                "remediation_hint": "synchronize system clock and reduce network latency/jitter until time-sync offset and RTT stay within configured limits, then rerun time_sync_probe",
                "threshold_breach_scope": "clock_skew_and_latency",
                "threshold_breach_sources": [
                    "time.google.com:offset_abs_ms=75.464>5,rtt_ms=162.015>120",
                    "time.cloudflare.com:offset_abs_ms=79.087>5",
                ],
                "threshold_breach_offset_sources": ["time.google.com", "time.cloudflare.com"],
                "threshold_breach_latency_sources": ["time.google.com"],
                "threshold_breach_estimated_offset_ms": 77.275,
                "threshold_breach_estimated_rtt_ms": 162.015,
                "ok_sources": 0,
                "available_sources": 2,
            },
            "selected_micro": [
                {
                    "symbol": "SOLUSDT",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 71,
                    "evidence_score": 0.55,
                    "cvd_context_note": "range_expanded_but_close_failed_to_hold|time_sync_risk",
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
            "2026-03-10T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["time_sync_classification"] == "threshold_breach"
    assert payload["time_sync_threshold_breach_scope"] == "clock_skew_and_latency"
    assert payload["time_sync_threshold_breach_estimated_offset_ms"] == 77.275
    assert payload["time_sync_threshold_breach_estimated_rtt_ms"] == 162.015
    assert payload["time_sync_blocker_detail"] == (
        "scope=clock_skew_and_latency; est_offset_ms=77.275; est_rtt_ms=162.015; "
        "time.google.com:offset_abs_ms=75.464>5,rtt_ms=162.015>120; "
        "time.cloudflare.com:offset_abs_ms=79.087>5"
    )
    assert payload["time_sync_threshold_breach_sources"] == [
        "time.google.com:offset_abs_ms=75.464>5,rtt_ms=162.015>120",
        "time.cloudflare.com:offset_abs_ms=79.087>5",
    ]


def test_build_crypto_cvd_semantic_snapshot_appends_environment_report_to_threshold_breach(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    source_path = artifact_dir / "20260310T094850Z_micro_capture.json"
    env_report_path = review_dir / "20260310T095000Z_system_time_sync_environment_report.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-10",
            "captured_at_utc": "2026-03-10T09:48:50Z",
            "status": "degraded",
            "pass": False,
            "symbols_requested": 1,
            "symbols_selected": 1,
            "cvd_semantics": {
                "context_counts": {"failed_auction": 1},
                "trust_counts": {"single_exchange_low": 1},
                "veto_hint_counts": {"low_sample_or_gap_risk": 1},
            },
            "system_time_sync": {
                "status": "degraded",
                "pass": False,
                "classification": "threshold_breach",
                "remediation_hint": "synchronize system clock and reduce network latency/jitter until time-sync offset and RTT stay within configured limits, then rerun time_sync_probe",
                "threshold_breach_scope": "clock_skew_and_latency",
                "threshold_breach_sources": [
                    "time.google.com:offset_abs_ms=75.464>5,rtt_ms=162.015>120",
                ],
                "threshold_breach_estimated_offset_ms": 75.464,
                "threshold_breach_estimated_rtt_ms": 162.015,
                "ok_sources": 0,
                "available_sources": 2,
            },
            "selected_micro": [
                {
                    "symbol": "SOLUSDT",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 71,
                    "evidence_score": 0.55,
                    "cvd_context_note": "range_expanded_but_close_failed_to_hold|time_sync_risk",
                }
            ],
        },
    )
    _write_json(
        env_report_path,
        {
            "status": "ok",
            "classification": "timed_ntp_via_fake_ip",
            "blocker_detail": "timed_source=NTP; ntp_ip=198.18.0.64; delay_ms=237.656; fake_ip_hosts=time.apple.com,time.windows.com",
            "remediation_hint": "exclude macOS timed / UDP 123 from Clash TUN fake-ip handling or provide a direct NTP path, then rerun time_sync_probe",
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
            "--micro-capture-file",
            str(source_path),
            "--time-sync-environment-file",
            str(env_report_path),
            "--now",
            "2026-03-10T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["time_sync_environment_classification"] == "timed_ntp_via_fake_ip"
    assert "env=timed_ntp_via_fake_ip:timed_source=NTP; ntp_ip=198.18.0.64" in payload["time_sync_blocker_detail"]
    assert "env: exclude macOS timed / UDP 123 from Clash TUN fake-ip handling" in payload[
        "time_sync_remediation_hint"
    ]


def test_build_crypto_cvd_semantic_snapshot_prefers_newer_cleared_probe_over_stale_micro_capture(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    time_sync_dir = tmp_path / "artifacts" / "time_sync"
    source_path = artifact_dir / "20260314T145512Z_micro_capture.json"
    env_report_path = review_dir / "20260315T100157Z_system_time_sync_environment_report.json"
    probe_path = time_sync_dir / "20260315T100159Z_time_sync_probe.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-14",
            "captured_at_utc": "2026-03-14T14:55:12Z",
            "status": "degraded",
            "pass": False,
            "environment_diagnostics": {
                "status": "environment_blocked",
                "classification": "timed_apns_fallback",
                "blocker_detail": "timed_source=APNS",
                "remediation_hint": "old env hint",
            },
            "cvd_semantics": {
                "context_counts": {"continuation": 1},
                "trust_counts": {"single_exchange_ok": 1},
                "veto_hint_counts": {},
            },
            "system_time_sync": {
                "status": "degraded",
                "pass": False,
                "classification": "threshold_breach",
                "remediation_hint": "old time sync hint",
                "threshold_breach_scope": "clock_skew_only",
                "threshold_breach_sources": ["ntp.aliyun.com:offset_abs_ms=83.205>5"],
                "threshold_breach_estimated_offset_ms": 81.261,
                "ok_sources": 0,
                "available_sources": 2,
            },
            "selected_micro": [
                {
                    "symbol": "SOLUSDT",
                    "cvd_context_mode": "continuation",
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 71,
                    "evidence_score": 0.55,
                    "cvd_context_note": "buyers_attacking|time_sync_risk",
                    "micro_alignment": 0.18,
                }
            ],
        },
    )
    _write_json(
        env_report_path,
        {
            "status": "ok",
            "classification": "none",
            "blocker_detail": "",
            "remediation_hint": "",
        },
    )
    _write_json(
        probe_path,
        {
            "status": "ok",
            "pass": True,
            "classification": "none",
            "captured_at_utc": "2026-03-15T10:01:59Z",
            "ok_sources": 2,
            "available_sources": 2,
            "remediation_hint": "",
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
            "--micro-capture-file",
            str(source_path),
            "--now",
            "2026-03-15T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["time_sync_probe_override_active"] is True
    assert payload["time_sync_classification"] == "none"
    assert payload["time_sync_blocker_detail"] == ""
    assert payload["environment_classification"] == "none"
    assert payload["trend_confirmation_watch"] == ["SOLUSDT"]
    row = payload["symbols"][0]
    assert row["time_sync_ok"] is True
    assert "time_sync_risk" not in row["active_reasons"]


def test_build_crypto_cvd_semantic_snapshot_marks_drift_risk_when_reference_is_stale(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "shortline:",
                "  cvd_local_window_minutes: 15",
                "  cvd_reference_max_age_minutes: 15",
                "  cvd_drift_guard_enabled: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_path = artifact_dir / "20260310T094850Z_micro_capture.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-10",
            "captured_at_utc": "2026-03-10T09:48:50Z",
            "status": "ok",
            "pass": True,
            "symbols_requested": 1,
            "symbols_selected": 1,
            "cvd_semantics": {
                "context_counts": {"reversal": 1},
                "trust_counts": {"single_exchange_ok": 1},
                "veto_hint_counts": {},
            },
            "selected_micro": [
                {
                    "symbol": "BTCUSDT",
                    "cvd_context_mode": "reversal",
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "",
                    "time_sync_ok": True,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 95,
                    "evidence_score": 0.91,
                    "micro_alignment": -0.22,
                    "cvd_reference_age_minutes": 42,
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
            "--config",
            str(config_path),
            "--now",
            "2026-03-10T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    row = payload["symbols"][0]
    assert row["classification"] == "watch_only"
    assert row["cvd_drift_risk"] is True
    assert row["cvd_attack_side"] == "sellers"
    assert row["cvd_locality_status"] == "outside_local_window"
    assert "cvd_drift_risk" in row["active_reasons"]
