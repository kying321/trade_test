from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_openclaw_provider_pool_health.py"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def test_build_openclaw_provider_pool_health_detects_expiry_duplication_and_service_drift(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    usage_path = tmp_path / "usage.txt"
    doctor_path = tmp_path / "doctor.txt"
    models_path = tmp_path / "models.json"
    gateway_path = tmp_path / "gateway.json"
    handoff_path = tmp_path / "handoff.json"
    rate_limit_path = tmp_path / "rate.txt"

    _write(
        usage_path,
        "\n".join(
            [
                "OpenClaw status",
                "Usage:",
                "  Antigravity: Token expired",
                "  Codex: Token expired",
                "Update available (npm 2026.3.13). Run: openclaw update",
                "│ Agents          │ 3 · 2 bootstrapping · sessions 40 · default main active 9m ago │",
                "│ Gateway service │ systemd not installed │",
            ]
        ),
    )
    _write(
        doctor_path,
        "\n".join(
            [
                "OpenClaw doctor",
                "- openai:x666_backup: cooldown (50m) — Wait for cooldown or switch provider.",
            ]
        ),
    )
    _write_json(
        models_path,
        {
            "defaultModel": "openai/gpt-5.4",
            "resolvedDefault": "openai/gpt-5.4",
            "auth": {
                "providers": [
                    {
                        "provider": "openai",
                        "effective": {"kind": "profiles"},
                        "profiles": {
                            "count": 2,
                            "labels": [
                                "openai:x666_primary=token:sk-AAAA",
                                "openai:x666_backup=token:sk-AAAA [cooldown 50m]",
                            ],
                        },
                    }
                ],
                "unusableProfiles": [
                    {
                        "profileId": "openai:x666_backup",
                        "provider": "openai",
                        "kind": "cooldown",
                    }
                ],
            },
        },
    )
    _write_json(
        gateway_path,
        {
            "service": {"loaded": False, "runtime": {"state": "inactive", "subState": "dead"}},
            "port": {
                "port": 18790,
                "status": "busy",
                "listeners": [{"pid": 1, "command": "openclaw-gateway"}],
            },
            "rpc": {"ok": True},
            "gateway": {"bindHost": "127.0.0.1", "bindMode": "loopback"},
        },
    )
    _write_json(
        handoff_path,
        {
            "ready_check": {
                "ready": False,
                "reason": "portfolio_margin_um_read_only_mode",
                "reasons": ["risk_guard_blocked", "ops_live_gate_blocked"],
            },
            "operator_handoff": {
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
                "next_focus_area": "gate",
                "next_focus_reason": "ops_live_gate_blocked",
                "remote_live_diagnosis": {
                    "status": "profitability_confirmed_but_auto_live_blocked",
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                    "blocking_layers": ["ops_live_gate", "risk_guard"],
                },
            },
        },
    )
    _write(
        rate_limit_path,
        "\n".join(
            [
                "Profile openai:x666_primary timed out (possible rate limit). Trying next account...",
                "⚠️ API rate limit reached. Please try again later.",
            ]
        ),
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--status-usage-file",
            str(usage_path),
            "--models-status-file",
            str(models_path),
            "--doctor-file",
            str(doctor_path),
            "--gateway-status-file",
            str(gateway_path),
            "--handoff-file",
            str(handoff_path),
            "--rate-limit-file",
            str(rate_limit_path),
            "--remote-host",
            "43.153.148.242",
            "--now",
            "2026-03-16T14:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["top_blocker_code"] == "provider_token_expired"
    assert payload["status_usage"]["expired_providers"] == ["Antigravity", "Codex"]
    assert payload["doctor"]["cooldowns"][0]["profile_id"] == "openai:x666_backup"
    assert payload["models_status"]["duplicate_lanes"][0]["provider"] == "openai"
    assert payload["gateway_status"]["service_drift"] is True
    assert payload["rate_limit_evidence"]["api_rate_limit_reached_count"] == 1
    assert payload["remote_live_handoff"]["handoff_state"] == "ops_live_gate_blocked"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_openclaw_provider_pool_health_reports_shadow_ready_when_clean(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    usage_path = tmp_path / "usage.txt"
    doctor_path = tmp_path / "doctor.txt"
    models_path = tmp_path / "models.json"
    gateway_path = tmp_path / "gateway.json"
    handoff_path = tmp_path / "handoff.json"

    _write(
        usage_path,
        "\n".join(
            [
                "OpenClaw status",
                "│ Agents          │ 1 · 0 bootstrapping · sessions 2 · default main active 1m ago │",
                "│ Gateway service │ systemd installed │",
            ]
        ),
    )
    _write(doctor_path, "OpenClaw doctor\n- No channel security warnings detected.\n")
    _write_json(
        models_path,
        {
            "defaultModel": "openai/gpt-5.4",
            "resolvedDefault": "openai/gpt-5.4",
            "auth": {
                "providers": [
                    {
                        "provider": "openai",
                        "effective": {"kind": "profiles"},
                        "profiles": {
                            "count": 2,
                            "labels": [
                                "openai:team01=token:sk-AAAA",
                                "openai:team02=token:sk-BBBB",
                            ],
                        },
                    }
                ],
                "unusableProfiles": [],
            },
        },
    )
    _write_json(
        gateway_path,
        {
            "service": {"loaded": True, "runtime": {"state": "active", "subState": "running"}},
            "port": {"port": 18790, "status": "busy", "listeners": [{"pid": 2}]},
            "rpc": {"ok": True},
            "gateway": {"bindHost": "127.0.0.1", "bindMode": "loopback"},
        },
    )
    _write_json(
        handoff_path,
        {
            "ready_check": {"ready": True, "reason": "", "reasons": []},
            "operator_handoff": {
                "handoff_state": "ready_for_canary",
                "operator_status_triplet": "runtime-ok / gate-ok / risk-guard-ok",
                "next_focus_area": "canary",
                "next_focus_reason": "ready_for_canary",
                "remote_live_diagnosis": {
                    "status": "formal_live_possible",
                    "brief": "formal_live_possible:portfolio_margin_um",
                    "blocking_layers": [],
                },
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--status-usage-file",
            str(usage_path),
            "--models-status-file",
            str(models_path),
            "--doctor-file",
            str(doctor_path),
            "--gateway-status-file",
            str(gateway_path),
            "--handoff-file",
            str(handoff_path),
            "--now",
            "2026-03-16T14:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["top_blocker_code"] == "pool_ready_for_shadow_use"
    assert payload["pool_decision"] == "shadow_ready_provider_pool_stable"
    assert payload["models_status"]["duplicate_lanes"] == []
    assert payload["gateway_status"]["service_drift"] is False


def test_build_openclaw_provider_pool_health_captures_local_proxy_route_and_embeddings_gap(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    usage_path = tmp_path / "usage.txt"
    doctor_path = tmp_path / "doctor.txt"
    models_path = tmp_path / "models.json"
    gateway_path = tmp_path / "gateway.json"
    handoff_path = tmp_path / "handoff.json"
    probe_path = tmp_path / "newapi_probe.json"
    embeddings_probe_path = tmp_path / "embeddings.txt"

    _write(
        usage_path,
        "\n".join(
            [
                "OpenClaw status",
                "│ Agents          │ 1 · 0 bootstrapping · sessions 2 · default main active 1m ago │",
                "│ Gateway service │ launchd installed │",
            ]
        ),
    )
    _write(doctor_path, "OpenClaw doctor\n- No channel security warnings detected.\n")
    _write_json(
        models_path,
        {
            "defaultModel": "openai/gpt-5.4",
            "resolvedDefault": "openai/gpt-5.4",
            "auth": {
                "providers": [
                    {
                        "provider": "openai",
                        "effective": {"kind": "profiles"},
                        "profiles": {
                            "count": 2,
                            "labels": [
                                "openai:cliproxy_primary=token:sk-LOCALA",
                                "openai:cliproxy_backup=token:sk-LOCALB",
                            ],
                        },
                    }
                ],
                "unusableProfiles": [],
            },
        },
    )
    _write_json(
        gateway_path,
        {
            "service": {"loaded": True, "runtime": {"state": "active", "subState": "running"}},
            "port": {"port": 18789, "status": "busy", "listeners": [{"pid": 2}]},
            "rpc": {"ok": True},
            "gateway": {"bindHost": "127.0.0.1", "bindMode": "loopback"},
        },
    )
    _write_json(
        handoff_path,
        {
            "ready_check": {"ready": True, "reason": "", "reasons": []},
            "operator_handoff": {
                "handoff_state": "ready_for_canary",
                "operator_status_triplet": "runtime-ok / gate-ok / risk-guard-ok",
                "next_focus_area": "canary",
                "next_focus_reason": "ready_for_canary",
                "remote_live_diagnosis": {
                    "status": "formal_live_possible",
                    "brief": "formal_live_possible:portfolio_margin_um",
                    "blocking_layers": [],
                },
            },
        },
    )
    _write_json(
        probe_path,
        {
            "base_url": "http://127.0.0.1:8317",
            "gate": {
                "status": "pass",
                "gate_ok": True,
                "required_effective": ["gpt-5.4"],
                "available_effective": ["gpt-5.4", "gemini-3.1-pro-preview-bs"],
                "required_missing_models": [],
                "optional_failing_models": [],
            },
            "summary": [
                {
                    "requested_model": "gpt-5.4",
                    "effective_model": "gpt-5.4",
                    "tier": "required",
                    "success": 1,
                    "failure": 0,
                }
            ],
        },
    )
    _write(embeddings_probe_path, "404 page not found\n")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--status-usage-file",
            str(usage_path),
            "--models-status-file",
            str(models_path),
            "--doctor-file",
            str(doctor_path),
            "--gateway-status-file",
            str(gateway_path),
            "--handoff-file",
            str(handoff_path),
            "--newapi-model-probe-file",
            str(probe_path),
            "--embeddings-probe-file",
            str(embeddings_probe_path),
            "--now",
            "2026-03-17T10:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["top_blocker_code"] == "pool_ready_for_shadow_use"
    assert payload["local_proxy_route_health"]["base_url"] == "http://127.0.0.1:8317"
    assert payload["local_proxy_route_health"]["gate_ok"] is True
    assert payload["embeddings_route_health"]["status"] == "unsupported_not_found"
    assert "configure_embeddings_backend_for_fallbacks" in payload["recommended_next_actions"]
