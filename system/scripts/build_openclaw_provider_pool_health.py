#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_file(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_text_file(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def parse_usage_text(text: str) -> dict[str, Any]:
    expired = re.findall(r"^\s*([A-Za-z0-9_-]+): Token expired\s*$", text, flags=re.M)
    update_available = "Update available" in text
    dashboard_match = re.search(r"Dashboard\s+│\s+([^\n│]+)", text)
    gateway_match = re.search(r"Gateway\s+│\s+([^\n]+)", text)
    gateway_service_match = re.search(r"Gateway service\s+│\s+([^\n]+)", text)
    node_service_match = re.search(r"Node service\s+│\s+([^\n]+)", text)
    agents_match = re.search(r"Agents\s+│\s+(\d+)\s+·\s+(\d+)\s+bootstrapping\s+·\s+sessions\s+(\d+)", text)
    sessions_match = re.search(r"Sessions\s+│\s+(\d+)\s+active", text)
    return {
        "expired_providers": expired,
        "update_available": update_available,
        "dashboard": (dashboard_match.group(1).strip() if dashboard_match else ""),
        "gateway_brief": (gateway_match.group(1).strip() if gateway_match else ""),
        "gateway_service_brief": (
            gateway_service_match.group(1).strip() if gateway_service_match else ""
        ),
        "node_service_brief": (
            node_service_match.group(1).strip() if node_service_match else ""
        ),
        "agents": int(agents_match.group(1)) if agents_match else 0,
        "bootstrapping_agents": int(agents_match.group(2)) if agents_match else 0,
        "sessions": int(agents_match.group(3)) if agents_match else (
            int(sessions_match.group(1)) if sessions_match else 0
        ),
    }


def parse_doctor_text(text: str) -> dict[str, Any]:
    cooldowns = []
    for profile_id, cooldown in re.findall(
        r"-\s+([A-Za-z0-9:_-]+): cooldown \(([^)]+)\)", text
    ):
        cooldowns.append({"profile_id": profile_id, "cooldown": cooldown})
    return {
        "cooldowns": cooldowns,
    }


def _normalize_profile_label(label: str) -> str:
    raw = str(label or "").strip()
    if "=" not in raw:
        return raw
    return raw.split("=", 1)[1].split(" [", 1)[0].strip()


def parse_models_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    providers = []
    duplicate_lanes = []
    for row in source.get("auth", {}).get("providers", []):
        if not isinstance(row, dict):
            continue
        provider = str(row.get("provider") or "").strip()
        profiles = row.get("profiles") if isinstance(row.get("profiles"), dict) else {}
        labels = [str(item).strip() for item in profiles.get("labels", []) if str(item).strip()]
        normalized = [_normalize_profile_label(item) for item in labels]
        unique_count = len(set(normalized))
        count = int(profiles.get("count") or len(labels) or 0)
        duplicate = count > 1 and unique_count < count
        provider_row = {
            "provider": provider,
            "effective_kind": str((row.get("effective") or {}).get("kind") or ""),
            "profile_count": count,
            "labels": labels,
            "unique_lane_count": unique_count,
            "duplicate_lane_value_detected": duplicate,
        }
        providers.append(provider_row)
        if duplicate:
            duplicate_lanes.append(
                {
                    "provider": provider,
                    "profile_count": count,
                    "unique_lane_count": unique_count,
                    "labels": labels,
                }
            )
    unusable = [
        dict(item)
        for item in source.get("auth", {}).get("unusableProfiles", [])
        if isinstance(item, dict)
    ]
    return {
        "default_model": str(source.get("defaultModel") or ""),
        "resolved_default_model": str(source.get("resolvedDefault") or ""),
        "providers": providers,
        "provider_count": len(providers),
        "duplicate_lanes": duplicate_lanes,
        "unusable_profiles": unusable,
        "unusable_profile_count": len(unusable),
    }


def parse_gateway_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    service = source.get("service") if isinstance(source.get("service"), dict) else {}
    runtime = service.get("runtime") if isinstance(service.get("runtime"), dict) else {}
    port = source.get("port") if isinstance(source.get("port"), dict) else {}
    rpc = source.get("rpc") if isinstance(source.get("rpc"), dict) else {}
    service_loaded = bool(service.get("loaded"))
    port_busy = str(port.get("status") or "").strip() == "busy"
    rpc_ok = bool(rpc.get("ok"))
    return {
        "service_loaded": service_loaded,
        "runtime_state": str(runtime.get("state") or ""),
        "runtime_sub_state": str(runtime.get("subState") or ""),
        "rpc_ok": rpc_ok,
        "port_busy": port_busy,
        "listener_count": len(port.get("listeners") or []),
        "bind_host": str((source.get("gateway") or {}).get("bindHost") or ""),
        "bind_mode": str((source.get("gateway") or {}).get("bindMode") or ""),
        "port": int(port.get("port") or 0) if str(port.get("port") or "").strip() else 0,
        "service_drift": rpc_ok and port_busy and not service_loaded,
        "service_issues": [
            str(item.get("code") or "")
            for item in (service.get("configAudit") or {}).get("issues", [])
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        ],
    }


def parse_newapi_model_probe(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    if not source:
        return {
            "checked": False,
            "base_url": "",
            "gate_status": "not_checked",
            "gate_ok": True,
            "required_effective": [],
            "available_effective": [],
            "optional_failing_models": [],
            "required_missing_models": [],
            "summary_rows": [],
        }
    gate = source.get("gate") if isinstance(source.get("gate"), dict) else {}
    summary = source.get("summary") if isinstance(source.get("summary"), list) else []
    return {
        "checked": True,
        "base_url": str(source.get("base_url") or ""),
        "gate_status": str(gate.get("status") or ""),
        "gate_ok": bool(gate.get("gate_ok")),
        "required_effective": [str(item).strip() for item in gate.get("required_effective", []) if str(item).strip()],
        "available_effective": [str(item).strip() for item in gate.get("available_effective", []) if str(item).strip()],
        "optional_failing_models": [
            str(item).strip() for item in gate.get("optional_failing_models", []) if str(item).strip()
        ],
        "required_missing_models": [
            str(item).strip() for item in gate.get("required_missing_models", []) if str(item).strip()
        ],
        "summary_rows": [
            {
                "requested_model": str(item.get("requested_model") or ""),
                "effective_model": str(item.get("effective_model") or ""),
                "tier": str(item.get("tier") or ""),
                "success": int(item.get("success") or 0),
                "failure": int(item.get("failure") or 0),
            }
            for item in summary
            if isinstance(item, dict)
        ],
    }


def parse_embeddings_probe_text(text: str) -> dict[str, Any]:
    body = str(text or "").strip()
    lower = body.lower()
    status = "unknown"
    supported = False
    detail = ""
    if not body:
        status = "not_checked"
        supported = True
        detail = "empty probe payload"
    elif "404 page not found" in lower:
        status = "unsupported_not_found"
        detail = "endpoint returned 404 page not found"
    elif '"embedding"' in body or '"data"' in body:
        status = "supported"
        supported = True
        detail = "embedding response payload detected"
    elif "unknown provider for model" in lower:
        status = "unsupported_unknown_provider"
        detail = "proxy rejected requested embedding model"
    else:
        status = "unsupported_or_unknown"
        detail = body[:200]
    return {
        "checked": status != "not_checked",
        "status": status,
        "supported": supported,
        "detail": detail,
    }


def parse_handoff(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    ready = source.get("ready_check") if isinstance(source.get("ready_check"), dict) else {}
    operator = source.get("operator_handoff") if isinstance(source.get("operator_handoff"), dict) else {}
    diagnosis = (
        operator.get("remote_live_diagnosis")
        if isinstance(operator.get("remote_live_diagnosis"), dict)
        else {}
    )
    return {
        "ready": bool(ready.get("ready")),
        "ready_reason": str(ready.get("reason") or ""),
        "ready_reasons": [str(item).strip() for item in ready.get("reasons", []) if str(item).strip()],
        "handoff_state": str(operator.get("handoff_state") or ""),
        "operator_status_triplet": str(operator.get("operator_status_triplet") or ""),
        "next_focus_area": str(operator.get("next_focus_area") or ""),
        "next_focus_reason": str(operator.get("next_focus_reason") or ""),
        "diagnosis_status": str(diagnosis.get("status") or ""),
        "diagnosis_brief": str(diagnosis.get("brief") or ""),
        "diagnosis_blocking_layers": [
            str(item).strip() for item in diagnosis.get("blocking_layers", []) if str(item).strip()
        ],
    }


def parse_rate_limit_text(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matched = [line for line in lines if "rate limit" in line.lower() or "quota" in line.lower()]
    api_limit_lines = [line for line in matched if "API rate limit reached" in line]
    timeout_lines = [line for line in matched if "possible rate limit" in line]
    return {
        "matched_line_count": len(matched),
        "api_rate_limit_reached_count": len(api_limit_lines),
        "possible_rate_limit_timeout_count": len(timeout_lines),
        "latest_line": matched[-1] if matched else "",
    }


def stability_score(
    usage: dict[str, Any],
    doctor: dict[str, Any],
    models: dict[str, Any],
    gateway: dict[str, Any],
    handoff: dict[str, Any],
    rate_limits: dict[str, Any],
    local_proxy_route: dict[str, Any],
    embeddings_route: dict[str, Any],
) -> int:
    score = 100
    if local_proxy_route.get("checked") and not local_proxy_route.get("gate_ok", True):
        score -= 25
    score -= min(30, 15 * len(usage.get("expired_providers", [])))
    score -= min(20, 10 * len(doctor.get("cooldowns", [])))
    score -= min(20, 10 * len(models.get("duplicate_lanes", [])))
    if gateway.get("service_drift"):
        score -= 20
    if usage.get("update_available"):
        score -= 5
    if not handoff.get("ready"):
        score -= 15
    if "ops_live_gate" in ",".join(handoff.get("diagnosis_blocking_layers", [])):
        score -= 10
    if embeddings_route.get("checked") and not embeddings_route.get("supported", True):
        score -= 5
    if rate_limits.get("api_rate_limit_reached_count", 0) > 0:
        score -= 15
    elif rate_limits.get("possible_rate_limit_timeout_count", 0) > 0:
        score -= 8
    return max(0, min(100, score))


def top_blocker(
    usage: dict[str, Any],
    doctor: dict[str, Any],
    models: dict[str, Any],
    gateway: dict[str, Any],
    handoff: dict[str, Any],
    rate_limits: dict[str, Any],
    local_proxy_route: dict[str, Any],
    embeddings_route: dict[str, Any],
) -> tuple[str, str, str]:
    if local_proxy_route.get("checked") and not local_proxy_route.get("gate_ok", True):
        missing = ",".join(local_proxy_route.get("required_missing_models", []))
        return (
            "local_proxy_route_failed",
            "Repair local CLIProxyAPI required model routing before relying on OpenClaw",
            f"base_url={local_proxy_route.get('base_url')}; required_missing={missing or 'none'}; gate_status={local_proxy_route.get('gate_status')}",
        )
    if usage.get("expired_providers"):
        providers = ",".join(usage["expired_providers"])
        return (
            "provider_token_expired",
            "Refresh expired provider tokens before relying on the pool",
            f"expired_providers={providers}",
        )
    if doctor.get("cooldowns"):
        items = ",".join(str(item.get("profile_id") or "") for item in doctor["cooldowns"])
        return (
            "provider_profile_cooldown",
            "Wait for provider cooldown or rotate to independent lanes",
            f"cooldown_profiles={items}",
        )
    if models.get("duplicate_lanes"):
        items = ",".join(str(item.get("provider") or "") for item in models["duplicate_lanes"])
        return (
            "provider_lane_duplication",
            "Split duplicate provider lanes into independent credentials",
            f"duplicate_providers={items}",
        )
    if gateway.get("service_drift"):
        return (
            "gateway_service_drift",
            "Install or repair gateway service supervision before trading use",
            "rpc_ok=true but service_loaded=false and port_busy=true",
        )
    if rate_limits.get("api_rate_limit_reached_count", 0) > 0:
        return (
            "provider_rate_limit_reached",
            "Reduce concurrency and add cooldown-aware lane rotation",
            f"api_rate_limit_reached_count={rate_limits.get('api_rate_limit_reached_count', 0)}",
        )
    if not handoff.get("ready"):
        layers = ",".join(handoff.get("diagnosis_blocking_layers", []))
        return (
            "remote_live_gate_blocked",
            "Clear remote live gate and risk guard before live rollout",
            f"handoff_state={handoff.get('handoff_state')}; layers={layers}",
        )
    return (
        "pool_ready_for_shadow_use",
        "Provider pool is stable enough for shadow use",
        "no critical provider or gateway blockers detected",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a source-owned OpenClaw provider pool health artifact from status/doctor/models/gateway outputs."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--status-usage-file", default="")
    parser.add_argument("--models-status-file", default="")
    parser.add_argument("--doctor-file", default="")
    parser.add_argument("--gateway-status-file", default="")
    parser.add_argument("--handoff-file", default="")
    parser.add_argument("--rate-limit-file", default="")
    parser.add_argument("--newapi-model-probe-file", default="")
    parser.add_argument("--embeddings-probe-file", default="")
    parser.add_argument("--remote-host", default="")
    parser.add_argument("--pool-label", default="openclaw_provider_pool")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    usage_path = Path(args.status_usage_file).expanduser().resolve() if str(args.status_usage_file).strip() else None
    models_path = Path(args.models_status_file).expanduser().resolve() if str(args.models_status_file).strip() else None
    doctor_path = Path(args.doctor_file).expanduser().resolve() if str(args.doctor_file).strip() else None
    gateway_path = Path(args.gateway_status_file).expanduser().resolve() if str(args.gateway_status_file).strip() else None
    handoff_path = Path(args.handoff_file).expanduser().resolve() if str(args.handoff_file).strip() else None
    rate_limit_path = Path(args.rate_limit_file).expanduser().resolve() if str(args.rate_limit_file).strip() else None
    newapi_probe_path = (
        Path(args.newapi_model_probe_file).expanduser().resolve()
        if str(args.newapi_model_probe_file).strip()
        else None
    )
    embeddings_probe_path = (
        Path(args.embeddings_probe_file).expanduser().resolve()
        if str(args.embeddings_probe_file).strip()
        else None
    )

    usage = parse_usage_text(load_text_file(usage_path))
    doctor = parse_doctor_text(load_text_file(doctor_path))
    models = parse_models_status(load_json_file(models_path))
    gateway = parse_gateway_status(load_json_file(gateway_path))
    handoff = parse_handoff(load_json_file(handoff_path))
    rate_limits = parse_rate_limit_text(load_text_file(rate_limit_path))
    local_proxy_route = parse_newapi_model_probe(load_json_file(newapi_probe_path))
    embeddings_route = parse_embeddings_probe_text(load_text_file(embeddings_probe_path))

    blocker_code, blocker_title, blocker_detail = top_blocker(
        usage, doctor, models, gateway, handoff, rate_limits, local_proxy_route, embeddings_route
    )
    score = stability_score(
        usage,
        doctor,
        models,
        gateway,
        handoff,
        rate_limits,
        local_proxy_route,
        embeddings_route,
    )
    ok = blocker_code == "pool_ready_for_shadow_use"
    status = blocker_code if ok else f"{blocker_code}_blocked"

    out: dict[str, Any] = {
        "action": "build_openclaw_provider_pool_health",
        "ok": ok,
        "status": status,
        "generated_at": fmt_utc(now_ts),
        "pool_label": str(args.pool_label or "openclaw_provider_pool"),
        "remote_host": str(args.remote_host or ""),
        "stability_score": score,
        "top_blocker_code": blocker_code,
        "top_blocker_title": blocker_title,
        "top_blocker_detail": blocker_detail,
        "pool_decision": (
            "shadow_only_until_provider_pool_stabilizes"
            if not ok
            else "shadow_ready_provider_pool_stable"
        ),
        "recommended_next_actions": [
            action
            for action in [
                "refresh_expired_provider_tokens" if usage.get("expired_providers") else "",
                "wait_or_rotate_cooldown_profiles" if doctor.get("cooldowns") else "",
                "separate_duplicate_provider_lanes" if models.get("duplicate_lanes") else "",
                "install_or_repair_gateway_service" if gateway.get("service_drift") else "",
                "repair_local_cliproxy_required_model_route"
                if local_proxy_route.get("checked") and not local_proxy_route.get("gate_ok", True)
                else "",
                "configure_embeddings_backend_for_fallbacks"
                if embeddings_route.get("checked") and not embeddings_route.get("supported", True)
                else "",
                "reduce_concurrency_and_add_jitter" if rate_limits.get("matched_line_count", 0) > 0 else "",
                "clear_remote_live_gate_and_risk_guard" if not handoff.get("ready") else "",
            ]
            if action
        ],
        "status_usage": {
            "expired_providers": usage.get("expired_providers", []),
            "update_available": usage.get("update_available", False),
            "gateway_brief": usage.get("gateway_brief", ""),
            "gateway_service_brief": usage.get("gateway_service_brief", ""),
            "node_service_brief": usage.get("node_service_brief", ""),
            "agents": usage.get("agents", 0),
            "bootstrapping_agents": usage.get("bootstrapping_agents", 0),
            "sessions": usage.get("sessions", 0),
        },
        "doctor": doctor,
        "models_status": models,
        "gateway_status": gateway,
        "local_proxy_route_health": local_proxy_route,
        "embeddings_route_health": embeddings_route,
        "remote_live_handoff": handoff,
        "rate_limit_evidence": rate_limits,
        "source_files": {
            "status_usage_file": str(usage_path) if usage_path else "",
            "models_status_file": str(models_path) if models_path else "",
            "doctor_file": str(doctor_path) if doctor_path else "",
            "gateway_status_file": str(gateway_path) if gateway_path else "",
            "handoff_file": str(handoff_path) if handoff_path else "",
            "rate_limit_file": str(rate_limit_path) if rate_limit_path else "",
            "newapi_model_probe_file": str(newapi_probe_path) if newapi_probe_path else "",
            "embeddings_probe_file": str(embeddings_probe_path) if embeddings_probe_path else "",
        },
        "artifact_status_label": status,
        "artifact_label": f"openclaw-provider-pool-health:{status}",
        "artifact_tags": [
            "openclaw",
            "provider-pool",
            "health",
            status,
        ],
        "artifact": "",
        "checksum": "",
    }

    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_openclaw_provider_pool_health.json"
    checksum_path = review_dir / f"{stamp}_openclaw_provider_pool_health_checksum.json"
    out["artifact"] = str(artifact_path)
    out["checksum"] = str(checksum_path)

    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload = {
        "generated_at": fmt_utc(now_ts),
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "files": [
            {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
                "size_bytes": int(artifact_path.stat().st_size),
            }
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
