#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
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


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dedupe_text(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_remote_execution_actor_canary_gate.json",
        "*_remote_execution_actor_canary_gate.md",
        "*_remote_execution_actor_canary_gate_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Execution Actor Canary Gate",
            "",
            f"- brief: `{text(payload.get('canary_gate_brief'))}`",
            f"- status: `{text(payload.get('canary_gate_status'))}`",
            f"- decision: `{text(payload.get('canary_gate_decision'))}`",
            f"- arm_state: `{text(payload.get('arm_state'))}`",
            f"- guarded_probe: `{text(payload.get('guarded_exec_probe_status'))}`",
            f"- transport_sla: `{text(payload.get('transport_sla_brief'))}`",
            f"- policy: `{text(payload.get('policy_brief'))}`",
            f"- ack: `{text(payload.get('ack_brief'))}`",
            f"- review_head: `{text(payload.get('review_head_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    transport_sla_path: Path,
    transport_sla_payload: dict[str, Any],
    guarded_transport_path: Path,
    guarded_transport_payload: dict[str, Any],
    actor_path: Path,
    actor_payload: dict[str, Any],
    ack_path: Path,
    ack_payload: dict[str, Any],
    policy_path: Path,
    policy_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    cross_market_path: Path,
    cross_market_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = (
        text(actor_payload.get("route_symbol"))
        or text(ack_payload.get("route_symbol"))
        or text(policy_payload.get("route_symbol"))
        or text(as_dict(cross_market_payload.get("review_head")).get("symbol"))
        or "-"
    )
    remote_market = (
        text(actor_payload.get("remote_market"))
        or text(ack_payload.get("remote_market"))
        or text(policy_payload.get("remote_market"))
        or "-"
    )
    policy_decision = text(policy_payload.get("policy_decision"))
    policy_status = text(policy_payload.get("policy_status"))
    transport_sla_status = text(transport_sla_payload.get("transport_sla_status"))
    guarded_transport_status = text(guarded_transport_payload.get("guarded_transport_status"))
    guarded_exec_probe_status = text(
        transport_sla_payload.get("guarded_exec_probe_status")
    ) or text(guarded_transport_payload.get("guarded_exec_probe_status"))
    guarded_exec_probe_artifact = text(
        transport_sla_payload.get("guarded_exec_probe_artifact")
    ) or text(guarded_transport_payload.get("guarded_exec_probe_artifact"))
    ack_status = text(ack_payload.get("ack_status"))
    blocked_gate_names = dedupe_text(
        [
            text(as_dict(row).get("name"))
            for row in as_list(live_gate_payload.get("blockers"))
            if text(as_dict(row).get("status")) == "blocked"
        ]
    )
    risk_reason_codes = dedupe_text([text(code) for code in as_list(policy_payload.get("risk_reason_codes"))])
    review_head = as_dict(cross_market_payload.get("review_head"))
    review_head_brief = ":".join(
        [
            text(review_head.get("status")) or "review",
            text(review_head.get("area")),
            text(review_head.get("symbol")),
            text(review_head.get("action")),
            text(review_head.get("priority_score")),
        ]
    ).strip(":")
    review_head_symbol = text(review_head.get("symbol"))
    review_head_action = text(review_head.get("action"))
    queue_status = text(policy_payload.get("queue_status"))
    scope_router_status = text(policy_payload.get("scope_router_status"))
    ticket_match_brief = text(policy_payload.get("ticket_match_brief"))
    alignment_status = text(cross_market_payload.get("remote_live_operator_alignment_status"))
    runtime_boundary_blocked = (
        policy_decision == "hold_requested_runtime_promotion"
        or transport_sla_status == "shadow_transport_sla_runtime_boundary_blocked_no_send"
        or guarded_transport_status == "guarded_transport_preview_runtime_boundary_blocked"
    )

    guardian_clear = not blocked_gate_names and not risk_reason_codes and policy_decision not in {
        "reject_until_guardian_clear",
        "accept_shadow_learning_only",
        "observe_without_transport",
    }
    transport_ready = transport_sla_status not in {
        "shadow_transport_sla_blocked_no_send",
        "shadow_transport_sla_learning_only_no_send",
        "shadow_transport_sla_probe_candidate_no_send",
        "shadow_transport_sla_pending_guard_clear",
        "shadow_transport_sla_unknown",
        "",
    } and guarded_transport_status not in {
        "guarded_transport_preview_blocked",
        "guarded_transport_preview_learning_only",
        "guarded_transport_preview_probe_candidate_blocked",
        "guarded_transport_preview_unknown",
        "",
    }
    candidate_trade_ready = (
        queue_status not in {"queued_wait_trade_readiness", ""}
        and "not_trade_ready" not in scope_router_status
        and not ticket_match_brief.startswith("stale_artifact")
        and review_head_action not in {"deprioritize_flow", "watch_priority_until_long_window_confirms"}
    )
    review_head_match = review_head_symbol.upper() == symbol.upper() if symbol and review_head_symbol else False

    if transport_sla_status == "shadow_transport_sla_probe_completed_no_send":
        canary_gate_status = "shadow_canary_gate_probe_completed_no_send"
        canary_gate_decision = "review_probe_evidence_before_canary_promotion"
        arm_state = "probe_completed_review_only"
    elif transport_sla_status == "shadow_transport_sla_probe_controlled_block_no_send":
        canary_gate_status = "shadow_canary_gate_probe_controlled_block"
        canary_gate_decision = "clear_probe_controlled_block_before_canary"
        arm_state = "not_armed_probe_controlled_block"
    elif transport_sla_status == "shadow_transport_sla_probe_timeout_no_send":
        canary_gate_status = "shadow_canary_gate_probe_timeout"
        canary_gate_decision = "inspect_probe_timeout_before_canary"
        arm_state = "not_armed_probe_timeout"
    elif transport_sla_status == "shadow_transport_sla_probe_error_no_send":
        canary_gate_status = "shadow_canary_gate_probe_error"
        canary_gate_decision = "inspect_probe_error_before_canary"
        arm_state = "not_armed_probe_error"
    elif transport_sla_status == "shadow_transport_sla_probe_candidate_no_send":
        canary_gate_status = "shadow_canary_gate_probe_candidate_blocked"
        canary_gate_decision = "deny_canary_until_probe_candidate_clears"
        arm_state = "not_armed_probe_candidate_blocked"
    elif runtime_boundary_blocked:
        canary_gate_status = "shadow_canary_gate_runtime_boundary_blocked"
        canary_gate_decision = "deny_canary_until_runtime_boundary_clears"
        arm_state = "not_armed_runtime_boundary_blocked"
    elif guardian_clear and transport_ready and candidate_trade_ready and review_head_match:
        canary_gate_status = "shadow_canary_gate_ready_preview"
        canary_gate_decision = "review_guarded_canary_eligibility"
        arm_state = "shadow_review_only_not_live_armed"
    elif not guardian_clear:
        canary_gate_status = "shadow_canary_gate_blocked"
        canary_gate_decision = "deny_canary_until_guardian_clear"
        arm_state = "not_armed_guardian_blocked"
    elif not transport_ready:
        canary_gate_status = "shadow_canary_gate_transport_unproven"
        canary_gate_decision = "hold_canary_until_transport_samples_exist"
        arm_state = "not_armed_transport_unproven"
    else:
        canary_gate_status = "shadow_canary_gate_candidate_not_ready"
        canary_gate_decision = "keep_shadow_only_until_route_quality_improves"
        arm_state = "not_armed_candidate_not_ready"

    canary_gate_brief = ":".join([canary_gate_status, symbol or "-", arm_state, remote_market or "-"])
    blocker_segments = dedupe_text(
        [
            text(policy_payload.get("blocker_detail")),
            ",".join(blocked_gate_names),
            ",".join(risk_reason_codes),
            text(transport_sla_payload.get("blocker_detail")),
            text(ack_payload.get("blocker_detail")),
            text(cross_market_payload.get("remote_live_takeover_gate_blocker_detail")),
            f"review_head={review_head_brief}" if review_head_brief else "",
            f"alignment={alignment_status}" if alignment_status else "",
        ]
    )
    done_when = (
        "guardian blockers clear, policy accepts a fresh route, transport SLA records non-blocked send/ack/fill samples, "
        "and the current review head remains scope-correct before any guarded canary is considered"
    )
    return {
        "action": "build_remote_execution_actor_canary_gate",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "canary_gate_status": canary_gate_status,
        "canary_gate_brief": canary_gate_brief,
        "canary_gate_decision": canary_gate_decision,
        "arm_state": arm_state,
        "next_transition": "remote_orderflow_quality_report",
        "route_symbol": symbol,
        "remote_market": remote_market,
        "policy_brief": text(policy_payload.get("policy_brief")),
        "policy_status": policy_status,
        "policy_decision": policy_decision,
        "transport_sla_brief": text(transport_sla_payload.get("transport_sla_brief")),
        "transport_sla_status": transport_sla_status,
        "guarded_exec_probe_status": guarded_exec_probe_status,
        "guarded_exec_probe_artifact": guarded_exec_probe_artifact,
        "guarded_transport_brief": text(guarded_transport_payload.get("guarded_transport_brief")),
        "guarded_transport_status": guarded_transport_status,
        "ack_brief": text(ack_payload.get("ack_brief")),
        "ack_status": ack_status,
        "review_head_brief": review_head_brief,
        "review_head_match": review_head_match,
        "queue_status": queue_status,
        "scope_router_status": scope_router_status,
        "ticket_match_brief": ticket_match_brief,
        "remote_live_operator_alignment_status": alignment_status,
        "blocked_gate_names": blocked_gate_names,
        "risk_reason_codes": risk_reason_codes,
        "guardian_clear": guardian_clear,
        "transport_ready": transport_ready,
        "candidate_trade_ready": candidate_trade_ready,
        "blocker_detail": " | ".join(blocker_segments),
        "done_when": done_when,
        "artifacts": {
            "remote_execution_transport_sla": str(transport_sla_path),
            "remote_execution_actor_guarded_transport": str(guarded_transport_path),
            "remote_execution_actor_state": str(actor_path),
            "remote_execution_ack_state": str(ack_path),
            "remote_orderflow_policy_state": str(policy_path),
            "live_gate_blocker_report": str(live_gate_path),
            "cross_market_operator_state": str(cross_market_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution actor canary gate state.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    transport_sla_path = find_latest(review_dir, "*_remote_execution_transport_sla.json")
    guarded_transport_path = find_latest(review_dir, "*_remote_execution_actor_guarded_transport.json")
    actor_path = find_latest(review_dir, "*_remote_execution_actor_state.json")
    ack_path = find_latest(review_dir, "*_remote_execution_ack_state.json")
    policy_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json")
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    cross_market_path = find_latest(review_dir, "*_cross_market_operator_state.json")
    missing = [
        name
        for name, path in (
            ("remote_execution_transport_sla", transport_sla_path),
            ("remote_execution_actor_guarded_transport", guarded_transport_path),
            ("remote_execution_actor_state", actor_path),
            ("remote_execution_ack_state", ack_path),
            ("remote_orderflow_policy_state", policy_path),
            ("live_gate_blocker_report", live_gate_path),
            ("cross_market_operator_state", cross_market_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        transport_sla_path=transport_sla_path,
        transport_sla_payload=load_json_mapping(transport_sla_path),
        guarded_transport_path=guarded_transport_path,
        guarded_transport_payload=load_json_mapping(guarded_transport_path),
        actor_path=actor_path,
        actor_payload=load_json_mapping(actor_path),
        ack_path=ack_path,
        ack_payload=load_json_mapping(ack_path),
        policy_path=policy_path,
        policy_payload=load_json_mapping(policy_path),
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path),
        cross_market_path=cross_market_path,
        cross_market_payload=load_json_mapping(cross_market_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_actor_canary_gate.json"
    markdown = review_dir / f"{stamp}_remote_execution_actor_canary_gate.md"
    checksum = review_dir / f"{stamp}_remote_execution_actor_canary_gate_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
                "generated_at_utc": fmt_utc(reference_now),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
