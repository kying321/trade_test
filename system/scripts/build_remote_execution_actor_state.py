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
        "*_remote_execution_actor_state.json",
        "*_remote_execution_actor_state.md",
        "*_remote_execution_actor_state_checksum.json",
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
            "# Remote Execution Actor State",
            "",
            f"- brief: `{text(payload.get('actor_brief'))}`",
            f"- status: `{text(payload.get('actor_status'))}`",
            f"- service: `{text(payload.get('actor_service_name'))}`",
            f"- backing_service: `{text(payload.get('backing_service_name'))}`",
            f"- transport_phase: `{text(payload.get('transport_phase'))}`",
            f"- ack: `{text(payload.get('ack_brief'))}`",
            f"- policy: `{text(payload.get('policy_brief'))}`",
            f"- executor: `{text(payload.get('executor_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    identity_path: Path,
    identity_payload: dict[str, Any],
    intent_path: Path,
    intent_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    policy_path: Path,
    policy_payload: dict[str, Any],
    executor_state_path: Path,
    executor_state_payload: dict[str, Any],
    ack_path: Path,
    ack_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = (
        text(ack_payload.get("route_symbol"))
        or text(policy_payload.get("route_symbol"))
        or text(intent_payload.get("preferred_route_symbol"))
        or "-"
    )
    action = (
        text(ack_payload.get("route_action"))
        or text(policy_payload.get("route_action"))
        or text(intent_payload.get("preferred_route_action"))
    )
    remote_market = (
        text(ack_payload.get("remote_market"))
        or text(policy_payload.get("remote_market"))
        or text(intent_payload.get("remote_market"))
        or text(identity_payload.get("ready_check_scope_market"))
        or "-"
    )
    ack_status = text(ack_payload.get("ack_status"))
    ack_decision = text(ack_payload.get("ack_decision"))
    transport_state = text(ack_payload.get("transport_state"))
    guarded_exec_probe_status = text(ack_payload.get("guarded_exec_probe_status")) or text(
        executor_state_payload.get("guarded_exec_probe_status")
    )
    guarded_exec_probe_artifact = text(ack_payload.get("guarded_exec_probe_artifact")) or text(
        executor_state_payload.get("guarded_exec_probe_artifact")
    )
    executor_status = text(executor_state_payload.get("executor_status"))
    policy_decision = text(policy_payload.get("policy_decision"))
    blocker_detail = (
        text(ack_payload.get("blocker_detail"))
        or text(policy_payload.get("blocker_detail"))
        or text(executor_state_payload.get("blocker_detail"))
        or text(intent_payload.get("blocker_detail"))
    )
    executor_runtime_boundary_status = text(executor_state_payload.get("runtime_boundary_status"))

    if guarded_exec_probe_status == "probe_completed":
        actor_status = "shadow_actor_guarded_probe_completed"
        transport_phase = "guarded_probe_completed_no_live_send"
        next_transition = "remote_execution_actor_guarded_transport"
    elif guarded_exec_probe_status in {
        "probe_controlled_block",
        "canary_controlled_block",
    } or guarded_exec_probe_status.startswith("downgraded_probe_"):
        actor_status = "shadow_actor_guarded_probe_controlled_block"
        transport_phase = "guarded_probe_controlled_block_no_live_send"
        next_transition = "remote_execution_actor_guarded_transport"
    elif guarded_exec_probe_status == "probe_timeout":
        actor_status = "shadow_actor_guarded_probe_timeout"
        transport_phase = "guarded_probe_timeout_no_live_send"
        next_transition = "remote_execution_actor_guarded_transport"
    elif guarded_exec_probe_status in {"probe_error", "probe_panic"}:
        actor_status = "shadow_actor_guarded_probe_error"
        transport_phase = "guarded_probe_error_no_live_send"
        next_transition = "remote_execution_actor_guarded_transport"
    elif (
        ack_status == "shadow_guarded_probe_candidate_ack_recorded"
        or executor_status in {"spot_live_guarded_probe_capable", "spot_live_guarded_probe_pending"}
    ):
        actor_status = "shadow_actor_guarded_probe_candidate"
        transport_phase = "guarded_probe_candidate_no_live_send"
        next_transition = "remote_execution_actor_guarded_transport"
    elif (
        ack_status == "shadow_runtime_boundary_ack_recorded"
        or executor_runtime_boundary_status == "requested_runtime_not_implemented"
        or policy_decision == "hold_requested_runtime_promotion"
    ):
        actor_status = "shadow_actor_runtime_boundary_blocked"
        transport_phase = "runtime_boundary_blocked_no_transport"
        next_transition = "remote_execution_actor_guarded_transport"
    elif ack_status == "shadow_fill_ack_recorded":
        actor_status = "shadow_actor_fill_recorded"
        transport_phase = "fill_recorded"
        next_transition = "remote_execution_transport_sla"
    elif ack_status == "shadow_learning_ack_recorded" or policy_decision == "accept_shadow_learning_only":
        actor_status = "shadow_actor_learning_only"
        transport_phase = "shadow_learning_no_transport"
        next_transition = "remote_execution_actor_guarded_transport"
    elif ack_status == "shadow_transport_pending_ack":
        actor_status = "shadow_actor_waiting_guard_clear"
        transport_phase = "candidate_pending_guard_clear"
        next_transition = "remote_execution_actor_guarded_transport"
    elif ack_status == "shadow_duplicate_ack_reused":
        actor_status = "shadow_actor_duplicate_intent"
        transport_phase = "duplicate_reused_without_send"
        next_transition = "remote_execution_actor_guarded_transport"
    elif ack_decision in {
        "record_reject_without_transport",
        "observe_without_transport",
    } or policy_decision == "reject_until_guardian_clear":
        actor_status = "shadow_actor_ready_policy_blocked"
        transport_phase = "shadow_only_no_transport"
        next_transition = "remote_execution_actor_guarded_transport"
    else:
        actor_status = "shadow_actor_state_unknown"
        transport_phase = "inspect_transport_boundary"
        next_transition = "remote_execution_actor_guarded_transport"

    actor_brief = ":".join([actor_status, symbol or "-", transport_phase, remote_market or "-"])
    done_when = (
        "fresh guardian-approved queued_ticket_ready intents advance through the actor boundary with explicit "
        "send/ack/fill truth, while guardian veto remains independent and shadow-only intents never bypass policy"
    )
    return {
        "action": "build_remote_execution_actor_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "actor_status": actor_status,
        "actor_brief": actor_brief,
        "actor_service_name": "remote_execution_actor.service",
        "backing_service_name": text(executor_state_payload.get("service_name"))
        or "openclaw-orderflow-executor.service",
        "transport_phase": transport_phase,
        "next_transition": next_transition,
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "ack_brief": text(ack_payload.get("ack_brief")),
        "ack_status": ack_status,
        "ack_decision": ack_decision,
        "transport_state": transport_state,
        "policy_brief": text(policy_payload.get("policy_brief")),
        "policy_status": text(policy_payload.get("policy_status")),
        "policy_decision": policy_decision,
        "executor_brief": text(executor_state_payload.get("executor_brief")),
        "executor_status": executor_status,
        "executor_mode": text(executor_state_payload.get("service_mode")),
        "executor_runtime_boundary_status": text(
            executor_state_payload.get("runtime_boundary_status")
        ),
        "executor_runtime_boundary_reason_codes": [
            text(code)
            for code in list(executor_state_payload.get("runtime_boundary_reason_codes") or [])
            if text(code)
        ],
        "executor_unit_preview_path": text(executor_state_payload.get("unit_preview_path")),
        "executor_heartbeat_status": text(executor_state_payload.get("heartbeat_status")),
        "guarded_exec_probe_status": guarded_exec_probe_status,
        "guarded_exec_probe_artifact": guarded_exec_probe_artifact,
        "intent_brief": text(intent_payload.get("queue_brief")),
        "journal_brief": text(journal_payload.get("journal_brief")),
        "journal_status": text(journal_payload.get("journal_status")),
        "idempotency_key_brief": text(ack_payload.get("idempotency_key"))
        or text(executor_state_payload.get("idempotency_key_brief"))
        or text(journal_payload.get("last_entry_key")),
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "remote_execution_identity_state": str(identity_path),
            "remote_intent_queue": str(intent_path),
            "remote_execution_journal": str(journal_path),
            "remote_orderflow_policy_state": str(policy_path),
            "openclaw_orderflow_executor_state": str(executor_state_path),
            "remote_execution_ack_state": str(ack_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution actor state.")
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
    identity_path = find_latest(review_dir, "*_remote_execution_identity_state.json")
    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    journal_path = find_latest(review_dir, "*_remote_execution_journal.json")
    policy_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json")
    executor_state_path = find_latest(review_dir, "*_openclaw_orderflow_executor_state.json")
    ack_path = find_latest(review_dir, "*_remote_execution_ack_state.json")
    missing = [
        name
        for name, path in (
            ("remote_execution_identity_state", identity_path),
            ("remote_intent_queue", intent_path),
            ("remote_execution_journal", journal_path),
            ("remote_orderflow_policy_state", policy_path),
            ("openclaw_orderflow_executor_state", executor_state_path),
            ("remote_execution_ack_state", ack_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        identity_path=identity_path,
        identity_payload=load_json_mapping(identity_path),
        intent_path=intent_path,
        intent_payload=load_json_mapping(intent_path),
        journal_path=journal_path,
        journal_payload=load_json_mapping(journal_path),
        policy_path=policy_path,
        policy_payload=load_json_mapping(policy_path),
        executor_state_path=executor_state_path,
        executor_state_payload=load_json_mapping(executor_state_path),
        ack_path=ack_path,
        ack_payload=load_json_mapping(ack_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_actor_state.json"
    markdown = review_dir / f"{stamp}_remote_execution_actor_state.md"
    checksum = review_dir / f"{stamp}_remote_execution_actor_state_checksum.json"

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
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
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
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
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
