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
        "*_remote_execution_ack_state.json",
        "*_remote_execution_ack_state.md",
        "*_remote_execution_ack_state_checksum.json",
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
            "# Remote Execution Ack State",
            "",
            f"- brief: `{text(payload.get('ack_brief'))}`",
            f"- status: `{text(payload.get('ack_status'))}`",
            f"- decision: `{text(payload.get('ack_decision'))}`",
            f"- transport_state: `{text(payload.get('transport_state'))}`",
            f"- fill_status: `{text(payload.get('fill_status'))}`",
            f"- guarded_probe: `{text(payload.get('guarded_exec_probe_status'))}`",
            f"- policy: `{text(payload.get('policy_brief'))}`",
            f"- executor: `{text(payload.get('executor_brief'))}`",
            f"- heartbeat: `{text(payload.get('heartbeat_brief'))}`",
            f"- idempotency: `{text(payload.get('idempotency_key'))}` ({text(payload.get('idempotency_status'))})",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    policy_path: Path,
    policy_payload: dict[str, Any],
    executor_state_path: Path,
    executor_state_payload: dict[str, Any],
    heartbeat_path: Path,
    heartbeat_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = (
        text(policy_payload.get("route_symbol"))
        or text(heartbeat_payload.get("intent_symbol"))
        or text(journal_payload.get("intent_symbol"))
        or "-"
    )
    action = (
        text(policy_payload.get("route_action"))
        or text(heartbeat_payload.get("intent_action"))
        or text(journal_payload.get("intent_action"))
    )
    remote_market = (
        text(policy_payload.get("remote_market"))
        or text(heartbeat_payload.get("remote_market"))
        or text(journal_payload.get("remote_market"))
    )
    policy_status = text(policy_payload.get("policy_status"))
    policy_decision = text(policy_payload.get("policy_decision"))
    policy_recommendation = text(policy_payload.get("policy_recommendation"))
    executor_status = text(executor_state_payload.get("executor_status"))
    executor_brief = text(executor_state_payload.get("executor_brief"))
    heartbeat_status = text(heartbeat_payload.get("executor_status"))
    heartbeat_brief = text(heartbeat_payload.get("executor_brief"))
    executor_action = text(heartbeat_payload.get("executor_action"))
    executor_runtime_boundary_status = text(
        executor_state_payload.get("runtime_boundary_status")
    ) or text(heartbeat_payload.get("executor_runtime_boundary_status"))
    guarded_exec_probe_status = text(heartbeat_payload.get("guarded_exec_probe_status")) or text(
        executor_state_payload.get("guarded_exec_probe_status")
    )
    guarded_exec_probe_artifact = text(
        heartbeat_payload.get("guarded_exec_probe_artifact")
    ) or text(executor_state_payload.get("guarded_exec_probe_artifact"))
    idempotency_key = (
        text(heartbeat_payload.get("idempotency_key"))
        or text(executor_state_payload.get("idempotency_key_brief"))
        or text(journal_payload.get("last_entry_key"))
        or text(journal_payload.get("entry_key"))
    )
    idempotency_status = text(heartbeat_payload.get("idempotency_status"))
    journal_entry_key = text(journal_payload.get("last_entry_key")) or text(
        journal_payload.get("entry_key")
    )
    fill_status = text(heartbeat_payload.get("fill_status")) or text(
        journal_payload.get("fill_status")
    )
    blocker_detail = text(policy_payload.get("blocker_detail")) or text(
        heartbeat_payload.get("blocker_detail")
    ) or text(journal_payload.get("blocker_detail"))

    if fill_status and not fill_status.startswith("no_fill_"):
        ack_status = "shadow_fill_ack_recorded"
        ack_decision = "record_fill_ack"
        transport_state = "fill_recorded"
        ack_reason = fill_status
    elif guarded_exec_probe_status == "probe_completed":
        ack_status = "shadow_guarded_probe_ack_recorded"
        ack_decision = "record_guarded_probe_without_live_transport"
        transport_state = "probe_completed_no_live_send"
        ack_reason = guarded_exec_probe_status
    elif guarded_exec_probe_status in {
        "probe_controlled_block",
        "canary_controlled_block",
    } or guarded_exec_probe_status.startswith("downgraded_probe_"):
        ack_status = "shadow_guarded_probe_blocked_ack_recorded"
        ack_decision = "record_guarded_probe_controlled_block_without_live_transport"
        transport_state = "probe_controlled_block_no_live_send"
        ack_reason = guarded_exec_probe_status
    elif guarded_exec_probe_status == "probe_timeout":
        ack_status = "shadow_guarded_probe_timeout_ack_recorded"
        ack_decision = "record_guarded_probe_timeout_without_live_transport"
        transport_state = "probe_timeout_no_live_send"
        ack_reason = guarded_exec_probe_status
    elif guarded_exec_probe_status in {"probe_error", "probe_panic"}:
        ack_status = "shadow_guarded_probe_error_ack_recorded"
        ack_decision = "record_guarded_probe_error_without_live_transport"
        transport_state = "probe_error_no_live_send"
        ack_reason = guarded_exec_probe_status
    elif (
        executor_runtime_boundary_status == "guarded_probe_only_runtime"
        or executor_status in {"spot_live_guarded_probe_capable", "spot_live_guarded_probe_pending"}
    ):
        ack_status = "shadow_guarded_probe_candidate_ack_recorded"
        ack_decision = "record_guarded_probe_candidate_without_transport"
        transport_state = "probe_candidate_blocked_no_live_send"
        ack_reason = guarded_exec_probe_status or executor_action or policy_recommendation or executor_status
    elif (
        policy_decision == "hold_requested_runtime_promotion"
        or executor_action == "idle_requested_mode_not_implemented"
        or executor_runtime_boundary_status == "requested_runtime_not_implemented"
    ):
        ack_status = "shadow_runtime_boundary_ack_recorded"
        ack_decision = "record_runtime_boundary_block_without_transport"
        transport_state = "not_sent_runtime_boundary_blocked"
        ack_reason = (
            policy_recommendation
            or executor_action
            or executor_runtime_boundary_status
            or policy_decision
        )
    elif policy_decision == "accept_shadow_learning_only":
        ack_status = "shadow_learning_ack_recorded"
        ack_decision = "record_learning_without_transport"
        transport_state = "not_sent_learning_only"
        ack_reason = policy_recommendation or policy_decision or policy_status
    elif policy_decision == "reject_until_guardian_clear":
        ack_status = "shadow_no_send_ack_recorded"
        ack_decision = "record_reject_without_transport"
        transport_state = "not_sent_policy_blocked"
        ack_reason = policy_recommendation or policy_decision or policy_status
    elif idempotency_status == "duplicate_intent_seen":
        ack_status = "shadow_duplicate_ack_reused"
        ack_decision = "reuse_previous_ack_state"
        transport_state = "not_sent_duplicate_intent"
        ack_reason = idempotency_status
    elif executor_action == "shadow_ready_wait_guard_clear":
        ack_status = "shadow_transport_pending_ack"
        ack_decision = "wait_for_transport_ack"
        transport_state = "ready_not_sent_guard_clear_pending"
        ack_reason = executor_action
    elif executor_action == "observe_only_no_transport":
        ack_status = "shadow_no_send_ack_recorded"
        ack_decision = "observe_without_transport"
        transport_state = "not_sent_shadow_observe_only"
        ack_reason = executor_action
    else:
        ack_status = "shadow_ack_state_unknown"
        ack_decision = "inspect_executor_transport_path"
        transport_state = "unknown"
        ack_reason = policy_decision or executor_action or executor_status or "unknown"

    ack_brief = ":".join([ack_status, symbol or "-", transport_state, fill_status or "-"])
    done_when = (
        "ack state advances only after a fresh guardian-approved intent reaches explicit transport ack/fill transitions; until then keep reject/no-send decisions source-owned"
    )
    return {
        "action": "build_remote_execution_ack_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "ack_status": ack_status,
        "ack_brief": ack_brief,
        "ack_decision": ack_decision,
        "ack_reason": ack_reason,
        "transport_state": transport_state,
        "fill_status": fill_status or "unknown",
        "guarded_exec_probe_status": guarded_exec_probe_status,
        "guarded_exec_probe_artifact": guarded_exec_probe_artifact,
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "policy_brief": text(policy_payload.get("policy_brief")),
        "policy_status": policy_status,
        "policy_decision": policy_decision,
        "policy_recommendation": policy_recommendation,
        "executor_brief": executor_brief,
        "executor_status": executor_status,
        "executor_runtime_boundary_status": executor_runtime_boundary_status,
        "heartbeat_brief": heartbeat_brief,
        "heartbeat_status": heartbeat_status,
        "executor_action": executor_action,
        "idempotency_key": idempotency_key,
        "idempotency_status": idempotency_status,
        "journal_brief": text(journal_payload.get("journal_brief")),
        "journal_status": text(journal_payload.get("journal_status")),
        "journal_entry_key": journal_entry_key or idempotency_key,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "remote_orderflow_policy_state": str(policy_path),
            "openclaw_orderflow_executor_state": str(executor_state_path),
            "openclaw_orderflow_executor_heartbeat": str(heartbeat_path),
            "remote_execution_journal": str(journal_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution ack state.")
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
    policy_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json")
    executor_state_path = find_latest(review_dir, "*_openclaw_orderflow_executor_state.json")
    heartbeat_path = find_latest(review_dir, "*_openclaw_orderflow_executor_heartbeat.json")
    journal_path = find_latest(review_dir, "*_remote_execution_journal.json")
    missing = [
        name
        for name, path in (
            ("remote_orderflow_policy_state", policy_path),
            ("openclaw_orderflow_executor_state", executor_state_path),
            ("openclaw_orderflow_executor_heartbeat", heartbeat_path),
            ("remote_execution_journal", journal_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        policy_path=policy_path,
        policy_payload=load_json_mapping(policy_path),
        executor_state_path=executor_state_path,
        executor_state_payload=load_json_mapping(executor_state_path),
        heartbeat_path=heartbeat_path,
        heartbeat_payload=load_json_mapping(heartbeat_path),
        journal_path=journal_path,
        journal_payload=load_json_mapping(journal_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_ack_state.json"
    markdown = review_dir / f"{stamp}_remote_execution_ack_state.md"
    checksum = review_dir / f"{stamp}_remote_execution_ack_state_checksum.json"

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
