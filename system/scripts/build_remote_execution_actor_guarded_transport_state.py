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
        "*_remote_execution_actor_guarded_transport.json",
        "*_remote_execution_actor_guarded_transport.md",
        "*_remote_execution_actor_guarded_transport_checksum.json",
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
            "# Remote Execution Actor Guarded Transport State",
            "",
            f"- brief: `{text(payload.get('guarded_transport_brief'))}`",
            f"- status: `{text(payload.get('guarded_transport_status'))}`",
            f"- decision: `{text(payload.get('guarded_transport_decision'))}`",
            f"- send_state: `{text(payload.get('send_state'))}`",
            f"- guarded_probe: `{text(payload.get('guarded_exec_probe_status'))}`",
            f"- actor: `{text(payload.get('actor_brief'))}`",
            f"- policy: `{text(payload.get('policy_brief'))}`",
            f"- ack: `{text(payload.get('ack_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    actor_path: Path,
    actor_payload: dict[str, Any],
    policy_path: Path,
    policy_payload: dict[str, Any],
    ack_path: Path,
    ack_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = (
        text(actor_payload.get("route_symbol"))
        or text(ack_payload.get("route_symbol"))
        or text(policy_payload.get("route_symbol"))
        or "-"
    )
    remote_market = (
        text(actor_payload.get("remote_market"))
        or text(ack_payload.get("remote_market"))
        or text(policy_payload.get("remote_market"))
        or "-"
    )
    actor_status = text(actor_payload.get("actor_status"))
    actor_phase = text(actor_payload.get("transport_phase"))
    policy_decision = text(policy_payload.get("policy_decision"))
    ack_status = text(ack_payload.get("ack_status"))
    ack_decision = text(ack_payload.get("ack_decision"))
    guarded_exec_probe_status = text(actor_payload.get("guarded_exec_probe_status")) or text(
        ack_payload.get("guarded_exec_probe_status")
    )
    guarded_exec_probe_artifact = text(actor_payload.get("guarded_exec_probe_artifact")) or text(
        ack_payload.get("guarded_exec_probe_artifact")
    )

    if actor_phase == "guarded_probe_completed_no_live_send":
        guarded_transport_status = "guarded_transport_preview_probe_completed"
        guarded_transport_decision = "keep_transport_disarmed_after_guarded_probe"
        send_state = "probe_completed_no_live_send"
    elif actor_phase == "guarded_probe_controlled_block_no_live_send":
        guarded_transport_status = "guarded_transport_preview_probe_controlled_block"
        guarded_transport_decision = "keep_transport_disarmed_after_guarded_probe_controlled_block"
        send_state = "probe_controlled_block_no_live_send"
    elif actor_phase == "guarded_probe_timeout_no_live_send":
        guarded_transport_status = "guarded_transport_preview_probe_timeout"
        guarded_transport_decision = "keep_transport_disarmed_after_guarded_probe_timeout"
        send_state = "probe_timeout_no_live_send"
    elif actor_phase == "guarded_probe_error_no_live_send":
        guarded_transport_status = "guarded_transport_preview_probe_error"
        guarded_transport_decision = "keep_transport_disarmed_after_guarded_probe_error"
        send_state = "probe_error_no_live_send"
    elif (
        actor_phase == "guarded_probe_candidate_no_live_send"
        or ack_status == "shadow_guarded_probe_candidate_ack_recorded"
    ):
        guarded_transport_status = "guarded_transport_preview_probe_candidate_blocked"
        guarded_transport_decision = "keep_transport_disarmed_probe_candidate"
        send_state = "probe_candidate_blocked_no_live_send"
    elif (
        actor_phase == "runtime_boundary_blocked_no_transport"
        or ack_status == "shadow_runtime_boundary_ack_recorded"
        or policy_decision == "hold_requested_runtime_promotion"
    ):
        guarded_transport_status = "guarded_transport_preview_runtime_boundary_blocked"
        guarded_transport_decision = "do_not_arm_transport_runtime_boundary_blocked"
        send_state = "not_armed_runtime_boundary_blocked"
    elif actor_phase == "shadow_learning_no_transport" or policy_decision == "accept_shadow_learning_only":
        guarded_transport_status = "guarded_transport_preview_learning_only"
        guarded_transport_decision = "keep_transport_disarmed_learning_only"
        send_state = "not_armed_learning_only"
    elif actor_phase == "shadow_only_no_transport" or policy_decision == "reject_until_guardian_clear":
        guarded_transport_status = "guarded_transport_preview_blocked"
        guarded_transport_decision = "do_not_arm_transport_policy_blocked"
        send_state = "not_armed_policy_blocked"
    elif actor_phase == "candidate_pending_guard_clear" or ack_status == "shadow_transport_pending_ack":
        guarded_transport_status = "guarded_transport_preview_pending_guard_clear"
        guarded_transport_decision = "wait_for_guardian_clear_before_send"
        send_state = "armed_shadow_wait_guard_clear"
    elif actor_status == "shadow_actor_fill_recorded":
        guarded_transport_status = "guarded_transport_preview_fill_recorded"
        guarded_transport_decision = "observe_fill_without_transport_change"
        send_state = "fill_recorded"
    else:
        guarded_transport_status = "guarded_transport_preview_unknown"
        guarded_transport_decision = "inspect_actor_boundary"
        send_state = "unknown"

    guarded_transport_brief = ":".join(
        [guarded_transport_status, symbol or "-", send_state, remote_market or "-"]
    )
    done_when = (
        "guardian-approved intents can cross the actor transport boundary with explicit send attempt metadata, "
        "while policy veto still prevents blocked routes from arming transport"
    )
    return {
        "action": "build_remote_execution_actor_guarded_transport_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "guarded_transport_status": guarded_transport_status,
        "guarded_transport_brief": guarded_transport_brief,
        "guarded_transport_decision": guarded_transport_decision,
        "send_state": send_state,
        "next_transition": "remote_execution_transport_sla",
        "route_symbol": symbol,
        "remote_market": remote_market,
        "actor_brief": text(actor_payload.get("actor_brief")),
        "actor_status": actor_status,
        "actor_service_name": text(actor_payload.get("actor_service_name")),
        "actor_transport_phase": actor_phase,
        "policy_brief": text(policy_payload.get("policy_brief")),
        "policy_status": text(policy_payload.get("policy_status")),
        "policy_decision": policy_decision,
        "ack_brief": text(ack_payload.get("ack_brief")),
        "ack_status": ack_status,
        "ack_decision": ack_decision,
        "guarded_exec_probe_status": guarded_exec_probe_status,
        "guarded_exec_probe_artifact": guarded_exec_probe_artifact,
        "blocker_detail": text(actor_payload.get("blocker_detail"))
        or text(ack_payload.get("blocker_detail"))
        or text(policy_payload.get("blocker_detail")),
        "done_when": done_when,
        "artifacts": {
            "remote_execution_actor_state": str(actor_path),
            "remote_orderflow_policy_state": str(policy_path),
            "remote_execution_ack_state": str(ack_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution actor guarded transport state.")
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
    actor_path = find_latest(review_dir, "*_remote_execution_actor_state.json")
    policy_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json")
    ack_path = find_latest(review_dir, "*_remote_execution_ack_state.json")
    missing = [
        name
        for name, path in (
            ("remote_execution_actor_state", actor_path),
            ("remote_orderflow_policy_state", policy_path),
            ("remote_execution_ack_state", ack_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        actor_path=actor_path,
        actor_payload=load_json_mapping(actor_path),
        policy_path=policy_path,
        policy_payload=load_json_mapping(policy_path),
        ack_path=ack_path,
        ack_payload=load_json_mapping(ack_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_actor_guarded_transport.json"
    markdown = review_dir / f"{stamp}_remote_execution_actor_guarded_transport.md"
    checksum = review_dir / f"{stamp}_remote_execution_actor_guarded_transport_checksum.json"

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
