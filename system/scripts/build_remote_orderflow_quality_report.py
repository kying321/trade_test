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
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")


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


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name))


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
        "*_remote_orderflow_quality_report.json",
        "*_remote_orderflow_quality_report.md",
        "*_remote_orderflow_quality_report_checksum.json",
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
            "# Remote Orderflow Quality Report",
            "",
            f"- brief: `{text(payload.get('quality_brief'))}`",
            f"- status: `{text(payload.get('quality_status'))}`",
            f"- recommendation: `{text(payload.get('quality_recommendation'))}`",
            f"- score: `{text(payload.get('quality_score'))}`",
            f"- shadow_learning_score: `{text(payload.get('shadow_learning_score'))}`",
            f"- execution_readiness_score: `{text(payload.get('execution_readiness_score'))}`",
            f"- transport_observability_score: `{text(payload.get('transport_observability_score'))}`",
            f"- canary_gate: `{text(payload.get('canary_gate_brief'))}`",
            f"- feedback: `{text(payload.get('feedback_brief'))}`",
            f"- policy: `{text(payload.get('policy_brief'))}`",
            f"- ack: `{text(payload.get('ack_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def clamp_score(value: int) -> int:
    return max(0, min(100, value))


def build_payload(
    *,
    feedback_path: Path,
    feedback_payload: dict[str, Any],
    policy_path: Path,
    policy_payload: dict[str, Any],
    ack_path: Path,
    ack_payload: dict[str, Any],
    actor_path: Path,
    actor_payload: dict[str, Any],
    transport_sla_path: Path,
    transport_sla_payload: dict[str, Any],
    canary_gate_path: Path,
    canary_gate_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = (
        text(canary_gate_payload.get("route_symbol"))
        or text(feedback_payload.get("route_symbol"))
        or text(policy_payload.get("route_symbol"))
        or text(journal_payload.get("intent_symbol"))
        or "-"
    )
    remote_market = (
        text(canary_gate_payload.get("remote_market"))
        or text(feedback_payload.get("remote_market"))
        or text(policy_payload.get("remote_market"))
        or "-"
    )
    feedback_status = text(feedback_payload.get("feedback_status"))
    policy_decision = text(policy_payload.get("policy_decision"))
    ack_status = text(ack_payload.get("ack_status"))
    transport_sla_status = text(transport_sla_payload.get("transport_sla_status"))
    canary_gate_status = text(canary_gate_payload.get("canary_gate_status"))
    queue_age_status = text(feedback_payload.get("queue_age_status"))
    ticket_artifact_status = text(feedback_payload.get("ticket_artifact_status"))
    guardian_blocked_count = int(feedback_payload.get("guardian_blocked_count") or 0)
    no_fill_count = int(feedback_payload.get("no_fill_count") or 0)

    shadow_learning_score = 0
    if feedback_status:
        shadow_learning_score += 20
    if text(policy_payload.get("policy_brief")):
        shadow_learning_score += 15
    if ack_status:
        shadow_learning_score += 15
    if text(actor_payload.get("actor_status")):
        shadow_learning_score += 10
    if transport_sla_status:
        shadow_learning_score += 10
    if canary_gate_status:
        shadow_learning_score += 10
    if text(journal_payload.get("journal_status")):
        shadow_learning_score += 10
    if queue_age_status in {"queue_fresh", "queue_warming"}:
        shadow_learning_score += 5
    if ticket_artifact_status.startswith("fresh_artifact"):
        shadow_learning_score += 5
    if "guardian_blocked" in feedback_status:
        shadow_learning_score -= 10
    if ticket_artifact_status.startswith("stale_artifact"):
        shadow_learning_score -= 10
    if no_fill_count > 0:
        shadow_learning_score -= 10
    shadow_learning_score = clamp_score(shadow_learning_score)

    execution_readiness_score = 100
    if "guardian_blocked" in feedback_status or policy_decision == "reject_until_guardian_clear":
        execution_readiness_score -= 35
    if ticket_artifact_status.startswith("stale_artifact"):
        execution_readiness_score -= 20
    if guardian_blocked_count > 0:
        execution_readiness_score -= 15
    if no_fill_count > 0:
        execution_readiness_score -= 10
    if ack_status == "shadow_no_send_ack_recorded":
        execution_readiness_score -= 10
    if transport_sla_status == "shadow_transport_sla_blocked_no_send":
        execution_readiness_score -= 10
    if "blocked" in canary_gate_status:
        execution_readiness_score -= 10
    if queue_age_status == "queue_warming":
        execution_readiness_score -= 5
    elif queue_age_status == "queue_aging_high":
        execution_readiness_score -= 10
    elif queue_age_status == "queue_stale":
        execution_readiness_score -= 15
    execution_readiness_score = clamp_score(execution_readiness_score)

    transport_observability_score = 0
    if text(journal_payload.get("journal_status")):
        transport_observability_score += 25
    if ack_status:
        transport_observability_score += 25
    if transport_sla_status:
        transport_observability_score += 25
    if text(actor_payload.get("actor_status")):
        transport_observability_score += 15
    if canary_gate_status:
        transport_observability_score += 10
    if ack_status == "shadow_no_send_ack_recorded":
        transport_observability_score -= 10
    transport_observability_score = clamp_score(transport_observability_score)

    quality_score = clamp_score(
        round(
            (shadow_learning_score * 0.5)
            + (execution_readiness_score * 0.35)
            + (transport_observability_score * 0.15)
        )
    )

    if canary_gate_status == "shadow_canary_gate_ready_preview" and quality_score >= 80:
        quality_status = "quality_ready_for_guarded_canary"
        quality_recommendation = "review_guarded_canary_candidate"
    elif (
        policy_decision == "accept_shadow_learning_only"
        and shadow_learning_score >= 60
        and transport_observability_score >= 80
    ):
        quality_status = "quality_learning_only_shadow_viable"
        quality_recommendation = "continue_shadow_learning_until_guardian_clear"
    elif "blocked" in canary_gate_status or policy_decision == "reject_until_guardian_clear":
        quality_status = "quality_degraded_guardian_blocked_shadow_only"
        quality_recommendation = "keep_downranked_shadow_until_guardian_clear"
    elif quality_score >= 50:
        quality_status = "quality_watch_shadow_only"
        quality_recommendation = "continue_shadow_sampling_before_canary"
    else:
        quality_status = "quality_degraded_shadow_only"
        quality_recommendation = "improve_route_quality_before_canary"

    quality_brief = ":".join(
        [quality_status, symbol or "-", f"score_{quality_score}", remote_market or "-"]
    )
    done_when = (
        "fresh intents produce repeatable guardian-approved ack/fill samples, queue aging stays controlled, "
        "and the canary gate remains ready long enough to justify a guarded canary review"
    )
    blocker_detail = (
        text(feedback_payload.get("blocker_detail"))
        or text(policy_payload.get("blocker_detail"))
        or text(ack_payload.get("blocker_detail"))
        or text(journal_payload.get("blocker_detail"))
    )
    return {
        "action": "build_remote_orderflow_quality_report",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "quality_status": quality_status,
        "quality_brief": quality_brief,
        "quality_recommendation": quality_recommendation,
        "quality_score": quality_score,
        "shadow_learning_score": shadow_learning_score,
        "execution_readiness_score": execution_readiness_score,
        "transport_observability_score": transport_observability_score,
        "next_transition": "live_boundary_hold",
        "route_symbol": symbol,
        "remote_market": remote_market,
        "feedback_brief": text(feedback_payload.get("feedback_brief")),
        "feedback_status": feedback_status,
        "policy_brief": text(policy_payload.get("policy_brief")),
        "policy_status": text(policy_payload.get("policy_status")),
        "policy_decision": policy_decision,
        "ack_brief": text(ack_payload.get("ack_brief")),
        "ack_status": ack_status,
        "actor_brief": text(actor_payload.get("actor_brief")),
        "actor_status": text(actor_payload.get("actor_status")),
        "transport_sla_brief": text(transport_sla_payload.get("transport_sla_brief")),
        "transport_sla_status": transport_sla_status,
        "canary_gate_brief": text(canary_gate_payload.get("canary_gate_brief")),
        "canary_gate_status": canary_gate_status,
        "queue_age_status": queue_age_status,
        "ticket_artifact_status": ticket_artifact_status,
        "guardian_blocked_count": guardian_blocked_count,
        "no_fill_count": no_fill_count,
        "recent_outcomes": list(as_list(feedback_payload.get("recent_outcomes"))),
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "remote_orderflow_feedback": str(feedback_path),
            "remote_orderflow_policy_state": str(policy_path),
            "remote_execution_ack_state": str(ack_path),
            "remote_execution_actor_state": str(actor_path),
            "remote_execution_transport_sla": str(transport_sla_path),
            "remote_execution_actor_canary_gate": str(canary_gate_path),
            "remote_execution_journal": str(journal_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote orderflow quality report.")
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
    feedback_path = find_latest(review_dir, "*_remote_orderflow_feedback.json")
    policy_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json")
    ack_path = find_latest(review_dir, "*_remote_execution_ack_state.json")
    actor_path = find_latest(review_dir, "*_remote_execution_actor_state.json")
    transport_sla_path = find_latest(review_dir, "*_remote_execution_transport_sla.json")
    canary_gate_path = find_latest(review_dir, "*_remote_execution_actor_canary_gate.json")
    journal_path = find_latest(review_dir, "*_remote_execution_journal.json")
    missing = [
        name
        for name, path in (
            ("remote_orderflow_feedback", feedback_path),
            ("remote_orderflow_policy_state", policy_path),
            ("remote_execution_ack_state", ack_path),
            ("remote_execution_actor_state", actor_path),
            ("remote_execution_transport_sla", transport_sla_path),
            ("remote_execution_actor_canary_gate", canary_gate_path),
            ("remote_execution_journal", journal_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        feedback_path=feedback_path,
        feedback_payload=load_json_mapping(feedback_path),
        policy_path=policy_path,
        policy_payload=load_json_mapping(policy_path),
        ack_path=ack_path,
        ack_payload=load_json_mapping(ack_path),
        actor_path=actor_path,
        actor_payload=load_json_mapping(actor_path),
        transport_sla_path=transport_sla_path,
        transport_sla_payload=load_json_mapping(transport_sla_path),
        canary_gate_path=canary_gate_path,
        canary_gate_payload=load_json_mapping(canary_gate_path),
        journal_path=journal_path,
        journal_payload=load_json_mapping(journal_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_orderflow_quality_report.json"
    markdown = review_dir / f"{stamp}_remote_orderflow_quality_report.md"
    checksum = review_dir / f"{stamp}_remote_orderflow_quality_report_checksum.json"
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
