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
REVIEW_BLOCKING_ACTIONS = {
    "deprioritize_flow",
    "watch_priority_until_long_window_confirms",
    "refresh_source_before_use",
    "consider_refresh_before_promotion",
}
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text_value = str(raw or "").strip()
    if not text_value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
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
        "*_remote_live_boundary_hold.json",
        "*_remote_live_boundary_hold.md",
        "*_remote_live_boundary_hold_checksum.json",
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


def review_head_brief(cross_market_payload: dict[str, Any]) -> str:
    head = as_dict(cross_market_payload.get("review_head"))
    if head:
        return ":".join(
            [
                text(head.get("status")) or "review",
                text(head.get("area")),
                text(head.get("symbol")),
                text(head.get("action")),
                text(head.get("priority_score")),
            ]
        ).strip(":")
    return text(cross_market_payload.get("review_head_brief"))


def time_sync_mode(*, time_sync_blocked: bool, shadow_learning_allowed: bool) -> str:
    if time_sync_blocked and shadow_learning_allowed:
        return "promotion_blocked_shadow_learning_allowed"
    if time_sync_blocked:
        return "promotion_blocked_shadow_learning_unavailable"
    if shadow_learning_allowed:
        return "time_sync_clear_shadow_learning_allowed"
    return "time_sync_clear_shadow_learning_unavailable"


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Live Boundary Hold",
            "",
            f"- brief: `{text(payload.get('hold_brief'))}`",
            f"- status: `{text(payload.get('hold_status'))}`",
            f"- decision: `{text(payload.get('hold_decision'))}`",
            f"- next_transition: `{text(payload.get('next_transition'))}`",
            f"- time_sync_mode: `{text(payload.get('time_sync_mode'))}`",
            f"- canary_gate: `{text(payload.get('canary_gate_brief'))}`",
            f"- quality_report: `{text(payload.get('quality_brief'))}`",
            f"- policy: `{text(payload.get('policy_brief'))}`",
            f"- review_head: `{text(payload.get('review_head_brief'))}`",
            f"- shadow_clock_evidence: `{text(payload.get('remote_shadow_clock_evidence_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    canary_gate_path: Path,
    canary_gate_payload: dict[str, Any],
    quality_report_path: Path,
    quality_report_payload: dict[str, Any],
    policy_path: Path,
    policy_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    cross_market_path: Path,
    cross_market_payload: dict[str, Any],
    shadow_clock_path: Path | None,
    shadow_clock_payload: dict[str, Any],
    time_sync_verification_path: Path | None,
    time_sync_verification_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    review_head = as_dict(cross_market_payload.get("review_head"))
    symbol = (
        text(quality_report_payload.get("route_symbol"))
        or text(canary_gate_payload.get("route_symbol"))
        or text(policy_payload.get("route_symbol"))
        or text(review_head.get("symbol"))
        or "-"
    )
    remote_market = (
        text(quality_report_payload.get("remote_market"))
        or text(canary_gate_payload.get("remote_market"))
        or text(policy_payload.get("remote_market"))
        or "-"
    )
    blocked_gate_names = dedupe_text(
        [
            text(as_dict(row).get("name"))
            for row in as_list(live_gate_payload.get("blockers"))
            if text(as_dict(row).get("status")) == "blocked"
        ]
    )
    policy_decision = text(policy_payload.get("policy_decision"))
    canary_gate_status = text(canary_gate_payload.get("canary_gate_status"))
    quality_status = text(quality_report_payload.get("quality_status"))
    review_action = text(review_head.get("action"))
    review_head_brief_text = review_head_brief(cross_market_payload)
    review_head_blocker_detail = (
        text(cross_market_payload.get("review_head_blocker_detail"))
        or text(review_head.get("blocker_detail"))
    )
    review_head_done_when = (
        text(cross_market_payload.get("review_head_done_when"))
        or text(review_head.get("done_when"))
    )
    time_sync_verification_brief = text(time_sync_verification_payload.get("verification_brief"))
    time_sync_cleared = bool(time_sync_verification_payload.get("cleared", False))
    time_sync_blocked = (bool(time_sync_verification_payload) and not time_sync_cleared) or (
        "time-sync=" in review_head_blocker_detail
    )
    remote_shadow_clock_evidence_brief = text(shadow_clock_payload.get("evidence_brief"))
    remote_shadow_clock_evidence_status = text(shadow_clock_payload.get("evidence_status"))
    remote_shadow_clock_shadow_learning_allowed = bool(
        shadow_clock_payload.get("shadow_learning_allowed", False)
    )
    current_time_sync_mode = time_sync_mode(
        time_sync_blocked=time_sync_blocked,
        shadow_learning_allowed=remote_shadow_clock_shadow_learning_allowed,
    )
    guardian_blocked = bool(blocked_gate_names) or policy_decision in {
        "reject_until_guardian_clear",
        "accept_shadow_learning_only",
        "observe_without_transport",
    }
    review_blocked = bool(review_head_blocker_detail) or review_action in REVIEW_BLOCKING_ACTIONS
    canary_ready = canary_gate_status == "shadow_canary_gate_ready_preview"
    quality_ready = quality_status == "quality_ready_for_guarded_canary"

    hold_reason_codes = dedupe_text(
        blocked_gate_names
        + ([policy_decision] if policy_decision else [])
        + (["canary_gate_not_ready"] if not canary_ready else [])
        + (["quality_not_ready"] if not quality_ready else [])
        + (["time_sync_blocked"] if time_sync_blocked else [])
        + (["review_head_not_ready"] if review_blocked else [])
    )
    policy_summary = ":".join(
        [
            part
            for part in [
                text(policy_payload.get("policy_status")),
                policy_decision,
                text(policy_payload.get("ticket_match_brief")),
            ]
            if part
        ]
    )

    if guardian_blocked and review_blocked:
        hold_mode = "guardian_review_blocked"
        next_transition = "guardian_blocker_clearance"
    elif guardian_blocked:
        hold_mode = "guardian_blocked"
        next_transition = "guardian_blocker_clearance"
    elif time_sync_blocked or review_blocked:
        hold_mode = "review_blocked"
        next_transition = "review_head_clearance"
    elif canary_ready and quality_ready:
        hold_mode = "ready_for_guarded_canary_review"
        next_transition = "guarded_canary_review"
    else:
        hold_mode = "candidate_not_ready"
        next_transition = "shadow_quality_improvement"

    ready_for_review = next_transition == "guarded_canary_review"
    hold_status = "live_boundary_hold_review_ready" if ready_for_review else "live_boundary_hold_active"
    hold_decision = (
        "review_guarded_canary_after_clearance" if ready_for_review else "keep_shadow_transport_only"
    )
    hold_brief = ":".join([hold_status, symbol or "-", hold_mode, remote_market or "-"])
    blocker_detail = " | ".join(
        dedupe_text(
            [
                ",".join(blocked_gate_names),
                policy_summary,
                text(canary_gate_payload.get("canary_gate_brief")),
                text(quality_report_payload.get("quality_brief")),
                review_head_blocker_detail,
                time_sync_verification_brief,
                remote_shadow_clock_evidence_brief,
            ]
        )
    )
    done_when_parts = dedupe_text(
        [
            "guardian blockers clear and the policy no longer rejects the remote route"
            if guardian_blocked
            else "",
            "review head stops advertising a blocker and no longer requires deprioritize/watch-only handling"
            if review_blocked
            else "",
            "system time sync repair verification clears and the review head no longer carries a time-sync blocker"
            if time_sync_blocked
            else "",
            "canary gate remains ready and the quality report graduates from shadow-only degradation"
            if not (canary_ready and quality_ready)
            else "",
            review_head_done_when,
            text(quality_report_payload.get("done_when")),
        ]
    )
    done_when = " | ".join(done_when_parts)

    return {
        "action": "build_remote_live_boundary_hold",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "hold_status": hold_status,
        "hold_brief": hold_brief,
        "hold_decision": hold_decision,
        "hold_mode": hold_mode,
        "next_transition": next_transition,
        "route_symbol": symbol,
        "remote_market": remote_market,
        "guardian_blocked": guardian_blocked,
        "review_blocked": review_blocked,
        "time_sync_blocked": time_sync_blocked,
        "time_sync_mode": current_time_sync_mode,
        "canary_ready": canary_ready,
        "quality_ready": quality_ready,
        "hold_reason_codes": hold_reason_codes,
        "canary_gate_brief": text(canary_gate_payload.get("canary_gate_brief")),
        "canary_gate_status": canary_gate_status,
        "quality_brief": text(quality_report_payload.get("quality_brief")),
        "quality_status": quality_status,
        "quality_recommendation": text(quality_report_payload.get("quality_recommendation")),
        "policy_brief": text(policy_payload.get("policy_brief")),
        "policy_status": text(policy_payload.get("policy_status")),
        "policy_decision": policy_decision,
        "review_head_brief": review_head_brief_text,
        "review_head_action": review_action,
        "review_head_blocker_detail": review_head_blocker_detail,
        "review_head_done_when": review_head_done_when,
        "time_sync_verification_brief": time_sync_verification_brief,
        "time_sync_verification_status": text(time_sync_verification_payload.get("status")),
        "time_sync_verification_cleared": time_sync_cleared,
        "remote_shadow_clock_evidence_brief": remote_shadow_clock_evidence_brief,
        "remote_shadow_clock_evidence_status": remote_shadow_clock_evidence_status,
        "remote_shadow_clock_shadow_learning_allowed": remote_shadow_clock_shadow_learning_allowed,
        "blocked_gate_names": blocked_gate_names,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "remote_execution_actor_canary_gate": str(canary_gate_path),
            "remote_orderflow_quality_report": str(quality_report_path),
            "remote_orderflow_policy_state": str(policy_path),
            "live_gate_blocker_report": str(live_gate_path),
            "cross_market_operator_state": str(cross_market_path),
            "remote_shadow_clock_evidence": str(shadow_clock_path) if shadow_clock_path else "",
            "system_time_sync_repair_verification_report": (
                str(time_sync_verification_path) if time_sync_verification_path else ""
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote live boundary hold artifact.")
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

    canary_gate_path = find_latest(review_dir, "*_remote_execution_actor_canary_gate.json")
    quality_report_path = find_latest(review_dir, "*_remote_orderflow_quality_report.json")
    policy_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json")
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    cross_market_path = find_latest(review_dir, "*_cross_market_operator_state.json")
    shadow_clock_path = find_latest(review_dir, "*_remote_shadow_clock_evidence.json")
    time_sync_verification_path = find_latest(
        review_dir, "*_system_time_sync_repair_verification_report.json"
    )

    missing = [
        name
        for name, path in (
            ("remote_execution_actor_canary_gate", canary_gate_path),
            ("remote_orderflow_quality_report", quality_report_path),
            ("remote_orderflow_policy_state", policy_path),
            ("live_gate_blocker_report", live_gate_path),
            ("cross_market_operator_state", cross_market_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        canary_gate_path=canary_gate_path,
        canary_gate_payload=load_json_mapping(canary_gate_path),
        quality_report_path=quality_report_path,
        quality_report_payload=load_json_mapping(quality_report_path),
        policy_path=policy_path,
        policy_payload=load_json_mapping(policy_path),
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path),
        cross_market_path=cross_market_path,
        cross_market_payload=load_json_mapping(cross_market_path),
        shadow_clock_path=shadow_clock_path,
        shadow_clock_payload=(
            load_json_mapping(shadow_clock_path)
            if shadow_clock_path is not None and shadow_clock_path.exists()
            else {}
        ),
        time_sync_verification_path=time_sync_verification_path,
        time_sync_verification_payload=(
            load_json_mapping(time_sync_verification_path)
            if time_sync_verification_path is not None and time_sync_verification_path.exists()
            else {}
        ),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_live_boundary_hold.json"
    markdown = review_dir / f"{stamp}_remote_live_boundary_hold.md"
    checksum = review_dir / f"{stamp}_remote_live_boundary_hold_checksum.json"
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
