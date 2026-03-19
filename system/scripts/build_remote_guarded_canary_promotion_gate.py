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
FUTURE_STAMP_GRACE_MINUTES = 5


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


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    try:
        return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(
    review_dir: Path,
    pattern: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: artifact_sort_key(item, reference_now))


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
        "*_remote_guarded_canary_promotion_gate.json",
        "*_remote_guarded_canary_promotion_gate.md",
        "*_remote_guarded_canary_promotion_gate_checksum.json",
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
            "# Remote Guarded Canary Promotion Gate",
            "",
            f"- brief: `{text(payload.get('promotion_gate_brief'))}`",
            f"- status: `{text(payload.get('promotion_gate_status'))}`",
            f"- decision: `{text(payload.get('promotion_gate_decision'))}`",
            f"- shadow_learning_decision: `{text(payload.get('shadow_learning_decision'))}`",
            f"- time_sync_mode: `{text(payload.get('time_sync_mode'))}`",
            f"- shadow_clock_evidence: `{text(payload.get('remote_shadow_clock_evidence_brief'))}`",
            f"- canary_gate: `{text(payload.get('canary_gate_brief'))}`",
            f"- quality_report: `{text(payload.get('quality_brief'))}`",
            f"- guardian_clearance: `{text(payload.get('guardian_clearance_brief'))}`",
            f"- blocker: `{text(payload.get('promotion_blocker_detail'))}`",
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
    live_boundary_hold_path: Path,
    live_boundary_hold_payload: dict[str, Any],
    guardian_clearance_path: Path,
    guardian_clearance_payload: dict[str, Any],
    shadow_clock_path: Path | None,
    shadow_clock_payload: dict[str, Any],
    time_sync_verification_path: Path | None,
    time_sync_verification_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    route_symbol = (
        text(canary_gate_payload.get("route_symbol"))
        or text(quality_report_payload.get("route_symbol"))
        or text(live_boundary_hold_payload.get("route_symbol"))
        or text(guardian_clearance_payload.get("route_symbol"))
        or "-"
    )
    remote_market = (
        text(canary_gate_payload.get("remote_market"))
        or text(quality_report_payload.get("remote_market"))
        or text(live_boundary_hold_payload.get("remote_market"))
        or text(guardian_clearance_payload.get("remote_market"))
        or "-"
    )
    canary_gate_status = text(canary_gate_payload.get("canary_gate_status"))
    quality_status = text(quality_report_payload.get("quality_status"))
    guardian_blocked = bool(live_boundary_hold_payload.get("guardian_blocked", False))
    review_blocked = bool(live_boundary_hold_payload.get("review_blocked", False))
    time_sync_blocked = bool(live_boundary_hold_payload.get("time_sync_blocked", False)) or not bool(
        time_sync_verification_payload.get("cleared", False)
    )
    time_sync_mode = text(live_boundary_hold_payload.get("time_sync_mode")) or text(
        guardian_clearance_payload.get("time_sync_mode")
    )
    shadow_learning_allowed = bool(shadow_clock_payload.get("shadow_learning_allowed", False)) or bool(
        live_boundary_hold_payload.get("remote_shadow_clock_shadow_learning_allowed", False)
    )
    canary_ready_preview = canary_gate_status == "shadow_canary_gate_ready_preview"
    quality_ready = quality_status == "quality_ready_for_guarded_canary"
    promotion_ready = (
        canary_ready_preview
        and quality_ready
        and not guardian_blocked
        and not review_blocked
        and not time_sync_blocked
    )

    if promotion_ready:
        promotion_gate_status = "guarded_canary_promotion_ready_review"
        promotion_gate_decision = "review_guarded_canary_promotion"
        shadow_learning_decision = "shadow_learning_continues_until_human_review"
        promotion_blocker_code = ""
        promotion_blocker_title = ""
        promotion_blocker_target_artifact = ""
        promotion_blocker_detail = ""
    elif time_sync_blocked and shadow_learning_allowed:
        promotion_gate_status = "guarded_canary_promotion_blocked_shadow_learning_allowed"
        promotion_gate_decision = "block_promotion_continue_shadow_learning"
        shadow_learning_decision = "continue_shadow_learning_collect_feedback"
        promotion_blocker_code = text(guardian_clearance_payload.get("top_blocker_code")) or "time_sync_clearance"
        promotion_blocker_title = text(guardian_clearance_payload.get("top_blocker_title")) or (
            "Repair time sync before any orderflow promotion"
        )
        promotion_blocker_target_artifact = text(
            guardian_clearance_payload.get("top_blocker_target_artifact")
        ) or "system_time_sync_repair_verification_report"
        promotion_blocker_detail = text(guardian_clearance_payload.get("top_blocker_detail")) or text(
            time_sync_verification_payload.get("verification_brief")
        )
    elif guardian_blocked or review_blocked:
        promotion_gate_status = "guarded_canary_promotion_blocked_guardian_review"
        promotion_gate_decision = "clear_guardian_review_blockers_before_promotion"
        shadow_learning_decision = (
            "continue_shadow_learning_collect_feedback"
            if shadow_learning_allowed
            else "shadow_learning_diagnostic_only"
        )
        promotion_blocker_code = text(guardian_clearance_payload.get("top_blocker_code")) or (
            "guardian_review_clearance"
        )
        promotion_blocker_title = text(guardian_clearance_payload.get("top_blocker_title")) or (
            "Clear guardian blockers before any guarded canary review"
        )
        promotion_blocker_target_artifact = text(
            guardian_clearance_payload.get("top_blocker_target_artifact")
        ) or "remote_guardian_blocker_clearance"
        promotion_blocker_detail = text(guardian_clearance_payload.get("top_blocker_detail")) or text(
            live_boundary_hold_payload.get("blocker_detail")
        )
    else:
        promotion_gate_status = "guarded_canary_promotion_blocked_candidate_not_ready"
        promotion_gate_decision = "improve_canary_quality_before_promotion"
        shadow_learning_decision = (
            "continue_shadow_learning_collect_feedback"
            if shadow_learning_allowed
            else "shadow_learning_diagnostic_only"
        )
        promotion_blocker_code = "canary_quality_readiness"
        promotion_blocker_title = "Improve canary and quality readiness before guarded promotion"
        promotion_blocker_target_artifact = "remote_orderflow_quality_report"
        promotion_blocker_detail = " | ".join(
            dedupe_text(
                [
                    text(canary_gate_payload.get("blocker_detail")),
                    text(quality_report_payload.get("blocker_detail")),
                    text(live_boundary_hold_payload.get("blocker_detail")),
                ]
            )
        )

    promotion_gate_brief = ":".join(
        [promotion_gate_status, route_symbol or "-", text(promotion_gate_decision), remote_market or "-"]
    )
    done_when = (
        text(guardian_clearance_payload.get("top_blocker_done_when"))
        or text(live_boundary_hold_payload.get("done_when"))
        or "guardian blockers, review blockers, and time-sync promotion blockers clear while canary and quality remain ready"
    )
    return {
        "action": "build_remote_guarded_canary_promotion_gate",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "promotion_gate_status": promotion_gate_status,
        "promotion_gate_brief": promotion_gate_brief,
        "promotion_gate_decision": promotion_gate_decision,
        "shadow_learning_decision": shadow_learning_decision,
        "promotion_ready": promotion_ready,
        "shadow_learning_allowed": shadow_learning_allowed,
        "guardian_blocked": guardian_blocked,
        "review_blocked": review_blocked,
        "time_sync_blocked": time_sync_blocked,
        "time_sync_mode": time_sync_mode,
        "route_symbol": route_symbol,
        "remote_market": remote_market,
        "canary_gate_brief": text(canary_gate_payload.get("canary_gate_brief")),
        "canary_gate_status": canary_gate_status,
        "quality_brief": text(quality_report_payload.get("quality_brief")),
        "quality_status": quality_status,
        "guardian_clearance_brief": text(guardian_clearance_payload.get("clearance_brief")),
        "live_boundary_hold_brief": text(live_boundary_hold_payload.get("hold_brief")),
        "remote_shadow_clock_evidence_brief": text(shadow_clock_payload.get("evidence_brief"))
        or text(live_boundary_hold_payload.get("remote_shadow_clock_evidence_brief"))
        or text(guardian_clearance_payload.get("remote_shadow_clock_evidence_brief")),
        "remote_shadow_clock_evidence_status": text(shadow_clock_payload.get("evidence_status"))
        or text(live_boundary_hold_payload.get("remote_shadow_clock_evidence_status"))
        or text(guardian_clearance_payload.get("remote_shadow_clock_evidence_status")),
        "promotion_blocker_code": promotion_blocker_code,
        "promotion_blocker_title": promotion_blocker_title,
        "promotion_blocker_target_artifact": promotion_blocker_target_artifact,
        "promotion_blocker_detail": promotion_blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "remote_execution_actor_canary_gate": str(canary_gate_path),
            "remote_orderflow_quality_report": str(quality_report_path),
            "remote_live_boundary_hold": str(live_boundary_hold_path),
            "remote_guardian_blocker_clearance": str(guardian_clearance_path),
            "remote_shadow_clock_evidence": str(shadow_clock_path) if shadow_clock_path else "",
            "system_time_sync_repair_verification_report": str(time_sync_verification_path)
            if time_sync_verification_path
            else "",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote guarded canary promotion gate.")
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

    canary_gate_path = find_latest(
        review_dir, "*_remote_execution_actor_canary_gate.json", reference_now
    )
    quality_report_path = find_latest(
        review_dir, "*_remote_orderflow_quality_report.json", reference_now
    )
    live_boundary_hold_path = find_latest(review_dir, "*_remote_live_boundary_hold.json", reference_now)
    guardian_clearance_path = find_latest(
        review_dir, "*_remote_guardian_blocker_clearance.json", reference_now
    )
    shadow_clock_path = find_latest(review_dir, "*_remote_shadow_clock_evidence.json", reference_now)
    time_sync_verification_path = find_latest(
        review_dir, "*_system_time_sync_repair_verification_report.json"
    )

    missing = [
        name
        for name, path in (
            ("remote_execution_actor_canary_gate", canary_gate_path),
            ("remote_orderflow_quality_report", quality_report_path),
            ("remote_live_boundary_hold", live_boundary_hold_path),
            ("remote_guardian_blocker_clearance", guardian_clearance_path),
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
        live_boundary_hold_path=live_boundary_hold_path,
        live_boundary_hold_payload=load_json_mapping(live_boundary_hold_path),
        guardian_clearance_path=guardian_clearance_path,
        guardian_clearance_payload=load_json_mapping(guardian_clearance_path),
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
    artifact = review_dir / f"{stamp}_remote_guarded_canary_promotion_gate.json"
    markdown = review_dir / f"{stamp}_remote_guarded_canary_promotion_gate.md"
    checksum = review_dir / f"{stamp}_remote_guarded_canary_promotion_gate_checksum.json"
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
