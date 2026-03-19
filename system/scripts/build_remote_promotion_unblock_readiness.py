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


def join_unique(parts: list[Any]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = text(part)
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return " | ".join(ordered)


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
        "*_remote_promotion_unblock_readiness.json",
        "*_remote_promotion_unblock_readiness.md",
        "*_remote_promotion_unblock_readiness_checksum.json",
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
            "# Remote Promotion Unblock Readiness",
            "",
            f"- brief: `{text(payload.get('readiness_brief'))}`",
            f"- status: `{text(payload.get('readiness_status'))}`",
            f"- decision: `{text(payload.get('readiness_decision'))}`",
            f"- remote_preconditions_status: `{text(payload.get('remote_preconditions_status'))}`",
            f"- primary_blocker_scope: `{text(payload.get('primary_blocker_scope'))}`",
            f"- primary_local_repair_title: `{text(payload.get('primary_local_repair_title'))}`",
            f"- primary_local_repair_target_artifact: `{text(payload.get('primary_local_repair_target_artifact'))}`",
            f"- primary_local_repair_plan_brief: `{text(payload.get('primary_local_repair_plan_brief'))}`",
            f"- primary_local_repair_environment_classification: `{text(payload.get('primary_local_repair_environment_classification'))}`",
            f"- promotion_gate: `{text(payload.get('promotion_gate_brief'))}`",
            f"- shadow_learning_continuity: `{text(payload.get('continuity_brief'))}`",
            f"- quality_report: `{text(payload.get('quality_brief'))}`",
            f"- guardian_clearance: `{text(payload.get('guardian_clearance_brief'))}`",
            f"- time_sync_verification: `{text(payload.get('time_sync_verification_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    promotion_gate_path: Path,
    promotion_gate_payload: dict[str, Any],
    continuity_path: Path,
    continuity_payload: dict[str, Any],
    guardian_clearance_path: Path,
    guardian_clearance_payload: dict[str, Any],
    quality_report_path: Path,
    quality_report_payload: dict[str, Any],
    time_sync_verification_path: Path,
    time_sync_verification_payload: dict[str, Any],
    time_sync_repair_plan_path: Path,
    time_sync_repair_plan_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    route_symbol = (
        text(continuity_payload.get("route_symbol"))
        or text(promotion_gate_payload.get("route_symbol"))
        or text(guardian_clearance_payload.get("route_symbol"))
        or "-"
    )
    remote_market = (
        text(continuity_payload.get("remote_market"))
        or text(promotion_gate_payload.get("remote_market"))
        or text(guardian_clearance_payload.get("remote_market"))
        or "-"
    )

    continuity_status = text(continuity_payload.get("continuity_status"))
    continuity_brief = text(continuity_payload.get("continuity_brief"))
    promotion_gate_status = text(promotion_gate_payload.get("promotion_gate_status"))
    promotion_gate_brief = text(promotion_gate_payload.get("promotion_gate_brief"))
    guardian_brief = text(guardian_clearance_payload.get("clearance_brief"))
    quality_brief = text(quality_report_payload.get("quality_brief"))
    time_sync_verification_brief = text(time_sync_verification_payload.get("verification_brief"))
    time_sync_repair_plan_brief = text(time_sync_repair_plan_payload.get("plan_brief"))
    time_sync_environment_classification = text(
        time_sync_repair_plan_payload.get("environment_classification")
    )
    time_sync_environment_blocker_detail = text(
        time_sync_repair_plan_payload.get("environment_blocker_detail")
    )
    time_sync_environment_remediation_hint = text(
        time_sync_repair_plan_payload.get("environment_remediation_hint")
    )
    top_blocker_target_artifact = text(guardian_clearance_payload.get("top_blocker_target_artifact")) or text(
        promotion_gate_payload.get("promotion_blocker_target_artifact")
    )
    top_blocker_code = text(guardian_clearance_payload.get("top_blocker_code")) or text(
        promotion_gate_payload.get("promotion_blocker_code")
    )
    top_blocker_next_action = text(guardian_clearance_payload.get("top_blocker_next_action")) or text(
        promotion_gate_payload.get("promotion_blocker_next_action")
    )
    top_blocker_title = text(guardian_clearance_payload.get("top_blocker_title")) or text(
        promotion_gate_payload.get("promotion_blocker_title")
    )
    top_blocker_detail = text(guardian_clearance_payload.get("top_blocker_detail")) or text(
        promotion_gate_payload.get("promotion_blocker_detail")
    )

    quality_score = int(quality_report_payload.get("quality_score") or 0)
    shadow_learning_score = int(quality_report_payload.get("shadow_learning_score") or 0)
    continuity_stable = continuity_status == "shadow_learning_continuity_stable"
    quality_shadow_viable = quality_score >= 40 and shadow_learning_score >= 60
    shadow_learning_continues = text(promotion_gate_payload.get("shadow_learning_decision")) == (
        "continue_shadow_learning_collect_feedback"
    )
    promotion_shadow_blocked = promotion_gate_status in {
        "guarded_canary_promotion_blocked_shadow_learning_allowed",
        "guarded_canary_promotion_blocked_guardian_review",
    } and (
        promotion_gate_status == "guarded_canary_promotion_blocked_shadow_learning_allowed"
        or shadow_learning_continues
    )
    time_sync_blocked = not bool(time_sync_verification_payload.get("cleared", False))
    local_time_sync_primary = (
        time_sync_blocked and top_blocker_target_artifact == "system_time_sync_repair_verification_report"
    )

    if continuity_stable and quality_shadow_viable and promotion_shadow_blocked and local_time_sync_primary:
        readiness_status = "local_time_sync_primary_blocker_shadow_ready"
        readiness_decision = "repair_local_time_sync_then_review_guarded_canary"
        remote_preconditions_status = "shadow_ready_remote_preconditions_viable"
        primary_blocker_scope = "local_admin_repair"
        primary_local_repair_title = "Repair local time sync to unlock guarded canary review"
        done_when = (
            "shadow learning continuity remains stable, quality stays viable, and the local time-sync verification clears "
            "so guarded canary review can move from local repair gate into promotion review"
        )
        if time_sync_environment_classification == "timed_ntp_via_fake_ip":
            readiness_decision = "repair_local_fake_ip_ntp_path_then_review_guarded_canary"
            primary_blocker_scope = "timed_ntp_via_fake_ip"
            primary_local_repair_title = (
                "Repair local fake-ip NTP path to unlock guarded canary review"
            )
    elif continuity_stable and promotion_shadow_blocked:
        if top_blocker_code == "guardian_ticket_actionability":
            readiness_status = "shadow_ready_ticket_actionability_blocked"
            if top_blocker_next_action == "generate_fresh_crypto_signal_source_before_rebuild_tickets":
                readiness_decision = (
                    "generate_fresh_crypto_signal_source_then_review_guarded_canary"
                )
            elif top_blocker_next_action == "rebuild_tickets_with_newer_signal_candidate":
                readiness_decision = (
                    "rebuild_tickets_with_newer_signal_candidate_then_review_guarded_canary"
                )
            elif (
                top_blocker_target_artifact == "crypto_shortline_liquidity_event_trigger"
                and top_blocker_next_action
                == "monitor_persistent_orderflow_pressure_for_liquidity_sweep"
            ):
                readiness_decision = (
                    "monitor_persistent_orderflow_pressure_for_liquidity_sweep_then_review_guarded_canary"
                )
            elif (
                top_blocker_target_artifact == "crypto_shortline_liquidity_event_trigger"
                and top_blocker_next_action
                == "refresh_shortline_execution_gate_after_liquidity_event"
            ):
                readiness_decision = (
                    "refresh_shortline_execution_gate_after_liquidity_event_then_review_guarded_canary"
                )
            elif (
                top_blocker_target_artifact
                in {
                    "crypto_shortline_profile_location_watch",
                    "crypto_shortline_pattern_router",
                    "crypto_shortline_live_bars_snapshot",
                    "crypto_shortline_liquidity_event_trigger",
                    "crypto_shortline_liquidity_sweep_watch",
                    "crypto_shortline_execution_quality_watch",
                    "crypto_shortline_slippage_snapshot",
                    "crypto_shortline_fill_capacity_watch",
                    "crypto_shortline_cvd_confirmation_watch",
                    "crypto_shortline_price_reference_watch",
                    "crypto_shortline_signal_quality_watch",
                    "crypto_shortline_sizing_watch",
                    "crypto_shortline_setup_transition_watch",
                    "crypto_shortline_gate_stack_progress",
                }
                and top_blocker_next_action.endswith(
                    ("_then_recheck_execution_gate", "_then_refresh_gate_stack")
                )
            ):
                readiness_decision = (
                    top_blocker_next_action.replace(
                        "_then_recheck_execution_gate",
                        "_then_review_guarded_canary",
                    ).replace(
                        "_then_refresh_gate_stack",
                        "_then_review_guarded_canary",
                    )
                )
            else:
                readiness_decision = "resolve_ticket_actionability_then_review_guarded_canary"
            remote_preconditions_status = "shadow_ready_remote_preconditions_viable"
            primary_blocker_scope = "guardian_ticket_actionability"
            done_when = (
                "a fresh actionable ticket row exists for the route symbol, guardian reject reasons clear, "
                "and guarded canary review can advance from shadow learning into promotion review"
            )
        else:
            readiness_status = "shadow_ready_mixed_blockers_remaining"
            readiness_decision = "clear_remaining_blockers_before_promotion"
            remote_preconditions_status = "shadow_ready_partial"
            primary_blocker_scope = top_blocker_code or "mixed_local_and_remote"
            done_when = (
                "shadow learning continuity remains stable, quality stays viable, and the remaining promotion blockers clear "
                "so guarded canary review can move into promotion review"
            )
        primary_local_repair_title = top_blocker_title
    elif not continuity_stable:
        readiness_status = "shadow_learning_path_not_ready"
        readiness_decision = "repair_shadow_learning_before_promotion"
        remote_preconditions_status = "shadow_path_not_ready"
        primary_blocker_scope = "remote_learning_path"
        primary_local_repair_title = top_blocker_title
        done_when = "shadow learning continuity becomes stable enough to support guarded canary review"
    elif not quality_shadow_viable:
        readiness_status = "shadow_quality_not_ready"
        readiness_decision = "improve_shadow_quality_before_promotion"
        remote_preconditions_status = "shadow_quality_not_ready"
        primary_blocker_scope = "remote_quality"
        primary_local_repair_title = top_blocker_title
        done_when = "quality score and shadow-learning score recover into the promotion-review range"
    else:
        readiness_status = "promotion_unblock_state_unclear"
        readiness_decision = "inspect_promotion_gate_and_clearance"
        remote_preconditions_status = "unknown"
        primary_blocker_scope = "unclear"
        primary_local_repair_title = top_blocker_title
        done_when = "promotion gate and guardian clearance converge on one explicit unblock path"

    primary_local_repair_detail = top_blocker_detail
    if primary_blocker_scope in {"local_admin_repair", "timed_ntp_via_fake_ip"}:
        primary_local_repair_detail = join_unique(
            [
                time_sync_verification_brief,
                (
                    f"repair_plan={time_sync_repair_plan_brief}"
                    if time_sync_repair_plan_brief
                    else ""
                ),
                (
                    f"time_sync_env={time_sync_environment_classification}:{time_sync_environment_blocker_detail}"
                    if time_sync_environment_classification and time_sync_environment_blocker_detail
                    else ""
                ),
                (
                    f"time_sync_fix_hint={time_sync_environment_remediation_hint}"
                    if time_sync_environment_remediation_hint
                    else ""
                ),
            ]
        ) or top_blocker_detail
    primary_local_repair_required = primary_blocker_scope in {
        "local_admin_repair",
        "timed_ntp_via_fake_ip",
    }
    exposed_primary_local_repair_plan_brief = (
        time_sync_repair_plan_brief if primary_local_repair_required else ""
    )
    exposed_primary_local_repair_environment_classification = (
        time_sync_environment_classification if primary_local_repair_required else ""
    )
    exposed_primary_local_repair_environment_blocker_detail = (
        time_sync_environment_blocker_detail if primary_local_repair_required else ""
    )
    exposed_primary_local_repair_environment_remediation_hint = (
        time_sync_environment_remediation_hint if primary_local_repair_required else ""
    )

    readiness_brief = ":".join(
        [readiness_status, route_symbol or "-", readiness_decision, remote_market or "-"]
    )
    blocker_detail = join_unique(
        [
            primary_local_repair_detail,
            f"quality_score={quality_score}",
            f"shadow_learning_score={shadow_learning_score}",
            promotion_gate_brief,
            continuity_brief,
        ]
    )
    return {
        "action": "build_remote_promotion_unblock_readiness",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "remote_market": remote_market,
        "readiness_status": readiness_status,
        "readiness_brief": readiness_brief,
        "readiness_decision": readiness_decision,
        "remote_preconditions_status": remote_preconditions_status,
        "primary_blocker_scope": primary_blocker_scope,
        "primary_local_repair_required": primary_local_repair_required,
        "primary_local_repair_title": primary_local_repair_title,
        "primary_local_repair_target_artifact": top_blocker_target_artifact,
        "primary_local_repair_detail": primary_local_repair_detail,
        "primary_local_repair_plan_brief": exposed_primary_local_repair_plan_brief,
        "primary_local_repair_environment_classification": exposed_primary_local_repair_environment_classification,
        "primary_local_repair_environment_blocker_detail": exposed_primary_local_repair_environment_blocker_detail,
        "primary_local_repair_environment_remediation_hint": exposed_primary_local_repair_environment_remediation_hint,
        "promotion_gate_brief": promotion_gate_brief,
        "continuity_brief": continuity_brief,
        "guardian_clearance_brief": guardian_brief,
        "quality_brief": quality_brief,
        "time_sync_verification_brief": time_sync_verification_brief,
        "quality_score": quality_score,
        "shadow_learning_score": shadow_learning_score,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "remote_guarded_canary_promotion_gate": str(promotion_gate_path),
            "remote_shadow_learning_continuity": str(continuity_path),
            "remote_guardian_blocker_clearance": str(guardian_clearance_path),
            "remote_orderflow_quality_report": str(quality_report_path),
            "system_time_sync_repair_verification_report": str(time_sync_verification_path),
            "system_time_sync_repair_plan": str(time_sync_repair_plan_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build remote promotion unblock readiness from shadow-learning and guardian artifacts."
    )
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

    promotion_gate_path = find_latest(
        review_dir, "*_remote_guarded_canary_promotion_gate.json", reference_now
    )
    continuity_path = find_latest(review_dir, "*_remote_shadow_learning_continuity.json", reference_now)
    guardian_clearance_path = find_latest(
        review_dir, "*_remote_guardian_blocker_clearance.json", reference_now
    )
    quality_report_path = find_latest(
        review_dir, "*_remote_orderflow_quality_report.json", reference_now
    )
    time_sync_verification_path = find_latest(
        review_dir, "*_system_time_sync_repair_verification_report.json", reference_now
    )
    time_sync_repair_plan_path = find_latest(
        review_dir, "*_system_time_sync_repair_plan.json", reference_now
    )

    missing = [
        name
        for name, path in (
            ("remote_guarded_canary_promotion_gate", promotion_gate_path),
            ("remote_shadow_learning_continuity", continuity_path),
            ("remote_guardian_blocker_clearance", guardian_clearance_path),
            ("remote_orderflow_quality_report", quality_report_path),
            ("system_time_sync_repair_verification_report", time_sync_verification_path),
            ("system_time_sync_repair_plan", time_sync_repair_plan_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        promotion_gate_path=promotion_gate_path,
        promotion_gate_payload=load_json_mapping(promotion_gate_path),
        continuity_path=continuity_path,
        continuity_payload=load_json_mapping(continuity_path),
        guardian_clearance_path=guardian_clearance_path,
        guardian_clearance_payload=load_json_mapping(guardian_clearance_path),
        quality_report_path=quality_report_path,
        quality_report_payload=load_json_mapping(quality_report_path),
        time_sync_verification_path=time_sync_verification_path,
        time_sync_verification_payload=load_json_mapping(time_sync_verification_path),
        time_sync_repair_plan_path=time_sync_repair_plan_path,
        time_sync_repair_plan_payload=load_json_mapping(time_sync_repair_plan_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_promotion_unblock_readiness.json"
    markdown = review_dir / f"{stamp}_remote_promotion_unblock_readiness.md"
    checksum = review_dir / f"{stamp}_remote_promotion_unblock_readiness_checksum.json"

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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
