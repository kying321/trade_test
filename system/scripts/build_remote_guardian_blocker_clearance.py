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
FUTURE_STAMP_GRACE_MINUTES = 5


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
        "*_remote_guardian_blocker_clearance.json",
        "*_remote_guardian_blocker_clearance.md",
        "*_remote_guardian_blocker_clearance_checksum.json",
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


def blocked_gate_names(live_gate_payload: dict[str, Any]) -> list[str]:
    return dedupe_text(
        [
            text(as_dict(row).get("name"))
            for row in as_list(live_gate_payload.get("blockers"))
            if text(as_dict(row).get("status")) == "blocked"
        ]
    )


def review_head(cross_market_payload: dict[str, Any]) -> dict[str, Any]:
    return as_dict(cross_market_payload.get("review_head"))


def review_head_brief(cross_market_payload: dict[str, Any]) -> str:
    head = review_head(cross_market_payload)
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


def build_blocker_items(
    *,
    symbol: str,
    remote_market: str,
    live_gate_payload: dict[str, Any],
    cross_market_payload: dict[str, Any],
    feedback_payload: dict[str, Any],
    policy_payload: dict[str, Any],
    ticket_actionability_payload: dict[str, Any],
    time_sync_verification_payload: dict[str, Any],
    time_sync_repair_plan_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    head = review_head(cross_market_payload)
    head_action = text(head.get("action"))
    head_blocker_detail = (
        text(cross_market_payload.get("review_head_blocker_detail"))
        or text(head.get("blocker_detail"))
    )
    head_done_when = text(cross_market_payload.get("review_head_done_when")) or text(
        head.get("done_when")
    )
    policy_decision = text(policy_payload.get("policy_decision"))
    feedback_ticket_brief = (
        text(feedback_payload.get("ticket_match_brief")) or text(policy_payload.get("ticket_match_brief"))
    )
    feedback_ticket_status = (
        text(feedback_payload.get("ticket_artifact_status"))
        or ("stale_artifact" if "stale_artifact" in feedback_ticket_brief else "")
    )
    feedback_status = text(feedback_payload.get("feedback_status"))
    risk_reason_codes = dedupe_text(
        [text(code) for code in as_list(policy_payload.get("risk_reason_codes"))]
        + [text(feedback_payload.get("dominant_guard_reason"))]
    )
    gates = blocked_gate_names(live_gate_payload)
    time_sync_verification_brief = text(time_sync_verification_payload.get("verification_brief"))
    time_sync_cleared = bool(time_sync_verification_payload.get("cleared", False))
    time_sync_blocked = (bool(time_sync_verification_payload) and not time_sync_cleared) or (
        "time-sync=" in head_blocker_detail
    )
    time_sync_plan_brief = text(time_sync_repair_plan_payload.get("plan_brief"))
    time_sync_environment_classification = text(
        time_sync_repair_plan_payload.get("environment_classification")
    )
    time_sync_environment_blocker_detail = text(
        time_sync_repair_plan_payload.get("environment_blocker_detail")
    )
    time_sync_environment_remediation_hint = text(
        time_sync_repair_plan_payload.get("environment_remediation_hint")
    )
    time_sync_blocker_code = "time_sync_clearance"
    time_sync_blocker_title = "Repair time sync before any orderflow promotion"
    time_sync_next_action = "run_manual_time_repair_then_verify"
    if time_sync_environment_classification == "timed_ntp_via_fake_ip":
        time_sync_blocker_code = "timed_ntp_via_fake_ip_clearance"
        time_sync_blocker_title = "Repair fake-ip NTP path before any orderflow promotion"
        time_sync_next_action = "repair_fake_ip_ntp_path_then_verify"
    time_sync_detail_parts = [
        time_sync_verification_brief or head_blocker_detail,
        time_sync_plan_brief,
    ]
    if time_sync_environment_classification and time_sync_environment_blocker_detail:
        time_sync_detail_parts.append(
            f"env={time_sync_environment_classification}:{time_sync_environment_blocker_detail}"
        )
    if time_sync_environment_remediation_hint:
        time_sync_detail_parts.append(f"fix_hint={time_sync_environment_remediation_hint}")

    items: list[dict[str, Any]] = []

    items.append(
        {
            "priority": 1,
            "blocker_code": time_sync_blocker_code,
            "title": time_sync_blocker_title,
            "status": "blocked" if time_sync_blocked else "cleared",
            "target_artifact": "system_time_sync_repair_verification_report",
            "owner": "system_time_sync",
            "change_class": "LIVE_GUARD_ONLY",
            "clearance_credit": 35,
            "blocker_detail": " | ".join(part for part in time_sync_detail_parts if text(part)),
            "next_action": time_sync_next_action
            if time_sync_blocked
            else "time_sync_clear",
            "done_when": "system time sync repair verification clears and the review head no longer carries a time-sync blocker",
        }
    )

    ticket_actionability_brief = text(ticket_actionability_payload.get("ticket_actionability_brief"))
    if ticket_actionability_brief:
        ticket_blocked = not bool(ticket_actionability_payload.get("actionable_ready", False))
        ticket_title = (
            text(ticket_actionability_payload.get("blocker_title"))
            or "Resolve ticket actionability before guarded canary review"
        )
        ticket_target_artifact = (
            text(ticket_actionability_payload.get("blocker_target_artifact"))
            or "remote_ticket_actionability_state"
        )
        ticket_detail = (
            text(ticket_actionability_payload.get("blocker_detail"))
            or ticket_actionability_brief
        )
        ticket_next_action = (
            text(ticket_actionability_payload.get("next_action"))
            or "resolve_ticket_actionability_before_promotion"
        )
        ticket_done_when = (
            text(ticket_actionability_payload.get("done_when"))
            or "ticket actionability state becomes ready for the current remote route"
        )
    else:
        ticket_blocked = (
            policy_decision in {"reject_until_guardian_clear", "accept_shadow_learning_only"}
            or "ticket_missing" in feedback_ticket_brief
            or feedback_ticket_status.startswith("stale_artifact")
            or feedback_status.startswith("downrank_guardian_blocked")
        )
        ticket_title = "Refresh actionable ticket and clear guardian reject"
        ticket_target_artifact = "remote_intent_queue"
        ticket_detail = " | ".join(
            [
                part
                for part in [
                    text(policy_payload.get("policy_brief")),
                    feedback_ticket_brief,
                    ",".join(risk_reason_codes),
                ]
                if part
            ]
        )
        ticket_next_action = (
            "refresh_signal_ticket_and_clear_guardian_reject"
            if ticket_blocked
            else "ticket_ready_guardian_clear"
        )
        ticket_done_when = (
            "remote intent queue references a fresh actionable ticket and policy stops rejecting the route"
        )
    items.append(
        {
            "priority": 2,
            "blocker_code": "guardian_ticket_actionability",
            "title": ticket_title,
            "status": "blocked" if ticket_blocked else "cleared",
            "target_artifact": ticket_target_artifact,
            "owner": "risk_guard",
            "change_class": "LIVE_GUARD_ONLY",
            "clearance_credit": 30,
            "blocker_detail": ticket_detail,
            "next_action": ticket_next_action,
            "done_when": ticket_done_when,
        }
    )

    review_blocked = bool(head_blocker_detail) or head_action in REVIEW_BLOCKING_ACTIONS
    items.append(
        {
            "priority": 3,
            "blocker_code": "review_head_tradeability",
            "title": "Promote review head from bias-only to tradeable candidate",
            "status": "blocked" if review_blocked else "cleared",
            "target_artifact": "cross_market_review_head",
            "owner": "research_execution_boundary",
            "change_class": "RESEARCH_ONLY",
            "clearance_credit": 20,
            "blocker_detail": head_blocker_detail,
            "next_action": "clear_bias_only_route_requirements" if review_blocked else "review_head_tradeable",
            "done_when": head_done_when
            or "review head no longer advertises deprioritize/watch-only handling and blocker detail clears",
        }
    )

    ops_live_gate_blocked = "ops_live_gate" in gates
    items.append(
        {
            "priority": 4,
            "blocker_code": "ops_live_gate_clearance",
            "title": "Clear ops live gate before any guarded canary review",
            "status": "blocked" if ops_live_gate_blocked else "cleared",
            "target_artifact": "live_gate_blocker_report",
            "owner": "remote_ops",
            "change_class": "LIVE_GUARD_ONLY",
            "clearance_credit": 10,
            "blocker_detail": ",".join(gates),
            "next_action": "clear_ops_live_gate_condition"
            if ops_live_gate_blocked
            else "ops_live_gate_clear",
            "done_when": "live gate blocker report no longer lists ops_live_gate as blocked",
        }
    )

    risk_guard_blocked = "risk_guard" in gates
    items.append(
        {
            "priority": 5,
            "blocker_code": "risk_guard_alignment",
            "title": "Align risk guard candidate with current route",
            "status": "blocked" if risk_guard_blocked else "cleared",
            "target_artifact": "live_gate_blocker_report",
            "owner": "risk_guard",
            "change_class": "LIVE_GUARD_ONLY",
            "clearance_credit": 5,
            "blocker_detail": ",".join(risk_reason_codes) or ",".join(gates),
            "next_action": "align_risk_guard_with_route_symbol" if risk_guard_blocked else "risk_guard_aligned",
            "done_when": "live gate blocker report no longer lists risk_guard as blocked for the current remote route",
        }
    )

    for item in items:
        item["route_symbol"] = symbol
        item["remote_market"] = remote_market
    return items


def build_payload(
    *,
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    cross_market_path: Path,
    cross_market_payload: dict[str, Any],
    feedback_path: Path,
    feedback_payload: dict[str, Any],
    policy_path: Path,
    policy_payload: dict[str, Any],
    ticket_actionability_path: Path | None,
    ticket_actionability_payload: dict[str, Any],
    quality_path: Path,
    quality_payload: dict[str, Any],
    boundary_hold_path: Path,
    boundary_hold_payload: dict[str, Any],
    shadow_clock_path: Path | None,
    shadow_clock_payload: dict[str, Any],
    time_sync_verification_path: Path | None,
    time_sync_verification_payload: dict[str, Any],
    time_sync_repair_plan_path: Path | None,
    time_sync_repair_plan_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = (
        text(boundary_hold_payload.get("route_symbol"))
        or text(quality_payload.get("route_symbol"))
        or text(policy_payload.get("route_symbol"))
        or text(as_dict(cross_market_payload.get("review_head")).get("symbol"))
        or "-"
    )
    remote_market = (
        text(boundary_hold_payload.get("remote_market"))
        or text(quality_payload.get("remote_market"))
        or text(policy_payload.get("remote_market"))
        or "-"
    )
    items = build_blocker_items(
        symbol=symbol,
        remote_market=remote_market,
        live_gate_payload=live_gate_payload,
        cross_market_payload=cross_market_payload,
        feedback_payload=feedback_payload,
        policy_payload=policy_payload,
        ticket_actionability_payload=ticket_actionability_payload,
        time_sync_verification_payload=time_sync_verification_payload,
        time_sync_repair_plan_payload=time_sync_repair_plan_payload,
    )
    blocked_items = [row for row in items if text(row.get("status")) == "blocked"]
    total_credit = sum(int(row.get("clearance_credit") or 0) for row in items) or 1
    cleared_credit = sum(
        int(row.get("clearance_credit") or 0) for row in items if text(row.get("status")) == "cleared"
    )
    clearance_score = max(0, min(100, round((cleared_credit / total_credit) * 100)))
    status = "guardian_blocker_clearance_ready" if not blocked_items else "guardian_blocker_clearance_blocked"
    top_item = blocked_items[0] if blocked_items else items[0]
    shadow_learning_allowed = bool(shadow_clock_payload.get("shadow_learning_allowed", False))
    sync_mode = time_sync_mode(
        time_sync_blocked=text(top_item.get("blocker_code"))
        in {"time_sync_clearance", "timed_ntp_via_fake_ip_clearance"}
        or any(
            text(row.get("blocker_code"))
            in {"time_sync_clearance", "timed_ntp_via_fake_ip_clearance"}
            and text(row.get("status")) == "blocked"
            for row in items
        ),
        shadow_learning_allowed=shadow_learning_allowed,
    )
    brief = ":".join([status, symbol or "-", f"{len(blocked_items)}_blocked", remote_market or "-"])
    done_when = (
        "all guardian clearance items are marked cleared, policy no longer rejects the route, "
        "and the live boundary hold can move from shadow-only review to guarded canary review"
    )
    boundary_hold_blocker_detail = text(boundary_hold_payload.get("blocker_detail"))
    if text(top_item.get("blocker_code")) != "time_sync_clearance" and text(
        top_item.get("blocker_code")
    ) != "timed_ntp_via_fake_ip_clearance":
        boundary_hold_blocker_detail = ""
    blocker_detail = " | ".join(
        dedupe_text(
            [
                text(top_item.get("blocker_detail")),
                boundary_hold_blocker_detail,
                text(quality_payload.get("blocker_detail")),
                text(shadow_clock_payload.get("evidence_brief")),
            ]
        )
    )
    return {
        "action": "build_remote_guardian_blocker_clearance",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "clearance_status": status,
        "clearance_brief": brief,
        "clearance_score": clearance_score,
        "route_symbol": symbol,
        "remote_market": remote_market,
        "blocked_count": len(blocked_items),
        "total_count": len(items),
        "top_blocker_code": text(top_item.get("blocker_code")),
        "top_blocker_title": text(top_item.get("title")),
        "top_blocker_target_artifact": text(top_item.get("target_artifact")),
        "top_blocker_next_action": text(top_item.get("next_action")),
        "top_blocker_done_when": text(top_item.get("done_when")),
        "top_blocker_detail": text(top_item.get("blocker_detail")),
        "quality_brief": text(quality_payload.get("quality_brief")),
        "quality_score": quality_payload.get("quality_score"),
        "policy_brief": text(policy_payload.get("policy_brief")),
        "policy_decision": text(policy_payload.get("policy_decision")),
        "feedback_brief": text(feedback_payload.get("feedback_brief")),
        "ticket_actionability_brief": text(ticket_actionability_payload.get("ticket_actionability_brief")),
        "ticket_actionability_status": text(ticket_actionability_payload.get("ticket_actionability_status")),
        "ticket_actionability_decision": text(
            ticket_actionability_payload.get("ticket_actionability_decision")
        ),
        "remote_shadow_clock_evidence_brief": text(shadow_clock_payload.get("evidence_brief")),
        "remote_shadow_clock_evidence_status": text(shadow_clock_payload.get("evidence_status")),
        "remote_shadow_clock_shadow_learning_allowed": shadow_learning_allowed,
        "time_sync_mode": sync_mode,
        "hold_brief": text(boundary_hold_payload.get("hold_brief")),
        "review_head_brief": review_head_brief(cross_market_payload),
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "clearance_items": items,
        "artifacts": {
            "live_gate_blocker_report": str(live_gate_path),
            "cross_market_operator_state": str(cross_market_path),
            "remote_orderflow_feedback": str(feedback_path),
            "remote_orderflow_policy_state": str(policy_path),
            "remote_ticket_actionability_state": str(ticket_actionability_path)
            if ticket_actionability_path
            else "",
            "remote_orderflow_quality_report": str(quality_path),
            "remote_live_boundary_hold": str(boundary_hold_path),
            "remote_shadow_clock_evidence": str(shadow_clock_path) if shadow_clock_path else "",
            "system_time_sync_repair_verification_report": (
                str(time_sync_verification_path) if time_sync_verification_path else ""
            ),
            "system_time_sync_repair_plan": str(time_sync_repair_plan_path)
            if time_sync_repair_plan_path
            else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Remote Guardian Blocker Clearance",
        "",
        f"- brief: `{text(payload.get('clearance_brief'))}`",
        f"- score: `{text(payload.get('clearance_score'))}`",
        f"- time_sync_mode: `{text(payload.get('time_sync_mode'))}`",
        f"- shadow_clock_evidence: `{text(payload.get('remote_shadow_clock_evidence_brief')) or '-'}`",
        f"- ticket_actionability: `{text(payload.get('ticket_actionability_brief')) or '-'}`",
        f"- top_blocker: `{text(payload.get('top_blocker_code'))}` -> `{text(payload.get('top_blocker_target_artifact'))}`",
        f"- next_action: `{text(payload.get('top_blocker_next_action'))}`",
        f"- blocker: `{text(payload.get('top_blocker_detail'))}`",
        f"- done_when: `{text(payload.get('done_when'))}`",
        "",
        "## Clearance Items",
    ]
    for row in as_list(payload.get("clearance_items")):
        item = as_dict(row)
        lines.append(
            "- "
            + f"`P{text(item.get('priority'))}` "
            + f"`{text(item.get('blocker_code'))}` "
            + f"status=`{text(item.get('status'))}` "
            + f"target=`{text(item.get('target_artifact'))}` "
            + f"next_action=`{text(item.get('next_action'))}` "
            + f"detail=`{text(item.get('blocker_detail'))}`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote guardian blocker clearance artifact.")
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
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json", reference_now)
    cross_market_path = find_latest(review_dir, "*_cross_market_operator_state.json", reference_now)
    feedback_path = find_latest(review_dir, "*_remote_orderflow_feedback.json", reference_now)
    policy_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json", reference_now)
    ticket_actionability_path = find_latest(
        review_dir, "*_remote_ticket_actionability_state.json", reference_now
    )
    quality_path = find_latest(review_dir, "*_remote_orderflow_quality_report.json", reference_now)
    boundary_hold_path = find_latest(review_dir, "*_remote_live_boundary_hold.json", reference_now)
    shadow_clock_path = find_latest(review_dir, "*_remote_shadow_clock_evidence.json", reference_now)
    time_sync_verification_path = find_latest(
        review_dir, "*_system_time_sync_repair_verification_report.json"
    )
    time_sync_repair_plan_path = find_latest(
        review_dir, "*_system_time_sync_repair_plan.json", reference_now
    )
    missing = [
        name
        for name, path in (
            ("live_gate_blocker_report", live_gate_path),
            ("cross_market_operator_state", cross_market_path),
            ("remote_orderflow_feedback", feedback_path),
            ("remote_orderflow_policy_state", policy_path),
            ("remote_orderflow_quality_report", quality_path),
            ("remote_live_boundary_hold", boundary_hold_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path),
        cross_market_path=cross_market_path,
        cross_market_payload=load_json_mapping(cross_market_path),
        feedback_path=feedback_path,
        feedback_payload=load_json_mapping(feedback_path),
        policy_path=policy_path,
        policy_payload=load_json_mapping(policy_path),
        ticket_actionability_path=ticket_actionability_path,
        ticket_actionability_payload=load_json_mapping(ticket_actionability_path)
        if ticket_actionability_path is not None and ticket_actionability_path.exists()
        else {},
        quality_path=quality_path,
        quality_payload=load_json_mapping(quality_path),
        boundary_hold_path=boundary_hold_path,
        boundary_hold_payload=load_json_mapping(boundary_hold_path),
        shadow_clock_path=shadow_clock_path,
        shadow_clock_payload=load_json_mapping(shadow_clock_path)
        if shadow_clock_path is not None and shadow_clock_path.exists()
        else {},
        time_sync_verification_path=time_sync_verification_path,
        time_sync_verification_payload=load_json_mapping(time_sync_verification_path)
        if time_sync_verification_path is not None and time_sync_verification_path.exists()
        else {},
        time_sync_repair_plan_path=time_sync_repair_plan_path,
        time_sync_repair_plan_payload=load_json_mapping(time_sync_repair_plan_path)
        if time_sync_repair_plan_path is not None and time_sync_repair_plan_path.exists()
        else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_guardian_blocker_clearance.json"
    markdown = review_dir / f"{stamp}_remote_guardian_blocker_clearance.md"
    checksum = review_dir / f"{stamp}_remote_guardian_blocker_clearance_checksum.json"
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
