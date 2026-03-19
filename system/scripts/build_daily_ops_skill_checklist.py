#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_PLAYBOOK_PATH = SYSTEM_ROOT / "docs" / "FENLIE_SKILL_USAGE_PLAYBOOK.md"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5

SKILL_LIBRARY: list[dict[str, str]] = [
    {
        "id": "source_ownership_review",
        "label": "Source Ownership Review",
        "skill_path": "/Users/jokenrobot/.codex/skills/fenlie-source-ownership-review",
        "command": (
            "python3 /Users/jokenrobot/.codex/skills/fenlie-source-ownership-review/scripts/"
            "run_source_ownership_review.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie"
        ),
        "success_focus": "Confirm source-vs-consumer drift is clean before further changes.",
    },
    {
        "id": "cross_market_refresh_audit",
        "label": "Cross-Market Refresh Audit",
        "skill_path": "/Users/jokenrobot/.codex/skills/fenlie-cross-market-refresh-audit",
        "command": (
            "python3 /Users/jokenrobot/.codex/skills/fenlie-cross-market-refresh-audit/scripts/"
            "run_cross_market_refresh_audit.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie "
            "--skip-downstream-refresh"
        ),
        "success_focus": "Verify review/operator/repair lanes and action queue semantics after source changes.",
    },
    {
        "id": "remote_live_guard_diagnostics",
        "label": "Remote Live Guard Diagnostics",
        "skill_path": "/Users/jokenrobot/.codex/skills/fenlie-remote-live-guard-diagnostics",
        "command": (
            "python3 /Users/jokenrobot/.codex/skills/fenlie-remote-live-guard-diagnostics/scripts/"
            "run_remote_live_guard_diagnostics.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie "
            "--skip-downstream-refresh"
        ),
        "success_focus": "Verify scope alignment, takeover gate, clearing, and repair queue.",
    },
    {
        "id": "operator_panel_refresh",
        "label": "Operator Panel Refresh",
        "skill_path": "/Users/jokenrobot/.codex/skills/fenlie-operator-panel-refresh",
        "command": (
            "python3 /Users/jokenrobot/.codex/skills/fenlie-operator-panel-refresh/scripts/"
            "run_operator_panel_refresh.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie"
        ),
        "success_focus": "Refresh monitoring panel so UI reflects latest source-owned state.",
    },
]


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


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    future_cutoff = reference_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = -1.0
    return (0 if is_future else 1, artifact_stamp(path), mtime, path.name)


def latest_review_json_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime) -> Path | None:
    candidates: list[Path] = []
    for item in review_dir.glob(f"*_{suffix}.json"):
        if not item.exists():
            continue
        candidates.append(item)
    ranked = sorted(candidates, key=lambda item: artifact_sort_key(item, reference_now), reverse=True)
    for path in ranked:
        if path.exists():
            return path
    return None


def load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime,
) -> tuple[list[str], list[str]]:
    cutoff = now_dt - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda item: item.stat().st_mtime, reverse=True)
    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
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


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return int(default)


def leading_status(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    return text.split(":", 1)[0].strip()


def build_skill_rows(
    *,
    hot_brief: dict[str, Any],
    cross_market: dict[str, Any],
    panel_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    review_head_brief = safe_text(hot_brief.get("cross_market_review_head_brief"))
    panel_summary = dict(panel_payload.get("summary") or {})
    review_head_status = safe_text(hot_brief.get("cross_market_review_head_status")) or leading_status(
        review_head_brief
        or cross_market.get("review_head_lane_brief")
        or panel_summary.get("review_head_brief")
    )
    remote_live_gate_brief = safe_text(hot_brief.get("cross_market_remote_live_takeover_gate_brief"))
    repair_count = safe_int(hot_brief.get("cross_market_operator_repair_backlog_count"))
    panel_head = safe_text(panel_summary.get("review_head_brief") or panel_payload.get("review_head_brief"))
    rows: list[dict[str, Any]] = []
    for idx, definition in enumerate(SKILL_LIBRARY, start=1):
        skill_id = definition["id"]
        status = "standby"
        why = "Routine ops check."
        if skill_id == "source_ownership_review":
            status = "run_now" if review_head_status == "refresh_required" else "verify_clean"
            why = (
                f"Current review head is `{review_head_brief}`."
                if review_head_brief
                else "Use first whenever source ownership is in doubt."
            )
        elif skill_id == "cross_market_refresh_audit":
            status = "run_now" if review_head_status in {"refresh_required", "review"} else "verify_only"
            why = f"Current review head brief: `{review_head_brief or '-'}`."
        elif skill_id == "remote_live_guard_diagnostics":
            status = "run_now" if remote_live_gate_brief or repair_count > 0 else "optional"
            why = f"Current remote-live gate: `{remote_live_gate_brief or '-'}`."
        elif skill_id == "operator_panel_refresh":
            status = "run_after_source_checks"
            why = (
                f"Current panel review head: `{panel_head or review_head_brief or '-'}`."
            )
        rows.append(
            {
                "rank": idx,
                "skill_id": skill_id,
                "label": definition["label"],
                "status": status,
                "why": why,
                "command": definition["command"],
                "skill_path": definition["skill_path"],
                "success_focus": definition["success_focus"],
            }
        )
    return rows


def checklist_brief(rows: list[dict[str, Any]], limit: int = 4) -> str:
    parts = [
        f"{int(row.get('rank') or 0)}:{safe_text(row.get('skill_id'))}:{safe_text(row.get('status'))}"
        for row in rows[:limit]
    ]
    if not parts:
        return "-"
    return " | ".join(parts) + (f" | +{len(rows) - limit}" if len(rows) > limit else "")


def derive_priority_repair(
    hot_brief: dict[str, Any],
    promotion_unblock_readiness: dict[str, Any] | None = None,
) -> dict[str, str]:
    readiness = dict(promotion_unblock_readiness or {})
    readiness_status = safe_text(readiness.get("readiness_status"))
    if readiness_status in {
        "local_time_sync_primary_blocker_shadow_ready",
        "shadow_ready_ticket_actionability_blocked",
    }:
        symbol = safe_text(readiness.get("route_symbol"))
        scope = safe_text(readiness.get("primary_blocker_scope")) or (
            "local_admin_repair"
            if readiness_status == "local_time_sync_primary_blocker_shadow_ready"
            else "guardian_ticket_actionability"
        )
        decision = (
            safe_text(readiness.get("readiness_decision"))
            or (
                "repair_local_time_sync_then_review_guarded_canary"
                if readiness_status == "local_time_sync_primary_blocker_shadow_ready"
                else "resolve_ticket_actionability_then_review_guarded_canary"
            )
        )
        brief = (
            f"{decision}:{symbol}:{scope}"
            if symbol
            else f"{decision}:{scope}"
        )
        return {
            "status": "run_now",
            "brief": brief,
            "blocker_detail": safe_text(readiness.get("primary_local_repair_detail"))
            or safe_text(readiness.get("blocker_detail"))
            or safe_text(readiness.get("readiness_brief")),
            "done_when": safe_text(readiness.get("done_when")),
            "target_artifact": safe_text(readiness.get("primary_local_repair_target_artifact")),
            "source_artifact": safe_text(readiness.get("artifact")),
            "source_brief": safe_text(readiness.get("readiness_brief")),
        }
    review_head_brief = safe_text(hot_brief.get("cross_market_review_head_brief"))
    blocker_detail = safe_text(hot_brief.get("cross_market_review_head_blocker_detail"))
    done_when = safe_text(hot_brief.get("cross_market_review_head_done_when"))
    if "time-sync=threshold_breach:" not in blocker_detail:
        return {
            "status": "",
            "brief": "",
            "blocker_detail": "",
            "done_when": "",
            "target_artifact": "",
            "source_artifact": "",
            "source_brief": "",
        }
    symbol = safe_text(hot_brief.get("cross_market_review_head_symbol")) or review_head_brief.split(":")[2] if review_head_brief.count(":") >= 2 else ""
    scope = ""
    marker = "time-sync=threshold_breach:scope="
    if marker in blocker_detail:
        scope = blocker_detail.split(marker, 1)[1].split(";", 1)[0].strip()
    brief = f"repair_time_sync:{symbol}:{scope or 'threshold_breach'}" if symbol else f"repair_time_sync:{scope or 'threshold_breach'}"
    return {
        "status": "run_now",
        "brief": brief,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "target_artifact": "system_time_sync_repair_verification_report",
        "source_artifact": "",
        "source_brief": "",
    }


def attach_repair_plan(
    *,
    review_dir: Path,
    reference_now: dt.datetime,
    priority_repair: dict[str, str],
) -> dict[str, str]:
    if not safe_text(priority_repair.get("brief")) or safe_text(
        priority_repair.get("target_artifact")
    ) != "system_time_sync_repair_verification_report":
        return {
            "plan_artifact": "",
            "plan_brief": "",
        }
    plan_path = latest_review_json_artifact(review_dir, "system_time_sync_repair_plan", reference_now)
    plan_payload = load_json_mapping(plan_path) if plan_path else {}
    return {
        "plan_artifact": str(plan_path) if plan_path else "",
        "plan_brief": safe_text(plan_payload.get("plan_brief")),
    }


def attach_repair_verification(
    *,
    review_dir: Path,
    reference_now: dt.datetime,
    priority_repair: dict[str, str],
) -> dict[str, str]:
    if not safe_text(priority_repair.get("brief")) or safe_text(
        priority_repair.get("target_artifact")
    ) != "system_time_sync_repair_verification_report":
        return {
            "verification_artifact": "",
            "verification_status": "",
            "verification_brief": "",
        }
    verification_path = latest_review_json_artifact(review_dir, "system_time_sync_repair_verification_report", reference_now)
    verification_payload = load_json_mapping(verification_path) if verification_path else {}
    return {
        "verification_artifact": str(verification_path) if verification_path else "",
        "verification_status": safe_text(verification_payload.get("status")),
        "verification_brief": safe_text(verification_payload.get("verification_brief")),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    rows = [dict(row) for row in list(payload.get("skills") or []) if isinstance(row, dict)]
    lines = [
        "# Daily Ops Skill Checklist",
        "",
        f"- generated_at_utc: `{safe_text(payload.get('generated_at_utc'))}`",
        f"- operator_head: `{safe_text(payload.get('operator_head_brief'))}`",
        f"- review_head: `{safe_text(payload.get('review_head_brief'))}`",
        f"- repair_head: `{safe_text(payload.get('repair_head_brief'))}`",
        f"- remote_live_gate: `{safe_text(payload.get('remote_live_gate_brief'))}`",
        f"- lane_state: `{safe_text(payload.get('lane_state_brief'))}`",
        f"- priority_repair: `{safe_text(payload.get('priority_repair_brief')) or '-'}`",
        f"- checklist_brief: `{safe_text(payload.get('checklist_brief'))}`",
        "",
    ]
    if safe_text(payload.get("priority_repair_status")):
        lines.extend(
            [
                "## Priority Repair",
                "",
                f"- status: `{safe_text(payload.get('priority_repair_status'))}`",
                f"- brief: `{safe_text(payload.get('priority_repair_brief'))}`",
                f"- blocker_detail: `{safe_text(payload.get('priority_repair_blocker_detail'))}`",
                f"- done_when: `{safe_text(payload.get('priority_repair_done_when'))}`",
                f"- target_artifact: `{safe_text(payload.get('priority_repair_target_artifact')) or '-'}`",
                f"- source_artifact: `{safe_text(payload.get('priority_repair_source_artifact')) or '-'}`",
                f"- source_brief: `{safe_text(payload.get('priority_repair_source_brief')) or '-'}`",
                f"- plan_artifact: `{safe_text(payload.get('priority_repair_plan_artifact')) or '-'}`",
                f"- verification_status: `{safe_text(payload.get('priority_repair_verification_status')) or '-'}`",
                f"- verification_brief: `{safe_text(payload.get('priority_repair_verification_brief')) or '-'}`",
                f"- verification_artifact: `{safe_text(payload.get('priority_repair_verification_artifact')) or '-'}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Skills",
        "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {int(row.get('rank') or 0)}. {safe_text(row.get('label'))}",
                f"- status: `{safe_text(row.get('status'))}`",
                f"- why: {safe_text(row.get('why'))}",
                f"- command: `{safe_text(row.get('command'))}`",
                f"- success_focus: {safe_text(row.get('success_focus'))}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a daily Fenlie ops skill checklist.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--playbook-path", type=Path, default=DEFAULT_PLAYBOOK_PATH)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-keep", type=int, default=20)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    args = parser.parse_args()

    reference_now = parse_now(args.now)
    review_dir = args.review_dir.resolve()
    playbook_path = args.playbook_path.resolve()

    hot_path = latest_review_json_artifact(review_dir, "hot_universe_operator_brief", reference_now)
    cross_path = latest_review_json_artifact(review_dir, "cross_market_operator_state", reference_now)
    panel_path = latest_review_json_artifact(review_dir, "operator_task_visual_panel", reference_now)
    if hot_path is None or cross_path is None:
        raise FileNotFoundError("missing required hot_brief or cross_market artifact")

    hot_payload = load_json_mapping(hot_path)
    cross_payload = load_json_mapping(cross_path)
    panel_payload = load_json_mapping(panel_path) if panel_path else {}
    promotion_unblock_readiness_path = latest_review_json_artifact(
        review_dir, "remote_promotion_unblock_readiness", reference_now
    )
    promotion_unblock_readiness_payload = (
        load_json_mapping(promotion_unblock_readiness_path)
        if promotion_unblock_readiness_path
        else {}
    )

    rows = build_skill_rows(hot_brief=hot_payload, cross_market=cross_payload, panel_payload=panel_payload)
    priority_repair = derive_priority_repair(
        hot_payload,
        promotion_unblock_readiness=promotion_unblock_readiness_payload,
    )
    priority_repair_plan = attach_repair_plan(
        review_dir=review_dir,
        reference_now=reference_now,
        priority_repair=priority_repair,
    )
    priority_repair_verification = attach_repair_verification(
        review_dir=review_dir,
        reference_now=reference_now,
        priority_repair=priority_repair,
    )

    summary_panel = dict(panel_payload.get("summary") or {})
    payload: dict[str, Any] = {
        "action": "build_daily_ops_skill_checklist",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "playbook_path": str(playbook_path),
        "hot_brief_artifact": str(hot_path),
        "cross_market_artifact": str(cross_path),
        "operator_panel_artifact": str(panel_path) if panel_path else "",
        "operator_head_brief": safe_text(
            hot_payload.get("cross_market_operator_head_brief") or summary_panel.get("operator_head_brief")
        ),
        "review_head_brief": safe_text(
            hot_payload.get("cross_market_review_head_brief") or summary_panel.get("review_head_brief")
        ),
        "repair_head_brief": safe_text(
            hot_payload.get("cross_market_operator_repair_head_brief") or summary_panel.get("repair_head_brief")
        ),
        "remote_live_gate_brief": safe_text(
            hot_payload.get("cross_market_remote_live_takeover_gate_brief")
            or summary_panel.get("remote_live_gate_brief")
        ),
        "lane_state_brief": safe_text(
            hot_payload.get("cross_market_operator_backlog_state_brief")
            or summary_panel.get("lane_state_brief")
        ),
        "priority_repair_status": safe_text(priority_repair.get("status")),
        "priority_repair_brief": safe_text(priority_repair.get("brief")),
        "priority_repair_blocker_detail": safe_text(priority_repair.get("blocker_detail")),
        "priority_repair_done_when": safe_text(priority_repair.get("done_when")),
        "priority_repair_target_artifact": safe_text(priority_repair.get("target_artifact")),
        "priority_repair_source_artifact": safe_text(priority_repair.get("source_artifact")),
        "priority_repair_source_brief": safe_text(priority_repair.get("source_brief")),
        "priority_repair_plan_artifact": safe_text(priority_repair_plan.get("plan_artifact")),
        "priority_repair_plan_brief": safe_text(priority_repair_plan.get("plan_brief")),
        "priority_repair_verification_artifact": safe_text(priority_repair_verification.get("verification_artifact")),
        "priority_repair_verification_status": safe_text(priority_repair_verification.get("verification_status")),
        "priority_repair_verification_brief": safe_text(priority_repair_verification.get("verification_brief")),
        "skills": rows,
        "checklist_brief": checklist_brief(rows),
    }

    stem = f"{reference_now.strftime('%Y%m%dT%H%M%SZ')}_daily_ops_skill_checklist"
    artifact_path = review_dir / f"{stem}.json"
    markdown_path = review_dir / f"{stem}.md"
    checksum_path = review_dir / f"{stem}_checksum.json"

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(build_markdown(payload), encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "markdown": str(markdown_path),
                "markdown_sha256": sha256_file(markdown_path),
                "generated_at_utc": fmt_utc(reference_now),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="daily_ops_skill_checklist",
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
        now_dt=reference_now,
    )

    payload.update(
        {
            "artifact": str(artifact_path),
            "markdown": str(markdown_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "markdown": str(markdown_path),
                "markdown_sha256": sha256_file(markdown_path),
                "generated_at_utc": fmt_utc(reference_now),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
