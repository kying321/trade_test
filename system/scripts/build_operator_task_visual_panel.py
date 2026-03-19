#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_DASHBOARD_DIST = SYSTEM_ROOT / "dashboard" / "web" / "dist"


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
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def latest_review_json_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        review_dir.glob(f"*_{suffix}.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def load_optional_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = load_json_mapping(path)
    except Exception:
        return {}
    payload.setdefault("artifact", str(path))
    return payload


def resolve_source_path(
    *,
    source_payload: dict[str, Any],
    artifact_key: str,
    review_dir: Path,
    suffix: str,
    reference_now: dt.datetime,
) -> Path | None:
    explicit = Path(str(source_payload.get(artifact_key) or "")).expanduser()
    if str(source_payload.get(artifact_key) or "").strip() and explicit.exists():
        return explicit
    return latest_review_json_artifact(review_dir, suffix, reference_now)


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
    now_dt: dt.datetime | None = None,
) -> tuple[list[str], list[str]]:
    effective_now = now_dt or now_utc()
    cutoff = effective_now - dt.timedelta(hours=max(1.0, ttl_hours))
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
    for path in survivors[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def compact(parts: list[str]) -> str:
    return " | ".join([str(part).strip() for part in parts if str(part).strip()])


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def safe_row(row: dict[str, Any] | None) -> dict[str, Any]:
    return dict(row or {})


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return list(value) if isinstance(value, tuple) else []


def path_brief(path_text: str) -> str:
    path = Path(path_text)
    if not path_text:
        return "-"
    if len(path.parts) >= 4:
        return str(Path(*path.parts[-4:]))
    return path_text


def build_lane_cards(cross_market: dict[str, Any]) -> list[dict[str, Any]]:
    counts = dict(cross_market.get("operator_backlog_state_counts") or {})
    totals = dict(cross_market.get("operator_backlog_priority_totals") or {})
    cards: list[dict[str, Any]] = []
    for state in ("waiting", "review", "watch", "blocked", "repair"):
        cards.append(
            {
                "state": state,
                "count": safe_int(counts.get(state)),
                "priority_total": safe_int(totals.get(state)),
                "brief": safe_text(cross_market.get(f"operator_{state}_lane_brief")),
                "head_symbol": safe_text(cross_market.get(f"operator_{state}_lane_head_symbol")),
                "head_action": safe_text(cross_market.get(f"operator_{state}_lane_head_action")),
            }
        )
    return cards


def build_impact_rows(
    *,
    hot_brief: dict[str, Any],
    cross_market: dict[str, Any],
) -> list[dict[str, Any]]:
    operator_head = safe_row(cross_market.get("operator_head_lane")).get("head") or {}
    review_head = safe_row(cross_market.get("review_head_lane")).get("head") or {}
    repair_head = safe_row(cross_market.get("operator_repair_head_lane")).get("head") or {}
    return [
        {
            "lane": "operator_head",
            "symbol": safe_text(operator_head.get("symbol")),
            "action": safe_text(operator_head.get("action")),
            "impact_scope": "primary slot, action queue rank 1, operator head lane, remote-live alignment",
            "blocker_detail": safe_text(operator_head.get("blocker_detail")),
            "done_when": safe_text(operator_head.get("done_when")),
        },
        {
            "lane": "review_head",
            "symbol": safe_text(review_head.get("symbol")),
            "action": safe_text(review_head.get("action")),
            "impact_scope": "secondary focus, review head lane, action queue rank 3, cross-market review backlog",
            "blocker_detail": safe_text(review_head.get("blocker_detail")),
            "done_when": safe_text(review_head.get("done_when")),
        },
        {
            "lane": "repair_head",
            "symbol": safe_text(repair_head.get("symbol")),
            "action": safe_text(repair_head.get("action")),
            "impact_scope": "repair queue head, action queue rank 4, remote-live clearing, live gate repair backlog",
            "blocker_detail": safe_text(hot_brief.get("cross_market_operator_repair_head_done_when") or repair_head.get("reason")),
            "done_when": safe_text(repair_head.get("clear_when") or repair_head.get("done_when")),
        },
    ]


def build_source_rows(
    *,
    hot_brief: dict[str, Any],
    source_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    definitions = [
        (
            "cross_market",
            "Cross-Market Operator State",
            safe_text(hot_brief.get("source_cross_market_operator_state_artifact")),
            safe_text(hot_brief.get("source_cross_market_operator_state_status")),
            safe_text(hot_brief.get("source_cross_market_operator_state_as_of")),
            safe_text(hot_brief.get("source_cross_market_operator_state_snapshot_brief")),
        ),
        (
            "crypto_route_refresh",
            "Crypto Route Refresh",
            safe_text(hot_brief.get("source_crypto_route_refresh_artifact")),
            safe_text(hot_brief.get("source_crypto_route_refresh_status")),
            safe_text(hot_brief.get("source_crypto_route_refresh_as_of")),
            safe_text(hot_brief.get("source_crypto_route_refresh_reuse_gate_brief")),
        ),
        (
            "live_gate_blocker",
            "Live Gate Blocker",
            safe_text(hot_brief.get("source_live_gate_blocker_artifact")),
            safe_text(source_payloads.get("live_gate_blocker", {}).get("status")),
            safe_text(source_payloads.get("live_gate_blocker", {}).get("as_of")),
            safe_text(hot_brief.get("source_live_gate_blocker_remote_live_diagnosis_brief")),
        ),
        (
            "remote_live_handoff",
            "Remote Live Handoff",
            safe_text(hot_brief.get("source_remote_live_handoff_artifact")),
            safe_text(source_payloads.get("remote_live_handoff", {}).get("status")),
            safe_text(source_payloads.get("remote_live_handoff", {}).get("as_of")),
            safe_text(hot_brief.get("source_remote_live_handoff_ready_check_scope_brief")),
        ),
        (
            "remote_live_history",
            "Remote Live History Audit",
            safe_text(hot_brief.get("source_remote_live_history_audit_artifact")),
            safe_text(source_payloads.get("remote_live_history", {}).get("status")),
            safe_text(source_payloads.get("remote_live_history", {}).get("generated_at_utc")),
            safe_text(hot_brief.get("source_remote_live_history_audit_window_brief")),
        ),
        (
            "brooks_refresh",
            "Brooks Structure Refresh",
            safe_text(hot_brief.get("source_brooks_structure_refresh_artifact")),
            safe_text(source_payloads.get("brooks_refresh", {}).get("status")),
            safe_text(source_payloads.get("brooks_refresh", {}).get("as_of")),
            safe_text(hot_brief.get("source_brooks_structure_refresh_brief")),
        ),
        (
            "brooks_queue",
            "Brooks Review Queue",
            safe_text(hot_brief.get("source_brooks_structure_review_queue_artifact")),
            safe_text(source_payloads.get("brooks_queue", {}).get("status")),
            safe_text(source_payloads.get("brooks_queue", {}).get("as_of")),
            safe_text(hot_brief.get("source_brooks_structure_review_queue_brief")),
        ),
        (
            "time_sync_repair_plan",
            "System Time Sync Repair Plan",
            safe_text(hot_brief.get("source_system_time_sync_repair_plan_artifact")),
            safe_text(hot_brief.get("source_system_time_sync_repair_plan_status")),
            safe_text(source_payloads.get("time_sync_repair_plan", {}).get("generated_at_utc")),
            safe_text(hot_brief.get("source_system_time_sync_repair_plan_brief")),
        ),
        (
            "time_sync_repair_verification",
            "System Time Sync Repair Verification",
            safe_text(hot_brief.get("source_system_time_sync_repair_verification_artifact")),
            safe_text(hot_brief.get("source_system_time_sync_repair_verification_status")),
            safe_text(source_payloads.get("time_sync_repair_verification", {}).get("generated_at_utc")),
            safe_text(hot_brief.get("source_system_time_sync_repair_verification_brief")),
        ),
        (
            "openclaw_orderflow_blueprint",
            "OpenClaw Orderflow Blueprint",
            safe_text(hot_brief.get("source_openclaw_orderflow_blueprint_artifact")),
            safe_text(hot_brief.get("source_openclaw_orderflow_blueprint_status")),
            safe_text(source_payloads.get("openclaw_orderflow_blueprint", {}).get("generated_at_utc")),
            safe_text(hot_brief.get("source_openclaw_orderflow_blueprint_brief")),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for key, label, artifact, status, as_of, brief in definitions:
        rows.append(
            {
                "key": key,
                "label": label,
                "artifact": artifact,
                "artifact_brief": path_brief(artifact),
                "status": status or "-",
                "as_of": as_of or "-",
                "brief": brief or "-",
            }
        )
    return rows


def build_chain_data(
    *,
    hot_brief: dict[str, Any],
    cross_market: dict[str, Any],
) -> dict[str, Any]:
    nodes = [
        {"id": "commodity", "label": "Commodity Lanes", "kind": "source", "x": 90, "y": 110, "brief": safe_text(hot_brief.get("cross_market_operator_waiting_lane_brief")) or "-"},
        {"id": "brooks", "label": "Brooks Refresh", "kind": "source", "x": 90, "y": 220, "brief": safe_text(hot_brief.get("source_brooks_structure_refresh_brief")) or "-"},
        {"id": "crypto", "label": "Crypto Route", "kind": "source", "x": 90, "y": 330, "brief": safe_text(hot_brief.get("source_crypto_route_refresh_reuse_gate_brief")) or "-"},
        {"id": "live_gate", "label": "Live Gate", "kind": "source", "x": 90, "y": 440, "brief": safe_text(hot_brief.get("source_live_gate_blocker_remote_live_diagnosis_brief")) or "-"},
        {"id": "cross_market", "label": "Cross-Market State", "kind": "aggregate", "x": 360, "y": 275, "brief": safe_text(hot_brief.get("source_cross_market_operator_state_snapshot_brief")) or "-"},
        {"id": "hot_brief", "label": "Hot Operator Brief", "kind": "aggregate", "x": 610, "y": 275, "brief": safe_text(hot_brief.get("operator_action_queue_brief")) or "-"},
        {"id": "primary", "label": "Primary Head", "kind": "head", "x": 890, "y": 110, "brief": safe_text(hot_brief.get("cross_market_operator_head_brief")) or "-"},
        {"id": "review", "label": "Review Head", "kind": "head", "x": 890, "y": 250, "brief": safe_text(hot_brief.get("cross_market_review_head_brief")) or "-"},
        {"id": "repair", "label": "Repair Head", "kind": "head", "x": 890, "y": 390, "brief": safe_text(hot_brief.get("cross_market_operator_repair_head_brief")) or "-"},
        {"id": "remote", "label": "Remote Takeover Gate", "kind": "gate", "x": 1160, "y": 250, "brief": safe_text(hot_brief.get("cross_market_remote_live_takeover_gate_brief")) or "-"},
    ]
    edges = [
        {"from": "commodity", "to": "cross_market", "label": "waiting lane"},
        {"from": "brooks", "to": "cross_market", "label": "review lane"},
        {"from": "crypto", "to": "cross_market", "label": "review lane"},
        {"from": "live_gate", "to": "cross_market", "label": "repair lane"},
        {"from": "cross_market", "to": "hot_brief", "label": "aggregated stack"},
        {"from": "hot_brief", "to": "primary", "label": "slot primary"},
        {"from": "hot_brief", "to": "review", "label": "slot secondary"},
        {"from": "hot_brief", "to": "repair", "label": "repair slot"},
        {"from": "cross_market", "to": "remote", "label": "takeover gate"},
        {"from": "live_gate", "to": "remote", "label": "live blockers"},
    ]
    return {"nodes": nodes, "edges": edges, "lane_order": safe_text(cross_market.get("operator_state_lane_priority_order_brief"))}


def build_panel_payload(
    *,
    review_dir: Path,
    dashboard_dist: Path,
    reference_now: dt.datetime,
) -> dict[str, Any]:
    hot_brief_path = latest_review_json_artifact(review_dir, "hot_universe_operator_brief", reference_now)
    if hot_brief_path is None:
        raise RuntimeError("operator_task_visual_panel_missing_hot_brief")
    hot_brief = load_json_mapping(hot_brief_path)
    hot_brief.setdefault("artifact", str(hot_brief_path))

    cross_market_path = resolve_source_path(
        source_payload=hot_brief,
        artifact_key="source_cross_market_operator_state_artifact",
        review_dir=review_dir,
        suffix="cross_market_operator_state",
        reference_now=reference_now,
    )
    if cross_market_path is None:
        raise RuntimeError("operator_task_visual_panel_missing_cross_market")
    cross_market = load_json_mapping(cross_market_path)
    cross_market.setdefault("artifact", str(cross_market_path))

    source_paths = {
        "crypto_route_refresh": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_crypto_route_refresh_artifact",
            review_dir=review_dir,
            suffix="crypto_route_refresh",
            reference_now=reference_now,
        ),
        "live_gate_blocker": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_live_gate_blocker_artifact",
            review_dir=review_dir,
            suffix="live_gate_blocker_report",
            reference_now=reference_now,
        ),
        "remote_live_handoff": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_remote_live_handoff_artifact",
            review_dir=review_dir,
            suffix="remote_live_handoff",
            reference_now=reference_now,
        ),
        "remote_live_history": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_remote_live_history_audit_artifact",
            review_dir=review_dir,
            suffix="remote_live_history_audit",
            reference_now=reference_now,
        ),
        "brooks_refresh": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_brooks_structure_refresh_artifact",
            review_dir=review_dir,
            suffix="brooks_structure_refresh",
            reference_now=reference_now,
        ),
        "brooks_queue": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_brooks_structure_review_queue_artifact",
            review_dir=review_dir,
            suffix="brooks_structure_review_queue",
            reference_now=reference_now,
        ),
        "brooks_route_report": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_brooks_route_report_artifact",
            review_dir=review_dir,
            suffix="brooks_price_action_route_report",
            reference_now=reference_now,
        ),
        "brooks_execution_plan": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_brooks_execution_plan_artifact",
            review_dir=review_dir,
            suffix="brooks_price_action_execution_plan",
            reference_now=reference_now,
        ),
        "time_sync_repair_plan": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_system_time_sync_repair_plan_artifact",
            review_dir=review_dir,
            suffix="system_time_sync_repair_plan",
            reference_now=reference_now,
        ),
        "time_sync_repair_verification": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_system_time_sync_repair_verification_artifact",
            review_dir=review_dir,
            suffix="system_time_sync_repair_verification_report",
            reference_now=reference_now,
        ),
        "openclaw_orderflow_blueprint": resolve_source_path(
            source_payload=hot_brief,
            artifact_key="source_openclaw_orderflow_blueprint_artifact",
            review_dir=review_dir,
            suffix="openclaw_orderflow_blueprint",
            reference_now=reference_now,
        ),
    }
    source_payloads = {key: load_optional_payload(path) for key, path in source_paths.items()}

    promotion_unblock_brief = safe_text(
        hot_brief.get("source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief")
    )
    promotion_unblock_status = safe_text(
        hot_brief.get("source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_status")
    )
    promotion_unblock_scope = safe_text(
        hot_brief.get("source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_blocker_scope")
    )
    promotion_unblock_title = safe_text(
        hot_brief.get("source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_title")
    )
    promotion_unblock_target_artifact = safe_text(
        hot_brief.get("source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_target_artifact")
    )
    promotion_unblock_local_repair_required = bool(
        hot_brief.get("source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_required")
    )
    priority_repair_uses_promotion_unblock = bool(
        promotion_unblock_brief
        and promotion_unblock_title
        and promotion_unblock_target_artifact
        and promotion_unblock_target_artifact != "system_time_sync_repair_verification_report"
        and not promotion_unblock_local_repair_required
        and promotion_unblock_scope
        and promotion_unblock_scope != "time_sync"
    )
    priority_repair_plan_payload: dict[str, Any]
    if priority_repair_uses_promotion_unblock:
        priority_repair_plan_payload = {
            "status": promotion_unblock_status or "blocked",
            "done_when": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_done_when"
                )
            ),
        }
        priority_repair_plan_brief = promotion_unblock_brief
        priority_repair_plan_artifact = promotion_unblock_target_artifact
        priority_repair_plan_admin_required = False
    else:
        priority_repair_plan_payload = source_payloads.get("time_sync_repair_plan") or {}
        priority_repair_plan_brief = safe_text(hot_brief.get("source_system_time_sync_repair_plan_brief"))
        priority_repair_plan_artifact = safe_text(hot_brief.get("source_system_time_sync_repair_plan_artifact"))
        priority_repair_plan_admin_required = bool(
            hot_brief.get("source_system_time_sync_repair_plan_admin_required")
        )

    operator_head_lane = safe_row(cross_market.get("operator_head_lane"))
    review_head_lane = safe_row(cross_market.get("review_head_lane"))
    repair_head_lane = safe_row(cross_market.get("operator_repair_head_lane"))
    operator_action_queue = list(cross_market.get("operator_action_queue") or hot_brief.get("operator_action_queue") or [])
    operator_action_checklist = list(cross_market.get("operator_action_checklist") or hot_brief.get("operator_action_checklist") or [])
    operator_focus_slots = list(cross_market.get("operator_focus_slots") or hot_brief.get("operator_focus_slots") or [])
    operator_repair_queue = list(cross_market.get("operator_repair_queue") or hot_brief.get("operator_repair_queue") or [])
    review_backlog = list(cross_market.get("review_backlog") or [])
    operator_backlog = list(cross_market.get("operator_backlog") or [])

    payload: dict[str, Any] = {
        "action": "build_operator_task_visual_panel",
        "status": "ok",
        "ok": True,
        "generated_at_utc": fmt_utc(reference_now),
        "review_dir": str(review_dir),
        "dashboard_dist": str(dashboard_dist),
        "source_artifacts": {
            "hot_brief": str(hot_brief_path),
            "cross_market": str(cross_market_path),
            **{key: str(path or "") for key, path in source_paths.items()},
        },
        "summary": {
            "operator_head": safe_row(operator_head_lane.get("head")),
            "review_head": safe_row(review_head_lane.get("head")),
            "repair_head": safe_row(repair_head_lane.get("head")),
            "operator_head_brief": safe_text(hot_brief.get("cross_market_operator_head_brief")),
            "review_head_brief": safe_text(hot_brief.get("cross_market_review_head_brief")),
            "repair_head_brief": safe_text(hot_brief.get("cross_market_operator_repair_head_brief")),
            "remote_live_gate_brief": safe_text(hot_brief.get("cross_market_remote_live_takeover_gate_brief")),
            "remote_live_clearing_brief": safe_text(hot_brief.get("cross_market_remote_live_takeover_clearing_brief")),
            "lane_state_brief": safe_text(hot_brief.get("cross_market_operator_backlog_state_brief")),
            "lane_priority_order_brief": safe_text(hot_brief.get("cross_market_operator_lane_priority_order_brief")),
            "action_queue_brief": safe_text(hot_brief.get("operator_action_queue_brief")),
            "crypto_refresh_reuse_brief": safe_text(hot_brief.get("source_crypto_route_refresh_reuse_gate_brief")),
            "remote_live_history_brief": safe_text(hot_brief.get("source_remote_live_history_audit_window_brief")),
            "remote_live_diagnosis_brief": safe_text(hot_brief.get("source_live_gate_blocker_remote_live_diagnosis_brief")),
            "brooks_refresh_brief": safe_text(hot_brief.get("source_brooks_structure_refresh_brief")),
            "brooks_selected_routes_brief": safe_text(hot_brief.get("source_brooks_route_report_selected_routes_brief")),
            "priority_repair_plan_brief": priority_repair_plan_brief,
            "priority_repair_plan_artifact": priority_repair_plan_artifact,
            "priority_repair_plan_admin_required": priority_repair_plan_admin_required,
            "priority_repair_verification_brief": safe_text(
                hot_brief.get("source_system_time_sync_repair_verification_brief")
            ),
            "priority_repair_verification_artifact": safe_text(
                hot_brief.get("source_system_time_sync_repair_verification_artifact")
            ),
            "priority_repair_verification_cleared": bool(
                hot_brief.get("source_system_time_sync_repair_verification_cleared")
            ),
            "openclaw_blueprint_brief": safe_text(hot_brief.get("source_openclaw_orderflow_blueprint_brief")),
            "openclaw_blueprint_artifact": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_artifact")
            ),
            "openclaw_current_life_stage": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_current_life_stage")
            ),
            "openclaw_target_life_stage": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_target_life_stage")
            ),
            "openclaw_intent_queue_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_intent_queue_brief")
            ),
            "openclaw_intent_queue_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_intent_queue_status")
            ),
            "openclaw_intent_queue_recommendation": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_intent_queue_recommendation")
            ),
            "openclaw_execution_journal_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_journal_brief")
            ),
            "openclaw_execution_journal_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_journal_status")
            ),
            "openclaw_execution_journal_append_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_journal_append_status")
            ),
            "openclaw_orderflow_feedback_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief")
            ),
            "openclaw_orderflow_feedback_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_feedback_status")
            ),
            "openclaw_orderflow_feedback_recommendation": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_feedback_recommendation")
            ),
            "openclaw_orderflow_policy_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief")
            ),
            "openclaw_orderflow_policy_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_policy_status")
            ),
            "openclaw_orderflow_policy_decision": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_policy_decision")
            ),
            "openclaw_execution_ack_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_ack_brief")
            ),
            "openclaw_execution_ack_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_ack_status")
            ),
            "openclaw_execution_ack_decision": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_ack_decision")
            ),
            "openclaw_canary_gate_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_brief")
            ),
            "openclaw_canary_gate_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_status")
            ),
            "openclaw_canary_gate_decision": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_execution_actor_canary_gate_decision")
            ),
            "openclaw_quality_report_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_brief")
            ),
            "openclaw_quality_report_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_status")
            ),
            "openclaw_quality_report_recommendation": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_recommendation")
            ),
            "openclaw_quality_report_score": safe_int(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_orderflow_quality_report_score")
            ),
            "openclaw_quality_shadow_learning_score": safe_int(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_orderflow_quality_shadow_learning_score"
                )
            ),
            "openclaw_quality_execution_readiness_score": safe_int(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_orderflow_quality_execution_readiness_score"
                )
            ),
            "openclaw_quality_transport_observability_score": safe_int(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_orderflow_quality_transport_observability_score"
                )
            ),
            "openclaw_live_boundary_hold_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_live_boundary_hold_brief")
            ),
            "openclaw_live_boundary_hold_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_live_boundary_hold_status")
            ),
            "openclaw_live_boundary_hold_decision": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_live_boundary_hold_decision")
            ),
            "openclaw_live_boundary_hold_next_transition": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_live_boundary_hold_next_transition")
            ),
            "openclaw_promotion_gate_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_brief"
                )
            ),
            "openclaw_promotion_gate_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_status"
                )
            ),
            "openclaw_promotion_gate_decision": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_decision"
                )
            ),
            "openclaw_promotion_gate_blocker_title": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_title"
                )
            ),
            "openclaw_promotion_gate_blocker_code": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code"
                )
            ),
            "openclaw_promotion_gate_blocker_target_artifact": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_target_artifact"
                )
            ),
            "openclaw_shadow_learning_continuity_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief"
                )
            ),
            "openclaw_shadow_learning_continuity_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_status"
                )
            ),
            "openclaw_shadow_learning_continuity_decision": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_decision"
                )
            ),
            "openclaw_shadow_learning_continuity_blocker_detail": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_blocker_detail"
                )
            ),
            "openclaw_promotion_unblock_readiness_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief"
                )
            ),
            "openclaw_promotion_unblock_readiness_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_status"
                )
            ),
            "openclaw_promotion_unblock_readiness_decision": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_decision"
                )
            ),
            "openclaw_promotion_unblock_primary_blocker_scope": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_blocker_scope"
                )
            ),
            "openclaw_promotion_unblock_primary_local_repair_title": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_title"
                )
            ),
            "openclaw_promotion_unblock_primary_local_repair_target_artifact": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_target_artifact"
                )
            ),
            "openclaw_promotion_unblock_primary_local_repair_plan_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_plan_brief"
                )
            ),
            "openclaw_promotion_unblock_primary_local_repair_environment_classification": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_environment_classification"
                )
            ),
            "openclaw_promotion_unblock_primary_local_repair_environment_blocker_detail": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_environment_blocker_detail"
                )
            ),
            "openclaw_ticket_actionability_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_ticket_actionability_brief"
                )
            ),
            "openclaw_ticket_actionability_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_ticket_actionability_status"
                )
            ),
            "openclaw_ticket_actionability_decision": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_ticket_actionability_decision"
                )
            ),
            "openclaw_ticket_actionability_next_action": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_ticket_actionability_next_action"
                )
            ),
            "openclaw_ticket_actionability_next_action_target_artifact": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_ticket_actionability_next_action_target_artifact"
                )
            ),
            "openclaw_shortline_backtest_slice_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_brief"
                )
            ),
            "openclaw_shortline_backtest_slice_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_status"
                )
            ),
            "openclaw_shortline_backtest_slice_decision": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_decision"
                )
            ),
            "openclaw_shortline_backtest_slice_selected_symbol": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_selected_symbol"
                )
            ),
            "openclaw_shortline_backtest_slice_universe_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_backtest_slice_universe_brief"
                )
            ),
            "openclaw_shortline_cross_section_backtest_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_brief"
                )
            ),
            "openclaw_shortline_cross_section_backtest_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_status"
                )
            ),
            "openclaw_shortline_cross_section_backtest_decision": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_decision"
                )
            ),
            "openclaw_shortline_cross_section_backtest_selected_edge_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_crypto_shortline_cross_section_backtest_selected_edge_status"
                )
            ),
            "openclaw_remote_time_sync_mode": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_time_sync_mode")
            ),
            "openclaw_remote_shadow_clock_evidence_brief": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_shadow_clock_evidence_brief")
            ),
            "openclaw_remote_shadow_clock_evidence_status": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_remote_shadow_clock_evidence_status")
            ),
            "openclaw_remote_shadow_clock_shadow_learning_allowed": bool(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_shadow_clock_shadow_learning_allowed"
                )
            ),
            "openclaw_guardian_clearance_brief": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_brief"
                )
            ),
            "openclaw_guardian_clearance_status": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_status"
                )
            ),
            "openclaw_guardian_clearance_score": safe_int(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_score"
                )
            ),
            "openclaw_guardian_clearance_top_blocker_code": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code"
                )
            ),
            "openclaw_guardian_clearance_top_blocker_title": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_title"
                )
            ),
            "openclaw_guardian_clearance_top_blocker_target_artifact": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_target_artifact"
                )
            ),
            "openclaw_guardian_clearance_top_blocker_next_action": safe_text(
                hot_brief.get(
                    "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_next_action"
                )
            ),
            "openclaw_top_backlog_title": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_top_backlog_title")
            ),
            "openclaw_top_backlog_target_artifact": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_top_backlog_target_artifact")
            ),
            "openclaw_top_backlog_why": safe_text(
                hot_brief.get("source_openclaw_orderflow_blueprint_top_backlog_why")
            ),
        },
        "lane_cards": build_lane_cards(cross_market),
        "focus_slots": operator_focus_slots,
        "action_queue": operator_action_queue,
        "action_checklist": operator_action_checklist,
        "repair_queue": operator_repair_queue,
        "review_backlog": review_backlog,
        "operator_backlog": operator_backlog,
        "chain": build_chain_data(hot_brief=hot_brief, cross_market=cross_market),
        "impact_rows": build_impact_rows(hot_brief=hot_brief, cross_market=cross_market),
        "source_rows": build_source_rows(hot_brief=hot_brief, source_payloads=source_payloads),
        "head_lanes": {
            "operator": operator_head_lane,
            "review": review_head_lane,
            "repair": repair_head_lane,
        },
        "priority_repair_plan": priority_repair_plan_payload,
        "priority_repair_verification": source_payloads.get("time_sync_repair_verification") or {},
        "openclaw_orderflow_blueprint": source_payloads.get("openclaw_orderflow_blueprint") or {},
        "control_chain": safe_list(
            (source_payloads.get("openclaw_orderflow_blueprint") or {}).get("control_chain")
        ),
        "continuous_optimization_backlog": safe_list(
            (source_payloads.get("openclaw_orderflow_blueprint") or {}).get(
                "continuous_optimization_backlog"
            )
        ),
    }
    return payload


def pill_class(state: str) -> str:
    mapping = {
        "waiting": "state-waiting",
        "review": "state-review",
        "watch": "state-watch",
        "blocked": "state-blocked",
        "repair": "state-repair",
        "ready": "state-ready",
    }
    return mapping.get(str(state or "").strip().lower(), "state-neutral")


def control_chain_state(row: dict[str, Any]) -> str:
    status = safe_text(row.get("source_status")).lower()
    decision = safe_text(row.get("source_decision")).lower()
    combined = f"{status} {decision}"
    if any(token in combined for token in ("blocked", "degraded", "repair", "veto", "missing")):
        return "repair"
    if any(token in combined for token in ("ready", "stable", "ok", "cleared")):
        return "ready"
    if any(token in combined for token in ("wait", "review", "monitor", "queued", "shadow")):
        return "review"
    return "neutral"


def card_html(title: str, brief: str, meta: list[str], state: str = "neutral") -> str:
    meta_html = "".join([f"<li>{html.escape(item)}</li>" for item in meta if item])
    return (
        f'<section class="panel-card {pill_class(state)}">'
        f"<div class=\"card-title\">{html.escape(title)}</div>"
        f"<div class=\"card-brief\">{html.escape(brief or '-') }</div>"
        f"<ul class=\"card-meta\">{meta_html}</ul>"
        f"</section>"
    )


def table_rows(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "".join([f"<th>{html.escape(label)}</th>" for _, label in columns])
    body_parts: list[str] = []
    for row in rows:
        body_parts.append(
            "<tr>"
            + "".join(
                [
                    "<td>"
                    + html.escape(str(row.get(key, "") if row.get(key, "") is not None else ""))
                    + "</td>"
                    for key, _ in columns
                ]
            )
            + "</tr>"
        )
    return (
        "<table class=\"grid-table\"><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body_parts)
        + "</tbody></table>"
    )


def render_chain_svg(chain: dict[str, Any]) -> str:
    node_lookup = {str(node.get("id")): node for node in chain.get("nodes", [])}
    svg_parts = [
        '<svg class="chain-svg" viewBox="0 0 1280 560" role="img" aria-label="Operator chain transmission graph">'
    ]
    for edge in chain.get("edges", []):
        start = node_lookup.get(str(edge.get("from")))
        end = node_lookup.get(str(edge.get("to")))
        if not start or not end:
            continue
        x1 = int(start.get("x") or 0) + 120
        y1 = int(start.get("y") or 0) + 34
        x2 = int(end.get("x") or 0)
        y2 = int(end.get("y") or 0) + 34
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        svg_parts.append(
            f'<path d="M{x1},{y1} C{mid_x},{y1} {mid_x},{y2} {x2},{y2}" class="chain-edge" />'
        )
        svg_parts.append(
            f'<text x="{mid_x}" y="{mid_y - 6}" class="chain-edge-label">{html.escape(str(edge.get("label") or ""))}</text>'
        )
    for node in chain.get("nodes", []):
        x = int(node.get("x") or 0)
        y = int(node.get("y") or 0)
        kind = safe_text(node.get("kind")) or "source"
        label = safe_text(node.get("label")) or "-"
        brief = safe_text(node.get("brief")) or "-"
        svg_parts.append(f'<g transform="translate({x},{y})" class="node-group node-{html.escape(kind)}">')
        svg_parts.append('<rect width="220" height="72" rx="18" class="chain-node" />')
        svg_parts.append(f'<text x="16" y="24" class="chain-node-title">{html.escape(label)}</text>')
        svg_parts.append(f'<text x="16" y="46" class="chain-node-brief">{html.escape(brief[:56])}</text>')
        svg_parts.append("</g>")
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def render_html(payload: dict[str, Any]) -> str:
    summary = dict(payload.get("summary") or {})
    operator_head = dict(summary.get("operator_head") or {})
    review_head = dict(summary.get("review_head") or {})
    repair_head = dict(summary.get("repair_head") or {})
    lane_cards = list(payload.get("lane_cards") or [])
    focus_slots = list(payload.get("focus_slots") or [])
    action_checklist = list(payload.get("action_checklist") or [])
    repair_queue = list(payload.get("repair_queue") or [])
    impact_rows = list(payload.get("impact_rows") or [])
    source_rows = list(payload.get("source_rows") or [])
    chain = dict(payload.get("chain") or {})
    control_chain = list(payload.get("control_chain") or [])
    continuous_optimization_backlog = list(payload.get("continuous_optimization_backlog") or [])
    priority_repair_plan = dict(payload.get("priority_repair_plan") or {})
    priority_repair_verification = dict(payload.get("priority_repair_verification") or {})
    priority_repair_plan_html = card_html(
        title="Time Sync Repair Plan",
        brief=safe_text(summary.get("priority_repair_plan_brief")) or "-",
        meta=[
            f"admin_required={bool(summary.get('priority_repair_plan_admin_required'))}",
            f"artifact={path_brief(safe_text(summary.get('priority_repair_plan_artifact')))}",
            f"done_when={safe_text(priority_repair_plan.get('done_when')) or '-'}",
        ],
        state="repair" if safe_text(summary.get("priority_repair_plan_brief")) else "neutral",
    )
    priority_repair_verification_html = card_html(
        title="Time Sync Repair Verification",
        brief=safe_text(summary.get("priority_repair_verification_brief")) or "-",
        meta=[
            f"cleared={bool(summary.get('priority_repair_verification_cleared'))}",
            f"artifact={path_brief(safe_text(summary.get('priority_repair_verification_artifact')))}",
            f"status={safe_text(priority_repair_verification.get('status')) or '-'}",
        ],
        state="ready"
        if bool(summary.get("priority_repair_verification_cleared"))
        else ("repair" if safe_text(summary.get("priority_repair_verification_brief")) else "neutral"),
    )
    openclaw_blueprint_html = card_html(
        title="OpenClaw Evolution",
        brief=safe_text(summary.get("openclaw_blueprint_brief")) or "-",
        meta=[
            f"life_stage={safe_text(summary.get('openclaw_current_life_stage')) or '-'} -> {safe_text(summary.get('openclaw_target_life_stage')) or '-'}",
            f"intent_queue={safe_text(summary.get('openclaw_intent_queue_brief')) or '-'}",
            f"execution_ack={safe_text(summary.get('openclaw_execution_ack_brief')) or '-'}",
            f"canary_gate={safe_text(summary.get('openclaw_canary_gate_brief')) or '-'}",
            f"quality={safe_text(summary.get('openclaw_quality_report_brief')) or '-'}",
            f"recommendation={safe_text(summary.get('openclaw_intent_queue_recommendation')) or '-'}",
            f"top_backlog={safe_text(summary.get('openclaw_top_backlog_target_artifact')) or '-'}:{safe_text(summary.get('openclaw_top_backlog_title')) or '-'}",
            f"artifact={path_brief(safe_text(summary.get('openclaw_blueprint_artifact')))}",
        ],
        state="review" if safe_text(summary.get("openclaw_blueprint_brief")) else "neutral",
    )

    lane_cards_html = "".join(
        [
            card_html(
                title=str(card.get("state") or "-").upper(),
                brief=f"count={safe_int(card.get('count'))} | total={safe_int(card.get('priority_total'))}",
                meta=[
                    f"head={safe_text(card.get('head_symbol')) or '-'}:{safe_text(card.get('head_action')) or '-'}",
                    safe_text(card.get("brief")) or "-",
                ],
                state=safe_text(card.get("state")) or "neutral",
            )
            for card in lane_cards
        ]
    )
    focus_slots_html = "".join(
        [
            card_html(
                title=f"{safe_text(slot.get('slot')).upper()} | {safe_text(slot.get('symbol')) or '-'}",
                brief=f"{safe_text(slot.get('action')) or '-'} | {safe_text(slot.get('priority_tier')) or '-'}:{safe_int(slot.get('priority_score'))}",
                meta=[
                    f"reason={safe_text(slot.get('reason')) or '-'}",
                    f"done_when={safe_text(slot.get('done_when')) or '-'}",
                    f"source={path_brief(safe_text(slot.get('source_artifact')))}",
                ],
                state=safe_text(slot.get("state")) or "neutral",
            )
            for slot in focus_slots
        ]
    )
    control_chain_html = "".join(
        [
            card_html(
                title=f"{safe_text(row.get('label')) or '-'} | {safe_text(row.get('stage')) or '-'}",
                brief=safe_text(row.get("source_brief")) or "-",
                meta=[
                    f"status={safe_text(row.get('source_status')) or '-'}",
                    f"decision={safe_text(row.get('source_decision')) or '-'}",
                    f"target={safe_text(row.get('next_target_artifact')) or '-'}",
                    f"artifact={path_brief(safe_text(row.get('source_artifact')))}",
                    f"fields={', '.join([safe_text(item) for item in list(row.get('interface_fields') or []) if safe_text(item)]) or '-'}",
                ],
                state=control_chain_state(row),
            )
            for row in control_chain
        ]
    )
    continuous_optimization_rows = [
        {
            **dict(row),
            "interface_fields_brief": ", ".join(
                [safe_text(item) for item in list(row.get("interface_fields") or []) if safe_text(item)]
            )
            or "-",
        }
        for row in continuous_optimization_backlog
    ]

    action_table = table_rows(
        action_checklist,
        [
            ("rank", "#"),
            ("state", "State"),
            ("symbol", "Symbol"),
            ("action", "Action"),
            ("priority_score", "Priority"),
            ("reason", "Reason"),
            ("done_when", "Done When"),
        ],
    )
    repair_table = table_rows(
        repair_queue,
        [
            ("rank", "#"),
            ("area", "Area"),
            ("symbol", "Symbol"),
            ("action", "Action"),
            ("priority_score", "Priority"),
            ("command", "Command"),
            ("clear_when", "Clear When"),
        ],
    )
    impact_table = table_rows(
        impact_rows,
        [
            ("lane", "Lane"),
            ("symbol", "Symbol"),
            ("action", "Action"),
            ("impact_scope", "Impact Range"),
            ("blocker_detail", "Blocker"),
            ("done_when", "Done When"),
        ],
    )
    source_table = table_rows(
        source_rows,
        [
            ("label", "Source"),
            ("status", "Status"),
            ("as_of", "As Of"),
            ("brief", "Brief"),
            ("artifact_brief", "Artifact"),
        ],
    )
    continuous_optimization_table = table_rows(
        continuous_optimization_rows,
        [
            ("priority", "Priority"),
            ("label", "Layer"),
            ("title", "Title"),
            ("target_artifact", "Target Artifact"),
            ("source_status", "Current Status"),
            ("interface_fields_brief", "Interface Fields"),
        ],
    )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Fenlie Operator Task Visual Panel</title>
    <style>
      :root {{
        --bg: #0f1722;
        --bg-soft: #152133;
        --panel: #1a2740;
        --panel-alt: #21314f;
        --line: rgba(255,255,255,0.09);
        --text: #edf2ff;
        --muted: #9fb0cb;
        --accent: #7ee0c6;
        --waiting: #f7c66b;
        --review: #74b8ff;
        --watch: #b6b9c3;
        --blocked: #ff7b72;
        --repair: #ff9a55;
        --ready: #7ee0c6;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background:
          radial-gradient(circle at top left, rgba(126,224,198,0.12), transparent 28%),
          radial-gradient(circle at top right, rgba(116,184,255,0.10), transparent 24%),
          linear-gradient(180deg, #0c1320 0%, var(--bg) 100%);
        color: var(--text);
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      }}
      .page {{
        max-width: 1480px;
        margin: 0 auto;
        padding: 28px 24px 64px;
      }}
      .hero {{
        display: grid;
        grid-template-columns: 1.6fr 1fr;
        gap: 18px;
        margin-bottom: 18px;
      }}
      .hero-main, .hero-side, .section {{
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.22);
      }}
      .eyebrow {{
        color: var(--accent);
        font-size: 12px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 700;
      }}
      h1 {{
        margin: 10px 0 12px;
        font-size: 34px;
        line-height: 1.05;
      }}
      .hero-copy {{
        color: var(--muted);
        line-height: 1.6;
        max-width: 920px;
      }}
      .signal-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
      }}
      .chip {{
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255,255,255,0.04);
        color: var(--text);
        font-size: 12px;
        line-height: 1.2;
      }}
      .hero-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }}
      .hero-stat {{
        padding: 14px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
      }}
      .hero-stat-label {{
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }}
      .hero-stat-value {{
        margin-top: 8px;
        font-size: 18px;
        font-weight: 700;
      }}
      .section {{
        margin-top: 18px;
      }}
      .section h2 {{
        margin: 0 0 14px;
        font-size: 18px;
      }}
      .section-lead {{
        color: var(--muted);
        margin: 0 0 16px;
        line-height: 1.55;
      }}
      .card-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
      }}
      .panel-card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
      }}
      .card-title {{
        font-size: 12px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
      }}
      .card-brief {{
        margin-top: 10px;
        font-size: 16px;
        line-height: 1.4;
        font-weight: 700;
      }}
      .card-meta {{
        list-style: none;
        padding: 0;
        margin: 12px 0 0;
        display: grid;
        gap: 8px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.4;
      }}
      .state-waiting {{ box-shadow: inset 0 0 0 1px rgba(247,198,107,0.28); }}
      .state-review {{ box-shadow: inset 0 0 0 1px rgba(116,184,255,0.28); }}
      .state-watch {{ box-shadow: inset 0 0 0 1px rgba(182,185,195,0.22); }}
      .state-blocked {{ box-shadow: inset 0 0 0 1px rgba(255,123,114,0.34); }}
      .state-repair {{ box-shadow: inset 0 0 0 1px rgba(255,154,85,0.34); }}
      .state-ready {{ box-shadow: inset 0 0 0 1px rgba(126,224,198,0.34); }}
      .state-neutral {{ box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08); }}
      .grid-table {{
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 16px;
        border: 1px solid var(--line);
      }}
      .grid-table th, .grid-table td {{
        padding: 12px 10px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
        font-size: 13px;
      }}
      .grid-table th {{
        background: var(--panel-alt);
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 11px;
      }}
      .grid-table tbody tr:nth-child(odd) {{
        background: rgba(255,255,255,0.01);
      }}
      .split {{
        display: grid;
        grid-template-columns: 1.1fr 0.9fr;
        gap: 18px;
      }}
      .chain-shell {{
        overflow-x: auto;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        padding: 10px;
      }}
      .chain-svg {{
        width: 100%;
        min-width: 1200px;
        height: auto;
      }}
      .chain-edge {{
        fill: none;
        stroke: rgba(255,255,255,0.22);
        stroke-width: 2;
      }}
      .chain-edge-label {{
        fill: var(--muted);
        font-size: 11px;
        text-anchor: middle;
      }}
      .chain-node {{
        fill: #18243a;
        stroke: rgba(255,255,255,0.10);
        stroke-width: 1.2;
      }}
      .node-source .chain-node {{ fill: #17273b; }}
      .node-aggregate .chain-node {{ fill: #1e3150; }}
      .node-head .chain-node {{ fill: #2b2444; }}
      .node-gate .chain-node {{ fill: #3b2a21; }}
      .chain-node-title {{
        fill: var(--text);
        font-size: 13px;
        font-weight: 700;
      }}
      .chain-node-brief {{
        fill: var(--muted);
        font-size: 11px;
      }}
      .footer-note {{
        margin-top: 18px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
      }}
      code {{
        font-family: "JetBrains Mono", "SFMono-Regular", monospace;
        font-size: 12px;
        background: rgba(255,255,255,0.05);
        padding: 2px 6px;
        border-radius: 8px;
      }}
      @media (max-width: 1100px) {{
        .hero, .split {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <section class="hero">
        <div class="hero-main">
          <div class="eyebrow">Fenlie Operator Window</div>
          <h1>Task Visual Management Panel</h1>
          <p class="hero-copy">
            这块面板直接读当前交易系统的 source artifacts，重点展示三件事：
            现在谁是全局 operator head，链路是怎样从 source 传到 action queue 的，以及当前动作会影响哪些 lane、gate 和下游判断。
          </p>
          <div class="signal-strip">
            <div class="chip">generated={html.escape(str(payload.get("generated_at_utc") or "-"))}</div>
            <div class="chip">lane_order={html.escape(str(summary.get("lane_priority_order_brief") or "-"))}</div>
            <div class="chip">crypto_refresh={html.escape(str(summary.get("crypto_refresh_reuse_brief") or "-"))}</div>
            <div class="chip">remote_live={html.escape(str(summary.get("remote_live_diagnosis_brief") or "-"))}</div>
            <div class="chip">brooks={html.escape(str(summary.get("brooks_refresh_brief") or "-"))}</div>
          </div>
        </div>
        <aside class="hero-side">
          <div class="hero-grid">
            <div class="hero-stat">
              <div class="hero-stat-label">Operator Head</div>
              <div class="hero-stat-value">{html.escape(str(summary.get("operator_head_brief") or "-"))}</div>
            </div>
            <div class="hero-stat">
              <div class="hero-stat-label">Review Head</div>
              <div class="hero-stat-value">{html.escape(str(summary.get("review_head_brief") or "-"))}</div>
            </div>
            <div class="hero-stat">
              <div class="hero-stat-label">Repair Head</div>
              <div class="hero-stat-value">{html.escape(str(summary.get("repair_head_brief") or "-"))}</div>
            </div>
            <div class="hero-stat">
              <div class="hero-stat-label">Remote Gate</div>
              <div class="hero-stat-value">{html.escape(str(summary.get("remote_live_gate_brief") or "-"))}</div>
            </div>
          </div>
        </aside>
      </section>

      <section class="section">
        <h2>Lane Pressure Board</h2>
        <p class="section-lead">
          这里不是虚构进度条，而是系统当前真实的任务负载分层。你能直接看到各 lane 的数量、总优先级和当前 head。
        </p>
        <div class="card-grid">{lane_cards_html}</div>
      </section>

      <section class="section">
        <h2>Focus Slots</h2>
        <p class="section-lead">
          顶层 `primary / followup / secondary` 槽位已经由 cross-market source 驱动。当前第三槽位是 Brooks 结构头，不再是旧的 crypto fallback。
        </p>
        <div class="card-grid">{focus_slots_html}</div>
      </section>

      <section class="section">
        <h2>OpenClaw Refactor Blueprint</h2>
        <p class="section-lead">
          这里显示云端 OpenClaw 当前生命阶段、目标阶段，以及最先要补的执行器官，避免云端执行重构继续停在孤立文档里。
        </p>
        <div class="card-grid">{openclaw_blueprint_html}</div>
      </section>

      <section class="section">
        <h2>Transmission Control Chain</h2>
        <p class="section-lead">
          这组卡片直接来自 blueprint 的 source-owned control chain，按研究、信号、风控、执行、对账、复盘六层展示当前状态、下一目标和前端应消费的接口字段。
        </p>
        <div class="card-grid">{control_chain_html}</div>
      </section>

      <section class="section">
        <h2>Continuous Optimization Backlog</h2>
        <p class="section-lead">
          持续优化不是口号，这里把传导控制链的长期优化任务落成待办节点，便于后续继续迭代而不丢接口契约。
        </p>
        {continuous_optimization_table}
      </section>

      <section class="section">
        <h2>Priority Repair Plan</h2>
        <p class="section-lead">
          当前最高优先级环境修复已经有独立计划件。这里直接显示 plan brief、是否需要管理员权限，以及对应 artifact。
        </p>
        <div class="card-grid">{priority_repair_plan_html}{priority_repair_verification_html}</div>
      </section>

      <section class="section split">
        <div>
          <h2>Action Checklist</h2>
          <p class="section-lead">这就是当前系统真正按顺序要看的动作队列。</p>
          {action_table}
        </div>
        <div>
          <h2>Repair Queue</h2>
          <p class="section-lead">远端 live 接管相关的修复项已经独立成 repair lane，并且能看到具体 command / clear_when。</p>
          {repair_table}
        </div>
      </section>

      <section class="section">
        <h2>Chain Transmission</h2>
        <p class="section-lead">
          这张图不是理论架构图，而是当前 source 到 head 的真实传导。左边是 source layers，中间是 cross-market 聚合和 hot brief，右边是最终 heads 和 remote takeover gate。
        </p>
        <div class="chain-shell">{render_chain_svg(chain)}</div>
      </section>

      <section class="section split">
        <div>
          <h2>Impact Range</h2>
          <p class="section-lead">
            当前 head 不只是一个 symbol/action，它会继续影响 focus slots、action queue 排序、remote live clearing 和 review lane。
          </p>
          {impact_table}
        </div>
        <div>
          <h2>Source Freshness</h2>
          <p class="section-lead">
            所有可视化都直接指向 source artifact。你可以用这一栏快速判断当前页面是不是在读错版本、读旧版本，或者 source 本身是不是 stale。
          </p>
          {source_table}
        </div>
      </section>

      <p class="footer-note">
        Stable dashboard files are written to <code>{html.escape(str(payload.get("dashboard_dist") or "-"))}</code>.
        Source audit files are stamped into <code>{html.escape(str(payload.get("review_dir") or "-"))}</code>.
      </p>
    </div>
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a source-driven operator task visual panel.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--dashboard-dist", type=Path, default=DEFAULT_DASHBOARD_DIST)
    parser.add_argument("--now", default="", help="Explicit UTC reference time in ISO8601 form.")
    parser.add_argument("--artifact-keep", type=int, default=20)
    parser.add_argument("--artifact-ttl-hours", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_now = parse_now(args.now)
    review_dir = args.review_dir.expanduser().resolve()
    dashboard_dist = args.dashboard_dist.expanduser().resolve()
    dashboard_dist.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    payload = build_panel_payload(
        review_dir=review_dir,
        dashboard_dist=dashboard_dist,
        reference_now=reference_now,
    )

    stamp = reference_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_operator_task_visual_panel.json"
    html_path = review_dir / f"{stamp}_operator_task_visual_panel.html"
    checksum_path = review_dir / f"{stamp}_operator_task_visual_panel_checksum.json"

    html_text = render_html(payload)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact_path),
        "html": str(html_path),
        "sha256_artifact": sha256_file(artifact_path),
        "sha256_html": sha256_file(html_path),
        "generated_at_utc": payload.get("generated_at_utc"),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stable_html = dashboard_dist / "operator_task_visual_panel.html"
    stable_json = dashboard_dist / "operator_task_visual_panel_data.json"
    stable_html.write_text(html_text, encoding="utf-8")
    stable_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload["artifact"] = str(artifact_path)
    payload["html"] = str(html_path)
    payload["dashboard_html"] = str(stable_html)
    payload["dashboard_json"] = str(stable_json)

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="operator_task_visual_panel",
        current_paths=[artifact_path, html_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=reference_now,
    )
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
