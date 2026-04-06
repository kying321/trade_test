#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

EVENT_CRISIS_REQUIRED_SUMMARY_KEYS = (
    "event_crisis_primary_theater_brief",
    "event_crisis_dominant_chain_brief",
    "event_crisis_safety_margin_brief",
    "event_crisis_hard_boundary_brief",
)


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


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot resolve system root from {workspace}")


def run_json(*, name: str, cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"returncode={proc.returncode}"
        raise RuntimeError(f"{name}_failed: {detail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name}_invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name}_invalid_payload")
    return payload


def event_summary_has_geostrategy_fields(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return all(safe_text(payload.get(key)) for key in EVENT_CRISIS_REQUIRED_SUMMARY_KEYS)


def ensure_event_crisis_summary_ready(
    *,
    system_root: Path,
    review_dir: Path,
    runtime_now: dt.datetime,
) -> dict[str, Any] | None:
    summary_path = review_dir / "latest_event_crisis_operator_summary.json"
    if event_summary_has_geostrategy_fields(summary_path):
        return None
    return run_json(
        name="run_event_crisis_pipeline",
        cmd=[
            "python3",
            str(system_root / "scripts" / "run_event_crisis_pipeline.py"),
            "--mode",
            "snapshot",
            "--output-root",
            str(system_root / "output"),
            "--now",
            fmt_utc(runtime_now),
        ],
    )


def sync_copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return {"src": str(src), "dst": str(dst), "copied": False, "reason": "source_missing"}
    shutil.copy2(src, dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "copied": True,
        "size_bytes": dst.stat().st_size,
    }


def degraded_external_refresh_payload(exc: Exception) -> dict[str, Any]:
    detail = safe_text(exc)
    takeaway = detail.split(":", 1)[1].strip() if ":" in detail else detail
    return {
        "status": "degraded_request_failed",
        "artifact_json": "",
        "external_intelligence_path": "",
        "recommended_brief": "",
        "takeaway": takeaway,
        "error": detail,
    }


def build_summary(
    *,
    workspace: Path,
    public_dir: Path,
    dist_dir: Path,
    external_refresh_payload: dict[str, Any],
    cpa_control_payload: dict[str, Any],
    panel_payload: dict[str, Any],
    snapshot_payload: dict[str, Any],
    feedback_payload: dict[str, Any],
    sync_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = dict(panel_payload.get("summary") or {})
    return {
        "ok": True,
        "workspace": str(workspace),
        "mode": "operator_panel_refresh",
        "panel_review_json": safe_text(panel_payload.get("artifact")),
        "panel_review_html": safe_text(panel_payload.get("html") or panel_payload.get("html_artifact")),
        "panel_public_html": str(public_dir / "operator_task_visual_panel.html"),
        "panel_public_json": str(public_dir / "operator_task_visual_panel_data.json"),
        "panel_dist_html": str(dist_dir / "operator_task_visual_panel.html"),
        "panel_dist_json": str(dist_dir / "operator_task_visual_panel_data.json"),
        "snapshot_public": str(public_dir / "data" / "fenlie_dashboard_snapshot.json"),
        "snapshot_internal_public": str(public_dir / "data" / "fenlie_dashboard_internal_snapshot.json"),
        "snapshot_dist": str(dist_dir / "data" / "fenlie_dashboard_snapshot.json"),
        "snapshot_internal_dist": str(dist_dir / "data" / "fenlie_dashboard_internal_snapshot.json"),
        "operator_head_brief": safe_text(summary.get("operator_head_brief")),
        "review_head_brief": safe_text(summary.get("review_head_brief")),
        "repair_head_brief": safe_text(summary.get("repair_head_brief")),
        "remote_live_gate_brief": safe_text(summary.get("remote_live_gate_brief")),
        "lane_state_brief": safe_text(summary.get("lane_state_brief")),
        "lane_priority_order_brief": safe_text(summary.get("lane_priority_order_brief")),
        "action_queue_brief": safe_text(summary.get("action_queue_brief")),
        "crypto_refresh_reuse_brief": safe_text(summary.get("crypto_refresh_reuse_brief")),
        "remote_live_history_brief": safe_text(summary.get("remote_live_history_brief")),
        "brooks_refresh_brief": safe_text(summary.get("brooks_refresh_brief")),
        "event_crisis_primary_theater_brief": safe_text(summary.get("event_crisis_primary_theater_brief")),
        "event_crisis_dominant_chain_brief": safe_text(summary.get("event_crisis_dominant_chain_brief")),
        "event_crisis_safety_margin_brief": safe_text(summary.get("event_crisis_safety_margin_brief")),
        "event_crisis_hard_boundary_brief": safe_text(summary.get("event_crisis_hard_boundary_brief")),
        "commodity_reasoning_primary_scenario_brief": safe_text(summary.get("commodity_reasoning_primary_scenario_brief")),
        "commodity_reasoning_primary_chain_brief": safe_text(summary.get("commodity_reasoning_primary_chain_brief")),
        "commodity_reasoning_range_scope_brief": safe_text(summary.get("commodity_reasoning_range_scope_brief")),
        "commodity_reasoning_boundary_strength_brief": safe_text(summary.get("commodity_reasoning_boundary_strength_brief")),
        "commodity_reasoning_invalidator_brief": safe_text(summary.get("commodity_reasoning_invalidator_brief")),
        "feedback_projection_artifact": safe_text(feedback_payload.get("artifact")),
        "feedback_projection_latest_artifact": safe_text(feedback_payload.get("latest_artifact")),
        "external_intelligence_status": safe_text(external_refresh_payload.get("status")),
        "external_intelligence_refresh_artifact": safe_text(external_refresh_payload.get("artifact_json")),
        "external_intelligence_snapshot_artifact": safe_text(external_refresh_payload.get("external_intelligence_path")),
        "external_intelligence_recommended_brief": safe_text(external_refresh_payload.get("recommended_brief")),
        "external_intelligence_takeaway": safe_text(external_refresh_payload.get("takeaway")),
        "cpa_control_plane_status": safe_text(cpa_control_payload.get("status")),
        "cpa_control_plane_artifact": safe_text(cpa_control_payload.get("artifact_json")),
        "cpa_control_plane_public_path": safe_text(cpa_control_payload.get("public_path")),
        "snapshot_outputs": list(snapshot_payload.get("outputs") or []),
        "sync_results": sync_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Fenlie operator panel and dashboard snapshots into public, then sync non-sensitive outputs into dist.",
    )
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--now", help="Explicit UTC timestamp for deterministic artifact selection.")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    dashboard_root = system_root / "dashboard" / "web"
    review_dir = system_root / "output" / "review"
    public_dir = dashboard_root / "public"
    dist_dir = dashboard_root / "dist"
    runtime_now = parse_now(args.now)

    public_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    ensure_event_crisis_summary_ready(
        system_root=system_root,
        review_dir=review_dir,
        runtime_now=runtime_now,
    )

    try:
        external_refresh_payload = run_json(
            name="run_external_intelligence_refresh",
            cmd=[
                "python3",
                str(system_root / "scripts" / "run_external_intelligence_refresh.py"),
                "--workspace",
                str(workspace),
                "--public-dir",
                str(public_dir),
                "--now",
                fmt_utc(runtime_now),
                "--skip-dashboard-snapshot",
            ],
        )
    except RuntimeError as exc:
        external_refresh_payload = degraded_external_refresh_payload(exc)

    panel_payload = run_json(
        name="build_operator_task_visual_panel",
        cmd=[
            "python3",
            str(system_root / "scripts" / "build_operator_task_visual_panel.py"),
            "--review-dir",
            str(review_dir),
            "--dashboard-dist",
            str(public_dir),
            "--now",
            fmt_utc(runtime_now),
        ],
    )

    feedback_payload = run_json(
        name="build_conversation_feedback_projection_internal",
        cmd=[
            "python3",
            str(system_root / "scripts" / "build_conversation_feedback_projection_internal.py"),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(runtime_now),
        ],
    )

    cpa_control_payload = run_json(
        name="build_cpa_control_plane_snapshot",
        cmd=[
            "python3",
            str(system_root / "scripts" / "build_cpa_control_plane_snapshot.py"),
            "--workspace",
            str(workspace),
            "--public-dir",
            str(public_dir),
        ],
    )

    snapshot_payload = run_json(
        name="build_dashboard_frontend_snapshot",
        cmd=[
            "python3",
            str(system_root / "scripts" / "build_dashboard_frontend_snapshot.py"),
            "--workspace",
            str(workspace),
            "--public-dir",
            str(public_dir),
        ],
    )

    sync_results = [
        sync_copy(public_dir / "operator_task_visual_panel.html", dist_dir / "operator_task_visual_panel.html"),
        sync_copy(public_dir / "operator_task_visual_panel_data.json", dist_dir / "operator_task_visual_panel_data.json"),
        sync_copy(public_dir / "data" / "cpa_control_plane_snapshot.json", dist_dir / "data" / "cpa_control_plane_snapshot.json"),
        sync_copy(public_dir / "data" / "fenlie_dashboard_snapshot.json", dist_dir / "data" / "fenlie_dashboard_snapshot.json"),
        sync_copy(public_dir / "data" / "fenlie_dashboard_internal_snapshot.json", dist_dir / "data" / "fenlie_dashboard_internal_snapshot.json"),
    ]

    print(
        json.dumps(
            build_summary(
                workspace=workspace,
                public_dir=public_dir,
                dist_dir=dist_dir,
                external_refresh_payload=external_refresh_payload,
                cpa_control_payload=cpa_control_payload,
                panel_payload=panel_payload,
                snapshot_payload=snapshot_payload,
                feedback_payload=feedback_payload,
                sync_results=sync_results,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
