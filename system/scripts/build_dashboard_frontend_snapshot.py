#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
SENSITIVE_PATH_PATTERNS = (
    "live_takeover",
    "remote_live",
    "openclaw",
    "execution_preview",
    "paper_execution",
    "notification_preview",
    "panic_close",
    "order_routing",
    "fund",
    "capital",
)
SENSITIVE_KEY_PATTERNS = (
    "secret",
    "token",
    "password",
    "authorization",
    "cookie",
    "signature",
    "passphrase",
    "private_key",
    "api_key",
    "webhook",
)

POSITIVE_TEXT = ("ok", "allowed", "active", "passed", "ready", "mixed_positive", "promising")
NEGATIVE_TEXT = ("blocked", "failed", "error", "demoted", "negative", "harmful", "overblocking", "not_promising")
WARNING_TEXT = ("watch", "candidate", "mixed", "partial", "warning", "pending", "inconclusive", "local", "defer")
WORKSPACE_DEFAULT_ARTIFACT_PRIORITY = (
    "price_action_breakout_pullback",
    "price_action_exit_risk",
    "hold_selection_handoff",
    "cross_section_backtest",
)


@dataclass(frozen=True)
class SurfaceSpec:
    key: str
    output_name: str
    expose_absolute_paths: bool
    redaction_level: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build a read-only non-sensitive dashboard snapshot for the web frontend.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--public-dir", type=Path, default=workspace_default / "system" / "dashboard" / "web" / "public")
    parser.add_argument("--review-dir", type=Path, default=workspace_default / "system" / "output" / "review")
    parser.add_argument("--artifacts-dir", type=Path, default=workspace_default / "system" / "output" / "artifacts")
    parser.add_argument("--max-catalog", type=int, default=80)
    parser.add_argument("--max-backtests", type=int, default=10)
    parser.add_argument("--max-equity-points", type=int, default=400)
    return parser.parse_args()


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def sort_key(path: Path) -> tuple[str, float, str]:
    return (artifact_stamp(path), path.stat().st_mtime, path.name)


def resolved_path_key(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def latest_by_glob(directory: Path, pattern: str) -> Path | None:
    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def latest_review_suffix(review_dir: Path, suffix: str) -> Path | None:
    return latest_by_glob(review_dir, f"*_{suffix}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return payload


def deep_get(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def path_is_sensitive(path: Path) -> bool:
    lowered = path.name.lower()
    return any(pattern in lowered for pattern in SENSITIVE_PATH_PATTERNS)


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if any(pattern in lowered for pattern in SENSITIVE_KEY_PATTERNS):
                continue
            sanitized[str(key)] = sanitize_value(child)
        return sanitized
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def path_locator(path: Path, workspace: Path, review_dir: Path, artifacts_dir: Path, public_dir: Path) -> str:
    resolved = path.resolve()
    candidates = (
        (review_dir.resolve(), "review"),
        (artifacts_dir.resolve(), "artifacts"),
        (public_dir.resolve(), "dashboard_public"),
        ((workspace / "system").resolve(), "system"),
        (workspace.resolve(), "workspace"),
    )
    for base, prefix in candidates:
        try:
            relative = resolved.relative_to(base)
            return f"{prefix}/{relative.as_posix()}"
        except ValueError:
            continue
    return path.name


def surface_path(path: Path, surface: SurfaceSpec, workspace: Path, review_dir: Path, artifacts_dir: Path, public_dir: Path) -> str:
    locator = path_locator(path, workspace, review_dir, artifacts_dir, public_dir)
    return str(path) if surface.expose_absolute_paths else locator


def sample_curve(curve: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(curve) <= max_points:
        return curve
    if max_points <= 1:
        return [curve[-1]]
    step = (len(curve) - 1) / float(max_points - 1)
    sampled = []
    for idx in range(max_points):
        sampled.append(curve[round(idx * step)])
    return sampled


def summarize_review_payload(
    label: str,
    payload: Any,
    path: Path,
    category: str,
    payload_key: str | None,
    display_path: str,
    artifact_layer: str,
    artifact_group: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "id": payload_key or path.name,
        "label": label,
        "path": display_path,
        "category": category,
        "payload_key": payload_key,
        "artifact_layer": artifact_layer,
        "artifact_group": artifact_group,
    }
    if isinstance(payload, dict):
        inferred_status = payload.get("status")
        if not inferred_status and label == "system_scheduler_state":
            inferred_status = deep_get(payload, "interval_tasks.micro_capture.status")
        if not inferred_status and label == "mode_stress_matrix":
            inferred_status = "ok" if payload.get("best_mode") else ""
        summary.update(
            {
                "action": payload.get("action") or payload.get("generated_at"),
                "status": inferred_status or ("ok" if payload.get("ok") else ""),
                "change_class": payload.get("change_class"),
                "generated_at_utc": payload.get("generated_at_utc") or payload.get("generated_at") or payload.get("as_of"),
                "research_decision": payload.get("research_decision"),
                "recommended_brief": payload.get("recommended_brief"),
                "takeaway": payload.get("takeaway") or payload.get("comparison_takeaway"),
                "top_level_keys": list(payload.keys())[:18],
            }
        )
    elif isinstance(payload, list):
        summary["status"] = "list"
        summary["top_level_keys"] = [f"len={len(payload)}"]
    return summary


def status_tone(value: str | None) -> str:
    text = str(value or "").lower()
    if not text:
        return "neutral"
    if any(token in text for token in POSITIVE_TEXT):
        return "positive"
    if any(token in text for token in NEGATIVE_TEXT):
        return "negative"
    if any(token in text for token in WARNING_TEXT):
        return "warning"
    return "neutral"


def payload_entry(
    label: str,
    path: Path | None,
    category: str,
    artifact_group: str,
    *,
    surface: SurfaceSpec,
    workspace: Path,
    review_dir: Path,
    artifacts_dir: Path,
    public_dir: Path,
) -> tuple[str, dict[str, Any]] | None:
    if path is None or not path.exists() or path_is_sensitive(path):
        return None
    raw = load_json(path)
    payload = sanitize_value(raw)
    display_path = surface_path(path, surface, workspace, review_dir, artifacts_dir, public_dir)
    key = label
    return key, {
        "label": label.replace("_", " "),
        "path": display_path,
        "category": category,
        "payload": payload,
        "summary": summarize_review_payload(label.replace("_", " "), payload, path, category, label, display_path, "canonical", artifact_group),
    }


def infer_category(path: Path) -> str:
    lowered = path.name.lower()
    if "operator_task_visual_panel" in lowered:
        return "operator-panel"
    if "backtest" in lowered:
        return "backtest"
    if "sim_only" in lowered:
        return "sim-only"
    if "hold_" in lowered or "price_action" in lowered or "geometry_delta" in lowered:
        return "research"
    return "review"


def build_catalog(
    review_dir: Path,
    artifact_payloads: dict[str, dict[str, Any]],
    selected_paths: dict[str, tuple[Path | None, str, str]],
    max_catalog: int,
    *,
    surface: SurfaceSpec,
    workspace: Path,
    artifacts_dir: Path,
    public_dir: Path,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, entry in artifact_payloads.items():
        rows.append(entry["summary"])
        selected_path = selected_paths.get(key, (None, "", ""))[0]
        path_key = resolved_path_key(selected_path)
        if path_key:
            seen.add(path_key)

    patterns = (
        "latest_*.json",
        "*_sim_only.json",
        "*_backtest*.json",
        "*_operator_task_visual_panel.json",
        "*_recent_strategy_backtest_comparison_report.json",
    )
    extra_paths: list[Path] = []
    for pattern in patterns:
        extra_paths.extend(path for path in review_dir.glob(pattern) if path.is_file())
    extra_paths = sorted({path for path in extra_paths if not path_is_sensitive(path)}, key=sort_key, reverse=True)

    for path in extra_paths:
        path_key = resolved_path_key(path)
        if path_key in seen:
            continue
        payload = sanitize_value(load_json(path))
        display_path = surface_path(path, surface, workspace, review_dir, artifacts_dir, public_dir)
        rows.append(summarize_review_payload(path.stem, payload, path, infer_category(path), None, display_path, "archive", "archive"))
        seen.add(path_key)
        if len(rows) >= max_catalog:
            break
    return rows[:max_catalog]


def build_domain_summary(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in catalog:
        category = str(item.get("category") or "uncategorized")
        row = grouped.setdefault(
            category,
            {
                "id": category,
                "label": category,
                "count": 0,
                "payload_count": 0,
                "positive": 0,
                "warning": 0,
                "negative": 0,
                "neutral": 0,
                "headline": "",
            },
        )
        row["count"] += 1
        if item.get("payload_key"):
            row["payload_count"] += 1
        row[status_tone(str(item.get("status") or item.get("research_decision") or ""))] += 1
        if not row["headline"]:
            row["headline"] = str(item.get("research_decision") or item.get("takeaway") or item.get("recommended_brief") or "")
    return sorted(grouped.values(), key=lambda row: (-int(row["count"]), str(row["label"])))


def build_interface_catalog(ui_routes: dict[str, Any], route_contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in route_contract.get("interface_catalog", []):
        if not isinstance(item, dict):
            continue
        url_key = str(item.get("url_key") or item.get("id") or "")
        rows.append(
            {
                "id": item.get("id") or url_key,
                "label": item.get("label") or url_key,
                "kind": item.get("kind") or "local-static",
                "url": ui_routes.get(url_key, item.get("url")),
                "description": item.get("description") or "",
            }
        )
    return rows


def build_public_topology(ui_routes: dict[str, Any], route_contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in route_contract.get("public_topology", []):
        if not isinstance(item, dict):
            continue
        url_key = str(item.get("url_key") or "")
        url = ui_routes.get(url_key, item.get("url"))
        row = copy.deepcopy(item)
        row["id"] = item.get("id") or url_key or str(url or "")
        row["url"] = url or ""
        rows.append(row)
    return rows


def build_source_heads(artifact_payloads: dict[str, dict[str, Any]], source_head_contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in source_head_contract.get("source_heads", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "")
        label = str(item.get("label") or key)
        if not key:
            continue
        entry = artifact_payloads.get(key)
        if not entry:
            continue
        payload = entry.get("payload") if isinstance(entry, dict) else {}
        summary = entry.get("summary") if isinstance(entry, dict) else {}
        result[key] = {
            "label": label,
            "status": summary.get("status"),
            "research_decision": summary.get("research_decision"),
            "recommended_brief": summary.get("recommended_brief"),
            "takeaway": summary.get("takeaway"),
            "path": entry.get("path"),
            "summary": summary.get("recommended_brief") or summary.get("research_decision") or summary.get("takeaway"),
            "active_baseline": payload.get("active_baseline") if isinstance(payload, dict) else None,
            "local_candidate": payload.get("local_candidate") if isinstance(payload, dict) else None,
            "transfer_watch": payload.get("transfer_watch") if isinstance(payload, dict) else None,
            "return_candidate": payload.get("return_candidate") if isinstance(payload, dict) else None,
            "return_watch": payload.get("return_watch") if isinstance(payload, dict) else None,
            "generated_at_utc": summary.get("generated_at_utc"),
        }
    return result


def ensure_event_artifacts(selected_paths: dict[str, tuple[Path | None, str, str]], review_dir: Path) -> None:
    event_artifacts = [
        ("event_regime_snapshot", "research", "event_insight", "event_regime_snapshot.json"),
        ("event_crisis_analogy", "research", "event_insight", "event_crisis_analogy.json"),
        ("event_asset_shock_map", "research", "event_insight", "event_asset_shock_map.json"),
        ("event_crisis_operator_summary", "research", "event_insight", "event_crisis_operator_summary.json"),
    ]
    for artifact_id, category, artifact_group, suffix in event_artifacts:
        if artifact_id in selected_paths:
            continue
        candidate = latest_review_suffix(review_dir, suffix)
        if candidate:
            selected_paths[artifact_id] = (candidate, category, artifact_group)


def build_backtests(
    artifacts_dir: Path,
    max_backtests: int,
    max_points: int,
    *,
    surface: SurfaceSpec,
    workspace: Path,
    review_dir: Path,
    public_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = sorted(artifacts_dir.glob("backtest_*.json"), key=sort_key, reverse=True)
    for path in candidates[:max_backtests]:
        payload = sanitize_value(load_json(path))
        if not isinstance(payload, dict):
            continue
        curve = payload.get("equity_curve") if isinstance(payload.get("equity_curve"), list) else []
        rows.append(
            {
                "id": path.stem,
                "label": path.name.replace(".json", ""),
                "path": surface_path(path, surface, workspace, review_dir, artifacts_dir, public_dir),
                "start": payload.get("start"),
                "end": payload.get("end"),
                "total_return": payload.get("total_return"),
                "annual_return": payload.get("annual_return"),
                "max_drawdown": payload.get("max_drawdown"),
                "win_rate": payload.get("win_rate"),
                "profit_factor": payload.get("profit_factor"),
                "expectancy": payload.get("expectancy"),
                "trades": payload.get("trades"),
                "violations": payload.get("violations"),
                "positive_window_ratio": payload.get("positive_window_ratio"),
                "equity_curve_sample": sample_curve(curve, max_points),
            }
        )
    return rows


def build_surface_contracts(route_contract: dict[str, Any], surface: SurfaceSpec) -> dict[str, Any]:
    contracts = copy.deepcopy(route_contract.get("surface_contracts", {}))
    public_contract = contracts.setdefault("public", {})
    internal_contract = contracts.setdefault("internal", {})

    public_contract.setdefault("availability", "active")
    public_contract.setdefault("redaction_level", "public_summary")
    public_contract.setdefault("notes", "公开数据面仅暴露非敏感摘要与安全路径定位。")

    internal_contract.setdefault("fallback_snapshot_path", public_contract.get("snapshot_path", "/data/fenlie_dashboard_snapshot.json"))
    internal_contract.setdefault("deploy_policy", "strip_from_public_dist")

    if surface.key == "internal":
        internal_contract["availability"] = "active"
        internal_contract["redaction_level"] = surface.redaction_level
        internal_contract["notes"] = "内部只读快照已挂载（本地或私有环境）。"
    else:
        internal_contract["availability"] = "local_only"
        internal_contract["notes"] = "内部快照仅在本地/私有预览环境保留，公开部署前剥离。"

    return contracts


def workspace_focus_defaults_for_group(artifact_group: str) -> tuple[str | None, str | None]:
    if artifact_group == "system_anchor":
        return "orchestration", "freshness"
    return "lab-review", "research-heads"


def build_workspace_default_focus(artifact_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not artifact_payloads:
        return {}

    preferred_id = ""
    for artifact_id in WORKSPACE_DEFAULT_ARTIFACT_PRIORITY:
        if artifact_id in artifact_payloads:
            preferred_id = artifact_id
            break

    if not preferred_id:
        for artifact_id, entry in artifact_payloads.items():
            summary = entry.get("summary") if isinstance(entry, dict) else {}
            if isinstance(summary, dict) and summary.get("artifact_group") == "research_mainline":
                preferred_id = artifact_id
                break

    if not preferred_id:
        preferred_id = next(iter(artifact_payloads.keys()), "")

    if not preferred_id:
        return {}

    entry = artifact_payloads.get(preferred_id, {})
    summary = entry.get("summary") if isinstance(entry, dict) else {}
    artifact_group = str(summary.get("artifact_group") or "")
    panel, section = workspace_focus_defaults_for_group(artifact_group)
    focus = {
        "artifact": preferred_id,
        "group": artifact_group or None,
        "search_scope": "title",
        "panel": panel,
        "section": section,
    }
    return {key: value for key, value in focus.items() if value}


def load_internal_feedback_projection(review_dir: Path) -> dict[str, Any] | None:
    path = review_dir / "latest_conversation_feedback_projection_internal.json"
    if not path.exists():
        return None
    payload = sanitize_value(load_json(path))
    if not isinstance(payload, dict):
        return None
    return payload


def build_surface_snapshot(
    *,
    surface: SurfaceSpec,
    workspace: Path,
    public_dir: Path,
    review_dir: Path,
    artifacts_dir: Path,
    selected_paths: dict[str, tuple[Path | None, str, str]],
    route_contract: dict[str, Any],
    source_head_contract: dict[str, Any],
    max_catalog: int,
    max_backtests: int,
    max_equity_points: int,
) -> dict[str, Any]:
    ensure_event_artifacts(selected_paths, review_dir)
    generated_at = utc_now()
    artifact_payloads: dict[str, dict[str, Any]] = {}
    for label, (path, category, artifact_group) in selected_paths.items():
        entry = payload_entry(
            label,
            path,
            category,
            artifact_group,
            surface=surface,
            workspace=workspace,
            review_dir=review_dir,
            artifacts_dir=artifacts_dir,
            public_dir=public_dir,
        )
        if entry:
            artifact_payloads[entry[0]] = entry[1]

    catalog = build_catalog(
        review_dir,
        artifact_payloads,
        selected_paths,
        max_catalog=max_catalog,
        surface=surface,
        workspace=workspace,
        artifacts_dir=artifacts_dir,
        public_dir=public_dir,
    )
    backtests = build_backtests(
        artifacts_dir,
        max_backtests=max_backtests,
        max_points=max_equity_points,
        surface=surface,
        workspace=workspace,
        review_dir=review_dir,
        public_dir=public_dir,
    )
    domain_summary = build_domain_summary(catalog)
    ui_routes = route_contract.get("ui_routes", {})
    interface_catalog = build_interface_catalog(ui_routes, route_contract)
    public_topology = build_public_topology(ui_routes, route_contract)
    source_heads = build_source_heads(artifact_payloads, source_head_contract)

    snapshot = {
        "action": "build_dashboard_frontend_snapshot",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "surface": surface.key,
        "generated_at_utc": generated_at,
    "workspace": surface.expose_absolute_paths and str(workspace) or workspace.name,
        "ui_routes": ui_routes,
        "meta": {
            "generated_at_utc": generated_at,
            "artifact_payload_count": len(artifact_payloads),
            "catalog_count": len(catalog),
            "backtest_artifact_count": len(backtests),
            "change_class": "RESEARCH_ONLY",
            "surface_key": surface.key,
            "redaction_level": surface.redaction_level,
        },
        "artifact_payloads": artifact_payloads,
        "catalog": catalog,
        "domain_summary": domain_summary,
        "interface_catalog": interface_catalog,
        "public_topology": public_topology,
        "source_heads": source_heads,
        "workspace_default_focus": build_workspace_default_focus(artifact_payloads),
        "experience_contract": route_contract.get("experience_contract", {}),
        "surface_contracts": build_surface_contracts(route_contract, surface),
        "backtest_artifacts": backtests,
    }
    if surface.key == "internal":
        snapshot["conversation_feedback_projection"] = load_internal_feedback_projection(review_dir)

    output_path = public_dir / "data" / surface.output_name
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "surface": surface.key,
        "path": str(output_path),
        "artifact_payload_count": len(artifact_payloads),
        "catalog_count": len(catalog),
        "backtest_artifact_count": len(backtests),
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    public_dir = args.public_dir.expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    config_dir = workspace / "system" / "config"
    artifact_contract = load_config(config_dir / "dashboard_snapshot_artifact_selection.json")
    route_contract = load_config(config_dir / "dashboard_snapshot_ui_routes.json")
    source_head_contract = load_config(config_dir / "dashboard_snapshot_source_heads.json")

    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "data").mkdir(parents=True, exist_ok=True)

    operator_panel_json = None
    for candidate in (
        public_dir / "operator_task_visual_panel_data.json",
        workspace / "system" / "dashboard" / "web" / "dist" / "operator_task_visual_panel_data.json",
        latest_review_suffix(review_dir, "operator_task_visual_panel.json"),
    ):
        if candidate and Path(candidate).exists():
            operator_panel_json = Path(candidate)
            break

    selected_paths: dict[str, tuple[Path | None, str, str]] = {}
    for item in artifact_contract.get("artifact_selection", []):
        if not isinstance(item, dict):
            continue
        artifact_id = str(item.get("id") or "")
        category = str(item.get("category") or "review")
        artifact_group = str(item.get("artifact_group") or "research_support")
        path_mode = str(item.get("path_mode") or "")
        value = str(item.get("value") or "")
        resolved_path: Path | None = None
        if path_mode == "operator_panel_auto":
            resolved_path = operator_panel_json
        elif path_mode == "latest_review_suffix":
            resolved_path = latest_review_suffix(review_dir, value)
        elif path_mode == "review_path":
            resolved_path = review_dir / value
        elif path_mode == "workspace_path":
            resolved_path = workspace / value
        if artifact_id:
            selected_paths[artifact_id] = (resolved_path, category, artifact_group)

    outputs = []
    for surface in (
        SurfaceSpec(key="public", output_name="fenlie_dashboard_snapshot.json", expose_absolute_paths=False, redaction_level="public_summary"),
        SurfaceSpec(key="internal", output_name="fenlie_dashboard_internal_snapshot.json", expose_absolute_paths=True, redaction_level="full_internal"),
    ):
        outputs.append(
            build_surface_snapshot(
                surface=surface,
                workspace=workspace,
                public_dir=public_dir,
                review_dir=review_dir,
                artifacts_dir=artifacts_dir,
                selected_paths=selected_paths,
                route_contract=route_contract,
                source_head_contract=source_head_contract,
                max_catalog=int(args.max_catalog),
                max_backtests=int(args.max_backtests),
                max_equity_points=int(args.max_equity_points),
            )
        )
    print(json.dumps({"ok": True, "outputs": outputs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
