#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
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


def summarize_review_payload(label: str, payload: Any, path: Path, category: str, payload_key: str | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "id": path.name,
        "label": label,
        "path": str(path),
        "category": category,
        "payload_key": payload_key,
    }
    if isinstance(payload, dict):
        summary.update(
            {
                "action": payload.get("action") or payload.get("generated_at"),
                "status": payload.get("status") or ("ok" if payload.get("ok") else ""),
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


def payload_entry(label: str, path: Path | None, category: str) -> tuple[str, dict[str, Any]] | None:
    if path is None or not path.exists() or path_is_sensitive(path):
        return None
    raw = load_json(path)
    payload = sanitize_value(raw)
    key = label
    return key, {
        "label": label.replace("_", " "),
        "path": str(path),
        "category": category,
        "payload": payload,
        "summary": summarize_review_payload(label.replace("_", " "), payload, path, category, label),
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


def build_catalog(review_dir: Path, artifact_payloads: dict[str, dict[str, Any]], max_catalog: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, entry in artifact_payloads.items():
        rows.append(entry["summary"])
        seen.add(entry["path"])

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
        if str(path) in seen:
            continue
        payload = sanitize_value(load_json(path))
        rows.append(summarize_review_payload(path.stem, payload, path, infer_category(path), None))
        seen.add(str(path))
        if len(rows) >= max_catalog:
            break
    return rows[:max_catalog]


def build_backtests(artifacts_dir: Path, max_backtests: int, max_points: int) -> list[dict[str, Any]]:
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
                "path": str(path),
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


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    public_dir = args.public_dir.expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()

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

    selected_paths = {
        "operator_panel": (operator_panel_json, "operator-panel"),
        "recent_strategy_backtests": (latest_review_suffix(review_dir, "recent_strategy_backtest_comparison_report.json"), "backtest"),
        "backtest_snapshot": (latest_review_suffix(review_dir, "backtest_snapshot_breakdown.json"), "backtest"),
        "price_action_breakout_pullback": (latest_review_suffix(review_dir, "price_action_breakout_pullback_sim_only.json"), "sim-only"),
        "price_action_exit_risk": (latest_review_suffix(review_dir, "price_action_breakout_pullback_exit_risk_sim_only.json"), "sim-only"),
        "price_action_exit_hold_robustness": (review_dir / "latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json", "sim-only"),
        "price_action_exit_rider_triage": (review_dir / "latest_price_action_breakout_pullback_exit_rider_triage_sim_only.json", "sim-only"),
        "hold_selection_handoff": (review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json", "research"),
        "hold_selection_gate": (review_dir / "latest_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.json", "research"),
        "hold_family_transfer": (review_dir / "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json", "research"),
        "hold_router_transfer": (review_dir / "latest_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json", "research"),
        "hold_frontier_report": (review_dir / "latest_price_action_breakout_pullback_hold_frontier_report_sim_only.json", "research"),
        "hold_frontier_cost_sensitivity": (review_dir / "latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json", "research"),
        "hold_candidate_divergence": (review_dir / "latest_price_action_breakout_pullback_hold_candidate_divergence_casebook_sim_only.json", "research"),
        "geometry_delta_breakout": (latest_review_suffix(review_dir, "geometry_delta_breakout_sim_only.json"), "sim-only"),
        "cross_section_backtest": (latest_review_suffix(review_dir, "crypto_shortline_cross_section_backtest.json"), "backtest"),
    }

    artifact_payloads: dict[str, dict[str, Any]] = {}
    for label, (path, category) in selected_paths.items():
        entry = payload_entry(label, path, category)
        if entry:
            artifact_payloads[entry[0]] = entry[1]

    catalog = build_catalog(review_dir, artifact_payloads, max_catalog=int(args.max_catalog))
    backtests = build_backtests(artifacts_dir, max_backtests=int(args.max_backtests), max_points=int(args.max_equity_points))

    snapshot = {
        "action": "build_dashboard_frontend_snapshot",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": utc_now(),
        "workspace": str(workspace),
        "ui_routes": {
            "frontend_dev": "http://127.0.0.1:5173",
            "operator_panel": "/operator_task_visual_panel.html",
            "snapshot_json": "/data/fenlie_dashboard_snapshot.json",
        },
        "meta": {
            "generated_at_utc": utc_now(),
            "artifact_payload_count": len(artifact_payloads),
            "catalog_count": len(catalog),
            "backtest_artifact_count": len(backtests),
            "change_class": "RESEARCH_ONLY",
        },
        "artifact_payloads": artifact_payloads,
        "catalog": catalog,
        "backtest_artifacts": backtests,
    }

    output_path = public_dir / "data" / "fenlie_dashboard_snapshot.json"
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "path": str(output_path), "artifact_payload_count": len(artifact_payloads), "catalog_count": len(catalog), "backtest_artifact_count": len(backtests)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
