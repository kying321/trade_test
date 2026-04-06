#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def workspace_to_system_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    return workspace if workspace.name == "system" else workspace / "system"


def review_dir_for_workspace(workspace: Path) -> Path:
    return workspace_to_system_root(workspace) / "output" / "review"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_is_active(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    status = str(payload.get("status") or "").strip().lower()
    ok_value = payload.get("ok")
    if status in {"ok", "partial"} and ok_value is not False:
        return True
    if bool(ok_value) and not status.startswith("blocked"):
        return True
    return False


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
    lines = [
        "# External Intelligence Snapshot",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- status: `{payload['status']}`",
        f"- ok: `{str(bool(payload['ok'])).lower()}`",
        f"- change_class: `{payload['change_class']}`",
        f"- sources_total: `{summary.get('sources_total', 0)}`",
        f"- recommended_brief: `{payload.get('recommended_brief') or '-'}`",
        f"- takeaway: `{payload.get('takeaway') or '-'}`",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(review_dir: Path, payload: dict[str, Any], stamp: str) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"{stamp}_external_intelligence_snapshot.json"
    md_path = review_dir / f"{stamp}_external_intelligence_snapshot.md"
    latest_json = review_dir / "latest_external_intelligence_snapshot.json"
    latest_md = review_dir / "latest_external_intelligence_snapshot.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def run_snapshot(*, workspace: Path) -> dict[str, Any]:
    generated_at = now_utc()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    review_dir = review_dir_for_workspace(workspace)

    jin10_path = review_dir / "latest_jin10_mcp_snapshot.json"
    axios_path = review_dir / "latest_axios_site_snapshot.json"
    jin10 = load_json(jin10_path) if jin10_path.exists() else {}
    axios = load_json(axios_path) if axios_path.exists() else {}

    active_sources: list[str] = []
    if source_is_active(jin10):
        active_sources.append("jin10")
    if source_is_active(axios):
        active_sources.append("axios")

    jin10_summary = jin10.get("summary", {}) if source_is_active(jin10) and isinstance(jin10.get("summary", {}), dict) else {}
    axios_summary = axios.get("summary", {}) if source_is_active(axios) and isinstance(axios.get("summary", {}), dict) else {}
    quote_watch = [
        str((row or {}).get("name") or (row or {}).get("code") or "").strip()
        for row in list(jin10_summary.get("quote_watch") or [])
        if isinstance(row, dict)
    ]
    takeaways = [
        str(jin10.get("takeaway") or "").strip() if source_is_active(jin10) else "",
        str(axios.get("takeaway") or "").strip() if source_is_active(axios) else "",
    ]
    takeaways = [value for value in takeaways if value]

    payload: dict[str, Any] = {
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "external_intelligence_snapshot",
        "change_class": "RESEARCH_ONLY",
        "ok": bool(active_sources),
        "status": "ok" if len(active_sources) == 2 else "partial" if active_sources else "blocked_missing_sources",
        "sources": {
            "jin10": {
                "status": jin10.get("status"),
                "recommended_brief": jin10.get("recommended_brief"),
                "takeaway": jin10.get("takeaway"),
                "path": str(jin10_path) if jin10 else "",
            },
            "axios": {
                "status": axios.get("status"),
                "recommended_brief": axios.get("recommended_brief"),
                "takeaway": axios.get("takeaway"),
                "path": str(axios_path) if axios else "",
            },
        },
        "summary": {
            "sources_total": len(active_sources),
            "active_sources": active_sources,
            "calendar_total": jin10_summary.get("calendar_total", 0),
            "high_importance_count": jin10_summary.get("high_importance_count", 0),
            "flash_total": jin10_summary.get("flash_total", 0),
            "quote_watch": quote_watch,
            "axios_news_total": axios_summary.get("news_total", 0),
            "axios_local_total": axios_summary.get("local_total", 0),
            "axios_national_total": axios_summary.get("national_total", 0),
            "top_titles": list(axios_summary.get("top_titles") or [])[:3],
            "top_keywords": list(axios_summary.get("top_keywords") or [])[:5],
        },
        "recommended_brief": " | ".join(
            [
                f"sources={len(active_sources)}",
                f"calendar={jin10_summary.get('calendar_total', 0)}",
                f"flash={jin10_summary.get('flash_total', 0)}",
                f"quotes={len(quote_watch)}",
                f"news={axios_summary.get('news_total', 0)}",
            ]
        ),
        "takeaway": " ｜ ".join(takeaways),
    }

    json_path, md_path = write_artifacts(review_dir, payload, stamp)
    payload["artifact_json"] = str(json_path)
    payload["artifact_md"] = str(md_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified external intelligence snapshot from sidecars.")
    parser.add_argument("--workspace", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_snapshot(workspace=Path(args.workspace).expanduser().resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
