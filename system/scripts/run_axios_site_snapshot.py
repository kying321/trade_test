#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.research.axios_site_client import AxiosSiteClient, build_summary


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def workspace_to_system_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    return workspace if workspace.name == "system" else workspace / "system"


def review_dir_for_workspace(workspace: Path) -> Path:
    return workspace_to_system_root(workspace) / "output" / "review"


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Axios Site Snapshot",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- status: `{payload['status']}`",
        f"- ok: `{str(bool(payload['ok'])).lower()}`",
        f"- change_class: `{payload['change_class']}`",
        f"- news_total: `{payload['summary']['news_total']}`",
        f"- local_total: `{payload['summary']['local_total']}`",
        f"- national_total: `{payload['summary']['national_total']}`",
        f"- takeaway: `{payload['takeaway'] or '-'} `",
    ]
    if payload["summary"]["top_titles"]:
        lines.append(f"- top_titles: `{ ' ｜ '.join(payload['summary']['top_titles']) }`")
    if payload["summary"]["top_keywords"]:
        lines.append(f"- top_keywords: `{ ', '.join(payload['summary']['top_keywords']) }`")
    return "\n".join(lines) + "\n"


def write_artifacts(review_dir: Path, payload: dict[str, Any], stamp: str) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"{stamp}_axios_site_snapshot.json"
    md_path = review_dir / f"{stamp}_axios_site_snapshot.md"
    latest_json = review_dir / "latest_axios_site_snapshot.json"
    latest_md = review_dir / "latest_axios_site_snapshot.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def run_snapshot(*, workspace: Path, limit: int = 20) -> dict[str, Any]:
    generated_at = now_utc()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    review_dir = review_dir_for_workspace(workspace)

    client = AxiosSiteClient()
    entries = client.fetch_news_entries(limit=limit)
    summary = build_summary(entries)
    payload: dict[str, Any] = {
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "axios_site_snapshot",
        "change_class": "RESEARCH_ONLY",
        "ok": True,
        "status": "ok",
        "entries": entries,
        "summary": summary,
        "recommended_brief": f"axios news={summary['news_total']} | local={summary['local_total']} | national={summary['national_total']}",
        "takeaway": summary.get("takeaway") or "",
    }
    json_path, md_path = write_artifacts(review_dir, payload, stamp)
    payload["artifact_json"] = str(json_path)
    payload["artifact_md"] = str(md_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Axios public-site research snapshot.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_snapshot(workspace=Path(args.workspace).expanduser().resolve(), limit=int(args.limit))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
