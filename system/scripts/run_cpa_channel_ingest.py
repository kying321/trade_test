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

from lie_engine.cpa_channels.ingest_pipeline import ingest_bundles_to_store


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def workspace_to_system_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    return workspace if workspace.name == "system" else workspace / "system"


def review_dir_for_workspace(workspace: Path) -> Path:
    return workspace_to_system_root(workspace) / "output" / "review"


def artifacts_dir_for_workspace(workspace: Path) -> Path:
    return workspace_to_system_root(workspace) / "output" / "artifacts" / "cpa_channels"


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CPA Channel Ingest",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- status: `{payload['status']}`",
        f"- ok: `{str(bool(payload['ok'])).lower()}`",
        f"- change_class: `{payload['change_class']}`",
        f"- accounts_total: `{payload['accounts_total']}`",
        f"- run_id: `{payload['run_id']}`",
        f"- store_path: `{payload['store_path']}`",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(review_dir: Path, payload: dict[str, Any], stamp: str) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"{stamp}_cpa_channel_ingest.json"
    md_path = review_dir / f"{stamp}_cpa_channel_ingest.md"
    latest_json = review_dir / "latest_cpa_channel_ingest.json"
    latest_md = review_dir / "latest_cpa_channel_ingest.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def run_ingest(
    *,
    workspace: Path,
    input_file: Path,
    store_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    generated_at = now_utc()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    review_dir = review_dir_for_workspace(workspace)
    artifacts_dir = artifacts_dir_for_workspace(workspace)
    resolved_store = store_path or (artifacts_dir / "cpa_channels.sqlite3")
    resolved_output = output_dir or (artifacts_dir / "authfiles")

    kernel_result = ingest_bundles_to_store(
        input_file=input_file,
        store_path=resolved_store,
        output_dir=resolved_output,
    )
    payload: dict[str, Any] = {
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "cpa_channel_ingest",
        "change_class": "LIVE_GUARD_ONLY",
        "ok": True,
        "status": "ok",
        "run_id": kernel_result["run_id"],
        "accounts_total": kernel_result["accounts_total"],
        "accounts": kernel_result["accounts"],
        "store_path": str(resolved_store),
        "output_dir": str(resolved_output),
        "exported_files": [str(row.get("exported_file") or "") for row in kernel_result["accounts"]],
        "input_file": str(input_file),
    }
    artifact_json, artifact_md = write_artifacts(review_dir, payload, stamp)
    payload["artifact_json"] = str(artifact_json)
    payload["artifact_md"] = str(artifact_md)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-only CPA channel ingest/store/export pipeline.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--store-path", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_ingest(
        workspace=Path(args.workspace).expanduser().resolve(),
        input_file=Path(args.input_file).expanduser().resolve(),
        store_path=Path(args.store_path).expanduser().resolve() if str(args.store_path).strip() else None,
        output_dir=Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
