#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def workspace_to_system_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    return workspace if workspace.name == "system" else workspace / "system"


def review_dir_for_workspace(workspace: Path) -> Path:
    return workspace_to_system_root(workspace) / "output" / "review"


def load_snapshot_action(workspace: Path, action_id: str) -> dict[str, Any]:
    review_dir = review_dir_for_workspace(workspace)
    snapshot_path = review_dir / "latest_cpa_control_plane_snapshot.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for action in list(payload.get("guarded_actions") or []):
        if isinstance(action, dict) and str(action.get("id") or "") == action_id:
            return action
    raise KeyError(f"guarded_action_not_found:{action_id}")


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CPA Guarded Action Receipt",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- status: `{payload['status']}`",
        f"- ok: `{str(bool(payload['ok'])).lower()}`",
        f"- action_id: `{payload['action_id']}`",
        f"- action_label: `{payload['action_label']}`",
        f"- risk_class: `{payload['risk_class']}`",
        f"- returncode: `{payload['returncode']}`",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(review_dir: Path, payload: dict[str, Any], stamp: str, action_id: str) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"{stamp}_cpa_guarded_action_receipt_{action_id}.json"
    md_path = review_dir / f"{stamp}_cpa_guarded_action_receipt_{action_id}.md"
    latest_json = review_dir / f"latest_cpa_guarded_action_receipt_{action_id}.json"
    latest_md = review_dir / f"latest_cpa_guarded_action_receipt_{action_id}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def run_action(*, workspace: Path, action_id: str, execute: bool) -> dict[str, Any]:
    generated_at = now_utc()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    review_dir = review_dir_for_workspace(workspace)
    action = load_snapshot_action(workspace, action_id)
    command = str(action.get("command") or "").strip()
    if not command:
        raise ValueError(f"guarded_action_command_missing:{action_id}")

    payload: dict[str, Any] = {
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "cpa_guarded_action",
        "change_class": "LIVE_GUARD_ONLY",
        "ok": True,
        "status": "dry_run",
        "action_id": action_id,
        "action_label": str(action.get("label") or action_id),
        "risk_class": str(action.get("risk_class") or "LIVE_GUARD_ONLY"),
        "command": command,
        "returncode": 0,
        "stdout_preview": "",
        "stderr_preview": "",
    }

    if execute:
        proc = subprocess.run(["/bin/zsh", "-lc", command], text=True, capture_output=True, check=False)
        payload["returncode"] = int(proc.returncode)
        payload["stdout_preview"] = (proc.stdout or "").strip()[:2000]
        payload["stderr_preview"] = (proc.stderr or "").strip()[:2000]
        payload["status"] = "ok" if proc.returncode == 0 else "failed"
        payload["ok"] = proc.returncode == 0

    artifact_json, artifact_md = write_artifacts(review_dir, payload, stamp, action_id)
    payload["artifact_json"] = str(artifact_json)
    payload["artifact_md"] = str(artifact_md)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a source-owned CPA guarded action.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_action(
        workspace=Path(args.workspace).expanduser().resolve(),
        action_id=str(args.action_id),
        execute=bool(args.execute),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
