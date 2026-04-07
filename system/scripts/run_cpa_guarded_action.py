#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_snapshot_builder():
    module_path = SYSTEM_ROOT / "scripts" / "build_cpa_control_plane_snapshot.py"
    spec = importlib.util.spec_from_file_location("build_cpa_control_plane_snapshot_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot_load_snapshot_builder:{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def refresh_control_snapshot(*, workspace: Path, public_dir: Path) -> dict[str, Any]:
    module = _load_snapshot_builder()
    return module.build_snapshot(workspace=workspace, public_dir=public_dir)


def parse_stdout_json(stdout: str) -> Any:
    text = str(stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_structured_summary(action_id: str, payload: Any) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    if action_id == "retry_candidate_pipeline":
        refresh = data.get("refresh") if isinstance(data.get("refresh"), dict) else {}
        acceptance = refresh.get("acceptance") if isinstance(refresh.get("acceptance"), dict) else {}
        inventory = data.get("inventory") if isinstance(data.get("inventory"), dict) else {}
        summary = inventory.get("summary") if isinstance(inventory.get("summary"), dict) else {}
        return {
            "attempted_count": int(data.get("attempted_count") or 0),
            "successful_count": len(list(data.get("successful_emails") or [])),
            "failed_count": len(list(data.get("failed_emails") or [])),
            "accepted": bool(acceptance.get("accepted", data.get("accepted", False))),
            "retry_candidate_remaining": int(summary.get("retry_candidate") or 0),
        }
    if action_id == "active_target_sync_success20":
        acceptance = data.get("acceptance") if isinstance(data.get("acceptance"), dict) else {}
        return {
            "exported_active_count": int(data.get("exported_active_count") or 0),
            "uploaded_count": int(data.get("uploaded_count") or 0),
            "synced_count": int(data.get("synced_count") or 0),
            "accepted": bool(acceptance.get("accepted", False)),
            "missing_active_paths_count": len(list(data.get("missing_active_paths") or [])),
        }
    if action_id == "acceptance_replay_success20":
        return {
            "accepted": bool(data.get("accepted", False)),
            "active_target_authfiles": int(data.get("active_target_authfiles") or 0),
            "store_verified_target_accounts": int(data.get("store_verified_target_accounts") or 0),
            "missing_in_proxy_count": len(list(data.get("missing_in_proxy") or [])),
            "missing_in_store_count": len(list(data.get("missing_in_store") or [])),
        }
    return {}


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


def run_action(*, workspace: Path, action_id: str, execute: bool, refresh_control_plane_after: bool = False) -> dict[str, Any]:
    generated_at = now_utc()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    review_dir = review_dir_for_workspace(workspace)
    public_dir = workspace_to_system_root(workspace) / "dashboard" / "web" / "public"
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
        "structured_summary": {},
        "control_snapshot_refresh": {},
    }

    if execute:
        proc = subprocess.run(["/bin/zsh", "-lc", command], text=True, capture_output=True, check=False)
        payload["returncode"] = int(proc.returncode)
        payload["stdout_preview"] = (proc.stdout or "").strip()[:2000]
        payload["stderr_preview"] = (proc.stderr or "").strip()[:2000]
        payload["status"] = "ok" if proc.returncode == 0 else "failed"
        payload["ok"] = proc.returncode == 0
        payload["structured_summary"] = build_structured_summary(action_id, parse_stdout_json(proc.stdout))
        if refresh_control_plane_after:
            payload["control_snapshot_refresh"] = refresh_control_snapshot(workspace=workspace, public_dir=public_dir)

    artifact_json, artifact_md = write_artifacts(review_dir, payload, stamp, action_id)
    payload["artifact_json"] = str(artifact_json)
    payload["artifact_md"] = str(artifact_md)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a source-owned CPA guarded action.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh-control-plane", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_action(
        workspace=Path(args.workspace).expanduser().resolve(),
        action_id=str(args.action_id),
        execute=bool(args.execute),
        refresh_control_plane_after=bool(args.refresh_control_plane),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
