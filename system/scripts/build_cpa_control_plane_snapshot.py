#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_MAC_TOOLS_DATA_ROOT = Path("/Users/jokenrobot/Downloads/Folders/MAC工具/data")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def workspace_to_system_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    return workspace if workspace.name == "system" else workspace / "system"


def review_dir_for_workspace(workspace: Path) -> Path:
    return workspace_to_system_root(workspace) / "output" / "review"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# CPA Control Plane Snapshot",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- status: `{payload['status']}`",
        f"- ok: `{str(bool(payload['ok'])).lower()}`",
        f"- change_class: `{payload['change_class']}`",
        f"- historical_success_total: `{summary['historical_success_total']}`",
        f"- inventory_total: `{summary['inventory_total']}`",
        f"- retry_candidate_total: `{summary['retry_candidate_total']}`",
        f"- blocked_about_you_total: `{summary['blocked_about_you_total']}`",
        f"- no_retry_deactivated_total: `{summary['no_retry_deactivated_total']}`",
        f"- new_unmounted_total: `{summary['new_unmounted_total']}`",
        f"- latest_kernel_accounts_total: `{summary['latest_kernel_accounts_total']}`",
    ]
    return "\n".join(lines) + "\n"


def write_review_artifacts(review_dir: Path, payload: dict[str, Any], stamp: str) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"{stamp}_cpa_control_plane_snapshot.json"
    md_path = review_dir / f"{stamp}_cpa_control_plane_snapshot.md"
    latest_json = review_dir / "latest_cpa_control_plane_snapshot.json"
    latest_md = review_dir / "latest_cpa_control_plane_snapshot.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def build_snapshot(*, workspace: Path, public_dir: Path, source_root: Path = DEFAULT_MAC_TOOLS_DATA_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    review_dir = review_dir_for_workspace(workspace)
    source_root = source_root.expanduser().resolve()

    success_rows = read_csv_rows(source_root / "registered_success_active20.csv")
    new_unmounted_rows = read_csv_rows(source_root / "registered_new_unmounted.csv")
    review_queue_rows = read_csv_rows(source_root / "cpa_non_active_review_queue.csv")
    deactivated_rows = read_csv_rows(source_root / "cpa_no_retry_deactivated_accounts.csv")
    inventory = read_json(source_root / "cpa_account_inventory.json")
    acceptance = read_json(source_root / "active20_acceptance_20260329_212018.json")
    latest_kernel = read_json(review_dir / "latest_cpa_channel_ingest.json")

    inventory_rows = inventory if isinstance(inventory, list) else []
    acceptance_map = acceptance if isinstance(acceptance, dict) else {}
    kernel_map = latest_kernel if isinstance(latest_kernel, dict) else {}
    historical_success_emails = [str(row.get("email") or "").strip().lower() for row in success_rows if str(row.get("email") or "").strip()]

    bucket_counts: dict[str, int] = {}
    inventory_bucket_by_email: dict[str, str] = {}
    for row in inventory_rows:
        if not isinstance(row, dict):
            continue
        email = str(row.get("email") or "").strip().lower()
        bucket = str(row.get("bucket") or "").strip()
        if not bucket:
            continue
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if email:
            inventory_bucket_by_email[email] = bucket

    historical_success_in_usable_active_total = sum(1 for email in historical_success_emails if inventory_bucket_by_email.get(email) == "usable_active")
    historical_success_non_active_total = sum(
        1
        for email in historical_success_emails
        if email in inventory_bucket_by_email and inventory_bucket_by_email.get(email) != "usable_active"
    )
    historical_success_missing_from_inventory_total = sum(1 for email in historical_success_emails if email not in inventory_bucket_by_email)
    tools_root = source_root.parent
    retry_candidate_total = bucket_counts.get("retry_candidate", 0)
    guarded_actions = [
        {
            "id": "acceptance_replay_success20",
            "label": "验收回放 / 历史成功20",
            "risk_class": "LIVE_GUARD_ONLY",
            "description": "先用历史成功快照重放 acceptance，不直接信当前 live 管理面。",
            "command": f"cd {tools_root} && python3 check_five_account_acceptance.py --csv data/registered_success_active20.csv --target-count 20 --store data/pipeline_store.sqlite3 --pretty",
        },
        {
            "id": "active_target_sync_success20",
            "label": "重建 active 子集 / success20",
            "risk_class": "LIVE_GUARD_ONLY",
            "description": "把历史 success20 重新推送到 CPA 并同步入库。",
            "command": f"cd {tools_root} && python3 run_active_target_sync.py --csv data/registered_success_active20.csv --output-csv data/active_target_accounts.csv --store data/pipeline_store.sqlite3 --target-count 20 --pretty",
        },
        {
            "id": "retry_candidate_pipeline",
            "label": "重试 retry_candidate 队列",
            "risk_class": "LIVE_GUARD_ONLY",
            "description": "只对 review queue 中 retry_candidate 运行受控 OAuth 重挂载。",
            "command": f"cd {tools_root} && python3 run_retry_candidate_pipeline.py --csv registered_accounts.csv --review-queue data/cpa_non_active_review_queue.csv --output-csv data/active_target_accounts.csv --store data/pipeline_store.sqlite3 --max-attempts {max(retry_candidate_total, 1)} --pretty",
        },
    ]

    payload: dict[str, Any] = {
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "cpa_control_plane_snapshot",
        "change_class": "LIVE_GUARD_ONLY",
        "ok": bool(success_rows or inventory_rows or kernel_map),
        "status": "ok" if success_rows or inventory_rows or kernel_map else "blocked_missing_sources",
        "source_root": str(source_root),
        "summary": {
            "historical_success_total": len(success_rows),
            "new_unmounted_total": len(new_unmounted_rows),
            "inventory_total": len(inventory_rows),
            "review_queue_total": len(review_queue_rows),
            "no_retry_deactivated_total": len(deactivated_rows),
            "retry_candidate_total": bucket_counts.get("retry_candidate", 0),
            "blocked_about_you_total": bucket_counts.get("blocked_about_you", 0),
            "active_target_authfiles": int(acceptance_map.get("active_target_authfiles") or 0),
            "registered_valid_accounts": int(acceptance_map.get("registered_valid_accounts") or 0),
            "latest_kernel_accounts_total": int(kernel_map.get("accounts_total") or 0),
            "historical_success_in_usable_active_total": historical_success_in_usable_active_total,
            "historical_success_non_active_total": historical_success_non_active_total,
            "historical_success_missing_from_inventory_total": historical_success_missing_from_inventory_total,
        },
        "bucket_counts": bucket_counts,
        "historical_success_emails": historical_success_emails,
        "new_unmounted_emails": [str(row.get("email") or "").strip() for row in new_unmounted_rows if str(row.get("email") or "").strip()],
        "review_queue_preview": review_queue_rows[:10],
        "deactivated_preview": deactivated_rows[:10],
        "acceptance": acceptance_map,
        "latest_kernel_run_id": str(kernel_map.get("run_id") or ""),
        "guarded_actions": guarded_actions,
    }

    artifact_json, artifact_md = write_review_artifacts(review_dir, payload, stamp)
    public_data_dir = public_dir / "data"
    public_data_dir.mkdir(parents=True, exist_ok=True)
    public_path = public_data_dir / "cpa_control_plane_snapshot.json"
    public_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload["artifact_json"] = str(artifact_json)
    payload["artifact_md"] = str(artifact_md)
    payload["public_path"] = str(public_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-owned CPA control plane snapshot from handoff artifacts.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--public-dir", default="")
    parser.add_argument("--source-root", default=str(DEFAULT_MAC_TOOLS_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    public_dir = (
        Path(args.public_dir).expanduser().resolve()
        if str(args.public_dir).strip()
        else workspace_to_system_root(workspace) / "dashboard" / "web" / "public"
    )
    payload = build_snapshot(
        workspace=workspace,
        public_dir=public_dir,
        source_root=Path(args.source_root).expanduser().resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
