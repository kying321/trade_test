#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


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


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# External Intelligence Refresh",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- status: `{payload['status']}`",
        f"- ok: `{str(bool(payload['ok'])).lower()}`",
        f"- change_class: `{payload['change_class']}`",
        f"- stamp: `{payload['stamp']}`",
        f"- jin10_status: `{payload['jin10_status']}`",
        f"- axios_status: `{payload['axios_status']}`",
        f"- external_status: `{payload['external_status']}`",
        f"- recommended_brief: `{payload.get('recommended_brief') or '-'}`",
        f"- takeaway: `{payload.get('takeaway') or '-'}`",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(review_dir: Path, payload: dict[str, Any], stamp: str) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"{stamp}_external_intelligence_refresh.json"
    md_path = review_dir / f"{stamp}_external_intelligence_refresh.md"
    latest_json = review_dir / "latest_external_intelligence_refresh.json"
    latest_md = review_dir / "latest_external_intelligence_refresh.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def build_summary(
    *,
    workspace: Path,
    public_dir: Path,
    runtime_now: dt.datetime,
    snapshot_skipped: bool,
    jin10_payload: dict[str, Any],
    axios_payload: dict[str, Any],
    external_payload: dict[str, Any],
    snapshot_payload: dict[str, Any],
) -> dict[str, Any]:
    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    dashboard_outputs: list[Any] = []
    for item in list(snapshot_payload.get("outputs") or []):
        if isinstance(item, dict):
            dashboard_outputs.append(item)
            continue
        text = str(item).strip()
        if text:
            dashboard_outputs.append(text)
    jin10_status = str(jin10_payload.get("status") or "")
    axios_status = str(axios_payload.get("status") or "")
    external_status = str(external_payload.get("status") or "")

    if snapshot_skipped:
        fully_ready = jin10_status == "ok" and axios_status == "ok" and external_status == "ok"
        partially_ready = external_status in {"ok", "partial"}
    else:
        fully_ready = (
            jin10_status == "ok"
            and axios_status == "ok"
            and external_status == "ok"
            and bool(dashboard_outputs)
        )
        partially_ready = bool(dashboard_outputs) and external_status in {"ok", "partial"}
    status = "ok" if fully_ready else "partial" if partially_ready else "blocked"

    return {
        "generated_at_utc": fmt_utc(runtime_now),
        "mode": "external_intelligence_refresh",
        "change_class": "RESEARCH_ONLY",
        "ok": partially_ready,
        "status": status,
        "stamp": stamp,
        "snapshot_skipped": bool(snapshot_skipped),
        "workspace": str(workspace),
        "public_dir": str(public_dir),
        "refresh_order": [
            "run_jin10_mcp_snapshot",
            "run_axios_site_snapshot",
            "run_external_intelligence_snapshot",
        ]
        + ([] if snapshot_skipped else ["build_dashboard_frontend_snapshot"]),
        "jin10_status": jin10_status,
        "jin10_path": str(jin10_payload.get("artifact_json") or ""),
        "axios_status": axios_status,
        "axios_path": str(axios_payload.get("artifact_json") or ""),
        "external_status": external_status,
        "external_intelligence_path": str(external_payload.get("artifact_json") or ""),
        "dashboard_outputs": dashboard_outputs,
        "recommended_brief": str(external_payload.get("recommended_brief") or ""),
        "takeaway": str(external_payload.get("takeaway") or ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external intelligence refresh chain and rebuild dashboard snapshot.")
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--public-dir", default="", help="Dashboard public dir; defaults to system/dashboard/web/public.")
    parser.add_argument("--now", help="Explicit UTC timestamp for deterministic artifact stamping.")
    parser.add_argument("--jin10-token-env", default="JIN10_MCP_BEARER_TOKEN")
    parser.add_argument("--axios-limit", type=int, default=20)
    parser.add_argument("--skip-dashboard-snapshot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    review_dir = system_root / "output" / "review"
    public_dir = (
        Path(args.public_dir).expanduser().resolve()
        if str(args.public_dir).strip()
        else system_root / "dashboard" / "web" / "public"
    )
    runtime_now = parse_now(args.now)

    jin10_payload = run_json(
        name="run_jin10_mcp_snapshot",
        cmd=[
            sys.executable,
            str(system_root / "scripts" / "run_jin10_mcp_snapshot.py"),
            "--workspace",
            str(workspace),
            "--token-env",
            str(args.jin10_token_env),
        ],
    )
    axios_payload = run_json(
        name="run_axios_site_snapshot",
        cmd=[
            sys.executable,
            str(system_root / "scripts" / "run_axios_site_snapshot.py"),
            "--workspace",
            str(workspace),
            "--limit",
            str(int(args.axios_limit)),
        ],
    )
    external_payload = run_json(
        name="run_external_intelligence_snapshot",
        cmd=[
            sys.executable,
            str(system_root / "scripts" / "run_external_intelligence_snapshot.py"),
            "--workspace",
            str(workspace),
        ],
    )
    snapshot_payload: dict[str, Any] = {}
    if not bool(args.skip_dashboard_snapshot):
        snapshot_payload = run_json(
            name="build_dashboard_frontend_snapshot",
            cmd=[
                sys.executable,
                str(system_root / "scripts" / "build_dashboard_frontend_snapshot.py"),
                "--workspace",
                str(workspace),
                "--public-dir",
                str(public_dir),
            ],
        )

    payload = build_summary(
        workspace=workspace,
        public_dir=public_dir,
        runtime_now=runtime_now,
        snapshot_skipped=bool(args.skip_dashboard_snapshot),
        jin10_payload=jin10_payload,
        axios_payload=axios_payload,
        external_payload=external_payload,
        snapshot_payload=snapshot_payload,
    )
    artifact_json, artifact_md = write_artifacts(review_dir, payload, payload["stamp"])
    payload["artifact_json"] = str(artifact_json)
    payload["artifact_md"] = str(artifact_md)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
