#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
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


def fmt_stamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot_resolve_system_root:{workspace}")


def sort_key(path: Path) -> tuple[str, float, str]:
    return (path.name, path.stat().st_mtime, path.name)


def latest_exit_risk_artifact(review_dir: Path) -> Path:
    candidates = [
        path
        for path in review_dir.glob("*_price_action_breakout_pullback_exit_risk_sim_only.json")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError("no_exit_risk_sim_only_artifact_found")
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def require_path(path: Path, code: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(code)
    return path


def current_python_executable() -> str:
    return sys.executable or "python3"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the SIM_ONLY ETH exit/risk canonical handoff from the latest source-owned review artifacts."
    )
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--review-dir", default="", help="Optional explicit review directory override.")
    parser.add_argument("--now", help="Explicit UTC timestamp used to derive the builder stamp.")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    review_dir = Path(args.review_dir).expanduser().resolve() if str(args.review_dir).strip() else system_root / "output" / "review"
    stamp = fmt_stamp(parse_now(args.now))

    exit_risk_path = latest_exit_risk_artifact(review_dir)
    forward_blocker_path = require_path(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json",
        "missing_exit_risk_forward_blocker_latest",
    )
    forward_consensus_path = require_path(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json",
        "missing_exit_risk_forward_consensus_latest",
    )
    break_even_sidecar_path = require_path(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json",
        "missing_exit_risk_break_even_sidecar_latest",
    )
    tail_capacity_path = require_path(
        review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json",
        "missing_exit_risk_forward_tail_capacity_latest",
    )

    payload = run_json(
        name="build_exit_risk_handoff",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
            "--exit-risk-path",
            str(exit_risk_path),
            "--forward-blocker-path",
            str(forward_blocker_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--break-even-sidecar-path",
            str(break_even_sidecar_path),
            "--tail-capacity-path",
            str(tail_capacity_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            stamp,
        ],
    )

    print(
        json.dumps(
            {
                "ok": True,
                "mode": "exit_risk_handoff_refresh_sim_only",
                "change_class": "SIM_ONLY",
                "stamp": stamp,
                "workspace": str(workspace),
                "review_dir": str(review_dir),
                "exit_risk_path": str(exit_risk_path),
                "forward_blocker_path": str(forward_blocker_path),
                "forward_consensus_path": str(forward_consensus_path),
                "break_even_sidecar_path": str(break_even_sidecar_path),
                "tail_capacity_path": str(tail_capacity_path),
                "json_path": payload.get("json_path"),
                "latest_json_path": payload.get("latest_json_path"),
                "research_decision": payload.get("research_decision"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
