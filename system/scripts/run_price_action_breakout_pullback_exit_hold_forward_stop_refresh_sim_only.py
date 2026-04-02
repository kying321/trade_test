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


def text(value: Any) -> str:
    return str(value or "").strip()


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot_resolve_system_root:{workspace}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the SIM_ONLY hold forward stop-condition artifact from the latest source-owned hold evidence chain."
    )
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--review-dir", default="", help="Optional explicit review directory override.")
    parser.add_argument("--forward-capacity-path", default="")
    parser.add_argument("--overlap-sidecar-path", default="")
    parser.add_argument("--handoff-path", default="")
    parser.add_argument("--window-consensus-path", default="")
    parser.add_argument("--now", help="Explicit UTC timestamp used to derive the builder stamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    review_dir = Path(args.review_dir).expanduser().resolve() if text(args.review_dir) else system_root / "output" / "review"
    stamp = fmt_stamp(parse_now(args.now))

    forward_capacity_path = (
        Path(args.forward_capacity_path).expanduser().resolve()
        if text(args.forward_capacity_path)
        else require_path(
            review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json",
            "missing_exit_hold_forward_window_capacity_latest",
        )
    )
    overlap_sidecar_path = (
        Path(args.overlap_sidecar_path).expanduser().resolve()
        if text(args.overlap_sidecar_path)
        else require_path(
            review_dir / "latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json",
            "missing_exit_hold_overlap_sidecar_latest",
        )
    )
    handoff_path = (
        Path(args.handoff_path).expanduser().resolve()
        if text(args.handoff_path)
        else require_path(
            review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
            "missing_hold_selection_handoff_latest",
        )
    )
    window_consensus_path = (
        Path(args.window_consensus_path).expanduser().resolve()
        if text(args.window_consensus_path)
        else require_path(
            review_dir / "latest_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.json",
            "missing_exit_hold_window_consensus_latest",
        )
    )

    payload = run_json(
        name="build_exit_hold_forward_stop",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py"),
            "--forward-capacity-path",
            str(forward_capacity_path),
            "--overlap-sidecar-path",
            str(overlap_sidecar_path),
            "--handoff-path",
            str(handoff_path),
            "--window-consensus-path",
            str(window_consensus_path),
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
                "mode": "exit_hold_forward_stop_refresh_sim_only",
                "change_class": "SIM_ONLY",
                "stamp": stamp,
                "workspace": str(workspace),
                "review_dir": str(review_dir),
                "forward_capacity_path": str(forward_capacity_path),
                "overlap_sidecar_path": str(overlap_sidecar_path),
                "handoff_path": str(handoff_path),
                "window_consensus_path": str(window_consensus_path),
                "json_path": payload.get("json_path"),
                "latest_json_path": payload.get("latest_json_path"),
                "research_decision": payload.get("research_decision"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
