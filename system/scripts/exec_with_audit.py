#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one command and append execution event into output/logs/command_exec.ndjson"
    )
    parser.add_argument("--root", default=None, help="System root path (default: infer from script path)")
    parser.add_argument("--source", default="manual", help="Event source label, e.g. launchd/manual/ci")
    parser.add_argument("--tag", default="", help="Optional event tag")
    parser.add_argument(
        "--no-pass-through",
        action="store_true",
        help="Do not mirror child stdout/stderr to current terminal",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after '--', e.g. -- lie test-all --fast")
    return parser.parse_args()


def _normalize_command_tokens(tokens: list[str]) -> list[str]:
    out = list(tokens)
    if out and out[0] == "--":
        out = out[1:]
    return out


_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _split_env_prefix(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    env_map: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not _ENV_ASSIGNMENT_RE.match(token):
            break
        key, value = token.split("=", 1)
        env_map[key] = value
        idx += 1
    return env_map, tokens[idx:]


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main() -> None:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parents[1]
    logs_path = root / "output" / "logs" / "command_exec.ndjson"

    raw_tokens = _normalize_command_tokens([str(x) for x in args.command])
    env_prefix, cmd = _split_env_prefix(raw_tokens)
    if not cmd:
        print("ERROR: missing command. Use: exec_with_audit.py -- <your command>", file=sys.stderr)
        raise SystemExit(2)

    started_at = datetime.now().astimezone()
    child_env = os.environ.copy()
    if env_prefix:
        child_env.update(env_prefix)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=child_env)
    ended_at = datetime.now().astimezone()
    duration_ms = int((ended_at - started_at).total_seconds() * 1000)

    if not bool(args.no_pass_through):
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)

    event = {
        "timestamp": ended_at.isoformat(),
        "started_at": started_at.isoformat(),
        "duration_ms": duration_ms,
        "source": str(args.source).strip() or "manual",
        "tag": str(args.tag).strip(),
        "cwd": os.getcwd(),
        "command": shlex.join(cmd),
        "argv": cmd,
        "env_overrides": sorted(env_prefix.keys()),
        "returncode": int(proc.returncode),
        "ok": bool(proc.returncode == 0),
    }
    _append_event(logs_path, event)
    raise SystemExit(int(proc.returncode))


if __name__ == "__main__":
    main()
