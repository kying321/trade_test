#!/usr/bin/env python3
"""Ensure only one OpenClaw gateway LaunchAgent is active.

Default policy:
- prefer `ai.openclaw.bot` (wrapper-aware launch path)
- if both `ai.openclaw.bot` and `ai.openclaw.gateway` are loaded, boot out the non-preferred one

This script is intentionally local/offline and returns a compact JSON payload.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


KNOWN_LABELS = ("ai.openclaw.bot", "ai.openclaw.gateway")


@dataclass
class CmdResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: List[str]) -> CmdResult:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return CmdResult(
        cmd=cmd,
        returncode=int(p.returncode),
        stdout=(p.stdout or ""),
        stderr=(p.stderr or ""),
    )


def _parse_launchctl_list(text: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw in (text or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(None, 2)
        if len(parts) != 3:
            continue
        pid_raw, status_raw, label = parts
        if label not in KNOWN_LABELS:
            continue
        pid: Optional[int] = None
        if pid_raw.isdigit():
            pid = int(pid_raw)
        out[label] = {
            "pid": pid,
            "status_raw": status_raw,
            "raw": raw,
        }
    return out


def _launchctl_state() -> Dict[str, Dict[str, Any]]:
    r = _run(["launchctl", "list"])
    if r.returncode != 0:
        return {}
    return _parse_launchctl_list(r.stdout)


def _resolve_lsof() -> Optional[str]:
    from_path = shutil.which("lsof")
    if from_path:
        return from_path
    for candidate in ("/usr/sbin/lsof", "/sbin/lsof", "/usr/bin/lsof"):
        p = Path(candidate)
        if p.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _listeners(port: int) -> List[int]:
    lsof_bin = _resolve_lsof()
    if not lsof_bin:
        return []
    r = _run([lsof_bin, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    if r.returncode != 0:
        return []
    out: List[int] = []
    for raw in (r.stdout or "").splitlines():
        raw = raw.strip()
        if raw.isdigit():
            out.append(int(raw))
    return sorted(set(out))


def _bootout(uid: int, label: str, dry_run: bool) -> Dict[str, Any]:
    cmd = ["launchctl", "bootout", f"gui/{uid}/{label}"]
    if dry_run:
        return {"action": "bootout", "label": label, "dry_run": True, "returncode": 0}
    r = _run(cmd)
    return {
        "action": "bootout",
        "label": label,
        "dry_run": False,
        "returncode": int(r.returncode),
        "stderr_tail": (r.stderr or "")[-200:],
    }


def _snapshot(port: int) -> Dict[str, Any]:
    st = _launchctl_state()
    loaded = sorted(st.keys())
    return {
        "loaded_labels": loaded,
        "state": st,
        "listener_pids": _listeners(port),
    }


def _evaluate(preferred: str, snap: Dict[str, Any]) -> Dict[str, Any]:
    loaded = list(snap.get("loaded_labels") or [])
    preferred_loaded = preferred in loaded
    if len(loaded) == 0:
        return {"status": "degraded", "reason": "no_gateway_service_loaded"}
    if len(loaded) > 1:
        return {"status": "degraded", "reason": "multi_gateway_services_loaded"}
    if not preferred_loaded:
        return {"status": "degraded", "reason": "non_preferred_gateway_loaded"}
    return {"status": "ok", "reason": "single_preferred_gateway_loaded"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preferred", default=os.getenv("PI_PREFERRED_GATEWAY_LABEL", "ai.openclaw.gateway"))
    ap.add_argument("--port", type=int, default=18789)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    preferred = str(args.preferred)
    if preferred not in KNOWN_LABELS:
        preferred = "ai.openclaw.bot"
    uid = os.getuid()

    pre = _snapshot(port=int(args.port))
    actions: List[Dict[str, Any]] = []
    loaded = list(pre.get("loaded_labels") or [])

    if len(loaded) > 1:
        for label in loaded:
            if label == preferred:
                continue
            actions.append(_bootout(uid=uid, label=label, dry_run=bool(args.dry_run)))

    post = _snapshot(port=int(args.port)) if not args.dry_run else pre
    eva = _evaluate(preferred=preferred, snap=post)

    payload = {
        "envelope_version": "1.0",
        "domain": "gateway_singleton_guard",
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preferred": preferred,
        "port": int(args.port),
        "dry_run": bool(args.dry_run),
        "pre": pre,
        "actions": actions,
        "post": post,
        "status": eva["status"],
        "reason": eva["reason"],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
