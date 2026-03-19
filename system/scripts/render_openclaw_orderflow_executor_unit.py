#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def build_exec_start(args: argparse.Namespace) -> str:
    cmd = [
        "/usr/bin/env",
        "PYTHONPATH=src",
        "/usr/bin/python3",
        "scripts/openclaw_orderflow_executor.py",
        "--config",
        "config.yaml",
        "--output-root",
        "output",
        "--review-dir",
        "output/review",
        "--poll-seconds",
        str(max(1, int(args.poll_seconds))),
        "--executor-timeout-seconds",
        str(max(5, int(args.executor_timeout_seconds))),
        "--mode",
        str(args.mode).strip() or "shadow_guarded",
    ]
    if args.max_loops is not None:
        cmd.extend(["--max-loops", str(max(0, int(args.max_loops)))])
    return shlex.join(cmd)


def render_unit(args: argparse.Namespace) -> str:
    raw_workdir = Path(str(args.project_dir)).expanduser()
    workdir = raw_workdir if raw_workdir.is_absolute() else raw_workdir.resolve()
    log_dir = workdir / "output" / "logs"
    state_dir = workdir / "output" / "state"
    review_dir = workdir / "output" / "review"
    artifacts_dir = workdir / "output" / "artifacts"
    exec_start = build_exec_start(args)
    lines = [
        "[Unit]",
        "Description=Fenlie OpenClaw Orderflow Executor",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"User={str(args.user).strip()}",
        f"WorkingDirectory={workdir}",
        "Environment=PYTHONUNBUFFERED=1",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        f"ExecStart={exec_start}",
        "Restart=always",
        "RestartSec=5",
        "KillSignal=SIGTERM",
        "TimeoutStopSec=5",
        "UMask=0077",
        "RemoveIPC=true",
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "PrivateDevices=true",
        "DevicePolicy=closed",
        "ProtectSystem=strict",
        "ProtectHostname=true",
        "ProtectControlGroups=true",
        "ProtectKernelTunables=true",
        "ProtectKernelModules=true",
        "ProtectKernelLogs=true",
        "ProtectClock=true",
        "ProtectProc=invisible",
        "ProtectHome=read-only",
        "ProcSubset=pid",
        "PrivateUsers=true",
        "PrivateNetwork=true",
        "IPAddressDeny=any",
        "RestrictNamespaces=true",
        "RestrictSUIDSGID=true",
        "RestrictRealtime=true",
        "LockPersonality=true",
        "SystemCallArchitectures=native",
        "SystemCallFilter=@system-service",
        "SystemCallFilter=~@resources",
        "SystemCallFilter=~@privileged",
        "MemoryDenyWriteExecute=true",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "RestrictAddressFamilies=AF_UNIX",
        f"ReadWritePaths={log_dir} {state_dir} {review_dir} {artifacts_dir}",
        "StandardOutput=journal",
        "StandardError=journal",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a systemd unit for the OpenClaw orderflow executor.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--executor-timeout-seconds", type=int, default=5)
    parser.add_argument("--mode", default="shadow_guarded")
    parser.add_argument("--max-loops", type=int, default=0)
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()
    text = render_unit(args)
    output_path = str(args.output_path).strip()
    if output_path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
