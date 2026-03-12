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
        "scripts/live_risk_daemon.py",
        "--config",
        "config.yaml",
        "--output-root",
        "output",
        "--review-dir",
        "output/review",
        "--poll-seconds",
        str(max(1, int(args.poll_seconds))),
        "--guard-timeout-seconds",
        str(max(5, int(args.guard_timeout_seconds))),
        "--history-limit",
        str(max(1, int(args.history_limit))),
        "--ticket-freshness-seconds",
        str(max(1, int(args.ticket_freshness_seconds))),
        "--panic-cooldown-seconds",
        str(max(1, int(args.panic_cooldown_seconds))),
        "--max-daily-loss-ratio",
        f"{float(args.max_daily_loss_ratio):.6f}",
        "--max-open-exposure-ratio",
        f"{float(args.max_open_exposure_ratio):.6f}",
        "--ticket-symbols",
        str(args.ticket_symbols),
        "--ticket-max-age-days",
        str(max(1, int(args.ticket_max_age_days))),
    ]
    if str(args.date).strip():
        cmd.extend(["--date", str(args.date).strip()])
    if args.ticket_min_confidence is not None:
        cmd.extend(["--ticket-min-confidence", str(float(args.ticket_min_confidence))])
    if args.ticket_min_convexity is not None:
        cmd.extend(["--ticket-min-convexity", str(float(args.ticket_min_convexity))])
    if float(args.ticket_equity_usdt) > 0.0:
        cmd.extend(["--ticket-equity-usdt", f"{float(args.ticket_equity_usdt):.8f}"])
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
        "Description=Fenlie Live Risk Daemon",
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
    parser = argparse.ArgumentParser(description="Render a systemd unit for the Fenlie live risk daemon.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--date", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--guard-timeout-seconds", type=int, default=45)
    parser.add_argument("--history-limit", type=int, default=12)
    parser.add_argument("--ticket-freshness-seconds", type=int, default=900)
    parser.add_argument("--panic-cooldown-seconds", type=int, default=1800)
    parser.add_argument("--max-daily-loss-ratio", type=float, default=0.05)
    parser.add_argument("--max-open-exposure-ratio", type=float, default=0.50)
    parser.add_argument("--ticket-symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSD")
    parser.add_argument("--ticket-equity-usdt", type=float, default=0.0)
    parser.add_argument("--ticket-min-confidence", type=float, default=None)
    parser.add_argument("--ticket-min-convexity", type=float, default=None)
    parser.add_argument("--ticket-max-age-days", type=int, default=14)
    args = parser.parse_args()
    print(render_unit(args), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
