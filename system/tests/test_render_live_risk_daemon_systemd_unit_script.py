from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "render_live_risk_daemon_systemd_unit.py"
    spec = importlib.util.spec_from_file_location("render_live_risk_daemon_systemd_unit", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_render_live_risk_daemon_unit_contains_expected_execstart() -> None:
    mod = _load_module()
    args = mod.argparse.Namespace(
        project_dir="/home/ubuntu/openclaw-system",
        user="ubuntu",
        date="",
        poll_seconds=60,
        guard_timeout_seconds=45,
        history_limit=12,
        ticket_freshness_seconds=900,
        panic_cooldown_seconds=1800,
        max_daily_loss_ratio=0.05,
        max_open_exposure_ratio=0.50,
        ticket_symbols="BTCUSDT,ETHUSDT",
        ticket_equity_usdt=0.0,
        ticket_min_confidence=None,
        ticket_min_convexity=None,
        ticket_max_age_days=14,
    )
    text = mod.render_unit(args)
    assert "Description=Fenlie Live Risk Daemon" in text
    assert "User=ubuntu" in text
    assert "WorkingDirectory=/home/ubuntu/openclaw-system" in text
    assert "scripts/live_risk_daemon.py" in text
    assert "--ticket-symbols BTCUSDT,ETHUSDT" in text
    assert "--ticket-max-age-days 14" in text
    assert "Restart=always" in text
    assert "Environment=PYTHONDONTWRITEBYTECODE=1" in text
    assert "ProtectSystem=strict" in text
    assert "PrivateTmp=true" in text
    assert "DevicePolicy=closed" in text
    assert "MemoryDenyWriteExecute=true" in text
    assert "RestrictAddressFamilies=AF_UNIX" in text
    assert "CapabilityBoundingSet=" in text
    assert "AmbientCapabilities=" in text
    assert "ProtectHostname=true" in text
    assert "ProtectProc=invisible" in text
    assert "ProtectHome=read-only" in text
    assert "ProcSubset=pid" in text
    assert "PrivateUsers=true" in text
    assert "PrivateNetwork=true" in text
    assert "IPAddressDeny=any" in text
    assert "RestrictNamespaces=true" in text
    assert "RemoveIPC=true" in text
    assert "SystemCallFilter=@system-service" in text
    assert "SystemCallFilter=~@resources" in text
    assert "SystemCallFilter=~@privileged" in text


def test_render_live_risk_daemon_unit_includes_optional_thresholds() -> None:
    mod = _load_module()
    old_argv = sys.argv
    sys.argv = [
        "render_live_risk_daemon_systemd_unit.py",
        "--project-dir",
        "/home/ubuntu/openclaw-system",
        "--user",
        "ubuntu",
        "--date",
        "2026-03-06",
        "--ticket-min-confidence",
        "61",
        "--ticket-min-convexity",
        "3.5",
        "--ticket-equity-usdt",
        "20",
    ]
    try:
        parser = mod.argparse.ArgumentParser()
    finally:
        sys.argv = old_argv
    _ = parser

    args = mod.argparse.Namespace(
        project_dir="/home/ubuntu/openclaw-system",
        user="ubuntu",
        date="2026-03-06",
        poll_seconds=60,
        guard_timeout_seconds=45,
        history_limit=12,
        ticket_freshness_seconds=900,
        panic_cooldown_seconds=1800,
        max_daily_loss_ratio=0.05,
        max_open_exposure_ratio=0.50,
        ticket_symbols="BTCUSDT,ETHUSDT",
        ticket_equity_usdt=20.0,
        ticket_min_confidence=61.0,
        ticket_min_convexity=3.5,
        ticket_max_age_days=14,
    )
    exec_start = mod.build_exec_start(args)
    assert "--date 2026-03-06" in exec_start
    assert "--ticket-min-confidence 61.0" in exec_start
    assert "--ticket-min-convexity 3.5" in exec_start
    assert "--ticket-equity-usdt 20.00000000" in exec_start


def test_render_live_risk_daemon_unit_whitelists_only_runtime_output_paths() -> None:
    mod = _load_module()
    args = mod.argparse.Namespace(
        project_dir="/home/ubuntu/openclaw-system",
        user="ubuntu",
        date="",
        poll_seconds=60,
        guard_timeout_seconds=45,
        history_limit=12,
        ticket_freshness_seconds=900,
        panic_cooldown_seconds=1800,
        max_daily_loss_ratio=0.05,
        max_open_exposure_ratio=0.50,
        ticket_symbols="BTCUSDT,ETHUSDT",
        ticket_equity_usdt=0.0,
        ticket_min_confidence=None,
        ticket_min_convexity=None,
        ticket_max_age_days=14,
    )
    text = mod.render_unit(args)
    assert (
        "ReadWritePaths=/home/ubuntu/openclaw-system/output/logs "
        "/home/ubuntu/openclaw-system/output/state "
        "/home/ubuntu/openclaw-system/output/review "
        "/home/ubuntu/openclaw-system/output/artifacts"
    ) in text
    assert "RestrictAddressFamilies=AF_UNIX" in text
    assert "ProtectKernelTunables=true" in text
    assert "ProtectKernelModules=true" in text
    assert "ProtectKernelLogs=true" in text
    assert "ProtectClock=true" in text
    assert "DevicePolicy=closed" in text
    assert "LockPersonality=true" in text
    assert "MemoryDenyWriteExecute=true" in text
    assert "ProtectHostname=true" in text
    assert "ProtectProc=invisible" in text
    assert "ProtectHome=read-only" in text
    assert "ProcSubset=pid" in text
    assert "PrivateUsers=true" in text
    assert "PrivateNetwork=true" in text
    assert "IPAddressDeny=any" in text
    assert "RestrictNamespaces=true" in text
    assert "RemoveIPC=true" in text
    assert "SystemCallFilter=@system-service" in text
    assert "SystemCallFilter=~@resources" in text
    assert "SystemCallFilter=~@privileged" in text
