#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_POLL_SECONDS = 60
DEFAULT_GUARD_TIMEOUT_SECONDS = 45
DEFAULT_HISTORY_LIMIT = 12


def _load_guard_module():
    script_dir = Path(__file__).resolve().parent
    mod_path = script_dir / "live_risk_guard.py"
    spec = importlib.util.spec_from_file_location("live_risk_guard", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load risk guard module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LG = _load_guard_module()
resolve_path = LG.resolve_path
run_json_command = LG.run_json_command
sha256_file = LG.sha256_file
write_json = LG.write_json


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def pid_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def build_guard_cmd(*, system_root: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(system_root / "scripts" / "live_risk_guard.py"),
        "--config",
        str(resolve_path(args.config, anchor=system_root)),
        "--output-root",
        str(resolve_path(args.output_root, anchor=system_root)),
        "--review-dir",
        str(resolve_path(args.review_dir, anchor=system_root)),
        "--ticket-freshness-seconds",
        str(max(1, int(args.ticket_freshness_seconds))),
        "--panic-cooldown-seconds",
        str(max(1, int(args.panic_cooldown_seconds))),
        "--max-daily-loss-ratio",
        f"{float(args.max_daily_loss_ratio):.6f}",
        "--max-open-exposure-ratio",
        f"{float(args.max_open_exposure_ratio):.6f}",
        "--refresh-tickets",
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
    return cmd


def summarize_guard_result(result: dict[str, Any], *, started_at_utc: str, finished_at_utc: str) -> dict[str, Any]:
    payload = result.get("payload", {}) if isinstance(result.get("payload", {}), dict) else {}
    reasons = payload.get("reasons", []) if isinstance(payload.get("reasons", []), list) else []
    ticket_refresh = payload.get("ticket_refresh", {}) if isinstance(payload.get("ticket_refresh", {}), dict) else {}
    return {
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "returncode": int(result.get("returncode", 0)),
        "timeout": bool(result.get("timeout", False)),
        "status": str(payload.get("status", "unknown")),
        "allowed": bool(payload.get("allowed", False)),
        "artifact": str(payload.get("artifact", "")),
        "fuse_path": str(payload.get("fuse_path", "")),
        "reasons": [str(x) for x in reasons[:8]],
        "ticket_refresh_returncode": int(ticket_refresh.get("returncode", 0) or 0),
        "ticket_refresh_artifact": str((ticket_refresh.get("payload", {}) if isinstance(ticket_refresh.get("payload", {}), dict) else {}).get("json", "")),
    }


def bounded_history(rows: list[dict[str, Any]], row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    out = list(rows)
    out.append(row)
    return out[-max(1, int(limit)) :]


def write_state(*, state_path: Path, payload: dict[str, Any]) -> Path:
    checksum_path = state_path.with_name(f"{state_path.stem}_checksum.json")
    payload["updated_at_utc"] = now_utc_iso()
    write_json(state_path, payload)
    digest, size_bytes = sha256_file(state_path)
    write_json(
        checksum_path,
        {
            "generated_at_utc": now_utc_iso(),
            "files": [{"path": str(state_path), "sha256": digest, "size_bytes": int(size_bytes)}],
        },
    )
    return checksum_path


def install_signal_handlers(stop_state: dict[str, Any]) -> None:
    def _handler(signum, frame) -> None:
        _ = frame
        stop_state["requested"] = True
        try:
            sig_name = signal.Signals(signum).name.lower()
        except Exception:
            sig_name = str(signum)
        stop_state["reason"] = f"signal_{sig_name}"

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> int:
    system_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a lightweight live risk daemon that keeps the fuse fresh.")
    parser.add_argument("--date", default="")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--review-dir", default="output/review")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--guard-timeout-seconds", type=int, default=DEFAULT_GUARD_TIMEOUT_SECONDS)
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--ticket-freshness-seconds", type=int, default=LG.DEFAULT_TICKET_FRESHNESS_SECONDS)
    parser.add_argument("--panic-cooldown-seconds", type=int, default=LG.DEFAULT_PANIC_COOLDOWN_SECONDS)
    parser.add_argument("--max-daily-loss-ratio", type=float, default=LG.DEFAULT_MAX_DAILY_LOSS_RATIO)
    parser.add_argument("--max-open-exposure-ratio", type=float, default=LG.DEFAULT_MAX_OPEN_EXPOSURE_RATIO)
    parser.add_argument("--ticket-symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSD")
    parser.add_argument("--ticket-equity-usdt", type=float, default=0.0)
    parser.add_argument("--ticket-min-confidence", type=float, default=None)
    parser.add_argument("--ticket-min-convexity", type=float, default=None)
    parser.add_argument("--ticket-max-age-days", type=int, default=14)
    parser.add_argument("--state-path", default="")
    args = parser.parse_args()

    output_root = resolve_path(args.output_root, anchor=system_root)
    review_dir = resolve_path(args.review_dir, anchor=system_root)
    state_path = resolve_path(args.state_path, anchor=system_root) if str(args.state_path).strip() else (output_root / "state" / "live_risk_daemon.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    stop_state: dict[str, Any] = {"requested": False, "reason": ""}
    install_signal_handlers(stop_state)

    payload: dict[str, Any] = {
        "generated_at_utc": now_utc_iso(),
        "action": "live_risk_daemon",
        "status": "running",
        "running": True,
        "pid": int(os.getpid()),
        "workspace": str(system_root.parent),
        "system_root": str(system_root),
        "config": str(resolve_path(args.config, anchor=system_root)),
        "output_root": str(output_root),
        "review_dir": str(review_dir),
        "poll_seconds": int(max(1, int(args.poll_seconds))),
        "max_cycles": int(max(0, int(args.max_cycles))),
        "guard_timeout_seconds": int(max(5, int(args.guard_timeout_seconds))),
        "history_limit": int(max(1, int(args.history_limit))),
        "cycles_completed": 0,
        "recent_cycles": [],
        "last_guard": {},
        "next_run_at_utc": now_utc_iso(),
        "stop_reason": "",
        "governance": {
            "bounded_history_limit": int(max(1, int(args.history_limit))),
            "bounded_state_files": 2,
        },
    }

    try:
        checksum_path = write_state(state_path=state_path, payload=payload)
        while True:
            started_at_utc = now_utc_iso()
            guard_result = run_json_command(
                cmd=build_guard_cmd(system_root=system_root, args=args),
                cwd=system_root,
                timeout_seconds=max(5.0, float(args.guard_timeout_seconds)),
            )
            finished_at_utc = now_utc_iso()
            summary = summarize_guard_result(guard_result, started_at_utc=started_at_utc, finished_at_utc=finished_at_utc)
            payload["cycles_completed"] = int(payload.get("cycles_completed", 0)) + 1
            payload["last_guard"] = summary
            payload["recent_cycles"] = bounded_history(
                payload.get("recent_cycles", []) if isinstance(payload.get("recent_cycles", []), list) else [],
                summary,
                limit=max(1, int(args.history_limit)),
            )
            payload["last_cycle_started_at_utc"] = started_at_utc
            payload["last_cycle_finished_at_utc"] = finished_at_utc

            if bool(stop_state.get("requested", False)):
                payload["stop_reason"] = str(stop_state.get("reason", "signal_requested"))
                break
            if int(args.max_cycles) > 0 and int(payload.get("cycles_completed", 0)) >= int(args.max_cycles):
                payload["stop_reason"] = "max_cycles_reached"
                break

            next_run = now_utc() + timedelta(seconds=max(1, int(args.poll_seconds)))
            payload["next_run_at_utc"] = next_run.strftime("%Y-%m-%dT%H:%M:%SZ")
            checksum_path = write_state(state_path=state_path, payload=payload)
            while now_utc() < next_run:
                if bool(stop_state.get("requested", False)):
                    payload["stop_reason"] = str(stop_state.get("reason", "signal_requested"))
                    break
                time.sleep(min(1.0, max(0.05, (next_run - now_utc()).total_seconds())))
            if bool(payload.get("stop_reason", "")):
                break

        payload["status"] = "stopped"
        payload["running"] = False
        payload["next_run_at_utc"] = None
        payload["stopped_at_utc"] = now_utc_iso()
        checksum_path = write_state(state_path=state_path, payload=payload)
        out = dict(payload)
        out["state_path"] = str(state_path)
        out["checksum"] = str(checksum_path)
        out["pid_alive"] = bool(payload.get("running", False)) and bool(pid_is_alive(int(payload.get("pid", 0))))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        error_payload = dict(payload)
        error_payload["status"] = "error"
        error_payload["running"] = False
        error_payload["stop_reason"] = error_payload.get("stop_reason", "") or "daemon_error"
        error_payload["error"] = str(exc)
        error_payload["stopped_at_utc"] = now_utc_iso()
        checksum_path = write_state(state_path=state_path, payload=error_payload)
        out = dict(error_payload)
        out["state_path"] = str(state_path)
        out["checksum"] = str(checksum_path)
        out["pid_alive"] = False
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
