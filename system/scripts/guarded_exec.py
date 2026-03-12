#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_IDEMPOTENCY_TTL_SECONDS = 1800
DEFAULT_IDEMPOTENCY_MAX_ENTRIES = 500
DEFAULT_ARTIFACT_TTL_HOURS = 168


def _load_takeover_module():
    script_dir = Path(__file__).resolve().parent
    mod_path = script_dir / "binance_live_takeover.py"
    spec = importlib.util.spec_from_file_location("binance_live_takeover", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load takeover module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TAKEOVER = _load_takeover_module()
PanicTriggered = TAKEOVER.PanicTriggered
RunHalfhourMutex = TAKEOVER.RunHalfhourMutex
load_recent_risk_fuse = TAKEOVER.load_recent_risk_fuse
panic_close_all = TAKEOVER.panic_close_all
read_json = TAKEOVER.read_json
write_json = TAKEOVER.write_json


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc_compact() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def resolve_workspace(raw: str) -> tuple[Path, Path]:
    if str(raw).strip():
        base = Path(str(raw).strip()).expanduser()
        workspace = base.resolve() if base.exists() else base.absolute()
    else:
        workspace = Path(__file__).resolve().parents[1]
    if (workspace / "system").is_dir():
        return workspace, workspace / "system"
    if (workspace / "scripts").is_dir() and (workspace / "src").is_dir():
        return workspace.parent, workspace
    raise FileNotFoundError(f"unable to resolve system root from workspace: {workspace}")


def parse_utc_text(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
        except Exception:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def parse_json_payload(text: str) -> dict[str, Any]:
    clean = str(text or "").strip()
    if not clean:
        return {}
    try:
        payload = json.loads(clean)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    for line in reversed([x.strip() for x in clean.splitlines() if x.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def run_json_command(*, cmd: list[str], cwd: Path, timeout_seconds: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "timeout": True,
            "returncode": 124,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "payload": {},
        }
    except OSError as exc:
        return {
            "ok": False,
            "timeout": False,
            "returncode": 126,
            "stdout": "",
            "stderr": str(exc),
            "payload": {},
            "spawn_error": str(exc),
        }
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    payload = parse_json_payload(stdout)
    return {
        "ok": int(proc.returncode) == 0,
        "timeout": False,
        "returncode": int(proc.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "payload": payload,
    }


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


def evict_old_guard_artifacts(*, directory: Path, now_dt: datetime, ttl_hours: float, keep_names: set[str]) -> int:
    cutoff = now_dt.timestamp() - (float(ttl_hours) * 3600.0)
    removed = 0
    for pattern in ("*_trade_live_exec_guard.json", "*_trade_live_exec_guard_checksum.json"):
        for path in directory.glob(pattern):
            if path.name in keep_names:
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
            except Exception:
                continue
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
    return removed


def load_idempotency_entries(path: Path, *, ttl_seconds: int, now_dt: datetime) -> list[dict[str, Any]]:
    payload = read_json(path, {})
    rows = payload.get("entries", []) if isinstance(payload, dict) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key", "")).strip()
        ts = parse_utc_text(str(row.get("ts_utc", "")))
        if not key or ts is None:
            continue
        age = (now_dt - ts).total_seconds()
        if age > max(1, int(ttl_seconds)):
            continue
        out.append({"key": key, "ts_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "age_seconds": float(max(0.0, age))})
    return out


def apply_idempotency_guard(
    *,
    ledger_path: Path,
    key: str,
    ttl_seconds: int,
    max_entries: int,
) -> dict[str, Any]:
    now_dt = now_utc()
    rows = load_idempotency_entries(ledger_path, ttl_seconds=ttl_seconds, now_dt=now_dt)
    duplicate = any(str(row.get("key", "")) == key for row in rows)
    if not duplicate:
        rows.append({"key": key, "ts_utc": now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "age_seconds": 0.0})
    trimmed = rows[-max(1, int(max_entries)) :]
    write_json(
        ledger_path,
        {
            "updated_at_utc": now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entries": [{"key": str(row.get("key", "")), "ts_utc": str(row.get("ts_utc", ""))} for row in trimmed],
        },
    )
    return {
        "key": key,
        "duplicate": bool(duplicate),
        "ttl_seconds": int(max(1, int(ttl_seconds))),
        "path": str(ledger_path),
        "entries": int(len(trimmed)),
    }


def takeover_cmd(
    *,
    system_root: Path,
    args: argparse.Namespace,
    effective_mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(system_root / "scripts" / "binance_live_takeover.py"),
        "--config",
        str(system_root / "config.yaml"),
        "--output-root",
        str(system_root / "output"),
        "--market",
        str(args.market),
        "--rate-limit-per-minute",
        str(max(1, int(args.rate_limit_per_minute))),
        "--timeout-ms",
        str(min(5000, max(100, int(args.timeout_ms)))),
        "--canary-quote-usdt",
        f"{float(args.canary_quote_usdt):.4f}",
        "--max-drawdown",
        f"{float(args.max_drawdown):.6f}",
        "--trade-window-hours",
        str(max(1, int(args.trade_window_hours))),
        "--risk-fuse-max-age-seconds",
        str(max(1, int(args.risk_fuse_max_age_seconds))),
        "--activate-config",
        "--skip-mutex",
    ]
    if str(args.date).strip():
        cmd.extend(["--date", str(args.date).strip()])
    if bool(args.allow_daemon_env_fallback):
        cmd.append("--allow-daemon-env-fallback")
    if str(args.order_symbol).strip():
        cmd.extend(["--order-symbol", str(args.order_symbol).strip().upper()])
    if str(args.order_side).strip():
        cmd.extend(["--order-side", str(args.order_side).strip().upper()])
    if float(args.order_quantity) > 0.0:
        cmd.extend(["--order-quantity", f"{float(args.order_quantity):.8f}"])
    if effective_mode == "canary":
        cmd.append("--allow-live-order")
    return cmd


def risk_guard_cmd(*, system_root: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(system_root / "scripts" / "live_risk_guard.py"),
        "--config",
        str(system_root / "config.yaml"),
        "--output-root",
        str(system_root / "output"),
        "--review-dir",
        str(system_root / "output" / "review"),
        "--ticket-freshness-seconds",
        str(max(1, int(args.risk_fuse_max_age_seconds))),
        "--panic-cooldown-seconds",
        str(max(1, int(args.panic_cooldown_seconds))),
        "--max-daily-loss-ratio",
        f"{float(args.max_daily_loss_ratio):.6f}",
        "--max-open-exposure-ratio",
        f"{float(args.max_open_exposure_ratio):.6f}",
        "--skip-mutex",
    ]
    if bool(args.refresh_tickets):
        cmd.extend(
            [
                "--refresh-tickets",
                "--ticket-symbols",
                str(args.ticket_symbols),
                "--ticket-max-age-days",
                str(max(1, int(args.ticket_max_age_days))),
            ]
        )
        if str(args.date).strip():
            cmd.extend(["--date", str(args.date).strip()])
        if args.ticket_min_confidence is not None:
            cmd.extend(["--ticket-min-confidence", str(float(args.ticket_min_confidence))])
        if args.ticket_min_convexity is not None:
            cmd.extend(["--ticket-min-convexity", str(float(args.ticket_min_convexity))])
        if float(args.ticket_equity_usdt) > 0.0:
            cmd.extend(["--ticket-equity-usdt", f"{float(args.ticket_equity_usdt):.8f}"])
    return cmd


def cached_risk_guard_result(*, output_root: Path, max_age_seconds: int) -> dict[str, Any] | None:
    fuse = load_recent_risk_fuse(output_root=output_root, max_age_seconds=max_age_seconds)
    status = str(fuse.get("status", "")).strip().lower()
    if not bool(fuse.get("fresh", False)):
        return None
    if status in {"missing", "stale", "invalid"}:
        return None
    reasons = [str(x) for x in fuse.get("reasons", []) if str(x)]
    payload = {
        "generated_at_utc": now_utc_iso(),
        "action": "live_risk_guard",
        "allowed": bool(fuse.get("allowed", False)),
        "status": "pass" if bool(fuse.get("allowed", False)) else "blocked",
        "reasons": reasons,
        "source": "fuse_cache",
        "fuse_path": str(fuse.get("path", "")),
        "fuse_age_seconds": float(fuse.get("age_seconds", 0.0) or 0.0),
        "fuse_artifact": str(fuse.get("artifact", "")),
        "fuse_checksum": str(fuse.get("checksum", "")),
        "backup_intel": fuse.get("backup_intel", {}) if isinstance(fuse.get("backup_intel", {}), dict) else {},
    }
    return {
        "source": "fuse_cache",
        "cached": True,
        "ok": bool(payload.get("allowed", False)),
        "timeout": False,
        "returncode": 0 if bool(payload.get("allowed", False)) else 3,
        "stdout": "",
        "stderr": "",
        "payload": payload,
    }


def classify_controlled_failure(payload: dict[str, Any]) -> bool:
    steps = payload.get("steps", {}) if isinstance(payload.get("steps", {}), dict) else {}
    canary = steps.get("canary_order", {}) if isinstance(steps.get("canary_order", {}), dict) else {}
    reason = str(canary.get("reason", ""))
    if reason in {
        "signal_missing",
        "risk_guard_blocked",
        "precheck_quantity_below_min_qty",
        "precheck_insufficient_quote_balance",
        "precheck_insufficient_base_balance",
        "notional_floor_above_budget",
        "exchange_reject",
        "allow_live_order_false",
    }:
        return True
    mode = str(payload.get("mode", ""))
    return mode in {"live_ready_signal_blocked", "live_ready_risk_blocked", "degraded_read_only"}


def build_guard_artifact(
    *,
    review_dir: Path,
    requested_mode: str,
    effective_mode: str,
    downgrade_reason: str,
    idempotency: dict[str, Any],
    ticket_refresh: dict[str, Any],
    risk_guard: dict[str, Any],
    takeover: dict[str, Any],
    artifact_ttl_hours: int,
) -> tuple[Path, Path, dict[str, Any]]:
    payload = {
        "generated_at_utc": now_utc_iso(),
        "action": "trade_live_exec_guard",
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "status": "",
        "executed": False,
        "downgrade_reason": downgrade_reason,
        "idempotency": idempotency,
        "ticket_refresh": {
            "source": str(ticket_refresh.get("source", "")),
            "skipped": bool(ticket_refresh.get("skipped", False)),
            "returncode": int(ticket_refresh.get("returncode", 0)),
            "payload": ticket_refresh.get("payload", {}),
        },
        "risk_guard": {
            "source": str(risk_guard.get("source", "")),
            "cached": bool(risk_guard.get("cached", False)),
            "returncode": int(risk_guard.get("returncode", 0)),
            "payload": risk_guard.get("payload", {}),
        },
        "takeover": {
            "returncode": int(takeover.get("returncode", 0)),
            "payload": takeover.get("payload", {}),
            "stdout_excerpt": str(takeover.get("stdout", ""))[:2000],
            "stderr_excerpt": str(takeover.get("stderr", ""))[:2000],
        },
    }
    takeover_payload = takeover.get("payload", {}) if isinstance(takeover.get("payload", {}), dict) else {}
    steps = takeover_payload.get("steps", {}) if isinstance(takeover_payload.get("steps", {}), dict) else {}
    canary = steps.get("canary_order", {}) if isinstance(steps.get("canary_order", {}), dict) else {}
    payload["executed"] = bool(canary.get("executed", False))
    if bool(idempotency.get("duplicate", False)):
        payload["status"] = "idempotent_skip"
    elif effective_mode == "probe" and requested_mode == "canary" and downgrade_reason:
        payload["status"] = f"downgraded_probe_{downgrade_reason}"
    elif effective_mode == "probe":
        payload["status"] = "probe_completed"
    elif bool(payload.get("executed", False)):
        payload["status"] = "canary_executed"
    elif classify_controlled_failure(takeover_payload):
        payload["status"] = "canary_controlled_block"
    elif str(takeover_payload.get("panic", "")):
        payload["status"] = "panic"
    else:
        payload["status"] = "error"

    stamp = now_utc_compact()
    out_json = review_dir / f"{stamp}_trade_live_exec_guard.json"
    out_checksum = review_dir / f"{stamp}_trade_live_exec_guard_checksum.json"
    write_json(out_json, payload)
    evicted = evict_old_guard_artifacts(
        directory=review_dir,
        now_dt=now_utc(),
        ttl_hours=max(1, int(artifact_ttl_hours)),
        keep_names={out_json.name, out_checksum.name},
    )
    payload["governance"] = {
        "artifact_ttl_hours": int(max(1, int(artifact_ttl_hours))),
        "evicted_files": int(evicted),
    }
    payload["artifact"] = str(out_json)
    write_json(out_json, payload)
    digest, size_bytes = sha256_file(out_json)
    write_json(
        out_checksum,
        {
            "generated_at_utc": now_utc_iso(),
            "files": [{"path": str(out_json), "sha256": digest, "size_bytes": int(size_bytes)}],
        },
    )
    payload["checksum"] = str(out_checksum)
    write_json(out_json, payload)
    return out_json, out_checksum, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded local execution wrapper for Binance takeover.")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--mode", choices=["probe", "canary"], default="probe")
    parser.add_argument("--market", choices=["spot", "futures_usdm"], default="spot")
    parser.add_argument("--canary-quote-usdt", type=float, default=5.0)
    parser.add_argument("--rate-limit-per-minute", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--max-drawdown", type=float, default=0.05)
    parser.add_argument("--trade-window-hours", type=int, default=24)
    parser.add_argument("--allow-live-order", action="store_true")
    parser.add_argument("--allow-daemon-env-fallback", action="store_true")
    parser.add_argument("--order-symbol", default="")
    parser.add_argument("--order-side", choices=["BUY", "SELL"], default="")
    parser.add_argument("--order-quantity", type=float, default=0.0)
    parser.add_argument("--risk-intent", choices=["normal", "reduce_only"], default="normal")
    parser.add_argument("--risk-fuse-max-age-seconds", type=int, default=300)
    parser.add_argument("--panic-cooldown-seconds", type=int, default=1800)
    parser.add_argument("--max-daily-loss-ratio", type=float, default=0.05)
    parser.add_argument("--max-open-exposure-ratio", type=float, default=0.50)
    parser.add_argument("--refresh-tickets", action="store_true")
    parser.add_argument("--ticket-symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSD")
    parser.add_argument("--ticket-equity-usdt", type=float, default=0.0)
    parser.add_argument("--ticket-min-confidence", type=float, default=None)
    parser.add_argument("--ticket-min-convexity", type=float, default=None)
    parser.add_argument("--ticket-max-age-days", type=int, default=14)
    parser.add_argument("--idempotency-ttl-seconds", type=int, default=DEFAULT_IDEMPOTENCY_TTL_SECONDS)
    parser.add_argument("--idempotency-max-entries", type=int, default=DEFAULT_IDEMPOTENCY_MAX_ENTRIES)
    parser.add_argument("--artifact-ttl-hours", type=int, default=DEFAULT_ARTIFACT_TTL_HOURS)
    parser.add_argument("--mutex-timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    _, system_root = resolve_workspace(args.workspace)
    output_root = system_root / "output"
    review_dir = output_root / "review"
    state_dir = output_root / "state"
    review_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    requested_mode = str(args.mode)
    effective_mode = requested_mode
    downgrade_reason = ""
    timeout_seconds = max(10.0, min(120.0, float(max(100, int(args.timeout_ms))) / 1000.0 * 6.0))
    as_of = str(args.date).strip() or date.today().isoformat()
    ledger_path = state_dir / "trade_live_exec_guard_idempotency.json"
    idempotency_key = hashlib.sha256(
        (
            f"{as_of}|{requested_mode}|{args.market}|{str(args.order_symbol).upper()}|"
            f"{str(args.order_side).upper()}|{float(args.order_quantity):.8f}|"
            f"{float(args.canary_quote_usdt):.4f}|{args.risk_intent}|{bool(args.allow_live_order)}"
        ).encode("utf-8")
    ).hexdigest()[:28]
    use_idempotency = requested_mode != "probe"

    artifact_payload: dict[str, Any]
    try:
        with RunHalfhourMutex(
            output_root=output_root,
            owner=f"guarded_exec:{requested_mode}:{as_of}",
            timeout_seconds=float(args.mutex_timeout_seconds),
        ):
            if use_idempotency:
                idempotency = apply_idempotency_guard(
                    ledger_path=ledger_path,
                    key=idempotency_key,
                    ttl_seconds=max(1, int(args.idempotency_ttl_seconds)),
                    max_entries=max(100, int(args.idempotency_max_entries)),
                )
            else:
                rows = load_idempotency_entries(
                    ledger_path,
                    ttl_seconds=max(1, int(args.idempotency_ttl_seconds)),
                    now_dt=now_utc(),
                )
                idempotency = {
                    "key": idempotency_key,
                    "duplicate": False,
                    "ttl_seconds": max(1, int(args.idempotency_ttl_seconds)),
                    "path": str(ledger_path),
                    "entries": len(rows),
                    "applied": False,
                }
            if bool(idempotency.get("duplicate", False)):
                out_json, _, artifact_payload = build_guard_artifact(
                    review_dir=review_dir,
                    requested_mode=requested_mode,
                    effective_mode="skipped",
                    downgrade_reason="",
                    idempotency=idempotency,
                    ticket_refresh={"source": "", "skipped": True, "returncode": 0, "payload": {}},
                    risk_guard={"source": "", "cached": False, "returncode": 0, "payload": {}},
                    takeover={"returncode": 0, "payload": {}},
                    artifact_ttl_hours=max(1, int(args.artifact_ttl_hours)),
                )
                artifact_payload["artifact"] = str(out_json)
                print(json.dumps(artifact_payload, ensure_ascii=False, indent=2))
                return 0

            ticket_refresh = {"source": "", "skipped": True, "returncode": 0, "payload": {}}
            risk_guard = cached_risk_guard_result(
                output_root=output_root,
                max_age_seconds=max(1, int(args.risk_fuse_max_age_seconds)),
            )
            if risk_guard is None:
                risk_guard = run_json_command(
                    cmd=risk_guard_cmd(system_root=system_root, args=args),
                    cwd=system_root,
                    timeout_seconds=timeout_seconds,
                )
                risk_guard["source"] = "live_risk_guard"
                risk_guard["cached"] = False
                rg_payload = risk_guard.get("payload", {}) if isinstance(risk_guard.get("payload", {}), dict) else {}
                rg_ticket_refresh = rg_payload.get("ticket_refresh", {}) if isinstance(rg_payload.get("ticket_refresh", {}), dict) else {}
                ticket_refresh = {
                    "source": "live_risk_guard",
                    "skipped": not bool(args.refresh_tickets),
                    "returncode": int(rg_ticket_refresh.get("returncode", 0) or 0),
                    "payload": rg_ticket_refresh,
                }
            else:
                ticket_refresh = {
                    "source": "fuse_cache",
                    "skipped": True,
                    "returncode": 0,
                    "payload": {
                        "reason": "fresh_risk_fuse",
                        "fuse_path": str((risk_guard.get("payload", {}) if isinstance(risk_guard.get("payload", {}), dict) else {}).get("fuse_path", "")),
                    },
                }
            if bool(risk_guard.get("timeout", False)) or int(risk_guard.get("returncode", 0)) not in {0, 3}:
                panic_close_all(
                    output_root,
                    reason="guarded_exec_risk_guard_failure",
                    detail=f"rc={risk_guard.get('returncode', 0)}; stderr={str(risk_guard.get('stderr', ''))[:400]}",
                )
            risk_blocked = int(risk_guard.get("returncode", 0)) == 3
            if requested_mode == "canary":
                if not bool(args.allow_live_order):
                    effective_mode = "probe"
                    downgrade_reason = "allow_live_order_false"
                elif args.risk_intent != "reduce_only" and risk_blocked:
                    effective_mode = "probe"
                    downgrade_reason = "risk_guard_blocked"

            takeover = run_json_command(
                cmd=takeover_cmd(system_root=system_root, args=args, effective_mode=effective_mode),
                cwd=system_root,
                timeout_seconds=timeout_seconds,
            )
            takeover_payload = takeover.get("payload", {}) if isinstance(takeover.get("payload", {}), dict) else {}
            if bool(takeover.get("timeout", False)):
                panic_close_all(output_root, reason="guarded_exec_takeover_timeout", detail="binance_live_takeover subprocess timeout")
            if int(takeover.get("returncode", 0)) != 0 and not takeover_payload:
                panic_close_all(
                    output_root,
                    reason="guarded_exec_takeover_failure",
                    detail=f"rc={takeover.get('returncode', 0)}; stderr={str(takeover.get('stderr', ''))[:400]}",
                )

            _, _, artifact_payload = build_guard_artifact(
                review_dir=review_dir,
                requested_mode=requested_mode,
                effective_mode=effective_mode,
                downgrade_reason=downgrade_reason,
                idempotency=idempotency,
                ticket_refresh=ticket_refresh,
                risk_guard=risk_guard,
                takeover=takeover,
                artifact_ttl_hours=max(1, int(args.artifact_ttl_hours)),
            )

            if int(takeover.get("returncode", 0)) != 0 and not classify_controlled_failure(takeover_payload) and not str(takeover_payload.get("panic", "")):
                panic_close_all(
                    output_root,
                    reason="guarded_exec_takeover_unclassified_failure",
                    detail=f"rc={takeover.get('returncode', 0)}; stdout={str(takeover.get('stdout', ''))[:400]}",
                )

    except PanicTriggered as exc:
        artifact_payload = {
            "generated_at_utc": now_utc_iso(),
            "action": "trade_live_exec_guard",
            "status": "panic",
            "executed": False,
            "panic": str(exc),
            "workspace": str(system_root.parent),
            "output_root": str(output_root),
        }
        print(json.dumps(artifact_payload, ensure_ascii=False, indent=2))
        return 2
    except Exception as exc:
        artifact_payload = {
            "generated_at_utc": now_utc_iso(),
            "action": "trade_live_exec_guard",
            "status": "error",
            "executed": False,
            "error": str(exc),
            "workspace": str(system_root.parent),
            "output_root": str(output_root),
        }
        print(json.dumps(artifact_payload, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(artifact_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
