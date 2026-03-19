#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_TICKET_FRESHNESS_SECONDS = 900
DEFAULT_PANIC_COOLDOWN_SECONDS = 1800
DEFAULT_MAX_DAILY_LOSS_RATIO = 0.05
DEFAULT_MAX_OPEN_EXPOSURE_RATIO = 0.50
DEFAULT_ARTIFACT_TTL_HOURS = 168
DEFAULT_BACKUP_INTEL_REL_PATH = "state/backup_web_intel.json"
DEFAULT_BACKUP_INTEL_MAX_AGE_SECONDS = 7200
DEFAULT_BACKUP_INTEL_REQUIRED_AUTHORITY = "risk_only"
DEFAULT_BACKUP_INTEL_BLOCK_SEVERITIES = ("high",)
DEFAULT_TAKEOVER_MARKET = "spot"
SUPPORTED_TAKEOVER_MARKETS = {"spot", "futures_usdm", "portfolio_margin_um"}


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
RunHalfhourMutex = TAKEOVER.RunHalfhourMutex
choose_canary_ticket = TAKEOVER.choose_canary_ticket
latest_tickets_path = TAKEOVER.latest_tickets_path
load_config = TAKEOVER.load_config
load_whitelist_symbols = TAKEOVER.load_whitelist_symbols
read_json = TAKEOVER.read_json
to_float = TAKEOVER.to_float
write_json = TAKEOVER.write_json


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc_compact() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def normalize_takeover_market(raw: Any, *, default: str = DEFAULT_TAKEOVER_MARKET) -> str:
    market = str(raw or "").strip().lower()
    if market in SUPPORTED_TAKEOVER_MARKETS:
        return market
    return str(default).strip().lower() or DEFAULT_TAKEOVER_MARKET


def load_takeover_market(cfg: dict[str, Any]) -> str:
    validation = cfg.get("validation", {}) if isinstance(cfg.get("validation", {}), dict) else {}
    return normalize_takeover_market(validation.get("binance_live_takeover_market", DEFAULT_TAKEOVER_MARKET))


def load_latest_takeover_payload(*, output_root: Path, market: str) -> tuple[dict[str, Any], str]:
    review_dir = output_root / "review"
    market_name = normalize_takeover_market(market)
    candidates = [
        review_dir / f"latest_binance_live_takeover_{market_name}.json",
        review_dir / "latest_binance_live_takeover.json",
    ]
    for path in candidates:
        payload = read_json(path, {})
        if not isinstance(payload, dict) or not payload:
            continue
        steps = payload.get("steps", {}) if isinstance(payload.get("steps", {}), dict) else {}
        acct = steps.get("account_overview", {}) if isinstance(steps.get("account_overview", {}), dict) else {}
        payload_market_raw = payload.get("market") or acct.get("market")
        payload_market = normalize_takeover_market(payload_market_raw) if str(payload_market_raw).strip() else ""
        if path.name == "latest_binance_live_takeover.json" and payload_market and payload_market != market_name:
            continue
        return payload, str(path)
    return {}, ""


def parse_utc_text(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    for candidate in (
        text,
        text.replace("Z", "+00:00"),
    ):
        try:
            parsed = datetime.fromisoformat(candidate)
        except Exception:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


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
        }
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    return {
        "ok": int(proc.returncode) == 0,
        "timeout": False,
        "returncode": int(proc.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "payload": parse_json_payload(stdout),
    }


def ticket_refresh_cmd(*, system_root: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(system_root / "scripts" / "build_order_ticket.py"),
        "--config",
        str(resolve_path(args.config, anchor=system_root)),
        "--output-root",
        str(resolve_path(args.output_root, anchor=system_root)),
        "--review-dir",
        str(resolve_path(args.review_dir, anchor=system_root)),
        "--output-dir",
        str(resolve_path(args.review_dir, anchor=system_root)),
        "--symbols",
        str(args.ticket_symbols),
        "--max-age-days",
        str(max(1, int(args.ticket_max_age_days))),
    ]
    if str(args.date).strip():
        cmd.extend(["--date", str(args.date).strip()])
    if args.ticket_min_confidence is not None:
        cmd.extend(["--min-confidence", str(float(args.ticket_min_confidence))])
    if args.ticket_min_convexity is not None:
        cmd.extend(["--min-convexity", str(float(args.ticket_min_convexity))])
    if float(args.ticket_equity_usdt) > 0.0:
        cmd.extend(["--equity-usdt", f"{float(args.ticket_equity_usdt):.8f}"])
    return cmd


def evict_old_risk_artifacts(*, directory: Path, now_dt: datetime, ttl_hours: float, keep_names: set[str]) -> int:
    cutoff = now_dt.timestamp() - (float(ttl_hours) * 3600.0)
    removed = 0
    for pattern in ("*_live_risk_guard.json", "*_live_risk_guard_checksum.json"):
        for path in directory.glob(pattern):
            if path.name in keep_names:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            if mtime >= cutoff:
                continue
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
    return removed


def load_recent_panic_state(*, output_root: Path, now_dt: datetime, cooldown_seconds: int) -> dict[str, Any]:
    path = output_root / "state" / "panic_close_all.json"
    if not path.exists():
        return {
            "active": False,
            "path": str(path),
            "reason": "",
            "detail": "",
            "age_seconds": None,
        }
    payload = read_json(path, {})
    ts = parse_utc_text(str(payload.get("ts_utc", "")))
    if ts is None:
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            ts = now_dt
    age_seconds = max(0.0, (now_dt - ts).total_seconds())
    return {
        "active": bool(age_seconds <= max(1, int(cooldown_seconds))),
        "path": str(path),
        "reason": str(payload.get("reason", "")),
        "detail": str(payload.get("detail", "")),
        "age_seconds": float(age_seconds),
        "ts_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def estimate_position_notional(row: dict[str, Any]) -> float:
    notional = abs(to_float(row.get("notional", 0.0), 0.0))
    if notional > 0.0:
        return float(notional)
    qty = abs(to_float(row.get("qty", row.get("positionAmt", 0.0)), 0.0))
    price = abs(to_float(row.get("market_price", row.get("markPrice", 0.0)), 0.0))
    return float(qty * price)


def load_exposure_snapshot(output_root: Path, *, market: str) -> dict[str, Any]:
    takeover, latest_takeover = load_latest_takeover_payload(output_root=output_root, market=market)
    steps = takeover.get("steps", {}) if isinstance(takeover, dict) else {}
    acct = steps.get("account_overview", {}) if isinstance(steps.get("account_overview", {}), dict) else {}
    snapshot_meta = steps.get("live_snapshot", {}) if isinstance(steps.get("live_snapshot", {}), dict) else {}
    snapshot_path = Path(str(snapshot_meta.get("path", "")).strip()) if str(snapshot_meta.get("path", "")).strip() else None
    if snapshot_path is not None and not snapshot_path.is_absolute():
        snapshot_path = output_root.parent / snapshot_path
    snapshot = read_json(snapshot_path, {}) if snapshot_path is not None and snapshot_path.exists() else {}
    positions = snapshot.get("positions", []) if isinstance(snapshot.get("positions", []), list) else []
    open_exposure_notional = float(sum(estimate_position_notional(row) for row in positions if isinstance(row, dict)))
    closed_pnl = to_float(snapshot.get("closed_pnl", snapshot_meta.get("closed_pnl", 0.0)), 0.0)
    quote_available = to_float(acct.get("quote_available", 0.0), 0.0)
    equity_proxy = max(0.0, quote_available) + max(0.0, open_exposure_notional)
    open_exposure_ratio = (open_exposure_notional / equity_proxy) if equity_proxy > 1e-12 else (1.0 if open_exposure_notional > 0.0 else 0.0)
    daily_loss_ratio = (abs(min(0.0, closed_pnl)) / equity_proxy) if equity_proxy > 1e-12 else (1.0 if closed_pnl < 0.0 else 0.0)
    return {
        "latest_takeover_path": str(latest_takeover),
        "market": normalize_takeover_market(acct.get("market") or takeover.get("market") or market),
        "live_snapshot_path": str(snapshot_path) if snapshot_path is not None else "",
        "quote_available": float(quote_available),
        "open_exposure_notional": float(open_exposure_notional),
        "open_exposure_ratio": float(open_exposure_ratio),
        "closed_pnl": float(closed_pnl),
        "daily_loss_ratio": float(daily_loss_ratio),
        "equity_proxy_usdt": float(equity_proxy),
        "open_position_count": int(len(positions)),
    }


def normalize_text_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def normalize_symbol(raw: Any) -> str:
    return str(raw or "").strip().upper()


def load_backup_intel_policy(cfg: dict[str, Any], *, output_root: Path) -> dict[str, Any]:
    validation = cfg.get("validation", {}) if isinstance(cfg.get("validation", {}), dict) else {}
    rel_path = str(validation.get("backup_web_intel_artifact_path", DEFAULT_BACKUP_INTEL_REL_PATH)).strip()
    path = Path(rel_path) if rel_path else Path(DEFAULT_BACKUP_INTEL_REL_PATH)
    if not path.is_absolute():
        path = output_root / path
    severities = [x.lower() for x in normalize_text_list(validation.get("backup_web_intel_block_severities", list(DEFAULT_BACKUP_INTEL_BLOCK_SEVERITIES)))]
    if not severities:
        severities = [x.lower() for x in DEFAULT_BACKUP_INTEL_BLOCK_SEVERITIES]
    return {
        "enabled": bool(validation.get("backup_web_intel_enabled", False)),
        "artifact_path": path,
        "max_age_seconds": max(60, int(to_float(validation.get("backup_web_intel_max_age_seconds", DEFAULT_BACKUP_INTEL_MAX_AGE_SECONDS), DEFAULT_BACKUP_INTEL_MAX_AGE_SECONDS))),
        "block_on_no_trade": bool(validation.get("backup_web_intel_block_on_no_trade", True)),
        "block_on_bias_conflict": bool(validation.get("backup_web_intel_block_on_bias_conflict", True)),
        "block_severities": severities,
        "required_authority": str(validation.get("backup_web_intel_required_authority", DEFAULT_BACKUP_INTEL_REQUIRED_AUTHORITY)).strip().lower() or DEFAULT_BACKUP_INTEL_REQUIRED_AUTHORITY,
    }


def evaluate_backup_intel(
    *,
    policy: dict[str, Any],
    ticket: dict[str, Any],
    now_dt: datetime,
) -> dict[str, Any]:
    path = Path(policy.get("artifact_path", ""))
    summary: dict[str, Any] = {
        "enabled": bool(policy.get("enabled", False)),
        "artifact_path": str(path),
        "status": "disabled",
        "active": False,
        "age_seconds": None,
        "generated_at_utc": "",
        "expires_at_utc": "",
        "authority": "",
        "required_authority": str(policy.get("required_authority", "")),
        "fallback_use_allowed": None,
        "ticket_symbol": normalize_symbol(ticket.get("symbol", "")),
        "ticket_side": str(ticket.get("side", "")).strip().upper(),
        "matched_no_trade_symbols": [],
        "matched_bias_conflicts": [],
        "matched_risk_flags": [],
        "blocking_reasons": [],
    }
    if not bool(policy.get("enabled", False)):
        return summary
    if not path.exists():
        summary["status"] = "missing"
        return summary
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        summary["status"] = "invalid_payload"
        return summary

    generated_dt = parse_utc_text(str(payload.get("generated_at", "") or payload.get("generated_at_utc", "") or payload.get("updated_at_utc", "")))
    if generated_dt is None:
        try:
            generated_dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            generated_dt = None
    expires_dt = parse_utc_text(str(payload.get("expires_at", "") or payload.get("expires_at_utc", "")))
    age_seconds: float | None = None
    if generated_dt is not None:
        age_seconds = max(0.0, (now_dt - generated_dt).total_seconds())
    summary["generated_at_utc"] = generated_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if generated_dt is not None else ""
    summary["expires_at_utc"] = expires_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if expires_dt is not None else ""
    summary["age_seconds"] = float(age_seconds) if age_seconds is not None else None

    if (expires_dt is not None and now_dt > expires_dt) or (age_seconds is not None and age_seconds > float(policy.get("max_age_seconds", DEFAULT_BACKUP_INTEL_MAX_AGE_SECONDS))):
        summary["status"] = "stale"
        return summary

    authority = str(payload.get("fallback_trade_authority", "")).strip().lower()
    fallback_use_allowed = bool(payload.get("fallback_use_allowed", True))
    summary["authority"] = authority
    summary["fallback_use_allowed"] = bool(fallback_use_allowed)
    if not fallback_use_allowed:
        summary["status"] = "inactive_not_allowed"
        return summary
    required_authority = str(policy.get("required_authority", "")).strip().lower()
    if required_authority and authority != required_authority:
        summary["status"] = "inactive_invalid_authority"
        return summary

    status_text = str(payload.get("status", "ok")).strip().lower()
    if status_text in {"source_unavailable", "insufficient_evidence", "error"}:
        summary["status"] = f"inactive_{status_text}"
        return summary

    ticket_symbol = str(summary.get("ticket_symbol", ""))
    ticket_side = str(summary.get("ticket_side", ""))
    no_trade_rows = payload.get("no_trade_list", []) if isinstance(payload.get("no_trade_list", []), list) else []
    bias_rows = payload.get("candidate_biases", []) if isinstance(payload.get("candidate_biases", []), list) else []
    risk_flags = payload.get("risk_flags", []) if isinstance(payload.get("risk_flags", []), list) else []
    matched_no_trade: list[dict[str, Any]] = []
    matched_bias_conflicts: list[dict[str, Any]] = []
    matched_risk_flags: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []

    if ticket_symbol:
        for row in no_trade_rows:
            if not isinstance(row, dict):
                continue
            symbol = normalize_symbol(row.get("symbol", row.get("instrument", "")))
            if symbol in {"", "ALL", "GLOBAL", "ANY"} or symbol == ticket_symbol:
                matched_no_trade.append(
                    {
                        "symbol": symbol or "GLOBAL",
                        "reason": str(row.get("reason", "")).strip(),
                    }
                )
        for row in bias_rows:
            if not isinstance(row, dict):
                continue
            symbol = normalize_symbol(row.get("symbol", row.get("instrument", "")))
            if symbol != ticket_symbol:
                continue
            bias = str(row.get("bias", "")).strip().lower()
            conflict = (ticket_side == "BUY" and bias in {"short_bias", "no_trade"}) or (
                ticket_side == "SELL" and bias in {"long_bias", "no_trade"}
            )
            if conflict:
                matched_bias_conflicts.append(
                    {
                        "symbol": symbol,
                        "bias": bias,
                        "thesis_type": str(row.get("thesis_type", "")).strip(),
                    }
                )

    block_severities = set(str(x).strip().lower() for x in policy.get("block_severities", []) if str(x).strip())
    for row in risk_flags:
        if not isinstance(row, dict):
            continue
        severity = str(row.get("severity", "")).strip().lower()
        if severity not in block_severities:
            continue
        matched_risk_flags.append(
            {
                "code": str(row.get("code", "")).strip(),
                "severity": severity,
                "message": str(row.get("message", "")).strip(),
            }
        )

    if bool(policy.get("block_on_no_trade", True)) and matched_no_trade:
        blocking_reasons.append("backup_intel_no_trade_symbol")
    if bool(policy.get("block_on_bias_conflict", True)) and matched_bias_conflicts:
        blocking_reasons.append("backup_intel_bias_conflict")
    if matched_risk_flags:
        blocking_reasons.append("backup_intel_high_risk_flag")

    summary["status"] = "active"
    summary["active"] = True
    summary["matched_no_trade_symbols"] = matched_no_trade
    summary["matched_bias_conflicts"] = matched_bias_conflicts
    summary["matched_risk_flags"] = matched_risk_flags
    summary["blocking_reasons"] = blocking_reasons
    return summary


def build_payload(
    *,
    config_path: Path,
    output_root: Path,
    review_dir: Path,
    ticket_freshness_seconds: int,
    panic_cooldown_seconds: int,
    max_daily_loss_ratio: float,
    max_open_exposure_ratio: float,
) -> dict[str, Any]:
    now_dt = now_utc()
    cfg = load_config(config_path)
    takeover_market = load_takeover_market(cfg)
    whitelist = load_whitelist_symbols(cfg)
    ticket = choose_canary_ticket(output_root=output_root, whitelist=whitelist, market=takeover_market)
    ticket_path = latest_tickets_path(output_root)
    ticket_age_seconds: float | None = None
    if ticket_path is not None and ticket_path.exists():
        ticket_age_seconds = max(0.0, (now_dt - datetime.fromtimestamp(ticket_path.stat().st_mtime, tz=timezone.utc)).total_seconds())
    panic_state = load_recent_panic_state(output_root=output_root, now_dt=now_dt, cooldown_seconds=panic_cooldown_seconds)
    exposure = load_exposure_snapshot(output_root, market=takeover_market)
    backup_intel_policy = load_backup_intel_policy(cfg, output_root=output_root)
    backup_intel = evaluate_backup_intel(policy=backup_intel_policy, ticket=ticket, now_dt=now_dt)

    reasons: list[str] = []
    if bool(ticket.get("signal_missing", False)):
        reasons.append(f"ticket_missing:{ticket.get('reason', 'unknown')}")
    if ticket_age_seconds is None:
        reasons.append("ticket_artifact_missing")
    elif ticket_age_seconds > float(max(1, int(ticket_freshness_seconds))):
        reasons.append("ticket_artifact_stale")
    if bool(panic_state.get("active", False)):
        reasons.append("panic_cooldown_active")
    if float(exposure.get("daily_loss_ratio", 0.0)) > float(max_daily_loss_ratio):
        reasons.append("daily_loss_limit_breached")
    if float(exposure.get("open_exposure_ratio", 0.0)) > float(max_open_exposure_ratio):
        reasons.append("open_exposure_above_cap")
    for reason in [str(x) for x in backup_intel.get("blocking_reasons", []) if str(x)]:
        if reason not in reasons:
            reasons.append(reason)

    allowed = len(reasons) == 0
    return {
        "generated_at_utc": now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": "live_risk_guard",
        "allowed": bool(allowed),
        "status": "pass" if allowed else "blocked",
        "reasons": reasons,
        "config": str(config_path),
        "output_root": str(output_root),
        "review_dir": str(review_dir),
        "whitelist": whitelist,
        "ticket_selection": {
            "path": str(ticket_path) if ticket_path is not None else "",
            "age_seconds": float(ticket_age_seconds) if ticket_age_seconds is not None else None,
            "ticket_freshness_seconds": int(max(1, int(ticket_freshness_seconds))),
            "selected": ticket,
        },
        "backup_intel": backup_intel,
        "panic_state": panic_state,
        "exposure": exposure,
        "limits": {
            "panic_cooldown_seconds": int(max(1, int(panic_cooldown_seconds))),
            "max_daily_loss_ratio": float(max_daily_loss_ratio),
            "max_open_exposure_ratio": float(max_open_exposure_ratio),
        },
    }


def main() -> int:
    system_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate independent live risk gate before execution.")
    parser.add_argument("--date", default="")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--review-dir", default="output/review")
    parser.add_argument("--ticket-freshness-seconds", type=int, default=DEFAULT_TICKET_FRESHNESS_SECONDS)
    parser.add_argument("--panic-cooldown-seconds", type=int, default=DEFAULT_PANIC_COOLDOWN_SECONDS)
    parser.add_argument("--max-daily-loss-ratio", type=float, default=DEFAULT_MAX_DAILY_LOSS_RATIO)
    parser.add_argument("--max-open-exposure-ratio", type=float, default=DEFAULT_MAX_OPEN_EXPOSURE_RATIO)
    parser.add_argument("--refresh-tickets", action="store_true")
    parser.add_argument("--ticket-symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSD")
    parser.add_argument("--ticket-equity-usdt", type=float, default=0.0)
    parser.add_argument("--ticket-min-confidence", type=float, default=None)
    parser.add_argument("--ticket-min-convexity", type=float, default=None)
    parser.add_argument("--ticket-max-age-days", type=int, default=14)
    parser.add_argument("--artifact-ttl-hours", type=int, default=DEFAULT_ARTIFACT_TTL_HOURS)
    parser.add_argument("--mutex-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--skip-mutex", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config, anchor=system_root)
    output_root = resolve_path(args.output_root, anchor=system_root)
    review_dir = resolve_path(args.review_dir, anchor=system_root)
    review_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any]
    rc = 0
    owner = f"live_risk_guard:{now_utc_iso()}"
    timeout_seconds = 30.0
    try:
        lock_cm = TAKEOVER.nullcontext() if bool(args.skip_mutex) else RunHalfhourMutex(
            output_root=output_root,
            owner=owner,
            timeout_seconds=float(args.mutex_timeout_seconds),
        )
        with lock_cm:
            ticket_refresh = {"returncode": 0, "payload": {}}
            if bool(args.refresh_tickets):
                ticket_refresh = run_json_command(
                    cmd=ticket_refresh_cmd(system_root=system_root, args=args),
                    cwd=system_root,
                    timeout_seconds=timeout_seconds,
                )
                if bool(ticket_refresh.get("timeout", False)) or int(ticket_refresh.get("returncode", 0)) != 0:
                    raise RuntimeError(f"ticket_refresh_failed rc={int(ticket_refresh.get('returncode', 0))}")

            payload = build_payload(
                config_path=config_path,
                output_root=output_root,
                review_dir=review_dir,
                ticket_freshness_seconds=max(1, int(args.ticket_freshness_seconds)),
                panic_cooldown_seconds=max(1, int(args.panic_cooldown_seconds)),
                max_daily_loss_ratio=max(0.0, float(args.max_daily_loss_ratio)),
                max_open_exposure_ratio=max(0.0, float(args.max_open_exposure_ratio)),
            )
            payload["ticket_refresh"] = {
                "returncode": int(ticket_refresh.get("returncode", 0)),
                "payload": ticket_refresh.get("payload", {}),
            }

            stamp = now_utc_compact()
            out_json = review_dir / f"{stamp}_live_risk_guard.json"
            out_checksum = review_dir / f"{stamp}_live_risk_guard_checksum.json"
            write_json(out_json, payload)
            evicted = evict_old_risk_artifacts(
                directory=review_dir,
                now_dt=now_utc(),
                ttl_hours=max(1, int(args.artifact_ttl_hours)),
                keep_names={out_json.name, out_checksum.name},
            )
            payload["governance"] = {
                "artifact_ttl_hours": int(max(1, int(args.artifact_ttl_hours))),
                "evicted_files": int(evicted),
            }
            payload["artifact"] = str(out_json)
            write_json(out_json, payload)
            json_sha, json_size = sha256_file(out_json)
            write_json(
                out_checksum,
                {
                    "generated_at_utc": now_utc_iso(),
                    "files": [{"path": str(out_json), "sha256": json_sha, "size_bytes": int(json_size)}],
                },
            )
            fuse_state = {
                "updated_at_utc": now_utc_iso(),
                "allowed": bool(payload.get("allowed", False)),
                "status": str(payload.get("status", "blocked")),
                "reasons": list(payload.get("reasons", [])),
                "artifact": str(out_json),
                "checksum": str(out_checksum),
                "backup_intel": payload.get("backup_intel", {}),
            }
            write_json(state_dir / "live_risk_fuse.json", fuse_state)
            payload["fuse_path"] = str(state_dir / "live_risk_fuse.json")
            write_json(out_json, payload)
            rc = 0 if bool(payload.get("allowed", False)) else 3
    except Exception as exc:
        payload = {
            "generated_at_utc": now_utc_iso(),
            "action": "live_risk_guard",
            "allowed": False,
            "status": "error",
            "reasons": ["guard_execution_error"],
            "error": str(exc),
            "config": str(config_path),
            "output_root": str(output_root),
            "review_dir": str(review_dir),
        }
        rc = 2

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
