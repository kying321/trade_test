#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_STATE_PATH = SYSTEM_ROOT / "output" / "state" / "spot_paper_state.json"
DEFAULT_LEDGER_PATH = SYSTEM_ROOT / "output" / "logs" / "paper_execution_ledger.jsonl"
DEFAULT_PULSE_LOCK_PATH = SYSTEM_ROOT / "output" / "state" / "run-halfhour-pulse.lock"


def parse_ts(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def build_state_fingerprint(state: dict[str, Any]) -> str:
    normalized = json.dumps(
        state if isinstance(state, dict) else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def acquire_lock(path: Path, *, timeout_sec: float, retry_sec: float) -> tuple[Any | None, bool, float]:
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl  # type: ignore
    except Exception:
        return None, False, max(0.0, time.monotonic() - started)
    lockf = path.open("a+", encoding="utf-8")
    while True:
        try:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lockf, True, max(0.0, time.monotonic() - started)
        except BlockingIOError:
            if (time.monotonic() - started) >= max(0.0, timeout_sec):
                try:
                    lockf.close()
                except Exception:
                    pass
                return None, False, max(0.0, time.monotonic() - started)
            time.sleep(max(0.01, retry_sec))
        except Exception:
            try:
                lockf.close()
            except Exception:
                pass
            return None, False, max(0.0, time.monotonic() - started)


def release_lock(lockf: Any | None) -> None:
    if lockf is None:
        return
    try:
        import fcntl  # type: ignore

        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lockf.close()
    except Exception:
        pass


def backup_state(path: Path, current: dict[str, Any]) -> Path:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bdir = path.parent / "backup"
    bdir.mkdir(parents=True, exist_ok=True)
    out = bdir / f"spot_paper_state_backfill_{ts}.json"
    out.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def load_sell_events(ledger_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not ledger_path.exists():
        return events
    for raw in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("domain") or "").strip() != "paper_execution":
            continue
        if str(payload.get("side") or "").strip().upper() != "SELL":
            continue
        ts = parse_ts(payload.get("ts"))
        if ts is None:
            continue
        pnl = safe_float(payload.get("realized_pnl_change"), 0.0)
        events.append(
            {
                "ts": ts,
                "ts_text": fmt_utc(ts),
                "realized_pnl_change": round(pnl, 10),
                "symbol": str(payload.get("symbol") or "").strip().upper(),
                "route": str(payload.get("route") or "").strip(),
            }
        )
    events.sort(key=lambda row: row["ts"])
    return events


def find_candidates(sell_events: list[dict[str, Any]]) -> dict[str, Any]:
    latest_sell = sell_events[-1] if sell_events else None
    latest_negative = None
    trailing_negative_streak = 0

    for row in reversed(sell_events):
        pnl = safe_float(row.get("realized_pnl_change"), 0.0)
        if latest_negative is None and pnl < 0:
            latest_negative = row
        if pnl < 0:
            trailing_negative_streak += 1
            continue
        break

    return {
        "sell_events": len(sell_events),
        "latest_sell": latest_sell,
        "latest_negative_sell": latest_negative,
        "trailing_negative_streak": trailing_negative_streak,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill spot_paper_state.last_loss_ts from paper_execution_ledger with strict streak validation."
    )
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--pulse-lock-path", default=str(DEFAULT_PULSE_LOCK_PATH))
    parser.add_argument("--allow-latest-loss-fallback", action="store_true")
    parser.add_argument("--expected-state-fingerprint", default="")
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--lock-timeout-sec", type=float, default=2.0)
    parser.add_argument("--lock-retry-sec", type=float, default=0.05)
    parser.add_argument("--write", action="store_true", help="Persist backfilled last_loss_ts. Default is dry-run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    state_path = Path(str(args.state_path)).expanduser().resolve()
    ledger_path = Path(str(args.ledger_path)).expanduser().resolve()
    pulse_lock_path = Path(str(args.pulse_lock_path)).expanduser().resolve()

    state_raw = load_json(state_path)
    state_fingerprint = build_state_fingerprint(state_raw)
    expected_state_fingerprint = str(args.expected_state_fingerprint or "").strip() or None
    current_streak = safe_int(state_raw.get("consecutive_losses"), 0)
    current_last_loss_ts = str(state_raw.get("last_loss_ts") or "").strip() or None
    sell_events = load_sell_events(ledger_path)
    candidates = find_candidates(sell_events)

    latest_negative = candidates["latest_negative_sell"]
    trailing_negative_streak = int(candidates["trailing_negative_streak"])
    selected_method = None
    selected_ts = None
    reasons: list[str] = []

    if current_streak <= 0:
        reasons.append("no_active_consecutive_loss_streak")
    if current_last_loss_ts and not bool(args.force_overwrite):
        reasons.append("last_loss_ts_already_present")
    if int(candidates["sell_events"]) == 0:
        reasons.append("ledger_sell_events_missing")
    if latest_negative is None:
        reasons.append("ledger_negative_sell_missing")

    if not reasons:
        if trailing_negative_streak == current_streak and latest_negative is not None:
            selected_method = "ledger_trailing_streak_match"
            selected_ts = str(latest_negative["ts_text"])
        elif bool(args.allow_latest_loss_fallback) and latest_negative is not None:
            selected_method = "ledger_latest_negative_fallback"
            selected_ts = str(latest_negative["ts_text"])
            reasons.append("streak_match_bypassed_via_latest_loss_fallback")
        else:
            reasons.append(
                f"trailing_negative_streak_mismatch(expected={current_streak},actual={trailing_negative_streak})"
            )

    eligible = selected_ts is not None and all(
        not reason.startswith("trailing_negative_streak_mismatch") and reason not in {
            "no_active_consecutive_loss_streak",
            "last_loss_ts_already_present",
            "ledger_sell_events_missing",
            "ledger_negative_sell_missing",
        }
        for reason in reasons
    )

    out = {
        "action": "backfill_paper_last_loss_ts",
        "ok": eligible,
        "eligible": eligible,
        "write_requested": bool(args.write),
        "write_performed": False,
        "state_path": str(state_path),
        "state_fingerprint": state_fingerprint,
        "expected_state_fingerprint": expected_state_fingerprint,
        "state_fingerprint_match": None if expected_state_fingerprint is None else state_fingerprint == expected_state_fingerprint,
        "ledger_path": str(ledger_path),
        "pulse_lock_path": str(pulse_lock_path),
        "current_streak": current_streak,
        "current_last_loss_ts": current_last_loss_ts,
        "allow_latest_loss_fallback": bool(args.allow_latest_loss_fallback),
        "selected_method": selected_method,
        "selected_last_loss_ts": selected_ts,
        "sell_event_count": int(candidates["sell_events"]),
        "trailing_negative_streak": trailing_negative_streak,
        "latest_sell": {
            "ts": candidates["latest_sell"]["ts_text"],
            "realized_pnl_change": candidates["latest_sell"]["realized_pnl_change"],
        }
        if isinstance(candidates["latest_sell"], dict)
        else None,
        "latest_negative_sell": {
            "ts": latest_negative["ts_text"],
            "realized_pnl_change": latest_negative["realized_pnl_change"],
        }
        if isinstance(latest_negative, dict)
        else None,
        "reasons": reasons,
        "lock_acquired": None,
        "lock_wait_sec": None,
        "backup_path": None,
    }

    if not bool(args.write):
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if not eligible:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 3

    lockf = None
    try:
        lockf, lock_acquired, lock_wait_sec = acquire_lock(
            pulse_lock_path,
            timeout_sec=max(0.0, float(args.lock_timeout_sec)),
            retry_sec=max(0.01, float(args.lock_retry_sec)),
        )
        out["lock_acquired"] = bool(lock_acquired)
        out["lock_wait_sec"] = round(lock_wait_sec, 6)
        if not lock_acquired:
            out["ok"] = False
            out["eligible"] = False
            out["reasons"] = ["pulse_lock_timeout"]
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 4

        current = load_json(state_path)
        current_state_fingerprint = build_state_fingerprint(current)
        out["state_fingerprint"] = current_state_fingerprint
        out["state_fingerprint_match"] = (
            None if expected_state_fingerprint is None else current_state_fingerprint == expected_state_fingerprint
        )
        if expected_state_fingerprint is not None and current_state_fingerprint != expected_state_fingerprint:
            out["ok"] = False
            out["eligible"] = False
            out["reasons"] = ["state_fingerprint_mismatch"]
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 5
        current_now_streak = safe_int(current.get("consecutive_losses"), 0)
        current_now_last_loss_ts = str(current.get("last_loss_ts") or "").strip() or None
        if current_now_streak != current_streak:
            out["ok"] = False
            out["eligible"] = False
            out["reasons"] = [f"state_streak_changed(expected={current_streak},actual={current_now_streak})"]
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 5
        if current_now_last_loss_ts and not bool(args.force_overwrite):
            out["ok"] = False
            out["eligible"] = False
            out["reasons"] = ["last_loss_ts_already_present"]
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 5

        backup_path = backup_state(state_path, current)
        current["last_loss_ts"] = selected_ts
        state_path.write_text(json.dumps(current, ensure_ascii=False) + "\n", encoding="utf-8")
        out["write_performed"] = True
        out["backup_path"] = str(backup_path)
        out["state_after"] = current
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    finally:
        release_lock(lockf)


if __name__ == "__main__":
    raise SystemExit(main())
