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
DEFAULT_ACK_PATH = SYSTEM_ROOT / "output" / "state" / "paper_consecutive_loss_ack.json"
DEFAULT_CHECKSUM_PATH = SYSTEM_ROOT / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"
DEFAULT_PULSE_LOCK_PATH = SYSTEM_ROOT / "output" / "state" / "run_halfhour_pulse.lock"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    lockf = path.open("w")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a short-lived manual ack for the local paper consecutive-loss guardrail."
    )
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--ack-path", default=str(DEFAULT_ACK_PATH))
    parser.add_argument("--checksum-path", default=str(DEFAULT_CHECKSUM_PATH))
    parser.add_argument("--pulse-lock-path", default=str(DEFAULT_PULSE_LOCK_PATH))
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--cooldown-hours", type=float, default=12.0)
    parser.add_argument("--allow-missing-last-loss-ts", action="store_true")
    parser.add_argument("--expected-state-fingerprint", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--lock-timeout-sec", type=float, default=2.0)
    parser.add_argument("--lock-retry-sec", type=float, default=0.05)
    parser.add_argument("--write", action="store_true", help="Persist ack artifact. Default is dry-run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    now_utc = dt.datetime.now(dt.timezone.utc)
    state_path = Path(str(args.state_path)).expanduser().resolve()
    ack_path = Path(str(args.ack_path)).expanduser().resolve()
    checksum_path = Path(str(args.checksum_path)).expanduser().resolve()
    pulse_lock_path = Path(str(args.pulse_lock_path)).expanduser().resolve()

    state_raw = load_state(state_path)
    state_fingerprint = build_state_fingerprint(state_raw)
    expected_state_fingerprint = str(args.expected_state_fingerprint or "").strip() or None
    streak = safe_int(state_raw.get("consecutive_losses"), 0)
    last_loss_ts_raw = str(state_raw.get("last_loss_ts") or "").strip() or None
    last_loss_ts = parse_ts(last_loss_ts_raw)
    cooldown_required_hours = max(0.0, float(args.cooldown_hours))
    cooldown_elapsed_hours: float | None = None
    reasons: list[str] = []

    if streak <= 0:
        reasons.append("no_active_consecutive_loss_streak")
    if last_loss_ts is not None:
        cooldown_elapsed_hours = max(
            0.0,
            (now_utc - last_loss_ts).total_seconds() / 3600.0,
        )
        if cooldown_elapsed_hours < cooldown_required_hours:
            reasons.append("cooldown_active")
    elif not bool(args.allow_missing_last_loss_ts):
        reasons.append("last_loss_ts_missing")

    eligible = len(reasons) == 0
    ack_payload = {
        "generated_at": fmt_utc(now_utc),
        "expires_at": fmt_utc(now_utc + dt.timedelta(hours=max(1.0, float(args.ttl_hours)))),
        "guardrail": "consecutive_loss_stop",
        "use_limit": 1,
        "uses_remaining": 1,
        "active": True,
        "streak_snapshot": int(streak),
        "cooldown_hours_required": round(cooldown_required_hours, 4),
        "last_loss_ts": fmt_utc(last_loss_ts) if last_loss_ts is not None else None,
        "allow_missing_last_loss_ts": bool(args.allow_missing_last_loss_ts),
        "note": str(args.note or "").strip(),
    }

    out = {
        "action": "ack_paper_consecutive_loss_guardrail",
        "ok": eligible,
        "eligible": eligible,
        "write_requested": bool(args.write),
        "write_performed": False,
        "state_path": str(state_path),
        "state_fingerprint": state_fingerprint,
        "expected_state_fingerprint": expected_state_fingerprint,
        "state_fingerprint_match": None if expected_state_fingerprint is None else state_fingerprint == expected_state_fingerprint,
        "ack_path": str(ack_path),
        "checksum_path": str(checksum_path),
        "pulse_lock_path": str(pulse_lock_path),
        "current_streak": int(streak),
        "last_loss_ts": fmt_utc(last_loss_ts) if last_loss_ts is not None else None,
        "cooldown_hours_required": round(cooldown_required_hours, 4),
        "cooldown_elapsed_hours": round(cooldown_elapsed_hours, 4) if cooldown_elapsed_hours is not None else None,
        "allow_missing_last_loss_ts": bool(args.allow_missing_last_loss_ts),
        "reasons": reasons,
        "ack_payload_preview": ack_payload,
        "lock_acquired": None,
        "lock_wait_sec": None,
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

        state_raw = load_state(state_path)
        state_fingerprint = build_state_fingerprint(state_raw)
        out["state_fingerprint"] = state_fingerprint
        out["state_fingerprint_match"] = (
            None if expected_state_fingerprint is None else state_fingerprint == expected_state_fingerprint
        )
        if expected_state_fingerprint is not None and state_fingerprint != expected_state_fingerprint:
            out["ok"] = False
            out["eligible"] = False
            out["reasons"] = ["state_fingerprint_mismatch"]
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 5

        streak = safe_int(state_raw.get("consecutive_losses"), 0)
        last_loss_ts_raw = str(state_raw.get("last_loss_ts") or "").strip() or None
        last_loss_ts = parse_ts(last_loss_ts_raw)
        out["current_streak"] = int(streak)
        out["last_loss_ts"] = fmt_utc(last_loss_ts) if last_loss_ts is not None else None
        cooldown_elapsed_hours = None
        reasons = []
        if streak <= 0:
            reasons.append("no_active_consecutive_loss_streak")
        if last_loss_ts is not None:
            cooldown_elapsed_hours = max(
                0.0,
                (now_utc - last_loss_ts).total_seconds() / 3600.0,
            )
            if cooldown_elapsed_hours < cooldown_required_hours:
                reasons.append("cooldown_active")
        elif not bool(args.allow_missing_last_loss_ts):
            reasons.append("last_loss_ts_missing")
        eligible = len(reasons) == 0
        ack_payload = {
            "generated_at": fmt_utc(now_utc),
            "expires_at": fmt_utc(now_utc + dt.timedelta(hours=max(1.0, float(args.ttl_hours)))),
            "guardrail": "consecutive_loss_stop",
            "use_limit": 1,
            "uses_remaining": 1,
            "active": True,
            "streak_snapshot": int(streak),
            "cooldown_hours_required": round(cooldown_required_hours, 4),
            "last_loss_ts": fmt_utc(last_loss_ts) if last_loss_ts is not None else None,
            "allow_missing_last_loss_ts": bool(args.allow_missing_last_loss_ts),
            "note": str(args.note or "").strip(),
        }
        out["eligible"] = eligible
        out["ok"] = eligible
        out["cooldown_elapsed_hours"] = round(cooldown_elapsed_hours, 4) if cooldown_elapsed_hours is not None else None
        out["reasons"] = reasons
        out["ack_payload_preview"] = ack_payload
        if not eligible:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 3

        ack_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        ack_path.write_text(json.dumps(ack_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checksum_payload = {
            "generated_at": fmt_utc(now_utc),
            "artifact": str(ack_path),
            "sha256": sha256_file(ack_path),
        }
        checksum_path.write_text(
            json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        out["write_performed"] = True
        out["checksum_sha256"] = checksum_payload["sha256"]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    finally:
        release_lock(lockf)


if __name__ == "__main__":
    raise SystemExit(main())
