#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def resolve_system_root() -> Path:
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


def load_native_module():
    script_path = SYSTEM_ROOT / "scripts" / "backtest_binance_indicator_combo_native_crypto.py"
    spec = importlib.util.spec_from_file_location("indicator_combo_native_crypto_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable_to_load_native_crypto_backtest_script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


native = load_native_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native crypto backtest under a subprocess timeout guard.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--symbol-group", default="custom")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback-bars", type=int, default=300)
    parser.add_argument("--sample-windows", type=int, default=3)
    parser.add_argument("--window-bars", type=int, default=40)
    parser.add_argument("--hold-bars", type=int, default=4)
    parser.add_argument("--binance-limit", type=int, default=300)
    parser.add_argument("--binance-period", default="1h")
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--per-symbol-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--summarize-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return native.now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def build_inner_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(SYSTEM_ROOT / "scripts" / "backtest_binance_indicator_combo_native_crypto.py"),
        "--review-dir",
        str(Path(args.review_dir).expanduser().resolve()),
        "--symbols",
        str(args.symbols),
        "--symbol-group",
        str(args.symbol_group),
        "--interval",
        str(args.interval),
        "--lookback-bars",
        str(int(args.lookback_bars)),
        "--sample-windows",
        str(int(args.sample_windows)),
        "--window-bars",
        str(int(args.window_bars)),
        "--hold-bars",
        str(int(args.hold_bars)),
        "--binance-limit",
        str(int(args.binance_limit)),
        "--binance-period",
        str(args.binance_period),
        "--rpm",
        str(int(args.rpm)),
        "--timeout-ms",
        str(int(args.timeout_ms)),
        "--per-symbol-timeout-seconds",
        str(float(args.per_symbol_timeout_seconds)),
        "--summarize-timeout-seconds",
        str(float(args.summarize_timeout_seconds)),
        "--artifact-ttl-hours",
        str(float(args.artifact_ttl_hours)),
        "--artifact-keep",
        str(int(args.artifact_keep)),
        "--now",
        native.fmt_utc(parse_now(args.now)) or "",
    ]


def excerpt(text: str, limit: int = 800) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3] + "..."


def build_guard_failure_payload(
    *,
    runtime_now: dt.datetime,
    args: argparse.Namespace,
    status: str,
    error_text: str,
    stdout_text: str = "",
    stderr_text: str = "",
    returncode: int | None = None,
) -> dict[str, Any]:
    completed_symbols: list[str] = []
    error_items = [
        {
            "stage": "subprocess_guard",
            "symbol": "",
            "status": status,
            "error": error_text,
        }
    ]
    measured_takeaway, practitioner_note = native.build_partial_takeaway(
        completed_symbols=completed_symbols,
        error_items=error_items,
    )
    return {
        "ok": False,
        "status": "partial_failure",
        "as_of": native.fmt_utc(runtime_now),
        "symbol_group": str(args.symbol_group),
        "interval": str(args.interval),
        "lookback_bars": int(args.lookback_bars),
        "sample_windows": int(args.sample_windows),
        "window_bars": int(args.window_bars),
        "hold_bars": int(args.hold_bars),
        "per_symbol_timeout_seconds": max(1.0, float(args.per_symbol_timeout_seconds)),
        "summarize_timeout_seconds": max(1.0, float(args.summarize_timeout_seconds)),
        "job_timeout_seconds": max(1.0, float(args.job_timeout_seconds)),
        "coverage": [],
        "completed_symbols": completed_symbols,
        "completed_symbol_count": 0,
        "timed_out_symbols": [],
        "timed_out_symbol_count": 0,
        "failed_symbols": [],
        "failed_symbol_count": 0,
        "error_items": error_items,
        "error_count": 1,
        "source_notes": [],
        "native_crypto_family": {"ranked_combos": [], "discarded_combos": []},
        "native_crypto_takeaway": measured_takeaway,
        "native_crypto_practitioner_note": practitioner_note,
        "control_note": "Guard wrapper terminated or rejected the inner native backtest process before a complete artifact was produced.",
        "guarded_command": build_inner_command(args),
        "guarded_returncode": returncode,
        "guarded_stdout_excerpt": excerpt(stdout_text),
        "guarded_stderr_excerpt": excerpt(stderr_text),
        "artifact_label": f"binance-indicator-combo-native-crypto:{str(args.symbol_group)}:partial_failure_guarded",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_now = parse_now(args.now)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    group_stem = native.artifact_stem_for_group(native.safe_group_slug(str(args.symbol_group)))
    cmd = build_inner_command(args)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SYSTEM_ROOT),
            text=True,
            capture_output=True,
            check=False,
            timeout=max(1.0, float(args.job_timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        payload = build_guard_failure_payload(
            runtime_now=runtime_now,
            args=args,
            status="timed_out",
            error_text=f"TimeoutError:job_timeout_seconds_exceeded:{max(1.0, float(args.job_timeout_seconds)):g}",
            stdout_text=str(exc.stdout or ""),
            stderr_text=str(exc.stderr or ""),
        )
        payload = native.write_artifacts(
            review_dir=review_dir,
            runtime_now=runtime_now,
            group_stem=group_stem,
            artifact_keep=int(args.artifact_keep),
            artifact_ttl_hours=float(args.artifact_ttl_hours),
            payload=payload,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = build_guard_failure_payload(
                runtime_now=runtime_now,
                args=args,
                status="failed",
                error_text="inner_script_returned_success_but_stdout_was_not_json",
                stdout_text=proc.stdout,
                stderr_text=proc.stderr,
                returncode=proc.returncode,
            )
            payload = native.write_artifacts(
                review_dir=review_dir,
                runtime_now=runtime_now,
                group_stem=group_stem,
                artifact_keep=int(args.artifact_keep),
                artifact_ttl_hours=float(args.artifact_ttl_hours),
                payload=payload,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    payload = build_guard_failure_payload(
        runtime_now=runtime_now,
        args=args,
        status="failed",
        error_text=f"inner_script_failed:returncode:{proc.returncode}",
        stdout_text=proc.stdout,
        stderr_text=proc.stderr,
        returncode=proc.returncode,
    )
    payload = native.write_artifacts(
        review_dir=review_dir,
        runtime_now=runtime_now,
        group_stem=group_stem,
        artifact_keep=int(args.artifact_keep),
        artifact_ttl_hours=float(args.artifact_ttl_hours),
        payload=payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
