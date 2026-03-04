#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


@contextmanager
def _run_halfhour_mutex(output_root: Path, timeout_seconds: float):
    lock_path = output_root / "state" / "run-halfhour-pulse.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = lock_path.open("a+", encoding="utf-8")
    try:
        import fcntl

        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"run-halfhour-pulse mutex timeout: {timeout_seconds:.1f}s")
                time.sleep(0.1)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fd.close()


def _run_step(
    *,
    name: str,
    cmd: list[str],
    timeout_seconds: int,
    logs_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = logs_dir / f"review_cycle_guarded_{ts}_{name}.out.log"
    err_log = logs_dir / f"review_cycle_guarded_{ts}_{name}.err.log"
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        out_log.write_text(stdout, encoding="utf-8")
        err_log.write_text(stderr, encoding="utf-8")
        return {
            "name": name,
            "timed_out": False,
            "returncode": int(proc.returncode),
            "elapsed_sec": round(time.time() - t0, 3),
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "stdout_log": str(out_log),
            "stderr_log": str(err_log),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        out_log.write_text(stdout, encoding="utf-8")
        err_log.write_text(stderr, encoding="utf-8")
        return {
            "name": name,
            "timed_out": True,
            "returncode": None,
            "elapsed_sec": round(time.time() - t0, 3),
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "stdout_log": str(out_log),
            "stderr_log": str(err_log),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run review-cycle in guarded split steps with per-step timeouts.")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--output-root", default="output", help="Output root path.")
    parser.add_argument("--ops-window-days", type=int, default=14)
    parser.add_argument("--review-timeout-seconds", type=int, default=180)
    parser.add_argument("--gate-timeout-seconds", type=int, default=120)
    parser.add_argument("--ops-timeout-seconds", type=int, default=120)
    parser.add_argument("--mutex-timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    as_of = date.fromisoformat(str(args.date))
    output_root = Path(args.output_root).resolve()
    logs_dir = output_root / "logs"
    review_dir = output_root / "review"
    logs_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    steps = [
        (
            "review",
            [
                "python",
                "-m",
                "lie_engine.cli",
                "--config",
                str(args.config),
                "review",
                "--date",
                as_of.isoformat(),
            ],
            int(args.review_timeout_seconds),
        ),
        (
            "gate_report",
            [
                "python",
                "-m",
                "lie_engine.cli",
                "--config",
                str(args.config),
                "gate-report",
                "--date",
                as_of.isoformat(),
            ],
            int(args.gate_timeout_seconds),
        ),
        (
            "ops_report",
            [
                "python",
                "-m",
                "lie_engine.cli",
                "--config",
                str(args.config),
                "ops-report",
                "--date",
                as_of.isoformat(),
                "--window-days",
                str(int(args.ops_window_days)),
            ],
            int(args.ops_timeout_seconds),
        ),
    ]

    results: list[dict[str, Any]] = []
    started_at = datetime.now().isoformat()
    with _run_halfhour_mutex(output_root=output_root, timeout_seconds=float(args.mutex_timeout_seconds)):
        for name, cmd, timeout_s in steps:
            results.append(
                _run_step(
                    name=name,
                    cmd=cmd,
                    timeout_seconds=timeout_s,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

    ok = all((not bool(x.get("timed_out", False))) and int(x.get("returncode", 1)) == 0 for x in results)
    payload = {
        "date": as_of.isoformat(),
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(),
        "ok": bool(ok),
        "steps": results,
    }
    out_path = review_dir / f"{as_of.isoformat()}_review_cycle_guarded.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
