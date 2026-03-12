#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
DEFAULT_WORKSPACE_SYSTEM_ROOT = Path.home() / ".openclaw" / "workspaces" / "pi" / "fenlie-system"
DEFAULT_LAB_PARENT = DEFAULT_WORKSPACE_SYSTEM_ROOT / "output" / "review"
DEFAULT_RUNTIME_CORE = SYSTEM_ROOT / "runtime" / "pi" / "scripts" / "lie_spot_halfhour_core.py"


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


def fmt_utc(ts: dt.datetime | None) -> str | None:
    if ts is None:
        return None
    return ts.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(cmd: list[str], *, env: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except Exception:
        payload = {
            "ok": False,
            "status": "invalid_json_stdout",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    payload.setdefault("returncode", proc.returncode)
    if proc.stderr.strip():
        payload.setdefault("stderr", proc.stderr.strip())
    return proc.returncode, payload


def load_runtime_module(runtime_script_path: Path):
    runtime_dir = str(runtime_script_path.parent)
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    module_name = "local_pi_recovery_lab_runtime_core"
    spec = importlib.util.spec_from_file_location(module_name, runtime_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"runtime_core_spec_missing:{runtime_script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def prune_lab_dirs(parent: Path, *, current_dir: Path, keep: int, ttl_hours: float) -> dict[str, list[str]]:
    parent.mkdir(parents=True, exist_ok=True)
    dirs = sorted(
        [p for p in parent.glob("*_local_pi_recovery_lab") if p.is_dir()],
        key=lambda p: p.name,
    )
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=max(1.0, ttl_hours))
    pruned_keep: list[str] = []
    pruned_age: list[str] = []
    survivors = [p for p in dirs if p != current_dir]
    while len(survivors) > max(0, keep - 1):
        stale = survivors.pop(0)
        shutil.rmtree(stale, ignore_errors=True)
        pruned_keep.append(str(stale))
    for candidate in list(survivors):
        try:
            mtime = dt.datetime.fromtimestamp(candidate.stat().st_mtime, tz=dt.timezone.utc)
        except Exception:
            continue
        if mtime < cutoff:
            shutil.rmtree(candidate, ignore_errors=True)
            pruned_age.append(str(candidate))
    return {"pruned_keep": pruned_keep, "pruned_age": pruned_age}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated local PI recovery lab using copied paper state and verify ack archive closure without mutating the real workspace."
    )
    parser.add_argument("--workspace-system-root", default=str(DEFAULT_WORKSPACE_SYSTEM_ROOT))
    parser.add_argument("--lab-parent-dir", default=str(DEFAULT_LAB_PARENT))
    parser.add_argument("--runtime-core-path", default=str(DEFAULT_RUNTIME_CORE))
    parser.add_argument("--artifact-ttl-hours", type=float, default=72.0)
    parser.add_argument("--keep-labs", type=int, default=6)
    parser.add_argument("--allow-fallback-write", action="store_true")
    parser.add_argument("--cycle-ts", default="")
    parser.add_argument("--now", default="")
    return parser


def classify_lab_status(out: dict[str, Any]) -> str:
    status = str(out.get("status") or "").strip()
    if status == "ok":
        return "lab-ok"
    if status == "missing_required_inputs":
        return "missing-inputs"
    if status in {
        "initial_status_failed",
        "post_backfill_status_failed",
        "post_ack_status_failed",
        "archive_visibility_failed",
        "runtime_consume_failed",
    }:
        return "lab-failed"
    if status in {"fallback_write_disabled", "ack_not_ready_after_lab_write"}:
        return "lab-blocked"
    if status in {"backfill_failed", "ack_failed"}:
        return "lab-write-failed"
    return status or "unknown"


def build_projection_validation(out: dict[str, Any]) -> dict[str, Any]:
    initial_status = out.get("initial_status") if isinstance(out.get("initial_status"), dict) else {}
    write_projection = (
        initial_status.get("write_projection")
        if isinstance(initial_status.get("write_projection"), dict)
        else {}
    )
    projected_steps = (
        write_projection.get("projected_steps")
        if isinstance(write_projection.get("projected_steps"), list)
        else []
    )
    actual_steps: list[dict[str, Any]] = []

    backfill_result = out.get("backfill_result") if isinstance(out.get("backfill_result"), dict) else {}
    if bool(backfill_result.get("write_performed")):
        selected_method = str(backfill_result.get("selected_method") or "").strip()
        simulated_step = (
            "strict_backfill_write"
            if selected_method == "ledger_tail_streak_exact"
            else "fallback_backfill_write"
        )
        actual_steps.append(
            {
                "simulated_step": simulated_step,
                "selected_method": selected_method or None,
                "selected_last_loss_ts": backfill_result.get("selected_last_loss_ts"),
            }
        )

    ack_result = out.get("ack_result") if isinstance(out.get("ack_result"), dict) else {}
    if bool(ack_result.get("write_performed")):
        actual_steps.append(
            {
                "simulated_step": "ack_write",
                "ack_path": ack_result.get("ack_path"),
            }
        )

    runtime_consume = out.get("runtime_consume") if isinstance(out.get("runtime_consume"), dict) else {}
    if bool(runtime_consume.get("consume_ok")) and bool(runtime_consume.get("archive_ok")):
        actual_steps.append(
            {
                "simulated_step": "ack_consumed_archive_ok",
                "archive_reason": runtime_consume.get("archive_reason"),
            }
        )

    compared_steps: list[dict[str, Any]] = []
    full_sequence_match = True
    for idx, actual in enumerate(actual_steps):
        projected = projected_steps[idx] if idx < len(projected_steps) else None
        projected_name = (
            str(projected.get("simulated_step") or "").strip()
            if isinstance(projected, dict)
            else None
        )
        actual_name = str(actual.get("simulated_step") or "").strip()
        step_match = projected_name == actual_name
        if not step_match:
            full_sequence_match = False
        compared_steps.append(
            {
                "index": idx,
                "projected_simulated_step": projected_name,
                "actual_simulated_step": actual_name,
                "match": step_match,
            }
        )

    write_step_names = {"strict_backfill_write", "fallback_backfill_write", "ack_write"}
    actual_write_steps = [
        str(step.get("simulated_step") or "").strip()
        for step in actual_steps
        if str(step.get("simulated_step") or "").strip() in write_step_names
    ]
    projected_write_steps = [
        str(step.get("simulated_step") or "").strip()
        for step in projected_steps
        if isinstance(step, dict) and str(step.get("simulated_step") or "").strip() in write_step_names
    ][: len(actual_write_steps)]
    write_prefix_match = actual_write_steps == projected_write_steps

    terminal_action = str(write_projection.get("terminal_action") or "").strip() or None
    final_status = out.get("final_status") if isinstance(out.get("final_status"), dict) else {}
    final_recovery_plan = (
        final_status.get("recovery_plan")
        if isinstance(final_status.get("recovery_plan"), dict)
        else {}
    )
    final_next_action = str(final_recovery_plan.get("next_action") or "").strip() or None
    terminal_reached = bool(actual_steps) and terminal_action == "full_cycle_gate_ready" and any(
        step.get("simulated_step") == "ack_consumed_archive_ok" for step in actual_steps
    )
    terminal_reason = (
        "lab_validated_backfill_ack_and_archive_but_not_full_cycle"
        if terminal_reached and terminal_action == "full_cycle_gate_ready"
        else "projection_not_fully_exercised"
    )

    return {
        "writes_enabled_assumption": bool(write_projection.get("writes_enabled_assumption")),
        "fallback_write_enabled_assumption": bool(
            write_projection.get("fallback_write_enabled_assumption")
        ),
        "projected_step_count": len(projected_steps),
        "actual_step_count": len(actual_steps),
        "projected_terminal_action": terminal_action,
        "final_next_action": final_next_action,
        "prefix_match": write_prefix_match,
        "full_sequence_match": full_sequence_match,
        "compared_steps": compared_steps,
        "projected_steps": projected_steps,
        "actual_steps": actual_steps,
        "terminal_reached": terminal_reached,
        "terminal_reason": terminal_reason,
    }


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = Path(args.workspace_system_root).expanduser().resolve()
    lab_parent = Path(args.lab_parent_dir).expanduser().resolve()
    runtime_core_path = Path(args.runtime_core_path).expanduser().resolve()
    now_utc = parse_ts(args.now) or dt.datetime.now(dt.timezone.utc)
    cycle_ts = fmt_utc(parse_ts(args.cycle_ts) or now_utc) or now_utc.isoformat()
    stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    lab_root = lab_parent / f"{stamp}_local_pi_recovery_lab"
    lab_state_dir = lab_root / "output" / "state"
    lab_log_dir = lab_root / "output" / "logs"
    lab_review_dir = lab_root / "output" / "review"
    lab_archive_dir = lab_review_dir / "paper_consecutive_loss_ack_archive"
    lab_state_dir.mkdir(parents=True, exist_ok=True)
    lab_log_dir.mkdir(parents=True, exist_ok=True)
    lab_archive_dir.mkdir(parents=True, exist_ok=True)

    source_state_path = workspace_root / "output" / "state" / "spot_paper_state.json"
    source_ack_path = workspace_root / "output" / "state" / "paper_consecutive_loss_ack.json"
    source_checksum_path = workspace_root / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"
    source_ledger_path = workspace_root / "output" / "logs" / "paper_execution_ledger.jsonl"

    lab_state_path = lab_state_dir / "spot_paper_state.json"
    lab_ack_path = lab_state_dir / "paper_consecutive_loss_ack.json"
    lab_checksum_path = lab_state_dir / "paper_consecutive_loss_ack_checksum.json"
    lab_pulse_lock_path = lab_state_dir / "run_halfhour_pulse.lock"
    lab_ledger_path = lab_log_dir / "paper_execution_ledger.jsonl"
    lab_archive_manifest_path = lab_archive_dir / "manifest.jsonl"

    copied_inputs = {
        "state": copy_if_exists(source_state_path, lab_state_path),
        "ack": copy_if_exists(source_ack_path, lab_ack_path),
        "ack_checksum": copy_if_exists(source_checksum_path, lab_checksum_path),
        "ledger": copy_if_exists(source_ledger_path, lab_ledger_path),
    }

    out: dict[str, Any] = {
        "action": "run_local_pi_recovery_lab",
        "ok": False,
        "status": "initializing",
        "artifact_status_label": None,
        "artifact_label": None,
        "artifact_tags": ["local-pi", "recovery-lab", "initializing"],
        "generated_at": fmt_utc(now_utc),
        "cycle_ts": cycle_ts,
        "workspace_system_root": str(workspace_root),
        "lab_root": str(lab_root),
        "copied_inputs": copied_inputs,
        "allow_fallback_write": bool(args.allow_fallback_write),
        "initial_status": None,
        "backfill_result": None,
        "post_backfill_status": None,
        "ack_result": None,
        "post_ack_status": None,
        "runtime_consume": None,
        "final_archive_status": None,
        "final_status": None,
        "projection_validation": None,
        "operator_note": None,
        "artifact": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }

    if not copied_inputs["state"] or not copied_inputs["ledger"]:
        out["status"] = "missing_required_inputs"
        out["artifact_status_label"] = classify_lab_status(out)
        out["artifact_label"] = f"recovery-lab:{out['status']}"
        out["artifact_tags"] = [
            "local-pi",
            "recovery-lab",
            str(out["status"] or "unknown"),
            str(out["artifact_status_label"] or "unknown"),
        ]
        report_path = lab_root / "report.json"
        checksum_path = lab_root / "report_checksum.json"
        report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        digest = sha256_file(report_path)
        checksum_path.write_text(
            json.dumps(
                {
                    "generated_at": fmt_utc(now_utc),
                    "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                    "files": [
                        {
                            "path": str(report_path),
                            "sha256": digest,
                            "size_bytes": int(report_path.stat().st_size),
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        out["artifact"] = str(report_path)
        out["checksum"] = str(checksum_path)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    base_env = os.environ.copy()
    base_env["FENLIE_SYSTEM_ROOT"] = str(SYSTEM_ROOT)
    base_env["LIE_SYSTEM_ROOT"] = str(SYSTEM_ROOT)

    status_cmd = [
        sys.executable,
        str(SYSTEM_ROOT / "scripts" / "paper_consecutive_loss_guardrail_status.py"),
        "--state-path",
        str(lab_state_path),
        "--ack-path",
        str(lab_ack_path),
        "--checksum-path",
        str(lab_checksum_path),
        "--ledger-path",
        str(lab_ledger_path),
        "--archive-dir",
        str(lab_archive_dir),
        "--archive-manifest-path",
        str(lab_archive_manifest_path),
        "--now",
        fmt_utc(now_utc) or "",
    ]
    rc, initial_status = run_json(status_cmd, env=base_env)
    out["initial_status"] = initial_status
    if rc != 0 or not bool(initial_status.get("ok")):
        out["status"] = "initial_status_failed"
    else:
        next_action = str(initial_status.get("recovery_plan", {}).get("next_action") or "").strip()
        status_for_ack = initial_status

        if next_action in {"write_strict_last_loss_ts_backfill", "review_fallback_last_loss_ts_backfill"}:
            backfill_cmd = [
                sys.executable,
                str(SYSTEM_ROOT / "scripts" / "backfill_paper_last_loss_ts.py"),
                "--state-path",
                str(lab_state_path),
                "--ledger-path",
                str(lab_ledger_path),
                "--pulse-lock-path",
                str(lab_pulse_lock_path),
                "--expected-state-fingerprint",
                str(initial_status.get("state_fingerprint") or ""),
                "--write",
            ]
            if next_action == "review_fallback_last_loss_ts_backfill":
                if not bool(args.allow_fallback_write):
                    out["status"] = "fallback_write_disabled"
                else:
                    backfill_cmd.append("--allow-latest-loss-fallback")
            if out["status"] == "initializing":
                rc, backfill_result = run_json(backfill_cmd, env=base_env)
                out["backfill_result"] = backfill_result
                if rc != 0 or not bool(backfill_result.get("ok")) or not bool(backfill_result.get("write_performed")):
                    out["status"] = "backfill_failed"
                else:
                    rc, post_backfill_status = run_json(status_cmd, env=base_env)
                    out["post_backfill_status"] = post_backfill_status
                    if rc != 0 or not bool(post_backfill_status.get("ok")):
                        out["status"] = "post_backfill_status_failed"
                    else:
                        status_for_ack = post_backfill_status
        if out["status"] == "initializing":
            ack_ready = bool(status_for_ack.get("ack", {}).get("eligible_for_apply_now"))
            next_action_for_ack = str(status_for_ack.get("recovery_plan", {}).get("next_action") or "").strip()
            if not ack_ready and next_action_for_ack == "write_manual_ack":
                ack_cmd = [
                    sys.executable,
                    str(SYSTEM_ROOT / "scripts" / "ack_paper_consecutive_loss_guardrail.py"),
                    "--state-path",
                    str(lab_state_path),
                    "--ack-path",
                    str(lab_ack_path),
                    "--checksum-path",
                    str(lab_checksum_path),
                    "--pulse-lock-path",
                    str(lab_pulse_lock_path),
                    "--ttl-hours",
                    "24",
                    "--cooldown-hours",
                    "12",
                    "--expected-state-fingerprint",
                    str(status_for_ack.get("state_fingerprint") or ""),
                    "--write",
                ]
                rc, ack_result = run_json(ack_cmd, env=base_env)
                out["ack_result"] = ack_result
                if rc != 0 or not bool(ack_result.get("ok")) or not bool(ack_result.get("write_performed")):
                    out["status"] = "ack_failed"
                else:
                    rc, post_ack_status = run_json(status_cmd, env=base_env)
                    out["post_ack_status"] = post_ack_status
                    if rc != 0 or not bool(post_ack_status.get("ok")):
                        out["status"] = "post_ack_status_failed"
                    else:
                        status_for_ack = post_ack_status
                        ack_ready = bool(status_for_ack.get("ack", {}).get("eligible_for_apply_now"))

            if out["status"] == "initializing":
                if not ack_ready:
                    out["status"] = "ack_not_ready_after_lab_write"
                else:
                    try:
                        runtime_mod = load_runtime_module(runtime_core_path)
                        runtime_mod.PAPER_CONSECUTIVE_LOSS_ACK_PATH = lab_ack_path
                        runtime_mod.PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH = lab_checksum_path
                        runtime_mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR = lab_archive_dir
                        runtime_mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH = lab_archive_manifest_path
                        runtime_mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES = max(1, int(args.keep_labs))
                        consume_result = runtime_mod.consume_consecutive_loss_ack(
                            ack_state={"applied": True},
                            cycle_ts=cycle_ts,
                        )
                        out["runtime_consume"] = consume_result
                    except Exception as exc:
                        out["runtime_consume"] = {
                            "ok": False,
                            "status": f"runtime_consume_failed:{exc.__class__.__name__}",
                        }
                        out["status"] = "runtime_consume_failed"

        archive_cmd = [
            sys.executable,
            str(SYSTEM_ROOT / "scripts" / "paper_consecutive_loss_ack_archive_status.py"),
            "--ack-path",
            str(lab_ack_path),
            "--checksum-path",
            str(lab_checksum_path),
            "--archive-dir",
            str(lab_archive_dir),
            "--archive-manifest-path",
            str(lab_archive_manifest_path),
            "--now",
            fmt_utc(now_utc) or "",
        ]
        _, final_archive_status = run_json(archive_cmd, env=base_env)
        out["final_archive_status"] = final_archive_status
        _, final_status = run_json(status_cmd, env=base_env)
        out["final_status"] = final_status
        out["projection_validation"] = build_projection_validation(out)

        if out["status"] == "initializing":
            archive_latest = final_archive_status.get("archive", {}).get("latest_archive")
            consume_ok = bool((out.get("runtime_consume") or {}).get("consume_ok"))
            archive_ok = bool((out.get("runtime_consume") or {}).get("archive_ok"))
            if consume_ok and archive_ok and isinstance(archive_latest, dict):
                out["status"] = "ok"
                out["ok"] = True
            else:
                out["status"] = "archive_visibility_failed"

    if bool(out.get("ok")):
        projection_validation = (
            out.get("projection_validation")
            if isinstance(out.get("projection_validation"), dict)
            else {}
        )
        out["operator_note"] = {
            "summary": "Isolated recovery lab validated fallback backfill, manual ack write, and runtime ack consume/archive without touching the real workspace.",
            "when_to_use": "Use this lab result as the last dry safety check before any real workspace recovery write.",
            "recommended_first_action": "Review projection_validation and confirm the fallback backfill candidate before enabling real writes.",
            "escalation_action": "If you proceed to real writes, keep auto snapshot enabled and use the generated rollback commands if post-write verification deviates.",
            "projection_prefix_match": bool(projection_validation.get("prefix_match")),
            "terminal_reason": projection_validation.get("terminal_reason"),
        }

    prune_result = prune_lab_dirs(
        lab_parent,
        current_dir=lab_root,
        keep=max(1, int(args.keep_labs)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    out["pruned_keep"] = prune_result["pruned_keep"]
    out["pruned_age"] = prune_result["pruned_age"]
    out["artifact_status_label"] = classify_lab_status(out)
    out["artifact_label"] = f"recovery-lab:{out['status']}"
    out["artifact_tags"] = [
        "local-pi",
        "recovery-lab",
        str(out["status"] or "unknown"),
        str(out["artifact_status_label"] or "unknown"),
    ]

    report_path = lab_root / "report.json"
    checksum_path = lab_root / "report_checksum.json"
    out["artifact"] = str(report_path)
    out["checksum"] = str(checksum_path)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(report_path)
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": fmt_utc(now_utc),
                "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                "files": [
                    {
                        "path": str(report_path),
                        "sha256": digest,
                        "size_bytes": int(report_path.stat().st_size),
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
