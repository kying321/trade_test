#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
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
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def unwrap_operator_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("operator_handoff")
    return nested if isinstance(nested, dict) else payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def parse_json_payload(raw: str) -> dict[str, Any]:
    clean = text(raw)
    if not clean:
        return {}
    try:
        payload = json.loads(clean)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    for line in reversed([item.strip() for item in clean.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_openclaw_orderflow_executor_heartbeat.json",
        "*_openclaw_orderflow_executor_heartbeat.md",
        "*_openclaw_orderflow_executor_heartbeat_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def derive_runtime_boundary(mode: str) -> dict[str, Any]:
    effective_mode = text(mode).lower() or "shadow_guarded"
    if effective_mode == "shadow_guarded":
        return {
            "status": "shadow_runtime_only",
            "reason_codes": ["shadow_executor_only_mode"],
            "done_when": (
                "implement a source-owned non-shadow remote send/ack/fill runtime before promoting beyond "
                "shadow_guarded"
            ),
        }
    if effective_mode in {"spot_live_guarded", "live_guarded", "live_send_guarded"}:
        return {
            "status": "guarded_probe_only_runtime",
            "reason_codes": ["guarded_probe_only_mode"],
            "done_when": (
                "use guarded probe evidence for promotion checks, then implement source-owned non-shadow "
                "send/ack/fill behavior before routing capital"
            ),
        }
    return {
        "status": "unsupported_runtime_mode_source",
        "reason_codes": ["unsupported_executor_mode_source"],
        "done_when": "set a supported executor mode source or implement runtime support for the requested mode",
    }


def resolve_executor_mode(review_dir: Path, explicit_mode: str | None) -> tuple[str, str]:
    mode_text = text(explicit_mode).lower()
    if mode_text:
        return mode_text, "arg"

    contract_path = find_latest(review_dir, "*_remote_execution_contract_state.json")
    if contract_path and contract_path.exists():
        contract_payload = load_json_mapping(contract_path)
        contract_mode = text(contract_payload.get("executor_mode")).lower()
        if contract_mode:
            return contract_mode, "contract_state"

    handoff_path = find_latest(review_dir, "*_remote_live_handoff.json")
    if handoff_path and handoff_path.exists():
        handoff_payload = unwrap_operator_handoff(load_json_mapping(handoff_path))
        execution_contract = as_dict(handoff_payload.get("execution_contract"))
        handoff_mode = text(
            execution_contract.get("executor_mode")
            or handoff_payload.get("openclaw_orderflow_executor_mode")
        ).lower()
        if handoff_mode:
            return handoff_mode, "handoff"

    bridge_context_path = find_latest(review_dir, "*_remote_live_bridge_context.json")
    if bridge_context_path and bridge_context_path.exists():
        bridge_context_payload = load_json_mapping(bridge_context_path)
        bridge_mode = text(bridge_context_payload.get("openclaw_orderflow_executor_mode")).lower()
        if bridge_mode:
            return bridge_mode, "bridge_context"

    return "shadow_guarded", "default"


def build_guarded_exec_probe_request(*, intent_payload: dict[str, Any], mode: str) -> dict[str, Any]:
    effective_mode = text(mode).lower()
    if effective_mode != "spot_live_guarded":
        return {"requested": False, "reason": "mode_not_spot_live_guarded"}
    remote_market = text(intent_payload.get("remote_market")).lower()
    if remote_market != "spot":
        return {"requested": False, "reason": "remote_market_not_spot"}
    execution_contract_mode = text(intent_payload.get("execution_contract_mode"))
    execution_contract_reason_codes = {
        text(code) for code in as_list(intent_payload.get("execution_contract_reason_codes")) if text(code)
    }
    guarded_probe_allowed = bool(intent_payload.get("execution_contract_guarded_probe_allowed", False)) or (
        execution_contract_mode == "guarded_probe_only"
        or "guarded_probe_only_mode" in execution_contract_reason_codes
    )
    if (
        not guarded_probe_allowed
        and execution_contract_mode != "promotion_requested"
        and "requested_executor_mode_not_implemented" not in execution_contract_reason_codes
    ):
        return {"requested": False, "reason": "execution_contract_not_probe_ready"}
    ticket = as_dict(intent_payload.get("ticket_selected_row"))
    if not ticket:
        return {"requested": False, "reason": "ticket_selected_row_missing"}
    if not bool(ticket.get("allowed", False)):
        return {"requested": False, "reason": "ticket_not_allowed"}
    execution = as_dict(ticket.get("execution"))
    if text(execution.get("mode")) != "SPOT_LONG_OR_SELL":
        return {"requested": False, "reason": "unsupported_execution_mode"}
    signal = as_dict(ticket.get("signal"))
    signal_side = text(signal.get("side")).upper()
    if signal_side == "LONG":
        order_side = "BUY"
    elif signal_side == "SHORT":
        order_side = "SELL"
    else:
        return {"requested": False, "reason": "invalid_signal_side"}
    sizing = as_dict(ticket.get("sizing"))
    quote_usdt = to_float(sizing.get("quote_usdt", 0.0), 0.0)
    if quote_usdt <= 0.0:
        return {"requested": False, "reason": "invalid_quote_usdt"}
    symbol = text(ticket.get("symbol")) or text(intent_payload.get("preferred_route_symbol"))
    if not symbol:
        return {"requested": False, "reason": "symbol_missing"}
    return {
        "requested": True,
        "market": "spot",
        "symbol": symbol.upper(),
        "order_side": order_side,
        "quote_usdt": float(quote_usdt),
        "ticket_date": text(ticket.get("date")),
        "execution_mode": text(execution.get("mode")),
        "signal_side": signal_side,
    }


def run_guarded_exec_probe(
    *,
    system_root: Path,
    request_payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    override = text(os.getenv("OPENCLAW_GUARDED_EXEC_SCRIPT"))
    script_path = (
        Path(override).expanduser().resolve()
        if override
        else (system_root / "scripts" / "guarded_exec.py").resolve()
    )
    cmd = [
        sys.executable,
        str(script_path),
        "--workspace",
        str(system_root),
        "--mode",
        "probe",
        "--market",
        "spot",
        "--order-symbol",
        text(request_payload.get("symbol")),
        "--order-side",
        text(request_payload.get("order_side")),
        "--canary-quote-usdt",
        f"{to_float(request_payload.get('quote_usdt', 0.0), 0.0):.8f}",
        "--timeout-ms",
        "5000",
    ]
    ticket_date = text(request_payload.get("ticket_date"))
    if ticket_date:
        cmd.extend(["--date", ticket_date])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(system_root),
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "requested": True,
            "status": "probe_timeout",
            "timeout": True,
            "returncode": 124,
            "payload": {},
            "stdout_excerpt": text(exc.stdout)[:2000],
            "stderr_excerpt": text(exc.stderr)[:2000],
            "script": str(script_path),
            "request": request_payload,
        }
    payload = parse_json_payload(str(proc.stdout or ""))
    status = text(payload.get("status"))
    if not status:
        if int(proc.returncode) == 0:
            status = "probe_completed"
        else:
            status = "probe_error"
    return {
        "requested": True,
        "status": status,
        "timeout": False,
        "returncode": int(proc.returncode),
        "payload": payload,
        "stdout_excerpt": text(proc.stdout)[:2000],
        "stderr_excerpt": text(proc.stderr)[:2000],
        "script": str(script_path),
        "request": request_payload,
    }


def build_executor_status(
    intent_payload: dict[str, Any],
    *,
    mode: str,
    runtime_boundary: dict[str, Any],
    guarded_exec_probe: dict[str, Any],
) -> tuple[str, str]:
    queue_status = text(intent_payload.get("queue_status"))
    symbol = text(intent_payload.get("preferred_route_symbol")) or "-"
    remote_market = text(intent_payload.get("remote_market")) or "-"
    boundary_status = text(runtime_boundary.get("status"))
    probe_status = text(guarded_exec_probe.get("status"))
    if probe_status == "probe_completed":
        return (
            "spot_live_guarded_probe_completed",
            f"spot_live_guarded_probe_completed:{symbol}:{queue_status or '-'}:{remote_market}",
        )
    if probe_status in {"probe_controlled_block", "canary_controlled_block"} or probe_status.startswith("downgraded_probe_"):
        return (
            "spot_live_guarded_probe_controlled_block",
            f"spot_live_guarded_probe_controlled_block:{symbol}:{queue_status or '-'}:{remote_market}",
        )
    if probe_status == "probe_timeout":
        return (
            "spot_live_guarded_probe_timeout",
            f"spot_live_guarded_probe_timeout:{symbol}:{queue_status or '-'}:{remote_market}",
        )
    if probe_status in {"error", "panic", "probe_error"}:
        return (
            "spot_live_guarded_probe_error",
            f"spot_live_guarded_probe_error:{symbol}:{queue_status or '-'}:{remote_market}",
        )
    if boundary_status == "guarded_probe_only_runtime":
        if queue_status == "queued_guarded_probe_ready":
            return (
                "spot_live_guarded_probe_pending",
                f"spot_live_guarded_probe_pending:{symbol}:{queue_status or '-'}:{remote_market}",
            )
        return (
            "spot_live_guarded_probe_capable",
            f"spot_live_guarded_probe_capable:{symbol}:{queue_status or '-'}:{remote_market}",
        )
    if boundary_status == "requested_runtime_not_implemented":
        return (
            "requested_mode_not_implemented",
            f"requested_mode_not_implemented:{symbol}:{queue_status or '-'}:{remote_market}:{text(mode) or '-'}",
        )
    if boundary_status == "unsupported_runtime_mode_source":
        return (
            "unsupported_executor_mode_source",
            f"unsupported_executor_mode_source:{symbol}:{queue_status or '-'}:{remote_market}:{text(mode) or '-'}",
        )
    if queue_status == "queued_execution_contract_blocked":
        return (
            "shadow_guarded_contract_blocked",
            f"shadow_guarded_contract_blocked:{symbol}:{queue_status}:{remote_market}",
        )
    if queue_status == "queued_ticket_ready":
        return (
            "shadow_guarded_ticket_ready",
            f"shadow_guarded_ticket_ready:{symbol}:{queue_status}:{remote_market}",
        )
    if queue_status == "queued_wait_trade_readiness":
        return (
            "shadow_guarded_idle",
            f"shadow_guarded_idle:{symbol}:{queue_status}:{remote_market}",
        )
    return ("shadow_guarded_no_intent", f"shadow_guarded_no_intent:{symbol}:{queue_status or '-'}:{remote_market}")


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OpenClaw Orderflow Executor Heartbeat",
            "",
            f"- executor: `{text(payload.get('executor_brief'))}`",
            f"- idempotency: `{text(payload.get('idempotency_status'))}`",
            f"- intent: `{text(payload.get('intent_brief'))}`",
            f"- journal: `{text(payload.get('journal_brief'))}`",
            f"- risk verdict: `{text(payload.get('risk_verdict_brief'))}`",
            f"- action: `{text(payload.get('executor_action'))}`",
            "",
        ]
    )


def build_payload(
    *,
    intent_path: Path,
    intent_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    previous_heartbeat: dict[str, Any],
    reference_now: dt.datetime,
    mode: str,
    mode_source: str,
    guarded_exec_probe: dict[str, Any],
) -> dict[str, Any]:
    runtime_boundary = derive_runtime_boundary(mode)
    executor_status, executor_brief = build_executor_status(
        intent_payload,
        mode=mode,
        runtime_boundary=runtime_boundary,
        guarded_exec_probe=guarded_exec_probe,
    )
    last_entry_key = text(journal_payload.get("last_entry_key")) or text(journal_payload.get("entry_key"))
    previous_entry_key = text(previous_heartbeat.get("last_intent_key"))
    idempotency_status = "duplicate_intent_seen" if previous_entry_key and previous_entry_key == last_entry_key else "fresh_intent_observed"
    risk_blocker = {}
    for row in list(live_gate_payload.get("blockers") or []):
        if isinstance(row, dict) and text(row.get("name")) == "risk_guard":
            risk_blocker = dict(row)
            break
    queue_status = text(intent_payload.get("queue_status"))
    probe_status = text(guarded_exec_probe.get("status"))
    if probe_status == "probe_completed":
        executor_action = "guarded_probe_completed"
    elif probe_status in {"probe_controlled_block", "canary_controlled_block"} or probe_status.startswith("downgraded_probe_"):
        executor_action = "guarded_probe_controlled_block"
    elif probe_status == "probe_timeout":
        executor_action = "guarded_probe_timeout"
    elif probe_status in {"error", "panic", "probe_error"}:
        executor_action = "guarded_probe_error"
    elif text(guarded_exec_probe.get("status")) == "skipped_duplicate_intent":
        executor_action = "guarded_probe_duplicate_skip"
    elif text(runtime_boundary.get("status")) == "guarded_probe_only_runtime":
        if queue_status == "queued_guarded_probe_ready":
            executor_action = "guarded_probe_pending"
        else:
            executor_action = "idle_guarded_probe_only_runtime"
    elif text(runtime_boundary.get("status")) == "requested_runtime_not_implemented":
        executor_action = "idle_requested_mode_not_implemented"
    elif text(runtime_boundary.get("status")) == "unsupported_runtime_mode_source":
        executor_action = "idle_unsupported_mode_source"
    elif queue_status == "queued_execution_contract_blocked":
        executor_action = "idle_execution_contract_blocked"
    elif queue_status == "queued_ticket_ready":
        executor_action = "shadow_ready_wait_guard_clear"
    elif queue_status == "queued_wait_trade_readiness":
        executor_action = "observe_only_no_transport"
    else:
        executor_action = "idle_no_intent"
    return {
        "action": "openclaw_orderflow_executor_heartbeat",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "executor_mode": mode,
        "executor_mode_source": mode_source,
        "executor_runtime_boundary_status": text(runtime_boundary.get("status")),
        "executor_runtime_boundary_reason_codes": runtime_boundary.get("reason_codes") or [],
        "executor_status": executor_status,
        "executor_brief": executor_brief,
        "executor_action": executor_action,
        "idempotency_key": last_entry_key or "-",
        "idempotency_status": idempotency_status,
        "intent_brief": text(intent_payload.get("queue_brief")),
        "intent_status": queue_status,
        "intent_symbol": text(intent_payload.get("preferred_route_symbol")),
        "intent_action": text(intent_payload.get("preferred_route_action")),
        "journal_brief": text(journal_payload.get("journal_brief")),
        "journal_status": text(journal_payload.get("journal_status")),
        "last_intent_key": last_entry_key,
        "risk_verdict_brief": text(journal_payload.get("risk_verdict_brief"))
        or ":".join(
            [
                text(risk_blocker.get("status")) or "unknown",
                ",".join([text(code) for code in list(risk_blocker.get("reason_codes") or []) if text(code)]),
            ]
        ).strip(":"),
        "fill_status": text(journal_payload.get("fill_status")) or "no_fill_execution_not_attempted",
        "guarded_exec_probe_status": probe_status,
        "guarded_exec_probe_returncode": int(guarded_exec_probe.get("returncode", 0) or 0),
        "guarded_exec_probe_timeout": bool(guarded_exec_probe.get("timeout", False)),
        "guarded_exec_probe_artifact": text(
            as_dict(guarded_exec_probe.get("payload")).get("artifact")
        ),
        "guarded_exec_probe_script": text(guarded_exec_probe.get("script")),
        "guarded_exec_probe_request": as_dict(guarded_exec_probe.get("request")),
        "blocker_detail": text(journal_payload.get("blocker_detail")) or text(intent_payload.get("blocker_detail")),
        "done_when": text(runtime_boundary.get("done_when"))
        or text(journal_payload.get("done_when"))
        or text(intent_payload.get("done_when")),
        "artifacts": {
            "remote_intent_queue": str(intent_path),
            "remote_execution_journal": str(journal_path),
            "live_gate_blocker_report": str(live_gate_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the guarded shadow OpenClaw orderflow executor.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--executor-timeout-seconds", type=int, default=5)
    parser.add_argument("--mode", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-loops", type=int, default=0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    loops = 1 if args.once else max(0, int(args.max_loops))
    keep_running = not args.once and loops == 0
    iteration = 0
    final_payload: dict[str, Any] = {}

    while True:
        current_now = reference_now + dt.timedelta(seconds=iteration)
        intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
        journal_path = find_latest(review_dir, "*_remote_execution_journal.json")
        live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
        previous_heartbeat_path = find_latest(review_dir, "*_openclaw_orderflow_executor_heartbeat.json")
        missing = [
            name
            for name, path in (
                ("remote_intent_queue", intent_path),
                ("remote_execution_journal", journal_path),
                ("live_gate_blocker_report", live_gate_path),
            )
            if path is None
        ]
        if missing:
            raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

        intent_payload = load_json_mapping(intent_path)
        journal_payload = load_json_mapping(journal_path)
        live_gate_payload = load_json_mapping(live_gate_path)
        previous_heartbeat = (
            load_json_mapping(previous_heartbeat_path)
            if previous_heartbeat_path is not None and previous_heartbeat_path.exists()
            else {}
        )
        executor_mode, executor_mode_source = resolve_executor_mode(review_dir, args.mode)
        last_entry_key = text(journal_payload.get("last_entry_key")) or text(journal_payload.get("entry_key"))
        duplicate_intent = bool(last_entry_key) and text(previous_heartbeat.get("last_intent_key")) == last_entry_key
        guarded_exec_probe_request = build_guarded_exec_probe_request(
            intent_payload=intent_payload,
            mode=executor_mode,
        )
        if duplicate_intent and bool(guarded_exec_probe_request.get("requested", False)):
            guarded_exec_probe = {
                "requested": False,
                "status": "skipped_duplicate_intent",
                "timeout": False,
                "returncode": 0,
                "payload": {},
                "script": "",
                "request": guarded_exec_probe_request,
            }
        elif bool(guarded_exec_probe_request.get("requested", False)):
            guarded_exec_probe = run_guarded_exec_probe(
                system_root=SYSTEM_ROOT,
                request_payload=guarded_exec_probe_request,
                timeout_seconds=max(1.0, float(args.executor_timeout_seconds)),
            )
        else:
            guarded_exec_probe = {
                "requested": False,
                "status": text(guarded_exec_probe_request.get("reason")) or "not_requested",
                "timeout": False,
                "returncode": 0,
                "payload": {},
                "script": "",
                "request": guarded_exec_probe_request,
            }

        payload = build_payload(
            intent_path=intent_path,
            intent_payload=intent_payload,
            journal_path=journal_path,
            journal_payload=journal_payload,
            live_gate_path=live_gate_path,
            live_gate_payload=live_gate_payload,
            previous_heartbeat=previous_heartbeat,
            reference_now=current_now,
            mode=executor_mode,
            mode_source=executor_mode_source,
            guarded_exec_probe=guarded_exec_probe,
        )

        stamp = current_now.strftime("%Y%m%dT%H%M%SZ")
        artifact = review_dir / f"{stamp}_openclaw_orderflow_executor_heartbeat.json"
        markdown = review_dir / f"{stamp}_openclaw_orderflow_executor_heartbeat.md"
        checksum = review_dir / f"{stamp}_openclaw_orderflow_executor_heartbeat_checksum.json"

        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        markdown.write_text(render_markdown(payload), encoding="utf-8")
        checksum.write_text(
            json.dumps(
                {
                    "artifact": str(artifact),
                    "artifact_sha256": sha256_file(artifact),
                    "markdown": str(markdown),
                    "markdown_sha256": sha256_file(markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        pruned_keep, pruned_age = prune_artifacts(
            review_dir,
            current_paths=[artifact, markdown, checksum],
            keep=max(1, int(args.artifact_keep)),
            ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
        )
        payload.update(
            {
                "artifact": str(artifact),
                "markdown": str(markdown),
                "checksum": str(checksum),
                "pruned_keep": pruned_keep,
                "pruned_age": pruned_age,
            }
        )
        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checksum.write_text(
            json.dumps(
                {
                    "artifact": str(artifact),
                    "artifact_sha256": sha256_file(artifact),
                    "markdown": str(markdown),
                    "markdown_sha256": sha256_file(markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        final_payload = payload
        iteration += 1
        if args.once:
            break
        if loops and iteration >= loops:
            break
        if not keep_running and iteration >= 1:
            break
        time.sleep(max(1, int(args.poll_seconds)))

    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
