#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
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
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


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


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "*_openclaw_orderflow_executor_state.json",
        "*_openclaw_orderflow_executor_state.md",
        "*_openclaw_orderflow_executor_state_checksum.json",
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


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OpenClaw Orderflow Executor State",
            "",
            f"- brief: `{text(payload.get('executor_brief'))}`",
            f"- service: `{text(payload.get('service_name'))}`",
            f"- heartbeat: `{text(payload.get('heartbeat_status'))}`",
            f"- unit preview: `{text(payload.get('unit_preview_path')) or '-'}`",
            f"- intent: `{text(payload.get('intent_brief'))}`",
            f"- journal: `{text(payload.get('journal_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    identity_path: Path,
    identity_payload: dict[str, Any],
    intent_path: Path,
    intent_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    heartbeat_path: Path | None,
    heartbeat_payload: dict[str, Any],
    unit_preview_path: Path | None,
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = text(intent_payload.get("preferred_route_symbol"))
    queue_status = text(intent_payload.get("queue_status"))
    remote_market = text(intent_payload.get("remote_market")) or text(
        identity_payload.get("ready_check_scope_market")
    ) or "portfolio_margin_um"
    heartbeat_status = text(heartbeat_payload.get("executor_status"))
    heartbeat_mode = text(heartbeat_payload.get("executor_mode")) or "shadow_guarded"
    heartbeat_mode_source = text(heartbeat_payload.get("executor_mode_source")) or "heartbeat"
    runtime_boundary_status = text(heartbeat_payload.get("executor_runtime_boundary_status"))
    runtime_boundary_reason_codes = [
        text(code)
        for code in list(heartbeat_payload.get("executor_runtime_boundary_reason_codes") or [])
        if text(code)
    ]
    guarded_exec_probe_status = text(heartbeat_payload.get("guarded_exec_probe_status"))
    guarded_exec_probe_artifact = text(heartbeat_payload.get("guarded_exec_probe_artifact"))
    has_unit_preview = unit_preview_path is not None and unit_preview_path.exists()
    if heartbeat_status in {
        "spot_live_guarded_probe_completed",
        "spot_live_guarded_probe_controlled_block",
        "spot_live_guarded_probe_timeout",
        "spot_live_guarded_probe_error",
        "spot_live_guarded_probe_pending",
        "spot_live_guarded_probe_capable",
    }:
        executor_status = heartbeat_status
    elif runtime_boundary_status == "guarded_probe_only_runtime":
        executor_status = "spot_live_guarded_probe_capable"
    elif runtime_boundary_status == "requested_runtime_not_implemented":
        executor_status = "executor_runtime_boundary_blocked"
    elif runtime_boundary_status == "unsupported_runtime_mode_source":
        executor_status = "executor_runtime_mode_source_invalid"
    elif heartbeat_status and has_unit_preview:
        executor_status = "shadow_guarded_executor_ready"
    elif has_unit_preview:
        executor_status = "shadow_guarded_unit_rendered"
    else:
        executor_status = "shadow_guarded_scaffold_pending_unit"
    executor_brief = ":".join([executor_status, symbol or "-", queue_status or "-", remote_market])
    done_when = (
        "executor heartbeat stays healthy, unit preview is rendered, and future queued_ticket_ready intents can flow into a dedicated actor without bypassing guardian veto"
    )
    return {
        "action": "build_openclaw_orderflow_executor_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "executor_status": executor_status,
        "executor_brief": executor_brief,
        "service_name": "openclaw-orderflow-executor.service",
        "service_mode": heartbeat_mode,
        "service_mode_source": heartbeat_mode_source,
        "runtime_boundary_status": runtime_boundary_status,
        "runtime_boundary_reason_codes": runtime_boundary_reason_codes,
        "entrypoint_script": str(SYSTEM_ROOT / "scripts" / "openclaw_orderflow_executor.py"),
        "renderer_script": str(SYSTEM_ROOT / "scripts" / "render_openclaw_orderflow_executor_unit.py"),
        "unit_preview_path": str(unit_preview_path) if unit_preview_path else "",
        "heartbeat_status": heartbeat_status or "-",
        "heartbeat_brief": text(heartbeat_payload.get("executor_brief")),
        "heartbeat_artifact": str(heartbeat_path) if heartbeat_path else "",
        "guarded_exec_probe_status": guarded_exec_probe_status,
        "guarded_exec_probe_artifact": guarded_exec_probe_artifact,
        "intent_brief": text(intent_payload.get("queue_brief")),
        "intent_status": queue_status,
        "journal_brief": text(journal_payload.get("journal_brief")),
        "journal_status": text(journal_payload.get("journal_status")),
        "idempotency_key_brief": text(heartbeat_payload.get("idempotency_key"))
        or text(journal_payload.get("last_entry_key"))
        or text(journal_payload.get("entry_key")),
        "ack_state_artifact_name": "remote_execution_ack_state.json",
        "blocker_detail": text(journal_payload.get("blocker_detail")) or text(intent_payload.get("blocker_detail")),
        "done_when": done_when,
        "artifacts": {
            "remote_execution_identity_state": str(identity_path),
            "remote_intent_queue": str(intent_path),
            "remote_execution_journal": str(journal_path),
            "executor_heartbeat": str(heartbeat_path) if heartbeat_path else "",
            "unit_preview": str(unit_preview_path) if unit_preview_path else "",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenClaw orderflow executor state.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--unit-preview-path", default="")
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    identity_path = find_latest(review_dir, "*_remote_execution_identity_state.json")
    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    journal_path = find_latest(review_dir, "*_remote_execution_journal.json")
    heartbeat_path = find_latest(review_dir, "*_openclaw_orderflow_executor_heartbeat.json")
    explicit_unit_path = Path(str(args.unit_preview_path).strip()).expanduser().resolve() if text(args.unit_preview_path) else None
    unit_preview_path = explicit_unit_path if explicit_unit_path and explicit_unit_path.exists() else find_latest(review_dir, "*_openclaw_orderflow_executor.service")
    missing = [
        name
        for name, path in (
            ("remote_execution_identity_state", identity_path),
            ("remote_intent_queue", intent_path),
            ("remote_execution_journal", journal_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        identity_path=identity_path,
        identity_payload=load_json_mapping(identity_path),
        intent_path=intent_path,
        intent_payload=load_json_mapping(intent_path),
        journal_path=journal_path,
        journal_payload=load_json_mapping(journal_path),
        heartbeat_path=heartbeat_path,
        heartbeat_payload=load_json_mapping(heartbeat_path)
        if heartbeat_path is not None and heartbeat_path.exists()
        else {},
        unit_preview_path=unit_preview_path,
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_openclaw_orderflow_executor_state.json"
    markdown = review_dir / f"{stamp}_openclaw_orderflow_executor_state.md"
    checksum = review_dir / f"{stamp}_openclaw_orderflow_executor_state_checksum.json"

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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
