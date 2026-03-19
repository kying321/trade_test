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
        "*_remote_execution_transport_sla.json",
        "*_remote_execution_transport_sla.md",
        "*_remote_execution_transport_sla_checksum.json",
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
            "# Remote Execution Transport SLA",
            "",
            f"- brief: `{text(payload.get('transport_sla_brief'))}`",
            f"- status: `{text(payload.get('transport_sla_status'))}`",
            f"- decision: `{text(payload.get('transport_sla_decision'))}`",
            f"- send_samples: `{text(payload.get('send_sample_count'))}`",
            f"- ack_samples: `{text(payload.get('ack_sample_count'))}`",
            f"- fill_samples: `{text(payload.get('fill_sample_count'))}`",
            f"- probe_samples: `{text(payload.get('probe_sample_count'))}`",
            f"- guarded_transport: `{text(payload.get('guarded_transport_brief'))}`",
            f"- ack: `{text(payload.get('ack_brief'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    guarded_transport_path: Path,
    guarded_transport_payload: dict[str, Any],
    ack_path: Path,
    ack_payload: dict[str, Any],
    actor_path: Path,
    actor_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = (
        text(guarded_transport_payload.get("route_symbol"))
        or text(actor_payload.get("route_symbol"))
        or "-"
    )
    remote_market = (
        text(guarded_transport_payload.get("remote_market"))
        or text(actor_payload.get("remote_market"))
        or "-"
    )
    guarded_status = text(guarded_transport_payload.get("guarded_transport_status"))
    guarded_exec_probe_status = text(
        guarded_transport_payload.get("guarded_exec_probe_status")
    ) or text(actor_payload.get("guarded_exec_probe_status"))
    guarded_exec_probe_artifact = text(
        guarded_transport_payload.get("guarded_exec_probe_artifact")
    ) or text(actor_payload.get("guarded_exec_probe_artifact"))
    ack_status = text(ack_payload.get("ack_status"))
    ack_decision = text(ack_payload.get("ack_decision"))

    if guarded_status == "guarded_transport_preview_probe_completed":
        transport_sla_status = "shadow_transport_sla_probe_completed_no_send"
        transport_sla_decision = "record_guarded_probe_sample_without_live_send"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 1
    elif guarded_status == "guarded_transport_preview_probe_controlled_block":
        transport_sla_status = "shadow_transport_sla_probe_controlled_block_no_send"
        transport_sla_decision = "record_guarded_probe_controlled_block_without_live_send"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 1
    elif guarded_status == "guarded_transport_preview_probe_timeout":
        transport_sla_status = "shadow_transport_sla_probe_timeout_no_send"
        transport_sla_decision = "inspect_guarded_probe_timeout_before_live_send"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 1
    elif guarded_status == "guarded_transport_preview_probe_error":
        transport_sla_status = "shadow_transport_sla_probe_error_no_send"
        transport_sla_decision = "inspect_guarded_probe_error_before_live_send"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 1
    elif guarded_status == "guarded_transport_preview_probe_candidate_blocked":
        transport_sla_status = "shadow_transport_sla_probe_candidate_no_send"
        transport_sla_decision = "record_probe_candidate_without_live_send"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 0
    elif guarded_status == "guarded_transport_preview_runtime_boundary_blocked":
        transport_sla_status = "shadow_transport_sla_runtime_boundary_blocked_no_send"
        transport_sla_decision = "implement_runtime_before_transport_sla"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 0
    elif guarded_status == "guarded_transport_preview_learning_only":
        transport_sla_status = "shadow_transport_sla_learning_only_no_send"
        transport_sla_decision = "accumulate_learning_samples_before_guard_clear"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 0
    elif guarded_status == "guarded_transport_preview_blocked":
        transport_sla_status = "shadow_transport_sla_blocked_no_send"
        transport_sla_decision = "define_sla_before_canary"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 0
    elif guarded_status == "guarded_transport_preview_pending_guard_clear":
        transport_sla_status = "shadow_transport_sla_pending_guard_clear"
        transport_sla_decision = "wait_for_first_guarded_send_sample"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 0
    elif ack_status == "shadow_fill_ack_recorded":
        transport_sla_status = "shadow_transport_sla_fill_observed"
        transport_sla_decision = "measure_first_fill_latency"
        send_sample_count = 1
        ack_sample_count = 1
        fill_sample_count = 1
        probe_sample_count = 0
    else:
        transport_sla_status = "shadow_transport_sla_unknown"
        transport_sla_decision = "inspect_transport_timing_inputs"
        send_sample_count = 0
        ack_sample_count = 0
        fill_sample_count = 0
        probe_sample_count = 0

    transport_sla_brief = ":".join(
        [transport_sla_status, symbol or "-", text(guarded_transport_payload.get("send_state")) or "-", remote_market or "-"]
    )
    done_when = (
        "first guarded send/ack/fill samples are recorded under the actor boundary and compared against an explicit latency/duplicate budget before any canary is armed"
    )
    return {
        "action": "build_remote_execution_transport_sla",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "transport_sla_status": transport_sla_status,
        "transport_sla_brief": transport_sla_brief,
        "transport_sla_decision": transport_sla_decision,
        "next_transition": "remote_execution_actor_canary_gate",
        "route_symbol": symbol,
        "remote_market": remote_market,
        "send_sample_count": send_sample_count,
        "ack_sample_count": ack_sample_count,
        "fill_sample_count": fill_sample_count,
        "probe_sample_count": probe_sample_count,
        "guarded_transport_brief": text(guarded_transport_payload.get("guarded_transport_brief")),
        "guarded_transport_status": guarded_status,
        "guarded_transport_decision": text(guarded_transport_payload.get("guarded_transport_decision")),
        "guarded_exec_probe_status": guarded_exec_probe_status,
        "guarded_exec_probe_artifact": guarded_exec_probe_artifact,
        "ack_brief": text(ack_payload.get("ack_brief")),
        "ack_status": ack_status,
        "ack_decision": ack_decision,
        "actor_brief": text(actor_payload.get("actor_brief")),
        "blocker_detail": text(actor_payload.get("blocker_detail"))
        or text(ack_payload.get("blocker_detail"))
        or text(guarded_transport_payload.get("blocker_detail")),
        "done_when": done_when,
        "artifacts": {
            "remote_execution_actor_guarded_transport": str(guarded_transport_path),
            "remote_execution_ack_state": str(ack_path),
            "remote_execution_actor_state": str(actor_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution transport SLA state.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    guarded_transport_path = find_latest(review_dir, "*_remote_execution_actor_guarded_transport.json")
    ack_path = find_latest(review_dir, "*_remote_execution_ack_state.json")
    actor_path = find_latest(review_dir, "*_remote_execution_actor_state.json")
    missing = [
        name
        for name, path in (
            ("remote_execution_actor_guarded_transport", guarded_transport_path),
            ("remote_execution_ack_state", ack_path),
            ("remote_execution_actor_state", actor_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        guarded_transport_path=guarded_transport_path,
        guarded_transport_payload=load_json_mapping(guarded_transport_path),
        ack_path=ack_path,
        ack_payload=load_json_mapping(ack_path),
        actor_path=actor_path,
        actor_payload=load_json_mapping(actor_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_transport_sla.json"
    markdown = review_dir / f"{stamp}_remote_execution_transport_sla.md"
    checksum = review_dir / f"{stamp}_remote_execution_transport_sla_checksum.json"

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
