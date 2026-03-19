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
    text_value = str(raw or "").strip()
    if not text_value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_utc(raw: Any) -> dt.datetime | None:
    text_value = str(raw or "").strip()
    if not text_value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
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


def dedupe_text(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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
        "*_remote_shadow_clock_evidence.json",
        "*_remote_shadow_clock_evidence.md",
        "*_remote_shadow_clock_evidence_checksum.json",
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


def build_payload(
    *,
    heartbeat_path: Path,
    heartbeat_payload: dict[str, Any],
    executor_state_path: Path,
    executor_state_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    identity_state_path: Path | None,
    identity_state_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    heartbeat_generated = parse_utc(heartbeat_payload.get("generated_at_utc"))
    executor_generated = parse_utc(executor_state_payload.get("generated_at_utc"))
    journal_generated = parse_utc(journal_payload.get("generated_at_utc"))
    journal_last_entry = parse_utc(as_dict(journal_payload.get("last_entry")).get("recorded_at_utc"))

    symbol_candidates = dedupe_text(
        [
            text(heartbeat_payload.get("intent_symbol")),
            text(journal_payload.get("intent_symbol")),
        ]
    )
    route_symbol = symbol_candidates[0] if symbol_candidates else "-"
    remote_market = text(journal_payload.get("remote_market")) or text(
        identity_state_payload.get("ready_check_scope_market")
    )
    symbol_alignment_ok = len(symbol_candidates) <= 1
    required_missing = [
        name
        for name, value in (
            ("heartbeat_generated_at_utc", heartbeat_generated),
            ("executor_generated_at_utc", executor_generated),
            ("journal_last_entry_recorded_at_utc", journal_last_entry),
        )
        if value is None
    ]
    timestamp_chain_ok = bool(
        heartbeat_generated
        and executor_generated
        and journal_last_entry
        and journal_last_entry <= heartbeat_generated <= executor_generated
    )

    if required_missing:
        evidence_status = "shadow_clock_evidence_missing"
    elif not symbol_alignment_ok or not remote_market:
        evidence_status = "shadow_clock_evidence_inconsistent"
    else:
        evidence_status = "shadow_clock_evidence_present"
    shadow_learning_allowed = evidence_status == "shadow_clock_evidence_present"
    timestamp_chain_brief = (
        "journal_last_entry<=heartbeat<=executor"
        if timestamp_chain_ok
        else "timestamp_chain_incomplete_or_out_of_order"
    )
    evidence_brief = ":".join(
        [
            evidence_status,
            route_symbol or "-",
            timestamp_chain_brief,
            remote_market or "-",
        ]
    )
    blocker_detail = (
        "remote executor heartbeat/executor/journal timestamps are present and aligned enough for "
        "shadow learning; this does not clear local promotion time-sync or guarded canary eligibility."
        if shadow_learning_allowed
        else "remote shadow clock evidence is missing or inconsistent; keep shadow learning diagnostic-only until timestamp chain is present."
    )
    return {
        "action": "build_remote_shadow_clock_evidence",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "evidence_status": evidence_status,
        "evidence_brief": evidence_brief,
        "route_symbol": route_symbol,
        "remote_market": remote_market,
        "shadow_learning_allowed": shadow_learning_allowed,
        "promotion_time_sync_dependency_artifact": "system_time_sync_repair_verification_report",
        "symbol_alignment_ok": symbol_alignment_ok,
        "timestamp_chain_ok": timestamp_chain_ok,
        "timestamp_chain_brief": timestamp_chain_brief,
        "heartbeat_generated_at_utc": fmt_utc(heartbeat_generated) if heartbeat_generated else "",
        "executor_generated_at_utc": fmt_utc(executor_generated) if executor_generated else "",
        "journal_generated_at_utc": fmt_utc(journal_generated) if journal_generated else "",
        "journal_last_entry_recorded_at_utc": fmt_utc(journal_last_entry) if journal_last_entry else "",
        "missing_timestamp_fields": required_missing,
        "blocker_detail": blocker_detail,
        "done_when": (
            "keep heartbeat/executor/journal timestamps present and aligned for shadow learning; "
            "local system time repair verification is still required before any promotion or guarded canary review"
        ),
        "artifacts": {
            "openclaw_orderflow_executor_heartbeat": str(heartbeat_path),
            "openclaw_orderflow_executor_state": str(executor_state_path),
            "remote_execution_journal": str(journal_path),
            "remote_execution_identity_state": str(identity_state_path) if identity_state_path else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Shadow Clock Evidence",
            "",
            f"- brief: `{text(payload.get('evidence_brief'))}`",
            f"- shadow_learning_allowed: `{text(payload.get('shadow_learning_allowed'))}`",
            f"- heartbeat_generated_at_utc: `{text(payload.get('heartbeat_generated_at_utc'))}`",
            f"- executor_generated_at_utc: `{text(payload.get('executor_generated_at_utc'))}`",
            f"- journal_last_entry_recorded_at_utc: `{text(payload.get('journal_last_entry_recorded_at_utc'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote shadow clock evidence artifact.")
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
    heartbeat_path = find_latest(review_dir, "*_openclaw_orderflow_executor_heartbeat.json")
    executor_state_path = find_latest(review_dir, "*_openclaw_orderflow_executor_state.json")
    journal_path = find_latest(review_dir, "*_remote_execution_journal.json")
    identity_state_path = find_latest(review_dir, "*_remote_execution_identity_state.json")

    missing = [
        name
        for name, path in (
            ("openclaw_orderflow_executor_heartbeat", heartbeat_path),
            ("openclaw_orderflow_executor_state", executor_state_path),
            ("remote_execution_journal", journal_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        heartbeat_path=heartbeat_path,
        heartbeat_payload=load_json_mapping(heartbeat_path),
        executor_state_path=executor_state_path,
        executor_state_payload=load_json_mapping(executor_state_path),
        journal_path=journal_path,
        journal_payload=load_json_mapping(journal_path),
        identity_state_path=identity_state_path,
        identity_state_payload=load_json_mapping(identity_state_path)
        if identity_state_path is not None and identity_state_path.exists()
        else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_shadow_clock_evidence.json"
    markdown = review_dir / f"{stamp}_remote_shadow_clock_evidence.md"
    checksum = review_dir / f"{stamp}_remote_shadow_clock_evidence_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
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
    payload["artifact"] = str(artifact)
    payload["markdown"] = str(markdown)
    payload["checksum"] = str(checksum)
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
