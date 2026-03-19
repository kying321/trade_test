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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def text(value: Any) -> str:
    return str(value or "").strip()


def join_nonempty(parts: list[str], *, sep: str = " | ") -> str:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = text(part)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return sep.join(out)


def append_jsonl_dedup(path: Path, row: dict[str, Any]) -> tuple[bool, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_count = 0
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        existing_count = len(lines)
        for line in reversed(lines[-10:]):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if text(payload.get("entry_key")) == text(row.get("entry_key")):
                return False, existing_count
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True, existing_count + 1


def read_last_entry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
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
        "*_remote_execution_journal.json",
        "*_remote_execution_journal.md",
        "*_remote_execution_journal_checksum.json",
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
    intent_queue_path: Path,
    intent_queue_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    identity_path: Path,
    identity_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    symbol = text(intent_queue_payload.get("preferred_route_symbol"))
    action = text(intent_queue_payload.get("preferred_route_action"))
    remote_market = text(intent_queue_payload.get("remote_market")) or text(
        identity_payload.get("ready_check_scope_market")
    )
    risk_blocker = {}
    for row in as_list(live_gate_payload.get("blockers")):
        if isinstance(row, dict) and text(row.get("name")) == "risk_guard":
            risk_blocker = dict(row)
            break
    verdict_status = text(risk_blocker.get("status")) or "unknown"
    verdict_brief = join_nonempty(
        [
            verdict_status,
            ",".join([text(code) for code in as_list(risk_blocker.get("reason_codes")) if text(code)]),
        ],
        sep=":"
    )
    execution_outcome = (
        "not_attempted_execution_contract_blocked"
        if text(intent_queue_payload.get("queue_status")) == "queued_execution_contract_blocked"
        else (
            "not_attempted_wait_trade_readiness"
            if text(intent_queue_payload.get("queue_status")) == "queued_wait_trade_readiness"
            else (
                "not_attempted_ticket_or_guard_blocked"
                if not bool(intent_queue_payload.get("intent_ready"))
                else "ready_for_executor_handoff"
            )
        )
    )
    entry_key = ":".join(
        [
            fmt_utc(reference_now),
            symbol or "-",
            action or "-",
            text(intent_queue_payload.get("queue_status")) or "-",
            text(intent_queue_payload.get("ticket_match_status")) or "-",
        ]
    )
    journal_entry = {
        "entry_key": entry_key,
        "entry_type": "remote_intent_snapshot",
        "recorded_at_utc": fmt_utc(reference_now),
        "remote_market": remote_market,
        "identity_brief": text(identity_payload.get("identity_brief")),
        "intent_brief": text(intent_queue_payload.get("queue_brief")),
        "intent_symbol": symbol,
        "intent_action": action,
        "intent_status": text(intent_queue_payload.get("queue_status")),
        "intent_recommendation": text(intent_queue_payload.get("queue_recommendation")),
        "ticket_match_brief": text(intent_queue_payload.get("ticket_match_brief")),
        "ticket_artifact_status": text(intent_queue_payload.get("ticket_artifact_status")),
        "guard_alignment_brief": text(intent_queue_payload.get("guard_alignment_brief")),
        "risk_verdict_brief": verdict_brief,
        "risk_reason_codes": [text(code) for code in as_list(risk_blocker.get("reason_codes")) if text(code)],
        "execution_outcome": execution_outcome,
        "fill_status": "no_fill_execution_not_attempted",
        "blocker_detail": text(intent_queue_payload.get("blocker_detail")),
        "done_when": text(intent_queue_payload.get("done_when")),
        "artifacts": {
            "remote_intent_queue": str(intent_queue_path),
            "live_gate_blocker_report": str(live_gate_path),
            "remote_execution_identity_state": str(identity_path),
            "signal_to_order_tickets": text(intent_queue_payload.get("ticket_artifact")),
        },
    }
    payload = {
        "action": "build_remote_execution_journal",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "journal_status": "intent_logged_guardian_blocked",
        "journal_brief": join_nonempty(
            [
                text(intent_queue_payload.get("queue_brief")),
                verdict_brief,
                execution_outcome,
            ]
        ),
        "entry_type": journal_entry["entry_type"],
        "entry_key": entry_key,
        "remote_market": remote_market,
        "intent_brief": text(intent_queue_payload.get("queue_brief")),
        "intent_symbol": symbol,
        "intent_action": action,
        "intent_status": text(intent_queue_payload.get("queue_status")),
        "intent_recommendation": text(intent_queue_payload.get("queue_recommendation")),
        "ticket_match_brief": text(intent_queue_payload.get("ticket_match_brief")),
        "ticket_artifact_status": text(intent_queue_payload.get("ticket_artifact_status")),
        "guard_alignment_brief": text(intent_queue_payload.get("guard_alignment_brief")),
        "risk_verdict_brief": verdict_brief,
        "execution_outcome": execution_outcome,
        "fill_status": "no_fill_execution_not_attempted",
        "blocker_detail": text(intent_queue_payload.get("blocker_detail")),
        "done_when": text(intent_queue_payload.get("done_when")),
        "artifacts": {
            "remote_intent_queue": str(intent_queue_path),
            "live_gate_blocker_report": str(live_gate_path),
            "remote_execution_identity_state": str(identity_path),
            "signal_to_order_tickets": text(intent_queue_payload.get("ticket_artifact")),
        },
    }
    return payload, journal_entry


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Execution Journal",
            "",
            f"- brief: `{text(payload.get('journal_brief'))}`",
            f"- intent: `{text(payload.get('intent_brief'))}`",
            f"- risk verdict: `{text(payload.get('risk_verdict_brief'))}`",
            f"- execution outcome: `{text(payload.get('execution_outcome'))}`",
            f"- fill status: `{text(payload.get('fill_status'))}`",
            f"- ticket state: `{text(payload.get('ticket_match_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution journal for OpenClaw.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--journal-path", default="")
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    intent_queue_path = find_latest(review_dir, "*_remote_intent_queue.json")
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    identity_path = find_latest(review_dir, "*_remote_execution_identity_state.json")
    missing = [
        name
        for name, path in (
            ("remote_intent_queue", intent_queue_path),
            ("live_gate_blocker_report", live_gate_path),
            ("remote_execution_identity_state", identity_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload, journal_entry = build_payload(
        intent_queue_path=intent_queue_path,
        intent_queue_payload=load_json_mapping(intent_queue_path),
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path),
        identity_path=identity_path,
        identity_payload=load_json_mapping(identity_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_journal.json"
    markdown = review_dir / f"{stamp}_remote_execution_journal.md"
    checksum = review_dir / f"{stamp}_remote_execution_journal_checksum.json"
    journal_path = (
        Path(args.journal_path).expanduser().resolve()
        if text(args.journal_path)
        else review_dir / "remote_execution_journal.jsonl"
    )
    appended, entry_count = append_jsonl_dedup(journal_path, journal_entry)
    last_entry = read_last_entry(journal_path)

    payload.update(
        {
            "journal_path": str(journal_path),
            "append_status": "appended" if appended else "duplicate_skipped",
            "entry_count": entry_count,
            "last_entry": last_entry,
            "last_entry_key": text(last_entry.get("entry_key")),
        }
    )

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
                "journal_path": str(journal_path),
                "journal_sha256": sha256_file(journal_path),
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
        keep=int(max(1, int(args.artifact_keep))),
        ttl_hours=float(max(1.0, float(args.artifact_ttl_hours))),
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_due_to_keep": pruned_keep,
            "pruned_due_to_age": pruned_age,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
