#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
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
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")


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


def parse_utc(raw: Any) -> dt.datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
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


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name))


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
        "*_remote_orderflow_feedback.json",
        "*_remote_orderflow_feedback.md",
        "*_remote_orderflow_feedback_checksum.json",
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


def read_journal_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def queue_age_status(age_seconds: float) -> str:
    if age_seconds <= 300.0:
        return "queue_fresh"
    if age_seconds <= 900.0:
        return "queue_warming"
    if age_seconds <= 3600.0:
        return "queue_aging_high"
    return "queue_stale"


def dominant_guard_reason(
    journal_payload: dict[str, Any], live_gate_payload: dict[str, Any]
) -> tuple[str, list[str]]:
    last_entry = as_dict(journal_payload.get("last_entry"))
    reasons = dedupe_text([text(code) for code in as_list(last_entry.get("risk_reason_codes"))])
    if not reasons:
        for row in as_list(live_gate_payload.get("blockers")):
            if not isinstance(row, dict):
                continue
            if text(row.get("name")) != "risk_guard":
                continue
            reasons = dedupe_text([text(code) for code in as_list(row.get("reason_codes"))])
            if reasons:
                break
    return (reasons[0] if reasons else "no_guard_reason_recorded", reasons)


def recent_feedback_counts(*, entries: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    symbol_entries = [
        dict(row)
        for row in entries
        if text(as_dict(row).get("intent_symbol")).upper() == symbol.upper()
    ]
    guardian_blocked = [
        row for row in symbol_entries if "blocked" in text(row.get("risk_verdict_brief")).lower()
    ]
    no_fill = [row for row in symbol_entries if text(row.get("fill_status")).startswith("no_fill")]
    recent_outcomes = dedupe_text([text(row.get("execution_outcome")) for row in symbol_entries])
    return {
        "entry_count": len(entries),
        "symbol_entry_count": len(symbol_entries),
        "guardian_blocked_count": len(guardian_blocked),
        "no_fill_count": len(no_fill),
        "recent_outcomes": recent_outcomes,
    }


def build_payload(
    *,
    intent_path: Path,
    intent_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    executor_state_path: Path,
    executor_state_payload: dict[str, Any],
    live_gate_path: Path | None,
    live_gate_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = text(intent_payload.get("preferred_route_symbol")) or text(journal_payload.get("intent_symbol"))
    action = text(intent_payload.get("preferred_route_action")) or text(journal_payload.get("intent_action"))
    remote_market = text(intent_payload.get("remote_market")) or text(journal_payload.get("remote_market"))
    queued_at = parse_utc(as_dict(journal_payload.get("last_entry")).get("recorded_at_utc")) or parse_utc(
        journal_payload.get("generated_at_utc")
    )
    age_seconds = max(0.0, (reference_now - (queued_at or reference_now)).total_seconds())
    age_status = queue_age_status(age_seconds)
    dominant_reason, reason_codes = dominant_guard_reason(journal_payload, live_gate_payload)
    journal_log_path = Path(text(journal_payload.get("journal_path"))) if text(journal_payload.get("journal_path")) else None
    journal_entries = read_journal_entries(journal_log_path) if journal_log_path else []
    counts = recent_feedback_counts(entries=journal_entries, symbol=symbol)
    execution_outcome = text(journal_payload.get("execution_outcome"))
    fill_status = text(journal_payload.get("fill_status"))
    guard_blocked = "blocked" in text(journal_payload.get("risk_verdict_brief")).lower() or bool(
        reason_codes
    )
    if fill_status.startswith("no_fill") and guard_blocked:
        feedback_status = "downrank_guardian_blocked_route"
        routing_impact = "downrank_current_route_until_guardian_clear"
        throttling_impact = "keep_executor_shadow_only"
        feedback_recommendation = "downrank_until_ticket_fresh_and_guardian_clear"
    elif fill_status.startswith("no_fill"):
        feedback_status = "downrank_no_fill_route"
        routing_impact = "downrank_current_route_until_fill_quality_improves"
        throttling_impact = "hold_shadow_executor"
        feedback_recommendation = "downrank_until_fill_or_better_queue_quality"
    else:
        feedback_status = "feedback_ready_for_route_learning"
        routing_impact = "use_fill_quality_for_lane_selection"
        throttling_impact = "allow_executor_quality_feedback"
        feedback_recommendation = "promote_fill_quality_into_route_scoring"
    blocker_detail = " | ".join(
        part
        for part in dedupe_text(
            [
                text(intent_payload.get("ticket_match_brief"))
                or text(journal_payload.get("ticket_match_brief")),
                text(intent_payload.get("guard_alignment_brief"))
                or text(journal_payload.get("guard_alignment_brief")),
                ",".join(reason_codes),
                text(journal_payload.get("execution_outcome")),
                text(journal_payload.get("fill_status")),
            ]
        )
        if part
    )
    return {
        "action": "build_remote_orderflow_feedback",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "feedback_status": feedback_status,
        "feedback_brief": ":".join([feedback_status, symbol or "-", age_status, dominant_reason]),
        "feedback_recommendation": feedback_recommendation,
        "routing_impact": routing_impact,
        "throttling_impact": throttling_impact,
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "queue_age_seconds": round(age_seconds, 3),
        "queue_age_status": age_status,
        "journal_status": text(journal_payload.get("journal_status")),
        "execution_outcome": execution_outcome,
        "fill_status": fill_status,
        "dominant_guard_reason": dominant_reason,
        "risk_reason_codes": reason_codes,
        "ticket_match_brief": text(intent_payload.get("ticket_match_brief"))
        or text(journal_payload.get("ticket_match_brief")),
        "ticket_artifact_status": text(intent_payload.get("ticket_artifact_status"))
        or text(journal_payload.get("ticket_artifact_status")),
        "guard_alignment_brief": text(intent_payload.get("guard_alignment_brief"))
        or text(journal_payload.get("guard_alignment_brief")),
        "executor_status": text(executor_state_payload.get("executor_status")),
        "executor_brief": text(executor_state_payload.get("executor_brief")),
        "heartbeat_status": text(executor_state_payload.get("heartbeat_status")),
        "idempotency_key_brief": text(executor_state_payload.get("idempotency_key_brief"))
        or text(journal_payload.get("last_entry_key")),
        "feedback_entry_count": counts["entry_count"],
        "feedback_symbol_entry_count": counts["symbol_entry_count"],
        "guardian_blocked_count": counts["guardian_blocked_count"],
        "no_fill_count": counts["no_fill_count"],
        "recent_outcomes": counts["recent_outcomes"],
        "blocker_detail": blocker_detail,
        "done_when": (
            "remote feedback records repeated guardian/no-fill patterns well enough to down-rank stale routes before they recycle through tickets, and future fills can feed slippage/quality back into selection"
        ),
        "artifacts": {
            "remote_intent_queue": str(intent_path),
            "remote_execution_journal": str(journal_path),
            "openclaw_orderflow_executor_state": str(executor_state_path),
            "live_gate_blocker_report": str(live_gate_path) if live_gate_path else "",
            "journal_log": str(journal_log_path) if journal_log_path else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Orderflow Feedback",
            "",
            f"- brief: `{text(payload.get('feedback_brief'))}`",
            f"- recommendation: `{text(payload.get('feedback_recommendation'))}`",
            f"- routing_impact: `{text(payload.get('routing_impact'))}`",
            f"- throttling_impact: `{text(payload.get('throttling_impact'))}`",
            f"- queue_age: `{text(payload.get('queue_age_status'))}` ({payload.get('queue_age_seconds')})",
            f"- dominant_guard_reason: `{text(payload.get('dominant_guard_reason'))}`",
            f"- ticket_state: `{text(payload.get('ticket_match_brief'))}`",
            f"- execution_outcome: `{text(payload.get('execution_outcome'))}`",
            f"- fill_status: `{text(payload.get('fill_status'))}`",
            f"- feedback_entries: `{payload.get('feedback_symbol_entry_count')}/{payload.get('feedback_entry_count')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote orderflow feedback for OpenClaw.")
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

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    journal_path = find_latest(review_dir, "*_remote_execution_journal.json")
    executor_state_path = find_latest(review_dir, "*_openclaw_orderflow_executor_state.json")
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    missing = [
        name
        for name, path in (
            ("remote_intent_queue", intent_path),
            ("remote_execution_journal", journal_path),
            ("openclaw_orderflow_executor_state", executor_state_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        intent_path=intent_path,
        intent_payload=load_json_mapping(intent_path),
        journal_path=journal_path,
        journal_payload=load_json_mapping(journal_path),
        executor_state_path=executor_state_path,
        executor_state_payload=load_json_mapping(executor_state_path),
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path) if live_gate_path else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_orderflow_feedback.json"
    markdown = review_dir / f"{stamp}_remote_orderflow_feedback.md"
    checksum = review_dir / f"{stamp}_remote_orderflow_feedback_checksum.json"

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
