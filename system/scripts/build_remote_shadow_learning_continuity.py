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
        "*_remote_shadow_learning_continuity.json",
        "*_remote_shadow_learning_continuity.md",
        "*_remote_shadow_learning_continuity_checksum.json",
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


def minutes_since(reference_now: dt.datetime, raw: Any) -> float | None:
    parsed = parse_utc(raw)
    if parsed is None:
        return None
    return max(0.0, round((reference_now - parsed).total_seconds() / 60.0, 3))


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Shadow Learning Continuity",
            "",
            f"- brief: `{text(payload.get('continuity_brief'))}`",
            f"- status: `{text(payload.get('continuity_status'))}`",
            f"- decision: `{text(payload.get('continuity_decision'))}`",
            f"- heartbeat_age_minutes: `{text(payload.get('heartbeat_age_minutes'))}`",
            f"- journal_age_minutes: `{text(payload.get('journal_last_entry_age_minutes'))}`",
            f"- quality_score: `{text(payload.get('quality_score'))}`",
            f"- promotion_gate: `{text(payload.get('promotion_gate_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    heartbeat_path: Path,
    heartbeat_payload: dict[str, Any],
    executor_state_path: Path,
    executor_state_payload: dict[str, Any],
    journal_path: Path,
    journal_payload: dict[str, Any],
    quality_report_path: Path,
    quality_report_payload: dict[str, Any],
    shadow_clock_path: Path,
    shadow_clock_payload: dict[str, Any],
    promotion_gate_path: Path,
    promotion_gate_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    route_symbol = (
        text(heartbeat_payload.get("intent_symbol"))
        or text(journal_payload.get("intent_symbol"))
        or text(promotion_gate_payload.get("route_symbol"))
        or "-"
    )
    remote_market = (
        text(journal_payload.get("remote_market"))
        or text(promotion_gate_payload.get("remote_market"))
        or "-"
    )
    heartbeat_age_minutes = minutes_since(reference_now, heartbeat_payload.get("generated_at_utc"))
    executor_age_minutes = minutes_since(reference_now, executor_state_payload.get("generated_at_utc"))
    journal_last_entry_age_minutes = minutes_since(
        reference_now, as_dict(journal_payload.get("last_entry")).get("recorded_at_utc")
    )
    quality_score = int(quality_report_payload.get("quality_score") or 0)
    shadow_learning_score = int(quality_report_payload.get("shadow_learning_score") or 0)
    shadow_clock_ok = bool(shadow_clock_payload.get("shadow_learning_allowed", False))
    promotion_gate_allows_shadow = text(promotion_gate_payload.get("shadow_learning_decision")) in {
        "continue_shadow_learning_collect_feedback",
        "shadow_learning_continues_until_human_review",
    }
    heartbeat_ok = heartbeat_age_minutes is not None and heartbeat_age_minutes <= 60
    executor_ok = executor_age_minutes is not None and executor_age_minutes <= 60
    journal_ok = journal_last_entry_age_minutes is not None and journal_last_entry_age_minutes <= 180
    quality_ok = quality_score >= 40 and shadow_learning_score >= 60
    executor_status = text(executor_state_payload.get("executor_status"))
    journal_status = text(journal_payload.get("journal_status"))

    if (
        heartbeat_ok
        and executor_ok
        and journal_ok
        and quality_ok
        and shadow_clock_ok
        and promotion_gate_allows_shadow
    ):
        continuity_status = "shadow_learning_continuity_stable"
        continuity_decision = "continue_shadow_learning_collect_feedback"
    elif promotion_gate_allows_shadow and (heartbeat_ok or journal_ok or quality_ok):
        continuity_status = "shadow_learning_continuity_degraded"
        continuity_decision = "repair_shadow_learning_path_while_blocked"
    else:
        continuity_status = "shadow_learning_continuity_broken"
        continuity_decision = "rebuild_shadow_learning_path_before_trusting_feedback"

    continuity_brief = ":".join(
        [continuity_status, route_symbol or "-", text(continuity_decision), remote_market or "-"]
    )
    blocker_detail = " | ".join(
        dedupe_text(
            [
                f"heartbeat_age_minutes={heartbeat_age_minutes}" if heartbeat_age_minutes is not None else "heartbeat_missing",
                f"executor_age_minutes={executor_age_minutes}" if executor_age_minutes is not None else "executor_missing",
                f"journal_last_entry_age_minutes={journal_last_entry_age_minutes}"
                if journal_last_entry_age_minutes is not None
                else "journal_missing",
                f"quality_score={quality_score}",
                f"shadow_learning_score={shadow_learning_score}",
                f"executor_status={executor_status}" if executor_status else "",
                f"journal_status={journal_status}" if journal_status else "",
                text(promotion_gate_payload.get("promotion_gate_brief")),
            ]
        )
    )
    done_when = (
        "executor heartbeat/executor state remain fresh, journal keeps appending recent shadow entries, "
        "quality stays above learning thresholds, and shadow clock evidence remains present while promotion is still blocked"
    )
    return {
        "action": "build_remote_shadow_learning_continuity",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "continuity_status": continuity_status,
        "continuity_brief": continuity_brief,
        "continuity_decision": continuity_decision,
        "route_symbol": route_symbol,
        "remote_market": remote_market,
        "heartbeat_age_minutes": heartbeat_age_minutes,
        "executor_age_minutes": executor_age_minutes,
        "journal_last_entry_age_minutes": journal_last_entry_age_minutes,
        "quality_score": quality_score,
        "shadow_learning_score": shadow_learning_score,
        "shadow_clock_ok": shadow_clock_ok,
        "promotion_gate_allows_shadow": promotion_gate_allows_shadow,
        "executor_status": executor_status,
        "journal_status": journal_status,
        "promotion_gate_brief": text(promotion_gate_payload.get("promotion_gate_brief")),
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "openclaw_orderflow_executor_heartbeat": str(heartbeat_path),
            "openclaw_orderflow_executor_state": str(executor_state_path),
            "remote_execution_journal": str(journal_path),
            "remote_orderflow_quality_report": str(quality_report_path),
            "remote_shadow_clock_evidence": str(shadow_clock_path),
            "remote_guarded_canary_promotion_gate": str(promotion_gate_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote shadow learning continuity.")
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
    quality_report_path = find_latest(review_dir, "*_remote_orderflow_quality_report.json")
    shadow_clock_path = find_latest(review_dir, "*_remote_shadow_clock_evidence.json")
    promotion_gate_path = find_latest(review_dir, "*_remote_guarded_canary_promotion_gate.json")

    missing = [
        name
        for name, path in (
            ("openclaw_orderflow_executor_heartbeat", heartbeat_path),
            ("openclaw_orderflow_executor_state", executor_state_path),
            ("remote_execution_journal", journal_path),
            ("remote_orderflow_quality_report", quality_report_path),
            ("remote_shadow_clock_evidence", shadow_clock_path),
            ("remote_guarded_canary_promotion_gate", promotion_gate_path),
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
        quality_report_path=quality_report_path,
        quality_report_payload=load_json_mapping(quality_report_path),
        shadow_clock_path=shadow_clock_path,
        shadow_clock_payload=load_json_mapping(shadow_clock_path),
        promotion_gate_path=promotion_gate_path,
        promotion_gate_payload=load_json_mapping(promotion_gate_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_shadow_learning_continuity.json"
    markdown = review_dir / f"{stamp}_remote_shadow_learning_continuity.md"
    checksum = review_dir / f"{stamp}_remote_shadow_learning_continuity_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
                "generated_at_utc": fmt_utc(reference_now),
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
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
