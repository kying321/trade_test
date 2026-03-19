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
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


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
        "*_crypto_signal_source_refresh_readiness.json",
        "*_crypto_signal_source_refresh_readiness.md",
        "*_crypto_signal_source_refresh_readiness_checksum.json",
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


def parse_iso_date(raw: Any) -> dt.date | None:
    text_value = text(raw)
    if not text_value:
        return None
    try:
        return dt.date.fromisoformat(text_value[:10])
    except ValueError:
        return None


def route_symbol(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()


def route_action(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )


def latest_candidate_date(paths: list[Path]) -> str:
    best = ""
    for path in paths:
        stem = path.name[:10]
        if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
            best = max(best, stem)
        elif len(path.name) >= 8 and path.name[:8].isdigit():
            candidate = f"{path.name[:4]}-{path.name[4:6]}-{path.name[6:8]}"
            best = max(best, candidate)
    return best


def build_payload(
    *,
    review_dir: Path,
    output_root: Path,
    freshness_payload: dict[str, Any],
    ticket_actionability_payload: dict[str, Any],
    intent_payload: dict[str, Any],
    operator_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = route_symbol(intent_payload, operator_payload)
    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    review_shortline = sorted(
        review_dir.glob("*_crypto_shortline_signal_source.json"), reverse=True
    )
    review_recent5 = sorted(review_dir.glob("*_strategy_recent5_signals.json"), reverse=True)
    daily_candidates = sorted((output_root / "daily").glob("*_signals.json"), reverse=True)
    state_daily_candidates = sorted((output_root / "state" / "output" / "daily").glob("*_signals.json"), reverse=True)
    latest_review_shortline_signal_source_date = latest_candidate_date(review_shortline)
    latest_review_recent5_date = latest_candidate_date(review_recent5)
    latest_daily_date = latest_candidate_date(daily_candidates)
    latest_state_daily_date = latest_candidate_date(state_daily_candidates)
    candidate_dates = [
        x
        for x in [
            latest_review_shortline_signal_source_date,
            latest_review_recent5_date,
            latest_daily_date,
            latest_state_daily_date,
        ]
        if x
    ]
    latest_candidate_artifact_date = max(candidate_dates) if candidate_dates else ""

    ticket_signal_source_artifact_date = text(
        freshness_payload.get("ticket_signal_source_artifact_date")
        or ticket_actionability_payload.get("ticket_signal_source_artifact_date")
    )
    route_signal_date = text(
        freshness_payload.get("route_signal_date")
        or ticket_actionability_payload.get("route_signal_date")
    )
    route_signal_age_days = (
        freshness_payload.get("route_signal_age_days")
        if freshness_payload.get("route_signal_age_days") is not None
        else ticket_actionability_payload.get("route_signal_age_days")
    )
    ticket_signal_source_age_days = (
        freshness_payload.get("ticket_signal_source_artifact_age_days")
        if freshness_payload.get("ticket_signal_source_artifact_age_days") is not None
        else ticket_actionability_payload.get("ticket_signal_source_age_days")
    )
    freshness_status = text(freshness_payload.get("freshness_status"))
    freshness_ok = bool(freshness_payload.get("freshness_ok", False))

    selected_source_date = parse_iso_date(ticket_signal_source_artifact_date)
    latest_candidate_date_obj = parse_iso_date(latest_candidate_artifact_date)
    newer_candidate_available = bool(
        selected_source_date is not None
        and latest_candidate_date_obj is not None
        and latest_candidate_date_obj > selected_source_date
    )

    if freshness_ok:
        readiness_status = "signal_source_refresh_not_required"
        readiness_decision = "signal_source_fresh_no_refresh_needed"
        refresh_needed = False
        brief = f"signal_source_refresh_not_required:{symbol}:{ticket_signal_source_artifact_date or '-'}"
        done_when = "signal source stays fresh enough for the current route"
    elif newer_candidate_available:
        readiness_status = "newer_signal_candidate_available"
        readiness_decision = "rebuild_tickets_with_newer_signal_candidate"
        refresh_needed = True
        brief = (
            f"newer_signal_candidate_available:{symbol}:{ticket_signal_source_artifact_date or '-'}"
            f"->{latest_candidate_artifact_date or '-'}"
        )
        done_when = "tickets are rebuilt from the newer crypto signal candidate and stale_signal disappears"
    else:
        readiness_status = "no_newer_crypto_signal_candidate_available"
        readiness_decision = "generate_fresh_crypto_signal_source_before_rebuild_tickets"
        refresh_needed = True
        brief = (
            f"no_newer_crypto_signal_candidate_available:{symbol}:{ticket_signal_source_artifact_date or '-'}"
            f":route_signal_date={route_signal_date or '-'}"
        )
        done_when = "a fresh crypto signal artifact newer than the current source exists and rebuilt tickets drop stale_signal"

    blocker_detail = " | ".join(
        part
        for part in [
            brief,
            (
                f"route_signal_age_days={route_signal_age_days}"
                if route_signal_age_days is not None
                else ""
            ),
            (
                f"ticket_signal_source_age_days={ticket_signal_source_age_days}"
                if ticket_signal_source_age_days is not None
                else ""
            ),
            (
                f"latest_review_shortline_signal_source_date={latest_review_shortline_signal_source_date}"
                if latest_review_shortline_signal_source_date
                else ""
            ),
            (
                f"latest_review_recent5_date={latest_review_recent5_date}"
                if latest_review_recent5_date
                else ""
            ),
            f"latest_daily_date={latest_daily_date}" if latest_daily_date else "",
            (
                f"latest_state_daily_date={latest_state_daily_date}"
                if latest_state_daily_date
                else ""
            ),
        ]
        if part
    )

    return {
        "action": "build_crypto_signal_source_refresh_readiness",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "readiness_status": readiness_status,
        "readiness_brief": brief,
        "readiness_decision": readiness_decision,
        "refresh_needed": refresh_needed,
        "newer_candidate_available": newer_candidate_available,
        "route_signal_date": route_signal_date,
        "route_signal_age_days": route_signal_age_days,
        "ticket_signal_source_artifact_date": ticket_signal_source_artifact_date,
        "ticket_signal_source_age_days": ticket_signal_source_age_days,
        "latest_review_shortline_signal_source_date": latest_review_shortline_signal_source_date,
        "latest_review_recent5_date": latest_review_recent5_date,
        "latest_daily_date": latest_daily_date,
        "latest_state_daily_date": latest_state_daily_date,
        "latest_candidate_artifact_date": latest_candidate_artifact_date,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "crypto_signal_source_freshness": text(freshness_payload.get("artifact")),
            "remote_ticket_actionability_state": text(ticket_actionability_payload.get("artifact")),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Signal Source Refresh Readiness",
            "",
            f"- brief: `{text(payload.get('readiness_brief'))}`",
            f"- decision: `{text(payload.get('readiness_decision'))}`",
            f"- refresh_needed: `{payload.get('refresh_needed')}`",
            f"- newer_candidate_available: `{payload.get('newer_candidate_available')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crypto signal source refresh readiness artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root).expanduser().resolve()
    reference_now = parse_now(args.now)

    freshness_path = find_latest(review_dir, "*_crypto_signal_source_freshness.json")
    ticket_actionability_path = find_latest(review_dir, "*_remote_ticket_actionability_state.json")
    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    if freshness_path is None or ticket_actionability_path is None or operator_path is None:
        missing = [
            name
            for name, path in (
                ("crypto_signal_source_freshness", freshness_path),
                ("remote_ticket_actionability_state", ticket_actionability_path),
                ("crypto_route_operator_brief", operator_path),
            )
            if path is None
        ]
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        review_dir=review_dir,
        output_root=output_root,
        freshness_payload=load_json_mapping(freshness_path),
        ticket_actionability_payload=load_json_mapping(ticket_actionability_path),
        intent_payload=load_json_mapping(intent_path)
        if intent_path is not None and intent_path.exists()
        else {},
        operator_payload=load_json_mapping(operator_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_signal_source_refresh_readiness.json"
    markdown = review_dir / f"{stamp}_crypto_signal_source_refresh_readiness.md"
    checksum = review_dir / f"{stamp}_crypto_signal_source_refresh_readiness_checksum.json"
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
