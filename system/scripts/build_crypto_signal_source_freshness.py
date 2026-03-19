#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SIGNAL_MAX_AGE_DAYS = 14


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


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dedupe_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_unique(parts: list[Any], *, sep: str = " | ") -> str:
    return sep.join(dedupe_text(parts))


def parse_utc(raw: Any) -> dt.datetime | None:
    text_value = text(raw)
    if not text_value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_iso_date(raw: Any) -> dt.date | None:
    text_value = text(raw)
    if not text_value:
        return None
    try:
        return dt.date.fromisoformat(text_value[:10])
    except ValueError:
        return None


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
        "*_crypto_signal_source_freshness.json",
        "*_crypto_signal_source_freshness.md",
        "*_crypto_signal_source_freshness_checksum.json",
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


def select_ticket_surface_path(review_dir: Path, route_symbol: str) -> Path | None:
    candidates = sorted(review_dir.glob("*_signal_to_order_tickets.json"), reverse=True)
    if not candidates:
        return None
    normalized_route = text(route_symbol).upper()
    for path in candidates:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        rows = [as_dict(row) for row in as_list(payload.get("tickets")) if isinstance(row, dict)]
        surface_symbols = {text(row.get("symbol")).upper() for row in rows if text(row.get("symbol"))}
        if normalized_route and normalized_route in surface_symbols:
            return path
    return candidates[0]


def load_ticket_surface(
    *,
    tickets_path: Path | None,
    route_symbol_value: str,
    reference_now: dt.datetime,
) -> dict[str, Any]:
    if tickets_path is None:
        return {
            "artifact": "",
            "artifact_status": "missing_artifact",
            "artifact_age_seconds": None,
            "signal_source_path": "",
            "signal_source_kind": "",
            "signal_source_selection_reason": "",
            "signal_source_artifact_date": "",
            "route_row_status": "row_missing",
            "route_row_reasons": [],
            "route_row_date": "",
            "route_row_age_days": None,
            "route_row": {},
            "surface_coverage_status": "missing_surface",
        }

    payload = load_json_mapping(tickets_path)
    signal_source = as_dict(payload.get("signal_source"))
    generated_at = parse_utc(payload.get("generated_at_utc"))
    if generated_at is None:
        generated_at = dt.datetime.fromtimestamp(tickets_path.stat().st_mtime, tz=dt.timezone.utc)
    age_seconds = max(0.0, (reference_now - generated_at).total_seconds())
    rows = [as_dict(row) for row in as_list(payload.get("tickets")) if isinstance(row, dict)]
    route_row = {}
    for row in rows:
        if text(row.get("symbol")).upper() == route_symbol_value.upper():
            route_row = row
            break
    route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))
    route_row_date = text(route_row.get("date"))
    route_row_age_days = route_row.get("age_days")
    surface_symbols = {text(row.get("symbol")).upper() for row in rows if text(row.get("symbol"))}
    if not rows:
        coverage_status = "missing_surface"
    elif route_symbol_value.upper() in surface_symbols:
        coverage_status = "route_symbol_present"
    else:
        coverage_status = "route_symbol_missing"
    return {
        "artifact": str(tickets_path),
        "artifact_status": "fresh_artifact",
        "artifact_age_seconds": age_seconds,
        "signal_source_path": text(signal_source.get("path")),
        "signal_source_kind": text(signal_source.get("kind")),
        "signal_source_selection_reason": text(signal_source.get("selection_reason")),
        "signal_source_artifact_date": text(signal_source.get("artifact_date")),
        "route_row_status": "row_present" if route_row else "row_missing",
        "route_row_reasons": route_row_reasons,
        "route_row_date": route_row_date,
        "route_row_age_days": route_row_age_days,
        "route_row": route_row,
        "surface_coverage_status": coverage_status,
    }


def build_payload(
    *,
    intent_queue_path: Path | None,
    intent_queue_payload: dict[str, Any],
    crypto_route_operator_path: Path,
    crypto_route_operator_payload: dict[str, Any],
    tickets_path: Path | None,
    reference_now: dt.datetime,
    signal_max_age_days: int,
) -> dict[str, Any]:
    symbol = route_symbol(intent_queue_payload, crypto_route_operator_payload)
    action = route_action(intent_queue_payload, crypto_route_operator_payload)
    remote_market = text(intent_queue_payload.get("remote_market")) or "portfolio_margin_um"
    ticket_surface = load_ticket_surface(
        tickets_path=tickets_path,
        route_symbol_value=symbol,
        reference_now=reference_now,
    )
    route_row_reasons = dedupe_text(as_list(ticket_surface.get("route_row_reasons")))
    route_signal_date = text(ticket_surface.get("route_row_date"))
    route_signal_age_days = ticket_surface.get("route_row_age_days")
    signal_source_artifact_date = text(ticket_surface.get("signal_source_artifact_date"))
    signal_source_artifact_age_days: int | None = None
    artifact_date = parse_iso_date(signal_source_artifact_date)
    if artifact_date is not None:
        signal_source_artifact_age_days = max(0, (reference_now.date() - artifact_date).days)

    if text(ticket_surface.get("route_row_status")) == "row_missing":
        freshness_status = "route_signal_row_missing"
        freshness_decision = "rebuild_tickets_with_route_signal_source"
        freshness_ok = False
        refresh_recommended = True
        done_when = "route symbol appears in the selected crypto ticket surface with a non-stale signal row"
        freshness_brief = f"route_signal_row_missing:{symbol}:{text(ticket_surface.get('surface_coverage_status'))}"
    elif "stale_signal" in route_row_reasons:
        freshness_status = "route_signal_row_stale"
        freshness_decision = "refresh_crypto_signal_source_then_rebuild_tickets"
        freshness_ok = False
        refresh_recommended = True
        freshness_brief = ":".join(
            [
                "route_signal_row_stale",
                symbol or "-",
                route_signal_date or "-",
                f"age_days={route_signal_age_days if route_signal_age_days is not None else '-'}",
                text(ticket_surface.get("signal_source_kind")) or "-",
            ]
        )
        done_when = (
            "selected crypto signal source yields a rebuilt ticket row for the route symbol "
            "without stale_signal in reasons"
        )
    else:
        freshness_status = "route_signal_row_fresh"
        freshness_decision = "signal_source_fresh_no_refresh_needed"
        freshness_ok = True
        refresh_recommended = False
        freshness_brief = ":".join(
            [
                "route_signal_row_fresh",
                symbol or "-",
                route_signal_date or "-",
                f"age_days={route_signal_age_days if route_signal_age_days is not None else '-'}",
                text(ticket_surface.get("signal_source_kind")) or "-",
            ]
        )
        done_when = "signal source remains fresh enough for the current route ticket surface"

    blocker_detail = join_unique(
        [
            freshness_brief,
            (
                "ticket_signal_source="
                + ":".join(
                    [
                        text(ticket_surface.get("signal_source_kind")) or "-",
                        Path(text(ticket_surface.get("signal_source_path"))).name
                        if text(ticket_surface.get("signal_source_path"))
                        else "-",
                        signal_source_artifact_date or "-",
                        (
                            f"artifact_age_days={signal_source_artifact_age_days}"
                            if signal_source_artifact_age_days is not None
                            else "artifact_age_days=-"
                        ),
                        text(ticket_surface.get("signal_source_selection_reason")) or "-",
                    ]
                )
            )
            if text(ticket_surface.get("signal_source_path"))
            or text(ticket_surface.get("signal_source_kind"))
            or signal_source_artifact_date
            else "",
            (
                f"route_signal_max_age_days={signal_max_age_days}"
                if signal_max_age_days > 0
                else ""
            ),
            ",".join(route_row_reasons),
        ]
    )

    return {
        "action": "build_crypto_signal_source_freshness",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "freshness_status": freshness_status,
        "freshness_brief": freshness_brief,
        "freshness_decision": freshness_decision,
        "freshness_ok": freshness_ok,
        "refresh_recommended": refresh_recommended,
        "done_when": done_when,
        "signal_max_age_days": int(signal_max_age_days),
        "ticket_artifact": text(ticket_surface.get("artifact")),
        "ticket_artifact_status": text(ticket_surface.get("artifact_status")),
        "ticket_artifact_age_seconds": ticket_surface.get("artifact_age_seconds"),
        "ticket_surface_coverage_status": text(ticket_surface.get("surface_coverage_status")),
        "ticket_signal_source_path": text(ticket_surface.get("signal_source_path")),
        "ticket_signal_source_kind": text(ticket_surface.get("signal_source_kind")),
        "ticket_signal_source_selection_reason": text(
            ticket_surface.get("signal_source_selection_reason")
        ),
        "ticket_signal_source_artifact_date": signal_source_artifact_date,
        "ticket_signal_source_artifact_age_days": signal_source_artifact_age_days,
        "route_row_status": text(ticket_surface.get("route_row_status")),
        "route_row_reasons": route_row_reasons,
        "route_signal_date": route_signal_date,
        "route_signal_age_days": route_signal_age_days,
        "blocker_detail": blocker_detail,
        "artifacts": {
            "remote_intent_queue": str(intent_queue_path) if intent_queue_path else "",
            "crypto_route_operator_brief": str(crypto_route_operator_path),
            "signal_to_order_tickets": text(ticket_surface.get("artifact")),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Signal Source Freshness",
            "",
            f"- brief: `{text(payload.get('freshness_brief'))}`",
            f"- decision: `{text(payload.get('freshness_decision'))}`",
            f"- refresh_recommended: `{payload.get('refresh_recommended')}`",
            f"- ticket_artifact: `{text(payload.get('ticket_artifact')) or '-'}`",
            f"- signal_source: `{text(payload.get('ticket_signal_source_kind')) or '-'}:{Path(text(payload.get('ticket_signal_source_path'))).name if text(payload.get('ticket_signal_source_path')) else '-'}`",
            f"- route_signal_date: `{text(payload.get('route_signal_date')) or '-'}`",
            f"- route_signal_age_days: `{payload.get('route_signal_age_days')}`",
            f"- blocker: `{text(payload.get('blocker_detail')) or '-'}`",
            f"- done_when: `{text(payload.get('done_when')) or '-'}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crypto signal source freshness artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--signal-max-age-days", type=int, default=DEFAULT_SIGNAL_MAX_AGE_DAYS)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_queue_path = find_latest(review_dir, "*_remote_intent_queue.json")
    crypto_route_operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    if crypto_route_operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_queue_payload = (
        load_json_mapping(intent_queue_path)
        if intent_queue_path is not None and intent_queue_path.exists()
        else {}
    )
    crypto_route_operator_payload = load_json_mapping(crypto_route_operator_path)
    symbol = route_symbol(intent_queue_payload, crypto_route_operator_payload)
    tickets_path = select_ticket_surface_path(review_dir, symbol)

    payload = build_payload(
        intent_queue_path=intent_queue_path,
        intent_queue_payload=intent_queue_payload,
        crypto_route_operator_path=crypto_route_operator_path,
        crypto_route_operator_payload=crypto_route_operator_payload,
        tickets_path=tickets_path,
        reference_now=reference_now,
        signal_max_age_days=args.signal_max_age_days,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_signal_source_freshness.json"
    markdown = review_dir / f"{stamp}_crypto_signal_source_freshness.md"
    checksum = review_dir / f"{stamp}_crypto_signal_source_freshness_checksum.json"
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
