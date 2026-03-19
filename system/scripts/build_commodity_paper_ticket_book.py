#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


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


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime | None = None,
) -> tuple[list[str], list[str]]:
    effective_now = now_dt or now_utc()
    cutoff = effective_now - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda item: item.stat().st_mtime, reverse=True)

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
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
    for path in survivors[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def _list_text(values: list[str], limit: int = 6) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def latest_commodity_paper_ticket_lane_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_commodity_paper_ticket_lane.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_paper_ticket_lane_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def _weight_hint(route_class: str, ticket_role: str, allow_paper_ticket: bool) -> float:
    if not allow_paper_ticket:
        return 0.0
    if route_class == "focus_primary":
        return 1.0 if ticket_role == "leader" else 0.65
    if route_class == "regime_filter":
        return 0.75 if ticket_role == "leader" else 0.5
    return 0.25 if ticket_role == "leader" else 0.1


def build_ticket_book(ticket_lane: dict[str, Any]) -> dict[str, Any]:
    lane_tickets = [row for row in ticket_lane.get("tickets", []) if isinstance(row, dict)]
    symbol_tickets: list[dict[str, Any]] = []
    actionable_batches: list[str] = []
    shadow_batches: list[str] = []

    for lane_ticket in lane_tickets:
        batch = str(lane_ticket.get("batch") or "").strip()
        route_class = str(lane_ticket.get("route_class") or "").strip()
        symbols = [str(x).strip().upper() for x in lane_ticket.get("symbols", []) if str(x).strip()]
        leaders = [str(x).strip().upper() for x in lane_ticket.get("leader_symbols", []) if str(x).strip()]
        allow_paper_ticket = bool(lane_ticket.get("allow_paper_ticket"))
        if allow_paper_ticket:
            actionable_batches.append(batch)
        else:
            shadow_batches.append(batch)
        leader_index = {symbol: idx + 1 for idx, symbol in enumerate(leaders)}
        for idx, symbol in enumerate(symbols, start=1):
            ticket_role = "leader" if symbol in leader_index else "sleeve"
            symbol_tickets.append(
                {
                    "ticket_id": f"commodity-paper-ticket:{batch}:{symbol}",
                    "parent_ticket_id": str(lane_ticket.get("ticket_id") or ""),
                    "batch": batch,
                    "symbol": symbol,
                    "route_class": route_class,
                    "batch_priority_rank": int(lane_ticket.get("priority_rank") or 0),
                    "symbol_rank": idx,
                    "leader_rank": int(leader_index.get(symbol) or 0),
                    "ticket_role": ticket_role,
                    "allow_paper_ticket": allow_paper_ticket,
                    "allow_live_ticket": False,
                    "execution_mode": "paper_only",
                    "ticket_status": "paper_ready" if allow_paper_ticket else "shadow_only",
                    "regime_gate": str(lane_ticket.get("regime_gate") or ""),
                    "weight_hint": _weight_hint(route_class, ticket_role, allow_paper_ticket),
                    "ticket_note": str(lane_ticket.get("ticket_note") or ""),
                }
            )

    actionable_tickets = [row for row in symbol_tickets if bool(row.get("allow_paper_ticket"))]
    next_ticket = actionable_tickets[0] if actionable_tickets else None
    stack_parts: list[str] = []
    if actionable_batches:
        stack_parts.append("paper-ready:" + ",".join(actionable_batches))
    if shadow_batches:
        stack_parts.append("shadow:" + ",".join(shadow_batches))

    return {
        "ticket_book_status": "paper-ready" if actionable_tickets else "paper-unavailable",
        "execution_mode": "paper_only",
        "actionable_batches": actionable_batches,
        "shadow_batches": shadow_batches,
        "ticket_count": len(symbol_tickets),
        "actionable_ticket_count": len(actionable_tickets),
        "shadow_ticket_count": len(symbol_tickets) - len(actionable_tickets),
        "next_ticket_id": str((next_ticket or {}).get("ticket_id") or ""),
        "next_ticket_batch": str((next_ticket or {}).get("batch") or ""),
        "next_ticket_symbol": str((next_ticket or {}).get("symbol") or ""),
        "next_ticket_regime_gate": str((next_ticket or {}).get("regime_gate") or ""),
        "next_ticket_weight_hint": float((next_ticket or {}).get("weight_hint") or 0.0),
        "ticket_book_stack_brief": " | ".join(stack_parts),
        "tickets": symbol_tickets,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Ticket Book",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_ticket_lane_artifact: `{payload.get('source_ticket_lane_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- ticket_book_status: `{payload.get('ticket_book_status') or ''}`",
        f"- actionable_batches: `{_list_text(payload.get('actionable_batches', []))}`",
        f"- shadow_batches: `{_list_text(payload.get('shadow_batches', []))}`",
        f"- next_ticket_id: `{payload.get('next_ticket_id') or '-'}`",
        f"- next_ticket_symbol: `{payload.get('next_ticket_symbol') or '-'}`",
        f"- next_ticket_regime_gate: `{payload.get('next_ticket_regime_gate') or '-'}`",
        f"- ticket_book_stack: `{payload.get('ticket_book_stack_brief') or ''}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Tickets"])
    for ticket in payload.get("tickets", []):
        if not isinstance(ticket, dict):
            continue
        lines.append(
            f"- `{ticket.get('ticket_id')}`: batch=`{ticket.get('batch')}` symbol=`{ticket.get('symbol')}` "
            f"role=`{ticket.get('ticket_role')}` status=`{ticket.get('ticket_status')}` "
            f"gate=`{ticket.get('regime_gate')}` weight=`{float(ticket.get('weight_hint', 0.0)):.2f}`"
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a per-symbol commodity paper ticket book.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    ticket_lane_path = latest_commodity_paper_ticket_lane_source(review_dir, runtime_now)
    ticket_lane_payload = json.loads(ticket_lane_path.read_text(encoding="utf-8"))
    book = build_ticket_book(ticket_lane_payload)

    summary_lines = [
        f"route-status: {ticket_lane_payload.get('route_status') or '-'}",
        f"ticket-book-status: {book.get('ticket_book_status') or '-'}",
        f"actionable-batches: {_list_text(book.get('actionable_batches', []))}",
        f"shadow-batches: {_list_text(book.get('shadow_batches', []))}",
        f"next-ticket-id: {book.get('next_ticket_id') or '-'}",
        f"next-ticket-symbol: {book.get('next_ticket_symbol') or '-'}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_ticket_book.json"
    md_path = review_dir / f"{stamp}_commodity_paper_ticket_book.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_ticket_book_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_ticket_lane_artifact": str(ticket_lane_path),
        "source_ticket_lane_status": str(ticket_lane_payload.get("status") or ""),
        "route_status": str(ticket_lane_payload.get("route_status") or ""),
        "route_stack_brief": str(ticket_lane_payload.get("ticket_stack_brief") or ticket_lane_payload.get("route_stack_brief") or ""),
        "commodity_focus_batch": str(ticket_lane_payload.get("next_ticket_batch") or ""),
        "commodity_focus_symbols": list(ticket_lane_payload.get("next_ticket_symbols") or []),
        **book,
        "summary_lines": summary_lines,
        "summary_text": " | ".join(summary_lines),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": payload["as_of"],
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="commodity_paper_ticket_book",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )

    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["files"][0]["sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
