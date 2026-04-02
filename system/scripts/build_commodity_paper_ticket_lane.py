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


def latest_commodity_execution_lane_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_commodity_execution_lane.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_execution_lane_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_hot_research_universe_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_hot_research_universe.json"))
    if not candidates:
        raise FileNotFoundError("no_hot_research_universe_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def _leader_symbols(batch_symbols: list[str], preferred: list[str]) -> list[str]:
    batch_set = {str(x).strip().upper() for x in batch_symbols if str(x).strip()}
    leaders = [str(x).strip().upper() for x in preferred if str(x).strip() and str(x).strip().upper() in batch_set]
    return leaders or [str(x).strip().upper() for x in batch_symbols if str(x).strip()]


def _regime_gate(route_class: str, batch: str) -> str:
    if route_class == "focus_primary":
        return "paper_only"
    if route_class == "regime_filter":
        if batch == "energy_liquids":
            return "strong_trend_only"
        return "regime_filter_required"
    return "shadow_only_no_allocation"


def _ticket_note(route_class: str, batch: str) -> str:
    if route_class == "focus_primary":
        return "Primary commodity sleeve. Route into paper tickets first; keep live disabled."
    if route_class == "regime_filter":
        if batch == "energy_liquids":
            return "Trend-only paper sleeve. Do not allocate when regime is range or unclear."
        return "Regime-filtered paper sleeve. Require regime confirmation before paper allocation."
    return "Shadow-only sleeve. Observe attribution and correlation; do not allocate."


def build_tickets(lane: dict[str, Any], universe: dict[str, Any]) -> dict[str, Any]:
    batch_map = {
        str(key).strip(): [str(v).strip().upper() for v in values if str(v).strip()]
        for key, values in dict(universe.get("batches") or {}).items()
        if isinstance(values, list)
    }

    primary_batches = [str(x).strip() for x in lane.get("focus_primary_batches", []) if str(x).strip()]
    regime_batches = [str(x).strip() for x in lane.get("focus_with_regime_filter_batches", []) if str(x).strip()]
    shadow_batches = [str(x).strip() for x in lane.get("shadow_only_batches", []) if str(x).strip()]
    leader_primary = [str(x).strip().upper() for x in lane.get("leader_symbols_primary", []) if str(x).strip()]
    leader_regime = [str(x).strip().upper() for x in lane.get("leader_symbols_regime_filter", []) if str(x).strip()]

    tickets: list[dict[str, Any]] = []
    missing_batches: list[str] = []

    def append_ticket(batch: str, route_class: str, rank: int) -> None:
        symbols = batch_map.get(batch, [])
        if not symbols:
            missing_batches.append(batch)
            return
        leaders = _leader_symbols(symbols, leader_primary if route_class == "focus_primary" else leader_regime)
        allow_paper = route_class != "shadow_only"
        ticket = {
            "ticket_id": f"commodity-paper:{batch}",
            "batch": batch,
            "route_class": route_class,
            "priority_rank": rank,
            "symbols": symbols,
            "leader_symbols": leaders,
            "allow_paper_ticket": allow_paper,
            "allow_live_ticket": False,
            "paper_execution_mode": "paper_only",
            "regime_gate": _regime_gate(route_class, batch),
            "stage": "paper_ticket_lane",
            "ticket_status": "paper_ready" if allow_paper else "shadow_only",
            "ticket_note": _ticket_note(route_class, batch),
        }
        tickets.append(ticket)

    rank = 1
    for batch in primary_batches:
        append_ticket(batch, "focus_primary", rank)
        rank += 1
    for batch in regime_batches:
        append_ticket(batch, "regime_filter", rank)
        rank += 1
    for batch in shadow_batches:
        append_ticket(batch, "shadow_only", rank)
        rank += 1

    paper_ready = [ticket["batch"] for ticket in tickets if ticket["allow_paper_ticket"]]
    shadow_only = [ticket["batch"] for ticket in tickets if not ticket["allow_paper_ticket"]]
    next_ticket = next((ticket for ticket in tickets if ticket["allow_paper_ticket"]), None)

    parts: list[str] = []
    if paper_ready:
        parts.append("paper-ready:" + ",".join(paper_ready))
    if shadow_only:
        parts.append("shadow:" + ",".join(shadow_only))

    return {
        "ticket_status": "paper-ready" if paper_ready else "paper-unavailable",
        "execution_mode": "paper_only",
        "paper_ready_batches": paper_ready,
        "shadow_only_batches": shadow_only,
        "missing_batches": missing_batches,
        "next_ticket_batch": str((next_ticket or {}).get("batch") or ""),
        "next_ticket_symbols": list((next_ticket or {}).get("leader_symbols") or (next_ticket or {}).get("symbols") or []),
        "ticket_stack_brief": " | ".join(parts),
        "tickets": tickets,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Ticket Lane",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_lane_artifact: `{payload.get('source_lane_artifact') or ''}`",
        f"- source_universe_artifact: `{payload.get('source_universe_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- ticket_status: `{payload.get('ticket_status') or ''}`",
        f"- ticket_stack: `{payload.get('ticket_stack_brief') or ''}`",
        f"- next_ticket_batch: `{payload.get('next_ticket_batch') or '-'}`",
        f"- next_ticket_symbols: `{_list_text(payload.get('next_ticket_symbols', []))}`",
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
            f"- `{ticket.get('ticket_id')}`: batch=`{ticket.get('batch')}` "
            f"class=`{ticket.get('route_class')}` status=`{ticket.get('ticket_status')}` "
            f"gate=`{ticket.get('regime_gate')}` symbols=`{_list_text(ticket.get('symbols', []))}` "
            f"leaders=`{_list_text(ticket.get('leader_symbols', []))}`"
        )
        note = str(ticket.get("ticket_note") or "").strip()
        if note:
            lines.append(f"  - {note}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a paper-only commodity ticket lane artifact.")
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

    lane_path = latest_commodity_execution_lane_source(review_dir, runtime_now)
    universe_path = latest_hot_research_universe_source(review_dir, runtime_now)
    lane_payload = json.loads(lane_path.read_text(encoding="utf-8"))
    universe_payload = json.loads(universe_path.read_text(encoding="utf-8"))

    ticket_lane = build_tickets(lane_payload, universe_payload)
    summary_lines = [
        f"route-status: {lane_payload.get('route_status') or '-'}",
        f"ticket-status: {ticket_lane.get('ticket_status') or '-'}",
        f"paper-ready: {_list_text(ticket_lane.get('paper_ready_batches', []))}",
        f"shadow-only: {_list_text(ticket_lane.get('shadow_only_batches', []))}",
        f"next-ticket-batch: {ticket_lane.get('next_ticket_batch') or '-'}",
        f"next-ticket-symbols: {_list_text(ticket_lane.get('next_ticket_symbols', []))}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_ticket_lane.json"
    md_path = review_dir / f"{stamp}_commodity_paper_ticket_lane.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_ticket_lane_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_lane_artifact": str(lane_path),
        "source_lane_status": str(lane_payload.get("status") or ""),
        "source_universe_artifact": str(universe_path),
        "source_universe_status": str(universe_payload.get("status") or ""),
        "route_status": str(lane_payload.get("route_status") or ""),
        "route_stack_brief": str(lane_payload.get("route_stack_brief") or ""),
        "commodity_focus_batch": str(lane_payload.get("next_focus_batch") or ""),
        "commodity_focus_symbols": list(lane_payload.get("next_focus_symbols") or []),
        **ticket_lane,
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
        stem="commodity_paper_ticket_lane",
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
