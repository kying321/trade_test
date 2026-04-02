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


def _route_class_priority(value: Any) -> int:
    route_class = str(value or "").strip().lower()
    if route_class == "focus_primary":
        return 0
    if route_class in {"regime_filter", "focus_with_regime_filter"}:
        return 1
    if route_class == "shadow_only":
        return 3
    return 2


def latest_commodity_paper_ticket_book_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_commodity_paper_ticket_book.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_paper_ticket_book_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def build_execution_preview(ticket_book: dict[str, Any]) -> dict[str, Any]:
    tickets = [row for row in ticket_book.get("tickets", []) if isinstance(row, dict)]
    grouped: dict[str, dict[str, Any]] = {}
    for ticket in tickets:
        batch = str(ticket.get("batch") or "").strip()
        if not batch:
            continue
        entry = grouped.setdefault(
            batch,
            {
                "batch": batch,
                "route_class": str(ticket.get("route_class") or "").strip(),
                "regime_gate": str(ticket.get("regime_gate") or "").strip(),
                "execution_mode": "paper_only",
                "preview_status": "shadow_only",
                "allow_paper_execution": False,
                "ticket_ids": [],
                "symbols": [],
                "leader_symbols": [],
                "weight_hint_sum": 0.0,
                "ticket_count": 0,
                "actionable_ticket_count": 0,
            },
        )
        ticket_id = str(ticket.get("ticket_id") or "").strip()
        symbol = str(ticket.get("symbol") or "").strip().upper()
        ticket_role = str(ticket.get("ticket_role") or "").strip()
        allow_paper_ticket = bool(ticket.get("allow_paper_ticket"))
        if ticket_id:
            entry["ticket_ids"].append(ticket_id)
        if symbol:
            entry["symbols"].append(symbol)
            if ticket_role == "leader":
                entry["leader_symbols"].append(symbol)
        entry["weight_hint_sum"] = float(entry["weight_hint_sum"]) + float(ticket.get("weight_hint") or 0.0)
        entry["ticket_count"] = int(entry["ticket_count"]) + 1
        if allow_paper_ticket:
            entry["allow_paper_execution"] = True
            entry["preview_status"] = "paper_execution_ready"
            entry["actionable_ticket_count"] = int(entry["actionable_ticket_count"]) + 1

    preview_batches = sorted(
        grouped.values(),
        key=lambda row: (
            0 if bool(row.get("allow_paper_execution")) else 1,
            _route_class_priority(row.get("route_class")),
            -float(row.get("weight_hint_sum") or 0.0),
            str(row.get("batch") or ""),
        ),
    )
    preview_ready_batches = [str(row.get("batch") or "") for row in preview_batches if bool(row.get("allow_paper_execution"))]
    shadow_only_batches = [str(row.get("batch") or "") for row in preview_batches if not bool(row.get("allow_paper_execution"))]
    next_preview = next((row for row in preview_batches if bool(row.get("allow_paper_execution"))), None)
    stack_parts: list[str] = []
    if preview_ready_batches:
        stack_parts.append("paper-execution-ready:" + ",".join(preview_ready_batches))
    if shadow_only_batches:
        stack_parts.append("shadow:" + ",".join(shadow_only_batches))

    return {
        "execution_preview_status": "paper-execution-ready" if preview_ready_batches else "paper-execution-unavailable",
        "execution_mode": "paper_only",
        "preview_ready_batches": preview_ready_batches,
        "shadow_only_batches": shadow_only_batches,
        "preview_batch_count": len(preview_batches),
        "next_execution_batch": str((next_preview or {}).get("batch") or ""),
        "next_execution_symbols": list((next_preview or {}).get("symbols") or []),
        "next_execution_ticket_ids": list((next_preview or {}).get("ticket_ids") or []),
        "next_execution_regime_gate": str((next_preview or {}).get("regime_gate") or ""),
        "next_execution_weight_hint_sum": float((next_preview or {}).get("weight_hint_sum") or 0.0),
        "preview_stack_brief": " | ".join(stack_parts),
        "preview_batches": preview_batches,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Preview",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_ticket_book_artifact: `{payload.get('source_ticket_book_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- ticket_book_status: `{payload.get('ticket_book_status') or ''}`",
        f"- execution_preview_status: `{payload.get('execution_preview_status') or ''}`",
        f"- preview_ready_batches: `{_list_text(payload.get('preview_ready_batches', []))}`",
        f"- shadow_only_batches: `{_list_text(payload.get('shadow_only_batches', []))}`",
        f"- next_execution_batch: `{payload.get('next_execution_batch') or '-'}`",
        f"- next_execution_symbols: `{_list_text(payload.get('next_execution_symbols', []))}`",
        f"- next_execution_regime_gate: `{payload.get('next_execution_regime_gate') or '-'}`",
        f"- preview_stack: `{payload.get('preview_stack_brief') or ''}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Preview Batches"])
    for row in payload.get("preview_batches", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('batch')}`: status=`{row.get('preview_status')}` symbols=`{_list_text(row.get('symbols', []))}` "
            f"leaders=`{_list_text(row.get('leader_symbols', []))}` weight=`{float(row.get('weight_hint_sum', 0.0)):.2f}`"
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a batch-level commodity paper execution preview.")
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

    ticket_book_path = latest_commodity_paper_ticket_book_source(review_dir, runtime_now)
    ticket_book_payload = json.loads(ticket_book_path.read_text(encoding="utf-8"))
    preview = build_execution_preview(ticket_book_payload)
    summary_lines = [
        f"route-status: {ticket_book_payload.get('route_status') or '-'}",
        f"ticket-book-status: {ticket_book_payload.get('ticket_book_status') or '-'}",
        f"execution-preview-status: {preview.get('execution_preview_status') or '-'}",
        f"preview-ready-batches: {_list_text(preview.get('preview_ready_batches', []))}",
        f"shadow-only-batches: {_list_text(preview.get('shadow_only_batches', []))}",
        f"next-execution-batch: {preview.get('next_execution_batch') or '-'}",
        f"next-execution-symbols: {_list_text(preview.get('next_execution_symbols', []))}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_preview.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_preview.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_preview_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_ticket_book_artifact": str(ticket_book_path),
        "route_status": str(ticket_book_payload.get("route_status") or ""),
        "ticket_book_status": str(ticket_book_payload.get("ticket_book_status") or ""),
        "summary_lines": summary_lines,
        **preview,
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
        stem="commodity_paper_execution_preview",
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
