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


def latest_commodity_paper_ticket_book_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_commodity_paper_ticket_book.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_paper_ticket_book_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_preview_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_commodity_paper_execution_preview.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_paper_execution_preview_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def build_execution_artifact(ticket_book: dict[str, Any], execution_preview: dict[str, Any]) -> dict[str, Any]:
    batch = str(execution_preview.get("next_execution_batch") or "").strip()
    ticket_ids = {str(x).strip() for x in execution_preview.get("next_execution_ticket_ids", []) if str(x).strip()}
    ticket_rows = [row for row in ticket_book.get("tickets", []) if isinstance(row, dict)]
    selected_items: list[dict[str, Any]] = []
    for ticket in ticket_rows:
        ticket_id = str(ticket.get("ticket_id") or "").strip()
        if not ticket_id or ticket_id not in ticket_ids:
            continue
        symbol = str(ticket.get("symbol") or "").strip().upper()
        execution_id = f"commodity-paper-execution:{batch}:{symbol}"
        selected_items.append(
            {
                "execution_id": execution_id,
                "source_ticket_id": ticket_id,
                "batch": batch,
                "symbol": symbol,
                "route_class": str(ticket.get("route_class") or "").strip(),
                "batch_priority_rank": int(ticket.get("batch_priority_rank", 0) or 0),
                "symbol_rank": int(ticket.get("symbol_rank", 0) or 0),
                "leader_rank": int(ticket.get("leader_rank", 0) or 0),
                "ticket_role": str(ticket.get("ticket_role") or "").strip(),
                "execution_mode": "paper_only",
                "execution_status": "planned",
                "allow_paper_execution": bool(ticket.get("allow_paper_ticket")),
                "allow_live_execution": False,
                "allow_proxy_price_reference_execution": bool(ticket.get("allow_paper_ticket")),
                "execution_price_normalization_mode": "paper_proxy_reference" if bool(ticket.get("allow_paper_ticket")) else "",
                "regime_gate": str(ticket.get("regime_gate") or "").strip(),
                "weight_hint": float(ticket.get("weight_hint") or 0.0),
                "execution_note": str(ticket.get("ticket_note") or "").strip(),
            }
        )

    selected_items.sort(
        key=lambda row: (
            int(row.get("batch_priority_rank", 0) or 0),
            int(row.get("symbol_rank", 0) or 0),
            str(row.get("symbol") or ""),
        )
    )
    execution_item_count = len(selected_items)
    actionable_item_count = len([row for row in selected_items if bool(row.get("allow_paper_execution"))])
    execution_stack_brief = ""
    if batch:
        execution_stack_brief = (
            f"paper-execution-artifact:{batch}:{_list_text([row.get('symbol', '') for row in selected_items], limit=10)}"
        )
    return {
        "execution_artifact_status": "paper-execution-artifact-ready" if execution_item_count else "paper-execution-artifact-empty",
        "execution_mode": "paper_only",
        "execution_batch": batch,
        "execution_symbols": [str(row.get("symbol") or "") for row in selected_items],
        "execution_ticket_ids": [str(row.get("source_ticket_id") or "") for row in selected_items],
        "execution_regime_gate": str(execution_preview.get("next_execution_regime_gate") or ""),
        "execution_weight_hint_sum": float(execution_preview.get("next_execution_weight_hint_sum") or 0.0),
        "execution_item_count": execution_item_count,
        "actionable_execution_item_count": actionable_item_count,
        "execution_stack_brief": execution_stack_brief,
        "execution_items": selected_items,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Artifact",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_ticket_book_artifact: `{payload.get('source_ticket_book_artifact') or ''}`",
        f"- source_execution_preview_artifact: `{payload.get('source_execution_preview_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- ticket_book_status: `{payload.get('ticket_book_status') or ''}`",
        f"- execution_preview_status: `{payload.get('execution_preview_status') or ''}`",
        f"- execution_artifact_status: `{payload.get('execution_artifact_status') or ''}`",
        f"- execution_batch: `{payload.get('execution_batch') or '-'}`",
        f"- execution_symbols: `{_list_text(payload.get('execution_symbols', []))}`",
        f"- execution_regime_gate: `{payload.get('execution_regime_gate') or '-'}`",
        f"- execution_weight_hint_sum: `{float(payload.get('execution_weight_hint_sum', 0.0) or 0.0):.2f}`",
        f"- execution_stack: `{payload.get('execution_stack_brief') or ''}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Execution Items"])
    for row in payload.get("execution_items", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('symbol')}`: status=`{row.get('execution_status')}` role=`{row.get('ticket_role')}` "
            f"weight=`{float(row.get('weight_hint', 0.0) or 0.0):.2f}` gate=`{row.get('regime_gate') or '-'}` "
            f"proxy-ref=`{str(bool(row.get('allow_proxy_price_reference_execution', False))).lower()}`"
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a concrete commodity paper execution artifact for the next batch.")
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
    execution_preview_path = latest_commodity_paper_execution_preview_source(review_dir, runtime_now)
    ticket_book_payload = json.loads(ticket_book_path.read_text(encoding="utf-8"))
    execution_preview_payload = json.loads(execution_preview_path.read_text(encoding="utf-8"))
    execution_artifact = build_execution_artifact(ticket_book_payload, execution_preview_payload)
    summary_lines = [
        f"route-status: {ticket_book_payload.get('route_status') or '-'}",
        f"ticket-book-status: {ticket_book_payload.get('ticket_book_status') or '-'}",
        f"execution-preview-status: {execution_preview_payload.get('execution_preview_status') or '-'}",
        f"execution-artifact-status: {execution_artifact.get('execution_artifact_status') or '-'}",
        f"execution-batch: {execution_artifact.get('execution_batch') or '-'}",
        f"execution-symbols: {_list_text(execution_artifact.get('execution_symbols', []))}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_artifact.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_artifact.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_artifact_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_ticket_book_artifact": str(ticket_book_path),
        "source_execution_preview_artifact": str(execution_preview_path),
        "route_status": str(ticket_book_payload.get("route_status") or ""),
        "ticket_book_status": str(ticket_book_payload.get("ticket_book_status") or ""),
        "execution_preview_status": str(execution_preview_payload.get("execution_preview_status") or ""),
        "summary_lines": summary_lines,
        **execution_artifact,
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
        stem="commodity_paper_execution_artifact",
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
