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


def latest_commodity_paper_execution_artifact_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path:
    candidates = list(review_dir.glob("*_commodity_paper_execution_artifact.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_paper_execution_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def build_execution_queue(execution_artifact: dict[str, Any]) -> dict[str, Any]:
    execution_items = [row for row in execution_artifact.get("execution_items", []) if isinstance(row, dict)]
    queued_items: list[dict[str, Any]] = []
    for index, row in enumerate(execution_items, start=1):
        allow_paper_execution = bool(row.get("allow_paper_execution", True))
        queued = dict(row)
        queued["source_execution_status"] = str(row.get("execution_status") or "")
        queued["execution_status"] = "queued" if allow_paper_execution else "blocked"
        queued["queue_rank"] = index
        queued_items.append(queued)

    actionable_items = [row for row in queued_items if bool(row.get("allow_paper_execution", True))]
    next_item = actionable_items[0] if actionable_items else (queued_items[0] if queued_items else {})
    execution_batch = str(execution_artifact.get("execution_batch") or "").strip()
    execution_symbols = [str(x).strip().upper() for x in execution_artifact.get("execution_symbols", []) if str(x).strip()]
    queue_stack_brief = ""
    if execution_batch:
        queue_stack_brief = (
            f"paper-execution-queue:{execution_batch}:{_list_text(execution_symbols, limit=10)}"
        )
    return {
        "execution_queue_status": "paper-execution-queued" if actionable_items else "paper-execution-queue-empty",
        "execution_mode": str(execution_artifact.get("execution_mode") or "paper_only"),
        "execution_batch": execution_batch,
        "execution_symbols": execution_symbols,
        "execution_ticket_ids": [str(x).strip() for x in execution_artifact.get("execution_ticket_ids", []) if str(x).strip()],
        "execution_regime_gate": str(execution_artifact.get("execution_regime_gate") or "").strip(),
        "execution_weight_hint_sum": float(execution_artifact.get("execution_weight_hint_sum", 0.0) or 0.0),
        "execution_item_count": len(queued_items),
        "actionable_execution_item_count": len(actionable_items),
        "queue_depth": len(queued_items),
        "actionable_queue_depth": len(actionable_items),
        "next_execution_id": str(next_item.get("execution_id") or "").strip(),
        "next_execution_symbol": str(next_item.get("symbol") or "").strip().upper(),
        "queue_stack_brief": queue_stack_brief,
        "queued_items": queued_items,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Queue",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_execution_artifact: `{payload.get('source_execution_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- ticket_book_status: `{payload.get('ticket_book_status') or ''}`",
        f"- execution_preview_status: `{payload.get('execution_preview_status') or ''}`",
        f"- execution_artifact_status: `{payload.get('execution_artifact_status') or ''}`",
        f"- execution_queue_status: `{payload.get('execution_queue_status') or ''}`",
        f"- execution_batch: `{payload.get('execution_batch') or '-'}`",
        f"- execution_symbols: `{_list_text(payload.get('execution_symbols', []))}`",
        f"- execution_regime_gate: `{payload.get('execution_regime_gate') or '-'}`",
        f"- next_execution_id: `{payload.get('next_execution_id') or '-'}`",
        f"- next_execution_symbol: `{payload.get('next_execution_symbol') or '-'}`",
        f"- queue_depth: `{int(payload.get('queue_depth', 0) or 0)}`",
        f"- actionable_queue_depth: `{int(payload.get('actionable_queue_depth', 0) or 0)}`",
        f"- queue_stack: `{payload.get('queue_stack_brief') or ''}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Queued Items"])
    for row in payload.get("queued_items", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('symbol')}`: queue_rank=`{int(row.get('queue_rank', 0) or 0)}` "
            f"status=`{row.get('execution_status')}` source_status=`{row.get('source_execution_status')}` "
            f"weight=`{float(row.get('weight_hint', 0.0) or 0.0):.2f}` gate=`{row.get('regime_gate') or '-'}` "
            f"proxy-ref=`{str(bool(row.get('allow_proxy_price_reference_execution', False))).lower()}`"
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a queue-oriented commodity paper execution artifact.")
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

    execution_artifact_path = latest_commodity_paper_execution_artifact_source(review_dir, runtime_now)
    execution_artifact_payload = json.loads(execution_artifact_path.read_text(encoding="utf-8"))
    execution_queue = build_execution_queue(execution_artifact_payload)
    summary_lines = [
        f"route-status: {execution_artifact_payload.get('route_status') or '-'}",
        f"ticket-book-status: {execution_artifact_payload.get('ticket_book_status') or '-'}",
        f"execution-preview-status: {execution_artifact_payload.get('execution_preview_status') or '-'}",
        f"execution-artifact-status: {execution_artifact_payload.get('execution_artifact_status') or '-'}",
        f"execution-queue-status: {execution_queue.get('execution_queue_status') or '-'}",
        f"execution-batch: {execution_queue.get('execution_batch') or '-'}",
        f"execution-symbols: {_list_text(execution_queue.get('execution_symbols', []))}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_queue.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_queue.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_queue_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_execution_artifact": str(execution_artifact_path),
        "route_status": str(execution_artifact_payload.get("route_status") or ""),
        "ticket_book_status": str(execution_artifact_payload.get("ticket_book_status") or ""),
        "execution_preview_status": str(execution_artifact_payload.get("execution_preview_status") or ""),
        "execution_artifact_status": str(execution_artifact_payload.get("execution_artifact_status") or ""),
        "summary_lines": summary_lines,
        **execution_queue,
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
        stem="commodity_paper_execution_queue",
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
