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


def _fmt_num(raw: Any) -> str:
    try:
        value = float(raw)
    except Exception:
        text = str(raw or "").strip()
        return text or "-"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _paper_execution_status(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    evidence_snapshot = row.get("paper_execution_evidence_snapshot")
    executed_plan = evidence_snapshot.get("executed_plan", {}) if isinstance(evidence_snapshot, dict) else {}
    return str(row.get("paper_execution_status") or executed_plan.get("status") or "").strip().upper()


def latest_commodity_paper_execution_review_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path:
    candidates = list(review_dir.glob("*_commodity_paper_execution_review.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_paper_execution_review")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def resolve_explicit_path(raw: str | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_execution_retro(execution_review: dict[str, Any]) -> dict[str, Any]:
    review_items = [row for row in execution_review.get("review_items", []) if isinstance(row, dict)]
    retro_items: list[dict[str, Any]] = []
    for row in review_items:
        review_status = str(row.get("review_status") or "").strip()
        paper_status = _paper_execution_status(row)
        evidence_present = bool(row.get("paper_execution_evidence_present"))
        if review_status == "awaiting_paper_execution_close_evidence":
            retro_status = "awaiting_paper_execution_close_evidence"
        elif review_status == "awaiting_paper_execution_review":
            if evidence_present and paper_status == "OPEN":
                retro_status = "awaiting_paper_execution_close_evidence"
            else:
                retro_status = "awaiting_paper_execution_retro"
        elif review_status == "awaiting_paper_execution_fill":
            retro_status = "awaiting_paper_execution_fill"
        else:
            retro_status = "blocked"
        retro_items.append(
            {
                **dict(row),
                "retro_status": retro_status,
                "retro_note": str(row.get("execution_note") or row.get("review_status") or "").strip(),
            }
        )

    actionable_items = [row for row in retro_items if str(row.get("retro_status") or "") == "awaiting_paper_execution_retro"]
    close_evidence_items = [
        row
        for row in retro_items
        if str(row.get("retro_status") or "") == "awaiting_paper_execution_close_evidence"
    ]
    fill_waiting_items = [row for row in retro_items if str(row.get("retro_status") or "") == "awaiting_paper_execution_fill"]
    retro_pending_symbols = [
        str(row.get("symbol") or "").strip().upper() for row in actionable_items if str(row.get("symbol") or "").strip()
    ]
    close_evidence_pending_symbols = [
        str(row.get("symbol") or "").strip().upper()
        for row in close_evidence_items
        if str(row.get("symbol") or "").strip()
    ]
    fill_evidence_pending_symbols = [
        str(row.get("symbol") or "").strip().upper()
        for row in fill_waiting_items
        if str(row.get("symbol") or "").strip()
    ]
    next_item = actionable_items[0] if actionable_items else {}
    next_close_item = close_evidence_items[0] if close_evidence_items else {}
    next_fill_item = fill_waiting_items[0] if fill_waiting_items else {}
    execution_batch = str(execution_review.get("execution_batch") or "").strip()
    execution_symbols = [str(x).strip().upper() for x in execution_review.get("execution_symbols", []) if str(x).strip()]
    retro_stack_brief = ""
    if execution_batch:
        retro_stack_brief = f"paper-execution-retro:{execution_batch}:{_list_text(execution_symbols, limit=10)}"
    execution_retro_status = "paper-execution-retro-pending"
    if actionable_items and close_evidence_items and fill_waiting_items:
        execution_retro_status = "paper-execution-retro-pending-close-fill-remainder"
    elif actionable_items and close_evidence_items:
        execution_retro_status = "paper-execution-retro-pending-close-remainder"
    elif actionable_items and fill_waiting_items:
        execution_retro_status = "paper-execution-retro-pending-fill-remainder"
    elif close_evidence_items and fill_waiting_items:
        execution_retro_status = "paper-execution-close-evidence-pending-fill-remainder"
    elif close_evidence_items:
        execution_retro_status = "paper-execution-close-evidence-pending"
    elif not actionable_items:
        execution_retro_status = (
            "paper-execution-awaiting-fill-evidence" if fill_waiting_items else "paper-execution-retro-empty"
        )
    return {
        "execution_retro_status": execution_retro_status,
        "execution_mode": str(execution_review.get("execution_mode") or "paper_only"),
        "execution_batch": execution_batch,
        "execution_symbols": execution_symbols,
        "execution_ticket_ids": [str(x).strip() for x in execution_review.get("execution_ticket_ids", []) if str(x).strip()],
        "execution_regime_gate": str(execution_review.get("execution_regime_gate") or "").strip(),
        "execution_weight_hint_sum": float(execution_review.get("execution_weight_hint_sum", 0.0) or 0.0),
        "execution_item_count": int(execution_review.get("execution_item_count", 0) or 0),
        "actionable_execution_item_count": int(execution_review.get("actionable_execution_item_count", 0) or 0),
        "queue_depth": int(execution_review.get("queue_depth", 0) or 0),
        "actionable_queue_depth": int(execution_review.get("actionable_queue_depth", 0) or 0),
        "review_item_count": int(execution_review.get("review_item_count", 0) or 0),
        "actionable_review_item_count": int(execution_review.get("actionable_review_item_count", 0) or 0),
        "retro_item_count": len(retro_items),
        "actionable_retro_item_count": len(actionable_items),
        "retro_pending_symbols": retro_pending_symbols,
        "close_evidence_pending_count": len(close_evidence_items),
        "close_evidence_pending_symbols": close_evidence_pending_symbols,
        "fill_evidence_pending_count": len(fill_waiting_items),
        "fill_evidence_pending_symbols": fill_evidence_pending_symbols,
        "next_retro_execution_id": str(next_item.get("execution_id") or "").strip(),
        "next_retro_execution_symbol": str(next_item.get("symbol") or "").strip().upper(),
        "next_close_evidence_execution_id": str(next_close_item.get("execution_id") or "").strip(),
        "next_close_evidence_execution_symbol": str(next_close_item.get("symbol") or "").strip().upper(),
        "next_fill_evidence_execution_id": str(next_fill_item.get("execution_id") or "").strip(),
        "next_fill_evidence_execution_symbol": str(next_fill_item.get("symbol") or "").strip().upper(),
        "retro_stack_brief": retro_stack_brief,
        "retro_items": retro_items,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Retro",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_execution_review_artifact: `{payload.get('source_execution_review_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- ticket_book_status: `{payload.get('ticket_book_status') or ''}`",
        f"- execution_preview_status: `{payload.get('execution_preview_status') or ''}`",
        f"- execution_artifact_status: `{payload.get('execution_artifact_status') or ''}`",
        f"- execution_queue_status: `{payload.get('execution_queue_status') or ''}`",
        f"- execution_review_status: `{payload.get('execution_review_status') or ''}`",
        f"- execution_retro_status: `{payload.get('execution_retro_status') or ''}`",
        f"- execution_batch: `{payload.get('execution_batch') or '-'}`",
        f"- execution_symbols: `{_list_text(payload.get('execution_symbols', []))}`",
        f"- execution_regime_gate: `{payload.get('execution_regime_gate') or '-'}`",
        f"- next_retro_execution_id: `{payload.get('next_retro_execution_id') or '-'}`",
        f"- next_retro_execution_symbol: `{payload.get('next_retro_execution_symbol') or '-'}`",
        f"- next_close_evidence_execution_id: `{payload.get('next_close_evidence_execution_id') or '-'}`",
        f"- next_close_evidence_execution_symbol: `{payload.get('next_close_evidence_execution_symbol') or '-'}`",
        f"- next_fill_evidence_execution_id: `{payload.get('next_fill_evidence_execution_id') or '-'}`",
        f"- next_fill_evidence_execution_symbol: `{payload.get('next_fill_evidence_execution_symbol') or '-'}`",
        f"- retro_item_count: `{int(payload.get('retro_item_count', 0) or 0)}`",
        f"- actionable_retro_item_count: `{int(payload.get('actionable_retro_item_count', 0) or 0)}`",
        f"- retro_pending_symbols: `{_list_text(payload.get('retro_pending_symbols', []))}`",
        f"- close_evidence_pending_count: `{int(payload.get('close_evidence_pending_count', 0) or 0)}`",
        f"- close_evidence_pending_symbols: `{_list_text(payload.get('close_evidence_pending_symbols', []))}`",
        f"- fill_evidence_pending_count: `{int(payload.get('fill_evidence_pending_count', 0) or 0)}`",
        f"- fill_evidence_pending_symbols: `{_list_text(payload.get('fill_evidence_pending_symbols', []))}`",
        f"- retro_stack: `{payload.get('retro_stack_brief') or ''}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Retro Items"])
    for row in payload.get("retro_items", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('symbol')}`: queue_rank=`{int(row.get('queue_rank', 0) or 0)}` "
            f"retro_status=`{row.get('retro_status')}` review_status=`{row.get('review_status')}` "
            f"evidence=`{'yes' if row.get('paper_execution_evidence_present') else 'no'}` "
            f"weight=`{float(row.get('weight_hint', 0.0) or 0.0):.2f}` gate=`{row.get('regime_gate') or '-'}`"
            + (
                " "
                + " ".join(
                    [
                        f"entry=`{_fmt_num(row.get('paper_entry_price'))}`",
                        f"stop=`{_fmt_num(row.get('paper_stop_price'))}`",
                        f"target=`{_fmt_num(row.get('paper_target_price'))}`",
                        f"quote=`{_fmt_num(row.get('paper_quote_usdt'))}`",
                        f"ref=`{row.get('paper_signal_price_reference_source') or '-'}`",
                    ]
                )
                if row.get("paper_execution_evidence_present")
                else ""
            )
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a retro-oriented commodity paper execution artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--execution-review-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    execution_review_path = resolve_explicit_path(args.execution_review_json) or latest_commodity_paper_execution_review_source(
        review_dir, runtime_now
    )
    execution_review_payload = json.loads(execution_review_path.read_text(encoding="utf-8"))
    execution_retro = build_execution_retro(execution_review_payload)
    summary_lines = [
        f"route-status: {execution_review_payload.get('route_status') or '-'}",
        f"ticket-book-status: {execution_review_payload.get('ticket_book_status') or '-'}",
        f"execution-preview-status: {execution_review_payload.get('execution_preview_status') or '-'}",
        f"execution-artifact-status: {execution_review_payload.get('execution_artifact_status') or '-'}",
        f"execution-queue-status: {execution_review_payload.get('execution_queue_status') or '-'}",
        f"execution-review-status: {execution_review_payload.get('execution_review_status') or '-'}",
        f"execution-retro-status: {execution_retro.get('execution_retro_status') or '-'}",
        f"execution-batch: {execution_retro.get('execution_batch') or '-'}",
        f"execution-symbols: {_list_text(execution_retro.get('execution_symbols', []))}",
        f"actionable-retro-item-count: {int(execution_retro.get('actionable_retro_item_count', 0) or 0)}",
        f"retro-pending-symbols: {_list_text(execution_retro.get('retro_pending_symbols', []))}",
        f"close-evidence-pending-count: {int(execution_retro.get('close_evidence_pending_count', 0) or 0)}",
        f"close-evidence-pending-symbols: {_list_text(execution_retro.get('close_evidence_pending_symbols', []))}",
        f"fill-evidence-pending-count: {int(execution_retro.get('fill_evidence_pending_count', 0) or 0)}",
        f"fill-evidence-pending-symbols: {_list_text(execution_retro.get('fill_evidence_pending_symbols', []))}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_retro.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_retro.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_retro_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_execution_review_artifact": str(execution_review_path),
        "route_status": str(execution_review_payload.get("route_status") or ""),
        "ticket_book_status": str(execution_review_payload.get("ticket_book_status") or ""),
        "execution_preview_status": str(execution_review_payload.get("execution_preview_status") or ""),
        "execution_artifact_status": str(execution_review_payload.get("execution_artifact_status") or ""),
        "execution_queue_status": str(execution_review_payload.get("execution_queue_status") or ""),
        "execution_review_status": str(execution_review_payload.get("execution_review_status") or ""),
        "summary_lines": summary_lines,
        **execution_retro,
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
        stem="commodity_paper_execution_retro",
        current_paths=[json_path, md_path, checksum_path],
        keep=int(max(1, args.artifact_keep)),
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
