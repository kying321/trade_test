#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
POSITION_SNAPSHOT_FIELDS = [
    "open_date",
    "symbol",
    "side",
    "size_pct",
    "risk_pct",
    "entry_price",
    "stop_price",
    "target_price",
    "runtime_mode",
    "status",
    "source_execution_id",
    "source_ticket_id",
    "bridge_idempotency_key",
    "quote_usdt",
    "signal_date",
    "regime_gate",
    "execution_price_normalization_mode",
    "paper_proxy_price_normalized",
    "signal_price_reference_kind",
    "signal_price_reference_source",
    "signal_price_reference_provider",
    "signal_price_reference_symbol",
]
LEDGER_SNAPSHOT_FIELDS = [
    "ts",
    "symbol",
    "action",
    "decision",
    "route",
    "side",
    "qty",
    "mark_px",
    "fill_px",
    "notional_usdt",
    "order_mode",
    "bridge_execution_id",
    "bridge_idempotency_key",
    "source_ticket_id",
    "execution_price_normalization_mode",
    "paper_proxy_price_normalized",
    "signal_price_reference_kind",
    "signal_price_reference_source",
    "signal_price_reference_provider",
    "signal_price_reference_symbol",
]
PLAN_SNAPSHOT_FIELDS = [
    "date",
    "open_date",
    "symbol",
    "side",
    "direction",
    "runtime_mode",
    "mode",
    "size_pct",
    "risk_pct",
    "entry_price",
    "stop_price",
    "target_price",
    "status",
    "bridge_execution_id",
    "bridge_idempotency_key",
    "source_ticket_id",
    "execution_price_normalization_mode",
    "paper_proxy_price_normalized",
    "signal_price_reference_source",
]


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
    evidence_rows = [
        row.get("paper_execution_evidence_snapshot", {}).get("position", {})
        if isinstance(row.get("paper_execution_evidence_snapshot"), dict)
        else {},
        row.get("paper_execution_evidence_snapshot", {}).get("executed_plan", {})
        if isinstance(row.get("paper_execution_evidence_snapshot"), dict)
        else {},
        row.get("paper_execution_evidence_snapshot", {}).get("trade_plan", {})
        if isinstance(row.get("paper_execution_evidence_snapshot"), dict)
        else {},
        row.get("paper_execution_evidence_snapshot", {}).get("ledger", {})
        if isinstance(row.get("paper_execution_evidence_snapshot"), dict)
        else {},
        row,
    ]
    return str(_first_present([x for x in evidence_rows if isinstance(x, dict)], "status") or "").strip().upper()


def _snapshot_row(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields if field in row and row.get(field) is not None}


def _empty_detail_maps() -> dict[str, dict[str, dict[str, Any]]]:
    return {"by_execution": {}, "by_symbol": {}}


def _register_detail_row(
    store: dict[str, dict[str, dict[str, Any]]],
    row: dict[str, Any],
    *,
    execution_keys: list[str],
    symbol_key: str = "symbol",
) -> None:
    for key_name in execution_keys:
        execution_id = str(row.get(key_name) or "").strip()
        if execution_id:
            store["by_execution"][execution_id] = row
            break
    symbol = str(row.get(symbol_key) or "").strip().upper()
    if symbol:
        store["by_symbol"][symbol] = row


def _pick_detail_row(
    store: dict[str, dict[str, dict[str, Any]]] | None,
    *,
    execution_id: str,
    symbol: str,
) -> dict[str, Any]:
    if not store:
        return {}
    execution_text = str(execution_id or "").strip()
    symbol_text = str(symbol or "").strip().upper()
    if execution_text and execution_text in store["by_execution"]:
        return dict(store["by_execution"][execution_text])
    if symbol_text and symbol_text in store["by_symbol"]:
        return dict(store["by_symbol"][symbol_text])
    return {}


def _first_present(rows: list[dict[str, Any]], field: str) -> Any:
    for row in rows:
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def latest_commodity_paper_execution_queue_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path:
    candidates = list(review_dir.glob("*_commodity_paper_execution_queue.json"))
    if not candidates:
        raise FileNotFoundError("no_commodity_paper_execution_queue")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def default_paper_execution_ledger_path(review_dir: Path) -> Path:
    return review_dir.parent / "logs" / "paper_execution_ledger.jsonl"


def default_paper_positions_path(review_dir: Path) -> Path:
    return review_dir.parent / "artifacts" / "paper_positions_open.json"


def default_paper_db_path(review_dir: Path) -> Path:
    return review_dir.parent / "artifacts" / "lie_engine.db"


def resolve_optional_path(raw: str | None, default_path: Path) -> Path | None:
    text = str(raw or "").strip()
    path = Path(text).expanduser().resolve() if text else default_path
    return path if path.exists() else None


def resolve_explicit_path(raw: str | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_ledger_symbol_counts(path: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path is None:
        return counts
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("domain") or "").strip() != "paper_execution":
                continue
            symbol = str(payload.get("symbol") or "").strip().upper()
            if symbol:
                counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def load_open_position_symbol_counts(path: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path is None:
        return counts
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return counts
    positions = payload.get("positions", []) if isinstance(payload, dict) else []
    for row in positions:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def load_open_position_details(path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    details = _empty_detail_maps()
    if path is None:
        return details
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return details
    positions = payload.get("positions", []) if isinstance(payload, dict) else []
    for row in positions:
        if not isinstance(row, dict):
            continue
        snapshot = _snapshot_row(row, POSITION_SNAPSHOT_FIELDS)
        _register_detail_row(details, snapshot, execution_keys=["source_execution_id", "bridge_execution_id"])
    return details


def load_executed_plan_symbol_counts(path: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path is None:
        return counts
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute("SELECT symbol, COUNT(*) AS cnt FROM executed_plans GROUP BY symbol").fetchall()
    except sqlite3.Error:
        return counts
    for symbol, cnt in rows:
        symbol_text = str(symbol or "").strip().upper()
        if symbol_text:
            counts[symbol_text] = int(cnt or 0)
    return counts


def load_latest_ledger_details(path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    details = _empty_detail_maps()
    if path is None:
        return details
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("domain") or "").strip() != "paper_execution":
                continue
            snapshot = _snapshot_row(payload, LEDGER_SNAPSHOT_FIELDS)
            _register_detail_row(details, snapshot, execution_keys=["bridge_execution_id", "source_execution_id"])
    return details


def load_sqlite_table_details(path: Path | None, table: str) -> dict[str, dict[str, dict[str, Any]]]:
    details = _empty_detail_maps()
    if path is None:
        return details
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f'SELECT rowid, * FROM "{table}" ORDER BY rowid ASC').fetchall()
    except sqlite3.Error:
        return details
    for row in rows:
        payload = dict(row)
        snapshot = _snapshot_row(payload, PLAN_SNAPSHOT_FIELDS)
        _register_detail_row(details, snapshot, execution_keys=["bridge_execution_id", "source_execution_id"])
    return details


def build_execution_review(
    execution_queue: dict[str, Any],
    *,
    ledger_counts: dict[str, int] | None = None,
    open_position_counts: dict[str, int] | None = None,
    executed_plan_counts: dict[str, int] | None = None,
    position_details: dict[str, dict[str, dict[str, Any]]] | None = None,
    ledger_details: dict[str, dict[str, dict[str, Any]]] | None = None,
    trade_plan_details: dict[str, dict[str, dict[str, Any]]] | None = None,
    executed_plan_details: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    queued_items = [row for row in execution_queue.get("queued_items", []) if isinstance(row, dict)]
    ledger_counts = dict(ledger_counts or {})
    open_position_counts = dict(open_position_counts or {})
    executed_plan_counts = dict(executed_plan_counts or {})
    review_items: list[dict[str, Any]] = []
    for row in queued_items:
        execution_id = str(row.get("execution_id") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        allow_paper_execution = bool(row.get("allow_paper_execution", True))
        execution_status = str(row.get("execution_status") or "").strip()
        ledger_event_count = int(ledger_counts.get(symbol, 0) or 0)
        open_position_count = int(open_position_counts.get(symbol, 0) or 0)
        executed_plan_count = int(executed_plan_counts.get(symbol, 0) or 0)
        evidence_present = bool(ledger_event_count or open_position_count or executed_plan_count)
        position_snapshot = _pick_detail_row(position_details, execution_id=execution_id, symbol=symbol)
        ledger_snapshot = _pick_detail_row(ledger_details, execution_id=execution_id, symbol=symbol)
        trade_plan_snapshot = _pick_detail_row(trade_plan_details, execution_id=execution_id, symbol=symbol)
        executed_plan_snapshot = _pick_detail_row(executed_plan_details, execution_id=execution_id, symbol=symbol)
        evidence_rows = [position_snapshot, executed_plan_snapshot, trade_plan_snapshot, ledger_snapshot]
        paper_execution_status = _paper_execution_status(
            {
                "paper_execution_evidence_snapshot": {
                    "position": position_snapshot,
                    "ledger": ledger_snapshot,
                    "trade_plan": trade_plan_snapshot,
                    "executed_plan": executed_plan_snapshot,
                }
            }
        )
        if allow_paper_execution and execution_status == "queued":
            if not evidence_present:
                review_status = "awaiting_paper_execution_fill"
            elif paper_execution_status == "OPEN":
                review_status = "awaiting_paper_execution_close_evidence"
            else:
                review_status = "awaiting_paper_execution_review"
        else:
            review_status = "blocked"
        review_items.append(
            {
                **dict(row),
                "symbol": symbol,
                "review_status": review_status,
                "route_class": str(row.get("route_class") or row.get("source_execution_status") or "").strip(),
                "ticket_role": str(row.get("ticket_role") or "paper_candidate").strip(),
                "execution_note": str(row.get("execution_note") or row.get("source_execution_status") or "").strip(),
                "paper_execution_ledger_event_count": ledger_event_count,
                "paper_open_position_count": open_position_count,
                "paper_executed_plan_count": executed_plan_count,
                "paper_execution_evidence_present": evidence_present,
                "paper_execution_evidence_snapshot": {
                    "position": position_snapshot,
                    "ledger": ledger_snapshot,
                    "trade_plan": trade_plan_snapshot,
                    "executed_plan": executed_plan_snapshot,
                },
                "paper_open_date": _first_present(evidence_rows, "open_date") or _first_present(evidence_rows, "date"),
                "paper_execution_side": _first_present(evidence_rows, "side")
                or _first_present(evidence_rows, "direction"),
                "paper_size_pct": _first_present(evidence_rows, "size_pct"),
                "paper_risk_pct": _first_present(evidence_rows, "risk_pct"),
                "paper_entry_price": _first_present(evidence_rows, "entry_price")
                or _first_present(evidence_rows, "fill_px"),
                "paper_stop_price": _first_present(evidence_rows, "stop_price"),
                "paper_target_price": _first_present(evidence_rows, "target_price"),
                "paper_quote_usdt": _first_present(evidence_rows, "quote_usdt")
                or _first_present(evidence_rows, "notional_usdt"),
                "paper_execution_status": paper_execution_status or _first_present(evidence_rows, "status"),
                "paper_runtime_mode": _first_present(evidence_rows, "runtime_mode")
                or _first_present(evidence_rows, "mode"),
                "paper_order_mode": _first_present(evidence_rows, "order_mode"),
                "paper_fill_ts": _first_present(evidence_rows, "ts"),
                "paper_signal_date": _first_present(evidence_rows, "signal_date")
                or _first_present(evidence_rows, "date"),
                "paper_bridge_idempotency_key": _first_present(evidence_rows, "bridge_idempotency_key"),
                "paper_execution_price_normalization_mode": _first_present(
                    evidence_rows, "execution_price_normalization_mode"
                ),
                "paper_proxy_price_normalized": _first_present(evidence_rows, "paper_proxy_price_normalized"),
                "paper_signal_price_reference_kind": _first_present(evidence_rows, "signal_price_reference_kind"),
                "paper_signal_price_reference_source": _first_present(evidence_rows, "signal_price_reference_source"),
                "paper_signal_price_reference_provider": _first_present(
                    evidence_rows, "signal_price_reference_provider"
                ),
                "paper_signal_price_reference_symbol": _first_present(evidence_rows, "signal_price_reference_symbol"),
            }
        )

    actionable_items = [row for row in review_items if str(row.get("review_status") or "") == "awaiting_paper_execution_review"]
    close_evidence_items = [
        row
        for row in review_items
        if str(row.get("review_status") or "") == "awaiting_paper_execution_close_evidence"
    ]
    fill_waiting_items = [row for row in review_items if str(row.get("review_status") or "") == "awaiting_paper_execution_fill"]
    review_pending_symbols = [
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
    execution_batch = str(execution_queue.get("execution_batch") or "").strip()
    execution_symbols = [str(x).strip().upper() for x in execution_queue.get("execution_symbols", []) if str(x).strip()]
    review_stack_brief = ""
    if execution_batch:
        review_stack_brief = f"paper-execution-review:{execution_batch}:{_list_text(execution_symbols, limit=10)}"
    execution_review_status = "paper-execution-review-pending"
    if actionable_items and close_evidence_items and fill_waiting_items:
        execution_review_status = "paper-execution-review-pending-close-fill-remainder"
    elif actionable_items and close_evidence_items:
        execution_review_status = "paper-execution-review-pending-close-remainder"
    elif actionable_items and fill_waiting_items:
        execution_review_status = "paper-execution-review-pending-fill-remainder"
    elif close_evidence_items and fill_waiting_items:
        execution_review_status = "paper-execution-close-evidence-pending-fill-remainder"
    elif close_evidence_items:
        execution_review_status = "paper-execution-close-evidence-pending"
    elif not actionable_items:
        execution_review_status = (
            "paper-execution-awaiting-fill-evidence" if fill_waiting_items else "paper-execution-review-empty"
        )
    return {
        "execution_review_status": execution_review_status,
        "execution_mode": str(execution_queue.get("execution_mode") or "paper_only"),
        "execution_batch": execution_batch,
        "execution_symbols": execution_symbols,
        "execution_ticket_ids": [str(x).strip() for x in execution_queue.get("execution_ticket_ids", []) if str(x).strip()],
        "execution_regime_gate": str(execution_queue.get("execution_regime_gate") or "").strip(),
        "execution_weight_hint_sum": float(execution_queue.get("execution_weight_hint_sum", 0.0) or 0.0),
        "execution_item_count": int(execution_queue.get("execution_item_count", 0) or 0),
        "actionable_execution_item_count": int(execution_queue.get("actionable_execution_item_count", 0) or 0),
        "queue_depth": int(execution_queue.get("queue_depth", 0) or 0),
        "actionable_queue_depth": int(execution_queue.get("actionable_queue_depth", 0) or 0),
        "review_item_count": len(review_items),
        "actionable_review_item_count": len(actionable_items),
        "review_pending_symbols": review_pending_symbols,
        "close_evidence_pending_count": len(close_evidence_items),
        "close_evidence_pending_symbols": close_evidence_pending_symbols,
        "fill_evidence_pending_count": len(fill_waiting_items),
        "fill_evidence_pending_symbols": fill_evidence_pending_symbols,
        "next_review_execution_id": str(next_item.get("execution_id") or "").strip(),
        "next_review_execution_symbol": str(next_item.get("symbol") or "").strip().upper(),
        "next_close_evidence_execution_id": str(next_close_item.get("execution_id") or "").strip(),
        "next_close_evidence_execution_symbol": str(next_close_item.get("symbol") or "").strip().upper(),
        "next_fill_evidence_execution_id": str(next_fill_item.get("execution_id") or "").strip(),
        "next_fill_evidence_execution_symbol": str(next_fill_item.get("symbol") or "").strip().upper(),
        "review_stack_brief": review_stack_brief,
        "review_items": review_items,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Review",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_execution_queue_artifact: `{payload.get('source_execution_queue_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- ticket_book_status: `{payload.get('ticket_book_status') or ''}`",
        f"- execution_preview_status: `{payload.get('execution_preview_status') or ''}`",
        f"- execution_artifact_status: `{payload.get('execution_artifact_status') or ''}`",
        f"- execution_queue_status: `{payload.get('execution_queue_status') or ''}`",
        f"- execution_review_status: `{payload.get('execution_review_status') or ''}`",
        f"- execution_batch: `{payload.get('execution_batch') or '-'}`",
        f"- execution_symbols: `{_list_text(payload.get('execution_symbols', []))}`",
        f"- execution_regime_gate: `{payload.get('execution_regime_gate') or '-'}`",
        f"- next_review_execution_id: `{payload.get('next_review_execution_id') or '-'}`",
        f"- next_review_execution_symbol: `{payload.get('next_review_execution_symbol') or '-'}`",
        f"- next_close_evidence_execution_id: `{payload.get('next_close_evidence_execution_id') or '-'}`",
        f"- next_close_evidence_execution_symbol: `{payload.get('next_close_evidence_execution_symbol') or '-'}`",
        f"- next_fill_evidence_execution_id: `{payload.get('next_fill_evidence_execution_id') or '-'}`",
        f"- next_fill_evidence_execution_symbol: `{payload.get('next_fill_evidence_execution_symbol') or '-'}`",
        f"- review_item_count: `{int(payload.get('review_item_count', 0) or 0)}`",
        f"- actionable_review_item_count: `{int(payload.get('actionable_review_item_count', 0) or 0)}`",
        f"- review_pending_symbols: `{_list_text(payload.get('review_pending_symbols', []))}`",
        f"- close_evidence_pending_count: `{int(payload.get('close_evidence_pending_count', 0) or 0)}`",
        f"- close_evidence_pending_symbols: `{_list_text(payload.get('close_evidence_pending_symbols', []))}`",
        f"- fill_evidence_pending_count: `{int(payload.get('fill_evidence_pending_count', 0) or 0)}`",
        f"- fill_evidence_pending_symbols: `{_list_text(payload.get('fill_evidence_pending_symbols', []))}`",
        f"- review_stack: `{payload.get('review_stack_brief') or ''}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Review Items"])
    for row in payload.get("review_items", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('symbol')}`: queue_rank=`{int(row.get('queue_rank', 0) or 0)}` "
            f"review_status=`{row.get('review_status')}` execution_status=`{row.get('execution_status')}` "
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
    parser = argparse.ArgumentParser(description="Build a review-oriented commodity paper execution artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--execution-queue-json", default="")
    parser.add_argument("--paper-ledger-path", default="")
    parser.add_argument("--paper-positions-path", default="")
    parser.add_argument("--paper-db-path", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)
    paper_ledger_path = resolve_optional_path(args.paper_ledger_path, default_paper_execution_ledger_path(review_dir))
    paper_positions_path = resolve_optional_path(args.paper_positions_path, default_paper_positions_path(review_dir))
    paper_db_path = resolve_optional_path(args.paper_db_path, default_paper_db_path(review_dir))

    execution_queue_path = resolve_explicit_path(args.execution_queue_json) or latest_commodity_paper_execution_queue_source(
        review_dir, runtime_now
    )
    execution_queue_payload = json.loads(execution_queue_path.read_text(encoding="utf-8"))
    execution_review = build_execution_review(
        execution_queue_payload,
        ledger_counts=load_ledger_symbol_counts(paper_ledger_path),
        open_position_counts=load_open_position_symbol_counts(paper_positions_path),
        executed_plan_counts=load_executed_plan_symbol_counts(paper_db_path),
        position_details=load_open_position_details(paper_positions_path),
        ledger_details=load_latest_ledger_details(paper_ledger_path),
        trade_plan_details=load_sqlite_table_details(paper_db_path, "trade_plans"),
        executed_plan_details=load_sqlite_table_details(paper_db_path, "executed_plans"),
    )
    summary_lines = [
        f"route-status: {execution_queue_payload.get('route_status') or '-'}",
        f"ticket-book-status: {execution_queue_payload.get('ticket_book_status') or '-'}",
        f"execution-preview-status: {execution_queue_payload.get('execution_preview_status') or '-'}",
        f"execution-artifact-status: {execution_queue_payload.get('execution_artifact_status') or '-'}",
        f"execution-queue-status: {execution_queue_payload.get('execution_queue_status') or '-'}",
        f"execution-review-status: {execution_review.get('execution_review_status') or '-'}",
        f"execution-batch: {execution_review.get('execution_batch') or '-'}",
        f"execution-symbols: {_list_text(execution_review.get('execution_symbols', []))}",
        f"actionable-review-item-count: {int(execution_review.get('actionable_review_item_count', 0) or 0)}",
        f"review-pending-symbols: {_list_text(execution_review.get('review_pending_symbols', []))}",
        f"close-evidence-pending-count: {int(execution_review.get('close_evidence_pending_count', 0) or 0)}",
        f"close-evidence-pending-symbols: {_list_text(execution_review.get('close_evidence_pending_symbols', []))}",
        f"fill-evidence-pending-count: {int(execution_review.get('fill_evidence_pending_count', 0) or 0)}",
        f"fill-evidence-pending-symbols: {_list_text(execution_review.get('fill_evidence_pending_symbols', []))}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_review.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_review.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_review_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_execution_queue_artifact": str(execution_queue_path),
        "source_paper_execution_ledger_path": str(paper_ledger_path) if paper_ledger_path else "",
        "source_paper_positions_path": str(paper_positions_path) if paper_positions_path else "",
        "source_paper_db_path": str(paper_db_path) if paper_db_path else "",
        "route_status": str(execution_queue_payload.get("route_status") or ""),
        "ticket_book_status": str(execution_queue_payload.get("ticket_book_status") or ""),
        "execution_preview_status": str(execution_queue_payload.get("execution_preview_status") or ""),
        "execution_artifact_status": str(execution_queue_payload.get("execution_artifact_status") or ""),
        "execution_queue_status": str(execution_queue_payload.get("execution_queue_status") or ""),
        "summary_lines": summary_lines,
        **execution_review,
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
        stem="commodity_paper_execution_review",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )
    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "report": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
