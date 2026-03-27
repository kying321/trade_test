#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Any

import pandas as pd


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
BRIDGE_RUNTIME_MODE = "commodity_queue_bridge"
BRIDGE_EVENT_SOURCE = "commodity_queue_bridge"
PARTIAL_BRIDGED_PROXY_STATUS = "bridge_partially_bridged_proxy_remainder"
PARTIAL_BRIDGED_STALE_STATUS = "bridge_partially_bridged_stale_remainder"
PARTIAL_BRIDGED_MISSING_STATUS = "bridge_partially_bridged_missing_remainder"


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


def _mapping_text(values: dict[str, Any], limit: int = 6) -> str:
    items = [
        f"{str(key).strip().upper()}:{str(value).strip()}"
        for key, value in sorted((values or {}).items())
        if str(key).strip() and str(value).strip()
    ]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def latest_review_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        raise FileNotFoundError(f"no_{suffix}")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_signal_tickets_artifact(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    return latest_review_artifact(review_dir, "signal_to_order_tickets", reference_now)


def default_output_root(review_dir: Path) -> Path:
    return review_dir.parent


def default_paper_positions_path(output_root: Path) -> Path:
    return output_root / "artifacts" / "paper_positions_open.json"


def default_paper_ledger_path(output_root: Path) -> Path:
    return output_root / "logs" / "paper_execution_ledger.jsonl"


def default_paper_db_path(output_root: Path) -> Path:
    return output_root / "artifacts" / "lie_engine.db"


def run_halfhour_mutex_path(output_root: Path) -> Path:
    return output_root / "state" / "run-halfhour-pulse.lock"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def run_halfhour_mutex(*, output_root: Path, timeout_seconds: float):
    lock_path = run_halfhour_mutex_path(output_root)
    ensure_parent(lock_path)
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                holder = lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                holder = ""
            if holder:
                try:
                    os.kill(int(holder), 0)
                except (ValueError, ProcessLookupError, OSError):
                    lock_path.unlink(missing_ok=True)
                    continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run-halfhour-pulse mutex timeout: {float(timeout_seconds):.1f}s")
            time.sleep(0.1)
    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_sqlite_conn(db_path: Path | str, max_retries: int = 3) -> sqlite3.Connection:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_path, timeout=15.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 15000;")
            return conn
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                last_err = exc
                time.sleep(2**attempt)
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("failed_to_connect_sqlite")


def append_sqlite(db_path: Path, table: str, df: pd.DataFrame) -> None:
    if df is None or len(df.columns) == 0:
        return
    ensure_parent(db_path)
    data = df.copy()
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
    with get_sqlite_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
        exists = cur.fetchone() is not None
        if exists:
            cur.execute(f'PRAGMA table_info("{table}")')
            existing_cols = [str(row[1]) for row in cur.fetchall()]
            existing_set = set(existing_cols)
            for col in data.columns:
                if col in existing_set:
                    continue
                sql_type = "REAL" if pd.api.types.is_numeric_dtype(data[col]) else "TEXT"
                safe_col = str(col).replace('"', '""')
                cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{safe_col}" {sql_type}')
                existing_cols.append(str(col))
                existing_set.add(str(col))
            for col in existing_cols:
                if col not in data.columns:
                    data[col] = None
            data = data[existing_cols]
        data.to_sql(table, conn, if_exists="append", index=False)
        conn.commit()


def load_existing_positions(path: Path) -> list[dict[str, Any]]:
    payload = load_json_mapping(path)
    rows = payload.get("positions", []) if isinstance(payload.get("positions"), list) else []
    return [dict(row) for row in rows if isinstance(row, dict)]


def save_positions(path: Path, *, as_of: str, positions: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps({"as_of": as_of, "positions": positions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_existing_ledger_state(path: Path) -> tuple[set[str], set[str]]:
    execution_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    if not path.exists():
        return execution_ids, idempotency_keys
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
            execution_id = str(payload.get("bridge_execution_id") or "").strip()
            if execution_id:
                execution_ids.add(execution_id)
            key = str(payload.get("bridge_idempotency_key") or "").strip()
            if key:
                idempotency_keys.add(key)
    return execution_ids, idempotency_keys


def sqlite_existing_bridge_state(path: Path, table: str) -> tuple[set[str], set[str]]:
    execution_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    if not path.exists():
        return execution_ids, idempotency_keys
    try:
        with get_sqlite_conn(path) as conn:
            cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            if not cols:
                return execution_ids, idempotency_keys
            col_names = {str(row[1]) for row in cols}
            if "bridge_execution_id" in col_names:
                rows = conn.execute(
                    f'SELECT DISTINCT bridge_execution_id FROM "{table}" WHERE bridge_execution_id IS NOT NULL'
                ).fetchall()
                execution_ids.update(str(row[0]).strip() for row in rows if str(row[0]).strip())
            if "bridge_idempotency_key" in col_names:
                rows = conn.execute(
                    f'SELECT DISTINCT bridge_idempotency_key FROM "{table}" WHERE bridge_idempotency_key IS NOT NULL'
                ).fetchall()
                idempotency_keys.update(str(row[0]).strip() for row in rows if str(row[0]).strip())
    except sqlite3.Error:
        return execution_ids, idempotency_keys
    return execution_ids, idempotency_keys


def load_existing_bridge_state(
    *,
    positions_path: Path,
    ledger_path: Path,
    sqlite_path: Path,
) -> tuple[set[str], set[str]]:
    execution_ids: set[str] = set()
    idempotency_keys: set[str] = set()

    for row in load_existing_positions(positions_path):
        execution_id = str(row.get("source_execution_id") or row.get("bridge_execution_id") or "").strip()
        if execution_id:
            execution_ids.add(execution_id)
        key = str(row.get("bridge_idempotency_key") or "").strip()
        if key:
            idempotency_keys.add(key)

    ledger_ids, ledger_keys = load_existing_ledger_state(ledger_path)
    execution_ids.update(ledger_ids)
    idempotency_keys.update(ledger_keys)

    for table in ("trade_plans", "executed_plans"):
        table_ids, table_keys = sqlite_existing_bridge_state(sqlite_path, table)
        execution_ids.update(table_ids)
        idempotency_keys.update(table_keys)
    return execution_ids, idempotency_keys


def normalize_side(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if text in {"BUY", "LONG"}:
        return "LONG"
    if text in {"SELL", "SHORT"}:
        return "SHORT"
    return ""


def safe_float(raw: Any) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if pd.isna(value):
        return 0.0
    return float(value)


def safe_int(raw: Any) -> int | None:
    try:
        value = int(raw)
    except Exception:
        return None
    return int(value)


def parse_iso_date(raw: Any) -> dt.date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if "T" in text:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.date()
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def compute_size_and_risk_pct(ticket: dict[str, Any]) -> tuple[float, float, float]:
    sizing = dict(ticket.get("sizing") or {})
    levels = dict(ticket.get("levels") or {})
    equity_usdt = max(0.0, safe_float(sizing.get("equity_usdt")))
    quote_usdt = max(0.0, safe_float(sizing.get("quote_usdt")))
    risk_budget_usdt = max(0.0, safe_float(sizing.get("risk_budget_usdt")))
    entry_price = max(0.0, safe_float(levels.get("entry_price")))
    stop_price = max(0.0, safe_float(levels.get("stop_price")))
    if equity_usdt > 0.0:
        size_pct = 100.0 * quote_usdt / equity_usdt
        risk_pct = 100.0 * risk_budget_usdt / equity_usdt
    else:
        per_unit_risk_pct = abs(entry_price - stop_price) / max(entry_price, 1e-9) * 100.0 if entry_price > 0.0 else 0.0
        max_alloc_pct = max(0.0, safe_float(sizing.get("max_alloc_pct")))
        size_pct = max_alloc_pct * 100.0
        risk_pct = per_unit_risk_pct * (size_pct / 100.0)
    return float(size_pct), float(risk_pct), float(quote_usdt)


def bridge_idempotency_key(*, execution_id: str, ticket: dict[str, Any]) -> str:
    levels = dict(ticket.get("levels") or {})
    signal = dict(ticket.get("signal") or {})
    seed = "|".join(
        [
            execution_id,
            str(ticket.get("date") or "").strip(),
            normalize_side(signal.get("side")),
            f"{safe_float(levels.get('entry_price')):.8f}",
            f"{safe_float(levels.get('stop_price')):.8f}",
            f"{safe_float(levels.get('target_price')):.8f}",
            BRIDGE_RUNTIME_MODE,
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def resolve_explicit_path(raw: str | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def parse_price_reference_source(raw: Any) -> tuple[str, str]:
    text = str(raw or "").strip()
    if not text:
        return "", ""
    if ":" not in text:
        return "", text
    provider, reference_symbol = text.split(":", 1)
    return provider.strip(), reference_symbol.strip()


def build_commodity_directional_signals(
    *,
    review_dir: Path,
    output_root: Path,
    as_of: dt.date,
    symbols: list[str],
) -> Path:
    script_path = review_dir.parents[1] / "scripts" / "build_commodity_directional_signals.py"
    cmd = [
        "python3",
        str(script_path),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--date",
        as_of.isoformat(),
        "--symbols",
        ",".join(symbols),
        "--max-age-days",
        "14",
        "--enable-state-carry",
        "--state-carry-max-age-days",
        "5",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"commodity_signal_build_failed: {proc.stderr.strip() or proc.stdout.strip()}")
    payload = json.loads(proc.stdout)
    json_path = Path(str(payload.get("json") or "")).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError("commodity_signal_build_output_missing")
    return json_path


def build_signal_tickets(
    *,
    review_dir: Path,
    output_root: Path,
    as_of: dt.date,
    symbols: list[str],
) -> tuple[Path, Path]:
    commodity_signal_path = build_commodity_directional_signals(
        review_dir=review_dir,
        output_root=output_root,
        as_of=as_of,
        symbols=symbols,
    )
    script_path = review_dir.parents[1] / "scripts" / "build_order_ticket.py"
    cmd = [
        "python3",
        str(script_path),
        "--date",
        as_of.isoformat(),
        "--signals-json",
        str(commodity_signal_path),
        "--symbols",
        ",".join(symbols),
        "--output-root",
        str(output_root),
        "--review-dir",
        str(review_dir),
        "--output-dir",
        str(review_dir),
        "--min-confidence",
        "14",
        "--min-convexity",
        "1.2",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"signal_ticket_build_failed: {proc.stderr.strip() or proc.stdout.strip()}")
    payload = json.loads(proc.stdout)
    json_path = Path(str(payload.get("json") or "")).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError("signal_ticket_build_output_missing")
    return json_path, commodity_signal_path


def write_empty_signal_tickets(
    *,
    review_dir: Path,
    runtime_now: dt.datetime,
    as_of: dt.date,
) -> Path:
    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_signal_to_order_tickets.json"
    payload = {
        "generated_at_utc": fmt_utc(runtime_now),
        "as_of": as_of.isoformat(),
        "symbols": [],
        "tickets": [],
        "summary": {
            "ticket_count": 0,
            "allowed_count": 0,
            "missing_count": 0,
            "stale_count": 0,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json_path


def resolve_signal_ticket_artifact(
    *,
    review_dir: Path,
    output_root: Path,
    queue_payload: dict[str, Any],
    runtime_now: dt.datetime,
    explicit_signal_tickets_path: Path | None,
    build_tickets: bool,
) -> tuple[Path, Path | None]:
    if explicit_signal_tickets_path is not None:
        return explicit_signal_tickets_path, None
    execution_symbols = [str(x).strip().upper() for x in queue_payload.get("execution_symbols", []) if str(x).strip()]
    queue_as_of = parse_now(queue_payload.get("as_of"))
    if not execution_symbols:
        return write_empty_signal_tickets(
            review_dir=review_dir,
            runtime_now=runtime_now,
            as_of=queue_as_of.date() if queue_as_of else runtime_now.date(),
        ), None
    if build_tickets and execution_symbols:
        return build_signal_tickets(
            review_dir=review_dir,
            output_root=output_root,
            as_of=queue_as_of.date() if queue_as_of else runtime_now.date(),
            symbols=execution_symbols,
        )
    return latest_signal_tickets_artifact(review_dir, runtime_now), None


def signal_ticket_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tickets = payload.get("tickets", []) if isinstance(payload.get("tickets"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in tickets:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            out[symbol] = dict(row)
    return out


def build_bridge_items(
    *,
    queue_payload: dict[str, Any],
    signal_ticket_payload: dict[str, Any],
    existing_execution_ids: set[str],
    existing_idempotency_keys: set[str],
) -> dict[str, Any]:
    queued_items = [dict(row) for row in queue_payload.get("queued_items", []) if isinstance(row, dict)]
    tickets_by_symbol = signal_ticket_map(signal_ticket_payload)
    bridge_items: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    already_present_count = 0
    signal_missing_count = 0
    signal_stale_count = 0
    signal_proxy_price_only_count = 0
    signal_stale_age_days: dict[str, int] = {}
    ticket_as_of_date = parse_iso_date(signal_ticket_payload.get("as_of")) or parse_iso_date(queue_payload.get("as_of"))

    for row in queued_items:
        execution_id = str(row.get("execution_id") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        ticket = dict(tickets_by_symbol.get(symbol) or {})
        signal = dict(ticket.get("signal") or {})
        levels = dict(ticket.get("levels") or {})
        allow_paper_execution = bool(row.get("allow_paper_execution"))
        allow_live_execution = bool(row.get("allow_live_execution"))
        allow_proxy_price_reference_execution = bool(row.get("allow_proxy_price_reference_execution"))
        execution_price_normalization_mode = str(row.get("execution_price_normalization_mode") or "").strip()
        side = normalize_side(signal.get("side"))
        signal_execution_price_ready = bool(signal.get("execution_price_ready", True))
        signal_price_reference_kind = str(signal.get("price_reference_kind") or "").strip()
        signal_price_reference_source = str(signal.get("price_reference_source") or "").strip()
        signal_price_reference_provider, signal_price_reference_symbol = parse_price_reference_source(
            signal_price_reference_source
        )
        signal_date_text = str(ticket.get("date") or "").strip()
        signal_date = parse_iso_date(signal_date_text)
        signal_age_days = safe_int(ticket.get("age_days"))
        if signal_age_days is None and signal_date is not None and ticket_as_of_date is not None:
            signal_age_days = max(0, int((ticket_as_of_date - signal_date).days))
        entry_price = safe_float(levels.get("entry_price"))
        stop_price = safe_float(levels.get("stop_price"))
        target_price = safe_float(levels.get("target_price"))
        allowed = bool(ticket.get("allowed"))
        item_reasons: list[str] = []
        ignored_ticket_reasons: list[str] = []
        paper_proxy_price_normalized = False
        if not ticket:
            item_reasons.append("signal_ticket_missing")
        elif not allowed:
            raw_reasons = [str(x).strip() for x in ticket.get("reasons", []) if str(x).strip()]
            filtered_reasons = list(raw_reasons)
            if allow_paper_execution and not allow_live_execution:
                filtered_reasons = [
                    reason for reason in filtered_reasons if reason not in {"unsupported_symbol", "size_below_min_notional"}
                ]
                if allow_proxy_price_reference_execution and signal_price_reference_source:
                    filtered_reasons = [reason for reason in filtered_reasons if reason != "proxy_price_reference_only"]
                    paper_proxy_price_normalized = "proxy_price_reference_only" in raw_reasons
                ignored_ticket_reasons = [reason for reason in raw_reasons if reason not in filtered_reasons]
            if filtered_reasons:
                item_reasons.extend(filtered_reasons)
            elif not raw_reasons:
                item_reasons.append("signal_not_allowed")
        else:
            if not side:
                item_reasons.append("signal_side_missing")
            if min(entry_price, stop_price, target_price) <= 0.0:
                item_reasons.append("signal_levels_invalid")
            if (
                allow_paper_execution
                and not allow_live_execution
                and allow_proxy_price_reference_execution
                and signal_price_reference_source
                and not signal_execution_price_ready
            ):
                paper_proxy_price_normalized = True
        size_pct, risk_pct, quote_usdt = compute_size_and_risk_pct(ticket) if ticket else (0.0, 0.0, 0.0)
        key = bridge_idempotency_key(execution_id=execution_id, ticket=ticket or {}) if execution_id and ticket else ""
        already_present = bool(execution_id and execution_id in existing_execution_ids) or bool(
            key and key in existing_idempotency_keys
        )
        if already_present:
            bridge_status = "already_bridged"
            already_present_count += 1
        elif item_reasons:
            blocked_count += 1
            has_proxy_price_only = "proxy_price_reference_only" in item_reasons
            has_stale_signal = "stale_signal" in item_reasons
            if has_proxy_price_only:
                signal_proxy_price_only_count += 1
            if has_stale_signal:
                signal_stale_count += 1
                if symbol and signal_age_days is not None:
                    signal_stale_age_days[symbol] = int(signal_age_days)
            if has_proxy_price_only:
                bridge_status = "blocked_proxy_price_reference_only"
            elif has_stale_signal:
                bridge_status = "blocked_stale_directional_signal"
            else:
                bridge_status = "blocked_missing_directional_signal"
                signal_missing_count += 1
        else:
            bridge_status = "bridge_ready"
            ready_count += 1
        bridge_items.append(
            {
                **row,
                "bridge_status": bridge_status,
                "bridge_reasons": item_reasons,
                "signal_ticket_present": bool(ticket),
                "signal_ticket_allowed": allowed,
                "signal_date": signal_date_text,
                "signal_age_days": signal_age_days,
                "signal_source_side": side,
                "signal_execution_price_ready": signal_execution_price_ready,
                "signal_price_reference_kind": signal_price_reference_kind,
                "signal_price_reference_source": signal_price_reference_source,
                "signal_price_reference_provider": signal_price_reference_provider,
                "signal_price_reference_symbol": signal_price_reference_symbol,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "size_pct": float(size_pct),
                "risk_pct": float(risk_pct),
                "quote_usdt": float(quote_usdt),
                "bridge_idempotency_key": key,
                "already_present": already_present,
                "allow_proxy_price_reference_execution": allow_proxy_price_reference_execution,
                "execution_price_normalization_mode": execution_price_normalization_mode,
                "paper_proxy_price_normalized": paper_proxy_price_normalized,
                "paper_only_ignored_ticket_reasons": ignored_ticket_reasons,
                "signal_ticket_reasons": [str(x).strip() for x in ticket.get("reasons", []) if str(x).strip()]
                if ticket
                else [],
            }
        )

    next_ready = next((row for row in bridge_items if str(row.get("bridge_status") or "") == "bridge_ready"), {})
    next_blocked = next(
        (
            row
            for row in bridge_items
            if str(row.get("bridge_status") or "") in {
                "blocked_missing_directional_signal",
                "blocked_stale_directional_signal",
                "blocked_proxy_price_reference_only",
            }
        ),
        {},
    )
    already_bridged_symbols = [
        str(row.get("symbol") or "").strip().upper()
        for row in bridge_items
        if str(row.get("bridge_status") or "") == "already_bridged" and str(row.get("symbol") or "").strip()
    ]
    if ready_count > 0:
        bridge_status = "bridge_ready"
    elif already_present_count > 0 and blocked_count == 0:
        bridge_status = "bridge_noop_already_bridged"
    elif already_present_count > 0 and signal_proxy_price_only_count > 0:
        bridge_status = PARTIAL_BRIDGED_PROXY_STATUS
    elif already_present_count > 0 and signal_stale_count > 0:
        bridge_status = PARTIAL_BRIDGED_STALE_STATUS
    elif already_present_count > 0 and signal_missing_count > 0:
        bridge_status = PARTIAL_BRIDGED_MISSING_STATUS
    elif signal_proxy_price_only_count > 0:
        bridge_status = "blocked_proxy_price_reference_only"
    elif signal_stale_count > 0:
        bridge_status = "blocked_stale_directional_signal"
    elif signal_missing_count > 0:
        bridge_status = "blocked_missing_directional_signal"
    else:
        bridge_status = "bridge_empty"

    return {
        "bridge_status": bridge_status,
        "bridge_items": bridge_items,
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "already_present_count": already_present_count,
        "signal_missing_count": signal_missing_count,
        "signal_stale_count": signal_stale_count,
        "signal_stale_age_days": signal_stale_age_days,
        "signal_proxy_price_only_count": signal_proxy_price_only_count,
        "already_bridged_symbols": already_bridged_symbols,
        "next_ready_execution_id": str(next_ready.get("execution_id") or "").strip(),
        "next_ready_symbol": str(next_ready.get("symbol") or "").strip().upper(),
        "next_blocked_execution_id": str(next_blocked.get("execution_id") or "").strip(),
        "next_blocked_symbol": str(next_blocked.get("symbol") or "").strip().upper(),
    }


def apply_bridge(
    *,
    runtime_now: dt.datetime,
    as_of: dt.date,
    output_root: Path,
    positions_path: Path,
    ledger_path: Path,
    sqlite_path: Path,
    bridge_items: list[dict[str, Any]],
    mutex_timeout_seconds: float,
) -> dict[str, Any]:
    applied_rows: list[dict[str, Any]] = []
    if not bridge_items:
        return {
            "positions_written": 0,
            "ledger_rows_written": 0,
            "trade_plan_rows_written": 0,
            "executed_plan_rows_written": 0,
            "applied_execution_ids": [],
        }
    with run_halfhour_mutex(output_root=output_root, timeout_seconds=mutex_timeout_seconds):
        existing_execution_ids, existing_idempotency_keys = load_existing_bridge_state(
            positions_path=positions_path,
            ledger_path=ledger_path,
            sqlite_path=sqlite_path,
        )
        positions = load_existing_positions(positions_path)
        ledger_rows: list[dict[str, Any]] = []
        trade_plan_rows: list[dict[str, Any]] = []
        executed_plan_rows: list[dict[str, Any]] = []

        for row in bridge_items:
            if str(row.get("bridge_status") or "") != "bridge_ready":
                continue
            execution_id = str(row.get("execution_id") or "").strip()
            key = str(row.get("bridge_idempotency_key") or "").strip()
            if execution_id in existing_execution_ids or (key and key in existing_idempotency_keys):
                continue
            side = str(row.get("signal_source_side") or "").strip().upper()
            entry_price = safe_float(row.get("entry_price"))
            stop_price = safe_float(row.get("stop_price"))
            target_price = safe_float(row.get("target_price"))
            quote_usdt = max(0.0, safe_float(row.get("quote_usdt")))
            qty = quote_usdt / max(entry_price, 1e-9) if entry_price > 0.0 else 0.0
            position_row = {
                "open_date": as_of.isoformat(),
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "side": side,
                "size_pct": float(row.get("size_pct") or 0.0),
                "risk_pct": float(row.get("risk_pct") or 0.0),
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "runtime_mode": BRIDGE_RUNTIME_MODE,
                "status": "OPEN",
                "source_execution_id": execution_id,
                "source_ticket_id": str(row.get("source_ticket_id") or "").strip(),
                "bridge_idempotency_key": key,
                "quote_usdt": quote_usdt,
                "signal_date": str(row.get("signal_date") or "").strip(),
                "regime_gate": str(row.get("regime_gate") or "").strip(),
                "execution_price_normalization_mode": str(row.get("execution_price_normalization_mode") or "").strip(),
                "paper_proxy_price_normalized": bool(row.get("paper_proxy_price_normalized", False)),
                "signal_price_reference_kind": str(row.get("signal_price_reference_kind") or "").strip(),
                "signal_price_reference_source": str(row.get("signal_price_reference_source") or "").strip(),
                "signal_price_reference_provider": str(row.get("signal_price_reference_provider") or "").strip(),
                "signal_price_reference_symbol": str(row.get("signal_price_reference_symbol") or "").strip(),
            }
            positions.append(position_row)
            ledger_rows.append(
                {
                    "domain": "paper_execution",
                    "ts": fmt_utc(runtime_now),
                    "event_source": BRIDGE_EVENT_SOURCE,
                    "symbol": position_row["symbol"],
                    "action": "OPEN",
                    "decision": "bridge_apply",
                    "route": BRIDGE_RUNTIME_MODE,
                    "side": side,
                    "qty": float(round(qty, 8)),
                    "mark_px": float(round(entry_price, 8)),
                    "fill_px": float(round(entry_price, 8)),
                    "signed_slippage_bps": 0.0,
                    "fee_rate": 0.0,
                    "fee_usdt": 0.0,
                    "notional_usdt": float(round(quote_usdt, 8)),
                    "realized_pnl_change": 0.0,
                    "paper_daily_realized_pnl_before": 0.0,
                    "paper_daily_realized_pnl_after": 0.0,
                    "paper_equity_after": None,
                    "order_mode": "paper_bridge_proxy_reference"
                    if bool(row.get("paper_proxy_price_normalized", False))
                    else "paper_bridge",
                    "bridge_execution_id": execution_id,
                    "bridge_idempotency_key": key,
                    "source_ticket_id": position_row["source_ticket_id"],
                    "batch": str(row.get("batch") or "").strip(),
                    "execution_price_normalization_mode": position_row["execution_price_normalization_mode"],
                    "paper_proxy_price_normalized": position_row["paper_proxy_price_normalized"],
                    "signal_price_reference_kind": position_row["signal_price_reference_kind"],
                    "signal_price_reference_source": position_row["signal_price_reference_source"],
                    "signal_price_reference_provider": position_row["signal_price_reference_provider"],
                    "signal_price_reference_symbol": position_row["signal_price_reference_symbol"],
                }
            )
            reason = (
                f"commodity_queue_bridge execution_id={execution_id} "
                f"source_ticket_id={position_row['source_ticket_id']} regime_gate={position_row['regime_gate']}"
            ).strip()
            trade_plan_rows.append(
                {
                    "date": as_of.isoformat(),
                    "symbol": position_row["symbol"],
                    "side": side,
                    "size_pct": position_row["size_pct"],
                    "risk_pct": position_row["risk_pct"],
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "hedge_leg": None,
                    "reason": reason,
                    "status": "OPEN",
                    "bridge_execution_id": execution_id,
                    "bridge_idempotency_key": key,
                    "source_ticket_id": position_row["source_ticket_id"],
                    "runtime_mode": BRIDGE_RUNTIME_MODE,
                    "execution_price_normalization_mode": position_row["execution_price_normalization_mode"],
                    "paper_proxy_price_normalized": position_row["paper_proxy_price_normalized"],
                    "signal_price_reference_source": position_row["signal_price_reference_source"],
                }
            )
            executed_plan_rows.append(
                {
                    "date": as_of.isoformat(),
                    "open_date": as_of.isoformat(),
                    "symbol": position_row["symbol"],
                    "side": side,
                    "direction": side,
                    "runtime_mode": BRIDGE_RUNTIME_MODE,
                    "mode": "paper",
                    "size_pct": position_row["size_pct"],
                    "risk_pct": position_row["risk_pct"],
                    "entry_price": entry_price,
                    "exit_price": None,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "exit_reason": "",
                    "hold_days": 0,
                    "holding_days": 0,
                    "status": "OPEN",
                    "bridge_execution_id": execution_id,
                    "bridge_idempotency_key": key,
                    "source_ticket_id": position_row["source_ticket_id"],
                    "execution_price_normalization_mode": position_row["execution_price_normalization_mode"],
                    "paper_proxy_price_normalized": position_row["paper_proxy_price_normalized"],
                    "signal_price_reference_source": position_row["signal_price_reference_source"],
                }
            )
            existing_execution_ids.add(execution_id)
            if key:
                existing_idempotency_keys.add(key)
            applied_rows.append(position_row)

        if applied_rows:
            save_positions(positions_path, as_of=fmt_utc(runtime_now) or as_of.isoformat(), positions=positions)
            append_jsonl(ledger_path, ledger_rows)
            append_sqlite(sqlite_path, "trade_plans", pd.DataFrame(trade_plan_rows))
            append_sqlite(sqlite_path, "executed_plans", pd.DataFrame(executed_plan_rows))

    return {
        "positions_written": len(applied_rows),
        "ledger_rows_written": len(applied_rows),
        "trade_plan_rows_written": len(applied_rows),
        "executed_plan_rows_written": len(applied_rows),
        "applied_execution_ids": [str(row.get("source_execution_id") or "") for row in applied_rows if str(row.get("source_execution_id") or "").strip()],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Bridge",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- bridge_status: `{payload.get('bridge_status') or ''}`",
        f"- apply_mode: `{payload.get('apply_mode') or ''}`",
        f"- source_execution_queue_artifact: `{payload.get('source_execution_queue_artifact') or ''}`",
        f"- source_commodity_signal_artifact: `{payload.get('source_commodity_signal_artifact') or ''}`",
        f"- source_signal_tickets_artifact: `{payload.get('source_signal_tickets_artifact') or ''}`",
        f"- execution_batch: `{payload.get('execution_batch') or '-'}`",
        f"- execution_symbols: `{_list_text(payload.get('execution_symbols', []))}`",
        f"- next_ready_execution_id: `{payload.get('next_ready_execution_id') or '-'}`",
        f"- next_ready_symbol: `{payload.get('next_ready_symbol') or '-'}`",
        f"- next_blocked_execution_id: `{payload.get('next_blocked_execution_id') or '-'}`",
        f"- next_blocked_symbol: `{payload.get('next_blocked_symbol') or '-'}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Bridge Items"])
    for row in payload.get("bridge_items", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('symbol')}`: bridge_status=`{row.get('bridge_status')}` "
            f"side=`{row.get('signal_source_side') or '-'}` "
            f"signal_date=`{row.get('signal_date') or '-'}` "
            f"signal_age_days=`{row.get('signal_age_days') if row.get('signal_age_days') is not None else '-'}` "
            f"entry=`{float(row.get('entry_price', 0.0) or 0.0):.4f}` "
            f"stop=`{float(row.get('stop_price', 0.0) or 0.0):.4f}` "
            f"target=`{float(row.get('target_price', 0.0) or 0.0):.4f}` "
            f"price_ref=`{row.get('signal_price_reference_source') or '-'}` "
            f"proxy_norm=`{str(bool(row.get('paper_proxy_price_normalized', False))).lower()}` "
            f"reasons=`{_list_text(row.get('bridge_reasons', []))}`"
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge commodity paper execution queue into paper execution evidence when directional tickets exist.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--execution-queue-json", default="")
    parser.add_argument("--signal-tickets-json", default="")
    parser.add_argument("--skip-signal-build", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--mutex-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--paper-positions-path", default="")
    parser.add_argument("--paper-ledger-path", default="")
    parser.add_argument("--paper-db-path", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(str(args.output_root).strip()).expanduser().resolve() if str(args.output_root).strip() else default_output_root(review_dir)
    runtime_now = parse_now(args.now)

    queue_path = resolve_explicit_path(args.execution_queue_json) or latest_review_artifact(
        review_dir, "commodity_paper_execution_queue", runtime_now
    )
    queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
    explicit_signal_tickets_path = resolve_explicit_path(args.signal_tickets_json)
    signal_tickets_path, commodity_signal_path = resolve_signal_ticket_artifact(
        review_dir=review_dir,
        output_root=output_root,
        queue_payload=queue_payload,
        runtime_now=runtime_now,
        explicit_signal_tickets_path=explicit_signal_tickets_path,
        build_tickets=not bool(args.skip_signal_build),
    )
    signal_tickets_payload = json.loads(signal_tickets_path.read_text(encoding="utf-8"))

    positions_path = resolve_explicit_path(args.paper_positions_path) or default_paper_positions_path(output_root)
    ledger_path = resolve_explicit_path(args.paper_ledger_path) or default_paper_ledger_path(output_root)
    sqlite_path = resolve_explicit_path(args.paper_db_path) or default_paper_db_path(output_root)

    existing_execution_ids, existing_idempotency_keys = load_existing_bridge_state(
        positions_path=positions_path,
        ledger_path=ledger_path,
        sqlite_path=sqlite_path,
    )
    bridge = build_bridge_items(
        queue_payload=queue_payload,
        signal_ticket_payload=signal_tickets_payload,
        existing_execution_ids=existing_execution_ids,
        existing_idempotency_keys=existing_idempotency_keys,
    )
    apply_result = {
        "positions_written": 0,
        "ledger_rows_written": 0,
        "trade_plan_rows_written": 0,
        "executed_plan_rows_written": 0,
        "applied_execution_ids": [],
    }
    if bool(args.apply) and bridge.get("ready_count", 0):
        ready_items = [
            row
            for row in bridge.get("bridge_items", [])
            if isinstance(row, dict) and str(row.get("bridge_status") or "") == "bridge_ready"
        ]
        queue_as_of = parse_now(queue_payload.get("as_of"))
        apply_result = apply_bridge(
            runtime_now=runtime_now,
            as_of=queue_as_of.date() if queue_as_of else runtime_now.date(),
            output_root=output_root,
            positions_path=positions_path,
            ledger_path=ledger_path,
            sqlite_path=sqlite_path,
            bridge_items=ready_items,
            mutex_timeout_seconds=float(args.mutex_timeout_seconds),
        )
        if apply_result.get("positions_written", 0):
            bridge["bridge_status"] = "paper_execution_bridged"
        else:
            bridge["bridge_status"] = "bridge_noop_already_bridged"

    summary_lines = [
        f"bridge-status: {bridge.get('bridge_status') or '-'}",
        f"execution-batch: {queue_payload.get('execution_batch') or '-'}",
        f"execution-symbols: {_list_text(queue_payload.get('execution_symbols', []))}",
        f"ready-count: {int(bridge.get('ready_count', 0) or 0)}",
        f"blocked-count: {int(bridge.get('blocked_count', 0) or 0)}",
        f"already-present-count: {int(bridge.get('already_present_count', 0) or 0)}",
        f"signal-missing-count: {int(bridge.get('signal_missing_count', 0) or 0)}",
        f"signal-stale-count: {int(bridge.get('signal_stale_count', 0) or 0)}",
        f"signal-stale-age-days: {_mapping_text(bridge.get('signal_stale_age_days', {}))}",
        f"signal-proxy-price-only-count: {int(bridge.get('signal_proxy_price_only_count', 0) or 0)}",
        f"positions-written: {int(apply_result.get('positions_written', 0) or 0)}",
        f"ledger-rows-written: {int(apply_result.get('ledger_rows_written', 0) or 0)}",
        f"trade-plan-rows-written: {int(apply_result.get('trade_plan_rows_written', 0) or 0)}",
        f"executed-plan-rows-written: {int(apply_result.get('executed_plan_rows_written', 0) or 0)}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_bridge.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_bridge.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_bridge_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "apply_mode": "apply" if bool(args.apply) else "dry_run",
        "source_execution_queue_artifact": str(queue_path),
        "source_commodity_signal_artifact": str(commodity_signal_path) if commodity_signal_path else "",
        "source_signal_tickets_artifact": str(signal_tickets_path),
        "source_signal_tickets_summary": dict(signal_tickets_payload.get("summary") or {}),
        "execution_batch": str(queue_payload.get("execution_batch") or ""),
        "execution_symbols": [str(x).strip().upper() for x in queue_payload.get("execution_symbols", []) if str(x).strip()],
        "source_paper_positions_path": str(positions_path),
        "source_paper_ledger_path": str(ledger_path),
        "source_paper_db_path": str(sqlite_path),
        **bridge,
        **apply_result,
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
        stem="commodity_paper_execution_bridge",
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
