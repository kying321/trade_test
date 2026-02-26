from __future__ import annotations

from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
import json
import sqlite3
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def write_parquet_optional(path: Path, df: pd.DataFrame) -> bool:
    ensure_parent(path)
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def configure_sqlite_connection(
    conn: sqlite3.Connection,
    *,
    busy_timeout_ms: int = 10000,
    wal: bool = True,
    synchronous: str = "NORMAL",
) -> dict[str, Any]:
    out: dict[str, Any] = {"busy_timeout_ms": int(max(0, int(busy_timeout_ms))), "journal_mode": "", "synchronous": ""}
    with closing(conn.cursor()) as cur:
        if int(out["busy_timeout_ms"]) > 0:
            cur.execute(f"PRAGMA busy_timeout={int(out['busy_timeout_ms'])}")
        if wal:
            try:
                row = cur.execute("PRAGMA journal_mode=WAL").fetchone()
                out["journal_mode"] = str((row[0] if row else "")).lower()
            except Exception:
                out["journal_mode"] = "unavailable"
        if str(synchronous).strip():
            sync = str(synchronous).strip().upper()
            try:
                cur.execute(f"PRAGMA synchronous={sync}")
                row = cur.execute("PRAGMA synchronous").fetchone()
                out["synchronous"] = str((row[0] if row else "")).lower()
            except Exception:
                out["synchronous"] = "unavailable"
    return out


def _safe_ident(name: str) -> str:
    return str(name).replace('"', '""')


def _sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    with closing(conn.cursor()) as cur:
        rows = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [str(r[0]) for r in rows if isinstance(r, (tuple, list)) and r]


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    safe_table = _safe_ident(table)
    with closing(conn.cursor()) as cur:
        rows = cur.execute(f'PRAGMA table_info("{safe_table}")').fetchall()
    out: list[str] = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) < 2:
            continue
        col = str(row[1]).strip()
        if col:
            out.append(col)
    return out


def collect_sqlite_stats(db_path: Path, *, tables: list[str] | None = None) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "exists": False,
            "db_path": str(db_path),
            "file_bytes": 0,
            "page_count": 0,
            "page_size": 0,
            "freelist_count": 0,
            "journal_mode": "missing",
            "tables": [],
        }

    out: dict[str, Any] = {
        "exists": True,
        "db_path": str(db_path),
        "file_bytes": int(db_path.stat().st_size),
        "page_count": 0,
        "page_size": 0,
        "freelist_count": 0,
        "journal_mode": "",
        "tables": [],
    }
    with closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
        configure_sqlite_connection(conn)
        with closing(conn.cursor()) as cur:
            out["page_count"] = int((cur.execute("PRAGMA page_count").fetchone() or [0])[0] or 0)
            out["page_size"] = int((cur.execute("PRAGMA page_size").fetchone() or [0])[0] or 0)
            out["freelist_count"] = int((cur.execute("PRAGMA freelist_count").fetchone() or [0])[0] or 0)
            out["journal_mode"] = str((cur.execute("PRAGMA journal_mode").fetchone() or [""])[0]).lower()

        all_tables = _sqlite_tables(conn)
        wanted = set(str(x).strip() for x in (tables or []) if str(x).strip())
        target_tables = [t for t in all_tables if (not wanted) or (t in wanted)]
        table_rows: list[dict[str, Any]] = []
        for table in target_tables:
            safe_table = _safe_ident(table)
            with closing(conn.cursor()) as cur:
                count = int((cur.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone() or [0])[0] or 0)
            table_rows.append({"table": table, "rows": int(count)})
        table_rows.sort(key=lambda x: int(x.get("rows", 0)), reverse=True)
        out["tables"] = table_rows
    return out


def apply_sqlite_retention(
    db_path: Path,
    *,
    as_of: date,
    retention_days: int,
    tables: list[str] | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    if int(retention_days) <= 0:
        raise ValueError("retention_days must be positive")

    out: dict[str, Any] = {
        "enabled": True,
        "apply": bool(apply),
        "retention_days": int(retention_days),
        "cutoff_date": (as_of - timedelta(days=int(retention_days))).isoformat(),
        "db_path": str(db_path),
        "status": "ok",
        "tables": [],
        "deleted_rows": 0,
        "eligible_rows": 0,
        "missing_tables": [],
    }
    if not db_path.exists():
        out["status"] = "skipped"
        out["reason"] = "sqlite_missing"
        return out

    cutoff = as_of - timedelta(days=int(retention_days))
    candidates = ("as_of", "date", "ts", "entry_date", "exit_date", "start", "end")
    with closing(sqlite3.connect(db_path, timeout=60.0)) as conn:
        configure_sqlite_connection(conn)
        all_tables = _sqlite_tables(conn)
        wanted = [str(x).strip() for x in (tables or []) if str(x).strip()]
        target_tables = wanted if wanted else all_tables
        for t in target_tables:
            if t not in all_tables:
                out["missing_tables"].append(t)
                continue
            cols = _sqlite_table_columns(conn, t)
            col_map = {c.lower(): c for c in cols}
            date_col = ""
            for c in candidates:
                if c.lower() in col_map:
                    date_col = col_map[c.lower()]
                    break
            safe_table = _safe_ident(t)
            table_entry: dict[str, Any] = {
                "table": t,
                "date_column": date_col,
                "rows_before": 0,
                "eligible_rows": 0,
                "deleted_rows": 0,
                "status": "ok",
            }
            with closing(conn.cursor()) as cur:
                table_entry["rows_before"] = int((cur.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone() or [0])[0] or 0)
                if not date_col:
                    table_entry["status"] = "skipped_no_date_column"
                else:
                    safe_col = _safe_ident(date_col)
                    eligible_sql = f'SELECT COUNT(*) FROM "{safe_table}" WHERE date("{safe_col}") < ?'
                    eligible_rows = int((cur.execute(eligible_sql, (cutoff.isoformat(),)).fetchone() or [0])[0] or 0)
                    table_entry["eligible_rows"] = int(eligible_rows)
                    out["eligible_rows"] = int(out["eligible_rows"]) + int(eligible_rows)
                    if apply and eligible_rows > 0:
                        delete_sql = f'DELETE FROM "{safe_table}" WHERE date("{safe_col}") < ?'
                        cur.execute(delete_sql, (cutoff.isoformat(),))
                        conn.commit()
                        rows_after = int((cur.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone() or [0])[0] or 0)
                        deleted_rows = max(0, int(table_entry["rows_before"]) - rows_after)
                        table_entry["deleted_rows"] = int(deleted_rows)
                        out["deleted_rows"] = int(out["deleted_rows"]) + int(deleted_rows)
            out["tables"].append(table_entry)
    return out


def run_sqlite_vacuum_analyze(
    db_path: Path,
    *,
    run_vacuum: bool = False,
    run_analyze: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "apply": bool(apply),
        "run_vacuum": bool(run_vacuum),
        "run_analyze": bool(run_analyze),
        "vacuum_executed": False,
        "analyze_executed": False,
        "status": "ok",
        "reason": "",
        "db_path": str(db_path),
    }
    if not db_path.exists():
        out["status"] = "skipped"
        out["reason"] = "sqlite_missing"
        return out
    if not apply:
        out["status"] = "skipped"
        out["reason"] = "dry_run"
        return out

    with closing(sqlite3.connect(db_path, timeout=120.0)) as conn:
        configure_sqlite_connection(conn)
        with closing(conn.cursor()) as cur:
            if run_analyze:
                cur.execute("ANALYZE")
                conn.commit()
                out["analyze_executed"] = True
            if run_vacuum:
                cur.execute("VACUUM")
                out["vacuum_executed"] = True
    return out


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
    with closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
        configure_sqlite_connection(conn)
        with closing(conn.cursor()) as cur:
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
