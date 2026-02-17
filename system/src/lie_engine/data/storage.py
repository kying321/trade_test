from __future__ import annotations

from contextlib import closing
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
    with closing(sqlite3.connect(db_path)) as conn:
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
