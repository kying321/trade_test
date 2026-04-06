from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CpaTokenStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    chatgpt_account_id TEXT DEFAULT '',
                    chatgpt_user_id TEXT DEFAULT '',
                    plan_type TEXT DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'imported',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tokens (
                    account_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    value TEXT NOT NULL,
                    expires_at TEXT DEFAULT '',
                    last_refresh TEXT DEFAULT '',
                    source_file TEXT DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (account_id, kind)
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    run_id TEXT DEFAULT '',
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_run_events_column(conn, "run_id", "TEXT DEFAULT ''")

    def _ensure_run_events_column(self, conn: sqlite3.Connection, column_name: str, column_spec: str) -> None:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(run_events)").fetchall()}
        if column_name in columns:
            return
        conn.execute(f"ALTER TABLE run_events ADD COLUMN {column_name} {column_spec}")

    def upsert_account(
        self,
        *,
        email: str,
        provider: str,
        chatgpt_account_id: str = "",
        chatgpt_user_id: str = "",
        plan_type: str = "",
        status: str = "imported",
    ) -> int:
        now = _now_iso()
        normalized_email = str(email).strip().lower()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO accounts (
                    email, provider, chatgpt_account_id, chatgpt_user_id,
                    plan_type, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    provider=excluded.provider,
                    chatgpt_account_id=CASE
                        WHEN excluded.chatgpt_account_id <> '' THEN excluded.chatgpt_account_id
                        ELSE accounts.chatgpt_account_id
                    END,
                    chatgpt_user_id=CASE
                        WHEN excluded.chatgpt_user_id <> '' THEN excluded.chatgpt_user_id
                        ELSE accounts.chatgpt_user_id
                    END,
                    plan_type=CASE
                        WHEN excluded.plan_type <> '' THEN excluded.plan_type
                        ELSE accounts.plan_type
                    END,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    normalized_email,
                    str(provider).strip(),
                    str(chatgpt_account_id).strip(),
                    str(chatgpt_user_id).strip(),
                    str(plan_type).strip(),
                    str(status).strip() or "imported",
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT id FROM accounts WHERE email = ?", (normalized_email,)).fetchone()
        if row is None:
            raise RuntimeError(f"account upsert failed: {normalized_email}")
        return int(row["id"])

    def upsert_token(
        self,
        account_id: int,
        *,
        kind: str,
        value: str,
        expires_at: str = "",
        last_refresh: str = "",
        source_file: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tokens (
                    account_id, kind, value, expires_at, last_refresh, source_file, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, kind) DO UPDATE SET
                    value=excluded.value,
                    expires_at=excluded.expires_at,
                    last_refresh=excluded.last_refresh,
                    source_file=excluded.source_file,
                    updated_at=excluded.updated_at
                """,
                (
                    int(account_id),
                    str(kind).strip(),
                    str(value),
                    str(expires_at).strip(),
                    str(last_refresh).strip(),
                    str(source_file).strip(),
                    _now_iso(),
                ),
            )

    def _get_account_snapshot(self, email: str, *, include_tokens: bool) -> dict[str, Any]:
        normalized_email = str(email).strip().lower()
        with self._connect() as conn:
            account_row = conn.execute("SELECT * FROM accounts WHERE email = ?", (normalized_email,)).fetchone()
            if account_row is None:
                return {"account": None, "tokens": {}}
            token_rows = []
            if include_tokens:
                token_rows = conn.execute(
                    "SELECT * FROM tokens WHERE account_id = ? ORDER BY kind ASC",
                    (int(account_row["id"]),),
                ).fetchall()
        return {
            "account": dict(account_row),
            "tokens": {str(row["kind"]): dict(row) for row in token_rows},
        }

    def get_account_snapshot(self, email: str) -> dict[str, Any]:
        return self._get_account_snapshot(email, include_tokens=False)

    def get_account_snapshot_raw(self, email: str) -> dict[str, Any]:
        return self._get_account_snapshot(email, include_tokens=True)

    def record_run_event(
        self,
        account_id: int,
        *,
        run_id: str = "",
        stage: str,
        status: str,
        detail: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_events (account_id, run_id, stage, status, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(account_id),
                    str(run_id).strip(),
                    str(stage).strip(),
                    str(status).strip(),
                    str(detail).strip(),
                    _now_iso(),
                ),
            )

    def list_run_events(self, account_id: int, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if run_id is None:
                rows = conn.execute(
                    "SELECT * FROM run_events WHERE account_id = ? ORDER BY id ASC",
                    (int(account_id),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM run_events WHERE account_id = ? AND run_id = ? ORDER BY id ASC",
                    (int(account_id), str(run_id).strip()),
                ).fetchall()
        return [dict(row) for row in rows]
