from __future__ import annotations

import gc
from contextlib import closing
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.storage import (
    append_sqlite,
    apply_sqlite_retention,
    collect_sqlite_stats,
    configure_sqlite_connection,
    run_sqlite_vacuum_analyze,
)


class StorageTests(unittest.TestCase):
    def test_append_sqlite_does_not_leave_unclosed_connection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "tmp.db"
            df = pd.DataFrame({"k": [1, 2], "v": [0.1, 0.2]})

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ResourceWarning)
                append_sqlite(db, "t", df)
                gc.collect()

            leaks = [w for w in captured if isinstance(w.message, ResourceWarning) and "unclosed database" in str(w.message)]
            self.assertEqual(leaks, [])

    def test_append_sqlite_auto_adds_missing_columns_for_existing_table(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "tmp.db"
            append_sqlite(db, "review_runs", pd.DataFrame({"as_of": ["2026-02-12"], "pass_gate": [1]}))
            append_sqlite(
                db,
                "review_runs",
                pd.DataFrame(
                    {
                        "as_of": ["2026-02-13"],
                        "pass_gate": [1],
                        "change_reasons": ['{"signal_confidence_min":"recover"}'],
                    }
                ),
            )
            import sqlite3

            with closing(sqlite3.connect(db)) as conn:
                rows = pd.read_sql_query("SELECT as_of, pass_gate, change_reasons FROM review_runs ORDER BY as_of", conn)
            self.assertEqual(len(rows), 2)
            self.assertTrue(pd.isna(rows.loc[0, "change_reasons"]))
            self.assertIn("recover", str(rows.loc[1, "change_reasons"]))

    def test_configure_sqlite_connection_sets_busy_timeout_and_wal(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "tmp.db"
            with closing(sqlite3.connect(db)) as conn:
                out = configure_sqlite_connection(conn, busy_timeout_ms=4321, wal=True)
                with closing(conn.cursor()) as cur:
                    busy_timeout = int((cur.execute("PRAGMA busy_timeout").fetchone() or [0])[0] or 0)
                    journal_mode = str((cur.execute("PRAGMA journal_mode").fetchone() or [""])[0]).lower()
                self.assertEqual(int(out.get("busy_timeout_ms", 0)), 4321)
                self.assertEqual(busy_timeout, 4321)
                self.assertIn(journal_mode, {"wal", "memory", "delete"})

    def test_apply_sqlite_retention_dry_run_and_apply(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "tmp.db"
            append_sqlite(
                db,
                "review_runs",
                pd.DataFrame(
                    {
                        "as_of": ["2026-01-01", "2026-01-20", "2026-02-13"],
                        "pass_gate": [0, 1, 1],
                    }
                ),
            )
            dry_run = apply_sqlite_retention(
                db,
                as_of=date(2026, 2, 13),
                retention_days=20,
                tables=["review_runs"],
                apply=False,
            )
            self.assertEqual(int(dry_run.get("eligible_rows", 0)), 2)
            self.assertEqual(int(dry_run.get("deleted_rows", 0)), 0)

            with closing(sqlite3.connect(db)) as conn:
                count_before = int(pd.read_sql_query("SELECT COUNT(*) AS c FROM review_runs", conn)["c"].iloc[0])
            self.assertEqual(count_before, 3)

            applied = apply_sqlite_retention(
                db,
                as_of=date(2026, 2, 13),
                retention_days=20,
                tables=["review_runs"],
                apply=True,
            )
            self.assertEqual(int(applied.get("deleted_rows", 0)), 2)
            with closing(sqlite3.connect(db)) as conn:
                count_after = int(pd.read_sql_query("SELECT COUNT(*) AS c FROM review_runs", conn)["c"].iloc[0])
            self.assertEqual(count_after, 1)

    def test_collect_stats_and_vacuum_analyze(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "tmp.db"
            append_sqlite(db, "quality", pd.DataFrame({"as_of": ["2026-02-13"], "passed": [1]}))

            before = collect_sqlite_stats(db, tables=["quality"])
            self.assertTrue(bool(before.get("exists", False)))
            self.assertEqual(str((before.get("tables", [{}])[0] if before.get("tables") else {}).get("table", "")), "quality")

            dry = run_sqlite_vacuum_analyze(db, run_vacuum=True, run_analyze=True, apply=False)
            self.assertEqual(str(dry.get("reason", "")), "dry_run")

            applied = run_sqlite_vacuum_analyze(db, run_vacuum=True, run_analyze=True, apply=True)
            self.assertTrue(bool(applied.get("vacuum_executed", False)))
            self.assertTrue(bool(applied.get("analyze_executed", False)))


if __name__ == "__main__":
    unittest.main()
