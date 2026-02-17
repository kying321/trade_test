from __future__ import annotations

import gc
from contextlib import closing
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.storage import append_sqlite


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


if __name__ == "__main__":
    unittest.main()
