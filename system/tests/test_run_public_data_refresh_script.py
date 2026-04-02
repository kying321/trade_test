from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import sqlite3

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_public_data_refresh.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_public_data_refresh_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PublicDataRefreshScriptTests(unittest.TestCase):
    def test_main_refreshes_sqlite_feature_and_research_cache(self) -> None:
        import tempfile
        from io import StringIO
        from unittest.mock import patch

        mod = load_module()
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "workspace"
            system_root = workspace / "system"
            output_root = system_root / "output"
            sqlite_path = output_root / "artifacts" / "lie_engine.db"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(sqlite_path)
            try:
                conn.execute("CREATE TABLE macro (date TEXT, source TEXT, cpi_yoy REAL)")
                conn.executemany(
                    "INSERT INTO macro (date, source, cpi_yoy) VALUES (?, ?, ?)",
                    [
                        ("2026-03-20", "public_macro_news", 0.1),
                        ("2026-03-20", "binance_spot_public", 0.0),
                        ("2026-03-20", "bybit_spot_public", 0.0),
                        ("2026-03-20", "open_source_primary", 1.1),
                        ("2026-03-20", "open_source_secondary", 1.2),
                        ("2026-03-20", "binance.macro_proxy", 0.0),
                    ],
                )
                conn.commit()
            finally:
                conn.close()
            feature_path = output_root / "artifacts" / "feature" / "2026-03-30_bars_feature.parquet"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            feature_path.write_text("parquet", encoding="utf-8")
            cache_meta = output_root / "artifacts" / "research_cache" / "demo_meta.json"
            cache_meta.parent.mkdir(parents=True, exist_ok=True)
            cache_meta.write_text("{}", encoding="utf-8")

            ingest_calls: list[dict[str, object]] = []
            bundle_calls: list[dict[str, object]] = []

            class FakeEngine:
                def __init__(self, config_path=None) -> None:
                    _ = config_path
                    self.ctx = SimpleNamespace(
                        root=system_root,
                        output_dir=output_root,
                        sqlite_path=sqlite_path,
                        config_path=system_root / "config.yaml",
                    )

                def _core_symbols(self) -> list[str]:
                    return ["BTCUSDT", "ETHUSDT"]

                def _persisted_aux_symbols(self) -> list[str]:
                    return ["BU2606"]

                def _persisted_ingestion_symbols(self, symbols: list[str]) -> list[str]:
                    return list(dict.fromkeys(list(symbols) + ["BU2606"]))

                def _run_ingestion(self, as_of, symbols):  # noqa: ANN001
                    ingest_calls.append({"as_of": as_of.isoformat(), "symbols": list(symbols)})
                    result = SimpleNamespace(
                        normalized_bars=pd.DataFrame(
                            {
                                "ts": pd.to_datetime(["2026-03-30", "2026-03-30"]),
                                "symbol": ["BTCUSDT", "BU2606"],
                                "close": [100.0, 200.0],
                            }
                        ),
                        macro=pd.DataFrame({"date": pd.to_datetime(["2026-03-20"]), "source": ["public_macro_news"]}),
                        news=[{"event_id": "evt-1"}],
                        sentiment={"btc_return_24h": 0.01},
                    )
                    bars = result.normalized_bars[result.normalized_bars["symbol"] == "BTCUSDT"].copy()
                    return bars, result

            def fake_load_real_data_bundle(**kwargs):  # noqa: ANN001
                bundle_calls.append(dict(kwargs))
                return SimpleNamespace(
                    universe=["BTCUSDT", "ETHUSDT", "BU2606"],
                    bars=pd.DataFrame({"symbol": ["BTCUSDT", "BU2606"]}),
                    news_records=0,
                    report_records=0,
                    fetch_stats={"cache_path": str(cache_meta), "cache_hit": False},
                )

            stdout = StringIO()
            with (
                patch.object(mod, "LieEngine", FakeEngine),
                patch.object(mod, "load_real_data_bundle", fake_load_real_data_bundle),
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_public_data_refresh.py",
                        "--workspace",
                        str(workspace),
                        "--date",
                        "2026-03-30",
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mod.main()

            payload = json.loads(stdout.getvalue())

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["as_of"], "2026-03-30")
            self.assertEqual(payload["requested_symbols"], ["BTCUSDT", "ETHUSDT"])
            self.assertEqual(payload["persist_symbols"], ["BTCUSDT", "ETHUSDT", "BU2606"])
            self.assertEqual(payload["sqlite_path"], str(sqlite_path))
            self.assertEqual(payload["feature_path"], str(feature_path))
            self.assertEqual(payload["research_cache_meta"], str(cache_meta))
            self.assertEqual(payload["research_cache_universe"], ["BTCUSDT", "ETHUSDT", "BU2606"])
            self.assertEqual(ingest_calls, [{"as_of": "2026-03-30", "symbols": ["BTCUSDT", "ETHUSDT"]}])
            self.assertEqual(len(bundle_calls), 1)
            self.assertEqual(bundle_calls[0]["core_symbols"], ["BTCUSDT", "ETHUSDT", "BU2606"])
            self.assertEqual(bundle_calls[0]["strict_cutoff"].isoformat(), "2026-03-30")
            self.assertEqual(bundle_calls[0]["review_days"], 0)
            self.assertIs(bundle_calls[0]["include_post_review"], False)
            self.assertEqual(payload["macro_cleanup"]["rows_removed"], 5)
            self.assertEqual(
                payload["macro_cleanup"]["removed_sources"],
                [
                    "binance.macro_proxy",
                    "binance_spot_public",
                    "bybit_spot_public",
                    "open_source_primary",
                    "open_source_secondary",
                ],
            )

            artifact_path = system_root / "output" / "artifacts" / "public_data_refresh" / "2026-03-30_public_data_refresh.json"
            self.assertTrue(artifact_path.exists())
            self.assertEqual(json.loads(artifact_path.read_text(encoding="utf-8"))["research_cache_meta"], str(cache_meta))
            conn = sqlite3.connect(sqlite_path)
            try:
                rows = pd.read_sql_query("SELECT date, source, cpi_yoy FROM macro ORDER BY source", conn)
            finally:
                conn.close()
            self.assertEqual(rows.to_dict("records"), [{"date": "2026-03-20", "source": "public_macro_news", "cpi_yoy": 0.1}])


if __name__ == "__main__":
    unittest.main()
