from __future__ import annotations

from datetime import datetime
import json
import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.pipeline import DataBus
from lie_engine.models import NewsEvent


class _MiniProvider:
    def __init__(self, name: str, sentiment: dict[str, float]) -> None:
        self.name = str(name)
        self._sentiment = dict(sentiment)

    def fetch_ohlcv(self, symbol: str, start, end, freq: str = "1d"):  # type: ignore[no-untyped-def]
        ts = pd.to_datetime([str(start)])
        return pd.DataFrame(
            {
                "ts": ts,
                "symbol": [symbol],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "source": [self.name],
                "asset_class": ["equity"],
            }
        )

    def fetch_macro(self, start, end):  # type: ignore[no-untyped-def]
        return pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"])

    def fetch_news(self, start_ts, end_ts, lang: str):  # type: ignore[no-untyped-def]
        return []

    def fetch_sentiment_factors(self, as_of):  # type: ignore[no-untyped-def]
        return dict(self._sentiment)

    def fetch_l2(self, symbol, start_ts, end_ts, depth: int = 20):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def fetch_trades(self, symbol, start_ts, end_ts, limit: int = 2000):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class DataPipelineTests(unittest.TestCase):
    def test_news_normalization_uses_source_reliability_and_category_extraction(self) -> None:
        bus = DataBus(
            providers=[],
            output_dir=Path("."),
            sqlite_path=Path("tmp.db"),
            completeness_min=0.99,
            conflict_max=0.005,
        )
        events = [
            NewsEvent(
                event_id="e1",
                ts=datetime(2026, 2, 13, 8, 30),
                title="国务院发布稳增长政策",
                content="政策强调产业升级和流动性支持",
                lang="zh",
                source="gov_official_feed",
                category="",
                confidence=0.75,
                entities=["A股"],
                importance=0.70,
            ),
            NewsEvent(
                event_id="e2",
                ts=datetime(2026, 2, 13, 8, 35),
                title="市场传闻",
                content="匿名渠道称将有变化",
                lang="zh",
                source="random_blog",
                category="",
                confidence=0.75,
                entities=["A股"],
                importance=0.70,
            ),
        ]

        normalized = bus._normalize_news(events)
        self.assertEqual(normalized[0].category, "政策")
        self.assertGreater(normalized[0].confidence, normalized[1].confidence)

    def test_source_confidence_distinguishes_good_and_bad_source(self) -> None:
        bus = DataBus(
            providers=[],
            output_dir=Path("."),
            sqlite_path=Path("tmp.db"),
            completeness_min=0.99,
            conflict_max=0.005,
            source_confidence_min=0.75,
            low_confidence_source_ratio_max=0.40,
        )

        ts = pd.to_datetime(["2026-02-12", "2026-02-13"])
        raw_bars = pd.DataFrame(
            {
                "ts": list(ts) * 2,
                "symbol": ["300750", "300750", "300750", "300750"],
                "open": [100.0, 101.0, 130.0, 132.0],
                "high": [101.0, 102.0, 131.0, 133.0],
                "low": [99.0, 100.0, 129.0, 131.0],
                "close": [100.0, 101.0, 130.0, 132.0],
                "volume": [1000.0, 1100.0, 900.0, 950.0],
                "source": ["open_source_primary", "open_source_primary", "random_blog", "random_blog"],
                "asset_class": ["equity", "equity", "equity", "equity"],
            }
        )
        news = bus._normalize_news(
            [
                NewsEvent(
                    event_id="n1",
                    ts=datetime(2026, 2, 13, 8, 0),
                    title="政策利好",
                    content="官方政策出台",
                    lang="zh",
                    source="open_source_primary",
                    category="政策",
                    confidence=0.9,
                    entities=["A股"],
                    importance=0.8,
                ),
                NewsEvent(
                    event_id="n2",
                    ts=datetime(2026, 2, 13, 8, 5),
                    title="传闻",
                    content="匿名消息",
                    lang="zh",
                    source="random_blog",
                    category="其他",
                    confidence=0.3,
                    entities=["A股"],
                    importance=0.5,
                ),
            ]
        )
        macro = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-02-01",
                        "2026-02-01",
                        "2026-02-01",
                        "2026-03-01",
                        "2026-03-01",
                        "2026-03-01",
                    ]
                ),
                "cpi_yoy": [0.9, 3.5, 1.0, 1.0, 3.8, 1.1],
                "ppi_yoy": [-1.1, -3.4, -1.0, -1.0, -3.2, -1.1],
                "lpr_1y": [3.45, 4.20, 3.46, 3.45, 4.30, 3.46],
                "source": [
                    "open_source_primary",
                    "random_blog",
                    "reference_feed",
                    "open_source_primary",
                    "random_blog",
                    "reference_feed",
                ],
            }
        )
        conf = bus._evaluate_source_confidence(
            raw_bars=raw_bars,
            macro=macro,
            news=news,
            sentiment_factor_count={"open_source_primary": 4, "random_blog": 0},
        )
        self.assertIn("open_source_primary", conf.by_source)
        self.assertIn("random_blog", conf.by_source)
        self.assertGreater(conf.by_source["open_source_primary"], conf.by_source["random_blog"])
        src_detail = {d.source: d for d in conf.details}
        self.assertGreater(src_detail["open_source_primary"].macro_consistency, src_detail["random_blog"].macro_consistency)

    def test_ingest_merges_sentiment_across_sources_with_prefixed_keys(self) -> None:
        p1 = _MiniProvider(name="src_a", sentiment={"pcr_50etf": 1.10, "btc_return_24h": -0.01})
        p2 = _MiniProvider(name="src_b", sentiment={"pcr_50etf": 0.90, "btc_return_24h": 0.03})
        bus = DataBus(
            providers=[p1, p2],  # type: ignore[list-item]
            output_dir=Path("."),
            sqlite_path=Path("tmp.db"),
            completeness_min=0.99,
            conflict_max=0.005,
        )
        out = bus.ingest(
            symbols=["TEST1"],
            start=datetime(2026, 2, 13).date(),
            end=datetime(2026, 2, 13).date(),
            start_ts=datetime(2026, 2, 13, 0, 0),
            end_ts=datetime(2026, 2, 13, 23, 59),
            langs=("zh",),
        )
        self.assertIn("src_a.pcr_50etf", out.sentiment)
        self.assertIn("src_b.pcr_50etf", out.sentiment)
        self.assertIn("pcr_50etf", out.sentiment)
        self.assertAlmostEqual(float(out.sentiment["pcr_50etf"]), 1.0, places=6)
        self.assertIn("btc_return_24h", out.sentiment)
        self.assertAlmostEqual(float(out.sentiment["btc_return_24h"]), 0.01, places=6)

    def test_ingest_writes_timing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = _MiniProvider(name="src_a", sentiment={"pcr_50etf": 1.10})
            p2 = _MiniProvider(name="src_b", sentiment={"pcr_50etf": 0.90})
            bus = DataBus(
                providers=[p1, p2],  # type: ignore[list-item]
                output_dir=root / "output",
                sqlite_path=root / "output" / "artifacts.db",
                completeness_min=0.99,
                conflict_max=0.005,
            )
            out = bus.ingest(
                symbols=["TEST1"],
                start=datetime(2026, 2, 13).date(),
                end=datetime(2026, 2, 13).date(),
                start_ts=datetime(2026, 2, 13, 0, 0),
                end_ts=datetime(2026, 2, 13, 23, 59),
                langs=("zh",),
            )
            self.assertFalse(out.normalized_bars.empty)
            timing_path = root / "output" / "review" / "2026-02-13_data_ingest_timing.json"
            self.assertTrue(timing_path.exists())
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertEqual(str(timing.get("status", "")), "completed")
            stage_names = [str(x.get("name", "")) for x in timing.get("stages", []) if isinstance(x, dict)]
            self.assertIn("collect_bars", stage_names)
            self.assertIn("collect_enrichments", stage_names)
            self.assertIn("evaluate_quality", stage_names)
            collect_stage = next(
                x for x in timing.get("stages", []) if isinstance(x, dict) and str(x.get("name", "")) == "collect_bars"
            )
            providers = collect_stage.get("summary", {}).get("providers", [])
            self.assertEqual(len(providers), 2)
            first_provider = providers[0]
            self.assertIn("symbols", first_provider)
            self.assertIn("slowest_symbols", first_provider)
            symbol_rows = first_provider.get("symbols", [])
            self.assertEqual(len(symbol_rows), 1)
            self.assertEqual(str(symbol_rows[0].get("symbol", "")), "TEST1")
            self.assertEqual(str(symbol_rows[0].get("outcome", "")), "rows")

    def test_persist_writes_timing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = _MiniProvider(name="src_a", sentiment={"pcr_50etf": 1.10})
            bus = DataBus(
                providers=[p1],  # type: ignore[list-item]
                output_dir=root / "output",
                sqlite_path=root / "output" / "artifacts.db",
                completeness_min=0.99,
                conflict_max=0.005,
            )
            out = bus.ingest(
                symbols=["TEST1"],
                start=datetime(2026, 2, 13).date(),
                end=datetime(2026, 2, 13).date(),
                start_ts=datetime(2026, 2, 13, 0, 0),
                end_ts=datetime(2026, 2, 13, 23, 59),
                langs=("zh",),
            )
            bus.persist(datetime(2026, 2, 13).date(), out)
            timing_path = root / "output" / "review" / "2026-02-13_data_persist_timing.json"
            self.assertTrue(timing_path.exists())
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertEqual(str(timing.get("status", "")), "completed")
            stage_names = [str(x.get("name", "")) for x in timing.get("stages", []) if isinstance(x, dict)]
            self.assertIn("write_artifacts", stage_names)
            self.assertIn("append_sqlite_core", stage_names)
            self.assertIn("append_sqlite_quality", stage_names)


if __name__ == "__main__":
    unittest.main()
