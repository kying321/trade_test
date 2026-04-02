from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.pipeline import DataBus, IngestionResult
from lie_engine.data.quality import DataQualityReport, SourceConfidenceItem, SourceConfidenceReport
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
    def test_persist_is_idempotent_for_source_owned_tables(self) -> None:
        from tempfile import TemporaryDirectory
        import sqlite3

        with TemporaryDirectory() as td:
            root = Path(td)
            bus = DataBus(
                providers=[],
                output_dir=root,
                sqlite_path=root / "artifacts" / "lie_engine.db",
                completeness_min=0.99,
                conflict_max=0.005,
            )
            result = IngestionResult(
                raw_bars=pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2026-03-30"]),
                        "symbol": ["BTCUSDT"],
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.0],
                        "volume": [1000.0],
                        "source": ["binance_spot_public"],
                        "asset_class": ["crypto"],
                    }
                ),
                normalized_bars=pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2026-03-30"]),
                        "symbol": ["BTCUSDT"],
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.0],
                        "volume": [1000.0],
                        "asset_class": ["crypto"],
                        "source_count": [2],
                        "data_conflict_flag": [0],
                    }
                ),
                conflicts=pd.DataFrame(columns=["ts", "symbol", "field", "values", "max_diff_pct"]),
                macro=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-03-20"]),
                        "cpi_yoy": [0.1],
                        "ppi_yoy": [-2.0],
                        "lpr_1y": [3.0],
                        "source": ["public_macro_news"],
                    }
                ),
                sentiment={"btc_return_24h": 0.01},
                news=[
                    NewsEvent(
                        event_id="evt-1",
                        ts=datetime(2026, 3, 30, 10, 0),
                        title="test title",
                        content="test content",
                        lang="zh",
                        source="public_macro_news",
                        category="宏观",
                        confidence=0.8,
                        entities=["BTCUSDT"],
                        importance=0.7,
                    )
                ],
                source_confidence=SourceConfidenceReport(
                    overall_score=0.8,
                    by_source={"public_macro_news": 0.8},
                    low_confidence_sources=[],
                    details=[
                        SourceConfidenceItem(
                            source="public_macro_news",
                            score=0.8,
                            base_reliability=0.8,
                            bar_consistency=1.0,
                            bar_coverage=1.0,
                            news_confidence=0.8,
                            sentiment_coverage=1.0,
                            bars_rows=1,
                            news_events=1,
                            sentiment_factors=1,
                            macro_consistency=1.0,
                            macro_coverage=1.0,
                            macro_rows=1,
                        )
                    ],
                ),
                quality=DataQualityReport(
                    completeness=1.0,
                    unresolved_conflict_ratio=0.0,
                    source_confidence_score=0.8,
                    low_confidence_source_ratio=0.0,
                    source_confidence={"public_macro_news": 0.8},
                    flags=[],
                ),
            )

            bus.persist(as_of=datetime(2026, 3, 30).date(), result=result)
            bus.persist(as_of=datetime(2026, 3, 30).date(), result=result)

            conn = sqlite3.connect(root / "artifacts" / "lie_engine.db")
            try:
                counts = {
                    table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in [
                        "bars_normalized",
                        "macro",
                        "news_events",
                        "sentiment_snapshot",
                        "source_confidence",
                        "quality",
                    ]
                }
            finally:
                conn.close()
            self.assertEqual(
                counts,
                {
                    "bars_normalized": 1,
                    "macro": 1,
                    "news_events": 1,
                    "sentiment_snapshot": 1,
                    "source_confidence": 1,
                    "quality": 1,
                },
            )

    def test_persist_feature_parquet_includes_latest_macro_snapshot(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            root = Path(td)
            bus = DataBus(
                providers=[],
                output_dir=root,
                sqlite_path=root / "artifacts" / "lie_engine.db",
                completeness_min=0.99,
                conflict_max=0.005,
            )
            result = IngestionResult(
                raw_bars=pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2026-03-30"]),
                        "symbol": ["BU2606"],
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.0],
                        "volume": [1000.0],
                        "source": ["public_macro_news"],
                        "asset_class": ["future"],
                    }
                ),
                normalized_bars=pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2026-03-30"]),
                        "symbol": ["BU2606"],
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.0],
                        "volume": [1000.0],
                        "asset_class": ["future"],
                        "source_count": [1],
                        "data_conflict_flag": [0],
                    }
                ),
                conflicts=pd.DataFrame(columns=["ts", "symbol", "field", "values", "max_diff_pct"]),
                macro=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-03-20"]),
                        "cpi_yoy": [0.1],
                        "ppi_yoy": [-2.0],
                        "lpr_1y": [3.0],
                        "fixed_asset_investment_cum": [52721.0],
                        "consumer_confidence_index": [90.6],
                        "source": ["public_macro_news"],
                        "fixed_asset_investment_source": ["nbs:fixed_asset_investment"],
                    }
                ),
                sentiment={},
                news=[],
                source_confidence=SourceConfidenceReport(overall_score=1.0, by_source={}, low_confidence_sources=[], details=[]),
                quality=DataQualityReport(
                    completeness=1.0,
                    unresolved_conflict_ratio=0.0,
                    source_confidence_score=1.0,
                    low_confidence_source_ratio=0.0,
                    source_confidence={},
                    flags=[],
                ),
            )

            bus.persist(as_of=datetime(2026, 3, 30).date(), result=result)
            feature_path = root / "artifacts" / "feature" / "2026-03-30_bars_feature.parquet"
            feature_df = pd.read_parquet(feature_path)

            self.assertIn("cpi_yoy", feature_df.columns)
            self.assertIn("fixed_asset_investment_cum", feature_df.columns)
            self.assertIn("consumer_confidence_index", feature_df.columns)
            self.assertNotIn("source", feature_df.columns)
            self.assertNotIn("fixed_asset_investment_source", feature_df.columns)
            self.assertAlmostEqual(float(feature_df.iloc[0]["cpi_yoy"]), 0.1, places=6)
            self.assertAlmostEqual(float(feature_df.iloc[0]["fixed_asset_investment_cum"]), 52721.0, places=6)
            self.assertAlmostEqual(float(feature_df.iloc[0]["consumer_confidence_index"]), 90.6, places=6)

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


if __name__ == "__main__":
    unittest.main()
