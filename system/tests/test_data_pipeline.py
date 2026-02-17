from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.pipeline import DataBus
from lie_engine.models import NewsEvent


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


if __name__ == "__main__":
    unittest.main()
