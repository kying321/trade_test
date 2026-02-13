from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()

