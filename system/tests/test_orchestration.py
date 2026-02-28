from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.models import NewsEvent, RegimeLabel
from lie_engine.orchestration import build_guard_assessment, estimate_factor_contrib_120d
from tests.helpers import make_multi_symbol_bars


@dataclass
class _FakeProvider:
    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]:
        return {
            "pcr_50etf": 1.1,
            "iv_50etf": 0.29,
            "northbound_netflow": -6e8,
            "margin_balance_chg": -0.01,
        }


@dataclass
class _FakeIngest:
    macro: pd.DataFrame
    news: list[NewsEvent]


class OrchestrationTests(unittest.TestCase):
    def test_build_guard_assessment_blocks_on_major_event(self) -> None:
        news = [
            NewsEvent(
                event_id="n1",
                ts=datetime(2026, 2, 13, 9, 0),
                title="FOMC emergency statement",
                content="rate path revised",
                lang="en",
                source="official_feed",
                category="政策",
                confidence=0.95,
                entities=["Fed"],
                importance=0.95,
            )
        ]
        trades = pd.DataFrame({"pnl": [-1.0, -0.8, -0.2]})
        out = build_guard_assessment(
            as_of=date(2026, 2, 13),
            regime=RegimeLabel.STRONG_TREND,
            atr_z=0.3,
            quality_passed=True,
            sentiment={},
            news=news,
            recent_trades=trades,
            lookback_hours=24,
            cooldown_losses=3,
            black_swan_threshold=70.0,
        )
        self.assertTrue(out.trade_blocked)
        self.assertTrue(out.major_event_window)

    def test_estimate_factor_contrib_outputs_required_keys(self) -> None:
        bars = make_multi_symbol_bars()
        ingest = _FakeIngest(
            macro=pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-03-01", "2025-06-01", "2025-09-01"]),
                    "cpi_yoy": [0.5, 0.6, 0.7],
                    "ppi_yoy": [-1.0, -0.8, -0.5],
                    "lpr_1y": [3.45, 3.4, 3.35],
                }
            ),
            news=[
                NewsEvent(
                    event_id="x",
                    ts=datetime(2025, 7, 2, 10, 0),
                    title="policy",
                    content="support",
                    lang="en",
                    source="official",
                    category="政策",
                    confidence=0.8,
                    entities=["A股"],
                    importance=0.7,
                )
            ],
        )
        contrib = estimate_factor_contrib_120d(
            bars=bars,
            ingest=ingest,
            providers=[_FakeProvider()],
            lookback_days=120,
        )
        for key in ["macro", "industry", "news", "sentiment", "fundamental", "technical"]:
            self.assertIn(key, contrib)
            self.assertGreaterEqual(float(contrib[key]), 0.0)


if __name__ == "__main__":
    unittest.main()

