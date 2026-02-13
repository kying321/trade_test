from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.research.optimizer import run_research_backtest
from lie_engine.research.real_data import RealDataBundle
from tests.helpers import make_multi_symbol_bars


class ResearchTests(unittest.TestCase):
    def test_run_research_backtest_with_mock_bundle(self) -> None:
        import lie_engine.research.optimizer as opt_mod

        bars = make_multi_symbol_bars()
        bars["asset_class"] = bars["asset_class"].astype(str)
        bars["source"] = "mock"
        idx = sorted(pd.to_datetime(bars["ts"]).dt.date.unique())
        news_daily = pd.Series([0.1] * len(idx), index=idx, dtype=float)
        report_daily = pd.Series([0.05] * len(idx), index=idx, dtype=float)
        bundle = RealDataBundle(
            bars=bars,
            universe=sorted(set(bars["symbol"])),
            news_daily=news_daily,
            report_daily=report_daily,
            news_records=50,
            report_records=80,
            fetch_stats={"mocked": True},
        )

        original_loader = opt_mod.load_real_data_bundle
        opt_mod.load_real_data_bundle = lambda **kwargs: bundle  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                summary = run_research_backtest(
                    output_root=Path(td),
                    core_symbols=["300750", "002050", "513130", "LC2603"],
                    start=date(2025, 1, 1),
                    end=date(2025, 12, 31),
                    hours_budget=0.01,
                    max_symbols=10,
                    report_symbol_cap=5,
                    workers=2,
                    max_trials_per_mode=2,
                    seed=7,
                )
                self.assertGreaterEqual(len(summary.mode_summaries), 1)
                self.assertTrue(Path(summary.output_dir).exists())
                self.assertTrue((Path(summary.output_dir) / "summary.json").exists())
                self.assertTrue((Path(summary.output_dir) / "report.md").exists())
        finally:
            opt_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]

    def test_load_real_data_bundle_enforces_cutoff_and_review_split(self) -> None:
        import lie_engine.research.real_data as rd

        cutoff = date(2026, 2, 13)
        original_load_universe = rd.load_universe
        original_fetch_one = rd._fetch_one_symbol
        original_fetch_nr = rd.fetch_symbol_news_and_reports

        def _fake_load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
            return ["000001"]

        def _fake_fetch_one(symbol: str, start: date, end: date):
            ts = pd.date_range(start=start, end=end, freq="D")
            df = pd.DataFrame(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.0,
                    "volume": 1000.0,
                    "source": "mock",
                    "asset_class": "equity",
                }
            )
            return symbol, df, None

        def _fake_fetch_nr(symbol: str, start: date, end: date):
            news = pd.DataFrame(
                [
                    {"date": date(2026, 2, 12), "symbol": symbol, "news_score": 0.1},
                    {"date": date(2026, 2, 13), "symbol": symbol, "news_score": 0.2},
                    {"date": date(2026, 2, 14), "symbol": symbol, "news_score": -0.1},
                    {"date": date(2026, 2, 15), "symbol": symbol, "news_score": 0.3},
                ]
            )
            report = pd.DataFrame(
                [
                    {"date": date(2026, 2, 13), "symbol": symbol, "report_score": 0.5},
                    {"date": date(2026, 2, 14), "symbol": symbol, "report_score": -0.2},
                    {"date": date(2026, 2, 15), "symbol": symbol, "report_score": 0.4},
                ]
            )
            return news, report

        rd.load_universe = _fake_load_universe  # type: ignore[assignment]
        rd._fetch_one_symbol = _fake_fetch_one  # type: ignore[assignment]
        rd.fetch_symbol_news_and_reports = _fake_fetch_nr  # type: ignore[assignment]
        try:
            bundle = rd.load_real_data_bundle(
                core_symbols=["000001"],
                start=date(2026, 2, 10),
                end=cutoff,
                max_symbols=1,
                report_symbol_cap=1,
                workers=2,
                cache_dir=None,
                strict_cutoff=cutoff,
                review_days=2,
                include_post_review=True,
            )
            max_bar_date = pd.to_datetime(bundle.bars["ts"]).dt.date.max()
            self.assertLessEqual(max_bar_date, cutoff)
            self.assertTrue(all(d <= cutoff for d in bundle.news_daily.index.tolist()))
            self.assertTrue(all(d <= cutoff for d in bundle.report_daily.index.tolist()))
            self.assertTrue(all(cutoff < d <= date(2026, 2, 15) for d in bundle.review_news_daily.index.tolist()))
            self.assertTrue(all(cutoff < d <= date(2026, 2, 15) for d in bundle.review_report_daily.index.tolist()))
            self.assertEqual(bundle.review_news_records, 2)
            self.assertEqual(bundle.review_report_records, 2)
            self.assertTrue(bool(bundle.fetch_stats.get("strict_cutoff_enforced", False)))
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]
            rd.fetch_symbol_news_and_reports = original_fetch_nr  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
