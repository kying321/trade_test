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
from lie_engine.research.strategy_lab import run_strategy_lab
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
            self.assertFalse(bundle.review_bars.empty)
            self.assertTrue(all(pd.to_datetime(bundle.review_bars["ts"]).dt.date > cutoff))
            self.assertTrue(all(d <= cutoff for d in bundle.news_daily.index.tolist()))
            self.assertTrue(all(d <= cutoff for d in bundle.report_daily.index.tolist()))
            self.assertTrue(all(cutoff < d <= date(2026, 2, 15) for d in bundle.review_news_daily.index.tolist()))
            self.assertTrue(all(cutoff < d <= date(2026, 2, 15) for d in bundle.review_report_daily.index.tolist()))
            self.assertEqual(bundle.review_news_records, 2)
            self.assertEqual(bundle.review_report_records, 2)
            self.assertTrue(bool(bundle.fetch_stats.get("strict_cutoff_enforced", False)))
            self.assertTrue(str(bundle.cutoff_ts).startswith("2026-02-13T"))
            self.assertIn("2026-02-13", str(bundle.bar_max_ts))
            self.assertIn("2026-02-13", str(bundle.news_max_ts))
            self.assertIn("2026-02-13", str(bundle.report_max_ts))
            self.assertIn("2026-02-15", str(bundle.review_bar_max_ts))
            self.assertIn("2026-02-15", str(bundle.review_news_max_ts))
            self.assertIn("2026-02-15", str(bundle.review_report_max_ts))
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]
            rd.fetch_symbol_news_and_reports = original_fetch_nr  # type: ignore[assignment]

    def test_run_strategy_lab_with_mock_bundle(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        bars = make_multi_symbol_bars()
        bars["asset_class"] = bars["asset_class"].astype(str)
        bars["source"] = "mock"
        idx = sorted(pd.to_datetime(bars["ts"]).dt.date.unique())
        news_daily = pd.Series([0.12] * len(idx), index=idx, dtype=float)
        report_daily = pd.Series([0.08] * len(idx), index=idx, dtype=float)
        review_bars = bars.groupby("symbol", as_index=False).tail(8).copy()
        review_bars["ts"] = pd.to_datetime(review_bars["ts"]) + pd.Timedelta(days=12)
        review_idx = sorted(pd.to_datetime(review_bars["ts"]).dt.date.unique())
        review_news_daily = pd.Series([0.10] * len(review_idx), index=review_idx, dtype=float)
        review_report_daily = pd.Series([0.06] * len(review_idx), index=review_idx, dtype=float)
        bundle = RealDataBundle(
            bars=bars,
            review_bars=review_bars,
            universe=sorted(set(bars["symbol"])),
            news_daily=news_daily,
            report_daily=report_daily,
            news_records=40,
            report_records=60,
            fetch_stats={"mocked": True, "strict_cutoff_enforced": True},
            cutoff_date=date(2025, 12, 31),
            review_days=3,
            review_news_daily=review_news_daily,
            review_report_daily=review_report_daily,
            review_news_records=int(len(review_idx)),
            review_report_records=int(len(review_idx)),
        )

        original_loader = sl_mod.load_real_data_bundle
        sl_mod.load_real_data_bundle = lambda **kwargs: bundle  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                summary = run_strategy_lab(
                    output_root=Path(td),
                    core_symbols=["300750", "002050", "513130", "LC2603"],
                    start=date(2025, 1, 1),
                    end=date(2025, 12, 31),
                    max_symbols=10,
                    report_symbol_cap=5,
                    workers=2,
                    review_days=3,
                    candidate_count=4,
                )
                self.assertGreaterEqual(len(summary.candidates), 1)
                self.assertTrue(summary.best_candidate)
                self.assertGreater(summary.review_bars_rows, 0)
                self.assertTrue(Path(summary.output_dir).exists())
                self.assertTrue((Path(summary.output_dir) / "summary.json").exists())
                self.assertTrue((Path(summary.output_dir) / "report.md").exists())
                self.assertTrue((Path(summary.output_dir) / "best_strategy.yaml").exists())
                self.assertIn("review_metrics", summary.candidates[0].to_dict())
                self.assertEqual(summary.cutoff_ts, "2025-12-31T23:59:59")
                self.assertEqual(summary.bar_max_ts, "")
                self.assertEqual(summary.news_max_ts, "")
                self.assertEqual(summary.report_max_ts, "")
        finally:
            sl_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
