from __future__ import annotations

from datetime import date, datetime
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.research.optimizer import run_research_backtest
from lie_engine.research.real_data import RealDataBundle
from lie_engine.research.strategy_lab import run_strategy_lab
from lie_engine.models import BacktestResult
from tests.helpers import make_multi_symbol_bars


def _load_theory_ablation_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_theory_ablation.py"
    spec = importlib.util.spec_from_file_location("run_theory_ablation_for_tests", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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
                first_mode = summary.mode_summaries[0].mode
                self.assertTrue((Path(summary.output_dir) / first_mode / "best_trade_journal.csv").exists())
                self.assertTrue((Path(summary.output_dir) / first_mode / "best_holding_daily_symbol_exposure.csv").exists())
                self.assertTrue(any((Path(summary.output_dir) / first_mode).glob("trial_*_trade_journal.csv")))
                self.assertTrue(any((Path(summary.output_dir) / first_mode).glob("trial_*_holding_daily_symbol_exposure.csv")))
                self.assertTrue(bool(summary.mode_summaries[0].to_dict().get("trial_artifacts", [])))
        finally:
            opt_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]

    def test_run_research_backtest_passes_symbol_level_factor_series_to_optimizer(self) -> None:
        import lie_engine.research.optimizer as opt_mod

        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "symbol": ["BU2606", "BU2606"],
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.0, 1.1],
                "volume": [1000.0, 1200.0],
                "source": ["mock", "mock"],
                "asset_class": ["future", "future"],
            }
        )
        idx = [date(2025, 1, 2), date(2025, 1, 3)]
        bundle = RealDataBundle(
            bars=bars,
            universe=["BU2606"],
            news_daily=pd.Series([0.0, 0.0], index=idx, dtype=float),
            report_daily=pd.Series([0.0, 0.0], index=idx, dtype=float),
            news_daily_by_symbol=pd.DataFrame(
                {
                    "date": idx,
                    "symbol": ["BU2606", "BU2606"],
                    "news_score": [0.6, -0.2],
                }
            ),
            report_daily_by_symbol=pd.DataFrame(
                {
                    "date": idx,
                    "symbol": ["BU2606", "BU2606"],
                    "report_score": [0.1, 0.3],
                }
            ),
            news_records=2,
            report_records=2,
            fetch_stats={"mocked": True},
            cutoff_date=date(2025, 1, 3),
        )

        captured: dict[str, Any] = {}

        original_loader = opt_mod.load_real_data_bundle
        original_optimize = opt_mod._optimize_one_mode

        def _fake_optimize_one_mode(**kwargs):  # type: ignore[no-untyped-def]
            captured["news_daily"] = kwargs["news_daily"]
            captured["report_daily"] = kwargs["report_daily"]
            return opt_mod.ModeOptimizationSummary(
                mode=kwargs["mode"],
                trials=0,
                best_score=0.0,
                best_params={},
                best_metrics={},
                budget_seconds=0.0,
                elapsed_seconds=0.0,
                trial_log_path="",
            )

        opt_mod.load_real_data_bundle = lambda **kwargs: bundle  # type: ignore[assignment]
        opt_mod._optimize_one_mode = _fake_optimize_one_mode  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                summary = run_research_backtest(
                    output_root=Path(td),
                    core_symbols=["BU2606"],
                    start=date(2025, 1, 1),
                    end=date(2025, 1, 3),
                    hours_budget=0.001,
                    max_symbols=1,
                    report_symbol_cap=1,
                    workers=1,
                    max_trials_per_mode=1,
                    seed=7,
                    modes=["ultra_short"],
                )
            self.assertTrue(summary.mode_summaries is not None)
            self.assertAlmostEqual(float(captured["news_daily"].iloc[0]), 0.6, places=6)
            self.assertAlmostEqual(float(captured["news_daily"].iloc[1]), -0.2, places=6)
            self.assertAlmostEqual(float(captured["report_daily"].iloc[0]), 0.1, places=6)
            self.assertAlmostEqual(float(captured["report_daily"].iloc[1]), 0.3, places=6)
        finally:
            opt_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]
            opt_mod._optimize_one_mode = original_optimize  # type: ignore[assignment]

    def test_optimize_one_mode_passes_backtest_daily_symbol_exposure_into_factor_alignment(self) -> None:
        import time
        import numpy as np
        import lie_engine.research.optimizer as opt_mod
        from lie_engine.models import BacktestResult

        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "symbol": ["BU2606", "BU2606"],
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.0, 1.1],
                "volume": [1000.0, 1100.0],
                "source": ["mock", "mock"],
                "asset_class": ["future", "future"],
            }
        )
        idx = [date(2025, 1, 2), date(2025, 1, 3)]
        captured: dict[str, Any] = {}

        original_run_backtest = opt_mod.run_event_backtest
        original_factor_alignment = opt_mod._factor_alignment

        def _fake_run_event_backtest(*args, **kwargs):  # type: ignore[no-untyped-def]
            _ = (args, kwargs)
            return BacktestResult(
                start=date(2025, 1, 1),
                end=date(2025, 1, 3),
                total_return=0.01,
                annual_return=0.10,
                max_drawdown=0.02,
                win_rate=0.6,
                profit_factor=1.2,
                expectancy=0.01,
                trades=1,
                violations=0,
                positive_window_ratio=1.0,
                equity_curve=[{"date": "2025-01-02", "equity": 1.0}, {"date": "2025-01-03", "equity": 1.02}],
                by_asset={},
                daily_symbol_exposure=[{"date": "2025-01-02", "symbol": "BU2606", "weight": 1.0}],
            )

        def _fake_factor_alignment(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured["daily_symbol_exposure"] = kwargs.get("daily_symbol_exposure")
            captured["news_daily_by_symbol"] = kwargs.get("news_daily_by_symbol")
            captured["report_daily_by_symbol"] = kwargs.get("report_daily_by_symbol")
            return 0.0

        opt_mod.run_event_backtest = _fake_run_event_backtest  # type: ignore[assignment]
        opt_mod._factor_alignment = _fake_factor_alignment  # type: ignore[assignment]
        try:
            summary = opt_mod._optimize_one_mode(
                mode="ultra_short",
                space=opt_mod.MODE_SPACES["ultra_short"],
                bars=bars,
                start=date(2025, 1, 1),
                end=date(2025, 1, 3),
                news_daily=pd.Series([0.0, 0.0], index=idx, dtype=float),
                report_daily=pd.Series([0.0, 0.0], index=idx, dtype=float),
                news_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606", "BU2606"], "news_score": [0.6, -0.2]}),
                report_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606", "BU2606"], "report_score": [0.1, 0.3]}),
                deadline_ts=time.time() + 1.0,
                max_trials=1,
                rng=np.random.default_rng(7),
                out_dir=Path(tempfile.mkdtemp()),
            )
            self.assertIsNotNone(summary)
            self.assertEqual(captured["daily_symbol_exposure"][0]["symbol"], "BU2606")
            self.assertFalse(captured["news_daily_by_symbol"].empty)
            self.assertFalse(captured["report_daily_by_symbol"].empty)
        finally:
            opt_mod.run_event_backtest = original_run_backtest  # type: ignore[assignment]
            opt_mod._factor_alignment = original_factor_alignment  # type: ignore[assignment]

    def test_optimize_one_mode_prefers_holding_daily_symbol_exposure(self) -> None:
        import time
        import numpy as np
        import lie_engine.research.optimizer as opt_mod
        from lie_engine.models import BacktestResult

        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "symbol": ["BU2606", "BU2606"],
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.0, 1.1],
                "volume": [1000.0, 1100.0],
                "source": ["mock", "mock"],
                "asset_class": ["future", "future"],
            }
        )
        idx = [date(2025, 1, 2), date(2025, 1, 3)]
        captured: dict[str, Any] = {}

        original_run_backtest = opt_mod.run_event_backtest
        original_factor_alignment = opt_mod._factor_alignment

        def _fake_run_event_backtest(*args, **kwargs):  # type: ignore[no-untyped-def]
            _ = (args, kwargs)
            return BacktestResult(
                start=date(2025, 1, 1),
                end=date(2025, 1, 3),
                total_return=0.01,
                annual_return=0.10,
                max_drawdown=0.02,
                win_rate=0.6,
                profit_factor=1.2,
                expectancy=0.01,
                trades=1,
                violations=0,
                positive_window_ratio=1.0,
                equity_curve=[{"date": "2025-01-02", "equity": 1.0}, {"date": "2025-01-03", "equity": 1.02}],
                by_asset={},
                daily_symbol_exposure=[{"date": "2025-01-02", "symbol": "WRONG", "weight": 1.0}],
                holding_daily_symbol_exposure=[{"date": "2025-01-02", "symbol": "BU2606", "weight": 1.0}],
            )

        def _fake_factor_alignment(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured["daily_symbol_exposure"] = kwargs.get("daily_symbol_exposure")
            return 0.0

        opt_mod.run_event_backtest = _fake_run_event_backtest  # type: ignore[assignment]
        opt_mod._factor_alignment = _fake_factor_alignment  # type: ignore[assignment]
        try:
            opt_mod._optimize_one_mode(
                mode="ultra_short",
                space=opt_mod.MODE_SPACES["ultra_short"],
                bars=bars,
                start=date(2025, 1, 1),
                end=date(2025, 1, 3),
                news_daily=pd.Series([0.0, 0.0], index=idx, dtype=float),
                report_daily=pd.Series([0.0, 0.0], index=idx, dtype=float),
                news_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606", "BU2606"], "news_score": [0.6, -0.2]}),
                report_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606", "BU2606"], "report_score": [0.1, 0.3]}),
                deadline_ts=time.time() + 1.0,
                max_trials=1,
                rng=np.random.default_rng(7),
                out_dir=Path(tempfile.mkdtemp()),
            )
            self.assertEqual(captured["daily_symbol_exposure"][0]["symbol"], "BU2606")
        finally:
            opt_mod.run_event_backtest = original_run_backtest  # type: ignore[assignment]
            opt_mod._factor_alignment = original_factor_alignment  # type: ignore[assignment]

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

    def test_load_universe_crypto_mode_keeps_core_symbols_only(self) -> None:
        import lie_engine.research.real_data as rd

        original_cons = rd.ak.index_stock_cons_csindex

        def _unexpected_cons_call(symbol: str):
            raise AssertionError(f"index constituent fetch should not run in crypto mode: {symbol}")

        rd.ak.index_stock_cons_csindex = _unexpected_cons_call  # type: ignore[assignment]
        try:
            out = rd.load_universe(core_symbols=["btcusdt", "ETHUSDT", "btc-usdt"], max_symbols=10)
            self.assertEqual(out, ["BTCUSDT", "ETHUSDT"])
        finally:
            rd.ak.index_stock_cons_csindex = original_cons  # type: ignore[assignment]

    def test_fetch_one_symbol_supports_crypto_pair(self) -> None:
        import lie_engine.research.real_data as rd

        original_get_provider = rd._get_binance_provider

        class _FakeProvider:
            def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2026-02-10", "2026-02-11"]),
                        "symbol": [symbol, symbol],
                        "open": [100.0, 101.0],
                        "high": [102.0, 103.0],
                        "low": [99.0, 100.0],
                        "close": [101.0, 102.0],
                        "volume": [1000.0, 1200.0],
                        "source": ["mock_binance", "mock_binance"],
                        "asset_class": ["equity", "equity"],
                    }
                )

        rd._get_binance_provider = lambda: _FakeProvider()  # type: ignore[assignment]
        try:
            symbol, df, err = rd._fetch_one_symbol("btc-usdt", date(2026, 2, 10), date(2026, 2, 11))
            self.assertEqual(symbol, "BTCUSDT")
            self.assertIsNone(err)
            self.assertFalse(df.empty)
            self.assertTrue((df["symbol"] == "BTCUSDT").all())
            self.assertTrue((df["asset_class"] == "crypto").all())
            self.assertIn("open", df.columns)
            self.assertIn("source", df.columns)
        finally:
            rd._get_binance_provider = original_get_provider  # type: ignore[assignment]

    def test_fetch_crypto_daily_falls_back_to_bybit_when_binance_empty(self) -> None:
        import lie_engine.research.real_data as rd

        original_get_binance = rd._get_binance_provider
        original_get_bybit = getattr(rd, "_get_bybit_provider", None)

        class _EmptyProvider:
            def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
                _ = (symbol, start, end, freq)
                return pd.DataFrame(
                    columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"]
                )

        class _FallbackProvider:
            def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
                _ = (start, end, freq)
                return pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2026-03-29", "2026-03-30"]),
                        "symbol": [symbol, symbol],
                        "open": [100.0, 101.0],
                        "high": [101.0, 102.0],
                        "low": [99.0, 100.0],
                        "close": [100.5, 101.5],
                        "volume": [1000.0, 1200.0],
                        "source": ["bybit_spot_public", "bybit_spot_public"],
                        "asset_class": ["crypto", "crypto"],
                    }
                )

        rd._get_binance_provider = lambda: _EmptyProvider()  # type: ignore[assignment]
        rd._get_bybit_provider = lambda: _FallbackProvider()  # type: ignore[attr-defined,assignment]
        try:
            out = rd.fetch_crypto_daily("BTCUSDT", date(2026, 3, 29), date(2026, 3, 30))
            self.assertFalse(out.empty)
            self.assertEqual(set(out["symbol"]), {"BTCUSDT"})
            self.assertEqual(set(out["asset_class"]), {"crypto"})
            self.assertEqual(set(out["source"]), {"bybit_spot_public"})
        finally:
            rd._get_binance_provider = original_get_binance  # type: ignore[assignment]
            if original_get_bybit is not None:
                rd._get_bybit_provider = original_get_bybit  # type: ignore[attr-defined,assignment]

    def test_public_news_helpers_support_crypto_and_future_symbols(self) -> None:
        import lie_engine.research.real_data as rd
        from lie_engine.models import NewsEvent
        future_event = NewsEvent(
            event_id="n1",
            ts=datetime(2026, 3, 30, 9, 0),
            title="[SHMET] 沥青库存去化",
            content="BU2606 沥青库存下降，成本支撑增强，短线利好。",
            lang="zh",
            source="public_macro_news",
            category="产业链",
            confidence=0.8,
            entities=["BU2606"],
            importance=0.7,
        )
        crypto_event = NewsEvent(
            event_id="n2",
            ts=datetime(2026, 3, 30, 10, 0),
            title="[Baidu] 比特币风险偏好回升",
            content="BTCUSDT 资金回流，突破前高，情绪改善。",
            lang="zh",
            source="public_macro_news",
            category="宏观",
            confidence=0.78,
            entities=["BTCUSDT"],
            importance=0.68,
        )
        self.assertTrue(rd._public_news_matches_symbol(future_event, "BU2606"))
        self.assertTrue(rd._public_news_matches_symbol(crypto_event, "BTCUSDT"))
        self.assertGreater(rd._public_news_score(future_event), 0.0)
        self.assertGreater(rd._public_news_score(crypto_event), 0.0)

    def test_public_news_helpers_allow_macro_geopolitical_fallback_for_crypto(self) -> None:
        import lie_engine.research.real_data as rd
        from lie_engine.models import NewsEvent

        event = NewsEvent(
            event_id="n3",
            ts=datetime(2026, 3, 30, 3, 0),
            title="[SHMET] 伊朗首都德黑兰发生爆炸",
            content="地缘冲突升级，风险偏好承压，避险情绪上升。",
            lang="zh",
            source="public_macro_news",
            category="产业链",
            confidence=0.74,
            entities=[],
            importance=0.66,
        )
        self.assertTrue(rd._public_news_matches_symbol(event, "BTCUSDT"))
        self.assertLess(rd._public_news_score(event, relevance=0.45), 0.0)

    def test_public_news_helpers_parse_warehouse_delta_direction_for_commodities(self) -> None:
        import lie_engine.research.real_data as rd
        from lie_engine.models import NewsEvent

        decrease_event = NewsEvent(
            event_id="n4",
            ts=datetime(2026, 3, 30, 15, 10),
            title="[Baidu] 中国3月30日上期所每日仓单变动-铜(吨)",
            content="中国 | 中国3月30日上期所每日仓单变动-铜(吨) | 预期=nan | 前值=-9365",
            lang="zh",
            source="public_macro_news",
            category="宏观",
            confidence=0.82,
            entities=[],
            importance=0.75,
        )
        increase_event = NewsEvent(
            event_id="n5",
            ts=datetime(2026, 3, 30, 15, 10),
            title="[Baidu] 中国3月30日上期所每日仓单变动-白银(千克)",
            content="中国 | 中国3月30日上期所每日仓单变动-白银(千克) | 预期=nan | 前值=1500",
            lang="zh",
            source="public_macro_news",
            category="宏观",
            confidence=0.82,
            entities=[],
            importance=0.60,
        )

        self.assertGreater(rd._public_news_score(decrease_event, relevance=0.75), 0.0)
        self.assertLess(rd._public_news_score(increase_event, relevance=0.75), 0.0)

    def test_public_news_helpers_do_not_match_unrelated_warehouse_events_for_bu(self) -> None:
        import lie_engine.research.real_data as rd
        from lie_engine.models import NewsEvent

        event = NewsEvent(
            event_id="n6",
            ts=datetime(2026, 3, 30, 15, 10),
            title="[Baidu] 中国3月30日上期所每日仓单变动-铜(吨)",
            content="中国 | 中国3月30日上期所每日仓单变动-铜(吨) | 预期=nan | 前值=-9365",
            lang="zh",
            source="public_macro_news",
            category="宏观",
            confidence=0.82,
            entities=[],
            importance=0.75,
        )

        self.assertFalse(rd._public_news_matches_symbol(event, "BU2606"))
        self.assertTrue(rd._public_news_matches_symbol(event, "CU2606"))

    def test_load_real_data_bundle_collects_public_news_for_crypto_and_future_universe(self) -> None:
        import lie_engine.research.real_data as rd

        original_load_universe = rd.load_universe
        original_fetch_one = rd._fetch_one_symbol
        original_fetch_nr = rd.fetch_symbol_news_and_reports

        def _fake_load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
            _ = max_symbols
            return ["BTCUSDT", "BU2606"]

        def _fake_fetch_one(symbol: str, start: date, end: date):
            _ = (start, end)
            df = pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-03-29", "2026-03-30"]),
                    "symbol": [symbol, symbol],
                    "open": [1.0, 1.1],
                    "high": [1.1, 1.2],
                    "low": [0.9, 1.0],
                    "close": [1.0, 1.1],
                    "volume": [1000.0, 1200.0],
                    "source": ["mock", "mock"],
                    "asset_class": ["crypto" if symbol == "BTCUSDT" else "future"] * 2,
                }
            )
            return symbol, df, None

        def _fake_fetch_nr(symbol: str, start: date, end: date):
            _ = (start, end)
            return (
                pd.DataFrame(
                    [
                        {
                            "date": date(2026, 3, 30),
                            "symbol": symbol,
                            "news_score": 0.4 if symbol == "BTCUSDT" else 0.2,
                        }
                    ]
                ),
                pd.DataFrame(columns=["date", "symbol", "report_score"]),
            )

        rd.load_universe = _fake_load_universe  # type: ignore[assignment]
        rd._fetch_one_symbol = _fake_fetch_one  # type: ignore[assignment]
        rd.fetch_symbol_news_and_reports = _fake_fetch_nr  # type: ignore[assignment]
        try:
            bundle = rd.load_real_data_bundle(
                core_symbols=["BTCUSDT", "BU2606"],
                start=date(2026, 3, 29),
                end=date(2026, 3, 30),
                max_symbols=2,
                report_symbol_cap=2,
                workers=2,
                cache_dir=None,
                strict_cutoff=date(2026, 3, 30),
                review_days=0,
                include_post_review=False,
            )
            self.assertEqual(bundle.universe, ["BTCUSDT", "BU2606"])
            self.assertEqual(bundle.news_records, 2)
            self.assertFalse(bundle.news_daily.empty)
            self.assertAlmostEqual(float(bundle.news_daily.iloc[-1]), 0.3, places=6)
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]
            rd.fetch_symbol_news_and_reports = original_fetch_nr  # type: ignore[assignment]

    def test_load_real_data_bundle_news_daily_ignores_zero_score_coverage_rows(self) -> None:
        import lie_engine.research.real_data as rd

        original_load_universe = rd.load_universe
        original_fetch_one = rd._fetch_one_symbol
        original_fetch_nr = rd.fetch_symbol_news_and_reports

        def _fake_load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
            _ = max_symbols
            return ["BTCUSDT", "BU2606"]

        def _fake_fetch_one(symbol: str, start: date, end: date):
            _ = (start, end)
            df = pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-03-30"]),
                    "symbol": [symbol],
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.0],
                    "volume": [1000.0],
                    "source": ["mock"],
                    "asset_class": ["crypto" if symbol == "BTCUSDT" else "future"],
                }
            )
            return symbol, df, None

        def _fake_fetch_nr(symbol: str, start: date, end: date):
            _ = (start, end)
            rows = [
                {"date": date(2026, 3, 30), "symbol": symbol, "news_score": 0.0},
                {"date": date(2026, 3, 30), "symbol": symbol, "news_score": -0.6 if symbol == "BU2606" else 0.3},
            ]
            return (
                pd.DataFrame(rows),
                pd.DataFrame(columns=["date", "symbol", "report_score"]),
            )

        rd.load_universe = _fake_load_universe  # type: ignore[assignment]
        rd._fetch_one_symbol = _fake_fetch_one  # type: ignore[assignment]
        rd.fetch_symbol_news_and_reports = _fake_fetch_nr  # type: ignore[assignment]
        try:
            bundle = rd.load_real_data_bundle(
                core_symbols=["BTCUSDT", "BU2606"],
                start=date(2026, 3, 30),
                end=date(2026, 3, 30),
                max_symbols=2,
                report_symbol_cap=2,
                workers=2,
                cache_dir=None,
                strict_cutoff=date(2026, 3, 30),
                review_days=0,
                include_post_review=False,
            )
            self.assertEqual(bundle.news_records, 4)
            self.assertFalse(bundle.news_daily.empty)
            self.assertAlmostEqual(float(bundle.news_daily.iloc[-1]), -0.15, places=6)
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]
            rd.fetch_symbol_news_and_reports = original_fetch_nr  # type: ignore[assignment]

    def test_load_real_data_bundle_exposes_symbol_level_news_daily_and_equal_weight_aggregate(self) -> None:
        import lie_engine.research.real_data as rd

        original_load_universe = rd.load_universe
        original_fetch_one = rd._fetch_one_symbol
        original_fetch_nr = rd.fetch_symbol_news_and_reports

        def _fake_load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
            _ = max_symbols
            return ["BTCUSDT", "BU2606"]

        def _fake_fetch_one(symbol: str, start: date, end: date):
            _ = (start, end)
            df = pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-03-30"]),
                    "symbol": [symbol],
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.0],
                    "volume": [1000.0],
                    "source": ["mock"],
                    "asset_class": ["crypto" if symbol == "BTCUSDT" else "future"],
                }
            )
            return symbol, df, None

        def _fake_fetch_nr(symbol: str, start: date, end: date):
            _ = (start, end)
            if symbol == "BTCUSDT":
                rows = [
                    {"date": date(2026, 3, 30), "symbol": symbol, "news_score": 0.0},
                    {"date": date(2026, 3, 30), "symbol": symbol, "news_score": 0.6},
                    {"date": date(2026, 3, 30), "symbol": symbol, "news_score": 0.6},
                ]
            else:
                rows = [
                    {"date": date(2026, 3, 30), "symbol": symbol, "news_score": -0.3},
                ]
            return (
                pd.DataFrame(rows),
                pd.DataFrame(columns=["date", "symbol", "report_score"]),
            )

        rd.load_universe = _fake_load_universe  # type: ignore[assignment]
        rd._fetch_one_symbol = _fake_fetch_one  # type: ignore[assignment]
        rd.fetch_symbol_news_and_reports = _fake_fetch_nr  # type: ignore[assignment]
        try:
            bundle = rd.load_real_data_bundle(
                core_symbols=["BTCUSDT", "BU2606"],
                start=date(2026, 3, 30),
                end=date(2026, 3, 30),
                max_symbols=2,
                report_symbol_cap=2,
                workers=2,
                cache_dir=None,
                strict_cutoff=date(2026, 3, 30),
                review_days=0,
                include_post_review=False,
            )
            self.assertFalse(bundle.news_daily_by_symbol.empty)
            by_symbol = bundle.news_daily_by_symbol.sort_values(["date", "symbol"]).reset_index(drop=True)
            self.assertEqual(list(by_symbol["symbol"]), ["BTCUSDT", "BU2606"])
            self.assertAlmostEqual(float(by_symbol.loc[by_symbol["symbol"] == "BTCUSDT", "news_score"].iloc[0]), 0.6, places=6)
            self.assertAlmostEqual(float(by_symbol.loc[by_symbol["symbol"] == "BU2606", "news_score"].iloc[0]), -0.3, places=6)
            self.assertFalse(bundle.news_daily.empty)
            self.assertAlmostEqual(float(bundle.news_daily.iloc[-1]), 0.15, places=6)
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]
            rd.fetch_symbol_news_and_reports = original_fetch_nr  # type: ignore[assignment]

    def test_fetch_symbol_news_and_reports_supports_public_reports_for_crypto_and_future(self) -> None:
        import lie_engine.research.real_data as rd
        from lie_engine.models import NewsEvent
        from unittest.mock import patch

        def _fake_load_public_news_events(start: date, end: date, lang: str = "zh"):  # noqa: ANN001
            _ = (start, end, lang)
            return [
                NewsEvent(
                    event_id="n-shmet",
                    ts=datetime(2026, 3, 30, 9, 0),
                    title="[SHMET] 伊朗首都德黑兰发生爆炸",
                    content="地缘冲突升级，原油链承压，风险偏好下降。",
                    lang="zh",
                    source="public_macro_news",
                    category="产业链",
                    confidence=0.74,
                    entities=["原油"],
                    importance=0.66,
                ),
                NewsEvent(
                    event_id="n-cctv",
                    ts=datetime(2026, 3, 30, 19, 0),
                    title="[CCTV] 今年前两个月我国电子商务稳定发展",
                    content="数字消费持续活跃，产业电商成为增长主动力，稳增长政策继续推进。",
                    lang="zh",
                    source="public_macro_news",
                    category="政策",
                    confidence=0.84,
                    entities=[],
                    importance=0.72,
                ),
            ]

        with patch.object(rd, "_load_public_news_events", new=_fake_load_public_news_events):
            future_news, future_report = rd.fetch_symbol_news_and_reports("BU2606", date(2026, 3, 29), date(2026, 3, 30))
            crypto_news, crypto_report = rd.fetch_symbol_news_and_reports("BTCUSDT", date(2026, 3, 29), date(2026, 3, 30))

        self.assertFalse(future_news.empty)
        self.assertFalse(future_report.empty)
        self.assertFalse(crypto_news.empty)
        self.assertFalse(crypto_report.empty)
        self.assertLess(float(future_news["news_score"].iloc[0]), 0.0)
        self.assertGreater(float(future_report["report_score"].iloc[0]), 0.0)
        self.assertLess(float(crypto_news["news_score"].iloc[0]), 0.0)
        self.assertGreater(float(crypto_report["report_score"].iloc[0]), 0.0)

    def test_fetch_symbol_news_and_reports_keeps_zero_scored_public_news_coverage(self) -> None:
        import lie_engine.research.real_data as rd
        from lie_engine.models import NewsEvent
        from unittest.mock import patch

        def _fake_load_public_news_events(start: date, end: date, lang: str = "zh"):  # noqa: ANN001
            _ = (start, end, lang)
            return [
                NewsEvent(
                    event_id="n-baidu-warehouse",
                    ts=datetime(2026, 3, 30, 15, 10),
                    title="[Baidu] 中国3月30日上期所每日仓单变动-原油(桶)",
                    content="中国 | 中国3月30日上期所每日仓单变动-原油(桶) | 预期=nan | 前值=0",
                    lang="zh",
                    source="public_macro_news",
                    category="宏观",
                    confidence=0.82,
                    entities=[],
                    importance=0.75,
                ),
            ]

        with patch.object(rd, "_load_public_news_events", new=_fake_load_public_news_events):
            future_news, future_report = rd.fetch_symbol_news_and_reports("BU2606", date(2026, 3, 30), date(2026, 3, 30))

        self.assertFalse(future_news.empty)
        self.assertTrue(future_report.empty)
        self.assertAlmostEqual(float(future_news["news_score"].iloc[0]), 0.0, places=6)

    def test_load_real_data_bundle_collects_public_reports_for_crypto_and_future_universe(self) -> None:
        import lie_engine.research.real_data as rd
        from lie_engine.models import NewsEvent
        from unittest.mock import patch

        original_load_universe = rd.load_universe
        original_fetch_one = rd._fetch_one_symbol

        def _fake_load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
            _ = max_symbols
            return ["BTCUSDT", "BU2606"]

        def _fake_fetch_one(symbol: str, start: date, end: date):
            _ = (start, end)
            df = pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-03-29", "2026-03-30"]),
                    "symbol": [symbol, symbol],
                    "open": [1.0, 1.1],
                    "high": [1.1, 1.2],
                    "low": [0.9, 1.0],
                    "close": [1.0, 1.1],
                    "volume": [1000.0, 1200.0],
                    "source": ["mock", "mock"],
                    "asset_class": ["crypto" if symbol == "BTCUSDT" else "future"] * 2,
                }
            )
            return symbol, df, None

        def _fake_load_public_news_events(start: date, end: date, lang: str = "zh"):  # noqa: ANN001
            _ = (start, end, lang)
            return [
                NewsEvent(
                    event_id="n-shmet",
                    ts=datetime(2026, 3, 30, 9, 0),
                    title="[SHMET] 伊朗首都德黑兰发生爆炸",
                    content="地缘冲突升级，原油链承压，风险偏好下降。",
                    lang="zh",
                    source="public_macro_news",
                    category="产业链",
                    confidence=0.74,
                    entities=["原油"],
                    importance=0.66,
                ),
                NewsEvent(
                    event_id="n-cctv",
                    ts=datetime(2026, 3, 30, 19, 0),
                    title="[CCTV] 今年前两个月我国电子商务稳定发展",
                    content="数字消费持续活跃，产业电商成为增长主动力，稳增长政策继续推进。",
                    lang="zh",
                    source="public_macro_news",
                    category="政策",
                    entities=[],
                    confidence=0.84,
                    importance=0.72,
                ),
            ]

        rd.load_universe = _fake_load_universe  # type: ignore[assignment]
        rd._fetch_one_symbol = _fake_fetch_one  # type: ignore[assignment]
        try:
            with patch.object(rd, "_load_public_news_events", new=_fake_load_public_news_events):
                bundle = rd.load_real_data_bundle(
                    core_symbols=["BTCUSDT", "BU2606"],
                    start=date(2026, 3, 29),
                    end=date(2026, 3, 30),
                    max_symbols=2,
                    report_symbol_cap=2,
                    workers=2,
                    cache_dir=None,
                    strict_cutoff=date(2026, 3, 30),
                    review_days=0,
                    include_post_review=False,
                )
            self.assertEqual(bundle.universe, ["BTCUSDT", "BU2606"])
            self.assertGreater(bundle.news_records, 0)
            self.assertGreater(bundle.report_records, 0)
            self.assertFalse(bundle.news_daily.empty)
            self.assertFalse(bundle.report_daily.empty)
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]

    def test_load_real_data_bundle_cache_ttl_zero_forces_refresh(self) -> None:
        import lie_engine.research.real_data as rd

        original_load_universe = rd.load_universe
        original_fetch_one = rd._fetch_one_symbol
        original_fetch_nr = rd.fetch_symbol_news_and_reports

        def _fake_load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
            _ = (core_symbols, max_symbols)
            return ["BTCUSDT"]

        call_state = {"n": 0}

        def _fake_fetch_one(symbol: str, start: date, end: date):
            _ = (start, end)
            call_state["n"] += 1
            close_value = 100.0 if call_state["n"] == 1 else 200.0
            df = pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-03-30"]),
                    "symbol": [symbol],
                    "open": [close_value],
                    "high": [close_value],
                    "low": [close_value],
                    "close": [close_value],
                    "volume": [1000.0],
                    "source": ["mock"],
                    "asset_class": ["crypto"],
                }
            )
            return symbol, df, None

        def _fake_fetch_nr(symbol: str, start: date, end: date):
            _ = (symbol, start, end)
            return (
                pd.DataFrame([{"date": date(2026, 3, 30), "symbol": "BTCUSDT", "news_score": 0.1}]),
                pd.DataFrame(columns=["date", "symbol", "report_score"]),
            )

        rd.load_universe = _fake_load_universe  # type: ignore[assignment]
        rd._fetch_one_symbol = _fake_fetch_one  # type: ignore[assignment]
        rd.fetch_symbol_news_and_reports = _fake_fetch_nr  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                cache_dir = Path(td)
                bundle_1 = rd.load_real_data_bundle(
                    core_symbols=["BTCUSDT"],
                    start=date(2026, 3, 30),
                    end=date(2026, 3, 30),
                    max_symbols=1,
                    report_symbol_cap=1,
                    workers=1,
                    cache_dir=cache_dir,
                    cache_ttl_hours=8.0,
                    strict_cutoff=date(2026, 3, 30),
                    review_days=0,
                    include_post_review=False,
                )
                bundle_2 = rd.load_real_data_bundle(
                    core_symbols=["BTCUSDT"],
                    start=date(2026, 3, 30),
                    end=date(2026, 3, 30),
                    max_symbols=1,
                    report_symbol_cap=1,
                    workers=1,
                    cache_dir=cache_dir,
                    cache_ttl_hours=0.0,
                    strict_cutoff=date(2026, 3, 30),
                    review_days=0,
                    include_post_review=False,
                )
            self.assertAlmostEqual(float(bundle_1.bars["close"].iloc[0]), 100.0, places=6)
            self.assertAlmostEqual(float(bundle_2.bars["close"].iloc[0]), 200.0, places=6)
            self.assertFalse(bool(bundle_2.fetch_stats.get("cache_hit", False)))
            self.assertGreaterEqual(call_state["n"], 2)
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]
            rd.fetch_symbol_news_and_reports = original_fetch_nr  # type: ignore[assignment]

    def test_load_real_data_bundle_attaches_public_macro_features_to_bars_and_cache(self) -> None:
        import lie_engine.research.real_data as rd

        original_load_universe = rd.load_universe
        original_fetch_one = rd._fetch_one_symbol
        original_fetch_nr = rd.fetch_symbol_news_and_reports
        original_get_provider = rd._get_public_news_provider

        def _fake_load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
            _ = (core_symbols, max_symbols)
            return ["BU2606"]

        def _fake_fetch_one(symbol: str, start: date, end: date):
            _ = (start, end)
            df = pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-03-30"]),
                    "symbol": [symbol],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.0],
                    "volume": [1000.0],
                    "source": ["mock"],
                    "asset_class": ["future"],
                }
            )
            return symbol, df, None

        def _fake_fetch_nr(symbol: str, start: date, end: date):
            _ = (symbol, start, end)
            return (
                pd.DataFrame(columns=["date", "symbol", "news_score"]),
                pd.DataFrame(columns=["date", "symbol", "report_score"]),
            )

        class _FakePublicProvider:
            def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
                _ = (start, end)
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-03-29"]),
                        "cpi_yoy": [0.2],
                        "fixed_asset_investment_cum": [52721.0],
                        "consumer_confidence_index": [90.6],
                        "source": ["public_macro_news"],
                        "fixed_asset_investment_source": ["nbs:fixed_asset_investment"],
                    }
                )

            def fetch_news(self, start_ts, end_ts, lang: str = "zh"):  # type: ignore[no-untyped-def]
                _ = (start_ts, end_ts, lang)
                return []

        rd.load_universe = _fake_load_universe  # type: ignore[assignment]
        rd._fetch_one_symbol = _fake_fetch_one  # type: ignore[assignment]
        rd.fetch_symbol_news_and_reports = _fake_fetch_nr  # type: ignore[assignment]
        rd._get_public_news_provider = lambda: _FakePublicProvider()  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                cache_dir = Path(td)
                bundle = rd.load_real_data_bundle(
                    core_symbols=["BU2606"],
                    start=date(2026, 3, 29),
                    end=date(2026, 3, 30),
                    max_symbols=1,
                    report_symbol_cap=1,
                    workers=1,
                    cache_dir=cache_dir,
                    strict_cutoff=date(2026, 3, 30),
                    review_days=0,
                    include_post_review=False,
                )

                self.assertIn("cpi_yoy", bundle.bars.columns)
                self.assertIn("fixed_asset_investment_cum", bundle.bars.columns)
                self.assertIn("consumer_confidence_index", bundle.bars.columns)
                self.assertAlmostEqual(float(bundle.bars.iloc[0]["cpi_yoy"]), 0.2, places=6)
                self.assertAlmostEqual(float(bundle.bars.iloc[0]["fixed_asset_investment_cum"]), 52721.0, places=6)
                self.assertAlmostEqual(float(bundle.bars.iloc[0]["consumer_confidence_index"]), 90.6, places=6)

                cache_files = sorted(cache_dir.glob("*_bars.parquet"))
                self.assertEqual(len(cache_files), 1)
                cached_bars = pd.read_parquet(cache_files[0])
                self.assertIn("cpi_yoy", cached_bars.columns)
                self.assertIn("fixed_asset_investment_cum", cached_bars.columns)
                self.assertIn("consumer_confidence_index", cached_bars.columns)
        finally:
            rd.load_universe = original_load_universe  # type: ignore[assignment]
            rd._fetch_one_symbol = original_fetch_one  # type: ignore[assignment]
            rd.fetch_symbol_news_and_reports = original_fetch_nr  # type: ignore[assignment]
            rd._get_public_news_provider = original_get_provider  # type: ignore[assignment]

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
                self.assertTrue((Path(summary.output_dir) / "best_trade_journal.csv").exists())
                self.assertTrue((Path(summary.output_dir) / "best_holding_daily_symbol_exposure.csv").exists())
                self.assertTrue(any(Path(summary.output_dir).glob("candidate_*_trade_journal.csv")))
                self.assertTrue(any(Path(summary.output_dir).glob("candidate_*_holding_daily_symbol_exposure.csv")))
                self.assertIn("review_metrics", summary.candidates[0].to_dict())
                self.assertIn("trade_journal_path", summary.candidates[0].to_dict())
                self.assertIn("holding_exposure_path", summary.candidates[0].to_dict())
                self.assertEqual(summary.cutoff_ts, "2025-12-31T23:59:59")
                self.assertEqual(summary.bar_max_ts, "")
                self.assertEqual(summary.news_max_ts, "")
                self.assertEqual(summary.report_max_ts, "")
                self.assertTrue(bool(summary.term_registry.get("exists", False)))
                self.assertGreaterEqual(int(summary.term_registry.get("atoms_total", 0)), 1)
                self.assertAlmostEqual(float(summary.max_drawdown_target), 0.05, places=6)
                self.assertAlmostEqual(float(summary.review_max_drawdown_target), 0.07, places=6)
                self.assertAlmostEqual(float(summary.drawdown_soft_band), 0.03, places=6)
        finally:
            sl_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]

    def test_run_strategy_lab_uses_symbol_level_factor_series(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]),
                "symbol": ["BU2606"] * 4,
                "open": [1.0, 1.1, 1.2, 1.3],
                "high": [1.1, 1.2, 1.3, 1.4],
                "low": [0.9, 1.0, 1.1, 1.2],
                "close": [1.0, 1.1, 1.2, 1.25],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0],
                "source": ["mock"] * 4,
                "asset_class": ["future"] * 4,
            }
        )
        idx = sorted(pd.to_datetime(bars["ts"]).dt.date.unique())
        bundle = RealDataBundle(
            bars=bars,
            universe=["BU2606"],
            news_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            report_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            news_daily_by_symbol=pd.DataFrame(
                {
                    "date": idx,
                    "symbol": ["BU2606"] * len(idx),
                    "news_score": [0.4, 0.5, -0.2, 0.1],
                }
            ),
            report_daily_by_symbol=pd.DataFrame(
                {
                    "date": idx,
                    "symbol": ["BU2606"] * len(idx),
                    "report_score": [0.3, 0.2, 0.1, 0.0],
                }
            ),
            news_records=4,
            report_records=4,
            fetch_stats={"mocked": True},
            cutoff_date=date(2025, 1, 5),
            review_days=0,
        )

        captured: dict[str, Any] = {}

        original_loader = sl_mod.load_real_data_bundle
        original_report_insights = sl_mod._report_insights

        def _capture_report_insights(news_daily, report_daily):  # type: ignore[no-untyped-def]
            captured["news_daily"] = news_daily
            captured["report_daily"] = report_daily
            return original_report_insights(news_daily, report_daily)

        sl_mod.load_real_data_bundle = lambda **kwargs: bundle  # type: ignore[assignment]
        sl_mod._report_insights = _capture_report_insights  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                summary = run_strategy_lab(
                    output_root=Path(td),
                    core_symbols=["BU2606"],
                    start=date(2025, 1, 1),
                    end=date(2025, 1, 5),
                    candidate_count=1,
                    workers=1,
                )
            self.assertTrue(summary.best_candidate is not None)
            self.assertAlmostEqual(float(captured["news_daily"].iloc[0]), 0.4, places=6)
            self.assertAlmostEqual(float(captured["report_daily"].iloc[0]), 0.3, places=6)
        finally:
            sl_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]
            sl_mod._report_insights = original_report_insights  # type: ignore[assignment]

    def test_run_strategy_lab_passes_backtest_daily_symbol_exposure_into_factor_alignment(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod
        from lie_engine.models import BacktestResult

        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]),
                "symbol": ["BU2606"] * 4,
                "open": [1.0, 1.1, 1.2, 1.3],
                "high": [1.1, 1.2, 1.3, 1.4],
                "low": [0.9, 1.0, 1.1, 1.2],
                "close": [1.0, 1.1, 1.2, 1.25],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0],
                "source": ["mock"] * 4,
                "asset_class": ["future"] * 4,
            }
        )
        idx = sorted(pd.to_datetime(bars["ts"]).dt.date.unique())
        bundle = RealDataBundle(
            bars=bars,
            universe=["BU2606"],
            news_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            report_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            news_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606"] * len(idx), "news_score": [0.4, 0.5, -0.2, 0.1]}),
            report_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606"] * len(idx), "report_score": [0.3, 0.2, 0.1, 0.0]}),
            news_records=4,
            report_records=4,
            fetch_stats={"mocked": True},
            cutoff_date=date(2025, 1, 5),
            review_days=0,
        )

        captured: dict[str, Any] = {}
        original_loader = sl_mod.load_real_data_bundle
        original_factor_alignment = sl_mod._factor_alignment
        original_run_backtest = sl_mod.run_event_backtest

        def _fake_factor_alignment(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured["daily_symbol_exposure"] = kwargs.get("daily_symbol_exposure")
            captured["news_daily_by_symbol"] = kwargs.get("news_daily_by_symbol")
            return 0.0

        def _fake_run_event_backtest(*args, **kwargs):  # type: ignore[no-untyped-def]
            _ = (args, kwargs)
            return BacktestResult(
                start=date(2025, 1, 1),
                end=date(2025, 1, 5),
                total_return=0.01,
                annual_return=0.10,
                max_drawdown=0.02,
                win_rate=0.6,
                profit_factor=1.2,
                expectancy=0.01,
                trades=1,
                violations=0,
                positive_window_ratio=1.0,
                equity_curve=[{"date": "2025-01-02", "equity": 1.0}, {"date": "2025-01-05", "equity": 1.02}],
                by_asset={},
                daily_symbol_exposure=[{"date": "2025-01-02", "symbol": "BU2606", "weight": 1.0}],
            )

        sl_mod.load_real_data_bundle = lambda **kwargs: bundle  # type: ignore[assignment]
        sl_mod._factor_alignment = _fake_factor_alignment  # type: ignore[assignment]
        sl_mod.run_event_backtest = _fake_run_event_backtest  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                summary = run_strategy_lab(
                    output_root=Path(td),
                    core_symbols=["BU2606"],
                    start=date(2025, 1, 1),
                    end=date(2025, 1, 5),
                    max_symbols=1,
                    report_symbol_cap=1,
                    workers=1,
                    review_days=0,
                    candidate_count=1,
                )
            self.assertTrue(summary.best_candidate is not None)
            self.assertEqual(captured["daily_symbol_exposure"][0]["symbol"], "BU2606")
            self.assertFalse(captured["news_daily_by_symbol"].empty)
        finally:
            sl_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]
            sl_mod._factor_alignment = original_factor_alignment  # type: ignore[assignment]
            sl_mod.run_event_backtest = original_run_backtest  # type: ignore[assignment]

    def test_run_strategy_lab_prefers_holding_daily_symbol_exposure(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod
        from lie_engine.models import BacktestResult

        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]),
                "symbol": ["BU2606"] * 4,
                "open": [1.0, 1.1, 1.2, 1.3],
                "high": [1.1, 1.2, 1.3, 1.4],
                "low": [0.9, 1.0, 1.1, 1.2],
                "close": [1.0, 1.1, 1.2, 1.25],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0],
                "source": ["mock"] * 4,
                "asset_class": ["future"] * 4,
            }
        )
        idx = sorted(pd.to_datetime(bars["ts"]).dt.date.unique())
        bundle = RealDataBundle(
            bars=bars,
            universe=["BU2606"],
            news_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            report_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            news_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606"] * len(idx), "news_score": [0.4, 0.5, -0.2, 0.1]}),
            report_daily_by_symbol=pd.DataFrame({"date": idx, "symbol": ["BU2606"] * len(idx), "report_score": [0.3, 0.2, 0.1, 0.0]}),
            news_records=4,
            report_records=4,
            fetch_stats={"mocked": True},
            cutoff_date=date(2025, 1, 5),
            review_days=0,
        )

        captured: dict[str, Any] = {}
        original_loader = sl_mod.load_real_data_bundle
        original_factor_alignment = sl_mod._factor_alignment
        original_run_backtest = sl_mod.run_event_backtest

        def _fake_factor_alignment(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured["daily_symbol_exposure"] = kwargs.get("daily_symbol_exposure")
            return 0.0

        def _fake_run_event_backtest(*args, **kwargs):  # type: ignore[no-untyped-def]
            _ = (args, kwargs)
            return BacktestResult(
                start=date(2025, 1, 1),
                end=date(2025, 1, 5),
                total_return=0.01,
                annual_return=0.10,
                max_drawdown=0.02,
                win_rate=0.6,
                profit_factor=1.2,
                expectancy=0.01,
                trades=1,
                violations=0,
                positive_window_ratio=1.0,
                equity_curve=[{"date": "2025-01-02", "equity": 1.0}, {"date": "2025-01-05", "equity": 1.02}],
                by_asset={},
                daily_symbol_exposure=[{"date": "2025-01-02", "symbol": "WRONG", "weight": 1.0}],
                holding_daily_symbol_exposure=[{"date": "2025-01-02", "symbol": "BU2606", "weight": 1.0}],
            )

        sl_mod.load_real_data_bundle = lambda **kwargs: bundle  # type: ignore[assignment]
        sl_mod._factor_alignment = _fake_factor_alignment  # type: ignore[assignment]
        sl_mod.run_event_backtest = _fake_run_event_backtest  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                run_strategy_lab(
                    output_root=Path(td),
                    core_symbols=["BU2606"],
                    start=date(2025, 1, 1),
                    end=date(2025, 1, 5),
                    max_symbols=1,
                    report_symbol_cap=1,
                    workers=1,
                    review_days=0,
                    candidate_count=1,
                )
            self.assertEqual(captured["daily_symbol_exposure"][0]["symbol"], "BU2606")
        finally:
            sl_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]
            sl_mod._factor_alignment = original_factor_alignment  # type: ignore[assignment]
            sl_mod.run_event_backtest = original_run_backtest  # type: ignore[assignment]

    def test_run_strategy_lab_dd_target_acceptance_gate(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        bars = make_multi_symbol_bars()
        bars["asset_class"] = bars["asset_class"].astype(str)
        bars["source"] = "mock"
        idx = sorted(pd.to_datetime(bars["ts"]).dt.date.unique())
        news_daily = pd.Series([0.0] * len(idx), index=idx, dtype=float)
        report_daily = pd.Series([0.0] * len(idx), index=idx, dtype=float)
        bundle = RealDataBundle(
            bars=bars,
            review_bars=pd.DataFrame(columns=bars.columns),
            universe=sorted(set(bars["symbol"])),
            news_daily=news_daily,
            report_daily=report_daily,
            news_records=0,
            report_records=0,
            fetch_stats={"mocked": True, "strict_cutoff_enforced": True},
            cutoff_date=date(2025, 12, 31),
            review_days=0,
        )

        original_loader = sl_mod.load_real_data_bundle
        original_bt = sl_mod.run_event_backtest

        def _fake_backtest(*, bars, start, end, cfg, trend_thr, mean_thr, atr_extreme):  # type: ignore[no-untyped-def]
            exp = float(getattr(cfg, "exposure_scale", 1.0))
            mdd = 0.045 if exp <= 0.21 else 0.065
            ann = 0.18 if exp <= 0.21 else 0.28
            return BacktestResult(
                start=start,
                end=end,
                total_return=ann,
                annual_return=ann,
                max_drawdown=mdd,
                win_rate=0.60,
                profit_factor=1.30,
                expectancy=0.01,
                trades=12,
                violations=0,
                positive_window_ratio=0.85,
                equity_curve=[{"date": start.isoformat(), "equity": 1.0}, {"date": end.isoformat(), "equity": 1.0 + ann}],
                by_asset={},
            )

        sl_mod.load_real_data_bundle = lambda **kwargs: bundle  # type: ignore[assignment]
        sl_mod.run_event_backtest = _fake_backtest  # type: ignore[assignment]
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
                    review_days=0,
                    candidate_count=6,
                    max_drawdown_target=0.05,
                    review_max_drawdown_target=0.07,
                    drawdown_soft_band=0.03,
                )
                accepted = [c for c in summary.candidates if bool(c.accepted)]
                rejected = [c for c in summary.candidates if not bool(c.accepted)]
                self.assertGreaterEqual(len(accepted), 1)
                self.assertGreaterEqual(len(rejected), 1)
                self.assertTrue(all(float(c.validation_metrics.get("max_drawdown", 1.0)) <= 0.05 for c in accepted))
                self.assertTrue(any(float(c.validation_metrics.get("max_drawdown", 0.0)) > 0.05 for c in rejected))
                self.assertAlmostEqual(float(summary.max_drawdown_target), 0.05, places=6)
                self.assertAlmostEqual(float(summary.review_max_drawdown_target), 0.07, places=6)
                self.assertAlmostEqual(float(summary.drawdown_soft_band), 0.03, places=6)
        finally:
            sl_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]
            sl_mod.run_event_backtest = original_bt  # type: ignore[assignment]

    def test_strategy_lab_market_insights_include_brooks_proxies(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        bars = make_multi_symbol_bars()
        out = sl_mod._market_insights(bars)  # type: ignore[attr-defined]
        for key in (
            "brooks_trend_bar_z",
            "brooks_micro_channel_bias",
            "brooks_two_legged_bias",
            "brooks_exhaustion_z",
            "wyckoff_accumulation_bias",
            "wyckoff_distribution_bias",
            "vpa_effort_result_bias",
            "vpa_climax_z",
        ):
            self.assertIn(key, out)
            v = float(out.get(key, 0.0))
            self.assertTrue(v == v)
            self.assertLess(abs(v), 10.0)

    def test_strategy_lab_candidate_generation_brooks_support_hazard_shift(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        report = {
            "news_bias_z": 0.0,
            "report_bias_z": 0.0,
            "news_report_agreement": 0.1,
        }
        base_market = {
            "trend_strength_z": 0.2,
            "volatility_z": 0.1,
            "tail_risk_z": 0.0,
            "brooks_trend_bar_z": 0.0,
            "brooks_micro_channel_bias": 0.0,
            "brooks_two_legged_bias": 0.0,
            "brooks_exhaustion_z": 0.0,
            "wyckoff_accumulation_bias": 0.0,
            "wyckoff_distribution_bias": 0.0,
            "vpa_effort_result_bias": 0.0,
            "vpa_climax_z": 0.0,
        }
        support_market = dict(base_market)
        support_market.update(
            {
                "brooks_trend_bar_z": 1.4,
                "brooks_micro_channel_bias": 0.8,
                "brooks_two_legged_bias": 0.9,
                "brooks_exhaustion_z": -0.2,
                "wyckoff_accumulation_bias": 0.8,
                "wyckoff_distribution_bias": -0.2,
                "vpa_effort_result_bias": 0.7,
                "vpa_climax_z": -0.2,
            }
        )
        hazard_market = dict(base_market)
        hazard_market.update(
            {
                "brooks_trend_bar_z": 0.2,
                "brooks_micro_channel_bias": -0.6,
                "brooks_two_legged_bias": 0.1,
                "brooks_exhaustion_z": 1.5,
                "wyckoff_accumulation_bias": -0.2,
                "wyckoff_distribution_bias": 0.9,
                "vpa_effort_result_bias": -0.7,
                "vpa_climax_z": 1.8,
            }
        )

        base = sl_mod._generate_candidates(market=base_market, report=report, candidate_count=1, exposure_cap=0.35)[0]  # type: ignore[attr-defined]
        support = sl_mod._generate_candidates(market=support_market, report=report, candidate_count=1, exposure_cap=0.35)[0]  # type: ignore[attr-defined]
        hazard = sl_mod._generate_candidates(market=hazard_market, report=report, candidate_count=1, exposure_cap=0.35)[0]  # type: ignore[attr-defined]

        self.assertGreater(float(support.get("theory_brooks_weight", 0.0)), float(base.get("theory_brooks_weight", 0.0)))
        self.assertGreater(float(hazard.get("theory_penalty_max", 0.0)), float(base.get("theory_penalty_max", 0.0)))
        self.assertLess(float(hazard.get("theory_conflict_fuse", 1.0)), float(base.get("theory_conflict_fuse", 1.0)))
        self.assertGreater(float(support.get("theory_wyckoff_weight", 0.0)), float(base.get("theory_wyckoff_weight", 0.0)))
        self.assertGreater(float(support.get("theory_vpa_weight", 0.0)), float(base.get("theory_vpa_weight", 0.0)))
        self.assertLessEqual(int(hazard.get("hold_days", 0)), int(support.get("hold_days", 0)))
        self.assertGreater(float(support.get("brooks_support_score", 0.0)), float(hazard.get("brooks_support_score", 0.0)))
        self.assertGreater(float(hazard.get("brooks_hazard_score", 0.0)), float(support.get("brooks_hazard_score", 0.0)))
        self.assertGreater(float(hazard.get("wyckoff_hazard_score", 0.0)), float(support.get("wyckoff_hazard_score", 0.0)))
        self.assertGreater(float(support.get("vpa_support_score", 0.0)), float(hazard.get("vpa_support_score", 0.0)))

    def test_strategy_lab_candidate_generation_low_activity_relaxes_thresholds(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        market = {
            "trend_strength_z": 0.1,
            "volatility_z": 0.0,
            "tail_risk_z": 0.0,
            "brooks_trend_bar_z": 0.1,
            "brooks_micro_channel_bias": 0.0,
            "brooks_two_legged_bias": 0.0,
            "brooks_exhaustion_z": 0.0,
            "wyckoff_accumulation_bias": 0.0,
            "wyckoff_distribution_bias": 0.0,
            "vpa_effort_result_bias": 0.0,
            "vpa_climax_z": 0.0,
        }
        report = {
            "news_bias_z": 0.0,
            "report_bias_z": 0.0,
            "news_report_agreement": 0.0,
        }
        normal = sl_mod._generate_candidates(  # type: ignore[attr-defined]
            market=market,
            report=report,
            candidate_count=1,
            exposure_cap=0.20,
            low_activity_mode=False,
        )[0]
        relaxed = sl_mod._generate_candidates(  # type: ignore[attr-defined]
            market=market,
            report=report,
            candidate_count=1,
            exposure_cap=0.20,
            low_activity_mode=True,
        )[0]
        self.assertLess(float(relaxed.get("signal_confidence_min", 0.0)), float(normal.get("signal_confidence_min", 0.0)))
        self.assertLess(float(relaxed.get("convexity_min", 0.0)), float(normal.get("convexity_min", 0.0)))
        self.assertLessEqual(int(relaxed.get("hold_days", 99)), int(normal.get("hold_days", 0)))
        self.assertGreaterEqual(int(relaxed.get("max_daily_trades", 0)), int(normal.get("max_daily_trades", 0)))

    def test_strategy_lab_candidate_generation_crypto_low_activity_relaxes_more(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        market = {
            "trend_strength_z": 0.1,
            "volatility_z": 0.0,
            "tail_risk_z": 0.0,
            "brooks_trend_bar_z": 0.1,
            "brooks_micro_channel_bias": 0.0,
            "brooks_two_legged_bias": 0.0,
            "brooks_exhaustion_z": 0.0,
            "wyckoff_accumulation_bias": 0.0,
            "wyckoff_distribution_bias": 0.0,
            "vpa_effort_result_bias": 0.0,
            "vpa_climax_z": 0.0,
        }
        report = {
            "news_bias_z": 0.0,
            "report_bias_z": 0.0,
            "news_report_agreement": 0.0,
        }
        low = sl_mod._generate_candidates(  # type: ignore[attr-defined]
            market=market,
            report=report,
            candidate_count=1,
            exposure_cap=0.20,
            low_activity_mode=True,
            crypto_mode=False,
        )[0]
        crypto_low = sl_mod._generate_candidates(  # type: ignore[attr-defined]
            market=market,
            report=report,
            candidate_count=1,
            exposure_cap=0.20,
            low_activity_mode=True,
            crypto_mode=True,
        )[0]
        self.assertLess(float(crypto_low.get("signal_confidence_min", 0.0)), float(low.get("signal_confidence_min", 0.0)))
        self.assertLess(float(crypto_low.get("convexity_min", 0.0)), float(low.get("convexity_min", 0.0)))
        self.assertLessEqual(int(crypto_low.get("hold_days", 99)), int(low.get("hold_days", 0)))
        self.assertGreaterEqual(int(crypto_low.get("max_daily_trades", 0)), int(low.get("max_daily_trades", 0)))

    def test_strategy_lab_candidate_generation_crypto_mode_prefers_crypto_templates(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        market = {
            "trend_strength_z": 0.0,
            "volatility_z": 0.0,
            "tail_risk_z": 0.0,
            "brooks_trend_bar_z": 0.0,
            "brooks_micro_channel_bias": 0.0,
            "brooks_two_legged_bias": 0.0,
            "brooks_exhaustion_z": 0.0,
            "wyckoff_accumulation_bias": 0.0,
            "wyckoff_distribution_bias": 0.0,
            "vpa_effort_result_bias": 0.0,
            "vpa_climax_z": 0.0,
        }
        report = {
            "news_bias_z": 0.0,
            "report_bias_z": 0.0,
            "news_report_agreement": 0.0,
        }
        crypto = sl_mod._generate_candidates(  # type: ignore[attr-defined]
            market=market,
            report=report,
            candidate_count=2,
            exposure_cap=0.20,
            low_activity_mode=False,
            crypto_mode=True,
        )
        self.assertTrue(str(crypto[0].get("name", "")).startswith("crypto_tactical_"))
        self.assertTrue(str(crypto[1].get("name", "")).startswith("crypto_swing_flow_"))
    def test_strategy_lab_resolve_exposure_cap_prefers_feasible_ablation(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            run = out_root / "research" / "theory_ablation_20260304_120000"
            run.mkdir(parents=True, exist_ok=True)
            payload = {
                "start": "2025-01-01",
                "end": "2025-12-31",
                "risk_feasible": True,
                "best_feasible_case": {
                    "name": "exp_auto_best",
                    "exposure_scale": 0.11,
                },
            }
            (run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            cap, meta = sl_mod._resolve_exposure_cap(  # type: ignore[attr-defined]
                output_root=out_root,
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                max_drawdown_target=0.05,
                market={
                    "brooks_exhaustion_z": 0.0,
                    "brooks_trend_bar_z": 0.0,
                    "brooks_two_legged_bias": 0.0,
                },
            )
            self.assertTrue(bool(meta.get("ablation_found", False)))
            self.assertLess(float(cap), float(meta.get("base_cap", 1.0)))
            self.assertAlmostEqual(float(meta.get("ablation_cap", 0.0)), 0.1265, places=4)
            self.assertAlmostEqual(float(cap), 0.1265, places=4)

    def test_strategy_lab_resolve_exposure_cap_hazard_multiplier(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            calm_cap, calm_meta = sl_mod._resolve_exposure_cap(  # type: ignore[attr-defined]
                output_root=out_root,
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                max_drawdown_target=0.05,
                market={
                    "brooks_exhaustion_z": 0.1,
                    "brooks_trend_bar_z": 0.0,
                    "brooks_two_legged_bias": 0.0,
                },
            )
            hazard_cap, hazard_meta = sl_mod._resolve_exposure_cap(  # type: ignore[attr-defined]
                output_root=out_root,
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                max_drawdown_target=0.05,
                market={
                    "brooks_exhaustion_z": 2.5,
                    "brooks_trend_bar_z": -0.2,
                    "brooks_two_legged_bias": -0.4,
                },
            )
            self.assertFalse(bool(calm_meta.get("ablation_found", False)))
            self.assertFalse(bool(hazard_meta.get("ablation_found", False)))
            self.assertLess(float(hazard_meta.get("hazard_multiplier", 1.0)), float(calm_meta.get("hazard_multiplier", 1.0)))
            self.assertLess(float(hazard_cap), float(calm_cap))

    def test_strategy_lab_resolve_exposure_cap_accepts_covering_ablation_window(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)
            run = out_root / "research" / "theory_ablation_20260304_120000"
            run.mkdir(parents=True, exist_ok=True)
            payload = {
                "start": "2025-01-01",
                "end": "2025-12-31",
                "risk_feasible": True,
                "best_feasible_case": {
                    "name": "exp_auto_best",
                    "exposure_scale": 0.09,
                },
            }
            (run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            cap, meta = sl_mod._resolve_exposure_cap(  # type: ignore[attr-defined]
                output_root=out_root,
                start=date(2025, 7, 5),
                end=date(2025, 12, 31),
                max_drawdown_target=0.05,
                market={
                    "brooks_exhaustion_z": 0.0,
                    "brooks_trend_bar_z": 0.0,
                    "brooks_two_legged_bias": 0.0,
                },
            )
            self.assertTrue(bool(meta.get("ablation_found", False)))
            self.assertEqual(int(meta.get("ablation_match_priority", -1)), 1)
            self.assertGreaterEqual(int(meta.get("ablation_match_gap_days", -1)), 0)
            self.assertLess(float(cap), float(meta.get("base_cap", 1.0)))

    def test_strategy_lab_score_penalizes_zero_trade_case(self) -> None:
        import lie_engine.research.strategy_lab as sl_mod

        train_bt = BacktestResult(
            start=date(2025, 1, 1),
            end=date(2025, 6, 30),
            total_return=0.10,
            annual_return=0.10,
            max_drawdown=0.03,
            win_rate=0.55,
            profit_factor=1.2,
            expectancy=0.01,
            trades=12,
            violations=0,
            positive_window_ratio=0.80,
            equity_curve=[{"date": "2025-01-01", "equity": 1.0}, {"date": "2025-06-30", "equity": 1.1}],
            by_asset={},
        )
        zero_trade_valid = BacktestResult(
            start=date(2025, 7, 1),
            end=date(2025, 12, 31),
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            trades=0,
            violations=0,
            positive_window_ratio=1.0,
            equity_curve=[{"date": "2025-07-01", "equity": 1.0}, {"date": "2025-12-31", "equity": 1.0}],
            by_asset={},
        )
        active_valid = BacktestResult(
            start=date(2025, 7, 1),
            end=date(2025, 12, 31),
            total_return=0.03,
            annual_return=0.03,
            max_drawdown=0.02,
            win_rate=0.5,
            profit_factor=1.1,
            expectancy=0.005,
            trades=8,
            violations=0,
            positive_window_ratio=0.70,
            equity_curve=[{"date": "2025-07-01", "equity": 1.0}, {"date": "2025-12-31", "equity": 1.03}],
            by_asset={},
        )

        score_zero = float(
            sl_mod._score_candidate(  # type: ignore[attr-defined]
                train_bt=train_bt,
                valid_bt=zero_trade_valid,
                align_train=0.0,
                align_valid=0.0,
                review_bt=None,
                align_review=0.0,
                robustness_score=0.0,
                max_drawdown_target=0.05,
                review_max_drawdown_target=0.07,
                drawdown_soft_band=0.03,
            )
        )
        score_active = float(
            sl_mod._score_candidate(  # type: ignore[attr-defined]
                train_bt=train_bt,
                valid_bt=active_valid,
                align_train=0.0,
                align_valid=0.0,
                review_bt=None,
                align_review=0.0,
                robustness_score=0.0,
                max_drawdown_target=0.05,
                review_max_drawdown_target=0.07,
                drawdown_soft_band=0.03,
            )
        )
        self.assertLess(score_zero, score_active)

    def test_theory_ablation_auto_exposure_search_fields(self) -> None:
        ab_mod = _load_theory_ablation_module()

        def _mk_case(name: str, exposure: float):  # type: ignore[no-untyped-def]
            mdd = float(exposure * 0.25)
            breached = bool(mdd > 0.05)
            return ab_mod.AblationCase(
                name=name,
                theory_enabled=False,
                exposure_scale=float(exposure),
                theory_ict_weight=1.0,
                theory_brooks_weight=1.0,
                theory_lie_weight=1.2,
                theory_confidence_boost_max=5.0,
                theory_penalty_max=6.0,
                theory_min_confluence=0.38,
                theory_conflict_fuse=0.72,
                annual_return=float(exposure),
                max_drawdown=float(mdd),
                drawdown_excess=float(max(0.0, mdd - 0.05)),
                target_breached=breached,
                positive_window_ratio=0.5,
                profit_factor=1.0,
                win_rate=0.5,
                trades=1,
                violations=0,
                sharpe_like=0.0,
                objective=float(exposure if not breached else -1.0),
            )

        best_case, cfg = ab_mod._auto_exposure_search(
            auto_low=0.08,
            auto_high=0.40,
            auto_iters=6,
            theory_enabled=False,
            evaluate_case=_mk_case,
        )
        self.assertIsNotNone(best_case)
        assert best_case is not None
        self.assertTrue(bool(cfg.get("enabled", False)))
        self.assertFalse(bool(cfg.get("theory_enabled", True)))
        self.assertGreaterEqual(int(len(cfg.get("trace", []))), 7)
        self.assertAlmostEqual(float(cfg.get("best_exposure", 0.0)), float(best_case.exposure_scale), places=6)
        self.assertEqual(str(cfg.get("best_case", {}).get("name", "")), str(best_case.name))
        self.assertLessEqual(float(best_case.max_drawdown), 0.05)

    def test_theory_ablation_weight_grid_supports_legacy_and_extended(self) -> None:
        ab_mod = _load_theory_ablation_module()
        legacy = ab_mod._parse_weight_grid("1.0:1.1:1.2")
        extended = ab_mod._parse_weight_grid("1.0:1.1:1.2:0.8:0.7")
        self.assertEqual(len(legacy), 1)
        self.assertEqual(len(extended), 1)
        self.assertEqual(len(legacy[0]), 5)
        self.assertEqual(len(extended[0]), 5)
        self.assertAlmostEqual(float(legacy[0][3]), 0.0, places=6)
        self.assertAlmostEqual(float(legacy[0][4]), 0.0, places=6)
        self.assertAlmostEqual(float(extended[0][3]), 0.8, places=6)
        self.assertAlmostEqual(float(extended[0][4]), 0.7, places=6)

    def test_theory_ablation_execute_with_watchdog_retry_success(self) -> None:
        ab_mod = _load_theory_ablation_module()

        base_case = ab_mod.AblationCase(
            name="ok_case",
            theory_enabled=False,
            exposure_scale=0.2,
            theory_ict_weight=1.0,
            theory_brooks_weight=1.0,
            theory_lie_weight=1.2,
            theory_confidence_boost_max=5.0,
            theory_penalty_max=6.0,
            theory_min_confluence=0.38,
            theory_conflict_fuse=0.72,
            annual_return=0.02,
            max_drawdown=0.04,
            drawdown_excess=0.0,
            target_breached=False,
            positive_window_ratio=0.7,
            profit_factor=1.1,
            win_rate=0.5,
            trades=3,
            violations=0,
            sharpe_like=0.2,
            objective=0.1,
        )
        call_count = {"n": 0}

        def _flaky():  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            return base_case

        out, errors = ab_mod._execute_with_watchdog(
            label="flaky_case",
            timeout_seconds=0,
            retry_times=2,
            run_fn=_flaky,
        )
        self.assertIsNotNone(out)
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("transient", errors[0])

    def test_theory_ablation_auto_exposure_search_abort_on_error(self) -> None:
        ab_mod = _load_theory_ablation_module()

        counter = {"n": 0}

        def _eval_case(name: str, exposure: float):  # type: ignore[no-untyped-def]
            counter["n"] += 1
            if counter["n"] == 2:
                raise RuntimeError("probe_failed")
            return ab_mod.AblationCase(
                name=name,
                theory_enabled=False,
                exposure_scale=float(exposure),
                theory_ict_weight=1.0,
                theory_brooks_weight=1.0,
                theory_lie_weight=1.2,
                theory_confidence_boost_max=5.0,
                theory_penalty_max=6.0,
                theory_min_confluence=0.38,
                theory_conflict_fuse=0.72,
                annual_return=0.01,
                max_drawdown=0.03,
                drawdown_excess=0.0,
                target_breached=False,
                positive_window_ratio=0.6,
                profit_factor=1.1,
                win_rate=0.4,
                trades=2,
                violations=0,
                sharpe_like=0.1,
                objective=0.05,
            )

        best_case, cfg = ab_mod._auto_exposure_search(
            auto_low=0.08,
            auto_high=0.20,
            auto_iters=3,
            theory_enabled=False,
            evaluate_case=_eval_case,
        )
        self.assertIsNotNone(best_case)
        self.assertTrue(bool(cfg.get("aborted", False)))
        self.assertGreaterEqual(len(cfg.get("errors", [])), 1)
        self.assertIn("probe_failed", str(cfg.get("errors", [""])[0]))

    def test_theory_ablation_render_markdown_includes_execution_guard(self) -> None:
        ab_mod = _load_theory_ablation_module()
        sample = ab_mod.AblationCase(
            name="baseline_off",
            theory_enabled=False,
            exposure_scale=0.2,
            theory_ict_weight=1.0,
            theory_brooks_weight=1.0,
            theory_lie_weight=1.2,
            theory_confidence_boost_max=5.0,
            theory_penalty_max=6.0,
            theory_min_confluence=0.38,
            theory_conflict_fuse=0.72,
            annual_return=0.01,
            max_drawdown=0.03,
            drawdown_excess=0.0,
            target_breached=False,
            positive_window_ratio=0.6,
            profit_factor=1.1,
            win_rate=0.4,
            trades=2,
            violations=0,
            sharpe_like=0.1,
            objective=0.05,
        )
        md = ab_mod._render_markdown(
            started_at="2026-03-04T00:00:00",
            ended_at="2026-03-04T00:00:01",
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
            symbols=["300750"],
            bars_rows=100,
            cases=[sample],
            data_source="mock",
            max_drawdown_target=0.05,
            drawdown_soft_band=0.03,
            auto_exposure_search={"enabled": False},
            bundle_load_guard={"requested_workers": 2, "effective_workers": 1, "attempts": 2, "timeout_seconds": 60, "fallback_to_single_worker": True},
            execution_guard={"case_timeout_seconds": 30, "case_retry_times": 1, "failure_count": 2, "fuse_max_failures": 3, "fuse_triggered": True},
        )
        self.assertIn("case_timeout_seconds", md)
        self.assertIn("fuse_triggered", md)

    def test_theory_ablation_bundle_load_recovery_single_worker(self) -> None:
        ab_mod = _load_theory_ablation_module()

        bars = make_multi_symbol_bars()
        bars["asset_class"] = bars["asset_class"].astype(str)
        bars["source"] = "mock"
        idx = sorted(pd.to_datetime(bars["ts"]).dt.date.unique())
        bundle_ok = RealDataBundle(
            bars=bars,
            universe=sorted(set(bars["symbol"])),
            news_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            report_daily=pd.Series([0.0] * len(idx), index=idx, dtype=float),
            news_records=0,
            report_records=0,
            fetch_stats={"mocked": True},
        )

        calls: list[int] = []
        original_loader = ab_mod.load_real_data_bundle

        def _fake_loader(**kwargs):  # type: ignore[no-untyped-def]
            w = int(kwargs.get("workers", 0))
            calls.append(w)
            if w > 1:
                raise TimeoutError("simulated_timeout")
            return bundle_ok

        ab_mod.load_real_data_bundle = _fake_loader  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                bundle, guard = ab_mod._load_bundle_with_recovery(
                    core_symbols=["300750", "002050"],
                    start=date(2025, 1, 1),
                    end=date(2025, 12, 31),
                    max_symbols=4,
                    report_symbol_cap=2,
                    workers=2,
                    cache_dir=Path(td),
                    cache_ttl_hours=8.0,
                    strict_cutoff=date(2025, 12, 31),
                    review_days=0,
                    include_post_review=False,
                    timeout_seconds=5,
                    allow_single_worker_recovery=True,
                )
            self.assertIs(bundle, bundle_ok)
            self.assertEqual(calls, [2, 1])
            self.assertEqual(int(guard.get("requested_workers", 0)), 2)
            self.assertEqual(int(guard.get("effective_workers", 0)), 1)
            self.assertTrue(bool(guard.get("fallback_to_single_worker", False)))
            self.assertEqual(int(guard.get("attempts", 0)), 2)
            self.assertGreaterEqual(len(guard.get("errors", [])), 1)
        finally:
            ab_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
