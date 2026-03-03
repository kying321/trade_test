from __future__ import annotations

from datetime import date
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

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
                self.assertTrue(bool(summary.term_registry.get("exists", False)))
                self.assertGreaterEqual(int(summary.term_registry.get("atoms_total", 0)), 1)
                self.assertAlmostEqual(float(summary.max_drawdown_target), 0.05, places=6)
                self.assertAlmostEqual(float(summary.review_max_drawdown_target), 0.07, places=6)
                self.assertAlmostEqual(float(summary.drawdown_soft_band), 0.03, places=6)
        finally:
            sl_mod.load_real_data_bundle = original_loader  # type: ignore[assignment]

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
