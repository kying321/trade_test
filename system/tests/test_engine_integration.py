from __future__ import annotations

from contextlib import closing
import sys
from pathlib import Path
import unittest
from datetime import date, datetime, timedelta, timezone
import tempfile
import json
import sqlite3
from unittest.mock import patch

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.engine import LieEngine
from lie_engine.models import BacktestResult, ReviewDelta
from tests.helpers import make_bars, make_multi_symbol_bars


class EngineIntegrationTests(unittest.TestCase):
    def _make_engine(self) -> tuple[LieEngine, Path]:
        project_root = Path(__file__).resolve().parents[1]
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp_root = Path(td.name)

        cfg_data = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
        cfg_data["paths"] = {"output": "output", "sqlite": "output/artifacts/lie_engine.db"}
        cfg_data.setdefault("validation", {})
        cfg_data["validation"]["review_autorun_strategy_lab_if_missing"] = False
        cfg_data["validation"]["use_mode_profiles"] = False
        cfg_data["validation"]["review_backtest_lookback_days"] = 540
        cfg_data["validation"]["style_drift_adaptive_enabled"] = False
        cfg_data["validation"]["micro_cross_source_build_missing_provider"] = False
        cfg_path = tmp_root / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")
        return LieEngine(config_path=cfg_path), tmp_root

    def test_run_eod_outputs_contract_files(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        out = eng.run_eod(d)
        self.assertIn("briefing", out)
        self.assertIn("manifest", out)
        daily_dir = tmp_root / "output" / "daily"
        self.assertTrue((daily_dir / "2026-02-13_briefing.md").exists())
        self.assertTrue((daily_dir / "2026-02-13_signals.json").exists())
        self.assertTrue((daily_dir / "2026-02-13_positions.csv").exists())
        self.assertTrue((daily_dir / "2026-02-13_mode_feedback.json").exists())
        self.assertTrue((tmp_root / "output" / "artifacts" / "paper_positions_open.json").exists())
        self.assertTrue((tmp_root / "output" / "artifacts" / "broker_snapshot" / "2026-02-13.json").exists())
        self.assertTrue((tmp_root / "output" / "artifacts" / "manifests" / "eod_2026-02-13.json").exists())
        self.assertIn("mode_feedback", out)
        self.assertIn("broker_snapshot", out)
        self.assertIn("system_time_sync_probe", out)

    def test_run_eod_mode_feedback_includes_microstructure_summary(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_eod(d)
        mode_feedback_path = tmp_root / "output" / "daily" / "2026-02-13_mode_feedback.json"
        payload = json.loads(mode_feedback_path.read_text(encoding="utf-8"))
        self.assertIn("microstructure", payload)
        micro = payload.get("microstructure", {})
        self.assertIn("enabled", micro)
        self.assertIn("symbols_total", micro)
        self.assertIn("symbols_with_data", micro)
        self.assertIn("coverage", micro)
        self.assertIn("symbols_schema_ok", micro)
        self.assertIn("symbols_schema_fail", micro)
        self.assertIn("symbols_time_sync_available", micro)
        self.assertIn("symbols_time_sync_ok", micro)
        self.assertIn("symbols_time_sync_breach", micro)
        self.assertIn("cross_source_audit", micro)
        self.assertIn("time_sync", payload)
        self.assertIn("status", payload.get("time_sync", {}))
        cs = micro.get("cross_source_audit", {})
        self.assertIn("quality_7d", cs)
        self.assertIn("quality_report_path", cs)
        report_path = Path(str(cs.get("quality_report_path", "")))
        self.assertTrue(report_path.exists())

    def test_run_time_sync_probe_with_mock_sntp(self) -> None:
        eng, tmp_root = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["system_time_sync_probe_enabled"] = True
        eng.settings.raw["validation"]["system_time_sync_hard_fuse_enabled"] = True
        eng.settings.raw["validation"]["system_time_sync_primary_source"] = "time.google.com"
        eng.settings.raw["validation"]["system_time_sync_secondary_source"] = "time.cloudflare.com"
        eng.settings.raw["validation"]["system_time_sync_probe_timeout_seconds"] = 1.0
        eng.settings.raw["validation"]["system_time_sync_max_offset_ms"] = 5
        eng.settings.raw["validation"]["system_time_sync_max_rtt_ms"] = 120
        eng.settings.raw["validation"]["system_time_sync_min_ok_sources"] = 1

        fake_output = "\n".join(
            [
                "selected:",
                "sntp_exchange {",
                "  delay: 0000.0000 (0.050000000)",
                "}",
                "+0.001000 +/- 0.050000 time.google.com 198.18.0.1",
            ]
        )

        class _FakeProc:
            returncode = 0
            stdout = fake_output
            stderr = ""

        with patch("lie_engine.engine.shutil.which", return_value="/usr/bin/sntp"), patch(
            "lie_engine.engine.subprocess.run",
            return_value=_FakeProc(),
        ):
            out = eng.run_time_sync_probe(date(2026, 2, 13))

        self.assertEqual(str(out.get("status", "")), "ok")
        self.assertTrue(bool(out.get("pass", False)))
        self.assertGreaterEqual(int(out.get("ok_sources", 0)), 1)
        with closing(sqlite3.connect(tmp_root / "output" / "artifacts" / "lie_engine.db")) as conn:
            daily = pd.read_sql_query("SELECT status, ok_sources FROM system_time_sync_daily", conn)
            rows = pd.read_sql_query("SELECT source, ok, offset_ms, rtt_ms FROM system_time_sync_source_state", conn)
        self.assertFalse(daily.empty)
        self.assertFalse(rows.empty)

    def test_collect_micro_factor_map_builds_missing_cross_source_providers(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng.providers = []
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["micro_cross_source_build_missing_provider"] = True
        eng.settings.raw["validation"]["micro_cross_source_primary"] = "binance_spot_public"
        eng.settings.raw["validation"]["micro_cross_source_secondary"] = "bybit_spot_public"
        eng.settings.raw["validation"]["microstructure_symbols"] = ["BTCUSDT"]
        eng.settings.raw["validation"]["micro_min_trade_count"] = 1

        def _mk_provider(name: str):
            class _P:
                def __init__(self, n: str) -> None:
                    self.name = n

                def fetch_l2(self, symbol, start_ts, end_ts, depth=20):  # noqa: ANN001
                    now_ms = int(datetime(2026, 2, 13, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
                    return pd.DataFrame(
                        [
                            {
                                "exchange": n,
                                "symbol": symbol,
                                "event_ts_ms": now_ms,
                                "recv_ts_ms": now_ms,
                                "seq": 100,
                                "prev_seq": 99,
                                "bids": [[100.0, 1.0]],
                                "asks": [[100.2, 1.2]],
                                "source": n,
                            }
                        ]
                    )

                def fetch_trades(self, symbol, start_ts, end_ts, limit=2000):  # noqa: ANN001
                    base_ms = int(datetime(2026, 2, 13, 11, 59, 0, tzinfo=timezone.utc).timestamp() * 1000)
                    return pd.DataFrame(
                        [
                            {
                                "exchange": n,
                                "symbol": symbol,
                                "trade_id": f"{n}-1",
                                "event_ts_ms": base_ms,
                                "recv_ts_ms": base_ms + 10,
                                "price": 100.1,
                                "qty": 0.5,
                                "side": "BUY",
                                "source": n,
                            },
                            {
                                "exchange": n,
                                "symbol": symbol,
                                "trade_id": f"{n}-2",
                                "event_ts_ms": base_ms + 200,
                                "recv_ts_ms": base_ms + 210,
                                "price": 100.2,
                                "qty": 0.4,
                                "side": "SELL",
                                "source": n,
                            },
                        ]
                    )

                def fetch_time_sync_sample(self):  # noqa: ANN001
                    return {"source": n, "offset_ms": 1, "rtt_ms": 50}

            n = str(name)
            return _P(n)

        provider_map = {
            "binance_spot_public": _mk_provider("binance_spot_public"),
            "bybit_spot_public": _mk_provider("bybit_spot_public"),
        }
        eng._build_provider_by_name = lambda name: provider_map.get(str(name))  # type: ignore[method-assign]

        out, source_rows = eng._collect_micro_factor_map_with_rows(as_of=d, symbols=["BTCUSDT"])
        self.assertIn("BTCUSDT", out)
        sources = {str(x.get("source", "")) for x in source_rows}
        self.assertIn("binance_spot_public", sources)
        self.assertIn("bybit_spot_public", sources)

    def test_run_micro_capture_writes_artifact_and_sqlite_rows(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)

        eng._collect_micro_factor_map_with_rows = lambda as_of, symbols, override_symbols=None: (  # type: ignore[method-assign]
            {
                "BTCUSDT": {
                    "source": "binance_spot_public",
                    "has_data": True,
                    "schema_ok": True,
                    "sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 120,
                    "evidence_score": 0.9,
                    "queue_imbalance": 0.2,
                    "ofi_norm": 0.1,
                    "micro_alignment": 0.15,
                    "max_trade_gap_ms": 200,
                    "sync_skew_ms": 8.0,
                    "time_sync_available": True,
                    "time_sync_ok": True,
                    "time_sync_offset_ms": 1,
                    "time_sync_rtt_ms": 40,
                    "time_sync_max_offset_ms": 5,
                    "time_sync_max_rtt_ms": 120,
                }
            },
            [
                {
                    "symbol": "BTCUSDT",
                    "source": "binance_spot_public",
                    "has_data": True,
                    "schema_ok": True,
                    "sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 120,
                    "evidence_score": 0.9,
                    "time_sync_available": True,
                    "time_sync_ok": True,
                    "time_sync_offset_ms": 1,
                    "time_sync_rtt_ms": 40,
                    "time_sync_max_offset_ms": 5,
                    "time_sync_max_rtt_ms": 120,
                }
            ],
        )
        eng._collect_cross_source_audit = lambda as_of, symbols: {  # type: ignore[method-assign]
            "enabled": True,
            "active": True,
            "status": "ok",
            "primary_source": "binance_spot_public",
            "secondary_source": "bybit_spot_public",
            "symbols_selected": 1,
            "symbols_audited": 1,
            "symbols_insufficient": 0,
            "fail_ratio": 0.0,
            "rows": [
                {
                    "symbol": "BTCUSDT",
                    "auditable": True,
                    "row_status": "audited_pass",
                    "sync_ok": True,
                    "gap_ok": True,
                }
            ],
        }
        eng._collect_system_time_sync_probe = lambda as_of: {  # type: ignore[method-assign]
            "enabled": True,
            "active": True,
            "status": "ok",
            "pass": True,
            "ok_sources": 1,
            "available_sources": 2,
            "sources": [{"source": "time.google.com", "available": True, "ok": True}],
            "artifact_path": "",
        }

        out = eng.run_micro_capture(as_of=d, symbols=["BTCUSDT"])
        self.assertEqual(str(out.get("status", "")), "ok")
        artifact_path = Path(str(out.get("artifact", "")))
        self.assertTrue(artifact_path.exists())
        with closing(sqlite3.connect(tmp_root / "output" / "artifacts" / "lie_engine.db")) as conn:
            runs = pd.read_sql_query("SELECT status, pass, symbols_selected FROM micro_capture_runs", conn)
            src = pd.read_sql_query("SELECT symbol, source FROM micro_capture_source_state", conn)
            sel = pd.read_sql_query("SELECT symbol, source FROM micro_capture_symbol_state", conn)
            cross = pd.read_sql_query("SELECT symbol, audit_status FROM micro_capture_cross_source", conn)
        self.assertFalse(runs.empty)
        self.assertFalse(src.empty)
        self.assertFalse(sel.empty)
        self.assertFalse(cross.empty)

    def test_market_factor_state_includes_cross_section_style(self) -> None:
        eng, _ = self._make_engine()
        bars = make_multi_symbol_bars()
        as_of = pd.to_datetime(bars["ts"]).max().date()
        regime = eng._regime_from_bars(as_of=as_of, bars=bars)
        state = eng._market_factor_state(
            sentiment={
                "pcr_50etf": 0.95,
                "iv_50etf": 0.21,
                "northbound_netflow": 1.5e8,
                "margin_balance_chg": 0.002,
            },
            regime=regime,
            bars=bars,
        )
        self.assertIn("value_preference", state)
        self.assertIn("size_preference", state)
        self.assertIn("dividend_preference", state)
        self.assertGreaterEqual(float(state.get("style_sample_size", 0.0)), 4.0)
        self.assertGreaterEqual(float(state.get("style_weight", 0.0)), 0.2)
        self.assertLessEqual(float(state.get("style_weight", 0.0)), 0.55)

    def test_mode_history_stats_aggregates_backtest_manifests(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 14)
        manifest_dir = tmp_root / "output" / "artifacts" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "backtest_2025-01-01_2026-02-10.json").write_text(
            json.dumps(
                {
                    "run_type": "backtest",
                    "run_id": "2025-01-01_2026-02-10",
                    "metrics": {
                        "annual_return": 0.10,
                        "max_drawdown": 0.12,
                        "win_rate": 0.50,
                        "profit_factor": 1.30,
                        "trades": 120,
                        "runtime_mode": "swing",
                    },
                    "checks": {"violations": 0},
                }
            ),
            encoding="utf-8",
        )
        (manifest_dir / "backtest_2025-06-01_2026-02-12.json").write_text(
            json.dumps(
                {
                    "run_type": "backtest",
                    "run_id": "2025-06-01_2026-02-12",
                    "metrics": {
                        "annual_return": 0.06,
                        "max_drawdown": 0.16,
                        "win_rate": 0.45,
                        "profit_factor": 1.10,
                        "trades": 80,
                        "runtime_mode": "swing",
                    },
                    "checks": {"violations": 1},
                }
            ),
            encoding="utf-8",
        )

        out = eng._mode_history_stats(as_of=d, lookback_days=365)
        self.assertIn("modes", out)
        self.assertIn("swing", out["modes"])
        swing = out["modes"]["swing"]
        self.assertEqual(int(swing["samples"]), 2)
        self.assertEqual(int(swing["total_violations"]), 1)

    def test_run_review_applies_mode_health_guard(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["mode_health_min_samples"] = 1
        eng.settings.raw["validation"]["mode_health_min_profit_factor"] = 1.0
        eng.settings.raw["validation"]["mode_health_min_win_rate"] = 0.40
        eng.settings.raw["validation"]["mode_health_max_drawdown_max"] = 0.18
        eng.settings.raw["validation"]["mode_health_max_violations"] = 0

        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.12,
            annual_return=0.06,
            max_drawdown=0.12,
            win_rate=0.52,
            profit_factor=1.3,
            expectancy=0.01,
            trades=120,
            violations=0,
            positive_window_ratio=0.85,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }
        eng._resolve_review_runtime_mode = lambda start, as_of: "swing"  # type: ignore[method-assign]
        eng._mode_history_stats = lambda as_of: {  # type: ignore[method-assign]
            "as_of": as_of.isoformat(),
            "lookback_days": 365,
            "modes": {
                "swing": {
                    "samples": 3,
                    "avg_profit_factor": 0.8,
                    "avg_win_rate": 0.35,
                    "worst_drawdown": 0.22,
                    "total_violations": 2,
                }
            },
        }

        review = eng.run_review(d)
        self.assertFalse(review.pass_gate)
        self.assertIn("MODE_HEALTH_DEGRADED", review.defects)
        self.assertEqual(int(review.parameter_changes.get("max_daily_trades", 0)), 1)
        self.assertEqual(int(review.parameter_changes.get("hold_days", 0)), 3)

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("runtime_mode"), "swing")
        self.assertIn("mode_health", payload)
        self.assertFalse(bool(payload.get("mode_health", {}).get("passed", True)))

    def test_run_review_applies_mode_adaptive_update(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["mode_adaptive_update_enabled"] = True
        eng.settings.raw["validation"]["mode_adaptive_update_min_samples"] = 1
        eng.settings.raw["validation"]["mode_adaptive_update_step"] = 0.12
        eng.settings.raw["validation"]["mode_adaptive_good_profit_factor"] = 1.20
        eng.settings.raw["validation"]["mode_adaptive_bad_profit_factor"] = 1.00
        eng.settings.raw["validation"]["mode_adaptive_good_win_rate"] = 0.50
        eng.settings.raw["validation"]["mode_adaptive_bad_win_rate"] = 0.40
        eng.settings.raw["validation"]["mode_adaptive_good_drawdown_max"] = 0.12
        eng.settings.raw["validation"]["mode_adaptive_bad_drawdown_max"] = 0.18

        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.15,
            annual_return=0.08,
            max_drawdown=0.10,
            win_rate=0.56,
            profit_factor=1.5,
            expectancy=0.02,
            trades=150,
            violations=0,
            positive_window_ratio=0.90,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }
        eng._resolve_review_runtime_mode = lambda start, as_of: "swing"  # type: ignore[method-assign]
        eng._mode_history_stats = lambda as_of: {  # type: ignore[method-assign]
            "as_of": as_of.isoformat(),
            "lookback_days": 365,
            "modes": {
                "swing": {
                    "samples": 5,
                    "avg_profit_factor": 1.45,
                    "avg_win_rate": 0.55,
                    "worst_drawdown": 0.10,
                    "total_violations": 0,
                }
            },
        }

        review = eng.run_review(d)
        self.assertTrue(any("mode_adaptive_update=" in n for n in review.notes))
        self.assertIn("convexity_min", review.parameter_changes)
        self.assertIn("hold_days", review.parameter_changes)
        self.assertIn("max_daily_trades", review.parameter_changes)

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        self.assertIn("mode_adaptive", payload)
        self.assertTrue(bool(payload.get("mode_adaptive", {}).get("applied", False)))
        self.assertEqual(str(payload.get("mode_adaptive", {}).get("direction", "")), "expand")

    def test_run_review_emits_style_diagnostics(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)

        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.12,
            annual_return=0.06,
            max_drawdown=0.10,
            win_rate=0.52,
            profit_factor=1.3,
            expectancy=0.01,
            trades=120,
            violations=0,
            positive_window_ratio=0.85,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }

        symbols = ["300750", "002050", "603026", "000830", "600519", "601318", "000001", "600000"]
        frames = []
        for i, symbol in enumerate(symbols):
            frame = make_bars(symbol, n=280, trend=0.03 + 0.01 * i, seed=100 + i)
            frame["market_cap"] = float((i + 1) * 1e10)
            frame["pe_ttm"] = float(10 + 5 * i)
            frame["dividend_yield"] = float(0.005 * i)
            frames.append(frame)
        bars = pd.concat(frames, ignore_index=True)
        eng._run_ingestion_range = lambda start, end, symbols: (bars, None)  # type: ignore[method-assign]

        review = eng.run_review(d)
        self.assertIn("style_spreads", review.style_diagnostics)
        self.assertIn("style_drift_score", review.style_diagnostics)
        self.assertTrue(any("style_diag=" in n for n in review.notes))

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        self.assertIn("style_diagnostics", payload)
        self.assertIn("style_spreads", payload.get("style_diagnostics", {}))

    def test_run_review_applies_style_drift_adaptive_guard(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "style_drift_adaptive_enabled": True,
                "style_drift_adaptive_confidence_step_max": 6.0,
                "style_drift_adaptive_trade_reduction_max": 2,
                "style_drift_adaptive_hold_reduction_max": 2,
                "style_drift_adaptive_trigger_ratio": 1.0,
                "style_drift_adaptive_ratio_for_max": 2.0,
                "style_drift_adaptive_block_ratio": 1.8,
            }
        )
        eng._resolve_review_runtime_mode = lambda start, as_of: "swing"  # type: ignore[method-assign]
        eng._mode_history_stats = lambda as_of: {  # type: ignore[method-assign]
            "as_of": as_of.isoformat(),
            "lookback_days": 365,
            "modes": {},
        }
        eng._review_style_diagnostics = lambda start, as_of: {  # type: ignore[method-assign]
            "active": True,
            "drift_gap_max": 0.01,
            "style_drift_score": 0.02,
            "alerts": ["style_drift:momentum"],
            "block_on_alert": False,
        }
        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.12,
            annual_return=0.06,
            max_drawdown=0.10,
            win_rate=0.52,
            profit_factor=1.3,
            expectancy=0.01,
            trades=120,
            violations=0,
            positive_window_ratio=0.85,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }

        review = eng.run_review(d)
        self.assertIn("STYLE_DRIFT_SEVERE", review.defects)
        self.assertFalse(bool(review.pass_gate))
        self.assertGreater(float(review.parameter_changes.get("signal_confidence_min", 0.0)), 60.0)
        self.assertLessEqual(float(review.parameter_changes.get("max_daily_trades", 9.0)), 1.0)
        self.assertLess(float(review.parameter_changes.get("hold_days", 99.0)), 5.0)
        self.assertTrue(any("style_drift_guard=" in n for n in review.notes))

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        self.assertIn("style_drift_guard", payload)
        self.assertTrue(bool(payload.get("style_drift_guard", {}).get("blocked", False)))

    def test_run_eod_blocks_new_positions_under_major_event_window(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng._major_event_window = lambda as_of, news: True  # type: ignore[method-assign]
        eng._loss_cooldown_active = lambda recent_trades: False  # type: ignore[method-assign]
        eng._black_swan_assessment = lambda regime, sentiment, news: (10.0, [], False)  # type: ignore[method-assign]

        out = eng.run_eod(d)
        self.assertEqual(out["plans"], 0)
        self.assertTrue(any("重大事件窗口" in x for x in out["non_trade_reasons"]))

    def test_run_eod_closes_paper_positions_into_executed_plans(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        eng.run_eod(d1)
        state_path = tmp_root / "output" / "artifacts" / "paper_positions_open.json"
        symbol = eng._core_symbols()[0]
        state_path.write_text(
            json.dumps(
                {
                    "as_of": d1.isoformat(),
                    "positions": [
                        {
                            "open_date": d1.isoformat(),
                            "symbol": symbol,
                            "side": "LONG",
                            "size_pct": 5.0,
                            "risk_pct": 1.0,
                            "entry_price": 100.0,
                            "stop_price": 1.0,
                            "target_price": 1000.0,
                            "runtime_mode": "swing",
                            "hold_days": 1,
                            "status": "OPEN",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        d2 = date(2026, 2, 14)
        out = eng.run_eod(d2)
        self.assertIn("closed_trades", out)
        self.assertGreaterEqual(int(out.get("closed_trades", 0)), 1)
        with closing(sqlite3.connect(tmp_root / "output" / "artifacts" / "lie_engine.db")) as conn:
            df = pd.read_sql_query("SELECT symbol, runtime_mode, mode, pnl, exit_reason FROM executed_plans", conn)
        self.assertFalse(df.empty)
        self.assertIn("runtime_mode", df.columns)
        self.assertIn("mode", df.columns)
        self.assertIn("pnl", df.columns)
        self.assertTrue((df["symbol"] == symbol).any())

        broker_path = tmp_root / "output" / "artifacts" / "broker_snapshot" / "2026-02-14.json"
        self.assertTrue(broker_path.exists())
        broker_payload = json.loads(broker_path.read_text(encoding="utf-8"))
        self.assertEqual(str(broker_payload.get("source", "")), "paper_engine")
        self.assertEqual(str(broker_payload.get("date", "")), "2026-02-14")
        self.assertIn("open_positions", broker_payload)
        self.assertIn("closed_pnl", broker_payload)

    def test_run_eod_uses_live_adapter_broker_snapshot_when_configured(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["broker_snapshot_source_mode"] = "live_adapter"
        eng.settings.raw["validation"]["broker_snapshot_live_inbox"] = "output/artifacts/broker_live_inbox"
        eng.settings.raw["validation"]["broker_snapshot_live_fallback_to_paper"] = False

        inbox = tmp_root / "output" / "artifacts" / "broker_live_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "2026-02-13.json").write_text(
            json.dumps(
                {
                    "source": "mock_live_broker",
                    "open_positions": 2,
                    "closed_count": 1,
                    "closed_pnl": 123.45,
                    "positions": [
                        {"symbol": "300750", "side": "LONG", "qty": 10, "entry_price": 100.0, "market_price": 101.0}
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.run_eod(d)
        broker_path = Path(str(out.get("broker_snapshot", "")))
        self.assertTrue(broker_path.exists())
        payload = json.loads(broker_path.read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("source", "")), "live_adapter")
        self.assertEqual(str(payload.get("adapter_source", "")), "mock_live_broker")
        self.assertEqual(int(payload.get("open_positions", 0)), 2)
        self.assertAlmostEqual(float(payload.get("closed_pnl", 0.0)), 123.45, places=6)

    def test_run_eod_live_adapter_uses_binance_mapping_profile(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["broker_snapshot_source_mode"] = "live_adapter"
        eng.settings.raw["validation"]["broker_snapshot_live_inbox"] = "output/artifacts/broker_live_inbox"
        eng.settings.raw["validation"]["broker_snapshot_live_fallback_to_paper"] = False
        eng.settings.raw["validation"]["broker_snapshot_live_mapping_profile"] = "binance"

        inbox = tmp_root / "output" / "artifacts" / "broker_live_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "2026-02-13.json").write_text(
            json.dumps(
                {
                    "broker": "binance_futures",
                    "summary": {"open_positions": 1, "closed_count": 3, "realized_pnl": 88.5},
                    "account": {
                        "positions": [
                            {
                                "symbol": "BTCUSDT",
                                "positionAmt": "-0.75",
                                "entryPrice": "50000",
                                "markPrice": "51000",
                                "positionSide": "BOTH",
                                "state": "OPEN",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.run_eod(d)
        broker_path = Path(str(out.get("broker_snapshot", "")))
        self.assertTrue(broker_path.exists())
        payload = json.loads(broker_path.read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("mapping_profile", "")), "binance")
        self.assertEqual(str(payload.get("adapter_source", "")), "binance_futures")
        self.assertEqual(int(payload.get("open_positions", 0)), 1)
        self.assertEqual(int(payload.get("closed_count", 0)), 3)
        self.assertAlmostEqual(float(payload.get("closed_pnl", 0.0)), 88.5, places=6)
        positions = payload.get("positions", [])
        self.assertEqual(len(positions), 1)
        row = positions[0]
        self.assertEqual(str(row.get("symbol", "")), "BTCUSDT")
        self.assertEqual(str(row.get("side", "")), "SHORT")
        self.assertAlmostEqual(float(row.get("qty", 0.0)), 0.75, places=6)
        self.assertAlmostEqual(float(row.get("entry_price", 0.0)), 50000.0, places=6)
        self.assertAlmostEqual(float(row.get("market_price", 0.0)), 51000.0, places=6)
        self.assertAlmostEqual(float(row.get("notional", 0.0)), 38250.0, places=6)

    def test_run_eod_live_adapter_uses_ibkr_mapping_profile(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["broker_snapshot_source_mode"] = "live_adapter"
        eng.settings.raw["validation"]["broker_snapshot_live_inbox"] = "output/artifacts/broker_live_inbox"
        eng.settings.raw["validation"]["broker_snapshot_live_fallback_to_paper"] = False
        eng.settings.raw["validation"]["broker_snapshot_live_mapping_profile"] = "ibkr"

        inbox = tmp_root / "output" / "artifacts" / "broker_live_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "2026-02-13.json").write_text(
            json.dumps(
                {
                    "source": "ibkr_gateway",
                    "summary": {"open_positions": 1, "closed_count": 2, "realized_pnl": -45.6},
                    "portfolio": [
                        {
                            "contract": {"symbol": "AAPL"},
                            "position": "5",
                            "avgCost": "100",
                            "marketPrice": "102",
                            "marketValue": "510",
                            "state": "ACTIVE",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.run_eod(d)
        broker_path = Path(str(out.get("broker_snapshot", "")))
        self.assertTrue(broker_path.exists())
        payload = json.loads(broker_path.read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("mapping_profile", "")), "ibkr")
        self.assertEqual(str(payload.get("adapter_source", "")), "ibkr_gateway")
        self.assertEqual(int(payload.get("open_positions", 0)), 1)
        self.assertEqual(int(payload.get("closed_count", 0)), 2)
        self.assertAlmostEqual(float(payload.get("closed_pnl", 0.0)), -45.6, places=6)
        positions = payload.get("positions", [])
        self.assertEqual(len(positions), 1)
        row = positions[0]
        self.assertEqual(str(row.get("symbol", "")), "AAPL")
        self.assertEqual(str(row.get("side", "")), "LONG")
        self.assertAlmostEqual(float(row.get("qty", 0.0)), 5.0, places=6)
        self.assertAlmostEqual(float(row.get("entry_price", 0.0)), 100.0, places=6)
        self.assertAlmostEqual(float(row.get("market_price", 0.0)), 102.0, places=6)
        self.assertAlmostEqual(float(row.get("notional", 0.0)), 510.0, places=6)

    def test_run_eod_live_adapter_uses_ctp_mapping_profile_and_side_enum(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["broker_snapshot_source_mode"] = "live_adapter"
        eng.settings.raw["validation"]["broker_snapshot_live_inbox"] = "output/artifacts/broker_live_inbox"
        eng.settings.raw["validation"]["broker_snapshot_live_fallback_to_paper"] = False
        eng.settings.raw["validation"]["broker_snapshot_live_mapping_profile"] = "ctp"

        inbox = tmp_root / "output" / "artifacts" / "broker_live_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "2026-02-13.json").write_text(
            json.dumps(
                {
                    "broker": "ctp_gateway",
                    "summary": {"open_positions": 1, "closed_count": 1, "realized_pnl": 12.0},
                    "position_list": [
                        {
                            "instrument": "rb2405",
                            "posi_direction": "3",
                            "position": "2",
                            "open_price": "3500",
                            "last_price": "3490",
                            "status": "OPEN",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.run_eod(d)
        broker_path = Path(str(out.get("broker_snapshot", "")))
        self.assertTrue(broker_path.exists())
        payload = json.loads(broker_path.read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("mapping_profile", "")), "ctp")
        positions = payload.get("positions", [])
        self.assertEqual(len(positions), 1)
        row = positions[0]
        self.assertEqual(str(row.get("symbol", "")), "rb2405")
        self.assertEqual(str(row.get("side", "")), "SHORT")
        self.assertAlmostEqual(float(row.get("qty", 0.0)), 2.0, places=6)

    def test_symbol_exposure_snapshot_uses_latest_date_only(self) -> None:
        eng, _ = self._make_engine()
        db_path = eng.ctx.sqlite_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute(
                "INSERT INTO latest_positions (date, symbol, size_pct, status) VALUES ('2026-02-13', '300750', 12.0, 'ACTIVE')"
            )
            conn.execute(
                "INSERT INTO latest_positions (date, symbol, size_pct, status) VALUES ('2026-02-14', '300750', 5.0, 'ACTIVE')"
            )
            conn.commit()
        by_symbol, _, total = eng._symbol_exposure_snapshot()
        self.assertAlmostEqual(float(by_symbol.get("300750", 0.0)), 5.0, places=6)
        self.assertAlmostEqual(float(total), 5.0, places=6)

    def test_run_slot_and_session(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)

        slot_out = eng.run_slot(as_of=d, slot="08:40")
        self.assertEqual(slot_out["slot"], "premarket")
        self.assertIn("result", slot_out)

        session_out = eng.run_session(as_of=d, include_review=False)
        self.assertIn("steps", session_out)
        self.assertIn("eod", session_out["steps"])
        self.assertTrue(session_out["steps"]["review_cycle"].get("skipped"))
        lock_path = tmp_root / "output" / "state" / "run-halfhour-pulse.lock"
        self.assertTrue(lock_path.exists())

    def test_run_slot_micro_capture_alias(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_micro_capture = lambda as_of, symbols=None: {  # type: ignore[method-assign]
            "as_of": as_of.isoformat(),
            "symbols": list(symbols or []),
            "status": "ok",
            "pass": True,
        }
        out = eng.run_slot(as_of=d, slot="micro-capture")
        self.assertEqual(out["slot"], "micro-capture")
        result = out.get("result", {})
        self.assertEqual(str(result.get("as_of", "")), "2026-02-13")

    def test_run_premarket_includes_source_confidence(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        out = eng.run_premarket(as_of=d)
        self.assertIn("source_confidence", out)
        self.assertIn("overall_score", out["source_confidence"])
        self.assertIn("runtime_mode", out)
        self.assertIn("mode_health", out)
        self.assertIn("risk_control", out)
        self.assertIn("risk_multiplier", out)
        self.assertIn("manifest", out)
        self.assertTrue(Path(str(out["manifest"])).exists())

    def test_run_intraday_includes_risk_control(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        out = eng.run_intraday_check(as_of=d, slot="10:30")
        self.assertEqual(out["slot"], "10:30")
        self.assertIn("runtime_mode", out)
        self.assertIn("mode_health", out)
        self.assertIn("risk_control", out)
        self.assertIn("risk_multiplier", out)
        self.assertIn("manifest", out)
        self.assertTrue(Path(str(out["manifest"])).exists())

    def test_stable_replay_one_day(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        out = eng.stable_replay_check(as_of=d, days=1)
        self.assertEqual(out["replay_days"], 1)
        self.assertIn("checks", out)
        self.assertEqual(len(out["checks"]), 1)

    def test_gate_report_generation(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_review(d)
        gate = eng.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertIn("checks", gate)
        self.assertIn("passed", gate)
        self.assertIn("slot_anomaly", gate)
        self.assertIn("reconcile_drift", gate)
        self.assertIn("style_drift", gate)
        self.assertIn("style_drift_ok", gate["checks"])
        self.assertIn("rollback_recommendation", gate)
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_gate_report.json").exists())

    def test_review_backtest_start_prefers_lookback_days(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 14)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["review_backtest_lookback_days"] = 365
        eng.settings.raw["validation"]["review_backtest_start_date"] = "2020-01-01"
        self.assertEqual(eng._review_backtest_start(as_of=d), date(2025, 2, 14))

    def test_review_backtest_start_rejects_future_start_date(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 14)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].pop("review_backtest_lookback_days", None)
        eng.settings.raw["validation"]["review_backtest_start_date"] = "2030-01-01"
        self.assertEqual(eng._review_backtest_start(as_of=d), date(2015, 1, 1))

    def test_gate_report_clears_stale_alert_on_pass(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)

        review_delta = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        review_delta.parent.mkdir(parents=True, exist_ok=True)
        review_delta.write_text("pass_gate: true\n", encoding="utf-8")

        alert = tmp_root / "output" / "logs" / "review_loop_alert_2026-02-13.json"
        alert.parent.mkdir(parents=True, exist_ok=True)
        alert.write_text("{\"passed\": false}\n", encoding="utf-8")

        eng._quality_snapshot = lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0}  # type: ignore[method-assign]
        eng._backtest_snapshot = lambda as_of: {"positive_window_ratio": 1.0, "max_drawdown": 0.10, "violations": 0}  # type: ignore[method-assign]
        eng.health_check = lambda as_of=None, require_review=True: {"status": "healthy", "checks": {}, "missing": []}  # type: ignore[method-assign]
        eng.stable_replay_check = lambda as_of, days=None: {"passed": True, "replay_days": 3, "checks": []}  # type: ignore[method-assign]

        gate = eng.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(gate["passed"])
        self.assertFalse(alert.exists())

    def test_ops_report_generation(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_eod(d)
        eng.run_review(d)
        report = eng.ops_report(as_of=d, window_days=3)
        self.assertIn("status", report)
        self.assertIn("history", report)
        self.assertIn("mode_drift", report)
        self.assertIn("reconcile_drift", report)
        self.assertIn("rollback_recommendation", report)
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_ops_report.json").exists())
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_ops_report.md").exists())

    def test_state_stability_flags_micro_capture_degraded_days(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["mode_switch_window_days"] = 5
        eng.settings.raw["validation"]["ops_state_min_samples"] = 3
        eng.settings.raw["validation"]["ops_micro_capture_degraded_days_max"] = 0
        eng.settings.raw["validation"]["ops_micro_capture_insufficient_days_max"] = 5
        eng.settings.raw["validation"]["ops_micro_capture_quality_fail_days_max"] = 5
        eng.settings.raw["validation"]["ops_micro_capture_multiplier_floor"] = 0.70
        eng.settings.raw["validation"]["ops_micro_capture_pass_ratio_min"] = 0.70
        eng.settings.raw["validation"]["ops_micro_capture_schema_ok_ratio_min"] = 0.90
        eng.settings.raw["validation"]["ops_micro_capture_time_sync_ok_ratio_min"] = 0.90
        eng.settings.raw["validation"]["ops_micro_capture_cross_source_fail_ratio_max"] = 0.35
        daily_dir = tmp_root / "output" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            day = d - timedelta(days=i)
            payload = {
                "runtime_mode": "base",
                "risk_control": {
                    "risk_multiplier": 0.95,
                    "source_confidence_score": 0.90,
                    "micro_capture_multiplier": 0.80,
                    "micro_capture_reason": "degraded" if i == 0 else "healthy",
                    "micro_capture_stats": {
                        "run_count": 5,
                        "pass_ratio": 0.95,
                        "avg_cross_source_fail_ratio": 0.10,
                        "avg_selected_schema_ok_ratio": 0.95,
                        "avg_selected_time_sync_ok_ratio": 0.95,
                    },
                },
                "mode_health": {"passed": True},
                "microstructure": {
                    "symbols_schema_fail": 0,
                    "gate_reasons": [],
                    "cross_source_audit": {"active": True, "fail_ratio": 0.0},
                },
                "time_sync": {"active": True, "pass": True, "ok_sources": 2, "available_sources": 2},
            }
            (daily_dir / f"{day.isoformat()}_mode_feedback.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

        state = eng._release_orchestrator()._state_stability_metrics(as_of=d)
        checks = state.get("checks", {}) if isinstance(state.get("checks", {}), dict) else {}
        alerts = state.get("alerts", []) if isinstance(state.get("alerts", []), list) else []
        self.assertTrue(bool(state.get("active", False)))
        self.assertFalse(bool(checks.get("micro_capture_degraded_days_ok", True)))
        self.assertIn("micro_capture_degraded_days_high", alerts)

    def test_run_slot_ops(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_review(d)
        out = eng.run_slot(as_of=d, slot="ops")
        self.assertEqual(out["slot"], "ops")
        self.assertIn("result", out)

    def test_run_daemon_dry_run(self) -> None:
        eng, tmp_root = self._make_engine()
        out = eng.run_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=1, dry_run=True)
        self.assertTrue(out["dry_run"])
        self.assertIn("slots", out)
        self.assertFalse((tmp_root / "output" / "logs" / "scheduler_state.json").exists())

    def test_architecture_audit_generates_files(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_eod(d)
        audit = eng.architecture_audit(as_of=d)
        self.assertIn("status", audit)
        self.assertIn("config", audit)
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_architecture_audit.json").exists())
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_architecture_audit.md").exists())

    def test_dependency_audit_generates_files(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        audit = eng.dependency_audit(as_of=d)
        self.assertIn("ok", audit)
        self.assertIn("edges", audit)
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_dependency_audit.json").exists())
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_dependency_audit.md").exists())

    def test_run_review_cycle(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        alert = tmp_root / "output" / "logs" / "review_loop_alert_2026-02-13.json"
        if alert.exists():
            alert.unlink()
        out = eng.run_review_cycle(as_of=d, max_rounds=0, ops_window_days=1)
        self.assertIn("review_loop", out)
        self.assertIn("gate_report", out)
        self.assertIn("ops_report", out)
        self.assertTrue(out["review_loop"].get("skipped"))
        self.assertFalse(alert.exists())

    def test_filter_model_path_bars_removes_conflicts(self) -> None:
        eng, _ = self._make_engine()
        df = pd.DataFrame(
            {
                "symbol": ["300750", "300750"],
                "ts": ["2026-02-12", "2026-02-13"],
                "close": [100.0, 101.0],
                "data_conflict_flag": [1, 0],
            }
        )
        filtered = eng._filter_model_path_bars(df)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(int(filtered["data_conflict_flag"].iloc[0]), 0)

    def test_review_until_pass_failure_writes_defect_plan(self) -> None:
        d = date(2026, 2, 13)
        eng, tmp_root = self._make_engine()
        alert = tmp_root / "output" / "logs" / "review_loop_alert_2026-02-13.json"

        eng.run_review = lambda as_of: ReviewDelta(  # type: ignore[method-assign]
            as_of=as_of,
            parameter_changes={},
            factor_weights={},
            defects=["review_fail"],
            pass_gate=False,
            notes=[],
        )
        eng.test_all = lambda: {"returncode": 1, "stdout": "", "stderr": "test_dummy ... FAIL"}  # type: ignore[method-assign]
        eng.gate_report = lambda as_of, run_tests=False, run_review_if_missing=False: {  # type: ignore[method-assign]
            "passed": False,
            "checks": {
                "review_pass_gate": False,
                "tests_ok": False,
                "health_ok": False,
                "stable_replay_ok": False,
                "data_completeness_ok": False,
                "unresolved_conflict_ok": False,
                "positive_window_ratio_ok": False,
                "max_drawdown_ok": False,
                "risk_violations_ok": False,
            },
            "metrics": {
                "max_drawdown": 0.25,
                "positive_window_ratio": 0.50,
            },
            "stable_replay": {"replay_days": 3},
        }

        out = eng.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        self.assertTrue(alert.exists())
        self.assertIn("defect_plan", out["rounds"][0])
        defect_json = Path(out["rounds"][0]["defect_plan"]["json"])
        defect_md = Path(out["rounds"][0]["defect_plan"]["md"])
        self.assertTrue(defect_json.exists())
        self.assertTrue(defect_md.exists())

    def test_review_until_pass_runs_fast_then_full_via_engine(self) -> None:
        d = date(2026, 2, 13)
        eng, tmp_root = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["review_loop_fast_test_enabled"] = True
        eng.settings.raw["validation"]["review_loop_fast_ratio"] = 0.2
        eng.settings.raw["validation"]["review_loop_fast_shard_index"] = 0
        eng.settings.raw["validation"]["review_loop_fast_shard_total"] = 1
        eng.settings.raw["validation"]["review_loop_fast_seed"] = "seed-z"
        eng.settings.raw["validation"]["review_loop_fast_then_full"] = True

        review_delta = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"

        def _fake_run_review(as_of: date) -> ReviewDelta:
            review_delta.parent.mkdir(parents=True, exist_ok=True)
            review_delta.write_text("pass_gate: true\nmode_health:\n  passed: true\n", encoding="utf-8")
            return ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
                notes=[],
            )

        calls: list[dict[str, object]] = []

        def _fake_test_all(**kwargs):
            calls.append(dict(kwargs))
            mode = "fast" if kwargs.get("fast") else "full"
            return {"returncode": 0, "summary_line": f"error=none; mode={mode}", "tests_ran": 1, "failed_tests": []}

        eng.run_review = _fake_run_review  # type: ignore[method-assign]
        eng.test_all = _fake_test_all  # type: ignore[method-assign]
        eng._quality_snapshot = lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0}  # type: ignore[method-assign]
        eng._backtest_snapshot = lambda as_of: {"positive_window_ratio": 1.0, "max_drawdown": 0.10, "violations": 0}  # type: ignore[method-assign]
        eng.health_check = lambda as_of=None, require_review=True: {"status": "healthy", "checks": {}, "missing": []}  # type: ignore[method-assign]
        eng.stable_replay_check = lambda as_of, days=None: {"passed": True, "replay_days": 3, "checks": []}  # type: ignore[method-assign]

        out = eng.review_until_pass(as_of=d, max_rounds=1)
        self.assertTrue(out["passed"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(bool(calls[0].get("fast", False)))
        self.assertFalse(bool(calls[1].get("fast", False)))
        self.assertEqual(out["rounds"][0]["tests_mode"], "fast+full")
        self.assertIn("mode=fast", out["rounds"][0]["fast_tests"]["summary_line"])
        self.assertIn("mode=full", out["rounds"][0]["full_tests"]["summary_line"])

    def test_run_review_writes_audit_fields(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_review(d)
        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        data = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        self.assertIn("previous_parameters", data)
        self.assertIn("parameter_deltas", data)
        self.assertIn("rollback_anchor", data)
        self.assertIn("factor_contrib_120d", data)

    def test_run_review_promotes_artifact_governance_baseline_with_rollback_anchor(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        eng.run_review(d1)
        eng.run_review(d2)

        p1 = tmp_root / "output" / "review" / "2026-02-13_baseline_promotion.json"
        p2 = tmp_root / "output" / "review" / "2026-02-14_baseline_promotion.json"
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())

        b1 = json.loads(p1.read_text(encoding="utf-8"))
        b2 = json.loads(p2.read_text(encoding="utf-8"))
        self.assertTrue(bool(b1.get("promoted", False)))
        self.assertTrue(bool(b2.get("promoted", False)))
        self.assertTrue(str(b1.get("snapshot_path", "")).endswith(".yaml"))
        self.assertTrue(str(b2.get("snapshot_path", "")).endswith(".yaml"))
        self.assertEqual(str(b2.get("rollback_anchor", "")), str(b1.get("snapshot_path", "")))

        active = tmp_root / "output" / "artifacts" / "baselines" / "artifact_governance" / "active_baseline.yaml"
        self.assertTrue(active.exists())
        active_payload = yaml.safe_load(active.read_text(encoding="utf-8"))
        self.assertEqual(str(active_payload.get("snapshot_path", "")), str(b2.get("snapshot_path", "")))

    def test_baseline_rollback_drill_restores_rollback_anchor_and_writes_audit(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        p1 = tmp_root / "output" / "review" / "2026-02-13_baseline_promotion.json"
        p2 = tmp_root / "output" / "review" / "2026-02-14_baseline_promotion.json"
        b1 = json.loads(p1.read_text(encoding="utf-8"))
        b2 = json.loads(p2.read_text(encoding="utf-8"))

        out = eng.baseline_rollback_drill(as_of=d3)
        self.assertTrue(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "ok")
        self.assertEqual(str(out.get("target_anchor", "")), str(b2.get("rollback_anchor", "")))

        active = tmp_root / "output" / "artifacts" / "baselines" / "artifact_governance" / "active_baseline.yaml"
        self.assertTrue(active.exists())
        active_payload = yaml.safe_load(active.read_text(encoding="utf-8"))
        self.assertEqual(str(active_payload.get("snapshot_path", "")), str(b1.get("snapshot_path", "")))

        audit_json = tmp_root / "output" / "review" / "2026-02-15_baseline_rollback_drill.json"
        audit_md = tmp_root / "output" / "review" / "2026-02-15_baseline_rollback_drill.md"
        self.assertTrue(audit_json.exists())
        self.assertTrue(audit_md.exists())
        audit_payload = json.loads(audit_json.read_text(encoding="utf-8"))
        self.assertEqual(str(audit_payload.get("active_before", "")), str(b2.get("snapshot_path", "")))
        self.assertEqual(str(audit_payload.get("active_after", "")), str(b1.get("snapshot_path", "")))

    def test_baseline_rollback_drill_dry_run_preflight_only(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        p1 = tmp_root / "output" / "review" / "2026-02-13_baseline_promotion.json"
        p2 = tmp_root / "output" / "review" / "2026-02-14_baseline_promotion.json"
        b1 = json.loads(p1.read_text(encoding="utf-8"))
        b2 = json.loads(p2.read_text(encoding="utf-8"))

        out = eng.baseline_rollback_drill(as_of=d3, dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "dry_run_ok")
        self.assertEqual(str(out.get("active_after", "")), str(b1.get("snapshot_path", "")))

        preflight = out.get("preflight", {})
        self.assertTrue(bool(preflight.get("active_baseline_exists_ok", False)))
        self.assertTrue(bool(preflight.get("active_baseline_payload_ok", False)))
        self.assertTrue(bool(preflight.get("rollback_anchor_present_ok", False)))
        self.assertTrue(bool(preflight.get("rollback_anchor_exists_ok", False)))
        self.assertTrue(bool(preflight.get("rollback_anchor_payload_ok", False)))
        self.assertTrue(bool(preflight.get("active_baseline_schema_ok", False)))
        self.assertTrue(bool(preflight.get("rollback_anchor_schema_ok", False)))
        self.assertEqual(preflight.get("errors", []), [])

        active = tmp_root / "output" / "artifacts" / "baselines" / "artifact_governance" / "active_baseline.yaml"
        self.assertTrue(active.exists())
        active_payload = yaml.safe_load(active.read_text(encoding="utf-8"))
        self.assertEqual(str(active_payload.get("snapshot_path", "")), str(b2.get("snapshot_path", "")))
        self.assertEqual(str(out.get("backup_path", "")), "")

    def test_baseline_rollback_drill_preflight_lints_anchor_payload(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        invalid_anchor = tmp_root / "output" / "artifacts" / "baselines" / "artifact_governance" / "history" / "invalid_anchor.yaml"
        invalid_anchor.parent.mkdir(parents=True, exist_ok=True)
        invalid_anchor.write_text(
            yaml.safe_dump({"source": "invalid_payload"}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, anchor=str(invalid_anchor), dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "rollback_anchor_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertTrue(bool(preflight.get("active_baseline_exists_ok", False)))
        self.assertTrue(bool(preflight.get("active_baseline_payload_ok", False)))
        self.assertTrue(bool(preflight.get("rollback_anchor_present_ok", False)))
        self.assertTrue(bool(preflight.get("rollback_anchor_exists_ok", False)))
        self.assertFalse(bool(preflight.get("rollback_anchor_payload_ok", True)))
        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        fields = {str(item.get("field", "")) for item in errors if isinstance(item, dict)}
        self.assertTrue({"as_of", "profiles", "snapshot_path"}.issubset(fields))

    def test_baseline_rollback_drill_preflight_lints_anchor_schema_types_and_formats(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        invalid_anchor = tmp_root / "output" / "artifacts" / "baselines" / "artifact_governance" / "history" / "invalid_schema_anchor.yaml"
        invalid_anchor.parent.mkdir(parents=True, exist_ok=True)
        invalid_anchor.write_text(
            yaml.safe_dump(
                {
                    "as_of": "2026/02/13",
                    "profiles": "primary",
                    "snapshot_path": 123,
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, anchor=str(invalid_anchor), dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "rollback_anchor_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertFalse(bool(preflight.get("rollback_anchor_schema_ok", True)))
        self.assertFalse(bool(preflight.get("rollback_anchor_payload_ok", True)))
        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        code_by_field = {
            str(item.get("field", "")): str(item.get("code", ""))
            for item in errors
            if isinstance(item, dict)
        }
        self.assertEqual(code_by_field.get("as_of"), "invalid_format")
        self.assertEqual(code_by_field.get("profiles"), "invalid_type")
        self.assertEqual(code_by_field.get("snapshot_path"), "invalid_type")

    def test_baseline_rollback_drill_preflight_lints_anchor_profile_subschema(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        invalid_anchor = tmp_root / "output" / "artifacts" / "baselines" / "artifact_governance" / "history" / "invalid_profile_schema_anchor.yaml"
        invalid_anchor.parent.mkdir(parents=True, exist_ok=True)
        invalid_anchor.write_text(
            yaml.safe_dump(
                {
                    "as_of": "2026-02-13",
                    "profiles": {
                        "temporal_autofix_patch": {
                            "json_glob": "*_temporal_autofix_patch.json",
                            "md_glob": "*_temporal_autofix_patch.md",
                            "checksum_index_filename": "temporal_autofix_patch_checksum_index.json",
                            "retention_days": "thirty",
                            "checksum_index_enabled": "yes",
                        }
                    },
                    "snapshot_path": str(invalid_anchor),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, anchor=str(invalid_anchor), dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "rollback_anchor_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertFalse(bool(preflight.get("rollback_anchor_schema_ok", True)))
        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        code_by_field = {
            str(item.get("field", "")): str(item.get("code", ""))
            for item in errors
            if isinstance(item, dict)
        }
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.retention_days"), "invalid_type")
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.checksum_index_enabled"), "invalid_type")

    def test_baseline_rollback_drill_preflight_lints_anchor_profile_path_fields(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        invalid_anchor = tmp_root / "output" / "artifacts" / "baselines" / "artifact_governance" / "history" / "invalid_profile_path_fields_anchor.yaml"
        invalid_anchor.parent.mkdir(parents=True, exist_ok=True)
        invalid_anchor.write_text(
            yaml.safe_dump(
                {
                    "as_of": "2026-02-13",
                    "profiles": {
                        "temporal_autofix_patch": {
                            "json_glob": "",
                            "md_glob": 123,
                            "checksum_index_filename": " ",
                            "retention_days": 30,
                            "checksum_index_enabled": True,
                        }
                    },
                    "snapshot_path": str(invalid_anchor),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, anchor=str(invalid_anchor), dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "rollback_anchor_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertFalse(bool(preflight.get("rollback_anchor_schema_ok", True)))
        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        code_by_field = {
            str(item.get("field", "")): str(item.get("code", ""))
            for item in errors
            if isinstance(item, dict)
        }
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.json_glob"), "empty_item")
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.md_glob"), "invalid_type")
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.checksum_index_filename"), "empty_item")

    def test_baseline_rollback_drill_preflight_lints_active_profile_path_fields(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        active_baseline = (
            tmp_root
            / "output"
            / "artifacts"
            / "baselines"
            / "artifact_governance"
            / "active_baseline.yaml"
        )
        self.assertTrue(active_baseline.exists())
        active_payload = yaml.safe_load(active_baseline.read_text(encoding="utf-8"))
        self.assertTrue(isinstance(active_payload, dict))
        active_payload["profiles"] = {
            "temporal_autofix_patch": {
                "json_glob": "",
                "md_glob": 123,
                "checksum_index_filename": " ",
                "retention_days": 30,
                "checksum_index_enabled": True,
            }
        }
        active_baseline.write_text(
            yaml.safe_dump(active_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "active_baseline_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertFalse(bool(preflight.get("active_baseline_schema_ok", True)))
        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        code_by_field = {
            str(item.get("field", "")): str(item.get("code", ""))
            for item in errors
            if isinstance(item, dict) and str(item.get("source", "")) == "active_baseline"
        }
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.json_glob"), "empty_item")
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.md_glob"), "invalid_type")
        self.assertEqual(code_by_field.get("profiles.temporal_autofix_patch.checksum_index_filename"), "empty_item")

    def test_baseline_rollback_drill_preflight_lints_mixed_profile_path_fields_in_stable_order(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        active_baseline = (
            tmp_root
            / "output"
            / "artifacts"
            / "baselines"
            / "artifact_governance"
            / "active_baseline.yaml"
        )
        self.assertTrue(active_baseline.exists())
        active_payload = yaml.safe_load(active_baseline.read_text(encoding="utf-8"))
        self.assertTrue(isinstance(active_payload, dict))
        active_payload["profiles"] = {
            "temporal_autofix_patch": {
                "json_glob": "",
                "md_glob": 123,
                "checksum_index_filename": " ",
                "retention_days": 30,
                "checksum_index_enabled": True,
            }
        }
        active_baseline.write_text(
            yaml.safe_dump(active_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        invalid_anchor = (
            tmp_root
            / "output"
            / "artifacts"
            / "baselines"
            / "artifact_governance"
            / "history"
            / "invalid_profile_path_fields_anchor_mixed.yaml"
        )
        invalid_anchor.parent.mkdir(parents=True, exist_ok=True)
        invalid_anchor.write_text(
            yaml.safe_dump(
                {
                    "as_of": "2026-02-13",
                    "profiles": {
                        "temporal_autofix_patch": {
                            "json_glob": "",
                            "md_glob": 123,
                            "checksum_index_filename": " ",
                            "retention_days": 30,
                            "checksum_index_enabled": True,
                        }
                    },
                    "snapshot_path": str(invalid_anchor),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, anchor=str(invalid_anchor), dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "active_baseline_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertFalse(bool(preflight.get("active_baseline_schema_ok", True)))
        self.assertFalse(bool(preflight.get("rollback_anchor_schema_ok", True)))

        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        self.assertEqual(
            [
                (
                    str(item.get("source", "")),
                    str(item.get("field", "")),
                    str(item.get("code", "")),
                )
                for item in errors
                if isinstance(item, dict)
            ],
            [
                (
                    "active_baseline",
                    "profiles.temporal_autofix_patch.json_glob",
                    "empty_item",
                ),
                (
                    "active_baseline",
                    "profiles.temporal_autofix_patch.md_glob",
                    "invalid_type",
                ),
                (
                    "active_baseline",
                    "profiles.temporal_autofix_patch.checksum_index_filename",
                    "empty_item",
                ),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.json_glob",
                    "empty_item",
                ),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.md_glob",
                    "invalid_type",
                ),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.checksum_index_filename",
                    "empty_item",
                ),
            ],
        )

    def test_baseline_rollback_drill_preflight_lints_mixed_path_and_non_path_fields_in_stable_order(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        active_baseline = (
            tmp_root
            / "output"
            / "artifacts"
            / "baselines"
            / "artifact_governance"
            / "active_baseline.yaml"
        )
        self.assertTrue(active_baseline.exists())
        active_payload = yaml.safe_load(active_baseline.read_text(encoding="utf-8"))
        self.assertTrue(isinstance(active_payload, dict))
        active_payload.update(
            {
                "as_of": "2026/02/14",
                "profiles": {
                    "temporal_autofix_patch": {
                        "json_glob": "",
                        "md_glob": 123,
                        "checksum_index_filename": " ",
                        "retention_days": 30,
                        "checksum_index_enabled": True,
                    }
                },
                "snapshot_path": 12345,
            }
        )
        active_baseline.write_text(
            yaml.safe_dump(active_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        invalid_anchor = (
            tmp_root
            / "output"
            / "artifacts"
            / "baselines"
            / "artifact_governance"
            / "history"
            / "invalid_profile_path_and_schema_fields_anchor_mixed.yaml"
        )
        invalid_anchor.parent.mkdir(parents=True, exist_ok=True)
        invalid_anchor.write_text(
            yaml.safe_dump(
                {
                    "as_of": "2026/02/13",
                    "profiles": {
                        "temporal_autofix_patch": {
                            "json_glob": "",
                            "md_glob": 123,
                            "checksum_index_filename": " ",
                            "retention_days": 30,
                            "checksum_index_enabled": True,
                        }
                    },
                    "snapshot_path": 67890,
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, anchor=str(invalid_anchor), dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "active_baseline_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertFalse(bool(preflight.get("active_baseline_schema_ok", True)))
        self.assertFalse(bool(preflight.get("rollback_anchor_schema_ok", True)))

        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        self.assertEqual(
            [
                (
                    str(item.get("source", "")),
                    str(item.get("field", "")),
                    str(item.get("code", "")),
                )
                for item in errors
                if isinstance(item, dict)
            ],
            [
                ("active_baseline", "as_of", "invalid_format"),
                (
                    "active_baseline",
                    "profiles.temporal_autofix_patch.json_glob",
                    "empty_item",
                ),
                (
                    "active_baseline",
                    "profiles.temporal_autofix_patch.md_glob",
                    "invalid_type",
                ),
                (
                    "active_baseline",
                    "profiles.temporal_autofix_patch.checksum_index_filename",
                    "empty_item",
                ),
                ("active_baseline", "snapshot_path", "invalid_type"),
                ("rollback_anchor", "as_of", "invalid_format"),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.json_glob",
                    "empty_item",
                ),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.md_glob",
                    "invalid_type",
                ),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.checksum_index_filename",
                    "empty_item",
                ),
                ("rollback_anchor", "snapshot_path", "invalid_type"),
            ],
        )

    def test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order(self) -> None:
        eng, tmp_root = self._make_engine()
        d1 = date(2026, 2, 13)
        d2 = date(2026, 2, 14)
        d3 = date(2026, 2, 15)
        eng.run_review(d1)
        eng.run_review(d2)

        active_baseline = (
            tmp_root
            / "output"
            / "artifacts"
            / "baselines"
            / "artifact_governance"
            / "active_baseline.yaml"
        )
        self.assertTrue(active_baseline.exists())
        active_payload = yaml.safe_load(active_baseline.read_text(encoding="utf-8"))
        self.assertTrue(isinstance(active_payload, dict))
        active_payload.update(
            {
                "as_of": " ",
                "profiles": {},
                "snapshot_path": " ",
            }
        )
        active_baseline.write_text(
            yaml.safe_dump(active_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        invalid_anchor = (
            tmp_root
            / "output"
            / "artifacts"
            / "baselines"
            / "artifact_governance"
            / "history"
            / "invalid_missing_required_and_profile_path_anchor_mixed.yaml"
        )
        invalid_anchor.parent.mkdir(parents=True, exist_ok=True)
        invalid_anchor.write_text(
            yaml.safe_dump(
                {
                    "as_of": "2026-02-13",
                    "profiles": {
                        "temporal_autofix_patch": {
                            "json_glob": "",
                            "md_glob": 123,
                            "checksum_index_filename": " ",
                            "retention_days": 30,
                            "checksum_index_enabled": True,
                        }
                    },
                    "snapshot_path": str(invalid_anchor),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        out = eng.baseline_rollback_drill(as_of=d3, anchor=str(invalid_anchor), dry_run=True)
        self.assertFalse(bool(out.get("executed", False)))
        self.assertEqual(str(out.get("reason", "")), "active_baseline_invalid_payload")
        preflight = out.get("preflight", {})
        self.assertFalse(bool(preflight.get("active_baseline_schema_ok", True)))
        self.assertFalse(bool(preflight.get("rollback_anchor_schema_ok", True)))

        errors = preflight.get("errors", [])
        self.assertTrue(isinstance(errors, list))
        self.assertEqual(
            [
                (
                    str(item.get("source", "")),
                    str(item.get("field", "")),
                    str(item.get("code", "")),
                )
                for item in errors
                if isinstance(item, dict)
            ],
            [
                ("active_baseline", "as_of", "missing_required_field"),
                ("active_baseline", "profiles", "missing_required_field"),
                ("active_baseline", "snapshot_path", "missing_required_field"),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.json_glob",
                    "empty_item",
                ),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.md_glob",
                    "invalid_type",
                ),
                (
                    "rollback_anchor",
                    "profiles.temporal_autofix_patch.checksum_index_filename",
                    "empty_item",
                ),
            ],
        )

    def test_run_review_writes_slot_regime_tuning_artifact(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_slot_regime_tune_enabled": True,
                "ops_slot_regime_tune_window_days": 60,
                "ops_slot_regime_tune_min_days": 1,
                "ops_slot_regime_tune_step": 0.5,
                "ops_slot_regime_tune_buffer": 0.1,
                "ops_slot_regime_tune_floor": 0.1,
                "ops_slot_regime_tune_ceiling": 0.9,
            }
        )

        manifest_dir = tmp_root / "output" / "artifacts" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            quality_passed = i == 0
            (manifest_dir / f"eod_{day.isoformat()}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": quality_passed},
                        "metrics": {"regime": "震荡", "risk_multiplier": 1.0},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.12,
            annual_return=0.06,
            max_drawdown=0.10,
            win_rate=0.52,
            profit_factor=1.2,
            expectancy=0.01,
            trades=120,
            violations=0,
            positive_window_ratio=0.82,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }

        eng.run_review(d)
        tuning_path = tmp_root / "output" / "artifacts" / "slot_regime_thresholds_live.yaml"
        self.assertTrue(tuning_path.exists())
        tuning = yaml.safe_load(tuning_path.read_text(encoding="utf-8"))
        self.assertIn("ops_slot_eod_quality_anomaly_ratio_max_by_regime", tuning)
        self.assertIn("range", tuning["ops_slot_eod_quality_anomaly_ratio_max_by_regime"])
        self.assertIn("changed", tuning)

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        delta = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        self.assertIn("slot_regime_tuning", delta)
        self.assertTrue(bool(delta["slot_regime_tuning"].get("applied", False)))

    def test_run_review_skips_slot_regime_tuning_when_missing_ratio_high(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_slot_regime_tune_enabled": True,
                "ops_slot_regime_tune_window_days": 60,
                "ops_slot_regime_tune_min_days": 1,
                "ops_slot_regime_tune_step": 0.5,
                "ops_slot_regime_tune_buffer": 0.1,
                "ops_slot_regime_tune_floor": 0.1,
                "ops_slot_regime_tune_ceiling": 0.9,
                "ops_slot_regime_tune_missing_ratio_hard_cap": 0.60,
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 1,
            }
        )

        manifest_dir = tmp_root / "output" / "artifacts" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            (manifest_dir / f"eod_{day.isoformat()}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {"regime": "震荡", "risk_multiplier": 1.0},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.12,
            annual_return=0.06,
            max_drawdown=0.10,
            win_rate=0.52,
            profit_factor=1.2,
            expectancy=0.01,
            trades=120,
            violations=0,
            positive_window_ratio=0.82,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }

        eng.run_review(d)
        tuning_path = tmp_root / "output" / "artifacts" / "slot_regime_thresholds_live.yaml"
        self.assertTrue(tuning_path.exists())
        tuning = yaml.safe_load(tuning_path.read_text(encoding="utf-8"))
        self.assertTrue(bool(tuning.get("skipped", False)))
        self.assertEqual(str(tuning.get("skip_reason", "")), "slot_missing_ratio_high")
        self.assertGreater(float(tuning.get("slot_missing_ratio", 0.0)), 0.60)

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        delta = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        slot_tuning = delta.get("slot_regime_tuning", {})
        self.assertFalse(bool(slot_tuning.get("applied", True)))
        self.assertTrue(bool(slot_tuning.get("skipped", False)))
        self.assertEqual(str(slot_tuning.get("reason", "")), "slot_missing_ratio_high")

    def test_run_review_merges_strategy_lab_candidate(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        manifest_dir = tmp_root / "output" / "artifacts" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        summary_dir = tmp_root / "output" / "research" / "strategy_lab_local"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "cutoff_date": "2026-02-12",
                    "cutoff_ts": "2026-02-12T23:59:59",
                    "bar_max_ts": "2026-02-12T15:00:00",
                    "news_max_ts": "2026-02-12T23:59:59",
                    "report_max_ts": "2026-02-12T23:59:59",
                    "data_fetch_stats": {"strict_cutoff_enforced": True},
                    "best_candidate": {
                        "name": "trend_convex_01",
                        "accepted": True,
                        "params": {
                            "signal_confidence_min": 55.0,
                            "convexity_min": 2.4,
                            "hold_days": 8,
                            "max_daily_trades": 3,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_path = manifest_dir / "strategy_lab_20260213_000000.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "artifacts": {"summary": str(summary_path)},
                    "checks": {"strict_cutoff_enforced": True},
                    "metadata": {"cutoff_date": "2026-02-12"},
                }
            ),
            encoding="utf-8",
        )

        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.10,
            annual_return=0.05,
            max_drawdown=0.12,
            win_rate=0.48,
            profit_factor=1.4,
            expectancy=0.01,
            trades=100,
            violations=0,
            positive_window_ratio=0.80,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }

        review = eng.run_review(d)
        self.assertIn("convexity_min", review.parameter_changes)
        self.assertIn("hold_days", review.parameter_changes)
        self.assertIn("max_daily_trades", review.parameter_changes)
        self.assertTrue(any("strategy-lab" in str(v) for v in review.change_reasons.values()))

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        self.assertIn("strategy_lab_candidate", payload)
        self.assertEqual(
            payload.get("strategy_lab_candidate", {}).get("candidate", {}).get("name", ""),
            "trend_convex_01",
        )

    def test_load_latest_strategy_candidate_rejects_temporal_leak(self) -> None:
        eng, tmp_root = self._make_engine()
        as_of = date(2026, 2, 13)
        manifest_dir = tmp_root / "output" / "artifacts" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        summary_dir = tmp_root / "output" / "research" / "strategy_lab_local"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "cutoff_date": "2026-02-12",
                    "cutoff_ts": "2026-02-12T23:59:59",
                    "bar_max_ts": "2026-02-13T09:31:00",
                    "news_max_ts": "2026-02-12T23:59:59",
                    "report_max_ts": "2026-02-12T23:59:59",
                    "data_fetch_stats": {"strict_cutoff_enforced": True},
                    "best_candidate": {
                        "name": "should_be_rejected",
                        "accepted": True,
                        "params": {
                            "signal_confidence_min": 55.0,
                            "convexity_min": 2.4,
                            "hold_days": 8,
                            "max_daily_trades": 3,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (manifest_dir / "strategy_lab_20260213_000000.json").write_text(
            json.dumps(
                {
                    "artifacts": {"summary": str(summary_path)},
                    "checks": {"strict_cutoff_enforced": True},
                    "metadata": {"cutoff_date": "2026-02-12"},
                }
            ),
            encoding="utf-8",
        )

        payload = eng._load_latest_strategy_candidate(as_of)  # type: ignore[attr-defined]
        self.assertEqual(payload, {})

    def test_run_backtest_uses_live_parameter_overrides(self) -> None:
        import lie_engine.engine as eng_mod

        eng, tmp_root = self._make_engine()
        live = tmp_root / "output" / "artifacts" / "params_live.yaml"
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(
            yaml.safe_dump(
                {
                    "signal_confidence_min": 52.0,
                    "convexity_min": 2.2,
                    "hold_days": 9,
                    "max_daily_trades": 4,
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        eng._run_ingestion_range = lambda start, end, symbols: (pd.DataFrame(), None)  # type: ignore[method-assign]
        captured: dict[str, object] = {}
        original = eng_mod.run_walk_forward_backtest

        def _fake_run_walk_forward_backtest(**kwargs):
            captured["cfg"] = kwargs.get("cfg_template")
            return BacktestResult(
                start=kwargs["start"],
                end=kwargs["end"],
                total_return=0.0,
                annual_return=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                trades=0,
                violations=0,
                positive_window_ratio=1.0,
                equity_curve=[],
                by_asset={},
            )

        eng_mod.run_walk_forward_backtest = _fake_run_walk_forward_backtest  # type: ignore[assignment]
        try:
            eng.run_backtest(start=date(2026, 1, 1), end=date(2026, 2, 13))
        finally:
            eng_mod.run_walk_forward_backtest = original  # type: ignore[assignment]

        cfg = captured["cfg"]
        self.assertEqual(int(getattr(cfg, "hold_days", 0)), 9)
        self.assertEqual(int(getattr(cfg, "max_daily_trades", 0)), 4)
        self.assertAlmostEqual(float(getattr(cfg, "signal_confidence_min", 0.0)), 52.0, places=6)
        self.assertAlmostEqual(float(getattr(cfg, "convexity_min", 0.0)), 2.2, places=6)

    def test_run_mode_stress_matrix_outputs_artifacts(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        out = eng.run_mode_stress_matrix(
            as_of=d,
            modes=["ultra_short", "swing", "long"],
            windows=[{"name": "quick_window", "start": "2026-01-01", "end": "2026-01-31"}],
        )
        self.assertIn("paths", out)
        self.assertIn("manifest", out)
        json_path = Path(str(out.get("paths", {}).get("json", "")))
        md_path = Path(str(out.get("paths", {}).get("md", "")))
        manifest_path = Path(str(out.get("manifest", "")))
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertTrue(manifest_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(int(payload.get("window_count", 0)), 1)
        self.assertEqual(int(payload.get("mode_count", 0)), 3)
        self.assertEqual(len(payload.get("mode_summary", [])), 3)
        self.assertEqual(str(payload.get("windows", [])[0].get("name", "")), "quick_window")
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_mode_stress_matrix.md").exists())
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_mode_stress_matrix.json").exists())

    def test_run_review_autoruns_strategy_lab_when_missing_candidate(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["review_autorun_strategy_lab_if_missing"] = True

        eng.run_backtest = lambda start, end: BacktestResult(  # type: ignore[method-assign]
            start=start,
            end=end,
            total_return=0.10,
            annual_return=0.05,
            max_drawdown=0.12,
            win_rate=0.48,
            profit_factor=1.4,
            expectancy=0.01,
            trades=100,
            violations=0,
            positive_window_ratio=0.80,
            equity_curve=[],
            by_asset={},
        )
        eng._estimate_factor_contrib_120d = lambda as_of: {  # type: ignore[method-assign]
            "macro": 0.2,
            "industry": 0.2,
            "news": 0.2,
            "sentiment": 0.2,
            "fundamental": 0.1,
            "technical": 0.1,
        }

        state = {"n": 0}

        def _fake_load_candidate(as_of: date) -> dict[str, object]:
            if state["n"] == 0:
                state["n"] += 1
                return {}
            return {
                "cutoff_date": "2026-02-12",
                "manifest_path": "/tmp/mock_strategy_manifest.json",
                "candidate": {
                    "name": "trend_convex_01",
                    "accepted": True,
                    "params": {
                        "signal_confidence_min": 55.0,
                        "convexity_min": 2.4,
                        "hold_days": 8,
                        "max_daily_trades": 3,
                    },
                },
            }

        called: dict[str, object] = {}

        def _fake_run_strategy_lab(**kwargs):
            called["kwargs"] = kwargs
            return {
                "manifest": "/tmp/mock_strategy_manifest.json",
                "candidates": [{"name": "trend_convex_01"}],
            }

        eng._load_latest_strategy_candidate = _fake_load_candidate  # type: ignore[method-assign]
        eng.run_strategy_lab = _fake_run_strategy_lab  # type: ignore[method-assign]

        review = eng.run_review(d)
        self.assertIn("kwargs", called)
        self.assertTrue(any("strategy_lab_autorun=" in n for n in review.notes))
        self.assertIn("hold_days", review.parameter_changes)

    def test_engine_provider_profile(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp_root = Path(td.name)

        cfg_data = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
        cfg_data["paths"] = {"output": "output", "sqlite": "output/artifacts/lie_engine.db"}
        cfg_data["data"] = {"provider_profile": "opensource_primary"}
        cfg_path = tmp_root / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")

        eng = LieEngine(config_path=cfg_path)
        self.assertEqual(len(eng.providers), 1)
        self.assertEqual(getattr(eng.providers[0], "name", ""), "open_source_primary")

    def test_engine_provider_profile_binance_public(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp_root = Path(td.name)

        cfg_data = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
        cfg_data["paths"] = {"output": "output", "sqlite": "output/artifacts/lie_engine.db"}
        cfg_data["data"] = {"provider_profile": "binance_spot_public"}
        cfg_data["universe"] = {
            "core": [
                {"symbol": "BTCUSDT", "asset_class": "crypto", "theme": "crypto"},
            ],
            "max_dynamic_additions": 0,
        }
        cfg_path = tmp_root / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")

        eng = LieEngine(config_path=cfg_path)
        self.assertEqual(len(eng.providers), 1)
        self.assertEqual(getattr(eng.providers[0], "name", ""), "binance_spot_public")

    def test_engine_provider_profile_dual_binance_bybit_public(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp_root = Path(td.name)

        cfg_data = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
        cfg_data["paths"] = {"output": "output", "sqlite": "output/artifacts/lie_engine.db"}
        cfg_data["data"] = {"provider_profile": "dual_binance_bybit_public"}
        cfg_data["universe"] = {
            "core": [
                {"symbol": "BTCUSDT", "asset_class": "crypto", "theme": "crypto"},
            ],
            "max_dynamic_additions": 0,
        }
        cfg_path = tmp_root / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")

        eng = LieEngine(config_path=cfg_path)
        self.assertEqual(len(eng.providers), 2)
        self.assertEqual(getattr(eng.providers[0], "name", ""), "binance_spot_public")
        self.assertEqual(getattr(eng.providers[1], "name", ""), "bybit_spot_public")

    def test_collect_micro_factor_map_uses_configured_symbols(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["microstructure_signal_enabled"] = True
        eng.settings.raw["validation"]["microstructure_symbols"] = ["BTCUSDT"]
        eng.settings.raw["validation"]["micro_min_trade_count"] = 1
        eng.settings.raw["validation"]["micro_time_sync_hard_fuse_enabled"] = True
        eng.settings.raw["validation"]["micro_time_sync_max_offset_ms"] = 5
        eng.settings.raw["validation"]["micro_time_sync_max_rtt_ms"] = 120

        class _FakeMicroProvider:
            name = "fake_micro"

            def fetch_l2(self, symbol, start_ts, end_ts, depth=20):  # type: ignore[no-untyped-def]
                if str(symbol) != "BTCUSDT":
                    return pd.DataFrame()
                return pd.DataFrame(
                    [
                        {
                            "exchange": "fake",
                            "symbol": "BTCUSDT",
                            "event_ts_ms": 1700000000000,
                            "recv_ts_ms": 1700000000001,
                            "seq": 10,
                            "prev_seq": 9,
                            "bids": [[100.0, 2.0]],
                            "asks": [[100.2, 1.5]],
                            "source": "fake_micro",
                        }
                    ]
                )

            def fetch_trades(self, symbol, start_ts, end_ts, limit=2000):  # type: ignore[no-untyped-def]
                if str(symbol) != "BTCUSDT":
                    return pd.DataFrame()
                rows = []
                for i in range(12):
                    rows.append(
                        {
                            "exchange": "fake",
                            "symbol": "BTCUSDT",
                            "trade_id": str(i),
                            "event_ts_ms": 1700000000000 + i * 200,
                            "recv_ts_ms": 1700000000000 + i * 200 + 1,
                            "price": 100.0 + i * 0.01,
                            "qty": 0.1 + 0.01 * i,
                            "side": "BUY" if i % 2 == 0 else "SELL",
                            "source": "fake_micro",
                        }
                    )
                return pd.DataFrame(rows)

            def fetch_time_sync_sample(self):  # type: ignore[no-untyped-def]
                return {
                    "source": "fake_micro",
                    "server_ts_ms": 1700000000000,
                    "local_send_ts_ms": 1699999999998,
                    "local_recv_ts_ms": 1700000000002,
                    "local_mid_ts_ms": 1700000000000,
                    "rtt_ms": 4,
                    "offset_ms": 0,
                    "offset_abs_ms": 0,
                }

        eng.providers = [_FakeMicroProvider()]  # type: ignore[assignment]
        out = eng._collect_micro_factor_map(as_of=date(2026, 2, 13), symbols=["300750"])
        self.assertIn("BTCUSDT", out)
        self.assertTrue(bool(out["BTCUSDT"].get("has_data", False)))
        self.assertTrue(bool(out["BTCUSDT"].get("time_sync_available", False)))
        self.assertTrue(bool(out["BTCUSDT"].get("time_sync_ok", False)))
        self.assertNotIn("300750", out)

    def test_collect_micro_factor_map_clamps_today_window_to_now(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["microstructure_signal_enabled"] = True
        eng.settings.raw["validation"]["microstructure_symbols"] = ["BTCUSDT"]
        eng.settings.raw["validation"]["micro_min_trade_count"] = 1

        seen_end_ts: list[datetime] = []

        class _WindowProbeProvider:
            name = "window_probe"

            def fetch_l2(self, symbol, start_ts, end_ts, depth=20):  # type: ignore[no-untyped-def]
                seen_end_ts.append(end_ts)
                return pd.DataFrame(
                    [
                        {
                            "exchange": "probe",
                            "symbol": "BTCUSDT",
                            "event_ts_ms": 1700000000000,
                            "recv_ts_ms": 1700000000001,
                            "seq": 10,
                            "prev_seq": 9,
                            "bids": [[100.0, 1.0]],
                            "asks": [[100.1, 1.0]],
                            "source": "window_probe",
                        }
                    ]
                )

            def fetch_trades(self, symbol, start_ts, end_ts, limit=2000):  # type: ignore[no-untyped-def]
                rows = []
                for i in range(5):
                    rows.append(
                        {
                            "exchange": "probe",
                            "symbol": "BTCUSDT",
                            "trade_id": str(i),
                            "event_ts_ms": 1700000000000 + i * 100,
                            "recv_ts_ms": 1700000000000 + i * 100 + 1,
                            "price": 100.0,
                            "qty": 0.1,
                            "side": "BUY",
                            "source": "window_probe",
                        }
                    )
                return pd.DataFrame(rows)

        eng.providers = [_WindowProbeProvider()]  # type: ignore[assignment]
        eng._collect_micro_factor_map(as_of=date.today(), symbols=["BTCUSDT"])
        self.assertTrue(bool(seen_end_ts))
        now_utc = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        self.assertLessEqual(seen_end_ts[0], now_utc + timedelta(seconds=2))

    def test_collect_cross_source_audit_builds_missing_providers_from_profile(self) -> None:
        import lie_engine.engine as eng_mod

        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["micro_cross_source_audit_enabled"] = True
        eng.settings.raw["validation"]["micro_cross_source_primary"] = "fake_primary"
        eng.settings.raw["validation"]["micro_cross_source_secondary"] = "fake_secondary"
        eng.settings.raw["validation"]["micro_cross_source_build_missing_provider"] = True
        eng.settings.raw["validation"]["micro_cross_source_symbols"] = ["BTCUSDT"]
        eng.settings.raw["validation"]["micro_cross_source_audit_symbol_cap"] = 1
        eng.settings.raw["validation"]["micro_cross_source_trade_limit"] = 200
        eng.settings.raw["validation"]["micro_cross_source_window_ms"] = 200
        eng.settings.raw["validation"]["micro_cross_source_tolerance_ms"] = 80
        eng.settings.raw["validation"]["micro_continuous_gap_ms"] = 2500

        class _FakeTradeProvider:
            def __init__(self, name: str, skew_ms: int = 0) -> None:
                self.name = name
                self.skew_ms = int(skew_ms)

            def fetch_trades(self, symbol, start_ts, end_ts, limit=2000):  # type: ignore[no-untyped-def]
                rows = []
                base = 1700000000000
                for i in range(20):
                    event_ts = base + i * 200 + self.skew_ms
                    rows.append(
                        {
                            "exchange": self.name,
                            "symbol": str(symbol),
                            "trade_id": f"{self.name}-{i}",
                            "event_ts_ms": event_ts,
                            "recv_ts_ms": event_ts + 1,
                            "price": 100.0 + i * 0.02,
                            "qty": 0.2,
                            "side": "BUY" if i % 2 == 0 else "SELL",
                            "source": self.name,
                        }
                    )
                return pd.DataFrame(rows)

        original = eng_mod.build_provider_stack

        def _fake_build_provider_stack(profile):  # type: ignore[no-untyped-def]
            p = str(profile).strip().lower()
            if p == "fake_primary":
                return [_FakeTradeProvider(name="fake_primary", skew_ms=0)]
            if p == "fake_secondary":
                return [_FakeTradeProvider(name="fake_secondary", skew_ms=0)]
            raise ValueError(f"unsupported profile: {profile}")

        eng_mod.build_provider_stack = _fake_build_provider_stack  # type: ignore[assignment]
        try:
            out = eng._collect_cross_source_audit(as_of=date(2026, 2, 13), symbols=["300750"])
        finally:
            eng_mod.build_provider_stack = original  # type: ignore[assignment]

        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        self.assertEqual(int(out.get("symbols_audited", 0)), 1)
        self.assertLessEqual(float(out.get("fail_ratio", 1.0)), 0.0)

    def test_collect_cross_source_audit_marks_insufficient_samples_without_fail(self) -> None:
        import lie_engine.engine as eng_mod

        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["micro_cross_source_audit_enabled"] = True
        eng.settings.raw["validation"]["micro_cross_source_primary"] = "fake_primary"
        eng.settings.raw["validation"]["micro_cross_source_secondary"] = "fake_secondary"
        eng.settings.raw["validation"]["micro_cross_source_build_missing_provider"] = True
        eng.settings.raw["validation"]["micro_cross_source_symbols"] = ["BTCUSDT"]
        eng.settings.raw["validation"]["micro_cross_source_audit_symbol_cap"] = 1
        eng.settings.raw["validation"]["micro_cross_source_min_rows_per_side"] = 2
        eng.settings.raw["validation"]["micro_cross_source_trade_limit"] = 100

        class _EmptyTradeProvider:
            def __init__(self, name: str) -> None:
                self.name = name

            def fetch_trades(self, symbol, start_ts, end_ts, limit=2000):  # type: ignore[no-untyped-def]
                return pd.DataFrame()

        original = eng_mod.build_provider_stack

        def _fake_build_provider_stack(profile):  # type: ignore[no-untyped-def]
            p = str(profile).strip().lower()
            if p == "fake_primary":
                return [_EmptyTradeProvider(name="fake_primary")]
            if p == "fake_secondary":
                return [_EmptyTradeProvider(name="fake_secondary")]
            raise ValueError(f"unsupported profile: {profile}")

        eng_mod.build_provider_stack = _fake_build_provider_stack  # type: ignore[assignment]
        try:
            out = eng._collect_cross_source_audit(as_of=date(2026, 2, 13), symbols=["300750"])
        finally:
            eng_mod.build_provider_stack = original  # type: ignore[assignment]

        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "insufficient_samples")
        self.assertEqual(int(out.get("symbols_selected", 0)), 1)
        self.assertEqual(int(out.get("symbols_audited", 0)), 0)
        self.assertEqual(int(out.get("symbols_insufficient", 0)), 1)
        self.assertEqual(float(out.get("fail_ratio", 1.0)), 0.0)
        self.assertEqual(out.get("fail_symbols", []), [])

    def test_collect_cross_source_audit_uses_frequency_normalized_gap_limit(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["micro_cross_source_audit_enabled"] = True
        eng.settings.raw["validation"]["micro_cross_source_primary"] = "fake_primary"
        eng.settings.raw["validation"]["micro_cross_source_secondary"] = "fake_secondary"
        eng.settings.raw["validation"]["micro_cross_source_symbols"] = ["BTCUSDT"]
        eng.settings.raw["validation"]["micro_cross_source_audit_symbol_cap"] = 1
        eng.settings.raw["validation"]["micro_cross_source_align_seconds"] = 6
        eng.settings.raw["validation"]["micro_cross_source_window_ms"] = 200
        eng.settings.raw["validation"]["micro_cross_source_trade_limit"] = 100
        eng.settings.raw["validation"]["micro_cross_source_tolerance_ms"] = 80
        eng.settings.raw["validation"]["micro_continuous_gap_ms"] = 1000
        eng.settings.raw["validation"]["micro_cross_source_adaptive_gap_enabled"] = True
        eng.settings.raw["validation"]["micro_cross_source_gap_freq_multiplier"] = 2.0
        eng.settings.raw["validation"]["micro_cross_source_gap_hist_window_days"] = 7
        eng.settings.raw["validation"]["micro_cross_source_gap_hist_quantile"] = 0.9
        eng.settings.raw["validation"]["micro_cross_source_gap_hist_multiplier"] = 1.1
        eng.settings.raw["validation"]["micro_cross_source_gap_limit_cap_ms"] = 10000
        eng.settings.raw["validation"]["micro_cross_source_min_rows_per_side"] = 1

        class _RateSkewProvider:
            def __init__(self, name: str, step_ms: int) -> None:
                self.name = name
                self.step_ms = int(step_ms)

            def fetch_trades(self, symbol, start_ts, end_ts, limit=2000):  # type: ignore[no-untyped-def]
                base = 1700000000000
                rows = []
                for i in range(30):
                    ts = base + i * self.step_ms
                    rows.append(
                        {
                            "exchange": self.name,
                            "symbol": str(symbol),
                            "trade_id": f"{self.name}-{i}",
                            "event_ts_ms": ts,
                            "recv_ts_ms": ts + 1,
                            "price": 100.0,
                            "qty": 0.1,
                            "side": "BUY" if i % 2 == 0 else "SELL",
                            "source": self.name,
                        }
                    )
                return pd.DataFrame(rows)

        eng.providers = [
            _RateSkewProvider(name="fake_primary", step_ms=200),
            _RateSkewProvider(name="fake_secondary", step_ms=2000),
        ]  # type: ignore[assignment]
        out = eng._collect_cross_source_audit(as_of=date(2026, 2, 13), symbols=["300750"])
        rows = out.get("rows", [])
        self.assertTrue(isinstance(rows, list) and rows)
        row = rows[0]
        self.assertGreater(int(row.get("effective_gap_limit_ms", 0)), 1000)
        self.assertGreaterEqual(int(row.get("gap_limit_from_freq_ms", 0)), 4000)
        self.assertTrue(bool(row.get("gap_ok", False)))
        self.assertEqual(str(row.get("row_status", "")), "audited_pass")
        self.assertEqual(str(out.get("status", "")), "ok")
        self.assertEqual(float(out.get("fail_ratio", 1.0)), 0.0)

    def test_execution_risk_control_applies_cross_source_stress_multiplier(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["execution_min_risk_multiplier"] = 0.2
        eng.settings.raw["validation"]["execution_cross_source_stress_risk_multiplier"] = 0.7
        eng.settings.raw["validation"]["execution_cross_source_stress_full_scale"] = 1.0
        out = eng._execution_risk_control(
            source_confidence_score=0.95,
            mode_health={"passed": True, "active": True},
            market_factor_state={"crypto_stress": 0.0, "cross_source_stress": 1.0},
        )
        self.assertAlmostEqual(float(out.get("cross_source_multiplier", 0.0)), 0.7, places=6)
        self.assertAlmostEqual(float(out.get("risk_multiplier", 0.0)), 0.7, places=6)

    def test_execution_risk_control_applies_micro_capture_degraded_multiplier(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["execution_min_risk_multiplier"] = 0.2
        eng.settings.raw["validation"]["execution_micro_capture_risk_enabled"] = True
        eng.settings.raw["validation"]["execution_micro_capture_risk_multiplier"] = 0.6
        eng.settings.raw["validation"]["execution_micro_capture_insufficient_sample_risk_multiplier"] = 0.9
        eng.settings.raw["validation"]["execution_micro_capture_lookback_days"] = 7
        eng.settings.raw["validation"]["execution_micro_capture_min_runs"] = 2
        eng.settings.raw["validation"]["execution_micro_capture_pass_ratio_min"] = 0.7
        eng.settings.raw["validation"]["execution_micro_capture_schema_ok_ratio_min"] = 0.9
        eng.settings.raw["validation"]["execution_micro_capture_time_sync_ok_ratio_min"] = 0.9
        eng.settings.raw["validation"]["execution_micro_capture_cross_source_fail_ratio_max"] = 0.35
        rows = pd.DataFrame(
            [
                {
                    "as_of": "2026-02-12",
                    "pass": 0,
                    "cross_source_fail_ratio": 0.9,
                    "selected_schema_ok_ratio": 0.2,
                    "selected_time_sync_ok_ratio": 0.3,
                },
                {
                    "as_of": "2026-02-13",
                    "pass": 0,
                    "cross_source_fail_ratio": 0.95,
                    "selected_schema_ok_ratio": 0.1,
                    "selected_time_sync_ok_ratio": 0.2,
                },
            ]
        )
        eng.ctx.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(eng.ctx.sqlite_path)) as conn:
            rows.to_sql("micro_capture_runs", conn, if_exists="replace", index=False)
        out = eng._execution_risk_control(
            source_confidence_score=0.95,
            mode_health={"passed": True, "active": True},
            market_factor_state={"crypto_stress": 0.0, "cross_source_stress": 0.0},
            as_of=date(2026, 2, 13),
        )
        self.assertEqual(str(out.get("micro_capture_reason", "")), "degraded")
        self.assertAlmostEqual(float(out.get("micro_capture_multiplier", 0.0)), 0.6, places=6)
        self.assertAlmostEqual(float(out.get("risk_multiplier", 0.0)), 0.6, places=6)
        stats = out.get("micro_capture_stats", {})
        self.assertEqual(int(stats.get("run_count", -1)), 2)

    def test_execution_risk_control_applies_micro_capture_insufficient_sample_multiplier(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["execution_min_risk_multiplier"] = 0.2
        eng.settings.raw["validation"]["execution_micro_capture_risk_enabled"] = True
        eng.settings.raw["validation"]["execution_micro_capture_risk_multiplier"] = 0.6
        eng.settings.raw["validation"]["execution_micro_capture_insufficient_sample_risk_multiplier"] = 0.85
        eng.settings.raw["validation"]["execution_micro_capture_lookback_days"] = 7
        eng.settings.raw["validation"]["execution_micro_capture_min_runs"] = 3
        rows = pd.DataFrame(
            [
                {
                    "as_of": "2026-02-13",
                    "pass": 1,
                    "cross_source_fail_ratio": 0.0,
                    "selected_schema_ok_ratio": 1.0,
                    "selected_time_sync_ok_ratio": 1.0,
                }
            ]
        )
        eng.ctx.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(eng.ctx.sqlite_path)) as conn:
            rows.to_sql("micro_capture_runs", conn, if_exists="replace", index=False)
        out = eng._execution_risk_control(
            source_confidence_score=0.95,
            mode_health={"passed": True, "active": True},
            market_factor_state={"crypto_stress": 0.0, "cross_source_stress": 0.0},
            as_of=date(2026, 2, 13),
        )
        self.assertEqual(str(out.get("micro_capture_reason", "")), "insufficient_samples")
        self.assertAlmostEqual(float(out.get("micro_capture_multiplier", 0.0)), 0.85, places=6)
        self.assertAlmostEqual(float(out.get("risk_multiplier", 0.0)), 0.85, places=6)

    def test_microstructure_gate_reasons_on_schema_fail(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["micro_schema_hard_fuse_enabled"] = True
        eng.settings.raw["validation"]["micro_schema_max_fail_symbols"] = 0
        reasons = eng._microstructure_gate_reasons(
            symbols=["BTCUSDT"],
            micro_factor_map={"BTCUSDT": {"schema_ok": False, "schema_issues": ["l2_missing:event_ts_ms"]}},
            cross_source_audit={"enabled": False, "active": False},
        )
        self.assertTrue(any("字段完整性熔断" in str(x) for x in reasons))

    def test_microstructure_gate_reasons_on_cross_source_fail_ratio(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["micro_cross_source_hard_fuse_enabled"] = True
        eng.settings.raw["validation"]["micro_cross_source_min_samples"] = 1
        eng.settings.raw["validation"]["micro_cross_source_max_fail_ratio"] = 0.0
        reasons = eng._microstructure_gate_reasons(
            symbols=["BTCUSDT"],
            micro_factor_map={},
            cross_source_audit={
                "enabled": True,
                "active": True,
                "symbols_audited": 1,
                "fail_ratio": 1.0,
                "fail_symbols": ["BTCUSDT"],
            },
        )
        self.assertTrue(any("跨源对齐熔断" in str(x) for x in reasons))

    def test_microstructure_gate_reasons_on_time_sync_breach(self) -> None:
        eng, _ = self._make_engine()
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"]["micro_time_sync_hard_fuse_enabled"] = True
        eng.settings.raw["validation"]["micro_time_sync_min_samples"] = 1
        eng.settings.raw["validation"]["micro_time_sync_max_offset_ms"] = 5
        eng.settings.raw["validation"]["micro_time_sync_max_rtt_ms"] = 120
        reasons = eng._microstructure_gate_reasons(
            symbols=["BTCUSDT"],
            micro_factor_map={
                "BTCUSDT": {
                    "has_data": True,
                    "time_sync_available": True,
                    "time_sync_offset_ms": 1000,
                    "time_sync_rtt_ms": 200,
                }
            },
            cross_source_audit={"enabled": False, "active": False},
        )
        self.assertTrue(any("时钟同步熔断" in str(x) for x in reasons))

    def test_run_strategy_lab_manifest(self) -> None:
        import lie_engine.engine as eng_mod

        eng, tmp_root = self._make_engine()
        run_dir = tmp_root / "output" / "research" / "strategy_lab_mock"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
        (run_dir / "best_strategy.yaml").write_text("name: mock\n", encoding="utf-8")

        class _Summary:
            output_dir = str(run_dir)

            def to_dict(self) -> dict[str, object]:
                return {
                    "universe_count": 4,
                    "bars_rows": 100,
                    "candidates": [{"name": "mock"}],
                    "best_candidate": {"name": "mock"},
                    "data_fetch_stats": {"strict_cutoff_enforced": True},
                    "cutoff_date": "2026-02-13",
                    "cutoff_ts": "2026-02-13T23:59:59",
                    "bar_max_ts": "2026-02-13T15:00:00",
                    "news_max_ts": "2026-02-13T23:59:59",
                    "report_max_ts": "2026-02-13T23:59:59",
                    "review_days": 3,
                }

        original = eng_mod.run_strategy_lab_pipeline
        eng_mod.run_strategy_lab_pipeline = lambda **kwargs: _Summary()  # type: ignore[assignment]
        try:
            out = eng.run_strategy_lab(start=date(2026, 1, 1), end=date(2026, 2, 13), candidate_count=2)
            self.assertIn("manifest", out)
            manifest_path = Path(str(out["manifest"]))
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            self.assertEqual(str(metadata.get("cutoff_ts", "")), "2026-02-13T23:59:59")
            self.assertEqual(str(metadata.get("bar_max_ts", "")), "2026-02-13T15:00:00")
            self.assertEqual(str(metadata.get("news_max_ts", "")), "2026-02-13T23:59:59")
            self.assertIn("term_registry_version", metadata)
            self.assertIn("term_registry_checksum_sha256", metadata)
            self.assertIn("term_registry_atoms_total", metadata)
            self.assertIn("term_registry_l2_atoms", metadata)
        finally:
            eng_mod.run_strategy_lab_pipeline = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
