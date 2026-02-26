from __future__ import annotations

from contextlib import closing
import hashlib
import sys
from pathlib import Path
import unittest
from datetime import date, datetime, timedelta
import tempfile
import json
import sqlite3

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

    def test_run_eod_persists_signals_with_date_column_for_retention(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_eod(d)
        db_path = tmp_root / "output" / "artifacts" / "lie_engine.db"
        with closing(sqlite3.connect(db_path)) as conn:
            cols = pd.read_sql_query("PRAGMA table_info(signals)", conn)["name"].astype(str).tolist()
            self.assertIn("date", cols)

    def test_run_ingestion_range_does_not_persist_by_default(self) -> None:
        eng, tmp_root = self._make_engine()
        bars, _ = eng._run_ingestion_range(
            start=date(2026, 1, 2),
            end=date(2026, 1, 31),
            symbols=eng._core_symbols(),
            persist=False,
        )
        self.assertFalse(bars.empty)
        db_path = tmp_root / "output" / "artifacts" / "lie_engine.db"
        if not db_path.exists():
            return
        with closing(sqlite3.connect(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='bars_normalized'")
            self.assertEqual(int(cur.fetchone()[0]), 0)

    def test_run_ingestion_range_persists_when_explicitly_enabled(self) -> None:
        eng, tmp_root = self._make_engine()
        bars, _ = eng._run_ingestion_range(
            start=date(2026, 1, 2),
            end=date(2026, 1, 31),
            symbols=eng._core_symbols(),
            persist=True,
        )
        self.assertFalse(bars.empty)
        db_path = tmp_root / "output" / "artifacts" / "lie_engine.db"
        self.assertTrue(db_path.exists())
        with closing(sqlite3.connect(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM bars_normalized")
            self.assertGreater(int(cur.fetchone()[0]), 0)

    def test_maintain_sqlite_outputs_report_paths(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_eod(d)
        out = eng.maintain_sqlite(
            as_of=d,
            retention_days=30,
            tables=["signals", "review_runs"],
            vacuum=False,
            analyze=False,
            apply=False,
        )
        self.assertIn("paths", out)
        paths = out.get("paths", {}) if isinstance(out.get("paths", {}), dict) else {}
        self.assertTrue(Path(str(paths.get("json", ""))).exists())
        self.assertTrue(Path(str(paths.get("md", ""))).exists())
        retention = out.get("retention", {}) if isinstance(out.get("retention", {}), dict) else {}
        self.assertEqual(str(retention.get("status", "")), "ok")

    def test_run_eod_manifest_binds_release_decision_snapshot(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = review_dir / f"{d.isoformat()}_release_decision_snapshot.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "date": d.isoformat(),
                    "decision_id": f"{d.isoformat()}_abc123def456",
                    "fingerprint": "abc123def4567890",
                }
            ),
            encoding="utf-8",
        )
        out = eng.run_eod(d)
        manifest_path = Path(str(out.get("manifest", "")))
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        self.assertEqual(str(metadata.get("release_decision_id", "")), f"{d.isoformat()}_abc123def456")
        self.assertEqual(
            Path(str(metadata.get("release_decision_snapshot", ""))).resolve(),
            snapshot_path.resolve(),
        )
        self.assertTrue(bool(metadata.get("release_decision_found", False)))
        self.assertEqual(str(out.get("release_decision_id", "")), f"{d.isoformat()}_abc123def456")

    def test_symbol_exposure_snapshot_handles_string_size_pct_rows(self) -> None:
        eng, tmp_root = self._make_engine()
        db_path = tmp_root / "output" / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        core_symbols = eng._core_symbols()
        sym_a = str(core_symbols[0]) if core_symbols else "600519"
        sym_b = str(core_symbols[1]) if len(core_symbols) >= 2 else "159915"
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct TEXT, status TEXT)"
            )
            conn.executemany(
                "INSERT INTO latest_positions (date, symbol, size_pct, status) VALUES (?, ?, ?, ?)",
                [
                    ("2026-02-13", sym_a, "0.12", "ACTIVE"),
                    ("2026-02-13", sym_b, "0.08", "ACTIVE"),
                    ("2026-02-13", "BAD", "not-a-number", "ACTIVE"),
                    ("2026-02-12", sym_a, "0.50", "ACTIVE"),
                ],
            )
            conn.commit()

        by_symbol, by_theme, total = eng._symbol_exposure_snapshot()
        self.assertAlmostEqual(float(total), 0.20, places=6)
        self.assertAlmostEqual(float(by_symbol.get(sym_a, 0.0)), 0.12, places=6)
        self.assertAlmostEqual(float(by_symbol.get(sym_b, 0.0)), 0.08, places=6)
        self.assertNotIn("BAD", by_symbol)
        self.assertGreaterEqual(float(sum(float(v) for v in by_theme.values())), 0.20)

    def test_run_eod_release_decision_staleness_policy_masks_stale_snapshot(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "release_decision_freshness_enabled": True,
                "release_decision_eod_max_staleness_hours": 1,
            }
        )
        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = review_dir / f"{d.isoformat()}_release_decision_snapshot.json"
        stale_generated_at = (datetime.now() - timedelta(hours=30)).isoformat()
        snapshot_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "date": d.isoformat(),
                    "decision_id": f"{d.isoformat()}_abc123def456",
                    "fingerprint": "abc123def4567890",
                    "generated_at": stale_generated_at,
                }
            ),
            encoding="utf-8",
        )

        out = eng.run_eod(d)
        manifest_path = Path(str(out.get("manifest", "")))
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        self.assertEqual(str(metadata.get("release_decision_id", "")), "")
        self.assertTrue(bool(metadata.get("release_decision_found", False)))
        self.assertFalse(bool(metadata.get("release_decision_fresh", True)))
        self.assertFalse(bool(metadata.get("release_decision_usable", True)))
        self.assertEqual(str(metadata.get("release_decision_stale_reason", "")), "stale_snapshot")
        self.assertEqual(str(out.get("release_decision_id", "")), "")
        self.assertFalse(bool(out.get("release_decision_fresh", True)))

    def test_stress_exec_approval_manifest_assists_proposal_id_and_writes(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        proposal_id = "abc123def4567890"
        proposal_path = review_dir / f"{d.isoformat()}_stress_matrix_execution_friction_trendline_autotune_proposal.json"
        proposal_path.write_text(
            json.dumps(
                {
                    "date": d.isoformat(),
                    "mode": "proposal_only",
                    "generated": True,
                    "proposal_id": proposal_id,
                    "apply_gate": {"allowed": False, "reason": "controlled_apply_pending"},
                    "proposal": {
                        "base_thresholds": {
                            "annual_drop_rise": 0.05,
                            "drawdown_rise_delta": 0.05,
                            "profit_factor_ratio_drop": 0.10,
                        },
                        "recommended_thresholds": {
                            "annual_drop_rise": 0.07,
                            "drawdown_rise_delta": 0.07,
                            "profit_factor_ratio_drop": 0.12,
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.stress_exec_controlled_apply_approval_manifest(as_of=d)
        self.assertTrue(bool(out.get("ok", False)))
        self.assertTrue(bool(out.get("written", False)))
        manifest_path = Path(str(out.get("manifest_path", "")))
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(str(manifest.get("proposal_id", "")), proposal_id)
        meta = manifest.get("meta", {}) if isinstance(manifest.get("meta", {}), dict) else {}
        self.assertTrue(bool(meta.get("proposal_id_assisted", False)))

    def test_stress_exec_approval_manifest_blocks_write_on_proposal_id_mismatch(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = review_dir / f"{d.isoformat()}_stress_matrix_execution_friction_trendline_autotune_proposal.json"
        proposal_path.write_text(
            json.dumps(
                {
                    "date": d.isoformat(),
                    "mode": "proposal_only",
                    "generated": True,
                    "proposal_id": "match_id_1234",
                    "apply_gate": {"allowed": False, "reason": "controlled_apply_pending"},
                    "proposal": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.stress_exec_controlled_apply_approval_manifest(as_of=d, proposal_id="mismatch_id_5678")
        self.assertFalse(bool(out.get("ok", True)))
        self.assertFalse(bool(out.get("written", False)))
        errors = set(str(x) for x in out.get("errors", []))
        self.assertIn("proposal_id_mismatch", errors)
        self.assertIn("write_blocked_by_lint", errors)
        manifest_path = Path(str(out.get("manifest_path", "")))
        self.assertFalse(manifest_path.exists())

    def test_frontend_hard_fail_approval_manifest_assists_proposal_id_and_writes(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 14)
        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        proposal_id = "frontend_hard_fail_id_1234"
        proposal_path = review_dir / f"{d.isoformat()}_frontend_snapshot_trend_hard_fail_proposal.json"
        proposal_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "frontend_snapshot_trend_hard_fail_promotion_proposal",
                    "generated": True,
                    "date": d.isoformat(),
                    "proposal_id": proposal_id,
                    "promotion_recommendation": "enable_hard_fail",
                    "burnin": {"active": True, "ok": True},
                    "patch": {"ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": True},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.frontend_hard_fail_controlled_apply_approval_manifest(as_of=d)
        self.assertTrue(bool(out.get("ok", False)))
        self.assertTrue(bool(out.get("written", False)))
        manifest_path = Path(str(out.get("manifest_path", "")))
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(str(manifest.get("type", "")), "frontend_snapshot_trend_hard_fail_approval")
        self.assertEqual(str(manifest.get("proposal_id", "")), proposal_id)
        meta = manifest.get("meta", {}) if isinstance(manifest.get("meta", {}), dict) else {}
        self.assertTrue(bool(meta.get("proposal_id_assisted", False)))

    def test_frontend_hard_fail_approval_manifest_blocks_write_on_proposal_id_mismatch(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 14)
        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = review_dir / f"{d.isoformat()}_frontend_snapshot_trend_hard_fail_proposal.json"
        proposal_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "frontend_snapshot_trend_hard_fail_promotion_proposal",
                    "generated": True,
                    "date": d.isoformat(),
                    "proposal_id": "frontend_match_id_1234",
                    "promotion_recommendation": "enable_hard_fail",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.frontend_hard_fail_controlled_apply_approval_manifest(
            as_of=d,
            proposal_id="frontend_mismatch_id_5678",
        )
        self.assertFalse(bool(out.get("ok", True)))
        self.assertFalse(bool(out.get("written", False)))
        errors = set(str(x) for x in out.get("errors", []))
        self.assertIn("proposal_id_mismatch", errors)
        self.assertIn("write_blocked_by_lint", errors)
        manifest_path = Path(str(out.get("manifest_path", "")))
        self.assertFalse(manifest_path.exists())

    def test_frontend_hard_fail_approval_manifest_validate_only_does_not_write(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 14)
        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = review_dir / f"{d.isoformat()}_frontend_snapshot_trend_hard_fail_proposal.json"
        proposal_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "frontend_snapshot_trend_hard_fail_promotion_proposal",
                    "generated": True,
                    "date": d.isoformat(),
                    "proposal_id": "frontend_validate_only_id_1",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out = eng.frontend_hard_fail_controlled_apply_approval_manifest(
            as_of=d,
            validate_only=True,
        )
        self.assertTrue(bool(out.get("ok", False)))
        self.assertFalse(bool(out.get("written", True)))
        manifest_path = Path(str(out.get("manifest_path", "")))
        self.assertFalse(manifest_path.exists())

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

    def test_run_eod_does_not_settle_future_open_date_positions(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        future_open = date(2026, 2, 15)
        symbol = eng._core_symbols()[0]

        state_path = tmp_root / "output" / "artifacts" / "paper_positions_open.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "as_of": future_open.isoformat(),
                    "positions": [
                        {
                            "open_date": future_open.isoformat(),
                            "symbol": symbol,
                            "side": "LONG",
                            "size_pct": 5.0,
                            "risk_pct": 1.0,
                            "entry_price": 100.0,
                            "stop_price": 1e9,
                            "target_price": 0.0,
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

        out = eng.run_eod(d)
        self.assertEqual(int(out.get("closed_trades", 0)), 0)
        with closing(sqlite3.connect(tmp_root / "output" / "artifacts" / "lie_engine.db")) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='executed_plans'")
            exists = cur.fetchone() is not None
            if exists:
                cur.execute("SELECT COUNT(*) FROM executed_plans WHERE date = ?", (d.isoformat(),))
                count = int(cur.fetchone()[0])
            else:
                count = 0
        self.assertEqual(count, 0)

    def test_run_eod_replay_clears_same_day_stale_latest_positions(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        db_path = tmp_root / "output" / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute(
                "INSERT INTO latest_positions (date, symbol, size_pct, status) VALUES (?, ?, ?, ?)",
                (d.isoformat(), "600519", 10.0, "ACTIVE"),
            )
            conn.commit()

        eng._major_event_window = lambda as_of, news: True  # type: ignore[method-assign]
        eng._loss_cooldown_active = lambda recent_trades: False  # type: ignore[method-assign]
        eng._black_swan_assessment = lambda regime, sentiment, news: (10.0, [], False)  # type: ignore[method-assign]
        out = eng.run_eod(d)
        self.assertEqual(int(out.get("plans", 0)), 0)

        with closing(sqlite3.connect(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM latest_positions WHERE date = ?", (d.isoformat(),))
            remaining = int(cur.fetchone()[0])
        self.assertEqual(remaining, 0)

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
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)

        slot_out = eng.run_slot(as_of=d, slot="08:40")
        self.assertEqual(slot_out["slot"], "premarket")
        self.assertIn("result", slot_out)

        session_out = eng.run_session(as_of=d, include_review=False)
        self.assertIn("steps", session_out)
        self.assertIn("eod", session_out["steps"])
        self.assertTrue(session_out["steps"]["review_cycle"].get("skipped"))

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

    def test_run_intraday_fallback_uses_config_first_slot(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("schedule", {})
        eng.settings.raw["schedule"]["intraday_slots"] = ["09:45", "13:15"]
        out = eng.run_intraday_check(as_of=d, slot="11:00")
        self.assertEqual(out["slot"], "09:45")

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

    def test_run_halfhour_pulse_dry_run(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        out = eng.run_halfhour_pulse(as_of=d, slot="11:00", dry_run=True)
        self.assertTrue(out["dry_run"])
        self.assertIn("due_slots", out)
        self.assertFalse((tmp_root / "output" / "logs" / "halfhour_pulse_state.json").exists())

    def test_run_halfhour_daemon_dry_run(self) -> None:
        eng, tmp_root = self._make_engine()
        out = eng.run_halfhour_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=1, dry_run=True)
        self.assertTrue(out["dry_run"])
        self.assertIn("would_run_pulse", out)
        self.assertFalse((tmp_root / "output" / "logs" / "halfhour_daemon_state.json").exists())

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

    def test_run_review_calibrates_degradation_params_from_gate_replay(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_degradation_calibration_enabled": True,
                "ops_degradation_calibration_window_days": 5,
                "ops_degradation_calibration_min_samples": 3,
                "ops_degradation_calibration_fp_target": 0.10,
                "ops_degradation_calibration_fn_target": 0.05,
                "ops_degradation_calibration_step_multiplier": 0.10,
                "ops_degradation_calibration_step_floor_ratio": 0.05,
                "ops_degradation_calibration_step_streak_days": 1,
                "ops_degradation_calibration_use_live_overrides": True,
                "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
            }
        )

        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            day = date.fromordinal(d.toordinal() - i)
            (review_dir / f"{day.isoformat()}_gate_report.json").write_text(
                json.dumps(
                    {
                        "slot_anomaly": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "premarket_anomaly_ratio": {
                                        "value": 0.10,
                                        "threshold": 0.50,
                                        "ok": False,
                                    }
                                },
                            },
                        },
                        "state_stability": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "switch_rate": {
                                        "value": 0.20,
                                        "threshold": 0.45,
                                        "ok": False,
                                    },
                                    "risk_multiplier_floor": {
                                        "value": 0.40,
                                        "threshold": 0.35,
                                        "ok": False,
                                    },
                                },
                            },
                        },
                    }
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

        review = eng.run_review(d)
        self.assertTrue(any("degradation_calibration=" in n for n in review.notes))

        live_path = tmp_root / "output" / "artifacts" / "degradation_params_live.yaml"
        self.assertTrue(live_path.exists())
        live_payload = yaml.safe_load(live_path.read_text(encoding="utf-8"))
        live_params = live_payload.get("params", {})
        self.assertGreater(float(live_params.get("ops_slot_degradation_soft_multiplier", 0.0)), 1.15)
        self.assertGreater(float(live_params.get("ops_state_degradation_soft_multiplier", 0.0)), 1.10)
        self.assertGreater(int(float(live_params.get("ops_slot_hysteresis_soft_streak_days", 0.0))), 2)
        self.assertLess(float(live_params.get("ops_state_degradation_floor_soft_ratio", 1.0)), 0.96)

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        delta_payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        calibration = delta_payload.get("degradation_calibration", {})
        self.assertTrue(bool(calibration.get("changed", False)))
        self.assertIn("domains", calibration)
        self.assertEqual(str(calibration.get("domains", {}).get("slot", {}).get("direction", "")), "relax")
        artifact_json = Path(str(calibration.get("artifacts", {}).get("json", "")))
        artifact_md = Path(str(calibration.get("artifacts", {}).get("md", "")))
        self.assertTrue(artifact_json.exists())
        self.assertTrue(artifact_md.exists())

    def test_run_review_rolls_back_degradation_params_on_fn_and_gate_fail_rise(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_degradation_calibration_enabled": True,
                "ops_degradation_calibration_window_days": 6,
                "ops_degradation_calibration_min_samples": 2,
                "ops_degradation_calibration_use_live_overrides": True,
                "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
                "ops_degradation_calibration_rollback_enabled": True,
                "ops_degradation_calibration_rollback_window_days": 6,
                "ops_degradation_calibration_rollback_recent_days": 2,
                "ops_degradation_calibration_rollback_min_samples": 2,
                "ops_degradation_calibration_rollback_fn_rise_min": 0.20,
                "ops_degradation_calibration_rollback_gate_fail_rise_min": 0.20,
                "ops_degradation_calibration_rollback_auto_promote_on_stable": True,
                "ops_degradation_calibration_rollback_stable_min_samples": 2,
                "ops_degradation_calibration_rollback_stable_fn_rate_max": 0.05,
                "ops_degradation_calibration_rollback_stable_gate_fail_ratio_max": 0.10,
                "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/degradation_snapshot/active.yaml",
                "ops_degradation_calibration_rollback_history_dir": "artifacts/degradation_snapshot/history",
            }
        )

        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            day = date.fromordinal(d.toordinal() - i)
            recent = i <= 1
            (review_dir / f"{day.isoformat()}_gate_report.json").write_text(
                json.dumps(
                    {
                        "passed": False if recent else True,
                        "slot_anomaly": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "premarket_anomaly_ratio": {
                                        "value": 0.80 if recent else 0.20,
                                        "threshold": 0.50,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                        "state_stability": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "switch_rate": {
                                        "value": 0.80 if recent else 0.20,
                                        "threshold": 0.45,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

        snapshot_active = tmp_root / "output" / "artifacts" / "degradation_snapshot" / "active.yaml"
        snapshot_active.parent.mkdir(parents=True, exist_ok=True)
        snapshot_active.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": "2026-02-10",
                    "promoted_at": "2026-02-10T20:30:00",
                    "params": {
                        "ops_slot_degradation_soft_multiplier": 1.60,
                        "ops_slot_degradation_hard_multiplier": 1.90,
                        "ops_slot_hysteresis_soft_streak_days": 5,
                        "ops_slot_hysteresis_hard_streak_days": 6,
                        "ops_state_degradation_soft_multiplier": 1.55,
                        "ops_state_degradation_hard_multiplier": 1.90,
                        "ops_state_degradation_floor_soft_ratio": 0.80,
                        "ops_state_degradation_floor_hard_ratio": 0.75,
                        "ops_state_hysteresis_soft_streak_days": 5,
                        "ops_state_hysteresis_hard_streak_days": 6,
                    },
                },
                allow_unicode=True,
                sort_keys=False,
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
        live_path = tmp_root / "output" / "artifacts" / "degradation_params_live.yaml"
        self.assertTrue(live_path.exists())
        live_payload = yaml.safe_load(live_path.read_text(encoding="utf-8"))
        live_params = live_payload.get("params", {})
        self.assertAlmostEqual(float(live_params.get("ops_slot_degradation_soft_multiplier", 0.0)), 1.60, places=6)
        self.assertAlmostEqual(float(live_params.get("ops_state_degradation_soft_multiplier", 0.0)), 1.55, places=6)

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        calibration = payload.get("degradation_calibration", {})
        rollback = calibration.get("rollback", {})
        self.assertTrue(bool(rollback.get("triggered", False)))
        self.assertTrue(bool(rollback.get("applied", False)))
        self.assertEqual(str(rollback.get("reason", "")), "rollback_to_stable_snapshot")
        source_path = str((rollback.get("snapshot", {}) or {}).get("source_path", ""))
        self.assertEqual(Path(source_path).resolve(), snapshot_active.resolve())
        rollback_artifact = Path(str(calibration.get("artifacts", {}).get("rollback_json", "")))
        self.assertTrue(rollback_artifact.exists())

    def test_run_review_blocks_rollback_on_snapshot_chain_checksum_mismatch(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_degradation_calibration_enabled": True,
                "ops_degradation_calibration_window_days": 6,
                "ops_degradation_calibration_min_samples": 2,
                "ops_degradation_calibration_use_live_overrides": True,
                "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
                "ops_degradation_calibration_rollback_enabled": True,
                "ops_degradation_calibration_rollback_window_days": 6,
                "ops_degradation_calibration_rollback_recent_days": 2,
                "ops_degradation_calibration_rollback_min_samples": 2,
                "ops_degradation_calibration_rollback_fn_rise_min": 0.20,
                "ops_degradation_calibration_rollback_gate_fail_rise_min": 0.20,
                "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/degradation_snapshot/active.yaml",
                "ops_degradation_calibration_rollback_history_dir": "artifacts/degradation_snapshot/history",
                "ops_degradation_calibration_snapshot_chain_enabled": True,
                "ops_degradation_calibration_snapshot_chain_require_active": True,
                "ops_degradation_calibration_snapshot_chain_require_checksum": True,
                "ops_degradation_calibration_snapshot_chain_max_depth": 8,
            }
        )

        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            day = date.fromordinal(d.toordinal() - i)
            recent = i <= 1
            (review_dir / f"{day.isoformat()}_gate_report.json").write_text(
                json.dumps(
                    {
                        "passed": False if recent else True,
                        "slot_anomaly": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "premarket_anomaly_ratio": {
                                        "value": 0.80 if recent else 0.20,
                                        "threshold": 0.50,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                        "state_stability": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "switch_rate": {
                                        "value": 0.80 if recent else 0.20,
                                        "threshold": 0.45,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

        snapshot_active = tmp_root / "output" / "artifacts" / "degradation_snapshot" / "active.yaml"
        snapshot_active.parent.mkdir(parents=True, exist_ok=True)
        snapshot_params = {
            "ops_slot_degradation_soft_multiplier": 1.95,
            "ops_slot_degradation_hard_multiplier": 2.00,
            "ops_slot_hysteresis_soft_streak_days": 6,
            "ops_slot_hysteresis_hard_streak_days": 7,
            "ops_state_degradation_soft_multiplier": 1.90,
            "ops_state_degradation_hard_multiplier": 2.00,
            "ops_state_degradation_floor_soft_ratio": 0.72,
            "ops_state_degradation_floor_hard_ratio": 0.70,
            "ops_state_hysteresis_soft_streak_days": 6,
            "ops_state_hysteresis_hard_streak_days": 7,
        }
        params_checksum = hashlib.sha1(
            json.dumps(snapshot_params, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        snapshot_active.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": "2026-02-10",
                    "promoted_at": "2026-02-10T20:30:00",
                    "params": snapshot_params,
                    "history_path": str(snapshot_active),
                    "snapshot_path": str(snapshot_active),
                    "rollback_anchor": "",
                    "params_checksum": params_checksum,
                    "chain_checksum": "tampered-chain-checksum",
                },
                allow_unicode=True,
                sort_keys=False,
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
        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        calibration = payload.get("degradation_calibration", {})
        rollback = calibration.get("rollback", {})
        self.assertTrue(bool(rollback.get("triggered", False)))
        self.assertFalse(bool(rollback.get("applied", True)))
        self.assertEqual(str(rollback.get("reason", "")), "triggered_but_snapshot_chain_invalid")
        integrity = (((rollback.get("snapshot", {}) or {}).get("integrity", {})) or {})
        self.assertFalse(bool(integrity.get("ok", True)))
        alerts = set((integrity.get("alerts", []) if isinstance(integrity.get("alerts", []), list) else []))
        self.assertIn("snapshot_chain_checksum_failed", alerts)

    def test_run_review_suppresses_rollback_during_cooldown(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_degradation_calibration_enabled": True,
                "ops_degradation_calibration_window_days": 6,
                "ops_degradation_calibration_min_samples": 2,
                "ops_degradation_calibration_use_live_overrides": True,
                "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
                "ops_degradation_calibration_rollback_enabled": True,
                "ops_degradation_calibration_rollback_window_days": 6,
                "ops_degradation_calibration_rollback_recent_days": 2,
                "ops_degradation_calibration_rollback_min_samples": 2,
                "ops_degradation_calibration_rollback_fn_rise_min": 0.20,
                "ops_degradation_calibration_rollback_gate_fail_rise_min": 0.20,
                "ops_degradation_calibration_rollback_cooldown_days": 3,
                "ops_degradation_calibration_rollback_promotion_cooldown_days": 7,
                "ops_degradation_calibration_rollback_hysteresis_window_days": 7,
                "ops_degradation_calibration_rollback_trigger_hysteresis_buffer": 0.02,
                "ops_degradation_calibration_rollback_stable_hysteresis_buffer": 0.01,
                "ops_degradation_calibration_rollback_auto_promote_on_stable": True,
                "ops_degradation_calibration_rollback_stable_min_samples": 2,
                "ops_degradation_calibration_rollback_stable_fn_rate_max": 0.05,
                "ops_degradation_calibration_rollback_stable_gate_fail_ratio_max": 0.10,
                "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/degradation_snapshot/active.yaml",
                "ops_degradation_calibration_rollback_history_dir": "artifacts/degradation_snapshot/history",
            }
        )

        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        prev_day = date.fromordinal(d.toordinal() - 1)
        (review_dir / f"{prev_day.isoformat()}_degradation_calibration_rollback.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "triggered": True,
                    "applied": True,
                    "reason": "rollback_to_stable_snapshot",
                    "snapshot_promotion": {"eligible": False, "promoted": False, "reason": "not_eligible"},
                }
            ),
            encoding="utf-8",
        )
        for i in range(6):
            day = date.fromordinal(d.toordinal() - i)
            recent = i <= 1
            (review_dir / f"{day.isoformat()}_gate_report.json").write_text(
                json.dumps(
                    {
                        "passed": False if recent else True,
                        "slot_anomaly": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "premarket_anomaly_ratio": {
                                        "value": 0.80 if recent else 0.20,
                                        "threshold": 0.50,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                        "state_stability": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "switch_rate": {
                                        "value": 0.80 if recent else 0.20,
                                        "threshold": 0.45,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

        snapshot_active = tmp_root / "output" / "artifacts" / "degradation_snapshot" / "active.yaml"
        snapshot_active.parent.mkdir(parents=True, exist_ok=True)
        snapshot_active.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": "2026-02-10",
                    "promoted_at": "2026-02-10T20:30:00",
                    "params": {
                        "ops_slot_degradation_soft_multiplier": 1.60,
                        "ops_slot_degradation_hard_multiplier": 1.90,
                        "ops_slot_hysteresis_soft_streak_days": 5,
                        "ops_slot_hysteresis_hard_streak_days": 6,
                        "ops_state_degradation_soft_multiplier": 1.55,
                        "ops_state_degradation_hard_multiplier": 1.90,
                        "ops_state_degradation_floor_soft_ratio": 0.80,
                        "ops_state_degradation_floor_hard_ratio": 0.75,
                        "ops_state_hysteresis_soft_streak_days": 5,
                        "ops_state_hysteresis_hard_streak_days": 6,
                    },
                },
                allow_unicode=True,
                sort_keys=False,
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
        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        rollback = payload.get("degradation_calibration", {}).get("rollback", {})
        self.assertTrue(bool(rollback.get("triggered_raw", False)))
        self.assertFalse(bool(rollback.get("triggered", False)))
        self.assertFalse(bool(rollback.get("applied", False)))
        self.assertEqual(str(rollback.get("reason", "")), "rollback_cooldown_active")
        cooldown = rollback.get("cooldown", {})
        self.assertTrue(bool(cooldown.get("rollback_active", False)))
        self.assertEqual(int(cooldown.get("last_rollback_days_ago", -1)), 1)

    def test_run_review_blocks_snapshot_promotion_during_cooldown(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_degradation_calibration_enabled": True,
                "ops_degradation_calibration_window_days": 5,
                "ops_degradation_calibration_min_samples": 2,
                "ops_degradation_calibration_use_live_overrides": True,
                "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
                "ops_degradation_calibration_rollback_enabled": True,
                "ops_degradation_calibration_rollback_window_days": 5,
                "ops_degradation_calibration_rollback_recent_days": 2,
                "ops_degradation_calibration_rollback_min_samples": 2,
                "ops_degradation_calibration_rollback_fn_rise_min": 0.20,
                "ops_degradation_calibration_rollback_gate_fail_rise_min": 0.20,
                "ops_degradation_calibration_rollback_cooldown_days": 1,
                "ops_degradation_calibration_rollback_promotion_cooldown_days": 5,
                "ops_degradation_calibration_rollback_hysteresis_window_days": 5,
                "ops_degradation_calibration_rollback_trigger_hysteresis_buffer": 0.02,
                "ops_degradation_calibration_rollback_stable_hysteresis_buffer": 0.01,
                "ops_degradation_calibration_rollback_auto_promote_on_stable": True,
                "ops_degradation_calibration_rollback_stable_min_samples": 2,
                "ops_degradation_calibration_rollback_stable_fn_rate_max": 0.20,
                "ops_degradation_calibration_rollback_stable_gate_fail_ratio_max": 0.10,
                "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/degradation_snapshot/active.yaml",
                "ops_degradation_calibration_rollback_history_dir": "artifacts/degradation_snapshot/history",
            }
        )

        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        prev_day = date.fromordinal(d.toordinal() - 1)
        (review_dir / f"{prev_day.isoformat()}_degradation_calibration_rollback.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "triggered": True,
                    "applied": True,
                    "reason": "rollback_to_stable_snapshot",
                    "snapshot_promotion": {"eligible": False, "promoted": False, "reason": "not_eligible"},
                }
            ),
            encoding="utf-8",
        )
        for i in range(5):
            day = date.fromordinal(d.toordinal() - i)
            (review_dir / f"{day.isoformat()}_gate_report.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "slot_anomaly": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "premarket_anomaly_ratio": {
                                        "value": 0.20,
                                        "threshold": 0.50,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                        "state_stability": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "switch_rate": {
                                        "value": 0.20,
                                        "threshold": 0.45,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                    }
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
        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        promotion = (
            payload.get("degradation_calibration", {})
            .get("rollback", {})
            .get("snapshot_promotion", {})
        )
        self.assertFalse(bool(promotion.get("eligible", False)))
        self.assertFalse(bool(promotion.get("promoted", False)))
        self.assertEqual(str(promotion.get("reason", "")), "promotion_cooldown_active")

    def test_run_review_promotes_degradation_snapshot_when_stable(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.settings.raw.setdefault("validation", {})
        eng.settings.raw["validation"].update(
            {
                "ops_degradation_calibration_enabled": True,
                "ops_degradation_calibration_window_days": 5,
                "ops_degradation_calibration_min_samples": 2,
                "ops_degradation_calibration_use_live_overrides": True,
                "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
                "ops_degradation_calibration_rollback_enabled": True,
                "ops_degradation_calibration_rollback_window_days": 5,
                "ops_degradation_calibration_rollback_recent_days": 2,
                "ops_degradation_calibration_rollback_min_samples": 2,
                "ops_degradation_calibration_rollback_fn_rise_min": 0.20,
                "ops_degradation_calibration_rollback_gate_fail_rise_min": 0.20,
                "ops_degradation_calibration_rollback_auto_promote_on_stable": True,
                "ops_degradation_calibration_rollback_stable_min_samples": 2,
                "ops_degradation_calibration_rollback_stable_fn_rate_max": 0.20,
                "ops_degradation_calibration_rollback_stable_gate_fail_ratio_max": 0.10,
                "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/degradation_snapshot/active.yaml",
                "ops_degradation_calibration_rollback_history_dir": "artifacts/degradation_snapshot/history",
            }
        )

        review_dir = tmp_root / "output" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            day = date.fromordinal(d.toordinal() - i)
            (review_dir / f"{day.isoformat()}_gate_report.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "slot_anomaly": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "premarket_anomaly_ratio": {
                                        "value": 0.20,
                                        "threshold": 0.50,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                        "state_stability": {
                            "active": True,
                            "degradation": {
                                "enabled": True,
                                "checks": {
                                    "switch_rate": {
                                        "value": 0.20,
                                        "threshold": 0.45,
                                        "ok": True,
                                    }
                                },
                            },
                        },
                    }
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
        snapshot_active = tmp_root / "output" / "artifacts" / "degradation_snapshot" / "active.yaml"
        self.assertTrue(snapshot_active.exists())
        snapshot_payload = yaml.safe_load(snapshot_active.read_text(encoding="utf-8"))
        self.assertIn("params", snapshot_payload)
        self.assertIn("history_path", snapshot_payload)
        self.assertTrue(Path(str(snapshot_payload.get("history_path", ""))).exists())

        delta_path = tmp_root / "output" / "review" / "2026-02-13_param_delta.yaml"
        payload = yaml.safe_load(delta_path.read_text(encoding="utf-8"))
        promotion = (
            payload.get("degradation_calibration", {})
            .get("rollback", {})
            .get("snapshot_promotion", {})
        )
        self.assertTrue(bool(promotion.get("promoted", False)))
        self.assertEqual(str(promotion.get("reason", "")), "promoted")

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
        friction = payload.get("execution_friction", {}) if isinstance(payload.get("execution_friction", {}), dict) else {}
        self.assertIn("scorecard", friction)
        self.assertGreaterEqual(len(friction.get("scenarios", [])), 1)
        self.assertIn("matrix", friction)
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
        finally:
            eng_mod.run_strategy_lab_pipeline = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
