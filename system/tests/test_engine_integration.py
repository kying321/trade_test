from __future__ import annotations

import sys
from pathlib import Path
import unittest
from datetime import date
import tempfile

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.engine import LieEngine
from lie_engine.models import ReviewDelta


class EngineIntegrationTests(unittest.TestCase):
    def _make_engine(self) -> tuple[LieEngine, Path]:
        project_root = Path(__file__).resolve().parents[1]
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp_root = Path(td.name)

        cfg_data = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
        cfg_data["paths"] = {"output": "output", "sqlite": "output/artifacts/lie_engine.db"}
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
        self.assertTrue((tmp_root / "output" / "artifacts" / "manifests" / "eod_2026-02-13.json").exists())

    def test_run_eod_blocks_new_positions_under_major_event_window(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng._major_event_window = lambda as_of, news: True  # type: ignore[method-assign]
        eng._loss_cooldown_active = lambda recent_trades: False  # type: ignore[method-assign]
        eng._black_swan_assessment = lambda regime, sentiment, news: (10.0, [], False)  # type: ignore[method-assign]

        out = eng.run_eod(d)
        self.assertEqual(out["plans"], 0)
        self.assertTrue(any("重大事件窗口" in x for x in out["non_trade_reasons"]))

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
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_gate_report.json").exists())

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
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_ops_report.json").exists())
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_ops_report.md").exists())

    def test_run_slot_ops(self) -> None:
        eng, _ = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_review(d)
        out = eng.run_slot(as_of=d, slot="ops")
        self.assertEqual(out["slot"], "ops")
        self.assertIn("result", out)

    def test_architecture_audit_generates_files(self) -> None:
        eng, tmp_root = self._make_engine()
        d = date(2026, 2, 13)
        eng.run_eod(d)
        audit = eng.architecture_audit(as_of=d)
        self.assertIn("status", audit)
        self.assertIn("config", audit)
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_architecture_audit.json").exists())
        self.assertTrue((tmp_root / "output" / "review" / "2026-02-13_architecture_audit.md").exists())

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


if __name__ == "__main__":
    unittest.main()
