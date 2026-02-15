from __future__ import annotations

from datetime import date
from pathlib import Path
import shutil
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.config import SystemSettings
from lie_engine.orchestration.observability import ObservabilityOrchestrator


class ObservabilityOrchestratorTests(unittest.TestCase):
    def _make_settings(self) -> SystemSettings:
        return SystemSettings(
            raw={
                "timezone": "Asia/Shanghai",
                "validation": {"required_stable_replay_days": 3},
            }
        )

    def test_health_check_review_required_toggle(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        day = date(2026, 2, 13).isoformat()
        sqlite_path = td / "artifacts" / "lie_engine.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_path.touch()

        daily = td / "daily"
        daily.mkdir(parents=True, exist_ok=True)
        (daily / f"{day}_briefing.md").write_text("# brief\n", encoding="utf-8")
        (daily / f"{day}_signals.json").write_text("{}\n", encoding="utf-8")
        (daily / f"{day}_positions.csv").write_text("symbol,size\n", encoding="utf-8")

        orch = ObservabilityOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            sqlite_path=sqlite_path,
            run_eod=lambda as_of: {},
        )

        no_review = orch.health_check(as_of=date(2026, 2, 13), require_review=False)
        with_review = orch.health_check(as_of=date(2026, 2, 13), require_review=True)

        self.assertEqual(no_review["status"], "healthy")
        self.assertEqual(with_review["status"], "degraded")
        self.assertIn("review_report", with_review["missing"])
        self.assertIn("review_delta", with_review["missing"])

    def test_stable_replay_invokes_run_eod_for_each_day(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        sqlite_path = td / "artifacts" / "lie_engine.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_path.touch()

        replay_dates: list[str] = []

        def run_eod(as_of: date) -> dict[str, object]:
            replay_dates.append(as_of.isoformat())
            day = as_of.isoformat()
            daily = td / "daily"
            review = td / "review"
            daily.mkdir(parents=True, exist_ok=True)
            review.mkdir(parents=True, exist_ok=True)
            (daily / f"{day}_briefing.md").write_text("# brief\n", encoding="utf-8")
            (daily / f"{day}_signals.json").write_text("{}\n", encoding="utf-8")
            (daily / f"{day}_positions.csv").write_text("symbol,size\n", encoding="utf-8")
            (review / f"{day}_review.md").write_text("# review\n", encoding="utf-8")
            (review / f"{day}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")
            return {"date": day}

        orch = ObservabilityOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            sqlite_path=sqlite_path,
            run_eod=run_eod,
        )

        out = orch.stable_replay_check(as_of=date(2026, 2, 13), days=2)
        self.assertTrue(out["passed"])
        self.assertEqual(out["replay_days"], 2)
        self.assertEqual(len(out["checks"]), 2)
        self.assertEqual(replay_dates, ["2026-02-13", "2026-02-12"])


if __name__ == "__main__":
    unittest.main()
