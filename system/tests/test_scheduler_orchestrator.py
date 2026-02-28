from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import shutil
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.config import SystemSettings
from lie_engine.orchestration.scheduler import SchedulerOrchestrator


class SchedulerOrchestratorTests(unittest.TestCase):
    def _make_settings(self, schedule: dict[str, object] | None = None) -> SystemSettings:
        return SystemSettings(
            raw={
                "timezone": "Asia/Shanghai",
                "schedule": schedule
                or {
                    "premarket": "08:40",
                    "intraday_slots": ["10:30", "14:30"],
                    "eod": "15:10",
                    "nightly_review": "20:30",
                },
                "validation": {"required_stable_replay_days": 3},
            }
        )

    def _make_orchestrator(self, output_dir: Path, settings: SystemSettings | None = None) -> tuple[SchedulerOrchestrator, dict[str, object]]:
        calls: dict[str, object] = {
            "premarket": 0,
            "intraday": [],
            "eod": 0,
            "review": [],
            "ops": [],
        }

        def run_premarket(as_of: date) -> dict[str, object]:
            calls["premarket"] = int(calls["premarket"]) + 1
            return {"phase": "premarket", "date": as_of.isoformat()}

        def run_intraday_check(as_of: date, slot: str) -> dict[str, object]:
            intraday_calls = calls["intraday"]
            assert isinstance(intraday_calls, list)
            intraday_calls.append(slot)
            return {"phase": "intraday", "date": as_of.isoformat(), "slot": slot}

        def run_eod(as_of: date) -> dict[str, object]:
            calls["eod"] = int(calls["eod"]) + 1
            return {"phase": "eod", "date": as_of.isoformat()}

        def run_review_cycle(as_of: date, max_rounds: int) -> dict[str, object]:
            review_calls = calls["review"]
            assert isinstance(review_calls, list)
            review_calls.append({"date": as_of.isoformat(), "max_rounds": max_rounds})
            return {"phase": "review", "date": as_of.isoformat(), "max_rounds": max_rounds}

        def run_ops_report(as_of: date, window_days: int) -> dict[str, object]:
            ops_calls = calls["ops"]
            assert isinstance(ops_calls, list)
            ops_calls.append({"date": as_of.isoformat(), "window_days": window_days})
            return {"phase": "ops", "date": as_of.isoformat(), "window_days": window_days}

        orchestrator = SchedulerOrchestrator(
            settings=settings or self._make_settings(),
            output_dir=output_dir,
            run_premarket=run_premarket,
            run_intraday_check=run_intraday_check,
            run_eod=run_eod,
            run_review_cycle=run_review_cycle,
            ops_report=run_ops_report,
        )
        return orchestrator, calls

    def test_run_slot_routes_named_and_time_slots(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        out_premarket = orch.run_slot(as_of=d, slot="08:40")
        out_ops = orch.run_slot(as_of=d, slot="ops")

        self.assertEqual(out_premarket["slot"], "premarket")
        self.assertEqual(out_ops["slot"], "ops")
        self.assertEqual(calls["premarket"], 1)
        self.assertEqual(calls["ops"], [{"date": "2026-02-13", "window_days": 3}])

    def test_run_slot_intraday_aliases(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        out1 = orch.run_slot(as_of=d, slot="intraday_1")
        out2 = orch.run_slot(as_of=d, slot="intraday_2")

        self.assertEqual(out1["slot"], "intraday:10:30")
        self.assertEqual(out2["slot"], "intraday:14:30")
        self.assertEqual(calls["intraday"], ["10:30", "14:30"])

    def test_run_session_respects_review_flag(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        out = orch.run_session(as_of=d, include_review=False, max_review_rounds=5)

        self.assertIn("steps", out)
        self.assertEqual(calls["premarket"], 1)
        self.assertEqual(calls["eod"], 1)
        self.assertEqual(calls["intraday"], ["10:30", "14:30"])
        self.assertEqual(calls["review"], [])
        self.assertTrue(out["steps"]["review_cycle"].get("skipped"))

    def test_run_daemon_persists_scheduler_state(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "23:59",
        }
        settings = self._make_settings(schedule=schedule)
        orch, calls = self._make_orchestrator(td, settings=settings)

        state = orch.run_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=1)
        state_path = td / "logs" / "scheduler_state.json"

        self.assertTrue(state_path.exists())
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["executed"], persisted["executed"])
        self.assertIn("premarket", state["executed"])
        self.assertGreaterEqual(int(calls["premarket"]), 1)

    def test_run_daemon_allows_same_trigger_time_for_eod_and_review(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "23:59",
            "intraday_slots": [],
            "eod": "00:00",
            "nightly_review": "00:00",
        }
        settings = self._make_settings(schedule=schedule)
        orch, calls = self._make_orchestrator(td, settings=settings)

        state = orch.run_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=2)

        self.assertIn("eod", state["executed"])
        self.assertIn("review", state["executed"])
        self.assertEqual(calls["eod"], 1)
        self.assertEqual(calls["review"], [{"date": state["date"], "max_rounds": 2}])

    def test_run_daemon_dry_run_does_not_execute_or_persist_state(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": ["00:00"],
            "eod": "00:00",
            "nightly_review": "00:00",
        }
        settings = self._make_settings(schedule=schedule)
        orch, calls = self._make_orchestrator(td, settings=settings)

        out = orch.run_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=2, dry_run=True)
        state_path = td / "logs" / "scheduler_state.json"

        self.assertTrue(out["dry_run"])
        self.assertFalse(state_path.exists())
        self.assertEqual(calls["premarket"], 0)
        self.assertEqual(calls["intraday"], [])
        self.assertEqual(calls["eod"], 0)
        self.assertEqual(calls["review"], [])
        self.assertEqual(set(out["would_execute"]), {"premarket", "intraday:00:00", "eod", "review"})


if __name__ == "__main__":
    unittest.main()
