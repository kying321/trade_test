from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import re
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch

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
            "health": [],
            "guardrail_burnin": [],
            "guardrail_drift": [],
            "compact": [],
            "restore_verify": [],
            "db_maintenance": [],
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

        def run_health_check(as_of: date, require_review: bool) -> dict[str, object]:
            health_calls = calls["health"]
            assert isinstance(health_calls, list)
            health_calls.append({"date": as_of.isoformat(), "require_review": require_review})
            return {"status": "healthy", "missing": [], "checks": {"ok": True}}

        def run_guardrail_burnin(
            as_of: date,
            days: int,
            run_stable_replay: bool,
            auto_tune: bool,
        ) -> dict[str, object]:
            burnin_calls = calls["guardrail_burnin"]
            assert isinstance(burnin_calls, list)
            burnin_calls.append(
                {
                    "date": as_of.isoformat(),
                    "days": int(days),
                    "run_stable_replay": bool(run_stable_replay),
                    "auto_tune": bool(auto_tune),
                }
            )
            return {
                "summary": {"active_days": int(days), "false_positive_ratio": 0.05},
                "live_overrides": {"applied": False},
                "paths": {"json": f"burnin/{as_of.isoformat()}.json"},
            }

        def run_guardrail_drift(as_of: date, window_days: int) -> dict[str, object]:
            drift_calls = calls["guardrail_drift"]
            assert isinstance(drift_calls, list)
            drift_calls.append({"date": as_of.isoformat(), "window_days": int(window_days)})
            return {
                "status": "ok",
                "alerts": [],
                "paths": {"json": f"drift/{as_of.isoformat()}.json"},
            }

        def run_compact_executed_plans(
            start: date,
            end: date,
            chunk_days: int,
            dry_run: bool,
            max_delete_rows: int | None,
        ) -> dict[str, object]:
            compact_calls = calls["compact"]
            assert isinstance(compact_calls, list)
            compact_calls.append(
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "chunk_days": int(chunk_days),
                    "dry_run": bool(dry_run),
                    "max_delete_rows": max_delete_rows,
                }
            )
            return {
                "run_id": f"{end.isoformat()}_compact",
                "status": "ok",
                "dry_run": bool(dry_run),
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "metrics": {"duplicate_rows_found": 0, "deleted_rows": 0},
                "paths": {"json": f"compact/{end.isoformat()}.json"},
                "rollback": {"available": False},
            }

        def verify_compaction_restore(run_id: str | None, keep_temp_db: bool) -> dict[str, object]:
            verify_calls = calls["restore_verify"]
            assert isinstance(verify_calls, list)
            verify_calls.append({"run_id": str(run_id or ""), "keep_temp_db": bool(keep_temp_db)})
            return {
                "run_id": str(run_id or ""),
                "status": "ok",
                "reason": "ok",
                "checks": {"restore_delta_match": True},
                "metrics": {"backup_rows": 1, "restored_rows_delta": 1},
                "paths": {"json": f"compact/{run_id}_verify.json"},
            }

        def run_db_maintenance(
            as_of: date,
            retention_days: int | None,
            tables: list[str] | None,
            vacuum: bool,
            analyze: bool,
            apply: bool,
        ) -> dict[str, object]:
            db_calls = calls["db_maintenance"]
            assert isinstance(db_calls, list)
            db_calls.append(
                {
                    "date": as_of.isoformat(),
                    "retention_days": retention_days,
                    "tables": list(tables or []),
                    "vacuum": bool(vacuum),
                    "analyze": bool(analyze),
                    "apply": bool(apply),
                }
            )
            return {
                "retention": {
                    "status": "ok",
                    "eligible_rows": 0,
                    "deleted_rows": 0,
                },
                "vacuum": {
                    "status": "ok",
                    "run_vacuum": bool(vacuum),
                    "run_analyze": bool(analyze),
                    "apply": bool(apply),
                },
                "stats": {
                    "before": {
                        "file_bytes": 123456,
                        "page_count": 1000,
                        "freelist_count": 10,
                    }
                },
                "paths": {"json": f"db/{as_of.isoformat()}.json"},
            }

        orchestrator = SchedulerOrchestrator(
            settings=settings or self._make_settings(),
            output_dir=output_dir,
            run_premarket=run_premarket,
            run_intraday_check=run_intraday_check,
            run_eod=run_eod,
            run_review_cycle=run_review_cycle,
            ops_report=run_ops_report,
            health_check=run_health_check,
            run_guardrail_burnin=run_guardrail_burnin,
            run_guardrail_threshold_drift_audit=run_guardrail_drift,
            run_compact_executed_plans=run_compact_executed_plans,
            verify_compaction_restore=verify_compaction_restore,
            run_db_maintenance=run_db_maintenance,
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

    def test_run_halfhour_pulse_executes_due_slots_and_persists_state(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        out = orch.run_halfhour_pulse(
            as_of=d,
            slot="11:00",
            max_review_rounds=2,
            max_slot_runs=2,
            ops_every_n_pulses=10,
        )

        self.assertFalse(out["duplicate_pulse"])
        self.assertEqual(out["run_slots"], ["premarket", "intraday:10:30"])
        self.assertIn("premarket", out["due_slots"])
        self.assertIn("intraday:10:30", out["due_slots"])
        self.assertEqual(calls["premarket"], 1)
        self.assertEqual(calls["intraday"], ["10:30"])
        self.assertEqual(calls["ops"], [{"date": "2026-02-13", "window_days": 3}])

        state_path = td / "logs" / "halfhour_pulse_state.json"
        self.assertTrue(state_path.exists())
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["date"], "2026-02-13")
        self.assertIn("11:00", payload["executed_pulses"])
        self.assertIn("premarket", payload["executed_slots"])
        self.assertIn("intraday:10:30", payload["executed_slots"])

    def test_run_halfhour_pulse_emits_event_envelope(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, _calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00", max_slot_runs=1)

        event = out.get("event_envelope", {})
        self.assertTrue(bool(out.get("trace_id", "")))
        self.assertEqual(str(event.get("trace_id", "")), str(out.get("trace_id", "")))
        self.assertEqual(str(event.get("traceparent", "")), str(out.get("traceparent", "")))
        self.assertTrue(bool(re.match(r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$", str(out.get("traceparent", "")))))
        self.assertTrue(bool(event.get("event_id", "")))
        self.assertTrue(bool(event.get("payload_hash", "")))
        stream_path = Path(str(out.get("event_stream_path", "")))
        self.assertTrue(stream_path.exists())
        lines = [x for x in stream_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertGreaterEqual(len(lines), 2)  # start + completed

    def test_run_slot_review_forwards_trace_context_when_callback_supports_it(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, _calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)
        captured: list[dict[str, str]] = []

        def review_with_trace(
            as_of: date,
            max_rounds: int,
            trace_id: str | None = None,
            parent_event_id: str | None = None,
        ) -> dict[str, object]:
            captured.append(
                {
                    "date": as_of.isoformat(),
                    "max_rounds": str(max_rounds),
                    "trace_id": str(trace_id or ""),
                    "parent_event_id": str(parent_event_id or ""),
                }
            )
            return {"phase": "review", "ok": True}

        orch.run_review_cycle = review_with_trace
        out = orch.run_slot(
            as_of=d,
            slot="review",
            max_review_rounds=2,
            trace_id="trace_abc",
            parent_event_id="evt_parent",
        )

        self.assertEqual(out["slot"], "review")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["trace_id"], "trace_abc")
        self.assertEqual(captured[0]["parent_event_id"], "evt_parent")

    def test_run_halfhour_pulse_duplicate_skips_without_force(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        first = orch.run_halfhour_pulse(as_of=d, slot="11:00", ops_every_n_pulses=10)
        second = orch.run_halfhour_pulse(as_of=d, slot="11:00", ops_every_n_pulses=10)

        self.assertFalse(first["duplicate_pulse"])
        self.assertTrue(second["duplicate_pulse"])
        self.assertTrue(second["skipped"])
        self.assertEqual(calls["premarket"], 1)
        self.assertEqual(calls["intraday"], ["10:30"])

    def test_run_halfhour_pulse_dry_run_does_not_mutate_state(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00", dry_run=True)

        self.assertTrue(out["dry_run"])
        self.assertEqual(calls["premarket"], 0)
        self.assertEqual(calls["intraday"], [])
        self.assertEqual(calls["ops"], [])
        self.assertEqual(calls["guardrail_burnin"], [])
        self.assertEqual(calls["guardrail_drift"], [])
        self.assertFalse((td / "logs" / "halfhour_pulse_state.json").exists())

    def test_run_halfhour_pulse_skips_when_scheduler_lock_unavailable(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)
        d = date(2026, 2, 13)

        with patch.object(SchedulerOrchestrator, "_acquire_scheduler_lock", return_value=(None, "busy")):
            out = orch.run_halfhour_pulse(as_of=d, slot="11:00", dry_run=False)

        self.assertTrue(bool(out.get("skipped", False)))
        self.assertEqual(str(out.get("reason", "")), "scheduler_locked")
        self.assertEqual(str(out.get("lock_status", "")), "busy")
        self.assertEqual(calls["premarket"], 0)
        self.assertEqual(calls["intraday"], [])
        self.assertEqual(calls["ops"], [])
        self.assertFalse((td / "logs" / "halfhour_pulse_state.json").exists())

    def test_run_halfhour_pulse_runs_weekly_guardrail_once(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # 2026-02-13
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_burnin_days": 5,
                "ops_weekly_guardrail_drift_window_days": 35,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        first = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        second = orch.run_halfhour_pulse(as_of=d, slot="11:30")

        weekly_first = first.get("weekly_guardrail", {})
        weekly_second = second.get("weekly_guardrail", {})
        self.assertTrue(bool(weekly_first.get("ran", False)))
        self.assertEqual(str(weekly_first.get("status", "")), "ok")
        self.assertFalse(bool(weekly_second.get("ran", True)))
        self.assertEqual(str(weekly_second.get("reason", "")), "already_ran_this_week")
        self.assertEqual(
            calls["guardrail_burnin"],
            [{"date": "2026-02-13", "days": 5, "run_stable_replay": True, "auto_tune": True}],
        )
        self.assertEqual(calls["guardrail_drift"], [{"date": "2026-02-13", "window_days": 35}])
        self.assertEqual(
            calls["compact"],
            [
                {
                    "start": "2025-08-18",
                    "end": "2026-02-13",
                    "chunk_days": 30,
                    "dry_run": True,
                    "max_delete_rows": None,
                }
            ],
        )
        maintenance_first = weekly_first.get("maintenance", {}) if isinstance(weekly_first.get("maintenance", {}), dict) else {}
        compact_first = maintenance_first.get("compact", {}) if isinstance(maintenance_first.get("compact", {}), dict) else {}
        self.assertTrue(bool(compact_first.get("ran", False)))
        self.assertEqual(str(compact_first.get("status", "")), "ok")
        weekly_state = td / "logs" / "weekly_guardrail_state.json"
        self.assertTrue(weekly_state.exists())

    def test_run_halfhour_pulse_weekly_guardrail_fails_on_burnin_coverage_gap(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # 2026-02-13
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_burnin_days": 5,
                "ops_weekly_guardrail_require_burnin_coverage": True,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        def _burnin_with_gap(as_of: date, days: int, run_stable_replay: bool, auto_tune: bool) -> dict[str, object]:
            burnin_calls = calls["guardrail_burnin"]
            assert isinstance(burnin_calls, list)
            burnin_calls.append(
                {
                    "date": as_of.isoformat(),
                    "days": int(days),
                    "run_stable_replay": bool(run_stable_replay),
                    "auto_tune": bool(auto_tune),
                }
            )
            return {
                "summary": {
                    "active_days": 1,
                    "min_samples": int(days),
                    "coverage_ok": False,
                    "false_positive_ratio": 0.0,
                },
                "live_overrides": {"applied": False},
                "paths": {"json": f"burnin/{as_of.isoformat()}.json"},
            }

        orch.run_guardrail_burnin = _burnin_with_gap
        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {})
        burnin = weekly.get("burnin", {}) if isinstance(weekly.get("burnin", {}), dict) else {}
        self.assertTrue(bool(weekly.get("ran", False)))
        self.assertEqual(str(weekly.get("status", "")), "error")
        self.assertEqual(str(burnin.get("status", "")), "insufficient_coverage")
        self.assertFalse(bool(burnin.get("coverage_ok", True)))
        self.assertTrue(bool(burnin.get("coverage_required", False)))
        weekly_state = td / "logs" / "weekly_guardrail_state.json"
        self.assertTrue(weekly_state.exists())
        payload = json.loads(weekly_state.read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("last_status", "")), "error")

    def test_run_halfhour_pulse_weekly_guardrail_runs_compaction_restore_verify_when_apply_enabled(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # 2026-02-13
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_compact_enabled": True,
                "ops_weekly_guardrail_compact_dry_run": False,
                "ops_weekly_guardrail_compact_verify_restore": True,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {}) if isinstance(out.get("weekly_guardrail", {}), dict) else {}
        maintenance = weekly.get("maintenance", {}) if isinstance(weekly.get("maintenance", {}), dict) else {}
        compact = maintenance.get("compact", {}) if isinstance(maintenance.get("compact", {}), dict) else {}
        restore_verify = (
            maintenance.get("restore_verify", {})
            if isinstance(maintenance.get("restore_verify", {}), dict)
            else {}
        )

        self.assertTrue(bool(weekly.get("ran", False)))
        self.assertEqual(str(weekly.get("status", "")), "ok")
        self.assertEqual(len(calls["compact"]), 1)
        self.assertEqual(bool((calls["compact"][0] if calls["compact"] else {}).get("dry_run", True)), False)
        self.assertEqual(len(calls["restore_verify"]), 1)
        self.assertTrue(bool(compact.get("ran", False)))
        self.assertEqual(str(compact.get("status", "")), "ok")
        self.assertTrue(bool(restore_verify.get("ran", False)))
        self.assertEqual(str(restore_verify.get("status", "")), "ok")

    def test_weekly_guardrail_compaction_controlled_apply_promotes_after_stability_window(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # Friday
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_compact_enabled": True,
                "ops_weekly_guardrail_compact_dry_run": True,
                "ops_weekly_guardrail_compact_max_delete_rows": 80,
                "ops_weekly_guardrail_compact_verify_restore": True,
                "ops_weekly_guardrail_compact_controlled_apply_enabled": True,
                "ops_weekly_guardrail_compact_controlled_apply_stability_weeks": 2,
                "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks": 2,
                "ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows": 50,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        weekly_state = td / "logs" / "weekly_guardrail_state.json"
        weekly_state.parent.mkdir(parents=True, exist_ok=True)
        weekly_state.write_text(
            json.dumps(
                {
                    "last_run_week": "2026-W06",
                    "last_run_date": "2026-02-06",
                    "last_status": "ok",
                    "history": [
                        {
                            "date": "2026-01-30",
                            "week_tag": "2026-W05",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": True},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": False,
                                    "status": "skipped",
                                    "reason": "compact_dry_run",
                                },
                            },
                        },
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": True},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": False,
                                    "status": "skipped",
                                    "reason": "compact_dry_run",
                                },
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {}) if isinstance(out.get("weekly_guardrail", {}), dict) else {}
        maintenance = weekly.get("maintenance", {}) if isinstance(weekly.get("maintenance", {}), dict) else {}
        compact = maintenance.get("compact", {}) if isinstance(maintenance.get("compact", {}), dict) else {}
        compact_policy = compact.get("policy", {}) if isinstance(compact.get("policy", {}), dict) else {}

        self.assertEqual(len(calls["compact"]), 1)
        compact_call = calls["compact"][0] if isinstance(calls["compact"], list) and calls["compact"] else {}
        self.assertEqual(bool(compact_call.get("dry_run", True)), False)
        self.assertEqual(int(compact_call.get("max_delete_rows", 0)), 50)
        self.assertEqual(str(compact_policy.get("mode", "")), "controlled_apply")
        self.assertEqual(str(compact_policy.get("reason", "")), "cadence_due")
        self.assertEqual(len(calls["restore_verify"]), 1)

    def test_weekly_guardrail_compaction_controlled_apply_respects_cadence(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # Friday
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_compact_enabled": True,
                "ops_weekly_guardrail_compact_dry_run": True,
                "ops_weekly_guardrail_compact_verify_restore": True,
                "ops_weekly_guardrail_compact_controlled_apply_enabled": True,
                "ops_weekly_guardrail_compact_controlled_apply_stability_weeks": 2,
                "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks": 2,
                "ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows": 100,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        weekly_state = td / "logs" / "weekly_guardrail_state.json"
        weekly_state.parent.mkdir(parents=True, exist_ok=True)
        weekly_state.write_text(
            json.dumps(
                {
                    "last_run_week": "2026-W06",
                    "last_run_date": "2026-02-06",
                    "last_status": "ok",
                    "history": [
                        {
                            "date": "2026-01-30",
                            "week_tag": "2026-W05",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": True},
                                "restore_verify": {"enabled": True, "ran": False, "status": "skipped"},
                            },
                        },
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": False},
                                "restore_verify": {"enabled": True, "ran": True, "status": "ok"},
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {}) if isinstance(out.get("weekly_guardrail", {}), dict) else {}
        maintenance = weekly.get("maintenance", {}) if isinstance(weekly.get("maintenance", {}), dict) else {}
        compact = maintenance.get("compact", {}) if isinstance(maintenance.get("compact", {}), dict) else {}
        compact_policy = compact.get("policy", {}) if isinstance(compact.get("policy", {}), dict) else {}
        restore_verify = (
            maintenance.get("restore_verify", {})
            if isinstance(maintenance.get("restore_verify", {}), dict)
            else {}
        )

        self.assertEqual(len(calls["compact"]), 1)
        compact_call = calls["compact"][0] if isinstance(calls["compact"], list) and calls["compact"] else {}
        self.assertEqual(bool(compact_call.get("dry_run", False)), True)
        self.assertEqual(str(compact_policy.get("reason", "")), "cadence_not_due")
        self.assertEqual(str(restore_verify.get("status", "")), "skipped")
        self.assertEqual(str(restore_verify.get("reason", "")), "compact_dry_run")
        self.assertEqual(len(calls["restore_verify"]), 0)

    def test_weekly_guardrail_compaction_controlled_apply_waits_for_stability_window(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # Friday
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_compact_enabled": True,
                "ops_weekly_guardrail_compact_dry_run": True,
                "ops_weekly_guardrail_compact_verify_restore": True,
                "ops_weekly_guardrail_compact_controlled_apply_enabled": True,
                "ops_weekly_guardrail_compact_controlled_apply_stability_weeks": 2,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {}) if isinstance(out.get("weekly_guardrail", {}), dict) else {}
        maintenance = weekly.get("maintenance", {}) if isinstance(weekly.get("maintenance", {}), dict) else {}
        compact = maintenance.get("compact", {}) if isinstance(maintenance.get("compact", {}), dict) else {}
        compact_policy = compact.get("policy", {}) if isinstance(compact.get("policy", {}), dict) else {}

        self.assertEqual(len(calls["compact"]), 1)
        compact_call = calls["compact"][0] if isinstance(calls["compact"], list) and calls["compact"] else {}
        self.assertEqual(bool(compact_call.get("dry_run", False)), True)
        self.assertEqual(str(compact_policy.get("reason", "")), "stability_window_insufficient")

    def test_weekly_guardrail_compaction_controlled_apply_rejects_stale_stability_window(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # Friday
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_compact_enabled": True,
                "ops_weekly_guardrail_compact_dry_run": True,
                "ops_weekly_guardrail_compact_verify_restore": True,
                "ops_weekly_guardrail_compact_controlled_apply_enabled": True,
                "ops_weekly_guardrail_compact_controlled_apply_stability_weeks": 2,
                "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks": 2,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        weekly_state = td / "logs" / "weekly_guardrail_state.json"
        weekly_state.parent.mkdir(parents=True, exist_ok=True)
        weekly_state.write_text(
            json.dumps(
                {
                    "last_run_week": "2026-W02",
                    "last_run_date": "2026-01-09",
                    "last_status": "ok",
                    "history": [
                        {
                            "date": "2026-01-02",
                            "week_tag": "2026-W01",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": True},
                                "restore_verify": {"enabled": True, "ran": False, "status": "skipped"},
                            },
                        },
                        {
                            "date": "2026-01-09",
                            "week_tag": "2026-W02",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": True},
                                "restore_verify": {"enabled": True, "ran": False, "status": "skipped"},
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {}) if isinstance(out.get("weekly_guardrail", {}), dict) else {}
        maintenance = weekly.get("maintenance", {}) if isinstance(weekly.get("maintenance", {}), dict) else {}
        compact = maintenance.get("compact", {}) if isinstance(maintenance.get("compact", {}), dict) else {}
        compact_policy = compact.get("policy", {}) if isinstance(compact.get("policy", {}), dict) else {}

        self.assertEqual(len(calls["compact"]), 1)
        compact_call = calls["compact"][0] if isinstance(calls["compact"], list) and calls["compact"] else {}
        self.assertEqual(bool(compact_call.get("dry_run", False)), True)
        self.assertEqual(str(compact_policy.get("reason", "")), "stability_window_stale")

    def test_run_halfhour_pulse_weekly_guardrail_runs_db_maintenance_dry_run_by_default(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # 2026-02-13
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_compact_enabled": False,
                "ops_weekly_guardrail_db_maintenance_enabled": True,
                "ops_weekly_guardrail_db_maintenance_dry_run": True,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {}) if isinstance(out.get("weekly_guardrail", {}), dict) else {}
        maintenance = weekly.get("maintenance", {}) if isinstance(weekly.get("maintenance", {}), dict) else {}
        db_payload = (
            maintenance.get("db_maintenance", {})
            if isinstance(maintenance.get("db_maintenance", {}), dict)
            else {}
        )

        self.assertTrue(bool(weekly.get("ran", False)))
        self.assertEqual(str(weekly.get("status", "")), "ok")
        self.assertEqual(len(calls["db_maintenance"]), 1)
        db_call = (
            calls["db_maintenance"][0]
            if isinstance(calls["db_maintenance"], list) and calls["db_maintenance"]
            else {}
        )
        self.assertEqual(bool(db_call.get("apply", True)), False)
        self.assertTrue(bool(db_payload.get("ran", False)))
        self.assertEqual(str(db_payload.get("status", "")), "ok")
        self.assertEqual(bool(db_payload.get("dry_run", False)), True)

    def test_weekly_guardrail_db_maintenance_promotes_to_apply_with_compact_controlled_apply(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # Friday
                "ops_weekly_guardrail_trigger_slot": "11:00",
                "ops_weekly_guardrail_compact_enabled": True,
                "ops_weekly_guardrail_compact_dry_run": True,
                "ops_weekly_guardrail_compact_verify_restore": True,
                "ops_weekly_guardrail_compact_controlled_apply_enabled": True,
                "ops_weekly_guardrail_compact_controlled_apply_stability_weeks": 2,
                "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks": 2,
                "ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows": 50,
                "ops_weekly_guardrail_db_maintenance_enabled": True,
                "ops_weekly_guardrail_db_maintenance_dry_run": True,
                "ops_weekly_guardrail_db_maintenance_apply_with_compact_controlled_apply": True,
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        weekly_state = td / "logs" / "weekly_guardrail_state.json"
        weekly_state.parent.mkdir(parents=True, exist_ok=True)
        weekly_state.write_text(
            json.dumps(
                {
                    "last_run_week": "2026-W06",
                    "last_run_date": "2026-02-06",
                    "last_status": "ok",
                    "history": [
                        {
                            "date": "2026-01-30",
                            "week_tag": "2026-W05",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": True},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": False,
                                    "status": "skipped",
                                    "reason": "compact_dry_run",
                                },
                            },
                        },
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": True},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": False,
                                    "status": "skipped",
                                    "reason": "compact_dry_run",
                                },
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {}) if isinstance(out.get("weekly_guardrail", {}), dict) else {}
        maintenance = weekly.get("maintenance", {}) if isinstance(weekly.get("maintenance", {}), dict) else {}
        compact = maintenance.get("compact", {}) if isinstance(maintenance.get("compact", {}), dict) else {}
        compact_policy = compact.get("policy", {}) if isinstance(compact.get("policy", {}), dict) else {}
        db_payload = (
            maintenance.get("db_maintenance", {})
            if isinstance(maintenance.get("db_maintenance", {}), dict)
            else {}
        )
        db_policy = db_payload.get("policy", {}) if isinstance(db_payload.get("policy", {}), dict) else {}

        self.assertEqual(len(calls["compact"]), 1)
        compact_call = calls["compact"][0] if isinstance(calls["compact"], list) and calls["compact"] else {}
        self.assertEqual(bool(compact_call.get("dry_run", True)), False)
        self.assertEqual(str(compact_policy.get("mode", "")), "controlled_apply")
        self.assertEqual(str(compact_policy.get("reason", "")), "cadence_due")

        self.assertEqual(len(calls["db_maintenance"]), 1)
        db_call = (
            calls["db_maintenance"][0]
            if isinstance(calls["db_maintenance"], list) and calls["db_maintenance"]
            else {}
        )
        self.assertEqual(bool(db_call.get("apply", False)), True)
        self.assertEqual(bool(db_payload.get("dry_run", True)), False)
        self.assertEqual(str(db_policy.get("compact_policy_mode", "")), "controlled_apply")
        self.assertEqual(str(db_policy.get("compact_policy_reason", "")), "cadence_due")

    def test_run_halfhour_pulse_weekly_guardrail_waits_for_trigger_slot(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        schedule = {
            "premarket": "00:00",
            "intraday_slots": [],
            "eod": "23:59",
            "nightly_review": "20:30",
        }
        settings = self._make_settings(schedule=schedule)
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_weekly_guardrail_enabled": True,
                "ops_weekly_guardrail_weekday": 5,  # 2026-02-13
                "ops_weekly_guardrail_trigger_slot": "22:30",
            }
        )
        orch, calls = self._make_orchestrator(td, settings=settings)
        d = date(2026, 2, 13)

        out = orch.run_halfhour_pulse(as_of=d, slot="11:00")
        weekly = out.get("weekly_guardrail", {})
        self.assertFalse(bool(weekly.get("ran", True)))
        self.assertEqual(str(weekly.get("reason", "")), "before_trigger_slot")
        self.assertEqual(calls["guardrail_burnin"], [])
        self.assertEqual(calls["guardrail_drift"], [])

    def test_run_halfhour_daemon_executes_once_per_bucket(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)

        class _Now1105(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                base = datetime(2026, 2, 13, 11, 5, 0)
                return base.replace(tzinfo=tz) if tz is not None else base

        class _Now1120(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                base = datetime(2026, 2, 13, 11, 20, 0)
                return base.replace(tzinfo=tz) if tz is not None else base

        with patch("lie_engine.orchestration.scheduler.datetime", _Now1105):
            first = orch.run_halfhour_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=2)
        with patch("lie_engine.orchestration.scheduler.datetime", _Now1120):
            second = orch.run_halfhour_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=2)

        self.assertEqual(first["last_bucket"], "11:00")
        self.assertEqual(second["last_bucket"], "11:00")
        self.assertEqual(calls["premarket"], 1)
        self.assertEqual(calls["intraday"], ["10:30"])
        self.assertEqual(calls["ops"], [{"date": "2026-02-13", "window_days": 3}])
        history = second.get("history", [])
        self.assertTrue(isinstance(history, list) and history)
        row = history[-1]
        readiness = row.get("weekly_controlled_apply", {}) if isinstance(row.get("weekly_controlled_apply", {}), dict) else {}
        self.assertIn("stability_weeks", readiness)
        self.assertIn("cadence_due", readiness)
        self.assertIn("effective_delete_budget", readiness)
        self.assertTrue((td / "logs" / "halfhour_daemon_state.json").exists())

    def test_run_halfhour_daemon_dry_run_does_not_mutate_state(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)

        class _Now1105(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                base = datetime(2026, 2, 13, 11, 5, 0)
                return base.replace(tzinfo=tz) if tz is not None else base

        with patch("lie_engine.orchestration.scheduler.datetime", _Now1105):
            out = orch.run_halfhour_daemon(
                poll_seconds=1,
                max_cycles=0,
                max_review_rounds=2,
                dry_run=True,
            )

        self.assertTrue(out["dry_run"])
        self.assertTrue(out["would_run_pulse"])
        self.assertEqual(calls["premarket"], 0)
        self.assertEqual(calls["intraday"], [])
        self.assertEqual(calls["ops"], [])
        self.assertFalse((td / "logs" / "halfhour_daemon_state.json").exists())
        self.assertFalse((td / "logs" / "halfhour_pulse_state.json").exists())
        readiness = out.get("weekly_controlled_apply", {}) if isinstance(out.get("weekly_controlled_apply", {}), dict) else {}
        self.assertIn("stability_weeks", readiness)
        self.assertIn("cadence_due", readiness)
        self.assertIn("effective_delete_budget", readiness)

    def test_run_halfhour_daemon_does_not_advance_bucket_when_pulse_locked(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, calls = self._make_orchestrator(td)

        class _Now1105(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                base = datetime(2026, 2, 13, 11, 5, 0)
                return base.replace(tzinfo=tz) if tz is not None else base

        locked_payload = {
            "date": "2026-02-13",
            "pulse_slot": "11:00",
            "observed_slot": "11:05",
            "duplicate_pulse": True,
            "skipped": True,
            "reason": "scheduler_locked",
            "lock_status": "busy",
            "run_slots": [],
            "pending_slots": [],
            "slot_errors": [],
            "ops": {},
            "weekly_guardrail": {},
        }
        with patch("lie_engine.orchestration.scheduler.datetime", _Now1105):
            with patch.object(SchedulerOrchestrator, "run_halfhour_pulse", return_value=locked_payload):
                state = orch.run_halfhour_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=2)

        self.assertIsNone(state.get("last_bucket"))
        history = state.get("history", [])
        self.assertTrue(isinstance(history, list) and history)
        row = history[-1]
        self.assertEqual(str(row.get("status", "")), "locked")
        self.assertEqual(str(row.get("reason", "")), "scheduler_locked")
        self.assertEqual(str(row.get("lock_status", "")), "busy")
        self.assertEqual(calls["premarket"], 0)
        self.assertEqual(calls["intraday"], [])
        self.assertEqual(calls["ops"], [])

    def test_run_daemon_returns_locked_when_scheduler_lock_unavailable(self) -> None:
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

        with patch.object(SchedulerOrchestrator, "_acquire_scheduler_lock", return_value=(None, "busy")):
            out = orch.run_daemon(poll_seconds=1, max_cycles=0, max_review_rounds=1, dry_run=False)

        self.assertTrue(bool(out.get("locked", False)))
        self.assertEqual(str(out.get("reason", "")), "scheduler_locked")
        self.assertEqual(str(out.get("lock_status", "")), "busy")
        self.assertEqual(calls["premarket"], 0)
        self.assertEqual(calls["intraday"], [])
        self.assertEqual(calls["ops"], [])

    def test_run_autorun_retro_writes_report_and_aggregates_metrics(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, _ = self._make_orchestrator(td)
        d = date(2026, 2, 13)
        logs = td / "logs"
        review = td / "review"
        logs.mkdir(parents=True, exist_ok=True)
        review.mkdir(parents=True, exist_ok=True)

        (logs / "halfhour_pulse_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "ts": "2026-02-13T11:00:00+08:00",
                            "slot_errors": [],
                            "ops": {"ran": True, "status": "ok"},
                            "health": {"ran": True, "status": "healthy"},
                            "weekly_guardrail": {"ran": True, "status": "ok"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (logs / "halfhour_daemon_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "ts": "2026-02-13T11:05:00+08:00",
                            "duplicate_pulse": False,
                            "slot_errors": [],
                            "ops": {"ran": True, "status": "ok"},
                            "weekly_guardrail": {"ran": False, "status": "skipped"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (logs / "weekly_guardrail_state.json").write_text(
            json.dumps({"history": [{"date": "2026-02-13", "status": "ok"}]}),
            encoding="utf-8",
        )
        (logs / "guard_state.json").write_text(json.dumps({"consecutive_bad": 0}), encoding="utf-8")
        (logs / "guard_loop_last.json").write_text(
            json.dumps({"recovery": {"mode": "light", "status": "ok"}}),
            encoding="utf-8",
        )
        (logs / "guard_loop_history.jsonl").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T11:30:00+08:00",
                    "pulse": {"ran": True, "status": "ran"},
                    "health": {"status": "ok"},
                    "recovery": {"mode": "light", "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (logs / "tests_20260213_113000.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")
        (review / "2026-02-13_gate_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
        (review / "2026-02-13_ops_report.json").write_text(json.dumps({"status": "green"}), encoding="utf-8")

        out = orch.run_autorun_retro(as_of=d, window_days=3)

        self.assertEqual(str(out.get("status")), "green")
        self.assertEqual(int(out["summary"]["pulse_entries"]), 1)
        self.assertEqual(int(out["summary"]["daemon_entries"]), 1)
        self.assertEqual(int(out["summary"]["weekly_entries"]), 1)
        self.assertEqual(int(out["summary"]["guard_loop_entries"]), 1)
        self.assertEqual(int(out["metrics"]["guard_recovery_light"]), 1)
        self.assertEqual(int(out["metrics"]["guard_recovery_errors"]), 0)
        self.assertTrue(Path(str(out["paths"]["json"])).exists())
        self.assertTrue(Path(str(out["paths"]["md"])).exists())

    def test_run_autorun_retro_marks_yellow_on_timeline_gap(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, _ = self._make_orchestrator(td)
        d = date(2026, 2, 13)
        logs = td / "logs"
        review = td / "review"
        logs.mkdir(parents=True, exist_ok=True)
        review.mkdir(parents=True, exist_ok=True)

        (logs / "halfhour_pulse_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "ts": "2026-02-13T01:00:00+08:00",
                            "slot_errors": [],
                            "ops": {"ran": False, "status": "skipped"},
                            "health": {"ran": True, "status": "healthy"},
                            "weekly_guardrail": {"ran": False, "status": "skipped"},
                        },
                        {
                            "ts": "2026-02-13T02:10:00+08:00",
                            "slot_errors": [],
                            "ops": {"ran": True, "status": "ok"},
                            "health": {"ran": True, "status": "healthy"},
                            "weekly_guardrail": {"ran": False, "status": "skipped"},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (logs / "halfhour_daemon_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {"ts": "2026-02-13T01:05:00+08:00", "duplicate_pulse": False, "slot_errors": []},
                        {"ts": "2026-02-13T02:05:00+08:00", "duplicate_pulse": False, "slot_errors": []},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (logs / "guard_loop_history.jsonl").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T02:30:00+08:00",
                    "pulse": {"ran": True, "status": "ok"},
                    "health": {"status": "healthy", "core_missing": []},
                    "recovery": {"mode": "none", "status": "skipped"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (logs / "tests_20260213_060000.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")
        (review / "2026-02-13_gate_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
        (review / "2026-02-13_ops_report.json").write_text(json.dumps({"status": "green"}), encoding="utf-8")

        out = orch.run_autorun_retro(as_of=d, window_days=1)

        self.assertEqual(str(out.get("status")), "yellow")
        self.assertGreater(int(out["metrics"]["pulse_gap_events_over_45m"]), 0)
        findings = [str(x) for x in out.get("findings", [])]
        self.assertTrue(any("时间断档" in item for item in findings))

    def test_run_autorun_retro_includes_cadence_lift_trend_deltas(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, _ = self._make_orchestrator(td)
        d = date(2026, 2, 13)
        logs = td / "logs"
        review = td / "review"
        logs.mkdir(parents=True, exist_ok=True)
        review.mkdir(parents=True, exist_ok=True)

        (logs / "halfhour_pulse_state.json").write_text(
            json.dumps({"history": [{"ts": "2026-02-13T11:00:00+08:00", "slot_errors": []}]}),
            encoding="utf-8",
        )
        (logs / "halfhour_daemon_state.json").write_text(
            json.dumps({"history": [{"ts": "2026-02-13T11:05:00+08:00", "duplicate_pulse": False, "slot_errors": []}]}),
            encoding="utf-8",
        )
        (logs / "guard_loop_history.jsonl").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T11:30:00+08:00",
                    "pulse": {"ran": True, "status": "ok"},
                    "health": {"status": "healthy", "core_missing": []},
                    "recovery": {"mode": "none", "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (logs / "tests_20260213_113000.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")

        trend_series = []
        for idx in range(5):
            trend_series.append(
                {
                    "date": f"2026-02-0{idx+4}",
                    "requested": True,
                    "applied": True,
                    "blocked_by_cooldown": False,
                    "requested_level": "soft",
                    "applied_level": "soft",
                    "reason_code": "guard_loop_cadence_non_apply_lift_soft",
                }
            )
        for idx in range(5):
            trend_series.append(
                {
                    "date": f"2026-02-1{idx}",
                    "requested": True,
                    "applied": False,
                    "blocked_by_cooldown": True,
                    "requested_level": "hard",
                    "applied_level": "none",
                    "reason_code": "guard_loop_cadence_non_apply_lift_cooldown",
                }
            )

        (review / "2026-02-13_gate_report.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "guard_loop_cadence_non_apply_lift_trend": {
                        "active": True,
                        "alerts": [],
                        "metrics": {"applied_rate": 0.5, "cooldown_block_rate": 0.5},
                        "series": trend_series,
                    },
                }
            ),
            encoding="utf-8",
        )
        (review / "2026-02-13_ops_report.json").write_text(json.dumps({"status": "green"}), encoding="utf-8")

        out = orch.run_autorun_retro(as_of=d, window_days=1)

        self.assertEqual(str(out.get("status")), "yellow")
        self.assertAlmostEqual(float(out["metrics"]["cadence_lift_applied_rate_delta"]), -1.0, places=6)
        self.assertAlmostEqual(float(out["metrics"]["cadence_lift_cooldown_block_rate_delta"]), 1.0, places=6)
        trend = out.get("cadence_lift_trend", {}) if isinstance(out.get("cadence_lift_trend", {}), dict) else {}
        self.assertTrue(bool(trend.get("active", False)))
        self.assertEqual(int(trend.get("recent_samples", 0)), 5)
        self.assertEqual(int(trend.get("prior_samples", 0)), 5)
        findings = [str(x) for x in out.get("findings", [])]
        self.assertTrue(any("应用率出现下行漂移" in item for item in findings))
        self.assertTrue(any("cooldown 阻塞率上升" in item for item in findings))
        md = Path(str(out["paths"]["md"])).read_text(encoding="utf-8")
        self.assertIn("Cadence Lift Trend Delta", md)

    def test_run_autorun_retro_marks_red_on_recovery_and_test_errors(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        orch, _ = self._make_orchestrator(td)
        d = date(2026, 2, 13)
        logs = td / "logs"
        logs.mkdir(parents=True, exist_ok=True)

        (logs / "guard_loop_history.jsonl").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T11:30:00+08:00",
                    "pulse": {"ran": False, "status": "skipped"},
                    "health": {"status": "degraded", "core_missing": ["daily_positions"]},
                    "recovery": {"mode": "heavy", "status": "error"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (logs / "tests_20260213_223000.json").write_text(json.dumps({"returncode": 1}), encoding="utf-8")

        out = orch.run_autorun_retro(as_of=d, window_days=1)

        self.assertEqual(str(out.get("status")), "red")
        self.assertEqual(int(out["metrics"]["guard_recovery_errors"]), 1)
        self.assertEqual(int(out["metrics"]["guard_health_core_degraded"]), 1)
        findings = [str(x) for x in out.get("findings", [])]
        self.assertTrue(any("test-all" in item for item in findings))


if __name__ == "__main__":
    unittest.main()
