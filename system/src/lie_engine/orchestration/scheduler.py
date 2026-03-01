from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import time as time_module
from typing import Any, Callable
from zoneinfo import ZoneInfo

from lie_engine.config import SystemSettings


@dataclass(slots=True)
class SchedulerOrchestrator:
    settings: SystemSettings
    output_dir: Path
    run_premarket: Callable[[date], dict[str, Any]]
    run_intraday_check: Callable[[date, str], dict[str, Any]]
    run_eod: Callable[[date], dict[str, Any]]
    run_review_cycle: Callable[[date, int], dict[str, Any]]
    ops_report: Callable[[date, int], dict[str, Any]]
    run_micro_capture: Callable[[date, list[str] | None], dict[str, Any]] | None = None
    _mutex_fd: Any = field(init=False, default=None, repr=False)
    _mutex_depth: int = field(init=False, default=0, repr=False)

    def _schedule_cfg(self) -> dict[str, Any]:
        schedule = self.settings.schedule
        return {
            "premarket": str(schedule.get("premarket", "08:40")),
            "intraday_slots": [str(x) for x in schedule.get("intraday_slots", ["10:30", "14:30"])],
            "eod": str(schedule.get("eod", "15:10")),
            "nightly_review": str(schedule.get("nightly_review", "20:30")),
        }

    @staticmethod
    def _parse_hhmm(slot: str) -> time:
        hh, mm = slot.split(":")
        return time(hour=int(hh), minute=int(mm))

    def _daemon_slots(self) -> list[dict[str, str]]:
        cfg = self._schedule_cfg()
        slots: list[dict[str, str]] = [{"id": "premarket", "slot": "premarket", "trigger": cfg["premarket"]}]
        for slot in cfg["intraday_slots"]:
            slots.append({"id": f"intraday:{slot}", "slot": slot, "trigger": slot})
        slots.append({"id": "eod", "slot": "eod", "trigger": cfg["eod"]})
        slots.append({"id": "review", "slot": "review", "trigger": cfg["nightly_review"]})
        return slots

    def _micro_capture_cfg(self) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("micro_capture_daemon_enabled", False))
        interval_minutes = max(5, int(val.get("micro_capture_daemon_interval_minutes", 30)))
        raw_symbols = val.get("micro_capture_daemon_symbols", [])
        symbols: list[str] = []
        if isinstance(raw_symbols, (list, tuple)):
            symbols = [str(x).strip().upper() for x in raw_symbols if str(x).strip()]
        elif isinstance(raw_symbols, str):
            symbols = [x.strip().upper() for x in raw_symbols.split(",") if x.strip()]
        return {
            "enabled": bool(enabled),
            "interval_minutes": int(interval_minutes),
            "symbols": symbols,
        }

    def _mutex_path(self) -> Path:
        path = self.output_dir / "state" / "run-halfhour-pulse.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _mutex_timeout_seconds(self) -> float:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        try:
            timeout = float(val.get("run_halfhour_pulse_mutex_timeout_seconds", 5.0))
        except Exception:
            timeout = 5.0
        return float(min(60.0, max(1.0, timeout)))

    @contextmanager
    def _run_halfhour_mutex(self, owner: str):
        self._mutex_depth += 1
        if self._mutex_depth > 1:
            try:
                yield
            finally:
                self._mutex_depth = max(0, self._mutex_depth - 1)
            return

        lock_path = self._mutex_path()
        fd = lock_path.open("a+", encoding="utf-8")
        timeout_seconds = self._mutex_timeout_seconds()
        deadline = time_module.monotonic() + timeout_seconds
        acquired = False
        try:
            while True:
                try:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time_module.monotonic() >= deadline:
                        raise TimeoutError(f"run-halfhour-pulse mutex timeout: {timeout_seconds:.1f}s")
                    time_module.sleep(0.05)
            payload = {
                "owner": str(owner),
                "pid": int(os.getpid()),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            fd.seek(0)
            fd.truncate()
            fd.write(json.dumps(payload, ensure_ascii=False, indent=2))
            fd.flush()
            os.fsync(fd.fileno())
            self._mutex_fd = fd
            yield
        finally:
            try:
                if acquired:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            finally:
                try:
                    fd.close()
                finally:
                    self._mutex_fd = None
                    self._mutex_depth = max(0, self._mutex_depth - 1)

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime | None:
        txt = str(value or "").strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(txt)
        except Exception:
            return None

    def _run_slot_unlocked(self, as_of: date, slot: str, max_review_rounds: int = 2) -> dict[str, Any]:
        cfg = self._schedule_cfg()
        slot = str(slot)

        if slot in {"premarket", cfg["premarket"]}:
            return {"slot": "premarket", "result": self.run_premarket(as_of)}

        if slot in {"eod", cfg["eod"]}:
            return {"slot": "eod", "result": self.run_eod(as_of)}

        if slot in {"review", cfg["nightly_review"]}:
            review_cycle = self.run_review_cycle(as_of, max_review_rounds)
            return {"slot": "review", "result": review_cycle}

        if slot in {"ops", "ops-report"}:
            replay_days = int(self.settings.validation.get("required_stable_replay_days", 3))
            ops = self.ops_report(as_of, replay_days)
            return {"slot": "ops", "result": ops}

        if slot in {"micro-capture", "micro_capture", "microcapture"}:
            if not callable(self.run_micro_capture):
                raise ValueError("micro-capture runner unavailable")
            cfg_mc = self._micro_capture_cfg()
            symbols = cfg_mc.get("symbols", [])
            out = self.run_micro_capture(as_of, symbols if isinstance(symbols, list) and symbols else None)
            return {"slot": "micro-capture", "result": out}

        intraday_slots = cfg["intraday_slots"]
        if slot in intraday_slots:
            return {"slot": f"intraday:{slot}", "result": self.run_intraday_check(as_of, slot)}

        if slot in {"intraday_1", "intraday-1"} and intraday_slots:
            target = intraday_slots[0]
            return {"slot": f"intraday:{target}", "result": self.run_intraday_check(as_of, target)}

        if slot in {"intraday_2", "intraday-2"} and len(intraday_slots) >= 2:
            target = intraday_slots[1]
            return {"slot": f"intraday:{target}", "result": self.run_intraday_check(as_of, target)}

        raise ValueError(f"Unsupported slot: {slot}")

    def run_slot(self, as_of: date, slot: str, max_review_rounds: int = 2) -> dict[str, Any]:
        owner = f"run-slot:{slot}:{as_of.isoformat()}"
        with self._run_halfhour_mutex(owner):
            return self._run_slot_unlocked(as_of=as_of, slot=slot, max_review_rounds=max_review_rounds)

    def _run_session_unlocked(self, as_of: date, include_review: bool = True, max_review_rounds: int = 2) -> dict[str, Any]:
        cfg = self._schedule_cfg()
        out: dict[str, Any] = {"date": as_of.isoformat(), "schedule": cfg, "steps": {}}

        out["steps"]["premarket"] = self.run_premarket(as_of)
        for slot in cfg["intraday_slots"]:
            out["steps"][f"intraday_{slot}"] = self.run_intraday_check(as_of, slot)
        out["steps"]["eod"] = self.run_eod(as_of)

        if include_review:
            out["steps"]["review_cycle"] = self.run_review_cycle(as_of, max_review_rounds)
        else:
            out["steps"]["review_cycle"] = {"skipped": True}
        return out

    def run_session(self, as_of: date, include_review: bool = True, max_review_rounds: int = 2) -> dict[str, Any]:
        owner = f"run-session:{as_of.isoformat()}"
        with self._run_halfhour_mutex(owner):
            return self._run_session_unlocked(as_of=as_of, include_review=include_review, max_review_rounds=max_review_rounds)

    def run_daemon(
        self,
        poll_seconds: int = 30,
        max_cycles: int | None = None,
        max_review_rounds: int = 2,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        slots = self._daemon_slots()
        tz = ZoneInfo(self.settings.timezone)
        state_path = self.output_dir / "logs" / "scheduler_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        def load_state() -> dict[str, Any]:
            if not state_path.exists():
                return {"date": None, "executed": [], "history": [], "interval_tasks": {}}
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                return {"date": None, "executed": [], "history": [], "interval_tasks": {}}

        def save_state(st: dict[str, Any]) -> None:
            state_path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

        state = load_state()
        cycles = 0
        mc_cfg = self._micro_capture_cfg()
        mc_enabled = bool(mc_cfg.get("enabled", False) and callable(self.run_micro_capture))
        mc_interval_minutes = int(mc_cfg.get("interval_minutes", 30))
        mc_symbols = mc_cfg.get("symbols", [])
        mc_slot_id = "interval:micro_capture"

        if dry_run:
            now = datetime.now(tz)
            day_str = now.date().isoformat()
            if state.get("date") != day_str:
                state = {"date": day_str, "executed": [], "history": [], "interval_tasks": state.get("interval_tasks", {})}
            executed_set = set(str(x) for x in state.get("executed", []))
            slots_view: list[dict[str, Any]] = []
            would_execute: list[str] = []
            for slot_cfg in slots:
                slot_id = slot_cfg["id"]
                slot_input = slot_cfg["slot"]
                trigger = slot_cfg["trigger"]
                already_executed = slot_id in executed_set or trigger in executed_set
                due = now.time() >= self._parse_hhmm(trigger)
                slots_view.append(
                    {
                        "slot_id": slot_id,
                        "slot": slot_input,
                        "trigger_time": trigger,
                        "due": due,
                        "already_executed": already_executed,
                    }
                )
                if due and not already_executed:
                    would_execute.append(slot_id)

            interval_tasks = state.get("interval_tasks", {}) if isinstance(state.get("interval_tasks", {}), dict) else {}
            mc_state = interval_tasks.get("micro_capture", {}) if isinstance(interval_tasks.get("micro_capture", {}), dict) else {}
            last_run_ts = self._parse_iso_datetime(mc_state.get("last_run_ts", ""))
            due_mc = bool(mc_enabled and (last_run_ts is None or now >= (last_run_ts + timedelta(minutes=mc_interval_minutes))))
            next_due = (
                now.isoformat()
                if (mc_enabled and last_run_ts is None)
                else (last_run_ts + timedelta(minutes=mc_interval_minutes)).isoformat()
                if (mc_enabled and last_run_ts is not None)
                else None
            )
            slots_view.append(
                {
                    "slot_id": mc_slot_id,
                    "slot": "micro-capture",
                    "trigger_time": f"every_{mc_interval_minutes}m",
                    "due": bool(due_mc),
                    "already_executed": False,
                    "enabled": bool(mc_enabled),
                    "last_run_ts": last_run_ts.isoformat() if isinstance(last_run_ts, datetime) else None,
                    "next_due_ts": next_due,
                }
            )
            if due_mc:
                would_execute.append(mc_slot_id)

            return {
                "date": day_str,
                "dry_run": True,
                "now": now.isoformat(),
                "state_path": str(state_path),
                "executed": state.get("executed", []),
                "would_execute": would_execute,
                "slots": slots_view,
            }

        while True:
            now = datetime.now(tz)
            day_str = now.date().isoformat()
            with self._run_halfhour_mutex(f"run-daemon-cycle:{now.isoformat()}"):
                state = load_state()
                if state.get("date") != day_str:
                    state = {
                        "date": day_str,
                        "executed": [],
                        "history": [],
                        "interval_tasks": state.get("interval_tasks", {}),
                    }

                executed_set = set(str(x) for x in state.get("executed", []))
                for slot_cfg in slots:
                    slot_id = slot_cfg["id"]
                    slot_input = slot_cfg["slot"]
                    trigger = slot_cfg["trigger"]

                    # Backward compatibility: older state may have stored raw HH:MM in executed.
                    if slot_id in executed_set or trigger in executed_set:
                        continue
                    if now.time() >= self._parse_hhmm(trigger):
                        try:
                            result = self._run_slot_unlocked(as_of=now.date(), slot=slot_input, max_review_rounds=max_review_rounds)
                            status = "ok"
                        except Exception as exc:
                            result = {"error": str(exc)}
                            status = "error"
                        state["executed"].append(slot_id)
                        state["history"].append(
                            {
                                "ts": now.isoformat(),
                                "slot_id": slot_id,
                                "slot": slot_input,
                                "trigger_time": trigger,
                                "status": status,
                                "result": result,
                            }
                        )
                        save_state(state)

                interval_tasks = state.get("interval_tasks", {}) if isinstance(state.get("interval_tasks", {}), dict) else {}
                mc_state = interval_tasks.get("micro_capture", {}) if isinstance(interval_tasks.get("micro_capture", {}), dict) else {}
                last_run_ts = self._parse_iso_datetime(mc_state.get("last_run_ts", ""))
                mc_due = bool(mc_enabled and (last_run_ts is None or now >= (last_run_ts + timedelta(minutes=mc_interval_minutes))))
                if mc_due:
                    try:
                        result = self.run_micro_capture(now.date(), mc_symbols if isinstance(mc_symbols, list) and mc_symbols else None)  # type: ignore[misc]
                        status = "ok"
                    except Exception as exc:
                        result = {"error": str(exc)}
                        status = "error"
                    interval_tasks["micro_capture"] = {
                        "enabled": bool(mc_enabled),
                        "interval_minutes": int(mc_interval_minutes),
                        "last_run_ts": now.isoformat(),
                        "status": status,
                        "symbols": mc_symbols if isinstance(mc_symbols, list) else [],
                    }
                    state["interval_tasks"] = interval_tasks
                    state["history"].append(
                        {
                            "ts": now.isoformat(),
                            "slot_id": mc_slot_id,
                            "slot": "micro-capture",
                            "trigger_time": f"every_{mc_interval_minutes}m",
                            "status": status,
                            "result": result,
                        }
                    )
                    save_state(state)

            if max_cycles is not None and cycles >= max_cycles:
                break

            cycles += 1
            time_module.sleep(max(1, int(poll_seconds)))

        return state
