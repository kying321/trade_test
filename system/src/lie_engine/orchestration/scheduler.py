from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import json
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

    def run_slot(self, as_of: date, slot: str, max_review_rounds: int = 2) -> dict[str, Any]:
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

    def run_session(self, as_of: date, include_review: bool = True, max_review_rounds: int = 2) -> dict[str, Any]:
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
                return {"date": None, "executed": [], "history": []}
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                return {"date": None, "executed": [], "history": []}

        def save_state(st: dict[str, Any]) -> None:
            state_path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

        state = load_state()
        cycles = 0

        if dry_run:
            now = datetime.now(tz)
            day_str = now.date().isoformat()
            if state.get("date") != day_str:
                state = {"date": day_str, "executed": [], "history": []}
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
            if state.get("date") != day_str:
                state = {"date": day_str, "executed": [], "history": []}

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
                        result = self.run_slot(as_of=now.date(), slot=slot_input, max_review_rounds=max_review_rounds)
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

            if max_cycles is not None and cycles >= max_cycles:
                break

            cycles += 1
            time_module.sleep(max(1, int(poll_seconds)))

        return state
