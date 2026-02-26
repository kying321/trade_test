from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
import time as time_module
from typing import IO
from typing import Any, Callable
from zoneinfo import ZoneInfo

from lie_engine.config import SystemSettings
from lie_engine.orchestration.events import append_event_envelope, build_event_envelope, derive_trace_id

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None


@dataclass(slots=True)
class SchedulerOrchestrator:
    settings: SystemSettings
    output_dir: Path
    run_premarket: Callable[[date], dict[str, Any]]
    run_intraday_check: Callable[[date, str], dict[str, Any]]
    run_eod: Callable[[date], dict[str, Any]]
    run_review_cycle: Callable[..., dict[str, Any]]
    ops_report: Callable[[date, int], dict[str, Any]]
    health_check: Callable[[date, bool], dict[str, Any]] | None = None
    run_guardrail_burnin: Callable[[date, int, bool, bool], dict[str, Any]] | None = None
    run_guardrail_threshold_drift_audit: Callable[[date, int], dict[str, Any]] | None = None
    run_compact_executed_plans: Callable[[date, date, int, bool, int | None], dict[str, Any]] | None = None
    verify_compaction_restore: Callable[[str | None, bool], dict[str, Any]] | None = None
    run_db_maintenance: Callable[[date, int | None, list[str] | None, bool, bool, bool], dict[str, Any]] | None = None

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

    @staticmethod
    def _format_hhmm(t: time) -> str:
        return f"{int(t.hour):02d}:{int(t.minute):02d}"

    @staticmethod
    def _pulse_bucket(t: time) -> time:
        minute = 30 if int(t.minute) >= 30 else 0
        return time(hour=int(t.hour), minute=minute)

    def _daemon_slots(self) -> list[dict[str, str]]:
        cfg = self._schedule_cfg()
        slots: list[dict[str, str]] = [{"id": "premarket", "slot": "premarket", "trigger": cfg["premarket"]}]
        for slot in cfg["intraday_slots"]:
            slots.append({"id": f"intraday:{slot}", "slot": slot, "trigger": slot})
        slots.append({"id": "eod", "slot": "eod", "trigger": cfg["eod"]})
        slots.append({"id": "review", "slot": "review", "trigger": cfg["nightly_review"]})
        return slots

    def _pulse_state_path(self) -> Path:
        return self.output_dir / "logs" / "halfhour_pulse_state.json"

    def _halfhour_daemon_state_path(self) -> Path:
        return self.output_dir / "logs" / "halfhour_daemon_state.json"

    def _scheduler_lock_path(self) -> Path:
        return self.output_dir / "logs" / "scheduler_exec.lock"

    def _weekly_guardrail_state_path(self) -> Path:
        return self.output_dir / "logs" / "weekly_guardrail_state.json"

    @staticmethod
    def _safe_bool(v: Any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float)):
            return bool(v)
        txt = str(v).strip().lower()
        if txt in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if txt in {"0", "false", "f", "no", "n", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _acquire_scheduler_lock(self) -> tuple[IO[str] | None, str]:
        path = self._scheduler_lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_fp: IO[str] | None = None
        try:
            lock_fp = path.open("a+", encoding="utf-8")
            if fcntl is None:
                return lock_fp, "unsupported"
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fp, "ok"
        except Exception as exc:
            if lock_fp is not None:
                try:
                    lock_fp.close()
                except Exception:
                    pass
            return None, f"{type(exc).__name__}:{exc}"

    def _release_scheduler_lock(self, lock_fp: IO[str] | None) -> None:
        if lock_fp is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_fp.close()
        except Exception:
            pass

    def _invoke_review_cycle(
        self,
        *,
        as_of: date,
        max_review_rounds: int,
        trace_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return self.run_review_cycle(
                as_of,
                max_review_rounds,
                trace_id=trace_id,
                parent_event_id=parent_event_id,
            )
        except TypeError:
            return self.run_review_cycle(as_of, max_review_rounds)

    def _emit_scheduler_event(
        self,
        *,
        as_of: date,
        source: str,
        event_type: str,
        payload: dict[str, Any],
        trace_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        envelope = build_event_envelope(
            source=source,
            event_type=event_type,
            payload=payload,
            as_of=as_of,
            trace_id=trace_id,
            parent_event_id=parent_event_id,
        )
        stream_path = append_event_envelope(
            output_dir=self.output_dir,
            envelope=envelope,
            payload=payload,
        )
        return envelope.to_dict(), str(stream_path)

    def _weekly_guardrail_cfg(self) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        schedule_cfg = self._schedule_cfg()
        trigger_slot_raw = str(
            val.get("ops_weekly_guardrail_trigger_slot", schedule_cfg.get("nightly_review", "20:30"))
        ).strip()
        try:
            trigger_slot = self._format_hhmm(self._parse_hhmm(trigger_slot_raw))
        except Exception:
            trigger_slot = str(schedule_cfg.get("nightly_review", "20:30"))
        weekday = int(val.get("ops_weekly_guardrail_weekday", 7))
        weekday = min(7, max(1, weekday))
        burnin_days = max(1, int(val.get("ops_weekly_guardrail_burnin_days", 7)))
        drift_window_days = max(7, int(val.get("ops_weekly_guardrail_drift_window_days", max(28, burnin_days * 4))))
        compact_max_delete_rows_raw = val.get("ops_weekly_guardrail_compact_max_delete_rows", None)
        compact_max_delete_rows = None
        if compact_max_delete_rows_raw not in {None, "", "none", "None", 0}:
            try:
                compact_max_delete_rows = max(1, int(compact_max_delete_rows_raw))
            except Exception:
                compact_max_delete_rows = None
        compact_controlled_budget_raw = val.get(
            "ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows",
            25000,
        )
        try:
            compact_controlled_budget = max(1, int(compact_controlled_budget_raw))
        except Exception:
            compact_controlled_budget = 25000
        db_tables_raw = val.get(
            "ops_weekly_guardrail_db_maintenance_tables",
            [
                "bars_normalized",
                "macro",
                "source_confidence",
                "quality",
                "signals",
                "trade_plans",
                "latest_positions",
                "executed_plans",
                "review_runs",
                "backtest_runs",
            ],
        )
        db_tables: list[str] = []
        if isinstance(db_tables_raw, list):
            db_tables = [str(x).strip() for x in db_tables_raw if str(x).strip()]
        elif isinstance(db_tables_raw, str):
            db_tables = [x.strip() for x in db_tables_raw.split(",") if x.strip()]
        db_retention_raw = val.get("ops_weekly_guardrail_db_maintenance_retention_days", 365)
        try:
            db_retention_days = max(1, int(db_retention_raw))
        except Exception:
            db_retention_days = 365
        return {
            "enabled": self._safe_bool(val.get("ops_weekly_guardrail_enabled", True), True),
            "weekday": int(weekday),
            "trigger_slot": trigger_slot,
            "burnin_days": int(burnin_days),
            "drift_window_days": int(drift_window_days),
            "auto_tune": self._safe_bool(val.get("ops_weekly_guardrail_auto_tune", True), True),
            "run_stable_replay": self._safe_bool(val.get("ops_weekly_guardrail_run_stable_replay", True), True),
            "require_ops_window": self._safe_bool(val.get("ops_weekly_guardrail_require_ops_window", True), True),
            "require_burnin_coverage": self._safe_bool(
                val.get("ops_weekly_guardrail_require_burnin_coverage", True),
                True,
            ),
            "compact": {
                "enabled": self._safe_bool(val.get("ops_weekly_guardrail_compact_enabled", True), True),
                "dry_run": self._safe_bool(val.get("ops_weekly_guardrail_compact_dry_run", True), True),
                "window_days": max(7, int(val.get("ops_weekly_guardrail_compact_window_days", 180))),
                "chunk_days": max(1, int(val.get("ops_weekly_guardrail_compact_chunk_days", 30))),
                "max_delete_rows": compact_max_delete_rows,
                "verify_restore": self._safe_bool(
                    val.get("ops_weekly_guardrail_compact_verify_restore", True),
                    True,
                ),
                "verify_keep_temp_db": self._safe_bool(
                    val.get("ops_weekly_guardrail_compact_verify_keep_temp_db", False),
                    False,
                ),
                "controlled_apply": {
                    "enabled": self._safe_bool(
                        val.get("ops_weekly_guardrail_compact_controlled_apply_enabled", True),
                        True,
                    ),
                    "stability_weeks": max(
                        1,
                        int(
                            val.get(
                                "ops_weekly_guardrail_compact_controlled_apply_stability_weeks",
                                2,
                            )
                        ),
                    ),
                    "cadence_weeks": max(
                        1,
                        int(
                            val.get(
                                "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks",
                                2,
                            )
                        ),
                    ),
                    "delete_budget_rows": int(compact_controlled_budget),
                    "require_restore_verify_pass": self._safe_bool(
                        val.get(
                            "ops_weekly_guardrail_compact_controlled_apply_require_restore_verify_pass",
                            True,
                        ),
                        True,
                    ),
                    "require_weekly_status_ok": self._safe_bool(
                        val.get(
                            "ops_weekly_guardrail_compact_controlled_apply_require_weekly_status_ok",
                            True,
                        ),
                        True,
                    ),
                },
            },
            "db_maintenance": {
                "enabled": self._safe_bool(val.get("ops_weekly_guardrail_db_maintenance_enabled", True), True),
                "dry_run": self._safe_bool(val.get("ops_weekly_guardrail_db_maintenance_dry_run", True), True),
                "retention_days": int(db_retention_days),
                "tables": list(db_tables),
                "vacuum": self._safe_bool(val.get("ops_weekly_guardrail_db_maintenance_vacuum", True), True),
                "analyze": self._safe_bool(val.get("ops_weekly_guardrail_db_maintenance_analyze", True), True),
                "apply_with_compact_controlled_apply": self._safe_bool(
                    val.get("ops_weekly_guardrail_db_maintenance_apply_with_compact_controlled_apply", True),
                    True,
                ),
                "size_warn_bytes": max(
                    1,
                    int(
                        self._safe_float(
                            val.get("ops_weekly_guardrail_db_maintenance_size_warn_bytes", 1_500_000_000),
                            1_500_000_000.0,
                        )
                    ),
                ),
                "freelist_warn_ratio": max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self._safe_float(
                                val.get("ops_weekly_guardrail_db_maintenance_freelist_warn_ratio", 0.20),
                                0.20,
                            )
                        ),
                    ),
                ),
                "eligible_rows_warn": max(
                    0,
                    int(
                        self._safe_float(
                            val.get("ops_weekly_guardrail_db_maintenance_eligible_rows_warn", 250000),
                            250000.0,
                        )
                    ),
                ),
            },
        }

    @staticmethod
    def _iso_week_tag(d: date) -> str:
        iso_year, iso_week, _ = d.isocalendar()
        return f"{int(iso_year)}-W{int(iso_week):02d}"

    @staticmethod
    def _week_tag_to_monday(week_tag: str) -> date | None:
        txt = str(week_tag or "").strip()
        if not txt or "-W" not in txt:
            return None
        try:
            y_raw, w_raw = txt.split("-W", 1)
            iso_year = int(y_raw)
            iso_week = int(w_raw)
            return date.fromisocalendar(iso_year, iso_week, 1)
        except Exception:
            return None

    def _weeks_between_tags(self, older: str, newer: str) -> int | None:
        older_day = self._week_tag_to_monday(older)
        newer_day = self._week_tag_to_monday(newer)
        if older_day is None or newer_day is None:
            return None
        delta_days = (newer_day - older_day).days
        if delta_days < 0:
            return None
        return int(delta_days // 7)

    def _compact_policy_decision(
        self,
        *,
        as_of: date,
        state_history: list[dict[str, Any]],
        compact_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        base_dry_run = bool(compact_cfg.get("dry_run", True))
        configured_max_delete_rows = compact_cfg.get("max_delete_rows", None)
        controlled = (
            compact_cfg.get("controlled_apply", {})
            if isinstance(compact_cfg.get("controlled_apply", {}), dict)
            else {}
        )
        controlled_enabled = bool(controlled.get("enabled", False))
        stability_weeks = max(1, int(self._safe_float(controlled.get("stability_weeks", 2), 2)))
        cadence_weeks = max(1, int(self._safe_float(controlled.get("cadence_weeks", 2), 2)))
        delete_budget_rows = max(1, int(self._safe_float(controlled.get("delete_budget_rows", 25000), 25000)))
        require_restore_verify_pass = bool(controlled.get("require_restore_verify_pass", True))
        require_weekly_status_ok = bool(controlled.get("require_weekly_status_ok", True))

        out: dict[str, Any] = {
            "mode": "dry_run",
            "reason": "base_dry_run",
            "base_dry_run": bool(base_dry_run),
            "controlled_apply_enabled": bool(controlled_enabled),
            "effective_dry_run": bool(base_dry_run),
            "configured_max_delete_rows": configured_max_delete_rows,
            "effective_max_delete_rows": configured_max_delete_rows,
            "stability_weeks": int(stability_weeks),
            "cadence_weeks": int(cadence_weeks),
            "delete_budget_rows": int(delete_budget_rows),
            "stable_window_passed": False,
            "stable_weeks_count": 0,
            "last_apply_week": "",
            "weeks_since_last_apply": None,
            "history_considered": 0,
            "history_compact_runs": 0,
            "history_recency_weeks": None,
        }
        if not base_dry_run:
            out.update(
                {
                    "mode": "forced_apply",
                    "reason": "base_apply_config",
                    "effective_dry_run": False,
                    "stable_window_passed": True,
                }
            )
            return out
        if not controlled_enabled:
            out["reason"] = "controlled_apply_disabled"
            return out

        compact_rows: list[dict[str, Any]] = []
        for entry in state_history:
            if not isinstance(entry, dict):
                continue
            maintenance = (
                entry.get("maintenance", {})
                if isinstance(entry.get("maintenance", {}), dict)
                else {}
            )
            compact = (
                maintenance.get("compact", {})
                if isinstance(maintenance.get("compact", {}), dict)
                else {}
            )
            if not bool(compact.get("ran", False)):
                continue
            week_tag = str(entry.get("week_tag", "")).strip()
            if not week_tag:
                parsed_date = self._parse_date_like(entry.get("date"))
                if parsed_date is not None:
                    week_tag = self._iso_week_tag(parsed_date)
            if not week_tag:
                continue
            restore_verify = (
                maintenance.get("restore_verify", {})
                if isinstance(maintenance.get("restore_verify", {}), dict)
                else {}
            )
            compact_status = str(compact.get("status", "")).strip().lower()
            weekly_status = str(entry.get("status", "")).strip().lower()
            restore_status = str(restore_verify.get("status", "")).strip().lower()
            compact_dry_run = bool(compact.get("dry_run", True))
            restore_enabled = bool(restore_verify.get("enabled", False))
            restore_pass = (not restore_enabled) or restore_status == "ok"
            if compact_dry_run:
                restore_pass = True
            stable = compact_status == "ok"
            if require_weekly_status_ok:
                stable = stable and weekly_status == "ok"
            if require_restore_verify_pass:
                stable = stable and bool(restore_pass)
            compact_rows.append(
                {
                    "week_tag": week_tag,
                    "stable": bool(stable),
                    "dry_run": bool(compact_dry_run),
                    "status": compact_status,
                    "weekly_status": weekly_status,
                }
            )

        out["history_compact_runs"] = int(len(compact_rows))
        if not compact_rows:
            out["reason"] = "stability_window_insufficient"
            out["history_considered"] = 0
            return out

        compact_rows.sort(
            key=lambda row: self._week_tag_to_monday(str(row.get("week_tag", ""))) or date.min,
        )
        current_week_tag = self._iso_week_tag(as_of)
        recent_rows = compact_rows[-stability_weeks:]
        out["history_considered"] = int(len(recent_rows))
        stable_window_passed = bool(len(recent_rows) >= stability_weeks and all(bool(r.get("stable", False)) for r in recent_rows))
        out["stable_window_passed"] = bool(stable_window_passed)
        out["stable_weeks_count"] = int(sum(1 for row in recent_rows if bool(row.get("stable", False))))
        if not stable_window_passed:
            out["reason"] = "stability_window_unstable" if len(recent_rows) >= stability_weeks else "stability_window_insufficient"
            return out

        oldest_recent_week = str((recent_rows[0] if recent_rows else {}).get("week_tag", "")).strip()
        history_recency_weeks: int | None = None
        if oldest_recent_week:
            history_recency_weeks = self._weeks_between_tags(oldest_recent_week, current_week_tag)
        out["history_recency_weeks"] = history_recency_weeks
        if history_recency_weeks is None or history_recency_weeks > stability_weeks:
            out["reason"] = "stability_window_stale"
            return out

        last_apply_week = ""
        for row in reversed(compact_rows):
            if not bool(row.get("dry_run", True)) and str(row.get("status", "")).lower() == "ok":
                last_apply_week = str(row.get("week_tag", "")).strip()
                break
        out["last_apply_week"] = last_apply_week
        weeks_since_last_apply: int | None = None
        if last_apply_week:
            weeks_since_last_apply = self._weeks_between_tags(last_apply_week, current_week_tag)
        out["weeks_since_last_apply"] = weeks_since_last_apply
        if weeks_since_last_apply is not None and weeks_since_last_apply < cadence_weeks:
            out["reason"] = "cadence_not_due"
            return out

        effective_max_delete_rows = int(delete_budget_rows)
        if configured_max_delete_rows is not None:
            try:
                effective_max_delete_rows = min(effective_max_delete_rows, max(1, int(configured_max_delete_rows)))
            except Exception:
                effective_max_delete_rows = int(delete_budget_rows)

        out.update(
            {
                "mode": "controlled_apply",
                "reason": "cadence_due",
                "effective_dry_run": False,
                "effective_max_delete_rows": int(effective_max_delete_rows),
            }
        )
        return out

    def _controlled_apply_readiness(
        self,
        *,
        as_of: date,
        state_history: list[dict[str, Any]],
        compact_cfg: dict[str, Any],
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_policy = policy if isinstance(policy, dict) else self._compact_policy_decision(
            as_of=as_of,
            state_history=state_history,
            compact_cfg=compact_cfg,
        )
        mode = str(resolved_policy.get("mode", "dry_run")).strip() or "dry_run"
        reason = str(resolved_policy.get("reason", "")).strip() or "unknown"
        raw_budget = resolved_policy.get(
            "effective_max_delete_rows",
            resolved_policy.get(
                "delete_budget_rows",
                compact_cfg.get("max_delete_rows", None),
            ),
        )
        effective_delete_budget: int | None = None
        if raw_budget not in {None, "", "none", "None"}:
            try:
                effective_delete_budget = max(1, int(raw_budget))
            except Exception:
                effective_delete_budget = None

        cadence_due = bool(mode == "controlled_apply" and reason == "cadence_due")
        return {
            "enabled": bool(resolved_policy.get("controlled_apply_enabled", False)),
            "mode": mode,
            "reason": reason,
            "stability_weeks": int(self._safe_float(resolved_policy.get("stability_weeks", 0), 0.0)),
            "cadence_weeks": int(self._safe_float(resolved_policy.get("cadence_weeks", 0), 0.0)),
            "cadence_due": bool(cadence_due),
            "effective_delete_budget": effective_delete_budget,
            "stable_window_passed": bool(resolved_policy.get("stable_window_passed", False)),
            "stable_weeks_count": int(self._safe_float(resolved_policy.get("stable_weeks_count", 0), 0.0)),
            "history_considered": int(self._safe_float(resolved_policy.get("history_considered", 0), 0.0)),
            "history_recency_weeks": (
                None
                if resolved_policy.get("history_recency_weeks") in {None, ""}
                else int(self._safe_float(resolved_policy.get("history_recency_weeks", 0), 0.0))
            ),
            "weeks_since_last_apply": (
                None
                if resolved_policy.get("weeks_since_last_apply") in {None, ""}
                else int(self._safe_float(resolved_policy.get("weeks_since_last_apply", 0), 0.0))
            ),
            "policy": resolved_policy,
        }

    def _weekly_controlled_apply_snapshot(self, *, as_of: date) -> dict[str, Any]:
        cfg = self._weekly_guardrail_cfg()
        compact_cfg = cfg.get("compact", {}) if isinstance(cfg.get("compact", {}), dict) else {}
        state_default: dict[str, Any] = {
            "last_run_week": None,
            "last_run_date": None,
            "last_status": None,
            "history": [],
        }
        state = self._load_state_json(self._weekly_guardrail_state_path(), state_default)
        history_raw = state.get("history", [])
        history = [entry for entry in history_raw if isinstance(entry, dict)] if isinstance(history_raw, list) else []
        return self._controlled_apply_readiness(
            as_of=as_of,
            state_history=history,
            compact_cfg=compact_cfg,
        )

    def _run_weekly_guardrail(
        self,
        *,
        as_of: date,
        observed_time: time,
        force: bool,
        dry_run: bool,
        ops_planned: bool,
    ) -> dict[str, Any]:
        cfg = self._weekly_guardrail_cfg()
        week_tag = self._iso_week_tag(as_of)
        state_path = self._weekly_guardrail_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_default: dict[str, Any] = {
            "last_run_week": None,
            "last_run_date": None,
            "last_status": None,
            "history": [],
        }
        state = self._load_state_json(state_path, state_default)
        history_raw = state.get("history", [])
        history = [entry for entry in history_raw if isinstance(entry, dict)] if isinstance(history_raw, list) else []
        compact_cfg = cfg.get("compact", {}) if isinstance(cfg.get("compact", {}), dict) else {}
        db_cfg = cfg.get("db_maintenance", {}) if isinstance(cfg.get("db_maintenance", {}), dict) else {}
        controlled_apply_readiness = self._controlled_apply_readiness(
            as_of=as_of,
            state_history=history,
            compact_cfg=compact_cfg,
        )

        callbacks_ready = self.run_guardrail_burnin is not None and self.run_guardrail_threshold_drift_audit is not None
        due = False
        reason = "unknown"
        if not bool(cfg["enabled"]):
            reason = "disabled"
        elif not callbacks_ready:
            reason = "callbacks_missing"
        elif str(state.get("last_run_week") or "") == week_tag and (not force):
            reason = "already_ran_this_week"
        elif bool(cfg["require_ops_window"]) and (not ops_planned) and (not force):
            reason = "ops_window_not_planned"
        elif (not force) and int(as_of.isoweekday()) != int(cfg["weekday"]):
            reason = "not_target_weekday"
        elif (not force) and self._parse_hhmm(str(cfg["trigger_slot"])) > observed_time:
            reason = "before_trigger_slot"
        else:
            due = True
            reason = "due"

        out: dict[str, Any] = {
            "enabled": bool(cfg["enabled"]),
            "callbacks_ready": bool(callbacks_ready),
            "due": bool(due),
            "ran": False,
            "status": "skipped",
            "reason": reason,
            "week_tag": week_tag,
            "state_path": str(state_path),
            "config": cfg,
            "controlled_apply_readiness": controlled_apply_readiness,
        }
        if not due or dry_run:
            return out

        burnin_payload: dict[str, Any] = {"ran": False}
        drift_payload: dict[str, Any] = {"ran": False}
        status = "ok"
        coverage_required = bool(cfg.get("require_burnin_coverage", True))
        if self.run_guardrail_burnin is not None:
            try:
                burnin = self.run_guardrail_burnin(
                    as_of,
                    int(cfg["burnin_days"]),
                    bool(cfg["run_stable_replay"]),
                    bool(cfg["auto_tune"]),
                )
                burnin_summary = burnin.get("summary", {}) if isinstance(burnin.get("summary", {}), dict) else {}
                burnin_live = burnin.get("live_overrides", {}) if isinstance(burnin.get("live_overrides", {}), dict) else {}
                low_cost_summary = (
                    burnin_summary.get("low_cost_replay", {})
                    if isinstance(burnin_summary.get("low_cost_replay", {}), dict)
                    else {}
                )
                low_cost_budget_audit = (
                    burnin.get("low_cost_replay_budget_audit", {})
                    if isinstance(burnin.get("low_cost_replay_budget_audit", {}), dict)
                    else (
                        burnin_summary.get("low_cost_replay_budget_audit", {})
                        if isinstance(burnin_summary.get("low_cost_replay_budget_audit", {}), dict)
                        else {}
                    )
                )
                try:
                    burnin_active_days = int(burnin_summary.get("active_days", 0))
                except Exception:
                    burnin_active_days = 0
                try:
                    burnin_min_samples = int(burnin_summary.get("min_samples", cfg.get("burnin_days", 1)))
                except Exception:
                    burnin_min_samples = int(cfg.get("burnin_days", 1))
                burnin_min_samples = max(1, int(burnin_min_samples))
                burnin_coverage_ok = bool(
                    burnin_summary.get("coverage_ok", burnin_active_days >= burnin_min_samples)
                )
                burnin_payload = {
                    "ran": True,
                    "status": "ok",
                    "paths": burnin.get("paths", {}),
                    "active_days": int(burnin_active_days),
                    "min_samples": int(burnin_min_samples),
                    "coverage_ok": bool(burnin_coverage_ok),
                    "coverage_required": bool(coverage_required),
                    "false_positive_ratio": float(burnin_summary.get("false_positive_ratio", 0.0)),
                    "low_cost_replay": {
                        "enabled": bool(low_cost_summary.get("enabled", False)),
                        "max_days_per_run": int(self._safe_float(low_cost_summary.get("max_days_per_run", 0), 0.0)),
                        "attempted_days": int(self._safe_float(low_cost_summary.get("attempted_days", 0), 0.0)),
                        "succeeded_days": int(self._safe_float(low_cost_summary.get("succeeded_days", 0), 0.0)),
                        "failed_days": int(self._safe_float(low_cost_summary.get("failed_days", 0), 0.0)),
                        "budget_skipped_days": int(
                            self._safe_float(low_cost_summary.get("budget_skipped_days", 0), 0.0)
                        ),
                        "recovery_ratio": (
                            float(low_cost_summary.get("recovery_ratio"))
                            if low_cost_summary.get("recovery_ratio") is not None
                            else None
                        ),
                        "failure_ratio": (
                            float(low_cost_summary.get("failure_ratio"))
                            if low_cost_summary.get("failure_ratio") is not None
                            else None
                        ),
                    },
                    "low_cost_replay_budget_audit": low_cost_budget_audit,
                    "live_override_applied": bool(burnin_live.get("applied", False)),
                }
                if coverage_required and (not burnin_coverage_ok):
                    status = "error"
                    burnin_payload["status"] = "insufficient_coverage"
                    burnin_payload["error"] = (
                        f"active_days={int(burnin_active_days)} below min_samples={int(burnin_min_samples)}"
                    )
            except Exception as exc:
                status = "error"
                burnin_payload = {"ran": True, "status": "error", "error": str(exc)}

        if self.run_guardrail_threshold_drift_audit is not None:
            try:
                drift = self.run_guardrail_threshold_drift_audit(as_of, int(cfg["drift_window_days"]))
                drift_payload = {
                    "ran": True,
                    "status": "ok",
                    "paths": drift.get("paths", {}),
                    "audit_status": str(drift.get("status", "unknown")),
                    "alerts": list(drift.get("alerts", [])) if isinstance(drift.get("alerts", []), list) else [],
                }
            except Exception as exc:
                status = "error"
                drift_payload = {"ran": True, "status": "error", "error": str(exc)}

        compact_enabled = bool(compact_cfg.get("enabled", False))
        compact_payload: dict[str, Any] = {"enabled": bool(compact_enabled), "ran": False}
        compact_policy: dict[str, Any] = {}
        restore_verify_payload: dict[str, Any] = {
            "enabled": bool(compact_cfg.get("verify_restore", False)),
            "ran": False,
        }
        db_enabled = bool(db_cfg.get("enabled", False))
        db_payload: dict[str, Any] = {"enabled": bool(db_enabled), "ran": False}
        maintenance_payload: dict[str, Any] = {
            "ran": False,
            "compact": compact_payload,
            "restore_verify": restore_verify_payload,
            "db_maintenance": db_payload,
        }
        if compact_enabled:
            maintenance_payload["ran"] = True
            if self.run_compact_executed_plans is None:
                compact_payload.update({"status": "skipped", "reason": "callback_missing"})
            else:
                compact_start = as_of - timedelta(days=max(1, int(compact_cfg.get("window_days", 180))) - 1)
                compact_end = as_of
                compact_policy = self._compact_policy_decision(
                    as_of=as_of,
                    state_history=history,
                    compact_cfg=compact_cfg,
                )
                controlled_apply_readiness = self._controlled_apply_readiness(
                    as_of=as_of,
                    state_history=history,
                    compact_cfg=compact_cfg,
                    policy=compact_policy,
                )
                compact_payload["policy"] = compact_policy
                compact_dry_run = bool(compact_policy.get("effective_dry_run", compact_cfg.get("dry_run", True)))
                compact_chunk_days = max(1, int(compact_cfg.get("chunk_days", 30)))
                compact_max_delete_rows = compact_policy.get(
                    "effective_max_delete_rows",
                    compact_cfg.get("max_delete_rows", None),
                )
                try:
                    compact_result = self.run_compact_executed_plans(
                        compact_start,
                        compact_end,
                        compact_chunk_days,
                        compact_dry_run,
                        compact_max_delete_rows if compact_max_delete_rows is None else int(compact_max_delete_rows),
                    )
                    compact_metrics = (
                        compact_result.get("metrics", {})
                        if isinstance(compact_result.get("metrics", {}), dict)
                        else {}
                    )
                    compact_payload.update(
                        {
                            "ran": True,
                            "status": str(compact_result.get("status", "unknown")),
                            "reason": str(compact_result.get("reason", "")),
                            "run_id": str(compact_result.get("run_id", "")),
                            "dry_run": bool(compact_result.get("dry_run", compact_dry_run)),
                            "window": compact_result.get(
                                "window",
                                {"start": compact_start.isoformat(), "end": compact_end.isoformat()},
                            ),
                            "metrics": compact_metrics,
                            "paths": compact_result.get("paths", {}),
                            "rollback": compact_result.get("rollback", {}),
                        }
                    )
                    if str(compact_payload.get("status", "")).lower() == "error":
                        status = "error"
                except Exception as exc:
                    status = "error"
                    compact_payload.update({"ran": True, "status": "error", "reason": f"exception:{exc}"})

            verify_enabled = bool(compact_cfg.get("verify_restore", False))
            restore_verify_payload["enabled"] = bool(verify_enabled)
            if verify_enabled:
                compact_status = str(compact_payload.get("status", "unknown")).lower()
                compact_ran = bool(compact_payload.get("ran", False))
                compact_dry_run = bool(compact_payload.get("dry_run", compact_cfg.get("dry_run", True)))
                if not compact_ran:
                    restore_verify_payload.update({"status": "skipped", "reason": "compact_not_ran"})
                elif compact_status == "error":
                    restore_verify_payload.update({"status": "skipped", "reason": "compact_error"})
                elif compact_dry_run:
                    restore_verify_payload.update({"status": "skipped", "reason": "compact_dry_run"})
                elif self.verify_compaction_restore is None:
                    status = "error"
                    restore_verify_payload.update({"status": "error", "reason": "callback_missing"})
                else:
                    try:
                        restore_result = self.verify_compaction_restore(
                            str(compact_payload.get("run_id", "")).strip() or None,
                            bool(compact_cfg.get("verify_keep_temp_db", False)),
                        )
                        restore_verify_payload.update(
                            {
                                "ran": True,
                                "status": str(restore_result.get("status", "unknown")),
                                "reason": str(restore_result.get("reason", "")),
                                "run_id": str(restore_result.get("run_id", "")),
                                "checks": (
                                    restore_result.get("checks", {})
                                    if isinstance(restore_result.get("checks", {}), dict)
                                    else {}
                                ),
                                "metrics": (
                                    restore_result.get("metrics", {})
                                    if isinstance(restore_result.get("metrics", {}), dict)
                                    else {}
                                ),
                                "paths": restore_result.get("paths", {}),
                            }
                        )
                        if str(restore_verify_payload.get("status", "")).lower() == "error":
                            status = "error"
                    except Exception as exc:
                        status = "error"
                        restore_verify_payload.update({"ran": True, "status": "error", "reason": f"exception:{exc}"})
            else:
                restore_verify_payload.update({"status": "skipped", "reason": "verify_disabled"})

        if db_enabled:
            maintenance_payload["ran"] = True
            if self.run_db_maintenance is None:
                db_payload.update({"status": "skipped", "reason": "callback_missing"})
            else:
                db_base_dry_run = bool(db_cfg.get("dry_run", True))
                db_apply_with_compact = bool(db_cfg.get("apply_with_compact_controlled_apply", True))
                db_effective_dry_run = bool(db_base_dry_run)
                if (not db_base_dry_run):
                    db_effective_dry_run = False
                elif db_apply_with_compact:
                    compact_policy_mode = str(compact_policy.get("mode", "")).strip().lower()
                    compact_reason = str(compact_policy.get("reason", "")).strip().lower()
                    compact_ok = str(compact_payload.get("status", "")).strip().lower() == "ok"
                    compact_apply_due = compact_policy_mode == "controlled_apply" and compact_reason == "cadence_due"
                    if compact_apply_due and compact_ok:
                        db_effective_dry_run = False
                db_tables = db_cfg.get("tables", [])
                if not isinstance(db_tables, list):
                    db_tables = []
                try:
                    db_result = self.run_db_maintenance(
                        as_of,
                        int(self._safe_float(db_cfg.get("retention_days", 365), 365.0)),
                        [str(x).strip() for x in db_tables if str(x).strip()] or None,
                        bool(db_cfg.get("vacuum", True)),
                        bool(db_cfg.get("analyze", True)),
                        not bool(db_effective_dry_run),
                    )
                    stats_payload = db_result.get("stats", {}) if isinstance(db_result.get("stats", {}), dict) else {}
                    stats_before = (
                        stats_payload.get("before", {})
                        if isinstance(stats_payload.get("before", {}), dict)
                        else {}
                    )
                    retention = (
                        db_result.get("retention", {})
                        if isinstance(db_result.get("retention", {}), dict)
                        else {}
                    )
                    vacuum = (
                        db_result.get("vacuum", {})
                        if isinstance(db_result.get("vacuum", {}), dict)
                        else {}
                    )
                    alerts: list[str] = []
                    size_warn_bytes = int(self._safe_float(db_cfg.get("size_warn_bytes", 1_500_000_000), 1_500_000_000.0))
                    freelist_warn_ratio = float(self._safe_float(db_cfg.get("freelist_warn_ratio", 0.20), 0.20))
                    eligible_rows_warn = int(self._safe_float(db_cfg.get("eligible_rows_warn", 250000), 250000.0))
                    file_bytes = int(self._safe_float(stats_before.get("file_bytes", 0), 0.0))
                    page_count = int(self._safe_float(stats_before.get("page_count", 0), 0.0))
                    freelist_count = int(self._safe_float(stats_before.get("freelist_count", 0), 0.0))
                    freelist_ratio = (float(freelist_count) / float(page_count)) if page_count > 0 else 0.0
                    eligible_rows = int(self._safe_float(retention.get("eligible_rows", 0), 0.0))
                    if file_bytes >= max(1, size_warn_bytes):
                        alerts.append("sqlite_size_warn")
                    if freelist_ratio >= max(0.0, min(1.0, freelist_warn_ratio)):
                        alerts.append("sqlite_freelist_warn")
                    if eligible_rows >= max(0, eligible_rows_warn):
                        alerts.append("sqlite_retention_backlog_warn")
                    db_status = "ok"
                    if str(retention.get("status", "")).strip().lower() == "error":
                        db_status = "error"
                    if str(vacuum.get("status", "")).strip().lower() == "error":
                        db_status = "error"
                    db_payload.update(
                        {
                            "ran": True,
                            "status": db_status,
                            "reason": str(retention.get("reason", "")),
                            "dry_run": bool(db_effective_dry_run),
                            "retention_days": int(self._safe_float(db_cfg.get("retention_days", 365), 365.0)),
                            "alerts": alerts,
                            "paths": db_result.get("paths", {}),
                            "stats_before": {
                                "file_bytes": int(file_bytes),
                                "page_count": int(page_count),
                                "freelist_count": int(freelist_count),
                                "freelist_ratio": float(freelist_ratio),
                            },
                            "retention": {
                                "eligible_rows": int(eligible_rows),
                                "deleted_rows": int(self._safe_float(retention.get("deleted_rows", 0), 0.0)),
                                "status": str(retention.get("status", "")),
                            },
                            "policy": {
                                "base_dry_run": bool(db_base_dry_run),
                                "effective_dry_run": bool(db_effective_dry_run),
                                "apply_with_compact_controlled_apply": bool(db_apply_with_compact),
                                "compact_policy_mode": str(compact_policy.get("mode", "")),
                                "compact_policy_reason": str(compact_policy.get("reason", "")),
                            },
                        }
                    )
                    if db_status == "error":
                        status = "error"
                except Exception as exc:
                    status = "error"
                    db_payload.update({"ran": True, "status": "error", "reason": f"exception:{exc}"})

        out.update(
            {
                "ran": True,
                "status": status,
                "reason": "executed",
                "burnin": burnin_payload,
                "drift_audit": drift_payload,
                "maintenance": maintenance_payload,
                "controlled_apply_readiness": controlled_apply_readiness,
            }
        )

        state["last_run_week"] = week_tag
        state["last_run_date"] = as_of.isoformat()
        state["last_status"] = status
        history = state.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "date": as_of.isoformat(),
                "week_tag": week_tag,
                "status": status,
                "burnin": burnin_payload,
                "drift_audit": drift_payload,
                "maintenance": maintenance_payload,
                "controlled_apply_readiness": controlled_apply_readiness,
            }
        )
        history_limit = max(24, int(self.settings.validation.get("weekly_guardrail_history_limit", 104)))
        state["history"] = history[-history_limit:]
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def _load_state_json(path: Path, default_payload: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default_payload)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default_payload)
        if not isinstance(payload, dict):
            return dict(default_payload)
        merged = dict(default_payload)
        merged.update(payload)
        return merged

    @staticmethod
    def _parse_date_like(raw: Any) -> date | None:
        txt = str(raw or "").strip()
        if not txt:
            return None
        try:
            return date.fromisoformat(txt)
        except Exception:
            pass
        try:
            return datetime.fromisoformat(txt).date()
        except Exception:
            pass
        if len(txt) >= 10:
            try:
                return date.fromisoformat(txt[:10])
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_event_epoch_seconds(raw: Any) -> float | None:
        if isinstance(raw, (int, float)):
            v = float(raw)
            if v <= 0:
                return None
            if v > 1_000_000_000_000:
                return v / 1000.0
            return v

        txt = str(raw or "").strip()
        if not txt:
            return None

        if txt.isdigit():
            try:
                v = float(txt)
                if v <= 0:
                    return None
                if v > 1_000_000_000_000:
                    return v / 1000.0
                return v
            except Exception:
                return None

        try:
            parsed = txt[:-1] + "+00:00" if txt.endswith("Z") else txt
            return float(datetime.fromisoformat(parsed).timestamp())
        except Exception:
            pass

        d = SchedulerOrchestrator._parse_date_like(txt)
        if d is None:
            return None
        try:
            return float(datetime.combine(d, time.min).timestamp())
        except Exception:
            return None

    def _compute_timeline_gap_metrics(
        self,
        entries: list[dict[str, Any]],
        *,
        threshold_minutes: float,
    ) -> dict[str, Any]:
        threshold = max(1.0, float(threshold_minutes))
        epochs: list[float] = []
        for row in entries:
            if not isinstance(row, dict):
                continue
            raw_ts: Any = row.get("ts")
            if raw_ts in {None, ""}:
                raw_ts = row.get("date")
            if raw_ts in {None, ""}:
                daemon = row.get("daemon", {})
                if isinstance(daemon, dict):
                    raw_ts = daemon.get("ts") or daemon.get("date")
            epoch = self._parse_event_epoch_seconds(raw_ts)
            if epoch is not None:
                epochs.append(float(epoch))

        if len(epochs) < 2:
            return {"event_count": len(epochs), "max_gap_minutes": 0.0, "gap_events_over_threshold": 0}

        gaps_minutes: list[float] = []
        sorted_epochs = sorted(epochs)
        for prev, cur in zip(sorted_epochs, sorted_epochs[1:]):
            if cur >= prev:
                gaps_minutes.append((cur - prev) / 60.0)

        if not gaps_minutes:
            return {"event_count": len(epochs), "max_gap_minutes": 0.0, "gap_events_over_threshold": 0}

        max_gap = max(gaps_minutes)
        over = sum(1 for x in gaps_minutes if x > threshold)
        return {
            "event_count": len(epochs),
            "max_gap_minutes": round(float(max_gap), 2),
            "gap_events_over_threshold": int(over),
        }

    def run_autorun_retro(self, as_of: date, window_days: int = 7) -> dict[str, Any]:
        d = as_of.isoformat()
        wd = max(1, int(window_days))
        start = as_of - timedelta(days=wd - 1)

        logs_dir = self.output_dir / "logs"
        review_dir = self.output_dir / "review"
        logs_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

        pulse_state = self._load_state_json(
            self._pulse_state_path(),
            {"date": None, "history": [], "executed_pulses": [], "executed_slots": []},
        )
        daemon_state = self._load_state_json(
            self._halfhour_daemon_state_path(),
            {"date": None, "history": [], "last_bucket": None, "cycles": 0},
        )
        weekly_state = self._load_state_json(
            self._weekly_guardrail_state_path(),
            {"last_run_week": None, "last_run_date": None, "last_status": None, "history": []},
        )
        guard_state = self._load_state_json(logs_dir / "guard_state.json", {})
        guard_last = self._load_state_json(logs_dir / "guard_loop_last.json", {})

        pulse_history_raw = pulse_state.get("history", [])
        daemon_history_raw = daemon_state.get("history", [])
        weekly_history_raw = weekly_state.get("history", [])
        pulse_history = pulse_history_raw if isinstance(pulse_history_raw, list) else []
        daemon_history = daemon_history_raw if isinstance(daemon_history_raw, list) else []
        weekly_history = weekly_history_raw if isinstance(weekly_history_raw, list) else []

        pulse_entries: list[dict[str, Any]] = []
        for entry in pulse_history:
            if not isinstance(entry, dict):
                continue
            e_date = self._parse_date_like(entry.get("ts"))
            if e_date is None:
                e_date = self._parse_date_like(entry.get("date"))
            if e_date is None:
                continue
            if start <= e_date <= as_of:
                pulse_entries.append(entry)

        daemon_entries: list[dict[str, Any]] = []
        for entry in daemon_history:
            if not isinstance(entry, dict):
                continue
            e_date = self._parse_date_like(entry.get("ts"))
            if e_date is None:
                e_date = self._parse_date_like(entry.get("date"))
            if e_date is None:
                continue
            if start <= e_date <= as_of:
                daemon_entries.append(entry)

        weekly_entries: list[dict[str, Any]] = []
        for entry in weekly_history:
            if not isinstance(entry, dict):
                continue
            e_date = self._parse_date_like(entry.get("date"))
            if e_date is None:
                e_date = self._parse_date_like(entry.get("ts"))
            if e_date is None:
                continue
            if start <= e_date <= as_of:
                weekly_entries.append(entry)

        pulse_slot_error_events = 0
        pulse_slot_error_items = 0
        pulse_ops_runs = 0
        pulse_ops_errors = 0
        pulse_health_error = 0
        pulse_health_nonhealthy = 0
        pulse_weekly_runs = 0
        pulse_weekly_errors = 0
        for entry in pulse_entries:
            slot_errors = entry.get("slot_errors", [])
            if isinstance(slot_errors, list) and slot_errors:
                pulse_slot_error_events += 1
                pulse_slot_error_items += len(slot_errors)
            ops = entry.get("ops", {}) if isinstance(entry.get("ops", {}), dict) else {}
            if bool(ops.get("ran", False)):
                pulse_ops_runs += 1
            if str(ops.get("status", "")).strip().lower() == "error":
                pulse_ops_errors += 1
            health = entry.get("health", {}) if isinstance(entry.get("health", {}), dict) else {}
            health_status = str(health.get("status", "")).strip().lower()
            if health_status == "error":
                pulse_health_error += 1
            if bool(health.get("ran", False)) and health_status not in {"", "healthy"}:
                pulse_health_nonhealthy += 1
            weekly_guardrail = (
                entry.get("weekly_guardrail", {}) if isinstance(entry.get("weekly_guardrail", {}), dict) else {}
            )
            if bool(weekly_guardrail.get("ran", False)):
                pulse_weekly_runs += 1
            if str(weekly_guardrail.get("status", "")).strip().lower() == "error":
                pulse_weekly_errors += 1

        daemon_duplicate_pulses = 0
        daemon_slot_error_events = 0
        daemon_ops_runs = 0
        daemon_weekly_runs = 0
        daemon_weekly_errors = 0
        for entry in daemon_entries:
            if bool(entry.get("duplicate_pulse", False)):
                daemon_duplicate_pulses += 1
            slot_errors = entry.get("slot_errors", [])
            if isinstance(slot_errors, list) and slot_errors:
                daemon_slot_error_events += 1
            ops = entry.get("ops", {}) if isinstance(entry.get("ops", {}), dict) else {}
            if bool(ops.get("ran", False)):
                daemon_ops_runs += 1
            weekly_guardrail = (
                entry.get("weekly_guardrail", {}) if isinstance(entry.get("weekly_guardrail", {}), dict) else {}
            )
            if bool(weekly_guardrail.get("ran", False)):
                daemon_weekly_runs += 1
            if str(weekly_guardrail.get("status", "")).strip().lower() == "error":
                daemon_weekly_errors += 1

        weekly_runs = 0
        weekly_errors = 0
        for entry in weekly_entries:
            weekly_runs += 1
            if str(entry.get("status", "")).strip().lower() != "ok":
                weekly_errors += 1

        guard_history_path = logs_dir / "guard_loop_history.jsonl"
        guard_entries: list[dict[str, Any]] = []
        if guard_history_path.exists():
            for line in guard_history_path.read_text(encoding="utf-8").splitlines():
                txt = line.strip()
                if not txt:
                    continue
                try:
                    row = json.loads(txt)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                row_date = self._parse_date_like(row.get("ts"))
                if row_date is None:
                    row_date = self._parse_date_like((row.get("daemon", {}) if isinstance(row.get("daemon", {}), dict) else {}).get("date"))
                if row_date is None:
                    continue
                if start <= row_date <= as_of:
                    guard_entries.append(row)

        guard_runs = len(guard_entries)
        guard_health_core_degraded = 0
        guard_health_expected_degraded = 0
        guard_recovery_errors = 0
        guard_recovery_light = 0
        guard_recovery_heavy = 0
        guard_pulse_skipped = 0
        guard_pulse_ran = 0
        for row in guard_entries:
            pulse = row.get("pulse", {}) if isinstance(row.get("pulse", {}), dict) else {}
            if bool(pulse.get("ran", False)):
                guard_pulse_ran += 1
            if str(pulse.get("status", "")).strip().lower() == "skipped":
                guard_pulse_skipped += 1

            health = row.get("health", {}) if isinstance(row.get("health", {}), dict) else {}
            if str(health.get("status", "")).strip().lower() == "degraded":
                core_missing = health.get("core_missing", [])
                if isinstance(core_missing, list) and len(core_missing) > 0:
                    guard_health_core_degraded += 1
                else:
                    guard_health_expected_degraded += 1

            recovery = row.get("recovery", {}) if isinstance(row.get("recovery", {}), dict) else {}
            mode = str(recovery.get("mode", "")).strip().lower()
            status = str(recovery.get("status", "")).strip().lower()
            if mode == "light":
                guard_recovery_light += 1
            elif mode == "heavy":
                guard_recovery_heavy += 1
            if status == "error":
                guard_recovery_errors += 1

        pulse_gap_metrics = self._compute_timeline_gap_metrics(pulse_entries, threshold_minutes=45.0)
        daemon_gap_metrics = self._compute_timeline_gap_metrics(daemon_entries, threshold_minutes=90.0)
        guard_gap_metrics = self._compute_timeline_gap_metrics(guard_entries, threshold_minutes=90.0)

        gate_path = review_dir / f"{d}_gate_report.json"
        ops_path = review_dir / f"{d}_ops_report.json"
        gate_payload = self._load_state_json(gate_path, {})
        ops_payload = self._load_state_json(ops_path, {})
        gate_exists = gate_path.exists() and isinstance(gate_payload, dict) and bool(gate_payload)
        ops_exists = ops_path.exists() and isinstance(ops_payload, dict) and bool(ops_payload)
        gate_passed = bool(gate_payload.get("passed", False)) if gate_exists else False
        ops_status = str(ops_payload.get("status", "missing")) if ops_exists else "missing"
        cadence_lift_trend_payload = (
            gate_payload.get("guard_loop_cadence_non_apply_lift_trend", {})
            if isinstance(gate_payload.get("guard_loop_cadence_non_apply_lift_trend", {}), dict)
            else {}
        )
        cadence_lift_trend_metrics = (
            cadence_lift_trend_payload.get("metrics", {})
            if isinstance(cadence_lift_trend_payload.get("metrics", {}), dict)
            else {}
        )
        cadence_lift_trend_series = (
            cadence_lift_trend_payload.get("series", [])
            if isinstance(cadence_lift_trend_payload.get("series", []), list)
            else []
        )
        cadence_lift_trend_alerts = (
            cadence_lift_trend_payload.get("alerts", [])
            if isinstance(cadence_lift_trend_payload.get("alerts", []), list)
            else []
        )
        cadence_lift_trend_active = bool(cadence_lift_trend_payload.get("active", False))
        cadence_lift_window_applied_rate = self._safe_float(
            cadence_lift_trend_metrics.get("applied_rate", 0.0),
            0.0,
        )
        cadence_lift_window_cooldown_block_rate = self._safe_float(
            cadence_lift_trend_metrics.get("cooldown_block_rate", 0.0),
            0.0,
        )
        cadence_lift_delta_window = min(5, len(cadence_lift_trend_series))
        cadence_lift_recent_rows = cadence_lift_trend_series[-cadence_lift_delta_window:] if cadence_lift_delta_window > 0 else []
        cadence_lift_prior_rows: list[dict[str, Any]] = []
        if cadence_lift_delta_window > 0:
            prior_slice = cadence_lift_trend_series[-2 * cadence_lift_delta_window : -cadence_lift_delta_window]
            if isinstance(prior_slice, list):
                cadence_lift_prior_rows = [
                    row for row in prior_slice if isinstance(row, dict)
                ]
        cadence_lift_recent_rows = [row for row in cadence_lift_recent_rows if isinstance(row, dict)]

        def _cadence_segment_rates(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
            if not rows:
                return 0.0, 0.0, 0
            total = len(rows)
            applied = sum(1 for row in rows if bool(row.get("applied", False)))
            requested = sum(1 for row in rows if bool(row.get("requested", False)))
            blocked = sum(1 for row in rows if bool(row.get("blocked_by_cooldown", False)))
            applied_rate = float(applied) / float(total) if total > 0 else 0.0
            cooldown_rate = float(blocked) / float(requested) if requested > 0 else 0.0
            return float(applied_rate), float(cooldown_rate), int(total)

        cadence_lift_recent_applied_rate, cadence_lift_recent_cooldown_rate, cadence_lift_recent_samples = (
            _cadence_segment_rates(cadence_lift_recent_rows)
        )
        cadence_lift_prior_applied_rate, cadence_lift_prior_cooldown_rate, cadence_lift_prior_samples = (
            _cadence_segment_rates(cadence_lift_prior_rows)
        )
        cadence_lift_applied_rate_delta = (
            float(cadence_lift_recent_applied_rate - cadence_lift_prior_applied_rate)
            if cadence_lift_prior_samples > 0
            else 0.0
        )
        cadence_lift_cooldown_rate_delta = (
            float(cadence_lift_recent_cooldown_rate - cadence_lift_prior_cooldown_rate)
            if cadence_lift_prior_samples > 0
            else 0.0
        )
        cadence_lift_reason_counts: dict[str, int] = {}
        for row in cadence_lift_trend_series:
            if not isinstance(row, dict):
                continue
            code = str(row.get("reason_code", "")).strip() or "NONE"
            cadence_lift_reason_counts[code] = int(cadence_lift_reason_counts.get(code, 0) + 1)
        cadence_lift_top_reasons = [
            {"reason_code": str(k), "count": int(v)}
            for k, v in sorted(cadence_lift_reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:5]
        ]

        latest_test_log_path = ""
        latest_test_payload: dict[str, Any] = {}
        test_logs = sorted(logs_dir.glob("tests_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if test_logs:
            latest_test_log_path = str(test_logs[0])
            latest_test_payload = self._load_state_json(test_logs[0], {})
        latest_test_returncode = int(
            (latest_test_payload.get("returncode", 1) if isinstance(latest_test_payload, dict) else 1)
        ) if latest_test_payload else None

        findings: list[str] = []
        next_actions: list[str] = []
        status = "green"

        if pulse_slot_error_items > 0 or pulse_ops_errors > 0 or pulse_health_error > 0:
            status = "red"
            findings.append("半小时调度存在执行错误（slot/ops/health）。")
            next_actions.append("优先检查 halfhour_pulse_state history 中的 slot_errors 与 ops.error，并复跑对应时段。")
        if daemon_slot_error_events > 0 or daemon_weekly_errors > 0:
            status = "red"
            findings.append("daemon 历史存在错误事件。")
            next_actions.append("核查 halfhour_daemon_state 与 weekly_guardrail_state，确认错误是否已恢复。")
        if guard_recovery_errors > 0:
            status = "red"
            findings.append("guard loop 恢复动作出现 error。")
            next_actions.append("检查 guard_loop_history 中 recovery.results 失败链路，修复后重放一轮守护任务。")
        timeline_gap_alerts = int(pulse_gap_metrics["gap_events_over_threshold"]) + int(
            daemon_gap_metrics["gap_events_over_threshold"]
        ) + int(guard_gap_metrics["gap_events_over_threshold"])
        timeline_max_gap = max(
            float(pulse_gap_metrics["max_gap_minutes"]),
            float(daemon_gap_metrics["max_gap_minutes"]),
            float(guard_gap_metrics["max_gap_minutes"]),
        )
        if timeline_gap_alerts > 0:
            if status == "green":
                status = "yellow"
            findings.append("自动执行链路存在时间断档（可能由休眠、额度耗尽或调度冲突引发）。")
            next_actions.append("检查电源休眠设置与 automation 错峰策略，并补跑遗漏窗口。")
        if timeline_max_gap >= 180.0:
            status = "red"
            findings.append("自动执行出现超长断档（>=180分钟），当日复盘与风控口径可能失真。")
            next_actions.append("立即补跑半小时脉冲、guard loop 与 autorun-retro，重建当日状态。")
        if (not gate_exists) or (not ops_exists):
            if status == "green":
                status = "yellow"
            findings.append("当日 gate/ops 报告缺失。")
            next_actions.append("补跑 lie gate-report 与 lie ops-report，保持自动执行产物完整。")
        elif (not gate_passed) or (str(ops_status).lower() == "red"):
            if status != "red":
                status = "yellow"
            findings.append("自动执行链路可运行，但 gate/ops 指标仍存在风险项。")
            next_actions.append("按 gate failed_checks 与 ops 红色项执行修复闭环。")
        if guard_health_core_degraded > 0:
            if status != "red":
                status = "yellow"
            findings.append("guard loop 观测到核心缺失导致 degraded。")
            next_actions.append("核对核心工件缺失原因（非时间窗预期），避免误降级。")
        if len(pulse_entries) == 0 and len(guard_entries) == 0:
            if status == "green":
                status = "yellow"
            findings.append("窗口内未观测到有效自动执行样本。")
            next_actions.append("检查 cron/automation 触发是否按预期执行。")
        if cadence_lift_trend_active and cadence_lift_prior_samples > 0:
            if cadence_lift_applied_rate_delta <= -0.15:
                if status == "green":
                    status = "yellow"
                findings.append("cadence rollback-lift 应用率出现下行漂移。")
                next_actions.append("降低 cooldown 阻塞并复核 cadence lift 阈值，避免恢复动作长期不落地。")
            if cadence_lift_cooldown_rate_delta >= 0.15:
                if status == "green":
                    status = "yellow"
                findings.append("cadence rollback-lift cooldown 阻塞率上升。")
                next_actions.append("检查近期 recovery 决策与冷却窗口设置，防止 hard/soft lift 被连续拦截。")
        if cadence_lift_trend_active and cadence_lift_trend_alerts:
            if status == "green":
                status = "yellow"
            findings.append("cadence lift 趋势监控存在告警。")
            next_actions.append("结合 gate 中 cadence-lift trend 告警逐项修复，再回放 guard loop 验证。")
        if latest_test_returncode is not None and latest_test_returncode != 0:
            status = "red"
            findings.append("最近一次 test-all 非零返回。")
            next_actions.append("优先处理最近测试失败并重新生成回归基线。")
        if not findings:
            findings.append("自动执行链路整体稳定，未发现阻断性异常。")
            next_actions.append("继续按固定窗口滚动复盘，并跟踪 gate/ops 漂移趋势。")

        summary = {
            "pulse_entries": int(len(pulse_entries)),
            "daemon_entries": int(len(daemon_entries)),
            "weekly_entries": int(len(weekly_entries)),
            "guard_loop_entries": int(guard_runs),
            "timeline_gap_alerts": int(timeline_gap_alerts),
            "gate_exists": bool(gate_exists),
            "gate_passed": bool(gate_passed) if gate_exists else None,
            "ops_exists": bool(ops_exists),
            "ops_status": str(ops_status),
            "cadence_lift_trend_active": bool(cadence_lift_trend_active),
            "cadence_lift_trend_alerts": int(len(cadence_lift_trend_alerts)),
            "latest_test_returncode": latest_test_returncode,
        }
        metrics = {
            "pulse_slot_error_events": int(pulse_slot_error_events),
            "pulse_slot_error_items": int(pulse_slot_error_items),
            "pulse_ops_runs": int(pulse_ops_runs),
            "pulse_ops_errors": int(pulse_ops_errors),
            "pulse_health_error": int(pulse_health_error),
            "pulse_health_nonhealthy": int(pulse_health_nonhealthy),
            "pulse_weekly_guardrail_runs": int(pulse_weekly_runs),
            "pulse_weekly_guardrail_errors": int(pulse_weekly_errors),
            "daemon_duplicate_pulses": int(daemon_duplicate_pulses),
            "daemon_slot_error_events": int(daemon_slot_error_events),
            "daemon_ops_runs": int(daemon_ops_runs),
            "daemon_weekly_guardrail_runs": int(daemon_weekly_runs),
            "daemon_weekly_guardrail_errors": int(daemon_weekly_errors),
            "weekly_runs": int(weekly_runs),
            "weekly_errors": int(weekly_errors),
            "guard_health_core_degraded": int(guard_health_core_degraded),
            "guard_health_expected_degraded": int(guard_health_expected_degraded),
            "guard_recovery_errors": int(guard_recovery_errors),
            "guard_recovery_light": int(guard_recovery_light),
            "guard_recovery_heavy": int(guard_recovery_heavy),
            "guard_pulse_ran": int(guard_pulse_ran),
            "guard_pulse_skipped": int(guard_pulse_skipped),
            "pulse_max_gap_minutes": float(pulse_gap_metrics["max_gap_minutes"]),
            "pulse_gap_events_over_45m": int(pulse_gap_metrics["gap_events_over_threshold"]),
            "daemon_max_gap_minutes": float(daemon_gap_metrics["max_gap_minutes"]),
            "daemon_gap_events_over_90m": int(daemon_gap_metrics["gap_events_over_threshold"]),
            "guard_max_gap_minutes": float(guard_gap_metrics["max_gap_minutes"]),
            "guard_gap_events_over_90m": int(guard_gap_metrics["gap_events_over_threshold"]),
            "cadence_lift_window_applied_rate": float(cadence_lift_window_applied_rate),
            "cadence_lift_window_cooldown_block_rate": float(cadence_lift_window_cooldown_block_rate),
            "cadence_lift_recent_applied_rate": float(cadence_lift_recent_applied_rate),
            "cadence_lift_recent_cooldown_block_rate": float(cadence_lift_recent_cooldown_rate),
            "cadence_lift_prior_applied_rate": float(cadence_lift_prior_applied_rate),
            "cadence_lift_prior_cooldown_block_rate": float(cadence_lift_prior_cooldown_rate),
            "cadence_lift_applied_rate_delta": float(cadence_lift_applied_rate_delta),
            "cadence_lift_cooldown_block_rate_delta": float(cadence_lift_cooldown_rate_delta),
            "cadence_lift_recent_samples": int(cadence_lift_recent_samples),
            "cadence_lift_prior_samples": int(cadence_lift_prior_samples),
        }

        out: dict[str, Any] = {
            "date": d,
            "status": status,
            "window_days": int(wd),
            "window": {"start": start.isoformat(), "end": d},
            "summary": summary,
            "metrics": metrics,
            "artifacts": {
                "gate_report_path": str(gate_path),
                "ops_report_path": str(ops_path),
                "latest_test_log_path": latest_test_log_path,
                "pulse_state_path": str(self._pulse_state_path()),
                "daemon_state_path": str(self._halfhour_daemon_state_path()),
                "weekly_guardrail_state_path": str(self._weekly_guardrail_state_path()),
                "guard_state_path": str(logs_dir / "guard_state.json"),
                "guard_last_path": str(logs_dir / "guard_loop_last.json"),
                "guard_history_path": str(guard_history_path),
            },
            "states": {
                "weekly_guardrail_last_run_week": str(weekly_state.get("last_run_week", "")),
                "weekly_guardrail_last_run_date": str(weekly_state.get("last_run_date", "")),
                "weekly_guardrail_last_status": str(weekly_state.get("last_status", "")),
                "guard_consecutive_bad": int(
                    (guard_state.get("consecutive_bad", 0) if isinstance(guard_state, dict) else 0)
                ),
                "guard_last_recovery_mode": str(
                    (
                        guard_last.get("recovery", {})
                        if isinstance(guard_last.get("recovery", {}), dict)
                        else {}
                    ).get("mode", "")
                )
                if isinstance(guard_last, dict)
                else "",
            },
            "cadence_lift_trend": {
                "active": bool(cadence_lift_trend_active),
                "alerts": [str(x) for x in cadence_lift_trend_alerts],
                "window_applied_rate": float(cadence_lift_window_applied_rate),
                "window_cooldown_block_rate": float(cadence_lift_window_cooldown_block_rate),
                "recent_applied_rate": float(cadence_lift_recent_applied_rate),
                "recent_cooldown_block_rate": float(cadence_lift_recent_cooldown_rate),
                "prior_applied_rate": float(cadence_lift_prior_applied_rate),
                "prior_cooldown_block_rate": float(cadence_lift_prior_cooldown_rate),
                "applied_rate_delta": float(cadence_lift_applied_rate_delta),
                "cooldown_block_rate_delta": float(cadence_lift_cooldown_rate_delta),
                "recent_samples": int(cadence_lift_recent_samples),
                "prior_samples": int(cadence_lift_prior_samples),
                "top_reason_codes": cadence_lift_top_reasons,
            },
            "findings": findings,
            "next_actions": next_actions[:8],
        }

        report_json = review_dir / f"{d}_autorun_retro.json"
        report_md = review_dir / f"{d}_autorun_retro.md"
        report_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            f"# 自动执行复盘 | {d}",
            "",
            f"- status: `{status}`",
            f"- window_days: `{int(wd)}` ({start.isoformat()} ~ {d})",
            f"- pulse/daemon/weekly/guard: `{summary['pulse_entries']}` / `{summary['daemon_entries']}` / `{summary['weekly_entries']}` / `{summary['guard_loop_entries']}`",
            f"- gate_passed: `{summary['gate_passed']}` | ops_status: `{summary['ops_status']}`",
            (
                "- cadence_lift(active/alerts): "
                + f"`{summary['cadence_lift_trend_active']}` / `{summary['cadence_lift_trend_alerts']}`"
            ),
            f"- latest_test_returncode: `{summary['latest_test_returncode']}`",
            "",
            "## 关键计数",
            (
                "- pulse(slot_error_events/items, ops_errors, health_nonhealthy): "
                + f"`{metrics['pulse_slot_error_events']}` / `{metrics['pulse_slot_error_items']}` / "
                + f"`{metrics['pulse_ops_errors']}` / `{metrics['pulse_health_nonhealthy']}`"
            ),
            (
                "- guard(core_degraded/expected_degraded, recovery_heavy/light/error): "
                + f"`{metrics['guard_health_core_degraded']}` / `{metrics['guard_health_expected_degraded']}` / "
                + f"`{metrics['guard_recovery_heavy']}` / `{metrics['guard_recovery_light']}` / `{metrics['guard_recovery_errors']}`"
            ),
            (
                "- timeline_gap_events(pulse>45m / daemon>90m / guard>90m): "
                + f"`{metrics['pulse_gap_events_over_45m']}` / `{metrics['daemon_gap_events_over_90m']}` / "
                + f"`{metrics['guard_gap_events_over_90m']}`"
            ),
            (
                "- timeline_max_gap_minutes(pulse/daemon/guard): "
                + f"`{metrics['pulse_max_gap_minutes']}` / `{metrics['daemon_max_gap_minutes']}` / `{metrics['guard_max_gap_minutes']}`"
            ),
            (
                "- cadence_lift_delta(applied/cooldown_block): "
                + f"`{metrics['cadence_lift_applied_rate_delta']:.2%}` / "
                + f"`{metrics['cadence_lift_cooldown_block_rate_delta']:.2%}`"
            ),
            "",
            "## Cadence Lift Trend Delta",
            (
                "- window(applied/cooldown_block): "
                + f"`{metrics['cadence_lift_window_applied_rate']:.2%}` / "
                + f"`{metrics['cadence_lift_window_cooldown_block_rate']:.2%}`"
            ),
            (
                "- recent(applied/cooldown_block/samples): "
                + f"`{metrics['cadence_lift_recent_applied_rate']:.2%}` / "
                + f"`{metrics['cadence_lift_recent_cooldown_block_rate']:.2%}` / "
                + f"`{metrics['cadence_lift_recent_samples']}`"
            ),
            (
                "- prior(applied/cooldown_block/samples): "
                + f"`{metrics['cadence_lift_prior_applied_rate']:.2%}` / "
                + f"`{metrics['cadence_lift_prior_cooldown_block_rate']:.2%}` / "
                + f"`{metrics['cadence_lift_prior_samples']}`"
            ),
            (
                "- delta(applied/cooldown_block): "
                + f"`{metrics['cadence_lift_applied_rate_delta']:.2%}` / "
                + f"`{metrics['cadence_lift_cooldown_block_rate_delta']:.2%}`"
            ),
            "",
            "## 发现",
        ]
        for item in findings:
            lines.append(f"- {item}")
        lines.extend(["", "## 下一步动作"])
        for item in next_actions[:8]:
            lines.append(f"- {item}")
        report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

        out["paths"] = {"json": str(report_json), "md": str(report_md)}
        return out

    def run_slot(
        self,
        as_of: date,
        slot: str,
        max_review_rounds: int = 2,
        trace_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        cfg = self._schedule_cfg()
        slot = str(slot)

        if slot in {"premarket", cfg["premarket"]}:
            return {"slot": "premarket", "result": self.run_premarket(as_of)}

        if slot in {"eod", cfg["eod"]}:
            return {"slot": "eod", "result": self.run_eod(as_of)}

        if slot in {"review", cfg["nightly_review"]}:
            review_cycle = self._invoke_review_cycle(
                as_of=as_of,
                max_review_rounds=max_review_rounds,
                trace_id=trace_id,
                parent_event_id=parent_event_id,
            )
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

    def run_halfhour_pulse(
        self,
        as_of: date,
        slot: str | None = None,
        max_review_rounds: int = 2,
        max_slot_runs: int = 2,
        slot_retry_max: int = 2,
        ops_every_n_pulses: int = 4,
        force: bool = False,
        dry_run: bool = False,
        trace_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        tz = ZoneInfo(self.settings.timezone)
        now = datetime.now(tz)
        observed_time = self._parse_hhmm(slot) if isinstance(slot, str) and str(slot).strip() else now.time()
        pulse_time = self._pulse_bucket(observed_time)
        observed_slot = self._format_hhmm(observed_time)
        pulse_slot = self._format_hhmm(pulse_time)
        lock_path = self._scheduler_lock_path()
        lock_status = "dry_run" if dry_run else "ok"
        lock_fp: IO[str] | None = None
        day_str = as_of.isoformat()
        weekly_controlled_apply_snapshot = self._weekly_controlled_apply_snapshot(as_of=as_of)
        resolved_trace_id = str(trace_id or derive_trace_id(source="scheduler.run_halfhour_pulse", as_of=day_str))
        if dry_run:
            pulse_start_event = build_event_envelope(
                source="scheduler.run_halfhour_pulse",
                event_type="scheduler.pulse.start",
                payload={
                    "pulse_slot": pulse_slot,
                    "observed_slot": observed_slot,
                    "dry_run": bool(dry_run),
                    "force": bool(force),
                },
                as_of=as_of,
                trace_id=resolved_trace_id,
                parent_event_id=parent_event_id,
            ).to_dict()
            pulse_stream_path = ""
        else:
            pulse_start_event, pulse_stream_path = self._emit_scheduler_event(
                as_of=as_of,
                source="scheduler.run_halfhour_pulse",
                event_type="scheduler.pulse.start",
                payload={
                    "pulse_slot": pulse_slot,
                    "observed_slot": observed_slot,
                    "dry_run": bool(dry_run),
                    "force": bool(force),
                },
                trace_id=resolved_trace_id,
                parent_event_id=parent_event_id,
            )
        pulse_parent_event_id = str(pulse_start_event.get("event_id", ""))

        def _attach_pulse_event(payload: dict[str, Any]) -> dict[str, Any]:
            slot_errors_count = 0
            if isinstance(payload.get("slot_errors", []), list):
                slot_errors_count = len(payload.get("slot_errors", []))
            status = "ok"
            if bool(payload.get("skipped", False)):
                status = "skipped"
            elif slot_errors_count > 0:
                status = "error"
            event_payload = {
                "status": status,
                "duplicate_pulse": bool(payload.get("duplicate_pulse", False)),
                "reason": str(payload.get("reason", "")),
                "dry_run": bool(payload.get("dry_run", False)),
                "force": bool(payload.get("force", False)),
                "pulse_slot": str(payload.get("pulse_slot", "")),
                "observed_slot": str(payload.get("observed_slot", "")),
                "lock_status": str(payload.get("lock_status", "")),
                "due_slots_count": len(payload.get("due_slots", [])) if isinstance(payload.get("due_slots", []), list) else 0,
                "run_slots_count": len(payload.get("run_slots", [])) if isinstance(payload.get("run_slots", []), list) else 0,
                "pending_slots_count": (
                    len(payload.get("pending_slots", [])) if isinstance(payload.get("pending_slots", []), list) else 0
                ),
                "slot_errors_count": int(slot_errors_count),
                "exhausted_retries_count": (
                    len(payload.get("exhausted_retries", []))
                    if isinstance(payload.get("exhausted_retries", []), list)
                    else 0
                ),
                "ops_status": str((payload.get("ops", {}) or {}).get("status", "")),
                "ops_planned": bool((payload.get("ops", {}) or {}).get("planned", False)),
                "health_status": str((payload.get("health", {}) or {}).get("status", "")),
            }
            if dry_run:
                event_envelope = build_event_envelope(
                    source="scheduler.run_halfhour_pulse",
                    event_type="scheduler.pulse.completed",
                    payload=event_payload,
                    as_of=as_of,
                    trace_id=resolved_trace_id,
                    parent_event_id=pulse_parent_event_id or parent_event_id,
                ).to_dict()
                event_stream_path = ""
            else:
                event_envelope, event_stream_path = self._emit_scheduler_event(
                    as_of=as_of,
                    source="scheduler.run_halfhour_pulse",
                    event_type="scheduler.pulse.completed",
                    payload=event_payload,
                    trace_id=resolved_trace_id,
                    parent_event_id=pulse_parent_event_id or parent_event_id,
                )
            payload["trace_id"] = resolved_trace_id
            payload["traceparent"] = str(event_envelope.get("traceparent", ""))
            payload["event_envelope"] = event_envelope
            payload["event_stream_path"] = event_stream_path
            payload["event_chain"] = {
                "start_event_id": pulse_parent_event_id,
                "final_event_id": str(event_envelope.get("event_id", "")),
                "start_stream_path": pulse_stream_path,
                "final_stream_path": event_stream_path,
            }
            return payload

        if not dry_run:
            lock_fp, lock_status = self._acquire_scheduler_lock()
            if lock_fp is None:
                return _attach_pulse_event(
                    {
                        "date": day_str,
                        "pulse_slot": pulse_slot,
                        "observed_slot": observed_slot,
                        "duplicate_pulse": True,
                        "skipped": True,
                        "reason": "scheduler_locked",
                        "lock_status": str(lock_status),
                        "state_path": str(self._pulse_state_path()),
                        "lock_path": str(lock_path),
                        "executed_pulses": [],
                        "executed_slots": [],
                        "weekly_controlled_apply": weekly_controlled_apply_snapshot,
                    }
                )

        try:
            state_path = self._pulse_state_path()
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_default = {
                "date": None,
                "executed_pulses": [],
                "executed_slots": [],
                "slot_failures": {},
                "history": [],
            }
            state = self._load_state_json(state_path, state_default)

            if str(state.get("date")) != day_str:
                state = dict(state_default)
                state["date"] = day_str

            executed_pulses = [str(x) for x in state.get("executed_pulses", [])]
            executed_slots = [str(x) for x in state.get("executed_slots", [])]
            slot_failures = state.get("slot_failures", {})
            if not isinstance(slot_failures, dict):
                slot_failures = {}

            if pulse_slot in set(executed_pulses) and not force:
                return _attach_pulse_event(
                    {
                        "date": day_str,
                        "pulse_slot": pulse_slot,
                        "observed_slot": observed_slot,
                        "duplicate_pulse": True,
                        "skipped": True,
                        "reason": "pulse_already_executed",
                        "lock_status": str(lock_status),
                        "state_path": str(state_path),
                        "lock_path": str(lock_path),
                        "executed_pulses": executed_pulses,
                        "executed_slots": executed_slots,
                        "weekly_controlled_apply": weekly_controlled_apply_snapshot,
                    }
                )

            max_slot_runs = max(1, int(max_slot_runs))
            slot_retry_max = max(1, int(slot_retry_max))
            ops_every_n_pulses = max(0, int(ops_every_n_pulses))
            executed_slots_set = set(executed_slots)

            due_slots: list[dict[str, str]] = []
            for slot_cfg in self._daemon_slots():
                trigger = slot_cfg["trigger"]
                slot_id = slot_cfg["id"]
                if slot_id in executed_slots_set:
                    continue
                if self._parse_hhmm(trigger) <= observed_time:
                    due_slots.append(slot_cfg)

            run_slots = due_slots[:max_slot_runs]
            pending_slots = [x["id"] for x in due_slots[max_slot_runs:]]
            run_results: list[dict[str, Any]] = []
            slot_errors: list[dict[str, str]] = []
            exhausted_retries: list[str] = []

            if not dry_run:
                for slot_cfg in run_slots:
                    slot_id = str(slot_cfg["id"])
                    try:
                        result = self.run_slot(
                            as_of=as_of,
                            slot=slot_cfg["slot"],
                            max_review_rounds=max_review_rounds,
                            trace_id=resolved_trace_id,
                            parent_event_id=pulse_parent_event_id,
                        )
                        run_results.append(
                            {
                                "slot_id": slot_id,
                                "status": "ok",
                                "result": result,
                            }
                        )
                        executed_slots_set.add(slot_id)
                        slot_failures.pop(slot_id, None)
                    except Exception as exc:
                        fail_count = int(slot_failures.get(slot_id, 0)) + 1
                        slot_failures[slot_id] = fail_count
                        slot_errors.append({"slot_id": slot_id, "error": str(exc), "fail_count": str(fail_count)})
                        run_results.append(
                            {
                                "slot_id": slot_id,
                                "status": "error",
                                "error": str(exc),
                                "fail_count": fail_count,
                            }
                        )
                        if fail_count >= slot_retry_max:
                            exhausted_retries.append(slot_id)
                            executed_slots_set.add(slot_id)

            health_payload: dict[str, Any] = {"enabled": bool(self.health_check), "ran": False}
            if self.health_check is not None and not dry_run:
                try:
                    raw = self.health_check(as_of, False)
                    health_payload = {
                        "enabled": True,
                        "ran": True,
                        "status": str(raw.get("status", "unknown")),
                        "missing_count": len(raw.get("missing", [])) if isinstance(raw.get("missing", []), list) else 0,
                        "check_count": len(raw.get("checks", {})) if isinstance(raw.get("checks", {}), dict) else 0,
                    }
                except Exception as exc:
                    health_payload = {"enabled": True, "ran": True, "status": "error", "error": str(exc)}

            pulse_index = len(executed_pulses) + 1
            slot_due_now = bool(run_slots)
            slot_executed_now = bool(slot_due_now and not dry_run)
            ops_due_by_cadence = ops_every_n_pulses > 0 and pulse_index % ops_every_n_pulses == 0
            run_ops = bool(slot_due_now or ops_due_by_cadence)
            ops_window_days = int(self.settings.validation.get("required_stable_replay_days", 3))
            ops_payload: dict[str, Any] = {
                "planned": run_ops,
                "ran": False,
                "window_days": int(ops_window_days),
                "reason": "slot_executed" if slot_executed_now else ("cadence" if ops_due_by_cadence else "none"),
            }
            if run_ops and not dry_run:
                try:
                    ops_result = self.ops_report(as_of, ops_window_days)
                    ops_payload["ran"] = True
                    ops_payload["status"] = "ok"
                    ops_payload["result_status"] = str(ops_result.get("status", "unknown"))
                except Exception as exc:
                    ops_payload["ran"] = True
                    ops_payload["status"] = "error"
                    ops_payload["error"] = str(exc)

            weekly_guardrail = self._run_weekly_guardrail(
                as_of=as_of,
                observed_time=observed_time,
                force=bool(force),
                dry_run=bool(dry_run),
                ops_planned=bool(run_ops),
            )
            weekly_controlled_apply = (
                weekly_guardrail.get("controlled_apply_readiness", {})
                if isinstance(weekly_guardrail.get("controlled_apply_readiness", {}), dict)
                else {}
            )
            if not weekly_controlled_apply:
                weekly_controlled_apply = weekly_controlled_apply_snapshot

            history_entry = {
                "ts": now.isoformat(),
                "pulse_slot": pulse_slot,
                "observed_slot": observed_slot,
                "dry_run": bool(dry_run),
                "force": bool(force),
                "due_slots": [x["id"] for x in due_slots],
                "run_slots": [x["id"] for x in run_slots],
                "pending_slots": pending_slots,
                "slot_errors": slot_errors,
                "exhausted_retries": exhausted_retries,
                "ops": ops_payload,
                "health": health_payload,
                "weekly_guardrail": weekly_guardrail,
                "weekly_controlled_apply": weekly_controlled_apply,
            }

            if not dry_run:
                state["date"] = day_str
                state["executed_pulses"] = sorted(set(executed_pulses + [pulse_slot]))
                state["executed_slots"] = sorted(executed_slots_set)
                state["slot_failures"] = slot_failures
                history = state.get("history", [])
                if not isinstance(history, list):
                    history = []
                history.append(history_entry)
                history_limit = max(24, int(self.settings.validation.get("pulse_state_history_limit", 192)))
                state["history"] = history[-history_limit:]
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            return _attach_pulse_event(
                {
                "date": day_str,
                "pulse_slot": pulse_slot,
                "observed_slot": observed_slot,
                "duplicate_pulse": False,
                "dry_run": bool(dry_run),
                "force": bool(force),
                "lock_status": str(lock_status),
                "state_path": str(state_path),
                "lock_path": str(lock_path),
                "due_slots": [x["id"] for x in due_slots],
                "run_slots": [x["id"] for x in run_slots],
                "pending_slots": pending_slots,
                "run_results": run_results,
                "slot_errors": slot_errors,
                "exhausted_retries": exhausted_retries,
                "slot_failures": slot_failures,
                "ops": ops_payload,
                "health": health_payload,
                "weekly_guardrail": weekly_guardrail,
                "weekly_controlled_apply": weekly_controlled_apply,
                }
            )
        finally:
            if not dry_run:
                self._release_scheduler_lock(lock_fp)

    def run_halfhour_daemon(
        self,
        poll_seconds: int = 30,
        max_cycles: int | None = None,
        max_review_rounds: int = 2,
        max_slot_runs: int = 2,
        slot_retry_max: int = 2,
        ops_every_n_pulses: int = 4,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        tz = ZoneInfo(self.settings.timezone)
        state_path = self._halfhour_daemon_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_default = {
            "date": None,
            "last_bucket": None,
            "history": [],
            "cycles": 0,
        }
        state = self._load_state_json(state_path, state_default)

        def _new_day_state(day_str: str) -> dict[str, Any]:
            return {
                "date": day_str,
                "last_bucket": None,
                "history": [],
                "cycles": 0,
            }

        def _prepare_state(now_dt: datetime, st: dict[str, Any]) -> dict[str, Any]:
            day_str = now_dt.date().isoformat()
            if str(st.get("date")) != day_str:
                return _new_day_state(day_str)
            return st

        if dry_run:
            now = datetime.now(tz)
            state = _prepare_state(now, state)
            observed_slot = self._format_hhmm(now.time())
            pulse_bucket = self._format_hhmm(self._pulse_bucket(now.time()))
            weekly_controlled_apply_snapshot = self._weekly_controlled_apply_snapshot(as_of=now.date())
            would_run_pulse = str(state.get("last_bucket")) != pulse_bucket
            pulse_preview: dict[str, Any]
            if would_run_pulse:
                pulse_preview = self.run_halfhour_pulse(
                    as_of=now.date(),
                    slot=observed_slot,
                    max_review_rounds=max_review_rounds,
                    max_slot_runs=max_slot_runs,
                    slot_retry_max=slot_retry_max,
                    ops_every_n_pulses=ops_every_n_pulses,
                    dry_run=True,
                )
                if "weekly_controlled_apply" not in pulse_preview:
                    pulse_preview["weekly_controlled_apply"] = weekly_controlled_apply_snapshot
            else:
                pulse_preview = {
                    "date": now.date().isoformat(),
                    "pulse_slot": pulse_bucket,
                    "observed_slot": observed_slot,
                    "duplicate_pulse": True,
                    "skipped": True,
                    "reason": "same_bucket",
                    "state_path": str(self._pulse_state_path()),
                    "weekly_controlled_apply": weekly_controlled_apply_snapshot,
                }
            preview_weekly_controlled_apply = (
                pulse_preview.get("weekly_controlled_apply", {})
                if isinstance(pulse_preview.get("weekly_controlled_apply", {}), dict)
                else weekly_controlled_apply_snapshot
            )
            return {
                "dry_run": True,
                "now": now.isoformat(),
                "date": now.date().isoformat(),
                "daemon_state_path": str(state_path),
                "pulse_state_path": str(self._pulse_state_path()),
                "current_bucket": pulse_bucket,
                "last_bucket": state.get("last_bucket"),
                "would_run_pulse": bool(would_run_pulse),
                "pulse_preview": pulse_preview,
                "weekly_controlled_apply": preview_weekly_controlled_apply,
            }

        cycles = 0
        while True:
            now = datetime.now(tz)
            state = _prepare_state(now, state)
            observed_slot = self._format_hhmm(now.time())
            pulse_bucket = self._format_hhmm(self._pulse_bucket(now.time()))
            weekly_controlled_apply_snapshot = self._weekly_controlled_apply_snapshot(as_of=now.date())
            should_run_pulse = str(state.get("last_bucket")) != pulse_bucket

            if should_run_pulse:
                pulse_result = self.run_halfhour_pulse(
                    as_of=now.date(),
                    slot=observed_slot,
                    max_review_rounds=max_review_rounds,
                    max_slot_runs=max_slot_runs,
                    slot_retry_max=slot_retry_max,
                    ops_every_n_pulses=ops_every_n_pulses,
                    dry_run=False,
                )
                pulse_reason = str(pulse_result.get("reason", "")).strip()
                pulse_locked = bool(pulse_result.get("skipped", False) and pulse_reason == "scheduler_locked")
                pulse_weekly_controlled_apply = (
                    pulse_result.get("weekly_controlled_apply", {})
                    if isinstance(pulse_result.get("weekly_controlled_apply", {}), dict)
                    else weekly_controlled_apply_snapshot
                )
                entry = {
                    "ts": now.isoformat(),
                    "date": now.date().isoformat(),
                    "observed_slot": observed_slot,
                    "pulse_bucket": pulse_bucket,
                    "status": "locked" if pulse_locked else "ok",
                    "duplicate_pulse": bool(pulse_result.get("duplicate_pulse", False)),
                    "reason": pulse_reason,
                    "lock_status": str(pulse_result.get("lock_status", "")),
                    "run_slots": list(pulse_result.get("run_slots", [])),
                    "pending_slots": list(pulse_result.get("pending_slots", [])),
                    "slot_errors": list(pulse_result.get("slot_errors", [])),
                    "ops": pulse_result.get("ops", {}),
                    "weekly_guardrail": pulse_result.get("weekly_guardrail", {}),
                    "weekly_controlled_apply": pulse_weekly_controlled_apply,
                    "trace_id": str(pulse_result.get("trace_id", "")),
                    "traceparent": str(pulse_result.get("traceparent", "")),
                    "event_id": str((pulse_result.get("event_envelope", {}) or {}).get("event_id", "")),
                }
                if not pulse_locked:
                    state["last_bucket"] = pulse_bucket
                state["weekly_controlled_apply"] = pulse_weekly_controlled_apply
                history = state.get("history", [])
                if not isinstance(history, list):
                    history = []
                history.append(entry)
                history_limit = max(48, int(self.settings.validation.get("halfhour_daemon_history_limit", 512)))
                state["history"] = history[-history_limit:]
                state["cycles"] = int(state.get("cycles", 0)) + 1
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            if max_cycles is not None and cycles >= max_cycles:
                break

            cycles += 1
            time_module.sleep(max(1, int(poll_seconds)))

        return state

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
        lock_status = "dry_run" if dry_run else "ok"
        lock_fp: IO[str] | None = None

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
                "lock_status": str(lock_status),
                "now": now.isoformat(),
                "state_path": str(state_path),
                "executed": state.get("executed", []),
                "would_execute": would_execute,
                "slots": slots_view,
            }

        lock_fp, lock_status = self._acquire_scheduler_lock()
        if lock_fp is None:
            now = datetime.now(tz)
            day_str = now.date().isoformat()
            return {
                "date": day_str,
                "dry_run": False,
                "locked": True,
                "reason": "scheduler_locked",
                "lock_status": str(lock_status),
                "state_path": str(state_path),
                "lock_path": str(self._scheduler_lock_path()),
                "executed": state.get("executed", []),
                "history": state.get("history", []),
            }

        try:
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
        finally:
            self._release_scheduler_lock(lock_fp)

        state["lock_status"] = str(lock_status)
        return state
