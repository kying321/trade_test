from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from contextlib import closing
import csv
import sqlite3
from typing import Any, Callable

import yaml

from lie_engine.config import SystemSettings
from lie_engine.data.storage import write_json, write_markdown
from lie_engine.models import ReviewDelta


@dataclass(slots=True)
class ReleaseOrchestrator:
    settings: SystemSettings
    output_dir: Path
    quality_snapshot: Callable[[date], dict[str, Any]]
    backtest_snapshot: Callable[[date], dict[str, Any]]
    run_review: Callable[[date], ReviewDelta]
    health_check: Callable[[date, bool], dict[str, Any]]
    stable_replay_check: Callable[[date, int | None], dict[str, Any]]
    test_all: Callable[..., dict[str, Any]]
    load_json_safely: Callable[[Path], dict[str, Any]]
    sqlite_path: Path | None = None

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _extract_failed_tests(test_payload: dict[str, Any]) -> list[str]:
        listed = test_payload.get("failed_tests", [])
        if isinstance(listed, list):
            out = [str(x).strip() for x in listed if str(x).strip()]
            if out:
                return out
        stderr = str(test_payload.get("stderr", "") or "")
        if not stderr:
            stderr = str(test_payload.get("stderr_excerpt", "") or "")
        stdout = str(test_payload.get("stdout", "") or "")
        if not stdout:
            stdout = str(test_payload.get("stdout_excerpt", "") or "")
        text = f"{stderr}\n{stdout}"
        failed: list[str] = []
        for line in text.splitlines():
            txt = line.strip()
            if txt.endswith("... FAIL") or txt.endswith("... ERROR"):
                failed.append(txt.split(" ... ")[0].strip())
            elif txt.startswith("FAIL: ") or txt.startswith("ERROR: "):
                failed.append(txt.split(": ", 1)[1].strip())
        out: list[str] = []
        seen: set[str] = set()
        for item in failed:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def gate_report(
        self,
        as_of: date,
        run_tests: bool = False,
        run_review_if_missing: bool = True,
    ) -> dict[str, Any]:
        d = as_of.isoformat()
        review_delta_path = self.output_dir / "review" / f"{d}_param_delta.yaml"
        if run_review_if_missing and not review_delta_path.exists():
            self.run_review(as_of)

        quality = self.quality_snapshot(as_of)
        backtest = self.backtest_snapshot(as_of)
        health = self.health_check(as_of, True)
        replay = self.stable_replay_check(as_of, None)
        state_stability = self._state_stability_metrics(as_of=as_of)
        state_active = bool(state_stability.get("active", False))
        state_checks = state_stability.get("checks", {}) if isinstance(state_stability.get("checks", {}), dict) else {}
        state_stability_ok = all(bool(v) for v in state_checks.values()) if state_active else True
        slot_anomaly = self._slot_anomaly_metrics(as_of=as_of)
        slot_active = bool(slot_anomaly.get("active", False))
        slot_checks = slot_anomaly.get("checks", {}) if isinstance(slot_anomaly.get("checks", {}), dict) else {}
        slot_anomaly_ok = all(bool(v) for v in slot_checks.values()) if slot_active else True
        mode_drift = self._mode_drift_metrics(as_of=as_of)
        drift_active = bool(mode_drift.get("active", False))
        drift_checks = mode_drift.get("checks", {}) if isinstance(mode_drift.get("checks", {}), dict) else {}
        mode_drift_ok = all(bool(v) for v in drift_checks.values()) if drift_active else True
        reconcile_drift = self._reconcile_drift_metrics(as_of=as_of)
        reconcile_active = bool(reconcile_drift.get("active", False))
        reconcile_checks = reconcile_drift.get("checks", {}) if isinstance(reconcile_drift.get("checks", {}), dict) else {}
        reconcile_drift_ok = all(bool(v) for v in reconcile_checks.values()) if reconcile_active else True

        tests_ok = True
        tests_payload: dict[str, Any] = {}
        if run_tests:
            tests_payload = self.test_all()
            tests_ok = bool(tests_payload.get("returncode", 1) == 0)

        review_pass = False
        mode_health_ok = True
        if review_delta_path.exists():
            try:
                review_delta = yaml.safe_load(review_delta_path.read_text(encoding="utf-8")) or {}
            except Exception:
                review_delta = {}
            review_pass = bool(review_delta.get("pass_gate", False))
            mode_health = review_delta.get("mode_health", {}) if isinstance(review_delta.get("mode_health", {}), dict) else {}
            mode_health_ok = bool(mode_health.get("passed", True))

        completeness = float(quality.get("completeness", 0.0))
        unresolved = float(quality.get("unresolved_conflict_ratio", 1.0))
        positive_ratio = float(backtest.get("positive_window_ratio", 0.0))
        max_drawdown = float(backtest.get("max_drawdown", 1.0))
        violations = int(backtest.get("violations", 999))

        completeness_ok = completeness >= float(self.settings.validation.get("data_completeness_min", 0.99))
        unresolved_ok = unresolved <= float(self.settings.validation.get("unresolved_conflict_max", 0.005))
        positive_ok = positive_ratio >= float(self.settings.validation.get("positive_window_ratio_min", 0.70))
        drawdown_ok = max_drawdown <= float(self.settings.validation.get("max_drawdown_max", 0.18))
        violations_ok = violations == 0
        health_ok = bool(health.get("status") == "healthy")
        replay_ok = bool(replay.get("passed", False))

        checks = {
            "review_pass_gate": review_pass,
            "mode_health_ok": mode_health_ok,
            "state_stability_ok": state_stability_ok,
            "slot_anomaly_ok": slot_anomaly_ok,
            "mode_drift_ok": mode_drift_ok,
            "reconcile_drift_ok": reconcile_drift_ok,
            "tests_ok": tests_ok,
            "health_ok": health_ok,
            "stable_replay_ok": replay_ok,
            "data_completeness_ok": completeness_ok,
            "unresolved_conflict_ok": unresolved_ok,
            "positive_window_ratio_ok": positive_ok,
            "max_drawdown_ok": drawdown_ok,
            "risk_violations_ok": violations_ok,
        }
        rollback_recommendation = self._rollback_recommendation(
            as_of=as_of,
            checks=checks,
            state_stability=state_stability,
            slot_anomaly=slot_anomaly,
            mode_drift=mode_drift,
            reconcile_drift=reconcile_drift,
        )
        checks["rollback_anchor_ready"] = bool(rollback_recommendation.get("anchor_ready", True))
        overall = all(checks.values())
        out = {
            "date": d,
            "passed": overall,
            "checks": checks,
            "metrics": {
                "completeness": completeness,
                "unresolved_conflict_ratio": unresolved,
                "positive_window_ratio": positive_ratio,
                "max_drawdown": max_drawdown,
                "violations": violations,
            },
            "health": health,
            "stable_replay": replay,
            "state_stability": state_stability,
            "slot_anomaly": slot_anomaly,
            "mode_drift": mode_drift,
            "reconcile_drift": reconcile_drift,
            "rollback_recommendation": rollback_recommendation,
            "tests": tests_payload if run_tests else {"skipped": True},
        }

        if overall:
            alert_path = self.output_dir / "logs" / f"review_loop_alert_{d}.json"
            if alert_path.exists():
                try:
                    alert_path.unlink()
                except OSError:
                    pass

        report_path = self.output_dir / "review" / f"{d}_gate_report.json"
        write_json(report_path, out)
        return out

    def _run_tests(
        self,
        *,
        fast: bool,
        fast_ratio: float,
        fast_shard_index: int,
        fast_shard_total: int,
        fast_seed: str,
    ) -> dict[str, Any]:
        if not fast:
            return self.test_all()
        try:
            return self.test_all(
                fast=True,
                fast_ratio=float(fast_ratio),
                fast_shard_index=int(fast_shard_index),
                fast_shard_total=int(fast_shard_total),
                fast_seed=str(fast_seed),
            )
        except TypeError:
            # Backward compatibility for legacy callables that don't accept kwargs.
            return self.test_all()

    def _latest_test_result(self) -> dict[str, Any]:
        logs_dir = self.output_dir / "logs"
        candidates = sorted(logs_dir.glob("tests_*.json"))
        if not candidates:
            return {"found": False}
        latest = candidates[-1]
        payload = self.load_json_safely(latest)
        return {
            "found": True,
            "path": str(latest),
            "returncode": payload.get("returncode"),
            "has_output": bool(payload.get("stdout") or payload.get("stderr") or payload.get("stdout_excerpt") or payload.get("stderr_excerpt")),
        }

    def _load_mode_feedback_series(self, *, as_of: date, window_days: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        wd = max(1, int(window_days))
        daily_dir = self.output_dir / "daily"
        for i in range(wd):
            day = as_of - timedelta(days=i)
            path = daily_dir / f"{day.isoformat()}_mode_feedback.json"
            payload = self.load_json_safely(path)
            if not payload:
                continue
            risk_control = payload.get("risk_control", {}) if isinstance(payload.get("risk_control", {}), dict) else {}
            mode_health = payload.get("mode_health", {}) if isinstance(payload.get("mode_health", {}), dict) else {}
            out.append(
                {
                    "date": day.isoformat(),
                    "runtime_mode": str(payload.get("runtime_mode", "")).strip(),
                    "risk_multiplier": self._safe_float(risk_control.get("risk_multiplier", 1.0), 1.0),
                    "source_confidence_score": self._safe_float(risk_control.get("source_confidence_score", 1.0), 1.0),
                    "mode_health_passed": bool(mode_health.get("passed", True)),
                }
            )
        out.reverse()
        return out

    def _effective_sqlite_path(self) -> Path:
        if self.sqlite_path is not None:
            return Path(self.sqlite_path)
        raw = str(self.settings.paths.get("sqlite", "output/artifacts/lie_engine.db")).strip()
        path = Path(raw)
        if path.is_absolute():
            return path
        return self.output_dir.parent / path

    def _latest_mode_feedback_payload(self, *, as_of: date, lookback_days: int = 30) -> dict[str, Any]:
        daily_dir = self.output_dir / "daily"
        for i in range(max(1, int(lookback_days))):
            day = as_of - timedelta(days=i)
            path = daily_dir / f"{day.isoformat()}_mode_feedback.json"
            payload = self.load_json_safely(path)
            if payload:
                return payload
        return {}

    def _slot_config(self) -> dict[str, Any]:
        schedule = self.settings.schedule if isinstance(self.settings.schedule, dict) else {}
        intraday_slots = schedule.get("intraday_slots", ["10:30", "14:30"])
        if not isinstance(intraday_slots, list):
            intraday_slots = ["10:30", "14:30"]
        out_slots: list[str] = []
        for raw in intraday_slots:
            txt = str(raw).strip()
            if txt:
                out_slots.append(txt)
        if not out_slots:
            out_slots = ["10:30", "14:30"]
        return {"intraday_slots": out_slots}

    def _slot_anomaly_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        cfg = self._slot_config()
        intraday_slots = list(cfg.get("intraday_slots", ["10:30", "14:30"]))
        window_days = max(1, int(val.get("ops_slot_window_days", 7)))
        min_samples = max(1, int(val.get("ops_slot_min_samples", 3)))
        pre_ratio_max = self._safe_float(val.get("ops_slot_premarket_anomaly_ratio_max", 0.50), 0.50)
        intraday_ratio_max = self._safe_float(val.get("ops_slot_intraday_anomaly_ratio_max", 0.50), 0.50)
        eod_ratio_max = self._safe_float(val.get("ops_slot_eod_anomaly_ratio_max", 0.50), 0.50)
        missing_ratio_max = self._safe_float(val.get("ops_slot_missing_ratio_max", 0.35), 0.35)
        source_floor = self._safe_float(
            val.get("ops_slot_source_confidence_floor", val.get("source_confidence_min", 0.75)),
            0.75,
        )
        risk_floor = self._safe_float(
            val.get("ops_slot_risk_multiplier_floor", val.get("execution_min_risk_multiplier", 0.20)),
            0.20,
        )

        def _slot_template() -> dict[str, Any]:
            return {"expected": 0, "observed": 0, "missing": 0, "anomalies": 0}

        slots = {
            "premarket": _slot_template(),
            "intraday": _slot_template(),
            "eod": _slot_template(),
        }
        series: list[dict[str, Any]] = []
        active_days = 0
        total_expected = 0
        total_observed = 0
        total_missing = 0
        total_anomalies = 0

        logs_dir = self.output_dir / "logs"
        manifest_dir = self.output_dir / "artifacts" / "manifests"

        for i in range(window_days):
            day = as_of - timedelta(days=i)
            dstr = day.isoformat()
            local = {
                "premarket": _slot_template(),
                "intraday": _slot_template(),
                "eod": _slot_template(),
            }
            day_alerts: list[str] = []

            # premarket
            local["premarket"]["expected"] += 1
            pre = self.load_json_safely(logs_dir / f"{dstr}_premarket.json")
            if pre:
                local["premarket"]["observed"] += 1
                quality = pre.get("quality", {}) if isinstance(pre.get("quality", {}), dict) else {}
                q_flags = quality.get("flags", []) if isinstance(quality.get("flags", []), list) else []
                score = self._safe_float(quality.get("source_confidence_score", pre.get("source_confidence_score", 1.0)), 1.0)
                risk_mult = self._safe_float(pre.get("risk_multiplier", 1.0), 1.0)
                reasons: list[str] = []
                if not bool(quality.get("passed", True)):
                    reasons.append("quality_failed")
                if q_flags:
                    reasons.append("quality_flags")
                if score < source_floor:
                    reasons.append("source_confidence_low")
                if risk_mult < risk_floor:
                    reasons.append("risk_multiplier_low")
                if reasons:
                    local["premarket"]["anomalies"] += 1
                    day_alerts.append("premarket:" + "+".join(reasons[:2]))
            else:
                local["premarket"]["missing"] += 1
                day_alerts.append("premarket:missing")

            # intraday
            for slot in intraday_slots:
                local["intraday"]["expected"] += 1
                intraday = self.load_json_safely(logs_dir / f"{dstr}_intraday_{slot.replace(':', '')}.json")
                if intraday:
                    local["intraday"]["observed"] += 1
                    q_flags = intraday.get("quality_flags", []) if isinstance(intraday.get("quality_flags", []), list) else []
                    score = self._safe_float(intraday.get("source_confidence_score", 1.0), 1.0)
                    risk_mult = self._safe_float(intraday.get("risk_multiplier", 1.0), 1.0)
                    reasons = []
                    if q_flags:
                        reasons.append("quality_flags")
                    if score < source_floor:
                        reasons.append("source_confidence_low")
                    if risk_mult < risk_floor:
                        reasons.append("risk_multiplier_low")
                    if reasons:
                        local["intraday"]["anomalies"] += 1
                        day_alerts.append("intraday:" + "+".join(reasons[:2]))
                else:
                    local["intraday"]["missing"] += 1
                    day_alerts.append(f"intraday:{slot}:missing")

            # eod manifest
            local["eod"]["expected"] += 1
            eod = self.load_json_safely(manifest_dir / f"eod_{dstr}.json")
            if eod:
                local["eod"]["observed"] += 1
                checks = eod.get("checks", {}) if isinstance(eod.get("checks", {}), dict) else {}
                metrics = eod.get("metrics", {}) if isinstance(eod.get("metrics", {}), dict) else {}
                reasons = []
                if not bool(checks.get("quality_passed", True)):
                    reasons.append("quality_failed")
                risk_mult = self._safe_float(metrics.get("risk_multiplier", 1.0), 1.0)
                if risk_mult < risk_floor:
                    reasons.append("risk_multiplier_low")
                if reasons:
                    local["eod"]["anomalies"] += 1
                    day_alerts.append("eod:" + "+".join(reasons[:2]))
            else:
                local["eod"]["missing"] += 1
                day_alerts.append("eod:missing")

            day_expected = sum(int(local[k]["expected"]) for k in local)
            day_observed = sum(int(local[k]["observed"]) for k in local)
            day_missing = sum(int(local[k]["missing"]) for k in local)
            day_anomalies = sum(int(local[k]["anomalies"]) for k in local)

            # only evaluate days that have at least one produced slot artifact
            if day_observed <= 0:
                continue

            active_days += 1
            total_expected += day_expected
            total_observed += day_observed
            total_missing += day_missing
            total_anomalies += day_anomalies
            for key in slots:
                for field in ("expected", "observed", "missing", "anomalies"):
                    slots[key][field] = int(slots[key][field]) + int(local[key][field])
            series.append(
                {
                    "date": dstr,
                    "expected": day_expected,
                    "observed": day_observed,
                    "missing": day_missing,
                    "anomalies": day_anomalies,
                    "alerts": day_alerts[:6],
                }
            )

        def _ratio(n: int, d: int) -> float:
            return float(n / d) if d > 0 else 0.0

        pre_ratio = _ratio(int(slots["premarket"]["anomalies"]), int(slots["premarket"]["expected"]))
        intraday_ratio = _ratio(int(slots["intraday"]["anomalies"]), int(slots["intraday"]["expected"]))
        eod_ratio = _ratio(int(slots["eod"]["anomalies"]), int(slots["eod"]["expected"]))
        missing_ratio = _ratio(total_missing, total_expected)

        active = active_days >= min_samples
        checks = {
            "missing_ratio_ok": True,
            "premarket_anomaly_ok": True,
            "intraday_anomaly_ok": True,
            "eod_anomaly_ok": True,
        }
        alerts: list[str] = []
        if active:
            checks["missing_ratio_ok"] = bool(missing_ratio <= missing_ratio_max)
            checks["premarket_anomaly_ok"] = bool(pre_ratio <= pre_ratio_max)
            checks["intraday_anomaly_ok"] = bool(intraday_ratio <= intraday_ratio_max)
            checks["eod_anomaly_ok"] = bool(eod_ratio <= eod_ratio_max)
            if not checks["missing_ratio_ok"]:
                alerts.append("slot_missing_ratio_high")
            if not checks["premarket_anomaly_ok"]:
                alerts.append("slot_premarket_anomaly_high")
            if not checks["intraday_anomaly_ok"]:
                alerts.append("slot_intraday_anomaly_high")
            if not checks["eod_anomaly_ok"]:
                alerts.append("slot_eod_anomaly_high")
        else:
            alerts.append("insufficient_slot_samples")

        for key in slots:
            expected = int(slots[key]["expected"])
            anomalies = int(slots[key]["anomalies"])
            missing = int(slots[key]["missing"])
            slots[key]["anomaly_ratio"] = _ratio(anomalies, expected)
            slots[key]["missing_ratio"] = _ratio(missing, expected)

        series.reverse()
        return {
            "active": active,
            "window_days": window_days,
            "samples": active_days,
            "min_samples": min_samples,
            "metrics": {
                "expected_slots": total_expected,
                "observed_slots": total_observed,
                "missing_slots": total_missing,
                "anomaly_slots": total_anomalies,
                "missing_ratio": missing_ratio,
                "premarket_anomaly_ratio": pre_ratio,
                "intraday_anomaly_ratio": intraday_ratio,
                "eod_anomaly_ratio": eod_ratio,
            },
            "thresholds": {
                "ops_slot_missing_ratio_max": missing_ratio_max,
                "ops_slot_premarket_anomaly_ratio_max": pre_ratio_max,
                "ops_slot_intraday_anomaly_ratio_max": intraday_ratio_max,
                "ops_slot_eod_anomaly_ratio_max": eod_ratio_max,
                "ops_slot_source_confidence_floor": source_floor,
                "ops_slot_risk_multiplier_floor": risk_floor,
            },
            "checks": checks,
            "alerts": alerts,
            "slots": slots,
            "series": series[-10:],
        }

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        n = float(num)
        d = float(den)
        return float(n / d) if d > 0 else 0.0

    @staticmethod
    def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
        try:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                    (str(table),),
                )
                return cur.fetchone() is not None
        except Exception:
            return False

    def _csv_plan_summary(self, *, day: str) -> dict[str, Any]:
        path = self.output_dir / "daily" / f"{day}_positions.csv"
        if not path.exists():
            return {"found": False, "path": str(path), "rows": 0, "active_rows": 0, "exposure": 0.0}
        rows = 0
        active_rows = 0
        exposure = 0.0
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows += 1
                    status = str(row.get("status", "ACTIVE") or "ACTIVE").strip().upper()
                    if status == "ACTIVE":
                        active_rows += 1
                    exposure += abs(self._safe_float(row.get("size_pct", 0.0), 0.0))
        except Exception:
            return {"found": False, "path": str(path), "rows": 0, "active_rows": 0, "exposure": 0.0}
        return {
            "found": True,
            "path": str(path),
            "rows": int(rows),
            "active_rows": int(active_rows),
            "exposure": float(exposure),
        }

    def _reconcile_drift_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        window_days = max(1, int(val.get("ops_reconcile_window_days", 7)))
        min_samples = max(1, int(val.get("ops_reconcile_min_samples", 3)))
        missing_ratio_max = self._safe_float(val.get("ops_reconcile_missing_ratio_max", 0.35), 0.35)
        plan_gap_ratio_max = self._safe_float(val.get("ops_reconcile_plan_gap_ratio_max", 0.10), 0.10)
        closed_count_gap_ratio_max = self._safe_float(val.get("ops_reconcile_closed_count_gap_ratio_max", 0.10), 0.10)
        closed_pnl_gap_abs_max = abs(self._safe_float(val.get("ops_reconcile_closed_pnl_gap_abs_max", 0.001), 0.001))
        open_gap_ratio_max = self._safe_float(val.get("ops_reconcile_open_gap_ratio_max", 0.25), 0.25)

        db_path = self._effective_sqlite_path()
        conn: sqlite3.Connection | None = None
        has_latest_positions = False
        has_executed_plans = False
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                has_latest_positions = self._sqlite_table_exists(conn, "latest_positions")
                has_executed_plans = self._sqlite_table_exists(conn, "executed_plans")
            except Exception:
                conn = None

        series: list[dict[str, Any]] = []
        samples = 0
        missing_days = 0
        plan_gap_breach_days = 0
        closed_count_gap_breach_days = 0
        closed_pnl_gap_breach_days = 0
        open_gap_breach_days = 0
        open_gap_samples = 0
        total_plan_gap = 0.0
        total_closed_count_gap = 0.0
        total_closed_pnl_gap = 0.0
        total_open_gap = 0.0

        manifest_dir = self.output_dir / "artifacts" / "manifests"
        paper_state = self.load_json_safely(self.output_dir / "artifacts" / "paper_positions_open.json")
        paper_state_as_of = str(paper_state.get("as_of", "")).strip()
        paper_positions = paper_state.get("positions", []) if isinstance(paper_state.get("positions", []), list) else []

        for i in range(window_days):
            day = as_of - timedelta(days=i)
            dstr = day.isoformat()
            manifest = self.load_json_safely(manifest_dir / f"eod_{dstr}.json")
            if not manifest:
                continue

            metrics = manifest.get("metrics", {}) if isinstance(manifest.get("metrics", {}), dict) else {}
            required_metric_keys = {"plans", "closed_trades", "closed_pnl", "open_positions"}
            if not required_metric_keys.issubset(set(metrics.keys())):
                continue

            samples += 1
            manifest_plans = int(self._safe_float(metrics.get("plans", 0), 0.0))
            manifest_closed_count = int(self._safe_float(metrics.get("closed_trades", 0), 0.0))
            manifest_closed_pnl = self._safe_float(metrics.get("closed_pnl", 0.0), 0.0)
            manifest_open_count = int(self._safe_float(metrics.get("open_positions", 0), 0.0))

            csv_summary = self._csv_plan_summary(day=dstr)
            plan_db_count: int | None = None
            closed_db_count: int | None = None
            closed_db_pnl: float | None = None
            day_missing = False
            day_alerts: list[str] = []
            plan_gap_breached = False
            closed_count_gap_breached = False
            closed_pnl_gap_breached = False
            open_gap_breached = False

            if conn is not None and has_latest_positions:
                try:
                    with closing(conn.cursor()) as cur:
                        cur.execute(
                            "SELECT COUNT(*) FROM latest_positions WHERE date = ?",
                            (dstr,),
                        )
                        row = cur.fetchone()
                        plan_db_count = int(row[0] if row else 0)
                except Exception:
                    plan_db_count = None
            if plan_db_count is None:
                if manifest_plans <= 0:
                    plan_db_count = 0
                else:
                    day_missing = True
                    day_alerts.append("latest_positions_missing")

            if conn is not None and has_executed_plans:
                try:
                    with closing(conn.cursor()) as cur:
                        cur.execute(
                            "SELECT COUNT(*), COALESCE(SUM(pnl), 0.0) FROM executed_plans "
                            "WHERE date = ? AND (status = 'CLOSED' OR status IS NULL)",
                            (dstr,),
                        )
                        row = cur.fetchone()
                        closed_db_count = int(row[0] if row else 0)
                        closed_db_pnl = self._safe_float(row[1] if row else 0.0, 0.0)
                except Exception:
                    closed_db_count = None
                    closed_db_pnl = None
            if closed_db_count is None:
                if manifest_closed_count <= 0 and abs(manifest_closed_pnl) <= 1e-12:
                    closed_db_count = 0
                    closed_db_pnl = 0.0
                else:
                    day_missing = True
                    day_alerts.append("executed_plans_missing")

            if not bool(csv_summary.get("found", False)):
                day_missing = True
                day_alerts.append("positions_csv_missing")

            plan_gap_ratio = 0.0
            closed_count_gap_ratio = 0.0
            closed_pnl_gap_abs = 0.0
            open_gap_ratio = 0.0

            if plan_db_count is not None:
                plan_gap_ratio = self._ratio(abs(plan_db_count - manifest_plans), max(1, manifest_plans))
                total_plan_gap += plan_gap_ratio
                if plan_gap_ratio > plan_gap_ratio_max:
                    plan_gap_breached = True
                    day_alerts.append("plan_count_gap_high")
                if bool(csv_summary.get("found", False)):
                    csv_count = int(self._safe_float(csv_summary.get("rows", 0), 0.0))
                    csv_db_gap = self._ratio(abs(csv_count - plan_db_count), max(1, plan_db_count))
                    if csv_db_gap > plan_gap_ratio_max:
                        plan_gap_breached = True
                        day_alerts.append("plan_csv_db_gap_high")

            if closed_db_count is not None and closed_db_pnl is not None:
                closed_count_gap_ratio = self._ratio(
                    abs(closed_db_count - manifest_closed_count),
                    max(1, manifest_closed_count),
                )
                closed_pnl_gap_abs = abs(closed_db_pnl - manifest_closed_pnl)
                total_closed_count_gap += closed_count_gap_ratio
                total_closed_pnl_gap += closed_pnl_gap_abs
                if closed_count_gap_ratio > closed_count_gap_ratio_max:
                    closed_count_gap_breached = True
                    day_alerts.append("closed_count_gap_high")
                if closed_pnl_gap_abs > closed_pnl_gap_abs_max:
                    closed_pnl_gap_breached = True
                    day_alerts.append("closed_pnl_gap_high")

            if day == as_of and paper_state_as_of == dstr:
                open_gap_samples += 1
                open_gap_ratio = self._ratio(abs(len(paper_positions) - manifest_open_count), max(1, manifest_open_count))
                total_open_gap += open_gap_ratio
                if open_gap_ratio > open_gap_ratio_max:
                    open_gap_breached = True
                    day_alerts.append("open_count_gap_high")
            elif day == as_of:
                if manifest_open_count <= 0:
                    open_gap_samples += 1
                    open_gap_ratio = 0.0
                else:
                    day_missing = True
                    day_alerts.append("paper_state_missing_or_stale")

            if plan_gap_breached:
                plan_gap_breach_days += 1
            if closed_count_gap_breached:
                closed_count_gap_breach_days += 1
            if closed_pnl_gap_breached:
                closed_pnl_gap_breach_days += 1
            if open_gap_breached:
                open_gap_breach_days += 1

            if day_missing:
                missing_days += 1

            series.append(
                {
                    "date": dstr,
                    "missing": bool(day_missing),
                    "manifest": {
                        "plans": manifest_plans,
                        "closed_trades": manifest_closed_count,
                        "closed_pnl": manifest_closed_pnl,
                        "open_positions": manifest_open_count,
                    },
                    "csv": {
                        "found": bool(csv_summary.get("found", False)),
                        "rows": int(self._safe_float(csv_summary.get("rows", 0), 0.0)),
                    },
                    "db": {
                        "latest_positions_rows": int(plan_db_count) if plan_db_count is not None else None,
                        "executed_closed_rows": int(closed_db_count) if closed_db_count is not None else None,
                        "executed_closed_pnl": float(closed_db_pnl) if closed_db_pnl is not None else None,
                    },
                    "gaps": {
                        "plan_count_ratio": plan_gap_ratio,
                        "closed_count_ratio": closed_count_gap_ratio,
                        "closed_pnl_abs": closed_pnl_gap_abs,
                        "open_count_ratio": open_gap_ratio,
                    },
                    "alerts": day_alerts[:8],
                }
            )

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        active = samples >= min_samples
        missing_ratio = self._ratio(missing_days, samples)
        plan_gap_breach_ratio = self._ratio(plan_gap_breach_days, samples)
        closed_count_gap_breach_ratio = self._ratio(closed_count_gap_breach_days, samples)
        closed_pnl_gap_breach_ratio = self._ratio(closed_pnl_gap_breach_days, samples)
        open_gap_breach_ratio = self._ratio(open_gap_breach_days, open_gap_samples) if open_gap_samples > 0 else 0.0
        avg_plan_gap_ratio = self._ratio(total_plan_gap, samples)
        avg_closed_count_gap_ratio = self._ratio(total_closed_count_gap, samples)
        avg_closed_pnl_gap_abs = self._ratio(total_closed_pnl_gap, samples)
        avg_open_gap_ratio = self._ratio(total_open_gap, open_gap_samples) if open_gap_samples > 0 else 0.0

        checks = {
            "missing_ratio_ok": True,
            "plan_count_gap_ok": True,
            "closed_count_gap_ok": True,
            "closed_pnl_gap_ok": True,
            "open_count_gap_ok": True,
        }
        alerts: list[str] = []
        if active:
            checks["missing_ratio_ok"] = bool(missing_ratio <= missing_ratio_max)
            checks["plan_count_gap_ok"] = bool(plan_gap_breach_ratio <= plan_gap_ratio_max)
            checks["closed_count_gap_ok"] = bool(closed_count_gap_breach_ratio <= closed_count_gap_ratio_max)
            checks["closed_pnl_gap_ok"] = bool(closed_pnl_gap_breach_ratio <= closed_count_gap_ratio_max)
            if open_gap_samples > 0:
                checks["open_count_gap_ok"] = bool(open_gap_breach_ratio <= open_gap_ratio_max)
            if not checks["missing_ratio_ok"]:
                alerts.append("reconcile_missing_ratio_high")
            if not checks["plan_count_gap_ok"]:
                alerts.append("reconcile_plan_count_gap_high")
            if not checks["closed_count_gap_ok"]:
                alerts.append("reconcile_closed_count_gap_high")
            if not checks["closed_pnl_gap_ok"]:
                alerts.append("reconcile_closed_pnl_gap_high")
            if not checks["open_count_gap_ok"]:
                alerts.append("reconcile_open_count_gap_high")
        else:
            alerts.append("insufficient_reconcile_samples")

        series.reverse()
        return {
            "active": active,
            "window_days": window_days,
            "samples": samples,
            "min_samples": min_samples,
            "metrics": {
                "missing_days": missing_days,
                "missing_ratio": missing_ratio,
                "plan_gap_breach_ratio": plan_gap_breach_ratio,
                "closed_count_gap_breach_ratio": closed_count_gap_breach_ratio,
                "closed_pnl_gap_breach_ratio": closed_pnl_gap_breach_ratio,
                "open_gap_breach_ratio": open_gap_breach_ratio,
                "avg_plan_gap_ratio": avg_plan_gap_ratio,
                "avg_closed_count_gap_ratio": avg_closed_count_gap_ratio,
                "avg_closed_pnl_gap_abs": avg_closed_pnl_gap_abs,
                "avg_open_gap_ratio": avg_open_gap_ratio,
                "open_gap_samples": open_gap_samples,
            },
            "thresholds": {
                "ops_reconcile_missing_ratio_max": missing_ratio_max,
                "ops_reconcile_plan_gap_ratio_max": plan_gap_ratio_max,
                "ops_reconcile_closed_count_gap_ratio_max": closed_count_gap_ratio_max,
                "ops_reconcile_closed_pnl_gap_abs_max": closed_pnl_gap_abs_max,
                "ops_reconcile_open_gap_ratio_max": open_gap_ratio_max,
            },
            "checks": checks,
            "alerts": alerts,
            "series": series[-10:],
        }

    def _rollback_candidates(self, *, as_of: date, lookback_days: int) -> list[dict[str, str]]:
        artifacts_dir = self.output_dir / "artifacts"
        review_dir = self.output_dir / "review"
        out: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        max_days = max(1, int(lookback_days))

        for path in sorted(artifacts_dir.glob("params_live_backup_*.yaml"), reverse=True):
            tag = path.stem.replace("params_live_backup_", "").strip()
            try:
                d = date.fromisoformat(tag)
            except Exception:
                continue
            if d > as_of:
                continue
            key = str(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            out.append({"date": d.isoformat(), "path": key, "source": "params_backup"})
            if len(out) >= max_days:
                break

        for i in range(max_days):
            day = as_of - timedelta(days=i)
            payload = self.load_json_safely(review_dir / f"{day.isoformat()}_param_delta.yaml")
            anchor = str(payload.get("rollback_anchor", "")).strip()
            if not anchor or anchor == "initial_seed":
                continue
            key = str(Path(anchor))
            if key in seen_paths:
                continue
            if not Path(anchor).exists():
                continue
            seen_paths.add(key)
            out.append({"date": day.isoformat(), "path": key, "source": "review_delta"})
            if len(out) >= max_days:
                break

        out.sort(key=lambda x: str(x.get("date", "")), reverse=True)
        return out

    def _rollback_recommendation(
        self,
        *,
        as_of: date,
        checks: dict[str, Any],
        state_stability: dict[str, Any],
        slot_anomaly: dict[str, Any],
        mode_drift: dict[str, Any],
        reconcile_drift: dict[str, Any],
    ) -> dict[str, Any]:
        score = 0
        reason_codes: list[str] = []

        if not bool(checks.get("risk_violations_ok", True)):
            score += 4
            reason_codes.append("risk_violations")
        if not bool(checks.get("max_drawdown_ok", True)):
            score += 3
            reason_codes.append("max_drawdown")
        if not bool(checks.get("stable_replay_ok", True)):
            score += 2
            reason_codes.append("stable_replay")
        if not bool(checks.get("health_ok", True)):
            score += 2
            reason_codes.append("health_degraded")
        if not bool(checks.get("state_stability_ok", True)):
            score += 2
            reason_codes.append("state_stability")
        if not bool(checks.get("mode_drift_ok", True)):
            score += 2
            reason_codes.append("mode_drift")
        if not bool(checks.get("reconcile_drift_ok", True)):
            score += 2
            reason_codes.append("reconcile_drift")
        if not bool(checks.get("slot_anomaly_ok", True)):
            score += 1
            reason_codes.append("slot_anomaly")
        if not bool(checks.get("tests_ok", True)):
            score += 1
            reason_codes.append("tests")
        if not bool(checks.get("review_pass_gate", True)):
            score += 1
            reason_codes.append("review_gate")

        hard_reasons = {"risk_violations", "max_drawdown"}
        has_hard_reason = any(code in hard_reasons for code in reason_codes)
        level = "none"
        if has_hard_reason or score >= 7:
            level = "hard"
        elif score >= 4:
            level = "soft"

        candidates = self._rollback_candidates(as_of=as_of, lookback_days=30)
        target_anchor = ""
        if level != "none":
            for item in candidates:
                tag = str(item.get("date", "")).strip()
                if tag and tag < as_of.isoformat():
                    target_anchor = str(item.get("path", "")).strip()
                    break
            if not target_anchor and candidates:
                target_anchor = str(candidates[0].get("path", "")).strip()

        anchor_ready = True
        if level != "none":
            anchor_ready = bool(target_anchor and Path(target_anchor).exists())
        action = "no_rollback"
        if level == "soft":
            action = "rollback_to_last_stable_anchor_after_partial_recheck"
        elif level == "hard":
            action = "rollback_now_and_lock_parameter_updates"

        state_alerts = state_stability.get("alerts", []) if isinstance(state_stability.get("alerts", []), list) else []
        slot_alerts = slot_anomaly.get("alerts", []) if isinstance(slot_anomaly.get("alerts", []), list) else []
        drift_alerts = mode_drift.get("alerts", []) if isinstance(mode_drift.get("alerts", []), list) else []
        reconcile_alerts = (
            reconcile_drift.get("alerts", []) if isinstance(reconcile_drift.get("alerts", []), list) else []
        )

        return {
            "active": level != "none",
            "level": level,
            "score": score,
            "reason_codes": reason_codes,
            "action": action,
            "target_anchor": target_anchor,
            "anchor_ready": bool(anchor_ready),
            "cooldown_days": 3 if level == "hard" else (1 if level == "soft" else 0),
            "candidates": candidates[:10],
            "alerts": list(state_alerts[:2]) + list(slot_alerts[:2]) + list(drift_alerts[:2]) + list(reconcile_alerts[:2]),
        }

    def _live_mode_metrics(self, *, as_of: date, window_days: int) -> dict[str, dict[str, float]]:
        db_path = self._effective_sqlite_path()
        if not db_path.exists():
            return {}

        start = as_of - timedelta(days=max(1, int(window_days)) - 1)
        sql = (
            "SELECT date, runtime_mode, mode, pnl "
            "FROM executed_plans "
            "WHERE date >= ? AND date <= ? "
            "ORDER BY date ASC"
        )
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cur:
                    cur.execute(sql, (start.isoformat(), as_of.isoformat()))
                    rows = cur.fetchall()
        except Exception:
            return {}

        buckets: dict[str, dict[str, float]] = {}
        for row in rows:
            mode_raw = str(row["runtime_mode"] or row["mode"] or "").strip()
            mode = mode_raw or "base"
            pnl = self._safe_float(row["pnl"], 0.0)
            b = buckets.setdefault(
                mode,
                {
                    "trades": 0.0,
                    "wins": 0.0,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                },
            )
            b["trades"] += 1.0
            if pnl > 0:
                b["wins"] += 1.0
                b["gross_profit"] += pnl
            elif pnl < 0:
                b["gross_loss"] += abs(pnl)

        out: dict[str, dict[str, float]] = {}
        for mode, b in buckets.items():
            trades = max(1.0, float(b["trades"]))
            gp = float(b["gross_profit"])
            gl = float(b["gross_loss"])
            if gl > 1e-9:
                pf = gp / gl
            else:
                pf = 10.0 if gp > 0 else 0.0
            out[mode] = {
                "trades": float(b["trades"]),
                "win_rate": float(b["wins"] / trades),
                "profit_factor": float(pf),
            }
        return out

    def _mode_drift_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        window_days = max(1, int(val.get("mode_drift_window_days", 120)))
        min_live_trades = max(1, int(val.get("mode_drift_min_live_trades", 30)))
        max_wr_gap = self._safe_float(val.get("mode_drift_win_rate_max_gap", 0.12), 0.12)
        max_pf_gap = self._safe_float(val.get("mode_drift_profit_factor_max_gap", 0.40), 0.40)
        focus_runtime_only = bool(val.get("mode_drift_focus_runtime_mode_only", True))

        latest_feedback = self._latest_mode_feedback_payload(as_of=as_of, lookback_days=max(30, window_days))
        runtime_mode = str(latest_feedback.get("runtime_mode", "")).strip()
        history = latest_feedback.get("history", {}) if isinstance(latest_feedback.get("history", {}), dict) else {}
        baseline_modes = history.get("modes", {}) if isinstance(history.get("modes", {}), dict) else {}
        live_modes = self._live_mode_metrics(as_of=as_of, window_days=window_days)

        scope_modes: list[str]
        if focus_runtime_only and runtime_mode:
            scope_modes = [runtime_mode]
        else:
            names = set(str(k).strip() for k in baseline_modes.keys()) | set(str(k).strip() for k in live_modes.keys())
            scope_modes = sorted([x for x in names if x])

        checks = {
            "samples_ok": True,
            "win_rate_gap_ok": True,
            "profit_factor_gap_ok": True,
        }
        alerts: list[str] = []
        mode_rows: dict[str, Any] = {}
        compared_modes = 0
        active_modes = 0

        for mode in scope_modes:
            baseline = baseline_modes.get(mode, {}) if isinstance(baseline_modes.get(mode, {}), dict) else {}
            live = live_modes.get(mode, {}) if isinstance(live_modes.get(mode, {}), dict) else {}
            base_wr = self._safe_float(baseline.get("avg_win_rate", 0.0), 0.0)
            base_pf = self._safe_float(baseline.get("avg_profit_factor", 0.0), 0.0)
            live_wr = self._safe_float(live.get("win_rate", 0.0), 0.0)
            live_pf = self._safe_float(live.get("profit_factor", 0.0), 0.0)
            live_trades = int(self._safe_float(live.get("trades", 0.0), 0.0))
            baseline_samples = int(self._safe_float(baseline.get("samples", 0.0), 0.0))

            row = {
                "baseline": {
                    "samples": baseline_samples,
                    "win_rate": base_wr,
                    "profit_factor": base_pf,
                },
                "live": {
                    "trades": live_trades,
                    "win_rate": live_wr,
                    "profit_factor": live_pf,
                },
                "gaps": {
                    "win_rate_abs": abs(live_wr - base_wr),
                    "profit_factor_abs": abs(live_pf - base_pf),
                },
                "checks": {
                    "baseline_ok": baseline_samples > 0,
                    "samples_ok": live_trades >= min_live_trades,
                    "win_rate_gap_ok": True,
                    "profit_factor_gap_ok": True,
                },
                "active": False,
                "reason": "",
            }

            if baseline_samples <= 0:
                row["reason"] = "missing_backtest_baseline"
                checks["samples_ok"] = False
                alerts.append(f"mode_drift_missing_baseline:{mode}")
            elif live_trades < min_live_trades:
                row["reason"] = "insufficient_live_trades"
                checks["samples_ok"] = False
                alerts.append(f"mode_drift_insufficient_live:{mode}")
            else:
                active_modes += 1
                row["active"] = True
                compared_modes += 1
                wr_gap = self._safe_float(row["gaps"]["win_rate_abs"], 0.0)
                pf_gap = self._safe_float(row["gaps"]["profit_factor_abs"], 0.0)
                row["checks"]["win_rate_gap_ok"] = bool(wr_gap <= max_wr_gap)
                row["checks"]["profit_factor_gap_ok"] = bool(pf_gap <= max_pf_gap)
                if not bool(row["checks"]["win_rate_gap_ok"]):
                    checks["win_rate_gap_ok"] = False
                    alerts.append(f"mode_drift_win_rate:{mode}")
                if not bool(row["checks"]["profit_factor_gap_ok"]):
                    checks["profit_factor_gap_ok"] = False
                    alerts.append(f"mode_drift_profit_factor:{mode}")
                row["reason"] = "ok" if (row["checks"]["win_rate_gap_ok"] and row["checks"]["profit_factor_gap_ok"]) else "drift_breach"
            mode_rows[mode] = row

        active = bool(compared_modes > 0)
        if not scope_modes:
            alerts.append("mode_drift_scope_empty")
        elif not active:
            alerts.append("mode_drift_inactive")

        return {
            "active": active,
            "window_days": window_days,
            "runtime_mode": runtime_mode,
            "focus_runtime_mode_only": focus_runtime_only,
            "min_live_trades": min_live_trades,
            "checks": checks,
            "thresholds": {
                "mode_drift_win_rate_max_gap": max_wr_gap,
                "mode_drift_profit_factor_max_gap": max_pf_gap,
            },
            "summary": {
                "scope_modes": int(len(scope_modes)),
                "compared_modes": int(compared_modes),
                "active_modes": int(active_modes),
            },
            "alerts": alerts,
            "modes": mode_rows,
        }

    def _state_stability_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        window_days = max(3, int(val.get("mode_switch_window_days", 20)))
        min_samples = max(1, int(val.get("ops_state_min_samples", 5)))
        switch_rate_max = self._safe_float(val.get("mode_switch_max_rate", 0.45), 0.45)
        risk_floor = self._safe_float(
            val.get("ops_risk_multiplier_floor", val.get("execution_min_risk_multiplier", 0.20)),
            0.20,
        )
        risk_drift_max = self._safe_float(val.get("ops_risk_multiplier_drift_max", 0.30), 0.30)
        source_floor = self._safe_float(
            val.get("ops_source_confidence_floor", val.get("source_confidence_min", 0.75)),
            0.75,
        )
        mode_health_fail_days_max = max(0, int(val.get("ops_mode_health_fail_days_max", 2)))

        rows = self._load_mode_feedback_series(as_of=as_of, window_days=window_days)
        samples = len(rows)
        modes = [str(x.get("runtime_mode", "")).strip() for x in rows if str(x.get("runtime_mode", "")).strip()]
        risk_values = [self._safe_float(x.get("risk_multiplier", 1.0), 1.0) for x in rows]
        source_values = [self._safe_float(x.get("source_confidence_score", 1.0), 1.0) for x in rows]
        mode_health_fail_days = sum(1 for x in rows if not bool(x.get("mode_health_passed", True)))

        switch_count = 0
        if len(modes) >= 2:
            switch_count = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])
        switch_rate = float(switch_count / max(1, len(modes) - 1)) if len(modes) >= 2 else 0.0

        risk_min = min(risk_values) if risk_values else 1.0
        risk_avg = (sum(risk_values) / len(risk_values)) if risk_values else 1.0
        source_min = min(source_values) if source_values else 1.0
        source_avg = (sum(source_values) / len(source_values)) if source_values else 1.0

        risk_drift = 0.0
        if len(risk_values) >= 6:
            risk_drift = (sum(risk_values[-3:]) / 3.0) - (sum(risk_values[-6:-3]) / 3.0)
        elif len(risk_values) >= 4:
            risk_drift = (sum(risk_values[-2:]) / 2.0) - (sum(risk_values[-4:-2]) / 2.0)

        active = samples >= min_samples
        checks = {
            "switch_rate_ok": True,
            "risk_multiplier_floor_ok": True,
            "risk_multiplier_drift_ok": True,
            "source_confidence_floor_ok": True,
            "mode_health_fail_days_ok": True,
        }
        alerts: list[str] = []
        if active:
            checks["switch_rate_ok"] = bool(switch_rate <= switch_rate_max)
            checks["risk_multiplier_floor_ok"] = bool(risk_min >= risk_floor)
            checks["risk_multiplier_drift_ok"] = bool(abs(risk_drift) <= risk_drift_max)
            checks["source_confidence_floor_ok"] = bool(source_min >= source_floor)
            checks["mode_health_fail_days_ok"] = bool(mode_health_fail_days <= mode_health_fail_days_max)
            if not checks["switch_rate_ok"]:
                alerts.append("mode_switch_rate_high")
            if not checks["risk_multiplier_floor_ok"]:
                alerts.append("risk_multiplier_too_low")
            if not checks["risk_multiplier_drift_ok"]:
                alerts.append("risk_multiplier_drift_high")
            if not checks["source_confidence_floor_ok"]:
                alerts.append("source_confidence_too_low")
            if not checks["mode_health_fail_days_ok"]:
                alerts.append("mode_health_fail_days_high")
        else:
            alerts.append("insufficient_mode_feedback_samples")

        return {
            "active": active,
            "window_days": window_days,
            "samples": samples,
            "min_samples": min_samples,
            "metrics": {
                "switch_count": switch_count,
                "switch_rate": switch_rate,
                "risk_multiplier_min": risk_min,
                "risk_multiplier_avg": risk_avg,
                "risk_multiplier_drift": risk_drift,
                "source_confidence_min": source_min,
                "source_confidence_avg": source_avg,
                "mode_health_fail_days": mode_health_fail_days,
            },
            "thresholds": {
                "mode_switch_max_rate": switch_rate_max,
                "ops_risk_multiplier_floor": risk_floor,
                "ops_risk_multiplier_drift_max": risk_drift_max,
                "ops_source_confidence_floor": source_floor,
                "ops_mode_health_fail_days_max": mode_health_fail_days_max,
            },
            "checks": checks,
            "alerts": alerts,
            "series": rows[-10:],
        }

    def ops_report(self, as_of: date, window_days: int = 7) -> dict[str, Any]:
        d = as_of.isoformat()
        wd = max(1, int(window_days))

        scheduler_state = self.load_json_safely(self.output_dir / "logs" / "scheduler_state.json")
        latest_tests = self._latest_test_result()
        gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)
        state_stability = self._state_stability_metrics(as_of=as_of)
        state_checks = state_stability.get("checks", {}) if isinstance(state_stability.get("checks", {}), dict) else {}
        state_all_ok = all(bool(v) for v in state_checks.values())
        slot_anomaly = gate.get("slot_anomaly", {}) if isinstance(gate.get("slot_anomaly", {}), dict) else {}
        slot_active = bool(slot_anomaly.get("active", False))
        slot_checks = slot_anomaly.get("checks", {}) if isinstance(slot_anomaly.get("checks", {}), dict) else {}
        slot_all_ok = all(bool(v) for v in slot_checks.values()) if slot_active else True
        mode_drift = gate.get("mode_drift", {}) if isinstance(gate.get("mode_drift", {}), dict) else {}
        drift_active = bool(mode_drift.get("active", False))
        drift_checks = mode_drift.get("checks", {}) if isinstance(mode_drift.get("checks", {}), dict) else {}
        drift_all_ok = all(bool(v) for v in drift_checks.values()) if drift_active else True
        reconcile_drift = gate.get("reconcile_drift", {}) if isinstance(gate.get("reconcile_drift", {}), dict) else {}
        reconcile_active = bool(reconcile_drift.get("active", False))
        reconcile_checks = (
            reconcile_drift.get("checks", {}) if isinstance(reconcile_drift.get("checks", {}), dict) else {}
        )
        reconcile_all_ok = all(bool(v) for v in reconcile_checks.values()) if reconcile_active else True
        rollback_rec = (
            gate.get("rollback_recommendation", {})
            if isinstance(gate.get("rollback_recommendation", {}), dict)
            else {}
        )
        rollback_level = str(rollback_rec.get("level", "none")).strip().lower() or "none"
        rollback_active = bool(rollback_rec.get("active", False))
        rollback_anchor_ready = bool(rollback_rec.get("anchor_ready", True))

        history = []
        healthy_days = 0
        for i in range(wd):
            day = as_of - timedelta(days=i)
            require_review = i == 0
            h = self.health_check(day, require_review)
            ok = h["status"] == "healthy"
            healthy_days += 1 if ok else 0
            history.append(
                {
                    "date": day.isoformat(),
                    "healthy": ok,
                    "missing": h.get("missing", []),
                }
            )

        history.reverse()
        health_ratio = healthy_days / wd
        status = "green"
        if (
            not gate["passed"]
            or health_ratio < 0.8
            or (bool(state_stability.get("active", False)) and not state_all_ok)
            or (slot_active and not slot_all_ok)
            or (drift_active and not drift_all_ok)
            or (reconcile_active and not reconcile_all_ok)
            or rollback_level == "hard"
            or (rollback_active and not rollback_anchor_ready)
        ):
            status = "red"
        elif (
            health_ratio < 1.0
            or (not bool(state_stability.get("active", False)))
            or (not slot_active)
            or (not drift_active)
            or (not reconcile_active)
            or rollback_level == "soft"
        ):
            status = "yellow"

        summary = {
            "date": d,
            "status": status,
            "window_days": wd,
            "health_ratio": health_ratio,
            "gate_passed": gate["passed"],
            "latest_tests": latest_tests,
            "scheduler": {
                "date": scheduler_state.get("date"),
                "executed_slots": scheduler_state.get("executed", []),
                "history_count": len(scheduler_state.get("history", [])),
            },
            "state_stability": state_stability,
            "slot_anomaly": slot_anomaly,
            "mode_drift": mode_drift,
            "reconcile_drift": reconcile_drift,
            "rollback_recommendation": rollback_rec,
            "history": history,
        }

        report_json = self.output_dir / "review" / f"{d}_ops_report.json"
        write_json(report_json, summary)

        lines: list[str] = []
        lines.append(f"# 运维状态报告 | {d}")
        lines.append("")
        lines.append(f"- 状态: `{status}`")
        lines.append(f"- 发布门槛通过: `{gate['passed']}`")
        lines.append(f"- 健康天数占比({wd}日): `{health_ratio:.2%}`")
        lines.append(f"- 最近测试: `{latest_tests.get('returncode', 'N/A')}`")
        lines.append(f"- 调度已执行槽位: `{', '.join(summary['scheduler']['executed_slots']) if summary['scheduler']['executed_slots'] else 'NONE'}`")
        lines.append("")
        lines.append("## 状态稳定性")
        lines.append(f"- active: `{state_stability.get('active', False)}`")
        lines.append(f"- samples: `{state_stability.get('samples', 0)}` / min=`{state_stability.get('min_samples', 0)}`")
        metrics = state_stability.get("metrics", {}) if isinstance(state_stability.get("metrics", {}), dict) else {}
        lines.append(
            "- switch_rate: "
            + f"`{self._safe_float(metrics.get('switch_rate', 0.0), 0.0):.2%}` "
            + f"(count={int(metrics.get('switch_count', 0))})"
        )
        lines.append(
            "- risk_multiplier(min/avg/drift): "
            + f"`{self._safe_float(metrics.get('risk_multiplier_min', 1.0), 1.0):.3f}` / "
            + f"`{self._safe_float(metrics.get('risk_multiplier_avg', 1.0), 1.0):.3f}` / "
            + f"`{self._safe_float(metrics.get('risk_multiplier_drift', 0.0), 0.0):+.3f}`"
        )
        lines.append(
            "- source_confidence(min/avg): "
            + f"`{self._safe_float(metrics.get('source_confidence_min', 1.0), 1.0):.2%}` / "
            + f"`{self._safe_float(metrics.get('source_confidence_avg', 1.0), 1.0):.2%}`"
        )
        lines.append(f"- mode_health_fail_days: `{int(metrics.get('mode_health_fail_days', 0))}`")
        lines.append(f"- alerts: `{', '.join(state_stability.get('alerts', [])) if state_stability.get('alerts') else 'NONE'}`")
        for k, v in state_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 分时异常")
        lines.append(f"- active: `{slot_anomaly.get('active', False)}`")
        lines.append(
            f"- samples: `{int(slot_anomaly.get('samples', 0))}` / min=`{int(slot_anomaly.get('min_samples', 0))}`"
        )
        slot_metrics = slot_anomaly.get("metrics", {}) if isinstance(slot_anomaly.get("metrics", {}), dict) else {}
        lines.append(
            "- slots(expected/observed/missing): "
            + f"`{int(slot_metrics.get('expected_slots', 0))}` / "
            + f"`{int(slot_metrics.get('observed_slots', 0))}` / "
            + f"`{int(slot_metrics.get('missing_slots', 0))}`"
        )
        lines.append(
            "- anomaly_ratio(pre/intra/eod): "
            + f"`{self._safe_float(slot_metrics.get('premarket_anomaly_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(slot_metrics.get('intraday_anomaly_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(slot_metrics.get('eod_anomaly_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            f"- missing_ratio: `{self._safe_float(slot_metrics.get('missing_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(f"- alerts: `{', '.join(slot_anomaly.get('alerts', [])) if slot_anomaly.get('alerts') else 'NONE'}`")
        for k, v in slot_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 模式漂移")
        lines.append(f"- active: `{mode_drift.get('active', False)}`")
        lines.append(
            f"- scope/compared: `{int((mode_drift.get('summary', {}) or {}).get('scope_modes', 0))}` / "
            + f"`{int((mode_drift.get('summary', {}) or {}).get('compared_modes', 0))}`"
        )
        lines.append(
            f"- runtime_mode: `{str(mode_drift.get('runtime_mode', '') or 'N/A')}` | "
            + f"focus_runtime_only=`{bool(mode_drift.get('focus_runtime_mode_only', True))}`"
        )
        lines.append(
            f"- min_live_trades: `{int(mode_drift.get('min_live_trades', 0))}` | "
            + f"window_days=`{int(mode_drift.get('window_days', 0))}`"
        )
        lines.append(f"- alerts: `{', '.join(mode_drift.get('alerts', [])) if mode_drift.get('alerts') else 'NONE'}`")
        for k, v in drift_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 对账漂移")
        lines.append(f"- active: `{reconcile_drift.get('active', False)}`")
        lines.append(
            f"- samples: `{int(reconcile_drift.get('samples', 0))}` / min=`{int(reconcile_drift.get('min_samples', 0))}`"
        )
        reconcile_metrics = (
            reconcile_drift.get("metrics", {}) if isinstance(reconcile_drift.get("metrics", {}), dict) else {}
        )
        lines.append(
            "- breach_ratio(plan/closed/open): "
            + f"`{self._safe_float(reconcile_metrics.get('plan_gap_breach_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('closed_count_gap_breach_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('open_gap_breach_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- missing_ratio: "
            + f"`{self._safe_float(reconcile_metrics.get('missing_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            f"- alerts: `{', '.join(reconcile_drift.get('alerts', [])) if reconcile_drift.get('alerts') else 'NONE'}`"
        )
        for k, v in reconcile_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 回滚建议")
        lines.append(f"- active: `{rollback_active}`")
        lines.append(f"- level: `{rollback_level}`")
        lines.append(f"- score: `{int(self._safe_float(rollback_rec.get('score', 0), 0))}`")
        lines.append(f"- action: `{str(rollback_rec.get('action', 'no_rollback'))}`")
        lines.append(f"- anchor_ready: `{rollback_anchor_ready}`")
        lines.append(f"- target_anchor: `{str(rollback_rec.get('target_anchor', '') or 'N/A')}`")
        reason_codes = rollback_rec.get("reason_codes", []) if isinstance(rollback_rec.get("reason_codes", []), list) else []
        lines.append(f"- reason_codes: `{', '.join(reason_codes) if reason_codes else 'NONE'}`")
        lines.append("")
        lines.append("## 最近健康历史")
        for item in history:
            lines.append(f"- {item['date']}: {'OK' if item['healthy'] else 'DEGRADED'} | missing={item['missing']}")
        lines.append("")
        lines.append("## 门槛检查")
        for k, v in gate["checks"].items():
            lines.append(f"- `{k}`: `{v}`")

        report_md = self.output_dir / "review" / f"{d}_ops_report.md"
        write_markdown(report_md, "\n".join(lines) + "\n")
        summary["paths"] = {"json": str(report_json), "md": str(report_md)}
        return summary

    def _build_defect_plan(
        self,
        as_of: date,
        round_no: int,
        review: ReviewDelta,
        tests: dict[str, Any],
        gate: dict[str, Any],
        state_stability: dict[str, Any] | None = None,
        slot_anomaly: dict[str, Any] | None = None,
        mode_drift: dict[str, Any] | None = None,
        reconcile_drift: dict[str, Any] | None = None,
        rollback_recommendation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checks = gate.get("checks", {})
        metrics = gate.get("metrics", {})
        failed_tests = self._extract_failed_tests(tests)

        defects: list[dict[str, Any]] = []
        if not bool(checks.get("data_completeness_ok", True)):
            defects.append(
                {
                    "category": "data",
                    "code": "DATA_COMPLETENESS",
                    "message": "关键字段完整率未达门槛",
                    "action": "优先修复抓取字段映射并重跑 run-premarket/run-eod。",
                }
            )
        if not bool(checks.get("unresolved_conflict_ok", True)):
            defects.append(
                {
                    "category": "data",
                    "code": "DATA_CONFLICT",
                    "message": "跨源冲突未决比例超限",
                    "action": "定位冲突字段来源并补充冲突仲裁规则后重跑。",
                }
            )
        if not bool(review.pass_gate):
            defects.append(
                {
                    "category": "model",
                    "code": "REVIEW_GATE",
                    "message": "复盘门槛未通过",
                    "action": "检查参数更新幅度和因子贡献，必要时回滚到上一个稳定参数。",
                }
            )
        if not bool(checks.get("mode_health_ok", True)):
            defects.append(
                {
                    "category": "model",
                    "code": "MODE_HEALTH",
                    "message": "当前运行模式历史表现退化，触发模式健康门禁",
                    "action": "收敛信号阈值并降低交易频率，补跑对应模式窗口回测后再放开。",
                }
            )
        if not bool(checks.get("slot_anomaly_ok", True)):
            defects.append(
                {
                    "category": "execution",
                    "code": "SLOT_ANOMALY",
                    "message": "固定时段执行链路出现异常聚集",
                    "action": "优先修复缺失/异常槽位数据后再恢复正常评审节奏。",
                }
            )
        if not bool(checks.get("mode_drift_ok", True)):
            defects.append(
                {
                    "category": "model",
                    "code": "MODE_DRIFT",
                    "message": "实盘与回测模式表现出现漂移",
                    "action": "先核验 live/backtest 口径一致性，再收紧该模式参数并复跑近窗回测。",
                }
            )
        if not bool(checks.get("reconcile_drift_ok", True)):
            defects.append(
                {
                    "category": "execution",
                    "code": "RECONCILE_DRIFT",
                    "message": "执行链路对账存在漂移",
                    "action": "优先核对 manifest/daily/sqlite 三方一致性并修复缺失写盘。",
                }
            )
        if not bool(checks.get("rollback_anchor_ready", True)):
            defects.append(
                {
                    "category": "risk",
                    "code": "ROLLBACK_ANCHOR_MISSING",
                    "message": "触发回滚条件但缺少可用回滚锚点",
                    "action": "立即补齐 params_live_backup 与 review rollback_anchor 审计链路。",
                }
            )
        if not bool(checks.get("positive_window_ratio_ok", True)):
            defects.append(
                {
                    "category": "model",
                    "code": "POSITIVE_WINDOW_RATIO",
                    "message": "样本外正收益窗口占比不达标",
                    "action": "收缩信号阈值与仓位，先做局部回测再跑全量 walk-forward。",
                }
            )
        if not bool(checks.get("max_drawdown_ok", True)):
            defects.append(
                {
                    "category": "risk",
                    "code": "MAX_DRAWDOWN",
                    "message": "最大回撤超过 18% 上限",
                    "action": "下调风险预算并加严止损/保护模式触发阈值。",
                }
            )
        if not bool(checks.get("risk_violations_ok", True)):
            defects.append(
                {
                    "category": "risk",
                    "code": "RISK_VIOLATION",
                    "message": "风险约束出现违规",
                    "action": "校验单笔/总暴露约束计算与执行器拦截逻辑。",
                }
            )
        if int(tests.get("returncode", 0)) != 0:
            defects.append(
                {
                    "category": "execution",
                    "code": "TEST_FAILURE",
                    "message": f"自动测试失败，失败用例数={len(failed_tests)}",
                    "action": "先跑失败用例局部回归，再跑 lie test-all 进行全量验收。",
                    "failed_tests": failed_tests[:20],
                }
            )
        if not bool(checks.get("stable_replay_ok", True)):
            defects.append(
                {
                    "category": "execution",
                    "code": "STABLE_REPLAY",
                    "message": "连续回放稳定性未达标",
                    "action": "定位失败日期的产物缺失或波动异常，修复后重跑 stable-replay。",
                }
            )
        if not bool(checks.get("health_ok", True)):
            defects.append(
                {
                    "category": "report",
                    "code": "HEALTH_DEGRADED",
                    "message": "输出工件不完整",
                    "action": "补跑缺失槽位并校验 output/daily 与 output/review 文件契约。",
                }
            )
        state_payload = state_stability if isinstance(state_stability, dict) else {}
        state_active = bool(state_payload.get("active", False))
        state_checks = state_payload.get("checks", {}) if isinstance(state_payload.get("checks", {}), dict) else {}
        state_metrics = state_payload.get("metrics", {}) if isinstance(state_payload.get("metrics", {}), dict) else {}
        state_thresholds = state_payload.get("thresholds", {}) if isinstance(state_payload.get("thresholds", {}), dict) else {}
        if state_active:
            if not bool(state_checks.get("switch_rate_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "STATE_MODE_SWITCH",
                        "message": (
                            "模式切换率过高："
                            + f"{self._safe_float(state_metrics.get('switch_rate', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(state_thresholds.get('mode_switch_max_rate', 1.0), 1.0):.2%}"
                        ),
                        "action": "降低状态切换敏感度（提高确认窗口/滞后阈值），并复跑近窗模式稳定性。",
                    }
                )
            if not bool(state_checks.get("risk_multiplier_floor_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "STATE_RISK_MULT_FLOOR",
                        "message": (
                            "执行风险乘子触及过低区间："
                            + f"min={self._safe_float(state_metrics.get('risk_multiplier_min', 1.0), 1.0):.3f} < "
                            + f"floor={self._safe_float(state_thresholds.get('ops_risk_multiplier_floor', 0.0), 0.0):.3f}"
                        ),
                        "action": "优先修复源置信度与模式健康退化根因，避免执行层持续被动降杠杆。",
                    }
                )
            if not bool(state_checks.get("risk_multiplier_drift_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "STATE_RISK_MULT_DRIFT",
                        "message": (
                            "执行风险乘子漂移超限："
                            + f"drift={self._safe_float(state_metrics.get('risk_multiplier_drift', 0.0), 0.0):+.3f}, "
                            + f"max={self._safe_float(state_thresholds.get('ops_risk_multiplier_drift_max', 0.0), 0.0):.3f}"
                        ),
                        "action": "检查近期状态切换与数据质量是否同步恶化，必要时进入保护模式并收敛交易频次。",
                    }
                )
            if not bool(state_checks.get("source_confidence_floor_ok", True)):
                defects.append(
                    {
                        "category": "data",
                        "code": "STATE_SOURCE_CONFIDENCE",
                        "message": (
                            "源置信度下穿门槛："
                            + f"min={self._safe_float(state_metrics.get('source_confidence_min', 1.0), 1.0):.2%} < "
                            + f"floor={self._safe_float(state_thresholds.get('ops_source_confidence_floor', 0.0), 0.0):.2%}"
                        ),
                        "action": "优先处理低置信源与跨源冲突仲裁，必要时临时降级数据源组合。",
                    }
                )
            if not bool(state_checks.get("mode_health_fail_days_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "STATE_MODE_HEALTH_DAYS",
                        "message": (
                            "模式健康失败天数超限："
                            + f"{int(self._safe_float(state_metrics.get('mode_health_fail_days', 0), 0))} > "
                            + f"{int(self._safe_float(state_thresholds.get('ops_mode_health_fail_days_max', 0), 0))}"
                        ),
                        "action": "收敛当前模式参数并补跑对应资产/模式窗口回测，确认恢复后再放开仓位。",
                    }
                )

        slot_payload = slot_anomaly if isinstance(slot_anomaly, dict) else {}
        slot_active = bool(slot_payload.get("active", False))
        slot_checks = slot_payload.get("checks", {}) if isinstance(slot_payload.get("checks", {}), dict) else {}
        slot_metrics = slot_payload.get("metrics", {}) if isinstance(slot_payload.get("metrics", {}), dict) else {}
        slot_thresholds = slot_payload.get("thresholds", {}) if isinstance(slot_payload.get("thresholds", {}), dict) else {}
        if slot_active:
            if not bool(slot_checks.get("missing_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_MISSING_RATIO",
                        "message": (
                            "分时槽位缺失率超限："
                            + f"{self._safe_float(slot_metrics.get('missing_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_missing_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "补齐缺失槽位日志/manifest，并核对调度器与数据写盘链路。",
                    }
                )
            if not bool(slot_checks.get("premarket_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_PREMARKET_ANOMALY",
                        "message": (
                            "盘前异常率超限："
                            + f"{self._safe_float(slot_metrics.get('premarket_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_premarket_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "优先修复盘前数据质量/保护模式触发路径，避免开盘前风控误触发。",
                    }
                )
            if not bool(slot_checks.get("intraday_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_INTRADAY_ANOMALY",
                        "message": (
                            "盘中异常率超限："
                            + f"{self._safe_float(slot_metrics.get('intraday_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_intraday_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "复核盘中校准任务的输入完整性与保护模式阈值，必要时降级盘中校准频率。",
                    }
                )
            if not bool(slot_checks.get("eod_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_EOD_ANOMALY",
                        "message": (
                            "收盘主计算异常率超限："
                            + f"{self._safe_float(slot_metrics.get('eod_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_eod_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "检查 EOD 主流程质量门禁与风险乘子链路，确保收盘工件稳定产出。",
                    }
                )

        drift_payload = mode_drift if isinstance(mode_drift, dict) else {}
        drift_active = bool(drift_payload.get("active", False))
        drift_checks = drift_payload.get("checks", {}) if isinstance(drift_payload.get("checks", {}), dict) else {}
        drift_modes = drift_payload.get("modes", {}) if isinstance(drift_payload.get("modes", {}), dict) else {}
        if drift_active:
            if not bool(drift_checks.get("win_rate_gap_ok", True)):
                offenders: list[str] = []
                for mode, row in drift_modes.items():
                    if not isinstance(row, dict):
                        continue
                    row_checks = row.get("checks", {}) if isinstance(row.get("checks", {}), dict) else {}
                    if bool(row.get("active", False)) and not bool(row_checks.get("win_rate_gap_ok", True)):
                        offenders.append(str(mode))
                defects.append(
                    {
                        "category": "model",
                        "code": "MODE_DRIFT_WIN_RATE",
                        "message": "模式胜率偏离回测基线",
                        "action": "优先收紧对应模式信号阈值，并校验成交成本/滑点建模是否低估。",
                        "modes": offenders[:10],
                    }
                )
            if not bool(drift_checks.get("profit_factor_gap_ok", True)):
                offenders = []
                for mode, row in drift_modes.items():
                    if not isinstance(row, dict):
                        continue
                    row_checks = row.get("checks", {}) if isinstance(row.get("checks", {}), dict) else {}
                    if bool(row.get("active", False)) and not bool(row_checks.get("profit_factor_gap_ok", True)):
                        offenders.append(str(mode))
                defects.append(
                    {
                        "category": "model",
                        "code": "MODE_DRIFT_PROFIT_FACTOR",
                        "message": "模式盈亏比偏离回测基线",
                        "action": "复核止损/止盈执行偏差与行情跳空影响，必要时降低模式风险乘子。",
                        "modes": offenders[:10],
                    }
                )

        reconcile_payload = reconcile_drift if isinstance(reconcile_drift, dict) else {}
        reconcile_active = bool(reconcile_payload.get("active", False))
        reconcile_checks = (
            reconcile_payload.get("checks", {}) if isinstance(reconcile_payload.get("checks", {}), dict) else {}
        )
        reconcile_metrics = (
            reconcile_payload.get("metrics", {}) if isinstance(reconcile_payload.get("metrics", {}), dict) else {}
        )
        reconcile_thresholds = (
            reconcile_payload.get("thresholds", {})
            if isinstance(reconcile_payload.get("thresholds", {}), dict)
            else {}
        )
        if reconcile_active:
            if not bool(reconcile_checks.get("missing_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_MISSING_RATIO",
                        "message": (
                            "对账缺失率超限："
                            + f"{self._safe_float(reconcile_metrics.get('missing_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(reconcile_thresholds.get('ops_reconcile_missing_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "优先补齐 daily/manifest/sqlite 缺失数据并重跑当日 EOD。",
                    }
                )
            if not bool(reconcile_checks.get("plan_count_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_PLAN_COUNT",
                        "message": "计划单数量对账漂移超限",
                        "action": "核查 latest_positions 与 daily positions 写入口径，修复后重跑对账。",
                    }
                )
            if not bool(reconcile_checks.get("closed_count_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_CLOSED_COUNT",
                        "message": "平仓笔数对账漂移超限",
                        "action": "核查 executed_plans 回写条件与 manifest 统计口径。",
                    }
                )
            if not bool(reconcile_checks.get("closed_pnl_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_CLOSED_PNL",
                        "message": "平仓盈亏对账漂移超限",
                        "action": "核查平仓收益写盘精度与汇总逻辑，修复后回放当日数据。",
                    }
                )
            if not bool(reconcile_checks.get("open_count_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_OPEN_COUNT",
                        "message": "当前持仓数量与 open_state 对账漂移超限",
                        "action": "重建 paper_positions_open 状态并校验 EOD open_positions 指标。",
                    }
                )

        rollback_payload = rollback_recommendation if isinstance(rollback_recommendation, dict) else {}
        rollback_level = str(rollback_payload.get("level", "none")).strip().lower() or "none"
        rollback_active = bool(rollback_payload.get("active", False))
        rollback_anchor_ready = bool(rollback_payload.get("anchor_ready", True))
        if rollback_level in {"soft", "hard"}:
            defects.append(
                {
                    "category": "risk",
                    "code": "ROLLBACK_RECOMMENDED",
                    "message": f"系统建议参数回滚（level={rollback_level}）",
                    "action": "按回滚建议执行参数回退并冻结自适应更新，完成局部回放后再恢复。",
                    "target_anchor": str(rollback_payload.get("target_anchor", "")),
                    "reason_codes": list(rollback_payload.get("reason_codes", []))[:10],
                }
            )
            if rollback_level == "hard":
                defects.append(
                    {
                        "category": "risk",
                        "code": "ROLLBACK_HARD",
                        "message": "触发硬回滚条件",
                        "action": "立即回滚并人工确认后再开放自动参数更新。",
                    }
                )
        if rollback_active and not rollback_anchor_ready:
            defects.append(
                {
                    "category": "risk",
                    "code": "ROLLBACK_ANCHOR_UNAVAILABLE",
                    "message": "回滚建议已触发但缺少可用锚点",
                    "action": "优先修复回滚锚点链路并补写备份文件。",
                }
            )

        if not defects:
            defects.append(
                {
                    "category": "unknown",
                    "code": "UNCLASSIFIED",
                    "message": "存在失败但未命中分类规则",
                    "action": "人工审查 gate_report 与 tests 日志后补充规则。",
                }
            )

        default_actions = [
            "仅针对缺陷模块执行局部测试/回放",
            "局部通过后执行 lie test-all",
            "门槛通过后重跑 gate-report 与 ops-report",
        ]
        next_actions = list(default_actions)
        if any(str(x.get("code", "")).startswith("STATE_") for x in defects):
            next_actions = [
                f"优先修复 state_stability 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "状态稳定后再进行参数修正与回测收敛",
            ] + default_actions
        if any(str(x.get("code", "")).startswith("SLOT_") for x in defects):
            next_actions = [
                f"优先修复 slot_anomaly 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "补齐缺失槽位后再进行策略回测与参数更新。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("MODE_DRIFT_") for x in defects):
            next_actions = [
                f"优先修复 mode_drift 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "确认 live/backtest 口径一致后再放开模式自适应更新",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("RECONCILE_") for x in defects):
            next_actions = [
                f"优先修复 reconcile_drift 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "对账一致后再执行参数更新与全量回测。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("ROLLBACK_") for x in defects):
            next_actions = [
                f"优先执行回滚建议并重跑 lie gate-report --date {as_of.isoformat()}",
                "回滚后先局部回放，再执行 lie test-all 与 review-loop。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions

        plan = {
            "date": as_of.isoformat(),
            "round": round_no,
            "defect_count": len(defects),
            "defects": defects,
            "metrics": metrics,
            "checks": checks,
            "state_stability": state_payload,
            "slot_anomaly": slot_payload,
            "mode_drift": drift_payload,
            "reconcile_drift": reconcile_payload,
            "rollback_recommendation": rollback_payload,
            "next_actions": next_actions,
        }

        review_dir = self.output_dir / "review"
        json_path = review_dir / f"{as_of.isoformat()}_defect_plan_round{round_no}.json"
        md_path = review_dir / f"{as_of.isoformat()}_defect_plan_round{round_no}.md"
        write_json(json_path, plan)

        lines: list[str] = []
        lines.append(f"# 缺陷修正计划 | {as_of.isoformat()} | Round {round_no}")
        lines.append("")
        lines.append(f"- 缺陷数量: `{len(defects)}`")
        lines.append(f"- 最大回撤: `{metrics.get('max_drawdown', 'N/A')}`")
        lines.append(f"- 正收益窗口占比: `{metrics.get('positive_window_ratio', 'N/A')}`")
        if state_payload:
            lines.append(
                "- 状态稳定性: "
                + f"active={state_payload.get('active', False)}, alerts={state_payload.get('alerts', [])}"
            )
        if slot_payload:
            lines.append(
                "- 分时异常: "
                + f"active={slot_payload.get('active', False)}, alerts={slot_payload.get('alerts', [])}"
            )
        if drift_payload:
            lines.append(
                "- 模式漂移: "
                + f"active={drift_payload.get('active', False)}, alerts={drift_payload.get('alerts', [])}"
            )
        if reconcile_payload:
            lines.append(
                "- 对账漂移: "
                + f"active={reconcile_payload.get('active', False)}, alerts={reconcile_payload.get('alerts', [])}"
            )
        if rollback_payload:
            lines.append(
                "- 回滚建议: "
                + f"level={rollback_payload.get('level', 'none')}, target={rollback_payload.get('target_anchor', '')}"
            )
        lines.append("")
        lines.append("## 缺陷分类")
        for item in defects:
            lines.append(f"- [{item['category']}] `{item['code']}`: {item['message']} | action={item['action']}")
            if item.get("failed_tests"):
                lines.append(f"- 失败用例: {', '.join(item['failed_tests'])}")
        lines.append("")
        lines.append("## 修正顺序")
        for idx, action in enumerate(plan["next_actions"], start=1):
            lines.append(f"{idx}. {action}")

        write_markdown(md_path, "\n".join(lines) + "\n")
        return {"json": str(json_path), "md": str(md_path)}

    def review_until_pass(self, as_of: date, max_rounds: int = 3) -> dict[str, Any]:
        alert_path = self.output_dir / "logs" / f"review_loop_alert_{as_of.isoformat()}.json"
        if int(max_rounds) <= 0:
            return {
                "passed": False,
                "skipped": True,
                "reason": "max_rounds must be >= 1",
                "rounds": [],
            }

        rounds = []
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        fast_enabled = bool(val.get("review_loop_fast_test_enabled", True))
        fast_ratio = float(val.get("review_loop_fast_ratio", 0.10))
        fast_shard_index = int(val.get("review_loop_fast_shard_index", 0))
        fast_shard_total = int(val.get("review_loop_fast_shard_total", 1))
        fast_seed = str(val.get("review_loop_fast_seed", "lie-fast-v1"))
        fast_then_full = bool(val.get("review_loop_fast_then_full", True))

        for i in range(int(max_rounds)):
            review = self.run_review(as_of)
            run_fast = bool(i == 0 and fast_enabled)
            tests = self._run_tests(
                fast=run_fast,
                fast_ratio=fast_ratio,
                fast_shard_index=fast_shard_index,
                fast_shard_total=fast_shard_total,
                fast_seed=fast_seed,
            )
            fast_tests = tests if run_fast else {}
            full_tests = tests if (not run_fast) else {}
            if run_fast and tests.get("returncode", 1) == 0 and fast_then_full:
                full_tests = self._run_tests(
                    fast=False,
                    fast_ratio=fast_ratio,
                    fast_shard_index=fast_shard_index,
                    fast_shard_total=fast_shard_total,
                    fast_seed=fast_seed,
                )
                tests = full_tests

            gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)
            state_stability = self._state_stability_metrics(as_of=as_of)
            state_checks = state_stability.get("checks", {}) if isinstance(state_stability.get("checks", {}), dict) else {}
            state_active = bool(state_stability.get("active", False))
            state_ok = all(bool(v) for v in state_checks.values()) if state_active else True
            slot_anomaly = gate.get("slot_anomaly", {}) if isinstance(gate.get("slot_anomaly", {}), dict) else {}
            slot_checks = slot_anomaly.get("checks", {}) if isinstance(slot_anomaly.get("checks", {}), dict) else {}
            slot_active = bool(slot_anomaly.get("active", False))
            slot_ok = all(bool(v) for v in slot_checks.values()) if slot_active else True
            mode_drift = gate.get("mode_drift", {}) if isinstance(gate.get("mode_drift", {}), dict) else {}
            drift_checks = mode_drift.get("checks", {}) if isinstance(mode_drift.get("checks", {}), dict) else {}
            drift_active = bool(mode_drift.get("active", False))
            drift_ok = all(bool(v) for v in drift_checks.values()) if drift_active else True
            reconcile_drift = gate.get("reconcile_drift", {}) if isinstance(gate.get("reconcile_drift", {}), dict) else {}
            reconcile_checks = (
                reconcile_drift.get("checks", {}) if isinstance(reconcile_drift.get("checks", {}), dict) else {}
            )
            reconcile_active = bool(reconcile_drift.get("active", False))
            reconcile_ok = all(bool(v) for v in reconcile_checks.values()) if reconcile_active else True
            rollback_rec = (
                gate.get("rollback_recommendation", {})
                if isinstance(gate.get("rollback_recommendation", {}), dict)
                else {}
            )
            ok = bool(gate["passed"] and tests["returncode"] == 0 and review.pass_gate)
            rounds.append(
                {
                    "round": i + 1,
                    "tests_mode": "fast+full" if (run_fast and fast_then_full and full_tests) else ("fast" if run_fast else "full"),
                    "pass_gate": review.pass_gate,
                    "tests_ok": tests["returncode"] == 0,
                    "stable_replay_ok": gate["checks"]["stable_replay_ok"],
                    "stable_replay_days": gate["stable_replay"]["replay_days"],
                    "gate_passed": gate["passed"],
                    "state_stability_active": state_active,
                    "state_stability_ok": state_ok,
                    "state_alerts": list(state_stability.get("alerts", [])),
                    "slot_anomaly_active": slot_active,
                    "slot_anomaly_ok": slot_ok,
                    "slot_alerts": list(slot_anomaly.get("alerts", [])),
                    "mode_drift_active": drift_active,
                    "mode_drift_ok": drift_ok,
                    "mode_drift_alerts": list(mode_drift.get("alerts", [])),
                    "reconcile_drift_active": reconcile_active,
                    "reconcile_drift_ok": reconcile_ok,
                    "reconcile_drift_alerts": list(reconcile_drift.get("alerts", [])),
                    "rollback_level": str(rollback_rec.get("level", "none")),
                    "rollback_action": str(rollback_rec.get("action", "no_rollback")),
                    "rollback_anchor_ready": bool(rollback_rec.get("anchor_ready", True)),
                }
            )
            if fast_tests:
                rounds[-1]["fast_tests"] = {
                    "returncode": int(fast_tests.get("returncode", 1)),
                    "summary_line": str(fast_tests.get("summary_line", "")),
                    "tests_ran": int(fast_tests.get("tests_ran", 0)),
                    "failed_tests": list(fast_tests.get("failed_tests", []))[:20],
                }
            if full_tests:
                rounds[-1]["full_tests"] = {
                    "returncode": int(full_tests.get("returncode", 1)),
                    "summary_line": str(full_tests.get("summary_line", "")),
                    "tests_ran": int(full_tests.get("tests_ran", 0)),
                    "failed_tests": list(full_tests.get("failed_tests", []))[:20],
                }
            if ok:
                release_path = self.output_dir / "artifacts" / f"release_ready_{as_of.isoformat()}.json"
                write_json(release_path, {"date": as_of.isoformat(), "passed": True, "rounds": rounds})
                if alert_path.exists():
                    try:
                        alert_path.unlink()
                    except OSError:
                        pass
                return {"passed": True, "skipped": False, "rounds": rounds}
            plan_paths = self._build_defect_plan(
                as_of=as_of,
                round_no=i + 1,
                review=review,
                tests=tests,
                gate=gate,
                state_stability=state_stability,
                slot_anomaly=slot_anomaly,
                mode_drift=mode_drift,
                reconcile_drift=reconcile_drift,
                rollback_recommendation=rollback_rec,
            )
            rounds[-1]["defect_plan"] = plan_paths
        fail_payload = {"passed": False, "skipped": False, "rounds": rounds}
        write_json(alert_path, fail_payload)
        return fail_payload

    def run_review_cycle(self, as_of: date, max_rounds: int = 2, ops_window_days: int | None = None) -> dict[str, Any]:
        replay_days = int(self.settings.validation.get("required_stable_replay_days", 3))
        ops_days = int(ops_window_days or replay_days)
        review_loop = self.review_until_pass(as_of=as_of, max_rounds=max_rounds)
        gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)
        ops = self.ops_report(as_of=as_of, window_days=ops_days)
        health = self.health_check(as_of, True)
        return {
            "date": as_of.isoformat(),
            "review_loop": review_loop,
            "gate_report": gate,
            "ops_report": ops,
            "health": health,
        }
