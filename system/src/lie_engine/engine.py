from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lie_engine.backtest import BacktestConfig, run_walk_forward_backtest
from lie_engine.config import SystemSettings, assert_valid_settings, load_settings
from lie_engine.data import DataBus, build_provider_stack
from lie_engine.data.storage import append_sqlite, write_csv, write_json, write_markdown
from lie_engine.models import BacktestResult, NewsEvent, RegimeLabel, RegimeState, ReviewDelta, TradePlan
from lie_engine.orchestration import (
    ArchitectureOrchestrator,
    DependencyOrchestrator,
    GuardAssessment,
    ObservabilityOrchestrator,
    ReleaseOrchestrator,
    SchedulerOrchestrator,
    TestingOrchestrator,
    estimate_factor_contrib_120d,
)
from lie_engine.orchestration.guards import black_swan_assessment, loss_cooldown_active, major_event_window
from lie_engine.regime import compute_atr_zscore, derive_regime_consensus, infer_hmm_state, latest_multi_scale_hurst
from lie_engine.research import run_research_backtest as run_research_pipeline
from lie_engine.research import run_strategy_lab as run_strategy_lab_pipeline
from lie_engine.reporting import render_daily_briefing, render_mode_stress_matrix, render_review_report, write_run_manifest
from lie_engine.review import ReviewThresholds, bounded_bayesian_update, build_review_delta
from lie_engine.risk import RiskManager, infer_edge_from_trades
from lie_engine.signal import SignalEngineConfig, expand_universe, scan_signals


@dataclass(slots=True)
class EngineContext:
    settings: SystemSettings
    root: Path
    output_dir: Path
    sqlite_path: Path


class LieEngine:
    def __init__(self, config_path: str | Path | None = None) -> None:
        settings = load_settings(config_path)
        assert_valid_settings(settings)
        root = Path(config_path).resolve().parent if config_path else Path(__file__).resolve().parents[2]
        output_dir = root / settings.paths.get("output", "output")
        sqlite_path = root / settings.paths.get("sqlite", "output/artifacts/lie_engine.db")
        self.ctx = EngineContext(settings=settings, root=root, output_dir=output_dir, sqlite_path=sqlite_path)

        data_cfg = settings.raw.get("data", {}) if isinstance(settings.raw, dict) else {}
        provider_profile = str(data_cfg.get("provider_profile", "opensource_dual"))
        self.providers = build_provider_stack(provider_profile)
        self.data_bus = DataBus(
            providers=self.providers,
            output_dir=self.ctx.output_dir,
            sqlite_path=self.ctx.sqlite_path,
            completeness_min=float(settings.validation.get("data_completeness_min", 0.99)),
            conflict_max=float(settings.validation.get("unresolved_conflict_max", 0.005)),
            source_confidence_min=float(settings.validation.get("source_confidence_min", 0.75)),
            low_confidence_source_ratio_max=float(settings.validation.get("low_confidence_source_ratio_max", 0.40)),
        )

        self.risk_manager = RiskManager(
            max_single_risk_pct=float(settings.risk.get("max_single_risk_pct", 2.0)),
            max_total_exposure_pct=float(settings.risk.get("max_total_exposure_pct", 50.0)),
            max_symbol_pct=float(settings.risk.get("max_symbol_pct", 15.0)),
            max_theme_pct=float(settings.risk.get("max_theme_pct", 25.0)),
        )

    @property
    def settings(self) -> SystemSettings:
        return self.ctx.settings

    def _core_symbols(self) -> list[str]:
        return [x["symbol"] for x in self.settings.universe.get("core", [])]

    def _theme_map(self) -> dict[str, str]:
        out = {}
        for item in self.settings.universe.get("core", []):
            out[item["symbol"]] = item.get("theme", "unknown")
        return out

    @staticmethod
    def _load_json_safely(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _clamp_float(v: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, v)))

    @staticmethod
    def _clamp_int(v: float | int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, int(round(float(v))))))

    @staticmethod
    def _parse_iso_date(value: Any) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value))
        except Exception:
            return None

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except Exception:
            return None

    @staticmethod
    def _parse_backtest_run_id_dates(run_id: str) -> tuple[date | None, date | None]:
        m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$", str(run_id).strip())
        if not m:
            return None, None
        try:
            return date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        except Exception:
            return None, None

    def _mode_history_stats(self, *, as_of: date, lookback_days: int | None = None) -> dict[str, Any]:
        lb = int(lookback_days) if isinstance(lookback_days, int) and lookback_days > 0 else int(self.settings.validation.get("mode_stats_lookback_days", 365))
        lb = max(30, lb)
        manifest_dir = self.ctx.output_dir / "artifacts" / "manifests"
        if not manifest_dir.exists():
            return {"as_of": as_of.isoformat(), "lookback_days": lb, "modes": {}}

        buckets: dict[str, list[dict[str, float]]] = {}
        for path in sorted(manifest_dir.glob("backtest_*.json")):
            payload = self._load_json_safely(path)
            if not payload:
                continue
            run_type = str(payload.get("run_type", "")).strip()
            if run_type and run_type != "backtest":
                continue
            run_id = str(payload.get("run_id", "")).strip()
            _, end_d = self._parse_backtest_run_id_dates(run_id)
            if end_d is None:
                _, end_d = self._parse_backtest_run_id_dates(path.stem.replace("backtest_", "", 1))
            if end_d is None:
                continue
            if end_d > as_of:
                continue
            if (as_of - end_d).days > lb:
                continue

            metrics = payload.get("metrics", {})
            checks = payload.get("checks", {})
            if not isinstance(metrics, dict):
                continue
            mode = str(metrics.get("runtime_mode", "base")).strip() or "base"
            row = {
                "annual_return": float(metrics.get("annual_return", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "trades": float(metrics.get("trades", 0.0)),
                "violations": float((checks or {}).get("violations", 0.0)),
            }
            buckets.setdefault(mode, []).append(row)

        modes: dict[str, Any] = {}
        for mode, rows in buckets.items():
            if not rows:
                continue
            n = float(len(rows))
            modes[mode] = {
                "samples": int(len(rows)),
                "avg_annual_return": float(sum(r["annual_return"] for r in rows) / n),
                "avg_win_rate": float(sum(r["win_rate"] for r in rows) / n),
                "avg_profit_factor": float(sum(r["profit_factor"] for r in rows) / n),
                "avg_trades": float(sum(r["trades"] for r in rows) / n),
                "worst_drawdown": float(max(r["max_drawdown"] for r in rows)),
                "total_violations": int(sum(int(r["violations"]) for r in rows)),
            }
        return {"as_of": as_of.isoformat(), "lookback_days": lb, "modes": modes}

    def _review_backtest_start(self, as_of: date) -> date:
        default_start = date(2015, 1, 1)
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}

        lookback_raw = val.get("review_backtest_lookback_days")
        if isinstance(lookback_raw, (int, float)) and int(lookback_raw) > 0:
            lookback_days = max(30, int(lookback_raw))
            return max(default_start, as_of - timedelta(days=lookback_days))

        start_raw = val.get("review_backtest_start_date")
        parsed = self._parse_iso_date(start_raw)
        if parsed is None:
            return default_start
        if parsed > as_of:
            return default_start
        return max(default_start, parsed)

    @staticmethod
    def _default_mode_profiles() -> dict[str, dict[str, float]]:
        return {
            "ultra_short": {
                "signal_confidence_min": 56.0,
                "convexity_min": 1.6,
                "hold_days": 2.0,
                "max_daily_trades": 4.0,
            },
            "swing": {
                "signal_confidence_min": 60.0,
                "convexity_min": 2.2,
                "hold_days": 8.0,
                "max_daily_trades": 2.0,
            },
            "long": {
                "signal_confidence_min": 54.0,
                "convexity_min": 2.8,
                "hold_days": 18.0,
                "max_daily_trades": 1.0,
            },
        }

    def _mode_from_regime(self, consensus: RegimeLabel) -> str:
        default_map = {
            RegimeLabel.STRONG_TREND: "long",
            RegimeLabel.WEAK_TREND: "swing",
            RegimeLabel.RANGE: "ultra_short",
            RegimeLabel.DOWNTREND: "swing",
            RegimeLabel.UNCERTAIN: "swing",
            RegimeLabel.EXTREME_VOL: "ultra_short",
        }
        raw_runtime = self.settings.raw.get("mode_runtime", {}) if isinstance(self.settings.raw, dict) else {}
        raw_map = raw_runtime.get("regime_to_mode", {}) if isinstance(raw_runtime, dict) else {}

        mode = default_map.get(consensus, "swing")
        if isinstance(raw_map, dict):
            for key in (consensus.value, consensus.name):
                if key in raw_map and str(raw_map.get(key)).strip():
                    mode = str(raw_map.get(key)).strip()
                    break
        if mode not in {"ultra_short", "swing", "long"}:
            mode = "swing"
        return mode

    def _resolve_runtime_params(self, *, regime, live_params: dict[str, float]) -> dict[str, float | str]:
        base = {
            "signal_confidence_min": self._clamp_float(
                float(live_params.get("signal_confidence_min", self.settings.thresholds.get("signal_confidence_min", 60.0))),
                20.0,
                95.0,
            ),
            "convexity_min": self._clamp_float(
                float(live_params.get("convexity_min", self.settings.thresholds.get("convexity_min", 3.0))),
                0.5,
                5.0,
            ),
            "hold_days": float(self._clamp_int(live_params.get("hold_days", 5.0), 1, 20)),
            "max_daily_trades": float(self._clamp_int(live_params.get("max_daily_trades", 2.0), 1, 5)),
        }
        if not bool(self.settings.validation.get("use_mode_profiles", False)):
            return {"mode": "base"} | base

        mode = self._mode_from_regime(regime.consensus)
        profiles = self._resolved_mode_profiles()
        profile = profiles.get(mode, profiles["swing"])
        blend = self._clamp_float(float(self.settings.validation.get("mode_profile_blend_with_live", 0.55)), 0.0, 1.0)
        merged_conf = (1.0 - blend) * float(base["signal_confidence_min"]) + blend * float(profile["signal_confidence_min"])
        merged_conv = (1.0 - blend) * float(base["convexity_min"]) + blend * float(profile["convexity_min"])
        merged_hold = (1.0 - blend) * float(base["hold_days"]) + blend * float(profile["hold_days"])
        merged_trades = (1.0 - blend) * float(base["max_daily_trades"]) + blend * float(profile["max_daily_trades"])
        return {
            "mode": mode,
            "signal_confidence_min": self._clamp_float(merged_conf, 20.0, 95.0),
            "convexity_min": self._clamp_float(merged_conv, 0.5, 5.0),
            "hold_days": float(self._clamp_int(merged_hold, 1, 20)),
            "max_daily_trades": float(self._clamp_int(merged_trades, 1, 5)),
        }

    def _resolved_mode_profiles(self) -> dict[str, dict[str, float]]:
        profiles = self._default_mode_profiles()
        raw_profiles = self.settings.raw.get("mode_profiles", {}) if isinstance(self.settings.raw, dict) else {}
        if isinstance(raw_profiles, dict):
            for mode_name, values in raw_profiles.items():
                if mode_name not in profiles or not isinstance(values, dict):
                    continue
                for key in ("signal_confidence_min", "convexity_min", "hold_days", "max_daily_trades"):
                    if key in values and isinstance(values.get(key), (int, float)):
                        profiles[mode_name][key] = float(values[key])
        return profiles

    @staticmethod
    def _regime_bucket(label: Any) -> str:
        txt = str(label or "").strip().lower()
        if not txt:
            return "unknown"
        if any(k in txt for k in ("极端波动", "extreme_vol", "extreme-vol", "extreme vol")):
            return "extreme_vol"
        if any(k in txt for k in ("强趋势", "弱趋势", "下跌趋势", "trend", "downtrend")):
            return "trend"
        if any(k in txt for k in ("震荡", "不确定", "range", "uncertain")):
            return "range"
        return "unknown"

    @staticmethod
    def _regime_threshold_map(raw: Any, *, default: float) -> dict[str, float]:
        out = {
            "trend": float(default),
            "range": float(default),
            "extreme_vol": float(default),
        }
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            bucket = LieEngine._regime_bucket(k)
            if bucket not in out:
                continue
            try:
                candidate = float(v)
            except Exception:
                continue
            if 0.0 <= candidate <= 1.0:
                out[bucket] = float(candidate)
        return out

    def _slot_regime_tuning_path(self) -> Path:
        return self.ctx.output_dir / "artifacts" / "slot_regime_thresholds_live.yaml"

    def _tune_slot_regime_thresholds(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("ops_slot_regime_tune_enabled", True))
        tuning_path = self._slot_regime_tuning_path()

        base_quality_default = float(val.get("ops_slot_eod_quality_anomaly_ratio_max", 0.50))
        base_risk_default = float(val.get("ops_slot_eod_risk_anomaly_ratio_max", 0.50))
        current_quality = self._regime_threshold_map(
            val.get("ops_slot_eod_quality_anomaly_ratio_max_by_regime", {}),
            default=base_quality_default,
        )
        current_risk = self._regime_threshold_map(
            val.get("ops_slot_eod_risk_anomaly_ratio_max_by_regime", {}),
            default=base_risk_default,
        )

        prev_live = {}
        if tuning_path.exists():
            try:
                prev_live = yaml.safe_load(tuning_path.read_text(encoding="utf-8")) or {}
            except Exception:
                prev_live = {}
        if isinstance(prev_live, dict):
            current_quality = self._regime_threshold_map(
                prev_live.get("ops_slot_eod_quality_anomaly_ratio_max_by_regime", current_quality),
                default=base_quality_default,
            )
            current_risk = self._regime_threshold_map(
                prev_live.get("ops_slot_eod_risk_anomaly_ratio_max_by_regime", current_risk),
                default=base_risk_default,
            )

        if not enabled:
            return {
                "enabled": False,
                "applied": False,
                "path": str(tuning_path),
                "ops_slot_eod_quality_anomaly_ratio_max_by_regime": current_quality,
                "ops_slot_eod_risk_anomaly_ratio_max_by_regime": current_risk,
                "reason": "disabled",
            }

        lookback_days = max(30, int(val.get("ops_slot_regime_tune_window_days", 180)))
        min_days = max(1, int(val.get("ops_slot_regime_tune_min_days", 20)))
        step = self._clamp_float(float(val.get("ops_slot_regime_tune_step", 0.12)), 0.01, 0.80)
        buffer = self._clamp_float(float(val.get("ops_slot_regime_tune_buffer", 0.08)), 0.0, 0.50)
        floor = self._clamp_float(float(val.get("ops_slot_regime_tune_floor", 0.10)), 0.0, 1.0)
        ceiling = self._clamp_float(float(val.get("ops_slot_regime_tune_ceiling", 0.80)), floor, 1.0)
        missing_ratio_hard_cap = self._clamp_float(
            float(val.get("ops_slot_regime_tune_missing_ratio_hard_cap", 0.80)),
            0.0,
            1.0,
        )
        risk_floor = self._clamp_float(
            float(val.get("ops_slot_risk_multiplier_floor", val.get("execution_min_risk_multiplier", 0.20))),
            0.0,
            1.0,
        )

        slot_missing_ratio = 0.0
        slot_missing_ratio_active = False
        slot_alerts: list[str] = []
        try:
            slot_anomaly = self._release_orchestrator()._slot_anomaly_metrics(as_of=as_of)
        except Exception:
            slot_anomaly = {}
        if isinstance(slot_anomaly, dict):
            slot_missing_ratio_active = bool(slot_anomaly.get("active", False))
            slot_metrics = slot_anomaly.get("metrics", {}) if isinstance(slot_anomaly.get("metrics", {}), dict) else {}
            slot_missing_ratio = float(slot_metrics.get("missing_ratio", 0.0))
            slot_alerts = list(slot_anomaly.get("alerts", [])) if isinstance(slot_anomaly.get("alerts", []), list) else []
        skip_for_missing = bool(slot_missing_ratio_active and slot_missing_ratio > missing_ratio_hard_cap)
        if skip_for_missing:
            payload = {
                "as_of": as_of.isoformat(),
                "generated_at": datetime.now().isoformat(),
                "lookback_days": int(lookback_days),
                "min_days": int(min_days),
                "step": float(step),
                "buffer": float(buffer),
                "floor": float(floor),
                "ceiling": float(ceiling),
                "risk_multiplier_floor": float(risk_floor),
                "ops_slot_regime_tune_missing_ratio_hard_cap": float(missing_ratio_hard_cap),
                "slot_missing_ratio": float(slot_missing_ratio),
                "slot_missing_ratio_active": bool(slot_missing_ratio_active),
                "slot_alerts": slot_alerts[:10],
                "ops_slot_eod_quality_anomaly_ratio_max_by_regime": current_quality,
                "ops_slot_eod_risk_anomaly_ratio_max_by_regime": current_risk,
                "buckets": {},
                "changed": False,
                "skipped": True,
                "skip_reason": "slot_missing_ratio_high",
            }
            write_markdown(tuning_path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
            return {
                "enabled": True,
                "applied": False,
                "path": str(tuning_path),
                "changed": False,
                "skipped": True,
                "reason": "slot_missing_ratio_high",
                "slot_missing_ratio": float(slot_missing_ratio),
                "ops_slot_regime_tune_missing_ratio_hard_cap": float(missing_ratio_hard_cap),
                "ops_slot_eod_quality_anomaly_ratio_max_by_regime": current_quality,
                "ops_slot_eod_risk_anomaly_ratio_max_by_regime": current_risk,
                "buckets": {},
            }

        stats = {
            "trend": {"days": 0, "quality_anomalies": 0, "risk_anomalies": 0},
            "range": {"days": 0, "quality_anomalies": 0, "risk_anomalies": 0},
            "extreme_vol": {"days": 0, "quality_anomalies": 0, "risk_anomalies": 0},
        }
        manifest_dir = self.ctx.output_dir / "artifacts" / "manifests"
        for i in range(lookback_days):
            day = as_of - timedelta(days=i)
            payload = self._load_json_safely(manifest_dir / f"eod_{day.isoformat()}.json")
            if not payload:
                continue
            metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
            checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
            bucket = self._regime_bucket(metrics.get("regime", ""))
            if bucket not in stats:
                continue
            stats[bucket]["days"] = int(stats[bucket]["days"]) + 1
            if not bool(checks.get("quality_passed", True)):
                stats[bucket]["quality_anomalies"] = int(stats[bucket]["quality_anomalies"]) + 1
            if float(metrics.get("risk_multiplier", 1.0)) < risk_floor:
                stats[bucket]["risk_anomalies"] = int(stats[bucket]["risk_anomalies"]) + 1

        tuned_quality = dict(current_quality)
        tuned_risk = dict(current_risk)
        summary_buckets: dict[str, Any] = {}
        changed = False
        for bucket in ("trend", "range", "extreme_vol"):
            days = int(stats[bucket]["days"])
            q_anom = int(stats[bucket]["quality_anomalies"])
            r_anom = int(stats[bucket]["risk_anomalies"])
            q_ratio = float(q_anom / days) if days > 0 else 0.0
            r_ratio = float(r_anom / days) if days > 0 else 0.0
            q_before = float(tuned_quality[bucket])
            r_before = float(tuned_risk[bucket])
            q_after = q_before
            r_after = r_before
            if days >= min_days:
                q_target = self._clamp_float(q_ratio + buffer, floor, ceiling)
                r_target = self._clamp_float(r_ratio + buffer, floor, ceiling)
                q_after = float(bounded_bayesian_update(q_before, q_target, floor, ceiling, step=step))
                r_after = float(bounded_bayesian_update(r_before, r_target, floor, ceiling, step=step))
                tuned_quality[bucket] = q_after
                tuned_risk[bucket] = r_after
                if abs(q_after - q_before) > 1e-9 or abs(r_after - r_before) > 1e-9:
                    changed = True
            summary_buckets[bucket] = {
                "days": days,
                "quality_anomaly_ratio": q_ratio,
                "risk_anomaly_ratio": r_ratio,
                "quality_before": q_before,
                "quality_after": q_after,
                "risk_before": r_before,
                "risk_after": r_after,
                "eligible": bool(days >= min_days),
            }

        payload = {
            "as_of": as_of.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "lookback_days": int(lookback_days),
            "min_days": int(min_days),
            "step": float(step),
            "buffer": float(buffer),
            "floor": float(floor),
            "ceiling": float(ceiling),
            "risk_multiplier_floor": float(risk_floor),
            "ops_slot_regime_tune_missing_ratio_hard_cap": float(missing_ratio_hard_cap),
            "slot_missing_ratio": float(slot_missing_ratio),
            "slot_missing_ratio_active": bool(slot_missing_ratio_active),
            "slot_alerts": slot_alerts[:10],
            "ops_slot_eod_quality_anomaly_ratio_max_by_regime": tuned_quality,
            "ops_slot_eod_risk_anomaly_ratio_max_by_regime": tuned_risk,
            "buckets": summary_buckets,
            "changed": bool(changed),
            "skipped": False,
        }
        write_markdown(tuning_path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
        return {
            "enabled": True,
            "applied": True,
            "path": str(tuning_path),
            "changed": bool(changed),
            "skipped": False,
            "ops_slot_regime_tune_missing_ratio_hard_cap": float(missing_ratio_hard_cap),
            "slot_missing_ratio": float(slot_missing_ratio),
            "ops_slot_eod_quality_anomaly_ratio_max_by_regime": tuned_quality,
            "ops_slot_eod_risk_anomaly_ratio_max_by_regime": tuned_risk,
            "buckets": summary_buckets,
        }

    def _load_latest_strategy_candidate(self, as_of: date) -> dict[str, Any]:
        if not bool(self.settings.validation.get("review_use_strategy_lab", True)):
            return {}
        lookback_days = int(self.settings.validation.get("strategy_lab_manifest_lookback_days", 45))
        manifest_dir = self.ctx.output_dir / "artifacts" / "manifests"
        if not manifest_dir.exists():
            return {}

        manifests = sorted(manifest_dir.glob("strategy_lab_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for mp in manifests:
            manifest = self._load_json_safely(mp)
            if not manifest:
                continue
            artifacts = manifest.get("artifacts", {})
            if not isinstance(artifacts, dict):
                continue
            summary_ref = artifacts.get("summary")
            if not summary_ref:
                continue
            summary_path = Path(str(summary_ref))
            if not summary_path.is_absolute():
                summary_path = self.ctx.root / summary_path
            summary = self._load_json_safely(summary_path)
            if not summary:
                continue
            metadata = manifest.get("metadata", {}) if isinstance(manifest.get("metadata", {}), dict) else {}
            checks = manifest.get("checks", {}) if isinstance(manifest.get("checks", {}), dict) else {}
            summary_fetch_stats = summary.get("data_fetch_stats", {})
            if not isinstance(summary_fetch_stats, dict):
                summary_fetch_stats = {}
            cutoff = self._parse_iso_date(summary.get("cutoff_date"))
            if cutoff is None:
                cutoff = self._parse_iso_date(metadata.get("cutoff_date"))
            if cutoff is None:
                continue
            if cutoff > as_of:
                continue
            if (as_of - cutoff).days > max(1, lookback_days):
                continue
            strict_cutoff_enforced = (
                bool(checks.get("strict_cutoff_enforced"))
                if "strict_cutoff_enforced" in checks
                else bool(summary_fetch_stats.get("strict_cutoff_enforced", False))
            )
            if not strict_cutoff_enforced:
                continue

            cutoff_ts_raw = str(summary.get("cutoff_ts") or metadata.get("cutoff_ts") or f"{cutoff.isoformat()}T23:59:59")
            cutoff_ts_dt = self._parse_iso_datetime(cutoff_ts_raw) or self._parse_iso_datetime(f"{cutoff.isoformat()}T23:59:59")
            if cutoff_ts_dt is None:
                continue

            def _temporal_field(name: str) -> str:
                return str(summary.get(name) or metadata.get(name) or summary_fetch_stats.get(name) or "").strip()

            bar_max_ts = _temporal_field("bar_max_ts")
            news_max_ts = _temporal_field("news_max_ts")
            report_max_ts = _temporal_field("report_max_ts")

            temporal_ok = True
            for ts_raw in (bar_max_ts, news_max_ts, report_max_ts):
                if not ts_raw:
                    continue
                ts_dt = self._parse_iso_datetime(ts_raw)
                ts_date = ts_dt.date() if ts_dt is not None else self._parse_iso_date(ts_raw)
                if ts_date is None or ts_date > cutoff:
                    temporal_ok = False
                    break
            if not temporal_ok:
                continue

            best = summary.get("best_candidate", {})
            if not isinstance(best, dict):
                best = {}
            if (not best) or (not bool(best.get("accepted", False))):
                cands = summary.get("candidates", [])
                if isinstance(cands, list):
                    accepted = [c for c in cands if isinstance(c, dict) and bool(c.get("accepted", False))]
                    if accepted:
                        accepted.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
                        best = accepted[0]
            if not isinstance(best, dict) or not best:
                continue
            params = best.get("params", {})
            if not isinstance(params, dict) or not params:
                continue
            return {
                "manifest_path": str(mp),
                "summary_path": str(summary_path),
                "cutoff_date": cutoff.isoformat(),
                "cutoff_ts": cutoff_ts_raw,
                "bar_max_ts": bar_max_ts,
                "news_max_ts": news_max_ts,
                "report_max_ts": report_max_ts,
                "strict_cutoff_enforced": strict_cutoff_enforced,
                "candidate": best,
            }
        return {}

    def _autorun_strategy_lab_for_review(self, as_of: date) -> dict[str, Any]:
        if not bool(self.settings.validation.get("review_autorun_strategy_lab_if_missing", False)):
            return {}
        lookback_days = max(60, int(self.settings.validation.get("strategy_lab_autorun_lookback_days", 180)))
        start = max(date(2015, 1, 1), as_of - timedelta(days=lookback_days))
        max_symbols = max(1, int(self.settings.validation.get("strategy_lab_autorun_max_symbols", 60)))
        report_symbol_cap = max(1, int(self.settings.validation.get("strategy_lab_autorun_report_symbol_cap", 20)))
        workers = max(1, int(self.settings.validation.get("strategy_lab_autorun_workers", 4)))
        review_days = max(0, int(self.settings.validation.get("strategy_lab_autorun_review_days", 3)))
        candidate_count = max(1, int(self.settings.validation.get("strategy_lab_autorun_candidate_count", 6)))
        return self.run_strategy_lab(
            start=start,
            end=as_of,
            max_symbols=max_symbols,
            report_symbol_cap=report_symbol_cap,
            workers=workers,
            review_days=review_days,
            candidate_count=candidate_count,
        )

    def _merge_strategy_candidate_into_review(
        self,
        *,
        review: ReviewDelta,
        current_params: dict[str, float],
        candidate_payload: dict[str, Any],
    ) -> None:
        cand = candidate_payload.get("candidate", {})
        if not isinstance(cand, dict):
            return
        params = cand.get("params", {})
        if not isinstance(params, dict):
            return

        step = self._clamp_float(float(self.settings.validation.get("strategy_lab_merge_step", 0.18)), 0.01, 0.60)
        updated: list[str] = []
        fields: list[tuple[str, float, float, bool]] = [
            ("signal_confidence_min", 30.0, 90.0, False),
            ("convexity_min", 1.0, 4.0, False),
            ("hold_days", 1.0, 20.0, True),
            ("max_daily_trades", 1.0, 5.0, True),
        ]
        for key, lo, hi, is_int in fields:
            if key not in params:
                continue
            target = float(params.get(key, 0.0))
            target = self._clamp_float(target, lo, hi)
            base = float(review.parameter_changes.get(key, current_params.get(key, target)))
            merged = bounded_bayesian_update(base, target, lo, hi, step=step)
            if is_int:
                review.parameter_changes[key] = float(self._clamp_int(merged, int(lo), int(hi)))
            else:
                review.parameter_changes[key] = float(merged)
            review.change_reasons[key] = f"融合 strategy-lab 最优候选 `{cand.get('name', 'unknown')}`（小步收敛，step={step:.2f}）"
            updated.append(key)

        if updated:
            review.notes.append(
                "strategy_lab_candidate="
                + f"{cand.get('name', 'unknown')}; cutoff={candidate_payload.get('cutoff_date', '')}; "
                + f"bar_max_ts={candidate_payload.get('bar_max_ts', '')}; "
                + f"news_max_ts={candidate_payload.get('news_max_ts', '')}; "
                + f"manifest={candidate_payload.get('manifest_path', '')}"
            )

    def _quality_snapshot(self, as_of: date) -> dict[str, Any]:
        path = self.ctx.output_dir / "artifacts" / f"{as_of.isoformat()}_quality.json"
        return self._load_json_safely(path)

    def _backtest_snapshot(self, as_of: date) -> dict[str, Any]:
        path = self.ctx.output_dir / "artifacts" / f"backtest_{date(2015, 1, 1).isoformat()}_{as_of.isoformat()}.json"
        return self._load_json_safely(path)

    @staticmethod
    def _filter_model_path_bars(bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty or "data_conflict_flag" not in bars.columns:
            return bars
        return bars[bars["data_conflict_flag"] == 0].copy()

    def _major_event_window(self, as_of: date, news: list[NewsEvent]) -> bool:
        lookback_hours = int(self.settings.validation.get("major_event_window_hours", 24))
        return major_event_window(as_of=as_of, news=news, lookback_hours=lookback_hours)

    def _loss_cooldown_active(self, recent_trades: pd.DataFrame) -> bool:
        cooldown_losses = int(self.settings.validation.get("cooldown_consecutive_losses", 3))
        return loss_cooldown_active(recent_trades=recent_trades, cooldown_losses=cooldown_losses)

    def _black_swan_assessment(self, regime, sentiment: dict[str, float], news: list[NewsEvent]) -> tuple[float, list[str], bool]:
        threshold = float(self.settings.validation.get("black_swan_score_threshold", 70.0))
        return black_swan_assessment(
            atr_z=float(regime.atr_z),
            sentiment=sentiment,
            news=news,
            threshold=threshold,
        )

    def _evaluate_guards(
        self,
        *,
        as_of: date,
        regime,
        quality_passed: bool,
        sentiment: dict[str, float],
        news: list[NewsEvent],
        recent_trades: pd.DataFrame,
    ) -> GuardAssessment:
        score, items, trigger = self._black_swan_assessment(regime=regime, sentiment=sentiment, news=news)
        event_window = self._major_event_window(as_of=as_of, news=news)
        cooldown = self._loss_cooldown_active(recent_trades=recent_trades)

        reasons: list[str] = []
        if event_window:
            reasons.append("重大事件窗口：暂停新增仓位")
        if not quality_passed:
            reasons.append("数据质量未通过门禁：进入保护模式并禁止开新仓")
        if regime.consensus in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL}:
            reasons.append("体制冲突/极端波动：不做方向强判")
        if cooldown:
            reasons.append("连亏冷却期未结束：暂停新增仓位")
        if trigger:
            reasons.append("黑天鹅评分超阈值：保护模式生效")

        return GuardAssessment(
            black_swan_score=score,
            black_swan_items=items,
            black_swan_trigger=trigger,
            major_event_window=event_window,
            cooldown_active=cooldown,
            non_trade_reasons=reasons,
        )

    def _estimate_factor_contrib_120d(self, as_of: date) -> dict[str, float]:
        lookback_days = int(self.settings.validation.get("factor_lookback_days", 120))
        start = as_of - timedelta(days=max(lookback_days * 2, 220))
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion_range(start=start, end=as_of, symbols=symbols)
        bars = self._filter_model_path_bars(bars)
        return estimate_factor_contrib_120d(
            bars=bars,
            ingest=ingest,
            providers=self.providers,
            lookback_days=lookback_days,
        )

    def _run_ingestion(self, as_of: date, symbols: list[str]) -> tuple[pd.DataFrame, Any]:
        start = as_of - timedelta(days=420)
        start_ts = datetime.combine(as_of, time(0, 0)) - timedelta(days=1)
        end_ts = datetime.combine(as_of, time(23, 59))

        result = self.data_bus.ingest(
            symbols=symbols,
            start=start,
            end=as_of,
            start_ts=start_ts,
            end_ts=end_ts,
            langs=("zh", "en"),
        )
        expanded = expand_universe(
            core_symbols=symbols,
            bars=result.normalized_bars,
            max_additions=int(self.settings.universe.get("max_dynamic_additions", 5)),
        )
        if set(expanded) - set(symbols):
            result = self.data_bus.ingest(
                symbols=expanded,
                start=start,
                end=as_of,
                start_ts=start_ts,
                end_ts=end_ts,
                langs=("zh", "en"),
            )
            symbols = expanded

        self.data_bus.persist(as_of=as_of, result=result)
        model_bars = self._filter_model_path_bars(result.normalized_bars)
        return model_bars, result

    def _run_ingestion_range(self, start: date, end: date, symbols: list[str]) -> tuple[pd.DataFrame, Any]:
        start_ts = datetime.combine(start, time(0, 0))
        end_ts = datetime.combine(end, time(23, 59))

        result = self.data_bus.ingest(
            symbols=symbols,
            start=start,
            end=end,
            start_ts=start_ts,
            end_ts=end_ts,
            langs=("zh", "en"),
        )
        expanded = expand_universe(
            core_symbols=symbols,
            bars=result.normalized_bars,
            max_additions=int(self.settings.universe.get("max_dynamic_additions", 5)),
        )
        if set(expanded) - set(symbols):
            result = self.data_bus.ingest(
                symbols=expanded,
                start=start,
                end=end,
                start_ts=start_ts,
                end_ts=end_ts,
                langs=("zh", "en"),
            )

        self.data_bus.persist(as_of=end, result=result)
        model_bars = self._filter_model_path_bars(result.normalized_bars)
        return model_bars, result

    def _regime_from_bars(self, as_of: date, bars: pd.DataFrame):
        if bars.empty:
            return derive_regime_consensus(
                as_of=as_of,
                hurst=0.5,
                hmm_probs={"bull": 0.33, "range": 0.34, "bear": 0.33},
                atr_z=0.0,
                trend_thr=float(self.settings.thresholds.get("hurst_trend", 0.6)),
                mean_thr=float(self.settings.thresholds.get("hurst_mean_revert", 0.4)),
                atr_extreme=float(self.settings.thresholds.get("atr_extreme", 2.0)),
            )

        market_proxy = bars.groupby("ts", as_index=False).agg({"close": "mean", "high": "mean", "low": "mean", "open": "mean", "volume": "mean"})
        market_proxy = market_proxy.sort_values("ts")

        hurst = latest_multi_scale_hurst(market_proxy["close"].to_numpy())
        hmm_probs = infer_hmm_state(market_proxy[["close", "volume"]].assign(high=market_proxy["close"], low=market_proxy["close"]))
        atr_z = compute_atr_zscore(market_proxy[["open", "high", "low", "close"]])

        regime = derive_regime_consensus(
            as_of=as_of,
            hurst=hurst,
            hmm_probs=hmm_probs,
            atr_z=atr_z,
            trend_thr=float(self.settings.thresholds.get("hurst_trend", 0.6)),
            mean_thr=float(self.settings.thresholds.get("hurst_mean_revert", 0.4)),
            atr_extreme=float(self.settings.thresholds.get("atr_extreme", 2.0)),
        )
        return regime

    def _load_recent_trades(self, n: int = 200) -> pd.DataFrame:
        if not self.ctx.sqlite_path.exists():
            return pd.DataFrame(columns=["date", "symbol", "pnl"])

        query = f"SELECT * FROM executed_plans ORDER BY rowid DESC LIMIT {int(n)}"
        with closing(sqlite3.connect(self.ctx.sqlite_path)) as conn:
            try:
                df = pd.read_sql_query(query, conn)
            except Exception:
                return pd.DataFrame(columns=["date", "symbol", "pnl"])
        return df

    def _paper_positions_state_path(self) -> Path:
        return self.ctx.output_dir / "artifacts" / "paper_positions_open.json"

    def _broker_snapshot_path(self, as_of: date) -> Path:
        return self.ctx.output_dir / "artifacts" / "broker_snapshot" / f"{as_of.isoformat()}.json"

    def _broker_snapshot_live_inbox_path(self, as_of: date) -> Path:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        raw = str(val.get("broker_snapshot_live_inbox", "output/artifacts/broker_live_inbox")).strip()
        path = Path(raw) if raw else Path("output/artifacts/broker_live_inbox")
        if not path.is_absolute():
            path = self.ctx.root / path
        return path / f"{as_of.isoformat()}.json"

    def _broker_snapshot_source_mode(self) -> str:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        mode = str(val.get("broker_snapshot_source_mode", "paper_engine")).strip().lower()
        if mode not in {"paper_engine", "live_adapter", "hybrid_prefer_live"}:
            return "paper_engine"
        return mode

    def _broker_snapshot_live_mapping_profile(self) -> str:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        profile = str(val.get("broker_snapshot_live_mapping_profile", "generic")).strip().lower()
        if profile not in {"generic", "ibkr", "binance", "ctp"}:
            return "generic"
        return profile

    @staticmethod
    def _deep_get_by_path(data: Any, path: str) -> Any:
        if not path:
            return None
        cur = data
        for part in str(path).split("."):
            key = str(part).strip()
            if key == "":
                return None
            if isinstance(cur, dict):
                if key not in cur:
                    return None
                cur = cur.get(key)
            else:
                return None
        return cur

    def _pick_first(self, data: Any, paths: list[str]) -> Any:
        for p in paths:
            value = self._deep_get_by_path(data, str(p))
            if value is not None:
                return value
        return None

    @staticmethod
    def _default_live_mapping_profiles() -> dict[str, dict[str, Any]]:
        return {
            "generic": {
                "source": ["source", "broker", "adapter_source"],
                "open_positions": ["open_positions", "open_count", "positions_count"],
                "closed_count": ["closed_count", "closed_trades", "summary.closed_count"],
                "closed_pnl": ["closed_pnl", "realized_pnl", "realizedPnL", "summary.realized_pnl"],
                "positions": ["positions", "open_positions_detail", "position_list"],
                "position_fields": {
                    "symbol": ["symbol", "ticker", "instrument", "instrument_id"],
                    "side": ["side", "direction", "position_side", "positionSide", "posSide"],
                    "qty": ["qty", "quantity", "position_qty", "size", "positionAmt", "position"],
                    "notional": ["notional", "notional_value", "marketValue", "market_value"],
                    "entry_price": ["entry_price", "avg_price", "average_price", "entryPrice", "avgCost"],
                    "market_price": ["market_price", "last_price", "mark_price", "markPrice", "marketPrice"],
                    "status": ["status", "state"],
                },
            },
            "ibkr": {
                "source": ["source", "broker", "adapter_source"],
                "open_positions": ["open_positions", "summary.open_positions"],
                "closed_count": ["closed_count", "summary.closed_count"],
                "closed_pnl": ["closed_pnl", "summary.realized_pnl", "realized_pnl"],
                "positions": ["positions", "portfolio"],
                "position_fields": {
                    "symbol": ["symbol", "contract.symbol", "localSymbol", "contract.localSymbol"],
                    "side": ["side", "direction"],
                    "qty": ["qty", "position", "quantity", "size"],
                    "notional": ["notional", "marketValue", "market_value"],
                    "entry_price": ["entry_price", "avgCost", "average_cost", "avg_price"],
                    "market_price": ["market_price", "marketPrice", "last_price", "markPrice"],
                    "status": ["status", "state"],
                },
            },
            "binance": {
                "source": ["source", "broker", "adapter_source"],
                "open_positions": ["open_positions", "summary.open_positions"],
                "closed_count": ["closed_count", "summary.closed_count"],
                "closed_pnl": ["closed_pnl", "realizedPnl", "summary.realized_pnl"],
                "positions": ["positions", "account.positions"],
                "position_fields": {
                    "symbol": ["symbol"],
                    "side": ["side", "positionSide"],
                    "qty": ["qty", "positionAmt", "amount", "size"],
                    "notional": ["notional", "notionalValue", "positionInitialMargin"],
                    "entry_price": ["entry_price", "entryPrice", "avgPrice"],
                    "market_price": ["market_price", "markPrice", "price", "lastPrice"],
                    "status": ["status", "state"],
                },
            },
            "ctp": {
                "source": ["source", "broker", "adapter_source"],
                "open_positions": ["open_positions", "summary.open_positions"],
                "closed_count": ["closed_count", "summary.closed_count"],
                "closed_pnl": ["closed_pnl", "realized_pnl", "summary.realized_pnl"],
                "positions": ["positions", "position_list"],
                "position_fields": {
                    "symbol": ["symbol", "instrument", "instrument_id"],
                    "side": ["side", "direction", "posi_direction"],
                    "qty": ["qty", "position", "volume"],
                    "notional": ["notional"],
                    "entry_price": ["entry_price", "open_price", "avg_price"],
                    "market_price": ["market_price", "last_price", "settlement_price"],
                    "status": ["status", "state"],
                },
            },
        }

    @staticmethod
    def _merge_live_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key in ("source", "open_positions", "closed_count", "closed_pnl", "positions"):
            if key in override and isinstance(override.get(key), list):
                merged[key] = [str(x) for x in override.get(key, []) if str(x).strip()]
        pf = dict(base.get("position_fields", {})) if isinstance(base.get("position_fields", {}), dict) else {}
        override_pf = override.get("position_fields", {})
        if isinstance(override_pf, dict):
            for f_key, f_paths in override_pf.items():
                if isinstance(f_paths, list):
                    pf[str(f_key)] = [str(x) for x in f_paths if str(x).strip()]
        merged["position_fields"] = pf
        return merged

    def _resolve_live_mapping_profile(self) -> tuple[str, dict[str, Any]]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        profile = self._broker_snapshot_live_mapping_profile()
        all_profiles = self._default_live_mapping_profiles()
        base = all_profiles.get(profile, all_profiles["generic"])
        override = val.get("broker_snapshot_live_mapping_fields", {})
        if isinstance(override, dict):
            base = self._merge_live_mapping(base, override)
        return profile, base

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _normalize_position_side(side_raw: Any, qty_signed: float) -> str:
        side = str(side_raw or "").strip().upper()
        if side in {"LONG", "BUY", "B", "L", "2"}:
            return "LONG"
        if side in {"SHORT", "SELL", "S", "-1", "3"}:
            return "SHORT"
        if side in {"FLAT", "NONE", "0"}:
            return "FLAT"
        if side in {"", "BOTH", "NET"}:
            if qty_signed < 0:
                return "SHORT"
            if qty_signed > 0:
                return "LONG"
            return "FLAT"
        if qty_signed < 0:
            return "SHORT"
        if qty_signed > 0:
            return "LONG"
        return "FLAT"

    def _normalize_live_broker_snapshot(
        self,
        *,
        as_of: date,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        profile, mapping = self._resolve_live_mapping_profile()
        positions_key_paths = mapping.get("positions", ["positions"])
        positions_raw = self._pick_first(payload, positions_key_paths if isinstance(positions_key_paths, list) else ["positions"])
        positions_in = positions_raw if isinstance(positions_raw, list) else []
        pf = mapping.get("position_fields", {}) if isinstance(mapping.get("position_fields", {}), dict) else {}
        positions: list[dict[str, Any]] = []
        for row in positions_in:
            if not isinstance(row, dict):
                continue
            qty_raw = self._pick_first(row, pf.get("qty", ["qty", "quantity"])) if isinstance(pf.get("qty", []), list) else row.get("qty", 0.0)
            qty_signed = self._to_float(qty_raw, 0.0)
            side_raw = self._pick_first(row, pf.get("side", ["side"])) if isinstance(pf.get("side", []), list) else row.get("side", "")
            side = self._normalize_position_side(side_raw, qty_signed)
            qty = abs(qty_signed)
            status_raw = self._pick_first(row, pf.get("status", ["status"])) if isinstance(pf.get("status", []), list) else row.get("status", "")
            status = str(status_raw or "").strip().upper() or "OPEN"
            if qty <= 1e-12 and status not in {"OPEN", "ACTIVE"}:
                continue
            symbol_raw = self._pick_first(row, pf.get("symbol", ["symbol"])) if isinstance(pf.get("symbol", []), list) else row.get("symbol", "")
            entry_price_raw = self._pick_first(row, pf.get("entry_price", ["entry_price"])) if isinstance(pf.get("entry_price", []), list) else row.get("entry_price", 0.0)
            market_price_raw = self._pick_first(row, pf.get("market_price", ["market_price"])) if isinstance(pf.get("market_price", []), list) else row.get("market_price", 0.0)
            notional_raw = self._pick_first(row, pf.get("notional", ["notional"])) if isinstance(pf.get("notional", []), list) else row.get("notional", 0.0)
            market_price = self._to_float(market_price_raw, 0.0)
            notional = self._to_float(notional_raw, 0.0)
            if abs(notional) <= 1e-12 and qty > 0 and market_price > 0:
                notional = qty * market_price
            positions.append(
                {
                    "symbol": str(symbol_raw or ""),
                    "side": side,
                    "qty": qty,
                    "notional": notional,
                    "entry_price": self._to_float(entry_price_raw, 0.0),
                    "market_price": market_price,
                    "status": status,
                }
            )
        open_positions_raw = self._pick_first(payload, mapping.get("open_positions", ["open_positions"]))
        open_positions = int(round(self._to_float(open_positions_raw, len(positions))))
        open_positions = max(0, open_positions)
        closed_pnl_raw = self._pick_first(payload, mapping.get("closed_pnl", ["closed_pnl"]))
        closed_count_raw = self._pick_first(payload, mapping.get("closed_count", ["closed_count"]))
        closed_pnl = self._to_float(closed_pnl_raw, 0.0)
        closed_count = int(round(self._to_float(closed_count_raw, 0.0)))
        closed_count = max(0, closed_count)
        source_raw = self._pick_first(payload, mapping.get("source", ["source"]))
        source = str(source_raw or "live_broker").strip() or "live_broker"

        return {
            "date": as_of.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "source": "live_adapter",
            "adapter_source": source,
            "mapping_profile": profile,
            "open_positions": open_positions,
            "closed_count": closed_count,
            "closed_pnl": closed_pnl,
            "positions": positions,
            "stats": {
                "raw_path": str(self._broker_snapshot_live_inbox_path(as_of)),
                "positions_rows_in": int(len(positions_in)),
                "positions_rows_out": int(len(positions)),
            },
        }

    def _load_open_paper_positions(self) -> list[dict[str, Any]]:
        payload = self._load_json_safely(self._paper_positions_state_path())
        rows = payload.get("positions", []) if isinstance(payload.get("positions", []), list) else []
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(dict(row))
        return out

    def _save_open_paper_positions(self, *, as_of: date, positions: list[dict[str, Any]]) -> None:
        write_json(
            self._paper_positions_state_path(),
            {
                "as_of": as_of.isoformat(),
                "positions": positions,
            },
        )

    def _write_paper_broker_snapshot(
        self,
        *,
        as_of: date,
        active_positions: list[dict[str, Any]],
        paper_exec_summary: dict[str, Any],
    ) -> Path:
        positions: list[dict[str, Any]] = []
        for row in active_positions:
            if not isinstance(row, dict):
                continue
            positions.append(
                {
                    "open_date": str(row.get("open_date", "")),
                    "symbol": str(row.get("symbol", "")),
                    "side": str(row.get("side", "")),
                    "size_pct": float(row.get("size_pct", 0.0)),
                    "risk_pct": float(row.get("risk_pct", 0.0)),
                    "entry_price": float(row.get("entry_price", 0.0)),
                    "stop_price": float(row.get("stop_price", 0.0)),
                    "target_price": float(row.get("target_price", 0.0)),
                    "runtime_mode": str(row.get("runtime_mode", "base")),
                    "status": str(row.get("status", "OPEN")),
                }
            )

        payload = {
            "date": as_of.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "source": "paper_engine",
            "open_positions": int(len(positions)),
            "closed_count": int((paper_exec_summary or {}).get("closed_count", 0)),
            "closed_pnl": float((paper_exec_summary or {}).get("closed_pnl", 0.0)),
            "positions": positions,
            "stats": {
                "open_before": int((paper_exec_summary or {}).get("open_before", 0)),
                "missing_symbol_count": int((paper_exec_summary or {}).get("missing_symbol_count", 0)),
            },
        }
        out_path = self._broker_snapshot_path(as_of)
        write_json(out_path, payload)
        return out_path

    def _resolve_and_write_broker_snapshot(
        self,
        *,
        as_of: date,
        active_positions: list[dict[str, Any]],
        paper_exec_summary: dict[str, Any],
    ) -> Path | None:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        mode = self._broker_snapshot_source_mode()
        fallback_to_paper = bool(val.get("broker_snapshot_live_fallback_to_paper", True))
        target_path = self._broker_snapshot_path(as_of)

        live_inbox = self._broker_snapshot_live_inbox_path(as_of)
        live_payload = self._load_json_safely(live_inbox)
        live_ready = bool(live_payload)
        prefer_live = mode in {"live_adapter", "hybrid_prefer_live"}

        if prefer_live and live_ready:
            normalized = self._normalize_live_broker_snapshot(as_of=as_of, payload=live_payload)
            write_json(target_path, normalized)
            return target_path

        if mode == "live_adapter" and (not live_ready) and (not fallback_to_paper):
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            return None

        paper_path = self._write_paper_broker_snapshot(
            as_of=as_of,
            active_positions=active_positions,
            paper_exec_summary=paper_exec_summary,
        )
        if mode == "live_adapter" and (not live_ready):
            payload = self._load_json_safely(paper_path)
            payload["source"] = "paper_engine_fallback"
            payload["fallback_reason"] = "live_snapshot_missing"
            payload["live_inbox"] = str(live_inbox)
            write_json(paper_path, payload)
        return paper_path

    @staticmethod
    def _bars_daily_snapshot(bars: pd.DataFrame) -> dict[str, dict[str, float]]:
        if bars.empty:
            return {}
        cols = {"symbol", "ts", "open", "high", "low", "close"}
        if not cols.issubset(set(bars.columns)):
            return {}
        snap = bars[["symbol", "ts", "open", "high", "low", "close"]].copy()
        snap["ts"] = pd.to_datetime(snap["ts"], errors="coerce")
        snap = snap.dropna(subset=["symbol", "ts"])
        if snap.empty:
            return {}
        for c in ("open", "high", "low", "close"):
            snap[c] = pd.to_numeric(snap[c], errors="coerce")
        snap = snap.dropna(subset=["open", "high", "low", "close"])
        if snap.empty:
            return {}
        snap = snap.sort_values(["symbol", "ts"])
        agg = (
            snap.groupby("symbol", as_index=False)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
            )
        )
        out: dict[str, dict[str, float]] = {}
        for row in agg.to_dict(orient="records"):
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            out[sym] = {
                "open": float(row.get("open", 0.0)),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("close", 0.0)),
            }
        return out

    def _settle_open_paper_positions(self, *, as_of: date, bars: pd.DataFrame) -> dict[str, Any]:
        positions = self._load_open_paper_positions()
        if not positions:
            return {
                "closed_df": pd.DataFrame(
                    columns=[
                        "date",
                        "open_date",
                        "symbol",
                        "side",
                        "direction",
                        "runtime_mode",
                        "mode",
                        "size_pct",
                        "risk_pct",
                        "entry_price",
                        "exit_price",
                        "pnl",
                        "pnl_pct",
                        "exit_reason",
                        "hold_days",
                        "holding_days",
                        "status",
                    ]
                ),
                "remaining_positions": [],
                "summary": {
                    "open_before": 0,
                    "closed_count": 0,
                    "closed_pnl": 0.0,
                    "missing_symbol_count": 0,
                },
            }

        snapshot = self._bars_daily_snapshot(bars)
        remaining: list[dict[str, Any]] = []
        closed_rows: list[dict[str, Any]] = []
        missing_symbol_count = 0

        for pos in positions:
            symbol = str(pos.get("symbol", "")).strip()
            side = str(pos.get("side", "LONG")).strip().upper()
            if not symbol or symbol not in snapshot:
                remaining.append(pos)
                missing_symbol_count += 1 if symbol else 0
                continue

            px = snapshot[symbol]
            entry = float(pos.get("entry_price", px["close"]))
            stop = float(pos.get("stop_price", entry))
            target = float(pos.get("target_price", entry))
            size_pct = float(pos.get("size_pct", 0.0))
            risk_pct = float(pos.get("risk_pct", 0.0))
            hold_days = max(1, int(pos.get("hold_days", 1)))
            open_date = self._parse_iso_date(pos.get("open_date")) or as_of
            holding_days = max(0, (as_of - open_date).days)
            runtime_mode = str(pos.get("runtime_mode", "base")).strip() or "base"

            exit_price = float(px["close"])
            exit_reason = ""

            if side == "SHORT":
                if float(px["high"]) >= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif float(px["low"]) <= target:
                    exit_price = target
                    exit_reason = "take_profit"
            else:
                if float(px["low"]) <= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif float(px["high"]) >= target:
                    exit_price = target
                    exit_reason = "take_profit"

            if not exit_reason and holding_days >= hold_days:
                exit_price = float(px["close"])
                exit_reason = "time_stop"

            if not exit_reason:
                remaining.append(pos)
                continue

            signed_ret = (exit_price - entry) / max(entry, 1e-9)
            if side == "SHORT":
                signed_ret = -signed_ret
            pnl = float((size_pct / 100.0) * signed_ret)

            closed_rows.append(
                {
                    "date": as_of.isoformat(),
                    "open_date": open_date.isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "direction": side,
                    "runtime_mode": runtime_mode,
                    "mode": runtime_mode,
                    "size_pct": size_pct,
                    "risk_pct": risk_pct,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": float(signed_ret),
                    "exit_reason": exit_reason,
                    "hold_days": hold_days,
                    "holding_days": holding_days,
                    "status": "CLOSED",
                }
            )

        closed_df = pd.DataFrame(closed_rows)
        if not closed_df.empty:
            append_sqlite(self.ctx.sqlite_path, "executed_plans", closed_df)

        return {
            "closed_df": closed_df,
            "remaining_positions": remaining,
            "summary": {
                "open_before": len(positions),
                "closed_count": int(len(closed_rows)),
                "closed_pnl": float(closed_df["pnl"].sum()) if not closed_df.empty else 0.0,
                "missing_symbol_count": int(missing_symbol_count),
            },
        }

    @staticmethod
    def _build_open_positions_from_plans(
        *,
        as_of: date,
        plans: list[TradePlan],
        runtime_mode: str,
        hold_days: int,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in plans:
            if str(p.status).upper() != "ACTIVE":
                continue
            out.append(
                {
                    "open_date": as_of.isoformat(),
                    "symbol": str(p.symbol),
                    "side": str(p.side.value),
                    "size_pct": float(p.size_pct),
                    "risk_pct": float(p.risk_pct),
                    "entry_price": float(p.entry_price),
                    "stop_price": float(p.stop_price),
                    "target_price": float(p.target_price),
                    "runtime_mode": str(runtime_mode or "base"),
                    "hold_days": int(max(1, hold_days)),
                    "status": "OPEN",
                }
            )
        return out

    def _symbol_exposure_snapshot(self) -> tuple[dict[str, float], dict[str, float], float]:
        if not self.ctx.sqlite_path.exists():
            return {}, {}, 0.0
        with closing(sqlite3.connect(self.ctx.sqlite_path)) as conn:
            try:
                pos = pd.read_sql_query(
                    "SELECT symbol, size_pct, status FROM latest_positions "
                    "WHERE date = (SELECT MAX(date) FROM latest_positions)",
                    conn,
                )
            except Exception:
                try:
                    pos = pd.read_sql_query("SELECT symbol, size_pct, status FROM latest_positions", conn)
                except Exception:
                    return {}, {}, 0.0
        if pos.empty:
            return {}, {}, 0.0

        pos = pos[pos["status"] == "ACTIVE"]
        by_symbol = pos.groupby("symbol")["size_pct"].sum().to_dict()
        theme_map = self._theme_map()
        by_theme: dict[str, float] = {}
        for sym, val in by_symbol.items():
            theme = theme_map.get(sym, "dynamic")
            by_theme.setdefault(theme, 0.0)
            by_theme[theme] += float(val)
        total = float(sum(by_symbol.values()))
        return by_symbol, by_theme, total

    def run_eod(self, as_of: date) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        paper_exec = self._settle_open_paper_positions(as_of=as_of, bars=bars)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
        live_params = self._load_live_params()
        runtime_params = self._resolve_runtime_params(regime=regime, live_params=live_params)

        signal_cfg = SignalEngineConfig(
            confidence_min=float(runtime_params["signal_confidence_min"]),
            convexity_min=float(runtime_params["convexity_min"]),
        )
        market_factor_state = self._market_factor_state(sentiment=ingest.sentiment, regime=regime, bars=bars)
        signals = scan_signals(
            bars=bars,
            regime=regime.consensus,
            cfg=signal_cfg,
            market_factor_state=market_factor_state,
        )

        recent_trades = self._load_recent_trades(300)
        win_rate, payoff = infer_edge_from_trades(recent_trades)
        guard = self._evaluate_guards(
            as_of=as_of,
            regime=regime,
            quality_passed=bool(ingest.quality.passed),
            sentiment=ingest.sentiment,
            news=ingest.news,
            recent_trades=recent_trades,
        )
        runtime_mode = str(runtime_params.get("mode", "base"))
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        risk_control = self._execution_risk_control(
            source_confidence_score=float(ingest.quality.source_confidence_score),
            mode_health=mode_health,
        )

        by_symbol, by_theme, used_exposure = self._symbol_exposure_snapshot()
        budget = self.risk_manager.build_budget(account_equity=1_000_000.0, used_exposure_pct=used_exposure)

        plans: list[TradePlan] = []
        theme_map = self._theme_map()

        if not guard.trade_blocked:
            for s in signals:
                theme = theme_map.get(s.symbol, "dynamic")
                plan = self.risk_manager.size_signal(
                    signal=s,
                    win_rate=win_rate,
                    payoff=payoff,
                    budget=budget,
                    symbol_exposure_pct=by_symbol.get(s.symbol, 0.0),
                    theme_exposure_pct=by_theme.get(theme, 0.0),
                    protection_mode=regime.protection_mode or guard.black_swan_trigger,
                    risk_multiplier=float(risk_control.get("risk_multiplier", 1.0)),
                )
                if plan:
                    plans.append(plan)
                    by_symbol[s.symbol] = by_symbol.get(s.symbol, 0.0) + plan.size_pct
                    by_theme[theme] = by_theme.get(theme, 0.0) + plan.size_pct
                    budget.used_exposure_pct += plan.size_pct

        next_events = [
            "08:40 盘前事件核查",
            "10:30 / 14:30 盘中信息校准",
            "20:30 盘后复盘和参数重估",
        ]
        avg_conf = float(pd.Series([float(s.confidence) for s in signals]).mean()) if signals else 0.0
        avg_conv = float(pd.Series([float(s.convexity_ratio) for s in signals]).mean()) if signals else 0.0
        mode_feedback = {
            "runtime_mode": runtime_mode,
            "runtime_params": {
                "signal_confidence_min": float(runtime_params["signal_confidence_min"]),
                "convexity_min": float(runtime_params["convexity_min"]),
                "hold_days": float(runtime_params["hold_days"]),
                "max_daily_trades": float(runtime_params["max_daily_trades"]),
            },
            "today": {
                "signals": int(len(signals)),
                "plans": int(len(plans)),
                "avg_confidence": avg_conf,
                "avg_convexity": avg_conv,
                "closed_trades": int((paper_exec.get("summary", {}) or {}).get("closed_count", 0)),
                "closed_pnl": float((paper_exec.get("summary", {}) or {}).get("closed_pnl", 0.0)),
            },
            "history": mode_history,
            "mode_health": mode_health,
            "risk_control": risk_control,
            "market_factor_state": market_factor_state,
        }

        daily_md = render_daily_briefing(
            as_of=as_of,
            regime=regime,
            signals=signals,
            plans=plans,
            quality=ingest.quality,
            black_swan_items=guard.black_swan_items,
            next_events=next_events,
            black_swan_score=guard.black_swan_score,
            non_trade_reasons=guard.non_trade_reasons,
            mode_feedback=mode_feedback,
        )

        daily_dir = self.ctx.output_dir / "daily"
        briefing_path = daily_dir / f"{as_of.isoformat()}_briefing.md"
        signals_path = daily_dir / f"{as_of.isoformat()}_signals.json"
        positions_path = daily_dir / f"{as_of.isoformat()}_positions.csv"
        mode_feedback_path = daily_dir / f"{as_of.isoformat()}_mode_feedback.json"

        write_markdown(briefing_path, daily_md)
        write_json(signals_path, [s.to_dict() for s in signals])
        write_json(mode_feedback_path, mode_feedback)
        plans_df = pd.DataFrame([p.to_dict() for p in plans])
        if plans_df.empty:
            plans_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "side",
                    "size_pct",
                    "risk_pct",
                    "entry_price",
                    "stop_price",
                    "target_price",
                    "hedge_leg",
                    "reason",
                    "status",
                ]
            )
        write_csv(positions_path, plans_df)

        signals_df = pd.DataFrame([s.to_dict() for s in signals])
        if signals_df.empty:
            signals_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "side",
                    "regime",
                    "position_score",
                    "structure_score",
                    "momentum_score",
                    "confidence",
                    "convexity_ratio",
                    "entry_price",
                    "stop_price",
                    "target_price",
                    "can_short",
                    "factor_exposure_score",
                    "factor_penalty",
                    "factor_flags",
                    "notes",
                ]
            )
        append_sqlite(self.ctx.sqlite_path, "signals", signals_df)
        append_sqlite(self.ctx.sqlite_path, "trade_plans", plans_df.assign(date=as_of.isoformat()))

        # latest positions snapshot for risk budgeting
        append_sqlite(
            self.ctx.sqlite_path,
            "latest_positions",
            plans_df.assign(date=as_of.isoformat()),
        )
        open_positions = self._build_open_positions_from_plans(
            as_of=as_of,
            plans=plans,
            runtime_mode=runtime_mode,
            hold_days=int(runtime_params["hold_days"]),
        )
        active_positions = list(paper_exec.get("remaining_positions", [])) + open_positions
        self._save_open_paper_positions(as_of=as_of, positions=active_positions)
        broker_snapshot_path = self._resolve_and_write_broker_snapshot(
            as_of=as_of,
            active_positions=active_positions,
            paper_exec_summary=(paper_exec.get("summary", {}) if isinstance(paper_exec.get("summary", {}), dict) else {}),
        )
        broker_snapshot_ref = str(broker_snapshot_path) if broker_snapshot_path is not None else ""
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="eod",
            run_id=as_of.isoformat(),
            artifacts={
                "briefing": str(briefing_path),
                "signals": str(signals_path),
                "positions": str(positions_path),
                "mode_feedback": str(mode_feedback_path),
                "broker_snapshot": broker_snapshot_ref,
            },
            metrics={
                "signals": int(len(signals)),
                "plans": int(len(plans)),
                "black_swan_score": float(guard.black_swan_score),
                "regime": str(regime.consensus.value),
                "runtime_mode": runtime_mode,
                "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
                "closed_trades": int((paper_exec.get("summary", {}) or {}).get("closed_count", 0)),
                "closed_pnl": float((paper_exec.get("summary", {}) or {}).get("closed_pnl", 0.0)),
                "open_positions": int(len(active_positions)),
            },
            checks={
                "quality_passed": bool(ingest.quality.passed),
                "trade_blocked": bool(guard.trade_blocked),
            },
        )

        return {
            "date": as_of.isoformat(),
            "regime": regime.to_dict(),
            "signals": len(signals),
            "plans": len(plans),
            "briefing": str(briefing_path),
            "manifest": str(manifest_path),
            "quality_passed": ingest.quality.passed,
            "source_confidence_score": float(ingest.quality.source_confidence_score),
            "low_confidence_source_ratio": float(ingest.quality.low_confidence_source_ratio),
            "low_confidence_sources": list(ingest.source_confidence.low_confidence_sources),
            "mode_feedback": str(mode_feedback_path),
            "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
            "non_trade_reasons": guard.non_trade_reasons,
            "black_swan_score": guard.black_swan_score,
            "runtime_mode": runtime_mode,
            "closed_trades": int((paper_exec.get("summary", {}) or {}).get("closed_count", 0)),
            "closed_pnl": float((paper_exec.get("summary", {}) or {}).get("closed_pnl", 0.0)),
            "open_positions": int(len(active_positions)),
            "broker_snapshot": broker_snapshot_ref,
        }

    def run_premarket(self, as_of: date) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
        market_factor_state = self._market_factor_state(sentiment=ingest.sentiment, regime=regime, bars=bars)
        live_params = self._load_live_params()
        runtime_params = self._resolve_runtime_params(regime=regime, live_params=live_params)
        runtime_mode = str(runtime_params.get("mode", "base"))
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        risk_control = self._execution_risk_control(
            source_confidence_score=float(ingest.quality.source_confidence_score),
            mode_health=mode_health,
        )
        recent_trades = self._load_recent_trades(300)
        guard = self._evaluate_guards(
            as_of=as_of,
            regime=regime,
            quality_passed=bool(ingest.quality.passed),
            sentiment=ingest.sentiment,
            news=ingest.news,
            recent_trades=recent_trades,
        )

        payload = {
            "date": as_of.isoformat(),
            "slot": "premarket",
            "quality": ingest.quality.to_dict(),
            "source_confidence": ingest.source_confidence.to_dict(),
            "regime": regime.to_dict(),
            "news_count": len(ingest.news),
            "black_swan_score": guard.black_swan_score,
            "major_event_window": guard.major_event_window,
            "runtime_mode": runtime_mode,
            "mode_health": mode_health,
            "risk_control": risk_control,
            "market_factor_state": market_factor_state,
            "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
            "protection_mode": regime.protection_mode or guard.black_swan_trigger or guard.major_event_window or (not ingest.quality.passed),
        }

        path = self.ctx.output_dir / "logs" / f"{as_of.isoformat()}_premarket.json"
        write_json(path, payload)
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="premarket",
            run_id=as_of.isoformat(),
            artifacts={"premarket_log": str(path)},
            metrics={
                "runtime_mode": runtime_mode,
                "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
                "source_confidence_score": float(ingest.quality.source_confidence_score),
                "black_swan_score": float(guard.black_swan_score),
                "news_count": int(len(ingest.news)),
            },
            checks={
                "quality_passed": bool(ingest.quality.passed),
                "major_event_window": bool(guard.major_event_window),
                "protection_mode": bool(payload.get("protection_mode", False)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def run_intraday_check(self, as_of: date, slot: str) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
        market_factor_state = self._market_factor_state(sentiment=ingest.sentiment, regime=regime, bars=bars)
        live_params = self._load_live_params()
        runtime_params = self._resolve_runtime_params(regime=regime, live_params=live_params)
        runtime_mode = str(runtime_params.get("mode", "base"))
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        risk_control = self._execution_risk_control(
            source_confidence_score=float(ingest.quality.source_confidence_score),
            mode_health=mode_health,
        )
        recent_trades = self._load_recent_trades(300)
        guard = self._evaluate_guards(
            as_of=as_of,
            regime=regime,
            quality_passed=bool(ingest.quality.passed),
            sentiment=ingest.sentiment,
            news=ingest.news,
            recent_trades=recent_trades,
        )

        if slot not in {"10:30", "14:30"}:
            slot = "10:30"

        payload = {
            "date": as_of.isoformat(),
            "slot": slot,
            "atr_z": regime.atr_z,
            "protection_mode": regime.protection_mode or guard.black_swan_trigger or guard.major_event_window or (not ingest.quality.passed),
            "quality_flags": ingest.quality.flags,
            "source_confidence_score": float(ingest.quality.source_confidence_score),
            "low_confidence_source_ratio": float(ingest.quality.low_confidence_source_ratio),
            "runtime_mode": runtime_mode,
            "mode_health": mode_health,
            "risk_control": risk_control,
            "market_factor_state": market_factor_state,
            "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
            "black_swan_score": guard.black_swan_score,
            "major_event_window": guard.major_event_window,
        }
        path = self.ctx.output_dir / "logs" / f"{as_of.isoformat()}_intraday_{slot.replace(':', '')}.json"
        write_json(path, payload)
        run_id = f"{as_of.isoformat()}_{slot.replace(':', '')}"
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="intraday_check",
            run_id=run_id,
            artifacts={"intraday_log": str(path)},
            metrics={
                "runtime_mode": runtime_mode,
                "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
                "source_confidence_score": float(ingest.quality.source_confidence_score),
                "black_swan_score": float(guard.black_swan_score),
                "atr_z": float(regime.atr_z),
            },
            checks={
                "quality_passed": bool(ingest.quality.passed),
                "major_event_window": bool(guard.major_event_window),
                "protection_mode": bool(payload.get("protection_mode", False)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def run_backtest(self, start: date, end: date) -> BacktestResult:
        symbols = self._core_symbols()
        # include dynamic candidates for wider stress
        symbols = expand_universe(symbols, pd.DataFrame(), int(self.settings.universe.get("max_dynamic_additions", 5)))
        bars, _ = self._run_ingestion_range(start=start, end=end, symbols=symbols)
        params = self._load_live_params()
        runtime_regime = self._regime_from_bars(as_of=end, bars=bars)
        runtime_params = self._resolve_runtime_params(regime=runtime_regime, live_params=params)
        signal_conf = float(runtime_params["signal_confidence_min"])
        convexity_min = float(runtime_params["convexity_min"])
        hold_days = self._clamp_int(runtime_params["hold_days"], 1, 20)
        max_daily_trades = self._clamp_int(runtime_params["max_daily_trades"], 1, 5)
        bt_cfg = BacktestConfig(
            signal_confidence_min=signal_conf,
            convexity_min=convexity_min,
            max_daily_trades=0 if signal_conf >= 90.0 else max_daily_trades,
            hold_days=hold_days,
        )

        result = run_walk_forward_backtest(
            bars=bars,
            start=start,
            end=end,
            trend_thr=float(self.settings.thresholds.get("hurst_trend", 0.6)),
            mean_thr=float(self.settings.thresholds.get("hurst_mean_revert", 0.4)),
            atr_extreme=float(self.settings.thresholds.get("atr_extreme", 2.0)),
            cfg_template=bt_cfg,
            train_years=3,
            valid_years=1,
            step_months=3,
        )

        out_path = self.ctx.output_dir / "artifacts" / f"backtest_{start.isoformat()}_{end.isoformat()}.json"
        write_json(out_path, result.to_dict())
        append_sqlite(self.ctx.sqlite_path, "backtest_runs", pd.DataFrame([result.to_dict()]))
        write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="backtest",
            run_id=f"{start.isoformat()}_{end.isoformat()}",
            artifacts={"result": str(out_path)},
            metrics={
                "annual_return": float(result.annual_return),
                "max_drawdown": float(result.max_drawdown),
                "win_rate": float(result.win_rate),
                "profit_factor": float(result.profit_factor),
                "trades": int(result.trades),
                "runtime_mode": str(runtime_params.get("mode", "base")),
            },
            checks={"violations": int(result.violations)},
        )
        return result

    @staticmethod
    def _default_mode_stress_windows() -> list[dict[str, str]]:
        return [
            {"name": "2015_crash", "start": "2015-01-01", "end": "2015-12-31"},
            {"name": "2020_pandemic", "start": "2020-01-01", "end": "2020-12-31"},
            {"name": "2022_geopolitical", "start": "2022-01-01", "end": "2022-12-31"},
            {"name": "extreme_gap", "start": "2024-01-01", "end": "2025-06-30"},
        ]

    def run_mode_stress_matrix(
        self,
        *,
        as_of: date,
        modes: list[str] | None = None,
        windows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        profiles = self._resolved_mode_profiles()
        default_modes = ["ultra_short", "swing", "long"]
        mode_candidates = modes if isinstance(modes, list) and modes else default_modes
        selected_modes: list[str] = []
        for raw in mode_candidates:
            m = str(raw).strip()
            if (not m) or (m in selected_modes):
                continue
            if m in profiles:
                selected_modes.append(m)
        if not selected_modes:
            selected_modes = ["swing"] if "swing" in profiles else [next(iter(profiles.keys()))]

        windows_in = windows if isinstance(windows, list) and windows else self._default_mode_stress_windows()
        parsed_windows: list[dict[str, Any]] = []
        for row in windows_in:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip() or "window"
            start = self._parse_iso_date(row.get("start"))
            end = self._parse_iso_date(row.get("end"))
            if start is None or end is None or end < start:
                continue
            if start > as_of:
                continue
            end_eff = min(end, as_of)
            parsed_windows.append(
                {
                    "name": name,
                    "start": start.isoformat(),
                    "end": end_eff.isoformat(),
                }
            )

        symbols = self._core_symbols()
        symbols = expand_universe(symbols, pd.DataFrame(), int(self.settings.universe.get("max_dynamic_additions", 5)))
        trend_thr = float(self.settings.thresholds.get("hurst_trend", 0.6))
        mean_thr = float(self.settings.thresholds.get("hurst_mean_revert", 0.4))
        atr_extreme = float(self.settings.thresholds.get("atr_extreme", 2.0))

        matrix_rows: list[dict[str, Any]] = []
        for w in parsed_windows:
            w_name = str(w.get("name", "window"))
            w_start = self._parse_iso_date(w.get("start"))
            w_end = self._parse_iso_date(w.get("end"))
            if w_start is None or w_end is None:
                continue
            bars, _ = self._run_ingestion_range(start=w_start, end=w_end, symbols=symbols)
            bars_empty = bool(bars.empty)
            for mode in selected_modes:
                profile = profiles.get(mode, profiles.get("swing", self._default_mode_profiles().get("swing", {})))
                cfg = BacktestConfig(
                    signal_confidence_min=float(profile.get("signal_confidence_min", 60.0)),
                    convexity_min=float(profile.get("convexity_min", 2.0)),
                    max_daily_trades=self._clamp_int(profile.get("max_daily_trades", 2.0), 1, 5),
                    hold_days=self._clamp_int(profile.get("hold_days", 5.0), 1, 20),
                )
                if bars_empty:
                    result = BacktestResult(
                        start=w_start,
                        end=w_end,
                        total_return=0.0,
                        annual_return=0.0,
                        max_drawdown=0.0,
                        win_rate=0.0,
                        profit_factor=0.0,
                        expectancy=0.0,
                        trades=0,
                        violations=0,
                        positive_window_ratio=0.0,
                        equity_curve=[],
                        by_asset={},
                    )
                    status = "no_data"
                else:
                    result = run_walk_forward_backtest(
                        bars=bars,
                        start=w_start,
                        end=w_end,
                        trend_thr=trend_thr,
                        mean_thr=mean_thr,
                        atr_extreme=atr_extreme,
                        cfg_template=cfg,
                        train_years=3,
                        valid_years=1,
                        step_months=3,
                    )
                    status = "ok"

                matrix_rows.append(
                    {
                        "mode": mode,
                        "window": w_name,
                        "start": w_start.isoformat(),
                        "end": w_end.isoformat(),
                        "status": status,
                        "total_return": float(result.total_return),
                        "annual_return": float(result.annual_return),
                        "max_drawdown": float(result.max_drawdown),
                        "win_rate": float(result.win_rate),
                        "profit_factor": float(result.profit_factor),
                        "expectancy": float(result.expectancy),
                        "trades": int(result.trades),
                        "violations": int(result.violations),
                        "positive_window_ratio": float(result.positive_window_ratio),
                    }
                )

        summary_rows: list[dict[str, Any]] = []
        for mode in selected_modes:
            rows = [x for x in matrix_rows if str(x.get("mode", "")) == mode and str(x.get("status", "")) == "ok"]
            if not rows:
                summary_rows.append(
                    {
                        "mode": mode,
                        "windows": 0,
                        "avg_annual_return": 0.0,
                        "worst_drawdown": 0.0,
                        "avg_win_rate": 0.0,
                        "avg_profit_factor": 0.0,
                        "avg_positive_window_ratio": 0.0,
                        "total_violations": 0,
                        "robustness_score": -1.0,
                    }
                )
                continue
            n = float(len(rows))
            avg_annual = float(sum(float(r.get("annual_return", 0.0)) for r in rows) / n)
            worst_drawdown = float(max(float(r.get("max_drawdown", 0.0)) for r in rows))
            avg_win = float(sum(float(r.get("win_rate", 0.0)) for r in rows) / n)
            avg_pf = float(sum(float(r.get("profit_factor", 0.0)) for r in rows) / n)
            avg_pwr = float(sum(float(r.get("positive_window_ratio", 0.0)) for r in rows) / n)
            total_violations = int(sum(int(r.get("violations", 0)) for r in rows))
            robustness_score = float(
                avg_annual
                - worst_drawdown
                + 0.10 * (avg_pf - 1.0)
                + 0.05 * (avg_pwr - 0.5)
                - 0.02 * float(total_violations)
            )
            summary_rows.append(
                {
                    "mode": mode,
                    "windows": int(len(rows)),
                    "avg_annual_return": avg_annual,
                    "worst_drawdown": worst_drawdown,
                    "avg_win_rate": avg_win,
                    "avg_profit_factor": avg_pf,
                    "avg_positive_window_ratio": avg_pwr,
                    "total_violations": total_violations,
                    "robustness_score": robustness_score,
                }
            )

        summary_rows.sort(key=lambda x: float(x.get("robustness_score", -1e9)), reverse=True)
        best_mode = str(summary_rows[0]["mode"]) if summary_rows else "N/A"

        payload: dict[str, Any] = {
            "date": as_of.isoformat(),
            "mode_count": int(len(selected_modes)),
            "window_count": int(len(parsed_windows)),
            "modes": selected_modes,
            "windows": parsed_windows,
            "matrix": matrix_rows,
            "mode_summary": summary_rows,
            "best_mode": best_mode,
        }

        review_dir = self.ctx.output_dir / "review"
        json_path = review_dir / f"{as_of.isoformat()}_mode_stress_matrix.json"
        md_path = review_dir / f"{as_of.isoformat()}_mode_stress_matrix.md"
        write_json(json_path, payload)
        write_markdown(md_path, render_mode_stress_matrix(as_of, payload))
        append_sqlite(
            self.ctx.sqlite_path,
            "mode_stress_matrix_runs",
            pd.DataFrame(
                [
                    {
                        "date": as_of.isoformat(),
                        "mode_count": int(len(selected_modes)),
                        "window_count": int(len(parsed_windows)),
                        "best_mode": best_mode,
                    }
                ]
            ),
        )
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="mode_stress_matrix",
            run_id=as_of.isoformat(),
            artifacts={
                "matrix_json": str(json_path),
                "matrix_md": str(md_path),
            },
            metrics={
                "mode_count": int(len(selected_modes)),
                "window_count": int(len(parsed_windows)),
                "best_mode": best_mode,
            },
            checks={"matrix_rows": int(len(matrix_rows))},
        )
        payload["paths"] = {"json": str(json_path), "md": str(md_path)}
        payload["manifest"] = str(manifest_path)
        return payload

    def run_research_backtest(
        self,
        *,
        start: date,
        end: date,
        hours_budget: float = 10.0,
        max_symbols: int = 120,
        report_symbol_cap: int = 40,
        workers: int = 8,
        max_trials_per_mode: int = 500,
        seed: int = 42,
        modes: list[str] | None = None,
        review_days: int = 5,
    ) -> dict[str, Any]:
        summary = run_research_pipeline(
            output_root=self.ctx.output_dir,
            core_symbols=self._core_symbols(),
            start=start,
            end=end,
            hours_budget=hours_budget,
            max_symbols=max_symbols,
            report_symbol_cap=report_symbol_cap,
            workers=workers,
            max_trials_per_mode=max_trials_per_mode,
            seed=seed,
            modes=modes,
            review_days=review_days,
        )
        payload = summary.to_dict()
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="research_backtest",
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            artifacts={
                "summary": str(Path(summary.output_dir) / "summary.json"),
                "report": str(Path(summary.output_dir) / "report.md"),
                "best_params": str(Path(summary.output_dir) / "best_params.yaml"),
            },
            metrics={
                "hours_budget": float(payload.get("hours_budget", 0.0)),
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                "universe_count": int(payload.get("universe_count", 0)),
                "bars_rows": int(payload.get("bars_rows", 0)),
            },
            checks={
                "strict_cutoff_enforced": bool(payload.get("data_fetch_stats", {}).get("strict_cutoff_enforced", False)),
                "mode_count": int(len(payload.get("mode_summaries", []))),
            },
            metadata={
                "cutoff_date": str(payload.get("cutoff_date", "")),
                "cutoff_ts": str(payload.get("cutoff_ts", "")),
                "bar_max_ts": str(payload.get("bar_max_ts", "")),
                "news_max_ts": str(payload.get("news_max_ts", "")),
                "report_max_ts": str(payload.get("report_max_ts", "")),
                "review_bar_max_ts": str(payload.get("review_bar_max_ts", "")),
                "review_news_max_ts": str(payload.get("review_news_max_ts", "")),
                "review_report_max_ts": str(payload.get("review_report_max_ts", "")),
                "review_days": int(payload.get("review_days", 0)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def run_strategy_lab(
        self,
        *,
        start: date,
        end: date,
        max_symbols: int = 120,
        report_symbol_cap: int = 40,
        workers: int = 8,
        review_days: int = 5,
        candidate_count: int = 10,
    ) -> dict[str, Any]:
        summary = run_strategy_lab_pipeline(
            output_root=self.ctx.output_dir,
            core_symbols=self._core_symbols(),
            start=start,
            end=end,
            max_symbols=max_symbols,
            report_symbol_cap=report_symbol_cap,
            workers=workers,
            review_days=review_days,
            candidate_count=candidate_count,
        )
        payload = summary.to_dict()
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="strategy_lab",
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            artifacts={
                "summary": str(Path(summary.output_dir) / "summary.json"),
                "report": str(Path(summary.output_dir) / "report.md"),
                "best_strategy": str(Path(summary.output_dir) / "best_strategy.yaml"),
            },
            metrics={
                "universe_count": int(payload.get("universe_count", 0)),
                "bars_rows": int(payload.get("bars_rows", 0)),
                "candidate_count": int(len(payload.get("candidates", []))),
            },
            checks={
                "has_best_candidate": bool(payload.get("best_candidate", {})),
                "strict_cutoff_enforced": bool(payload.get("data_fetch_stats", {}).get("strict_cutoff_enforced", False)),
            },
            metadata={
                "cutoff_date": str(payload.get("cutoff_date", "")),
                "cutoff_ts": str(payload.get("cutoff_ts", "")),
                "bar_max_ts": str(payload.get("bar_max_ts", "")),
                "news_max_ts": str(payload.get("news_max_ts", "")),
                "report_max_ts": str(payload.get("report_max_ts", "")),
                "review_bar_max_ts": str(payload.get("review_bar_max_ts", "")),
                "review_news_max_ts": str(payload.get("review_news_max_ts", "")),
                "review_report_max_ts": str(payload.get("review_report_max_ts", "")),
                "review_days": int(payload.get("review_days", 0)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def _load_live_params(self) -> dict[str, float]:
        p = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        if not p.exists():
            return {
                "win_rate": 0.45,
                "payoff": 2.0,
                "signal_confidence_min": float(self.settings.thresholds.get("signal_confidence_min", 60.0)),
                "convexity_min": float(self.settings.thresholds.get("convexity_min", 3.0)),
                "hold_days": 5.0,
                "max_daily_trades": 2.0,
            }
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    def _backup_live_params(self, as_of: date) -> str:
        p = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        if not p.exists():
            return "initial_seed"
        backup = self.ctx.output_dir / "artifacts" / f"params_live_backup_{as_of.isoformat()}.yaml"
        write_markdown(backup, p.read_text(encoding="utf-8"))
        return str(backup)

    def _resolve_review_runtime_mode(self, *, start: date, as_of: date) -> str:
        manifest_path = self.ctx.output_dir / "artifacts" / "manifests" / f"backtest_{start.isoformat()}_{as_of.isoformat()}.json"
        payload = self._load_json_safely(manifest_path)
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
        mode = str(metrics.get("runtime_mode", "base")).strip() or "base"
        return mode

    def _evaluate_mode_health(self, *, runtime_mode: str, mode_history: dict[str, Any]) -> dict[str, Any]:
        validation = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        modes = mode_history.get("modes", {}) if isinstance(mode_history.get("modes", {}), dict) else {}
        stats = modes.get(runtime_mode, {})
        if not isinstance(stats, dict):
            stats = {}

        samples = int(stats.get("samples", 0))
        min_samples = max(1, int(validation.get("mode_health_min_samples", 1)))
        min_profit_factor = float(validation.get("mode_health_min_profit_factor", 1.0))
        min_win_rate = float(validation.get("mode_health_min_win_rate", 0.40))
        max_drawdown = float(validation.get("mode_health_max_drawdown_max", validation.get("max_drawdown_max", 0.18)))
        max_violations = max(0, int(validation.get("mode_health_max_violations", 0)))

        if samples < min_samples:
            return {
                "passed": True,
                "active": False,
                "runtime_mode": runtime_mode,
                "reason": "insufficient_samples",
                "samples": samples,
                "min_samples": min_samples,
            }

        pf = float(stats.get("avg_profit_factor", 0.0))
        wr = float(stats.get("avg_win_rate", 0.0))
        dd = float(stats.get("worst_drawdown", 1.0))
        viol = int(stats.get("total_violations", 0))
        failed_rules: list[str] = []
        if pf < min_profit_factor:
            failed_rules.append("profit_factor")
        if wr < min_win_rate:
            failed_rules.append("win_rate")
        if dd > max_drawdown:
            failed_rules.append("max_drawdown")
        if viol > max_violations:
            failed_rules.append("violations")

        return {
            "passed": len(failed_rules) == 0,
            "active": True,
            "runtime_mode": runtime_mode,
            "samples": samples,
            "thresholds": {
                "min_profit_factor": min_profit_factor,
                "min_win_rate": min_win_rate,
                "max_drawdown_max": max_drawdown,
                "max_violations": max_violations,
                "min_samples": min_samples,
            },
            "stats": {
                "avg_profit_factor": pf,
                "avg_win_rate": wr,
                "worst_drawdown": dd,
                "total_violations": viol,
            },
            "failed_rules": failed_rules,
        }

    def _apply_mode_health_guard(
        self,
        *,
        review: ReviewDelta,
        current_params: dict[str, float],
        mode_health: dict[str, Any],
    ) -> None:
        if bool(mode_health.get("passed", True)):
            return
        if "MODE_HEALTH_DEGRADED" not in review.defects:
            review.defects.append("MODE_HEALTH_DEGRADED")
        review.pass_gate = False

        conf = float(review.parameter_changes.get("signal_confidence_min", current_params.get("signal_confidence_min", 60.0)))
        tightened_conf = self._clamp_float(conf + 3.0, 50.0, 90.0)
        review.parameter_changes["signal_confidence_min"] = tightened_conf
        review.change_reasons["signal_confidence_min"] = (
            str(review.change_reasons.get("signal_confidence_min", "")).strip()
            + "；模式健康劣化触发保护收敛（提升信号置信阈值）"
        ).strip("；")

        trades = self._clamp_int(
            review.parameter_changes.get("max_daily_trades", current_params.get("max_daily_trades", 2.0)),
            1,
            5,
        )
        reduced_trades = float(max(1, trades - 1))
        review.parameter_changes["max_daily_trades"] = reduced_trades
        review.change_reasons["max_daily_trades"] = "模式健康劣化触发保护收敛（下调单日交易次数）"

        hold_days = self._clamp_int(
            review.parameter_changes.get("hold_days", current_params.get("hold_days", 5.0)),
            1,
            20,
        )
        shorter_hold = float(max(1, hold_days - 2))
        review.parameter_changes["hold_days"] = shorter_hold
        review.change_reasons["hold_days"] = "模式健康劣化触发保护收敛（缩短持有期暴露）"

        failed_rules = mode_health.get("failed_rules", [])
        review.notes.append(
            "mode_health_guard="
            + f"runtime_mode={mode_health.get('runtime_mode', 'base')}; "
            + f"failed_rules={failed_rules}; samples={mode_health.get('samples', 0)}"
        )

    def _apply_mode_adaptive_update(
        self,
        *,
        review: ReviewDelta,
        current_params: dict[str, float],
        runtime_mode: str,
        mode_history: dict[str, Any],
    ) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("mode_adaptive_update_enabled", True))
        summary: dict[str, Any] = {
            "enabled": enabled,
            "applied": False,
            "runtime_mode": runtime_mode,
        }
        if not enabled:
            summary["reason"] = "disabled"
            return summary

        modes = mode_history.get("modes", {}) if isinstance(mode_history.get("modes", {}), dict) else {}
        stats = modes.get(runtime_mode, {})
        if not isinstance(stats, dict) or not stats:
            summary["reason"] = "missing_mode_stats"
            return summary

        min_samples = max(1, int(val.get("mode_adaptive_update_min_samples", 3)))
        samples = int(stats.get("samples", 0))
        summary["samples"] = samples
        summary["min_samples"] = min_samples
        if samples < min_samples:
            summary["reason"] = "insufficient_samples"
            return summary

        step = self._clamp_float(float(val.get("mode_adaptive_update_step", 0.08)), 0.01, 0.50)
        good_pf = float(val.get("mode_adaptive_good_profit_factor", 1.25))
        bad_pf = float(val.get("mode_adaptive_bad_profit_factor", 1.00))
        good_wr = self._clamp_float(float(val.get("mode_adaptive_good_win_rate", 0.50)), 0.0, 1.0)
        bad_wr = self._clamp_float(float(val.get("mode_adaptive_bad_win_rate", 0.42)), 0.0, 1.0)
        default_bad_dd = float(val.get("max_drawdown_max", 0.18))
        good_dd = self._clamp_float(float(val.get("mode_adaptive_good_drawdown_max", default_bad_dd * 0.67)), 0.0, 1.0)
        bad_dd = self._clamp_float(float(val.get("mode_adaptive_bad_drawdown_max", default_bad_dd)), 0.0, 1.0)

        pf = float(stats.get("avg_profit_factor", 0.0))
        wr = float(stats.get("avg_win_rate", 0.0))
        dd = float(stats.get("worst_drawdown", 1.0))
        viol = int(stats.get("total_violations", 0))
        summary["stats"] = {
            "avg_profit_factor": pf,
            "avg_win_rate": wr,
            "worst_drawdown": dd,
            "total_violations": viol,
        }

        direction = "neutral"
        if pf >= good_pf and wr >= good_wr and dd <= good_dd and viol == 0:
            direction = "expand"
        elif pf < bad_pf or wr < bad_wr or dd > bad_dd or viol > 0:
            direction = "tighten"
        summary["direction"] = direction
        summary["thresholds"] = {
            "good_profit_factor": good_pf,
            "bad_profit_factor": bad_pf,
            "good_win_rate": good_wr,
            "bad_win_rate": bad_wr,
            "good_drawdown_max": good_dd,
            "bad_drawdown_max": bad_dd,
            "step": step,
        }
        if direction == "neutral":
            summary["reason"] = "no_regime_shift_signal"
            return summary

        profiles = self._resolved_mode_profiles()
        profile = profiles.get(runtime_mode, profiles.get("swing", self._default_mode_profiles()["swing"]))

        base_conf = float(review.parameter_changes.get("signal_confidence_min", current_params.get("signal_confidence_min", 60.0)))
        base_conv = float(review.parameter_changes.get("convexity_min", current_params.get("convexity_min", self.settings.thresholds.get("convexity_min", 3.0))))
        base_hold = float(review.parameter_changes.get("hold_days", current_params.get("hold_days", 5.0)))
        base_trades = float(review.parameter_changes.get("max_daily_trades", current_params.get("max_daily_trades", 2.0)))

        if direction == "expand":
            evidence_conf = min(base_conf - 1.5, float(profile.get("signal_confidence_min", base_conf)))
            evidence_conv = min(base_conv - 0.10, float(profile.get("convexity_min", base_conv)))
            evidence_hold = max(base_hold + 1.0, float(profile.get("hold_days", base_hold)))
            evidence_trades = max(base_trades + 1.0, float(profile.get("max_daily_trades", base_trades)))
        else:
            evidence_conf = max(base_conf + 3.0, float(profile.get("signal_confidence_min", base_conf)))
            evidence_conv = max(base_conv + 0.25, float(profile.get("convexity_min", base_conv)))
            evidence_hold = min(base_hold - 1.0, float(profile.get("hold_days", base_hold)))
            evidence_trades = min(base_trades - 1.0, float(profile.get("max_daily_trades", base_trades)))

        new_conf = bounded_bayesian_update(base_conf, evidence_conf, 30.0, 90.0, step=step)
        new_conv = bounded_bayesian_update(base_conv, evidence_conv, 1.0, 4.0, step=step)
        new_hold = float(self._clamp_int(bounded_bayesian_update(base_hold, evidence_hold, 1.0, 20.0, step=step), 1, 20))
        new_trades = float(self._clamp_int(bounded_bayesian_update(base_trades, evidence_trades, 1.0, 5.0, step=step), 1, 5))

        review.parameter_changes["signal_confidence_min"] = float(new_conf)
        review.parameter_changes["convexity_min"] = float(new_conv)
        review.parameter_changes["hold_days"] = float(new_hold)
        review.parameter_changes["max_daily_trades"] = float(new_trades)
        reason = f"模式自适应更新（mode={runtime_mode}, direction={direction}, step={step:.2f}）"
        for key in ("signal_confidence_min", "convexity_min", "hold_days", "max_daily_trades"):
            prev_reason = str(review.change_reasons.get(key, "")).strip()
            review.change_reasons[key] = (prev_reason + "；" + reason).strip("；")
        review.notes.append(
            "mode_adaptive_update="
            + f"mode={runtime_mode}; direction={direction}; samples={samples}; "
            + f"pf={pf:.3f}; wr={wr:.2%}; dd={dd:.2%}; viol={viol}; step={step:.2f}"
        )

        summary["applied"] = True
        summary["updates"] = {
            "signal_confidence_min": float(new_conf),
            "convexity_min": float(new_conv),
            "hold_days": float(new_hold),
            "max_daily_trades": float(new_trades),
        }
        return summary

    def _execution_risk_control(
        self,
        *,
        source_confidence_score: float,
        mode_health: dict[str, Any],
    ) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        min_mult = self._clamp_float(float(val.get("execution_min_risk_multiplier", 0.2)), 0.0, 1.0)
        source_floor = self._clamp_float(float(val.get("source_confidence_floor_risk_multiplier", 0.35)), min_mult, 1.0)
        mode_degraded_mult = self._clamp_float(float(val.get("mode_health_risk_multiplier", 0.5)), min_mult, 1.0)
        mode_unknown_mult = self._clamp_float(float(val.get("mode_health_insufficient_sample_risk_multiplier", 0.85)), min_mult, 1.0)
        source_min = self._clamp_float(float(val.get("source_confidence_min", 0.75)), 0.01, 1.0)

        score = self._clamp_float(float(source_confidence_score), 0.0, 1.0)
        source_mult = 1.0
        if score < source_min:
            source_mult = source_floor + (1.0 - source_floor) * (score / source_min)
        source_mult = self._clamp_float(source_mult, min_mult, 1.0)

        mode_mult = 1.0
        mode_reason = "healthy"
        if not bool(mode_health.get("passed", True)):
            mode_mult = mode_degraded_mult
            mode_reason = "degraded"
        elif not bool(mode_health.get("active", False)):
            mode_mult = mode_unknown_mult
            mode_reason = "insufficient_samples"

        final_mult = self._clamp_float(source_mult * mode_mult, min_mult, 1.0)
        return {
            "risk_multiplier": final_mult,
            "source_multiplier": source_mult,
            "mode_multiplier": mode_mult,
            "mode_reason": mode_reason,
            "source_confidence_score": score,
        }

    @staticmethod
    def _rank_corr(a: pd.Series, b: pd.Series) -> float:
        x = pd.to_numeric(a, errors="coerce")
        y = pd.to_numeric(b, errors="coerce")
        frame = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(frame) < 4:
            return 0.0
        corr = frame["x"].rank().corr(frame["y"].rank())
        if corr is None or not np.isfinite(corr):
            return 0.0
        return float(corr)

    def _cross_section_style_state(self, bars: pd.DataFrame) -> dict[str, float]:
        neutral = {
            "momentum_preference": 0.55,
            "value_preference": 0.50,
            "size_preference": 0.50,
            "dividend_preference": 0.55,
            "crowding_aversion": 0.50,
            "style_signal_strength": 0.0,
            "style_sample_size": 0.0,
        }
        if bars.empty:
            return dict(neutral)

        work = bars.copy()
        work["ts"] = pd.to_datetime(work["ts"], errors="coerce")
        work = work.dropna(subset=["ts"]).sort_values(["symbol", "ts"])
        if work.empty:
            return dict(neutral)

        rows: list[dict[str, float]] = []
        for _, g in work.groupby("symbol"):
            g = g.tail(160).copy()
            if len(g) < 25:
                continue
            close = pd.to_numeric(g["close"], errors="coerce")
            volume = pd.to_numeric(g["volume"], errors="coerce")
            if close.dropna().empty:
                continue

            close_last = float(close.iloc[-1])
            close_20 = float(close.iloc[-21]) if len(close) >= 21 else float(close.iloc[0])
            close_60 = float(close.iloc[-61]) if len(close) >= 61 else float(close.iloc[0])
            ret20 = 0.0 if abs(close_20) <= 1e-9 else (close_last / close_20 - 1.0)
            ret60 = 0.0 if abs(close_60) <= 1e-9 else (close_last / close_60 - 1.0)
            ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())
            turnover20 = float((close * volume).tail(20).mean()) if not volume.dropna().empty else 1.0
            vol20 = float(close.pct_change().tail(20).std(ddof=0))

            last = g.iloc[-1]
            market_cap = pd.to_numeric(pd.Series([last.get("market_cap")]), errors="coerce").iloc[0]
            pe_ttm = pd.to_numeric(pd.Series([last.get("pe_ttm")]), errors="coerce").iloc[0]
            pb = pd.to_numeric(pd.Series([last.get("pb")]), errors="coerce").iloc[0]
            dividend_yield = pd.to_numeric(pd.Series([last.get("dividend_yield")]), errors="coerce").iloc[0]

            if np.isfinite(market_cap) and float(market_cap) > 0:
                size_score = -float(np.log10(max(1.0, float(market_cap))))
            else:
                size_score = -float(np.log10(max(1.0, turnover20)))

            if np.isfinite(pe_ttm) and float(pe_ttm) > 0:
                value_score = -float(np.log(max(1e-9, float(pe_ttm))))
            elif np.isfinite(pb) and float(pb) > 0:
                value_score = -float(np.log(max(1e-9, float(pb))))
            else:
                value_score = -(close_last / max(ma60, 1e-9) - 1.0)

            if np.isfinite(dividend_yield) and float(dividend_yield) >= 0:
                dividend_score = float(dividend_yield)
            else:
                dividend_score = -vol20

            crowd_score = float(np.log10(max(1.0, turnover20))) + 5.0 * vol20
            rows.append(
                {
                    "ret20": float(ret20),
                    "momentum_score": float(ret60),
                    "size_score": float(size_score),
                    "value_score": float(value_score),
                    "dividend_score": float(dividend_score),
                    "crowd_score": float(crowd_score),
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty or len(frame) < 4:
            out = dict(neutral)
            out["style_sample_size"] = float(len(frame))
            return out

        corr_mom = self._rank_corr(frame["momentum_score"], frame["ret20"])
        corr_size = self._rank_corr(frame["size_score"], frame["ret20"])
        corr_value = self._rank_corr(frame["value_score"], frame["ret20"])
        corr_div = self._rank_corr(frame["dividend_score"], frame["ret20"])
        corr_crowd = self._rank_corr(frame["crowd_score"], frame["ret20"])

        out = {
            "momentum_preference": self._clamp_float(0.55 + 0.45 * corr_mom, 0.0, 1.5),
            "value_preference": self._clamp_float(0.50 + 0.50 * corr_value, 0.0, 1.5),
            "size_preference": self._clamp_float(0.50 + 0.50 * corr_size, 0.0, 1.5),
            "dividend_preference": self._clamp_float(0.55 + 0.45 * corr_div, 0.0, 1.5),
            "crowding_aversion": self._clamp_float(0.50 - 0.50 * corr_crowd, 0.0, 1.5),
            "style_signal_strength": self._clamp_float(
                float(np.mean([abs(corr_mom), abs(corr_size), abs(corr_value), abs(corr_div), abs(corr_crowd)])),
                0.0,
                1.0,
            ),
            "style_sample_size": float(len(frame)),
        }
        return out

    @staticmethod
    def _style_feature_panel(bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return pd.DataFrame(
                columns=[
                    "ts",
                    "symbol",
                    "fwd_ret1",
                    "momentum_score",
                    "value_score",
                    "size_score",
                    "dividend_score",
                    "crowd_score",
                ]
            )

        work = bars.copy()
        work["ts"] = pd.to_datetime(work["ts"], errors="coerce")
        work = work.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
        if work.empty:
            return pd.DataFrame(columns=["ts", "symbol", "fwd_ret1", "momentum_score", "value_score", "size_score", "dividend_score", "crowd_score"])

        work["close"] = pd.to_numeric(work["close"], errors="coerce")
        work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
        work = work.dropna(subset=["close"])
        if work.empty:
            return pd.DataFrame(columns=["ts", "symbol", "fwd_ret1", "momentum_score", "value_score", "size_score", "dividend_score", "crowd_score"])

        by_symbol = work.groupby("symbol", group_keys=False)
        work["ret1"] = by_symbol["close"].pct_change()
        work["ret60"] = by_symbol["close"].pct_change(60)
        work["ma60"] = by_symbol["close"].transform(lambda s: s.rolling(60, min_periods=20).mean())
        work["vol20"] = by_symbol["ret1"].transform(lambda s: s.rolling(20, min_periods=10).std(ddof=0))

        turnover = work["close"] * work["volume"].fillna(0.0)
        work["turnover20"] = turnover.groupby(work["symbol"]).transform(lambda s: s.rolling(20, min_periods=10).mean())
        work["fwd_ret1"] = by_symbol["close"].shift(-1) / work["close"] - 1.0

        if "market_cap" in work.columns:
            mc = pd.to_numeric(work["market_cap"], errors="coerce")
        else:
            mc = pd.Series(np.nan, index=work.index, dtype=float)
        if "pe_ttm" in work.columns:
            pe = pd.to_numeric(work["pe_ttm"], errors="coerce")
        else:
            pe = pd.Series(np.nan, index=work.index, dtype=float)
        if "pb" in work.columns:
            pb = pd.to_numeric(work["pb"], errors="coerce")
        else:
            pb = pd.Series(np.nan, index=work.index, dtype=float)
        if "dividend_yield" in work.columns:
            divy = pd.to_numeric(work["dividend_yield"], errors="coerce")
        else:
            divy = pd.Series(np.nan, index=work.index, dtype=float)

        safe_turnover = work["turnover20"].replace(0.0, np.nan)
        work["size_score"] = np.where(
            (mc > 0) & np.isfinite(mc),
            -np.log10(np.maximum(1.0, mc)),
            -np.log10(np.maximum(1.0, safe_turnover.fillna(1.0))),
        )
        fallback_value = -(work["close"] / work["ma60"].replace(0.0, np.nan) - 1.0)
        work["value_score"] = np.where(
            (pe > 0) & np.isfinite(pe),
            -np.log(np.maximum(pe, 1e-9)),
            np.where((pb > 0) & np.isfinite(pb), -np.log(np.maximum(pb, 1e-9)), fallback_value),
        )
        work["momentum_score"] = work["ret60"]
        work["dividend_score"] = np.where((divy >= 0) & np.isfinite(divy), divy, -work["vol20"])
        work["crowd_score"] = np.log10(np.maximum(1.0, safe_turnover.fillna(1.0))) + 5.0 * work["vol20"].fillna(0.0)

        keep_cols = [
            "ts",
            "symbol",
            "fwd_ret1",
            "momentum_score",
            "value_score",
            "size_score",
            "dividend_score",
            "crowd_score",
        ]
        panel = work[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd_ret1"])
        return panel

    @staticmethod
    def _bucket_spread(day_df: pd.DataFrame, score_col: str, ret_col: str = "fwd_ret1") -> float | None:
        if day_df.empty or score_col not in day_df.columns or ret_col not in day_df.columns:
            return None
        frame = day_df[[score_col, ret_col]].dropna()
        if len(frame) < 6:
            return None
        q_lo = float(frame[score_col].quantile(0.25))
        q_hi = float(frame[score_col].quantile(0.75))
        if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
            return None
        lo = frame[frame[score_col] <= q_lo][ret_col]
        hi = frame[frame[score_col] >= q_hi][ret_col]
        if lo.empty or hi.empty:
            return None
        return float(hi.mean() - lo.mean())

    def _review_style_diagnostics(self, *, start: date, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        lookback_days = max(30, int(val.get("style_attribution_lookback_days", 180)))
        drift_window = max(5, int(val.get("style_drift_window_days", 20)))
        min_sample_days = max(8, int(val.get("style_drift_min_sample_days", 20)))
        drift_gap_max = self._clamp_float(float(val.get("style_drift_gap_max", 0.010)), 0.001, 0.10)
        block_on_alert = bool(val.get("style_drift_block_on_alert", False))

        fetch_start = max(start, as_of - timedelta(days=max(60, lookback_days * 2)))
        bars, _ = self._run_ingestion_range(start=fetch_start, end=as_of, symbols=self._core_symbols())
        panel = self._style_feature_panel(bars)
        if panel.empty:
            return {
                "active": False,
                "reason": "insufficient_panel",
                "lookback_days": int(lookback_days),
                "sample_days": 0,
                "drift_window_days": int(drift_window),
                "drift_gap_max": float(drift_gap_max),
                "block_on_alert": bool(block_on_alert),
            }

        panel = panel[panel["ts"].dt.date <= as_of].copy()
        if panel.empty:
            return {
                "active": False,
                "reason": "empty_after_cutoff",
                "lookback_days": int(lookback_days),
                "sample_days": 0,
                "drift_window_days": int(drift_window),
                "drift_gap_max": float(drift_gap_max),
                "block_on_alert": bool(block_on_alert),
            }

        style_map = {
            "momentum": "momentum_score",
            "value": "value_score",
            "size": "size_score",
            "dividend": "dividend_score",
        }
        daily = panel.groupby(panel["ts"].dt.date)
        spreads_by_style: dict[str, list[float]] = {k: [] for k in style_map}
        for _, day_df in daily:
            for style, col in style_map.items():
                spread = self._bucket_spread(day_df, score_col=col, ret_col="fwd_ret1")
                if spread is not None and np.isfinite(spread):
                    spreads_by_style[style].append(float(spread))

        summary: dict[str, dict[str, float]] = {}
        drift_by_style: dict[str, float] = {}
        for style, arr in spreads_by_style.items():
            if not arr:
                summary[style] = {"avg_spread": 0.0, "win_rate": 0.0, "sample_days": 0.0, "recent_avg": 0.0, "prev_avg": 0.0}
                drift_by_style[style] = 0.0
                continue
            ser = pd.Series(arr, dtype=float)
            avg_spread = float(ser.mean())
            win_rate = float((ser > 0.0).mean())
            recent = ser.tail(drift_window)
            prev = ser.iloc[max(0, len(ser) - drift_window * 2) : max(0, len(ser) - drift_window)]
            if prev.empty:
                prev = ser.head(max(1, len(ser) // 2))
            recent_avg = float(recent.mean()) if not recent.empty else 0.0
            prev_avg = float(prev.mean()) if not prev.empty else 0.0
            drift_gap = abs(recent_avg - prev_avg)
            drift_by_style[style] = float(drift_gap)
            summary[style] = {
                "avg_spread": avg_spread,
                "win_rate": win_rate,
                "sample_days": float(len(ser)),
                "recent_avg": recent_avg,
                "prev_avg": prev_avg,
            }

        dominant_style = "neutral"
        dominant_val = 0.0
        for style, stats in summary.items():
            v = float(stats.get("avg_spread", 0.0))
            if abs(v) > abs(dominant_val):
                dominant_val = v
                dominant_style = style
        dominant_direction = "long_high" if dominant_val >= 0 else "long_low"

        sample_days = int(max((len(x) for x in spreads_by_style.values()), default=0))
        drift_score = float(max(drift_by_style.values())) if drift_by_style else 0.0
        alerts = [f"style_drift:{k}" for k, v in drift_by_style.items() if float(v) > float(drift_gap_max)]
        active = sample_days >= min_sample_days
        if not active:
            alerts.append("style_drift_inactive")

        return {
            "active": bool(active),
            "lookback_days": int(lookback_days),
            "sample_days": int(sample_days),
            "drift_window_days": int(drift_window),
            "drift_gap_max": float(drift_gap_max),
            "block_on_alert": bool(block_on_alert),
            "dominant_style": dominant_style,
            "dominant_direction": dominant_direction,
            "dominant_spread": float(dominant_val),
            "style_spreads": summary,
            "style_drift_gap": {k: float(v) for k, v in drift_by_style.items()},
            "style_drift_score": float(drift_score),
            "alerts": alerts,
        }

    def _apply_style_drift_adaptive_guard(
        self,
        *,
        review: ReviewDelta,
        current_params: dict[str, float],
        style_diag: dict[str, Any],
    ) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("style_drift_adaptive_enabled", True))
        summary: dict[str, Any] = {
            "enabled": enabled,
            "active": bool(style_diag.get("active", False)),
            "applied": False,
            "blocked": False,
            "drift_ratio": 0.0,
            "intensity": 0.0,
            "confidence_step": 0.0,
            "trade_step": 0,
            "hold_step": 0,
            "alerts": [],
        }
        if not enabled:
            summary["reason"] = "disabled"
            return summary
        if not bool(style_diag.get("active", False)):
            summary["reason"] = "inactive"
            return summary

        drift_gap_max = self._clamp_float(float(style_diag.get("drift_gap_max", val.get("style_drift_gap_max", 0.01))), 1e-6, 1.0)
        drift_score = self._clamp_float(float(style_diag.get("style_drift_score", 0.0)), 0.0, 10.0)
        drift_ratio = float(drift_score / max(drift_gap_max, 1e-9))
        drift_alerts = [str(x) for x in style_diag.get("alerts", []) if str(x).startswith("style_drift:")]
        summary["drift_ratio"] = float(drift_ratio)
        summary["alerts"] = list(drift_alerts)

        trigger_ratio = self._clamp_float(float(val.get("style_drift_adaptive_trigger_ratio", 1.0)), 0.5, 5.0)
        ratio_for_max = self._clamp_float(float(val.get("style_drift_adaptive_ratio_for_max", 2.0)), trigger_ratio + 0.05, 8.0)
        block_ratio = self._clamp_float(float(val.get("style_drift_adaptive_block_ratio", 1.8)), 1.0, 12.0)
        conf_step_max = self._clamp_float(float(val.get("style_drift_adaptive_confidence_step_max", 6.0)), 0.0, 20.0)
        trade_step_max = self._clamp_int(float(val.get("style_drift_adaptive_trade_reduction_max", 2.0)), 0, 4)
        hold_step_max = self._clamp_int(float(val.get("style_drift_adaptive_hold_reduction_max", 2.0)), 0, 10)
        block_on_alert = bool(style_diag.get("block_on_alert", False))

        trigger = bool(drift_alerts) or bool(drift_ratio >= trigger_ratio)
        if not trigger:
            summary["reason"] = "no_trigger"
            return summary

        raw_intensity = (drift_ratio - trigger_ratio) / max(1e-6, ratio_for_max - trigger_ratio)
        intensity = self._clamp_float(raw_intensity, 0.0, 1.0)
        if drift_alerts and intensity < 0.20:
            intensity = 0.20
        summary["intensity"] = float(intensity)

        conf_step = 0.0
        if conf_step_max > 0.0:
            conf_step = conf_step_max * intensity
            if drift_alerts:
                conf_step = max(conf_step, 1.0)
        trade_step = 0
        if trade_step_max > 0:
            trade_step = int(round(trade_step_max * intensity))
            if drift_alerts and trade_step <= 0:
                trade_step = 1
            trade_step = int(min(trade_step_max, max(0, trade_step)))
        hold_step = 0
        if hold_step_max > 0:
            hold_step = int(round(hold_step_max * intensity))
            if drift_alerts and hold_step <= 0:
                hold_step = 1
            hold_step = int(min(hold_step_max, max(0, hold_step)))

        summary["confidence_step"] = float(conf_step)
        summary["trade_step"] = int(trade_step)
        summary["hold_step"] = int(hold_step)

        base_conf = float(review.parameter_changes.get("signal_confidence_min", current_params.get("signal_confidence_min", 60.0)))
        new_conf = float(self._clamp_float(base_conf + conf_step, 50.0, 90.0))
        review.parameter_changes["signal_confidence_min"] = new_conf

        base_trades = self._clamp_int(
            float(review.parameter_changes.get("max_daily_trades", current_params.get("max_daily_trades", 2.0))),
            1,
            5,
        )
        review.parameter_changes["max_daily_trades"] = float(max(1, base_trades - int(trade_step)))

        base_hold = self._clamp_int(
            float(review.parameter_changes.get("hold_days", current_params.get("hold_days", 5.0))),
            1,
            20,
        )
        review.parameter_changes["hold_days"] = float(max(1, base_hold - int(hold_step)))

        def _append_reason(key: str, msg: str) -> None:
            prior = str(review.change_reasons.get(key, "")).strip("；").strip()
            review.change_reasons[key] = (prior + "；" + msg).strip("；")

        guard_reason = (
            "风格漂移自适应收敛"
            + f"(ratio={drift_ratio:.2f}, intensity={intensity:.2f})"
        )
        _append_reason("signal_confidence_min", guard_reason + "，上调信号门槛")
        _append_reason("max_daily_trades", guard_reason + "，下调日内交易次数")
        _append_reason("hold_days", guard_reason + "，缩短持有暴露")

        blocked = bool((block_on_alert and bool(drift_alerts)) or (drift_ratio >= block_ratio))
        if blocked:
            review.pass_gate = False
            defect_code = "STYLE_DRIFT_SEVERE" if drift_ratio >= block_ratio else "STYLE_DRIFT_ALERT"
            if defect_code not in review.defects:
                review.defects.append(defect_code)

        review.notes.append(
            "style_drift_guard="
            + f"ratio={drift_ratio:.2f}; intensity={intensity:.2f}; "
            + f"conf_step={conf_step:.2f}; trade_step={trade_step}; hold_step={hold_step}; "
            + f"blocked={blocked}"
        )
        if drift_alerts:
            review.notes.append("style_drift_guard_alerts=" + ",".join(drift_alerts))

        summary["applied"] = True
        summary["blocked"] = bool(blocked)
        summary["block_on_alert"] = bool(block_on_alert)
        summary["block_ratio"] = float(block_ratio)
        summary["trigger_ratio"] = float(trigger_ratio)
        summary["ratio_for_max"] = float(ratio_for_max)
        summary["reason"] = "applied"
        return summary

    def _market_factor_state(
        self,
        *,
        sentiment: dict[str, float],
        regime: RegimeState,
        bars: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        def _f(key: str, default: float) -> float:
            try:
                out = float(sentiment.get(key, default))
            except (TypeError, ValueError):
                out = float(default)
            if not np.isfinite(out):
                return float(default)
            return out

        pcr = _f("pcr_50etf", 0.9)
        iv = _f("iv_50etf", 0.22)
        north = _f("northbound_netflow", 0.0)
        margin = _f("margin_balance_chg", 0.0)
        atr_z = float(max(0.0, regime.atr_z))
        style_state = self._cross_section_style_state(bars if isinstance(bars, pd.DataFrame) else pd.DataFrame())
        style_strength = self._clamp_float(float(style_state.get("style_signal_strength", 0.0)), 0.0, 1.0)
        style_weight = self._clamp_float(0.20 + 0.35 * style_strength, 0.20, 0.55)

        valuation_pressure = self._clamp_float(
            0.40 + max(0.0, pcr - 1.0) * 0.45 + max(0.0, iv - 0.24) * 1.10 + max(0.0, atr_z - 1.0) * 0.15,
            0.0,
            1.5,
        )
        momentum_preference = self._clamp_float(
            0.55 + np.tanh(north / 3e9) * 0.25 + np.tanh(margin * 15.0) * 0.20,
            0.0,
            1.5,
        )
        crowding_aversion = self._clamp_float(
            0.35 + max(0.0, iv - 0.24) * 1.20 + max(0.0, atr_z - 1.2) * 0.20,
            0.0,
            1.5,
        )
        small_cap_pressure = self._clamp_float(
            0.30 + max(0.0, -north / 3e9) * 0.35 + max(0.0, pcr - 1.05) * 0.30,
            0.0,
            1.5,
        )
        dividend_preference = self._clamp_float(
            0.50 + max(0.0, pcr - 1.0) * 0.12 + max(0.0, iv - 0.24) * 0.25 + np.tanh(-north / 3e9) * 0.08,
            0.0,
            1.5,
        )

        if regime.consensus in {RegimeLabel.RANGE, RegimeLabel.UNCERTAIN, RegimeLabel.DOWNTREND}:
            valuation_pressure = self._clamp_float(valuation_pressure + 0.12, 0.0, 1.5)
            small_cap_pressure = self._clamp_float(small_cap_pressure + 0.10, 0.0, 1.5)
            dividend_preference = self._clamp_float(dividend_preference + 0.08, 0.0, 1.5)

        style_value = self._clamp_float(0.30 + 0.90 * float(style_state.get("value_preference", 0.50)), 0.0, 1.5)
        style_mom = self._clamp_float(float(style_state.get("momentum_preference", 0.55)), 0.0, 1.5)
        style_crowd = self._clamp_float(float(style_state.get("crowding_aversion", 0.50)), 0.0, 1.5)
        style_small_cap_pressure = self._clamp_float(
            0.80 - 0.70 * float(style_state.get("size_preference", 0.50)),
            0.0,
            1.5,
        )
        style_dividend = self._clamp_float(float(style_state.get("dividend_preference", 0.55)), 0.0, 1.5)

        valuation_pressure = self._clamp_float(
            (1.0 - style_weight) * valuation_pressure + style_weight * style_value,
            0.0,
            1.5,
        )
        momentum_preference = self._clamp_float(
            (1.0 - style_weight) * momentum_preference + style_weight * style_mom,
            0.0,
            1.5,
        )
        crowding_aversion = self._clamp_float(
            (1.0 - style_weight) * crowding_aversion + style_weight * style_crowd,
            0.0,
            1.5,
        )
        small_cap_pressure = self._clamp_float(
            (1.0 - style_weight) * small_cap_pressure + style_weight * style_small_cap_pressure,
            0.0,
            1.5,
        )
        dividend_preference = self._clamp_float(
            (1.0 - style_weight) * dividend_preference + style_weight * style_dividend,
            0.0,
            1.5,
        )

        return {
            "valuation_pressure": float(valuation_pressure),
            "momentum_preference": float(momentum_preference),
            "crowding_aversion": float(crowding_aversion),
            "small_cap_pressure": float(small_cap_pressure),
            "dividend_preference": float(dividend_preference),
            "style_weight": float(style_weight),
            "style_signal_strength": float(style_strength),
            "style_sample_size": float(style_state.get("style_sample_size", 0.0)),
            "value_preference": float(style_state.get("value_preference", 0.50)),
            "size_preference": float(style_state.get("size_preference", 0.50)),
        }

    def run_review(self, as_of: date) -> ReviewDelta:
        start = self._review_backtest_start(as_of=as_of)
        bt = self.run_backtest(start=start, end=as_of)
        params = self._load_live_params()
        rollback_anchor = self._backup_live_params(as_of)
        factor_weights = {
            "macro": 0.20,
            "industry": 0.18,
            "news": 0.17,
            "sentiment": 0.15,
            "fundamental": 0.10,
            "technical": 0.20,
        }
        factor_contrib = self._estimate_factor_contrib_120d(as_of=as_of)
        thresholds = ReviewThresholds(
            positive_window_ratio_min=float(self.settings.validation.get("positive_window_ratio_min", 0.70)),
            max_drawdown_max=float(self.settings.validation.get("max_drawdown_max", 0.18)),
        )
        review = build_review_delta(
            as_of=as_of,
            backtest=bt,
            current_params=params,
            factor_weights=factor_weights,
            factor_contrib=factor_contrib,
            thresholds=thresholds,
        )
        style_diag = self._review_style_diagnostics(start=start, as_of=as_of)
        review.style_diagnostics = style_diag
        review.notes.append(
            "style_diag="
            + f"active={bool(style_diag.get('active', False))}; "
            + f"dominant={style_diag.get('dominant_style', 'neutral')}; "
            + f"dir={style_diag.get('dominant_direction', 'na')}; "
            + f"drift={float(style_diag.get('style_drift_score', 0.0)):.4f}; "
            + f"sample_days={int(style_diag.get('sample_days', 0))}"
        )
        if bool(style_diag.get("active", False)) and style_diag.get("alerts"):
            review.notes.append("style_diag_alerts=" + ",".join(str(x) for x in style_diag.get("alerts", [])))
        strategy_candidate = self._load_latest_strategy_candidate(as_of=as_of)
        strategy_lab_autorun: dict[str, Any] = {}
        if not strategy_candidate and bool(self.settings.validation.get("review_autorun_strategy_lab_if_missing", False)):
            strategy_lab_autorun = self._autorun_strategy_lab_for_review(as_of=as_of)
            if strategy_lab_autorun:
                strategy_candidate = self._load_latest_strategy_candidate(as_of=as_of)
                review.notes.append(
                    "strategy_lab_autorun="
                    + f"manifest={strategy_lab_autorun.get('manifest', '')}; "
                    + f"candidate_count={len(strategy_lab_autorun.get('candidates', []))}"
                )
        if strategy_candidate:
            self._merge_strategy_candidate_into_review(
                review=review,
                current_params=params,
                candidate_payload=strategy_candidate,
            )
        else:
            review.notes.append("No accepted strategy-lab candidate found within lookback window")

        runtime_mode = self._resolve_review_runtime_mode(start=start, as_of=as_of)
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        mode_adaptive = self._apply_mode_adaptive_update(
            review=review,
            current_params=params,
            runtime_mode=runtime_mode,
            mode_history=mode_history,
        )
        slot_regime_tuning = self._tune_slot_regime_thresholds(as_of=as_of)
        self._apply_mode_health_guard(
            review=review,
            current_params=params,
            mode_health=mode_health,
        )
        style_drift_guard = self._apply_style_drift_adaptive_guard(
            review=review,
            current_params=params,
            style_diag=style_diag,
        )
        style_diag["adaptive_guard"] = style_drift_guard
        review.style_diagnostics = style_diag
        review.notes.append(
            "mode_health="
            + f"runtime_mode={runtime_mode}; passed={mode_health.get('passed', True)}; "
            + f"active={mode_health.get('active', False)}"
        )
        review.notes.append(
            "slot_regime_tuning="
            + f"applied={bool(slot_regime_tuning.get('applied', False))}; "
            + f"changed={bool(slot_regime_tuning.get('changed', False))}; "
            + f"path={slot_regime_tuning.get('path', '')}"
        )
        review.rollback_anchor = rollback_anchor

        report = render_review_report(as_of=as_of, backtest=bt, review=review)
        review_dir = self.ctx.output_dir / "review"
        review_path = review_dir / f"{as_of.isoformat()}_review.md"
        delta_path = review_dir / f"{as_of.isoformat()}_param_delta.yaml"
        write_markdown(review_path, report)

        prev_params = params
        deltas = {
            k: float(review.parameter_changes.get(k, 0.0) - prev_params.get(k, 0.0))
            for k in review.parameter_changes.keys()
        }
        audit_payload = review.to_dict()
        audit_payload["previous_parameters"] = prev_params
        audit_payload["parameter_deltas"] = deltas
        audit_payload["impact_window_days"] = review.impact_window_days
        audit_payload["rollback_anchor"] = rollback_anchor
        audit_payload["strategy_lab_candidate"] = strategy_candidate
        audit_payload["strategy_lab_autorun"] = strategy_lab_autorun
        audit_payload["runtime_mode"] = runtime_mode
        audit_payload["mode_history"] = mode_history
        audit_payload["mode_health"] = mode_health
        audit_payload["mode_adaptive"] = mode_adaptive
        audit_payload["style_drift_guard"] = style_drift_guard
        audit_payload["slot_regime_tuning"] = slot_regime_tuning
        audit_payload["style_diagnostics"] = style_diag
        audit_payload["generated_at"] = datetime.now().isoformat()
        write_markdown(delta_path, yaml.safe_dump(audit_payload, allow_unicode=True, sort_keys=False))

        live_params_path = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        write_markdown(live_params_path, yaml.safe_dump(review.parameter_changes, allow_unicode=True, sort_keys=False))

        append_sqlite(self.ctx.sqlite_path, "review_runs", pd.DataFrame([audit_payload]))
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="review",
            run_id=as_of.isoformat(),
            artifacts={
                "review_report": str(review_path),
                "param_delta": str(delta_path),
                "live_params": str(live_params_path),
            },
            metrics={
                "pass_gate": bool(review.pass_gate),
                "defects": int(len(review.defects)),
                "changed_params": int(len(review.parameter_changes)),
                "runtime_mode": runtime_mode,
                "mode_health_passed": bool(mode_health.get("passed", True)),
                "mode_adaptive_applied": bool(mode_adaptive.get("applied", False)),
                "mode_adaptive_direction": str(mode_adaptive.get("direction", "none")),
                "slot_regime_tuning_changed": bool(slot_regime_tuning.get("changed", False)),
                "style_drift_score": float(style_diag.get("style_drift_score", 0.0)),
                "style_drift_alerts": int(
                    sum(1 for x in style_diag.get("alerts", []) if str(x).startswith("style_drift:"))
                ),
                "style_drift_guard_applied": bool(style_drift_guard.get("applied", False)),
                "style_drift_guard_blocked": bool(style_drift_guard.get("blocked", False)),
                "style_drift_guard_intensity": float(style_drift_guard.get("intensity", 0.0)),
            },
            checks={
                "positive_window_ratio": float(bt.positive_window_ratio),
                "max_drawdown": float(bt.max_drawdown),
                "violations": int(bt.violations),
            },
        )
        review.notes.append(f"manifest={manifest_path}")
        return review

    def test_all(
        self,
        *,
        fast: bool = False,
        fast_ratio: float = 0.10,
        fast_shard_index: int = 0,
        fast_shard_total: int = 1,
        fast_seed: str = "lie-fast-v1",
    ) -> dict[str, Any]:
        return self._testing_orchestrator().test_all(
            fast=fast,
            fast_ratio=fast_ratio,
            fast_shard_index=fast_shard_index,
            fast_shard_total=fast_shard_total,
            fast_seed=fast_seed,
        )

    def stable_replay_check(self, as_of: date, days: int | None = None) -> dict[str, Any]:
        return self._observability_orchestrator().stable_replay_check(as_of=as_of, days=days)

    def _release_orchestrator(self) -> ReleaseOrchestrator:
        return ReleaseOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            quality_snapshot=lambda d: self._quality_snapshot(d),
            backtest_snapshot=lambda d: self._backtest_snapshot(d),
            run_review=lambda d: self.run_review(d),
            health_check=lambda d, require_review: self.health_check(d, require_review=require_review),
            stable_replay_check=lambda d, days: self.stable_replay_check(d, days=days),
            test_all=lambda **kwargs: self.test_all(**kwargs),
            load_json_safely=lambda p: self._load_json_safely(p),
            sqlite_path=self.ctx.sqlite_path,
            run_stress_matrix=lambda d, modes: self.run_mode_stress_matrix(as_of=d, modes=modes),
        )

    def _architecture_orchestrator(self) -> ArchitectureOrchestrator:
        return ArchitectureOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            health_check=lambda d, require_review: self.health_check(d, require_review=require_review),
        )

    def _dependency_orchestrator(self) -> DependencyOrchestrator:
        return DependencyOrchestrator(
            settings=self.settings,
            source_root=self.ctx.root / "src",
            output_dir=self.ctx.output_dir,
        )

    def _observability_orchestrator(self) -> ObservabilityOrchestrator:
        return ObservabilityOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            sqlite_path=self.ctx.sqlite_path,
            run_eod=lambda d: self.run_eod(d),
        )

    def _testing_orchestrator(self) -> TestingOrchestrator:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        return TestingOrchestrator(
            root=self.ctx.root,
            output_dir=self.ctx.output_dir,
            timeout_seconds=max(30, int(val.get("test_all_timeout_seconds", 1800))),
        )

    def _scheduler_orchestrator(self) -> SchedulerOrchestrator:
        return SchedulerOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            run_premarket=lambda d: self.run_premarket(d),
            run_intraday_check=lambda d, slot: self.run_intraday_check(d, slot=slot),
            run_eod=lambda d: self.run_eod(d),
            run_review_cycle=lambda d, max_rounds: self.run_review_cycle(as_of=d, max_rounds=max_rounds),
            ops_report=lambda d, window_days: self.ops_report(as_of=d, window_days=window_days),
        )

    def gate_report(
        self,
        as_of: date,
        run_tests: bool = False,
        run_review_if_missing: bool = True,
    ) -> dict[str, Any]:
        return self._release_orchestrator().gate_report(
            as_of=as_of,
            run_tests=run_tests,
            run_review_if_missing=run_review_if_missing,
        )

    def ops_report(self, as_of: date, window_days: int = 7) -> dict[str, Any]:
        return self._release_orchestrator().ops_report(as_of=as_of, window_days=window_days)

    def review_until_pass(self, as_of: date, max_rounds: int = 3) -> dict[str, Any]:
        return self._release_orchestrator().review_until_pass(as_of=as_of, max_rounds=max_rounds)

    def run_review_cycle(self, as_of: date, max_rounds: int = 2, ops_window_days: int | None = None) -> dict[str, Any]:
        return self._release_orchestrator().run_review_cycle(
            as_of=as_of,
            max_rounds=max_rounds,
            ops_window_days=ops_window_days,
        )

    def run_slot(self, as_of: date, slot: str, max_review_rounds: int = 2) -> dict[str, Any]:
        return self._scheduler_orchestrator().run_slot(
            as_of=as_of,
            slot=slot,
            max_review_rounds=max_review_rounds,
        )

    def run_session(self, as_of: date, include_review: bool = True, max_review_rounds: int = 2) -> dict[str, Any]:
        return self._scheduler_orchestrator().run_session(
            as_of=as_of,
            include_review=include_review,
            max_review_rounds=max_review_rounds,
        )

    def run_daemon(
        self,
        poll_seconds: int = 30,
        max_cycles: int | None = None,
        max_review_rounds: int = 2,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._scheduler_orchestrator().run_daemon(
            poll_seconds=poll_seconds,
            max_cycles=max_cycles,
            max_review_rounds=max_review_rounds,
            dry_run=dry_run,
        )

    def health_check(self, as_of: date | None = None, require_review: bool = True) -> dict[str, Any]:
        return self._observability_orchestrator().health_check(as_of=as_of, require_review=require_review)

    def architecture_audit(self, as_of: date | None = None) -> dict[str, Any]:
        return self._architecture_orchestrator().architecture_audit(as_of=as_of)

    def dependency_audit(self, as_of: date | None = None) -> dict[str, Any]:
        return self._dependency_orchestrator().dependency_audit(as_of=as_of)
