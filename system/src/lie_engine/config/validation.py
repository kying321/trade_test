from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any
from zoneinfo import ZoneInfo

from lie_engine.config.settings import SystemSettings


HHMM_RE = re.compile(r"^\d{2}:\d{2}$")
EQUITY_RE = re.compile(r"^\d{6}$")
FUTURE_RE = re.compile(r"^[A-Z]{1,3}\d{4}$")
SUPPORTED_PROVIDER_PROFILES = {
    "opensource_dual",
    "opensource_primary",
    "hybrid_with_paid_placeholder",
    "paid_placeholder",
}


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    level: str  # error | warning
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "path": self.path, "message": self.message}


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _validate_hhmm(slot: Any) -> bool:
    if not isinstance(slot, str) or not HHMM_RE.match(slot):
        return False
    hh, mm = slot.split(":")
    return 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59


def _normalize_symbol_token(raw: Any) -> str:
    txt = str(raw or "").strip().upper()
    if not txt:
        return ""
    txt = "".join(txt.split())
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch for ch in txt if ch in allowed)


def _normalize_side_token(raw: Any) -> str:
    txt = str(raw or "").strip().upper()
    if txt in {"LONG", "BUY", "B", "1", "+1"}:
        return "LONG"
    if txt in {"SHORT", "SELL", "S", "-1"}:
        return "SHORT"
    if txt in {"FLAT", "BOTH", "NET", "NONE", "0", "NEUTRAL"}:
        return "FLAT"
    return ""


def validate_settings(settings: SystemSettings) -> dict[str, Any]:
    issues: list[ValidationIssue] = []

    tz = str(settings.timezone or "")
    try:
        ZoneInfo(tz)
    except Exception:
        issues.append(ValidationIssue("error", "timezone", f"无效时区: {tz!r}"))

    schedule = settings.schedule
    for key in ("premarket", "eod", "nightly_review"):
        if not _validate_hhmm(schedule.get(key)):
            issues.append(ValidationIssue("error", f"schedule.{key}", f"必须为 HH:MM，当前值: {schedule.get(key)!r}"))
    slots = schedule.get("intraday_slots", [])
    if not isinstance(slots, list) or not slots:
        issues.append(ValidationIssue("error", "schedule.intraday_slots", "必须是非空列表"))
    else:
        for i, slot in enumerate(slots):
            if not _validate_hhmm(slot):
                issues.append(ValidationIssue("error", f"schedule.intraday_slots[{i}]", f"必须为 HH:MM，当前值: {slot!r}"))

    thr = settings.thresholds
    hurst_trend = _as_float(thr.get("hurst_trend", 0.6))
    hurst_mean = _as_float(thr.get("hurst_mean_revert", 0.4))
    if not (0.0 < hurst_mean < hurst_trend < 1.0):
        issues.append(ValidationIssue("error", "thresholds.hurst_*", "需满足 0 < hurst_mean_revert < hurst_trend < 1"))
    if not (0.0 <= _as_float(thr.get("signal_confidence_min", 60.0)) <= 100.0):
        issues.append(ValidationIssue("error", "thresholds.signal_confidence_min", "必须在 [0, 100]"))
    if _as_float(thr.get("convexity_min", 3.0)) <= 0.0:
        issues.append(ValidationIssue("error", "thresholds.convexity_min", "必须 > 0"))
    if _as_float(thr.get("atr_extreme", 2.0)) <= 0.0:
        issues.append(ValidationIssue("error", "thresholds.atr_extreme", "必须 > 0"))

    risk = settings.risk
    max_single = _as_float(risk.get("max_single_risk_pct", 2.0))
    max_symbol = _as_float(risk.get("max_symbol_pct", 15.0))
    max_theme = _as_float(risk.get("max_theme_pct", 25.0))
    max_total = _as_float(risk.get("max_total_exposure_pct", 50.0))
    safety = _as_float(risk.get("safety_bucket_pct", 85.0))
    convex = _as_float(risk.get("convexity_bucket_pct", 15.0))
    for k, v in (
        ("risk.max_single_risk_pct", max_single),
        ("risk.max_symbol_pct", max_symbol),
        ("risk.max_theme_pct", max_theme),
        ("risk.max_total_exposure_pct", max_total),
    ):
        if not (0.0 <= v <= 100.0):
            issues.append(ValidationIssue("error", k, "必须在 [0, 100]"))
    if max_single > max_symbol:
        issues.append(ValidationIssue("error", "risk.max_single_risk_pct", "不能大于 risk.max_symbol_pct"))
    if max_symbol > max_total:
        issues.append(ValidationIssue("error", "risk.max_symbol_pct", "不能大于 risk.max_total_exposure_pct"))
    if max_theme > max_total:
        issues.append(ValidationIssue("warning", "risk.max_theme_pct", "建议不高于 risk.max_total_exposure_pct"))
    if not (0.0 <= safety <= 100.0 and 0.0 <= convex <= 100.0):
        issues.append(ValidationIssue("error", "risk.*_bucket_pct", "必须在 [0, 100]"))
    if abs((safety + convex) - 100.0) > 1e-6:
        issues.append(ValidationIssue("warning", "risk.*_bucket_pct", "建议 safety_bucket_pct + convexity_bucket_pct = 100"))

    val = settings.validation
    for k in ("data_completeness_min", "unresolved_conflict_max", "positive_window_ratio_min", "max_drawdown_max"):
        v = _as_float(val.get(k, 0.0))
        if not (0.0 <= v <= 1.0):
            issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    for k in ("source_confidence_min", "low_confidence_source_ratio_max"):
        v = _as_float(val.get(k, 0.0))
        if not (0.0 <= v <= 1.0):
            issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    execution_min_risk_multiplier = _as_float(val.get("execution_min_risk_multiplier", 0.2))
    source_confidence_floor_risk_multiplier = _as_float(val.get("source_confidence_floor_risk_multiplier", 0.35))
    mode_health_risk_multiplier = _as_float(val.get("mode_health_risk_multiplier", 0.5))
    mode_health_insufficient_sample_risk_multiplier = _as_float(
        val.get("mode_health_insufficient_sample_risk_multiplier", 0.85)
    )
    for k, v in (
        ("execution_min_risk_multiplier", execution_min_risk_multiplier),
        ("source_confidence_floor_risk_multiplier", source_confidence_floor_risk_multiplier),
        ("mode_health_risk_multiplier", mode_health_risk_multiplier),
        ("mode_health_insufficient_sample_risk_multiplier", mode_health_insufficient_sample_risk_multiplier),
    ):
        if not (0.0 <= v <= 1.0):
            issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if source_confidence_floor_risk_multiplier < execution_min_risk_multiplier:
        issues.append(
            ValidationIssue(
                "error",
                "validation.source_confidence_floor_risk_multiplier",
                "不能低于 validation.execution_min_risk_multiplier",
            )
        )
    if mode_health_risk_multiplier < execution_min_risk_multiplier:
        issues.append(
            ValidationIssue(
                "error",
                "validation.mode_health_risk_multiplier",
                "不能低于 validation.execution_min_risk_multiplier",
            )
        )
    if mode_health_insufficient_sample_risk_multiplier < execution_min_risk_multiplier:
        issues.append(
            ValidationIssue(
                "error",
                "validation.mode_health_insufficient_sample_risk_multiplier",
                "不能低于 validation.execution_min_risk_multiplier",
            )
        )
    if mode_health_risk_multiplier > mode_health_insufficient_sample_risk_multiplier:
        issues.append(
            ValidationIssue(
                "warning",
                "validation.mode_health_risk_multiplier",
                "建议不高于 validation.mode_health_insufficient_sample_risk_multiplier",
            )
        )
    for k in ("strategy_lab_merge_step", "mode_profile_blend_with_live"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if _as_int(val.get("required_stable_replay_days", 3)) < 1:
        issues.append(ValidationIssue("error", "validation.required_stable_replay_days", "必须 >= 1"))
    if _as_int(val.get("cooldown_consecutive_losses", 3)) < 1:
        issues.append(ValidationIssue("error", "validation.cooldown_consecutive_losses", "必须 >= 1"))
    if _as_int(val.get("major_event_window_hours", 24)) < 1:
        issues.append(ValidationIssue("error", "validation.major_event_window_hours", "必须 >= 1"))
    if _as_int(val.get("factor_lookback_days", 120)) < 20:
        issues.append(ValidationIssue("warning", "validation.factor_lookback_days", "建议 >= 20"))
    if "strategy_lab_manifest_lookback_days" in val and _as_int(val.get("strategy_lab_manifest_lookback_days", 45)) < 1:
        issues.append(ValidationIssue("error", "validation.strategy_lab_manifest_lookback_days", "必须 >= 1"))
    if "strategy_lab_autorun_lookback_days" in val and _as_int(val.get("strategy_lab_autorun_lookback_days", 180)) < 30:
        issues.append(ValidationIssue("warning", "validation.strategy_lab_autorun_lookback_days", "建议 >= 30"))
    for k in ("strategy_lab_autorun_candidate_count", "strategy_lab_autorun_max_symbols", "strategy_lab_autorun_report_symbol_cap", "strategy_lab_autorun_workers"):
        if k in val and _as_int(val.get(k, 1)) < 1:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 >= 1"))
    if "strategy_lab_autorun_review_days" in val and _as_int(val.get("strategy_lab_autorun_review_days", 0)) < 0:
        issues.append(ValidationIssue("error", "validation.strategy_lab_autorun_review_days", "必须 >= 0"))
    if "mode_stats_lookback_days" in val and _as_int(val.get("mode_stats_lookback_days", 0)) < 30:
        issues.append(ValidationIssue("error", "validation.mode_stats_lookback_days", "必须 >= 30"))
    if "mode_health_min_samples" in val and _as_int(val.get("mode_health_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.mode_health_min_samples", "必须 >= 1"))
    if "mode_health_min_profit_factor" in val and _as_float(val.get("mode_health_min_profit_factor", 0.0)) <= 0.0:
        issues.append(ValidationIssue("error", "validation.mode_health_min_profit_factor", "必须 > 0"))
    if "mode_health_min_win_rate" in val:
        wr = _as_float(val.get("mode_health_min_win_rate", 0.0))
        if not (0.0 <= wr <= 1.0):
            issues.append(ValidationIssue("error", "validation.mode_health_min_win_rate", "必须在 [0, 1]"))
    if "mode_health_max_drawdown_max" in val:
        dd = _as_float(val.get("mode_health_max_drawdown_max", 0.0))
        if not (0.0 <= dd <= 1.0):
            issues.append(ValidationIssue("error", "validation.mode_health_max_drawdown_max", "必须在 [0, 1]"))
    if "mode_health_max_violations" in val and _as_int(val.get("mode_health_max_violations", 0)) < 0:
        issues.append(ValidationIssue("error", "validation.mode_health_max_violations", "必须 >= 0"))
    if "mode_adaptive_update_min_samples" in val and _as_int(val.get("mode_adaptive_update_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.mode_adaptive_update_min_samples", "必须 >= 1"))
    if "mode_adaptive_update_step" in val:
        step = _as_float(val.get("mode_adaptive_update_step", 0.0))
        if not (0.0 < step <= 1.0):
            issues.append(ValidationIssue("error", "validation.mode_adaptive_update_step", "必须在 (0, 1]"))
    if "mode_adaptive_good_profit_factor" in val and _as_float(val.get("mode_adaptive_good_profit_factor", 0.0)) <= 0.0:
        issues.append(ValidationIssue("error", "validation.mode_adaptive_good_profit_factor", "必须 > 0"))
    if "mode_adaptive_bad_profit_factor" in val and _as_float(val.get("mode_adaptive_bad_profit_factor", 0.0)) <= 0.0:
        issues.append(ValidationIssue("error", "validation.mode_adaptive_bad_profit_factor", "必须 > 0"))
    if "mode_adaptive_good_profit_factor" in val and "mode_adaptive_bad_profit_factor" in val:
        good_pf = _as_float(val.get("mode_adaptive_good_profit_factor", 0.0))
        bad_pf = _as_float(val.get("mode_adaptive_bad_profit_factor", 0.0))
        if good_pf < bad_pf:
            issues.append(ValidationIssue("error", "validation.mode_adaptive_good_profit_factor", "必须 >= validation.mode_adaptive_bad_profit_factor"))
    for k in ("mode_adaptive_good_win_rate", "mode_adaptive_bad_win_rate", "mode_adaptive_good_drawdown_max", "mode_adaptive_bad_drawdown_max"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "mode_adaptive_good_win_rate" in val and "mode_adaptive_bad_win_rate" in val:
        good_wr = _as_float(val.get("mode_adaptive_good_win_rate", 0.0))
        bad_wr = _as_float(val.get("mode_adaptive_bad_win_rate", 0.0))
        if good_wr < bad_wr:
            issues.append(ValidationIssue("error", "validation.mode_adaptive_good_win_rate", "必须 >= validation.mode_adaptive_bad_win_rate"))
    if "mode_adaptive_good_drawdown_max" in val and "mode_adaptive_bad_drawdown_max" in val:
        good_dd = _as_float(val.get("mode_adaptive_good_drawdown_max", 0.0))
        bad_dd = _as_float(val.get("mode_adaptive_bad_drawdown_max", 0.0))
        if good_dd > bad_dd:
            issues.append(ValidationIssue("error", "validation.mode_adaptive_good_drawdown_max", "必须 <= validation.mode_adaptive_bad_drawdown_max"))
    if "mode_switch_window_days" in val and _as_int(val.get("mode_switch_window_days", 0)) < 3:
        issues.append(ValidationIssue("error", "validation.mode_switch_window_days", "必须 >= 3"))
    if "mode_drift_window_days" in val and _as_int(val.get("mode_drift_window_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.mode_drift_window_days", "必须 >= 1"))
    if "mode_drift_min_live_trades" in val and _as_int(val.get("mode_drift_min_live_trades", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.mode_drift_min_live_trades", "必须 >= 1"))
    if "mode_drift_win_rate_max_gap" in val:
        wr_gap = _as_float(val.get("mode_drift_win_rate_max_gap", 0.0))
        if not (0.0 <= wr_gap <= 1.0):
            issues.append(ValidationIssue("error", "validation.mode_drift_win_rate_max_gap", "必须在 [0, 1]"))
    if "mode_drift_profit_factor_max_gap" in val and _as_float(val.get("mode_drift_profit_factor_max_gap", 0.0)) <= 0.0:
        issues.append(ValidationIssue("error", "validation.mode_drift_profit_factor_max_gap", "必须 > 0"))
    if "mode_drift_focus_runtime_mode_only" in val and not isinstance(val.get("mode_drift_focus_runtime_mode_only"), bool):
        issues.append(ValidationIssue("error", "validation.mode_drift_focus_runtime_mode_only", "必须是布尔值"))
    if "style_attribution_lookback_days" in val and _as_int(val.get("style_attribution_lookback_days", 0)) < 20:
        issues.append(ValidationIssue("error", "validation.style_attribution_lookback_days", "必须 >= 20"))
    if "style_drift_window_days" in val and _as_int(val.get("style_drift_window_days", 0)) < 3:
        issues.append(ValidationIssue("error", "validation.style_drift_window_days", "必须 >= 3"))
    if "style_drift_min_sample_days" in val and _as_int(val.get("style_drift_min_sample_days", 0)) < 5:
        issues.append(ValidationIssue("error", "validation.style_drift_min_sample_days", "必须 >= 5"))
    if "style_drift_gap_max" in val:
        v = _as_float(val.get("style_drift_gap_max", 0.0))
        if not (0.0 < v <= 1.0):
            issues.append(ValidationIssue("error", "validation.style_drift_gap_max", "必须在 (0, 1]"))
    if "style_drift_block_on_alert" in val and not isinstance(val.get("style_drift_block_on_alert"), bool):
        issues.append(ValidationIssue("error", "validation.style_drift_block_on_alert", "必须是布尔值"))
    if "style_drift_adaptive_enabled" in val and not isinstance(val.get("style_drift_adaptive_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.style_drift_adaptive_enabled", "必须是布尔值"))
    if "style_drift_adaptive_confidence_step_max" in val and _as_float(
        val.get("style_drift_adaptive_confidence_step_max", 0.0)
    ) < 0.0:
        issues.append(ValidationIssue("error", "validation.style_drift_adaptive_confidence_step_max", "必须 >= 0"))
    if "style_drift_adaptive_trade_reduction_max" in val and _as_int(
        val.get("style_drift_adaptive_trade_reduction_max", 0)
    ) < 0:
        issues.append(ValidationIssue("error", "validation.style_drift_adaptive_trade_reduction_max", "必须 >= 0"))
    if "style_drift_adaptive_hold_reduction_max" in val and _as_int(
        val.get("style_drift_adaptive_hold_reduction_max", 0)
    ) < 0:
        issues.append(ValidationIssue("error", "validation.style_drift_adaptive_hold_reduction_max", "必须 >= 0"))
    if "style_drift_adaptive_trigger_ratio" in val and _as_float(
        val.get("style_drift_adaptive_trigger_ratio", 0.0)
    ) <= 0.0:
        issues.append(ValidationIssue("error", "validation.style_drift_adaptive_trigger_ratio", "必须 > 0"))
    if "style_drift_adaptive_ratio_for_max" in val and _as_float(
        val.get("style_drift_adaptive_ratio_for_max", 0.0)
    ) <= 0.0:
        issues.append(ValidationIssue("error", "validation.style_drift_adaptive_ratio_for_max", "必须 > 0"))
    if (
        "style_drift_adaptive_trigger_ratio" in val
        and "style_drift_adaptive_ratio_for_max" in val
        and _as_float(val.get("style_drift_adaptive_ratio_for_max", 0.0))
        <= _as_float(val.get("style_drift_adaptive_trigger_ratio", 0.0))
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.style_drift_adaptive_ratio_for_max",
                "必须 > validation.style_drift_adaptive_trigger_ratio",
            )
        )
    if "style_drift_adaptive_block_ratio" in val and _as_float(val.get("style_drift_adaptive_block_ratio", 0.0)) < 1.0:
        issues.append(ValidationIssue("error", "validation.style_drift_adaptive_block_ratio", "必须 >= 1"))
    if "style_drift_gate_enabled" in val and not isinstance(val.get("style_drift_gate_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.style_drift_gate_enabled", "必须是布尔值"))
    if "style_drift_gate_require_active" in val and not isinstance(val.get("style_drift_gate_require_active"), bool):
        issues.append(ValidationIssue("error", "validation.style_drift_gate_require_active", "必须是布尔值"))
    if "style_drift_gate_allow_alerts" in val and not isinstance(val.get("style_drift_gate_allow_alerts"), bool):
        issues.append(ValidationIssue("error", "validation.style_drift_gate_allow_alerts", "必须是布尔值"))
    if "style_drift_gate_max_alerts" in val and _as_int(val.get("style_drift_gate_max_alerts", 0)) < 0:
        issues.append(ValidationIssue("error", "validation.style_drift_gate_max_alerts", "必须 >= 0"))
    if "style_drift_gate_max_ratio" in val and _as_float(val.get("style_drift_gate_max_ratio", 0.0)) <= 0.0:
        issues.append(ValidationIssue("error", "validation.style_drift_gate_max_ratio", "必须 > 0"))
    if "style_drift_gate_hard_fail" in val and not isinstance(val.get("style_drift_gate_hard_fail"), bool):
        issues.append(ValidationIssue("error", "validation.style_drift_gate_hard_fail", "必须是布尔值"))
    if "style_drift_gate_lookback_days" in val and _as_int(val.get("style_drift_gate_lookback_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.style_drift_gate_lookback_days", "必须 >= 1"))
    if "ops_temporal_audit_enabled" in val and not isinstance(val.get("ops_temporal_audit_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_temporal_audit_enabled", "必须是布尔值"))
    if "ops_temporal_audit_lookback_days" in val and _as_int(val.get("ops_temporal_audit_lookback_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_temporal_audit_lookback_days", "必须 >= 1"))
    if "ops_temporal_audit_min_samples" in val and _as_int(val.get("ops_temporal_audit_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_temporal_audit_min_samples", "必须 >= 1"))
    for k in ("ops_temporal_audit_missing_ratio_max", "ops_temporal_audit_leak_ratio_max"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_temporal_audit_autofix_enabled" in val and not isinstance(
        val.get("ops_temporal_audit_autofix_enabled"), bool
    ):
        issues.append(ValidationIssue("error", "validation.ops_temporal_audit_autofix_enabled", "必须是布尔值"))
    if "ops_temporal_audit_autofix_max_writes" in val and _as_int(
        val.get("ops_temporal_audit_autofix_max_writes", -1)
    ) < 0:
        issues.append(ValidationIssue("error", "validation.ops_temporal_audit_autofix_max_writes", "必须 >= 0"))
    if "ops_temporal_audit_autofix_fix_strict_cutoff" in val and not isinstance(
        val.get("ops_temporal_audit_autofix_fix_strict_cutoff"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_temporal_audit_autofix_fix_strict_cutoff", "必须是布尔值")
        )
    if "ops_temporal_audit_autofix_require_safe" in val and not isinstance(
        val.get("ops_temporal_audit_autofix_require_safe"), bool
    ):
        issues.append(ValidationIssue("error", "validation.ops_temporal_audit_autofix_require_safe", "必须是布尔值"))
    if "ops_temporal_audit_autofix_patch_retention_days" in val and _as_int(
        val.get("ops_temporal_audit_autofix_patch_retention_days", 0)
    ) < 1:
        issues.append(ValidationIssue("error", "validation.ops_temporal_audit_autofix_patch_retention_days", "必须 >= 1"))
    if "ops_temporal_audit_autofix_patch_checksum_index_enabled" in val and not isinstance(
        val.get("ops_temporal_audit_autofix_patch_checksum_index_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_temporal_audit_autofix_patch_checksum_index_enabled",
                "必须是布尔值",
            )
        )
    if "ops_stress_matrix_trend_enabled" in val and not isinstance(val.get("ops_stress_matrix_trend_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_stress_matrix_trend_enabled", "必须是布尔值"))
    if "ops_stress_matrix_trend_window_runs" in val and _as_int(val.get("ops_stress_matrix_trend_window_runs", 0)) < 2:
        issues.append(ValidationIssue("error", "validation.ops_stress_matrix_trend_window_runs", "必须 >= 2"))
    if "ops_stress_matrix_trend_min_runs" in val and _as_int(val.get("ops_stress_matrix_trend_min_runs", 0)) < 2:
        issues.append(ValidationIssue("error", "validation.ops_stress_matrix_trend_min_runs", "必须 >= 2"))
    if "mode_stress_execution_friction_enabled" in val and not isinstance(
        val.get("mode_stress_execution_friction_enabled"), bool
    ):
        issues.append(ValidationIssue("error", "validation.mode_stress_execution_friction_enabled", "必须是布尔值"))
    if "mode_stress_execution_friction_scenarios" in val:
        raw_scenarios = val.get("mode_stress_execution_friction_scenarios")
        if not isinstance(raw_scenarios, list):
            issues.append(
                ValidationIssue("error", "validation.mode_stress_execution_friction_scenarios", "必须是列表")
            )
        else:
            for idx, row in enumerate(raw_scenarios):
                prefix = f"validation.mode_stress_execution_friction_scenarios[{idx}]"
                if not isinstance(row, dict):
                    issues.append(ValidationIssue("error", prefix, "必须是字典"))
                    continue
                if "execution_latency_days" in row and _as_int(row.get("execution_latency_days", -1)) < 0:
                    issues.append(ValidationIssue("error", f"{prefix}.execution_latency_days", "必须 >= 0"))
                if "latency_days" in row and _as_int(row.get("latency_days", -1)) < 0:
                    issues.append(ValidationIssue("error", f"{prefix}.latency_days", "必须 >= 0"))
                if "execution_friction_multiplier" in row and _as_float(
                    row.get("execution_friction_multiplier", -1.0)
                ) < 0.0:
                    issues.append(
                        ValidationIssue("error", f"{prefix}.execution_friction_multiplier", "必须 >= 0")
                    )
                if "friction_multiplier" in row and _as_float(row.get("friction_multiplier", -1.0)) < 0.0:
                    issues.append(ValidationIssue("error", f"{prefix}.friction_multiplier", "必须 >= 0"))
                if "execution_extra_slippage_bps" in row and _as_float(
                    row.get("execution_extra_slippage_bps", -1.0)
                ) < 0.0:
                    issues.append(
                        ValidationIssue("error", f"{prefix}.execution_extra_slippage_bps", "必须 >= 0")
                    )
                if "extra_slippage_bps" in row and _as_float(row.get("extra_slippage_bps", -1.0)) < 0.0:
                    issues.append(ValidationIssue("error", f"{prefix}.extra_slippage_bps", "必须 >= 0"))
    if "ops_stress_matrix_execution_friction_enabled" in val and not isinstance(
        val.get("ops_stress_matrix_execution_friction_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_stress_matrix_execution_friction_enabled", "必须是布尔值")
        )
    if "ops_stress_matrix_execution_friction_require_active" in val and not isinstance(
        val.get("ops_stress_matrix_execution_friction_require_active"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_require_active",
                "必须是布尔值",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_enabled" in val and not isinstance(
        val.get("ops_stress_matrix_execution_friction_trendline_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_enabled",
                "必须是布尔值",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_require_active" in val and not isinstance(
        val.get("ops_stress_matrix_execution_friction_trendline_require_active"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_require_active",
                "必须是布尔值",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_require_samples" in val and not isinstance(
        val.get("ops_stress_matrix_execution_friction_trendline_require_samples"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_require_samples",
                "必须是布尔值",
            )
        )
    for k in (
        "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled",
        "ops_stress_matrix_execution_friction_trendline_autotune_enabled",
        "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled",
        "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled",
        "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required",
        "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled",
        "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_enabled",
        "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_gate_hard_fail",
    ):
        if k in val and not isinstance(val.get(k), bool):
            issues.append(ValidationIssue("error", f"validation.{k}", "必须是布尔值"))
    for k in (
        "ops_stress_matrix_execution_friction_trendline_recent_runs",
        "ops_stress_matrix_execution_friction_trendline_prior_runs",
    ):
        if k in val and _as_int(val.get(k, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 >= 1"))
    if "ops_stress_matrix_execution_friction_trendline_max_staleness_days" in val and _as_int(
        val.get("ops_stress_matrix_execution_friction_trendline_max_staleness_days", -1)
    ) < 0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_max_staleness_days",
                "必须 >= 0",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days" in val and _as_int(
        val.get("ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days", -1)
    ) < 0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days",
                "必须 >= 0",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path" in val:
        raw_path = val.get("ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path",
                    "必须是非空字符串路径",
                )
            )
    if "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path" in val:
        raw_path = val.get("ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path",
                    "必须是非空字符串路径",
                )
            )
    if "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_retention_days" in val and _as_int(
        val.get("ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_retention_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_retention_days",
                "必须 >= 1",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days" in val and _as_int(
        val.get("ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days",
                "必须 >= 1",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max" in val:
        ratio = _as_float(
            val.get("ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max", -1.0),
            -1.0,
        )
        if not (0.0 <= ratio <= 1.0):
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max",
                    "必须在 [0,1] 之间",
                )
            )
    if "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max" in val:
        ratio = _as_float(
            val.get(
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max",
                -1.0,
            ),
            -1.0,
        )
        if not (0.0 <= ratio <= 1.0):
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max",
                    "必须在 [0,1] 之间",
                )
            )
    if "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs" in val and _as_int(
        val.get("ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs", 0)
    ) < 3:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs",
                "必须 >= 3",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions" in val and _as_int(
        val.get("ops_stress_matrix_execution_friction_trendline_autotune_min_transitions", 0)
    ) < 2:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_matrix_execution_friction_trendline_autotune_min_transitions",
                "必须 >= 2",
            )
        )
    if "ops_stress_matrix_execution_friction_trendline_min_runs" in val and _as_int(
        val.get("ops_stress_matrix_execution_friction_trendline_min_runs", 0)
    ) < 2:
        issues.append(
            ValidationIssue("error", "validation.ops_stress_matrix_execution_friction_trendline_min_runs", "必须 >= 2")
        )
    if "ops_stress_autorun_history_enabled" in val and not isinstance(val.get("ops_stress_autorun_history_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_history_enabled", "必须是布尔值"))
    if "ops_stress_autorun_history_window_days" in val and _as_int(
        val.get("ops_stress_autorun_history_window_days", 0)
    ) < 1:
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_history_window_days", "必须 >= 1"))
    if "ops_stress_autorun_history_min_rounds" in val and _as_int(
        val.get("ops_stress_autorun_history_min_rounds", 0)
    ) < 1:
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_history_min_rounds", "必须 >= 1"))
    if "ops_stress_autorun_history_no_trigger_min_payload_days" in val and _as_int(
        val.get("ops_stress_autorun_history_no_trigger_min_payload_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_stress_autorun_history_no_trigger_min_payload_days", "必须 >= 1")
        )
    if "ops_stress_autorun_history_retention_days" in val and _as_int(
        val.get("ops_stress_autorun_history_retention_days", 0)
    ) < 1:
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_history_retention_days", "必须 >= 1"))
    if "ops_stress_autorun_history_checksum_index_enabled" in val and not isinstance(
        val.get("ops_stress_autorun_history_checksum_index_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_stress_autorun_history_checksum_index_enabled", "必须是布尔值")
        )
    if "ops_stress_autorun_adaptive_monitor_enabled" in val and not isinstance(
        val.get("ops_stress_autorun_adaptive_monitor_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_stress_autorun_adaptive_monitor_enabled", "必须是布尔值")
        )
    if "ops_stress_autorun_adaptive_monitor_window_days" in val and _as_int(
        val.get("ops_stress_autorun_adaptive_monitor_window_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_stress_autorun_adaptive_monitor_window_days", "必须 >= 1")
        )
    if "ops_stress_autorun_adaptive_monitor_min_rounds" in val and _as_int(
        val.get("ops_stress_autorun_adaptive_monitor_min_rounds", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_stress_autorun_adaptive_monitor_min_rounds", "必须 >= 1")
        )
    if "ops_stress_autorun_adaptive_effective_base_ratio_floor" in val and _as_float(
        val.get("ops_stress_autorun_adaptive_effective_base_ratio_floor", 0.0)
    ) < 0.0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_autorun_adaptive_effective_base_ratio_floor",
                "必须 >= 0",
            )
        )
    if "ops_stress_autorun_adaptive_effective_base_ratio_ceiling" in val and _as_float(
        val.get("ops_stress_autorun_adaptive_effective_base_ratio_ceiling", 0.0)
    ) <= 0.0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_autorun_adaptive_effective_base_ratio_ceiling",
                "必须 > 0",
            )
        )
    for k in ("ops_stress_autorun_adaptive_throttle_ratio_max", "ops_stress_autorun_adaptive_expand_ratio_max"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if (
        "ops_stress_autorun_adaptive_effective_base_ratio_floor" in val
        and "ops_stress_autorun_adaptive_effective_base_ratio_ceiling" in val
    ):
        floor_v = _as_float(val.get("ops_stress_autorun_adaptive_effective_base_ratio_floor", 0.0))
        ceil_v = _as_float(val.get("ops_stress_autorun_adaptive_effective_base_ratio_ceiling", 0.0))
        if floor_v > ceil_v:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_stress_autorun_adaptive_effective_base_ratio_floor",
                    "不能大于 ops_stress_autorun_adaptive_effective_base_ratio_ceiling",
                )
            )
    if "ops_stress_autorun_reason_drift_enabled" in val and not isinstance(
        val.get("ops_stress_autorun_reason_drift_enabled"), bool
    ):
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_reason_drift_enabled", "必须是布尔值"))
    if "ops_stress_autorun_reason_drift_window_days" in val and _as_int(
        val.get("ops_stress_autorun_reason_drift_window_days", 0)
    ) < 1:
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_reason_drift_window_days", "必须 >= 1"))
    if "ops_stress_autorun_reason_drift_min_rounds" in val and _as_int(
        val.get("ops_stress_autorun_reason_drift_min_rounds", 0)
    ) < 2:
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_reason_drift_min_rounds", "必须 >= 2"))
    if "ops_stress_autorun_reason_drift_recent_rounds" in val and _as_int(
        val.get("ops_stress_autorun_reason_drift_recent_rounds", 0)
    ) < 1:
        issues.append(ValidationIssue("error", "validation.ops_stress_autorun_reason_drift_recent_rounds", "必须 >= 1"))
    if "ops_stress_autorun_reason_drift_retention_days" in val and _as_int(
        val.get("ops_stress_autorun_reason_drift_retention_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_stress_autorun_reason_drift_retention_days", "必须 >= 1")
        )
    if "ops_stress_autorun_reason_drift_checksum_index_enabled" in val and not isinstance(
        val.get("ops_stress_autorun_reason_drift_checksum_index_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_stress_autorun_reason_drift_checksum_index_enabled",
                "必须是布尔值",
            )
        )
    for k in ("ops_stress_autorun_reason_drift_mix_gap_max", "ops_stress_autorun_reason_drift_change_point_gap_max"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_stress_matrix_trend_window_runs" in val and "ops_stress_matrix_trend_min_runs" in val:
        wr = _as_int(val.get("ops_stress_matrix_trend_window_runs", 0))
        mr = _as_int(val.get("ops_stress_matrix_trend_min_runs", 0))
        if mr > wr:
            issues.append(ValidationIssue("error", "validation.ops_stress_matrix_trend_min_runs", "不能大于 ops_stress_matrix_trend_window_runs"))
    if (
        "ops_stress_matrix_execution_friction_trendline_min_runs" in val
        and (
            "ops_stress_matrix_execution_friction_trendline_recent_runs" in val
            or "ops_stress_matrix_execution_friction_trendline_prior_runs" in val
        )
    ):
        recent_runs = _as_int(val.get("ops_stress_matrix_execution_friction_trendline_recent_runs", 3))
        prior_runs = _as_int(val.get("ops_stress_matrix_execution_friction_trendline_prior_runs", 3))
        min_runs = _as_int(val.get("ops_stress_matrix_execution_friction_trendline_min_runs", 2))
        if min_runs > (recent_runs + prior_runs):
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_stress_matrix_execution_friction_trendline_min_runs",
                    "不能大于 recent_runs + prior_runs",
                )
            )
    for k in (
        "ops_stress_matrix_robustness_drop_max",
        "ops_stress_matrix_annual_return_drop_max",
        "ops_stress_matrix_drawdown_rise_max",
        "ops_stress_matrix_fail_ratio_max",
        "mode_stress_execution_friction_max_annual_drop",
        "mode_stress_execution_friction_max_drawdown_rise",
        "mode_stress_execution_friction_min_profit_factor_ratio",
        "mode_stress_execution_friction_max_fail_ratio",
        "mode_stress_execution_friction_max_positive_window_ratio_drop",
        "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise",
        "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise",
        "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop",
        "ops_stress_matrix_execution_friction_trendline_autotune_quantile",
        "ops_stress_matrix_execution_friction_trendline_autotune_step_max",
    ):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_slot_window_days" in val and _as_int(val.get("ops_slot_window_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_slot_window_days", "必须 >= 1"))
    if "ops_slot_min_samples" in val and _as_int(val.get("ops_slot_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_slot_min_samples", "必须 >= 1"))
    for k in (
        "ops_slot_missing_ratio_max",
        "ops_slot_premarket_anomaly_ratio_max",
        "ops_slot_intraday_anomaly_ratio_max",
        "ops_slot_eod_anomaly_ratio_max",
        "ops_slot_eod_quality_anomaly_ratio_max",
        "ops_slot_eod_risk_anomaly_ratio_max",
        "ops_slot_regime_tune_missing_ratio_hard_cap",
        "ops_slot_source_confidence_floor",
        "ops_slot_risk_multiplier_floor",
    ):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    for k in ("ops_slot_degradation_soft_multiplier", "ops_slot_degradation_hard_multiplier"):
        if k in val and _as_float(val.get(k, 0.0)) < 1.0:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 >= 1"))
    if "ops_slot_degradation_soft_multiplier" in val and "ops_slot_degradation_hard_multiplier" in val:
        soft = _as_float(val.get("ops_slot_degradation_soft_multiplier", 1.0))
        hard = _as_float(val.get("ops_slot_degradation_hard_multiplier", 1.0))
        if hard < soft:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_slot_degradation_hard_multiplier",
                    "不能小于 ops_slot_degradation_soft_multiplier",
                )
            )
    for k in ("ops_slot_hysteresis_soft_streak_days", "ops_slot_hysteresis_hard_streak_days"):
        if k in val and _as_int(val.get(k, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 >= 1"))
    if "ops_slot_degradation_enabled" in val and not isinstance(val.get("ops_slot_degradation_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_slot_degradation_enabled", "必须是布尔值"))
    if "ops_slot_hysteresis_enabled" in val and not isinstance(val.get("ops_slot_hysteresis_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_slot_hysteresis_enabled", "必须是布尔值"))
    for map_key in (
        "ops_slot_eod_quality_anomaly_ratio_max_by_regime",
        "ops_slot_eod_risk_anomaly_ratio_max_by_regime",
    ):
        if map_key in val:
            raw_map = val.get(map_key)
            if not isinstance(raw_map, dict):
                issues.append(ValidationIssue("error", f"validation.{map_key}", "必须是字典"))
            else:
                for bucket, raw_v in raw_map.items():
                    v = _as_float(raw_v, -1.0)
                    if not (0.0 <= v <= 1.0):
                        issues.append(ValidationIssue("error", f"validation.{map_key}.{bucket}", "必须在 [0, 1]"))
    if "ops_slot_use_live_regime_thresholds" in val and not isinstance(val.get("ops_slot_use_live_regime_thresholds"), bool):
        issues.append(ValidationIssue("error", "validation.ops_slot_use_live_regime_thresholds", "必须是布尔值"))
    if "ops_slot_regime_tune_enabled" in val and not isinstance(val.get("ops_slot_regime_tune_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_slot_regime_tune_enabled", "必须是布尔值"))
    if "ops_slot_regime_tune_window_days" in val and _as_int(val.get("ops_slot_regime_tune_window_days", 0)) < 30:
        issues.append(ValidationIssue("error", "validation.ops_slot_regime_tune_window_days", "必须 >= 30"))
    if "ops_slot_regime_tune_min_days" in val and _as_int(val.get("ops_slot_regime_tune_min_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_slot_regime_tune_min_days", "必须 >= 1"))
    for k in (
        "ops_slot_regime_tune_step",
        "ops_slot_regime_tune_buffer",
        "ops_slot_regime_tune_floor",
        "ops_slot_regime_tune_ceiling",
    ):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_slot_regime_tune_floor" in val and "ops_slot_regime_tune_ceiling" in val:
        f = _as_float(val.get("ops_slot_regime_tune_floor", 0.0))
        c = _as_float(val.get("ops_slot_regime_tune_ceiling", 0.0))
        if f > c:
            issues.append(ValidationIssue("error", "validation.ops_slot_regime_tune_floor", "不能大于 ops_slot_regime_tune_ceiling"))
    if "ops_reconcile_window_days" in val and _as_int(val.get("ops_reconcile_window_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_window_days", "必须 >= 1"))
    if "ops_reconcile_min_samples" in val and _as_int(val.get("ops_reconcile_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_min_samples", "必须 >= 1"))
    if "ops_reconcile_broker_row_diff_min_samples" in val and _as_int(
        val.get("ops_reconcile_broker_row_diff_min_samples", 0)
    ) < 1:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_broker_row_diff_min_samples", "必须 >= 1"))
    if "ops_reconcile_broker_row_diff_artifact_retention_days" in val and _as_int(
        val.get("ops_reconcile_broker_row_diff_artifact_retention_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_reconcile_broker_row_diff_artifact_retention_days",
                "必须 >= 1",
            )
        )
    for k in (
        "ops_reconcile_missing_ratio_max",
        "ops_reconcile_plan_gap_ratio_max",
        "ops_reconcile_closed_count_gap_ratio_max",
        "ops_reconcile_open_gap_ratio_max",
        "ops_reconcile_executed_dedup_pruned_ratio_max",
        "ops_reconcile_executed_dedup_days_ratio_max",
        "ops_reconcile_broker_missing_ratio_max",
        "ops_reconcile_broker_gap_ratio_max",
        "ops_reconcile_broker_contract_schema_invalid_ratio_max",
        "ops_reconcile_broker_contract_numeric_invalid_ratio_max",
        "ops_reconcile_broker_contract_symbol_invalid_ratio_max",
        "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max",
        "ops_reconcile_broker_row_diff_breach_ratio_max",
        "ops_reconcile_broker_row_diff_key_mismatch_max",
        "ops_reconcile_broker_row_diff_count_gap_max",
        "ops_reconcile_broker_row_diff_notional_gap_max",
        "ops_reconcile_broker_row_diff_alias_hit_rate_min",
        "ops_reconcile_broker_row_diff_unresolved_key_ratio_max",
    ):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_reconcile_closed_pnl_gap_abs_max" in val and _as_float(val.get("ops_reconcile_closed_pnl_gap_abs_max", 0.0)) < 0.0:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_closed_pnl_gap_abs_max", "必须 >= 0"))
    if "ops_reconcile_broker_pnl_gap_abs_max" in val and _as_float(val.get("ops_reconcile_broker_pnl_gap_abs_max", 0.0)) < 0.0:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_broker_pnl_gap_abs_max", "必须 >= 0"))
    if "ops_reconcile_executed_dedup_monitor_enabled" in val and not isinstance(
        val.get("ops_reconcile_executed_dedup_monitor_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_reconcile_executed_dedup_monitor_enabled", "必须是布尔值")
        )
    if "ops_reconcile_executed_dedup_gate_hard_fail" in val and not isinstance(
        val.get("ops_reconcile_executed_dedup_gate_hard_fail"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_reconcile_executed_dedup_gate_hard_fail", "必须是布尔值")
        )
    if "ops_reconcile_executed_dedup_restore_verify_enabled" in val and not isinstance(
        val.get("ops_reconcile_executed_dedup_restore_verify_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_reconcile_executed_dedup_restore_verify_enabled",
                "必须是布尔值",
            )
        )
    if "ops_reconcile_executed_dedup_restore_verify_hard_fail" in val and not isinstance(
        val.get("ops_reconcile_executed_dedup_restore_verify_hard_fail"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_reconcile_executed_dedup_restore_verify_hard_fail",
                "必须是布尔值",
            )
        )
    if "ops_reconcile_executed_dedup_restore_verify_max_age_days" in val and _as_int(
        val.get("ops_reconcile_executed_dedup_restore_verify_max_age_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_reconcile_executed_dedup_restore_verify_max_age_days",
                "必须 >= 1",
            )
        )
    if "ops_reconcile_executed_dedup_restore_verify_min_backup_rows" in val and _as_int(
        val.get("ops_reconcile_executed_dedup_restore_verify_min_backup_rows", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_reconcile_executed_dedup_restore_verify_min_backup_rows",
                "必须 >= 1",
            )
        )
    for k in (
        "ops_reconcile_broker_closed_pnl_abs_hard_max",
        "ops_reconcile_broker_position_qty_abs_hard_max",
        "ops_reconcile_broker_position_notional_abs_hard_max",
        "ops_reconcile_broker_price_abs_hard_max",
    ):
        if k in val and _as_float(val.get(k, 0.0)) <= 0.0:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 > 0"))
    if "ops_reconcile_require_broker_snapshot" in val and not isinstance(val.get("ops_reconcile_require_broker_snapshot"), bool):
        issues.append(ValidationIssue("error", "validation.ops_reconcile_require_broker_snapshot", "必须是布尔值"))
    if "ops_reconcile_broker_contract_emit_canonical_view" in val and not isinstance(
        val.get("ops_reconcile_broker_contract_emit_canonical_view"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_reconcile_broker_contract_emit_canonical_view", "必须是布尔值")
        )
    if "ops_reconcile_broker_contract_canonical_dir" in val and str(
        val.get("ops_reconcile_broker_contract_canonical_dir", "")
    ).strip() == "":
        issues.append(
            ValidationIssue("error", "validation.ops_reconcile_broker_contract_canonical_dir", "不能为空")
        )
    if "ops_reconcile_broker_row_diff_asof_only" in val and not isinstance(
        val.get("ops_reconcile_broker_row_diff_asof_only"), bool
    ):
        issues.append(ValidationIssue("error", "validation.ops_reconcile_broker_row_diff_asof_only", "必须是布尔值"))
    if "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled" in val and not isinstance(
        val.get("ops_reconcile_broker_row_diff_artifact_checksum_index_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_reconcile_broker_row_diff_artifact_checksum_index_enabled",
                "必须是布尔值",
            )
        )
    if "ops_reconcile_broker_row_diff_alias_monitor_enabled" in val and not isinstance(
        val.get("ops_reconcile_broker_row_diff_alias_monitor_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_reconcile_broker_row_diff_alias_monitor_enabled", "必须是布尔值")
        )
    if "ops_reconcile_broker_row_diff_symbol_alias_map" in val:
        alias_map = val.get("ops_reconcile_broker_row_diff_symbol_alias_map")
        if not isinstance(alias_map, dict):
            issues.append(
                ValidationIssue("error", "validation.ops_reconcile_broker_row_diff_symbol_alias_map", "必须是对象")
            )
        else:
            for key, value in alias_map.items():
                src = _normalize_symbol_token(key)
                dst = _normalize_symbol_token(value)
                if not src:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"validation.ops_reconcile_broker_row_diff_symbol_alias_map[{key}]",
                            "键规范化后不能为空",
                        )
                    )
                if not dst:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"validation.ops_reconcile_broker_row_diff_symbol_alias_map[{key}]",
                            "值规范化后不能为空",
                        )
                    )
    if "ops_reconcile_broker_row_diff_side_alias_map" in val:
        side_map = val.get("ops_reconcile_broker_row_diff_side_alias_map")
        if not isinstance(side_map, dict):
            issues.append(
                ValidationIssue("error", "validation.ops_reconcile_broker_row_diff_side_alias_map", "必须是对象")
            )
        else:
            for key, value in side_map.items():
                src = str(key or "").strip().upper()
                dst = _normalize_side_token(value)
                if not src:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "validation.ops_reconcile_broker_row_diff_side_alias_map",
                            "键不能为空",
                        )
                    )
                if not dst:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"validation.ops_reconcile_broker_row_diff_side_alias_map[{key}]",
                            "值必须可规范化为 LONG/SHORT/FLAT",
                        )
                    )
    if "ops_artifact_governance_profiles" in val:
        profiles = val.get("ops_artifact_governance_profiles")
        if not isinstance(profiles, dict):
            issues.append(ValidationIssue("error", "validation.ops_artifact_governance_profiles", "必须是对象"))
        else:
            allowed_keys = {
                "json_glob",
                "md_glob",
                "checksum_index_filename",
                "retention_days",
                "checksum_index_enabled",
            }
            for profile_name, profile_cfg in profiles.items():
                profile_key = str(profile_name or "").strip()
                path_base = f"validation.ops_artifact_governance_profiles[{profile_name}]"
                if not profile_key:
                    issues.append(ValidationIssue("error", path_base, "profile 名称不能为空"))
                    continue
                if not isinstance(profile_cfg, dict):
                    issues.append(ValidationIssue("error", path_base, "profile 配置必须是对象"))
                    continue
                for key in profile_cfg.keys():
                    if str(key) not in allowed_keys:
                        issues.append(
                            ValidationIssue(
                                "warning",
                                f"{path_base}.{key}",
                                "未知字段（将被忽略）",
                            )
                        )
                for key in ("json_glob", "md_glob", "checksum_index_filename"):
                    if key in profile_cfg:
                        raw = profile_cfg.get(key)
                        if not isinstance(raw, str):
                            issues.append(
                                ValidationIssue("error", f"{path_base}.{key}", "必须是字符串")
                            )
                        elif str(raw).strip() == "":
                            issues.append(
                                ValidationIssue("error", f"{path_base}.{key}", "不能为空")
                            )
                if "retention_days" in profile_cfg and _as_int(profile_cfg.get("retention_days", 0)) < 1:
                    issues.append(
                        ValidationIssue("error", f"{path_base}.retention_days", "必须 >= 1")
                    )
                if "checksum_index_enabled" in profile_cfg and not isinstance(
                    profile_cfg.get("checksum_index_enabled"), bool
                ):
                    issues.append(
                        ValidationIssue("error", f"{path_base}.checksum_index_enabled", "必须是布尔值")
                    )
    if "ops_artifact_governance_strict_mode_enabled" in val and not isinstance(
        val.get("ops_artifact_governance_strict_mode_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_artifact_governance_strict_mode_enabled",
                "必须是布尔值",
            )
        )
    if "ops_artifact_governance_baseline_snapshot_enabled" in val and not isinstance(
        val.get("ops_artifact_governance_baseline_snapshot_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_artifact_governance_baseline_snapshot_enabled",
                "必须是布尔值",
            )
        )
    if "ops_artifact_governance_baseline_auto_promote_on_review_pass" in val and not isinstance(
        val.get("ops_artifact_governance_baseline_auto_promote_on_review_pass"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_artifact_governance_baseline_auto_promote_on_review_pass",
                "必须是布尔值",
            )
        )
    for key in (
        "ops_artifact_governance_baseline_snapshot_path",
        "ops_artifact_governance_baseline_history_dir",
    ):
        if key in val:
            raw = val.get(key)
            if not isinstance(raw, str):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须是字符串"))
            elif str(raw).strip() == "":
                issues.append(ValidationIssue("error", f"validation.{key}", "不能为空"))
    if "ops_artifact_governance_profile_baseline" in val:
        baseline = val.get("ops_artifact_governance_profile_baseline")
        if not isinstance(baseline, dict):
            issues.append(ValidationIssue("error", "validation.ops_artifact_governance_profile_baseline", "必须是对象"))
        else:
            allowed_keys = {
                "json_glob",
                "md_glob",
                "checksum_index_filename",
                "retention_days",
                "checksum_index_enabled",
            }
            for profile_name, profile_cfg in baseline.items():
                profile_key = str(profile_name or "").strip()
                path_base = f"validation.ops_artifact_governance_profile_baseline[{profile_name}]"
                if not profile_key:
                    issues.append(ValidationIssue("error", path_base, "profile 名称不能为空"))
                    continue
                if not isinstance(profile_cfg, dict):
                    issues.append(ValidationIssue("error", path_base, "profile baseline 必须是对象"))
                    continue
                for key in profile_cfg.keys():
                    if str(key) not in allowed_keys:
                        issues.append(
                            ValidationIssue(
                                "warning",
                                f"{path_base}.{key}",
                                "未知字段（将被忽略）",
                            )
                        )
                for key in ("json_glob", "md_glob", "checksum_index_filename"):
                    if key in profile_cfg:
                        raw = profile_cfg.get(key)
                        if not isinstance(raw, str):
                            issues.append(
                                ValidationIssue("error", f"{path_base}.{key}", "必须是字符串")
                            )
                        elif str(raw).strip() == "":
                            issues.append(
                                ValidationIssue("error", f"{path_base}.{key}", "不能为空")
                            )
                if "retention_days" in profile_cfg and _as_int(profile_cfg.get("retention_days", 0)) < 1:
                    issues.append(
                        ValidationIssue("error", f"{path_base}.retention_days", "必须 >= 1")
                    )
                if "checksum_index_enabled" in profile_cfg and not isinstance(
                    profile_cfg.get("checksum_index_enabled"), bool
                ):
                    issues.append(
                        ValidationIssue("error", f"{path_base}.checksum_index_enabled", "必须是布尔值")
                    )
    if "ops_state_min_samples" in val and _as_int(val.get("ops_state_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_state_min_samples", "必须 >= 1"))
    if "ops_mode_health_fail_days_max" in val and _as_int(val.get("ops_mode_health_fail_days_max", 0)) < 0:
        issues.append(ValidationIssue("error", "validation.ops_mode_health_fail_days_max", "必须 >= 0"))
    if "ops_state_switch_rate_max_by_mode" in val:
        raw_map = val.get("ops_state_switch_rate_max_by_mode")
        if not isinstance(raw_map, dict):
            issues.append(ValidationIssue("error", "validation.ops_state_switch_rate_max_by_mode", "必须是字典"))
        else:
            for key, raw_v in raw_map.items():
                if str(key).strip() == "":
                    issues.append(
                        ValidationIssue("error", "validation.ops_state_switch_rate_max_by_mode", "键不能为空")
                    )
                if _as_float(raw_v, -1.0) <= 0.0:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"validation.ops_state_switch_rate_max_by_mode.{key}",
                            "必须 > 0",
                        )
                    )
    for k in ("ops_state_hysteresis_soft_streak_days", "ops_state_hysteresis_hard_streak_days"):
        if k in val and _as_int(val.get(k, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 >= 1"))
    for k in ("ops_state_degradation_soft_multiplier", "ops_state_degradation_hard_multiplier"):
        if k in val and _as_float(val.get(k, 0.0)) < 1.0:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 >= 1"))
    if "ops_state_degradation_soft_multiplier" in val and "ops_state_degradation_hard_multiplier" in val:
        soft = _as_float(val.get("ops_state_degradation_soft_multiplier", 1.0))
        hard = _as_float(val.get("ops_state_degradation_hard_multiplier", 1.0))
        if hard < soft:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_state_degradation_hard_multiplier",
                    "不能小于 ops_state_degradation_soft_multiplier",
                )
            )
    for k in ("ops_state_degradation_floor_soft_ratio", "ops_state_degradation_floor_hard_ratio"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_state_degradation_floor_soft_ratio" in val and "ops_state_degradation_floor_hard_ratio" in val:
        soft_r = _as_float(val.get("ops_state_degradation_floor_soft_ratio", 1.0))
        hard_r = _as_float(val.get("ops_state_degradation_floor_hard_ratio", 1.0))
        if hard_r > soft_r:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_state_degradation_floor_hard_ratio",
                    "不能大于 ops_state_degradation_floor_soft_ratio",
                )
            )
    if "ops_state_degradation_enabled" in val and not isinstance(val.get("ops_state_degradation_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_state_degradation_enabled", "必须是布尔值"))
    if "ops_state_hysteresis_enabled" in val and not isinstance(val.get("ops_state_hysteresis_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_state_hysteresis_enabled", "必须是布尔值"))
    if "ops_degradation_calibration_enabled" in val and not isinstance(
        val.get("ops_degradation_calibration_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_enabled", "必须是布尔值")
        )
    if "ops_degradation_calibration_use_live_overrides" in val and not isinstance(
        val.get("ops_degradation_calibration_use_live_overrides"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_use_live_overrides", "必须是布尔值")
        )
    if "ops_degradation_calibration_live_params_path" in val:
        raw = val.get("ops_degradation_calibration_live_params_path")
        if not isinstance(raw, str):
            issues.append(
                ValidationIssue("error", "validation.ops_degradation_calibration_live_params_path", "必须是字符串")
            )
        elif str(raw).strip() == "":
            issues.append(
                ValidationIssue("error", "validation.ops_degradation_calibration_live_params_path", "不能为空")
            )
    if "ops_degradation_calibration_rollback_enabled" in val and not isinstance(
        val.get("ops_degradation_calibration_rollback_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_rollback_enabled", "必须是布尔值")
        )
    if "ops_degradation_calibration_rollback_auto_promote_on_stable" in val and not isinstance(
        val.get("ops_degradation_calibration_rollback_auto_promote_on_stable"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_calibration_rollback_auto_promote_on_stable",
                "必须是布尔值",
            )
        )
    if "release_decision_freshness_enabled" in val and not isinstance(
        val.get("release_decision_freshness_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.release_decision_freshness_enabled",
                "必须是布尔值",
            )
        )
    if "release_decision_freshness_hard_fail" in val and not isinstance(
        val.get("release_decision_freshness_hard_fail"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.release_decision_freshness_hard_fail",
                "必须是布尔值",
            )
        )
    for key in (
        "release_decision_review_max_staleness_hours",
        "release_decision_gate_max_staleness_hours",
        "release_decision_eod_max_staleness_hours",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    if "ops_degradation_guardrail_dashboard_enabled" in val and not isinstance(
        val.get("ops_degradation_guardrail_dashboard_enabled"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_dashboard_enabled",
                "必须是布尔值",
            )
        )
    if "ops_degradation_guardrail_dashboard_hard_fail" in val and not isinstance(
        val.get("ops_degradation_guardrail_dashboard_hard_fail"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_dashboard_hard_fail",
                "必须是布尔值",
            )
        )
    if "ops_degradation_guardrail_dashboard_use_live_overrides" in val and not isinstance(
        val.get("ops_degradation_guardrail_dashboard_use_live_overrides"), bool
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_dashboard_use_live_overrides",
                "必须是布尔值",
            )
        )
    if "ops_degradation_guardrail_dashboard_live_params_path" in val:
        raw_live = val.get("ops_degradation_guardrail_dashboard_live_params_path")
        if not isinstance(raw_live, str):
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_degradation_guardrail_dashboard_live_params_path",
                    "必须是字符串",
                )
            )
        elif str(raw_live).strip() == "":
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_degradation_guardrail_dashboard_live_params_path",
                    "不能为空",
                )
            )
    if "ops_degradation_guardrail_dashboard_window_days" in val and _as_int(
        val.get("ops_degradation_guardrail_dashboard_window_days", 0)
    ) < 7:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_dashboard_window_days",
                "必须 >= 7",
            )
        )
    if "ops_degradation_guardrail_dashboard_min_samples" in val and _as_int(
        val.get("ops_degradation_guardrail_dashboard_min_samples", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_dashboard_min_samples",
                "必须 >= 1",
            )
        )
    for key in (
        "ops_degradation_guardrail_cooldown_hit_rate_max",
        "ops_degradation_guardrail_suppressed_trigger_density_max",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    if "ops_degradation_guardrail_promotion_latency_days_max" in val and _as_int(
        val.get("ops_degradation_guardrail_promotion_latency_days_max", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_promotion_latency_days_max",
                "必须 >= 1",
            )
        )
    if "ops_degradation_guardrail_false_positive_target_max" in val:
        fp_target = _as_float(val.get("ops_degradation_guardrail_false_positive_target_max", -1.0))
        if not (0.0 <= fp_target <= 1.0):
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_degradation_guardrail_false_positive_target_max",
                    "必须在 [0, 1]",
                )
            )
    for key in (
        "ops_degradation_guardrail_burnin_autofill_review_if_missing",
        "ops_degradation_guardrail_burnin_require_min_samples_for_tune",
        "ops_degradation_guardrail_burnin_light_backfill_enabled",
        "ops_degradation_guardrail_burnin_low_cost_replay_enabled",
        "ops_degradation_guardrail_burnin_use_live_overrides",
        "ops_degradation_guardrail_burnin_budget_audit_enabled",
        "ops_degradation_guardrail_burnin_budget_audit_auto_tune",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    if "ops_degradation_guardrail_burnin_live_params_path" in val:
        raw = val.get("ops_degradation_guardrail_burnin_live_params_path")
        if not isinstance(raw, str):
            issues.append(
                ValidationIssue("error", "validation.ops_degradation_guardrail_burnin_live_params_path", "必须是字符串")
            )
        elif str(raw).strip() == "":
            issues.append(
                ValidationIssue("error", "validation.ops_degradation_guardrail_burnin_live_params_path", "不能为空")
            )
    for key in (
        "ops_degradation_guardrail_burnin_light_backfill_max_days_per_run",
        "ops_degradation_guardrail_burnin_review_autofill_max_days_per_run",
        "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run",
        "ops_degradation_guardrail_burnin_budget_audit_min_days",
        "ops_degradation_guardrail_burnin_budget_audit_max_days",
    ):
        if key in val and _as_int(val.get(key, -1)) < 0:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 0"))
    if "ops_degradation_guardrail_burnin_budget_audit_step_days" in val and _as_int(
        val.get("ops_degradation_guardrail_burnin_budget_audit_step_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_burnin_budget_audit_step_days",
                "必须 >= 1",
            )
        )
    for key in (
        "ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min",
        "ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    if (
        "ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min" in val
        and "ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max" in val
        and _as_float(val.get("ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max", 0.0))
        > _as_float(val.get("ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min", 1.0))
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max",
                "不能大于 ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min",
            )
        )
    for key in (
        "ops_degradation_guardrail_threshold_drift_enabled",
        "ops_degradation_guardrail_threshold_drift_gate_hard_fail",
        "ops_degradation_guardrail_threshold_drift_require_active",
        "ops_degradation_guardrail_threshold_drift_autofix_enabled",
        "ops_degradation_guardrail_threshold_drift_autofix_on_missing",
        "ops_degradation_guardrail_threshold_drift_autofix_on_stale",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    if "ops_degradation_guardrail_threshold_drift_max_staleness_days" in val and _as_int(
        val.get("ops_degradation_guardrail_threshold_drift_max_staleness_days", -1)
    ) < 0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_threshold_drift_max_staleness_days",
                "必须 >= 0",
            )
        )
    if "ops_degradation_guardrail_threshold_drift_min_burnin_samples" in val and _as_int(
        val.get("ops_degradation_guardrail_threshold_drift_min_burnin_samples", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_threshold_drift_min_burnin_samples",
                "必须 >= 1",
            )
        )
    if "ops_degradation_guardrail_threshold_drift_autofix_window_days" in val and _as_int(
        val.get("ops_degradation_guardrail_threshold_drift_autofix_window_days", 0)
    ) < 7:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_threshold_drift_autofix_window_days",
                "必须 >= 7",
            )
        )
    for key in (
        "ops_degradation_guardrail_threshold_drift_warn_ratio",
        "ops_degradation_guardrail_threshold_drift_critical_ratio",
    ):
        if key in val and _as_float(val.get(key, -1.0)) < 0.0:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 0"))
    for key in (
        "ops_compaction_restore_trend_enabled",
        "ops_compaction_restore_trend_gate_hard_fail",
        "ops_compaction_restore_trend_require_active",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    if "ops_compaction_restore_trend_min_restore_required_runs" in val and _as_int(
        val.get("ops_compaction_restore_trend_min_restore_required_runs", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_compaction_restore_trend_min_restore_required_runs", "必须 >= 1")
        )
    for key in (
        "ops_guard_loop_frontend_snapshot_trend_enabled",
        "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail",
        "ops_guard_loop_frontend_snapshot_trend_require_active",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    for key in (
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_enabled",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_checksum_index_enabled",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_enabled",
        "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_checksum_index_enabled",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_enabled",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_gate_hard_fail",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    for key in (
        "ops_guard_loop_frontend_snapshot_trend_window_days",
        "ops_guard_loop_frontend_snapshot_trend_min_samples",
        "ops_guard_loop_frontend_snapshot_trend_max_failure_streak",
        "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    for key in (
        "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio",
        "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio",
        "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    if "ops_guard_loop_frontend_snapshot_recovery_antiflap_cooldown_hours" in val:
        cooldown_hours = _as_float(
            val.get("ops_guard_loop_frontend_snapshot_recovery_antiflap_cooldown_hours", -1.0)
        )
        if cooldown_hours < 0.0:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_cooldown_hours",
                    "必须 >= 0",
                )
            )
    for key in (
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_window_runs",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_max_escalations",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_min_timeout_streak",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_retention_days",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_min_long_samples",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_retention_days",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_window_days",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_samples",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_recent_days",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_prior_days",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_min_samples",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    if "ops_guard_loop_frontend_snapshot_trend_controlled_apply_max_apply_window_days" in val and _as_int(
        val.get("ops_guard_loop_frontend_snapshot_trend_controlled_apply_max_apply_window_days", -1)
    ) < 0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_max_apply_window_days",
                "必须 >= 0",
            )
        )
    if "ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path" in val:
        raw_path = str(
            val.get("ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path", "")
        ).strip()
        if not raw_path:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path",
                    "不能为空",
                )
            )
    if "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs" in val and _as_int(
        val.get("ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs", -1)
    ) < 0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs",
                "必须 >= 0",
            )
        )
    if "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio" in val:
        vv = _as_float(
            val.get(
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio",
                -1.0,
            )
        )
        if not (0.0 <= vv <= 1.0):
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio",
                    "必须在 [0, 1]",
                )
            )
    for key in (
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_non_apply_ratio",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_rollback_guard_ratio",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_apply_ratio",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_non_apply_ratio_rise",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_rollback_guard_ratio_rise",
        "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_apply_ratio_drop",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    for key in (
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_suppression_ratio_delta",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_replay_missed_ratio_delta",
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_reason_missing_ratio_delta",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    if (
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days" in val
        and "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days" in val
    ):
        short_days = _as_int(
            val.get("ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days", 1)
        )
        long_days = _as_int(
            val.get(
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days",
                1,
            )
        )
        if long_days < short_days:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days",
                    "不能小于 ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days",
                )
            )
    if (
        "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_window_runs" in val
        and "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_max_escalations" in val
    ):
        antiflap_window = _as_int(
            val.get("ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_window_runs", 1)
        )
        antiflap_max = _as_int(
            val.get("ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_max_escalations", 1)
        )
        if antiflap_max > antiflap_window:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_max_escalations",
                    "不能大于 ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_window_runs",
                )
            )
    if (
        "ops_degradation_guardrail_threshold_drift_warn_ratio" in val
        and "ops_degradation_guardrail_threshold_drift_critical_ratio" in val
        and _as_float(val.get("ops_degradation_guardrail_threshold_drift_critical_ratio", 0.0))
        < _as_float(val.get("ops_degradation_guardrail_threshold_drift_warn_ratio", 0.0))
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_guardrail_threshold_drift_critical_ratio",
                "不能小于 ops_degradation_guardrail_threshold_drift_warn_ratio",
            )
        )
    for key in (
        "ops_guard_loop_cadence_non_apply_rollback_lift_enabled",
        "ops_guard_loop_cadence_non_apply_rollback_lift_allow_upgrade_during_cooldown",
        "ops_guard_loop_cadence_non_apply_rollback_lift_force_heavy_hard",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    if "ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days" in val and _as_int(
        val.get("ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days", -1)
    ) < 0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days",
                "必须 >= 0",
            )
        )
    if "ops_guard_loop_cadence_non_apply_rollback_lift_lookback_days" in val and _as_int(
        val.get("ops_guard_loop_cadence_non_apply_rollback_lift_lookback_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_non_apply_rollback_lift_lookback_days",
                "必须 >= 1",
            )
        )
    for key in (
        "ops_guard_loop_cadence_non_apply_semantic_drift_window_days",
        "ops_guard_loop_cadence_non_apply_semantic_drift_min_samples",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    for key in (
        "ops_guard_loop_cadence_non_apply_semantic_drift_max_pulse_fail_ratio",
        "ops_guard_loop_cadence_non_apply_semantic_drift_max_gap_backfill_fail_ratio",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    for key in (
        "ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min",
        "ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    if (
        "ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min" in val
        and "ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min" in val
    ):
        light_min = _as_int(val.get("ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min", 1))
        heavy_min = _as_int(val.get("ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min", 1))
        if heavy_min < light_min:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min",
                    "不能小于 ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min",
                )
            )
    for key in (
        "ops_guard_loop_cadence_non_apply_lift_trend_enabled",
        "ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail",
        "ops_guard_loop_cadence_non_apply_lift_trend_require_active",
        "ops_guard_loop_cadence_non_apply_lift_trend_checksum_index_enabled",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    for key in (
        "ops_guard_loop_cadence_non_apply_lift_trend_window_days",
        "ops_guard_loop_cadence_non_apply_lift_trend_min_samples",
        "ops_guard_loop_cadence_non_apply_lift_trend_retention_days",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    for key in (
        "ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min",
        "ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    for key in (
        "ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max",
        "ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max",
        "ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min",
        "ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min",
    ):
        if key in val:
            vv = _as_float(val.get(key, -9.0))
            if not (-1.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [-1, 1]"))
    if (
        "ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max" in val
        and "ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max" in val
    ):
        light_applied = _as_float(val.get("ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max", -0.15))
        heavy_applied = _as_float(val.get("ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max", -0.30))
        if heavy_applied > light_applied:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max",
                    "不能大于 ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max",
                )
            )
    if (
        "ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min" in val
        and "ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min" in val
    ):
        light_cooldown = _as_float(val.get("ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min", 0.15))
        heavy_cooldown = _as_float(val.get("ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min", 0.30))
        if heavy_cooldown < light_cooldown:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min",
                    "不能小于 ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min",
                )
            )
    for key in (
        "ops_guard_loop_cadence_lift_trend_preset_drift_window_days",
        "ops_guard_loop_cadence_lift_trend_preset_drift_retention_days",
        "ops_guard_loop_cadence_lift_trend_preset_drift_min_samples",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    for key in (
        "ops_guard_loop_cadence_lift_trend_preset_drift_min_recovery_link_rate",
        "ops_guard_loop_cadence_lift_trend_preset_drift_min_retro_found_rate",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    if (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_enabled" in val
        and not isinstance(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_enabled"), bool)
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_enabled",
                "必须是布尔值",
            )
        )
    if (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_min_samples" in val
        and _as_int(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_min_samples", 0)) < 1
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_min_samples",
                "必须 >= 1",
            )
        )
    for key in (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_enabled",
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_enabled",
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_staleness_guard_enabled",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    if (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_apply_cooldown_days" in val
        and _as_int(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_apply_cooldown_days", -1)) < 0
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_apply_cooldown_days",
                "必须 >= 0",
            )
        )
    if (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_window_days" in val
        and _as_int(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_window_days", 0)) < 1
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_window_days",
                "必须 >= 1",
            )
        )
    if (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_max_staleness_days" in val
        and _as_int(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_max_staleness_days", -1)) < 0
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_max_staleness_days",
                "必须 >= 0",
            )
        )
    for key in (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_step_max",
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_low",
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high",
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_applied_gap_min",
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_cooldown_gap_min",
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_min_delta",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    if (
        "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_low" in val
        and "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high" in val
    ):
        low = _as_float(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_low", 0.0))
        high = _as_float(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high", 1.0))
        if high < low:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high",
                    "不能小于 ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_low",
                )
            )
    if (
        "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_enabled" in val
        and not isinstance(val.get("ops_guard_loop_cadence_lift_trend_preset_drift_trendline_enabled"), bool)
    ):
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_guard_loop_cadence_lift_trend_preset_drift_trendline_enabled",
                "必须是布尔值",
            )
        )
    for key in (
        "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_recent_days",
        "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_prior_days",
        "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_min_samples",
    ):
        if key in val and _as_int(val.get(key, 0)) < 1:
            issues.append(ValidationIssue("error", f"validation.{key}", "必须 >= 1"))
    for key in (
        "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_recovery_link_drop",
        "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_retro_found_drop",
    ):
        if key in val:
            vv = _as_float(val.get(key, -1.0))
            if not (0.0 <= vv <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    if "ops_weekly_guardrail_enabled" in val and not isinstance(val.get("ops_weekly_guardrail_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.ops_weekly_guardrail_enabled", "必须是布尔值"))
    if "ops_weekly_guardrail_weekday" in val:
        weekday = _as_int(val.get("ops_weekly_guardrail_weekday", 0))
        if not (1 <= weekday <= 7):
            issues.append(ValidationIssue("error", "validation.ops_weekly_guardrail_weekday", "必须在 [1, 7]"))
    if "ops_weekly_guardrail_trigger_slot" in val and not _validate_hhmm(val.get("ops_weekly_guardrail_trigger_slot")):
        issues.append(ValidationIssue("error", "validation.ops_weekly_guardrail_trigger_slot", "必须为 HH:MM"))
    if "ops_weekly_guardrail_burnin_days" in val and _as_int(val.get("ops_weekly_guardrail_burnin_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_weekly_guardrail_burnin_days", "必须 >= 1"))
    if "ops_weekly_guardrail_drift_window_days" in val and _as_int(
        val.get("ops_weekly_guardrail_drift_window_days", 0)
    ) < 7:
        issues.append(ValidationIssue("error", "validation.ops_weekly_guardrail_drift_window_days", "必须 >= 7"))
    for key in (
        "ops_weekly_guardrail_auto_tune",
        "ops_weekly_guardrail_run_stable_replay",
        "ops_weekly_guardrail_require_ops_window",
        "ops_weekly_guardrail_require_burnin_coverage",
        "ops_weekly_guardrail_compact_enabled",
        "ops_weekly_guardrail_compact_dry_run",
        "ops_weekly_guardrail_compact_verify_restore",
        "ops_weekly_guardrail_compact_verify_keep_temp_db",
        "ops_weekly_guardrail_compact_controlled_apply_enabled",
        "ops_weekly_guardrail_compact_controlled_apply_require_restore_verify_pass",
        "ops_weekly_guardrail_compact_controlled_apply_require_weekly_status_ok",
    ):
        if key in val and not isinstance(val.get(key), bool):
            issues.append(ValidationIssue("error", f"validation.{key}", "必须是布尔值"))
    if "ops_weekly_guardrail_compact_window_days" in val and _as_int(
        val.get("ops_weekly_guardrail_compact_window_days", 0)
    ) < 7:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_weekly_guardrail_compact_window_days",
                "必须 >= 7",
            )
        )
    if "ops_weekly_guardrail_compact_chunk_days" in val and _as_int(
        val.get("ops_weekly_guardrail_compact_chunk_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_weekly_guardrail_compact_chunk_days",
                "必须 >= 1",
            )
        )
    if "ops_weekly_guardrail_compact_controlled_apply_stability_weeks" in val and _as_int(
        val.get("ops_weekly_guardrail_compact_controlled_apply_stability_weeks", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_weekly_guardrail_compact_controlled_apply_stability_weeks",
                "必须 >= 1",
            )
        )
    if "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks" in val and _as_int(
        val.get("ops_weekly_guardrail_compact_controlled_apply_cadence_weeks", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_weekly_guardrail_compact_controlled_apply_cadence_weeks",
                "必须 >= 1",
            )
        )
    if "ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows" in val and _as_int(
        val.get("ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows",
                "必须 >= 1",
            )
        )
    if "ops_weekly_guardrail_compact_max_delete_rows" in val:
        raw = val.get("ops_weekly_guardrail_compact_max_delete_rows")
        raw_txt = str(raw).strip().lower()
        if raw not in {None, "", "none", "None"} and raw_txt not in {"none"} and _as_int(raw, 0) < 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_weekly_guardrail_compact_max_delete_rows",
                    "必须 >= 1 或为空",
                )
            )
    if "weekly_guardrail_history_limit" in val and _as_int(val.get("weekly_guardrail_history_limit", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.weekly_guardrail_history_limit", "必须 >= 1"))
    for key in (
        "ops_degradation_calibration_rollback_active_snapshot_path",
        "ops_degradation_calibration_rollback_history_dir",
    ):
        if key in val:
            raw = val.get(key)
            if not isinstance(raw, str):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须是字符串"))
            elif str(raw).strip() == "":
                issues.append(ValidationIssue("error", f"validation.{key}", "不能为空"))
    if "ops_degradation_calibration_window_days" in val and _as_int(
        val.get("ops_degradation_calibration_window_days", 0)
    ) < 3:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_window_days", "必须 >= 3")
        )
    if "ops_degradation_calibration_rollback_window_days" in val and _as_int(
        val.get("ops_degradation_calibration_rollback_window_days", 0)
    ) < 3:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_rollback_window_days", "必须 >= 3")
        )
    if "ops_degradation_calibration_rollback_recent_days" in val and _as_int(
        val.get("ops_degradation_calibration_rollback_recent_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_rollback_recent_days", "必须 >= 1")
        )
    if "ops_degradation_calibration_rollback_cooldown_days" in val and _as_int(
        val.get("ops_degradation_calibration_rollback_cooldown_days", -1)
    ) < 0:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_rollback_cooldown_days", "必须 >= 0")
        )
    if "ops_degradation_calibration_rollback_promotion_cooldown_days" in val and _as_int(
        val.get("ops_degradation_calibration_rollback_promotion_cooldown_days", -1)
    ) < 0:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_calibration_rollback_promotion_cooldown_days",
                "必须 >= 0",
            )
        )
    if "ops_degradation_calibration_rollback_hysteresis_window_days" in val and _as_int(
        val.get("ops_degradation_calibration_rollback_hysteresis_window_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue(
                "error",
                "validation.ops_degradation_calibration_rollback_hysteresis_window_days",
                "必须 >= 1",
            )
        )
    if "ops_degradation_calibration_rollback_window_days" in val and "ops_degradation_calibration_rollback_recent_days" in val:
        wd = _as_int(val.get("ops_degradation_calibration_rollback_window_days", 0))
        rd = _as_int(val.get("ops_degradation_calibration_rollback_recent_days", 0))
        if wd > 0 and rd > wd:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_degradation_calibration_rollback_recent_days",
                    "不能大于 ops_degradation_calibration_rollback_window_days",
                )
            )
    if "ops_degradation_calibration_min_samples" in val and _as_int(
        val.get("ops_degradation_calibration_min_samples", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_min_samples", "必须 >= 1")
        )
    if "ops_degradation_calibration_rollback_min_samples" in val and _as_int(
        val.get("ops_degradation_calibration_rollback_min_samples", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_rollback_min_samples", "必须 >= 1")
        )
    if "ops_degradation_calibration_rollback_stable_min_samples" in val and _as_int(
        val.get("ops_degradation_calibration_rollback_stable_min_samples", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_rollback_stable_min_samples", "必须 >= 1")
        )
    for key in (
        "ops_degradation_calibration_fp_target",
        "ops_degradation_calibration_fn_target",
        "ops_degradation_calibration_floor_ratio_min",
        "ops_degradation_calibration_floor_ratio_max",
        "ops_degradation_calibration_rollback_fn_rise_min",
        "ops_degradation_calibration_rollback_gate_fail_rise_min",
        "ops_degradation_calibration_rollback_stable_fn_rate_max",
        "ops_degradation_calibration_rollback_stable_gate_fail_ratio_max",
        "ops_degradation_calibration_rollback_trigger_hysteresis_buffer",
        "ops_degradation_calibration_rollback_stable_hysteresis_buffer",
    ):
        if key in val:
            ratio = _as_float(val.get(key, 0.0))
            if not (0.0 <= ratio <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 [0, 1]"))
    for key in (
        "ops_degradation_calibration_step_multiplier",
        "ops_degradation_calibration_step_floor_ratio",
    ):
        if key in val:
            ratio = _as_float(val.get(key, 0.0))
            if not (0.0 < ratio <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{key}", "必须在 (0, 1]"))
    if "ops_degradation_calibration_step_streak_days" in val and _as_int(
        val.get("ops_degradation_calibration_step_streak_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_step_streak_days", "必须 >= 1")
        )
    if "ops_degradation_calibration_multiplier_min" in val and _as_float(
        val.get("ops_degradation_calibration_multiplier_min", 0.0)
    ) < 1.0:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_multiplier_min", "必须 >= 1")
        )
    if "ops_degradation_calibration_multiplier_max" in val and _as_float(
        val.get("ops_degradation_calibration_multiplier_max", 0.0)
    ) < 1.0:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_multiplier_max", "必须 >= 1")
        )
    if "ops_degradation_calibration_multiplier_min" in val and "ops_degradation_calibration_multiplier_max" in val:
        min_mult = _as_float(val.get("ops_degradation_calibration_multiplier_min", 1.0))
        max_mult = _as_float(val.get("ops_degradation_calibration_multiplier_max", 1.0))
        if max_mult < min_mult:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_degradation_calibration_multiplier_max",
                    "不能小于 ops_degradation_calibration_multiplier_min",
                )
            )
    if "ops_degradation_calibration_floor_ratio_min" in val and "ops_degradation_calibration_floor_ratio_max" in val:
        min_ratio = _as_float(val.get("ops_degradation_calibration_floor_ratio_min", 0.0))
        max_ratio = _as_float(val.get("ops_degradation_calibration_floor_ratio_max", 0.0))
        if max_ratio < min_ratio:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_degradation_calibration_floor_ratio_max",
                    "不能小于 ops_degradation_calibration_floor_ratio_min",
                )
            )
    if "ops_degradation_calibration_streak_min" in val and _as_int(
        val.get("ops_degradation_calibration_streak_min", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_streak_min", "必须 >= 1")
        )
    if "ops_degradation_calibration_streak_max" in val and _as_int(
        val.get("ops_degradation_calibration_streak_max", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.ops_degradation_calibration_streak_max", "必须 >= 1")
        )
    if "ops_degradation_calibration_streak_min" in val and "ops_degradation_calibration_streak_max" in val:
        min_streak = _as_int(val.get("ops_degradation_calibration_streak_min", 1))
        max_streak = _as_int(val.get("ops_degradation_calibration_streak_max", 1))
        if max_streak < min_streak:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.ops_degradation_calibration_streak_max",
                    "不能小于 ops_degradation_calibration_streak_min",
                )
            )
    for k in ("mode_switch_max_rate", "ops_risk_multiplier_floor", "ops_risk_multiplier_drift_max", "ops_source_confidence_floor"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "broker_snapshot_source_mode" in val:
        mode = str(val.get("broker_snapshot_source_mode", "")).strip().lower()
        if mode not in {"paper_engine", "live_adapter", "hybrid_prefer_live"}:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.broker_snapshot_source_mode",
                    "必须是 paper_engine/live_adapter/hybrid_prefer_live",
                )
            )
    if "broker_snapshot_live_mapping_profile" in val:
        profile = str(val.get("broker_snapshot_live_mapping_profile", "")).strip().lower()
        if profile not in {"generic", "ibkr", "binance", "ctp"}:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.broker_snapshot_live_mapping_profile",
                    "必须是 generic/ibkr/binance/ctp",
                )
            )
    if "broker_snapshot_live_mapping_fields" in val:
        mapping = val.get("broker_snapshot_live_mapping_fields")
        if not isinstance(mapping, dict):
            issues.append(ValidationIssue("error", "validation.broker_snapshot_live_mapping_fields", "必须是字典"))
        else:
            for k in ("source", "open_positions", "closed_count", "closed_pnl", "positions"):
                if k in mapping and not isinstance(mapping.get(k), list):
                    issues.append(ValidationIssue("error", f"validation.broker_snapshot_live_mapping_fields.{k}", "必须是列表"))
            if "position_fields" in mapping:
                pf = mapping.get("position_fields")
                if not isinstance(pf, dict):
                    issues.append(ValidationIssue("error", "validation.broker_snapshot_live_mapping_fields.position_fields", "必须是字典"))
                else:
                    for f_key, f_value in pf.items():
                        if not isinstance(f_value, list):
                            issues.append(
                                ValidationIssue(
                                    "error",
                                    f"validation.broker_snapshot_live_mapping_fields.position_fields.{f_key}",
                                    "必须是列表",
                                )
                            )
    if "broker_snapshot_live_inbox" in val and str(val.get("broker_snapshot_live_inbox", "")).strip() == "":
        issues.append(ValidationIssue("error", "validation.broker_snapshot_live_inbox", "不能为空"))
    if "broker_snapshot_live_fallback_to_paper" in val and not isinstance(val.get("broker_snapshot_live_fallback_to_paper"), bool):
        issues.append(ValidationIssue("error", "validation.broker_snapshot_live_fallback_to_paper", "必须是布尔值"))
    if "test_all_timeout_seconds" in val and _as_int(val.get("test_all_timeout_seconds", 0)) < 30:
        issues.append(ValidationIssue("error", "validation.test_all_timeout_seconds", "必须 >= 30"))
    if "review_loop_fast_ratio" in val:
        rr = _as_float(val.get("review_loop_fast_ratio", 0.0))
        if not (0.0 < rr <= 1.0):
            issues.append(ValidationIssue("error", "validation.review_loop_fast_ratio", "必须在 (0, 1]"))
    if "review_loop_fast_shard_total" in val and _as_int(val.get("review_loop_fast_shard_total", 1)) < 1:
        issues.append(ValidationIssue("error", "validation.review_loop_fast_shard_total", "必须 >= 1"))
    if "review_loop_fast_shard_index" in val and _as_int(val.get("review_loop_fast_shard_index", 0)) < 0:
        issues.append(ValidationIssue("error", "validation.review_loop_fast_shard_index", "必须 >= 0"))
    if "review_loop_fast_shard_total" in val or "review_loop_fast_shard_index" in val:
        shard_total = _as_int(val.get("review_loop_fast_shard_total", 1))
        shard_index = _as_int(val.get("review_loop_fast_shard_index", 0))
        if shard_total > 0 and shard_index >= shard_total:
            issues.append(ValidationIssue("error", "validation.review_loop_fast_shard_index", "必须 < review_loop_fast_shard_total"))
    if "review_loop_timeout_fallback_enabled" in val and not isinstance(val.get("review_loop_timeout_fallback_enabled"), bool):
        issues.append(ValidationIssue("error", "validation.review_loop_timeout_fallback_enabled", "必须是布尔值"))
    if "review_loop_timeout_fallback_ratio" in val:
        rr = _as_float(val.get("review_loop_timeout_fallback_ratio", 0.0))
        if not (0.0 < rr <= 1.0):
            issues.append(ValidationIssue("error", "validation.review_loop_timeout_fallback_ratio", "必须在 (0, 1]"))
    if "review_loop_timeout_fallback_shard_total" in val and _as_int(val.get("review_loop_timeout_fallback_shard_total", 1)) < 1:
        issues.append(ValidationIssue("error", "validation.review_loop_timeout_fallback_shard_total", "必须 >= 1"))
    if "review_loop_timeout_fallback_shard_index" in val and _as_int(val.get("review_loop_timeout_fallback_shard_index", 0)) < 0:
        issues.append(ValidationIssue("error", "validation.review_loop_timeout_fallback_shard_index", "必须 >= 0"))
    if "review_loop_timeout_fallback_shard_total" in val or "review_loop_timeout_fallback_shard_index" in val:
        shard_total = _as_int(val.get("review_loop_timeout_fallback_shard_total", 1))
        shard_index = _as_int(val.get("review_loop_timeout_fallback_shard_index", 0))
        if shard_total > 0 and shard_index >= shard_total:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.review_loop_timeout_fallback_shard_index",
                    "必须 < review_loop_timeout_fallback_shard_total",
                )
            )
    if "review_loop_stress_matrix_autorun_enabled" in val and not isinstance(
        val.get("review_loop_stress_matrix_autorun_enabled"), bool
    ):
        issues.append(ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_enabled", "必须是布尔值"))
    if "review_loop_stress_matrix_autorun_on_mode_drift" in val and not isinstance(
        val.get("review_loop_stress_matrix_autorun_on_mode_drift"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_on_mode_drift", "必须是布尔值")
        )
    if "review_loop_stress_matrix_autorun_on_stress_breach" in val and not isinstance(
        val.get("review_loop_stress_matrix_autorun_on_stress_breach"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_on_stress_breach", "必须是布尔值")
        )
    if "review_loop_stress_matrix_autorun_max_runs" in val and _as_int(
        val.get("review_loop_stress_matrix_autorun_max_runs", -1)
    ) < 0:
        issues.append(ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_max_runs", "必须 >= 0"))
    if "review_loop_stress_matrix_autorun_cooldown_rounds" in val and _as_int(
        val.get("review_loop_stress_matrix_autorun_cooldown_rounds", -1)
    ) < 0:
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_cooldown_rounds", "必须 >= 0")
        )
    if "review_loop_stress_matrix_autorun_backoff_multiplier" in val:
        mult = _as_float(val.get("review_loop_stress_matrix_autorun_backoff_multiplier", 0.0))
        if not (mult >= 1.0):
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.review_loop_stress_matrix_autorun_backoff_multiplier",
                    "必须 >= 1.0",
                )
            )
    if "review_loop_stress_matrix_autorun_backoff_max_rounds" in val and _as_int(
        val.get("review_loop_stress_matrix_autorun_backoff_max_rounds", -1)
    ) < 0:
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_backoff_max_rounds", "必须 >= 0")
        )
    if "review_loop_stress_matrix_autorun_adaptive_enabled" in val and not isinstance(
        val.get("review_loop_stress_matrix_autorun_adaptive_enabled"), bool
    ):
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_adaptive_enabled", "必须是布尔值")
        )
    if "review_loop_stress_matrix_autorun_adaptive_window_days" in val and _as_int(
        val.get("review_loop_stress_matrix_autorun_adaptive_window_days", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_adaptive_window_days", "必须 >= 1")
        )
    if "review_loop_stress_matrix_autorun_adaptive_min_rounds" in val and _as_int(
        val.get("review_loop_stress_matrix_autorun_adaptive_min_rounds", 0)
    ) < 1:
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_adaptive_min_rounds", "必须 >= 1")
        )
    for k in (
        "review_loop_stress_matrix_autorun_adaptive_low_density_threshold",
        "review_loop_stress_matrix_autorun_adaptive_high_density_threshold",
    ):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if (
        "review_loop_stress_matrix_autorun_adaptive_low_density_threshold" in val
        and "review_loop_stress_matrix_autorun_adaptive_high_density_threshold" in val
    ):
        low_th = _as_float(val.get("review_loop_stress_matrix_autorun_adaptive_low_density_threshold", 0.0))
        high_th = _as_float(val.get("review_loop_stress_matrix_autorun_adaptive_high_density_threshold", 0.0))
        if low_th > high_th:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.review_loop_stress_matrix_autorun_adaptive_low_density_threshold",
                    "不能大于 review_loop_stress_matrix_autorun_adaptive_high_density_threshold",
                )
            )
    for k in (
        "review_loop_stress_matrix_autorun_adaptive_low_density_factor",
        "review_loop_stress_matrix_autorun_adaptive_high_density_factor",
    ):
        if k in val and _as_float(val.get(k, 0.0)) <= 0.0:
            issues.append(ValidationIssue("error", f"validation.{k}", "必须 > 0"))
    if "review_loop_stress_matrix_autorun_adaptive_min_runs_floor" in val and _as_int(
        val.get("review_loop_stress_matrix_autorun_adaptive_min_runs_floor", -1)
    ) < 0:
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_adaptive_min_runs_floor", "必须 >= 0")
        )
    if "review_loop_stress_matrix_autorun_adaptive_max_runs_cap" in val and _as_int(
        val.get("review_loop_stress_matrix_autorun_adaptive_max_runs_cap", -1)
    ) < 0:
        issues.append(
            ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_adaptive_max_runs_cap", "必须 >= 0")
        )
    if (
        "review_loop_stress_matrix_autorun_adaptive_min_runs_floor" in val
        and "review_loop_stress_matrix_autorun_adaptive_max_runs_cap" in val
    ):
        floor_runs = _as_int(val.get("review_loop_stress_matrix_autorun_adaptive_min_runs_floor", 0))
        cap_runs = _as_int(val.get("review_loop_stress_matrix_autorun_adaptive_max_runs_cap", 0))
        if cap_runs < floor_runs:
            issues.append(
                ValidationIssue(
                    "error",
                    "validation.review_loop_stress_matrix_autorun_adaptive_max_runs_cap",
                    "不能小于 review_loop_stress_matrix_autorun_adaptive_min_runs_floor",
                )
            )
    if "review_loop_stress_matrix_autorun_modes" in val:
        modes = val.get("review_loop_stress_matrix_autorun_modes")
        if not isinstance(modes, list):
            issues.append(ValidationIssue("error", "validation.review_loop_stress_matrix_autorun_modes", "必须是列表"))
        else:
            for i, item in enumerate(modes):
                if str(item).strip() == "":
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"validation.review_loop_stress_matrix_autorun_modes[{i}]",
                            "不能为空字符串",
                        )
                    )
    if "review_backtest_lookback_days" in val and _as_int(val.get("review_backtest_lookback_days", 0)) < 30:
        issues.append(ValidationIssue("error", "validation.review_backtest_lookback_days", "必须 >= 30"))
    if "review_backtest_start_date" in val:
        raw_start = str(val.get("review_backtest_start_date", "")).strip()
        try:
            date.fromisoformat(raw_start)
        except Exception:
            issues.append(ValidationIssue("error", "validation.review_backtest_start_date", "必须是 YYYY-MM-DD"))

    uni = settings.universe
    core = uni.get("core", [])
    if not isinstance(core, list) or not core:
        issues.append(ValidationIssue("error", "universe.core", "必须是非空列表"))
    else:
        for i, item in enumerate(core):
            if not isinstance(item, dict):
                issues.append(ValidationIssue("error", f"universe.core[{i}]", "必须是对象"))
                continue
            sym = str(item.get("symbol", "")).strip()
            if not (EQUITY_RE.match(sym) or FUTURE_RE.match(sym)):
                issues.append(ValidationIssue("error", f"universe.core[{i}].symbol", f"无效标的格式: {sym!r}"))
            if str(item.get("asset_class", "")).strip() == "":
                issues.append(ValidationIssue("warning", f"universe.core[{i}].asset_class", "建议显式配置 asset_class"))

    paths = settings.paths
    output = str(paths.get("output", "")).strip()
    sqlite = str(paths.get("sqlite", "")).strip()
    if output == "":
        issues.append(ValidationIssue("error", "paths.output", "不能为空"))
    if sqlite == "":
        issues.append(ValidationIssue("error", "paths.sqlite", "不能为空"))

    data_cfg = settings.raw.get("data", {}) if isinstance(settings.raw, dict) else {}
    provider_profile = str(data_cfg.get("provider_profile", "opensource_dual"))
    if provider_profile not in SUPPORTED_PROVIDER_PROFILES:
        issues.append(
            ValidationIssue(
                "error",
                "data.provider_profile",
                f"不支持的 provider_profile={provider_profile!r}，可选={sorted(SUPPORTED_PROVIDER_PROFILES)}",
            )
        )

    errors = [x for x in issues if x.level == "error"]
    warnings = [x for x in issues if x.level == "warning"]
    return {
        "ok": len(errors) == 0,
        "errors": [x.to_dict() for x in errors],
        "warnings": [x.to_dict() for x in warnings],
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
        },
    }


def assert_valid_settings(settings: SystemSettings) -> None:
    result = validate_settings(settings)
    if result["ok"]:
        return
    lines = ["配置校验失败："]
    for item in result.get("errors", []):
        lines.append(f"- [{item['path']}] {item['message']}")
    raise ValueError("\n".join(lines))
