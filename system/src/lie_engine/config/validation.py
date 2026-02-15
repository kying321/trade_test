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
    if "ops_slot_window_days" in val and _as_int(val.get("ops_slot_window_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_slot_window_days", "必须 >= 1"))
    if "ops_slot_min_samples" in val and _as_int(val.get("ops_slot_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_slot_min_samples", "必须 >= 1"))
    for k in (
        "ops_slot_missing_ratio_max",
        "ops_slot_premarket_anomaly_ratio_max",
        "ops_slot_intraday_anomaly_ratio_max",
        "ops_slot_eod_anomaly_ratio_max",
        "ops_slot_source_confidence_floor",
        "ops_slot_risk_multiplier_floor",
    ):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_reconcile_window_days" in val and _as_int(val.get("ops_reconcile_window_days", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_window_days", "必须 >= 1"))
    if "ops_reconcile_min_samples" in val and _as_int(val.get("ops_reconcile_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_min_samples", "必须 >= 1"))
    for k in (
        "ops_reconcile_missing_ratio_max",
        "ops_reconcile_plan_gap_ratio_max",
        "ops_reconcile_closed_count_gap_ratio_max",
        "ops_reconcile_open_gap_ratio_max",
    ):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
    if "ops_reconcile_closed_pnl_gap_abs_max" in val and _as_float(val.get("ops_reconcile_closed_pnl_gap_abs_max", 0.0)) < 0.0:
        issues.append(ValidationIssue("error", "validation.ops_reconcile_closed_pnl_gap_abs_max", "必须 >= 0"))
    if "ops_state_min_samples" in val and _as_int(val.get("ops_state_min_samples", 0)) < 1:
        issues.append(ValidationIssue("error", "validation.ops_state_min_samples", "必须 >= 1"))
    if "ops_mode_health_fail_days_max" in val and _as_int(val.get("ops_mode_health_fail_days_max", 0)) < 0:
        issues.append(ValidationIssue("error", "validation.ops_mode_health_fail_days_max", "必须 >= 0"))
    for k in ("mode_switch_max_rate", "ops_risk_multiplier_floor", "ops_risk_multiplier_drift_max", "ops_source_confidence_floor"):
        if k in val:
            v = _as_float(val.get(k, 0.0))
            if not (0.0 <= v <= 1.0):
                issues.append(ValidationIssue("error", f"validation.{k}", "必须在 [0, 1]"))
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
