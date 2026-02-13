from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from lie_engine.models import BacktestResult, ReviewDelta


@dataclass(slots=True)
class ReviewThresholds:
    positive_window_ratio_min: float
    max_drawdown_max: float


def bounded_bayesian_update(current: float, evidence: float, low: float, high: float, step: float = 0.12) -> float:
    prior = np.clip(current, low, high)
    target = np.clip(evidence, low, high)
    posterior = (1.0 - step) * prior + step * target
    return float(np.clip(posterior, low, high))


def renormalize_factor_weights(base_weights: dict[str, float], contributions: dict[str, float]) -> dict[str, float]:
    weights = {}
    for key, base in base_weights.items():
        contrib = max(0.0, contributions.get(key, base))
        weights[key] = bounded_bayesian_update(base, contrib, 0.01, 1.0, step=0.2)
    s = sum(weights.values())
    if s <= 1e-9:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: float(v / s) for k, v in weights.items()}


def build_review_delta(
    as_of: date,
    backtest: BacktestResult,
    current_params: dict[str, float],
    factor_weights: dict[str, float],
    factor_contrib: dict[str, float],
    thresholds: ReviewThresholds,
) -> ReviewDelta:
    defects: list[str] = []
    notes: list[str] = []

    pass_gate = True
    if backtest.positive_window_ratio < thresholds.positive_window_ratio_min:
        defects.append("MODEL_POSITIVE_WINDOW_RATIO_BELOW_GATE")
        pass_gate = False
    if backtest.max_drawdown > thresholds.max_drawdown_max:
        defects.append("RISK_MAX_DRAWDOWN_ABOVE_GATE")
        pass_gate = False
    if backtest.violations > 0:
        defects.append("RISK_POLICY_VIOLATION")
        pass_gate = False

    wr = bounded_bayesian_update(current_params.get("win_rate", 0.45), backtest.win_rate, 0.2, 0.8)
    pf = bounded_bayesian_update(current_params.get("payoff", 2.0), backtest.profit_factor, 0.8, 5.0)
    current_conf = float(current_params.get("signal_confidence_min", 60.0))
    if pass_gate:
        conf_min = bounded_bayesian_update(
            current_conf,
            65.0 if backtest.expectancy > 0 else 70.0,
            50.0,
            90.0,
            step=0.08,
        )
    else:
        # Recovery mode: apply bounded shrink toward a recoverable confidence band.
        # This avoids runaway tightening (e.g. jumping to 90) that can lock the system.
        repair_target = 62.0 if backtest.expectancy > 0 else 68.0
        conf_min = bounded_bayesian_update(
            current_conf,
            repair_target,
            50.0,
            90.0,
            step=0.55,
        )
        conf_min = float(np.clip(conf_min, 55.0, 70.0))

    parameter_changes = {
        "win_rate": wr,
        "payoff": pf,
        "signal_confidence_min": conf_min,
    }
    change_reasons = {
        "win_rate": "基于滚动样本外胜率做有界贝叶斯更新，避免过拟合漂移",
        "payoff": "基于利润因子收缩更新，抑制极端窗口的偶然性",
        "signal_confidence_min": "门槛通过时小步调整；失败时进入恢复带(55-70)并触发再验证",
    }

    weights = renormalize_factor_weights(factor_weights, factor_contrib)

    if pass_gate:
        notes.append("All review gates passed")
    else:
        notes.append("Gates failed; keep protection bias and trigger re-test")

    return ReviewDelta(
        as_of=as_of,
        parameter_changes=parameter_changes,
        factor_weights=weights,
        defects=defects,
        pass_gate=pass_gate,
        notes=notes,
        change_reasons=change_reasons,
        factor_contrib_120d={k: float(v) for k, v in factor_contrib.items()},
        impact_window_days=120,
    )
