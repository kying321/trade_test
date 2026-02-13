from __future__ import annotations

from dataclasses import dataclass

from lie_engine.models import RegimeLabel, RiskBudget, Side, SignalCandidate, TradePlan
from lie_engine.risk.kelly import compute_kelly_fraction


@dataclass(slots=True)
class RiskManager:
    max_single_risk_pct: float
    max_total_exposure_pct: float
    max_symbol_pct: float
    max_theme_pct: float

    def build_budget(self, account_equity: float, used_exposure_pct: float) -> RiskBudget:
        return RiskBudget(
            account_equity=account_equity,
            max_single_risk_pct=self.max_single_risk_pct,
            max_total_exposure_pct=self.max_total_exposure_pct,
            max_symbol_pct=self.max_symbol_pct,
            max_theme_pct=self.max_theme_pct,
            used_exposure_pct=used_exposure_pct,
        )

    def size_signal(
        self,
        signal: SignalCandidate,
        win_rate: float,
        payoff: float,
        budget: RiskBudget,
        symbol_exposure_pct: float,
        theme_exposure_pct: float,
        protection_mode: bool,
    ) -> TradePlan | None:
        if budget.available_exposure_pct <= 0:
            return None

        if signal.side == Side.FLAT:
            return None

        kelly = compute_kelly_fraction(win_rate=win_rate, payoff=payoff)
        confidence_factor = max(0.0, min(1.0, signal.confidence / 100.0))
        raw_size_pct = 100.0 * 0.5 * kelly * confidence_factor

        if protection_mode:
            raw_size_pct *= 0.5

        size_pct = min(raw_size_pct, budget.available_exposure_pct)
        size_pct = min(size_pct, self.max_symbol_pct - symbol_exposure_pct)
        size_pct = min(size_pct, self.max_theme_pct - theme_exposure_pct)

        if size_pct <= 0:
            return None

        per_unit_risk_pct = abs(signal.entry_price - signal.stop_price) / max(signal.entry_price, 1e-9) * 100.0
        risk_pct = per_unit_risk_pct * (size_pct / 100.0)
        if risk_pct > self.max_single_risk_pct:
            scale = self.max_single_risk_pct / max(risk_pct, 1e-9)
            size_pct *= scale
            risk_pct = self.max_single_risk_pct

        if size_pct <= 0.05:
            return None

        status = "ACTIVE"
        hedge_leg = None
        reason = f"½Kelly + 置信度({signal.confidence:.1f}%) + 风控约束"

        if signal.side == Side.SHORT and not signal.can_short:
            status = "TRANSLATED"
            hedge_leg = "Index hedge via IF/IH or inverse ETF"
            reason = (signal.notes or "") + "；" + reason

        if signal.regime in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL}:
            status = "BLOCKED"
            return None

        return TradePlan(
            symbol=signal.symbol,
            side=signal.side,
            size_pct=float(size_pct),
            risk_pct=float(risk_pct),
            entry_price=float(signal.entry_price),
            stop_price=float(signal.stop_price),
            target_price=float(signal.target_price),
            hedge_leg=hedge_leg,
            reason=reason,
            status=status,
        )
