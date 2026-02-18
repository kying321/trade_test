from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    CASH = "cash"
    HEDGE = "hedge"


class RegimeLabel(str, Enum):
    STRONG_TREND = "强趋势"
    WEAK_TREND = "弱趋势"
    RANGE = "震荡"
    DOWNTREND = "下跌趋势"
    UNCERTAIN = "不确定"
    EXTREME_VOL = "极端波动"


@dataclass(slots=True)
class MarketBar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    asset_class: AssetClass
    contract_month: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ts"] = self.ts.isoformat()
        data["asset_class"] = self.asset_class.value
        return data


@dataclass(slots=True)
class NewsEvent:
    event_id: str
    ts: datetime
    title: str
    content: str
    lang: str
    source: str
    category: str
    confidence: float
    entities: list[str] = field(default_factory=list)
    importance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ts"] = self.ts.isoformat()
        return data


@dataclass(slots=True)
class RegimeState:
    as_of: date
    hurst: float
    hmm_probs: dict[str, float]
    atr_z: float
    consensus: RegimeLabel
    protection_mode: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        data["consensus"] = self.consensus.value
        return data


@dataclass(slots=True)
class SignalCandidate:
    symbol: str
    side: Side
    regime: RegimeLabel
    position_score: float
    structure_score: float
    momentum_score: float
    confidence: float
    convexity_ratio: float
    entry_price: float
    stop_price: float
    target_price: float
    can_short: bool
    factor_exposure_score: float = 0.0
    factor_penalty: float = 0.0
    factor_flags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = self.side.value
        data["regime"] = self.regime.value
        return data


@dataclass(slots=True)
class TradePlan:
    symbol: str
    side: Side
    size_pct: float
    risk_pct: float
    entry_price: float
    stop_price: float
    target_price: float
    hedge_leg: str | None
    reason: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = self.side.value
        return data


@dataclass(slots=True)
class RiskBudget:
    account_equity: float
    max_single_risk_pct: float
    max_total_exposure_pct: float
    max_symbol_pct: float
    max_theme_pct: float
    used_exposure_pct: float

    @property
    def available_exposure_pct(self) -> float:
        return max(0.0, self.max_total_exposure_pct - self.used_exposure_pct)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["available_exposure_pct"] = self.available_exposure_pct
        return data


@dataclass(slots=True)
class BacktestResult:
    start: date
    end: date
    total_return: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    expectancy: float
    trades: int
    violations: int
    positive_window_ratio: float
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    by_asset: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"] = self.start.isoformat()
        data["end"] = self.end.isoformat()
        return data


@dataclass(slots=True)
class ReviewDelta:
    as_of: date
    parameter_changes: dict[str, float]
    factor_weights: dict[str, float]
    defects: list[str]
    pass_gate: bool
    notes: list[str] = field(default_factory=list)
    change_reasons: dict[str, str] = field(default_factory=dict)
    factor_contrib_120d: dict[str, float] = field(default_factory=dict)
    style_diagnostics: dict[str, Any] = field(default_factory=dict)
    impact_window_days: int = 120
    rollback_anchor: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        return data
