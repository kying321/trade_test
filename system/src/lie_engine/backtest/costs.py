from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AssetCost:
    fee_bps: float
    slippage_bps: float
    impact_bps: float
    borrow_bps_daily: float

    @property
    def roundtrip_bps(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps + self.impact_bps)


def default_cost_model(asset_class: str) -> AssetCost:
    if asset_class == "future":
        return AssetCost(fee_bps=1.2, slippage_bps=0.8, impact_bps=1.2, borrow_bps_daily=0.0)
    if asset_class == "option":
        return AssetCost(fee_bps=2.0, slippage_bps=2.0, impact_bps=1.5, borrow_bps_daily=0.0)
    if asset_class == "etf":
        return AssetCost(fee_bps=1.0, slippage_bps=1.2, impact_bps=0.8, borrow_bps_daily=0.0)
    if asset_class == "hedge":
        return AssetCost(fee_bps=1.2, slippage_bps=1.5, impact_bps=1.0, borrow_bps_daily=0.0)
    # equity default
    return AssetCost(fee_bps=1.5, slippage_bps=2.0, impact_bps=1.5, borrow_bps_daily=2.0)
