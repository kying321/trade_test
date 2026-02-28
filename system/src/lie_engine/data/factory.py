from __future__ import annotations

from typing import Any

from lie_engine.data.providers import (
    OpenSourcePrimaryProvider,
    OpenSourceSecondaryProvider,
    PaidProviderPlaceholder,
    TushareBinanceHybridProvider,
)


SUPPORTED_PROVIDER_PROFILES = {
    "opensource_dual",
    "opensource_primary",
    "tushare_binance_hybrid",
    "hybrid_with_paid_placeholder",
    "paid_placeholder",
}


def build_provider_stack(profile: str | None = None) -> list[Any]:
    p = str(profile or "opensource_dual").strip().lower()
    if p == "opensource_dual":
        return [OpenSourcePrimaryProvider(), OpenSourceSecondaryProvider()]
    if p == "opensource_primary":
        return [OpenSourcePrimaryProvider()]
    if p == "tushare_binance_hybrid":
        return [TushareBinanceHybridProvider()]
    if p == "hybrid_with_paid_placeholder":
        return [OpenSourcePrimaryProvider(), OpenSourceSecondaryProvider(), PaidProviderPlaceholder()]
    if p == "paid_placeholder":
        return [PaidProviderPlaceholder()]
    raise ValueError(f"Unsupported data.provider_profile: {profile!r}")
