from __future__ import annotations

from typing import Any

from lie_engine.data.providers import (
    BinanceSpotPublicProvider,
    BybitSpotPublicProvider,
    OpenSourcePrimaryProvider,
    OpenSourceSecondaryProvider,
    PaidProviderPlaceholder,
    PublicInternetResearchProvider,
)


SUPPORTED_PROVIDER_PROFILES = {
    "opensource_dual",
    "opensource_primary",
    "binance_spot_public",
    "bybit_spot_public",
    "dual_binance_bybit_public",
    "public_research_binance_bybit",
    "hybrid_opensource_binance",
    "hybrid_opensource_binance_bybit",
    "hybrid_with_paid_placeholder",
    "paid_placeholder",
}


def build_provider_stack(profile: str | None = None) -> list[Any]:
    p = str(profile or "opensource_dual").strip().lower()
    if p == "opensource_dual":
        return [OpenSourcePrimaryProvider(), OpenSourceSecondaryProvider()]
    if p == "opensource_primary":
        return [OpenSourcePrimaryProvider()]
    if p == "binance_spot_public":
        return [BinanceSpotPublicProvider()]
    if p == "bybit_spot_public":
        return [BybitSpotPublicProvider()]
    if p == "dual_binance_bybit_public":
        return [BinanceSpotPublicProvider(), BybitSpotPublicProvider()]
    if p == "public_research_binance_bybit":
        return [PublicInternetResearchProvider(), BinanceSpotPublicProvider(), BybitSpotPublicProvider()]
    if p == "hybrid_opensource_binance":
        return [OpenSourcePrimaryProvider(), BinanceSpotPublicProvider()]
    if p == "hybrid_opensource_binance_bybit":
        return [OpenSourcePrimaryProvider(), BinanceSpotPublicProvider(), BybitSpotPublicProvider()]
    if p == "hybrid_with_paid_placeholder":
        return [OpenSourcePrimaryProvider(), OpenSourceSecondaryProvider(), PaidProviderPlaceholder()]
    if p == "paid_placeholder":
        return [PaidProviderPlaceholder()]
    raise ValueError(f"Unsupported data.provider_profile: {profile!r}")
