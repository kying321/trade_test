from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

STRATEGY_CONFIG = {
    "tv_basis_btc_spot_perp_v1": {"symbol": "BTCUSDT"},
}
VALID_STRATEGY_IDS = set(STRATEGY_CONFIG)
VALID_EVENT_TYPES = {"entry_check", "exit_check"}


@dataclass(frozen=True)
class TvBasisWebhookSignal:
    strategy_id: str
    symbol: str
    event_type: str
    tv_timestamp: str
    alert_id: str | None = None


def _require_text(payload: Dict[str, Any], key: str) -> str:
    raw = payload.get(key)
    if raw is None:
        raise ValueError(f"missing {key}")
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{key} must be non-empty")
    return text


def _optional_text(payload: Dict[str, Any], key: str) -> str | None:
    raw = payload.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def parse_tv_basis_webhook_payload(payload: Dict[str, Any]) -> TvBasisWebhookSignal:
    strategy_id = _require_text(payload, "strategy_id")
    if strategy_id not in VALID_STRATEGY_IDS:
        raise ValueError(f"unknown strategy_id:{strategy_id}")
    symbol_override = _optional_text(payload, "symbol")
    event_type = _require_text(payload, "event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"invalid event_type:{event_type}")
    tv_timestamp = _require_text(payload, "tv_timestamp")
    alert_id = _optional_text(payload, "alert_id")
    symbol = STRATEGY_CONFIG[strategy_id]["symbol"].upper()
    if symbol_override is not None and symbol_override.upper() != symbol:
        raise ValueError(f"symbol mismatch:{symbol_override}")
    return TvBasisWebhookSignal(
        strategy_id=strategy_id,
        symbol=symbol,
        event_type=event_type,
        tv_timestamp=tv_timestamp,
        alert_id=alert_id,
    )
