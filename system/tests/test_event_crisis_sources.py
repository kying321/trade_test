from __future__ import annotations

import datetime as dt
from lie_engine.research import event_crisis_sources


def test_normalize_public_event_rows_keeps_timestamp_and_classes() -> None:
    now = dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc)
    rows = event_crisis_sources.normalize_public_event_rows(
        [
            {
                "event_id": "evt-1",
                "event_ts": "2026-03-25T12:00:00Z",
                "event_classes": ["credit_deterioration"],
            }
        ],
        default_ts=now,
    )

    assert rows[0]["event_ts_utc"].endswith("Z")
    assert rows[0]["event_ts_utc"] == "2026-03-25T12:00:00Z"
    assert rows[0]["event_classes"] == ["credit_deterioration"]


def test_default_priority_assets_include_core_assets() -> None:
    assets = set(event_crisis_sources.DEFAULT_PRIORITY_ASSETS)
    expected = {
        "BTC",
        "ETH",
        "SOL",
        "BNB",
        "GOLD",
        "UST_LONG",
        "OIL",
        "BANKS",
        "HIGH_YIELD",
    }
    assert expected.issubset(assets)


def test_normalize_public_event_rows_string_fields() -> None:
    now = dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc)
    rows = event_crisis_sources.normalize_public_event_rows(
        [
            {
                "event_id": "evt-2",
                "event_ts": "2026-03-25T12:00:00Z",
                "event_classes": "credit_deterioration",
                "regions": "global",
                "affected_assets": "BTC",
            }
        ],
        default_ts=now,
    )

    assert rows[0]["event_classes"] == ["credit_deterioration"]
    assert rows[0]["regions"] == ["global"]
    assert rows[0]["affected_assets"] == ["BTC"]


def test_normalize_market_inputs_handles_none() -> None:
    result = event_crisis_sources.normalize_market_inputs(
        {"credit_liquidity_stress_score": None, "breadth_score": ""}
    )
    assert result["credit_liquidity_stress_score"] == 0.45
    assert result["breadth_score"] == 0.3
