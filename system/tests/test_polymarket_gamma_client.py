from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lie_engine" / "research" / "polymarket_gamma_client.py"


def load_module():
    spec = importlib.util.spec_from_file_location("polymarket_gamma_client_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyResponse(io.BytesIO):
    def __init__(self, payload: object):
        super().__init__(json.dumps(payload).encode("utf-8"))
        self.headers = {}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_client_parses_active_markets_and_extracts_yes_price(monkeypatch) -> None:
    mod = load_module()
    payload = [
        {
            "question": "Will BTC hit $120k in April?",
            "url": "https://polymarket.com/event/btc-120k-april",
            "slug": "btc-120k-april",
            "category": "Crypto",
            "volume24hr": 120034.5,
            "liquidity": "80000.12",
            "outcomes": "[\"Yes\", \"No\"]",
            "outcomePrices": "[\"0.63\", \"0.37\"]",
            "endDate": "2026-04-30T00:00:00Z",
        },
        {
            "question": "Will the Fed cut before June?",
            "slug": "fed-cut-before-june",
            "category": "Economy",
            "volume24hr": 98321.1,
            "liquidity": "65000.00",
            "outcomes": "[\"Yes\", \"No\"]",
            "outcomePrices": "[\"0.28\", \"0.72\"]",
            "endDate": "2026-06-30T00:00:00Z",
        },
    ]

    monkeypatch.setattr(
        mod.urllib.request,
        "urlopen",
        lambda request, timeout=0, context=None: DummyResponse(payload),
    )

    client = mod.PolymarketGammaClient()
    rows = client.fetch_active_markets(limit=10)

    assert len(rows) == 2
    assert rows[0]["question"] == "Will BTC hit $120k in April?"
    assert rows[0]["yes_price"] == 0.63
    assert rows[0]["volume24hr"] == 120034.5
    assert rows[1]["question"] == "Will the Fed cut before June?"
    assert rows[1]["yes_price"] == 0.28
    assert rows[1]["category"] == "Economy"


def test_build_summary_rolls_up_sentiment_counts() -> None:
    mod = load_module()
    summary = mod.build_summary(
        [
            {
                "question": "Will BTC hit $120k in April?",
                "slug": "btc-120k-april",
                "category": "Crypto",
                "volume24hr": 120034.5,
                "liquidity": 80000.12,
                "yes_price": 0.63,
            },
            {
                "question": "Will the Fed cut before June?",
                "slug": "fed-cut-before-june",
                "category": "Economy",
                "volume24hr": 98321.1,
                "liquidity": 65000.0,
                "yes_price": 0.28,
            },
            {
                "question": "Will Trump visit China in 2026?",
                "slug": "trump-china-visit-2026",
                "category": "Politics",
                "volume24hr": 54000.0,
                "liquidity": 40000.0,
                "yes_price": 0.78,
            },
        ]
    )

    assert summary["markets_total"] == 3
    assert summary["binary_markets_total"] == 3
    assert summary["bullish_count"] == 1
    assert summary["bearish_count"] == 1
    assert summary["yes_price_avg"] == 0.5633
    assert summary["total_volume_24hr"] == 272355.6
    assert summary["top_categories"] == ["Crypto", "Economy", "Politics"]
    assert summary["top_titles"] == [
        "Will BTC hit $120k in April?",
        "Will the Fed cut before June?",
        "Will Trump visit China in 2026?",
    ]
    assert summary["takeaway"] == "Will BTC hit $120k in April?"
