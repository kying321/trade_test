from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any


GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _yes_price(outcomes: list[Any], prices: list[Any]) -> float | None:
    labels = [str(item or "").strip().lower() for item in outcomes]
    normalized_prices = [_safe_float(item) for item in prices]
    if len(labels) != len(normalized_prices) or not normalized_prices:
        return None
    if "yes" in labels:
        index = labels.index("yes")
        return normalized_prices[index]
    return None


class PolymarketGammaClient:
    def __init__(self, *, timeout_seconds: float = 5.0, user_agent: str = "Mozilla/5.0") -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.user_agent = str(user_agent).strip() or "Mozilla/5.0"
        self._ssl_context = ssl.create_default_context()

    def _fetch_json(self, url: str) -> list[dict[str, Any]]:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []

    def fetch_active_markets(self, limit: int = 20) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode(
            {
                "limit": max(int(limit), 20),
                "active": "true",
                "closed": "false",
            }
        )
        rows = self._fetch_json(f"{GAMMA_MARKETS_URL}?{query}")
        normalized: list[dict[str, Any]] = []
        for row in rows:
            outcomes = _safe_json_list(row.get("outcomes"))
            prices = _safe_json_list(row.get("outcomePrices"))
            slug = str(row.get("slug") or "").strip()
            normalized.append(
                {
                    "question": str(row.get("question") or "").strip(),
                    "slug": slug,
                    "url": f"https://polymarket.com/event/{slug}" if slug else "",
                    "category": str(row.get("category") or "").strip(),
                    "volume24hr": _safe_float(row.get("volume24hr") or row.get("volume24hrClob") or row.get("volume")),
                    "liquidity": _safe_float(row.get("liquidityNum") or row.get("liquidityClob") or row.get("liquidity")),
                    "yes_price": _yes_price(outcomes, prices),
                    "end_date": str(row.get("endDate") or "").strip(),
                }
            )
        normalized.sort(key=lambda row: row.get("volume24hr") or 0.0, reverse=True)
        return normalized[: max(1, int(limit))]


def build_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    category_counter: Counter[str] = Counter()
    top_titles: list[str] = []
    total_volume_24hr = 0.0
    bullish_count = 0
    bearish_count = 0
    binary_prices: list[float] = []

    for row in entries:
        category = str(row.get("category") or "").strip()
        if category:
            category_counter.update([category])
        question = str(row.get("question") or "").strip()
        if question:
            top_titles.append(question)
        volume = _safe_float(row.get("volume24hr"))
        total_volume_24hr += volume
        yes_price = row.get("yes_price")
        if isinstance(yes_price, (int, float)):
            binary_prices.append(float(yes_price))
            if float(yes_price) >= 0.7:
                bullish_count += 1
            if float(yes_price) <= 0.3:
                bearish_count += 1

    yes_price_avg = round(sum(binary_prices) / len(binary_prices), 4) if binary_prices else 0.0
    return {
        "markets_total": len(entries),
        "binary_markets_total": len(binary_prices),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "yes_price_avg": yes_price_avg,
        "total_volume_24hr": round(total_volume_24hr, 4),
        "top_categories": [category for category, _count in category_counter.most_common(5)],
        "top_titles": top_titles[:5],
        "takeaway": top_titles[0] if top_titles else "",
    }
