from __future__ import annotations

import ssl
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any


SITEMAP_INDEX_URL = "https://www.axios.com/sitemap.xml"
NEWS_SITEMAP_URL = "https://www.axios.com/sitemaps/news.xml"
LAST200_SITEMAP_URL = "https://www.axios.com/sitemaps/last200.xml"
XML_NAMESPACES = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


class AxiosSiteClient:
    def __init__(self, *, timeout_seconds: float = 5.0, user_agent: str = "Mozilla/5.0") -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.user_agent = str(user_agent).strip() or "Mozilla/5.0"
        self._ssl_context = ssl.create_default_context()

    def _fetch_text(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="ignore")

    def fetch_news_entries(self, limit: int = 20) -> list[dict[str, Any]]:
        text = self._fetch_text(NEWS_SITEMAP_URL)
        root = ET.fromstring(text)
        rows: list[dict[str, Any]] = []
        for node in root.findall("sm:url", XML_NAMESPACES):
            loc = (node.findtext("sm:loc", default="", namespaces=XML_NAMESPACES) or "").strip()
            news_node = node.find("news:news", XML_NAMESPACES)
            title = ""
            published_at = ""
            keywords: list[str] = []
            if news_node is not None:
                title = (news_node.findtext("news:title", default="", namespaces=XML_NAMESPACES) or "").strip()
                published_at = (
                    news_node.findtext("news:publication_date", default="", namespaces=XML_NAMESPACES) or ""
                ).strip()
                raw_keywords = (
                    news_node.findtext("news:keywords", default="", namespaces=XML_NAMESPACES) or ""
                ).strip()
                keywords = [part.strip() for part in raw_keywords.split(",") if part.strip()]
            if not loc or not title:
                continue
            rows.append(
                {
                    "title": title,
                    "url": loc,
                    "published_at": published_at,
                    "keywords": keywords,
                    "is_local": "/local/" in loc,
                }
            )
            if len(rows) >= int(limit):
                break
        return rows


def build_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    keyword_counter: Counter[str] = Counter()
    for row in entries:
        keyword_counter.update([str(keyword).strip() for keyword in list(row.get("keywords") or []) if str(keyword).strip()])

    top_titles = [str(row.get("title") or "").strip() for row in entries[:5] if str(row.get("title") or "").strip()]
    local_total = sum(1 for row in entries if bool(row.get("is_local")))
    national_total = max(0, len(entries) - local_total)
    takeaway = top_titles[0] if top_titles else ""
    return {
        "news_total": len(entries),
        "local_total": local_total,
        "national_total": national_total,
        "top_titles": top_titles,
        "top_keywords": [keyword for keyword, _count in keyword_counter.most_common(5)],
        "takeaway": takeaway,
    }
