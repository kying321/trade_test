from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lie_engine" / "research" / "axios_site_client.py"


def load_module():
    spec = importlib.util.spec_from_file_location("axios_site_client_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyResponse(io.BytesIO):
    def __init__(self, text: str):
        super().__init__(text.encode("utf-8"))
        self.headers = {}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_client_parses_news_sitemap_with_titles_and_keywords(monkeypatch) -> None:
    mod = load_module()
    news_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url>
    <loc>https://www.axios.com/2026/04/06/example-story</loc>
    <news:news>
      <news:publication>
        <news:name>Axios</news:name>
        <news:language>en</news:language>
      </news:publication>
      <news:publication_date>2026-04-06T13:15:05+00:00</news:publication_date>
      <news:title>Example headline</news:title>
      <news:keywords>Markets,Oil,Gold</news:keywords>
    </news:news>
  </url>
</urlset>"""

    monkeypatch.setattr(
        mod.urllib.request,
        "urlopen",
        lambda request, timeout=0, context=None: DummyResponse(news_xml),
    )

    client = mod.AxiosSiteClient()
    rows = client.fetch_news_entries(limit=10)

    assert len(rows) == 1
    assert rows[0]["title"] == "Example headline"
    assert rows[0]["url"] == "https://www.axios.com/2026/04/06/example-story"
    assert rows[0]["published_at"] == "2026-04-06T13:15:05+00:00"
    assert rows[0]["keywords"] == ["Markets", "Oil", "Gold"]
    assert rows[0]["is_local"] is False


def test_client_builds_summary_from_news_entries() -> None:
    mod = load_module()
    summary = mod.build_summary(
        [
            {
                "title": "Oil market braces for inventory shock",
                "url": "https://www.axios.com/2026/04/06/oil-shock",
                "published_at": "2026-04-06T13:15:05+00:00",
                "keywords": ["Oil", "Markets"],
                "is_local": False,
            },
            {
                "title": "Phoenix growth slows",
                "url": "https://www.axios.com/local/phoenix/2026/04/06/growth-slows",
                "published_at": "2026-04-06T12:15:05+00:00",
                "keywords": ["Phoenix", "Economy"],
                "is_local": True,
            },
        ]
    )

    assert summary["news_total"] == 2
    assert summary["local_total"] == 1
    assert summary["national_total"] == 1
    assert summary["top_titles"] == ["Oil market braces for inventory shock", "Phoenix growth slows"]
    assert summary["top_keywords"] == ["Oil", "Markets", "Phoenix", "Economy"]
    assert summary["takeaway"] == "Oil market braces for inventory shock"
