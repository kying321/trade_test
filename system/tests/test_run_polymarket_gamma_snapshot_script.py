from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_polymarket_gamma_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_polymarket_gamma_snapshot_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_snapshot_writes_public_sentiment_artifact(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fetch_active_markets(self, limit: int = 20):
            return [
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
            ]

    monkeypatch.setattr(mod, "PolymarketGammaClient", FakeClient)
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 4, 6, 15, 20, tzinfo=mod.dt.timezone.utc))

    result = mod.run_snapshot(workspace=workspace, limit=20)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["change_class"] == "RESEARCH_ONLY"
    assert result["mode"] == "polymarket_gamma_snapshot"
    assert result["summary"]["markets_total"] == 2
    assert result["summary"]["bullish_count"] == 0
    assert result["summary"]["bearish_count"] == 1
    assert result["takeaway"] == "Will BTC hit $120k in April?"
    assert Path(result["artifact_json"]).exists()
    assert Path(result["artifact_md"]).exists()
    payload = json.loads(Path(result["artifact_json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["top_categories"] == ["Crypto", "Economy"]
