from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_hot_research_universe.py"
)
SPEC = importlib.util.spec_from_file_location("build_hot_research_universe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_hot_research_universe_offline_builds_artifact(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--network-mode",
            "offline",
            "--crypto-limit",
            "5",
            "--now",
            "2026-03-10T10:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["source_tier"] == "static_fallback"
    assert payload["network_mode"] == "offline"
    assert payload["crypto"]["selected"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    assert payload["commodities"]["selected"] == ["XAUUSD", "XAGUSD", "WTIUSD", "BRENTUSD", "NATGAS", "COPPER"]
    assert payload["batches"]["crypto_hot"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    assert payload["batches"]["crypto_majors"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    assert "crypto_beta" not in payload["batches"]
    assert payload["batches"]["precious_metals"] == ["XAUUSD", "XAGUSD"]
    assert payload["batches"]["energy"] == ["WTIUSD", "BRENTUSD", "NATGAS"]
    assert payload["batches"]["energy_liquids"] == ["WTIUSD", "BRENTUSD"]
    assert payload["batches"]["energy_gas"] == ["NATGAS"]
    assert payload["batches"]["base_metals"] == ["COPPER"]
    assert payload["batches"]["metals_all"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["batches"]["metals_macro"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["batches"]["mixed_macro"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "XAUUSD", "XAGUSD", "WTIUSD", "BRENTUSD"]
    assert payload["batches"]["mixed_macro_expanded"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "XAUUSD",
        "XAGUSD",
        "WTIUSD",
        "BRENTUSD",
        "NATGAS",
        "COPPER",
    ]
    assert payload["artifact_status_label"] == "hot-universe-ok"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()
    assert Path(str(payload["report"])).exists()


def test_build_hot_research_universe_offline_emits_beta_batch_when_universe_is_large(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--network-mode",
            "offline",
            "--crypto-limit",
            "10",
            "--now",
            "2026-03-10T10:30:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["batches"]["crypto_majors"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
    assert payload["batches"]["crypto_beta"] == ["SUIUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"]
    assert payload["batches"]["mixed_macro_expanded"][:6] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "BNBUSDT",
    ]


def test_is_valid_hot_crypto_symbol_rejects_fiat_pairs() -> None:
    assert MODULE.is_valid_hot_crypto_symbol("EURUSDT") is False
    assert MODULE.is_valid_hot_crypto_symbol("GBPUSDT") is False
    assert MODULE.is_valid_hot_crypto_symbol("BTCUSDT") is True
