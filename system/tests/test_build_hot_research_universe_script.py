from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
import textwrap


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_hot_research_universe.py"
SPEC = importlib.util.spec_from_file_location("build_hot_research_universe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)


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
    assert payload["batches"]["metals_all"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert Path(str(payload["artifact"])).exists()


def test_build_hot_research_universe_reads_domestic_futures_paper_from_config(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            timezone: Asia/Shanghai
            universe:
              core:
                - {symbol: "BTCUSDT", asset_class: "crypto"}
              domestic_futures_paper:
                - {symbol: "BU2606", asset_class: "future", product: "asphalt", batch: "asphalt_cn", stage: "paper_only"}
            data:
              provider_profile: "dual_binance_bybit_public"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--network-mode",
            "offline",
            "--crypto-limit",
            "5",
            "--now",
            "2026-03-28T10:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["domestic_futures"]["selected"] == ["BU2606"]
    assert payload["domestic_futures"]["count"] == 1
    assert payload["batches"]["asphalt_cn"] == ["BU2606"]
    assert payload["batches"]["domestic_futures_cn"] == ["BU2606"]
    assert payload["commodities"]["selected"] == ["XAUUSD", "XAGUSD", "WTIUSD", "BRENTUSD", "NATGAS", "COPPER"]
