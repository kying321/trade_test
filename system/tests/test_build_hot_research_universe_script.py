from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


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
