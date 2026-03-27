from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_hot_universe_operator_brief.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_hot_universe_operator_brief_prefers_non_dry_run(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T100000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all", "precious_metals"],
                "research_queue_batches": ["crypto_hot"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "next_focus_symbol": "BNBUSDT",
                "next_retest_action": "rerun_bnb_native_long_window",
            },
        },
    )
    _write_json(
        review_dir / "20260310T110000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {"focus_primary_batches": []},
            "crypto_route_brief": {"operator_status": "watch-all"},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "ok"
    assert payload["source_mode"] == "single-hot-universe-source"
    assert payload["focus_primary_batches"] == ["metals_all", "precious_metals"]
    assert payload["research_queue_batches"] == ["crypto_hot"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"
    assert payload["crypto_next_retest_action"] == "rerun_bnb_native_long_window"
    assert "primary: metals_all, precious_metals" in payload["summary_text"]


def test_build_hot_universe_operator_brief_falls_back_to_rich_dry_run(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T100000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"focus_primary_batches": []},
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260310T110000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "research_queue_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "next_focus_symbol": "BNBUSDT",
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "next_focus_symbol": "BNBUSDT",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "dry_run"
    assert payload["focus_primary_batches"] == ["metals_all"]
    assert payload["research_queue_batches"] == ["crypto_majors"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"
