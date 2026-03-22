from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("hold_router_transfer_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_router_transfer_writes_blocked_artifact_when_router_hypothesis_has_no_best_router(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    long_dataset_path = review_dir / "long.csv"
    derivation_dataset_path = review_dir / "derivation.csv"
    base_artifact_path = review_dir / "base.json"
    router_artifact_path = review_dir / "router.json"

    long_dataset_path.write_text("ts,open,high,low,close,volume,symbol\n", encoding="utf-8")
    derivation_dataset_path.write_text("ts,open,high,low,close,volume,symbol\n", encoding="utf-8")
    base_artifact_path.write_text("{}", encoding="utf-8")
    router_artifact_path.write_text(
        json.dumps(
            {
                "research_decision": "no_clean_price_state_router_hypothesis",
                "best_router": {},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            long_dataset_path=str(long_dataset_path),
            derivation_dataset_path=str(derivation_dataset_path),
            base_artifact_path=str(base_artifact_path),
            router_artifact_path=str(router_artifact_path),
            review_dir=str(review_dir),
            symbol="ETHUSDT",
            stamp="20260321T123500Z",
        ),
    )
    monkeypatch.setattr(
        module.EXIT_MODULE,
        "load_base_entry_params",
        lambda _path, _symbol: (
            {
                "breakout_lookback": 40,
                "breakout_memory_bars": 16,
                "pullback_max_atr": 1.6,
                "stop_buffer_atr": 0.3,
                "target_r": 2.5,
                "max_hold_bars": 16,
                "structure_filter": "none",
            },
            {"focus_symbol": "ETHUSDT"},
        ),
    )

    signal_frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:15:00Z",
                    "2026-01-01T00:30:00Z",
                ],
                utc=True,
            )
        }
    )
    monkeypatch.setattr(module.ROUTER_MODULE, "load_signal_frame", lambda *_args, **_kwargs: signal_frame.copy())
    monkeypatch.setattr(
        module,
        "evaluate_window",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("evaluate_window should not run without frozen router")),
    )

    result = module.main()

    assert result == 0
    json_path = review_dir / "20260321T123500Z_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json"
    md_path = review_dir / "20260321T123500Z_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.md"

    assert json_path.exists()
    assert latest_json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))

    assert payload["research_decision"] == "router_transfer_failed_keep_pure_hold_selection"
    assert payload["frozen_router"] == {}
    assert payload["historical_transfer_challenge"]["eligible_grid_count"] == 0
    assert payload["future_tail_probe"]["eligible_grid_count"] == 0
    assert "decision=router_transfer_failed_keep_pure_hold_selection" in payload["recommended_brief"]
    assert latest_payload["research_decision"] == "router_transfer_failed_keep_pure_hold_selection"
