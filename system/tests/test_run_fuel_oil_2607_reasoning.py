from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_fuel_oil_2607_reasoning.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_fuel_oil_2607_reasoning_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_fuel_oil_2607_reasoning_writes_all_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    input_packet = {
        "generated_at_utc": "2026-04-07T17:00:00Z",
        "as_of_date": "2026-04-07",
        "contract_focus": "FU2607",
        "benchmark_contract": "SC2607",
        "deferred_contract": "FU2609",
        "coverage": {"missing_fields": ["refinery_margin"], "coverage_ratio": 0.82},
        "framework_fixups": ["dynamic_probabilities"],
        "price_snapshot": {
            "last_price": 4257.0,
            "prev_settle": 4242.0,
            "benchmark_last_price": 684.9,
            "calendar_spread": 26.0,
            "fuel_sc_ratio": 6.22,
        },
        "fundamental_snapshot": {
            "inventory_signal": "tightening",
            "freight_signal": "firm",
            "demand_signal": "firm",
            "fuel_oil_inventory": 182000.0,
            "fuel_oil_inventory_delta": -4500.0,
            "bcti_index": 820.0,
            "bdti_index": 1040.0,
            "cargo_volume_yoy": 4.8,
            "coastal_port_throughput_yoy": 5.1,
            "relative_strength_score": 0.66,
        },
        "technical_snapshot": {
            "last_price": 4257.0,
            "daily_trend": "up",
            "weekly_trend": "up",
            "support_levels": [4172.0, 4098.0],
            "resistance_levels": [4303.0, 4456.0],
            "atr14": 92.0,
            "rsi14": 61.5,
            "macd_hist": 18.2,
            "calendar_spread": 26.0,
            "benchmark_relative_strength_20d": 0.035,
        },
        "participant_snapshot": {
            "net_top2": 1693.0,
            "long_short_bias": "net_long",
        },
    }
    paths = module.run_pipeline(
        output_root=tmp_path,
        input_packet=input_packet,
        generated_at="2026-04-07T17:00:00Z",
    )
    expected = {
        "input_packet": "latest_fuel_oil_2607_input_packet.json",
        "scenario_tree": "latest_fuel_oil_2607_scenario_tree.json",
        "transmission_map": "latest_fuel_oil_2607_transmission_map.json",
        "validation_ring": "latest_fuel_oil_2607_validation_ring.json",
        "trade_space": "latest_fuel_oil_2607_trade_space.json",
        "strategy_matrix": "latest_fuel_oil_2607_strategy_matrix.json",
        "summary": "latest_fuel_oil_2607_summary.json",
    }
    for key, filename in expected.items():
        path = tmp_path / "review" / filename
        assert path.exists()
        assert paths[key] == path
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
