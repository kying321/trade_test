from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_commodity_paper_execution_gap_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_commodity_paper_execution_gap_report_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gap_report_treats_domestic_futures_paper_as_runtime_coverage(tmp_path: Path) -> None:
    mod = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            timezone: Asia/Shanghai
            universe:
              core:
                - {symbol: "BTCUSDT", asset_class: "crypto"}
              domestic_futures_paper:
                - {symbol: "BU2606", asset_class: "future", venue: "shfe", product: "asphalt", batch: "asphalt_cn", stage: "paper_only"}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config_symbols, config_asset_classes = mod.load_config_core_symbols(config_path)
    gap = mod.derive_gap_report(
        execution_queue={
            "execution_batch": "asphalt_cn",
            "execution_symbols": ["BU2606"],
            "next_execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
            "next_execution_symbol": "BU2606",
            "execution_queue_status": "paper-execution-queued",
        },
        execution_review={"execution_review_status": "paper-execution-close-evidence-pending"},
        execution_retro={"execution_retro_status": "paper-execution-close-evidence-pending"},
        execution_bridge={
            "bridge_status": "bridge_noop_already_bridged",
            "bridge_items": [
                {
                    "symbol": "BU2606",
                    "bridge_status": "already_bridged",
                    "bridge_reasons": [],
                }
            ],
        },
        config_core_symbols=config_symbols,
        config_asset_classes=config_asset_classes,
        trade_plan_counts={"BU2606": 1},
        executed_plan_counts={"BU2606": 1},
        ledger_counts={"BU2606": 1},
        open_position_counts={"BU2606": 1},
    )

    assert "BU2606" in config_symbols
    assert "future" in config_asset_classes
    assert gap["queue_symbols_missing_from_core_universe"] == []
    assert "core_universe_crypto_only" not in gap["gap_reason_codes"]
    assert "queue_symbols_missing_from_core_universe" not in gap["gap_reason_codes"]
    assert gap["gap_status"] == "gap_clear"
