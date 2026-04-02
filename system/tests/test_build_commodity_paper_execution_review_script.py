from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_commodity_paper_execution_review.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_commodity_paper_execution_review_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_execution_review_prefers_closed_symbol_snapshot_over_open_execution_snapshot() -> None:
    mod = _load_module()
    payload = mod.build_execution_review(
        {
            "execution_batch": "asphalt_cn",
            "execution_symbols": ["BU2606"],
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
                    "symbol": "BU2606",
                    "execution_status": "queued",
                    "allow_paper_execution": True,
                    "allow_live_execution": False,
                    "queue_rank": 1,
                }
            ],
        },
        ledger_counts={"BU2606": 1},
        open_position_counts={"BU2606": 0},
        executed_plan_counts={"BU2606": 2},
        position_details={"by_execution": {}, "by_symbol": {}},
        ledger_details={"by_execution": {}, "by_symbol": {}},
        trade_plan_details={
            "by_execution": {
                "commodity-paper-execution:asphalt_cn:BU2606": {
                    "symbol": "BU2606",
                    "status": "OPEN",
                    "bridge_execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
                }
            },
            "by_symbol": {"BU2606": {"symbol": "BU2606", "status": "OPEN"}},
        },
        executed_plan_details={
            "by_execution": {
                "commodity-paper-execution:asphalt_cn:BU2606": {
                    "symbol": "BU2606",
                    "status": "OPEN",
                    "bridge_execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
                }
            },
            "by_symbol": {
                "BU2606": {
                    "symbol": "BU2606",
                    "status": "CLOSED",
                    "exit_reason": "time_stop_no_market_data",
                }
            },
        },
    )

    item = payload["review_items"][0]
    assert item["paper_execution_status"] == "CLOSED"
    assert item["review_status"] == "awaiting_paper_execution_review"
