from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refresh_cross_market_operator_state.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cross_market_operator_state_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_hot_brief_source_falls_back_when_explicit_brief_was_pruned(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    fallback_path = review_dir / "20260313T125900Z_hot_universe_operator_brief.json"
    fallback_path.write_text(
        json.dumps({"status": "ok", "artifact": str(fallback_path)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    resolved = mod.resolve_hot_brief_source(
        review_dir=review_dir,
        commodity_refresh={"brief_artifact": str(review_dir / "20260313T124800Z_hot_universe_operator_brief.json")},
        reference_now=mod.parse_now("2026-03-13T13:00:00Z"),
    )

    assert resolved == fallback_path


def test_build_review_head_lane_marks_non_promotable_head_as_refresh_required() -> None:
    mod = _load_module()
    backlog = mod._build_review_backlog_from_rows(
        [
            {
                "area": "brooks_structure",
                "symbol": "SC2603",
                "action": "review_manual_stop_entry",
                "state": "review",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "reason": "second_entry_trend_continuation",
                "blocker_detail": "manual only | source freshness guard",
                "done_when": "SC2603 receives a fresh brooks artifact before promotion",
                "source_refresh_action": "consider_refresh_before_promotion",
                "promotion_ready": False,
            }
        ]
    )
    lane = mod._build_review_head_lane(backlog, "ready")
    assert lane["status"] == "refresh_required"
    assert lane["brief"] == "refresh_required:brooks_structure:SC2603:consider_refresh_before_promotion:96"
    assert lane["head"]["action"] == "consider_refresh_before_promotion"
    assert lane["head"]["review_action"] == "review_manual_stop_entry"
