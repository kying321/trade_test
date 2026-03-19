from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from pathlib import Path
import importlib.util


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_hot_universe_research.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_hot_universe_research_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_latest_review_artifact_ignores_future_stamped_file(tmp_path: Path) -> None:
    mod = _load_script_module()
    mod.now_utc = lambda: dt.datetime(2026, 3, 10, 15, 58, tzinfo=dt.timezone.utc)
    current = tmp_path / "20260310T155630Z_crypto_route_operator_brief.json"
    future = tmp_path / "20260310T234630Z_crypto_route_operator_brief.json"
    current.write_text(json.dumps({"marker": "current"}), encoding="utf-8")
    future.write_text(json.dumps({"marker": "future"}), encoding="utf-8")
    path = mod.latest_review_artifact(tmp_path, "crypto_route_operator_brief")
    assert path == current


def test_run_hot_universe_research_dry_run_uses_universe_batches(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    universe_path = tmp_path / "universe.json"
    _write_json(
        review_dir / "20260310T155800Z_commodity_paper_ticket_lane.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_status": "paper-ready",
            "execution_mode": "paper_only",
            "paper_ready_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "missing_batches": [],
            "next_ticket_batch": "metals_all",
            "next_ticket_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
            "ticket_stack_brief": "paper-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
            "tickets": [
                {"ticket_id": "commodity-paper:metals_all"},
                {"ticket_id": "commodity-paper:precious_metals"},
                {"ticket_id": "commodity-paper:energy_liquids"},
                {"ticket_id": "commodity-paper:commodities_benchmark"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T155820Z_commodity_paper_ticket_book.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_mode": "paper_only",
            "actionable_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "shadow_batches": ["commodities_benchmark"],
            "next_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
            "next_ticket_batch": "metals_all",
            "next_ticket_symbol": "XAUUSD",
            "next_ticket_regime_gate": "paper_only",
            "next_ticket_weight_hint": 1.0,
            "ticket_book_stack_brief": "paper-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
            "actionable_ticket_count": 7,
            "tickets": [
                {"ticket_id": "commodity-paper-ticket:metals_all:XAUUSD"},
                {"ticket_id": "commodity-paper-ticket:precious_metals:XAGUSD"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T155840Z_commodity_paper_execution_preview.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_mode": "paper_only",
            "preview_ready_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "preview_batch_count": 4,
            "next_execution_batch": "metals_all",
            "next_execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "next_execution_regime_gate": "paper_only",
            "next_execution_weight_hint_sum": 2.3,
            "preview_stack_brief": "paper-execution-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
        },
    )
    _write_json(
        review_dir / "20260310T155900Z_commodity_paper_execution_artifact.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "execution_item_count": 3,
            "actionable_execution_item_count": 3,
            "execution_stack_brief": "paper-execution-artifact:metals_all:XAUUSD, XAGUSD, COPPER",
            "execution_items": [
                {"execution_id": "commodity-paper-execution:metals_all:XAUUSD", "symbol": "XAUUSD"},
                {"execution_id": "commodity-paper-execution:metals_all:XAGUSD", "symbol": "XAGUSD"},
                {"execution_id": "commodity-paper-execution:metals_all:COPPER", "symbol": "COPPER"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T155905Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "execution_item_count": 3,
            "actionable_execution_item_count": 3,
            "queue_depth": 3,
            "actionable_queue_depth": 3,
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "source_execution_status": "planned",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T155907Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-review-pending",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "execution_item_count": 3,
            "actionable_execution_item_count": 3,
            "queue_depth": 3,
            "actionable_queue_depth": 3,
            "review_item_count": 3,
            "actionable_review_item_count": 3,
            "next_review_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_review_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
            "review_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "review_status": "awaiting_paper_execution_review",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T155909Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-review-pending",
            "execution_retro_status": "paper-execution-retro-pending",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "review_item_count": 3,
            "actionable_review_item_count": 3,
            "retro_item_count": 3,
            "actionable_retro_item_count": 3,
            "next_retro_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_retro_execution_symbol": "XAUUSD",
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_retro",
                    "review_status": "awaiting_paper_execution_review",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T155245Z_binance_indicator_symbol_route_handoff.json",
        {
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
            "next_focus_reason": "BNB degrades on long window.",
            "focus_window_gate": "blocked_until_long_window_confirms",
            "focus_window_verdict": "degrades_on_long_window",
            "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
            "watch_priority_symbols": ["BNBUSDT"],
            "watch_only_symbols": ["SOLUSDT"],
            "routes": [],
            "overall_takeaway": "Deploy majors and keep beta in watch mode.",
        },
    )
    _write_json(
        review_dir / "20260310T155400Z_crypto_route_brief.json",
        {
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
            "next_focus_reason": "BNB degrades on long window.",
            "focus_window_gate": "blocked_until_long_window_confirms",
            "focus_window_gate_reason": "BNB still needs longer-window confirmation before promotion.",
            "focus_window_verdict": "degrades_on_long_window",
            "focus_brief": "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing.",
            "next_retest_action": "rerun_bnb_native_long_window",
            "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
            "brief_lines": ["status: deploy-price-state-plus-beta-watch"],
            "brief_text": "status: deploy-price-state-plus-beta-watch",
        },
    )
    _write_json(
        review_dir / "20260310T155600Z_crypto_route_operator_brief.json",
        {
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
            "next_focus_reason": "BNB degrades on long window.",
            "focus_window_gate": "blocked_until_long_window_confirms",
            "focus_window_verdict": "degrades_on_long_window",
            "focus_window_floor": "positive_but_weaker",
            "price_state_window_floor": "negative",
            "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
            "xlong_flow_window_floor": "laggy_positive_only",
            "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            "focus_brief": "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing.",
            "next_retest_action": "rerun_bnb_native_long_window",
            "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
            "operator_lines": ["focus-gate: blocked_until_long_window_confirms"],
            "operator_text": "focus-gate: blocked_until_long_window_confirms",
        },
    )
    _write_json(
        review_dir / "20260310T155910Z_binance_indicator_bnb_flow_focus.json",
        {
            "symbol": "BNBUSDT",
            "operator_status": "watch_priority",
            "promotion_gate": "blocked_until_long_window_confirms",
            "promotion_gate_reason": "BNB still needs longer-window confirmation before promotion.",
            "action": "watch_priority_until_long_window_confirms",
            "action_reason": "BNB has the best short-window beta flow, but it degrades when the window is extended.",
            "flow_window_verdict": "degrades_on_long_window",
            "price_state_window_verdict": "degrades_on_long_window",
            "next_retest_action": "rerun_bnb_native_long_window",
            "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
            "brief": "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing.",
        },
    )
    _write_json(
        universe_path,
        {
            "action": "build_hot_research_universe",
            "source_tier": "coingecko",
            "crypto": {"selected": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
            "commodities": {"selected": ["XAUUSD", "WTIUSD", "COPPER"]},
            "batches": {
                "crypto_hot": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                "crypto_majors": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                "commodities_benchmark": ["XAUUSD", "WTIUSD", "COPPER"],
                "energy": ["WTIUSD"],
                "mixed_macro": ["BTCUSDT", "ETHUSDT", "XAUUSD", "WTIUSD"],
                "mixed_macro_expanded": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSD", "WTIUSD", "COPPER"],
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--universe-file",
            str(universe_path),
            "--start",
            "2025-01-01",
            "--end",
            "2025-12-31",
            "--batch",
            "crypto_hot",
            "--batch",
            "mixed_macro",
            "--batch",
            "energy",
            "--dry-run",
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
    assert payload["status"] == "dry_run"
    assert len(payload["batch_results"]) == 3
    assert payload["batch_results"][0]["batch"] == "crypto_hot"
    assert payload["batch_results"][0]["status"] == "planned"
    assert payload["batch_results"][1]["batch"] == "mixed_macro"
    assert payload["batch_results"][2]["batch"] == "energy"
    assert payload["batch_summary"]["planned_batches"] == ["crypto_hot", "mixed_macro", "energy"]
    assert payload["batch_playbook"]["preferred_batches"] == []
    assert payload["batch_playbook"]["market_regime_takeaways"][0]["signal"] == "prefer_validated_batches"
    assert payload["symbol_attribution"]["batch_symbol_rankings"][0]["ranked_symbols"] == []
    assert payload["regime_attribution"]["batch_regime_rankings"][0]["ranked_regimes"] == []
    assert payload["regime_playbook"]["batch_rules"][0]["execution_profile"] == "inactive"
    assert payload["microstructure_playbook"]["cvd_lite_supported_batches"] == ["crypto_hot"]
    assert payload["microstructure_playbook"]["cvd_lite_partial_batches"] == ["mixed_macro"]
    assert payload["microstructure_playbook"]["cvd_lite_unsupported_batches"] == ["energy"]
    assert payload["commodity_paper_ticket_lane"]["ticket_status"] == "paper-ready"
    assert payload["commodity_paper_ticket_lane"]["next_ticket_batch"] == "metals_all"
    assert payload["commodity_paper_ticket_lane"]["next_ticket_symbols"] == ["XAGUSD", "COPPER", "XAUUSD"]
    assert payload["commodity_paper_ticket_book"]["ticket_book_status"] == "paper-ready"
    assert payload["commodity_paper_ticket_book"]["next_ticket_id"] == "commodity-paper-ticket:metals_all:XAUUSD"
    assert payload["commodity_paper_ticket_book"]["next_ticket_symbol"] == "XAUUSD"
    assert payload["commodity_paper_ticket_book"]["actionable_ticket_count"] == 7
    assert payload["commodity_paper_execution_preview"]["execution_preview_status"] == "paper-execution-ready"
    assert payload["commodity_paper_execution_preview"]["next_execution_batch"] == "metals_all"
    assert payload["commodity_paper_execution_preview"]["next_execution_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["commodity_paper_execution_preview"]["next_execution_regime_gate"] == "paper_only"
    assert payload["commodity_paper_execution_artifact"]["execution_artifact_status"] == "paper-execution-artifact-ready"
    assert payload["commodity_paper_execution_artifact"]["execution_batch"] == "metals_all"
    assert payload["commodity_paper_execution_artifact"]["execution_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["commodity_paper_execution_artifact"]["execution_regime_gate"] == "paper_only"
    assert payload["commodity_paper_execution_artifact"]["actionable_execution_item_count"] == 3
    assert payload["commodity_paper_execution_queue"]["execution_queue_status"] == "paper-execution-queued"
    assert payload["commodity_paper_execution_queue"]["execution_batch"] == "metals_all"
    assert payload["commodity_paper_execution_queue"]["next_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_paper_execution_queue"]["next_execution_symbol"] == "XAUUSD"
    assert payload["commodity_paper_execution_queue"]["queue_depth"] == 3
    assert payload["commodity_paper_execution_queue"]["actionable_queue_depth"] == 3
    assert payload["commodity_paper_execution_review"]["execution_review_status"] == "paper-execution-review-pending"
    assert payload["commodity_paper_execution_review"]["next_review_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_paper_execution_review"]["next_review_execution_symbol"] == "XAUUSD"
    assert payload["commodity_paper_execution_retro"]["execution_retro_status"] == "paper-execution-retro-pending"
    assert payload["commodity_paper_execution_retro"]["next_retro_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_paper_execution_retro"]["next_retro_execution_symbol"] == "XAUUSD"
    assert payload["bnb_flow_focus"]["symbol"] == "BNBUSDT"
    assert payload["bnb_flow_focus"]["next_retest_action"] == "rerun_bnb_native_long_window"
    assert payload["crypto_symbol_route_handoff"]["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["crypto_symbol_route_handoff"]["focus_window_verdict"] == "degrades_on_long_window"
    assert payload["crypto_route_brief"]["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["crypto_route_brief"]["next_retest_action"] == "rerun_bnb_native_long_window"
    assert payload["crypto_route_brief"]["focus_window_verdict"] == "degrades_on_long_window"
    assert payload["crypto_route_operator_brief"]["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["crypto_route_operator_brief"]["focus_window_floor"] == "positive_but_weaker"
    assert payload["crypto_route_operator_brief"]["price_state_window_floor"] == "negative"
    assert payload["crypto_route_operator_brief"]["comparative_window_takeaway"].startswith("Long-window flow holds up better")
    assert payload["crypto_route_operator_brief"]["xlong_flow_window_floor"] == "laggy_positive_only"
    assert payload["crypto_route_operator_brief"]["xlong_comparative_window_takeaway"].startswith("Extra-long flow keeps a raw positive return")
    assert payload["crypto_route_operator_brief"]["next_retest_action"] == "rerun_bnb_native_long_window"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()
    assert Path(str(payload["report"])).exists()


def test_summarize_batch_results_classifies_validated_and_research_only() -> None:
    mod = _load_script_module()
    summary = mod.summarize_batch_results(
        [
            {
                "batch": "energy",
                "status": "research_and_lab_ok",
                "research_backtest": {
                    "best_score": 7.0,
                    "best_metrics": {"annual_return": 1.2, "trades": 12},
                },
                "strategy_lab": {"accepted_count": 2},
            },
            {
                "batch": "precious_metals",
                "status": "research_and_lab_ok",
                "research_backtest": {
                    "best_score": 4.0,
                    "best_metrics": {"annual_return": 0.8, "trades": 10},
                },
                "strategy_lab": {"accepted_count": 0},
            },
        ]
    )
    assert summary["validated_batches"] == ["energy"]
    assert summary["research_only_batches"] == ["precious_metals"]


def test_run_batches_timeout_marks_partial_failure(tmp_path: Path) -> None:
    mod = _load_script_module()
    if "fork" not in mod.mp.get_all_start_methods():
        return

    class _DummyResult:
        def to_dict(self) -> dict[str, object]:
            return {
                "output_dir": str(tmp_path / "out"),
                "manifest": str(tmp_path / "manifest.json"),
                "elapsed_seconds": 0.0,
                "universe_count": 1,
                "bars_rows": 1,
                "mode_summaries": [],
            }

    def _slow_research_backtest(**_: object) -> _DummyResult:
        time.sleep(1.2)
        return _DummyResult()

    mod._batch_execution_context_name = lambda: "fork"
    mod.run_research_backtest = _slow_research_backtest

    batch_results = mod.run_batches(
        universe_payload={},
        selected_batches={"crypto_hot": ["BTCUSDT"]},
        output_root=tmp_path,
        start=dt.date(2025, 1, 1),
        end=dt.date(2025, 1, 31),
        hours_budget=0.01,
        workers=1,
        max_trials_per_mode=1,
        review_days=1,
        run_strategy_lab_enabled=False,
        strategy_lab_candidate_count=1,
        seed=42,
        dry_run=False,
        batch_timeout_seconds=1.0,
    )

    assert len(batch_results) == 1
    assert batch_results[0]["status"] == "failed"
    assert batch_results[0]["timed_out"] is True
    assert batch_results[0]["effective_timeout_seconds"] == 1.0
    assert str(batch_results[0]["error"]).startswith("TimeoutError:batch_timeout_seconds_exceeded")
    assert mod.overall_run_status(batch_results=batch_results, dry_run=False) == "partial_failure"


def test_run_batches_scales_timeout_for_larger_symbol_sets(tmp_path: Path) -> None:
    mod = _load_script_module()
    if "fork" not in mod.mp.get_all_start_methods():
        return

    class _DummyResult:
        def to_dict(self) -> dict[str, object]:
            return {
                "output_dir": str(tmp_path / "out"),
                "manifest": str(tmp_path / "manifest.json"),
                "elapsed_seconds": 0.0,
                "universe_count": 8,
                "bars_rows": 8,
                "mode_summaries": [],
            }

    def _slow_research_backtest(**_: object) -> _DummyResult:
        time.sleep(1.2)
        return _DummyResult()

    mod._batch_execution_context_name = lambda: "fork"
    mod.run_research_backtest = _slow_research_backtest

    batch_results = mod.run_batches(
        universe_payload={},
        selected_batches={
            "crypto_hot": [
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
                "XRPUSDT",
                "DOGEUSDT",
                "BNBUSDT",
                "SUIUSDT",
                "ADAUSDT",
            ]
        },
        output_root=tmp_path,
        start=dt.date(2025, 1, 1),
        end=dt.date(2025, 1, 31),
        hours_budget=0.01,
        workers=1,
        max_trials_per_mode=1,
        review_days=1,
        run_strategy_lab_enabled=False,
        strategy_lab_candidate_count=1,
        seed=42,
        dry_run=False,
        batch_timeout_seconds=1.0,
    )

    assert len(batch_results) == 1
    assert batch_results[0]["status"] == "research_ok"
    assert batch_results[0]["timed_out"] is False
    assert batch_results[0]["effective_timeout_seconds"] == 2.0


def test_derive_batch_playbook_marks_preferred_fragile_and_regime_takeaways() -> None:
    mod = _load_script_module()
    summary = {
        "ranked_batches": [
            {
                "batch": "metals_all",
                "status_label": "validated",
                "research_score": 4.6,
                "research_annual_return": 3.3,
                "research_trades": 16,
                "accepted_count": 4,
                "takeaway": "Research and strategy_lab both support this batch.",
            },
            {
                "batch": "energy_liquids",
                "status_label": "validated",
                "research_score": 4.1,
                "research_annual_return": 3.0,
                "research_trades": 16,
                "accepted_count": 2,
                "takeaway": "Research and strategy_lab both support this batch.",
            },
            {
                "batch": "precious_metals",
                "status_label": "research_only",
                "research_score": 4.0,
                "research_annual_return": 2.6,
                "research_trades": 14,
                "accepted_count": 0,
                "takeaway": "Research is positive but strategy_lab still does not validate it.",
            },
            {
                "batch": "commodities_benchmark",
                "status_label": "validated",
                "research_score": 0.6,
                "research_annual_return": 0.34,
                "research_trades": 6,
                "accepted_count": 2,
                "takeaway": "Research and strategy_lab both support this batch. Trade count is thin; treat as fragile.",
            },
            {
                "batch": "energy_gas",
                "status_label": "deprioritize",
                "research_score": -1.2,
                "research_annual_return": 0.0,
                "research_trades": 0,
                "accepted_count": 0,
                "takeaway": "Neither research nor strategy_lab currently supports this batch.",
            },
            {
                "batch": "base_metals",
                "status_label": "deprioritize",
                "research_score": -1.2,
                "research_annual_return": 0.0,
                "research_trades": 0,
                "accepted_count": 0,
                "takeaway": "Neither research nor strategy_lab currently supports this batch.",
            },
        ]
    }
    playbook = mod.derive_batch_playbook(summary)
    assert [row["batch"] for row in playbook["preferred_batches"]] == ["metals_all", "energy_liquids"]
    assert [row["batch"] for row in playbook["fragile_batches"]] == ["commodities_benchmark"]
    assert [row["batch"] for row in playbook["research_queue_batches"]] == ["precious_metals"]
    assert [row["batch"] for row in playbook["avoid_batches"]] == ["energy_gas", "base_metals"]
    assert {row["signal"] for row in playbook["market_regime_takeaways"]} >= {
        "paired_liquids_outperform_single_gas",
        "mixed_metals_outperform_single_sleeves",
        "broad_basket_secondary_only",
    }


def test_derive_symbol_attribution_sorts_symbols_by_total_pnl() -> None:
    mod = _load_script_module()
    attribution = mod.derive_symbol_attribution(
        [
            {
                "batch": "metals_all",
                "research_backtest": {
                    "best_by_symbol": {
                        "XAUUSD": {"total_pnl": 0.22, "avg_pnl": 0.03, "trade_count": 7, "win_rate": 0.71},
                        "COPPER": {"total_pnl": -0.05, "avg_pnl": -0.01, "trade_count": 5, "win_rate": 0.4},
                        "XAGUSD": {"total_pnl": 0.12, "avg_pnl": 0.02, "trade_count": 6, "win_rate": 0.66},
                    }
                },
            }
        ]
    )
    ranked = attribution["batch_symbol_rankings"][0]["ranked_symbols"]
    assert [row["symbol"] for row in ranked] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert attribution["top_symbol_by_batch"]["metals_all"]["symbol"] == "XAUUSD"


def test_derive_batch_relationships_marks_shadowed_validated_batch() -> None:
    mod = _load_script_module()
    batch_results = [
        {
            "batch": "metals_all",
            "symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "research_backtest": {
                "best_score": 4.67,
                "best_metrics": {"annual_return": 3.30, "trades": 16},
            },
        },
        {
            "batch": "commodities_benchmark",
            "symbols": ["XAUUSD", "XAGUSD", "WTIUSD", "BRENTUSD", "NATGAS", "COPPER"],
            "research_backtest": {
                "best_score": 4.67,
                "best_metrics": {"annual_return": 3.30, "trades": 16},
            },
        },
        {
            "batch": "energy_liquids",
            "symbols": ["WTIUSD", "BRENTUSD"],
            "research_backtest": {
                "best_score": 4.15,
                "best_metrics": {"annual_return": 3.04, "trades": 16},
            },
        },
    ]
    playbook = {
        "preferred_batches": [
            {"batch": "metals_all"},
            {"batch": "commodities_benchmark"},
            {"batch": "energy_liquids"},
        ]
    }
    symbol_attribution = {
        "top_symbol_by_batch": {
            "metals_all": {"symbol": "XAGUSD"},
            "commodities_benchmark": {"symbol": "XAGUSD"},
            "energy_liquids": {"symbol": "BRENTUSD"},
        }
    }
    relationships = mod.derive_batch_relationships(batch_results, playbook, symbol_attribution)
    assert relationships["primary_batches"] == ["metals_all", "energy_liquids"]
    assert relationships["secondary_batches"] == ["commodities_benchmark"]
    assert relationships["shadow_pairs"][0]["primary_batch"] == "metals_all"
    assert relationships["shadow_pairs"][0]["shadowed_batch"] == "commodities_benchmark"


def test_derive_regime_attribution_sorts_regimes_and_symbols() -> None:
    mod = _load_script_module()
    attribution = mod.derive_regime_attribution(
        [
            {
                "batch": "metals_all",
                "research_backtest": {
                    "best_by_symbol_regime": {
                        "XAGUSD": {
                            "震荡": {"total_pnl": 0.20, "avg_pnl": 0.05, "trade_count": 4, "win_rate": 0.75},
                            "弱趋势": {"total_pnl": 0.08, "avg_pnl": 0.04, "trade_count": 2, "win_rate": 0.50},
                        },
                        "COPPER": {
                            "震荡": {"total_pnl": 0.10, "avg_pnl": 0.03, "trade_count": 3, "win_rate": 0.66},
                            "下跌趋势": {"total_pnl": 0.12, "avg_pnl": 0.06, "trade_count": 2, "win_rate": 1.00},
                        },
                    }
                },
            }
        ]
    )
    ranked_regimes = attribution["batch_regime_rankings"][0]["ranked_regimes"]
    assert [row["regime"] for row in ranked_regimes] == ["震荡", "下跌趋势", "弱趋势"]
    assert ranked_regimes[0]["ranked_symbols"][0]["symbol"] == "XAGUSD"
    assert attribution["dominant_regime_by_batch"]["metals_all"]["regime"] == "震荡"
    assert attribution["top_symbol_by_batch_regime"]["metals_all"]["震荡"]["symbol"] == "XAGUSD"


def test_derive_regime_playbook_marks_range_avoid_and_trend_only() -> None:
    mod = _load_script_module()
    batch_results = [
        {"batch": "energy_liquids", "outcome": {"label": "validated"}},
        {"batch": "precious_metals", "outcome": {"label": "validated"}},
        {"batch": "energy_gas", "outcome": {"label": "deprioritize"}},
    ]
    regime_attribution = {
        "batch_regime_rankings": [
            {
                "batch": "energy_liquids",
                "ranked_regimes": [
                    {
                        "regime": "强趋势",
                        "total_pnl": 1.2,
                        "trade_count": 8,
                        "ranked_symbols": [
                            {"symbol": "BRENTUSD", "total_pnl": 0.60},
                            {"symbol": "WTIUSD", "total_pnl": 0.59},
                        ],
                    },
                    {
                        "regime": "震荡",
                        "total_pnl": -0.2,
                        "trade_count": 8,
                        "ranked_symbols": [
                            {"symbol": "BRENTUSD", "total_pnl": -0.1},
                        ],
                    },
                ],
            },
            {
                "batch": "precious_metals",
                "ranked_regimes": [
                    {
                        "regime": "强趋势",
                        "total_pnl": 0.9,
                        "trade_count": 10,
                        "ranked_symbols": [
                            {"symbol": "XAGUSD", "total_pnl": 0.45},
                            {"symbol": "XAUUSD", "total_pnl": 0.44},
                        ],
                    }
                ],
            },
            {
                "batch": "energy_gas",
                "ranked_regimes": [],
            },
        ]
    }
    playbook = mod.derive_regime_playbook(batch_results, regime_attribution)
    rules = {row["batch"]: row for row in playbook["batch_rules"]}
    assert rules["energy_liquids"]["execution_profile"] == "range_avoid"
    assert rules["energy_liquids"]["avoid_regimes"] == ["震荡"]
    assert rules["energy_liquids"]["leader_symbols"] == ["BRENTUSD", "WTIUSD"]
    assert rules["precious_metals"]["execution_profile"] == "trend_only"
    assert rules["precious_metals"]["leader_symbols"] == ["XAGUSD", "XAUUSD"]
    assert rules["energy_gas"]["execution_profile"] == "inactive"


def test_derive_research_action_ladder_marks_primary_shadow_and_avoid() -> None:
    mod = _load_script_module()
    summary = {
        "ranked_batches": [
            {"batch": "metals_all", "status_label": "validated", "research_score": 4.6},
            {"batch": "commodities_benchmark", "status_label": "validated", "research_score": 4.6},
            {"batch": "energy_liquids", "status_label": "validated", "research_score": 4.1},
            {"batch": "energy_gas", "status_label": "deprioritize", "research_score": -1.2},
        ]
    }
    playbook = {
        "preferred_batches": [
            {"batch": "metals_all"},
            {"batch": "commodities_benchmark"},
            {"batch": "energy_liquids"},
        ],
        "fragile_batches": [],
        "research_queue_batches": [],
        "avoid_batches": [{"batch": "energy_gas"}],
    }
    regime_playbook = {
        "batch_rules": [
            {
                "batch": "metals_all",
                "execution_profile": "trend_only",
                "dominant_regime": "强趋势",
                "leader_symbols": ["XAGUSD", "COPPER"],
                "avoid_regimes": [],
            },
            {
                "batch": "commodities_benchmark",
                "execution_profile": "trend_only",
                "dominant_regime": "强趋势",
                "leader_symbols": ["XAGUSD", "COPPER"],
                "avoid_regimes": [],
            },
            {
                "batch": "energy_liquids",
                "execution_profile": "range_avoid",
                "dominant_regime": "强趋势",
                "leader_symbols": ["BRENTUSD", "WTIUSD"],
                "avoid_regimes": ["震荡"],
            },
            {
                "batch": "energy_gas",
                "execution_profile": "inactive",
                "dominant_regime": "",
                "leader_symbols": [],
                "avoid_regimes": [],
            },
        ]
    }
    relationships = {
        "primary_batches": ["metals_all", "energy_liquids"],
        "secondary_batches": ["commodities_benchmark"],
    }
    leader_profiles = {
        "by_batch": {
            "metals_all": {"dominant_regime_profile": {"leader_structure": "paired_symmetric", "takeaway": "Top contribution is shared almost evenly by XAGUSD/COPPER."}},
            "commodities_benchmark": {"dominant_regime_profile": {"leader_structure": "paired_symmetric", "takeaway": "Top contribution is shared almost evenly by XAGUSD/COPPER."}},
            "energy_liquids": {"dominant_regime_profile": {"leader_structure": "paired_symmetric", "takeaway": "Top contribution is shared almost evenly by BRENTUSD/WTIUSD."}},
            "energy_gas": {"dominant_regime_profile": {"leader_structure": "inactive", "takeaway": "No positive symbol contribution was observed."}},
        }
    }
    ladder = mod.derive_research_action_ladder(summary, playbook, regime_playbook, relationships, leader_profiles)
    assert ladder["focus_primary_batches"] == ["metals_all"]
    assert ladder["focus_with_regime_filter_batches"] == ["energy_liquids"]
    assert ladder["shadow_only_batches"] == ["commodities_benchmark"]
    assert ladder["avoid_batches"] == ["energy_gas"]
    assert ladder["paired_focus_batches"] == ["metals_all", "energy_liquids"]
    ranked = {row["batch"]: row for row in ladder["ranked_actions"]}
    assert ranked["metals_all"]["action"] == "focus_primary"
    assert ranked["metals_all"]["leader_structure"] == "paired_symmetric"
    assert ranked["energy_liquids"]["action"] == "focus_with_regime_filter"
    assert ranked["commodities_benchmark"]["action"] == "shadow_only"


def test_derive_microstructure_playbook_marks_full_partial_and_macro_focus() -> None:
    mod = _load_script_module()
    universe_payload = {
        "crypto": {"selected": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
    }
    batch_results = [
        {"batch": "crypto_majors", "symbols": ["BTCUSDT", "ETHUSDT"]},
        {"batch": "mixed_macro", "symbols": ["BTCUSDT", "ETHUSDT", "XAUUSD", "WTIUSD"]},
        {"batch": "metals_all", "symbols": ["XAUUSD", "XAGUSD", "COPPER"]},
    ]
    action_ladder = {
        "focus_now_batches": ["metals_all"],
        "research_queue_batches": ["crypto_majors", "mixed_macro"],
    }
    playbook = mod.derive_microstructure_playbook(universe_payload, batch_results, action_ladder)
    assert playbook["cvd_lite_supported_batches"] == ["crypto_majors"]
    assert playbook["cvd_lite_partial_batches"] == ["mixed_macro"]
    assert playbook["cvd_lite_unsupported_batches"] == ["metals_all"]
    assert playbook["cvd_priority_batches"] == ["crypto_majors", "mixed_macro"]
    assert playbook["focus_macro_only_batches"] == ["metals_all"]
    assert "commodity sleeves" in playbook["overall_takeaway"]


def test_derive_crypto_cvd_queue_profile_maps_trend_and_range_batches() -> None:
    mod = _load_script_module()
    batch_results = [
        {"batch": "crypto_hot", "outcome": {"label": "pilot_only"}},
        {"batch": "crypto_majors", "outcome": {"label": "pilot_only"}},
        {"batch": "mixed_macro", "outcome": {"label": "deprioritize"}},
    ]
    action_ladder = {
        "research_queue_batches": ["crypto_hot", "crypto_majors", "mixed_macro"],
    }
    regime_playbook = {
        "batch_rules": [
            {
                "batch": "crypto_hot",
                "dominant_regime": "下跌趋势",
                "execution_profile": "single_regime",
                "leader_symbols": ["ZECUSDT", "SOLUSDT"],
            },
            {
                "batch": "crypto_majors",
                "dominant_regime": "震荡",
                "execution_profile": "single_regime",
                "leader_symbols": ["ETHUSDT"],
            },
            {
                "batch": "mixed_macro",
                "dominant_regime": "震荡",
                "execution_profile": "single_regime",
                "leader_symbols": ["BTCUSDT"],
            },
        ]
    }
    leader_profiles = {
        "by_batch": {
            "crypto_hot": {"dominant_regime_profile": {"leader_structure": "dual_leader"}},
            "crypto_majors": {"dominant_regime_profile": {"leader_structure": "distributed"}},
            "mixed_macro": {"dominant_regime_profile": {"leader_structure": "single_leader"}},
        }
    }
    microstructure_playbook = {
        "cvd_priority_batches": ["crypto_hot", "crypto_majors"],
        "batch_profiles": [
            {
                "batch": "crypto_hot",
                "coverage_label": "cvd_lite_full",
                "coverage_ratio": 1.0,
                "cvd_eligible_symbols": ["BTCUSDT", "ETHUSDT"],
                "proxy_only_symbols": [],
            },
            {
                "batch": "crypto_majors",
                "coverage_label": "cvd_lite_full",
                "coverage_ratio": 1.0,
                "cvd_eligible_symbols": ["BTCUSDT", "ETHUSDT"],
                "proxy_only_symbols": [],
            },
            {
                "batch": "mixed_macro",
                "coverage_label": "cvd_lite_partial",
                "coverage_ratio": 0.5,
                "cvd_eligible_symbols": ["BTCUSDT"],
                "proxy_only_symbols": ["XAUUSD"],
            },
        ],
    }

    profile = mod.derive_crypto_cvd_queue_profile(
        batch_results,
        action_ladder,
        regime_playbook,
        leader_profiles,
        microstructure_playbook,
    )

    assert profile["priority_batches"] == ["crypto_hot", "crypto_majors"]
    assert profile["trend_confirmation_batches"] == ["crypto_hot"]
    assert profile["reversal_watch_batches"] == ["crypto_majors"]
    assert profile["mixed_bridge_filter_batches"] == ["mixed_macro"]
    by_batch = {row["batch"]: row for row in profile["batch_profiles"]}
    assert by_batch["crypto_hot"]["trust_requirement"] == "dual_leader_alignment"
    assert by_batch["crypto_hot"]["preferred_contexts"] == ["continuation", "failed_auction"]
    assert by_batch["crypto_majors"]["trust_requirement"] == "basket_consensus"
    assert by_batch["crypto_majors"]["preferred_contexts"] == ["reversal", "absorption", "failed_auction"]
    assert by_batch["mixed_macro"]["queue_mode"] == "mixed_bridge_filter"


def test_classify_leader_structure_marks_symmetric_and_single_leader() -> None:
    mod = _load_script_module()
    symmetric = mod.classify_leader_structure(
        [
            {"symbol": "BRENTUSD", "total_pnl": 0.50, "trade_count": 8},
            {"symbol": "WTIUSD", "total_pnl": 0.50, "trade_count": 8},
        ]
    )
    assert symmetric["leader_structure"] == "paired_symmetric"
    assert symmetric["leader_symbols"] == ["BRENTUSD", "WTIUSD"]

    single = mod.classify_leader_structure(
        [
            {"symbol": "SOLUSDT", "total_pnl": 0.70, "trade_count": 3},
            {"symbol": "BTCUSDT", "total_pnl": 0.20, "trade_count": 4},
            {"symbol": "ETHUSDT", "total_pnl": 0.10, "trade_count": 4},
        ]
    )
    assert single["leader_structure"] == "single_leader"
    assert single["leader_symbols"] == ["SOLUSDT"]


def test_derive_leader_profiles_tracks_overall_and_dominant_regime() -> None:
    mod = _load_script_module()
    leader_profiles = mod.derive_leader_profiles(
        {
            "batch_symbol_rankings": [
                {
                    "batch": "metals_all",
                    "ranked_symbols": [
                        {"symbol": "XAGUSD", "total_pnl": 0.33, "trade_count": 5},
                        {"symbol": "COPPER", "total_pnl": 0.31, "trade_count": 7},
                        {"symbol": "XAUUSD", "total_pnl": 0.27, "trade_count": 4},
                    ],
                }
            ]
        },
        {
            "batch_regime_rankings": [
                {
                    "batch": "metals_all",
                    "ranked_regimes": [
                        {
                            "regime": "强趋势",
                            "total_pnl": 0.91,
                            "ranked_symbols": [
                                {"symbol": "XAGUSD", "total_pnl": 0.33, "trade_count": 5},
                                {"symbol": "COPPER", "total_pnl": 0.31, "trade_count": 7},
                                {"symbol": "XAUUSD", "total_pnl": 0.27, "trade_count": 4},
                            ],
                        }
                    ],
                }
            ]
        },
    )
    profile = leader_profiles["by_batch"]["metals_all"]
    assert profile["overall_profile"]["leader_structure"] == "dual_leader"
    assert profile["dominant_regime_profile"]["leader_structure"] == "dual_leader"
    assert profile["dominant_regime"] == "强趋势"


def test_build_report_includes_crypto_symbol_routes_section() -> None:
    mod = _load_script_module()
    report = mod.build_report(
        universe_payload={"source_tier": "coingecko", "crypto": {"selected": ["BTCUSDT"]}, "commodities": {"selected": ["XAUUSD"]}},
        batch_results=[],
        dry_run=True,
        start=dt.date.fromisoformat("2025-01-01"),
        end=dt.date.fromisoformat("2025-12-31"),
        crypto_route_brief={
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_short_window_flow_priority",
            "brief_text": "status: deploy-price-state-plus-beta-watch\nfocus: BNBUSDT",
        },
        crypto_symbol_route_handoff={
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
            "watch_priority_symbols": ["BNBUSDT"],
            "watch_only_symbols": ["SOLUSDT"],
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_short_window_flow_priority",
            "overall_takeaway": "Deploy majors and watch beta.",
            "routes": [
                {
                    "symbol": "BTCUSDT",
                    "lane": "majors",
                    "deployment": "price_state_primary_only",
                    "action": "deploy_price_state_only",
                    "status_label": "deploy_now",
                    "reason": "majors stable",
                }
            ],
        },
    )
    assert "## Crypto Route Brief" in report
    assert "## Crypto Symbol Routes" in report
    assert "deploy-price-state-plus-beta-watch" in report
    assert "watch_short_window_flow_priority" in report
    assert "BTCUSDT" in report
