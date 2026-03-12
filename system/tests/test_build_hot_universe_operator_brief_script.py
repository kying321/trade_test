from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_hot_universe_operator_brief.py")


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
                "focus_with_regime_filter_batches": ["energy_liquids"],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": ["commodities_benchmark"],
                "avoid_batches": ["energy_gas"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_focus_reason": "BNB degrades on long window.",
                "focus_window_gate": "blocked_until_long_window_confirms",
                "focus_window_verdict": "degrades_on_long_window",
                "focus_brief": "BNB still needs long-window confirmation.",
                "next_retest_action": "rerun_bnb_native_long_window",
                "next_retest_reason": "Retest BNB on a longer native sample.",
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
    assert "research-queue: crypto_hot" in payload["summary_text"]


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
                "focus_with_regime_filter_batches": ["energy_liquids"],
                "research_queue_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
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
    assert payload["source_mode"] == "single-hot-universe-source"
    assert payload["focus_primary_batches"] == ["metals_all"]
    assert payload["research_queue_batches"] == ["crypto_majors"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"


def test_build_hot_universe_operator_brief_marks_research_queue_plus_crypto_deploy(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_beta"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
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
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_status"] == "research-queue-plus-crypto-deploy-watch"
    assert payload["research_queue_batches"] == ["crypto_majors", "crypto_hot"]
    assert "research-queue: crypto_majors, crypto_hot" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prefers_ok_over_richer_partial_failure_for_sources(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "partial_failure",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "research_queue_batches": ["crypto_hot"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_retest_action": "rerun_bnb_native_long_window",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"avoid_batches": ["crypto_hot"]},
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
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
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "ok"
    assert payload["source_artifact"] == str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["source_action_status"] == "ok"
    assert payload["source_action_artifact"] == str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["source_crypto_status"] == "ok"
    assert payload["source_crypto_artifact"] == str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["operator_research_embedding_quality_status"] == "avoid_only"
    assert payload["operator_research_embedding_quality_brief"] == "avoid_only:crypto_hot"
    assert payload["operator_research_embedding_active_batches"] == []
    assert payload["operator_research_embedding_avoid_batches"] == ["crypto_hot"]
    assert payload["operator_research_embedding_zero_trade_deprioritized_batches"] == []
    assert payload["operator_research_embedding_quality_done_when"] == (
        "latest hot_universe_research promotes at least one focus_primary or research_queue batch"
    )
    assert payload["operator_crypto_route_alignment_focus_slot"] == "next"
    assert payload["operator_crypto_route_alignment_status"] == "route_ahead_of_embedding"
    assert payload["operator_crypto_route_alignment_brief"] == "route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot"
    assert payload["operator_crypto_route_alignment_recovery_status"] == "recovery_completed_no_edge"
    assert payload["operator_crypto_route_alignment_recovery_brief"] == "recovery_completed_no_edge:crypto_hot"
    assert payload["operator_crypto_route_alignment_recovery_failed_batch_count"] == 0
    assert payload["operator_crypto_route_alignment_recovery_timed_out_batch_count"] == 0
    assert payload["operator_crypto_route_alignment_recovery_zero_trade_batches"] == ["crypto_hot"]
    assert payload["operator_crypto_route_alignment_cooldown_status"] == "cooldown_active_wait_for_new_market_data"
    assert payload["operator_crypto_route_alignment_cooldown_brief"] == (
        "cooldown_active_wait_for_new_market_data:>2026-03-10"
    )


def test_build_hot_universe_operator_brief_prefers_fresher_ok_over_older_richer_ok_sources(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"avoid_batches": ["crypto_hot"]},
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_retest_action": "rerun_bnb_native_long_window",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Older richer route artifact.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Older richer xlong artifact.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"avoid_batches": ["crypto_hot"]},
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-review",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch-priority:BNBUSDT",
                "next_focus_symbol": "SOLUSDT",
                "next_focus_action": "deprioritize_flow",
            },
            "crypto_route_operator_brief": {
                "comparative_window_takeaway": "Newer route artifact should win even if it is slightly less rich.",
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
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    expected_path = str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["source_artifact"] == expected_path
    assert payload["source_action_artifact"] == expected_path
    assert payload["source_crypto_artifact"] == expected_path
    assert payload["source_status"] == "ok"
    assert payload["source_crypto_status"] == "ok"
    assert payload["operator_crypto_route_alignment_cooldown_last_research_end_date"] == "2026-03-10"
    assert payload["operator_crypto_route_alignment_cooldown_next_eligible_end_date"] == "2026-03-11"
    assert payload["operator_crypto_route_alignment_cooldown_blocker_detail"] == (
        "latest clean crypto recovery already evaluated data through 2026-03-10 and still found no edge; "
        "rerunning before a later end date is unlikely to change the outcome"
    )
    assert payload["operator_crypto_route_alignment_cooldown_done_when"] == (
        "hot_universe_research end date advances beyond 2026-03-10 or crypto_route focus changes"
    )
    assert payload["operator_crypto_route_alignment_recipe_status"] == "deferred_by_cooldown"
    assert payload["operator_crypto_route_alignment_recipe_brief"] == "deferred_by_cooldown:2026-03-11"
    assert payload["operator_crypto_route_alignment_recipe_blocker_detail"] == (
        "latest clean crypto recovery already evaluated data through 2026-03-10 and still found no edge; "
        "rerunning before a later end date is unlikely to change the outcome"
    )
    assert payload["operator_crypto_route_alignment_recipe_done_when"] == (
        "hot_universe_research end date advances beyond 2026-03-10 or crypto_route focus changes"
    )
    assert payload["operator_crypto_route_alignment_recipe_ready_on_date"] == "2026-03-11"
    assert payload["operator_focus_slot_actionability_backlog_brief"] == "next:SOLUSDT:recovery_completed_no_edge"
    assert payload["operator_focus_slot_actionability_backlog_count"] == 1
    assert payload["operator_focus_slot_actionable_count"] == 2
    assert payload["operator_focus_slot_actionability_gate_brief"] == "actionability_guarded_by_content:2/3"
    assert payload["operator_focus_slot_actionability_gate_status"] == "actionability_guarded_by_content"
    assert payload["operator_focus_slot_actionability_gate_blocker_detail"] == (
        "SOLUSDT next content state remains blocked "
        "(route_ahead_of_embedding, recovery_completed_no_edge): "
        "deploy-price-state-plus-beta-review"
    )
    assert payload["operator_focus_slot_actionability_gate_done_when"] == "SOLUSDT completes deprioritize_flow"
    assert payload["operator_focus_slot_readiness_gate_ready_count"] == 1
    assert payload["operator_focus_slot_readiness_gate_brief"] == "readiness_guarded_by_source_freshness:1/3"
    assert payload["operator_focus_slot_readiness_gate_status"] == "readiness_guarded_by_source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocking_gate"] == "source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocker_detail"] == (
        "- followup source requires inspect_source_state (-, -, unknown)"
    )
    assert payload["operator_focus_slot_readiness_gate_done_when"] == "operator_focus_slot_refresh_backlog_count reaches 0"
    assert "research-embedding-quality: avoid_only:crypto_hot" in payload["summary_text"]
    assert "focus-slot-actionability-gate: actionability_guarded_by_content:2/3" in payload["summary_text"]
    assert "focus-slot-readiness-gate: readiness_guarded_by_source_freshness:1/3" in payload["summary_text"]
    assert "crypto-route-alignment-cooldown: cooldown_active_wait_for_new_market_data:>2026-03-10" in payload["summary_text"]
    assert "crypto-route-alignment-recovery-recipe: deferred_by_cooldown:2026-03-11" in payload["summary_text"]
    assert "crypto:SOLUSDT:deprioritize_flow" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prefers_dedicated_crypto_route_payload_over_embedded_route(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260312T054623Z_hot_universe_research.json",
        {
            "status": "ok",
            "end": "2026-03-12",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_hot", "crypto_majors", "crypto_beta"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260312T054632Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-12T05:46:34+00:00",
            "operator_status": "deploy-price-state-plus-beta-review",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch-priority:BNBUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-12T05:46:37Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_crypto_route_artifact"] == str(
        review_dir / "20260312T054632Z_crypto_route_operator_brief.json"
    )
    assert payload["crypto_route_status"] == "deploy-price-state-plus-beta-review"
    assert payload["crypto_route_stack_brief"] == (
        "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch-priority:BNBUSDT"
    )
    assert payload["crypto_focus_symbol"] == "SOLUSDT"
    assert payload["crypto_focus_action"] == "deprioritize_flow"
    assert payload["next_focus_area"] == "crypto_route"
    assert payload["next_focus_symbol"] == "SOLUSDT"
    assert payload["next_focus_action"] == "deprioritize_flow"
    assert payload["next_focus_source_artifact"] == str(
        review_dir / "20260312T054632Z_crypto_route_operator_brief.json"
    )


def test_build_hot_universe_operator_brief_emits_crypto_route_alignment_recovery_recipe(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    universe_path = review_dir / "20260310T120000Z_hot_research_universe.json"
    _write_json(
        universe_path,
        {
            "status": "ok",
            "batches": {
                "crypto_hot": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                "crypto_majors": ["BTCUSDT", "ETHUSDT"],
                "crypto_beta": ["BNBUSDT", "SOLUSDT"],
            },
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "ok",
            "start": "2026-03-07",
            "end": "2026-03-10",
            "universe_file": str(universe_path),
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_hot", "crypto_majors", "crypto_beta"],
            },
            "batch_summary": {
                "ranked_batches": [
                    {
                        "batch": "crypto_hot",
                        "status_label": "deprioritize",
                        "research_trades": 0,
                        "accepted_count": 0,
                    },
                    {
                        "batch": "crypto_majors",
                        "status_label": "deprioritize",
                        "research_trades": 0,
                        "accepted_count": 0,
                    },
                    {
                        "batch": "crypto_beta",
                        "status_label": "deprioritize",
                        "research_trades": 0,
                        "accepted_count": 0,
                    },
                ]
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T121500Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:15:00+00:00",
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending",
            "execution_retro_status": "paper-execution-retro-pending",
            "next_retro_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_retro_execution_symbol": "XAUUSD",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_retro",
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_crypto_route_alignment_status"] == "route_ahead_of_embedding"
    assert payload["operator_crypto_route_alignment_brief"] == (
        "route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta"
    )
    assert payload["operator_crypto_route_alignment_recovery_status"] == "recovery_completed_no_edge"
    assert payload["operator_crypto_route_alignment_recovery_brief"] == (
        "recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta"
    )
    assert payload["operator_crypto_route_alignment_recovery_zero_trade_batches"] == [
        "crypto_hot",
        "crypto_majors",
        "crypto_beta",
    ]
    assert payload["operator_crypto_route_alignment_recipe_window_days"] == 21
    assert payload["operator_crypto_route_alignment_recipe_target_batches"] == [
        "crypto_hot",
        "crypto_majors",
        "crypto_beta",
    ]
    assert payload["operator_crypto_route_alignment_recipe_expected_status"] == "ok"
    assert payload["operator_crypto_route_alignment_recipe_script"] == str(
        SCRIPT_PATH.parent / "run_hot_universe_research.py"
    )
    assert "--run-strategy-lab" in payload["operator_crypto_route_alignment_recipe_command_hint"]
    assert "--start 2026-02-18" in payload["operator_crypto_route_alignment_recipe_command_hint"]
    assert f"--universe-file {shlex.quote(str(universe_path))}" in payload["operator_crypto_route_alignment_recipe_command_hint"]
    assert payload["operator_crypto_route_alignment_recipe_followup_script"] == str(
        SCRIPT_PATH.parent / "refresh_commodity_paper_execution_state.py"
    )
    assert (
        payload["operator_crypto_route_alignment_recipe_verify_hint"]
        == "confirm operator_research_embedding_quality_status leaves avoid_only or operator_crypto_route_alignment_status leaves route_ahead_of_embedding"
    )


def test_build_hot_universe_operator_brief_merges_action_and_crypto_sources(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_beta"],
            },
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_retest_action": "wait_for_more_bnb_native_data",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
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
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_action_status"] == "ok"
    assert payload["source_crypto_status"] == "dry_run"
    assert payload["source_mode"] == "merged-action-crypto-sources"
    assert payload["research_queue_batches"] == ["crypto_majors", "crypto_hot"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"
    assert payload["crypto_focus_window_floor"] == "positive_but_weaker"
    assert payload["crypto_xlong_flow_window_floor"] == "laggy_positive_only"
    assert "crypto-xlong-flow-floor: laggy_positive_only" in payload["summary_text"]


def test_build_hot_universe_operator_brief_merges_commodity_lane(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_beta"],
            },
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122000Z_commodity_execution_lane.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "execution_mode": "paper_first",
            "focus_primary_batches": ["metals_all", "precious_metals"],
            "focus_with_regime_filter_batches": ["energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "leader_symbols_primary": ["XAGUSD", "COPPER", "XAUUSD"],
            "leader_symbols_regime_filter": ["BRENTUSD", "WTIUSD"],
            "next_focus_batch": "metals_all",
            "next_focus_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
            "next_stage": "paper_ticket_lane",
            "route_stack_brief": "paper-primary:metals_all,precious_metals | regime-filter:energy_liquids | shadow:commodities_benchmark",
        },
    )
    _write_json(
        review_dir / "20260310T122500Z_commodity_paper_ticket_lane.json",
        {
            "status": "ok",
            "ticket_status": "paper-ready",
            "ticket_stack_brief": "paper-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
            "paper_ready_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "next_ticket_batch": "metals_all",
            "next_ticket_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
            "tickets": [
                {"ticket_id": "commodity-paper:metals_all"},
                {"ticket_id": "commodity-paper:precious_metals"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122700Z_commodity_paper_ticket_book.json",
        {
            "status": "ok",
            "ticket_book_status": "paper-ready",
            "ticket_book_stack_brief": "paper-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
            "actionable_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "shadow_batches": ["commodities_benchmark"],
            "next_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
            "next_ticket_batch": "metals_all",
            "next_ticket_symbol": "XAUUSD",
            "actionable_ticket_count": 7,
            "tickets": [
                {"ticket_id": "commodity-paper-ticket:metals_all:XAUUSD"},
                {"ticket_id": "commodity-paper-ticket:precious_metals:XAGUSD"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122720Z_commodity_paper_execution_preview.json",
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
        review_dir / "20260310T122740Z_commodity_paper_execution_artifact.json",
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
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
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
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
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
            "execution_regime_gate": "paper_only",
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_mode"] == "merged-action-commodity-crypto-sources"
    assert payload["commodity_route_status"] == "paper-first"
    assert payload["commodity_ticket_status"] == "paper-ready"
    assert payload["commodity_ticket_book_status"] == "paper-ready"
    assert payload["commodity_focus_batch"] == "metals_all"
    assert payload["commodity_focus_symbols"] == ["XAGUSD", "COPPER", "XAUUSD"]
    assert payload["commodity_ticket_focus_batch"] == "metals_all"
    assert payload["commodity_ticket_focus_symbols"] == ["XAGUSD", "COPPER", "XAUUSD"]
    assert payload["commodity_next_ticket_id"] == "commodity-paper-ticket:metals_all:XAUUSD"
    assert payload["commodity_next_ticket_symbol"] == "XAUUSD"
    assert payload["commodity_actionable_ticket_count"] == 7
    assert payload["commodity_execution_preview_status"] == "paper-execution-ready"
    assert payload["commodity_next_execution_batch"] == "metals_all"
    assert payload["commodity_next_execution_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["commodity_next_execution_regime_gate"] == "paper_only"
    assert payload["commodity_execution_artifact_status"] == "paper-execution-artifact-ready"
    assert payload["commodity_execution_batch"] == "metals_all"
    assert payload["commodity_execution_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["commodity_actionable_execution_item_count"] == 3
    assert payload["commodity_execution_queue_status"] == "paper-execution-queued"
    assert payload["commodity_queue_depth"] == 3
    assert payload["commodity_actionable_queue_depth"] == 3
    assert payload["commodity_next_queue_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_queue_execution_symbol"] == "XAUUSD"
    assert payload["commodity_execution_review_status"] == "paper-execution-review-pending"
    assert payload["commodity_review_item_count"] == 3
    assert payload["commodity_actionable_review_item_count"] == 3
    assert payload["commodity_next_review_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_review_execution_symbol"] == "XAUUSD"
    assert payload["source_commodity_execution_queue_status"] == "ok"
    assert payload["source_commodity_execution_artifact_status"] == "ok"
    assert payload["source_commodity_execution_review_status"] == "ok"
    assert payload["operator_status"] == "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
    assert "commodity-route: paper-primary:metals_all,precious_metals | regime-filter:energy_liquids | shadow:commodities_benchmark" in payload["summary_text"]
    assert "commodity-ticket-status: paper-ready" in payload["summary_text"]
    assert "commodity-ticket-book-status: paper-ready" in payload["summary_text"]
    assert "commodity-execution-preview-status: paper-execution-ready" in payload["summary_text"]
    assert "commodity-execution-artifact-status: paper-execution-artifact-ready" in payload["summary_text"]
    assert "commodity-execution-review-status: paper-execution-review-pending" in payload["summary_text"]
    assert "commodity-execution-queue-status: paper-execution-queued" in payload["summary_text"]


def test_build_hot_universe_operator_brief_merges_commodity_execution_retro(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T120000Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_brief.json"),
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
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

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_retro_status"] == "paper-execution-retro-pending"
    assert payload["commodity_retro_item_count"] == 3
    assert payload["commodity_actionable_retro_item_count"] == 3
    assert payload["commodity_next_retro_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_retro_execution_symbol"] == "XAUUSD"
    assert payload["source_commodity_execution_retro_status"] == "ok"
    assert payload["operator_status"] == "commodity-paper-execution-retro-pending-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:retro:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_retro"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "review_paper_execution_retro"
    assert payload["next_focus_reason"] == "paper_execution_retro_pending"
    assert payload["secondary_focus_area"] == "crypto_route"
    assert payload["secondary_focus_target"] == "BNBUSDT"
    assert payload["secondary_focus_action"] == "watch_priority_until_long_window_confirms"
    assert payload["operator_crypto_route_alignment_status"] == "aligned"
    assert payload["operator_crypto_route_alignment_brief"] == "aligned:BNBUSDT:crypto_majors, crypto_hot"
    assert "crypto-route-alignment: aligned:BNBUSDT:crypto_majors, crypto_hot" in payload["summary_text"]
    assert "commodity-execution-retro-status: paper-execution-retro-pending" in payload["summary_text"]


def test_build_hot_universe_operator_brief_keeps_retro_focus_for_partial_fill_remainder(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_research_universe.json",
        {
            "status": "ok",
            "batches": {
                "crypto_hot": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                "crypto_majors": ["BTCUSDT", "ETHUSDT"],
                "crypto_beta": ["BNBUSDT", "SOLUSDT"],
                "metals_all": ["XAUUSD", "XAGUSD", "COPPER"],
            },
        },
    )
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T120000Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_brief.json"),
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending-fill-remainder",
            "actionable_review_item_count": 1,
            "review_pending_symbols": ["XAUUSD"],
            "fill_evidence_pending_count": 2,
            "fill_evidence_pending_symbols": ["XAGUSD", "COPPER"],
            "next_review_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_review_execution_symbol": "XAUUSD",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_fill_evidence_execution_symbol": "XAGUSD",
            "review_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "review_status": "awaiting_paper_execution_review",
                    "paper_execution_evidence_present": True,
                    "paper_entry_price": 5198.10009765625,
                    "paper_stop_price": 4847.7998046875,
                    "paper_target_price": 5758.58056640625,
                    "paper_quote_usdt": 0.15896067200583952,
                    "paper_execution_status": "OPEN",
                    "paper_signal_price_reference_source": "yfinance:GC=F",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending-fill-remainder",
            "execution_retro_status": "paper-execution-close-evidence-pending-fill-remainder",
            "actionable_retro_item_count": 0,
            "close_evidence_pending_count": 1,
            "close_evidence_pending_symbols": ["XAUUSD"],
            "fill_evidence_pending_count": 2,
            "fill_evidence_pending_symbols": ["XAGUSD", "COPPER"],
            "next_retro_execution_id": "",
            "next_retro_execution_symbol": "",
            "next_close_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_close_evidence_execution_symbol": "XAUUSD",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_fill_evidence_execution_symbol": "XAGUSD",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_close_evidence",
                    "paper_execution_evidence_present": True,
                    "paper_entry_price": 5198.10009765625,
                    "paper_stop_price": 4847.7998046875,
                    "paper_target_price": 5758.58056640625,
                    "paper_quote_usdt": 0.15896067200583952,
                    "paper_execution_status": "OPEN",
                    "paper_signal_price_reference_source": "yfinance:GC=F",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD and COPPER."],
            "queue_symbols_with_stale_directional_signal_dates": {
                "XAGUSD": "2026-01-26",
                "COPPER": "2026-01-29",
            },
            "queue_symbols_with_stale_directional_signal_age_days": {
                "XAGUSD": 42,
                "COPPER": 39,
            },
            "stale_directional_signal_watch_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "symbol": "XAGUSD",
                    "signal_date": "2026-01-26",
                    "signal_age_days": 42,
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:COPPER",
                    "symbol": "COPPER",
                    "signal_date": "2026-01-29",
                    "signal_age_days": 39,
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_partially_bridged_stale_remainder",
            "already_present_count": 1,
            "already_bridged_symbols": ["XAUUSD"],
            "signal_stale_count": 2,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir), "--now", "2026-03-10T12:30:00Z"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_review_status"] == "paper-execution-close-evidence-pending-fill-remainder"
    assert payload["commodity_execution_retro_status"] == "paper-execution-close-evidence-pending-fill-remainder"
    assert payload["operator_status"] == "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:close-evidence:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_close_evidence"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["next_focus_reason"] == "paper_execution_close_evidence_pending"
    assert payload["next_focus_state"] == "waiting"
    assert payload["next_focus_blocker_detail"] == (
        "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
    )
    assert payload["next_focus_done_when"] == (
        "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
    )
    assert payload["followup_focus_area"] == "commodity_fill_evidence"
    assert payload["followup_focus_target"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["followup_focus_symbol"] == "XAGUSD"
    assert payload["followup_focus_action"] == "wait_for_paper_execution_fill_evidence"
    assert payload["followup_focus_reason"] == "paper_execution_fill_evidence_pending"
    assert payload["followup_focus_state"] == "waiting"
    assert payload["followup_focus_blocker_detail"] == (
        "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26"
    )
    assert payload["followup_focus_done_when"] == "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols"
    assert payload["operator_focus_slots_brief"] == (
        "primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence"
        " | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence"
        " | secondary:watch:BNBUSDT:watch_priority_until_long_window_confirms"
    )
    assert payload["operator_focus_slot_sources_brief"] == (
        "primary:commodity_execution_retro | followup:commodity_execution_review | secondary:crypto_route"
    )
    assert payload["operator_focus_slot_status_brief"] == (
        "primary:ok@2026-03-10T12:27:55+00:00"
        " | followup:ok@2026-03-10T12:27:51+00:00"
        " | secondary:ok@2026-03-10T12:00:00+00:00"
    )
    assert payload["operator_focus_slot_recency_brief"] == (
        "primary:fresh:2m | followup:fresh:2m | secondary:carry_over:30m"
    )
    assert payload["operator_focus_slot_health_brief"] == (
        "primary:ready:read_current_artifact"
        " | followup:ready:read_current_artifact"
        " | secondary:carry_over_ok:consider_refresh_before_promotion"
    )
    assert payload["operator_focus_slot_refresh_backlog_brief"] == (
        "secondary:BNBUSDT:consider_refresh_before_promotion"
    )
    assert payload["operator_focus_slot_refresh_backlog_count"] == 1
    assert payload["operator_focus_slot_ready_count"] == 2
    assert payload["operator_focus_slot_total_count"] == 3
    assert payload["operator_focus_slot_promotion_gate_brief"] == (
        "promotion_guarded_by_source_freshness:2/3"
    )
    assert payload["operator_focus_slot_promotion_gate_status"] == "promotion_guarded_by_source_freshness"
    assert payload["operator_focus_slot_promotion_gate_blocker_detail"] == (
        "BNBUSDT secondary source requires consider_refresh_before_promotion "
        "(crypto_route, ok, carry_over, age=30m)"
    )
    assert payload["operator_focus_slot_promotion_gate_done_when"] == (
        "operator_focus_slot_refresh_backlog_count reaches 0"
    )
    assert payload["operator_focus_slot_readiness_gate_ready_count"] == 2
    assert payload["operator_focus_slot_readiness_gate_brief"] == (
        "readiness_guarded_by_source_freshness:2/3"
    )
    assert payload["operator_focus_slot_readiness_gate_status"] == "readiness_guarded_by_source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocking_gate"] == "source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocker_detail"] == (
        "BNBUSDT secondary source requires consider_refresh_before_promotion "
        "(crypto_route, ok, carry_over, age=30m)"
    )
    assert payload["operator_focus_slot_readiness_gate_done_when"] == (
        "operator_focus_slot_refresh_backlog_count reaches 0"
    )
    assert payload["operator_crypto_route_alignment_status"] == "aligned"
    assert payload["operator_crypto_route_alignment_brief"] == "aligned:BNBUSDT:crypto_hot"
    assert payload["operator_source_refresh_queue_brief"] == (
        "1:secondary:BNBUSDT:consider_refresh_before_promotion"
    )
    assert payload["operator_source_refresh_queue_count"] == 1
    assert payload["operator_source_refresh_checklist_brief"] == (
        "1:refresh_recommended:BNBUSDT:consider_refresh_before_promotion"
    )
    expected_refresh_recipe_script = str(SCRIPT_PATH.parent / "refresh_crypto_route_state.py")
    expected_refresh_recipe_command = shlex.join(
        [
            "python3",
            expected_refresh_recipe_script,
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(review_dir.parent),
            "--now",
            "2026-03-10T12:30:00Z",
        ]
    )
    expected_refresh_operator_script = str(SCRIPT_PATH.parent / "build_crypto_route_operator_brief.py")
    expected_refresh_operator_command = shlex.join(
        [
            "python3",
            expected_refresh_operator_script,
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ]
    )
    expected_refresh_recipe_artifact_kind = "crypto_route_brief"
    expected_refresh_recipe_artifact_path_hint = str(review_dir / "*_crypto_route_brief.json")
    expected_refresh_operator_artifact_kind = "crypto_route_operator_brief"
    expected_refresh_operator_artifact_path_hint = str(review_dir / "*_crypto_route_operator_brief.json")
    expected_refresh_research_script = str(SCRIPT_PATH.parent / "run_hot_universe_research.py")
    expected_refresh_research_command = shlex.join(
        [
            "python3",
            expected_refresh_research_script,
            "--output-root",
            str(review_dir.parent),
            "--review-dir",
            str(review_dir),
            "--start",
            "2026-03-07",
            "--end",
            "2026-03-10",
            "--now",
            "2026-03-10T12:30:00Z",
            "--universe-file",
            str(review_dir / "20260310T120000Z_hot_research_universe.json"),
            "--batch",
            "crypto_hot",
            "--batch",
            "crypto_majors",
            "--batch",
            "crypto_beta",
        ]
    )
    expected_refresh_followup_script = str(SCRIPT_PATH.parent / "refresh_commodity_paper_execution_state.py")
    expected_refresh_followup_command = shlex.join(
        [
            "python3",
            expected_refresh_followup_script,
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(review_dir.parent),
            "--context-path",
            str(review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"),
        ]
    )
    expected_refresh_research_artifact_kind = "hot_universe_research"
    expected_refresh_research_artifact_path_hint = str(review_dir / "*_hot_universe_research.json")
    expected_refresh_followup_artifact_kind = "commodity_paper_execution_refresh"
    expected_refresh_followup_artifact_path_hint = str(review_dir / "*_commodity_paper_execution_refresh.json")
    expected_refresh_step_checkpoint_brief = (
        "1:missing:crypto_route_brief | 2:carry_over:crypto_route_operator_brief"
        " | 3:carry_over:hot_universe_research | 4:missing:commodity_paper_execution_refresh"
    )
    expected_refresh_pipeline_pending_brief = (
        "1:refresh_crypto_route_brief:missing:crypto_route_brief"
        " | 2:refresh_crypto_route_operator_brief:carry_over:crypto_route_operator_brief"
        " | 3:refresh_hot_universe_research_embedding:carry_over:hot_universe_research"
        " | 4:refresh_commodity_handoff:missing:commodity_paper_execution_refresh"
    )
    assert payload["next_focus_source_kind"] == "commodity_execution_retro"
    assert payload["next_focus_source_artifact"] == str(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json"
    )
    assert payload["next_focus_source_status"] == "ok"
    assert payload["next_focus_source_as_of"] == "2026-03-10T12:27:55+00:00"
    assert payload["next_focus_source_age_minutes"] == 2
    assert payload["next_focus_source_recency"] == "fresh"
    assert payload["next_focus_source_health"] == "ready"
    assert payload["next_focus_source_refresh_action"] == "read_current_artifact"
    assert payload["followup_focus_source_kind"] == "commodity_execution_review"
    assert payload["followup_focus_source_artifact"] == str(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json"
    )
    assert payload["followup_focus_source_status"] == "ok"
    assert payload["followup_focus_source_as_of"] == "2026-03-10T12:27:51+00:00"
    assert payload["followup_focus_source_age_minutes"] == 2
    assert payload["followup_focus_source_recency"] == "fresh"
    assert payload["followup_focus_source_health"] == "ready"
    assert payload["followup_focus_source_refresh_action"] == "read_current_artifact"
    assert payload["secondary_focus_source_kind"] == "crypto_route"
    assert payload["secondary_focus_source_artifact"] == str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json")
    assert payload["secondary_focus_source_status"] == "ok"
    assert payload["secondary_focus_source_as_of"] == "2026-03-10T12:00:00+00:00"
    assert payload["secondary_focus_source_age_minutes"] == 30
    assert payload["secondary_focus_source_recency"] == "carry_over"
    assert payload["secondary_focus_source_health"] == "carry_over_ok"
    assert payload["secondary_focus_source_refresh_action"] == "consider_refresh_before_promotion"
    assert payload["operator_focus_slot_refresh_head_slot"] == "secondary"
    assert payload["operator_focus_slot_refresh_head_symbol"] == "BNBUSDT"
    assert payload["operator_focus_slot_refresh_head_action"] == "consider_refresh_before_promotion"
    assert payload["operator_focus_slot_refresh_head_health"] == "carry_over_ok"
    assert payload["operator_source_refresh_next_slot"] == "secondary"
    assert payload["operator_source_refresh_next_symbol"] == "BNBUSDT"
    assert payload["operator_source_refresh_next_action"] == "consider_refresh_before_promotion"
    assert payload["operator_source_refresh_next_source_kind"] == "crypto_route"
    assert payload["operator_source_refresh_next_source_health"] == "carry_over_ok"
    assert payload["operator_source_refresh_next_source_artifact"] == str(
        review_dir / "20260310T120000Z_crypto_route_operator_brief.json"
    )
    assert payload["operator_source_refresh_next_state"] == "refresh_recommended"
    assert payload["operator_source_refresh_next_blocker_detail"] == (
        "crypto_route artifact is ok and carry_over, age=30m"
    )
    assert payload["operator_source_refresh_next_done_when"] == (
        "BNBUSDT receives a fresh crypto_route artifact before promotion"
    )
    assert payload["operator_source_refresh_next_recipe_script"] == expected_refresh_recipe_script
    assert payload["operator_source_refresh_next_recipe_command_hint"] == expected_refresh_recipe_command
    assert payload["operator_source_refresh_next_recipe_expected_status"] == "ok"
    assert payload["operator_source_refresh_next_recipe_expected_artifact_kind"] == expected_refresh_recipe_artifact_kind
    assert payload["operator_source_refresh_next_recipe_expected_artifact_path_hint"] == (
        expected_refresh_recipe_artifact_path_hint
    )
    assert payload["operator_source_refresh_next_recipe_note"] == (
        "guarded entrypoint refreshes native crypto route sources before the remaining pipeline steps"
    )
    assert payload["operator_source_refresh_next_recipe_followup_script"] == expected_refresh_followup_script
    assert payload["operator_source_refresh_next_recipe_followup_command_hint"] == expected_refresh_followup_command
    assert payload["operator_source_refresh_next_recipe_verify_hint"] == (
        "rerun commodity refresh and confirm BNBUSDT leaves operator_source_refresh_queue"
    )
    assert payload["operator_source_refresh_next_recipe_steps_brief"] == (
        "1:refresh_crypto_route_brief | 2:refresh_crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding | 4:refresh_commodity_handoff"
    )
    assert payload["operator_source_refresh_next_recipe_step_checkpoint_brief"] == (
        expected_refresh_step_checkpoint_brief
    )
    assert payload["operator_source_refresh_pipeline_steps_brief"] == (
        "1:refresh_crypto_route_brief | 2:refresh_crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding | 4:refresh_commodity_handoff"
    )
    assert payload["operator_source_refresh_pipeline_step_checkpoint_brief"] == expected_refresh_step_checkpoint_brief
    assert payload["operator_source_refresh_pipeline_pending_brief"] == expected_refresh_pipeline_pending_brief
    assert payload["operator_source_refresh_pipeline_pending_count"] == 4
    assert payload["operator_source_refresh_pipeline_head_rank"] == "1"
    assert payload["operator_source_refresh_pipeline_head_name"] == "refresh_crypto_route_brief"
    assert payload["operator_source_refresh_pipeline_head_checkpoint_state"] == "missing"
    assert payload["operator_source_refresh_pipeline_head_expected_artifact_kind"] == "crypto_route_brief"
    assert payload["operator_source_refresh_pipeline_head_current_artifact"] == "-"
    assert payload["operator_source_refresh_next_recipe_steps"] == [
            {
                "rank": 1,
                "name": "refresh_crypto_route_brief",
                "script": expected_refresh_recipe_script,
                "command_hint": expected_refresh_recipe_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_recipe_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_recipe_artifact_path_hint,
                "current_artifact": "",
                "current_status": "",
                "current_as_of": "",
                "current_age_minutes": None,
                "current_recency": "unknown",
                "checkpoint_state": "missing",
            },
            {
                "rank": 2,
                "name": "refresh_crypto_route_operator_brief",
                "script": expected_refresh_operator_script,
                "command_hint": expected_refresh_operator_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_operator_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_operator_artifact_path_hint,
                "current_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
                "current_status": "ok",
                "current_as_of": "2026-03-10T12:00:00+00:00",
                "current_age_minutes": 30,
                "current_recency": "carry_over",
                "checkpoint_state": "carry_over",
            },
            {
                "rank": 3,
                "name": "refresh_hot_universe_research_embedding",
                "script": expected_refresh_research_script,
                "command_hint": expected_refresh_research_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_research_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_research_artifact_path_hint,
                "current_artifact": str(review_dir / "20260310T120000Z_hot_universe_research.json"),
                "current_status": "ok",
                "current_as_of": "2026-03-10T12:00:00+00:00",
                "current_age_minutes": 30,
                "current_recency": "carry_over",
                "checkpoint_state": "carry_over",
            },
            {
                "rank": 4,
                "name": "refresh_commodity_handoff",
                "script": expected_refresh_followup_script,
                "command_hint": expected_refresh_followup_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_followup_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_followup_artifact_path_hint,
                "current_artifact": "",
                "current_status": "",
                "current_as_of": "",
                "current_age_minutes": None,
                "current_recency": "unknown",
                "checkpoint_state": "missing",
            },
        ]
    assert payload["operator_focus_slot_refresh_backlog"] == [
        {
            "slot": "secondary",
            "symbol": "BNBUSDT",
            "action": "consider_refresh_before_promotion",
            "source_kind": "crypto_route",
            "source_status": "ok",
            "source_recency": "carry_over",
            "source_health": "carry_over_ok",
            "source_age_minutes": 30,
            "source_as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
        }
    ]
    assert payload["operator_source_refresh_queue"] == [
        {
            "rank": 1,
            "slot": "secondary",
            "symbol": "BNBUSDT",
            "action": "consider_refresh_before_promotion",
            "source_kind": "crypto_route",
            "source_status": "ok",
            "source_recency": "carry_over",
            "source_health": "carry_over_ok",
            "source_age_minutes": 30,
            "source_as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
        }
    ]
    assert payload["operator_source_refresh_checklist"] == [
        {
            "rank": 1,
            "slot": "secondary",
            "symbol": "BNBUSDT",
            "action": "consider_refresh_before_promotion",
            "source_kind": "crypto_route",
            "source_status": "ok",
            "source_recency": "carry_over",
            "source_health": "carry_over_ok",
            "source_age_minutes": 30,
            "source_as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
            "state": "refresh_recommended",
            "blocker_detail": "crypto_route artifact is ok and carry_over, age=30m",
            "done_when": "BNBUSDT receives a fresh crypto_route artifact before promotion",
            "recipe_script": expected_refresh_recipe_script,
            "recipe_command_hint": expected_refresh_recipe_command,
            "recipe_expected_status": "ok",
            "recipe_expected_artifact_kind": expected_refresh_recipe_artifact_kind,
            "recipe_expected_artifact_path_hint": expected_refresh_recipe_artifact_path_hint,
            "recipe_note": "guarded entrypoint refreshes native crypto route sources before the remaining pipeline steps",
            "recipe_followup_script": expected_refresh_followup_script,
            "recipe_followup_command_hint": expected_refresh_followup_command,
            "recipe_verify_hint": "rerun commodity refresh and confirm BNBUSDT leaves operator_source_refresh_queue",
            "recipe_steps_brief": "1:refresh_crypto_route_brief | 2:refresh_crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding | 4:refresh_commodity_handoff",
            "recipe_step_checkpoint_brief": expected_refresh_step_checkpoint_brief,
            "recipe_steps": [
                    {
                        "rank": 1,
                        "name": "refresh_crypto_route_brief",
                        "script": expected_refresh_recipe_script,
                        "command_hint": expected_refresh_recipe_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_recipe_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_recipe_artifact_path_hint,
                        "current_artifact": "",
                        "current_status": "",
                        "current_as_of": "",
                        "current_age_minutes": None,
                        "current_recency": "unknown",
                        "checkpoint_state": "missing",
                    },
                    {
                        "rank": 2,
                        "name": "refresh_crypto_route_operator_brief",
                        "script": expected_refresh_operator_script,
                        "command_hint": expected_refresh_operator_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_operator_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_operator_artifact_path_hint,
                        "current_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
                        "current_status": "ok",
                        "current_as_of": "2026-03-10T12:00:00+00:00",
                        "current_age_minutes": 30,
                        "current_recency": "carry_over",
                        "checkpoint_state": "carry_over",
                    },
                    {
                        "rank": 3,
                        "name": "refresh_hot_universe_research_embedding",
                        "script": expected_refresh_research_script,
                        "command_hint": expected_refresh_research_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_research_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_research_artifact_path_hint,
                        "current_artifact": str(review_dir / "20260310T120000Z_hot_universe_research.json"),
                        "current_status": "ok",
                        "current_as_of": "2026-03-10T12:00:00+00:00",
                        "current_age_minutes": 30,
                        "current_recency": "carry_over",
                        "checkpoint_state": "carry_over",
                    },
                    {
                        "rank": 4,
                        "name": "refresh_commodity_handoff",
                        "script": expected_refresh_followup_script,
                        "command_hint": expected_refresh_followup_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_followup_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_followup_artifact_path_hint,
                        "current_artifact": "",
                        "current_status": "",
                        "current_as_of": "",
                        "current_age_minutes": None,
                        "current_recency": "unknown",
                        "checkpoint_state": "missing",
                    },
                ],
            }
        ]
    assert payload["operator_focus_slots"] == [
        {
            "slot": "primary",
            "area": "commodity_execution_close_evidence",
            "target": "commodity-paper-execution:metals_all:XAUUSD",
            "symbol": "XAUUSD",
            "action": "wait_for_paper_execution_close_evidence",
            "reason": "paper_execution_close_evidence_pending",
            "state": "waiting",
            "blocker_detail": "paper execution evidence is present, but position is still OPEN; waiting for close evidence",
            "done_when": "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available",
            "source_kind": "commodity_execution_retro",
            "source_artifact": str(review_dir / "20260310T122755Z_commodity_paper_execution_retro.json"),
            "source_status": "ok",
            "source_as_of": "2026-03-10T12:27:55+00:00",
            "source_age_minutes": 2,
            "source_recency": "fresh",
            "source_health": "ready",
            "source_refresh_action": "read_current_artifact",
        },
        {
            "slot": "followup",
            "area": "commodity_fill_evidence",
            "target": "commodity-paper-execution:metals_all:XAGUSD",
            "symbol": "XAGUSD",
            "action": "wait_for_paper_execution_fill_evidence",
            "reason": "paper_execution_fill_evidence_pending",
            "state": "waiting",
            "blocker_detail": "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26",
            "done_when": "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols",
            "source_kind": "commodity_execution_review",
            "source_artifact": str(review_dir / "20260310T122751Z_commodity_paper_execution_review.json"),
            "source_status": "ok",
            "source_as_of": "2026-03-10T12:27:51+00:00",
            "source_age_minutes": 2,
            "source_recency": "fresh",
            "source_health": "ready",
            "source_refresh_action": "read_current_artifact",
        },
        {
            "slot": "secondary",
            "area": "crypto_route",
            "target": "BNBUSDT",
            "symbol": "BNBUSDT",
            "action": "watch_priority_until_long_window_confirms",
            "reason": "secondary_focus",
            "state": "watch",
            "blocker_detail": "long-window confirmation still missing",
            "done_when": "BNBUSDT upgrades from priority watch to deploy or leaves priority watch",
            "source_kind": "crypto_route",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
            "source_status": "ok",
            "source_as_of": "2026-03-10T12:00:00+00:00",
            "source_age_minutes": 30,
            "source_recency": "carry_over",
            "source_health": "carry_over_ok",
            "source_refresh_action": "consider_refresh_before_promotion",
        },
    ]
    assert payload["secondary_focus_area"] == "crypto_route"
    assert payload["secondary_focus_target"] == "BNBUSDT"
    assert payload["secondary_focus_symbol"] == "BNBUSDT"
    assert payload["secondary_focus_action"] == "watch_priority_until_long_window_confirms"
    assert payload["secondary_focus_reason"] == "secondary_focus"
    assert payload["secondary_focus_state"] == "watch"
    assert payload["secondary_focus_blocker_detail"] == "long-window confirmation still missing"
    assert payload["secondary_focus_done_when"] == (
        "BNBUSDT upgrades from priority watch to deploy or leaves priority watch"
    )
    assert payload["operator_action_queue_brief"] == (
        "1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence"
        " | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence"
        " | 3:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms"
    )
    assert payload["operator_action_checklist_brief"] == (
        "1:waiting:XAUUSD:wait_for_paper_execution_close_evidence"
        " | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence"
        " | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms"
    )
    assert "focus-slot-refresh-backlog: secondary:BNBUSDT:consider_refresh_before_promotion" in payload["summary_text"]
    assert "focus-slot-promotion-gate: promotion_guarded_by_source_freshness:2/3" in payload["summary_text"]
    assert "focus-slot-readiness-gate: readiness_guarded_by_source_freshness:2/3" in payload["summary_text"]
    assert "source-refresh-queue: 1:secondary:BNBUSDT:consider_refresh_before_promotion" in payload["summary_text"]
    assert "source-refresh-checklist: 1:refresh_recommended:BNBUSDT:consider_refresh_before_promotion" in payload["summary_text"]
    assert payload["operator_action_queue"] == [
        {
            "rank": 1,
            "area": "commodity_execution_close_evidence",
            "target": "commodity-paper-execution:metals_all:XAUUSD",
            "symbol": "XAUUSD",
            "action": "wait_for_paper_execution_close_evidence",
            "reason": "paper_execution_close_evidence_pending",
        },
        {
            "rank": 2,
            "area": "commodity_fill_evidence",
            "target": "commodity-paper-execution:metals_all:XAGUSD",
            "symbol": "XAGUSD",
            "action": "wait_for_paper_execution_fill_evidence",
            "reason": "paper_execution_fill_evidence_pending",
        },
        {
            "rank": 3,
            "area": "crypto_route",
            "target": "BNBUSDT",
            "symbol": "BNBUSDT",
            "action": "watch_priority_until_long_window_confirms",
            "reason": "secondary_focus",
        },
    ]
    assert payload["operator_action_checklist"] == [
        {
            "rank": 1,
            "area": "commodity_execution_close_evidence",
            "target": "commodity-paper-execution:metals_all:XAUUSD",
            "symbol": "XAUUSD",
            "action": "wait_for_paper_execution_close_evidence",
            "reason": "paper_execution_close_evidence_pending",
            "state": "waiting",
            "blocker_detail": "paper execution evidence is present, but position is still OPEN; waiting for close evidence",
            "done_when": "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available",
        },
        {
            "rank": 2,
            "area": "commodity_fill_evidence",
            "target": "commodity-paper-execution:metals_all:XAGUSD",
            "symbol": "XAGUSD",
            "action": "wait_for_paper_execution_fill_evidence",
            "reason": "paper_execution_fill_evidence_pending",
            "state": "waiting",
            "blocker_detail": "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26",
            "done_when": "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols",
        },
        {
            "rank": 3,
            "area": "crypto_route",
            "target": "BNBUSDT",
            "symbol": "BNBUSDT",
            "action": "watch_priority_until_long_window_confirms",
            "reason": "secondary_focus",
            "state": "watch",
            "blocker_detail": "long-window confirmation still missing",
            "done_when": "BNBUSDT upgrades from priority watch to deploy or leaves priority watch",
        },
    ]
    assert payload["commodity_remainder_focus_area"] == "commodity_fill_evidence"
    assert payload["commodity_remainder_focus_target"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["commodity_remainder_focus_symbol"] == "XAGUSD"
    assert payload["commodity_remainder_focus_action"] == "wait_for_paper_execution_fill_evidence"
    assert payload["commodity_remainder_focus_reason"] == "paper_execution_fill_evidence_pending"
    assert payload["commodity_remainder_focus_signal_date"] == "2026-01-26"
    assert payload["commodity_remainder_focus_signal_age_days"] == 42
    assert payload["commodity_next_fill_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["commodity_next_fill_evidence_execution_symbol"] == "XAGUSD"
    assert payload["commodity_fill_evidence_pending_count"] == 2
    assert payload["commodity_execution_bridge_stale_signal_dates"] == {
        "COPPER": "2026-01-29",
        "XAGUSD": "2026-01-26",
    }
    assert payload["commodity_execution_bridge_stale_signal_age_days"] == {
        "COPPER": 39,
        "XAGUSD": 42,
    }
    assert payload["commodity_stale_signal_watch_brief"] == "XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29"
    assert payload["commodity_stale_signal_watch_next_execution_id"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["commodity_stale_signal_watch_next_symbol"] == "XAGUSD"
    assert payload["commodity_stale_signal_watch_next_signal_date"] == "2026-01-26"
    assert payload["commodity_stale_signal_watch_next_signal_age_days"] == 42
    assert payload["commodity_focus_evidence_item_source"] == "retro"
    assert payload["commodity_focus_evidence_summary"]["paper_entry_price"] == 5198.10009765625
    assert payload["commodity_focus_evidence_summary"]["paper_stop_price"] == 4847.7998046875
    assert payload["commodity_focus_evidence_summary"]["paper_target_price"] == 5758.58056640625
    assert payload["commodity_focus_evidence_summary"]["paper_signal_price_reference_source"] == "yfinance:GC=F"
    assert payload["commodity_focus_lifecycle_status"] == "open_position_wait_close_evidence"
    assert payload["commodity_focus_lifecycle_brief"] == "open_position_wait_close_evidence:XAUUSD"
    assert payload["commodity_focus_lifecycle_blocker_detail"] == (
        "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
    )
    assert payload["commodity_focus_lifecycle_done_when"] == (
        "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
    )
    assert payload["commodity_execution_close_evidence_status"] == "close_evidence_pending"
    assert payload["commodity_execution_close_evidence_brief"] == "close_evidence_pending:XAUUSD"
    assert payload["commodity_execution_close_evidence_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_execution_close_evidence_symbol"] == "XAUUSD"
    assert payload["commodity_execution_close_evidence_blocker_detail"] == (
        "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
    )
    assert payload["commodity_execution_close_evidence_done_when"] == (
        "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
    )
    assert payload["commodity_review_pending_symbols"] == []
    assert payload["commodity_review_close_evidence_pending_count"] == 1
    assert payload["commodity_review_close_evidence_pending_symbols"] == ["XAUUSD"]
    assert payload["commodity_next_review_close_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_review_close_evidence_execution_symbol"] == "XAUUSD"
    assert payload["commodity_retro_pending_symbols"] == []
    assert payload["commodity_close_evidence_pending_count"] == 1
    assert payload["commodity_close_evidence_pending_symbols"] == ["XAUUSD"]
    assert payload["commodity_next_close_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_close_evidence_execution_symbol"] == "XAUUSD"
    assert payload["commodity_review_fill_evidence_pending_symbols"] == ["XAGUSD", "COPPER"]
    assert payload["commodity_retro_fill_evidence_pending_symbols"] == ["XAGUSD", "COPPER"]
    assert "commodity-execution-review-status: paper-execution-close-evidence-pending-fill-remainder" in payload["summary_text"]
    assert "commodity-execution-retro-status: paper-execution-close-evidence-pending-fill-remainder" in payload["summary_text"]
    assert "commodity-review-pending-symbols: -" in payload["summary_text"]
    assert "commodity-review-close-evidence-pending-count: 1" in payload["summary_text"]
    assert "commodity-review-close-evidence-pending-symbols: XAUUSD" in payload["summary_text"]
    assert "commodity-retro-pending-symbols: -" in payload["summary_text"]
    assert "commodity-close-evidence-pending-symbols: XAUUSD" in payload["summary_text"]
    assert "commodity-fill-evidence-pending-symbols: XAGUSD, COPPER" in payload["summary_text"]
    assert "next-focus-state: waiting" in payload["summary_text"]
    assert (
        "next-focus-blocker: paper execution evidence is present, but position is still OPEN; waiting for close evidence"
        in payload["summary_text"]
    )
    assert (
        "next-focus-done-when: XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
        in payload["summary_text"]
    )
    assert "followup-focus: commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence" in payload["summary_text"]
    assert "followup-focus-state: waiting" in payload["summary_text"]
    assert "followup-focus-blocker: paper execution fill evidence not written; stale directional signal 42d since 2026-01-26" in payload["summary_text"]
    assert "followup-focus-done-when: XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols" in payload["summary_text"]
    assert "secondary-focus: crypto_route:BNBUSDT:watch_priority_until_long_window_confirms" in payload["summary_text"]
    assert "secondary-focus-state: watch" in payload["summary_text"]
    assert "secondary-focus-blocker: long-window confirmation still missing" in payload["summary_text"]
    assert "secondary-focus-done-when: BNBUSDT upgrades from priority watch to deploy or leaves priority watch" in payload["summary_text"]
    assert "focus-slots: primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | secondary:watch:BNBUSDT:watch_priority_until_long_window_confirms" in payload["summary_text"]
    assert "focus-slot-sources: primary:commodity_execution_retro | followup:commodity_execution_review | secondary:crypto_route" in payload["summary_text"]
    assert "focus-slot-source-status: primary:ok@2026-03-10T12:27:55+00:00 | followup:ok@2026-03-10T12:27:51+00:00 | secondary:ok@2026-03-10T12:00:00+00:00" in payload["summary_text"]
    assert "focus-slot-source-recency: primary:fresh:2m | followup:fresh:2m | secondary:carry_over:30m" in payload["summary_text"]
    assert "focus-slot-source-health: primary:ready:read_current_artifact | followup:ready:read_current_artifact | secondary:carry_over_ok:consider_refresh_before_promotion" in payload["summary_text"]
    assert "action-queue: 1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence | 3:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms" in payload["summary_text"]
    assert "action-checklist: 1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-dates: COPPER:2026-01-29, XAGUSD:2026-01-26" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-age-days: COPPER:39, XAGUSD:42" in payload["summary_text"]
    assert "commodity-stale-signal-watch: XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-id: commodity-paper-execution:metals_all:XAGUSD" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-symbol: XAGUSD" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-signal-date: 2026-01-26" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-signal-age-days: 42" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-date: 2026-01-26" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-age-days: 42" in payload["summary_text"]
    assert "commodity-focus-paper-evidence: source=retro entry=5198.100098 stop=4847.799805 target=5758.580566 quote=0.158961 status=OPEN ref=yfinance:GC=F" in payload["summary_text"]
    assert "commodity-focus-lifecycle: open_position_wait_close_evidence:XAUUSD" in payload["summary_text"]
    assert "commodity-close-evidence: close_evidence_pending:XAUUSD" in payload["summary_text"]
    assert (
        "commodity-remainder-focus: commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence"
        in payload["summary_text"]
    )


def test_build_hot_universe_operator_brief_falls_back_to_queue_when_fill_evidence_missing(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all", "precious_metals"],
                "focus_with_regime_filter_batches": ["energy_liquids"],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": ["commodities_benchmark"],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
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
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-awaiting-fill-evidence",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_regime_gate": "paper_only",
            "review_item_count": 3,
            "actionable_review_item_count": 0,
            "fill_evidence_pending_count": 3,
            "next_review_execution_id": "",
            "next_review_execution_symbol": "",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
            "review_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "review_status": "awaiting_paper_execution_fill",
                    "paper_execution_evidence_present": False,
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-awaiting-fill-evidence",
            "execution_retro_status": "paper-execution-awaiting-fill-evidence",
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
            "retro_item_count": 3,
            "actionable_retro_item_count": 0,
            "fill_evidence_pending_count": 3,
            "next_retro_execution_id": "",
            "next_retro_execution_symbol": "",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_fill",
                    "review_status": "awaiting_paper_execution_fill",
                    "paper_execution_evidence_present": False,
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_review_status"] == "paper-execution-awaiting-fill-evidence"
    assert payload["commodity_execution_retro_status"] == "paper-execution-awaiting-fill-evidence"
    assert payload["commodity_next_review_execution_id"] == ""
    assert payload["commodity_next_retro_execution_id"] == ""
    assert payload["operator_status"] == "commodity-paper-execution-queued-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:queue:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_queue"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "inspect_paper_execution_queue"
    assert payload["next_focus_reason"] == "paper_execution_queued"
    assert "commodity-execution-review-status: paper-execution-awaiting-fill-evidence" in payload["summary_text"]
    assert "commodity-execution-retro-status: paper-execution-awaiting-fill-evidence" in payload["summary_text"]


def test_build_hot_universe_operator_brief_surfaces_commodity_gap_report(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-awaiting-fill-evidence",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_retro_status": "paper-execution-awaiting-fill-evidence",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": [
                "queue_symbols_missing_from_core_universe",
                "core_universe_crypto_only",
                "queue_symbols_missing_from_trade_plans",
            ],
            "root_cause_lines": [
                "Queue symbols are absent from config core universe: XAUUSD, XAGUSD, COPPER.",
                "Config core universe remains crypto-only.",
            ],
            "recommended_actions": [
                "Keep commodity queue in research/paper-planning mode until a real paper execution bridge exists."
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_commodity_execution_gap_status"] == "ok"
    assert payload["commodity_execution_gap_status"] == "blocking_gap_active"
    assert payload["commodity_execution_gap_decision"] == "do_not_assume_commodity_paper_execution_active"
    assert payload["commodity_execution_gap_reason_codes"] == [
        "queue_symbols_missing_from_core_universe",
        "core_universe_crypto_only",
        "queue_symbols_missing_from_trade_plans",
    ]
    assert payload["commodity_execution_gap_root_cause_lines"] == [
        "Queue symbols are absent from config core universe: XAUUSD, XAGUSD, COPPER.",
        "Config core universe remains crypto-only.",
    ]
    assert payload["commodity_execution_gap_recommended_actions"] == [
        "Keep commodity queue in research/paper-planning mode until a real paper execution bridge exists."
    ]
    assert payload["commodity_execution_gap_batch"] == ""
    assert payload["commodity_execution_gap_next_execution_id"] == ""
    assert payload["commodity_execution_gap_next_execution_symbol"] == ""
    assert payload["commodity_gap_focus_batch"] == "metals_all"
    assert payload["commodity_gap_focus_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_gap_focus_symbol"] == "XAUUSD"
    assert payload["operator_status"] == "commodity-paper-execution-gap-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:gap:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_gap"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "resolve_commodity_paper_execution_gap"
    assert payload["next_focus_reason"] == "commodity_execution_gap_active"
    assert "commodity-execution-gap-status: blocking_gap_active" in payload["summary_text"]
    assert "commodity-execution-gap-decision: do_not_assume_commodity_paper_execution_active" in payload["summary_text"]
    assert "commodity-gap-root-cause: Queue symbols are absent from config core universe: XAUUSD, XAGUSD, COPPER." in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_missing_directional_signal"],
            "root_cause_lines": ["Fresh signal-to-order tickets still return signal_not_found."],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_missing_directional_signal",
            "signal_missing_count": 3,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_blocked_symbol": "XAUUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_commodity_execution_bridge_status"] == "ok"
    assert payload["commodity_execution_bridge_status"] == "blocked_missing_directional_signal"
    assert payload["commodity_execution_bridge_next_blocked_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_execution_bridge_next_blocked_symbol"] == "XAUUSD"
    assert payload["commodity_execution_bridge_signal_missing_count"] == 3
    assert payload["commodity_execution_bridge_signal_stale_count"] == 0
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-blocked:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "restore_commodity_directional_signal"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_missing_directional_signal"
    assert "commodity-execution-bridge-status: blocked_missing_directional_signal" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_stale_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent; the latest bridgeable commodity signals are stale."],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_stale_directional_signal",
            "signal_missing_count": 0,
            "signal_stale_count": 3,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_blocked_symbol": "XAUUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "blocked_stale_directional_signal"
    assert payload["commodity_execution_bridge_signal_missing_count"] == 0
    assert payload["commodity_execution_bridge_signal_stale_count"] == 3
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-stale:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_action"] == "restore_commodity_directional_signal"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_stale_directional_signal"
    assert "commodity-execution-bridge-status: blocked_stale_directional_signal" in payload["summary_text"]
    assert "commodity-execution-bridge-signal-stale-count: 3" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_partial_stale_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD."],
            "queue_symbols_with_stale_directional_signal_dates": {"XAGUSD": "2026-01-26"},
            "queue_symbols_with_stale_directional_signal_age_days": {"XAGUSD": 42},
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_partially_bridged_stale_remainder",
            "already_present_count": 1,
            "already_bridged_symbols": ["XAUUSD"],
            "signal_missing_count": 0,
            "signal_stale_count": 1,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "bridge_partially_bridged_stale_remainder"
    assert payload["commodity_execution_bridge_already_present_count"] == 1
    assert payload["commodity_execution_bridge_already_bridged_symbols"] == ["XAUUSD"]
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-stale:XAGUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["next_focus_symbol"] == "XAGUSD"
    assert payload["next_focus_action"] == "restore_commodity_directional_signal"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_stale_directional_signal"
    assert payload["commodity_execution_bridge_stale_signal_dates"] == {"XAGUSD": "2026-01-26"}
    assert payload["commodity_execution_bridge_stale_signal_age_days"] == {"XAGUSD": 42}
    assert payload["commodity_remainder_focus_signal_date"] == "2026-01-26"
    assert payload["commodity_remainder_focus_signal_age_days"] == 42
    assert "commodity-execution-bridge-already-present-count: 1" in payload["summary_text"]
    assert "commodity-execution-bridge-already-bridged-symbols: XAUUSD" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-dates: XAGUSD:2026-01-26" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-age-days: XAGUSD:42" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-date: 2026-01-26" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-age-days: 42" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_proxy_price_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_proxy_price_reference_only", "queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": [
                "Commodity directional tickets still use proxy-market prices rather than executable instrument prices."
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_proxy_price_reference_only",
            "signal_missing_count": 0,
            "signal_stale_count": 3,
            "signal_proxy_price_only_count": 3,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_blocked_symbol": "XAUUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "blocked_proxy_price_reference_only"
    assert payload["commodity_execution_bridge_signal_missing_count"] == 0
    assert payload["commodity_execution_bridge_signal_stale_count"] == 3
    assert payload["commodity_execution_bridge_signal_proxy_price_only_count"] == 3
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-proxy:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_action"] == "normalize_commodity_execution_price_reference"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_proxy_price_reference_only"
    assert "commodity-execution-bridge-status: blocked_proxy_price_reference_only" in payload["summary_text"]
    assert "commodity-execution-bridge-signal-proxy-price-only-count: 3" in payload["summary_text"]


def test_build_hot_universe_operator_brief_keeps_review_focus_when_bridge_only_partially_blocked(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending",
            "actionable_review_item_count": 1,
            "next_review_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_review_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_retro_status": "paper-execution-awaiting-fill-evidence",
            "actionable_retro_item_count": 0,
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD and COPPER."],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_partially_bridged_stale_remainder",
            "already_present_count": 1,
            "already_bridged_symbols": ["XAUUSD"],
            "signal_missing_count": 0,
            "signal_stale_count": 2,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "bridge_partially_bridged_stale_remainder"
    assert payload["commodity_execution_bridge_already_present_count"] == 1
    assert payload["commodity_execution_bridge_already_bridged_symbols"] == ["XAUUSD"]
    assert payload["commodity_execution_review_status"] == "paper-execution-review-pending"
    assert payload["operator_status"] == "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:review:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_review"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "review_paper_execution"
    assert payload["next_focus_reason"] == "paper_execution_review_pending"
    assert "commodity-execution-bridge-status: bridge_partially_bridged_stale_remainder" in payload["summary_text"]


def test_build_hot_universe_operator_brief_uses_explicit_execution_artifacts(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    queue_path = review_dir / "20260311T084601Z_commodity_paper_execution_queue.json"
    review_path = review_dir / "20260311T084602Z_commodity_paper_execution_review.json"
    retro_path = review_dir / "20260311T084603Z_commodity_paper_execution_retro.json"
    gap_path = review_dir / "20260311T084604Z_commodity_paper_execution_gap_report.json"
    bridge_path = review_dir / "20260311T084600Z_commodity_paper_execution_bridge.json"
    _write_json(queue_path, {"status": "ok", "execution_queue_status": "paper-execution-queued", "execution_symbols": ["COPPER"]})
    _write_json(
        review_path,
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending",
            "actionable_review_item_count": 1,
            "next_review_execution_id": "commodity-paper-execution:metals_all:COPPER",
            "next_review_execution_symbol": "COPPER",
        },
    )
    _write_json(
        retro_path,
        {
            "status": "ok",
            "execution_retro_status": "paper-execution-retro-pending",
            "actionable_retro_item_count": 1,
            "next_retro_execution_id": "commodity-paper-execution:metals_all:COPPER",
            "next_retro_execution_symbol": "COPPER",
        },
    )
    _write_json(
        gap_path,
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD."],
        },
    )
    _write_json(
        bridge_path,
        {
            "status": "ok",
            "bridge_status": "blocked_stale_directional_signal",
            "signal_stale_count": 1,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--commodity-execution-queue-json",
            str(queue_path),
            "--commodity-execution-review-json",
            str(review_path),
            "--commodity-execution-retro-json",
            str(retro_path),
            "--commodity-execution-gap-json",
            str(gap_path),
            "--commodity-execution-bridge-json",
            str(bridge_path),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_commodity_execution_queue_artifact"] == str(queue_path.resolve())
    assert payload["source_commodity_execution_review_artifact"] == str(review_path.resolve())
    assert payload["source_commodity_execution_retro_artifact"] == str(retro_path.resolve())
    assert payload["source_commodity_execution_gap_artifact"] == str(gap_path.resolve())
    assert payload["source_commodity_execution_bridge_artifact"] == str(bridge_path.resolve())
    assert payload["commodity_next_retro_execution_symbol"] == "COPPER"
