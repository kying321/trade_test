from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_backtest_slice.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_backtest_slice_marks_bias_only_slice_for_cross_section_backtest(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  structure_timeframe: 4h
  execution_timeframe: 15m
  micro_structure_timeframes: [1m, 5m]
  holding_window_minutes: {min: 15, max: 180}
  trigger_stack:
    - 4h_profile_location
    - liquidity_sweep
    - 1m_5m_mss_or_choch
    - 15m_cvd_divergence_or_confirmation
    - fvg_ob_breaker_retest
    - 15m_reversal_or_breakout_candle
  session_liquidity_map:
    - asia_high_low
    - london_high_low
    - prior_day_high_low
    - equal_highs_lows
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT:deprioritize_flow",
            "focus_review_blocker_detail": (
                "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, "
                "fvg_ob_breaker_retest, cvd_confirmation"
            ),
            "focus_review_done_when": "liquidity_sweep + mss + cvd_confirmation clear",
            "shortline_market_state_brief": "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade",
            "shortline_execution_gate_brief": (
                "4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> "
                "15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle"
            ),
            "focus_execution_micro_veto": "watch_only:continuation:low_sample_or_gap_risk",
        },
    )
    _write_json(
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                        "cvd_confirmation",
                    ],
                    "micro_signals": {"context": "watch_only"},
                    "focus_execution_micro_reasons": ["trust_low", "low_sample_or_gap_risk"],
                },
                {"symbol": "BTCUSDT", "execution_state": "Bias_Only", "route_action": "watch"},
                {"symbol": "ETHUSDT", "execution_state": "Bias_Only", "route_action": "watch"},
                {"symbol": "BNBUSDT", "execution_state": "Bias_Only", "route_action": "watch"},
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "takeaway": "watch-only until micro recovers",
            "semantic_takeaway": "queue remains watch-only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
            "queue_stack_brief": "crypto_hot -> crypto_majors",
            "runtime_queue": [
                {
                    "batch": "crypto_hot",
                    "eligible_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
                    "matching_symbols": [
                        {
                            "symbol": "SOLUSDT",
                            "classification": "Bias_Only",
                            "cvd_context_mode": "watch_only",
                            "cvd_veto_hint": "low_sample_or_gap_risk",
                        }
                    ],
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
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:10:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["slice_status"] == "selected_watch_only_orderflow_slice"
    assert payload["research_decision"] == "run_watch_only_orderflow_cross_section_backtest"
    assert payload["selected_symbol"] == "SOLUSDT"
    assert payload["selected_focus_batch"] == "crypto_hot"
    assert payload["selected_focus_action"] == "defer_until_micro_recovers"
    assert payload["slice_universe"] == ["SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"]
    assert payload["trigger_stack"][0] == "4h_profile_location"
    assert payload["session_liquidity_map"][-1] == "equal_highs_lows"
    assert payload["comparison_rows"][0]["symbol"] == "SOLUSDT"
    assert payload["comparison_rows"][0]["missing_gates"] == [
        "liquidity_sweep",
        "mss",
        "fvg_ob_breaker_retest",
        "cvd_confirmation",
    ]
    assert "SOLUSDT remains Bias_Only" in payload["blocker_detail"]
