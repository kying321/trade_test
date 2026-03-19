from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_execution_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("crypto_shortline_execution_gate_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bars_for_symbol(symbol: str, *, setup_ready: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for day in range(1, 9):
        rows.append(
            {
                "ts": f"2026-03-{day:02d}",
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100.0,
                "source": "test",
                "asset_class": "crypto",
            }
        )
    if setup_ready:
        rows.extend(
            [
                {"ts": "2026-03-09", "symbol": symbol, "open": 99.4, "high": 100.8, "low": 95.0, "close": 100.4, "volume": 110.0, "source": "test", "asset_class": "crypto"},
                {"ts": "2026-03-10", "symbol": symbol, "open": 100.5, "high": 102.0, "low": 100.2, "close": 101.8, "volume": 120.0, "source": "test", "asset_class": "crypto"},
                {"ts": "2026-03-11", "symbol": symbol, "open": 102.7, "high": 103.4, "low": 102.5, "close": 103.0, "volume": 200.0, "source": "test", "asset_class": "crypto"},
                {"ts": "2026-03-12", "symbol": symbol, "open": 102.8, "high": 103.6, "low": 102.6, "close": 103.2, "volume": 220.0, "source": "test", "asset_class": "crypto"},
            ]
        )
    else:
        rows.extend(
            [
                {"ts": "2026-03-09", "symbol": symbol, "open": 99.8, "high": 100.5, "low": 99.4, "close": 99.9, "volume": 100.0, "source": "test", "asset_class": "crypto"},
                {"ts": "2026-03-10", "symbol": symbol, "open": 99.9, "high": 100.4, "low": 99.6, "close": 100.0, "volume": 100.0, "source": "test", "asset_class": "crypto"},
                {"ts": "2026-03-11", "symbol": symbol, "open": 100.0, "high": 100.3, "low": 99.7, "close": 99.8, "volume": 95.0, "source": "test", "asset_class": "crypto"},
                {"ts": "2026-03-12", "symbol": symbol, "open": 99.8, "high": 100.1, "low": 99.4, "close": 99.7, "volume": 90.0, "source": "test", "asset_class": "crypto"},
            ]
        )
    return rows


def test_script_builds_setup_ready_and_bias_only_rows(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bars_dir = output_root / "research" / "20260312_100000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    bars_path = bars_dir / "bars_used.csv"
    bars = pd.DataFrame(
        _bars_for_symbol("BTCUSDT", setup_ready=True)
        + _bars_for_symbol("ETHUSDT", setup_ready=False)
    )
    bars.to_csv(bars_path, index=False)

    route_path = review_dir / "20260312T100000Z_binance_indicator_symbol_route_handoff.json"
    _write_json(
        route_path,
        {
            "routes": [
                {"symbol": "BTCUSDT", "action": "deploy_price_state_only"},
                {"symbol": "ETHUSDT", "action": "watch_only"},
            ]
        },
    )
    micro_path = artifact_dir / "20260312T100000Z_micro_capture.json"
    _write_json(
        micro_path,
        {
            "selected_micro": [
                {
                    "symbol": "BTCUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 50,
                    "evidence_score": 0.95,
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "",
                    "cvd_context_mode": "continuation",
                    "micro_alignment": 0.35,
                },
                {
                    "symbol": "ETHUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 15,
                    "evidence_score": 0.40,
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "cvd_context_mode": "unclear",
                    "micro_alignment": 0.01,
                },
            ]
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "shortline:",
                "  default_market_state: Bias_Only",
                "  setup_ready_state: Setup_Ready",
                "  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade",
                "  profile_lookback_bars: 12",
                "  location_priority: [HVN, POC]",
                "  supported_symbols: [BTCUSDT, ETHUSDT]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--config",
            str(config_path),
            "--route-handoff-file",
            str(route_path),
            "--bars-file",
            str(bars_path),
            "--micro-capture-file",
            str(micro_path),
            "--now",
            "2026-03-12T10:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["gate_brief"] == "setup_ready:BTCUSDT"
    by_symbol = {row["symbol"]: row for row in payload["symbols"]}
    assert by_symbol["BTCUSDT"]["execution_state"] == "Setup_Ready"
    assert by_symbol["BTCUSDT"]["setup_direction"] == "LONG"
    assert by_symbol["ETHUSDT"]["execution_state"] == "Bias_Only"
    assert "route_state=watch:watch_only" in by_symbol["ETHUSDT"]["missing_gates"]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_script_blocks_stale_cvd_drift_even_if_structure_is_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bars_dir = output_root / "research" / "20260312_100000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    bars_path = bars_dir / "bars_used.csv"
    bars = pd.DataFrame(_bars_for_symbol("BTCUSDT", setup_ready=True))
    bars.to_csv(bars_path, index=False)

    route_path = review_dir / "20260312T100000Z_binance_indicator_symbol_route_handoff.json"
    _write_json(route_path, {"routes": [{"symbol": "BTCUSDT", "action": "deploy_price_state_only"}]})
    micro_path = artifact_dir / "20260312T100000Z_micro_capture.json"
    _write_json(
        micro_path,
        {
            "selected_micro": [
                {
                    "symbol": "BTCUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 50,
                    "evidence_score": 0.95,
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "",
                    "cvd_context_mode": "reversal",
                    "micro_alignment": 0.35,
                    "cvd_reference_age_minutes": 40,
                    "cvd_drift_risk": True,
                    "cvd_attack_side": "buyers",
                }
            ]
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "shortline:",
                "  default_market_state: Bias_Only",
                "  setup_ready_state: Setup_Ready",
                "  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade",
                "  profile_lookback_bars: 12",
                "  location_priority: [HVN, POC]",
                "  supported_symbols: [BTCUSDT]",
                "  cvd_local_window_minutes: 15",
                "  cvd_reference_max_age_minutes: 15",
                "  cvd_drift_guard_enabled: true",
                "  cvd_attack_confirmation_required: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--config",
            str(config_path),
            "--route-handoff-file",
            str(route_path),
            "--bars-file",
            str(bars_path),
            "--micro-capture-file",
            str(micro_path),
            "--now",
            "2026-03-12T10:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    row = payload["symbols"][0]
    assert row["execution_state"] == "Bias_Only"
    assert "cvd_local_window" in row["missing_gates"]
    assert "cvd_drift_guard" in row["missing_gates"]
    assert row["micro_signals"]["cvd_drift_risk"] is True
    assert row["micro_signals"]["attack_side"] == "buyers"


def test_script_prefers_live_bars_snapshot_overlay_for_route_symbol(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bars_dir = output_root / "research" / "20260312_100000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    base_bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(
        _bars_for_symbol("BTCUSDT", setup_ready=False)
        + _bars_for_symbol("ETHUSDT", setup_ready=False)
    ).to_csv(base_bars_path, index=False)

    live_bars_path = review_dir / "20260312T100100Z_crypto_shortline_live_bars_snapshot_bars.csv"
    pd.DataFrame(_bars_for_symbol("BTCUSDT", setup_ready=True)).to_csv(live_bars_path, index=False)
    live_snapshot_path = review_dir / "20260312T100100Z_crypto_shortline_live_bars_snapshot.json"
    _write_json(
        live_snapshot_path,
        {
            "snapshot_status": "bars_live_snapshot_ready",
            "route_symbol": "BTCUSDT",
            "bars_artifact": str(live_bars_path),
        },
    )

    route_path = review_dir / "20260312T100000Z_binance_indicator_symbol_route_handoff.json"
    _write_json(
        route_path,
        {
            "routes": [
                {"symbol": "BTCUSDT", "action": "deploy_price_state_only"},
                {"symbol": "ETHUSDT", "action": "watch_only"},
            ]
        },
    )
    micro_path = artifact_dir / "20260312T100000Z_micro_capture.json"
    _write_json(
        micro_path,
        {
            "selected_micro": [
                {
                    "symbol": "BTCUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 50,
                    "evidence_score": 0.95,
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "",
                    "cvd_context_mode": "continuation",
                    "micro_alignment": 0.35,
                },
                {
                    "symbol": "ETHUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 15,
                    "evidence_score": 0.40,
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "cvd_context_mode": "unclear",
                    "micro_alignment": 0.01,
                },
            ]
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "shortline:",
                "  default_market_state: Bias_Only",
                "  setup_ready_state: Setup_Ready",
                "  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade",
                "  profile_lookback_bars: 12",
                "  location_priority: [HVN, POC]",
                "  supported_symbols: [BTCUSDT, ETHUSDT]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--config",
            str(config_path),
            "--route-handoff-file",
            str(route_path),
            "--micro-capture-file",
            str(micro_path),
            "--now",
            "2026-03-12T10:02:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bars_source_kind"] == "merged_base_plus_live_snapshot"
    assert payload["live_bars_snapshot_artifact"] == str(live_snapshot_path)
    materialized_path = Path(payload["bars_artifact"])
    assert materialized_path.exists()
    merged = pd.read_csv(materialized_path)
    assert set(merged["symbol"].astype(str)) == {"BTCUSDT", "ETHUSDT"}
    btc_latest = merged.loc[merged["symbol"] == "BTCUSDT", "ts"].astype(str).max()
    assert btc_latest == "2026-03-12"
    by_symbol = {row["symbol"]: row for row in payload["symbols"]}
    assert by_symbol["BTCUSDT"]["execution_state"] == "Setup_Ready"
    assert by_symbol["ETHUSDT"]["execution_state"] == "Bias_Only"


def test_script_prefers_live_bars_snapshot_overlay_for_materialized_symbols(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bars_dir = output_root / "research" / "20260312_100000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    base_bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(
        _bars_for_symbol("BTCUSDT", setup_ready=False)
        + _bars_for_symbol("ETHUSDT", setup_ready=False)
    ).to_csv(base_bars_path, index=False)

    live_bars_path = review_dir / "20260312T100100Z_crypto_shortline_live_bars_snapshot_bars.csv"
    pd.DataFrame(
        _bars_for_symbol("BTCUSDT", setup_ready=True)
        + _bars_for_symbol("ETHUSDT", setup_ready=True)
    ).to_csv(live_bars_path, index=False)
    live_snapshot_path = review_dir / "20260312T100100Z_crypto_shortline_live_bars_snapshot.json"
    _write_json(
        live_snapshot_path,
        {
            "snapshot_status": "bars_live_snapshot_ready",
            "route_symbol": "BTCUSDT",
            "materialized_symbols": ["BTCUSDT", "ETHUSDT"],
            "bars_artifact": str(live_bars_path),
        },
    )

    route_path = review_dir / "20260312T100000Z_binance_indicator_symbol_route_handoff.json"
    _write_json(
        route_path,
        {
            "routes": [
                {"symbol": "BTCUSDT", "action": "deploy_price_state_only"},
                {"symbol": "ETHUSDT", "action": "deploy_price_state_only"},
            ]
        },
    )
    micro_path = artifact_dir / "20260312T100000Z_micro_capture.json"
    _write_json(
        micro_path,
        {
            "selected_micro": [
                {
                    "symbol": "BTCUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 50,
                    "evidence_score": 0.95,
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "",
                    "cvd_context_mode": "continuation",
                    "micro_alignment": 0.35,
                },
                {
                    "symbol": "ETHUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 50,
                    "evidence_score": 0.95,
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "",
                    "cvd_context_mode": "continuation",
                    "micro_alignment": 0.35,
                },
            ]
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "shortline:",
                "  default_market_state: Bias_Only",
                "  setup_ready_state: Setup_Ready",
                "  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade",
                "  profile_lookback_bars: 12",
                "  location_priority: [HVN, POC]",
                "  supported_symbols: [BTCUSDT, ETHUSDT]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--config",
            str(config_path),
            "--route-handoff-file",
            str(route_path),
            "--micro-capture-file",
            str(micro_path),
            "--now",
            "2026-03-12T10:02:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["bars_source_kind"] == "merged_base_plus_live_snapshot"
    by_symbol = {row["symbol"]: row for row in payload["symbols"]}
    assert by_symbol["BTCUSDT"]["execution_state"] == "Setup_Ready"
    assert by_symbol["ETHUSDT"]["execution_state"] == "Setup_Ready"


def test_script_adds_continuation_pattern_hint_for_bias_only_context(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bars_dir = output_root / "research" / "20260312_100000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(_bars_for_symbol("SOLUSDT", setup_ready=False)).to_csv(bars_path, index=False)

    route_path = review_dir / "20260312T100000Z_binance_indicator_symbol_route_handoff.json"
    _write_json(
        route_path,
        {"routes": [{"symbol": "SOLUSDT", "action": "deprioritize_flow"}]},
    )
    micro_path = artifact_dir / "20260312T100000Z_micro_capture.json"
    _write_json(
        micro_path,
        {
            "selected_micro": [
                {
                    "symbol": "SOLUSDT",
                    "schema_ok": True,
                    "time_sync_ok": True,
                    "gap_ok": True,
                    "trade_count": 500,
                    "evidence_score": 1.0,
                    "cvd_trust_tier_hint": "single_exchange_ok",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "cvd_context_mode": "continuation",
                    "micro_alignment": 0.14,
                    "key_level_confirmed": True,
                    "cvd_attack_side": "buyers",
                    "cvd_attack_presence": "buyers_attacking",
                    "cvd_attack_confirmation_ok": True,
                }
            ]
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "shortline:",
                "  default_market_state: Bias_Only",
                "  setup_ready_state: Setup_Ready",
                "  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade",
                "  profile_lookback_bars: 12",
                "  location_priority: [HVN, POC]",
                "  supported_symbols: [SOLUSDT]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--artifact-dir",
            str(artifact_dir),
            "--config",
            str(config_path),
            "--route-handoff-file",
            str(route_path),
            "--bars-file",
            str(bars_path),
            "--micro-capture-file",
            str(micro_path),
            "--now",
            "2026-03-12T10:03:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    row = payload["symbols"][0]
    assert row["execution_state"] == "Bias_Only"
    assert row["pattern_family_hint"] == "imbalance_continuation"
    assert row["pattern_stage_hint"] == "imbalance_retest"
    assert row["effective_missing_gates"] == [
        "fvg_ob_breaker_retest",
        "route_state=watch:deprioritize_flow",
    ]
    assert row["pattern_hint_brief"] == (
        "imbalance_continuation:imbalance_retest:"
        "fvg_ob_breaker_retest,route_state=watch:deprioritize_flow"
    )
