from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "live_risk_guard.py"
    spec = importlib.util.spec_from_file_location("live_risk_guard", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_ticket(review_dir: Path, *, allowed: bool = True) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "tickets": [
                    {
                        "symbol": "BTCUSDT",
                        "date": "2026-03-06",
                        "age_days": 0,
                        "allowed": bool(allowed),
                        "reasons": [] if allowed else ["size_below_min_notional"],
                        "signal": {"side": "LONG", "confidence": 72.0, "convexity_ratio": 3.6},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_takeover_snapshot(output_root: Path, *, quote_available: float = 20.0, open_notional: float = 2.0, closed_pnl: float = -0.2) -> None:
    review_dir = output_root / "review"
    artifacts_dir = output_root / "artifacts" / "broker_live_inbox"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    live_snapshot_path = artifacts_dir / "2026-03-06.json"
    live_snapshot_path.write_text(
        json.dumps(
            {
                "date": "2026-03-06",
                "generated_at": "2026-03-06T00:00:00Z",
                "source": "binance_spot",
                "open_positions": 1 if open_notional > 0 else 0,
                "closed_count": 1 if closed_pnl != 0 else 0,
                "closed_pnl": float(closed_pnl),
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "qty": 0.00003,
                        "market_price": 70000.0,
                        "notional": float(open_notional),
                    }
                ]
                if open_notional > 0
                else [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps(
            {
                "steps": {
                    "account_overview": {"quote_available": float(quote_available)},
                    "live_snapshot": {"path": str(live_snapshot_path), "closed_pnl": float(closed_pnl)},
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_backup_web_intel(
    output_root: Path,
    *,
    generated_at: str = "2099-01-01T00:00:00Z",
    expires_at: str = "2099-01-01T02:00:00Z",
    no_trade_list: list[dict[str, object]] | None = None,
    candidate_biases: list[dict[str, object]] | None = None,
    risk_flags: list[dict[str, object]] | None = None,
    status: str = "ok",
    authority: str = "risk_only",
    fallback_use_allowed: bool = True,
) -> None:
    state_dir = output_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "backup_web_intel.json").write_text(
        json.dumps(
            {
                "status": status,
                "generated_at": generated_at,
                "expires_at": expires_at,
                "fallback_use_allowed": bool(fallback_use_allowed),
                "fallback_trade_authority": authority,
                "no_trade_list": no_trade_list or [],
                "candidate_biases": candidate_biases or [],
                "risk_flags": risk_flags or [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_live_risk_guard_passes_with_fresh_ticket_and_small_exposure(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    _write_ticket(review_dir)
    _write_takeover_snapshot(output_root, quote_available=20.0, open_notional=2.0, closed_pnl=-0.2)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_guard.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--ticket-freshness-seconds",
            "900",
            "--max-daily-loss-ratio",
            "0.05",
            "--max-open-exposure-ratio",
            "0.50",
        ],
    )
    rc = mod.main()
    assert rc == 0

    generated = sorted(review_dir.glob("*_live_risk_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert bool(parsed.get("allowed", False)) is True
    assert str(parsed.get("status", "")) == "pass"
    fuse = json.loads((output_root / "state" / "live_risk_fuse.json").read_text(encoding="utf-8"))
    assert bool(fuse.get("allowed", False)) is True
    checksum = sorted(review_dir.glob("*_live_risk_guard_checksum.json"))
    assert checksum


def test_live_risk_guard_blocks_on_recent_panic_marker(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    _write_ticket(review_dir)
    _write_takeover_snapshot(output_root, quote_available=20.0, open_notional=1.0, closed_pnl=0.0)
    (output_root / "state").mkdir(parents=True, exist_ok=True)
    (output_root / "state" / "panic_close_all.json").write_text(
        json.dumps({"ts_utc": "2026-03-06T00:00:00Z", "reason": "socket_disconnect"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_guard.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--panic-cooldown-seconds",
            "999999999",
        ],
    )
    rc = mod.main()
    assert rc == 3

    generated = sorted(review_dir.glob("*_live_risk_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert bool(parsed.get("allowed", True)) is False
    assert "panic_cooldown_active" in list(parsed.get("reasons", []))
    fuse = json.loads((output_root / "state" / "live_risk_fuse.json").read_text(encoding="utf-8"))
    assert bool(fuse.get("allowed", True)) is False


def test_live_risk_guard_surfaces_ticket_market_scope_mismatch(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "tickets": [
                    {
                        "symbol": "SOLUSDT",
                        "date": "2026-03-06",
                        "age_days": 0,
                        "allowed": False,
                        "reasons": ["signal_market_scope_mismatch", "size_below_min_notional"],
                        "signal": {"side": "LONG", "confidence": 72.0, "convexity_ratio": 3.6},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_takeover_snapshot(output_root, quote_available=20.0, open_notional=0.0, closed_pnl=0.0)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "micro_capture_daemon_symbols": ["SOLUSDT"],
                    "binance_live_takeover_market": "spot",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_guard.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--ticket-freshness-seconds",
            "900",
        ],
    )
    rc = mod.main()
    assert rc == 3

    generated = sorted(review_dir.glob("*_live_risk_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert bool(parsed.get("allowed", True)) is False
    assert "ticket_missing:signal_market_scope_mismatch" in list(parsed.get("reasons", []))
    selected = parsed.get("ticket_selection", {}).get("selected", {})
    assert str(selected.get("reason", "")) == "signal_market_scope_mismatch"


def test_load_exposure_snapshot_prefers_market_specific_takeover_snapshot(tmp_path: Path) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    artifacts_dir = output_root / "artifacts" / "broker_live_inbox"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    spot_snapshot = artifacts_dir / "spot.json"
    pm_snapshot = artifacts_dir / "pm.json"
    spot_snapshot.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "qty": 0.00002,
                        "market_price": 100000.0,
                    }
                ],
                "closed_pnl": 0.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pm_snapshot.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "positionAmt": 0.001,
                        "markPrice": 73000.0,
                    }
                ],
                "closed_pnl": 12.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "latest_binance_live_takeover.json").write_text(
        json.dumps(
            {
                "market": "portfolio_margin_um",
                "steps": {
                    "account_overview": {"market": "portfolio_margin_um", "quote_available": 67.0},
                    "live_snapshot": {"path": str(pm_snapshot), "closed_pnl": 12.0},
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_binance_live_takeover_spot.json").write_text(
        json.dumps(
            {
                "market": "spot",
                "steps": {
                    "account_overview": {"market": "spot", "quote_available": 20.0},
                    "live_snapshot": {"path": str(spot_snapshot), "closed_pnl": 0.0},
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exposure = mod.load_exposure_snapshot(output_root, market="spot")
    assert str(exposure.get("latest_takeover_path", "")).endswith("latest_binance_live_takeover_spot.json")
    assert str(exposure.get("market", "")) == "spot"
    assert abs(float(exposure.get("quote_available", 0.0)) - 20.0) < 1e-12
    assert abs(float(exposure.get("open_exposure_notional", 0.0)) - 2.0) < 1e-12


def test_live_risk_guard_blocks_read_only_portfolio_margin_market(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    _write_ticket(review_dir, allowed=True)
    _write_takeover_snapshot(output_root, quote_available=20.0, open_notional=1.0, closed_pnl=0.0)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "micro_capture_daemon_symbols": ["BTCUSDT"],
                    "binance_live_takeover_market": "portfolio_margin_um",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_guard.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--ticket-freshness-seconds",
            "900",
            "--max-daily-loss-ratio",
            "0.05",
            "--max-open-exposure-ratio",
            "0.50",
        ],
    )
    rc = mod.main()
    assert rc == 3

    generated = sorted(review_dir.glob("*_live_risk_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert bool(parsed.get("allowed", True)) is False
    assert list(parsed.get("reasons", [])) == ["ticket_missing:target_market_read_only"]
    selected = ((parsed.get("ticket_selection", {}) or {}).get("selected", {}) or {})
    blocked = selected.get("blocked_candidate", {})
    assert isinstance(blocked, dict)
    assert list(blocked.get("ticket_reasons", [])) == ["target_market_read_only"]
    assert str(blocked.get("target_market", "")) == "portfolio_margin_um"


def test_live_risk_guard_refreshes_tickets_before_eval(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    _write_ticket(review_dir)
    _write_takeover_snapshot(output_root, quote_available=20.0, open_notional=1.0, closed_pnl=0.0)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        assert Path(cmd[1]).name == "build_order_ticket.py"
        return {
            "returncode": 0,
            "payload": {"summary": {"ticket_count": 1}},
            "stdout": "",
            "stderr": "",
            "timeout": False,
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_guard.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--refresh-tickets",
            "--ticket-symbols",
            "BTCUSDT,ETHUSDT",
            "--ticket-min-confidence",
            "61",
            "--ticket-min-convexity",
            "3.5",
            "--ticket-max-age-days",
            "9",
            "--skip-mutex",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert len(calls) == 1
    ticket_cmd = calls[0]
    assert ticket_cmd[ticket_cmd.index("--symbols") + 1] == "BTCUSDT,ETHUSDT"
    assert ticket_cmd[ticket_cmd.index("--min-confidence") + 1] == "61.0"
    assert ticket_cmd[ticket_cmd.index("--min-convexity") + 1] == "3.5"
    assert ticket_cmd[ticket_cmd.index("--max-age-days") + 1] == "9"

    generated = sorted(review_dir.glob("*_live_risk_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    ticket_refresh = parsed.get("ticket_refresh", {})
    assert isinstance(ticket_refresh, dict)
    assert int(ticket_refresh.get("returncode", 0)) == 0


def test_live_risk_guard_blocks_on_backup_web_intel_no_trade_symbol(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    _write_ticket(review_dir)
    _write_takeover_snapshot(output_root, quote_available=20.0, open_notional=1.0, closed_pnl=0.0)
    _write_backup_web_intel(
        output_root,
        no_trade_list=[{"symbol": "BTCUSDT", "reason": "event_uncertainty_high"}],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "micro_capture_daemon_symbols": ["BTCUSDT"],
                    "backup_web_intel_enabled": True,
                    "backup_web_intel_artifact_path": "state/backup_web_intel.json",
                    "backup_web_intel_max_age_seconds": 7200,
                    "backup_web_intel_block_on_no_trade": True,
                    "backup_web_intel_block_on_bias_conflict": True,
                    "backup_web_intel_block_severities": ["high"],
                    "backup_web_intel_required_authority": "risk_only",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_guard.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
        ],
    )
    rc = mod.main()
    assert rc == 3

    generated = sorted(review_dir.glob("*_live_risk_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert "backup_intel_no_trade_symbol" in list(parsed.get("reasons", []))
    backup = parsed.get("backup_intel", {})
    assert isinstance(backup, dict)
    assert bool(backup.get("active", False)) is True
    assert backup.get("status") == "active"
    assert backup.get("matched_no_trade_symbols") == [{"symbol": "BTCUSDT", "reason": "event_uncertainty_high"}]


def test_live_risk_guard_ignores_stale_backup_web_intel(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    _write_ticket(review_dir)
    _write_takeover_snapshot(output_root, quote_available=20.0, open_notional=1.0, closed_pnl=0.0)
    _write_backup_web_intel(
        output_root,
        generated_at="2026-03-01T00:00:00Z",
        expires_at="2026-03-01T01:00:00Z",
        no_trade_list=[{"symbol": "BTCUSDT", "reason": "should_be_ignored"}],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "micro_capture_daemon_symbols": ["BTCUSDT"],
                    "backup_web_intel_enabled": True,
                    "backup_web_intel_artifact_path": "state/backup_web_intel.json",
                    "backup_web_intel_max_age_seconds": 3600,
                    "backup_web_intel_block_on_no_trade": True,
                    "backup_web_intel_block_on_bias_conflict": True,
                    "backup_web_intel_block_severities": ["high"],
                    "backup_web_intel_required_authority": "risk_only",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_guard.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
        ],
    )
    rc = mod.main()
    assert rc == 0

    generated = sorted(review_dir.glob("*_live_risk_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert "backup_intel_no_trade_symbol" not in list(parsed.get("reasons", []))
    backup = parsed.get("backup_intel", {})
    assert isinstance(backup, dict)
    assert bool(backup.get("active", True)) is False
    assert backup.get("status") == "stale"
