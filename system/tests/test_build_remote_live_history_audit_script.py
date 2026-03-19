from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_remote_live_history_audit.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_remote_live_history_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_window_summary_reads_trade_telemetry_maps() -> None:
    mod = _load_module()
    payload = {
        "status": "probe_completed",
        "executed": False,
        "guarded_exec": {
            "artifact": "/tmp/guard.json",
            "takeover": {
                "payload": {
                    "market": "portfolio_margin_um",
                    "artifact": "/tmp/takeover.json",
                    "steps": {
                        "account_overview": {
                            "quote_available": 98.5,
                            "position_count": 1,
                        },
                        "live_snapshot": {
                            "open_positions": 1,
                            "closed_count": 4,
                            "closed_pnl": 12.75,
                        },
                        "trade_telemetry": {
                            "trades": 6,
                            "income_rows": 5,
                            "query_slice_hours": 168,
                            "trade_slice_count": 2,
                            "income_slice_count": 2,
                            "trade_count_by_symbol": {"BTCUSDT": 4, "ETHUSDT": 2},
                            "income_count_by_symbol": {"BTCUSDT": 3, "ETHUSDT": 2},
                            "income_pnl_by_symbol": {"BTCUSDT": 9.5, "ETHUSDT": 3.25},
                            "income_pnl_by_day": {"2026-03-11": 8.0, "2026-03-12": 4.75},
                        },
                        "risk_guard": {
                            "status": "ok",
                            "allowed": True,
                            "reasons": [],
                        },
                        "signal_selection": {
                            "blocked_candidate": {"symbol": "BTCUSDT"},
                        },
                    },
                }
            },
        },
    }

    out = mod.build_window_summary(window_hours=168, probe_payload=payload, probe_source="fixture")
    assert out["history_window_label"] == "7d"
    assert out["probe_status"] == "probe_completed"
    assert out["probe_mode"] == "direct"
    assert out["probe_transport"] == {}
    assert out["probe_returncode"] == 0
    assert out["probe_panic"] == ""
    assert out["probe_error_detail"] == ""
    assert out["trade_count"] == 6
    assert out["income_rows"] == 5
    assert out["query_slice_hours"] == 168
    assert out["trade_slice_count"] == 2
    assert out["income_slice_count"] == 2
    assert out["trade_count_by_symbol"] == {"BTCUSDT": 4, "ETHUSDT": 2}
    assert out["income_count_by_symbol"] == {"BTCUSDT": 3, "ETHUSDT": 2}
    assert out["income_pnl_by_symbol"] == {"BTCUSDT": 9.5, "ETHUSDT": 3.25}
    assert out["income_pnl_by_day"] == {"2026-03-11": 8.0, "2026-03-12": 4.75}


def test_build_window_summary_falls_back_to_guarded_panic_detail() -> None:
    mod = _load_module()
    payload = {
        "status": "panic",
        "executed": False,
        "guarded_exec": {
            "status": "panic",
            "panic": "panic_close_all:guarded_exec_takeover_unclassified_failure",
        },
        "probe_transport": {"capture_mode": "remote_async_capture"},
        "__probe_mode": "remote_async",
        "__probe_returncode": 0,
    }

    out = mod.build_window_summary(window_hours=720, probe_payload=payload, probe_source="fixture")
    assert out["history_window_label"] == "30d"
    assert out["probe_mode"] == "remote_async"
    assert out["probe_panic"] == "panic_close_all:guarded_exec_takeover_unclassified_failure"
    assert out["probe_error_detail"] == "panic_close_all:guarded_exec_takeover_unclassified_failure"


def test_main_builds_audit_from_probe_files(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    probe_24 = review_dir / "probe_24.json"
    probe_168 = review_dir / "probe_168.json"

    probe_24.write_text(
        json.dumps(
            {
                "status": "probe_completed",
                "executed": False,
                "guarded_exec": {
                    "artifact": str(review_dir / "guard_24.json"),
                    "takeover": {
                        "payload": {
                            "market": "portfolio_margin_um",
                            "artifact": str(review_dir / "takeover_24.json"),
                            "steps": {
                                "account_overview": {"quote_available": 12.0, "position_count": 1},
                                "live_snapshot": {"open_positions": 1, "closed_count": 2, "closed_pnl": 3.5},
                                "trade_telemetry": {
                                    "trades": 4,
                                    "income_rows": 3,
                                    "query_slice_hours": 24,
                                    "trade_slice_count": 1,
                                    "income_slice_count": 1,
                                    "trade_count_by_symbol": {"BTCUSDT": 4},
                                    "income_count_by_symbol": {"BTCUSDT": 3},
                                    "income_pnl_by_symbol": {"BTCUSDT": 3.5},
                                    "income_pnl_by_day": {"2026-03-12": 3.5},
                                },
                                "risk_guard": {"status": "ok", "allowed": True, "reasons": []},
                                "signal_selection": {"blocked_candidate": {"symbol": "BTCUSDT"}},
                            },
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    probe_168.write_text(
        json.dumps(
            {
                "status": "probe_failed",
                "executed": False,
                "guarded_exec": {
                    "artifact": str(review_dir / "guard_168.json"),
                    "takeover": {
                        "payload": {
                            "market": "portfolio_margin_um",
                            "artifact": str(review_dir / "takeover_168.json"),
                            "steps": {
                                "account_overview": {"quote_available": 10.0, "position_count": 0},
                                "live_snapshot": {"open_positions": 0, "closed_count": 0, "closed_pnl": 0.0},
                                "trade_telemetry": {
                                    "trades": 0,
                                    "income_rows": 0,
                                    "query_slice_hours": 168,
                                    "trade_slice_count": 1,
                                    "income_slice_count": 1,
                                    "trade_count_by_symbol": {},
                                    "income_count_by_symbol": {},
                                    "income_pnl_by_symbol": {},
                                    "income_pnl_by_day": {},
                                },
                                "risk_guard": {"status": "blocked", "allowed": False, "reasons": ["ticket_missing"]},
                                "signal_selection": {"blocked_candidate": {"symbol": "SOLUSDT"}},
                            },
                        }
                    },
                },
                "panic": "panic_close_all:guarded_exec_takeover_unclassified_failure",
                "__probe_returncode": 2,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_remote_live_history_audit.py",
            "--system-root",
            str(tmp_path),
            "--review-dir",
            str(review_dir),
            "--market",
            "portfolio_margin_um",
            "--windows-hours",
            "24,168",
            "--probe-json",
            f"24={probe_24}",
            "--probe-json",
            f"168={probe_168}",
            "--now",
            "2026-03-12T12:00:00Z",
        ],
    )
    rc = mod.main()

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "partial_failure"
    assert payload["ok"] is False
    assert payload["windows_hours"] == [24, 168]
    summaries = payload["window_summaries"]
    assert [row["window_hours"] for row in summaries] == [24, 168]
    assert summaries[0]["history_window_label"] == "24h"
    assert summaries[1]["history_window_label"] == "7d"
    assert summaries[0]["probe_mode"] == "direct"
    assert summaries[0]["income_pnl_by_symbol"] == {"BTCUSDT": 3.5}
    assert summaries[1]["probe_status"] == "probe_failed"
    assert summaries[1]["probe_returncode"] == 2
    assert summaries[1]["probe_panic"] == "panic_close_all:guarded_exec_takeover_unclassified_failure"
    assert summaries[1]["probe_error_detail"] == "panic_close_all:guarded_exec_takeover_unclassified_failure"

    artifact_path = Path(payload["artifact"])
    markdown_path = Path(payload["markdown_artifact"])
    latest_path = review_dir / "latest_remote_live_history_audit.json"
    checksum_path = review_dir / "20260312T120000Z_remote_live_history_audit_checksum.json"
    assert artifact_path.exists()
    assert markdown_path.exists()
    assert latest_path.exists()
    assert checksum_path.exists()

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Remote Live History Audit" in markdown
    assert "## 24h" in markdown
    assert "## 7d" in markdown
    assert '`{"BTCUSDT": 3.5}`' in markdown
    assert "- probe_panic: `panic_close_all:guarded_exec_takeover_unclassified_failure`" in markdown


def test_resolve_probe_mode_uses_remote_capture_for_30d() -> None:
    mod = _load_module()
    assert mod.resolve_probe_mode(requested="auto", window_hours=24) == "direct"
    assert mod.resolve_probe_mode(requested="auto", window_hours=168) == "direct"
    assert mod.resolve_probe_mode(requested="auto", window_hours=720) == "remote_async"
    assert mod.resolve_probe_mode(requested="remote_capture", window_hours=24) == "remote_capture"
    assert mod.resolve_probe_mode(requested="remote_async", window_hours=24) == "remote_async"
    assert mod.recommended_probe_rate_limit_per_minute(window_hours=24) == 10
    assert mod.recommended_probe_rate_limit_per_minute(window_hours=168) == 15
    assert mod.recommended_probe_rate_limit_per_minute(window_hours=720) == 30


def test_run_probe_sets_capture_mode_env(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    scripts_dir = system_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "openclaw_cloud_bridge.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = json.dumps({"status": "probe_completed", "probe_transport": {"capture_mode": "remote_stdout_capture"}})
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = dict(kwargs.get("env", {}))
        return _Proc()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    out = mod.run_probe(
        system_root=system_root,
        market="portfolio_margin_um",
        window_hours=720,
        timeout_seconds=45.0,
        probe_mode="remote_async",
    )

    assert out["returncode"] == 0
    payload = out["payload"]
    assert payload["status"] == "probe_completed"
    assert payload["__probe_mode"] == "remote_async"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["LIVE_TAKEOVER_PROBE_CAPTURE_MODE"] == "remote_async"
    assert env["LIVE_TAKEOVER_RATE_LIMIT_PER_MINUTE"] == "30"
