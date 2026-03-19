from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "guarded_exec.py"
    spec = importlib.util.spec_from_file_location("guarded_exec", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _DummyMutex:
    def __init__(self, **kwargs) -> None:
        _ = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = (exc_type, exc, tb)
        return False


def test_resolve_workspace_default_resolves_repo_system_root() -> None:
    mod = _load_module()
    workspace, system_root = mod.resolve_workspace("")
    assert system_root.name == "system"
    assert workspace == system_root.parent


def test_guarded_exec_downgrades_canary_to_probe_without_allow_live_order(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        script = Path(cmd[1]).name
        if script == "live_risk_guard.py":
            return {"returncode": 0, "payload": {"allowed": True, "status": "pass"}, "stdout": "", "stderr": "", "timeout": False}
        if script == "binance_live_takeover.py":
            return {
                "returncode": 0,
                "payload": {"mode": "live_ready", "steps": {"canary_order": {"executed": False, "reason": "allow_live_order_false"}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_exec.py",
            "--workspace",
            str(tmp_path),
            "--mode",
            "canary",
            "--market",
            "spot",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("requested_mode", "")) == "canary"
    assert str(parsed.get("effective_mode", "")) == "probe"
    assert str(parsed.get("status", "")) == "downgraded_probe_allow_live_order_false"


def test_guarded_exec_downgrades_canary_to_probe_when_risk_guard_blocks(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        script = Path(cmd[1]).name
        if script == "live_risk_guard.py":
            return {
                "returncode": 3,
                "payload": {"allowed": False, "status": "blocked", "reasons": ["ticket_missing:no_actionable_ticket"]},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        if script == "binance_live_takeover.py":
            return {
                "returncode": 0,
                "payload": {"mode": "live_ready", "steps": {"canary_order": {"executed": False, "reason": "allow_live_order_false"}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_exec.py",
            "--workspace",
            str(tmp_path),
            "--mode",
            "canary",
            "--market",
            "spot",
            "--allow-live-order",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("requested_mode", "")) == "canary"
    assert str(parsed.get("effective_mode", "")) == "probe"
    assert str(parsed.get("status", "")) == "downgraded_probe_risk_guard_blocked"
    assert int((parsed.get("risk_guard", {}) or {}).get("returncode", 0)) == 3


def test_guarded_exec_uses_fresh_fuse_cache_before_sync_risk_guard(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state" / "live_risk_fuse.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2099-01-01T00:00:00Z",
                "allowed": False,
                "status": "blocked",
                "reasons": ["ticket_missing:no_actionable_ticket"],
                "artifact": str(system_root / "output" / "review" / "risk.json"),
                "checksum": str(system_root / "output" / "review" / "risk_checksum.json"),
                "backup_intel": {
                    "enabled": True,
                    "status": "active",
                    "active": True,
                    "blocking_reasons": ["backup_intel_no_trade_symbol"],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "live_risk_guard.py":
            raise AssertionError("live_risk_guard.py should not run when fuse cache is fresh")
        if script == "binance_live_takeover.py":
            return {
                "returncode": 0,
                "payload": {"mode": "live_read_only", "steps": {"canary_order": {"executed": False}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_exec.py",
            "--workspace",
            str(tmp_path),
            "--mode",
            "probe",
            "--market",
            "spot",
            "--refresh-tickets",
            "--ticket-symbols",
            "BTCUSDT,ETHUSDT",
            "--ticket-min-confidence",
            "61",
            "--ticket-min-convexity",
            "3.5",
            "--ticket-max-age-days",
            "9",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert [Path(cmd[1]).name for cmd in calls] == ["binance_live_takeover.py"]

    generated = sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    ticket_refresh = parsed.get("ticket_refresh", {})
    assert isinstance(ticket_refresh, dict)
    assert int(ticket_refresh.get("returncode", 0)) == 0
    assert bool(ticket_refresh.get("skipped", False)) is True
    risk_guard = parsed.get("risk_guard", {})
    assert isinstance(risk_guard, dict)
    assert str(risk_guard.get("source", "")) == "fuse_cache"
    assert bool(risk_guard.get("cached", False)) is True
    payload = risk_guard.get("payload", {})
    assert isinstance(payload, dict)
    assert (payload.get("backup_intel", {}) or {}).get("status") == "active"


def test_guarded_exec_falls_back_to_sync_risk_guard_when_fuse_is_stale(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state" / "live_risk_fuse.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2000-01-01T00:00:00Z",
                "allowed": True,
                "status": "pass",
                "reasons": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "live_risk_guard.py":
            return {
                "returncode": 0,
                "payload": {"allowed": True, "status": "pass", "ticket_refresh": {"returncode": 0, "payload": {"json": "tickets.json"}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        if script == "binance_live_takeover.py":
            return {
                "returncode": 0,
                "payload": {"mode": "live_read_only", "steps": {"canary_order": {"executed": False}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_exec.py",
            "--workspace",
            str(tmp_path),
            "--mode",
            "probe",
            "--market",
            "spot",
            "--refresh-tickets",
            "--ticket-symbols",
            "BTCUSDT,ETHUSDT",
            "--ticket-min-confidence",
            "61",
            "--ticket-min-convexity",
            "3.5",
            "--ticket-max-age-days",
            "9",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert [Path(cmd[1]).name for cmd in calls] == [
        "live_risk_guard.py",
        "binance_live_takeover.py",
    ]
    guard_cmd = calls[0]
    assert "--refresh-tickets" in guard_cmd
    assert guard_cmd[guard_cmd.index("--ticket-symbols") + 1] == "BTCUSDT,ETHUSDT"
    assert guard_cmd[guard_cmd.index("--ticket-min-confidence") + 1] == "61.0"
    assert guard_cmd[guard_cmd.index("--ticket-min-convexity") + 1] == "3.5"
    assert guard_cmd[guard_cmd.index("--ticket-max-age-days") + 1] == "9"

    generated = sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    risk_guard = parsed.get("risk_guard", {})
    assert isinstance(risk_guard, dict)
    assert str(risk_guard.get("source", "")) == "live_risk_guard"
    assert bool(risk_guard.get("cached", True)) is False


def test_guarded_exec_probe_does_not_idempotent_skip_duplicate_runs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    calls: list[str] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        script = Path(cmd[1]).name
        calls.append(script)
        if script == "live_risk_guard.py":
            return {"returncode": 0, "payload": {"allowed": True, "status": "pass"}, "stdout": "", "stderr": "", "timeout": False}
        if script == "binance_live_takeover.py":
            return {
                "returncode": 0,
                "payload": {"mode": "live_read_only", "steps": {"canary_order": {"executed": False}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)

    argv = [
        "guarded_exec.py",
        "--workspace",
        str(tmp_path),
        "--mode",
        "probe",
        "--market",
        "spot",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0

    assert calls == [
        "live_risk_guard.py",
        "binance_live_takeover.py",
        "live_risk_guard.py",
        "binance_live_takeover.py",
    ]
    generated = sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("status", "")) == "probe_completed"
    idem = parsed.get("idempotency", {})
    assert isinstance(idem, dict)
    assert bool(idem.get("duplicate", False)) is False
    assert bool(idem.get("applied", True)) is False


def test_guarded_exec_allows_portfolio_margin_um_probe_market(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "live_risk_guard.py":
            return {"returncode": 0, "payload": {"allowed": True, "status": "pass"}, "stdout": "", "stderr": "", "timeout": False}
        if script == "binance_live_takeover.py":
            return {
                "returncode": 0,
                "payload": {"mode": "live_ready", "market": "portfolio_margin_um", "steps": {"canary_order": {"executed": False, "reason": "allow_live_order_false"}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_exec.py",
            "--workspace",
            str(tmp_path),
            "--mode",
            "probe",
            "--market",
            "portfolio_margin_um",
        ],
    )

    rc = mod.main()
    assert rc == 0
    takeover_cmd = calls[1]
    assert takeover_cmd[takeover_cmd.index("--market") + 1] == "portfolio_margin_um"
    parsed = json.loads(sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))[-1].read_text(encoding="utf-8"))
    assert str(((parsed.get("takeover") or {}).get("payload") or {}).get("market", "")) == "portfolio_margin_um"


def test_guarded_exec_treats_takeover_preflight_transport_block_as_controlled_failure(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        script = Path(cmd[1]).name
        if script == "live_risk_guard.py":
            return {"returncode": 0, "payload": {"allowed": True, "status": "pass"}, "stdout": "", "stderr": "", "timeout": False}
        if script == "binance_live_takeover.py":
            return {
                "returncode": 2,
                "payload": {
                    "mode": "probe_preflight_blocked",
                    "steps": {
                        "canary_order": {
                            "executed": False,
                            "reason": "preflight_transport_blocked",
                            "failed_step": "canary_plan",
                        }
                    },
                },
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_exec.py",
            "--workspace",
            str(tmp_path),
            "--mode",
            "probe",
            "--market",
            "spot",
        ],
    )

    rc = mod.main()
    assert rc == 0
    parsed = json.loads(sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("status", "")) == "probe_controlled_block"
    takeover = parsed.get("takeover", {})
    assert isinstance(takeover, dict)
    assert int(takeover.get("returncode", 0)) == 2
    assert str((((takeover.get("payload") or {}).get("steps") or {}).get("canary_order") or {}).get("reason", "")) == "preflight_transport_blocked"


def test_guarded_exec_treats_takeover_preflight_runtime_block_as_controlled_failure(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        script = Path(cmd[1]).name
        if script == "live_risk_guard.py":
            return {"returncode": 0, "payload": {"allowed": True, "status": "pass"}, "stdout": "", "stderr": "", "timeout": False}
        if script == "binance_live_takeover.py":
            return {
                "returncode": 2,
                "payload": {
                    "mode": "live_ready_preflight_blocked",
                    "steps": {
                        "canary_order": {
                            "executed": False,
                            "reason": "preflight_runtime_blocked",
                            "failed_step": "account_overview",
                        }
                    },
                },
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guarded_exec.py",
            "--workspace",
            str(tmp_path),
            "--mode",
            "probe",
            "--market",
            "spot",
        ],
    )

    rc = mod.main()
    assert rc == 0
    parsed = json.loads(sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("status", "")) == "probe_controlled_block"
    takeover = parsed.get("takeover", {})
    assert isinstance(takeover, dict)
    assert int(takeover.get("returncode", 0)) == 2
    assert str((((takeover.get("payload") or {}).get("steps") or {}).get("canary_order") or {}).get("reason", "")) == "preflight_runtime_blocked"


def test_guarded_exec_canary_still_idempotent_skips_duplicate_runs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)
    (system_root / "output" / "state").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))
    monkeypatch.setattr(mod, "RunHalfhourMutex", _DummyMutex)

    calls: list[str] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        script = Path(cmd[1]).name
        calls.append(script)
        if script == "live_risk_guard.py":
            return {"returncode": 0, "payload": {"allowed": True, "status": "pass"}, "stdout": "", "stderr": "", "timeout": False}
        if script == "binance_live_takeover.py":
            return {
                "returncode": 0,
                "payload": {"mode": "live_ready", "steps": {"canary_order": {"executed": False, "reason": "allow_live_order_false"}}},
                "stdout": "",
                "stderr": "",
                "timeout": False,
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)

    argv = [
        "guarded_exec.py",
        "--workspace",
        str(tmp_path),
        "--mode",
        "canary",
        "--market",
        "spot",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0

    assert calls == [
        "live_risk_guard.py",
        "binance_live_takeover.py",
    ]
    generated = sorted((system_root / "output" / "review").glob("*_trade_live_exec_guard.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("status", "")) == "idempotent_skip"
    idem = parsed.get("idempotency", {})
    assert isinstance(idem, dict)
    assert bool(idem.get("duplicate", False)) is True
