from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "run_ticketed_canary.py"
    spec = importlib.util.spec_from_file_location("run_ticketed_canary", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_ticketed_canary_skips_when_fast_plan_has_no_executable_candidate(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "build_order_ticket.py":
            return {"returncode": 0, "payload": {"json": str(system_root / "output" / "review" / "a.json")}, "stdout": "", "stderr": ""}
        if script == "build_fast_trade_skill_plan.py":
            return {
                "returncode": 0,
                "payload": {"selected": {"executable": False, "reason": "no_actionable_candidate"}},
                "stdout": "",
                "stderr": "",
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ticketed_canary.py",
            "--workspace",
            str(tmp_path),
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted((system_root / "output" / "review").glob("*_ticketed_canary_pipeline.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("status", "")) == "skipped_no_executable_candidate"
    assert len([cmd for cmd in calls if Path(cmd[1]).name == "guarded_exec.py"]) == 0


def test_run_ticketed_canary_executes_guarded_canary_and_auto_unwinds(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "build_order_ticket.py":
            return {"returncode": 0, "payload": {"json": str(system_root / "output" / "review" / "tickets.json")}, "stdout": "", "stderr": ""}
        if script == "build_fast_trade_skill_plan.py":
            return {
                "returncode": 0,
                "payload": {
                    "selected": {
                        "executable": True,
                        "symbol": "BTCUSDT",
                        "side": "BUY",
                        "quote_usdt": 5.5,
                    }
                },
                "stdout": "",
                "stderr": "",
            }
        if script == "guarded_exec.py":
            if "--risk-intent" in cmd and cmd[cmd.index("--risk-intent") + 1] == "reduce_only":
                return {
                    "returncode": 0,
                    "payload": {
                        "status": "canary_executed",
                        "takeover": {"payload": {"steps": {"canary_order": {"executed": True, "symbol": "BTCUSDT", "side": "SELL", "quantity": 0.00008}}}},
                    },
                    "stdout": "",
                    "stderr": "",
                }
            return {
                "returncode": 0,
                "payload": {
                    "status": "canary_executed",
                    "takeover": {"payload": {"steps": {"canary_order": {"executed": True, "symbol": "BTCUSDT", "side": "BUY", "quantity": 0.00008}}}},
                },
                "stdout": "",
                "stderr": "",
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ticketed_canary.py",
            "--workspace",
            str(tmp_path),
            "--run-live-canary",
            "--auto-unwind-live-canary",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted((system_root / "output" / "review").glob("*_ticketed_canary_pipeline.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert str(parsed.get("status", "")) == "canary_executed"
    auto_unwind = parsed.get("auto_unwind", {})
    assert isinstance(auto_unwind, dict)
    assert bool(auto_unwind.get("attempted", False)) is True

    guard_cmds = [cmd for cmd in calls if Path(cmd[1]).name == "guarded_exec.py"]
    assert len(guard_cmds) == 2
    unwind_cmd = guard_cmds[-1]
    assert "--risk-intent" in unwind_cmd
    assert unwind_cmd[unwind_cmd.index("--risk-intent") + 1] == "reduce_only"
    assert unwind_cmd[unwind_cmd.index("--order-side") + 1] == "SELL"


def test_run_ticketed_canary_uses_selected_quote_when_requested(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    (system_root / "output" / "review").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "resolve_workspace", lambda raw: (tmp_path, system_root))

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "build_order_ticket.py":
            return {"returncode": 0, "payload": {"json": str(system_root / "output" / "review" / "tickets.json")}, "stdout": "", "stderr": ""}
        if script == "build_fast_trade_skill_plan.py":
            return {
                "returncode": 0,
                "payload": {
                    "selected": {
                        "executable": True,
                        "symbol": "BTCUSDT",
                        "side": "BUY",
                        "quote_usdt": 5.5,
                    }
                },
                "stdout": "",
                "stderr": "",
            }
        if script == "guarded_exec.py":
            return {
                "returncode": 0,
                "payload": {"status": "probe_completed", "takeover": {"payload": {"steps": {"canary_order": {"executed": False}}}}},
                "stdout": "",
                "stderr": "",
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ticketed_canary.py",
            "--workspace",
            str(tmp_path),
            "--use-selected-quote",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted((system_root / "output" / "review").glob("*_ticketed_canary_pipeline.json"))
    assert generated
    parsed = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert float(parsed.get("effective_canary_quote_usdt", 0.0)) == 5.5
