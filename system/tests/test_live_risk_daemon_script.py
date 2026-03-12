from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "live_risk_daemon.py"
    spec = importlib.util.spec_from_file_location("live_risk_daemon", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_risk_daemon_runs_single_cycle_and_writes_state(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    output_root = system_root / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    config_path = system_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("thresholds: {}\n", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run_json_command(*, cmd, cwd, timeout_seconds):
        _ = (cwd, timeout_seconds)
        calls.append(list(cmd))
        return {
            "returncode": 3,
            "payload": {
                "status": "blocked",
                "allowed": False,
                "reasons": ["ticket_missing:no_actionable_ticket"],
                "artifact": str(review_dir / "risk.json"),
                "fuse_path": str(output_root / "state" / "live_risk_fuse.json"),
                "ticket_refresh": {"returncode": 0, "payload": {"json": str(review_dir / "tickets.json")}},
            },
            "stdout": "",
            "stderr": "",
            "timeout": False,
        }

    monkeypatch.setattr(mod, "run_json_command", _fake_run_json_command)
    monkeypatch.setattr(mod, "resolve_path", lambda raw, anchor: Path(raw) if Path(str(raw)).is_absolute() else (anchor / str(raw)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_daemon.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--max-cycles",
            "1",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert len(calls) == 1
    assert Path(calls[0][1]).name == "live_risk_guard.py"

    state_path = output_root / "state" / "live_risk_daemon.json"
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert bool(payload.get("running", True)) is False
    assert str(payload.get("status", "")) == "stopped"
    assert int(payload.get("cycles_completed", 0)) == 1
    last_guard = payload.get("last_guard", {})
    assert isinstance(last_guard, dict)
    assert int(last_guard.get("returncode", 0)) == 3
    checksum = output_root / "state" / "live_risk_daemon_checksum.json"
    assert checksum.exists()


def test_live_risk_daemon_bounds_recent_cycle_history(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    system_root = tmp_path / "system"
    output_root = system_root / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    config_path = system_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("thresholds: {}\n", encoding="utf-8")

    seq = iter(
        [
            {"returncode": 0, "payload": {"status": "pass", "allowed": True, "reasons": []}, "stdout": "", "stderr": "", "timeout": False},
            {"returncode": 3, "payload": {"status": "blocked", "allowed": False, "reasons": ["ticket_missing:no_actionable_ticket"]}, "stdout": "", "stderr": "", "timeout": False},
            {"returncode": 0, "payload": {"status": "pass", "allowed": True, "reasons": []}, "stdout": "", "stderr": "", "timeout": False},
        ]
    )

    monkeypatch.setattr(mod, "run_json_command", lambda **kwargs: next(seq))
    monkeypatch.setattr(mod, "resolve_path", lambda raw, anchor: Path(raw) if Path(str(raw)).is_absolute() else (anchor / str(raw)))
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_risk_daemon.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--max-cycles",
            "3",
            "--history-limit",
            "2",
            "--poll-seconds",
            "1",
        ],
    )
    rc = mod.main()
    assert rc == 0

    state_path = output_root / "state" / "live_risk_daemon.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert int(payload.get("cycles_completed", 0)) == 3
    recent = payload.get("recent_cycles", [])
    assert isinstance(recent, list)
    assert len(recent) == 2
    assert [str(row.get("status", "")) for row in recent] == ["blocked", "pass"]
