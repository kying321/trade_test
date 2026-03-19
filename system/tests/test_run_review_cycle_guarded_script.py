from __future__ import annotations

import importlib.util
from pathlib import Path

from unittest.mock import patch


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "run_review_cycle_guarded.py"
    spec = importlib.util.spec_from_file_location("run_review_cycle_guarded", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_propagates_parent_mutex_marker_to_child_env(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    captured_envs: list[dict[str, str]] = []

    def _fake_run_step(*, name, cmd, timeout_seconds, logs_dir, env):  # type: ignore[no-untyped-def]
        captured_envs.append(dict(env))
        return {
            "name": name,
            "timed_out": False,
            "returncode": 0,
            "elapsed_sec": 0.01,
            "stdout_len": 0,
            "stderr_len": 0,
            "stdout_log": str(logs_dir / f"{name}.out.log"),
            "stderr_log": str(logs_dir / f"{name}.err.log"),
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_review_cycle_guarded.py",
            "--date",
            "2026-02-13",
            "--config",
            "config.yaml",
            "--output-root",
            str(output_root),
            "--skip-health-check",
        ],
    )
    with patch.object(mod, "_run_step", side_effect=_fake_run_step):
        rc = mod.main()

    assert rc == 0
    assert captured_envs
    assert all(env.get(mod.RUN_HALFHOUR_MUTEX_HELD_ENV) == "run_review_cycle_guarded" for env in captured_envs)


def test_main_passes_fast_tests_only_to_review_loop(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    captured_cmds: list[list[str]] = []

    def _fake_run_step(*, name, cmd, timeout_seconds, logs_dir, env):  # type: ignore[no-untyped-def]
        captured_cmds.append(list(cmd))
        return {
            "name": name,
            "timed_out": False,
            "returncode": 0,
            "elapsed_sec": 0.01,
            "stdout_len": 0,
            "stderr_len": 0,
            "stdout_log": str(logs_dir / f"{name}.out.log"),
            "stderr_log": str(logs_dir / f"{name}.err.log"),
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_review_cycle_guarded.py",
            "--date",
            "2026-02-13",
            "--config",
            "config.yaml",
            "--output-root",
            str(output_root),
            "--fast-tests-only",
            "--skip-health-check",
        ],
    )
    with patch.object(mod, "_run_step", side_effect=_fake_run_step):
        rc = mod.main()

    assert rc == 0
    assert captured_cmds
    review_cmd = captured_cmds[0]
    assert "--fast-tests-only" in review_cmd


def test_main_defaults_gate_report_to_cached_stable_replay(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    captured_cmds: list[list[str]] = []

    def _fake_run_step(*, name, cmd, timeout_seconds, logs_dir, env):  # type: ignore[no-untyped-def]
        captured_cmds.append(list(cmd))
        return {
            "name": name,
            "timed_out": False,
            "returncode": 0,
            "elapsed_sec": 0.01,
            "stdout_len": 0,
            "stderr_len": 0,
            "stdout_log": str(logs_dir / f"{name}.out.log"),
            "stderr_log": str(logs_dir / f"{name}.err.log"),
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_review_cycle_guarded.py",
            "--date",
            "2026-02-13",
            "--config",
            "config.yaml",
            "--output-root",
            str(output_root),
            "--skip-health-check",
        ],
    )
    with patch.object(mod, "_run_step", side_effect=_fake_run_step):
        rc = mod.main()

    assert rc == 0
    gate_cmd = next(cmd for cmd in captured_cmds if "gate-report" in cmd)
    idx = gate_cmd.index("--stable-replay-mode")
    assert gate_cmd[idx + 1] == "cached"
