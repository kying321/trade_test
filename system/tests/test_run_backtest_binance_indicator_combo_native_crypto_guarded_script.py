from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_backtest_binance_indicator_combo_native_crypto_guarded.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("guarded_native_crypto_backtest_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_main_writes_guarded_partial_failure_artifact_on_timeout(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd=["python3", "x.py"], timeout=5.0, output="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    rc = module.main(
        [
            "--review-dir",
            str(tmp_path),
            "--symbol-group",
            "custom",
            "--job-timeout-seconds",
            "5",
            "--now",
            "2026-03-12T06:00:00+00:00",
        ]
    )
    assert rc == 0
    artifacts = sorted(tmp_path.glob("*_custom_binance_indicator_combo_native_crypto.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["status"] == "partial_failure"
    assert payload["artifact_label"].endswith(":partial_failure_guarded")
    assert payload["error_items"][0]["stage"] == "subprocess_guard"
    assert payload["error_items"][0]["status"] == "timed_out"


def test_main_passes_through_inner_json_when_success(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return subprocess.CompletedProcess(
            args=["python3", "x.py"],
            returncode=0,
            stdout=json.dumps({"ok": True, "status": "ok", "artifact": str(tmp_path / "ok.json")}),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    rc = module.main(
        [
            "--review-dir",
            str(tmp_path),
            "--symbol-group",
            "custom",
            "--now",
            "2026-03-12T06:00:00+00:00",
        ]
    )
    assert rc == 0
