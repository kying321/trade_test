from __future__ import annotations

import builtins
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class CliTests(unittest.TestCase):
    def test_validate_config_does_not_import_engine(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.daemon.test.yaml"
        argv = ["lie", "--config", str(config_path), "validate-config"]

        original_import = builtins.__import__

        def guarded_import(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "lie_engine.engine":
                raise AssertionError("validate-config must not import lie_engine.engine")
            return original_import(name, globals_, locals_, fromlist, level)

        sys.modules.pop("lie_engine.cli", None)
        with patch("builtins.__import__", side_effect=guarded_import):
            cli_mod = importlib.import_module("lie_engine.cli")
            out_buf = io.StringIO()
            with patch.object(sys, "argv", argv):
                with redirect_stdout(out_buf):
                    cli_mod.main()

        payload = json.loads(out_buf.getvalue())
        self.assertTrue(bool(payload.get("ok", False)))
        self.assertEqual(int(payload.get("summary", {}).get("errors", 0)), 0)

    def test_test_all_timeout_seconds_passes_through_cli(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.daemon.test.yaml"
        argv = [
            "lie",
            "--config",
            str(config_path),
            "test-all",
            "--fast",
            "--fast-ratio",
            "0.2",
            "--timeout-seconds",
            "120",
        ]

        calls: list[dict[str, object]] = []

        class _FakeEngine:
            def __init__(self, config_path: str | None = None) -> None:
                self.config_path = config_path

            def test_all(self, **kwargs: object) -> dict[str, object]:
                calls.append(dict(kwargs))
                return {"ok": True, "kwargs": kwargs}

        fake_engine_mod = types.ModuleType("lie_engine.engine")
        fake_engine_mod.LieEngine = _FakeEngine  # type: ignore[attr-defined]

        sys.modules.pop("lie_engine.cli", None)
        cli_mod = importlib.import_module("lie_engine.cli")
        out_buf = io.StringIO()
        with patch.dict(sys.modules, {"lie_engine.engine": fake_engine_mod}):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(out_buf):
                    cli_mod.main()

        payload = json.loads(out_buf.getvalue())
        self.assertTrue(bool(payload.get("ok", False)))
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertTrue(bool(call.get("fast", False)))
        self.assertAlmostEqual(float(call.get("fast_ratio", 0.0)), 0.2, places=6)
        self.assertEqual(int(call.get("timeout_seconds", 0)), 120)

    def test_review_loop_fast_tests_only_passes_through_cli(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.daemon.test.yaml"
        argv = [
            "lie",
            "--config",
            str(config_path),
            "review-loop",
            "--date",
            "2026-02-13",
            "--max-rounds",
            "2",
            "--fast-tests-only",
        ]

        calls: list[dict[str, object]] = []

        class _FakeEngine:
            def __init__(self, config_path: str | None = None) -> None:
                self.config_path = config_path

            def review_until_pass(self, **kwargs: object) -> dict[str, object]:
                calls.append(dict(kwargs))
                return {"ok": True, "kwargs": kwargs}

        fake_engine_mod = types.ModuleType("lie_engine.engine")
        fake_engine_mod.LieEngine = _FakeEngine  # type: ignore[attr-defined]

        sys.modules.pop("lie_engine.cli", None)
        cli_mod = importlib.import_module("lie_engine.cli")
        out_buf = io.StringIO()
        with patch.dict(sys.modules, {"lie_engine.engine": fake_engine_mod}):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(out_buf):
                    cli_mod.main()

        payload = json.loads(out_buf.getvalue())
        self.assertTrue(bool(payload.get("ok", False)))
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(int(call.get("max_rounds", 0)), 2)
        self.assertTrue(bool(call.get("fast_tests_only", False)))

    def test_gate_report_stable_replay_mode_passes_through_cli(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.daemon.test.yaml"
        argv = [
            "lie",
            "--config",
            str(config_path),
            "gate-report",
            "--date",
            "2026-02-13",
            "--stable-replay-mode",
            "cached",
        ]

        calls: list[dict[str, object]] = []

        class _FakeEngine:
            def __init__(self, config_path: str | None = None) -> None:
                self.config_path = config_path

            def gate_report(self, **kwargs: object) -> dict[str, object]:
                calls.append(dict(kwargs))
                return {"ok": True, "kwargs": kwargs}

        fake_engine_mod = types.ModuleType("lie_engine.engine")
        fake_engine_mod.LieEngine = _FakeEngine  # type: ignore[attr-defined]

        sys.modules.pop("lie_engine.cli", None)
        cli_mod = importlib.import_module("lie_engine.cli")
        out_buf = io.StringIO()
        with patch.dict(sys.modules, {"lie_engine.engine": fake_engine_mod}):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(out_buf):
                    cli_mod.main()

        payload = json.loads(out_buf.getvalue())
        self.assertTrue(bool(payload.get("ok", False)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(str(calls[0].get("stable_replay_mode", "")), "cached")


if __name__ == "__main__":
    unittest.main()
