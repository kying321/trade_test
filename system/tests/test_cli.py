from __future__ import annotations

import builtins
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main()
