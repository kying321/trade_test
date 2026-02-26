from __future__ import annotations

import importlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch


MODULE_NAME = "dashboard.api.main"
SYSTEM_PATH_ENV = "LIE_DASHBOARD_SYSTEM_PATH"


def _reload_dashboard_api_main():
    module = importlib.import_module(MODULE_NAME)
    return importlib.reload(module)


class DashboardApiPathResolutionTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop(SYSTEM_PATH_ENV, None)
        _reload_dashboard_api_main()

    def test_default_system_path_resolves_from_module_location(self) -> None:
        with patch.dict(os.environ, {SYSTEM_PATH_ENV: ""}, clear=False):
            module = _reload_dashboard_api_main()

        expected_root = Path(module.__file__).resolve().parents[2]
        self.assertEqual(Path(module.LIE_SYSTEM_PATH).resolve(), expected_root)
        self.assertEqual(
            Path(module.PARAMS_FILE).resolve(),
            expected_root / "config" / "params_live.yaml",
        )
        self.assertEqual(
            Path(module.REVIEW_DIR).resolve(),
            expected_root / "output" / "review",
        )

    def test_env_override_rebinds_paths_and_runbook_templates(self) -> None:
        fake_root = Path("/tmp/lie-dashboard-root")
        with patch.dict(os.environ, {SYSTEM_PATH_ENV: str(fake_root)}, clear=False):
            module = _reload_dashboard_api_main()

        resolved_root = fake_root.expanduser().resolve()
        self.assertEqual(Path(module.LIE_SYSTEM_PATH).resolve(), resolved_root)
        self.assertEqual(Path(module.LOGS_DIR), resolved_root / "output" / "logs")
        self.assertEqual(Path(module.HMM_JSON), resolved_root / "output" / "live_multimodal.json")
        template = module.runbook_template("triage_proposals_and_raise_review_capacity")
        self.assertIn(f"cd '{resolved_root}' && ", str(template.get("command", "")))


if __name__ == "__main__":
    unittest.main()
