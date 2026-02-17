from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.config import SystemSettings
from lie_engine.orchestration.dependency import DependencyOrchestrator


class DependencyOrchestratorTests(unittest.TestCase):
    def test_dependency_audit_outputs_files(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        source_root = Path(__file__).resolve().parents[1] / "src"
        settings = SystemSettings(raw={"timezone": "Asia/Shanghai"})
        orch = DependencyOrchestrator(settings=settings, source_root=source_root, output_dir=td)
        out = orch.dependency_audit()
        self.assertIn("ok", out)
        self.assertIn("edges", out)
        self.assertTrue(Path(out["paths"]["json"]).exists())
        self.assertTrue(Path(out["paths"]["md"]).exists())


if __name__ == "__main__":
    unittest.main()
