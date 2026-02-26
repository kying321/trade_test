from __future__ import annotations

from datetime import date
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
        self.assertIn("dashboard_adapter", out)
        self.assertIn("total_files_checked", out)
        self.assertTrue(Path(out["paths"]["json"]).exists())
        self.assertTrue(Path(out["paths"]["md"]).exists())

    def test_dependency_audit_flags_dashboard_adapter_violations(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        src_root = td / "src"
        lie_root = src_root / "lie_engine"
        dashboard_root = td / "dashboard" / "api"
        lie_root.mkdir(parents=True, exist_ok=True)
        dashboard_root.mkdir(parents=True, exist_ok=True)
        (lie_root / "__init__.py").write_text("", encoding="utf-8")
        (lie_root / "models.py").write_text("x = 1\n", encoding="utf-8")
        (dashboard_root / "main.py").write_text(
            "from lie_engine.engine import LieEngine\n",
            encoding="utf-8",
        )
        settings = SystemSettings(raw={"timezone": "Asia/Shanghai"})
        orch = DependencyOrchestrator(settings=settings, source_root=src_root, output_dir=td)
        out = orch.dependency_audit(as_of=date(2026, 2, 22))
        self.assertFalse(bool(out.get("ok", True)))
        dashboard_scan = (
            out.get("dashboard_adapter", {})
            if isinstance(out.get("dashboard_adapter", {}), dict)
            else {}
        )
        self.assertTrue(bool(dashboard_scan.get("enabled", False)))
        self.assertEqual(int(dashboard_scan.get("files_checked", 0)), 1)
        violations = (
            dashboard_scan.get("violations", [])
            if isinstance(dashboard_scan.get("violations", []), list)
            else []
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("lie_engine.engine", str(violations[0]))

    def test_dependency_audit_allows_missing_dashboard_adapter_directory(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        src_root = td / "src"
        lie_root = src_root / "lie_engine"
        lie_root.mkdir(parents=True, exist_ok=True)
        (lie_root / "__init__.py").write_text("", encoding="utf-8")
        (lie_root / "models.py").write_text("x = 1\n", encoding="utf-8")
        settings = SystemSettings(raw={"timezone": "Asia/Shanghai"})
        orch = DependencyOrchestrator(settings=settings, source_root=src_root, output_dir=td)
        out = orch.dependency_audit(as_of=date(2026, 2, 22))
        dashboard_scan = (
            out.get("dashboard_adapter", {})
            if isinstance(out.get("dashboard_adapter", {}), dict)
            else {}
        )
        self.assertFalse(bool(dashboard_scan.get("enabled", True)))
        self.assertEqual(int(dashboard_scan.get("files_checked", -1)), 0)
        self.assertEqual(dashboard_scan.get("violations", []), [])
        self.assertTrue(bool(out.get("ok", False)))


if __name__ == "__main__":
    unittest.main()
