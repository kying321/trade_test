from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.engine import LieEngine


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_orchestration_layer_does_not_import_engine(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "lie_engine" / "orchestration"
        py_files = [p for p in root.glob("*.py") if p.name != "__init__.py"]
        offenders: list[str] = []
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "lie_engine.engine":
                            offenders.append(str(path))
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "lie_engine.engine":
                        offenders.append(str(path))
        self.assertEqual(offenders, [], f"orchestration层不应依赖engine，违规: {offenders}")

    def test_engine_release_methods_delegate_to_orchestrator(self) -> None:
        targets = {
            "gate_report": "self._release_orchestrator().gate_report",
            "ops_report": "self._release_orchestrator().ops_report",
            "degradation_guardrail_burnin": "self._release_orchestrator().degradation_guardrail_burnin",
            "degradation_guardrail_threshold_drift_audit": "self._release_orchestrator().degradation_guardrail_threshold_drift_audit",
            "review_until_pass": "self._release_orchestrator().review_until_pass",
            "run_review_cycle": "self._release_orchestrator().run_review_cycle",
            "architecture_audit": "self._architecture_orchestrator().architecture_audit",
            "dependency_audit": "self._dependency_orchestrator().dependency_audit",
            "health_check": "self._observability_orchestrator().health_check",
            "stable_replay_check": "self._observability_orchestrator().stable_replay_check",
            "test_all": "self._testing_orchestrator().test_all",
            "run_slot": "self._scheduler_orchestrator().run_slot",
            "run_session": "self._scheduler_orchestrator().run_session",
            "run_autorun_retro": "self._scheduler_orchestrator().run_autorun_retro",
            "run_daemon": "self._scheduler_orchestrator().run_daemon",
            "run_halfhour_daemon": "self._scheduler_orchestrator().run_halfhour_daemon",
        }
        for method_name, needle in targets.items():
            src = inspect.getsource(getattr(LieEngine, method_name))
            self.assertIn(needle, src, f"{method_name} 应委托给对应 orchestrator")

    def test_dashboard_api_layer_does_not_import_engine_or_orchestration_modules(self) -> None:
        root = Path(__file__).resolve().parents[1] / "dashboard" / "api"
        py_files = [p for p in root.glob("*.py") if p.name != "__init__.py"]
        banned_prefixes = (
            "lie_engine.engine",
            "lie_engine.orchestration",
            "lie_engine.backtest",
            "lie_engine.signal",
            "lie_engine.risk",
            "lie_engine.regime",
            "lie_engine.research",
            "lie_engine.review",
        )
        offenders: list[str] = []
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = str(alias.name or "")
                        if imported.startswith(banned_prefixes):
                            offenders.append(f"{path}:{imported}")
                elif isinstance(node, ast.ImportFrom):
                    imported = str(node.module or "")
                    if imported.startswith(banned_prefixes):
                        offenders.append(f"{path}:{imported}")
        self.assertEqual(offenders, [], f"dashboard/api 应保持适配层，不得越层依赖内部模块: {offenders}")


if __name__ == "__main__":
    unittest.main()
