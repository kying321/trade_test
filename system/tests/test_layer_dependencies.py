from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class LayerDependencyTests(unittest.TestCase):
    def test_internal_dependency_whitelist(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "lie_engine"
        allow: dict[str, set[str]] = {
            "config": {"config"},
            "data": {"data", "models"},
            "regime": {"regime", "models"},
            "signal": {"signal", "models", "regime"},
            "risk": {"risk", "models"},
            "backtest": {"backtest", "models", "regime", "signal"},
            "review": {"review", "models"},
            "reporting": {"reporting", "models", "data"},
            "research": {"research", "models", "data", "backtest"},
            "orchestration": {"orchestration", "models", "config", "data"},
            "engine": {
                "engine",
                "config",
                "data",
                "models",
                "orchestration",
                "regime",
                "reporting",
                "research",
                "review",
                "risk",
                "signal",
                "backtest",
            },
            "cli": {"cli", "engine", "config", "models"},
            "models": {"models"},
        }

        violations: list[str] = []
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root)
            if rel.as_posix() == "__init__.py":
                continue

            if len(rel.parts) == 1:
                src_layer = rel.stem
            else:
                src_layer = rel.parts[0]
            if src_layer not in allow:
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module.startswith("lie_engine."):
                            dep_layer = module.split(".")[1]
                            if dep_layer not in allow[src_layer]:
                                violations.append(f"{rel}: {src_layer} -> {dep_layer}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith("lie_engine."):
                        dep_layer = module.split(".")[1]
                        if dep_layer not in allow[src_layer]:
                            violations.append(f"{rel}: {src_layer} -> {dep_layer}")

        self.assertEqual([], violations, "存在跨层违规依赖:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
