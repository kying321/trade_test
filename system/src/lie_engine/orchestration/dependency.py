from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from lie_engine.config import SystemSettings
from lie_engine.data.storage import write_json, write_markdown


LAYER_WHITELIST: dict[str, set[str]] = {
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


@dataclass(slots=True)
class DependencyOrchestrator:
    settings: SystemSettings
    source_root: Path
    output_dir: Path

    def _scan(self) -> dict[str, Any]:
        root = self.source_root / "lie_engine"
        violations: list[str] = []
        edges: dict[str, set[str]] = {}
        files_checked = 0

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
            if src_layer not in LAYER_WHITELIST:
                continue
            files_checked += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("lie_engine."):
                            dep_layer = alias.name.split(".")[1]
                            edges.setdefault(src_layer, set()).add(dep_layer)
                            if dep_layer not in LAYER_WHITELIST[src_layer]:
                                violations.append(f"{rel}: {src_layer} -> {dep_layer}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod.startswith("lie_engine."):
                        dep_layer = mod.split(".")[1]
                        edges.setdefault(src_layer, set()).add(dep_layer)
                        if dep_layer not in LAYER_WHITELIST[src_layer]:
                            violations.append(f"{rel}: {src_layer} -> {dep_layer}")

        edges_out = {k: sorted(v) for k, v in sorted(edges.items())}
        violations = sorted(set(violations))
        return {
            "ok": len(violations) == 0,
            "files_checked": files_checked,
            "edges": edges_out,
            "violations": violations,
            "whitelist": {k: sorted(v) for k, v in sorted(LAYER_WHITELIST.items())},
        }

    def dependency_audit(self, as_of: date | None = None) -> dict[str, Any]:
        tz = ZoneInfo(self.settings.timezone)
        d = (as_of or datetime.now(tz).date()).isoformat()
        scan = self._scan()
        payload = {"date": d, **scan}

        json_path = self.output_dir / "review" / f"{d}_dependency_audit.json"
        md_path = self.output_dir / "review" / f"{d}_dependency_audit.md"
        write_json(json_path, payload)

        lines: list[str] = []
        lines.append(f"# 依赖分层审计 | {d}")
        lines.append("")
        lines.append(f"- 结果: `{'PASS' if scan['ok'] else 'FAIL'}`")
        lines.append(f"- 检查文件数: `{scan['files_checked']}`")
        lines.append(f"- 违规数: `{len(scan['violations'])}`")
        lines.append("")
        lines.append("## 层级依赖图")
        for layer, deps in scan["edges"].items():
            lines.append(f"- `{layer}` -> `{', '.join(deps) if deps else 'NONE'}`")
        lines.append("")
        if scan["violations"]:
            lines.append("## 违规明细")
            for item in scan["violations"]:
                lines.append(f"- {item}")
            lines.append("")
        write_markdown(md_path, "\n".join(lines) + "\n")

        payload["paths"] = {"json": str(json_path), "md": str(md_path)}
        return payload
