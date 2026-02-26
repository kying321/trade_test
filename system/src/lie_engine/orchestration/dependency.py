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

DASHBOARD_ADAPTER_BANNED_PREFIXES: tuple[str, ...] = (
    "lie_engine.engine",
    "lie_engine.orchestration",
    "lie_engine.backtest",
    "lie_engine.signal",
    "lie_engine.risk",
    "lie_engine.regime",
    "lie_engine.research",
    "lie_engine.review",
)


@dataclass(slots=True)
class DependencyOrchestrator:
    settings: SystemSettings
    source_root: Path
    output_dir: Path

    def _scan_dashboard_adapter(self) -> dict[str, Any]:
        root = self.source_root.parent / "dashboard" / "api"
        if not root.exists():
            return {
                "enabled": False,
                "root": str(root),
                "files_checked": 0,
                "imports": [],
                "violations": [],
                "banned_prefixes": list(DASHBOARD_ADAPTER_BANNED_PREFIXES),
            }

        files_checked = 0
        imports: set[str] = set()
        violations: list[str] = []
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            files_checked += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = str(alias.name or "")
                        if imported.startswith("lie_engine."):
                            imports.add(imported)
                            if imported.startswith(DASHBOARD_ADAPTER_BANNED_PREFIXES):
                                violations.append(f"dashboard/api/{rel}: dashboard_api -> {imported}")
                elif isinstance(node, ast.ImportFrom):
                    imported = str(node.module or "")
                    if imported.startswith("lie_engine."):
                        imports.add(imported)
                        if imported.startswith(DASHBOARD_ADAPTER_BANNED_PREFIXES):
                            violations.append(f"dashboard/api/{rel}: dashboard_api -> {imported}")

        return {
            "enabled": True,
            "root": str(root),
            "files_checked": files_checked,
            "imports": sorted(imports),
            "violations": sorted(set(violations)),
            "banned_prefixes": list(DASHBOARD_ADAPTER_BANNED_PREFIXES),
        }

    def _scan(self) -> dict[str, Any]:
        root = self.source_root / "lie_engine"
        core_violations: list[str] = []
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
                                core_violations.append(f"{rel}: {src_layer} -> {dep_layer}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod.startswith("lie_engine."):
                        dep_layer = mod.split(".")[1]
                        edges.setdefault(src_layer, set()).add(dep_layer)
                        if dep_layer not in LAYER_WHITELIST[src_layer]:
                            core_violations.append(f"{rel}: {src_layer} -> {dep_layer}")

        edges_out = {k: sorted(v) for k, v in sorted(edges.items())}
        dashboard_scan = self._scan_dashboard_adapter()
        core_violations_out = sorted(set(core_violations))
        combined_violations = sorted(
            set(core_violations_out + list(dashboard_scan.get("violations", [])))
        )
        return {
            "ok": len(combined_violations) == 0,
            "files_checked": files_checked,
            "total_files_checked": files_checked + int(dashboard_scan.get("files_checked", 0) or 0),
            "edges": edges_out,
            "violations": combined_violations,
            "core_violations": core_violations_out,
            "dashboard_adapter": dashboard_scan,
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
        lines.append(f"- 检查文件数(core): `{scan['files_checked']}`")
        lines.append(
            f"- 检查文件数(dashboard_adapter): `{int((scan.get('dashboard_adapter', {}) or {}).get('files_checked', 0) or 0)}`"
        )
        lines.append(f"- 检查文件数(total): `{int(scan.get('total_files_checked', 0) or 0)}`")
        lines.append(f"- 违规数: `{len(scan['violations'])}`")
        lines.append("")
        lines.append("## 层级依赖图")
        for layer, deps in scan["edges"].items():
            lines.append(f"- `{layer}` -> `{', '.join(deps) if deps else 'NONE'}`")
        lines.append("")
        lines.append("## Dashboard Adapter Imports")
        dashboard_scan = scan.get("dashboard_adapter", {}) if isinstance(scan.get("dashboard_adapter", {}), dict) else {}
        lines.append(f"- enabled: `{bool(dashboard_scan.get('enabled', False))}`")
        imports = dashboard_scan.get("imports", []) if isinstance(dashboard_scan.get("imports", []), list) else []
        lines.append(f"- imports: `{', '.join(str(x) for x in imports) if imports else 'NONE'}`")
        lines.append("")
        if scan["violations"]:
            lines.append("## 违规明细")
            for item in scan["violations"]:
                lines.append(f"- {item}")
            lines.append("")
        write_markdown(md_path, "\n".join(lines) + "\n")

        payload["paths"] = {"json": str(json_path), "md": str(md_path)}
        return payload
