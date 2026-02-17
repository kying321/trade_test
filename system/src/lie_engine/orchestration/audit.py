from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from lie_engine.config import SystemSettings, validate_settings
from lie_engine.data.storage import write_json, write_markdown


@dataclass(slots=True)
class ArchitectureOrchestrator:
    settings: SystemSettings
    output_dir: Path
    health_check: Callable[[date, bool], dict[str, Any]]

    def architecture_audit(self, as_of: date | None = None) -> dict[str, Any]:
        tz = ZoneInfo(self.settings.timezone)
        target = as_of or datetime.now(tz).date()
        d = target.isoformat()
        config_report = validate_settings(self.settings)
        health = self.health_check(target, False)

        manifest_dir = self.output_dir / "artifacts" / "manifests"
        expected = {
            "eod_manifest": manifest_dir / f"eod_{d}.json",
            "backtest_manifest": manifest_dir / f"backtest_2015-01-01_{d}.json",
            "review_manifest": manifest_dir / f"review_{d}.json",
        }
        manifest_checks = {k: p.exists() for k, p in expected.items()}

        errors = int(config_report.get("summary", {}).get("errors", 0))
        missing_manifests = [k for k, v in manifest_checks.items() if not v]
        health_ok = str(health.get("status", "")) == "healthy"
        if errors > 0:
            status = "fail"
        elif (not health_ok) or missing_manifests:
            status = "warn"
        else:
            status = "pass"
        payload = {
            "date": d,
            "status": status,
            "config": config_report,
            "health": health,
            "manifests": manifest_checks,
            "missing_manifests": missing_manifests,
        }

        json_path = self.output_dir / "review" / f"{d}_architecture_audit.json"
        md_path = self.output_dir / "review" / f"{d}_architecture_audit.md"
        write_json(json_path, payload)
        lines = [
            f"# 架构审计报告 | {d}",
            "",
            f"- 状态: `{status}`",
            f"- 配置错误数: `{errors}`",
            f"- 配置告警数: `{config_report.get('summary', {}).get('warnings', 0)}`",
            f"- 健康状态: `{health.get('status', 'unknown')}`",
            f"- 缺失 Manifest 数: `{len(missing_manifests)}`",
            "",
            "## Manifest 检查",
        ]
        for k, v in manifest_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        if config_report.get("errors"):
            lines.append("## 配置错误")
            for item in config_report["errors"]:
                lines.append(f"- `{item['path']}`: {item['message']}")
            lines.append("")
        if config_report.get("warnings"):
            lines.append("## 配置告警")
            for item in config_report["warnings"]:
                lines.append(f"- `{item['path']}`: {item['message']}")
            lines.append("")
        write_markdown(md_path, "\n".join(lines) + "\n")
        payload["paths"] = {"json": str(json_path), "md": str(md_path)}
        return payload
