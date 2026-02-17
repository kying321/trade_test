from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SystemSettings:
    raw: dict[str, Any]

    @property
    def timezone(self) -> str:
        return self.raw.get("timezone", "Asia/Shanghai")

    @property
    def schedule(self) -> dict[str, Any]:
        return self.raw.get("schedule", {})

    @property
    def thresholds(self) -> dict[str, float]:
        return self.raw.get("thresholds", {})

    @property
    def risk(self) -> dict[str, float]:
        return self.raw.get("risk", {})

    @property
    def validation(self) -> dict[str, float]:
        return self.raw.get("validation", {})

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw.get("universe", {})

    @property
    def paths(self) -> dict[str, str]:
        return self.raw.get("paths", {})


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config.yaml"


def load_settings(path: str | Path | None = None) -> SystemSettings:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SystemSettings(raw=raw)
