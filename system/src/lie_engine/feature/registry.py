from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import yaml


_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,127}$")
_ALLOWED_SOURCE_LAYERS = {"raw", "normalized", "feature", "external"}


@dataclass(slots=True)
class FeatureSpec:
    name: str
    entity_keys: list[str]
    source_layer: str
    ttl_minutes: int
    owner: str
    version: str = "v1"
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name).strip(),
            "entity_keys": [str(x).strip() for x in self.entity_keys if str(x).strip()],
            "source_layer": str(self.source_layer).strip().lower(),
            "ttl_minutes": int(self.ttl_minutes),
            "owner": str(self.owner).strip(),
            "version": str(self.version).strip() or "v1",
            "description": str(self.description).strip(),
            "tags": {str(k): str(v) for k, v in dict(self.tags).items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSpec:
        return cls(
            name=str(payload.get("name", "")).strip(),
            entity_keys=list(payload.get("entity_keys", []) or []),
            source_layer=str(payload.get("source_layer", "")).strip().lower(),
            ttl_minutes=int(payload.get("ttl_minutes", 0) or 0),
            owner=str(payload.get("owner", "")).strip(),
            version=str(payload.get("version", "v1")).strip() or "v1",
            description=str(payload.get("description", "")).strip(),
            tags=dict(payload.get("tags", {}) or {}),
        )


class FeatureRegistry:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._items: dict[str, FeatureSpec] = {}
        self.schema_version = "v1"

    @staticmethod
    def _validate_spec(spec: FeatureSpec) -> None:
        if not _NAME_RE.match(str(spec.name).strip()):
            raise ValueError(f"invalid feature name: {spec.name!r}")
        if not isinstance(spec.entity_keys, list) or not spec.entity_keys:
            raise ValueError(f"feature {spec.name!r} must have at least one entity_key")
        for key in spec.entity_keys:
            txt = str(key).strip()
            if not _NAME_RE.match(txt):
                raise ValueError(f"feature {spec.name!r} has invalid entity_key: {key!r}")
        layer = str(spec.source_layer).strip().lower()
        if layer not in _ALLOWED_SOURCE_LAYERS:
            raise ValueError(
                f"feature {spec.name!r} has invalid source_layer={layer!r}; allowed={sorted(_ALLOWED_SOURCE_LAYERS)}"
            )
        if int(spec.ttl_minutes) <= 0:
            raise ValueError(f"feature {spec.name!r} ttl_minutes must be positive")
        if not str(spec.owner).strip():
            raise ValueError(f"feature {spec.name!r} owner is required")

    @staticmethod
    def _normalize_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(payload or {})
        feats = data.get("features", [])
        if not isinstance(feats, list):
            feats = []
        return {
            "version": str(data.get("version", "v1")).strip() or "v1",
            "updated_at": str(data.get("updated_at", "")).strip(),
            "features": feats,
        }

    def load(self) -> FeatureRegistry:
        if not self.path.exists():
            self._items = {}
            self.schema_version = "v1"
            return self
        payload = self._normalize_payload(yaml.safe_load(self.path.read_text(encoding="utf-8")) or {})
        self.schema_version = str(payload.get("version", "v1"))
        items: dict[str, FeatureSpec] = {}
        for row in payload.get("features", []):
            if not isinstance(row, dict):
                continue
            spec = FeatureSpec.from_dict(row)
            self._validate_spec(spec)
            if spec.name in items:
                raise ValueError(f"duplicate feature name: {spec.name}")
            items[spec.name] = spec
        self._items = items
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.schema_version or "v1",
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "features": [spec.to_dict() for spec in sorted(self._items.values(), key=lambda x: x.name)],
        }
        self.path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    def upsert(self, spec: FeatureSpec) -> None:
        self._validate_spec(spec)
        self._items[spec.name] = spec

    def remove(self, feature_name: str) -> bool:
        key = str(feature_name).strip()
        if key in self._items:
            del self._items[key]
            return True
        return False

    def get(self, feature_name: str) -> FeatureSpec | None:
        return self._items.get(str(feature_name).strip())

    def list(self) -> list[FeatureSpec]:
        return [self._items[k] for k in sorted(self._items)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.schema_version or "v1",
            "features": [x.to_dict() for x in self.list()],
        }
