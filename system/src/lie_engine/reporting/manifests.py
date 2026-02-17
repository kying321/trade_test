from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from lie_engine.data.storage import write_json


def write_run_manifest(
    *,
    output_dir: Path,
    run_type: str,
    run_id: str,
    artifacts: dict[str, str],
    metrics: dict[str, Any] | None = None,
    checks: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = output_dir / "artifacts" / "manifests" / f"{run_type}_{run_id}.json"
    payload = {
        "run_type": run_type,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "artifacts": artifacts,
        "metrics": metrics or {},
        "checks": checks or {},
        "metadata": metadata or {},
    }
    write_json(path, payload)
    return path
