from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "lie-local"


def test_lie_local_wrapper_runs_validate_config_without_global_install() -> None:
    config_path = SYSTEM_ROOT / "config.daemon.test.yaml"

    assert SCRIPT_PATH.exists(), "expected repo-owned local CLI wrapper at scripts/lie-local"
    assert os.access(SCRIPT_PATH, os.X_OK), "scripts/lie-local must be executable"

    proc = subprocess.run(
        [str(SCRIPT_PATH), "--config", str(config_path), "validate-config"],
        cwd=SYSTEM_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert bool(payload.get("ok", False)) is True
    assert int(payload.get("summary", {}).get("errors", 0)) == 0
