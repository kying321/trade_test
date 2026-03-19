from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "render_openclaw_orderflow_executor_unit.py"
    spec = importlib.util.spec_from_file_location("render_openclaw_orderflow_executor_unit", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_render_openclaw_orderflow_executor_unit_contains_expected_execstart() -> None:
    mod = _load_module()
    args = mod.argparse.Namespace(
        project_dir="/home/ubuntu/openclaw-system",
        user="ubuntu",
        poll_seconds=15,
        executor_timeout_seconds=5,
        mode="shadow_guarded",
        max_loops=0,
        output_path="",
    )
    text = mod.render_unit(args)
    assert "Description=Fenlie OpenClaw Orderflow Executor" in text
    assert "User=ubuntu" in text
    assert "WorkingDirectory=/home/ubuntu/openclaw-system" in text
    assert "scripts/openclaw_orderflow_executor.py" in text
    assert "--poll-seconds 15" in text
    assert "--executor-timeout-seconds 5" in text
    assert "--mode shadow_guarded" in text
    assert "ProtectSystem=strict" in text
    assert "PrivateNetwork=true" in text

