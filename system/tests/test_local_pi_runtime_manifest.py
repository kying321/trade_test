from __future__ import annotations

import json
from pathlib import Path


RUNTIME_PI_SCRIPTS = Path(__file__).resolve().parents[1] / "runtime" / "pi" / "scripts"
RUNTIME_PI_MANIFEST = Path(__file__).resolve().parents[1] / "runtime" / "pi" / "runtime_manifest.json"


def test_runtime_pi_hot_path_scripts_are_repo_managed() -> None:
    expected = {
        "binance_spot_exec.py",
        "cortex_control_kernel.py",
        "cortex_evaluator.py",
        "cortex_gate.py",
        "cron_health_snapshot.py",
        "cron_policy_apply_batch.py",
        "cron_policy_patch_draft.py",
        "digital_life_core.py",
        "envelope_lint.py",
        "gate_notify_trend.py",
        "gateway_singleton_guard.py",
        "hippocampus.py",
        "lie_root_resolver.py",
        "lie_spine_watchdog.py",
        "lie_spot_halfhour_core.py",
        "memory_fallback_search.py",
        "net_resilience.py",
        "neuro_guard_cycle.py",
        "pi_cycle_halfhour_launchd_runner.sh",
        "pi_cycle_orchestrator.py",
        "reset_paper_state.py",
        "signal_registry.py",
    }

    manifest_payload = json.loads(RUNTIME_PI_MANIFEST.read_text(encoding="utf-8"))
    manifest_files = set(manifest_payload["files"])
    actual = {path.name for path in RUNTIME_PI_SCRIPTS.iterdir() if path.is_file()}

    missing = sorted(expected - manifest_files)
    assert not missing, f"missing repo-managed runtime scripts: {missing}"
    assert manifest_files == actual, f"runtime manifest drift: manifest_only={sorted(manifest_files - actual)} actual_only={sorted(actual - manifest_files)}"
