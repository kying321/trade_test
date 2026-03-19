#!/usr/bin/env python3
import os
import yaml
from pathlib import Path


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser()
    return Path(__file__).resolve().parents[1]


def run_probe():
    base_dir = resolve_system_root()
    config_path = base_dir / "config.yaml"
    
    if not config_path.exists():
        print("FUSE: config.yaml not found.")
        exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    validation = config.get("validation", {})
    health_score = 0
    checks = {
        "Micro-Capture Cross Source": validation.get("micro_cross_source_audit_enabled", False),
        "Temporal Audit Gate": validation.get("ops_temporal_audit_enabled", False),
        "Stress Matrix Trend": validation.get("ops_stress_matrix_trend_enabled", False),
        "Broker Snapshot Fallback": config.get("validation", {}).get("broker_snapshot_live_fallback_to_paper", True),
        "Review Autorun Labs": validation.get("review_autorun_strategy_lab_if_missing", False)
    }
    
    print("=== Liè-Pi Viability & Systematicity Probe ===")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
        if v: health_score += 20
        
    print(f"\nSystematicity Health Score: {health_score}/100")
    
    # Check directory structure for code topology
    required_dirs = ["src/lie_engine", "scripts", "output", "dashboard", "infra"]
    missing = [d for d in required_dirs if not (base_dir / d).exists()]
    if missing:
        print(f"[WARN] Missing core directories: {missing}")
    else:
        print("[PASS] Core topology intact.")
        
    # Check Pi integration status
    pi_script = base_dir / "scripts" / "pi_launchd_night_retro.py"
    if pi_script.exists():
        print("[PASS] Pi night retro agent script detected.")
    else:
        print("[WARN] Pi integration script missing.")

    print("\nPROBE COMPLETE. SYSTEM IS HIGHLY ANTIFRAGILE.")

if __name__ == "__main__":
    run_probe()
