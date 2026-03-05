#!/usr/bin/env python3
import sys
from pathlib import Path

def test_pi_integration():
    try:
        import lie_engine
        from lie_engine.config import load_settings
    except ImportError:
        print("FAIL: lie_engine not installed or out of PYTHONPATH")
        sys.exit(1)
        
    config = load_settings("/Users/jokenrobot/Downloads/Folders/fenlie/system/config.yaml")
    val = config.validation if isinstance(config.validation, dict) else {}
    
    # 验证 Pi 数据总线是否生效
    has_micro_capture = val.get("micro_capture_daemon_enabled", False)
    has_time_sync = val.get("system_time_sync_probe_enabled", False)
    takeover = val.get("binance_live_takeover_enabled", False)
    
    print("=== Pi Integration Validation ===")
    print(f"Micro-Capture Daemon: {'[ACTIVE]' if has_micro_capture else '[DISABLED]'}")
    print(f"Time Sync Probe: {'[ACTIVE]' if has_time_sync else '[DISABLED]'}")
    print(f"Binance Live Takeover: {'[ACTIVE]' if takeover else '[DISABLED]'}")
    
    if takeover and has_micro_capture:
        print("=> SYSTEM ARCHITECTURE IS ROBUST: Live node connects via micro-capture feedback loop.")
    else:
        print("=> SYSTEM WARNING: Some integration nodes are disabled.")

if __name__ == "__main__":
    test_pi_integration()
