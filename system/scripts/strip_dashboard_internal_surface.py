#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Strip local-only internal dashboard artifacts from public dist before deploy.")
    parser.add_argument("--dist-dir", type=Path, default=workspace_default / "system" / "dashboard" / "web" / "dist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.expanduser().resolve()
    targets = [
        dist_dir / "data" / "fenlie_dashboard_internal_snapshot.json",
    ]
    removed: list[str] = []
    missing: list[str] = []

    for target in targets:
        if target.exists():
            target.unlink()
            removed.append(str(target))
        else:
            missing.append(str(target))

    print(json.dumps({"ok": True, "dist_dir": str(dist_dir), "removed": removed, "missing": missing}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
