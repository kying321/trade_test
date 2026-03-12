#!/usr/bin/env python3
"""Local fallback memory search (rg/grep style, offline).

Use when `openclaw memory search/status` is unavailable or degraded.
Search scope:
- MEMORY.md
- memory/**/*.md
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIMIT = 30


def _candidate_files(root: Path) -> List[Path]:
    files: List[Path] = []
    mem_md = root / "MEMORY.md"
    if mem_md.exists():
        files.append(mem_md)

    mem_dir = root / "memory"
    if mem_dir.exists():
        for p in sorted(mem_dir.rglob("*.md")):
            if p.is_file():
                files.append(p)
    return files


def _run_rg(query: str, files: List[Path], limit: int) -> List[Dict[str, Any]]:
    if not files:
        return []
    if shutil.which("rg") is None:
        return []

    cmd = ["rg", "-n", "--no-heading", "--color", "never", "-S", query] + [str(f) for f in files]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0, 1):  # 1 = no matches
        return []

    out: List[Dict[str, Any]] = []
    for raw in (p.stdout or "").splitlines():
        m = re.match(r"^(.*?):(\d+):(.*)$", raw)
        if not m:
            continue
        path_s, line_s, text = m.group(1), m.group(2), m.group(3)
        out.append(
            {
                "path": path_s,
                "line": int(line_s),
                "text": text.strip(),
            }
        )
        if len(out) >= limit:
            break
    return out


def _python_scan(query: str, files: List[Path], limit: int) -> List[Dict[str, Any]]:
    needle = query.lower()
    out: List[Dict[str, Any]] = []
    for p in files:
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            if needle in line.lower():
                out.append(
                    {
                        "path": str(p),
                        "line": idx,
                        "text": line.strip(),
                    }
                )
                if len(out) >= limit:
                    return out
    return out


def search_memory(query: str, root: Path, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")

    files = _candidate_files(root)
    hits = _run_rg(q, files, limit=limit)
    backend = "rg"
    if not hits:
        hits = _python_scan(q, files, limit=limit)
        backend = "python"

    return {
        "query": q,
        "root": str(root),
        "backend": backend,
        "files_scanned": len(files),
        "hits": hits,
        "hit_count": len(hits),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    payload = search_memory(args.query, root=root, limit=max(1, int(args.limit)))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

