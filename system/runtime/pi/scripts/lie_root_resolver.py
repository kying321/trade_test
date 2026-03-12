#!/usr/bin/env python3
"""Resolve LiE system root across migrated workspace layouts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


_THIS_FILE = Path(__file__).resolve()
_SCRIPT_PARENT = _THIS_FILE.parents[1]
_DEFAULT_CANDIDATES = (
    _SCRIPT_PARENT / "fenlie-system",
    _SCRIPT_PARENT,
    Path.cwd() / "fenlie-system",
    Path.cwd(),
)


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw).expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _looks_like_engine_root(path: Path) -> bool:
    return (path / "config.yaml").is_file() and (path / "src" / "lie_engine").is_dir()


def resolve_lie_system_root(*, extra_candidates: Iterable[Path] | None = None) -> Path:
    env_raw = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip()
    env_candidate = Path(env_raw).expanduser() if env_raw else None
    fenlie_env_raw = str(os.getenv("FENLIE_SYSTEM_ROOT", "")).strip()
    fenlie_env_candidate = Path(fenlie_env_raw).expanduser() if fenlie_env_raw else None

    # Explicit env always wins (tests and staged sandboxes rely on this override).
    if env_candidate is not None:
        return env_candidate

    candidates: List[Path] = []
    if extra_candidates is not None:
        candidates.extend(extra_candidates)
    if fenlie_env_candidate is not None:
        candidates.append(fenlie_env_candidate)
    candidates.extend(_DEFAULT_CANDIDATES)

    ordered = _dedupe_paths(candidates)
    for candidate in ordered:
        if _looks_like_engine_root(candidate):
            return candidate

    # Fallback: honor explicit env if provided (even if currently incomplete).
    if env_candidate is not None:
        return env_candidate

    # Last fallback prefers the colocated system root over ambient cwd.
    return _DEFAULT_CANDIDATES[0]
