from __future__ import annotations

from datetime import date
from pathlib import Path
import hashlib
from typing import Any

from lie_engine.data.storage import write_json


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), int(size)


def extract_iso_date_prefix(path: Path) -> date | None:
    name = str(path.name).strip()
    if len(name) < 10:
        return None
    prefix = name[:10]
    try:
        return date.fromisoformat(prefix)
    except Exception:
        return None


def collect_dated_artifact_pairs(
    *,
    directory: Path,
    json_glob: str,
    md_glob: str,
) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for p in sorted(directory.glob(str(json_glob))):
        d = extract_iso_date_prefix(p)
        if d is None:
            continue
        key = d.isoformat()
        out.setdefault(key, {})
        out[key]["json"] = p
    for p in sorted(directory.glob(str(md_glob))):
        d = extract_iso_date_prefix(p)
        if d is None:
            continue
        key = d.isoformat()
        out.setdefault(key, {})
        out[key]["md"] = p
    return out


def rotate_dated_artifact_pairs(
    *,
    as_of: date,
    pairs: dict[str, dict[str, Path]],
    retention_days: int,
) -> dict[str, Any]:
    keep_days = max(1, int(retention_days))
    rotated_out_dates: list[str] = []
    deleted_files = 0

    for dstr in sorted(pairs.keys()):
        try:
            d0 = date.fromisoformat(str(dstr))
        except Exception:
            continue
        age_days = (as_of - d0).days
        if age_days < keep_days:
            continue
        if age_days < 0:
            # Future artifacts are not touched by retention.
            continue
        item = pairs.get(dstr, {})
        for key in ("json", "md"):
            p = item.get(key)
            if isinstance(p, Path) and p.exists():
                p.unlink()
                deleted_files += 1
        rotated_out_dates.append(str(dstr))

    return {
        "retention_days": int(keep_days),
        "rotated_out_count": int(len(rotated_out_dates)),
        "rotated_out_dates": rotated_out_dates,
        "deleted_files": int(deleted_files),
    }


def write_dated_artifact_checksum_index(
    *,
    as_of: date,
    pairs: dict[str, dict[str, Path]],
    retention_days: int,
    rotated_out_dates: list[str],
    index_path: Path,
) -> dict[str, Any]:
    index_entries: list[dict[str, Any]] = []
    for dstr in sorted(pairs.keys(), reverse=True):
        item = pairs.get(dstr, {})
        row: dict[str, Any] = {
            "date": str(dstr),
            "json": "",
            "json_sha256": "",
            "json_bytes": 0,
            "md": "",
            "md_sha256": "",
            "md_bytes": 0,
            "pair_complete": False,
        }
        json_path = item.get("json")
        if isinstance(json_path, Path) and json_path.exists():
            row["json"] = str(json_path)
            digest, size = sha256_file(json_path)
            row["json_sha256"] = str(digest)
            row["json_bytes"] = int(size)
        md_path = item.get("md")
        if isinstance(md_path, Path) and md_path.exists():
            row["md"] = str(md_path)
            digest, size = sha256_file(md_path)
            row["md_sha256"] = str(digest)
            row["md_bytes"] = int(size)
        row["pair_complete"] = bool(row["json"] and row["md"])
        index_entries.append(row)

    index_payload = {
        "generated_for_date": as_of.isoformat(),
        "retention_days": int(max(1, int(retention_days))),
        "rotated_out_dates": [str(x) for x in rotated_out_dates],
        "entry_count": int(len(index_entries)),
        "entries": index_entries,
    }
    write_json(index_path, index_payload)
    return {
        "written": True,
        "path": str(index_path),
        "entries": int(len(index_entries)),
    }


def apply_dated_artifact_governance(
    *,
    as_of: date,
    directory: Path,
    json_glob: str,
    md_glob: str,
    retention_days: int,
    checksum_index_enabled: bool,
    checksum_index_filename: str,
) -> dict[str, Any]:
    keep_days = max(1, int(retention_days))
    index_enabled = bool(checksum_index_enabled)

    rotation_failed = False
    checksum_failed = False
    rotated_out_count = 0
    rotated_out_dates: list[str] = []
    checksum_written = False
    checksum_path = ""
    checksum_entries = 0
    reason = ""

    try:
        pairs = collect_dated_artifact_pairs(directory=directory, json_glob=json_glob, md_glob=md_glob)
        rotation = rotate_dated_artifact_pairs(as_of=as_of, pairs=pairs, retention_days=keep_days)
        rotated_out_count = int(rotation.get("rotated_out_count", 0))
        rotated_out_dates = [str(x) for x in rotation.get("rotated_out_dates", [])]
    except Exception as exc:
        rotation_failed = True
        reason = f"rotation_failed:{type(exc).__name__}:{exc}"

    if index_enabled and (not rotation_failed):
        try:
            pairs = collect_dated_artifact_pairs(directory=directory, json_glob=json_glob, md_glob=md_glob)
            checksum_meta = write_dated_artifact_checksum_index(
                as_of=as_of,
                pairs=pairs,
                retention_days=keep_days,
                rotated_out_dates=rotated_out_dates,
                index_path=directory / str(checksum_index_filename),
            )
            checksum_written = bool(checksum_meta.get("written", False))
            checksum_path = str(checksum_meta.get("path", ""))
            checksum_entries = int(checksum_meta.get("entries", 0))
        except Exception as exc:
            checksum_failed = True
            reason = f"checksum_index_write_failed:{type(exc).__name__}:{exc}"
    elif not index_enabled:
        checksum_written = False
        checksum_path = ""
        checksum_entries = 0
    else:
        checksum_written = False
        checksum_path = ""
        checksum_entries = 0
        checksum_failed = True
        if not reason:
            reason = "checksum_index_skipped_due_to_rotation_failure"

    return {
        "retention_days": int(keep_days),
        "rotated_out_count": int(rotated_out_count),
        "rotated_out_dates": rotated_out_dates,
        "rotation_failed": bool(rotation_failed),
        "checksum_index_enabled": bool(index_enabled),
        "checksum_index_written": bool(checksum_written),
        "checksum_index_path": str(checksum_path),
        "checksum_index_entries": int(checksum_entries),
        "checksum_index_failed": bool(checksum_failed),
        "reason": str(reason),
    }
