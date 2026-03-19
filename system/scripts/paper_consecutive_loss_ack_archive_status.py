#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_ACK_PATH = SYSTEM_ROOT / "output" / "state" / "paper_consecutive_loss_ack.json"
DEFAULT_CHECKSUM_PATH = (
    SYSTEM_ROOT / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"
)
DEFAULT_ARCHIVE_DIR = SYSTEM_ROOT / "output" / "review" / "paper_consecutive_loss_ack_archive"
DEFAULT_ARCHIVE_MANIFEST_PATH = DEFAULT_ARCHIVE_DIR / "manifest.jsonl"


def parse_ts(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def parse_checksum_sidecar(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    token = raw.split(None, 1)[0].strip()
    if len(token) == 64:
        return token.lower()
    return None


def summarize_ack_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": payload.get("generated_at"),
        "expires_at": payload.get("expires_at"),
        "active": bool(payload.get("active")),
        "uses_remaining": int(payload.get("uses_remaining") or 0),
        "use_limit": int(payload.get("use_limit") or 0),
        "streak_snapshot": int(payload.get("streak_snapshot") or 0),
        "last_loss_ts": payload.get("last_loss_ts"),
        "consumed_at": payload.get("consumed_at"),
        "consume_reason": payload.get("consume_reason"),
        "archived_at": payload.get("archived_at"),
        "archive_reason": payload.get("archive_reason"),
        "archive_cycle_ts": payload.get("archive_cycle_ts"),
        "note": payload.get("note"),
    }


def inspect_live_ack(ack_path: Path, checksum_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "present": ack_path.exists(),
        "path": str(ack_path),
        "size_bytes": int(ack_path.stat().st_size) if ack_path.exists() else 0,
        "checksum_path": str(checksum_path),
        "checksum_present": checksum_path.exists(),
        "checksum_valid": None,
        "checksum_sha256": None,
        "checksum_expected_sha256": None,
        "payload": None,
    }
    if not ack_path.exists():
        return out
    payload = load_json(ack_path)
    out["payload"] = summarize_ack_payload(payload)
    actual = sha256_file(ack_path)
    out["checksum_sha256"] = actual
    checksum_payload = load_json(checksum_path)
    expected = str(checksum_payload.get("sha256") or "").strip() or None
    out["checksum_expected_sha256"] = expected
    if expected is not None:
        out["checksum_valid"] = expected.lower() == actual.lower()
    return out


def load_manifest_tail(path: Path, tail_count: int) -> tuple[list[dict[str, Any]], int, int]:
    if not path.exists():
        return [], 0, 0
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed: list[dict[str, Any]] = []
    parse_errors = 0
    for line in lines[-max(1, tail_count) :]:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            parse_errors += 1
            continue
        if isinstance(payload, dict):
            parsed.append(payload)
        else:
            parse_errors += 1
    return parsed, len(lines), parse_errors


def inspect_archive(
    archive_dir: Path,
    manifest_path: Path,
    *,
    manifest_tail: int,
    archive_keep_files: int,
) -> dict[str, Any]:
    archives = sorted(archive_dir.glob("paper_consecutive_loss_ack_*.json"), key=lambda p: p.name)
    checksums = sorted(archive_dir.glob("paper_consecutive_loss_ack_*.json.sha256"), key=lambda p: p.name)
    latest = archives[-1] if archives else None
    manifest_tail_entries, manifest_line_count, manifest_parse_errors = load_manifest_tail(
        manifest_path, manifest_tail
    )
    manifest_events = [str(entry.get("event") or "").strip() for entry in manifest_tail_entries]
    latest_archive: dict[str, Any] | None = None
    if latest is not None:
        checksum_path = latest.with_suffix(latest.suffix + ".sha256")
        checksum_expected = parse_checksum_sidecar(checksum_path)
        checksum_actual = sha256_file(latest)
        latest_payload = load_json(latest)
        latest_archive = {
            "path": str(latest),
            "size_bytes": int(latest.stat().st_size),
            "checksum_path": str(checksum_path),
            "checksum_present": checksum_path.exists(),
            "checksum_expected_sha256": checksum_expected,
            "checksum_sha256": checksum_actual,
            "checksum_valid": checksum_expected.lower() == checksum_actual.lower()
            if checksum_expected is not None
            else None,
            "payload": summarize_ack_payload(latest_payload),
        }
    return {
        "dir": str(archive_dir),
        "present": archive_dir.exists(),
        "archive_keep_files": int(archive_keep_files),
        "file_count": len(archives),
        "checksum_count": len(checksums),
        "latest_archive": latest_archive,
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_path.exists(),
        "manifest_line_count": manifest_line_count,
        "manifest_parse_errors": manifest_parse_errors,
        "manifest_tail": manifest_tail_entries,
        "manifest_tail_count": len(manifest_tail_entries),
        "manifest_tail_events": manifest_events,
        "consumed_archive_events_in_tail": manifest_events.count("consumed_archive"),
        "purged_events_in_tail": manifest_events.count("purged"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only status summary for live consecutive-loss ack artifact and consumed archive trail."
    )
    parser.add_argument("--ack-path", default=str(DEFAULT_ACK_PATH))
    parser.add_argument("--checksum-path", default=str(DEFAULT_CHECKSUM_PATH))
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--archive-manifest-path", default=str(DEFAULT_ARCHIVE_MANIFEST_PATH))
    parser.add_argument(
        "--archive-keep-files",
        type=int,
        default=max(1, int(os.getenv("LIE_PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES", "12"))),
    )
    parser.add_argument("--manifest-tail", type=int, default=5)
    parser.add_argument("--now", default="", help="Optional ISO8601 UTC override for deterministic checks.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    now_utc = parse_ts(args.now) or dt.datetime.now(dt.timezone.utc)
    ack_path = Path(args.ack_path).expanduser().resolve()
    checksum_path = Path(args.checksum_path).expanduser().resolve()
    archive_dir = Path(args.archive_dir).expanduser().resolve()
    manifest_path = Path(args.archive_manifest_path).expanduser().resolve()

    out = {
        "action": "paper_consecutive_loss_ack_archive_status",
        "ok": True,
        "generated_at": fmt_utc(now_utc),
        "system_root": str(SYSTEM_ROOT),
        "live_ack": inspect_live_ack(ack_path, checksum_path),
        "archive": inspect_archive(
            archive_dir,
            manifest_path,
            manifest_tail=max(1, int(args.manifest_tail)),
            archive_keep_files=max(1, int(args.archive_keep_files)),
        ),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
