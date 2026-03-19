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
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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


def load_payload_file(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    clean = text.strip()
    if not clean:
        return None
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def find_latest_handoff(review_dir: Path) -> Path | None:
    files = sorted(
        review_dir.glob("*_remote_live_handoff.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def prune_previews(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_checksum: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    pruned_keep: list[str] = []
    pruned_age: list[str] = []
    review_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        review_dir.glob("*_remote_live_notification_preview*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    survivors: list[Path] = []
    protected = {current_artifact.name, current_checksum.name}
    for path in candidates:
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)
    artifact_like = [p for p in survivors if p.name.endswith(".json")]
    for path in artifact_like[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local read-only notification preview artifact from the latest remote live handoff."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--handoff-file", default="")
    parser.add_argument("--handoff-returncode", type=int, default=0)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    handoff_path = (
        Path(args.handoff_file).expanduser().resolve()
        if str(args.handoff_file).strip()
        else find_latest_handoff(review_dir)
    )
    handoff_payload = load_payload_file(handoff_path)
    operator_handoff = (
        handoff_payload.get("operator_handoff")
        if isinstance(handoff_payload, dict)
        and isinstance(handoff_payload.get("operator_handoff"), dict)
        else {}
    )
    notification = (
        operator_handoff.get("operator_notification")
        if isinstance(operator_handoff.get("operator_notification"), dict)
        else {}
    )
    templates = (
        operator_handoff.get("operator_notification_templates")
        if isinstance(operator_handoff.get("operator_notification_templates"), dict)
        else {}
    )
    handoff_state = str(operator_handoff.get("handoff_state") or "")

    status = "ok"
    ok = True
    if handoff_payload is None:
        status = "handoff_missing"
        ok = False
    elif not operator_handoff:
        status = "operator_handoff_missing"
        ok = False
    elif not notification:
        status = "notification_missing"
        ok = False
    elif not templates:
        status = "notification_templates_missing"
        ok = False

    out: dict[str, Any] = {
        "action": "build_remote_live_notification_preview",
        "ok": ok,
        "status": status,
        "generated_at": fmt_utc(now_ts),
        "handoff_returncode": int(args.handoff_returncode),
        "source_handoff_artifact": None if handoff_path is None else str(handoff_path),
        "handoff_state": handoff_state or None,
        "operator_status_triplet": operator_handoff.get("operator_status_triplet"),
        "next_focus_area": operator_handoff.get("next_focus_area"),
        "next_focus_reason": operator_handoff.get("next_focus_reason"),
        "focus_stack_brief": operator_handoff.get("focus_stack_brief"),
        "runtime_floor_brief": notification.get("runtime_floor_brief")
        if isinstance(notification, dict)
        else None,
        "notification": notification if notification else None,
        "notification_templates": templates if templates else None,
        "artifact_status_label": "notification-preview-ok" if ok else status,
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }
    suffix = handoff_state or status
    out["artifact_label"] = f"remote-live-notification-preview:{suffix}"
    out["artifact_tags"] = ["remote-live", "notification-preview", str(suffix)]

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_remote_live_notification_preview.json"
    checksum_path = review_dir / f"{stamp}_remote_live_notification_preview_checksum.json"
    out["artifact"] = str(artifact_path)
    out["checksum"] = str(checksum_path)

    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(artifact_path)
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": fmt_utc(now_ts),
                "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                "files": [
                    {
                        "path": str(artifact_path),
                        "sha256": digest,
                        "size_bytes": int(artifact_path.stat().st_size),
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_previews(
        review_dir,
        current_artifact=artifact_path,
        current_checksum=checksum_path,
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    out["pruned_keep"] = pruned_keep
    out["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
