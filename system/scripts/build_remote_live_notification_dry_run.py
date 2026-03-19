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


def find_latest_preview(review_dir: Path) -> Path | None:
    files = sorted(
        review_dir.glob("*_remote_live_notification_preview.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def prune_artifacts(
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
        review_dir.glob("*_remote_live_notification_dry_run*.json"),
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


def validate_telegram(template: dict[str, Any] | None) -> dict[str, Any]:
    template = template if isinstance(template, dict) else {}
    reasons: list[str] = []
    parse_mode = str(template.get("parse_mode") or "")
    text = str(template.get("text") or "")
    if parse_mode != "MarkdownV2":
        reasons.append("telegram_parse_mode_invalid")
    if not text:
        reasons.append("telegram_text_missing")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "request": {
            "method": "POST",
            "url": "https://api.telegram.org/bot<TOKEN>/sendMessage",
            "headers": {
                "Content-Type": "application/json",
            },
            "json_body": {
                "chat_id": "<CHAT_ID>",
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": bool(template.get("disable_web_page_preview", True)),
            },
        },
    }


def validate_feishu(template: dict[str, Any] | None) -> dict[str, Any]:
    template = template if isinstance(template, dict) else {}
    reasons: list[str] = []
    msg_type = str(template.get("msg_type") or "")
    content = template.get("content")
    content = content if isinstance(content, dict) else {}
    if msg_type != "text":
        reasons.append("feishu_msg_type_invalid")
    if not str(content.get("text") or ""):
        reasons.append("feishu_text_missing")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "request": {
            "method": "POST",
            "url": "https://open.feishu.cn/open-apis/bot/v2/hook/<TOKEN>",
            "headers": {
                "Content-Type": "application/json",
            },
            "json_body": {
                "msg_type": msg_type,
                "content": content,
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only dry-run notification artifact from the latest remote live notification preview."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--preview-file", default="")
    parser.add_argument("--preview-returncode", type=int, default=0)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    preview_path = (
        Path(args.preview_file).expanduser().resolve()
        if str(args.preview_file).strip()
        else find_latest_preview(review_dir)
    )
    preview_payload = load_payload_file(preview_path)
    templates = (
        preview_payload.get("notification_templates")
        if isinstance(preview_payload, dict)
        and isinstance(preview_payload.get("notification_templates"), dict)
        else {}
    )
    telegram_template = (
        templates.get("telegram") if isinstance(templates.get("telegram"), dict) else {}
    )
    feishu_template = (
        templates.get("feishu") if isinstance(templates.get("feishu"), dict) else {}
    )
    generic_template = (
        templates.get("generic") if isinstance(templates.get("generic"), dict) else {}
    )
    notification_payload = (
        preview_payload.get("notification")
        if isinstance(preview_payload, dict)
        and isinstance(preview_payload.get("notification"), dict)
        else {}
    )
    runtime_floor_brief = (
        str(preview_payload.get("runtime_floor_brief") or "").strip()
        if isinstance(preview_payload, dict)
        else ""
    ) or (
        str(notification_payload.get("runtime_floor_brief") or "").strip()
        or str(generic_template.get("runtime_floor_brief") or "").strip()
        or None
    )
    focus_stack_brief = (
        str(preview_payload.get("focus_stack_brief") or "").strip()
        if isinstance(preview_payload, dict)
        else ""
    ) or (
        str(notification_payload.get("focus_stack_brief") or "").strip()
        or str(generic_template.get("focus_stack_brief") or "").strip()
        or None
    )

    status = "ok"
    ok = True
    if preview_payload is None:
        status = "preview_missing"
        ok = False
    elif not templates:
        status = "notification_templates_missing"
        ok = False

    telegram = validate_telegram(telegram_template)
    feishu = validate_feishu(feishu_template)
    if ok and not (telegram["ok"] and feishu["ok"]):
        status = "template_validation_failed"
        ok = False

    out: dict[str, Any] = {
        "action": "build_remote_live_notification_dry_run",
        "ok": ok,
        "status": status,
        "generated_at": fmt_utc(now_ts),
        "preview_returncode": int(args.preview_returncode),
        "source_preview_artifact": None if preview_path is None else str(preview_path),
        "handoff_state": preview_payload.get("handoff_state") if isinstance(preview_payload, dict) else None,
        "operator_status_triplet": preview_payload.get("operator_status_triplet") if isinstance(preview_payload, dict) else None,
        "next_focus_area": preview_payload.get("next_focus_area") if isinstance(preview_payload, dict) else None,
        "focus_stack_brief": focus_stack_brief,
        "runtime_floor_brief": runtime_floor_brief,
        "telegram": telegram,
        "feishu": feishu,
        "artifact_status_label": "notification-dry-run-ok" if ok else status,
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }
    suffix = str(out.get("handoff_state") or status)
    out["artifact_label"] = f"remote-live-notification-dry-run:{suffix}"
    out["artifact_tags"] = ["remote-live", "notification-dry-run", suffix]

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_remote_live_notification_dry_run.json"
    checksum_path = review_dir / f"{stamp}_remote_live_notification_dry_run_checksum.json"
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

    pruned_keep, pruned_age = prune_artifacts(
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
