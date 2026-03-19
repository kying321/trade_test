#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request

from binance_live_takeover import PanicTriggered, panic_close_all


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_STATE_DIR = SYSTEM_ROOT / "output" / "state"
DEFAULT_DELIVERY = "none"
DEFAULT_TIMEOUT_MS = 5000
DEFAULT_RATE_LIMIT_PER_MINUTE = 6
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 1800
DEFAULT_IDEMPOTENCY_MAX_ENTRIES = 200


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


def find_latest_dry_run(review_dir: Path) -> Path | None:
    files = sorted(
        review_dir.glob("*_remote_live_notification_dry_run.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
            self.fd = None
        return False


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        review_dir.glob("*_remote_live_notification_send*.json"),
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


def normalize_delivery(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if value in {"", "none"}:
        return "none"
    if value in {"telegram", "feishu", "all"}:
        return value
    return "invalid"


def delivery_channels(delivery: str) -> list[str]:
    if delivery == "telegram":
        return ["telegram"]
    if delivery == "feishu":
        return ["feishu"]
    if delivery == "all":
        return ["telegram", "feishu"]
    return []


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_idempotency_entries(path: Path, *, ttl_seconds: int, now_ts: dt.datetime) -> list[dict[str, Any]]:
    rows = read_json_file(path, [])
    if not isinstance(rows, list):
        return []
    cutoff = now_ts - dt.timedelta(seconds=max(1, ttl_seconds))
    survivors: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts_text = str(row.get("ts") or "").strip()
        try:
            ts_value = dt.datetime.fromisoformat(ts_text)
        except Exception:
            continue
        if ts_value.tzinfo is None:
            ts_value = ts_value.replace(tzinfo=dt.timezone.utc)
        if ts_value >= cutoff:
            survivors.append(row)
    return survivors


def reserve_idempotency_key(
    *,
    ledger_path: Path,
    lock_path: Path,
    key_seed: str,
    ttl_seconds: int,
    max_entries: int,
    now_ts: dt.datetime,
) -> dict[str, Any]:
    with FileLock(lock_path):
        rows = load_idempotency_entries(ledger_path, ttl_seconds=ttl_seconds, now_ts=now_ts)
        idem_key = hashlib.sha256(key_seed.encode("utf-8")).hexdigest()
        duplicate = False
        existing_status = None
        for row in rows:
            if str(row.get("idempotency_key") or "") == idem_key and str(row.get("status") or "") in {
                "reserved",
                "sent",
            }:
                duplicate = True
                existing_status = str(row.get("status") or "")
                break
        if not duplicate:
            rows.append(
                {
                    "idempotency_key": idem_key,
                    "status": "reserved",
                    "ts": fmt_utc(now_ts),
                    "material_sha256": hashlib.sha256(key_seed.encode("utf-8")).hexdigest(),
                }
            )
            rows = rows[-max(1, int(max_entries)) :]
            write_json_file(ledger_path, rows)
        return {
            "enforced": True,
            "idempotency_key": idem_key,
            "duplicate": duplicate,
            "existing_status": existing_status,
            "ttl_seconds": max(1, int(ttl_seconds)),
        }


def finalize_idempotency_key(
    *,
    ledger_path: Path,
    lock_path: Path,
    idempotency_key: str,
    status: str,
    now_ts: dt.datetime,
) -> None:
    with FileLock(lock_path):
        rows = read_json_file(ledger_path, [])
        if not isinstance(rows, list):
            rows = []
        for row in reversed(rows):
            if str(row.get("idempotency_key") or "") == str(idempotency_key):
                row["status"] = str(status)
                row["finalized_at"] = fmt_utc(now_ts)
                break
        write_json_file(ledger_path, rows)


def acquire_bucket_token(
    *,
    bucket_state_path: Path,
    lock_path: Path,
    bucket_name: str,
    rate_limit_per_minute: int,
    now_ts: dt.datetime,
) -> dict[str, Any]:
    with FileLock(lock_path):
        state = read_json_file(bucket_state_path, {})
        if not isinstance(state, dict):
            state = {}
        buckets = state.get("buckets")
        if not isinstance(buckets, dict):
            buckets = {}
            state["buckets"] = buckets
        capacity = max(1, int(rate_limit_per_minute))
        refill_per_second = capacity / 60.0
        bucket = buckets.get(bucket_name)
        bucket = bucket if isinstance(bucket, dict) else {}
        tokens = float(bucket.get("tokens", capacity))
        last_refill = str(bucket.get("last_refill") or "")
        try:
            last_refill_ts = dt.datetime.fromisoformat(last_refill)
        except Exception:
            last_refill_ts = now_ts
        if last_refill_ts.tzinfo is None:
            last_refill_ts = last_refill_ts.replace(tzinfo=dt.timezone.utc)
        elapsed = max(0.0, (now_ts - last_refill_ts).total_seconds())
        tokens = min(float(capacity), float(tokens) + elapsed * refill_per_second)
        allowed = tokens >= 1.0
        retry_after_seconds = 0.0
        if allowed:
            tokens -= 1.0
        else:
            retry_after_seconds = max(0.0, (1.0 - tokens) / refill_per_second)
        buckets[bucket_name] = {
            "tokens": float(tokens),
            "capacity": int(capacity),
            "last_refill": fmt_utc(now_ts),
        }
        write_json_file(bucket_state_path, state)
        return {
            "allowed": allowed,
            "bucket_name": bucket_name,
            "capacity": int(capacity),
            "tokens_remaining": float(tokens),
            "retry_after_seconds": float(retry_after_seconds),
        }


def resolve_telegram_request(
    request_spec: dict[str, Any],
    *,
    token: str,
    chat_id: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if not token:
        reasons.append("telegram_token_missing")
    if not chat_id:
        reasons.append("telegram_chat_id_missing")
    method = str(request_spec.get("method") or "POST")
    url = str(request_spec.get("url") or "")
    json_body = request_spec.get("json_body")
    json_body = dict(json_body) if isinstance(json_body, dict) else {}
    if not url:
        reasons.append("telegram_request_url_missing")
    if not json_body:
        reasons.append("telegram_request_body_missing")
    if reasons:
        return None, reasons
    url = url.replace("<TOKEN>", token)
    json_body["chat_id"] = chat_id
    return {
        "method": method,
        "url": url,
        "headers": {"Content-Type": "application/json"},
        "json_body": json_body,
    }, reasons


def resolve_feishu_request(
    request_spec: dict[str, Any],
    *,
    hook_token: str,
    webhook_url: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    method = str(request_spec.get("method") or "POST")
    url = str(request_spec.get("url") or "")
    json_body = request_spec.get("json_body")
    json_body = dict(json_body) if isinstance(json_body, dict) else {}
    target_url = str(webhook_url or "").strip()
    if not target_url:
        if not hook_token:
            reasons.append("feishu_hook_token_missing")
        if url:
            target_url = url.replace("<TOKEN>", hook_token)
    if not target_url:
        reasons.append("feishu_request_url_missing")
    if not json_body:
        reasons.append("feishu_request_body_missing")
    if reasons:
        return None, reasons
    return {
        "method": method,
        "url": target_url,
        "headers": {"Content-Type": "application/json"},
        "json_body": json_body,
    }, reasons


def build_delivery_capabilities(
    *,
    delivery: str,
    telegram_request: dict[str, Any] | None,
    telegram_reasons: list[str],
    feishu_request: dict[str, Any] | None,
    feishu_reasons: list[str],
) -> tuple[dict[str, Any], str]:
    telegram_ready = bool(telegram_request) and not telegram_reasons
    feishu_ready = bool(feishu_request) and not feishu_reasons
    available_channels: list[str] = []
    blocked_channels: dict[str, list[str]] = {}
    if telegram_ready:
        available_channels.append("telegram")
    else:
        blocked_channels["telegram"] = list(telegram_reasons)
    if feishu_ready:
        available_channels.append("feishu")
    else:
        blocked_channels["feishu"] = list(feishu_reasons)

    readiness_label = "credentials-missing"
    if delivery == "none":
        readiness_label = "delivery-none"
    elif delivery == "telegram":
        readiness_label = "telegram-ready" if telegram_ready else "telegram-blocked"
    elif delivery == "feishu":
        readiness_label = "feishu-ready" if feishu_ready else "feishu-blocked"
    elif delivery == "all":
        if telegram_ready and feishu_ready:
            readiness_label = "all-ready"
        elif available_channels:
            readiness_label = "partial-ready"
        else:
            readiness_label = "all-blocked"

    return (
        {
            "delivery_requested": delivery,
            "telegram_configured": telegram_ready,
            "feishu_configured": feishu_ready,
            "available_channels": available_channels,
            "blocked_channels": blocked_channels,
        },
        readiness_label,
    )


def dispatch_request(
    *,
    channel: str,
    request_spec: dict[str, Any],
    timeout_ms: int,
    output_root: Path,
) -> dict[str, Any]:
    url = str(request_spec.get("url") or "")
    json_body = request_spec.get("json_body")
    headers = request_spec.get("headers")
    headers = headers if isinstance(headers, dict) else {}
    body = json.dumps(json_body if isinstance(json_body, dict) else {}, ensure_ascii=False).encode("utf-8")
    req = request.Request(url=url, data=body, headers={str(k): str(v) for k, v in headers.items()}, method="POST")
    timeout_seconds = min(5.0, max(0.1, float(max(100, int(timeout_ms))) / 1000.0))
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read(4096)
            text = raw.decode("utf-8", errors="replace")
            return {
                "sent": True,
                "http_status": int(getattr(response, "status", 200)),
                "response_excerpt": text[:512],
            }
    except urlerror.HTTPError as exc:
        body_excerpt = ""
        try:
            body_excerpt = exc.read(4096).decode("utf-8", errors="replace")[:512]
        except Exception:
            body_excerpt = str(exc)
        if channel == "telegram" and int(exc.code) == 409:
            panic_close_all(
                output_root,
                reason="remote_live_notification_telegram_409_conflict",
                detail=body_excerpt or str(exc),
            )
        return {
            "sent": False,
            "http_status": int(exc.code),
            "response_excerpt": body_excerpt,
            "error": str(exc),
        }
    except (urlerror.URLError, socket.timeout, TimeoutError) as exc:
        panic_close_all(
            output_root,
            reason=f"remote_live_notification_{channel}_transport_error",
            detail=str(exc),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send or preview remote live notifications with idempotency, token bucket, timeout, and artifact governance."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--dry-run-file", default="")
    parser.add_argument("--dry-run-returncode", type=int, default=0)
    parser.add_argument("--delivery", default=DEFAULT_DELIVERY)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--rate-limit-per-minute", type=int, default=DEFAULT_RATE_LIMIT_PER_MINUTE)
    parser.add_argument("--idempotency-ttl-seconds", type=int, default=DEFAULT_IDEMPOTENCY_TTL_SECONDS)
    parser.add_argument("--idempotency-max-entries", type=int, default=DEFAULT_IDEMPOTENCY_MAX_ENTRIES)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--telegram-token", default=str(os.getenv("REMOTE_LIVE_NOTIFICATION_TELEGRAM_TOKEN", "")))
    parser.add_argument("--telegram-chat-id", default=str(os.getenv("REMOTE_LIVE_NOTIFICATION_TELEGRAM_CHAT_ID", "")))
    parser.add_argument("--feishu-hook-token", default=str(os.getenv("REMOTE_LIVE_NOTIFICATION_FEISHU_HOOK_TOKEN", "")))
    parser.add_argument("--feishu-webhook-url", default=str(os.getenv("REMOTE_LIVE_NOTIFICATION_FEISHU_WEBHOOK_URL", "")))
    parser.add_argument("--now", default="")
    return parser


def make_channel_outcome(
    channel: str,
    *,
    selected: bool,
    request: dict[str, Any] | None = None,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "channel": channel,
        "selected": bool(selected),
        "ok": False,
        "sent": False,
        "would_send": bool(selected),
        "duplicate": False,
        "rate_limited": False,
        "request": request,
        "reasons": list(reasons or []),
        "http_status": None,
        "response_excerpt": None,
        "idempotency": None,
        "bucket": None,
    }


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    state_dir = Path(args.state_dir).expanduser().resolve()
    output_root = state_dir.parent
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    delivery = normalize_delivery(args.delivery)
    dry_run_path = (
        Path(args.dry_run_file).expanduser().resolve()
        if str(args.dry_run_file).strip()
        else find_latest_dry_run(review_dir)
    )
    dry_run_payload = load_payload_file(dry_run_path)

    ok = True
    status = "ok"
    if delivery == "invalid":
        status = "delivery_invalid"
        ok = False
    elif dry_run_payload is None:
        status = "dry_run_missing"
        ok = False

    telegram_source = (
        dry_run_payload.get("telegram")
        if isinstance(dry_run_payload, dict) and isinstance(dry_run_payload.get("telegram"), dict)
        else {}
    )
    feishu_source = (
        dry_run_payload.get("feishu")
        if isinstance(dry_run_payload, dict) and isinstance(dry_run_payload.get("feishu"), dict)
        else {}
    )
    requested_channels = delivery_channels(delivery)

    telegram_request, telegram_request_reasons = resolve_telegram_request(
        telegram_source.get("request") if isinstance(telegram_source.get("request"), dict) else {},
        token=str(args.telegram_token or "").strip(),
        chat_id=str(args.telegram_chat_id or "").strip(),
    )
    feishu_request, feishu_request_reasons = resolve_feishu_request(
        feishu_source.get("request") if isinstance(feishu_source.get("request"), dict) else {},
        hook_token=str(args.feishu_hook_token or "").strip(),
        webhook_url=str(args.feishu_webhook_url or "").strip(),
    )
    delivery_capabilities, delivery_readiness_label = build_delivery_capabilities(
        delivery=delivery,
        telegram_request=telegram_request,
        telegram_reasons=telegram_request_reasons,
        feishu_request=feishu_request,
        feishu_reasons=feishu_request_reasons,
    )

    out: dict[str, Any] = {
        "action": "send_remote_live_notification",
        "ok": ok,
        "status": status,
        "generated_at": fmt_utc(now_ts),
        "dry_run_returncode": int(args.dry_run_returncode),
        "source_dry_run_artifact": None if dry_run_path is None else str(dry_run_path),
        "handoff_state": dry_run_payload.get("handoff_state") if isinstance(dry_run_payload, dict) else None,
        "operator_status_triplet": dry_run_payload.get("operator_status_triplet") if isinstance(dry_run_payload, dict) else None,
        "next_focus_area": dry_run_payload.get("next_focus_area") if isinstance(dry_run_payload, dict) else None,
        "focus_stack_brief": dry_run_payload.get("focus_stack_brief")
        if isinstance(dry_run_payload, dict)
        else None,
        "runtime_floor_brief": dry_run_payload.get("runtime_floor_brief")
        if isinstance(dry_run_payload, dict)
        else None,
        "delivery_capabilities": delivery_capabilities,
        "delivery_readiness_label": delivery_readiness_label,
        "delivery_requested": delivery,
        "timeout_ms": min(5000, max(100, int(args.timeout_ms))),
        "rate_limit_per_minute": max(1, int(args.rate_limit_per_minute)),
        "telegram": make_channel_outcome(
            "telegram",
            selected="telegram" in requested_channels,
            request=telegram_request,
            reasons=telegram_request_reasons if "telegram" in requested_channels else [],
        ),
        "feishu": make_channel_outcome(
            "feishu",
            selected="feishu" in requested_channels,
            request=feishu_request,
            reasons=feishu_request_reasons if "feishu" in requested_channels else [],
        ),
        "channels_attempted": [],
        "channels_sent": [],
        "channels_skipped": [],
        "artifact_status_label": None,
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
        "panic": None,
    }

    suffix = str(out.get("handoff_state") or status)
    out["artifact_label"] = f"remote-live-notification-send:{suffix}"
    out["artifact_tags"] = ["remote-live", "notification-send", suffix]

    idempotency_ledger = state_dir / "remote_live_notification_send_idempotency.json"
    bucket_state_path = state_dir / "remote_live_notification_token_bucket.json"
    lock_path = state_dir / "remote_live_notification_send.lock"

    exit_code = 0
    try:
        if delivery == "none" and ok:
            status = "delivery_none"
        elif ok:
            for channel in requested_channels:
                channel_out = out[channel]
                channel_out["would_send"] = True
                out["channels_attempted"].append(channel)
                request_spec = channel_out.get("request")
                if not isinstance(request_spec, dict):
                    if not channel_out["reasons"]:
                        channel_out["reasons"].append("request_unresolved")
                    out["channels_skipped"].append(channel)
                    continue
                if channel_out["reasons"]:
                    out["channels_skipped"].append(channel)
                    continue
                idem_material = stable_json_dumps(
                    {
                        "channel": channel,
                        "handoff_state": out.get("handoff_state"),
                        "operator_status_triplet": out.get("operator_status_triplet"),
                        "request": request_spec,
                    }
                )
                idem = reserve_idempotency_key(
                    ledger_path=idempotency_ledger,
                    lock_path=lock_path,
                    key_seed=idem_material,
                    ttl_seconds=max(1, int(args.idempotency_ttl_seconds)),
                    max_entries=max(10, int(args.idempotency_max_entries)),
                    now_ts=now_ts,
                )
                channel_out["idempotency"] = idem
                if bool(idem.get("duplicate")):
                    channel_out["duplicate"] = True
                    channel_out["ok"] = True
                    channel_out["reasons"].append("idempotent_skip")
                    out["channels_skipped"].append(channel)
                    continue
                bucket = acquire_bucket_token(
                    bucket_state_path=bucket_state_path,
                    lock_path=lock_path,
                    bucket_name=channel,
                    rate_limit_per_minute=max(1, int(args.rate_limit_per_minute)),
                    now_ts=now_ts,
                )
                channel_out["bucket"] = bucket
                if not bool(bucket.get("allowed")):
                    channel_out["rate_limited"] = True
                    channel_out["reasons"].append("token_bucket_exhausted")
                    finalize_idempotency_key(
                        ledger_path=idempotency_ledger,
                        lock_path=lock_path,
                        idempotency_key=str(idem.get("idempotency_key") or ""),
                        status="failed",
                        now_ts=now_ts,
                    )
                    out["channels_skipped"].append(channel)
                    continue

                result = dispatch_request(
                    channel=channel,
                    request_spec=request_spec,
                    timeout_ms=min(5000, max(100, int(args.timeout_ms))),
                    output_root=output_root,
                )
                channel_out["sent"] = bool(result.get("sent"))
                channel_out["ok"] = bool(result.get("sent"))
                channel_out["http_status"] = result.get("http_status")
                channel_out["response_excerpt"] = result.get("response_excerpt")
                if channel_out["sent"]:
                    finalize_idempotency_key(
                        ledger_path=idempotency_ledger,
                        lock_path=lock_path,
                        idempotency_key=str(idem.get("idempotency_key") or ""),
                        status="sent",
                        now_ts=now_ts,
                    )
                    out["channels_sent"].append(channel)
                else:
                    if result.get("error"):
                        channel_out["reasons"].append("http_error")
                    finalize_idempotency_key(
                        ledger_path=idempotency_ledger,
                        lock_path=lock_path,
                        idempotency_key=str(idem.get("idempotency_key") or ""),
                        status="failed",
                        now_ts=now_ts,
                    )

            if out["channels_sent"] and len(out["channels_sent"]) == len(requested_channels):
                status = "sent"
            elif out["channels_sent"] and requested_channels:
                status = "partial_sent"
                ok = False
            elif requested_channels and all(out[ch]["duplicate"] for ch in requested_channels):
                status = "idempotent_skip"
            elif requested_channels and all(out[ch]["rate_limited"] for ch in requested_channels):
                status = "rate_limited"
                ok = False
            elif requested_channels and any(out[ch]["reasons"] for ch in requested_channels):
                status = "delivery_blocked"
                ok = False
    except PanicTriggered as exc:
        ok = False
        status = "panic_triggered"
        out["panic"] = {"message": str(exc)}
        exit_code = 3

    out["ok"] = ok
    out["status"] = status
    out["artifact_status_label"] = "notification-send-ok" if ok and status in {"delivery_none", "sent", "idempotent_skip"} else status

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_remote_live_notification_send.json"
    checksum_path = review_dir / f"{stamp}_remote_live_notification_send_checksum.json"
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
