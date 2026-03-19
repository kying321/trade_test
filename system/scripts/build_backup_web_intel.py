#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = "output"
DEFAULT_STATE_REL_PATH = "state/backup_web_intel.json"
DEFAULT_TTL_SECONDS = 7200
DEFAULT_AUTHORITY = "risk_only"
ALLOWED_BIASES = {"long_bias", "short_bias", "neutral", "no_trade"}
ALLOWED_SEVERITIES = {"low", "medium", "high", "critical"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_text(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
        except Exception:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def resolve_state_path(raw: str, *, output_root: Path) -> Path:
    path = Path(str(raw).strip()) if str(raw).strip() else Path(DEFAULT_STATE_REL_PATH)
    if path.is_absolute():
        return path
    return output_root / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


def ensure_dict_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def normalize_page_ref(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    title = str(raw.get("title", "")).strip()
    url = str(raw.get("url", "")).strip()
    if title:
        out["title"] = title
    if url:
        out["url"] = url
    return out


def normalize_candidate_biases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        bias = str(row.get("bias", "")).strip().lower()
        if bias not in ALLOWED_BIASES:
            continue
        symbol = str(row.get("symbol", row.get("instrument", ""))).strip().upper()
        if not symbol:
            continue
        clean: dict[str, Any] = {
            "symbol": symbol,
            "bias": bias,
            "thesis_type": str(row.get("thesis_type", "")).strip(),
            "timeframe": str(row.get("timeframe", "")).strip(),
            "strength": str(row.get("strength", "")).strip(),
            "ticket_ready": bool(row.get("ticket_ready", False)),
        }
        for key in ("why", "fact_ids", "blockers", "invalidators"):
            if isinstance(row.get(key), list):
                clean[key] = [str(x).strip() for x in row.get(key, []) if str(x).strip()]
        out.append(clean)
    return out


def normalize_no_trade(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol", row.get("instrument", ""))).strip().upper()
        reason = str(row.get("reason", "")).strip()
        if not symbol and not reason:
            continue
        out.append({"symbol": symbol, "reason": reason})
    return out


def normalize_risk_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        code = str(row.get("code", "")).strip()
        severity = str(row.get("severity", "")).strip().lower()
        if not code or severity not in ALLOWED_SEVERITIES:
            continue
        out.append(
            {
                "code": code,
                "severity": severity,
                "message": str(row.get("message", "")).strip(),
            }
        )
    return out


def normalize_input(payload: dict[str, Any], *, ttl_seconds: int, input_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("input_payload_must_be_dict")
    status = str(payload.get("status", "ok")).strip().lower() or "ok"
    generated_dt = parse_utc_text(str(payload.get("generated_at", "") or payload.get("generated_at_utc", ""))) or now_utc()
    expires_dt = parse_utc_text(str(payload.get("expires_at", "") or payload.get("expires_at_utc", "")))
    if expires_dt is None:
        expires_dt = generated_dt + timedelta(seconds=max(60, int(ttl_seconds)))

    input_authority = str(payload.get("fallback_trade_authority", "")).strip().lower()
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "generated_at": generated_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "normalized_at_utc": now_utc_iso(),
        "expires_at": expires_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_root": str(payload.get("source_root", "")).strip(),
        "selected_primary_page": normalize_page_ref(payload.get("selected_primary_page", {})),
        "supplemental_pages": [normalize_page_ref(x) for x in ensure_dict_list(payload.get("supplemental_pages", [])) if normalize_page_ref(x)],
        "stale_source": bool(payload.get("stale_source", False)),
        "fallback_use_allowed": bool(payload.get("fallback_use_allowed", True)),
        "fallback_trade_authority": DEFAULT_AUTHORITY,
        "input_fallback_trade_authority": input_authority,
        "authority_normalized": input_authority != DEFAULT_AUTHORITY,
        "market_state": ensure_dict_list(payload.get("market_state", [])),
        "evidence_table": ensure_dict_list(payload.get("evidence_table", [])),
        "candidate_biases": normalize_candidate_biases(ensure_dict_list(payload.get("candidate_biases", []))),
        "no_trade_list": normalize_no_trade(ensure_dict_list(payload.get("no_trade_list", []))),
        "risk_flags": normalize_risk_flags(ensure_dict_list(payload.get("risk_flags", []))),
        "event_watchlist": ensure_dict_list(payload.get("event_watchlist", [])),
        "unknowns": ensure_dict_list(payload.get("unknowns", [])),
        "generator": {
            "input_path": str(input_path),
            "ttl_seconds": int(max(60, int(ttl_seconds))),
        },
        "summary": {
            "candidate_bias_count": 0,
            "no_trade_count": 0,
            "risk_flag_count": 0,
        },
    }
    output["summary"] = {
        "candidate_bias_count": int(len(output.get("candidate_biases", []))),
        "no_trade_count": int(len(output.get("no_trade_list", []))),
        "risk_flag_count": int(len(output.get("risk_flags", []))),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and write bounded backup web intel state for risk-only fallback.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state-path", default=DEFAULT_STATE_REL_PATH)
    parser.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    args = parser.parse_args()

    output_root = Path(str(args.output_root)).expanduser().resolve()
    input_path = Path(str(args.input_json)).expanduser().resolve()
    state_path = resolve_state_path(str(args.state_path), output_root=output_root)
    checksum_path = state_path.with_name(f"{state_path.stem}_checksum.json")

    payload: dict[str, Any]
    rc = 0
    try:
        raw = read_json(input_path)
        if not isinstance(raw, dict):
            raise ValueError("input_json_not_dict")
        payload = normalize_input(raw, ttl_seconds=max(60, int(args.ttl_seconds)), input_path=input_path)
        write_json(state_path, payload)
        file_sha, file_size = sha256_file(state_path)
        write_json(
            checksum_path,
            {
                "generated_at_utc": now_utc_iso(),
                "files": [{"path": str(state_path), "sha256": file_sha, "size_bytes": int(file_size)}],
            },
        )
        payload["artifact"] = str(state_path)
        payload["checksum"] = str(checksum_path)
        write_json(state_path, payload)
        rc = 0
    except Exception as exc:
        payload = {
            "generated_at_utc": now_utc_iso(),
            "action": "build_backup_web_intel",
            "ok": False,
            "error": str(exc),
            "input_json": str(input_path),
            "state_path": str(state_path),
        }
        rc = 2

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
