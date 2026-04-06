from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_json_stream(raw: str) -> list[dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]

    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    idx = 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx >= len(raw):
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            items.append(obj)
        idx = end
    return items


def load_account_bundles(input_file: str | Path) -> list[dict[str, Any]]:
    raw = Path(input_file).read_text(encoding="utf-8")
    return _parse_json_stream(raw)


def normalize_account_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    email = str(bundle.get("email") or "").strip().lower()
    if not email:
        raise ValueError("account.email is required")
    access_token = str(bundle.get("access_token") or "").strip()
    if not access_token:
        raise ValueError("account.access_token is required")
    provider = str(bundle.get("provider") or bundle.get("type") or "codex").strip() or "codex"
    return {
        "email": email,
        "provider": provider,
        "type": provider,
        "access_token": access_token,
        "refresh_token": str(bundle.get("refresh_token") or "").strip(),
        "id_token": str(bundle.get("id_token") or "").strip(),
        "chatgpt_account_id": str(bundle.get("chatgpt_account_id") or bundle.get("account_id") or "").strip(),
        "chatgpt_user_id": str(bundle.get("chatgpt_user_id") or bundle.get("user_id") or "").strip(),
        "plan_type": str(bundle.get("plan_type") or "").strip(),
        "source_file": str(bundle.get("source_file") or "").strip(),
    }


def import_bundles(input_file: str | Path) -> list[dict[str, Any]]:
    return [normalize_account_bundle(item) for item in load_account_bundles(input_file)]
