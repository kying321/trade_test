from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lie_engine.cpa_channels.account_bundle import load_account_bundles, normalize_account_bundle


SHANGHAI_TZ = timezone(timedelta(hours=8))


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _format_dt(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def build_cpa_auth_payload(
    account: dict[str, Any],
    *,
    now: datetime | None = None,
    tz: timezone = SHANGHAI_TZ,
) -> dict[str, Any]:
    bundle = normalize_account_bundle(account)
    payload = _decode_jwt_payload(bundle["access_token"])
    auth_info = payload.get("https://api.openai.com/auth", {})
    auth_info = auth_info if isinstance(auth_info, dict) else {}

    account_id = (
        str(bundle.get("chatgpt_account_id") or "").strip()
        or str(account.get("account_id") or "").strip()
        or str(auth_info.get("chatgpt_account_id") or "").strip()
    )
    exp_value = payload.get("exp")
    exp_ts = int(exp_value) if isinstance(exp_value, int) or str(exp_value).isdigit() else 0
    current = (now or datetime.now(timezone.utc)).astimezone(tz)
    expired = _format_dt(datetime.fromtimestamp(exp_ts, tz=tz)) if exp_ts > 0 else ""

    return {
        "type": bundle["provider"],
        "email": bundle["email"],
        "expired": expired,
        "id_token": bundle["id_token"],
        "account_id": account_id,
        "access_token": bundle["access_token"],
        "last_refresh": _format_dt(current),
        "refresh_token": bundle["refresh_token"],
    }


def write_cpa_authfile(
    account: dict[str, Any],
    *,
    output_dir: str | Path,
    now: datetime | None = None,
    tz: timezone = SHANGHAI_TZ,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_cpa_auth_payload(account, now=now, tz=tz)
    file_path = out_dir / f"{payload['email']}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"file": str(file_path), "payload": payload}


def export_cpa_authfiles(
    *,
    input_file: str | Path,
    output_dir: str | Path,
    now: datetime | None = None,
    tz: timezone = SHANGHAI_TZ,
) -> dict[str, Any]:
    accounts = [normalize_account_bundle(item) for item in load_account_bundles(input_file)]
    written_files: list[str] = []
    for account in accounts:
        result = write_cpa_authfile(account, output_dir=output_dir, now=now, tz=tz)
        written_files.append(str(result["file"]))
    return {
        "count": len(written_files),
        "output_dir": str(Path(output_dir)),
        "files": written_files,
    }
