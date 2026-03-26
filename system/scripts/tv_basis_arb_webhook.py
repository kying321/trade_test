from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tv_basis_arb_common import TvBasisWebhookSignal, parse_tv_basis_webhook_payload


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sanitize_filename(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    result = "".join(ch if ch in allowed else "_" for ch in value)
    return result or "signal"


def _deterministic_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def _unique_suffix(signal: TvBasisWebhookSignal, payload: Dict[str, Any]) -> str:
    parts: list[str] = []
    if signal.alert_id:
        parts.append(_sanitize_filename(signal.alert_id))
    parts.append(_deterministic_hash(payload))
    parts.append(uuid.uuid4().hex[:6])
    return "_".join(parts)


def signal_artifact_path(output_root: Path, signal: TvBasisWebhookSignal, payload: Dict[str, Any]) -> Path:
    timestamp_safe = _sanitize_filename(signal.tv_timestamp)
    suffix = _unique_suffix(signal, payload)
    artifact_dir = output_root / "review" / "tv_basis_arb"
    return artifact_dir / f"{timestamp_safe}_{signal.strategy_id}_{signal.event_type}_{suffix}.json"


def _build_payload(signal: TvBasisWebhookSignal) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "strategy_id": signal.strategy_id,
        "symbol": signal.symbol,
        "event_type": signal.event_type,
        "tv_timestamp": signal.tv_timestamp,
    }
    if signal.alert_id is not None:
        data["alert_id"] = signal.alert_id
    return data


def write_signal_artifact(output_root: Path, signal: TvBasisWebhookSignal, payload: Dict[str, Any]) -> Path:
    target = signal_artifact_path(output_root, signal, payload)
    _write_json(target, _build_payload(signal))
    return target


def handle_webhook(payload: Dict[str, Any], *, output_root: Path | str) -> Path:
    root = Path(output_root)
    signal = parse_tv_basis_webhook_payload(payload)
    return write_signal_artifact(root, signal, payload)
