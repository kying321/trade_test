from __future__ import annotations

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
    return "".join(ch if ch in allowed else "_" for ch in value) or "signal"


def _unique_suffix(signal: TvBasisWebhookSignal) -> str:
    if signal.alert_id:
        return _sanitize_filename(signal.alert_id)
    return uuid.uuid4().hex[:8]


def signal_artifact_path(output_root: Path, signal: TvBasisWebhookSignal) -> Path:
    timestamp_safe = _sanitize_filename(signal.tv_timestamp)
    suffix = _unique_suffix(signal)
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


def write_signal_artifact(output_root: Path, signal: TvBasisWebhookSignal) -> Path:
    target = signal_artifact_path(output_root, signal)
    _write_json(target, _build_payload(signal))
    return target


def handle_webhook(payload: Dict[str, Any], *, output_root: Path | str) -> Path:
    root = Path(output_root)
    signal = parse_tv_basis_webhook_payload(payload)
    return write_signal_artifact(root, signal)
