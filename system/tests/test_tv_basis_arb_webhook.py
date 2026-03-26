from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_webhook_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "tv_basis_arb_webhook.py"
    spec = importlib.util.spec_from_file_location("tv_basis_arb_webhook", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_payload(strategy_id: str = "tv_basis_btc_spot_perp_v1") -> dict[str, str]:
    return {
        "strategy_id": strategy_id,
        "symbol": "BTCUSDT",
        "event_type": "entry_check",
        "tv_timestamp": "2026-03-26T12:30:00Z",
    }


def test_missing_strategy_id_rejected(tmp_path: Path) -> None:
    webhook = _load_webhook_module()
    payload = _minimal_payload()
    payload.pop("strategy_id")

    with pytest.raises(ValueError, match="strategy_id"):
        webhook.handle_webhook(payload, output_root=tmp_path)


def test_unknown_strategy_id_rejected(tmp_path: Path) -> None:
    webhook = _load_webhook_module()
    payload = _minimal_payload(strategy_id="unknown_strategy")

    with pytest.raises(ValueError, match="strategy_id"):
        webhook.handle_webhook(payload, output_root=tmp_path)


def test_minimal_payload_writes_signal_artifact(tmp_path: Path) -> None:
    webhook = _load_webhook_module()
    payload = _minimal_payload()
    artifact_path = webhook.handle_webhook(payload, output_root=tmp_path)

    assert artifact_path.exists()
    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert written["strategy_id"] == payload["strategy_id"]
    assert written["event_type"] == payload["event_type"]
    assert written["symbol"] == payload["symbol"]
    assert written["tv_timestamp"] == payload["tv_timestamp"]
