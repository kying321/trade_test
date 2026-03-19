from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "build_backup_web_intel.py"
    spec = importlib.util.spec_from_file_location("build_backup_web_intel", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_backup_web_intel_writes_state_and_checksum(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "generated_at": "2026-03-08T08:00:00Z",
                "source_root": "https://tiny-knave-962.notion.site/f36fa327d86b4b7c8fd4a5e68b48eb60",
                "selected_primary_page": {"title": "期货市场情报简报", "url": "https://example.test/page"},
                "fallback_use_allowed": True,
                "fallback_trade_authority": "advisory_only",
                "candidate_biases": [
                    {"symbol": "XAUUSD", "bias": "long_bias", "thesis_type": "macro_regime", "ticket_ready": False},
                    {"symbol": "BTCUSDT", "bias": "invalid_bias"},
                ],
                "no_trade_list": [{"symbol": "CL=F", "reason": "policy_intervention_risk"}],
                "risk_flags": [
                    {"code": "macro_conflict_high", "severity": "high", "message": "conflict"},
                    {"code": "ignored", "severity": "weird"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_backup_web_intel.py",
            "--input-json",
            str(input_path),
            "--output-root",
            str(output_root),
            "--ttl-seconds",
            "3600",
        ],
    )
    rc = mod.main()
    assert rc == 0

    state_path = output_root / "state" / "backup_web_intel.json"
    checksum_path = output_root / "state" / "backup_web_intel_checksum.json"
    assert state_path.exists()
    assert checksum_path.exists()

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["fallback_trade_authority"] == "risk_only"
    assert payload["input_fallback_trade_authority"] == "advisory_only"
    assert payload["authority_normalized"] is True
    assert payload["summary"]["candidate_bias_count"] == 1
    assert payload["summary"]["risk_flag_count"] == 1
    assert payload["artifact"] == str(state_path)
    assert payload["checksum"] == str(checksum_path)


def test_build_backup_web_intel_does_not_overwrite_last_good_state_on_invalid_input(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    state_path = output_root / "state" / "backup_web_intel.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"status": "ok", "generated_at": "2099-01-01T00:00:00Z"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    before = state_path.read_text(encoding="utf-8")

    invalid_input = tmp_path / "invalid.json"
    invalid_input.write_text("[1,2,3]\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_backup_web_intel.py",
            "--input-json",
            str(invalid_input),
            "--output-root",
            str(output_root),
        ],
    )
    rc = mod.main()
    assert rc == 2
    assert state_path.read_text(encoding="utf-8") == before
