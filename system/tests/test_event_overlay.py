from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from lie_engine.orchestration.event_overlay import load_event_live_guard_overlay


def _assert_fail_closed(payload: dict[str, object]) -> None:
    assert payload["risk_multiplier_override"] == 0.0
    assert payload["canary_freeze"] is True
    assert "event_overlay_fail_closed" in payload["override_reason_codes"]


def test_load_event_live_guard_overlay_clamps_and_parses(tmp_path: Path) -> None:
    target = tmp_path / "event_live_guard_overlay.json"
    target.write_text(
        "{\n"
        "  \"risk_multiplier_override\": 1.2,\n"
        "  \"canary_freeze\": \"true\",\n"
        "  \"override_reason_codes\": [\"sector_watch\"]\n"
        "}\n",
        encoding="utf-8",
    )

    payload = load_event_live_guard_overlay(target)

    assert payload["risk_multiplier_override"] == 1.0
    assert payload["canary_freeze"] is True
    assert payload["override_reason_codes"] == ["sector_watch"]


def test_load_event_live_guard_overlay_missing_file_fail_closed(tmp_path: Path) -> None:
    payload = load_event_live_guard_overlay(tmp_path / "missing.json")
    _assert_fail_closed(payload)


def test_load_event_live_guard_overlay_invalid_json_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "event_live_guard_overlay.json"
    target.write_text("{not json}", encoding="utf-8")
    payload = load_event_live_guard_overlay(target)
    _assert_fail_closed(payload)


def test_load_event_live_guard_overlay_non_dict_payload_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "event_live_guard_overlay.json"
    target.write_text("[1, 2, 3]", encoding="utf-8")
    payload = load_event_live_guard_overlay(target)
    _assert_fail_closed(payload)


def test_load_event_live_guard_overlay_expired_payload_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "event_live_guard_overlay.json"
    expired = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    target.write_text(
        "{\n"
        f"  \"valid_until\": \"{expired}\"\n"
        "}\n",
        encoding="utf-8",
    )
    payload = load_event_live_guard_overlay(target)
    _assert_fail_closed(payload)
