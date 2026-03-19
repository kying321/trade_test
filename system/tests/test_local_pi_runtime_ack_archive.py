from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


RUNTIME_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "pi"
    / "scripts"
    / "lie_spot_halfhour_core.py"
)


def _load_runtime_module():
    runtime_dir = str(RUNTIME_SCRIPT_PATH.parent)
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    module_name = "test_lie_spot_halfhour_core"
    spec = importlib.util.spec_from_file_location(module_name, RUNTIME_SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_consume_consecutive_loss_ack_archives_and_clears_live_ack(tmp_path: Path) -> None:
    mod = _load_runtime_module()
    mod.PAPER_CONSECUTIVE_LOSS_ACK_PATH = tmp_path / "paper_consecutive_loss_ack.json"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH = tmp_path / "paper_consecutive_loss_ack_checksum.json"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR = tmp_path / "archive"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH = mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR / "manifest.jsonl"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES = 3

    payload = {
        "generated_at": "2026-03-08T00:00:00+00:00",
        "expires_at": "2026-03-09T00:00:00+00:00",
        "guardrail": "consecutive_loss_stop",
        "use_limit": 1,
        "uses_remaining": 1,
        "active": True,
        "streak_snapshot": 4,
        "cooldown_hours_required": 12.0,
        "last_loss_ts": "2026-03-07T00:00:00+00:00",
        "allow_missing_last_loss_ts": False,
        "note": "manual-check",
    }
    assert mod._write_consecutive_loss_ack_payload(payload, now_dt=mod.dt.datetime(2026, 3, 8, tzinfo=mod.dt.timezone.utc))

    ack_state = {"applied": True}
    updated = mod.consume_consecutive_loss_ack(
        ack_state=ack_state,
        cycle_ts="2026-03-08T12:00:00+00:00",
    )

    assert updated["consume_ok"] is True
    assert updated["archive_attempted"] is True
    assert updated["archive_ok"] is True
    assert updated["consume_reason"] == "single_use_consumed_archived_and_cleared"
    assert updated["uses_remaining_after_consume"] == 0
    assert updated["active_after_consume"] is False
    assert updated["live_ack_cleared"] is True
    assert updated["live_checksum_cleared"] is True
    assert not mod.PAPER_CONSECUTIVE_LOSS_ACK_PATH.exists()
    assert not mod.PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH.exists()

    archive_path = Path(str(updated["archive_path"]))
    checksum_path = Path(str(updated["archive_checksum_path"]))
    assert archive_path.exists()
    assert checksum_path.exists()
    archived = json.loads(archive_path.read_text(encoding="utf-8"))
    assert archived["active"] is False
    assert archived["uses_remaining"] == 0
    assert archived["consume_reason"] == "single_use_consumed"
    assert archived["archived_at"]
    assert archived["archive_reason"] == "consumed"

    manifest_lines = mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    manifest_entry = json.loads(manifest_lines[0])
    assert manifest_entry["event"] == "consumed_archive"


def test_consecutive_loss_ack_archive_prunes_to_keep_limit(tmp_path: Path) -> None:
    mod = _load_runtime_module()
    mod.PAPER_CONSECUTIVE_LOSS_ACK_PATH = tmp_path / "paper_consecutive_loss_ack.json"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH = tmp_path / "paper_consecutive_loss_ack_checksum.json"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR = tmp_path / "archive"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH = mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR / "manifest.jsonl"
    mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES = 1

    base_payload = {
        "generated_at": "2026-03-08T00:00:00+00:00",
        "expires_at": "2026-03-09T00:00:00+00:00",
        "guardrail": "consecutive_loss_stop",
        "use_limit": 1,
        "uses_remaining": 0,
        "active": False,
        "streak_snapshot": 4,
        "cooldown_hours_required": 12.0,
        "last_loss_ts": "2026-03-07T00:00:00+00:00",
        "allow_missing_last_loss_ts": False,
        "note": "manual-check",
        "consumed_at": "2026-03-08T12:00:00+00:00",
        "consume_reason": "single_use_consumed",
    }
    for idx in range(3):
        archive_result = mod._archive_and_clear_consecutive_loss_ack_payload(
            dict(base_payload),
            now_dt=mod.dt.datetime(2026, 3, 8, 12, 0, idx, tzinfo=mod.dt.timezone.utc),
            cycle_ts=f"2026-03-08T12:00:0{idx}+00:00",
        )
        assert archive_result["archive_ok"] is True

    archives = sorted(mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR.glob("paper_consecutive_loss_ack_*.json"))
    checksums = sorted(mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR.glob("paper_consecutive_loss_ack_*.json.sha256"))
    assert len(archives) == 1
    assert len(checksums) == 1

    manifest_lines = mod.PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in manifest_lines]
    assert events.count("consumed_archive") == 3
    assert events.count("purged") == 2
