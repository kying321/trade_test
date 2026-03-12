from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "paper_consecutive_loss_ack_archive_status.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_status_reports_empty_state(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--ack-path",
            str(tmp_path / "paper_consecutive_loss_ack.json"),
            "--checksum-path",
            str(tmp_path / "paper_consecutive_loss_ack_checksum.json"),
            "--archive-dir",
            str(tmp_path / "archive"),
            "--archive-manifest-path",
            str(tmp_path / "archive" / "manifest.jsonl"),
            "--now",
            "2026-03-08T12:00:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["live_ack"]["present"] is False
    assert payload["archive"]["present"] is False
    assert payload["archive"]["file_count"] == 0
    assert payload["archive"]["manifest_present"] is False
    assert payload["archive"]["latest_archive"] is None


def test_archive_status_reports_live_ack_checksum_valid(tmp_path: Path) -> None:
    ack_path = tmp_path / "paper_consecutive_loss_ack.json"
    checksum_path = tmp_path / "paper_consecutive_loss_ack_checksum.json"
    ack_payload = {
        "generated_at": "2026-03-08T00:00:00+00:00",
        "expires_at": "2026-03-09T00:00:00+00:00",
        "active": True,
        "uses_remaining": 1,
        "use_limit": 1,
        "streak_snapshot": 4,
        "last_loss_ts": "2026-03-07T00:00:00+00:00",
    }
    ack_path.write_text(json.dumps(ack_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-08T00:00:00+00:00",
                "artifact": str(ack_path),
                "sha256": _sha256(ack_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--ack-path",
            str(ack_path),
            "--checksum-path",
            str(checksum_path),
            "--archive-dir",
            str(tmp_path / "archive"),
            "--archive-manifest-path",
            str(tmp_path / "archive" / "manifest.jsonl"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["live_ack"]["present"] is True
    assert payload["live_ack"]["checksum_present"] is True
    assert payload["live_ack"]["checksum_valid"] is True
    assert payload["live_ack"]["payload"]["active"] is True
    assert payload["live_ack"]["payload"]["streak_snapshot"] == 4


def test_archive_status_reports_latest_archive_and_manifest_tail(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True)
    manifest_path = archive_dir / "manifest.jsonl"
    archive_path = archive_dir / "paper_consecutive_loss_ack_20260308T120000_000000Z.json"
    archive_payload = {
        "generated_at": "2026-03-08T00:00:00+00:00",
        "expires_at": "2026-03-09T00:00:00+00:00",
        "active": False,
        "uses_remaining": 0,
        "use_limit": 1,
        "streak_snapshot": 4,
        "last_loss_ts": "2026-03-07T00:00:00+00:00",
        "consumed_at": "2026-03-08T12:00:00+00:00",
        "consume_reason": "single_use_consumed",
        "archived_at": "2026-03-08T12:00:01+00:00",
        "archive_reason": "consumed",
    }
    archive_path.write_text(
        json.dumps(archive_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_path = archive_path.with_suffix(".json.sha256")
    checksum_path.write_text(f"{_sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts_utc": "2026-03-08T12:00:01+00:00",
                        "event": "consumed_archive",
                        "path": str(archive_path),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "ts_utc": "2026-03-08T12:01:00+00:00",
                        "event": "purged",
                        "path": str(archive_dir / "older.json"),
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--ack-path",
            str(tmp_path / "paper_consecutive_loss_ack.json"),
            "--checksum-path",
            str(tmp_path / "paper_consecutive_loss_ack_checksum.json"),
            "--archive-dir",
            str(archive_dir),
            "--archive-manifest-path",
            str(manifest_path),
            "--manifest-tail",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    latest_archive = payload["archive"]["latest_archive"]
    assert payload["archive"]["present"] is True
    assert payload["archive"]["file_count"] == 1
    assert payload["archive"]["checksum_count"] == 1
    assert latest_archive["checksum_valid"] is True
    assert latest_archive["payload"]["archive_reason"] == "consumed"
    assert latest_archive["payload"]["consumed_at"] == "2026-03-08T12:00:00+00:00"
    assert payload["archive"]["manifest_present"] is True
    assert payload["archive"]["manifest_line_count"] == 2
    assert payload["archive"]["manifest_tail_count"] == 2
    assert payload["archive"]["consumed_archive_events_in_tail"] == 1
    assert payload["archive"]["purged_events_in_tail"] == 1
