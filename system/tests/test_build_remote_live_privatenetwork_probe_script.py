from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_live_privatenetwork_probe.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_live_privatenetwork_probe_reads_probe_payload(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    probe_path = tmp_path / "probe.json"
    _write_json(
        probe_path,
        {
            "action": "live-risk-daemon-privatenetwork-probe",
            "ok": True,
            "status": "compatible",
            "unit": "fenlie-live-risk-daemon-privatenetwork-probe-123",
            "returncode": 0,
            "properties": {"PrivateNetwork": True},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--probe-file",
            str(probe_path),
            "--now",
            "2026-03-09T05:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["artifact_status_label"] == "privatenetwork-compatible"
    assert payload["probe"]["status"] == "compatible"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_remote_live_privatenetwork_probe_reports_probe_failure(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    probe_path = tmp_path / "probe.json"
    _write_json(
        probe_path,
        {
            "action": "live-risk-daemon-privatenetwork-probe",
            "ok": False,
            "status": "incompatible",
            "returncode": 1,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--probe-file",
            str(probe_path),
            "--now",
            "2026-03-09T05:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "incompatible"
    assert payload["artifact_status_label"] == "incompatible"
