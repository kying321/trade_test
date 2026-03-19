from __future__ import annotations

import json
import subprocess
from pathlib import Path


SNAPSHOT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "snapshot_local_pi_recovery_state.py"
)
RESTORE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "restore_local_pi_recovery_state.py"
)


def test_restore_local_pi_recovery_state_dry_run_reports_plan(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    checkpoint_dir = workspace_root / "output" / "review" / "local_pi_recovery_checkpoints"
    state_path = workspace_root / "output" / "state" / "spot_paper_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 8}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    snap = subprocess.run(
        [
            "python3",
            str(SNAPSHOT_SCRIPT),
            "--workspace-system-root",
            str(workspace_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert snap.returncode == 0, snap.stderr
    checkpoint_manifest = json.loads(snap.stdout)["checkpoint_manifest"]

    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(RESTORE_SCRIPT),
            "--workspace-system-root",
            str(workspace_root),
            "--checkpoint",
            str(checkpoint_manifest),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "dry_run"
    assert payload["restore_plan"][0]["action"] == "restore_file"
    current = json.loads(state_path.read_text(encoding="utf-8"))
    assert current["consecutive_losses"] == 1


def test_restore_local_pi_recovery_state_write_restores_snapshot_and_removes_missing_ack(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    checkpoint_dir = workspace_root / "output" / "review" / "local_pi_recovery_checkpoints"
    state_path = workspace_root / "output" / "state" / "spot_paper_state.json"
    ack_path = workspace_root / "output" / "state" / "paper_consecutive_loss_ack.json"
    checksum_path = workspace_root / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 8}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    snap = subprocess.run(
        [
            "python3",
            str(SNAPSHOT_SCRIPT),
            "--workspace-system-root",
            str(workspace_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert snap.returncode == 0, snap.stderr
    snap_payload = json.loads(snap.stdout)
    checkpoint_manifest = snap_payload["checkpoint_manifest"]
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ack_path.write_text(json.dumps({"active": True}, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum_path.write_text(json.dumps({"sha256": "deadbeef"}, ensure_ascii=False) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(RESTORE_SCRIPT),
            "--workspace-system-root",
            str(workspace_root),
            "--checkpoint",
            str(checkpoint_manifest),
            "--write",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    restored = json.loads(state_path.read_text(encoding="utf-8"))
    assert restored["consecutive_losses"] == 8
    assert not ack_path.exists()
    assert not checksum_path.exists()
    assert payload["pre_restore_backup_dir"]

    mismatch_proc = subprocess.run(
        [
            "python3",
            str(RESTORE_SCRIPT),
            "--workspace-system-root",
            str(workspace_root),
            "--checkpoint",
            str(checkpoint_manifest),
            "--expected-current-state-fingerprint",
            "deadbeef",
            "--write",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert mismatch_proc.returncode == 5, mismatch_proc.stderr
    mismatch_payload = json.loads(mismatch_proc.stdout)
    assert mismatch_payload["ok"] is False
    assert mismatch_payload["status"] == "state_fingerprint_mismatch"
