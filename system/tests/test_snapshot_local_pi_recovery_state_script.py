from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "snapshot_local_pi_recovery_state.py"
)


def test_snapshot_local_pi_recovery_state_writes_checkpoint(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    state_path = workspace_root / "output" / "state" / "spot_paper_state.json"
    ack_path = workspace_root / "output" / "state" / "paper_consecutive_loss_ack.json"
    checksum_path = workspace_root / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"
    checkpoint_dir = workspace_root / "output" / "review" / "local_pi_recovery_checkpoints"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 8}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ack_path.write_text(
        json.dumps({"active": True, "uses_remaining": 1, "streak_snapshot": 8}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    checksum_path.write_text(
        json.dumps({"sha256": "deadbeef"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace-system-root",
            str(workspace_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--note",
            "before-real-write",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["lock_acquired"] is True
    manifest_path = Path(str(payload["checkpoint_manifest"]))
    checksum_path = Path(str(payload["checkpoint_checksum"]))
    assert manifest_path.exists()
    assert checksum_path.exists()
    files = payload["files"]
    assert len(files) == 3
    assert any(Path(str(entry["checkpoint_path"])).exists() for entry in files if entry["present"])


def test_snapshot_local_pi_recovery_state_dry_run_does_not_write(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    state_path = workspace_root / "output" / "state" / "spot_paper_state.json"
    checkpoint_dir = workspace_root / "output" / "review" / "local_pi_recovery_checkpoints"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"date": "2026-03-08", "consecutive_losses": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace-system-root",
            str(workspace_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
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
    assert not Path(str(payload["checkpoint_root"])).exists()
