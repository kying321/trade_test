from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_live_notification_preview.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_live_notification_preview_reads_handoff(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    handoff_path = tmp_path / "handoff.json"
    _write_json(
        handoff_path,
        {
            "action": "build_remote_live_handoff",
            "operator_handoff": {
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
                "next_focus_area": "gate",
                "next_focus_reason": "ops_live_gate_blocked",
                "focus_stack_brief": "gate -> risk_guard",
                "operator_notification": {
                    "level": "warning",
                    "title": "Remote gate is blocking live execution",
                    "body": "runtime-ok / gate-blocked / risk-guard-blocked; focus=gate",
                    "command": "cmd-one",
                    "tags": ["state:ops_live_gate_blocked"],
                    "focus_stack_brief": "gate -> risk_guard",
                    "runtime_floor_brief": "addrfam=AF_UNIX; clockdev=rtc-read-only",
                    "plain_text": "plain",
                    "markdown": "markdown",
                },
                "operator_notification_templates": {
                    "telegram": {"parse_mode": "MarkdownV2", "text": "text"},
                    "feishu": {"msg_type": "text", "content": {"text": "plain"}},
                    "generic": {"title": "Remote gate is blocking live execution"},
                },
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--handoff-file",
            str(handoff_path),
            "--now",
            "2026-03-09T03:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["handoff_state"] == "ops_live_gate_blocked"
    assert payload["operator_status_triplet"] == "runtime-ok / gate-blocked / risk-guard-blocked"
    assert payload["focus_stack_brief"] == "gate -> risk_guard"
    assert payload["runtime_floor_brief"] == "addrfam=AF_UNIX; clockdev=rtc-read-only"
    assert payload["notification"]["title"] == "Remote gate is blocking live execution"
    assert payload["notification_templates"]["telegram"]["parse_mode"] == "MarkdownV2"
    assert payload["artifact_label"] == "remote-live-notification-preview:ops_live_gate_blocked"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_remote_live_notification_preview_reports_missing_templates(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    handoff_path = tmp_path / "handoff.json"
    _write_json(
        handoff_path,
        {
            "action": "build_remote_live_handoff",
            "operator_handoff": {
                "handoff_state": "ops_live_gate_blocked",
                "operator_notification": {
                    "level": "warning",
                    "title": "Remote gate is blocking live execution",
                },
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--handoff-file",
            str(handoff_path),
            "--now",
            "2026-03-09T03:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "notification_templates_missing"
    assert payload["notification_templates"] is None
