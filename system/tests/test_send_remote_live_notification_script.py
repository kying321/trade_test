from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "send_remote_live_notification.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dry_run_payload() -> dict[str, object]:
    return {
        "action": "build_remote_live_notification_dry_run",
        "ok": True,
        "status": "ok",
        "handoff_state": "ops_live_gate_blocked",
        "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
        "next_focus_area": "gate",
        "focus_stack_brief": "gate -> risk_guard",
        "runtime_floor_brief": "addrfam=AF_UNIX; clockdev=rtc-read-only",
        "telegram": {
            "ok": True,
            "request": {
                "method": "POST",
                "url": "https://api.telegram.org/bot<TOKEN>/sendMessage",
                "headers": {"Content-Type": "application/json"},
                "json_body": {
                    "chat_id": "<CHAT_ID>",
                    "text": "telegram body",
                    "parse_mode": "MarkdownV2",
                    "disable_web_page_preview": True,
                },
            },
        },
        "feishu": {
            "ok": True,
            "request": {
                "method": "POST",
                "url": "https://open.feishu.cn/open-apis/bot/v2/hook/<TOKEN>",
                "headers": {"Content-Type": "application/json"},
                "json_body": {
                    "msg_type": "text",
                    "content": {"text": "feishu body"},
                },
            },
        },
    }


def test_send_remote_live_notification_delivery_none_builds_artifact(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    state_dir = tmp_path / "state"
    dry_run_path = tmp_path / "dry_run.json"
    _write_json(dry_run_path, _dry_run_payload())

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--state-dir",
            str(state_dir),
            "--dry-run-file",
            str(dry_run_path),
            "--delivery",
            "none",
            "--now",
            "2026-03-09T04:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "delivery_none"
    assert payload["artifact_status_label"] == "notification-send-ok"
    assert payload["focus_stack_brief"] == "gate -> risk_guard"
    assert payload["runtime_floor_brief"] == "addrfam=AF_UNIX; clockdev=rtc-read-only"
    assert payload["delivery_readiness_label"] == "delivery-none"
    assert payload["delivery_capabilities"]["telegram_configured"] is False
    assert payload["delivery_capabilities"]["feishu_configured"] is False
    assert payload["delivery_capabilities"]["available_channels"] == []
    assert payload["channels_attempted"] == []
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_send_remote_live_notification_reports_missing_credentials(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    state_dir = tmp_path / "state"
    dry_run_path = tmp_path / "dry_run.json"
    _write_json(dry_run_path, _dry_run_payload())

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--state-dir",
            str(state_dir),
            "--dry-run-file",
            str(dry_run_path),
            "--delivery",
            "telegram",
            "--now",
            "2026-03-09T04:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "delivery_blocked"
    assert payload["delivery_readiness_label"] == "telegram-blocked"
    assert payload["delivery_capabilities"]["telegram_configured"] is False
    assert payload["delivery_capabilities"]["feishu_configured"] is False
    assert payload["telegram"]["selected"] is True
    assert payload["telegram"]["sent"] is False
    assert payload["telegram"]["reasons"] == [
        "telegram_token_missing",
        "telegram_chat_id_missing",
    ]
