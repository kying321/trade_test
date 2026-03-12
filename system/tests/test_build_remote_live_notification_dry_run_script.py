from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_live_notification_dry_run.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_live_notification_dry_run_reads_preview(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    preview_path = tmp_path / "preview.json"
    _write_json(
        preview_path,
        {
            "action": "build_remote_live_notification_preview",
            "handoff_state": "ops_live_gate_blocked",
            "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "next_focus_area": "gate",
            "focus_stack_brief": "gate -> risk_guard",
            "runtime_floor_brief": "addrfam=AF_UNIX; clockdev=rtc-read-only",
            "notification_templates": {
                "telegram": {
                    "parse_mode": "MarkdownV2",
                    "text": "telegram body",
                    "disable_web_page_preview": True,
                },
                "feishu": {
                    "msg_type": "text",
                    "content": {"text": "feishu body"},
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
            "--preview-file",
            str(preview_path),
            "--now",
            "2026-03-09T04:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["focus_stack_brief"] == "gate -> risk_guard"
    assert payload["runtime_floor_brief"] == "addrfam=AF_UNIX; clockdev=rtc-read-only"
    assert payload["telegram"]["ok"] is True
    assert payload["telegram"]["request"]["url"] == "https://api.telegram.org/bot<TOKEN>/sendMessage"
    assert payload["feishu"]["ok"] is True
    assert payload["feishu"]["request"]["url"] == "https://open.feishu.cn/open-apis/bot/v2/hook/<TOKEN>"
    assert payload["artifact_label"] == "remote-live-notification-dry-run:ops_live_gate_blocked"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_remote_live_notification_dry_run_reports_template_validation_failure(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    preview_path = tmp_path / "preview.json"
    _write_json(
        preview_path,
        {
            "action": "build_remote_live_notification_preview",
            "handoff_state": "ops_live_gate_blocked",
            "notification_templates": {
                "telegram": {
                    "parse_mode": "HTML",
                    "text": "",
                },
                "feishu": {
                    "msg_type": "post",
                    "content": {},
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
            "--preview-file",
            str(preview_path),
            "--now",
            "2026-03-09T04:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "template_validation_failed"
    assert payload["telegram"]["reasons"] == [
        "telegram_parse_mode_invalid",
        "telegram_text_missing",
    ]
    assert payload["feishu"]["reasons"] == [
        "feishu_msg_type_invalid",
        "feishu_text_missing",
    ]
