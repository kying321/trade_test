from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_live_handoff.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_live_handoff_reports_ready_for_canary(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    journal_path = tmp_path / "journal.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": True,
            "reasons": [],
            "quote_available": 20.0,
            "ops_live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        journal_path,
        {
            "action": "live-risk-daemon-journal",
            "lines": ["line1", "line2", "line3"],
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {
                "returncode": 0,
                "overall_exposure": 3.2,
                "overall_rating": "OK",
                "findings": ["✗ ProtectHome="],
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--journal-file",
            str(journal_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["operator_handoff"]["handoff_state"] == "ready_for_canary"
    assert payload["operator_handoff"]["daemon_running"] is True
    assert payload["operator_handoff"]["ops_status"] == "passed"
    assert payload["operator_handoff"]["security_verify_ok"] is True
    assert payload["operator_handoff"]["security_exposure_score"] == 3.2
    assert payload["operator_handoff"]["security_acceptance_status"] == "accepted"
    assert payload["operator_handoff"]["security_acceptance_reasons"] == []
    assert payload["operator_handoff"]["security_acceptance_max_allowed_exposure"] == 4.0
    assert payload["operator_handoff"]["runtime_status_label"] == "runtime-ok"
    assert payload["operator_handoff"]["runtime_attention_reasons"] == []
    assert payload["operator_handoff"]["gate_status_label"] == "gate-ok"
    assert payload["operator_handoff"]["gate_attention_reasons"] == []
    assert payload["operator_handoff"]["risk_guard_status_label"] == "risk-guard-ok"
    assert payload["operator_handoff"]["risk_guard_attention_reasons"] == []
    assert (
        payload["operator_handoff"]["operator_status_triplet"]
        == "runtime-ok / gate-ok / risk-guard-ok"
    )
    assert (
        payload["operator_handoff"]["operator_status_quad"]
        == "runtime-ok / gate-ok / risk-guard-ok / notify-unknown"
    )
    assert (
        payload["operator_handoff"]["operator_status_summary"]
        == "runtime=runtime-ok; gate=gate-ok; risk_guard=risk-guard-ok"
    )
    assert payload["operator_handoff"]["next_focus_area"] == "canary"
    assert payload["operator_handoff"]["next_focus_reason"] == "ready_for_canary"
    assert payload["operator_handoff"]["focus_stack"] == [
        {"area": "canary", "reason": "ready_for_canary"},
        {"area": "notify", "reason": "notification_status_missing"},
    ]
    assert payload["operator_handoff"]["focus_stack_brief"] == "canary -> notify"
    assert (
        payload["operator_handoff"]["focus_stack_summary"]
        == "canary(ready_for_canary) -> notify(notification_status_missing)"
    )
    assert payload["operator_handoff"]["next_focus_command"].endswith(
        "scripts/openclaw_cloud_bridge.sh live-takeover-probe"
    )
    assert payload["operator_handoff"]["next_focus_commands"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-canary",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
    ]
    assert payload["operator_handoff"]["operator_playbook"]["active_focus"] == "canary"
    assert payload["operator_handoff"]["operator_playbook"]["active_reason"] == "ready_for_canary"
    assert payload["operator_handoff"]["operator_playbook"]["primary_sequence"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-canary",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
    ]
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["runtime"]["status_label"]
        == "runtime-ok"
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["runtime"]["address_family_floor"]
        == "unknown"
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["gate"]["status_label"]
        == "gate-ok"
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["risk_guard"]["status_label"]
        == "risk-guard-ok"
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["canary"]["status_label"]
        == "canary-ready"
    )
    assert "## Active Focus" in payload["operator_handoff"]["operator_playbook_md"]
    assert "- focus: `canary`" in payload["operator_handoff"]["operator_playbook_md"]
    assert "address-family floor: `unknown`" in payload["operator_handoff"]["operator_playbook_md"]
    assert "1. `cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe`" in payload["operator_handoff"]["operator_playbook_md"]
    assert "# Remote Live Handoff" in payload["operator_handoff"]["operator_handoff_md"]
    assert "- state: `ready_for_canary`" in payload["operator_handoff"]["operator_handoff_md"]
    assert "- status: `runtime-ok / gate-ok / risk-guard-ok`" in payload["operator_handoff"]["operator_handoff_md"]
    assert "- focus-stack: `canary -> notify`" in payload["operator_handoff"]["operator_handoff_md"]
    assert "- next focus: `canary`" in payload["operator_handoff"]["operator_handoff_md"]
    assert payload["operator_handoff"]["secondary_focus_area"] == "notify"
    assert payload["operator_handoff"]["secondary_focus_reason"] == "notification_status_missing"
    assert payload["operator_handoff"]["secondary_focus_command"].endswith(
        "scripts/openclaw_cloud_bridge.sh remote-live-notification-send"
    )
    assert payload["operator_handoff"]["remote_live_diagnosis"]["status"] == "formal_live_possible"
    assert payload["operator_handoff"]["remote_live_diagnosis"]["brief"] == "formal_live_possible:unknown"
    assert payload["operator_handoff"]["operator_handoff_brief"] == "\n".join(
        [
            "state: ready_for_canary",
            "status: runtime-ok / gate-ok / risk-guard-ok",
            "status4: runtime-ok / gate-ok / risk-guard-ok / notify-unknown",
            "focus-stack: canary -> notify",
            "addrfam: unknown",
            "history-diagnosis: formal_live_possible:unknown",
            "focus: canary",
            "reason: ready_for_canary",
            "cmd: cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
            "focus2: notify",
            "reason2: notification_status_missing",
        ]
    )
    assert payload["operator_handoff"]["operator_notification"] == {
        "level": "info",
        "title": "Remote live ready for canary",
        "body": "runtime-ok / gate-ok / risk-guard-ok; focus=canary; reason=ready_for_canary; addrfam=unknown; focusstack=canary -> notify; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
        "command": "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
        "tags": [
            "state:ready_for_canary",
            "focus:canary",
            "security:security-ok",
        ],
        "focus_stack_brief": "canary -> notify",
        "runtime_floor_brief": "",
        "plain_text": "[INFO] Remote live ready for canary\nruntime-ok / gate-ok / risk-guard-ok; focus=canary; reason=ready_for_canary; addrfam=unknown; focusstack=canary -> notify; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
        "markdown": "\n".join(
            [
                "**Remote live ready for canary**",
                "- level: `info`",
                "- state: `ready_for_canary`",
                "- status: `runtime-ok / gate-ok / risk-guard-ok`",
                "- focus-stack: `canary -> notify`",
                "- focus: `canary`",
                "- reason: `ready_for_canary`",
                "- command: `cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe`",
            ]
        ),
    }
    assert payload["operator_handoff"]["operator_notification_templates"] == {
        "telegram": {
            "parse_mode": "MarkdownV2",
            "text": "*Remote live ready for canary*\nlevel: `info`\nruntime\\-ok / gate\\-ok / risk\\-guard\\-ok; focus\\=canary; reason\\=ready\\_for\\_canary; addrfam\\=unknown; focusstack\\=canary \\-\\> notify; cmd\\=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw\\_cloud\\_bridge\\.sh live\\-takeover\\-probe\ncmd: `cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw\\_cloud\\_bridge\\.sh live\\-takeover\\-probe`\nfocus-stack: `canary \\-\\> notify`",
            "disable_web_page_preview": True,
        },
        "feishu": {
            "msg_type": "text",
            "content": {
                "text": "[INFO] Remote live ready for canary\nruntime-ok / gate-ok / risk-guard-ok; focus=canary; reason=ready_for_canary; addrfam=unknown; focusstack=canary -> notify; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe\nfocus-stack: canary -> notify"
            },
        },
        "generic": {
            "level": "info",
            "title": "Remote live ready for canary",
            "body": "runtime-ok / gate-ok / risk-guard-ok; focus=canary; reason=ready_for_canary; addrfam=unknown; focusstack=canary -> notify; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
            "command": "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-probe",
            "tags": [
                "state:ready_for_canary",
                "focus:canary",
                "security:security-ok",
            ],
            "focus_stack_brief": "canary -> notify",
            "runtime_floor_brief": "",
        },
    }


def test_build_remote_live_handoff_surfaces_remote_live_history_audit(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    journal_path = tmp_path / "journal.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["risk_guard_blocked"],
            "risk_guard": {"allowed": False, "status": "blocked", "reasons": ["ticket_missing:no_actionable_ticket"]},
            "ops_live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(journal_path, {"action": "live-risk-daemon-journal", "lines": ["line1"]})
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {"returncode": 0, "overall_exposure": 3.2, "overall_rating": "OK", "findings": []},
        },
    )
    _write_json(
        review_dir / "20260313T012000Z_remote_live_history_audit.json",
        {
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "market": "portfolio_margin_um",
            "status": "ok",
            "window_summaries": [
                {
                    "window_hours": 24,
                    "history_window_label": "24h",
                    "quote_available": -0.87,
                    "open_positions": 1,
                    "closed_pnl": 14.82,
                    "trade_count": 20,
                    "risk_guard_status": "blocked",
                    "risk_guard_reasons": ["ticket_missing:no_actionable_ticket", "open_exposure_above_cap"],
                    "blocked_candidate": {"symbol": "BNBUSDT"},
                },
                {
                    "window_hours": 720,
                    "history_window_label": "30d",
                    "closed_pnl": 18.79,
                    "trade_count": 38,
                    "income_pnl_by_symbol": {"BTCUSDT": 15.17, "ETHUSDT": 10.18},
                    "income_pnl_by_day": {"2026-03-12": 14.82},
                },
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--journal-file",
            str(journal_path),
            "--security-status-file",
            str(security_path),
            "--now",
            "2026-03-13T01:30:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    history = payload["operator_handoff"]["remote_live_history"]
    assert history["status"] == "ok"
    assert history["market"] == "portfolio_margin_um"
    assert history["window_brief"] == "24h:14.82pnl/20tr/1open | 30d:18.79pnl/38tr/0open"
    assert history["risk_guard_reasons"] == [
        "ticket_missing:no_actionable_ticket",
        "open_exposure_above_cap",
    ]
    assert history["blocked_candidate_symbol"] == "BNBUSDT"
    assert history["symbol_pnl_brief"] == "BTCUSDT:15.17, ETHUSDT:10.18"
    diagnosis = payload["operator_handoff"]["remote_live_diagnosis"]
    assert diagnosis["status"] == "profitability_confirmed_but_auto_live_blocked"
    assert diagnosis["brief"] == (
        "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:risk_guard"
    )
    assert diagnosis["profitability_confirmed"] is True
    assert diagnosis["profitability_window"] == "30d"
    assert diagnosis["blocking_layers"] == ["risk_guard"]
    assert "portfolio_margin_um remote history confirms realized profitability" in diagnosis[
        "blocker_detail"
    ]
    assert "history: 24h:14.82pnl/20tr/1open | 30d:18.79pnl/38tr/0open" in payload["operator_handoff"]["operator_handoff_brief"]
    assert (
        "history-diagnosis: profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:risk_guard"
        in payload["operator_handoff"]["operator_handoff_brief"]
    )
    assert "remote-live-history: `24h:14.82pnl/20tr/1open | 30d:18.79pnl/38tr/0open | risk_guard=blocked | reasons=ticket_missing:no_actionable_ticket, open_exposure_above_cap`" in payload["operator_handoff"]["operator_handoff_md"]
    assert (
        "remote-live-diagnosis: `profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:risk_guard"
        in payload["operator_handoff"]["operator_handoff_md"]
    )
    assert payload["operator_handoff"]["security_status_label"] == "security-ok"
    assert payload["operator_handoff"]["security_top_risks"] == []
    assert payload["operator_handoff"]["security_recommendations"] == []
    assert "live-takeover-ready-check" in " ".join(payload["operator_handoff"]["operator_commands"])
    assert "live-risk-daemon-security-status" in " ".join(payload["operator_handoff"]["operator_commands"])
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_remote_live_handoff_selects_portfolio_margin_scope_when_history_is_unified(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    ready_spot_path = tmp_path / "ready_spot.json"
    ready_portfolio_path = tmp_path / "ready_portfolio.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    journal_path = tmp_path / "journal.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_spot_path,
        {
            "action": "live-takeover-ready-check",
            "market": "spot",
            "ready": False,
            "reason": "insufficient_quote_balance",
            "reasons": ["insufficient_quote_balance"],
            "quote_available": 0.0,
            "required_quote": 5.0,
            "ops_live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        ready_portfolio_path,
        {
            "action": "live-takeover-ready-check",
            "market": "portfolio_margin_um",
            "ready": False,
            "reason": "portfolio_margin_um_read_only_mode",
            "reasons": ["portfolio_margin_um_read_only_mode", "risk_guard_blocked"],
            "quote_available": -0.8,
            "required_quote": 5.0,
            "risk_guard": {"allowed": False, "status": "blocked", "reasons": ["open_exposure_above_cap"]},
            "ops_live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": False,
            "status": "blocked",
            "reason_code": "ops_status_red",
            "live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
        },
    )
    _write_json(journal_path, {"action": "live-risk-daemon-journal", "lines": ["line1"]})
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {"returncode": 0, "overall_exposure": 3.2, "overall_rating": "OK", "findings": []},
        },
    )
    _write_json(
        review_dir / "20260313T012000Z_remote_live_history_audit.json",
        {
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "market": "portfolio_margin_um",
            "status": "ok",
            "window_summaries": [
                {
                    "window_hours": 24,
                    "history_window_label": "24h",
                    "quote_available": -0.87,
                    "open_positions": 1,
                    "closed_pnl": 14.82,
                    "trade_count": 20,
                    "risk_guard_status": "blocked",
                    "risk_guard_reasons": ["open_exposure_above_cap"],
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-spot-file",
            str(ready_spot_path),
            "--ready-check-spot-returncode",
            "3",
            "--ready-check-portfolio-margin-file",
            str(ready_portfolio_path),
            "--ready-check-portfolio-margin-returncode",
            "3",
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--journal-file",
            str(journal_path),
            "--security-status-file",
            str(security_path),
            "--now",
            "2026-03-13T01:30:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    operator = payload["operator_handoff"]
    assert payload["ready_check"]["market"] == "portfolio_margin_um"
    assert payload["ready_check_returncode"] == 3
    assert payload["ready_check_spot"]["market"] == "spot"
    assert payload["ready_check_portfolio_margin_um"]["market"] == "portfolio_margin_um"
    assert operator["ready_check_scope_market"] == "portfolio_margin_um"
    assert operator["ready_check_scope_source"] == "portfolio_margin_um"
    assert operator["ready_check_scope_brief"] == "portfolio_margin_um:portfolio_margin_um"
    assert operator["account_scope_alignment"]["status"] == "split_scope_spot_vs_portfolio_margin_um"
    assert operator["account_scope_alignment"]["blocking"] is False
    assert operator["remote_live_diagnosis"]["status"] == "profitability_confirmed_but_auto_live_blocked"
    assert operator["remote_live_diagnosis"]["brief"] == (
        "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
    )
    assert operator["remote_live_diagnosis"]["blocking_layers"] == [
        "ops_live_gate",
        "risk_guard",
    ]
    assert operator["ready_reasons"] == [
        "portfolio_margin_um_read_only_mode",
        "risk_guard_blocked",
    ]
    assert operator["gate_attention_reasons"] == [
        "ops_live_gate_blocked",
        "live_gate:ops_status_red",
    ]
    assert operator["handoff_state"] == "ops_live_gate_blocked"
    assert "scope: portfolio_margin_um:portfolio_margin_um" in operator["operator_handoff_brief"]
    assert "scope-align: split_scope_spot_vs_portfolio_margin_um" in operator["operator_handoff_brief"]
    assert (
        "history-diagnosis: profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
        in operator["operator_handoff_brief"]
    )
    assert "- ready-check scope: `portfolio_margin_um:portfolio_margin_um`" in operator["operator_handoff_md"]
    assert "- account-scope: `split_scope_spot_vs_portfolio_margin_um`" in operator["operator_handoff_md"]
    assert (
        "remote-live-diagnosis: `profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
        in operator["operator_handoff_md"]
    )


def test_build_remote_live_handoff_auto_discovers_persisted_review_inputs(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260313T012501Z_remote_live_ready_check_spot.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_ready_check_spot",
            "captured_at_utc": "2026-03-13T01:25:01Z",
            "returncode": 3,
            "payload": {
                "action": "live-takeover-ready-check",
                "market": "spot",
                "ready": False,
                "reason": "insufficient_quote_balance",
                "reasons": ["insufficient_quote_balance"],
                "quote_available": 0.0,
                "required_quote": 5.0,
                "ops_live_gate": {"ok": True, "blocking_reason_codes": []},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012502Z_remote_live_ready_check_portfolio_margin_um.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_ready_check_portfolio_margin_um",
            "captured_at_utc": "2026-03-13T01:25:02Z",
            "returncode": 3,
            "payload": {
                "action": "live-takeover-ready-check",
                "market": "portfolio_margin_um",
                "ready": False,
                "reason": "portfolio_margin_um_read_only_mode",
                "reasons": ["portfolio_margin_um_read_only_mode", "risk_guard_blocked"],
                "quote_available": -0.8,
                "required_quote": 5.0,
                "risk_guard": {
                    "allowed": False,
                    "status": "blocked",
                    "reasons": ["open_exposure_above_cap"],
                },
                "ops_live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012503Z_remote_live_risk_daemon_status.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_risk_daemon_status",
            "captured_at_utc": "2026-03-13T01:25:03Z",
            "returncode": 0,
            "payload": {
                "action": "live-risk-daemon-status",
                "mode": "systemd",
                "systemd": {"active_state": "active", "unit_file_state": "enabled"},
                "payload": {"running": True, "status": "running"},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012504Z_remote_live_ops_reconcile_status.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_ops_reconcile_status",
            "captured_at_utc": "2026-03-13T01:25:04Z",
            "returncode": 0,
            "payload": {
                "action": "live-ops-reconcile-status",
                "ok": False,
                "status": "blocked",
                "reason_code": "ops_status_red",
                "live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012505Z_remote_live_risk_daemon_journal.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_risk_daemon_journal",
            "captured_at_utc": "2026-03-13T01:25:05Z",
            "returncode": 0,
            "payload": {"action": "live-risk-daemon-journal", "lines": ["line1"]},
        },
    )
    _write_json(
        review_dir / "20260313T012506Z_remote_live_risk_daemon_security_status.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_risk_daemon_security_status",
            "captured_at_utc": "2026-03-13T01:25:06Z",
            "returncode": 0,
            "payload": {
                "action": "live-risk-daemon-security-status",
                "verify": {"ok": True, "returncode": 0},
                "security": {
                    "returncode": 0,
                    "overall_exposure": 3.0,
                    "overall_rating": "OK",
                    "findings": [],
                },
            },
        },
    )
    _write_json(
        review_dir / "20260313T012507Z_remote_live_bridge_context.json",
        {
            "action": "capture_remote_live_handoff_context",
            "capture_kind": "remote_live_bridge_context",
            "captured_at_utc": "2026-03-13T01:25:07Z",
            "remote_host": "43.153.148.242",
            "remote_user": "ubuntu",
            "remote_project_dir": "/home/ubuntu/openclaw-system",
            "security_accept_max_exposure": 2.5,
            "openclaw_orderflow_executor_mode": "shadow_guarded",
        },
    )
    _write_json(
        review_dir / "20260313T012000Z_remote_live_history_audit.json",
        {
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "market": "portfolio_margin_um",
            "status": "ok",
            "window_summaries": [
                {
                    "window_hours": 24,
                    "history_window_label": "24h",
                    "quote_available": -0.87,
                    "open_positions": 1,
                    "closed_pnl": 14.82,
                    "trade_count": 20,
                    "risk_guard_status": "blocked",
                    "risk_guard_reasons": ["open_exposure_above_cap"],
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-13T01:30:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    operator = payload["operator_handoff"]
    assert payload["ready_check_spot_returncode"] == 3
    assert payload["ready_check_portfolio_margin_um_returncode"] == 3
    assert payload["risk_daemon_status_returncode"] == 0
    assert payload["ops_status_returncode"] == 0
    assert payload["journal_returncode"] == 0
    assert payload["security_status_returncode"] == 0
    assert payload["ready_check"]["market"] == "portfolio_margin_um"
    assert payload["ready_check_returncode"] == 3
    assert payload["remote_host"] == "43.153.148.242"
    assert payload["remote_user"] == "ubuntu"
    assert payload["remote_project_dir"] == "/home/ubuntu/openclaw-system"
    assert payload["security_accept_max_exposure"] == 2.5
    assert payload["operator_handoff"]["execution_contract"]["status"] == "non_executable_contract"
    assert payload["operator_handoff"]["execution_contract"]["mode"] == "shadow_only"
    assert payload["operator_handoff"]["execution_contract"]["executor_mode"] == "shadow_guarded"
    assert payload["operator_handoff"]["execution_contract"]["executor_mode_source"] == "bridge_context"
    assert payload["operator_handoff"]["execution_contract"]["live_orders_allowed"] is False
    assert payload["operator_handoff"]["execution_contract"]["reason_codes"] == [
        "spot_remote_lane_missing",
        "portfolio_margin_um_read_only_mode",
        "shadow_executor_only_mode",
    ]
    assert operator["ready_check_scope_market"] == "portfolio_margin_um"
    assert "scope: portfolio_margin_um:portfolio_margin_um" in operator["operator_handoff_brief"]
    assert operator["security_acceptance_status"] == "review"
    assert operator["security_acceptance_max_allowed_exposure"] == 2.5
    assert operator["security_acceptance_reasons"] == [
        "security_exposure_above_threshold(3.0>2.5)"
    ]
    assert payload["source_artifacts"]["ready_check_spot"].endswith(
        "20260313T012501Z_remote_live_ready_check_spot.json"
    )
    assert payload["source_artifacts"]["ready_check_portfolio_margin_um"].endswith(
        "20260313T012502Z_remote_live_ready_check_portfolio_margin_um.json"
    )
    assert payload["source_artifacts"]["risk_daemon_status"].endswith(
        "20260313T012503Z_remote_live_risk_daemon_status.json"
    )
    assert payload["source_artifacts"]["ops_status"].endswith(
        "20260313T012504Z_remote_live_ops_reconcile_status.json"
    )
    assert payload["source_artifacts"]["bridge_context"].endswith(
        "20260313T012507Z_remote_live_bridge_context.json"
    )


def test_build_remote_live_handoff_prefers_explicit_spot_target_market(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260313T012501Z_remote_live_ready_check_spot.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_ready_check_spot",
            "captured_at_utc": "2026-03-13T01:25:01Z",
            "returncode": 3,
            "payload": {
                "action": "live-takeover-ready-check",
                "market": "spot",
                "ready": False,
                "reason": "insufficient_quote_balance",
                "reasons": ["insufficient_quote_balance"],
                "quote_available": 0.0,
                "required_quote": 5.0,
                "ops_live_gate": {"ok": True, "blocking_reason_codes": []},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012502Z_remote_live_ready_check_portfolio_margin_um.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_ready_check_portfolio_margin_um",
            "captured_at_utc": "2026-03-13T01:25:02Z",
            "returncode": 3,
            "payload": {
                "action": "live-takeover-ready-check",
                "market": "portfolio_margin_um",
                "ready": False,
                "reason": "portfolio_margin_um_read_only_mode",
                "reasons": ["portfolio_margin_um_read_only_mode", "risk_guard_blocked"],
                "quote_available": -0.8,
                "required_quote": 5.0,
                "risk_guard": {
                    "allowed": False,
                    "status": "blocked",
                    "reasons": ["open_exposure_above_cap"],
                },
                "ops_live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012503Z_remote_live_risk_daemon_status.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_risk_daemon_status",
            "captured_at_utc": "2026-03-13T01:25:03Z",
            "returncode": 0,
            "payload": {
                "action": "live-risk-daemon-status",
                "mode": "systemd",
                "systemd": {"active_state": "active", "unit_file_state": "enabled"},
                "payload": {"running": True, "status": "running"},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012504Z_remote_live_ops_reconcile_status.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_ops_reconcile_status",
            "captured_at_utc": "2026-03-13T01:25:04Z",
            "returncode": 0,
            "payload": {
                "action": "live-ops-reconcile-status",
                "ok": False,
                "status": "blocked",
                "reason_code": "ops_status_red",
                "live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
            },
        },
    )
    _write_json(
        review_dir / "20260313T012505Z_remote_live_risk_daemon_journal.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_risk_daemon_journal",
            "captured_at_utc": "2026-03-13T01:25:05Z",
            "returncode": 0,
            "payload": {"action": "live-risk-daemon-journal", "lines": ["line1"]},
        },
    )
    _write_json(
        review_dir / "20260313T012506Z_remote_live_risk_daemon_security_status.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_risk_daemon_security_status",
            "captured_at_utc": "2026-03-13T01:25:06Z",
            "returncode": 0,
            "payload": {
                "action": "live-risk-daemon-security-status",
                "verify": {"ok": True, "returncode": 0},
                "security": {
                    "returncode": 0,
                    "overall_exposure": 3.0,
                    "overall_rating": "OK",
                    "findings": [],
                },
            },
        },
    )
    _write_json(
        review_dir / "20260313T012507Z_remote_live_bridge_context.json",
        {
            "action": "capture_remote_live_handoff_context",
            "capture_kind": "remote_live_bridge_context",
            "captured_at_utc": "2026-03-13T01:25:07Z",
            "remote_host": "43.153.148.242",
            "remote_user": "ubuntu",
            "remote_project_dir": "/home/ubuntu/openclaw-system",
            "security_accept_max_exposure": 2.5,
            "live_takeover_market": "spot",
            "openclaw_orderflow_executor_mode": "shadow_guarded",
        },
    )
    _write_json(
        review_dir / "20260313T012000Z_remote_live_history_audit.json",
        {
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "market": "portfolio_margin_um",
            "status": "ok",
            "window_summaries": [
                {
                    "window_hours": 24,
                    "history_window_label": "24h",
                    "quote_available": -0.87,
                    "open_positions": 1,
                    "closed_pnl": 14.82,
                    "trade_count": 20,
                    "risk_guard_status": "blocked",
                    "risk_guard_reasons": ["open_exposure_above_cap"],
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-13T01:30:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    operator = payload["operator_handoff"]
    assert payload["ready_check"]["market"] == "spot"
    assert payload["ready_check_returncode"] == 3
    assert operator["target_live_takeover_market"] == "spot"
    assert operator["ready_check_scope_market"] == "spot"
    assert operator["ready_check_scope_source"] == "spot"
    assert payload["operator_handoff"]["execution_contract"]["reason_codes"] == [
        "shadow_executor_only_mode"
    ]
    assert payload["operator_handoff"]["execution_contract"]["executor_mode"] == "shadow_guarded"
    assert payload["operator_handoff"]["execution_contract"]["executor_mode_source"] == "bridge_context"
    assert payload["operator_handoff"]["execution_contract"]["brief"] == (
        "non_executable_contract:spot:spot:shadow_executor_only_mode"
    )


def test_build_remote_live_handoff_marks_requested_non_shadow_executor_mode_as_probe_only_contract(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260313T012501Z_remote_live_ready_check_spot.json",
        {
            "action": "capture_remote_live_handoff_input",
            "capture_kind": "remote_live_ready_check_spot",
            "captured_at_utc": "2026-03-13T01:25:01Z",
            "returncode": 3,
            "payload": {
                "action": "live-takeover-ready-check",
                "market": "spot",
                "ready": False,
                "reasons": ["risk_guard_blocked"],
                "ops_live_gate": {
                    "ok": False,
                    "blocking_reason_codes": ["ticket_missing:no_actionable_ticket"],
                },
            },
        },
    )
    _write_json(
        review_dir / "20260313T012507Z_remote_live_bridge_context.json",
        {
            "action": "capture_remote_live_handoff_context",
            "capture_kind": "remote_live_bridge_context",
            "captured_at_utc": "2026-03-13T01:25:07Z",
            "remote_host": "43.153.148.242",
            "remote_user": "ubuntu",
            "remote_project_dir": "/home/ubuntu/openclaw-system",
            "security_accept_max_exposure": 2.5,
            "live_takeover_market": "spot",
            "openclaw_orderflow_executor_mode": "spot_live_guarded",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-13T01:30:00+00:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    contract = payload["operator_handoff"]["execution_contract"]
    assert contract["status"] == "probe_only_contract"
    assert contract["mode"] == "guarded_probe_only"
    assert contract["executor_mode"] == "spot_live_guarded"
    assert contract["executor_mode_source"] == "bridge_context"
    assert contract["guarded_probe_allowed"] is True
    assert contract["reason_codes"] == ["guarded_probe_only_mode"]
    assert contract["live_orders_allowed"] is False


def test_build_remote_live_handoff_surfaces_notification_readiness_from_latest_send(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    journal_path = tmp_path / "journal.json"
    security_path = tmp_path / "security.json"
    notification_send_path = review_dir / "20260309T120000Z_remote_live_notification_send.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": True,
            "reasons": [],
            "ops_live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        journal_path,
        {
            "action": "live-risk-daemon-journal",
            "lines": ["line1"],
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {
                "returncode": 0,
                "overall_exposure": 3.2,
                "overall_rating": "OK",
                "findings": [],
            },
        },
    )
    _write_json(
        notification_send_path,
        {
            "action": "send_remote_live_notification",
            "status": "delivery_none",
            "delivery_readiness_label": "delivery-none",
            "delivery_capabilities": {
                "delivery_requested": "none",
                "telegram_configured": False,
                "feishu_configured": False,
                "available_channels": [],
                "blocked_channels": {
                    "telegram": ["telegram_token_missing"],
                    "feishu": ["feishu_hook_token_missing"],
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
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--journal-file",
            str(journal_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T02:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    handoff = payload["operator_handoff"]
    assert handoff["notification_delivery_status"] == "delivery_none"
    assert handoff["notification_delivery_readiness_label"] == "delivery-none"
    assert handoff["notification_status_label"] == "notify-disabled"
    assert handoff["notification_delivery_capabilities"]["delivery_requested"] == "none"
    assert "notify: delivery-none" in handoff["operator_handoff_brief"]
    assert "- notify: `delivery-none`" in handoff["operator_handoff_md"]
    assert (
        handoff["operator_status_quad"]
        == "runtime-ok / gate-ok / risk-guard-ok / notify-disabled"
    )


def test_build_remote_live_handoff_reports_risk_daemon_attention(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["risk_guard_blocked"],
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "inactive", "unit_file_state": "enabled"},
            "payload": {"running": False, "status": "stopped"},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": False, "returncode": 1},
            "security": {"returncode": 0, "overall_exposure": 6.1, "overall_rating": "MEDIUM"},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_handoff"]["handoff_state"] == "risk_daemon_attention"
    assert payload["operator_handoff"]["daemon_running"] is False
    assert payload["operator_handoff"]["security_acceptance_status"] == "failed"
    assert payload["operator_handoff"]["runtime_status_label"] == "runtime-failed"
    assert payload["operator_handoff"]["runtime_attention_reasons"] == [
        "daemon_not_running",
        "security_acceptance_failed",
    ]
    assert payload["operator_handoff"]["gate_status_label"] == "gate-ok"
    assert payload["operator_handoff"]["gate_attention_reasons"] == []
    assert payload["operator_handoff"]["risk_guard_status_label"] == "risk-guard-blocked"
    assert payload["operator_handoff"]["risk_guard_attention_reasons"] == [
        "risk_guard_blocked"
    ]
    assert (
        payload["operator_handoff"]["operator_status_triplet"]
        == "runtime-failed / gate-ok / risk-guard-blocked"
    )
    assert payload["operator_handoff"]["next_focus_area"] == "runtime"
    assert payload["operator_handoff"]["next_focus_reason"] == "daemon_not_running"
    assert payload["operator_handoff"]["next_focus_command"].endswith(
        "scripts/openclaw_cloud_bridge.sh live-risk-daemon-start"
    )
    assert payload["operator_handoff"]["next_focus_commands"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-start",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-status",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-journal",
    ]
    assert payload["operator_handoff"]["security_status_label"] == "security-review"
    assert payload["operator_handoff"]["security_attention_reasons"] == ["systemd_verify_failed"]
    assert payload["operator_handoff"]["security_acceptance_reasons"] == ["systemd_verify_failed"]
    assert payload["operator_handoff"]["security_top_risks"] == []
    assert payload["operator_handoff"]["security_recommendations"] == []
    assert "live-risk-daemon-start" in " ".join(payload["operator_handoff"]["operator_commands"])


def test_build_remote_live_handoff_updates_inet_recommendation_for_hardened_network(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["ops_live_gate_blocked"],
            "ops_live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
            "payload_alignment": {"aligned": True, "reasons": []},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {
                "returncode": 0,
                "overall_exposure": 2.1,
                "overall_rating": "OK",
                "findings": ["✗ RestrictAddressFamilies=~AF_(INET|INET6)"],
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T10:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_handoff"]["security_top_risks"] == ["internet_socket_access"]
    assert payload["operator_handoff"]["security_recommendations"] == [
        "Review whether RestrictAddressFamilies can drop AF_INET/AF_INET6 now that PrivateNetwork and IPAddressDeny are enabled."
    ]


def test_build_remote_live_handoff_distinguishes_syscall_filter_findings(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["ops_live_gate_blocked"],
            "ops_live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
            "payload_alignment": {"aligned": True, "reasons": []},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": False, "blocking_reason_codes": ["ops_status_red"]},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {
                "returncode": 0,
                "overall_exposure": 0.5,
                "overall_rating": "SAFE",
                "findings": [
                    "✗ SystemCallFilter=~@resources",
                    "✗ SystemCallFilter=~@privileged",
                    "✗ RootDirectory=/RootImage=",
                ],
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T11:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_handoff"]["security_top_risks"] == [
        "syscall_resources_allowed",
        "syscall_privileged_allowed",
        "host_root_filesystem",
    ]
    assert payload["operator_handoff"]["security_recommendations"] == [
        "Review whether @resources is still required in SystemCallFilter after measuring runtime behavior under the current hardening profile.",
        "Review whether @privileged can be narrowed further without breaking the daemon lifecycle or file ownership operations.",
        "If acceptable, review RootImage/RootDirectory isolation; otherwise keep ProtectSystem strict and document host-root dependency.",
    ]


def test_build_remote_live_handoff_maps_rtc_device_allow_to_protectclock_floor(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["ops_live_gate_blocked"],
            "ops_live_gate": {"ok": False, "blocking_reason_codes": ["slot_anomaly"]},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
            "payload_alignment": {"aligned": True, "reasons": []},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": False, "blocking_reason_codes": ["slot_anomaly"]},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {
                "returncode": 0,
                "overall_exposure": 0.3,
                "overall_rating": "SAFE",
                "findings": [
                    "✗ DeviceAllow= Service has a device ACL with some special devices: char-rtc:r 0.1",
                ],
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-10T00:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_handoff"]["security_top_risks"] == ["rtc_read_allowed"]
    assert payload["operator_handoff"]["security_recommendations"] == [
        "ProtectClock implies read-only RTC access via DeviceAllow=char-rtc:r. Treat this as the tested floor unless you are prepared to remove ProtectClock."
    ]
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["runtime"][
            "clock_device_floor"
        ]
        == "rtc-read-only"
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["runtime"][
            "clock_device_note"
        ]
        == "ProtectClock implies read-only RTC access via DeviceAllow=char-rtc:r. Treat this as the tested floor unless you are prepared to remove ProtectClock."
    )
    assert "clock-device floor: `rtc-read-only`" in payload["operator_handoff"][
        "operator_playbook_md"
    ]
    assert "- clock-device floor: `rtc-read-only`" in payload["operator_handoff"][
        "operator_handoff_md"
    ]
    assert "clockdev: rtc-read-only" in payload["operator_handoff"]["operator_handoff_brief"]
    assert (
        payload["operator_handoff"]["operator_notification"]["runtime_floor_brief"]
        == "clockdev=rtc-read-only"
    )
    assert (
        "clockdev=rtc-read-only" in payload["operator_handoff"]["operator_notification"]["body"]
    )
    assert "clockdev:rtc-read-only" in payload["operator_handoff"]["operator_notification"][
        "tags"
    ]
    assert payload["operator_handoff"]["operator_notification_templates"]["generic"][
        "runtime_floor_brief"
    ] == "clockdev=rtc-read-only"


def test_build_remote_live_handoff_reports_alignment_attention_when_payload_lags(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["risk_guard_blocked"],
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled", "main_pid": 222},
            "payload": {"running": True, "status": "running", "pid": 111},
            "payload_alignment": {
                "aligned": False,
                "systemd_active_state": "active",
                "systemd_main_pid": 222,
                "payload_pid": 111,
                "payload_running": True,
                "payload_pid_alive": True,
                "reasons": ["payload_pid_mismatch(payload=111,systemd=222)"],
            },
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {"returncode": 0, "overall_exposure": 3.0, "overall_rating": "OK"},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_handoff"]["handoff_state"] == "risk_daemon_attention"
    assert payload["operator_handoff"]["daemon_running"] is True
    assert payload["operator_handoff"]["daemon_payload_alignment_ok"] is False
    assert payload["operator_handoff"]["runtime_status_label"] == "runtime-review"
    assert payload["operator_handoff"]["runtime_attention_reasons"] == [
        "daemon_payload_unaligned"
    ]
    assert payload["operator_handoff"]["gate_status_label"] == "gate-ok"
    assert payload["operator_handoff"]["gate_attention_reasons"] == []
    assert payload["operator_handoff"]["risk_guard_status_label"] == "risk-guard-blocked"
    assert payload["operator_handoff"]["risk_guard_attention_reasons"] == [
        "risk_guard_blocked"
    ]
    assert (
        payload["operator_handoff"]["operator_status_triplet"]
        == "runtime-review / gate-ok / risk-guard-blocked"
    )
    assert payload["operator_handoff"]["next_focus_area"] == "runtime"
    assert payload["operator_handoff"]["next_focus_reason"] == "daemon_payload_unaligned"
    assert payload["operator_handoff"]["next_focus_command"].endswith(
        "scripts/openclaw_cloud_bridge.sh live-risk-daemon-status"
    )
    assert payload["operator_handoff"]["next_focus_commands"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-status",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-journal",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
    ]
    assert payload["operator_handoff"]["daemon_payload_alignment_reasons"] == [
        "payload_pid_mismatch(payload=111,systemd=222)"
    ]


def test_build_remote_live_handoff_uses_security_acceptance_payload_when_present(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"

    _write_json(ready_path, {"action": "live-takeover-ready-check", "ready": False, "reasons": []})
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": True, "blocking_reason_codes": []},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {"returncode": 0, "overall_exposure": 3.0, "overall_rating": "OK"},
            "security_acceptance": {
                "status": "review",
                "max_allowed_exposure": 2.5,
                "observed_exposure": 3.0,
                "observed_rating": "OK",
                "reasons": ["security_exposure_above_threshold(3.0>2.5)"],
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--security-accept-max-exposure",
            "4.0",
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_handoff"]["security_acceptance_status"] == "review"
    assert payload["operator_handoff"]["security_acceptance_max_allowed_exposure"] == 2.5
    assert payload["operator_handoff"]["security_acceptance_reasons"] == [
        "security_exposure_above_threshold(3.0>2.5)"
    ]
    assert payload["operator_handoff"]["runtime_status_label"] == "runtime-review"
    assert payload["operator_handoff"]["runtime_attention_reasons"] == [
        "security_acceptance_review"
    ]
    assert payload["operator_handoff"]["gate_status_label"] == "gate-ok"
    assert payload["operator_handoff"]["gate_attention_reasons"] == []
    assert payload["operator_handoff"]["risk_guard_status_label"] == "risk-guard-review"
    assert payload["operator_handoff"]["risk_guard_attention_reasons"] == [
        "risk_guard_status_missing"
    ]
    assert (
        payload["operator_handoff"]["operator_status_triplet"]
        == "runtime-review / gate-ok / risk-guard-review"
    )
    assert payload["operator_handoff"]["next_focus_area"] == "runtime"
    assert payload["operator_handoff"]["next_focus_reason"] == "security_acceptance_review"
    assert payload["operator_handoff"]["next_focus_command"].endswith(
        "scripts/openclaw_cloud_bridge.sh live-risk-daemon-security-status"
    )
    assert payload["operator_handoff"]["next_focus_commands"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-security-status",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-status",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
    ]
    assert payload["operator_handoff"]["security_status_label"] == "security-review"


def test_build_remote_live_handoff_reports_ops_live_gate_blocked_labels(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["risk_guard_blocked", "ops_live_gate_blocked"],
            "risk_guard": {
                "allowed": False,
                "status": "blocked",
                "reasons": ["ticket_missing:no_actionable_ticket"],
            },
            "ops_live_gate": {
                "ok": False,
                "blocking_reason_codes": ["rollback_hard", "slot_anomaly"],
            },
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": False, "blocking_reason_codes": ["rollback_hard", "slot_anomaly"]},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {"returncode": 0, "overall_exposure": 3.0, "overall_rating": "OK"},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T02:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_handoff"]["handoff_state"] == "ops_live_gate_blocked"
    assert payload["operator_handoff"]["runtime_status_label"] == "runtime-ok"
    assert payload["operator_handoff"]["gate_status_label"] == "gate-blocked"
    assert payload["operator_handoff"]["gate_attention_reasons"] == [
        "ops_live_gate_blocked",
        "live_gate:rollback_hard",
        "live_gate:slot_anomaly",
    ]
    assert payload["operator_handoff"]["risk_guard_status_label"] == "risk-guard-blocked"
    assert payload["operator_handoff"]["risk_guard_attention_reasons"] == [
        "risk_guard_blocked",
        "risk_guard:ticket_missing:no_actionable_ticket",
    ]
    assert (
        payload["operator_handoff"]["operator_status_triplet"]
        == "runtime-ok / gate-blocked / risk-guard-blocked"
    )
    assert (
        payload["operator_handoff"]["operator_status_quad"]
        == "runtime-ok / gate-blocked / risk-guard-blocked / notify-unknown"
    )
    assert payload["operator_handoff"]["next_focus_area"] == "gate"
    assert payload["operator_handoff"]["next_focus_reason"] == "ops_live_gate_blocked"
    assert payload["operator_handoff"]["focus_stack"] == [
        {"area": "gate", "reason": "ops_live_gate_blocked"},
        {"area": "risk_guard", "reason": "risk_guard_blocked"},
    ]
    assert payload["operator_handoff"]["focus_stack_brief"] == "gate -> risk_guard"
    assert (
        payload["operator_handoff"]["focus_stack_summary"]
        == "gate(ops_live_gate_blocked) -> risk_guard(risk_guard_blocked)"
    )
    assert payload["operator_handoff"]["next_focus_command"].endswith(
        "scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status"
    )
    assert payload["operator_handoff"]["next_focus_commands"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
    ]
    assert payload["operator_handoff"]["secondary_focus_area"] == "risk_guard"
    assert payload["operator_handoff"]["secondary_focus_reason"] == "risk_guard_blocked"
    assert payload["operator_handoff"]["secondary_focus_command"].endswith(
        "scripts/openclaw_cloud_bridge.sh live-takeover-ready-check"
    )
    assert payload["operator_handoff"]["secondary_focus_commands"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-takeover-ready-check",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-risk-daemon-status",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
    ]
    assert payload["operator_handoff"]["operator_playbook"]["active_focus"] == "gate"
    assert payload["operator_handoff"]["operator_playbook"]["active_reason"] == "ops_live_gate_blocked"
    assert payload["operator_handoff"]["operator_playbook"]["primary_sequence"] == [
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh",
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
    ]
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["gate"]["status_label"]
        == "gate-blocked"
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["gate"]["attention_reasons"]
        == [
            "ops_live_gate_blocked",
            "live_gate:rollback_hard",
            "live_gate:slot_anomaly",
        ]
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["risk_guard"]["status_label"]
        == "risk-guard-blocked"
    )
    assert (
        payload["operator_handoff"]["operator_playbook"]["sections"]["canary"]["status_label"]
        == "canary-blocked"
    )
    assert "## Active Focus" in payload["operator_handoff"]["operator_playbook_md"]
    assert "- focus: `gate`" in payload["operator_handoff"]["operator_playbook_md"]
    assert "reasons: `ops_live_gate_blocked`, `live_gate:rollback_hard`, `live_gate:slot_anomaly`" in payload["operator_handoff"]["operator_playbook_md"]
    assert "# Remote Live Handoff" in payload["operator_handoff"]["operator_handoff_md"]
    assert "- state: `ops_live_gate_blocked`" in payload["operator_handoff"]["operator_handoff_md"]
    assert "- focus-stack: `gate -> risk_guard`" in payload["operator_handoff"]["operator_handoff_md"]
    assert "- next focus: `gate`" in payload["operator_handoff"]["operator_handoff_md"]
    assert "security top risks" in payload["operator_handoff"]["operator_handoff_md"]
    assert (
        payload["operator_handoff"]["remote_live_diagnosis"]["status"]
        == "auto_live_blocked_without_profitability_confirmation"
    )
    assert payload["operator_handoff"]["operator_handoff_brief"] == "\n".join(
        [
            "state: ops_live_gate_blocked",
            "status: runtime-ok / gate-blocked / risk-guard-blocked",
            "status4: runtime-ok / gate-blocked / risk-guard-blocked / notify-unknown",
            "focus-stack: gate -> risk_guard",
            "addrfam: unknown",
            "history-diagnosis: auto_live_blocked_without_profitability_confirmation:unknown:ops_live_gate+risk_guard",
            "focus: gate",
            "reason: ops_live_gate_blocked",
            "cmd: cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
            "focus2: risk_guard",
            "reason2: risk_guard_blocked",
        ]
    )
    assert payload["operator_handoff"]["operator_notification"] == {
        "level": "warning",
        "title": "Remote gate is blocking live execution",
        "body": "runtime-ok / gate-blocked / risk-guard-blocked; focus=gate; reason=ops_live_gate_blocked; addrfam=unknown; focusstack=gate -> risk_guard; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
        "command": "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
        "tags": [
            "state:ops_live_gate_blocked",
            "focus:gate",
            "security:security-ok",
        ],
        "focus_stack_brief": "gate -> risk_guard",
        "runtime_floor_brief": "",
        "plain_text": "[WARNING] Remote gate is blocking live execution\nruntime-ok / gate-blocked / risk-guard-blocked; focus=gate; reason=ops_live_gate_blocked; addrfam=unknown; focusstack=gate -> risk_guard; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
        "markdown": "\n".join(
            [
                "**Remote gate is blocking live execution**",
                "- level: `warning`",
                "- state: `ops_live_gate_blocked`",
                "- status: `runtime-ok / gate-blocked / risk-guard-blocked`",
                "- focus-stack: `gate -> risk_guard`",
                "- focus: `gate`",
                "- reason: `ops_live_gate_blocked`",
                "- command: `cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status`",
            ]
        ),
    }
    assert payload["operator_handoff"]["operator_notification_templates"] == {
        "telegram": {
            "parse_mode": "MarkdownV2",
            "text": "*Remote gate is blocking live execution*\nlevel: `warning`\nruntime\\-ok / gate\\-blocked / risk\\-guard\\-blocked; focus\\=gate; reason\\=ops\\_live\\_gate\\_blocked; addrfam\\=unknown; focusstack\\=gate \\-\\> risk\\_guard; cmd\\=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw\\_cloud\\_bridge\\.sh live\\-ops\\-reconcile\\-status\ncmd: `cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw\\_cloud\\_bridge\\.sh live\\-ops\\-reconcile\\-status`\nfocus-stack: `gate \\-\\> risk\\_guard`",
            "disable_web_page_preview": True,
        },
        "feishu": {
            "msg_type": "text",
            "content": {
                "text": "[WARNING] Remote gate is blocking live execution\nruntime-ok / gate-blocked / risk-guard-blocked; focus=gate; reason=ops_live_gate_blocked; addrfam=unknown; focusstack=gate -> risk_guard; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status\nfocus-stack: gate -> risk_guard"
            },
        },
        "generic": {
            "level": "warning",
            "title": "Remote gate is blocking live execution",
            "body": "runtime-ok / gate-blocked / risk-guard-blocked; focus=gate; reason=ops_live_gate_blocked; addrfam=unknown; focusstack=gate -> risk_guard; cmd=cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
            "command": "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
            "tags": [
                "state:ops_live_gate_blocked",
                "focus:gate",
                "security:security-ok",
            ],
            "focus_stack_brief": "gate -> risk_guard",
            "runtime_floor_brief": "",
        },
    }


def test_build_remote_live_handoff_surfaces_noaf_incompatibility(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    ready_path = tmp_path / "ready.json"
    daemon_path = tmp_path / "daemon.json"
    ops_path = tmp_path / "ops.json"
    security_path = tmp_path / "security.json"
    noaf_path = tmp_path / "noaf.json"

    _write_json(
        ready_path,
        {
            "action": "live-takeover-ready-check",
            "ready": False,
            "reasons": ["ops_live_gate_blocked", "risk_guard_blocked"],
            "ops_live_gate": {"ok": False, "blocking_reason_codes": ["slot_anomaly"]},
        },
    )
    _write_json(
        daemon_path,
        {
            "action": "live-risk-daemon-status",
            "mode": "systemd",
            "systemd": {"active_state": "active", "unit_file_state": "enabled"},
            "payload": {"running": True, "status": "running"},
            "payload_alignment": {"aligned": True, "reasons": []},
        },
    )
    _write_json(
        ops_path,
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "live_gate": {"ok": False, "blocking_reason_codes": ["slot_anomaly"]},
        },
    )
    _write_json(
        security_path,
        {
            "action": "live-risk-daemon-security-status",
            "verify": {"ok": True, "returncode": 0},
            "security": {"returncode": 0, "overall_exposure": 0.3, "overall_rating": "SAFE", "findings": []},
        },
    )
    _write_json(
        noaf_path,
        {
            "action": "build_remote_live_noaf_probe",
            "ok": False,
            "status": "incompatible",
            "probe": {"status": "incompatible"},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--ready-check-file",
            str(ready_path),
            "--risk-daemon-status-file",
            str(daemon_path),
            "--ops-status-file",
            str(ops_path),
            "--security-status-file",
            str(security_path),
            "--noaf-probe-file",
            str(noaf_path),
            "--remote-host",
            "43.153.148.242",
            "--remote-user",
            "ubuntu",
            "--remote-project-dir",
            "/home/ubuntu/openclaw-system",
            "--now",
            "2026-03-09T12:30:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    handoff = payload["operator_handoff"]
    assert handoff["address_family_floor"] == "AF_UNIX"
    assert handoff["address_family_probe_status"] == "incompatible"
    assert "RestrictAddressFamilies=none probe is incompatible" in str(
        handoff["address_family_recommendation"]
    )
    assert handoff["address_family_probe"]["status"] == "incompatible"
    assert (
        handoff["operator_playbook"]["sections"]["runtime"]["address_family_floor"] == "AF_UNIX"
    )
    assert "address-family floor: `AF_UNIX`" in handoff["operator_playbook_md"]
    assert "- address-family floor: `AF_UNIX`" in handoff["operator_handoff_md"]
