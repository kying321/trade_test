from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import textwrap


def _extract_bridge_lib(tmp_path: Path) -> tuple[Path, Path]:
    system_root = Path(__file__).resolve().parents[1]
    script_path = system_root / "scripts" / "openclaw_cloud_bridge.sh"
    text = script_path.read_text(encoding="utf-8")
    marker = "\nif (( $# != 1 )); then\n"
    idx = text.rfind(marker)
    assert idx != -1
    lib_path = tmp_path / "openclaw_cloud_bridge.lib.sh"
    lib_path.write_text(text[:idx], encoding="utf-8")
    return lib_path, system_root


def _run_bridge_shell(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    lib_path, system_root = _extract_bridge_lib(tmp_path)
    env = os.environ.copy()
    env["FENLIE_SYSTEM_ROOT"] = str(system_root)
    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        source "{lib_path}"
        {body}
        """
    )
    return subprocess.run(
        ["bash", "-lc", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _stub_guarded_exec_payload() -> str:
    return json.dumps(
        {
            "artifact": "/tmp/guard.json",
            "risk_guard": {
                "returncode": 3,
                "payload": {
                    "allowed": False,
                    "status": "blocked",
                    "reasons": ["ticket_missing:no_actionable_ticket"],
                    "backup_intel": {
                        "enabled": True,
                        "status": "active",
                        "active": True,
                        "blocking_reasons": ["backup_intel_high_risk_flag"],
                    },
                },
            },
            "takeover": {
                "payload": {
                    "market": "spot",
                    "artifact": "/tmp/takeover.json",
                    "steps": {
                        "credentials": {"has_api_key": True, "has_api_secret": True},
                        "canary_plan": {
                            "quote_usdt": 5.0,
                            "effective_quote_usdt": 0.0,
                        },
                        "account_overview": {"quote_available": 10.0},
                    },
                }
            },
        },
        ensure_ascii=False,
    )


def _stub_guarded_exec_payload_risk_ok() -> str:
    return json.dumps(
        {
            "artifact": "/tmp/guard.json",
            "risk_guard": {
                "returncode": 0,
                "payload": {
                    "allowed": True,
                    "status": "allowed",
                    "reasons": [],
                },
            },
            "takeover": {
                "payload": {
                    "market": "spot",
                    "artifact": "/tmp/takeover.json",
                    "steps": {
                        "credentials": {"has_api_key": True, "has_api_secret": True},
                        "canary_plan": {
                            "quote_usdt": 5.0,
                            "effective_quote_usdt": 5.0,
                        },
                        "account_overview": {"quote_available": 10.0},
                    },
                }
            },
        },
        ensure_ascii=False,
    )


def _stub_ops_payload() -> str:
    return json.dumps(
        {
            "action": "live-ops-reconcile-status",
            "ok": False,
            "status": "blocked",
            "reason_code": "ops_reconcile_drift_blocked",
        },
        ensure_ascii=False,
    )


def _stub_ops_gate_failed_payload() -> str:
    return json.dumps(
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "reason_code": "",
            "gate_passed": False,
            "ops_status": "red",
        },
        ensure_ascii=False,
    )


def _stub_ops_live_gate_blocked_payload() -> str:
    return json.dumps(
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "reason_code": "",
            "gate_passed": False,
            "ops_status": "red",
            "gate_failed_checks": ["max_drawdown_ok", "stable_replay_ok"],
            "live_gate": {
                "ok": False,
                "required_checks": {
                    "review_pass_gate": True,
                    "mode_health_ok": True,
                    "health_ok": False,
                    "stable_replay_ok": False,
                    "reconcile_drift_ok": True,
                    "artifact_governance_ok": True,
                },
                "gate_failed_checks": ["health_ok", "stable_replay_ok", "max_drawdown_ok"],
                "blocking_reason_codes": ["rollback_hard", "max_drawdown", "stable_replay", "ops_status_red"],
                "rollback_active": True,
                "rollback_level": "hard",
                "rollback_anchor_ready": True,
                "rollback_action": "rollback_now_and_lock_parameter_updates",
                "rollback_reason_codes": ["max_drawdown", "stable_replay"],
                "ops_status": "red",
            },
        },
        ensure_ascii=False,
    )


def _stub_ops_live_gate_allowed_payload() -> str:
    return json.dumps(
        {
            "action": "live-ops-reconcile-status",
            "ok": True,
            "status": "passed",
            "reason_code": "",
            "gate_passed": False,
            "ops_status": "red",
            "gate_failed_checks": ["review_pass_gate"],
            "live_gate": {
                "ok": True,
                "required_checks": {
                    "mode_health_ok": True,
                    "runtime_health_ok": True,
                    "runtime_replay_ok": True,
                    "state_stability_ok": True,
                    "reconcile_drift_ok": True,
                    "artifact_governance_ok": True,
                },
                "gate_failed_checks": ["review_pass_gate"],
                "blocking_reason_codes": [],
                "rollback_active": False,
                "rollback_level": "none",
                "rollback_anchor_ready": True,
                "rollback_action": "no_rollback",
                "rollback_reason_codes": [],
                "ops_status": "red",
            },
        },
        ensure_ascii=False,
    )


def _stub_remote_live_handoff_payload() -> str:
    return json.dumps(
        {
            "action": "build_remote_live_handoff",
            "ok": True,
            "status": "ok",
            "operator_handoff": {
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
                "next_focus_area": "gate",
                "next_focus_reason": "ops_live_gate_blocked",
                "operator_notification": {
                    "level": "warning",
                    "title": "Remote gate is blocking live execution",
                    "body": "runtime-ok / gate-blocked / risk-guard-blocked; focus=gate",
                    "command": "cmd-one",
                    "tags": ["state:ops_live_gate_blocked"],
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
        ensure_ascii=False,
    )


def test_ready_check_returns_payload_and_nonzero_rc_without_exiting_shell(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        run_guarded_exec_remote() {{
          cat <<'JSON'
        {_stub_guarded_exec_payload()}
        JSON
        }}
        run_live_ops_reconcile_status_remote() {{
          cat <<'JSON'
        {_stub_ops_payload()}
        JSON
        }}
        tmp_out="$(mktemp)"
        set +e
        action_live_takeover_ready_check >"${{tmp_out}}"
        rc=$?
        set -e
        printf 'RC=%s\\n' "${{rc}}"
        cat "${{tmp_out}}"
        """,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines and lines[0] == "RC=3"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["ready"] is False
    assert payload["reasons"] == ["risk_guard_blocked", "ops_reconcile_drift_blocked"]
    assert payload["backup_intel"]["status"] == "active"


def test_ready_check_blocks_on_ops_gate_failure_even_when_reconcile_passes(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        run_guarded_exec_remote() {{
          cat <<'JSON'
        {_stub_guarded_exec_payload_risk_ok()}
        JSON
        }}
        run_live_ops_reconcile_status_remote() {{
          cat <<'JSON'
        {_stub_ops_gate_failed_payload()}
        JSON
        }}
        tmp_out="$(mktemp)"
        set +e
        action_live_takeover_ready_check >"${{tmp_out}}"
        rc=$?
        set -e
        printf 'RC=%s\\n' "${{rc}}"
        cat "${{tmp_out}}"
        """,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines and lines[0] == "RC=3"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["ready"] is False
    assert payload["reasons"] == ["ops_gate_failed", "ops_status_red"]
    assert payload["ops_gate"]["gate_passed"] is False
    assert payload["ops_gate"]["ops_status"] == "red"


def test_ready_check_prefers_structured_live_gate_block_reasons(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        run_guarded_exec_remote() {{
          cat <<'JSON'
        {_stub_guarded_exec_payload_risk_ok()}
        JSON
        }}
        run_live_ops_reconcile_status_remote() {{
          cat <<'JSON'
        {_stub_ops_live_gate_blocked_payload()}
        JSON
        }}
        tmp_out="$(mktemp)"
        set +e
        action_live_takeover_ready_check >"${{tmp_out}}"
        rc=$?
        set -e
        printf 'RC=%s\\n' "${{rc}}"
        cat "${{tmp_out}}"
        """,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines and lines[0] == "RC=3"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["ready"] is False
    assert payload["reasons"] == ["ops_live_gate_blocked"]
    assert payload["ops_live_gate"]["rollback_level"] == "hard"
    assert "max_drawdown" in set(payload["ops_live_gate"]["blocking_reason_codes"])
    assert "stable_replay" in set(payload["ops_live_gate"]["blocking_reason_codes"])


def test_ready_check_allows_structured_live_gate_when_ok(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        run_guarded_exec_remote() {{
          cat <<'JSON'
        {_stub_guarded_exec_payload_risk_ok()}
        JSON
        }}
        run_live_ops_reconcile_status_remote() {{
          cat <<'JSON'
        {_stub_ops_live_gate_allowed_payload()}
        JSON
        }}
        tmp_out="$(mktemp)"
        set +e
        action_live_takeover_ready_check >"${{tmp_out}}"
        rc=$?
        set -e
        printf 'RC=%s\\n' "${{rc}}"
        cat "${{tmp_out}}"
        """,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines and lines[0] == "RC=0"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["ready"] is True
    assert payload["reasons"] == []
    assert payload["ops_live_gate"]["ok"] is True


def test_canary_skips_not_ready_before_bridge_idempotency(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        run_guarded_exec_remote() {{
          cat <<'JSON'
        {_stub_guarded_exec_payload()}
        JSON
        }}
        run_live_ops_reconcile_status_remote() {{
          cat <<'JSON'
        {_stub_ops_payload()}
        JSON
        }}
        idempotency_guard() {{
          echo 'idempotency guard should not run for not-ready canary' >&2
          return 91
        }}
        tmp_out="$(mktemp)"
        set +e
        action_live_takeover_canary >"${{tmp_out}}"
        rc=$?
        set -e
        printf 'RC=%s\\n' "${{rc}}"
        cat "${{tmp_out}}"
        """,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines and lines[0] == "RC=0"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["status"] == "skipped_not_ready"
    assert payload["reasons"] == ["risk_guard_blocked", "ops_reconcile_drift_blocked"]
    assert "idempotency guard should not run" not in proc.stderr


def test_ensure_local_openclaw_runtime_model_action_repairs_temp_home_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    openclaw_dir = home / ".openclaw"
    openclaw_dir.mkdir(parents=True, exist_ok=True)
    cfg = openclaw_dir / "openclaw.json"
    cfg.write_text(
        json.dumps(
            {
                "models": {"providers": {"google": {"baseUrl": "http://127.0.0.1:9999/v1beta", "models": []}}},
                "agents": {"defaults": {"models": {}}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        export HOME="{home}"
        action_ensure_local_openclaw_runtime_model
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["changed"] is True
    out = json.loads(cfg.read_text(encoding="utf-8"))
    assert out["models"]["providers"]["openai"]["models"][0]["id"] == "gpt-5.4"
    assert "openai/gpt-5.4" in out["agents"]["defaults"]["models"]
    assert "gpt-5.4" in out["agents"]["defaults"]["models"]


def test_sync_local_pi_workspace_action_forwards_expected_args(tmp_path: Path) -> None:
    target_root = tmp_path / "pi" / "fenlie-system"
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{target_root}"
        local_pi_workspace_backup_keep="7"
        local_pi_workspace_backup_max_age_hours="24"
        local_pi_workspace_dry_run="true"
        local_pi_workspace_no_backup="true"
        python3() {{
          printf 'PYTHON3:%s\\n' "$*"
        }}
        action_sync_local_pi_workspace
        """,
    )

    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("PYTHON3:")
    assert "scripts/sync_local_pi_workspace.py" in line
    assert f"--target-root {target_root}" in line
    assert "--backup-keep 7" in line
    assert "--backup-max-age-hours 24" in line
    assert f"--pulse-lock-path {target_root}/output/state/run_halfhour_pulse.lock" in line
    assert "--dry-run" in line
    assert "--no-backup" in line


def test_publish_local_pi_runtime_scripts_action_forwards_expected_args(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    runtime_root = tmp_path / "repo-runtime"
    target_root = tmp_path / "pi" / "scripts"
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_runtime_scripts_source_root="{runtime_root}"
        local_pi_runtime_scripts_target_root="{target_root}"
        local_pi_runtime_scripts_backup_keep="6"
        local_pi_runtime_scripts_backup_max_age_hours="48"
        local_pi_runtime_scripts_dry_run="true"
        local_pi_runtime_scripts_no_backup="true"
        python3() {{
          printf 'PYTHON3:%s\\n' "$*"
        }}
        action_publish_local_pi_runtime_scripts
        """,
    )

    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("PYTHON3:")
    assert "scripts/publish_local_pi_runtime_scripts.py" in line
    assert f"--source-root {runtime_root}" in line
    assert f"--target-root {target_root}" in line
    assert f"--output-root {workspace_root}/output" in line
    assert f"--pulse-lock-path {workspace_root}/output/state/run_halfhour_pulse.lock" in line
    assert "--backup-keep 6" in line
    assert "--backup-max-age-hours 48" in line
    assert "--dry-run" in line
    assert "--no-backup" in line


def test_local_pi_consecutive_loss_guardrail_status_runs_local_script(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    scripts_dir = workspace_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "paper_consecutive_loss_guardrail_status.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}, ensure_ascii=False))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_consecutive_loss_stop_threshold="5"
        local_pi_consecutive_loss_ack_allow_missing_last_loss_ts="true"
        action_local_pi_consecutive_loss_guardrail_status
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    argv = list(payload["argv"])
    assert "--stop-threshold" in argv
    assert "5" in argv
    assert "--allow-missing-last-loss-ts" in argv


def test_local_pi_ack_archive_status_runs_local_script(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    scripts_dir = workspace_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "paper_consecutive_loss_ack_archive_status.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}, ensure_ascii=False))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_ack_archive_manifest_tail="7"
        action_local_pi_ack_archive_status
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    argv = list(payload["argv"])
    assert "--archive-dir" in argv
    assert f"{workspace_root}/output/review/paper_consecutive_loss_ack_archive" in argv
    assert "--archive-manifest-path" in argv
    assert f"{workspace_root}/output/review/paper_consecutive_loss_ack_archive/manifest.jsonl" in argv
    assert "--manifest-tail" in argv
    assert "7" in argv


def test_backfill_local_pi_last_loss_ts_runs_local_script(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    scripts_dir = workspace_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "backfill_paper_last_loss_ts.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}, ensure_ascii=False))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        LOCAL_PI_EXPECTED_STATE_FINGERPRINT="abc123"
        local_pi_last_loss_ts_backfill_allow_latest_loss_fallback="true"
        local_pi_last_loss_ts_backfill_write="true"
        action_backfill_local_pi_last_loss_ts
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    argv = list(payload["argv"])
    assert "--expected-state-fingerprint" in argv
    assert "abc123" in argv
    assert "--allow-latest-loss-fallback" in argv
    assert "--write" in argv


def test_prepare_local_pi_runtime_runs_sync_model_and_gate_smoke(tmp_path: Path) -> None:
    runner = tmp_path / "pi_cycle_halfhour_launchd_runner.sh"
    runner.write_text("#!/usr/bin/env bash\necho 'gate ok'\n", encoding="utf-8")
    runner.chmod(0o755)

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_launchd_runner_path="{runner}"
        action_sync_local_pi_workspace() {{
          cat <<'JSON'
        {json.dumps({"action": "sync_local_pi_workspace", "ok": True, "changed": False}, ensure_ascii=False)}
        JSON
        }}
        action_publish_local_pi_runtime_scripts() {{
          cat <<'JSON'
        {json.dumps({"action": "publish_local_pi_runtime_scripts", "ok": True, "changed": False}, ensure_ascii=False)}
        JSON
        }}
        action_ensure_local_openclaw_runtime_model() {{
          cat <<'JSON'
        {json.dumps({"action": "ensure_openclaw_runtime_model", "ok": True, "changed": False}, ensure_ascii=False)}
        JSON
        }}
        action_prepare_local_pi_runtime
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["sync"]["ok"] is True
    assert payload["runtime_scripts"]["ok"] is True
    assert payload["runtime_model"]["ok"] is True
    assert payload["gate_smoke"]["ok"] is True
    assert payload["gate_smoke"]["runner_path"] == str(runner)
    assert "gate ok" in "\n".join(payload["gate_smoke"]["log_tail"])


def test_smoke_local_pi_cycle_runs_prepare_then_full_cycle(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    review_dir = workspace_root / "output" / "review"
    review_dir.mkdir(parents=True)
    log_path = tmp_path / "pi_cycle_launchd.log"
    runner = tmp_path / "pi_cycle_halfhour_launchd_runner.sh"
    artifact_path = review_dir / "20260308T101500Z_pi_launchd_auto_retro.json"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"printf '%s\\n' '[2026-03-08T10:15:00Z] pi_cycle_launchd done rc=0' >> '{log_path}'",
                f"cat <<'JSON' > '{artifact_path}'",
                json.dumps({"status": "ok", "core_execution_status": "ok"}, ensure_ascii=False),
                "JSON",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_launchd_runner_path="{runner}"
        local_pi_launchd_log_path="{log_path}"
        action_prepare_local_pi_runtime() {{
          cat <<'JSON'
        {json.dumps({"action": "prepare_local_pi_runtime", "ok": True, "status": "ok"}, ensure_ascii=False)}
        JSON
        }}
        action_smoke_local_pi_cycle
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["prepare"]["ok"] is True
    assert payload["full_cycle"]["ok"] is True
    assert payload["full_cycle"]["runner_path"] == str(runner)
    assert payload["full_cycle"]["returncode"] == 0
    assert payload["full_cycle"]["latest_envelope"] == str(artifact_path)
    assert payload["full_cycle"]["retro_artifact_rotated"] is True
    assert "pi_cycle_launchd done rc=0" in "\n".join(payload["full_cycle"]["log_tail"])


def test_run_local_pi_recovery_lab_forwards_expected_args(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    review_dir = workspace_root / "output" / "review"
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_recovery_lab_parent_dir="{review_dir}"
        local_pi_recovery_lab_keep="9"
        local_pi_recovery_lab_ttl_hours="96"
        local_pi_recovery_lab_allow_fallback_write="true"
        python3() {{
          printf 'PYTHON3:%s\\n' "$*"
        }}
        action_run_local_pi_recovery_lab
        """,
    )

    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("PYTHON3:")
    assert "scripts/run_local_pi_recovery_lab.py" in line
    assert f"--workspace-system-root {workspace_root}" in line
    assert f"--lab-parent-dir {review_dir}" in line
    assert "--artifact-ttl-hours 96" in line
    assert "--keep-labs 9" in line
    assert "--allow-fallback-write" in line


def test_snapshot_local_pi_recovery_state_forwards_expected_args(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    checkpoint_dir = workspace_root / "output" / "review" / "local_pi_recovery_checkpoints"
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_recovery_checkpoint_dir="{checkpoint_dir}"
        local_pi_recovery_checkpoint_keep="15"
        local_pi_recovery_checkpoint_max_age_hours="240"
        local_pi_recovery_checkpoint_note="before-real-write"
        python3() {{
          printf 'PYTHON3:%s\\n' "$*"
        }}
        action_snapshot_local_pi_recovery_state
        """,
    )

    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("PYTHON3:")
    assert "scripts/snapshot_local_pi_recovery_state.py" in line
    assert f"--workspace-system-root {workspace_root}" in line
    assert f"--checkpoint-dir {checkpoint_dir}" in line
    assert "--checkpoint-keep 15" in line
    assert "--checkpoint-max-age-hours 240" in line
    assert "--note before-real-write" in line


def test_restore_local_pi_recovery_state_forwards_expected_args(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_recovery_restore_checkpoint="/tmp/checkpoint.json"
        local_pi_recovery_restore_write="false"
        local_pi_recovery_restore_expected_state_fingerprint="abc123"
        python3() {{
          printf 'PYTHON3:%s\\n' "$*"
        }}
        action_restore_local_pi_recovery_state
        """,
    )

    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("PYTHON3:")
    assert "scripts/restore_local_pi_recovery_state.py" in line
    assert "--checkpoint /tmp/checkpoint.json" in line
    assert "--expected-current-state-fingerprint abc123" in line
    assert "--dry-run" in line


def test_rollback_local_pi_recovery_state_uses_latest_checkpoint_when_not_set(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    older = checkpoint_dir / "20260308T120000Z_local_pi_recovery_checkpoint"
    latest = checkpoint_dir / "20260308T121000Z_local_pi_recovery_checkpoint"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)
    (older / "checkpoint.json").write_text("{}", encoding="utf-8")
    (latest / "checkpoint.json").write_text("{}", encoding="utf-8")
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_recovery_checkpoint_dir="{checkpoint_dir}"
        action_restore_local_pi_recovery_state() {{
          printf '%s\\n' "${{local_pi_recovery_restore_checkpoint}}"
        }}
        action_rollback_local_pi_recovery_state
        """,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(latest / "checkpoint.json")


def test_ack_local_pi_consecutive_loss_guardrail_runs_local_script(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    scripts_dir = workspace_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "ack_paper_consecutive_loss_guardrail.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}, ensure_ascii=False))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        LOCAL_PI_EXPECTED_STATE_FINGERPRINT="abc123"
        local_pi_consecutive_loss_ack_allow_missing_last_loss_ts="true"
        local_pi_consecutive_loss_ack_write="true"
        local_pi_consecutive_loss_ack_note="manual-check"
        action_ack_local_pi_consecutive_loss_guardrail
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    argv = list(payload["argv"])
    assert "--expected-state-fingerprint" in argv
    assert "abc123" in argv
    assert "--allow-missing-last-loss-ts" in argv
    assert "--write" in argv
    assert "--note" in argv
    assert "manual-check" in argv


def test_local_pi_recovery_handoff_runs_local_script(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pi" / "fenlie-system"
    scripts_dir = workspace_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "build_local_pi_recovery_handoff.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}, ensure_ascii=False))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_workspace_system_root="{workspace_root}"
        local_pi_recovery_handoff_keep="9"
        artifact_ttl_hours="72"
        action_local_pi_recovery_handoff
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    argv = list(payload["argv"])
    assert "--workspace-system-root" in argv
    assert str(workspace_root) in argv
    assert "--artifact-ttl-hours" in argv
    assert "72" in argv
    assert "--artifact-keep" in argv
    assert "9" in argv


def test_remote_live_handoff_runs_local_script(tmp_path: Path) -> None:
    body = """
        output_dir="__TMP__"
        cat >"__TMP__/20260309T042253Z_remote_live_noaf_probe.json" <<'JSON'
        {"action":"build_remote_live_noaf_probe","ok":false,"status":"incompatible"}
        JSON
        remote_live_handoff_keep="9"
        action_live_takeover_ready_check() {
          cat <<'JSON'
        {"action":"live-takeover-ready-check","ready":false,"reasons":["risk_guard_blocked"]}
        JSON
          return 3
        }
        action_live_risk_daemon_status() {
          cat <<'JSON'
        {"action":"live-risk-daemon-status","payload":{"running":true,"status":"running"}}
        JSON
        }
        action_live_ops_reconcile_status() {
          cat <<'JSON'
        {"action":"live-ops-reconcile-status","ok":true,"status":"passed"}
        JSON
        }
        action_live_risk_daemon_journal() {
          cat <<'JSON'
        {"action":"live-risk-daemon-journal","lines":["line1","line2"]}
        JSON
        }
        action_live_risk_daemon_security_status() {
          cat <<'JSON'
        {"action":"live-risk-daemon-security-status","verify":{"ok":true},"security":{"overall_exposure":3.2,"overall_rating":"OK"}}
        JSON
        }
        python3() {
          printf 'PYTHON3:%s\\n' "$*"
        }
        action_remote_live_handoff
        """.replace("__TMP__", str(tmp_path))
    proc = _run_bridge_shell(
        tmp_path,
        body,
    )

    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("PYTHON3:")
    assert "scripts/build_remote_live_handoff.py" in line
    assert "--ready-check-returncode 3" in line
    assert "--risk-daemon-status-returncode 0" in line
    assert "--ops-status-returncode 0" in line
    assert "--journal-returncode 0" in line
    assert "--security-status-returncode 0" in line
    assert f"--noaf-probe-file {tmp_path}/20260309T042253Z_remote_live_noaf_probe.json" in line
    assert "--security-accept-max-exposure 4.0" in line
    assert "--artifact-keep 9" in line


def test_live_risk_daemon_security_status_action_uses_remote_helper(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        run_live_risk_daemon_security_status_remote() {
          cat <<'JSON'
        {"action":"live-risk-daemon-security-status","verify":{"ok":true},"security":{"overall_exposure":3.2,"overall_rating":"OK"}}
        JSON
        }
        action_live_risk_daemon_security_status
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "live-risk-daemon-security-status"
    assert payload["verify"]["ok"] is True
    assert payload["security"]["overall_exposure"] == 3.2


def test_live_risk_daemon_install_service_command_includes_verify_and_security_checks(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        live_risk_daemon_security_accept_max_exposure="2.5"
        run_live_takeover_remote() {
          printf 'ACTION:%s\\n' "$1"
          printf 'CMD:%s\\n' "$2"
        }
        action_live_risk_daemon_install_service
        """,
    )

    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    assert "ACTION:live-risk-daemon-install-service" in text
    assert "systemd-analyze', 'verify'" in text
    assert "systemd-analyze', 'security'" in text
    assert "payload_alignment = evaluate_payload_alignment(payload)" in text
    assert "alignment_wait = {" in text
    assert "'verify': {" in text
    assert "'security': {" in text
    assert "'security_acceptance': {" in text
    assert "security_exposure_above_threshold" in text
    assert "'2.5' <<'PY'" in text


def test_apply_local_pi_recovery_step_previews_fallback_backfill(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        action_local_pi_consecutive_loss_guardrail_status() {{
          cat <<'JSON'
        {json.dumps({
            "action": "paper_consecutive_loss_guardrail_status",
            "ok": True,
            "state_fingerprint": "abc123",
            "recovery_plan": {
                "next_action": "review_fallback_last_loss_ts_backfill",
                "action_level": "review",
                "reason": "strict_backfill_unavailable_but_fallback_candidate_available",
            },
            "write_projection": {
                "write_chain_possible": True,
                "terminal_action": "full_cycle_gate_ready",
            },
        }, ensure_ascii=False)}
        JSON
        }}
        action_backfill_local_pi_last_loss_ts() {{
          printf '%s\\n' "{{\\"action\\":\\"backfill_local_pi_last_loss_ts\\",\\"ok\\":true,\\"allow_latest_loss_fallback\\":\\"${{local_pi_last_loss_ts_backfill_allow_latest_loss_fallback:-}}\\",\\"write\\":\\"${{local_pi_last_loss_ts_backfill_write:-}}\\",\\"expected_state_fingerprint\\":\\"${{LOCAL_PI_EXPECTED_STATE_FINGERPRINT:-}}\\"}}"
        }}
        action_apply_local_pi_recovery_step
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["next_action"] == "review_fallback_last_loss_ts_backfill"
    assert payload["execution_mode"] == "fallback_backfill_preview"
    assert payload["snapshot_required"] is False
    assert payload["snapshot_attempted"] is False
    assert payload["step_attempted"] is True
    assert payload["mutation_detected"] is False
    assert payload["verify_after_step"] is True
    assert payload["write_projection"]["terminal_action"] == "full_cycle_gate_ready"
    assert payload["step_result"]["expected_state_fingerprint"] == "abc123"
    assert payload["step_result"]["allow_latest_loss_fallback"] == "true"
    assert payload["step_result"]["write"] == "false"
    assert payload["post_status_payload"]["recovery_plan"]["next_action"] == "review_fallback_last_loss_ts_backfill"
    assert payload["post_next_action"] == "review_fallback_last_loss_ts_backfill"
    assert payload["post_write_projection"]["write_chain_possible"] is True


def test_apply_local_pi_recovery_step_writes_manual_ack_when_enabled(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_recovery_apply_write="true"
        action_local_pi_consecutive_loss_guardrail_status() {{
          cat <<'JSON'
        {json.dumps({
            "action": "paper_consecutive_loss_guardrail_status",
            "ok": True,
            "state_fingerprint": "abc123",
            "recovery_plan": {
                "next_action": "write_manual_ack",
                "action_level": "write",
                "reason": "manual_ack_eligible",
            },
        }, ensure_ascii=False)}
        JSON
        }}
        action_snapshot_local_pi_recovery_state() {{
          cat <<'JSON'
        {json.dumps({
            "action": "snapshot_local_pi_recovery_state",
            "ok": True,
            "checkpoint_root": "/tmp/checkpoint",
            "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json",
            "checkpoint_checksum": "/tmp/checkpoint/checkpoint_checksum.json",
            "state_fingerprint": "abc123",
        }, ensure_ascii=False)}
        JSON
        }}
        action_ack_local_pi_consecutive_loss_guardrail() {{
          printf '%s\\n' "{{\\"action\\":\\"ack_local_pi_consecutive_loss_guardrail\\",\\"ok\\":true,\\"write\\":\\"${{local_pi_consecutive_loss_ack_write:-}}\\",\\"expected_state_fingerprint\\":\\"${{LOCAL_PI_EXPECTED_STATE_FINGERPRINT:-}}\\"}}"
        }}
        action_apply_local_pi_recovery_step
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["next_action"] == "write_manual_ack"
    assert payload["execution_mode"] == "ack_write"
    assert payload["snapshot_required"] is True
    assert payload["snapshot_attempted"] is True
    assert payload["snapshot_result"]["checkpoint_root"] == "/tmp/checkpoint"
    assert payload["rollback_guidance"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert "rollback-local-pi-recovery-state" in payload["rollback_guidance"]["dry_run_command"]
    assert "LOCAL_PI_RECOVERY_RESTORE_WRITE=true" in payload["rollback_guidance"]["write_command"]
    assert payload["operator_note"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert "rollback path" in payload["operator_note"]["summary"]
    assert payload["step_attempted"] is True
    assert payload["mutation_detected"] is False
    assert payload["step_result"]["expected_state_fingerprint"] == "abc123"
    assert payload["step_result"]["write"] == "true"
    assert payload["post_status_payload"]["recovery_plan"]["next_action"] == "write_manual_ack"
    assert payload["post_next_action"] == "write_manual_ack"


def test_apply_local_pi_recovery_step_stops_when_auto_snapshot_fails(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_recovery_apply_write="true"
        action_local_pi_consecutive_loss_guardrail_status() {{
          cat <<'JSON'
        {json.dumps({
            "action": "paper_consecutive_loss_guardrail_status",
            "ok": True,
            "state_fingerprint": "abc123",
            "recovery_plan": {
                "next_action": "write_manual_ack",
                "action_level": "write",
                "reason": "manual_ack_eligible",
            },
        }, ensure_ascii=False)}
        JSON
        }}
        action_snapshot_local_pi_recovery_state() {{
          cat <<'JSON'
        {json.dumps({
            "action": "snapshot_local_pi_recovery_state",
            "ok": False,
            "status": "lock_timeout",
        }, ensure_ascii=False)}
        JSON
          return 9
        }}
        action_ack_local_pi_consecutive_loss_guardrail() {{
          printf '%s\\n' '{{"action":"ack_local_pi_consecutive_loss_guardrail","ok":true}}'
        }}
        tmp_out="$(mktemp)"
        rc=0
        action_apply_local_pi_recovery_step >"${{tmp_out}}" || rc=$?
        printf 'RC=%s\\n' "${{rc}}"
        cat "${{tmp_out}}"
        """,
    )

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == "RC=9"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["ok"] is False
    assert payload["status"] == "snapshot_failed"
    assert payload["execution_mode"] == "snapshot_failed"
    assert payload["snapshot_attempted"] is True
    assert payload["snapshot_result"]["status"] == "lock_timeout"
    assert payload["step_attempted"] is False


def test_run_local_pi_recovery_flow_stops_after_preview_step(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        local_pi_recovery_artifacts_enabled="false"
        action_apply_local_pi_recovery_step() {
          cat <<'JSON'
        {
          "action": "apply_local_pi_recovery_step",
          "ok": true,
          "next_action": "review_fallback_last_loss_ts_backfill",
          "action_level": "review",
          "step_attempted": true,
          "execution_mode": "fallback_backfill_preview",
          "write_projection": {
            "write_chain_possible": true,
            "terminal_action": "full_cycle_gate_ready"
          },
          "post_next_action": "review_fallback_last_loss_ts_backfill",
          "post_action_level": "review",
          "post_write_projection": {
            "write_chain_possible": true,
            "terminal_action": "full_cycle_gate_ready"
          },
          "post_status_payload": {
            "recovery_plan": {
              "next_action": "review_fallback_last_loss_ts_backfill",
              "action_level": "review"
            }
          }
        }
        JSON
        }
        action_run_local_pi_recovery_flow
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["step_count"] == 1
    assert payload["stopped_reason"] == "preview_mode"
    assert payload["final_next_action"] == "review_fallback_last_loss_ts_backfill"
    assert payload["initial_write_projection"]["terminal_action"] == "full_cycle_gate_ready"
    assert payload["final_write_projection"]["write_chain_possible"] is True


def test_run_local_pi_recovery_flow_chains_until_no_follow_up_action(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        local_pi_recovery_artifacts_enabled="false"
        local_pi_recovery_enforce_projection="false"
        recovery_counter="$(mktemp)"
        printf '0' >"${recovery_counter}"
        action_apply_local_pi_recovery_step() {
          count="$(cat "${recovery_counter}")"
          if [ "${count}" = "0" ]; then
            printf '1' >"${recovery_counter}"
            cat <<'JSON'
        {
          "action": "apply_local_pi_recovery_step",
          "ok": true,
          "next_action": "review_fallback_last_loss_ts_backfill",
          "action_level": "review",
          "step_attempted": true,
          "execution_mode": "fallback_backfill_write",
          "post_next_action": "write_manual_ack",
          "post_action_level": "write",
          "post_status_payload": {
            "recovery_plan": {
              "next_action": "write_manual_ack",
              "action_level": "write"
            }
          }
        }
        JSON
          elif [ "${count}" = "1" ]; then
            printf '2' >"${recovery_counter}"
            cat <<'JSON'
        {
          "action": "apply_local_pi_recovery_step",
          "ok": true,
          "next_action": "write_manual_ack",
          "action_level": "write",
          "step_attempted": true,
          "execution_mode": "ack_write",
          "post_next_action": "run_full_cycle_with_existing_ack",
          "post_action_level": "execute",
          "post_status_payload": {
            "recovery_plan": {
              "next_action": "run_full_cycle_with_existing_ack",
              "action_level": "execute"
            }
          }
        }
        JSON
          else
            cat <<'JSON'
        {
          "action": "apply_local_pi_recovery_step",
          "ok": true,
          "next_action": "run_full_cycle_with_existing_ack",
          "action_level": "execute",
          "step_attempted": true,
          "execution_mode": "full_cycle",
          "post_next_action": "",
          "post_action_level": "",
          "post_status_payload": {
            "recovery_plan": {
              "next_action": "",
              "action_level": ""
            }
          }
        }
        JSON
          fi
        }
        action_run_local_pi_recovery_flow
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["step_count"] == 3
    assert payload["stopped_reason"] == "no_follow_up_action"
    assert payload["final_next_action"] == "run_full_cycle_with_existing_ack"


def test_run_local_pi_recovery_flow_stops_on_projection_mismatch(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        local_pi_recovery_artifacts_enabled="false"
        action_apply_local_pi_recovery_step() {
          cat <<'JSON'
        {
          "action": "apply_local_pi_recovery_step",
          "ok": true,
          "next_action": "write_manual_ack",
          "action_level": "write",
          "step_attempted": true,
          "execution_mode": "ack_write",
          "write_projection": {
            "projected_steps": [
              {
                "simulated_step": "fallback_backfill_write",
                "projected_next_action": "run_full_cycle_with_existing_ack"
              }
            ]
          },
          "post_next_action": "run_full_cycle_with_existing_ack",
          "post_action_level": "execute",
          "post_status_payload": {
            "recovery_plan": {
              "next_action": "run_full_cycle_with_existing_ack",
              "action_level": "execute"
            }
          }
        }
        JSON
        }
        action_run_local_pi_recovery_flow
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["step_count"] == 1
    assert payload["projection_guard_enforced"] is True
    assert payload["stopped_reason"] == "projection_step_mismatch"
    assert payload["rollback_attempted"] is False


def test_run_local_pi_recovery_flow_attempts_rollback_after_failed_write_step(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        local_pi_recovery_artifacts_enabled="false"
        action_apply_local_pi_recovery_step() {
          cat <<'JSON'
        {
          "action": "apply_local_pi_recovery_step",
          "ok": false,
          "status": "step_failed",
          "next_action": "write_manual_ack",
          "action_level": "write",
          "snapshot_attempted": true,
          "snapshot_result": {
            "ok": true,
            "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json"
          },
          "step_attempted": true,
          "execution_mode": "ack_write",
          "mutation_detected": true,
          "mutation_detected_reasons": ["ack_presence_changed"],
          "step_result": {
            "write_performed": true
          }
        }
        JSON
          return 9
        }
        action_rollback_local_pi_recovery_state() {
          cat <<'JSON'
        {
          "action": "restore_local_pi_recovery_state",
          "ok": true,
          "status": "dry_run",
          "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json"
        }
        JSON
        }
        tmp_out="$(mktemp)"
        rc=0
        action_run_local_pi_recovery_flow >"${tmp_out}" || rc=$?
        printf 'RC=%s\\n' "${rc}"
        cat "${tmp_out}"
        """,
    )

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == "RC=9"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["ok"] is False
    assert payload["stopped_reason"] == "step_failed"
    assert payload["rollback_attempted"] is True
    assert payload["rollback_write_requested"] is False
    assert payload["rollback_checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert payload["rollback_result"]["status"] == "dry_run"
    assert payload["rollback_guidance"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert "rollback-local-pi-recovery-state" in payload["rollback_guidance"]["dry_run_command"]
    assert payload["operator_note"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert "Rollback hook was attempted" in payload["operator_note"]["summary"]


def test_run_local_pi_recovery_flow_skips_rollback_when_failed_write_has_no_mutation(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        local_pi_recovery_artifacts_enabled="false"
        action_apply_local_pi_recovery_step() {
          cat <<'JSON'
        {
          "action": "apply_local_pi_recovery_step",
          "ok": false,
          "status": "step_failed",
          "next_action": "write_manual_ack",
          "action_level": "write",
          "snapshot_attempted": true,
          "snapshot_result": {
            "ok": true,
            "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json"
          },
          "step_attempted": true,
          "execution_mode": "ack_write",
          "mutation_detected": false,
          "mutation_detected_reasons": []
        }
        JSON
          return 9
        }
        action_rollback_local_pi_recovery_state() {
          cat <<'JSON'
        {
          "action": "restore_local_pi_recovery_state",
          "ok": true,
          "status": "dry_run",
          "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json"
        }
        JSON
        }
        tmp_out="$(mktemp)"
        rc=0
        action_run_local_pi_recovery_flow >"${tmp_out}" || rc=$?
        printf 'RC=%s\\n' "${rc}"
        cat "${tmp_out}"
        """,
    )

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == "RC=9"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["ok"] is False
    assert payload["stopped_reason"] == "step_failed"
    assert payload["rollback_attempted"] is False
    assert payload["rollback_result"] is None


def test_run_local_pi_recovery_flow_writes_flow_and_step_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "review"
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        local_pi_recovery_artifacts_enabled="true"
        local_pi_recovery_artifact_dir="{artifact_dir}"
        action_apply_local_pi_recovery_step() {{
          cat <<'JSON'
        {{
          "action": "apply_local_pi_recovery_step",
          "ok": true,
          "next_action": "review_fallback_last_loss_ts_backfill",
          "action_level": "review",
          "snapshot_attempted": true,
          "snapshot_result": {{
            "ok": true,
            "checkpoint_root": "/tmp/checkpoint",
            "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json",
            "checkpoint_checksum": "/tmp/checkpoint/checkpoint_checksum.json",
            "state_fingerprint": "abc123"
          }},
          "rollback_guidance": {{
            "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json",
            "dry_run_command": "dry-run",
            "write_command": "write"
          }},
          "operator_note": {{
            "checkpoint_manifest": "/tmp/checkpoint/checkpoint.json",
            "summary": "stub-note"
          }},
          "step_attempted": true,
          "execution_mode": "fallback_backfill_preview",
          "post_next_action": "review_fallback_last_loss_ts_backfill",
          "post_action_level": "review",
          "post_status_payload": {{
            "recovery_plan": {{
              "next_action": "review_fallback_last_loss_ts_backfill",
              "action_level": "review"
            }}
          }}
        }}
        JSON
        }}
        action_run_local_pi_recovery_flow
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["artifact"]
    assert payload["checksum"]
    assert payload["checkpoint_count"] == 1
    assert payload["checkpoints"][0]["checkpoint_root"] == "/tmp/checkpoint"
    assert payload["rollback_guidance"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert payload["operator_note"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert payload["artifact_status_label"] == "preview"
    assert payload["artifact_label"] == "recovery-flow:preview_mode"
    assert "recovery-flow" in payload["artifact_tags"]
    assert len(payload["step_artifacts"]) == 1
    flow_artifact = Path(payload["artifact"])
    flow_checksum = Path(payload["checksum"])
    step_artifact = Path(payload["step_artifacts"][0]["artifact"])
    step_checksum = Path(payload["step_artifacts"][0]["checksum"])
    assert flow_artifact.exists()
    assert flow_checksum.exists()
    assert step_artifact.exists()
    assert step_checksum.exists()
    flow_payload = json.loads(flow_artifact.read_text(encoding="utf-8"))
    step_payload = json.loads(step_artifact.read_text(encoding="utf-8"))
    assert flow_payload["artifact_status_label"] == "preview"
    assert flow_payload["artifact_label"] == "recovery-flow:preview_mode"
    assert "recovery-flow" in flow_payload["artifact_tags"]
    assert step_payload["artifact_status_label"] == "preview"
    assert step_payload["artifact_label"] == "recovery-step:fallback_backfill_preview"
    assert "recovery-step" in step_payload["artifact_tags"]
    assert "preview" in step_payload["artifact_tags"]
    assert step_payload["operator_note"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"
    assert step_payload["rollback_guidance"]["checkpoint_manifest"] == "/tmp/checkpoint/checkpoint.json"


def test_remote_live_notification_preview_builds_preview_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        action_remote_live_handoff() {{
          cat <<'JSON'
        {_stub_remote_live_handoff_payload()}
        JSON
        }}
        action_remote_live_notification_preview
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_notification_preview"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["handoff_state"] == "ops_live_gate_blocked"
    assert payload["notification"]["title"] == "Remote gate is blocking live execution"
    assert payload["notification_templates"]["telegram"]["parse_mode"] == "MarkdownV2"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_remote_live_notification_dry_run_builds_request_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        action_remote_live_notification_preview() {{
          cat <<'JSON'
        {{
          "action": "build_remote_live_notification_preview",
          "ok": true,
          "status": "ok",
          "handoff_state": "ops_live_gate_blocked",
          "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
          "next_focus_area": "gate",
          "notification_templates": {{
            "telegram": {{
              "parse_mode": "MarkdownV2",
              "text": "telegram body",
              "disable_web_page_preview": true
            }},
            "feishu": {{
              "msg_type": "text",
              "content": {{"text": "feishu body"}}
            }}
          }}
        }}
        JSON
        }}
        action_remote_live_notification_dry_run
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_notification_dry_run"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["telegram"]["request"]["url"] == "https://api.telegram.org/bot<TOKEN>/sendMessage"
    assert payload["feishu"]["request"]["url"] == "https://open.feishu.cn/open-apis/bot/v2/hook/<TOKEN>"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_remote_live_notification_send_defaults_to_delivery_none(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        state_dir="{tmp_path}/state"
        remote_live_notification_delivery="none"
        action_remote_live_notification_dry_run() {{
          cat <<'JSON'
        {{
          "action": "build_remote_live_notification_dry_run",
          "ok": true,
          "status": "ok",
          "handoff_state": "ops_live_gate_blocked",
          "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
          "next_focus_area": "gate",
          "telegram": {{
            "ok": true,
            "request": {{
              "method": "POST",
              "url": "https://api.telegram.org/bot<TOKEN>/sendMessage",
              "headers": {{"Content-Type": "application/json"}},
              "json_body": {{
                "chat_id": "<CHAT_ID>",
                "text": "telegram body",
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": true
              }}
            }}
          }},
          "feishu": {{
            "ok": true,
            "request": {{
              "method": "POST",
              "url": "https://open.feishu.cn/open-apis/bot/v2/hook/<TOKEN>",
              "headers": {{"Content-Type": "application/json"}},
              "json_body": {{
                "msg_type": "text",
                "content": {{"text": "feishu body"}}
              }}
            }}
          }}
        }}
        JSON
        }}
        action_remote_live_notification_send
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "send_remote_live_notification"
    assert payload["ok"] is True
    assert payload["status"] == "delivery_none"
    assert payload["delivery_requested"] == "none"
    assert payload["channels_attempted"] == []
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_mdwe_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-mdwe-probe-123",
  "returncode": 0,
  "properties": {{
    "MemoryDenyWriteExecute": True,
    "NoNewPrivileges": True
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_mdwe_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_mdwe_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["MemoryDenyWriteExecute"] is True
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_protecthome_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-protecthome-probe-123",
  "returncode": 0,
  "properties": {{
    "ProtectHome": "read-only",
    "MemoryDenyWriteExecute": True
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_protecthome_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_protecthome_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["ProtectHome"] == "read-only"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_procsubset_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-procsubset-probe-123",
  "returncode": 0,
  "properties": {{
    "ProtectProc": "invisible",
    "ProtectHome": "read-only",
    "ProcSubset": "pid"
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_procsubset_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_procsubset_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["ProcSubset"] == "pid"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_privateusers_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-privateusers-probe-123",
  "returncode": 0,
  "properties": {{
    "PrivateUsers": True,
    "ProcSubset": "pid"
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_privateusers_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_privateusers_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["PrivateUsers"] is True
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_privatenetwork_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-privatenetwork-probe-123",
  "returncode": 0,
  "properties": {{
    "PrivateNetwork": True,
    "PrivateUsers": True
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_privatenetwork_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_privatenetwork_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["PrivateNetwork"] is True
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_ipdeny_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-ipdeny-probe-123",
  "returncode": 0,
  "properties": {{
    "IPAddressDeny": "any",
    "PrivateNetwork": True
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_ipdeny_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_ipdeny_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["IPAddressDeny"] == "any"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_afunix_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-afunix-probe-123",
  "returncode": 0,
  "properties": {{
    "RestrictAddressFamilies": "AF_UNIX",
    "IPAddressDeny": "any"
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_afunix_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_afunix_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["RestrictAddressFamilies"] == "AF_UNIX"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_devicepolicy_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-devicepolicy-probe-123",
  "returncode": 0,
  "properties": {{
    "DevicePolicy": "closed",
    "PrivateDevices": True
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_devicepolicy_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_devicepolicy_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["DevicePolicy"] == "closed"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_noaf_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-noaf-probe-123",
  "returncode": 0,
  "properties": {{
    "RestrictAddressFamilies": "none",
    "IPAddressDeny": "any",
    "PrivateNetwork": True
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_noaf_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_noaf_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["RestrictAddressFamilies"] == "none"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_syscallfilter_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-syscallfilter-probe-123",
  "returncode": 0,
  "properties": {{
    "SystemCallFilter": "@system-service",
    "RestrictAddressFamilies": "AF_UNIX"
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_syscallfilter_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_syscallfilter_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["SystemCallFilter"] == "@system-service"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_risk_daemon_syscallfilter_tight_probe_builds_artifact(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        f"""
        output_dir="{tmp_path}"
        run_live_takeover_remote() {{
          python3 - "$1" <<'PY'
import json
import sys
out = {{
  "action": sys.argv[1],
  "ok": True,
  "status": "compatible",
  "unit": "fenlie-live-risk-daemon-syscallfilter-tight-probe-123",
  "returncode": 0,
  "properties": {{
    "SystemCallFilter": ["@system-service", "~@resources", "~@privileged"],
    "RestrictAddressFamilies": "AF_UNIX"
  }}
}}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
        }}
        action_live_risk_daemon_syscallfilter_tight_probe
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "build_remote_live_syscallfilter_tight_probe"
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["probe"]["status"] == "compatible"
    assert payload["probe"]["properties"]["SystemCallFilter"] == [
        "@system-service",
        "~@resources",
        "~@privileged",
    ]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_live_ops_reconcile_refresh_requires_artifact_mtime_advance(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        ssh_exec() {
          if [[ "$1" == *"python3 - '2026-03-10' '7'"* ]]; then
            cat <<'JSON'
{
  "action": "live-ops-reconcile-refresh",
  "status": "started",
  "pid": 321,
  "started": true,
  "date": "2026-03-10",
  "window_days": 7,
  "artifact_path": "output/review/2026-03-10_ops_report.json",
  "artifact_existed_before": true,
  "artifact_mtime_ns_before": 100,
  "stdout_log": "stdout.log",
  "stderr_log": "stderr.log"
}
JSON
          elif [[ "$1" == *"python3 - '321' '2026-03-10'"* ]]; then
            cat <<'JSON'
{
  "action": "live-ops-reconcile-refresh",
  "pid": 321,
  "alive": false,
  "finished": true,
  "artifact_exists": true,
  "artifact_updated": false,
  "artifact_mtime_ns": 100,
  "artifact_mtime_ns_before": 100,
  "artifact_path": "output/review/2026-03-10_ops_report.json",
  "stdout_log": "stdout.log",
  "stderr_log": "stderr.log",
  "stdout_tail": [],
  "stderr_tail": ["stale artifact"]
}
JSON
          else
            echo "unexpected ssh_exec call: $1" >&2
            return 1
          fi
        }
        live_takeover_date=2026-03-10
        set +e
        out="$(run_live_ops_reconcile_refresh_remote)"
        rc=$?
        set -e
        printf 'RC=%s\n' "${rc}"
        printf '%s\n' "${out}"
        """,
    )

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines and lines[0] == "RC=2"
    payload = json.loads("\n".join(lines[1:]))
    assert payload["artifact_exists"] is True
    assert payload["artifact_updated"] is False
    assert payload["artifact_mtime_ns_before"] == 100


def test_live_ops_reconcile_refresh_calls_status_only_after_artifact_update(tmp_path: Path) -> None:
    proc = _run_bridge_shell(
        tmp_path,
        """
        ssh_exec() {
          if [[ "$1" == *"python3 - '2026-03-10' '7'"* ]]; then
            cat <<'JSON'
{
  "action": "live-ops-reconcile-refresh",
  "status": "started",
  "pid": 654,
  "started": true,
  "date": "2026-03-10",
  "window_days": 7,
  "artifact_path": "output/review/2026-03-10_ops_report.json",
  "artifact_existed_before": true,
  "artifact_mtime_ns_before": 100,
  "stdout_log": "stdout.log",
  "stderr_log": "stderr.log"
}
JSON
          elif [[ "$1" == *"python3 - '654' '2026-03-10'"* ]]; then
            cat <<'JSON'
{
  "action": "live-ops-reconcile-refresh",
  "pid": 654,
  "alive": false,
  "finished": true,
  "artifact_exists": true,
  "artifact_updated": true,
  "artifact_mtime_ns": 200,
  "artifact_mtime_ns_before": 100,
  "artifact_path": "output/review/2026-03-10_ops_report.json",
  "stdout_log": "stdout.log",
  "stderr_log": "stderr.log",
  "stdout_tail": ["ok"],
  "stderr_tail": []
}
JSON
          else
            echo "unexpected ssh_exec call: $1" >&2
            return 1
          fi
        }
        run_live_ops_reconcile_status_remote() {
          cat <<'JSON'
{
  "action": "live-ops-reconcile-status",
  "ok": true,
  "status": "passed",
  "artifact_path": "output/review/2026-03-10_ops_report.json"
}
JSON
        }
        live_takeover_date=2026-03-10
        out="$(run_live_ops_reconcile_refresh_remote)"
        printf '%s\n' "${out}"
        """,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "live-ops-reconcile-status"
    assert payload["ok"] is True
    assert payload["status"] == "passed"
