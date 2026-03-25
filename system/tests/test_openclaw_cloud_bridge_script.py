from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "openclaw_cloud_bridge.sh"


class TestOpenClawCloudBridgeScript:
    def _write_executable(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _fake_ssh_script(self, log_path: Path) -> str:
        return """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

log_path = Path(os.environ[\"FAKE_SSH_LOG\"])
log_path.parent.mkdir(parents=True, exist_ok=True)
argv = sys.argv[1:]
remote_cmd = argv[-1] if argv else ""
with log_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": argv, "remote_cmd": remote_cmd}) + "\\n")

mode = "default"
if "--mode autopilot-check" in remote_cmd:
    mode = os.environ.get("FAKE_SSH_AUTOPILOT_MODE", "blocked")
    if mode == "blocked":
        payload = {
            "ok": True,
            "mode": "autopilot_check",
            "autopilot_allowed": False,
            "reason": "daily_budget_cap_reached",
            "steps": {"budget": {"within_cap": False}},
        }
    elif mode == "string-false":
        payload = {"ok": True, "mode": "autopilot_check", "autopilot_allowed": "false", "reason": "typed_wrong"}
    elif mode == "object":
        payload = {"ok": True, "mode": "autopilot_check", "autopilot_allowed": {"blocked": True}, "reason": "typed_wrong"}
    elif mode == "invalid-json":
        print("{not-json")
        raise SystemExit(0)
    else:
        payload = {"ok": True, "mode": "autopilot_check", "autopilot_allowed": True}
    print(json.dumps(payload))
    raise SystemExit(0)

print(json.dumps({"ok": True, "remote_cmd": remote_cmd}))
"""

    def _run_bridge(self, action: str, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fakebin = td_path / "fakebin"
            ssh_log = td_path / "ssh_log.jsonl"
            self._write_executable(fakebin / "ssh", self._fake_ssh_script(ssh_log))
            env = dict(os.environ)
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["FAKE_SSH_LOG"] = str(ssh_log)
            env.setdefault("CLOUD_HOST", "127.0.0.1")
            env.setdefault("CLOUD_USER", "ubuntu")
            env.setdefault("CLOUD_PROJECT_DIR", "/remote/openclaw-system")
            if extra_env:
                env.update(extra_env)
            proc = subprocess.run(
                [str(SCRIPT_PATH), action],
                cwd=SYSTEM_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if ssh_log.exists():
                rows = [json.loads(line) for line in ssh_log.read_text(encoding="utf-8").splitlines() if line.strip()]
                proc.remote_cmds = [str(row.get("remote_cmd", "")) for row in rows]  # type: ignore[attr-defined]
            else:
                proc.remote_cmds = []  # type: ignore[attr-defined]
            return proc

    def _read_remote_cmds(self, proc: subprocess.CompletedProcess[str]) -> list[str]:
        return list(getattr(proc, "remote_cmds", []))

    def test_usage_lists_infra_canary_actions_and_env_defaults(self) -> None:
        proc = subprocess.run(
            [str(SCRIPT_PATH)],
            cwd=SYSTEM_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert proc.returncode == 2
        assert "infra-canary-probe" in proc.stdout
        assert "infra-canary-run" in proc.stdout
        assert "infra-canary-autopilot" in proc.stdout
        assert "INFRA_CANARY_SYMBOL" in proc.stdout
        assert "INFRA_CANARY_QUOTE_USDT" in proc.stdout
        assert "INFRA_CANARY_SINGLE_RUN_CAP_USDT" in proc.stdout
        assert "INFRA_CANARY_DAILY_BUDGET_CAP_USDT" in proc.stdout
        assert "INFRA_CANARY_ALLOW_DUST" in proc.stdout

    def test_infra_canary_probe_builds_remote_command_with_defaults(self) -> None:
        proc = self._run_bridge("infra-canary-probe")

        assert proc.returncode == 0, proc.stderr
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 1
        remote_cmd = remote_cmds[0]
        assert "scripts/binance_infra_canary.py" in remote_cmd
        assert "--mode probe" in remote_cmd
        assert "--symbol BTCUSDT" in remote_cmd
        assert "--quote-usdt 10" in remote_cmd
        assert "--single-run-cap-usdt 12" in remote_cmd
        assert "--daily-budget-cap-usdt 20" in remote_cmd
        assert "--allow-dust" in remote_cmd

    def test_infra_canary_run_builds_remote_command_with_run_mode(self) -> None:
        proc = self._run_bridge("infra-canary-run")

        assert proc.returncode == 0, proc.stderr
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 1
        remote_cmd = remote_cmds[0]
        assert "scripts/binance_infra_canary.py" in remote_cmd
        assert "--mode run" in remote_cmd
        assert "--symbol BTCUSDT" in remote_cmd
        assert "--quote-usdt 10" in remote_cmd
        assert "--single-run-cap-usdt 12" in remote_cmd
        assert "--daily-budget-cap-usdt 20" in remote_cmd
        assert "--allow-dust" in remote_cmd

    def test_infra_canary_probe_forwards_local_creds_and_daemon_env_fallback(self) -> None:
        proc = self._run_bridge(
            "infra-canary-probe",
            extra_env={
                "LIVE_TAKEOVER_FORWARD_LOCAL_CREDS": "true",
                "LIVE_TAKEOVER_ALLOW_DAEMON_ENV_FALLBACK": "true",
                "BINANCE_API_KEY": "local-key",
                "BINANCE_SECRET": "local-secret",
            },
        )

        assert proc.returncode == 0, proc.stderr
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 1
        remote_cmd = remote_cmds[0]
        assert "BINANCE_API_KEY=local-key" in remote_cmd
        assert "BINANCE_SECRET=local-secret" in remote_cmd
        assert "--allow-daemon-env-fallback" in remote_cmd
        assert "--mode probe" in remote_cmd

    def test_infra_canary_autopilot_skips_real_run_when_check_blocks(self) -> None:
        proc = self._run_bridge(
            "infra-canary-autopilot",
            extra_env={"FAKE_SSH_AUTOPILOT_MODE": "blocked"},
        )

        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["executed"] is False
        assert payload["status"] == "skipped_not_ready"
        assert payload["reason"] == "daily_budget_cap_reached"
        assert payload["autopilot_allowed"] is False
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 1
        assert "--mode autopilot-check" in remote_cmds[0]
        assert all("--mode run" not in cmd for cmd in remote_cmds)

    def test_infra_canary_autopilot_skips_safely_when_check_json_invalid(self) -> None:
        proc = self._run_bridge(
            "infra-canary-autopilot",
            extra_env={"FAKE_SSH_AUTOPILOT_MODE": "invalid-json"},
        )

        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["executed"] is False
        assert payload["status"] == "skipped_invalid_check_payload"
        assert payload["reason"] == "invalid_autopilot_check_json"
        assert payload["autopilot_allowed"] is False
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 1
        assert "--mode autopilot-check" in remote_cmds[0]
        assert all("--mode run" not in cmd for cmd in remote_cmds)

    def test_infra_canary_autopilot_runs_only_when_gate_allows(self) -> None:
        proc = self._run_bridge(
            "infra-canary-autopilot",
            extra_env={"FAKE_SSH_AUTOPILOT_MODE": "allowed"},
        )

        assert proc.returncode == 0, proc.stderr
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 2
        assert "--mode autopilot-check" in remote_cmds[0]
        assert "--mode run" in remote_cmds[1]

    def test_infra_canary_autopilot_passes_creds_and_fallback_to_check_and_run(self) -> None:
        proc = self._run_bridge(
            "infra-canary-autopilot",
            extra_env={
                "FAKE_SSH_AUTOPILOT_MODE": "allowed",
                "LIVE_TAKEOVER_FORWARD_LOCAL_CREDS": "true",
                "LIVE_TAKEOVER_ALLOW_DAEMON_ENV_FALLBACK": "true",
                "BINANCE_API_KEY": "local-key",
                "BINANCE_SECRET": "local-secret",
            },
        )

        assert proc.returncode == 0, proc.stderr
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 2
        for remote_cmd in remote_cmds:
            assert "BINANCE_API_KEY=local-key" in remote_cmd
            assert "BINANCE_SECRET=local-secret" in remote_cmd
            assert "--allow-daemon-env-fallback" in remote_cmd
        assert "--mode autopilot-check" in remote_cmds[0]
        assert "--mode run" in remote_cmds[1]

    def test_infra_canary_autopilot_skips_when_gate_field_is_string_false(self) -> None:
        proc = self._run_bridge(
            "infra-canary-autopilot",
            extra_env={"FAKE_SSH_AUTOPILOT_MODE": "string-false"},
        )

        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["executed"] is False
        assert payload["autopilot_allowed"] is False
        assert payload["status"] == "skipped_not_ready"
        assert payload["reason"] == "typed_wrong"
        assert payload["gate_field_error"]["actual_type"] == "str"
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 1
        assert "--mode autopilot-check" in remote_cmds[0]
        assert all("--mode run" not in cmd for cmd in remote_cmds)

    def test_infra_canary_autopilot_skips_when_gate_field_is_object(self) -> None:
        proc = self._run_bridge(
            "infra-canary-autopilot",
            extra_env={"FAKE_SSH_AUTOPILOT_MODE": "object"},
        )

        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["executed"] is False
        assert payload["autopilot_allowed"] is False
        assert payload["status"] == "skipped_not_ready"
        assert payload["reason"] == "typed_wrong"
        assert payload["gate_field_error"]["actual_type"] == "dict"
        remote_cmds = self._read_remote_cmds(proc)
        assert len(remote_cmds) == 1
        assert "--mode autopilot-check" in remote_cmds[0]
        assert all("--mode run" not in cmd for cmd in remote_cmds)
