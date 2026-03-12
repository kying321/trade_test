from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class NewApiModelProbeScriptTests(unittest.TestCase):
    def _run_probe(
        self,
        config_path: Path,
        *,
        disable_isolation_write: bool,
        home_dir: Path | None = None,
        api_key_env_name: str = "DUMMY_NEWAPI_KEY",
        inject_dummy_env_key: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "newapi_model_probe.sh"
        cmd = [
            "bash",
            str(script),
            "--base-url",
            "http://127.0.0.1:9",
            "--api-key-env",
            api_key_env_name,
            "--models",
            "gpt-5.4",
            "--required-models",
            "gpt-5.4",
            "--optional-models",
            "",
            "--samples",
            "1",
            "--retry-transient",
            "0",
            "--isolation-config-path",
            str(config_path),
            "--output-dir",
            "system/output/review",
        ]
        if disable_isolation_write:
            cmd.append("--disable-isolation-write")

        env = dict(os.environ)
        for key in ("NEWAPI_API_KEY", "X666_API_KEY", "OPENAI_API_KEY", "DUMMY_NEWAPI_KEY"):
            env.pop(key, None)
        if inject_dummy_env_key and api_key_env_name == "DUMMY_NEWAPI_KEY":
            env["DUMMY_NEWAPI_KEY"] = "sk-dummy-key-for-script-test"
        if home_dir is not None:
            env["HOME"] = str(home_dir)
        return subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_probe_disable_isolation_write_does_not_mutate_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            home = Path(td) / "home"
            home.mkdir(parents=True, exist_ok=True)
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")

            proc = self._run_probe(cfg, disable_isolation_write=True, home_dir=home)

            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("Isolation config write disabled", proc.stderr)
            text = cfg.read_text(encoding="utf-8")
            self.assertIn("binance_live_takeover_enabled: true", text)
            self.assertFalse((cfg.parent / f"{cfg.name}.bak").exists())

    def test_probe_default_isolation_write_mutates_target_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            home = Path(td) / "home"
            home.mkdir(parents=True, exist_ok=True)
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")

            proc = self._run_probe(cfg, disable_isolation_write=False, home_dir=home)

            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("Live Takeover isolated via", proc.stderr)
            text = cfg.read_text(encoding="utf-8")
            self.assertIn("binance_live_takeover_enabled: false", text)
            self.assertTrue((cfg.parent / f"{cfg.name}.bak").exists())

    def test_probe_artifact_records_attempt_trace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")
            home = Path(td) / "home"
            openclaw = home / ".openclaw"
            openclaw.mkdir(parents=True, exist_ok=True)
            # Add deterministic fallback key candidates to verify trace capture.
            (openclaw / "openclaw.json").write_text(
                json.dumps(
                    {
                        "skills": {
                            "entries": {
                                "openrouter": {"apiKey": "sk-dummy-fallback-key-1"},
                                "google-gemini": {"apiKey": "sk-dummy-fallback-key-2"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            proc = self._run_probe(cfg, disable_isolation_write=True, home_dir=home)
            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            artifact_path = Path(proc.stdout.strip().splitlines()[0].strip())
            self.assertTrue(artifact_path.exists(), msg=f"artifact_missing: {artifact_path}")
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            records = payload.get("records") or []
            self.assertTrue(records)
            row = records[0]
            trace = row.get("attempt_trace") or []
            self.assertTrue(isinstance(trace, list))
            self.assertEqual(int(row.get("attempts", 0)), len(trace))
            self.assertIn("first_http_status", row)
            self.assertIn("final_http_status", row)

    def test_probe_loads_openai_env_fallback_candidate_for_newapi(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")
            home = Path(td) / "home"
            openclaw = home / ".openclaw"
            openclaw.mkdir(parents=True, exist_ok=True)
            (openclaw / ".env").write_text("OPENAI_API_KEY=sk-dummy-openai-fallback-key\n", encoding="utf-8")

            proc = self._run_probe(
                cfg,
                disable_isolation_write=True,
                home_dir=home,
                api_key_env_name="NEWAPI_API_KEY",
                inject_dummy_env_key=False,
            )
            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            artifact_path = Path(proc.stdout.strip().splitlines()[0].strip())
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            sources = payload.get("api_key_sources") or []
            self.assertIn("OPENCLAW_ENV:OPENAI_API_KEY", sources)

    def test_probe_loads_auth_profiles_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")
            home = Path(td) / "home"
            profile_dir = home / ".openclaw" / "agents" / "main" / "agent"
            profile_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "auth-profiles.json").write_text(
                json.dumps(
                    {
                        "profiles": {
                            "openrouter:x666_primary": {
                                "providers": {
                                    "openrouter": {"apiKey": "sk-dummy-auth-profile-key"}
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            proc = self._run_probe(
                cfg,
                disable_isolation_write=True,
                home_dir=home,
                api_key_env_name="NEWAPI_API_KEY",
                inject_dummy_env_key=False,
            )
            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            artifact_path = Path(proc.stdout.strip().splitlines()[0].strip())
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            sources = payload.get("api_key_sources") or []
            self.assertIn("AUTH_PROFILES:openrouter:x666_primary:openrouter.apiKey", sources)

    def test_probe_loads_newapi_api_keys_csv_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")
            home = Path(td) / "home"
            openclaw = home / ".openclaw"
            openclaw.mkdir(parents=True, exist_ok=True)
            (openclaw / ".env").write_text(
                "NEWAPI_API_KEYS=sk-dummy-list-key-1, sk-dummy-list-key-2\n",
                encoding="utf-8",
            )

            proc = self._run_probe(
                cfg,
                disable_isolation_write=True,
                home_dir=home,
                api_key_env_name="NEWAPI_API_KEY",
                inject_dummy_env_key=False,
            )
            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            artifact_path = Path(proc.stdout.strip().splitlines()[0].strip())
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            sources = payload.get("api_key_sources") or []
            self.assertIn("OPENCLAW_ENV:NEWAPI_API_KEYS[1]", sources)
            self.assertIn("OPENCLAW_ENV:NEWAPI_API_KEYS[2]", sources)


if __name__ == "__main__":
    unittest.main()
