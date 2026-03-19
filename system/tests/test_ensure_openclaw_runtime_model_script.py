from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class EnsureOpenClawRuntimeModelScriptTests(unittest.TestCase):
    def _run(self, config_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "ensure_openclaw_runtime_model.py"
        cmd = ["python3", str(script), "--config", str(config_path), *extra]
        return subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_adds_openai_runtime_provider_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "openclaw.json"
            cfg.write_text(
                json.dumps(
                    {
                        "models": {"providers": {"google": {"baseUrl": "https://generativelanguage.googleapis.com/v1beta", "models": []}}},
                        "agents": {"defaults": {"model": {"primary": "openai/gpt-5.4", "fallbacks": []}, "models": {}}},
                    }
                ),
                encoding="utf-8",
            )

            proc = self._run(cfg)

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["changed"])
            self.assertTrue(payload["backup_path"])
            out = json.loads(cfg.read_text(encoding="utf-8"))
            provider = out["models"]["providers"]["openai"]
            self.assertEqual(provider["api"], "openai-responses")
            self.assertEqual(provider["baseUrl"], "http://127.0.0.1:8317/v1")
            self.assertEqual(provider["models"][0]["id"], "gpt-5.4")
            self.assertIn("openai/gpt-5.4", out["agents"]["defaults"]["models"])
            self.assertIn("gpt-5.4", out["agents"]["defaults"]["models"])

    def test_is_idempotent_when_runtime_model_already_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "openclaw.json"
            payload = {
                "models": {
                    "providers": {
                        "openai": {
                            "api": "openai-responses",
                            "baseUrl": "http://127.0.0.1:8317/v1",
                            "models": [
                                {
                                    "id": "gpt-5.4",
                                    "name": "gpt-5.4",
                                    "reasoning": True,
                                    "input": ["text", "image"],
                                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                                    "contextWindow": 391000,
                                    "maxTokens": 128000,
                                }
                            ],
                        }
                    }
                },
                "agents": {"defaults": {"models": {"openai/gpt-5.4": {}, "gpt-5.4": {}}}},
            }
            cfg.write_text(json.dumps(payload), encoding="utf-8")

            proc = self._run(cfg, "--no-backup")

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            out = json.loads(proc.stdout)
            self.assertFalse(out["changed"])
            self.assertIsNone(out["backup_path"])
            self.assertEqual(out["before_sha256"], out["after_sha256"])

    def test_preserves_existing_openai_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "openclaw.json"
            cfg.write_text(
                json.dumps(
                    {
                        "models": {
                            "providers": {
                                "openai": {
                                    "api": "openai-responses",
                                    "baseUrl": "http://127.0.0.1:8317/v1",
                                    "models": [{"id": "gpt-4o", "name": "gpt-4o"}],
                                }
                            }
                        },
                        "agents": {"defaults": {"models": {}}},
                    }
                ),
                encoding="utf-8",
            )

            proc = self._run(cfg, "--no-backup")

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            out = json.loads(cfg.read_text(encoding="utf-8"))
            model_ids = [row.get("id") for row in out["models"]["providers"]["openai"]["models"]]
            self.assertIn("gpt-4o", model_ids)
            self.assertIn("gpt-5.4", model_ids)


if __name__ == "__main__":
    unittest.main()
