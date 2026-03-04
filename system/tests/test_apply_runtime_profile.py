from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import yaml


def _load_apply_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "apply_runtime_profile.py"
    spec = importlib.util.spec_from_file_location("apply_runtime_profile_for_tests", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class ApplyRuntimeProfileTests(unittest.TestCase):
    def test_dry_run_does_not_mutate_params_live(self) -> None:
        mod = _load_apply_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_root = root / "output"
            params_path = output_root / "artifacts" / "params_live.yaml"
            params_path.parent.mkdir(parents=True, exist_ok=True)
            params_path.write_text(
                yaml.safe_dump({"signal_confidence_min": 70.0, "win_rate": 0.45}, allow_unicode=True),
                encoding="utf-8",
            )
            original = params_path.read_text(encoding="utf-8")

            profile_file = root / "runtime_profiles.yaml"
            profile_file.write_text(
                yaml.safe_dump(
                    {
                        "stable_profile": {
                            "runtime": {
                                "signal_confidence_min": 5.0,
                                "theory_enabled": True,
                                "theory_wyckoff_weight": 0.8,
                            }
                        }
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            out = mod.apply_runtime_profile(
                output_root=output_root,
                profile_file=profile_file,
                profile_name="stable_profile",
                write=False,
                allow_reflex_lock=False,
                timeout_seconds=1.0,
            )
            self.assertEqual(str(out.get("status", "")), "dry_run")
            self.assertGreaterEqual(int(out.get("changed_count", 0)), 1)
            self.assertEqual(params_path.read_text(encoding="utf-8"), original)

    def test_write_blocked_when_reflex_lock_active(self) -> None:
        mod = _load_apply_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_root = root / "output"
            params_path = output_root / "artifacts" / "params_live.yaml"
            params_path.parent.mkdir(parents=True, exist_ok=True)
            params_path.write_text(
                yaml.safe_dump({"signal_confidence_min": 70.0, "reflex_lock": "ACTIVE"}, allow_unicode=True),
                encoding="utf-8",
            )
            profile_file = root / "runtime_profiles.yaml"
            profile_file.write_text(
                yaml.safe_dump({"stable_profile": {"runtime": {"signal_confidence_min": 5.0}}}, allow_unicode=True),
                encoding="utf-8",
            )

            out = mod.apply_runtime_profile(
                output_root=output_root,
                profile_file=profile_file,
                profile_name="stable_profile",
                write=True,
                allow_reflex_lock=False,
                timeout_seconds=1.0,
            )
            self.assertEqual(str(out.get("status", "")), "blocked_reflex_lock")
            payload = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
            self.assertAlmostEqual(float(payload.get("signal_confidence_min", 0.0)), 70.0, places=6)

    def test_write_applies_runtime_and_creates_backup(self) -> None:
        mod = _load_apply_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_root = root / "output"
            params_path = output_root / "artifacts" / "params_live.yaml"
            params_path.parent.mkdir(parents=True, exist_ok=True)
            params_path.write_text(
                yaml.safe_dump({"signal_confidence_min": 70.0, "win_rate": 0.45}, allow_unicode=True),
                encoding="utf-8",
            )
            profile_file = root / "runtime_profiles.yaml"
            profile_file.write_text(
                yaml.safe_dump(
                    {"stable_profile": {"runtime": {"signal_confidence_min": 5.0, "theory_enabled": True}}},
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            out = mod.apply_runtime_profile(
                output_root=output_root,
                profile_file=profile_file,
                profile_name="stable_profile",
                write=True,
                allow_reflex_lock=False,
                timeout_seconds=1.0,
            )
            self.assertEqual(str(out.get("status", "")), "applied")
            backup_path = Path(str(out.get("backup_path", "")))
            self.assertTrue(backup_path.exists())
            payload = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
            self.assertAlmostEqual(float(payload.get("signal_confidence_min", 0.0)), 5.0, places=6)
            self.assertAlmostEqual(float(payload.get("theory_enabled", 0.0)), 1.0, places=6)
            self.assertAlmostEqual(float(payload.get("win_rate", 0.0)), 0.45, places=6)


if __name__ == "__main__":
    unittest.main()
