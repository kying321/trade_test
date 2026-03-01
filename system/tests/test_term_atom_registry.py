from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


class TermAtomRegistryTests(unittest.TestCase):
    def test_validate_term_atom_registry_script(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "validate_term_atoms.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--path", "config/term_atoms.yaml"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertTrue(bool(payload.get("ok", False)))
        self.assertGreaterEqual(int(payload.get("atoms_total", 0)), 10)
        self.assertGreaterEqual(int(payload.get("l2_atoms", 0)), 1)


if __name__ == "__main__":
    unittest.main()
