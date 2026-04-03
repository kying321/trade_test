from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard_lane_sync_guard.py"
AUDIT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "branch_governance_audit.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("dashboard_lane_sync_guard_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_guard_reports_lie_as_owner_and_rejects_blanket_sync(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    commands: list[tuple[str, ...]] = []

    def fake_run_git_text(repo_root: Path, *args: str) -> str:
        assert repo_root == tmp_path
        commands.append(tuple(args))
        lookup = {
            ("rev-parse", "--verify", "pi"): "36302e27048390274df5b936653842c43359ba49\n",
            ("rev-parse", "--verify", "lie"): "016af7acdf3d2c647c5394f70e15b537298841d3\n",
            ("ls-tree", "--name-only", "pi", "system/dashboard/web"): "",
            ("ls-tree", "--name-only", "lie", "system/dashboard/web"): "system/dashboard/web\n",
            ("merge-base", "pi", "lie"): "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd\n",
            ("rev-list", "--count", "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd..pi"): "24\n",
            ("rev-list", "--count", "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd..lie"): "78\n",
        }
        key = tuple(args)
        if key not in lookup:
            raise AssertionError(f"unexpected git args: {args!r}")
        return lookup[key]

    monkeypatch.setattr(mod, "run_git_text", fake_run_git_text)

    result = mod.evaluate_dashboard_lane_sync_guard(
        repo_root=tmp_path,
        primary_branches=["pi", "lie"],
    )

    assert result["pass"] is True
    assert result["blocking"] is False
    assert result["current_owner"] == "lie"
    assert result["recommended_action"] == "no_sync_record_ownership"
    assert result["reason_codes"] == [
        "single_owner_detected",
        "pi_missing_dashboard_tree",
        "lie_owns_dashboard_surface",
    ]
    assert result["branches"]["pi"] == {
        "exists": True,
        "has_dashboard_tree": False,
        "head": "36302e27048390274df5b936653842c43359ba49",
        "ahead_since_merge_base": 24,
        "resolved_ref": "pi",
    }
    assert result["branches"]["lie"] == {
        "exists": True,
        "has_dashboard_tree": True,
        "head": "016af7acdf3d2c647c5394f70e15b537298841d3",
        "ahead_since_merge_base": 78,
        "resolved_ref": "lie",
    }
    assert result["merge_base"] == "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd"
    assert ("ls-tree", "--name-only", "pi", "system/dashboard/web") in commands
    assert ("ls-tree", "--name-only", "lie", "system/dashboard/web") in commands


def test_evaluate_guard_skips_branches_without_common_merge_base(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()

    def fake_run_git_text(repo_root: Path, *args: str) -> str:
        assert repo_root == tmp_path
        lookup = {
            ("rev-parse", "--verify", "main"): "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
            ("rev-parse", "--verify", "pi"): "36302e27048390274df5b936653842c43359ba49\n",
            ("rev-parse", "--verify", "lie"): "016af7acdf3d2c647c5394f70e15b537298841d3\n",
            ("ls-tree", "--name-only", "main", "system/dashboard/web"): "",
            ("ls-tree", "--name-only", "pi", "system/dashboard/web"): "",
            ("ls-tree", "--name-only", "lie", "system/dashboard/web"): "system/dashboard/web\n",
            ("merge-base", "main", "lie"): subprocess.CalledProcessError(1, ["git", "merge-base", "main", "lie"]),
            ("merge-base", "pi", "lie"): "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd\n",
            ("rev-list", "--count", "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd..pi"): "24\n",
            ("rev-list", "--count", "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd..lie"): "78\n",
        }
        key = tuple(args)
        value = lookup.get(key)
        if value is None:
            raise AssertionError(f"unexpected git args: {args!r}")
        if isinstance(value, subprocess.CalledProcessError):
            raise value
        return value

    monkeypatch.setattr(mod, "run_git_text", fake_run_git_text)

    result = mod.evaluate_dashboard_lane_sync_guard(
        repo_root=tmp_path,
        primary_branches=["main", "pi", "lie"],
    )

    assert result["pass"] is True
    assert result["current_owner"] == "lie"
    assert result["recommended_action"] == "no_sync_record_ownership"
    assert result["branches"]["main"]["ahead_since_merge_base"] is None
    assert result["branches"]["pi"]["ahead_since_merge_base"] == 24
    assert result["branches"]["lie"]["ahead_since_merge_base"] == 78
    assert result["branches"]["main"]["resolved_ref"] == "main"
    assert result["branches"]["pi"]["resolved_ref"] == "pi"
    assert result["branches"]["lie"]["resolved_ref"] == "lie"
    assert result["merge_base"] == "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd"


def test_evaluate_guard_falls_back_to_origin_tracking_branches(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()

    def fake_run_git_text(repo_root: Path, *args: str) -> str:
        assert repo_root == tmp_path
        lookup = {
            ("rev-parse", "--verify", "pi"): subprocess.CalledProcessError(1, ["git", "rev-parse", "--verify", "pi"]),
            ("rev-parse", "--verify", "origin/pi"): "36302e27048390274df5b936653842c43359ba49\n",
            ("rev-parse", "--verify", "lie"): subprocess.CalledProcessError(1, ["git", "rev-parse", "--verify", "lie"]),
            ("rev-parse", "--verify", "origin/lie"): "016af7acdf3d2c647c5394f70e15b537298841d3\n",
            ("ls-tree", "--name-only", "origin/pi", "system/dashboard/web"): "",
            ("ls-tree", "--name-only", "origin/lie", "system/dashboard/web"): "system/dashboard/web\n",
            ("merge-base", "origin/pi", "origin/lie"): "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd\n",
            ("rev-list", "--count", "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd..origin/pi"): "24\n",
            ("rev-list", "--count", "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd..origin/lie"): "78\n",
        }
        key = tuple(args)
        value = lookup.get(key)
        if value is None:
            raise AssertionError(f"unexpected git args: {args!r}")
        if isinstance(value, subprocess.CalledProcessError):
            raise value
        return value

    monkeypatch.setattr(mod, "run_git_text", fake_run_git_text)

    result = mod.evaluate_dashboard_lane_sync_guard(
        repo_root=tmp_path,
        primary_branches=["pi", "lie"],
    )

    assert result["pass"] is True
    assert result["current_owner"] == "lie"
    assert result["recommended_action"] == "no_sync_record_ownership"
    assert result["reason_codes"] == [
        "single_owner_detected",
        "pi_missing_dashboard_tree",
        "lie_owns_dashboard_surface",
    ]
    assert result["branches"]["pi"] == {
        "exists": True,
        "has_dashboard_tree": False,
        "head": "36302e27048390274df5b936653842c43359ba49",
        "ahead_since_merge_base": 24,
        "resolved_ref": "origin/pi",
    }
    assert result["branches"]["lie"] == {
        "exists": True,
        "has_dashboard_tree": True,
        "head": "016af7acdf3d2c647c5394f70e15b537298841d3",
        "ahead_since_merge_base": 78,
        "resolved_ref": "origin/lie",
    }
    assert result["merge_base"] == "ab86d0d4abcdabcdabcdabcdabcdabcdabcdabcd"


def test_branch_governance_audit_embeds_dashboard_lane_sync_guard_with_overrides(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(parents=True)

    protection_script = stub_dir / "protection.sh"
    protection_script.write_text("#!/usr/bin/env bash\necho 'branch protection ok'\n", encoding="utf-8")
    protection_script.chmod(0o755)

    reaper_script = stub_dir / "reaper.sh"
    reaper_script.write_text("#!/usr/bin/env bash\necho 'reaper ok'\n", encoding="utf-8")
    reaper_script.chmod(0o755)

    guard_script = stub_dir / "guard.py"
    guard_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "print(json.dumps({",
                "  'pass': True,",
                "  'blocking': False,",
                "  'current_owner': 'lie',",
                "  'recommended_action': 'no_sync_record_ownership',",
                "  'reason_codes': ['single_owner_detected'],",
                "  'summary': 'stubbed guard payload',",
                "}, ensure_ascii=False))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    guard_script.chmod(0o755)

    output_rel = f"system/output/review/pytest_dashboard_lane_sync_guard_{tmp_path.name}"
    output_dir = repo_root / output_rel
    shutil.rmtree(output_dir, ignore_errors=True)

    env = os.environ.copy()
    env["FENLIE_BRANCH_PROTECTION_SCRIPT"] = str(protection_script)
    env["FENLIE_HOTFIX_REAPER_SCRIPT"] = str(reaper_script)
    env["FENLIE_DASHBOARD_LANE_SYNC_GUARD_SCRIPT"] = str(guard_script)

    proc = subprocess.run(
        ["bash", str(AUDIT_SCRIPT_PATH), "--repo", "example/repo", "--output-dir", output_rel],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) >= 2
    artifact_json = Path(stdout_lines[0])
    artifact_md = Path(stdout_lines[1])
    assert artifact_json.exists()
    assert artifact_md.exists()

    payload = json.loads(artifact_json.read_text(encoding="utf-8"))
    guard = payload["checks"]["dashboard_lane_sync_guard"]
    assert guard["pass"] is True
    assert guard["blocking"] is False
    assert guard["current_owner"] == "lie"
    assert guard["recommended_action"] == "no_sync_record_ownership"
    assert guard["reason_codes"] == ["single_owner_detected"]
    assert guard["summary"] == "stubbed guard payload"
    assert guard["return_code"] == 0
    assert guard["stderr"] == ""

    shutil.rmtree(output_dir, ignore_errors=True)
