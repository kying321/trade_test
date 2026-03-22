from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


class AutoGitSyncScriptTests(unittest.TestCase):
    def _write_executable(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_dry_run_prefers_repo_owned_lie_local_wrapper(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)

            call_log = repo_root / "lie_local_calls.log"
            self._write_executable(
                repo_root / "system" / "scripts" / "lie-local",
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        f"printf '%s\\n' \"$*\" >> {str(call_log)!r}",
                    ]
                )
                + "\n",
            )
            self._write_executable(
                repo_root / "fakebin" / "lie",
                "#!/usr/bin/env bash\n"
                "echo 'global lie must not be used during this test' >&2\n"
                "exit 91\n",
            )

            (repo_root / "system").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("repo-owned wrapper test\n", encoding="utf-8")

            env = dict(os.environ)
            env["PATH"] = f"{repo_root / 'fakebin'}{os.pathsep}{env.get('PATH', '')}"

            proc = subprocess.run(
                [str(script), "--dry-run", "--message", "test(auto-git-sync): prefer lie-local"],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertTrue(call_log.exists(), msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertEqual(call_log.read_text(encoding="utf-8").splitlines(), ["validate-config", "test-all"])
            self.assertNotIn("global lie must not be used", f"{proc.stdout}\n{proc.stderr}")

    def test_dry_run_falls_back_to_pythonpath_cli_when_wrapper_and_global_lie_are_unavailable(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)

            call_log = repo_root / "pythonpath_cli_calls.log"
            cli_module = repo_root / "system" / "src" / "lie_engine" / "cli.py"
            cli_module.parent.mkdir(parents=True, exist_ok=True)
            (cli_module.parent / "__init__.py").write_text("", encoding="utf-8")
            cli_module.write_text(
                "\n".join(
                    [
                        "from __future__ import annotations",
                        "import json",
                        "import sys",
                        "from pathlib import Path",
                        f"CALL_LOG = Path({str(call_log)!r})",
                        "CALL_LOG.parent.mkdir(parents=True, exist_ok=True)",
                        "with CALL_LOG.open('a', encoding='utf-8') as fh:",
                        "    fh.write(' '.join(sys.argv[1:]) + '\\n')",
                        "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}))",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (repo_root / "system" / "README.md").write_text("pythonpath fallback test\n", encoding="utf-8")

            env = dict(os.environ)
            env["PATH"] = os.pathsep.join(["/usr/bin", "/bin", "/usr/sbin", "/sbin"])

            proc = subprocess.run(
                [str(script), "--dry-run", "--message", "test(auto-git-sync): pythonpath fallback"],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertTrue(call_log.exists(), msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertEqual(call_log.read_text(encoding="utf-8").splitlines(), ["validate-config", "test-all"])

    def test_non_dry_run_creates_lie_commit_and_pushes_only_eligible_files(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)

            (repo_root / "system" / "README.md").write_text("updated readme\n", encoding="utf-8")
            (repo_root / "notes.txt").write_text("should stay untracked\n", encoding="utf-8")
            (repo_root / "system" / "output" / "artifacts").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "output" / "artifacts" / "ignored.json").write_text("{}", encoding="utf-8")

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--message",
                    "test(auto-git-sync): commit eligible files only",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            current_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self.assertEqual(current_branch, "lie")

            head_subject = self._git(repo_root, "log", "-1", "--pretty=%s").stdout.strip()
            self.assertEqual(head_subject, "test(auto-git-sync): commit eligible files only")

            committed_files = {
                line.strip()
                for line in self._git(repo_root, "show", "--name-only", "--pretty=format:", "HEAD").stdout.splitlines()
                if line.strip()
            }
            self.assertIn("system/README.md", committed_files)
            self.assertNotIn("notes.txt", committed_files)
            self.assertNotIn("system/output/artifacts/ignored.json", committed_files)

            remote_head = self._git(td_path, "--git-dir", str(origin_root), "rev-parse", "refs/heads/lie").stdout.strip()
            local_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()
            self.assertEqual(remote_head, local_head)

    def test_non_dry_run_blocked_branch_auto_falls_back_to_current_allowed_branch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)
            self._git(repo_root, "checkout", "-b", "lie")
            self._git(repo_root, "push", "-u", "origin", "lie")

            (repo_root / "system" / "README.md").write_text("blocked branch fallback to current allowed branch\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--branch",
                    "codex/blocked-branch",
                    "--message",
                    "test(auto-git-sync): fallback current allowed branch",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("auto-fallback to current branch 'lie'", proc.stderr)
            current_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self.assertEqual(current_branch, "lie")
            head_subject = self._git(repo_root, "log", "-1", "--pretty=%s").stdout.strip()
            self.assertEqual(head_subject, "test(auto-git-sync): fallback current allowed branch")

    def test_non_dry_run_blocked_branch_uses_explicit_allowed_fallback_branch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)
            self._git(repo_root, "checkout", "-b", "feature/local")

            (repo_root / "system" / "README.md").write_text("blocked branch explicit fallback\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--branch",
                    "codex/blocked-branch",
                    "--fallback-branch",
                    "lie",
                    "--message",
                    "test(auto-git-sync): explicit fallback branch",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("fallback to 'lie'", proc.stderr)
            current_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self.assertEqual(current_branch, "lie")
            head_subject = self._git(repo_root, "log", "-1", "--pretty=%s").stdout.strip()
            self.assertEqual(head_subject, "test(auto-git-sync): explicit fallback branch")
            remote_head = self._git(td_path, "--git-dir", str(origin_root), "rev-parse", "refs/heads/lie").stdout.strip()
            local_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()
            self.assertEqual(remote_head, local_head)

    def test_non_dry_run_excludes_progress_file_by_default(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system" / "docs").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            (repo_root / "system" / "docs" / "PROGRESS.md").write_text("seed\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md", "system/docs/PROGRESS.md")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)

            (repo_root / "system" / "docs" / "PROGRESS.md").write_text("progress only change\n", encoding="utf-8")
            before_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--message",
                    "test(auto-git-sync): exclude progress by default",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("No eligible files to commit", proc.stdout)
            after_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()
            self.assertEqual(after_head, before_head)

    def test_non_dry_run_include_progress_flag_commits_progress_file(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system" / "docs").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            (repo_root / "system" / "docs" / "PROGRESS.md").write_text("seed\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md", "system/docs/PROGRESS.md")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)

            (repo_root / "system" / "docs" / "PROGRESS.md").write_text("progress include change\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--include-progress",
                    "--message",
                    "test(auto-git-sync): include progress file",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            committed_files = {
                line.strip()
                for line in self._git(repo_root, "show", "--name-only", "--pretty=format:", "HEAD").stdout.splitlines()
                if line.strip()
            }
            self.assertIn("system/docs/PROGRESS.md", committed_files)
            head_subject = self._git(repo_root, "log", "-1", "--pretty=%s").stdout.strip()
            self.assertEqual(head_subject, "test(auto-git-sync): include progress file")

    def test_non_dry_run_no_review_files_flag_excludes_review_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system" / "output" / "review").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            (repo_root / "system" / "output" / "review" / "seed.json").write_text("{\"seed\":true}\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md", "system/output/review/seed.json")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)

            (repo_root / "system" / "output" / "review" / "latest.json").write_text("{\"review\":true}\n", encoding="utf-8")
            before_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--no-review-files",
                    "--message",
                    "test(auto-git-sync): exclude review files",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("No eligible files to commit", proc.stdout)
            after_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()
            self.assertEqual(after_head, before_head)

    def test_non_dry_run_excludes_test_logs_by_default(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system" / "output" / "logs").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            (repo_root / "system" / "output" / "logs" / "tests_seed.json").write_text("{\"seed\":true}\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md", "system/output/logs/tests_seed.json")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)

            (repo_root / "system" / "output" / "logs" / "tests_run.json").write_text("{\"log\":true}\n", encoding="utf-8")
            before_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--message",
                    "test(auto-git-sync): exclude logs by default",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("No eligible files to commit", proc.stdout)
            after_head = self._git(repo_root, "rev-parse", "HEAD").stdout.strip()
            self.assertEqual(after_head, before_head)

    def test_non_dry_run_include_logs_flag_commits_test_logs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            origin_root = td_path / "origin.git"
            repo_root = td_path / "repo"

            self._git(td_path, "init", "--bare", str(origin_root))
            self._git(td_path, "init", str(repo_root))
            self._git(repo_root, "config", "user.name", "Fenlie Test")
            self._git(repo_root, "config", "user.email", "fenlie@example.com")

            (repo_root / "system" / "output" / "logs").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("initial\n", encoding="utf-8")
            (repo_root / "system" / "output" / "logs" / "tests_seed.json").write_text("{\"seed\":true}\n", encoding="utf-8")
            self._git(repo_root, "add", "system/README.md", "system/output/logs/tests_seed.json")
            self._git(repo_root, "commit", "-m", "chore: seed")
            self._git(repo_root, "remote", "add", "origin", str(origin_root))
            initial_branch = self._git(repo_root, "branch", "--show-current").stdout.strip()
            self._git(repo_root, "push", "-u", "origin", initial_branch)

            (repo_root / "system" / "output" / "logs" / "tests_run.json").write_text("{\"log\":true}\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    str(script),
                    "--skip-tests",
                    "--include-logs",
                    "--message",
                    "test(auto-git-sync): include logs file",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            committed_files = {
                line.strip()
                for line in self._git(repo_root, "show", "--name-only", "--pretty=format:", "HEAD").stdout.splitlines()
                if line.strip()
            }
            self.assertIn("system/output/logs/tests_run.json", committed_files)
            head_subject = self._git(repo_root, "log", "-1", "--pretty=%s").stdout.strip()
            self.assertEqual(head_subject, "test(auto-git-sync): include logs file")

    def test_dry_run_blocked_branch_and_blocked_fallback_fail_fast(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "auto_git_sync.sh"

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
            (repo_root / "system").mkdir(parents=True, exist_ok=True)
            (repo_root / "system" / "README.md").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "add", "system/README.md"], cwd=repo_root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Fenlie Test"], cwd=repo_root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "fenlie@example.com"], cwd=repo_root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "chore: seed"], cwd=repo_root, check=True, capture_output=True, text=True)

            proc = subprocess.run(
                [
                    str(script),
                    "--dry-run",
                    "--branch",
                    "codex/blocked-branch",
                    "--fallback-branch",
                    "codex/also-blocked",
                    "--message",
                    "test(auto-git-sync): blocked fallback should fail",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("branch 'codex/blocked-branch' is blocked", proc.stderr)
            self.assertIn("fallback branch 'codex/also-blocked' is also blocked", proc.stderr)


if __name__ == "__main__":
    unittest.main()
