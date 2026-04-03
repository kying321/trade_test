#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence


DASHBOARD_TREE = "system/dashboard/web"
DOC_PATHS = [
    "system/docs/BRANCH_POLICY.md",
    "system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md",
    "system/docs/DASHBOARD_LANE_SYNC_MATRIX.md",
    "system/docs/PROGRESS.md",
]


def run_git_text(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def resolve_branch_ref(repo_root: Path, branch: str) -> str | None:
    candidates = [branch, f"origin/{branch}"]
    for candidate in candidates:
        try:
            run_git_text(repo_root, "rev-parse", "--verify", candidate)
        except subprocess.CalledProcessError:
            continue
        return candidate
    return None


def branch_snapshot(repo_root: Path, branch: str) -> dict[str, object]:
    resolved_ref = resolve_branch_ref(repo_root, branch)
    if not resolved_ref:
        return {
            "exists": False,
            "has_dashboard_tree": False,
            "head": "",
            "ahead_since_merge_base": None,
            "resolved_ref": "",
        }

    head = run_git_text(repo_root, "rev-parse", "--verify", resolved_ref).strip()
    tree_text = run_git_text(repo_root, "ls-tree", "--name-only", resolved_ref, DASHBOARD_TREE).strip()
    return {
        "exists": True,
        "has_dashboard_tree": bool(tree_text),
        "head": head,
        "ahead_since_merge_base": None,
        "resolved_ref": resolved_ref,
    }


def evaluate_dashboard_lane_sync_guard(repo_root: Path, primary_branches: Sequence[str]) -> dict[str, object]:
    branches: dict[str, dict[str, object]] = {}
    existing_branches: list[str] = []
    owner_branches: list[str] = []

    for branch in primary_branches:
        snapshot = branch_snapshot(repo_root, branch)
        branches[branch] = snapshot
        if bool(snapshot["exists"]):
            existing_branches.append(branch)
        if bool(snapshot["has_dashboard_tree"]):
            owner_branches.append(branch)
    comparable_branches = [branch for branch in primary_branches if bool(branches.get(branch, {}).get("exists"))]

    blocking = False
    pass_value = True
    current_owner = ""
    recommended_action = "manual_review_required"
    reason_codes: list[str] = []
    summary = ""

    if len(owner_branches) == 1:
        current_owner = owner_branches[0]
        recommended_action = "no_sync_record_ownership"
        reason_codes.append("single_owner_detected")
        for branch in comparable_branches:
            if branch == current_owner:
                reason_codes.append(f"{branch}_owns_dashboard_surface")
            elif not bool(branches[branch]["has_dashboard_tree"]):
                reason_codes.append(f"{branch}_missing_dashboard_tree")
        summary = (
            f"Detected a single dashboard owner branch `{current_owner}`; "
            "do not blanket-sync other primary branches."
        )
    elif len(owner_branches) == 0:
        pass_value = False
        recommended_action = "ownership_missing_manual_review"
        reason_codes.append("no_owner_detected")
        summary = "No primary branch currently carries the dashboard tree; manual review is required."
    else:
        pass_value = False
        recommended_action = "explicit_transfer_or_split_required"
        reason_codes.append("multiple_owners_detected")
        for branch in owner_branches:
            reason_codes.append(f"{branch}_owns_dashboard_surface")
        summary = "Multiple primary branches carry the dashboard tree; explicit ownership transfer/split is required."

    merge_base = ""
    pair_candidates: list[tuple[str, str]] = []
    if current_owner:
        for branch in comparable_branches:
            if branch != current_owner:
                pair_candidates.append((branch, current_owner))
    if not pair_candidates and len(comparable_branches) >= 2:
        for index, left in enumerate(comparable_branches[:-1]):
            for right in comparable_branches[index + 1 :]:
                pair_candidates.append((left, right))

    for left, right in pair_candidates:
        try:
            left_ref = str(branches[left].get("resolved_ref") or left)
            right_ref = str(branches[right].get("resolved_ref") or right)
            candidate_merge_base = run_git_text(repo_root, "merge-base", left_ref, right_ref).strip()
        except subprocess.CalledProcessError:
            continue
        if not candidate_merge_base:
            continue
        merge_base = candidate_merge_base
        for branch in {left, right}:
            branch_ref = str(branches[branch].get("resolved_ref") or branch)
            count_text = run_git_text(repo_root, "rev-list", "--count", f"{merge_base}..{branch_ref}").strip()
            branches[branch]["ahead_since_merge_base"] = int(count_text or "0")
        break

    return {
        "pass": pass_value,
        "blocking": blocking,
        "current_owner": current_owner,
        "recommended_action": recommended_action,
        "reason_codes": reason_codes,
        "summary": summary,
        "dashboard_tree": DASHBOARD_TREE,
        "primary_branches": list(primary_branches),
        "merge_base": merge_base,
        "branches": branches,
        "docs": DOC_PATHS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only dashboard lane sync guard.")
    parser.add_argument("--repo-root", default=".", help="Git repository root.")
    parser.add_argument(
        "--primary-branches",
        default="pi,lie",
        help="Comma-separated primary branches to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    primary_branches = [part.strip() for part in str(args.primary_branches).split(",") if part.strip()]
    payload = evaluate_dashboard_lane_sync_guard(repo_root=repo_root, primary_branches=primary_branches)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
