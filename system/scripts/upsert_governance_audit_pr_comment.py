#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


COMMENT_MARKER = "<!-- fenlie-governance-audit-advisory -->"
API_ROOT = "https://api.github.com"


def build_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "fenlie-governance-audit-comment",
    }


def api_request(method: str, url: str, token: str, payload: dict[str, object] | None = None) -> object:
    data = None
    headers = build_headers(token)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request) as response:  # noqa: S310
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def read_http_error_payload(exc: urllib.error.HTTPError) -> dict[str, object]:
    if exc.fp is None:
        return {}
    try:
        raw = exc.fp.read().decode("utf-8")
    except Exception:
        return {}
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def upsert_comment(*, repo: str, issue_number: int, token: str, comment_path: Path) -> dict[str, object]:
    body = comment_path.read_text(encoding="utf-8")
    comments_url = f"{API_ROOT}/repos/{repo}/issues/{issue_number}/comments?per_page=100"
    create_url = f"{API_ROOT}/repos/{repo}/issues/{issue_number}/comments"

    try:
        comments_payload = api_request("GET", comments_url, token)
        comments = comments_payload if isinstance(comments_payload, list) else []
        existing = next(
            (
                comment
                for comment in comments
                if isinstance(comment, dict) and COMMENT_MARKER in str(comment.get("body") or "")
            ),
            None,
        )

        if existing:
            comment_id = int(existing.get("id"))
            update_url = f"{API_ROOT}/repos/{repo}/issues/comments/{comment_id}"
            updated = api_request("PATCH", update_url, token, {"body": body})
            return {
                "action": "updated",
                "comment_id": int((updated or {}).get("id") or comment_id),
                "warning": "",
            }

        created = api_request("POST", create_url, token, {"body": body})
        return {
            "action": "created",
            "comment_id": int((created or {}).get("id") or 0),
            "warning": "",
        }
    except urllib.error.HTTPError as exc:
        payload = read_http_error_payload(exc)
        message = str(payload.get("message") or exc.reason or "").strip()
        if exc.code == 403 and "Resource not accessible by integration" in message:
            return {
                "action": "skipped",
                "comment_id": None,
                "warning": message,
            }
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert governance audit PR comment.")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--issue-number", required=True, type=int)
    parser.add_argument("--comment-file", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = str((__import__("os").environ.get("GH_TOKEN") or __import__("os").environ.get("GITHUB_TOKEN") or "")).strip()
    if not token:
        print("governance audit PR comment skipped: missing GH_TOKEN/GITHUB_TOKEN", file=sys.stderr)
        return 0

    result = upsert_comment(
        repo=str(args.repo),
        issue_number=int(args.issue_number),
        token=token,
        comment_path=Path(args.comment_file).expanduser().resolve(),
    )

    if result["action"] == "skipped":
        print(f"::warning::governance audit PR comment skipped: {result['warning']}")
        return 0

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
