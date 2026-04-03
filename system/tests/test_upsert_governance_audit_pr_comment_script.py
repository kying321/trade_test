from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "upsert_governance_audit_pr_comment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("upsert_governance_audit_pr_comment_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_upsert_comment_creates_new_comment_when_marker_missing(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    comment_path = tmp_path / "comment.md"
    comment_path.write_text("<!-- fenlie-governance-audit-advisory -->\nbody\n", encoding="utf-8")
    requests: list[tuple[str, str]] = []

    class DummyResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request):  # noqa: ANN001
        requests.append((request.get_method(), request.full_url))
        if request.get_method() == "GET":
            return DummyResponse(json.dumps([]).encode("utf-8"))
        if request.get_method() == "POST":
            payload = json.loads(request.data.decode("utf-8"))
            assert "<!-- fenlie-governance-audit-advisory -->" in payload["body"]
            return DummyResponse(json.dumps({"id": 123}).encode("utf-8"))
        raise AssertionError(f"unexpected method {request.get_method()}")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    result = mod.upsert_comment(
        repo="kying321/trade_test",
        issue_number=119,
        token="token",
        comment_path=comment_path,
    )

    assert result == {"action": "created", "comment_id": 123, "warning": ""}
    assert requests == [
        ("GET", "https://api.github.com/repos/kying321/trade_test/issues/119/comments?per_page=100"),
        ("POST", "https://api.github.com/repos/kying321/trade_test/issues/119/comments"),
    ]


def test_upsert_comment_degrades_on_resource_not_accessible(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    comment_path = tmp_path / "comment.md"
    comment_path.write_text("<!-- fenlie-governance-audit-advisory -->\nbody\n", encoding="utf-8")

    def fake_urlopen(request):  # noqa: ANN001
        payload = json.dumps({"message": "Resource not accessible by integration"}).encode("utf-8")
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(payload),
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    result = mod.upsert_comment(
        repo="kying321/trade_test",
        issue_number=119,
        token="token",
        comment_path=comment_path,
    )

    assert result["action"] == "skipped"
    assert result["comment_id"] is None
    assert "Resource not accessible by integration" in result["warning"]
