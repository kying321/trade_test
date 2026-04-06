from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lie_engine" / "cpa_channels" / "cpa_authfiles.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cpa_channel_authfile_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def encode_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}."


def test_build_cpa_auth_payload_uses_jwt_claims_and_normalized_email() -> None:
    mod = load_module()
    token = encode_jwt(
        {
            "exp": 1775400000,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_from_jwt",
            },
        }
    )

    payload = mod.build_cpa_auth_payload(
        {
            "email": "Alpha@Fuuu.fun ",
            "access_token": token,
            "refresh_token": "refresh-token",
            "id_token": "id-token",
            "provider": "codex",
        },
        now=mod.datetime(2026, 4, 7, 8, 30, tzinfo=mod.timezone.utc),
    )

    assert payload["email"] == "alpha@fuuu.fun"
    assert payload["type"] == "codex"
    assert payload["account_id"] == "acct_from_jwt"
    assert payload["refresh_token"] == "refresh-token"
    assert payload["id_token"] == "id-token"
    assert payload["last_refresh"].startswith("2026-04-07T16:30:00")
    assert payload["expired"] != ""


def test_export_cpa_authfiles_writes_one_file_per_account(tmp_path: Path) -> None:
    mod = load_module()
    access_token = encode_jwt({"exp": 1775400000})
    input_path = tmp_path / "bundle.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"email": "a@fuuu.fun", "access_token": access_token, "provider": "codex"}),
                json.dumps({"email": "b@fuuu.fun", "access_token": access_token, "provider": "codex"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = mod.export_cpa_authfiles(
        input_file=input_path,
        output_dir=tmp_path / "authfiles",
        now=mod.datetime(2026, 4, 7, 8, 30, tzinfo=mod.timezone.utc),
    )

    assert result["count"] == 2
    assert sorted(Path(path).name for path in result["files"]) == ["a@fuuu.fun.json", "b@fuuu.fun.json"]
    assert json.loads((tmp_path / "authfiles" / "a@fuuu.fun.json").read_text(encoding="utf-8"))["email"] == "a@fuuu.fun"
