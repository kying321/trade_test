from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cpa_channel_ingest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_cpa_channel_ingest_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def encode_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}."


def test_run_ingest_persists_bundle_and_writes_review_artifact(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    access_token = encode_jwt({"exp": 1775400000, "https://api.openai.com/auth": {"chatgpt_account_id": "acct_demo"}})
    input_path = workspace / "bundle.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        json.dumps(
            {
                "email": "alpha@fuuu.fun",
                "provider": "codex",
                "access_token": access_token,
                "refresh_token": "refresh-alpha",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mod.now_utc = lambda: mod.dt.datetime(2026, 4, 7, 9, 0, tzinfo=mod.dt.timezone.utc)
    result = mod.run_ingest(workspace=workspace, input_file=input_path)

    assert result["ok"] is True
    assert result["mode"] == "cpa_channel_ingest"
    assert result["change_class"] == "LIVE_GUARD_ONLY"
    assert result["accounts_total"] == 1
    assert result["run_id"] != ""
    assert result["store_path"].endswith("system/output/artifacts/cpa_channels/cpa_channels.sqlite3")
    assert result["exported_files"][0].endswith("alpha@fuuu.fun.json")
    assert Path(result["artifact_json"]).exists()
    assert Path(result["artifact_md"]).exists()
    latest_json = workspace / "system" / "output" / "review" / "latest_cpa_channel_ingest.json"
    assert latest_json.exists()
    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    assert payload["accounts"][0]["email"] == "alpha@fuuu.fun"
    assert payload["accounts"][0]["events"] == ["imported", "stored", "exported"]
