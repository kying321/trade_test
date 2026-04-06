from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lie_engine" / "cpa_channels" / "token_store.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cpa_channel_token_store_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_token_store_persists_account_tokens_and_run_events(tmp_path: Path) -> None:
    mod = load_module()
    store = mod.CpaTokenStore(tmp_path / "cpa_channels.sqlite3")

    account_id = store.upsert_account(
        email="alpha@fuuu.fun",
        provider="codex",
        chatgpt_account_id="acct_alpha",
        chatgpt_user_id="user_alpha",
        plan_type="plus",
        status="stored",
    )
    store.upsert_token(
        account_id,
        kind="access_token",
        value="access-token-value",
        expires_at="2026-04-08T00:00:00+08:00",
        source_file="/tmp/alpha.json",
    )
    store.record_run_event(
        account_id,
        run_id="run-001",
        stage="stored",
        status="ok",
        detail="bundle persisted",
    )

    snapshot = store.get_account_snapshot_raw("alpha@fuuu.fun")
    events = store.list_run_events(account_id, run_id="run-001")

    assert snapshot["account"]["email"] == "alpha@fuuu.fun"
    assert snapshot["account"]["provider"] == "codex"
    assert snapshot["tokens"]["access_token"]["value"] == "access-token-value"
    assert events == [
        {
            "id": 1,
            "account_id": account_id,
            "run_id": "run-001",
            "stage": "stored",
            "status": "ok",
            "detail": "bundle persisted",
            "created_at": events[0]["created_at"],
        }
    ]


def test_token_store_run_id_filter_excludes_other_runs(tmp_path: Path) -> None:
    mod = load_module()
    store = mod.CpaTokenStore(tmp_path / "cpa_channels.sqlite3")
    account_id = store.upsert_account(email="beta@fuuu.fun", provider="codex")
    store.record_run_event(account_id, run_id="run-a", stage="stored", status="ok")
    store.record_run_event(account_id, run_id="run-b", stage="exported", status="ok")

    run_a = store.list_run_events(account_id, run_id="run-a")
    all_rows = store.list_run_events(account_id)

    assert [row["run_id"] for row in run_a] == ["run-a"]
    assert [row["run_id"] for row in all_rows] == ["run-a", "run-b"]
