from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from lie_engine.cpa_channels.account_bundle import import_bundles
from lie_engine.cpa_channels.cpa_authfiles import write_cpa_authfile
from lie_engine.cpa_channels.token_store import CpaTokenStore


def _new_run_id() -> str:
    return uuid.uuid4().hex


def ingest_bundles_to_store(
    *,
    input_file: str | Path,
    store_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    bundles = import_bundles(input_file)
    store = CpaTokenStore(store_path)
    run_id = _new_run_id()
    accounts: list[dict[str, Any]] = []

    for bundle in bundles:
        account_id = store.upsert_account(
            email=bundle["email"],
            provider=bundle["provider"],
            chatgpt_account_id=bundle.get("chatgpt_account_id", ""),
            chatgpt_user_id=bundle.get("chatgpt_user_id", ""),
            plan_type=bundle.get("plan_type", ""),
            status="imported",
        )
        store.record_run_event(account_id, run_id=run_id, stage="imported", status="ok", detail=str(Path(input_file)))

        store.upsert_account(
            email=bundle["email"],
            provider=bundle["provider"],
            chatgpt_account_id=bundle.get("chatgpt_account_id", ""),
            chatgpt_user_id=bundle.get("chatgpt_user_id", ""),
            plan_type=bundle.get("plan_type", ""),
            status="stored",
        )
        if bundle.get("access_token"):
            store.upsert_token(
                account_id,
                kind="access_token",
                value=bundle["access_token"],
                source_file=str(input_file),
            )
        if bundle.get("refresh_token"):
            store.upsert_token(
                account_id,
                kind="refresh_token",
                value=bundle["refresh_token"],
                source_file=str(input_file),
            )
        if bundle.get("id_token"):
            store.upsert_token(
                account_id,
                kind="id_token",
                value=bundle["id_token"],
                source_file=str(input_file),
            )
        store.record_run_event(account_id, run_id=run_id, stage="stored", status="ok", detail="bundle persisted")

        export_result = write_cpa_authfile(bundle, output_dir=output_dir)
        store.record_run_event(
            account_id,
            run_id=run_id,
            stage="exported",
            status="ok",
            detail=Path(str(export_result["file"])).name,
        )

        snapshot = store.get_account_snapshot_raw(bundle["email"])
        events = [row["stage"] for row in store.list_run_events(account_id, run_id=run_id)]
        accounts.append(
            {
                "account_id": account_id,
                "email": bundle["email"],
                "provider": bundle["provider"],
                "exported_file": str(export_result["file"]),
                "events": events,
                "account": snapshot["account"],
            }
        )

    return {
        "run_id": run_id,
        "accounts_total": len(accounts),
        "accounts": accounts,
        "store_path": str(Path(store_path)),
        "output_dir": str(Path(output_dir)),
        "input_file": str(Path(input_file)),
    }
