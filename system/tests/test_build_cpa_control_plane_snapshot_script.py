from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_cpa_control_plane_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_cpa_control_plane_snapshot_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_snapshot_rolls_up_handoff_sources_and_kernel_state(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    public_dir = workspace / "system" / "dashboard" / "web" / "public"
    review_dir = workspace / "system" / "output" / "review"
    source_root = tmp_path / "mac_tools_data"
    public_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    (source_root / "registered_success_active20.csv").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "registered_success_active20.csv").write_text(
        "email,password,timestamp\n"
        "a@fuuu.fun,p1,2026-03-21 11:19:40\n"
        "b@fuuu.fun,p2,2026-03-21 11:24:07\n",
        encoding="utf-8",
    )
    (source_root / "registered_new_unmounted.csv").write_text(
        "email,password,timestamp\n"
        "c@fuuu.fun,p3,2026-03-29 18:51:17\n",
        encoding="utf-8",
    )
    (source_root / "cpa_non_active_review_queue.csv").write_text(
        "email,bucket,reason,mgmt_status,disabled,auth_name,has_local_authfile,local_authfile,artifact_html,artifact_title,artifact_flags,status_message,updated_at\n"
        "x@fuuu.fun,retry_candidate,needs remount,,False,,,data/oauth_x.html,确认一下你的年龄,about_you,,\n"
        "y@fuuu.fun,blocked_about_you,stuck on age/about-you,,False,,,data/oauth_y.html,确认一下你的年龄,about_you,,\n",
        encoding="utf-8",
    )
    (source_root / "cpa_no_retry_deactivated_accounts.csv").write_text(
        "email,bucket,reason,mgmt_status,disabled,auth_name,has_local_authfile,local_authfile,artifact_html,artifact_title,artifact_flags,status_message,updated_at\n"
        "z@fuuu.fun,no_retry_deactivated,deactivated,error,False,codex-z@fuuu.fun-free.json,True,/tmp/z.json,data/oauth_z.html,糟糕，出错了！,deactivated,token invalidated,2026-03-29T21:12:07+08:00\n",
        encoding="utf-8",
    )
    (source_root / "cpa_account_inventory.json").write_text(
        json.dumps(
            [
                {"email": "x@fuuu.fun", "bucket": "retry_candidate"},
                {"email": "y@fuuu.fun", "bucket": "blocked_about_you"},
                {"email": "z@fuuu.fun", "bucket": "no_retry_deactivated"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_root / "active20_acceptance_20260329_212018.json").write_text(
        json.dumps(
            {
                "target_count": 20,
                "registered_valid_accounts": 20,
                "active_target_authfiles": 20,
                "accepted": True,
                "store_verified_target_accounts": 20,
                "store_accepted": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_cpa_channel_ingest.json").write_text(
        json.dumps(
            {
                "mode": "cpa_channel_ingest",
                "status": "ok",
                "run_id": "run-kernel-1",
                "accounts_total": 1,
                "accounts": [
                    {"email": "phase1-demo@fuuu.fun", "events": ["imported", "stored", "exported"]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = mod.build_snapshot(
        workspace=workspace,
        public_dir=public_dir,
        source_root=source_root,
    )

    assert payload["ok"] is True
    assert payload["mode"] == "cpa_control_plane_snapshot"
    assert payload["summary"]["historical_success_total"] == 2
    assert payload["summary"]["new_unmounted_total"] == 1
    assert payload["summary"]["inventory_total"] == 3
    assert payload["summary"]["retry_candidate_total"] == 1
    assert payload["summary"]["blocked_about_you_total"] == 1
    assert payload["summary"]["no_retry_deactivated_total"] == 1
    assert payload["summary"]["active_target_authfiles"] == 20
    assert payload["summary"]["latest_kernel_accounts_total"] == 1
    assert payload["latest_kernel_run_id"] == "run-kernel-1"
    assert Path(payload["artifact_json"]).exists()
    assert (public_dir / "data" / "cpa_control_plane_snapshot.json").exists()
