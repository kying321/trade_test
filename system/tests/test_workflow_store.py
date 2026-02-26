from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.api.workflow_store import (  # noqa: E402
    default_write_session_context,
    evaluate_workflow_store_strict_mode,
    FileProposalWorkflowStore,
    resolve_proposal_workflow_store,
    WORKFLOW_STORE_ALLOW_FALLBACK_ENV,
    WORKFLOW_STORE_BACKEND_ENV,
    WORKFLOW_STORE_STRICT_MODE_ENV,
)


class WorkflowStoreTests(unittest.TestCase):
    def test_file_store_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            review_dir = Path(td)
            store = FileProposalWorkflowStore(review_dir=review_dir)
            now = datetime(2026, 2, 21, 10, 30, 0)
            record = {
                "proposal_id": "p_001",
                "status": "queued",
                "created_at": now.isoformat(),
                "proposer": "alice",
                "reason": "drift fix",
                "expected_impact": "lower drawdown",
                "rollback_anchor": "params_live_backup_2026-02-20.yaml",
                "changes": {"signal_confidence_min": 0.62},
                "status_history": [
                    {
                        "status": "queued",
                        "ts": now.isoformat(),
                        "actor": "alice",
                        "note": "init",
                    }
                ],
            }
            created = store.create_proposal(record)
            self.assertTrue(created.artifact_path.endswith("_param_proposal_p_001.json"))

            listed = store.list_proposals(limit=10)
            self.assertEqual(len(listed), 1)
            self.assertEqual(str(listed[0].record.get("proposal_id", "")), "p_001")
            fetched = store.get_proposal("p_001")
            self.assertIsNotNone(fetched)
            self.assertEqual(str(fetched.record.get("proposer", "")), "alice")

    def test_resolve_postgres_falls_back_to_file_when_dsn_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store, info = resolve_proposal_workflow_store(
                review_dir=td,
                file_glob="*_param_proposal_*.json",
                env={
                    WORKFLOW_STORE_BACKEND_ENV: "postgres",
                    WORKFLOW_STORE_ALLOW_FALLBACK_ENV: "true",
                },
            )
            self.assertIsInstance(store, FileProposalWorkflowStore)
            self.assertTrue(bool(info.get("degraded", False)))
            self.assertTrue(bool(info.get("fallback", False)))
            self.assertEqual(str(info.get("requested_backend", "")), "postgres")

    def test_resolve_postgres_without_fallback_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                resolve_proposal_workflow_store(
                    review_dir=td,
                    file_glob="*_param_proposal_*.json",
                    env={
                        WORKFLOW_STORE_BACKEND_ENV: "postgres",
                        WORKFLOW_STORE_ALLOW_FALLBACK_ENV: "false",
                    },
                )

    def test_default_write_session_context_prefers_proposer_when_user_missing(self) -> None:
        ctx = default_write_session_context(
            proposer="bob",
            env={},
        )
        self.assertEqual(str(ctx.get("role", "")), "platform_admin")
        self.assertEqual(str(ctx.get("visibility", "")), "internal")
        self.assertEqual(str(ctx.get("user", "")), "bob")

    def test_evaluate_workflow_store_strict_mode_flags_violation(self) -> None:
        policy = evaluate_workflow_store_strict_mode(
            store_info={
                "backend": "file",
                "requested_backend": "postgres",
                "degraded": True,
                "fallback": True,
            },
            env={WORKFLOW_STORE_STRICT_MODE_ENV: "true"},
        )
        self.assertTrue(bool(policy.get("active", False)))
        self.assertFalse(bool(policy.get("ok", True)))
        self.assertIn("backend_mismatch", str(policy.get("reason", "")))

    def test_evaluate_workflow_store_strict_mode_inactive_is_ok(self) -> None:
        policy = evaluate_workflow_store_strict_mode(
            store_info={"backend": "file"},
            env={WORKFLOW_STORE_STRICT_MODE_ENV: "false"},
        )
        self.assertFalse(bool(policy.get("active", True)))
        self.assertTrue(bool(policy.get("ok", False)))


if __name__ == "__main__":
    unittest.main()
