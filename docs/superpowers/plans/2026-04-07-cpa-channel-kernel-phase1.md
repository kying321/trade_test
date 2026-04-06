# CPA Channel Kernel Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the stable account-ingest/token-store/CPA-authfile-export components from the archived team session into Fenlie as a source-owned CPA channel kernel.

**Architecture:** Create a new Python package under `system/src/lie_engine/cpa_channels/` that owns bundle parsing, token/account persistence, run-event ledgering, and CPA auth-file export. Expose a single local-only ingest script that reads one or more account bundles, persists them to SQLite, exports CPA auth-files, and writes a review artifact for later CPA control-plane integration.

**Tech Stack:** Python 3, sqlite3, pytest, existing Fenlie `system/scripts` artifact pattern.

---

### Task 1: Add failing tests for the CPA channel kernel

**Files:**
- Create: `system/tests/test_cpa_channel_token_store.py`
- Create: `system/tests/test_cpa_channel_authfile_export.py`
- Create: `system/tests/test_run_cpa_channel_ingest_script.py`

- [ ] **Step 1: Write the failing token-store tests**
- [ ] **Step 2: Run `pytest -q system/tests/test_cpa_channel_token_store.py` and verify failure**
- [ ] **Step 3: Write the failing authfile-export tests**
- [ ] **Step 4: Run `pytest -q system/tests/test_cpa_channel_authfile_export.py` and verify failure**
- [ ] **Step 5: Write the failing ingest-runner tests**
- [ ] **Step 6: Run `pytest -q system/tests/test_run_cpa_channel_ingest_script.py` and verify failure**

### Task 2: Implement the CPA channel kernel package

**Files:**
- Create: `system/src/lie_engine/cpa_channels/__init__.py`
- Create: `system/src/lie_engine/cpa_channels/account_bundle.py`
- Create: `system/src/lie_engine/cpa_channels/token_store.py`
- Create: `system/src/lie_engine/cpa_channels/cpa_authfiles.py`
- Create: `system/src/lie_engine/cpa_channels/ingest_pipeline.py`

- [ ] **Step 1: Implement JSON/JSONL/JSON-stream account bundle loading**
- [ ] **Step 2: Implement SQLite-backed account/token/run-event store with run_id support**
- [ ] **Step 3: Implement JWT-derived CPA auth-file export helpers**
- [ ] **Step 4: Implement local-only ingest pipeline (import -> store -> export -> review summary)**
- [ ] **Step 5: Run token store and authfile-export tests until green**

### Task 3: Add a local-only script entrypoint

**Files:**
- Create: `system/scripts/run_cpa_channel_ingest.py`

- [ ] **Step 1: Implement CLI wrapper for workspace/input/store/output args**
- [ ] **Step 2: Make the script write a review artifact and latest symlink copy**
- [ ] **Step 3: Run `pytest -q system/tests/test_run_cpa_channel_ingest_script.py` until green**

### Task 4: Document and validate the Phase 1 kernel

**Files:**
- Modify: `system/docs/PROGRESS.md`

- [ ] **Step 1: Add a compact progress entry describing the new CPA channel kernel**
- [ ] **Step 2: Run `pytest -q system/tests/test_cpa_channel_token_store.py system/tests/test_cpa_channel_authfile_export.py system/tests/test_run_cpa_channel_ingest_script.py`**
- [ ] **Step 3: If useful, run one local ingest smoke with a fixture bundle and capture the artifact path**

