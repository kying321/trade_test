# Recommendations

1. Run a second pass that converts public-facing docs from absolute paths to repo-relative paths across:
   - `docs/investor/**`
   - `system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md`
   - `system/docs/OPENCLAW_CLOUD_BRIDGE.md`
2. Decide whether `system/output/review/**` belongs in the public packaging surface; if not, move or filter it.
3. Review `system/src/lie_engine/data/providers.py` request constants for provenance and rotation policy.
4. Add a lightweight pre-commit / CI check that rejects new `/Users/<name>/...` paths in public docs.
