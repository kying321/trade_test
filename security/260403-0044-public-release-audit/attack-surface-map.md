# Attack Surface Map

## Public Entry Surfaces
- `README.md`
- `docs/investor/README.md`
- `system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md`
- `system/docs/OPENCLAW_CLOUD_BRIDGE.md`

## Sensitive-ish Code Surfaces
- `system/src/lie_engine/data/providers.py`
- `system/dashboard/web/src/pages/CpaPage.tsx`
- `system/dashboard/web/src/cpa/model.ts`
- `system/dashboard/web/src/cloudflare-preflight.test.js`

## Bulk Disclosure Surfaces
- `system/output/review/**`
- `system/docs/**`
- `docs/investor/**`
- `system/tests/**`
- `system/scripts/**`
