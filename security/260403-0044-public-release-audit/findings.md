# Findings

## [MEDIUM][fixed-in-working-tree] Public README entry points leaked local absolute paths
- Location: `README.md`, `docs/investor/README.md`
- Evidence: prior tracked content referenced `/Users/jokenrobot/...` directly in public entry docs
- Attack scenario: external readers learn local username and workstation directory layout; public packaging appears private and host-bound
- Impact: information disclosure + poor portability on the main public surface
- Confidence: high
- STRIDE: information disclosure
- OWASP: A05 Security Misconfiguration
- Mitigation: replace public entry docs with repo-relative paths; keep host-specific instructions out of top-level public docs
- Status: fixed in current working tree, not yet committed

## [MEDIUM][open] Repo still contains widespread local-environment disclosure in tracked docs and artifacts
- Location: `docs/investor/**`, `system/docs/**`, `system/output/review/**`, selected `system/scripts/**` and `system/tests/**`
- Evidence: `git grep -nE '(/Users/|file:///|127\\.0\\.0\\.1|localhost)' -- .` returned 1695 hit lines across 153 tracked files after the public README cleanup
- Attack scenario: a public repo/package reveals operator username, local filesystem layout, internal review artifact names, bridge ports, and host assumptions
- Impact: non-credential information disclosure, reduced packaging quality, easier operator-environment fingerprinting
- Confidence: high
- STRIDE: information disclosure
- OWASP: A05 Security Misconfiguration
- Mitigation: split public vs internal docs, convert public docs to repo-relative paths, and exclude or sanitize review artifact indexes before broad publication

## [LOW][open] Provider request constants may be mistaken for secrets and require manual provenance review
- Location: `system/src/lie_engine/data/providers.py:224-227`
- Evidence: static headers include `x-app-id`, `x-csrf-token`, and `x-version`
- Attack scenario: public readers or scanners may flag these as leaked credentials; if provenance is not public-doc derived, they may become a future rotation burden
- Impact: currently no proof of exploitability, but they increase false-positive secret risk and release-review friction
- Confidence: medium
- STRIDE: information disclosure
- OWASP: A02 Cryptographic Failures / A05 Security Misconfiguration (classification-by-risk-context)
- Mitigation: document provenance or externalize non-essential constants into config/comments; if they are revocable provider-issued values, rotate before public promotion

## [INFO][expected] Token-shaped values in CPA page and Cloudflare tests are placeholders, not confirmed secrets
- Location: `system/dashboard/web/src/pages/CpaPage.tsx:115`, `system/dashboard/web/src/cloudflare-preflight.test.js:85-86,103-104`
- Evidence: values are `$TOKEN$`, `cached-oauth-token`, and `cached-refresh-token`
- Impact: no confirmed secret exposure from these literals
- Confidence: high
- STRIDE: none
- OWASP: none
- Mitigation: keep placeholders obviously synthetic and avoid production-like prefixes in future fixtures where practical
