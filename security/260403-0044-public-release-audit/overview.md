# Public Release Security Audit Overview

- Run mode: security
- Goal: audit tracked repository content for public-release sensitive information and exposure risk
- Scope: current `lie` branch working tree, excluding untracked `.codex/`
- Baseline metric: no confirmed hardcoded production credentials in tracked files
- Residual risk theme: local-environment disclosure remains widespread in docs, review artifacts, and some scripts
- Fixed during this pass: root `README.md` and `docs/investor/README.md` no longer expose `/Users/jokenrobot/...` absolute paths
