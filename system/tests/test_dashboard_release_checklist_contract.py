from __future__ import annotations

import json
from pathlib import Path


def test_release_checklist_includes_internal_alignment_smoke() -> None:
    package_json = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/package.json')
    payload = json.loads(package_json.read_text(encoding='utf-8'))
    script = payload['scripts']['verify:release-checklist']

    assert 'npm run smoke:workspace-routes -- --skip-build' in script
    assert 'npm run smoke:alignment-internal -- --skip-build' in script
    assert script.index('npm run smoke:workspace-routes -- --skip-build') < script.index('npm run smoke:alignment-internal -- --skip-build')


def test_runbook_release_checklist_mentions_internal_alignment_smoke() -> None:
    runbook = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_ENGINEERING_RUNBOOK.md').read_text(encoding='utf-8')

    assert 'npm run verify:release-checklist' in runbook
    assert 'npm run smoke:alignment-internal' in runbook
    assert 'internal alignment 浏览器 smoke' in runbook
