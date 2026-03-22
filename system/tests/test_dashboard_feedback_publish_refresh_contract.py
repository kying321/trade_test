from __future__ import annotations

import json
from pathlib import Path


def test_package_scripts_expose_feedback_publish_refresh() -> None:
    package_json = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/package.json')
    payload = json.loads(package_json.read_text(encoding='utf-8'))
    scripts = payload['scripts']

    assert 'feedback:publish' in scripts
    assert 'feedback:publish-refresh' in scripts
    command = scripts['feedback:publish-refresh']
    assert 'bash -lc' in command
    assert '"$@"' in command
    assert 'publish_conversation_feedback_autopublish_internal.py' in command
    assert 'run_operator_panel_refresh.py --workspace ../../..' in command


def test_runbook_mentions_feedback_publish_refresh() -> None:
    runbook = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_ENGINEERING_RUNBOOK.md').read_text(encoding='utf-8')

    assert 'npm run feedback:publish-refresh' in runbook
    assert '发布 + refresh 一条链' in runbook
