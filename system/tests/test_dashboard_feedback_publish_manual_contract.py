from __future__ import annotations

import json
from pathlib import Path


def test_package_scripts_expose_feedback_publish_manual_variants() -> None:
    package_json = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/package.json')
    payload = json.loads(package_json.read_text(encoding='utf-8'))
    scripts = payload['scripts']

    assert 'feedback:publish-manual' in scripts
    assert 'feedback:publish-manual-refresh' in scripts
    assert 'publish_conversation_feedback_manual_internal.py' in scripts['feedback:publish-manual']

    chained = scripts['feedback:publish-manual-refresh']
    assert 'bash -lc' in chained
    assert '"$@"' in chained
    assert 'publish_conversation_feedback_manual_internal.py' in chained
    assert 'run_operator_panel_refresh.py --workspace ../../..' in chained


def test_runbook_mentions_feedback_publish_manual_flow() -> None:
    runbook = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_ENGINEERING_RUNBOOK.md').read_text(encoding='utf-8')

    assert 'npm run feedback:publish-manual' in runbook
    assert 'npm run feedback:publish-manual-refresh' in runbook
    assert 'conversation_feedback_events_internal.jsonl' in runbook
