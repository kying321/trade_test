from __future__ import annotations

import json
from pathlib import Path


def test_release_checklist_includes_internal_alignment_and_terminal_focus_smoke() -> None:
    package_json = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/package.json')
    payload = json.loads(package_json.read_text(encoding='utf-8'))
    script = payload['scripts']['verify:release-checklist']
    public_surface_script = payload['scripts']['verify:public-surface']
    python_contracts = payload['scripts'].get('test:dashboard-python-contracts', '')

    assert python_contracts
    assert 'test_build_dashboard_frontend_snapshot_script.py' in python_contracts
    assert 'test_run_operator_panel_refresh_script.py' in python_contracts
    assert 'test_run_dashboard_workspace_artifacts_smoke_script.py' in python_contracts
    assert 'test_run_dashboard_public_topology_smoke_script.py' in python_contracts
    assert 'test_run_dashboard_public_acceptance_script.py' in python_contracts
    assert 'test_publish_conversation_feedback_autopublish_internal_script.py' in python_contracts
    assert 'test_publish_conversation_feedback_manual_internal_script.py' in python_contracts
    assert 'test_build_conversation_feedback_projection_internal_script.py' in python_contracts
    assert 'test_dashboard_feedback_publish_refresh_contract.py' in python_contracts
    assert 'test_dashboard_feedback_publish_manual_contract.py' in python_contracts
    assert 'test_dashboard_release_checklist_contract.py' in python_contracts
    assert 'node ./scripts/run-python.mjs' in public_surface_script
    assert 'python3 ' not in public_surface_script
    assert '--strict-topology-tls' in public_surface_script
    assert 'npm run smoke:workspace-routes -- --skip-build' in script
    assert 'npm run smoke:alignment-internal -- --skip-build' in script
    assert 'npm run smoke:terminal-internal-focus -- --skip-build' in script
    assert 'npm run test:dashboard-python-contracts' in script
    assert script.index('npm run smoke:workspace-routes -- --skip-build') < script.index('npm run smoke:alignment-internal -- --skip-build')
    assert script.index('npm run smoke:alignment-internal -- --skip-build') < script.index('npm run smoke:terminal-internal-focus -- --skip-build')
    assert script.index('npm run test:dashboard-python-contracts') < script.index('npm run verify:public-surface -- --skip-workspace-build')


def test_runbook_release_checklist_mentions_internal_alignment_smoke() -> None:
    runbook = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_ENGINEERING_RUNBOOK.md').read_text(encoding='utf-8')

    assert 'npm run verify:release-checklist' in runbook
    assert 'npm run test:dashboard-python-contracts' in runbook
    assert 'run-python.mjs' in runbook
    assert 'npm run smoke:alignment-internal' in runbook
    assert 'internal alignment 浏览器 smoke' in runbook
    assert 'npm run smoke:terminal-internal-focus' in runbook
    assert 'terminal/internal focus browser smoke' in runbook
    assert 'test_publish_conversation_feedback_autopublish_internal_script.py' in runbook
    assert 'test_publish_conversation_feedback_manual_internal_script.py' in runbook
    assert 'test_build_conversation_feedback_projection_internal_script.py' in runbook
    assert 'test_dashboard_feedback_publish_refresh_contract.py' in runbook
    assert 'test_dashboard_feedback_publish_manual_contract.py' in runbook
    assert 'test_run_dashboard_public_acceptance_script.py' in runbook


def test_public_deploy_doc_mentions_shared_python_launcher() -> None:
    deploy_doc = Path('/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md').read_text(encoding='utf-8')

    assert 'npm run verify:public-surface -- --skip-workspace-build' in deploy_doc
    assert 'node ./scripts/run-python.mjs' in deploy_doc
    assert 'CONDA_PREFIX/bin/python3' in deploy_doc
    assert 'run_dashboard_public_acceptance.py' in deploy_doc
