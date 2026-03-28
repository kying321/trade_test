import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

function loadScripts() {
  const packageJson = JSON.parse(readFileSync(join(process.cwd(), 'package.json'), 'utf8'));
  return packageJson.scripts ?? {};
}

describe('deploy script contract', () => {
  it('clears inherited proxy environment for Cloudflare wrangler commands', () => {
    const scripts = loadScripts();

    expect(scripts['cf:whoami']).toContain('env -u HTTP_PROXY -u HTTPS_PROXY');
    expect(scripts['cf:deploy']).toContain('env -u HTTP_PROXY -u HTTPS_PROXY');
    expect(scripts['cf:whoami']).toContain('cloudflare-wrangler.mjs whoami');
    expect(scripts['cf:deploy']).toContain('cloudflare-wrangler.mjs pages deploy');
  });

  it('exposes a single release-checklist verification entrypoint', () => {
    const scripts = loadScripts();

    expect(scripts['verify:release-checklist']).toBeTruthy();
    expect(scripts['test:dashboard-python-contracts']).toBeTruthy();
    expect(scripts['verify:release-checklist']).toContain('npm run build');
    expect(scripts['verify:release-checklist']).toContain('npm test');
    expect(scripts['verify:release-checklist']).toContain('npx tsc --noEmit');
    expect(scripts['test:dashboard-python-contracts']).toContain('node ./scripts/run-python.mjs');
    expect(scripts['test:dashboard-python-contracts']).toContain('-m pytest -q');
    expect(scripts['test:dashboard-python-contracts']).toContain('test_build_dashboard_frontend_snapshot_script.py');
    expect(scripts['test:dashboard-python-contracts']).toContain('test_run_operator_panel_refresh_script.py');
    expect(scripts['test:dashboard-python-contracts']).toContain('test_run_dashboard_workspace_artifacts_smoke_script.py');
    expect(scripts['test:dashboard-python-contracts']).toContain('test_run_dashboard_public_topology_smoke_script.py');
    expect(scripts['test:dashboard-python-contracts']).toContain('test_run_dashboard_public_acceptance_script.py');
    expect(scripts['test:dashboard-python-contracts']).toContain(
      'test_publish_conversation_feedback_autopublish_internal_script.py',
    );
    expect(scripts['test:dashboard-python-contracts']).toContain(
      'test_publish_conversation_feedback_manual_internal_script.py',
    );
    expect(scripts['test:dashboard-python-contracts']).toContain(
      'test_build_conversation_feedback_projection_internal_script.py',
    );
    expect(scripts['test:dashboard-python-contracts']).toContain('test_dashboard_feedback_publish_refresh_contract.py');
    expect(scripts['test:dashboard-python-contracts']).toContain('test_dashboard_feedback_publish_manual_contract.py');
    expect(scripts['test:dashboard-python-contracts']).toContain('test_dashboard_release_checklist_contract.py');
    expect(scripts['verify:release-checklist']).toContain('npm run test:dashboard-python-contracts');
    expect(scripts['verify:release-checklist']).toContain('npm run smoke:workspace-routes -- --skip-build');
    expect(scripts['verify:release-checklist']).toContain('npm run smoke:alignment-internal -- --skip-build');
    expect(scripts['verify:release-checklist']).toContain('npm run smoke:terminal-internal-focus -- --skip-build');
    expect(scripts['verify:release-checklist']).toContain('npm run verify:public-surface -- --skip-workspace-build');
    expect(
      scripts['verify:release-checklist'].indexOf('npm run test:dashboard-python-contracts'),
    ).toBeLessThan(scripts['verify:release-checklist'].indexOf('npm run verify:public-surface -- --skip-workspace-build'));
    expect(
      scripts['verify:release-checklist'].indexOf('npm run smoke:alignment-internal -- --skip-build'),
    ).toBeLessThan(scripts['verify:release-checklist'].indexOf('npm run smoke:terminal-internal-focus -- --skip-build'));
  });

  it('keeps feedback publish entrypoints visible to npm test and release workflows', () => {
    const scripts = loadScripts();

    expect(scripts['feedback:publish']).toContain('publish_conversation_feedback_autopublish_internal.py');
    expect(scripts['feedback:publish-refresh']).toContain('publish_conversation_feedback_autopublish_internal.py');
    expect(scripts['feedback:publish-refresh']).toContain('run_operator_panel_refresh.py --workspace ../../..');
    expect(scripts['feedback:publish-refresh']).toContain('"$@"');

    expect(scripts['feedback:publish-manual']).toContain('publish_conversation_feedback_manual_internal.py');
    expect(scripts['feedback:publish-manual-refresh']).toContain('publish_conversation_feedback_manual_internal.py');
    expect(scripts['feedback:publish-manual-refresh']).toContain('run_operator_panel_refresh.py --workspace ../../..');
    expect(scripts['feedback:publish-manual-refresh']).toContain('"$@"');
  });

  it('fails fast on Cloudflare auth before build/deploy', () => {
    const scripts = loadScripts();

    expect(scripts['cf:preflight']).toContain('cloudflare-preflight.mjs');
    expect(scripts['cf:deploy']).toContain('npm run cf:preflight');
    expect(scripts['cf:deploy']).toContain('npm run build');
    expect(scripts['cf:deploy'].indexOf('npm run cf:preflight')).toBeLessThan(scripts['cf:deploy'].indexOf('npm run build'));
  });

  it('exposes a dedicated manual-probe browser smoke entrypoint for internal alignment', () => {
    const scripts = loadScripts();

    expect(scripts['smoke:alignment-internal-manual-probe']).toContain('run_dashboard_workspace_artifacts_smoke.py');
    expect(scripts['smoke:alignment-internal-manual-probe']).toContain('--mode internal_alignment_manual_probe');
  });

  it('exposes a dedicated terminal/internal focus browser smoke entrypoint', () => {
    const scripts = loadScripts();

    expect(scripts['smoke:terminal-internal-focus']).toContain('run_dashboard_workspace_artifacts_smoke.py');
    expect(scripts['smoke:terminal-internal-focus']).toContain('--mode internal_terminal_focus');
  });

  it('exposes a dedicated graph-home browser smoke entrypoint', () => {
    const scripts = loadScripts();

    expect(scripts['smoke:graph-home']).toContain('run_dashboard_workspace_artifacts_smoke.py');
    expect(scripts['smoke:graph-home']).toContain('--mode graph_home');
  });

  it('exposes a dedicated graph-home narrow browser smoke entrypoint', () => {
    const scripts = loadScripts();

    expect(scripts['smoke:graph-home-narrow']).toContain('run_dashboard_workspace_artifacts_smoke.py');
    expect(scripts['smoke:graph-home-narrow']).toContain('--mode graph_home_narrow');
  });

  it('routes dashboard python entrypoints through the shared python launcher', () => {
    const scripts = loadScripts();
    const sharedLauncher = 'node ./scripts/run-python.mjs';
    const expectedScripts = [
      'refresh:panel',
      'refresh:data',
      'strip:internal',
      'verify:public-surface',
      'test:dashboard-python-contracts',
      'smoke:workspace-routes',
      'smoke:workspace-artifacts',
      'smoke:alignment-internal',
      'smoke:alignment-internal-manual-probe',
      'smoke:terminal-internal-focus',
      'smoke:graph-home',
      'smoke:graph-home-narrow',
      'feedback:publish',
      'feedback:publish-refresh',
      'feedback:publish-manual',
      'feedback:publish-manual-refresh',
    ];

    for (const name of expectedScripts) {
      expect(scripts[name]).toContain(sharedLauncher);
      expect(scripts[name]).not.toContain('python3 ');
    }
  });
});
