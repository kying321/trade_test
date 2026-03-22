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
    expect(scripts['verify:release-checklist']).toContain('npm run build');
    expect(scripts['verify:release-checklist']).toContain('npm test');
    expect(scripts['verify:release-checklist']).toContain('npx tsc --noEmit');
    expect(scripts['verify:release-checklist']).toContain('npm run smoke:workspace-routes -- --skip-build');
    expect(scripts['verify:release-checklist']).toContain('npm run smoke:alignment-internal -- --skip-build');
    expect(scripts['verify:release-checklist']).toContain('npm run verify:public-surface -- --skip-workspace-build');
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
});
