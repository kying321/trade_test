import { describe, expect, it } from 'vitest';
import { buildOpsLegacyLink, buildOverviewLink, buildResearchLegacyLink } from './route-contract';

function expectUrl(pathWithQuery: string, expectedPath: string, expectedQuery: Record<string, string>) {
  const [path, query = ''] = pathWithQuery.split('?');
  expect(path).toBe(expectedPath);
  const params = new URLSearchParams(query);
  expect(Object.fromEntries(params.entries())).toEqual(expectedQuery);
}

describe('route-contract', () => {
  it('keeps every shared query key when building overview-to-research links', () => {
    expect(
      buildResearchLegacyLink('artifacts', {
        artifact: 'price_action_breakout_pullback',
        group: 'research_mainline',
        view: 'public',
        symbol: 'ETHUSDT',
        window: 'forward-16',
      }),
    ).toBe(
      '/workspace/artifacts?artifact=price_action_breakout_pullback&group=research_mainline&view=public&symbol=ETHUSDT&window=forward-16',
    );
  });

  it('preserves panel section and row when navigating into terminal legacy routes', () => {
    expect(
      buildOpsLegacyLink('public', {
        panel: 'signal-risk',
        section: 'lane-pressure',
        row: 'ETHUSDT',
        view: 'public',
      }),
    ).toBe('/terminal/public?panel=signal-risk&section=lane-pressure&row=ETHUSDT&view=public');
  });

  it('builds overview links with shared query values intact', () => {
    expectUrl(
      buildOverviewLink({
        artifact: 'price_action_breakout_pullback',
        group: 'research_mainline',
        panel: 'lab-review',
        section: 'research-heads',
        row: 'price_action_breakout_pullback',
      }),
      '/overview',
      {
        artifact: 'price_action_breakout_pullback',
        group: 'research_mainline',
        panel: 'lab-review',
        section: 'research-heads',
        row: 'price_action_breakout_pullback',
      },
    );
  });
});
