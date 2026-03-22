import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadDashboardSurface } from './load-surface';

describe('loadDashboardSurface', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('falls back to public snapshot when internal surface is local-only and unavailable', async () => {
    const publicSnapshot = {
      surface_contracts: {
        internal: {
          availability: 'local_only',
        },
      },
    };

    vi.stubGlobal('fetch', vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.includes('fenlie_dashboard_internal_snapshot.json')) {
        return {
          ok: false,
          status: 404,
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => publicSnapshot,
      };
    }));

    const loaded = await loadDashboardSurface('internal');
    expect(loaded.requestedSurface).toBe('internal');
    expect(loaded.effectiveSurface).toBe('public');
    expect(loaded.snapshotPath).toBe('/data/fenlie_dashboard_snapshot.json');
    expect(loaded.warnings[0]).toContain('仅在本地/私有环境保留');
  });

  it('keeps internal request as an error when fallback is disabled', async () => {
    const fetchMock = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.includes('fenlie_dashboard_internal_snapshot.json')) {
        return {
          ok: false,
          status: 404,
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({}),
      };
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(loadDashboardSurface('internal', { allowFallback: false })).rejects.toThrow(
      '/data/fenlie_dashboard_internal_snapshot.json -> 404',
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
