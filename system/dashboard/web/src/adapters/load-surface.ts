import type { DashboardSnapshot, LoadedSurface, SurfaceKey } from '../types/contracts';
import { labelFor } from '../utils/dictionary';

const DEFAULT_PATHS: Record<SurfaceKey, string> = {
  public: '/data/fenlie_dashboard_snapshot.json',
  internal: '/data/fenlie_dashboard_internal_snapshot.json',
};
const t = (key: string) => labelFor(key);

type LoadSurfaceOptions = {
  allowFallback?: boolean;
};

async function fetchJson<T>(path: string): Promise<T> {
  const url = new URL(path, window.location.origin);
  url.searchParams.set('ts', String(Date.now()));
  const response = await fetch(url.toString(), { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function loadDashboardSurface(
  requestedSurface: SurfaceKey,
  options: LoadSurfaceOptions = {},
): Promise<LoadedSurface> {
  const { allowFallback = true } = options;
  const warnings: string[] = [];
  const primaryPath = DEFAULT_PATHS[requestedSurface];
  try {
    const snapshot = await fetchJson<DashboardSnapshot>(primaryPath);
    return {
      snapshot,
      requestedSurface,
      effectiveSurface: requestedSurface,
      snapshotPath: primaryPath,
      warnings,
    };
  } catch (error) {
    if (requestedSurface !== 'internal' || !allowFallback) {
      throw error;
    }
    const fallbackPath = DEFAULT_PATHS.public;
    const snapshot = await fetchJson<DashboardSnapshot>(fallbackPath);
    const internalAvailability = snapshot.surface_contracts?.internal?.availability;
    const reason = internalAvailability === 'local_only'
      ? t('surface_internal_local_only_notice')
      : `${t('surface_internal_unavailable_prefix')}：${String(error)}`;
    warnings.push(`${reason} ${t('surface_fallback_public_notice')}`);
    return {
      snapshot,
      requestedSurface,
      effectiveSurface: 'public',
      snapshotPath: fallbackPath,
      warnings,
    };
  }
}
