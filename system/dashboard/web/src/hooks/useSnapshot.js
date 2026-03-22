import { useCallback, useEffect, useMemo, useState } from 'react';
import { buildDisplayModel } from '../lib/display-model';
import { normalizeSnapshot } from '../lib/snapshot-normalizers';

const SNAPSHOT_PATH = '/data/fenlie_dashboard_snapshot.json';

export function useSnapshot() {
  const [snapshot, setSnapshot] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`${SNAPSHOT_PATH}?ts=${Date.now()}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`snapshot fetch failed: ${response.status}`);
      const payload = await response.json();
      setSnapshot(payload);
    } catch (err) {
      setError(err?.message || 'snapshot load failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const normalized = useMemo(() => buildDisplayModel(normalizeSnapshot(snapshot)), [snapshot]);

  return {
    snapshot,
    normalized,
    loading,
    error,
    refresh,
    snapshotPath: SNAPSHOT_PATH,
  };
}
