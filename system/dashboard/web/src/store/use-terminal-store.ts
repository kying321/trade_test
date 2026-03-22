import { create } from 'zustand';
import type { DashboardSnapshot, SurfaceKey, TerminalReadModel } from '../types/contracts';
import { loadDashboardSurface } from '../adapters/load-surface';
import { buildTerminalReadModel } from '../adapters/read-model';

interface TerminalStore {
  requestedSurface: SurfaceKey;
  effectiveSurface: SurfaceKey;
  snapshotPath: string;
  snapshot: DashboardSnapshot | null;
  model: TerminalReadModel | null;
  loading: boolean;
  error: string;
  refresh: (surface?: SurfaceKey, allowFallback?: boolean) => Promise<void>;
  setSurface: (surface: SurfaceKey) => void;
}

export const useTerminalStore = create<TerminalStore>((set, get) => ({
  requestedSurface: 'public',
  effectiveSurface: 'public',
  snapshotPath: '/data/fenlie_dashboard_snapshot.json',
  snapshot: null,
  model: null,
  loading: false,
  error: '',
  setSurface: (surface) => set({ requestedSurface: surface }),
  refresh: async (surface, allowFallback = true) => {
    const targetSurface = surface || get().requestedSurface;
    set({ loading: true, error: '', requestedSurface: targetSurface });
    try {
      const loaded = await loadDashboardSurface(targetSurface, { allowFallback });
      set({
        snapshot: loaded.snapshot,
        model: buildTerminalReadModel(loaded),
        effectiveSurface: loaded.effectiveSurface,
        snapshotPath: loaded.snapshotPath,
        loading: false,
      });
    } catch (error) {
      set({
        snapshot: null,
        model: null,
        effectiveSurface: targetSurface,
        snapshotPath: targetSurface === 'internal'
          ? '/data/fenlie_dashboard_internal_snapshot.json'
          : '/data/fenlie_dashboard_snapshot.json',
        loading: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
}));
