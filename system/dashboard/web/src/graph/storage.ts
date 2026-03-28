import { useCallback, useEffect, useMemo, useState } from 'react';
import type { GraphPipeline, GraphPipelineStore } from './types';

export const GRAPH_PIPELINE_STORAGE_KEY = 'graph_home_pipelines_v1';

function createDefaultStore(): GraphPipelineStore {
  return { version: 1, pipelines: [] };
}

export function loadGraphPipelineStore(): GraphPipelineStore {
  if (typeof window === 'undefined') return createDefaultStore();
  try {
    const raw = window.localStorage.getItem(GRAPH_PIPELINE_STORAGE_KEY);
    if (!raw) return createDefaultStore();
    const parsed = JSON.parse(raw) as GraphPipelineStore;
    if (parsed.version !== 1 || !Array.isArray(parsed.pipelines)) return createDefaultStore();
    return parsed;
  } catch {
    return createDefaultStore();
  }
}

export function persistGraphPipelineStore(store: GraphPipelineStore) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(GRAPH_PIPELINE_STORAGE_KEY, JSON.stringify(store));
}

export function useGraphPipelineStore() {
  const [store, setStore] = useState<GraphPipelineStore>(() => loadGraphPipelineStore());

  useEffect(() => {
    persistGraphPipelineStore(store);
  }, [store]);

  const pipelines = store.pipelines;
  const defaultPipelineId = pipelines.find((pipeline) => pipeline.isDefault)?.id || pipelines[0]?.id || '';

  const upsertPipeline = useCallback((pipeline: GraphPipeline) => {
    setStore((current) => {
      const existing = current.pipelines.findIndex((item) => item.id === pipeline.id);
      if (existing >= 0) {
        const next = [...current.pipelines];
        next[existing] = pipeline;
        return { ...current, pipelines: next };
      }
      return { ...current, pipelines: [...current.pipelines, pipeline] };
    });
  }, []);

  const setDefaultPipeline = useCallback((pipelineId: string) => {
    setStore((current) => ({
      ...current,
      pipelines: current.pipelines.map((pipeline) => ({
        ...pipeline,
        isDefault: pipeline.id === pipelineId,
      })),
    }));
  }, []);

  const removePipeline = useCallback((pipelineId: string) => {
    setStore((current) => ({
      ...current,
      pipelines: current.pipelines.filter((pipeline) => pipeline.id !== pipelineId),
    }));
  }, []);

  const value = useMemo(() => ({
    store,
    pipelines,
    defaultPipelineId,
    upsertPipeline,
    setDefaultPipeline,
    removePipeline,
  }), [defaultPipelineId, pipelines, removePipeline, setDefaultPipeline, store, upsertPipeline]);

  return value;
}
