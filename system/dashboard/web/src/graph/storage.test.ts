import { describe, expect, it } from 'vitest';
import { GRAPH_PIPELINE_STORAGE_KEY, loadGraphPipelineStore, persistGraphPipelineStore } from './storage';

describe('graph pipeline storage', () => {
  it('persists and reloads graph_home_pipelines_v1 from localStorage', () => {
    const store = {
      version: 1 as const,
      pipelines: [
        {
          id: 'pipeline-1',
          name: '我的交易管道',
          nodeIds: ['trade-hub', 'pipeline-research-judgment'],
          layout: 'radial' as const,
          updatedAt: '2026-03-28T12:00:00Z',
          isDefault: true,
        },
      ],
    };

    persistGraphPipelineStore(store);
    expect(window.localStorage.getItem(GRAPH_PIPELINE_STORAGE_KEY)).toContain('pipeline-1');
    expect(loadGraphPipelineStore()).toEqual(store);
  });
});
