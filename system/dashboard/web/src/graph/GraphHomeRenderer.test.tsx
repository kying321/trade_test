import { render, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { GraphHomeRenderer } from './GraphHomeRenderer';
import type { GraphEdge, GraphLayoutNode } from './types';

let lastProps: any = null;

vi.mock('echarts-for-react/lib/core', () => ({
  default: (props: any) => {
    lastProps = props;
    return <div data-testid="echarts-renderer-mock" />;
  },
}));

function createNodes(): GraphLayoutNode[] {
  return [
    {
      id: 'trade-hub',
      kind: 'strategy',
      label: '交易中枢',
      importance: 10,
      upstream: [],
      downstream: ['missing-source-node'],
      x: 0,
      y: 0,
      z: 24,
      depth: 0,
      intensity: 1,
      detail: { sourceContract: 'graph.center' },
    },
    {
      id: 'missing-source-node',
      kind: 'market',
      label: '国内商品',
      importance: 5,
      upstream: ['trade-hub'],
      downstream: [],
      x: 100,
      y: 0,
      z: 16,
      depth: 1,
      intensity: 0.8,
    },
    {
      id: 'source-owned-node',
      kind: 'artifact',
      label: 'ETH 15m 主线',
      importance: 6,
      upstream: ['trade-hub'],
      downstream: [],
      x: -100,
      y: 0,
      z: 16,
      depth: 1,
      intensity: 0.8,
      detail: { sourceContract: 'workspace.artifactRows' },
    },
  ];
}

function createEdges(): GraphEdge[] {
  return [
    { id: 'trade-hub->missing-source-node', source: 'trade-hub', target: 'missing-source-node', relation: 'feeds', strength: 1 },
    { id: 'trade-hub->source-owned-node', source: 'trade-hub', target: 'source-owned-node', relation: 'feeds', strength: 1 },
  ];
}

describe('GraphHomeRenderer', () => {
  beforeEach(() => {
    lastProps = null;
  });

  it('adds warning styling for active nodes missing source-owned detail', async () => {
    render(
      <GraphHomeRenderer
        nodes={createNodes()}
        edges={createEdges()}
        centerId="trade-hub"
        activeNodeIds={['trade-hub', 'missing-source-node']}
        activeEdgeIds={['trade-hub->missing-source-node']}
        onSelect={() => {}}
      />,
    );

    await waitFor(() => {
      expect(lastProps?.option).toBeTruthy();
    });

    const graphSeries = lastProps.option.series[0];
    const warningNode = graphSeries.data.find((node: any) => node.id === 'missing-source-node');
    const sourceOwnedNode = graphSeries.data.find((node: any) => node.id === 'source-owned-node');

    expect(warningNode.itemStyle.borderColor).toBe('#ff8b5b');
    expect(warningNode.itemStyle.borderWidth).toBe(3);
    expect(sourceOwnedNode.itemStyle.borderColor).toBeUndefined();
  });

  it('lets focus warning and dim layers be disabled independently', async () => {
    render(
      <GraphHomeRenderer
        nodes={createNodes()}
        edges={createEdges()}
        centerId="trade-hub"
        activeNodeIds={['trade-hub', 'missing-source-node']}
        activeEdgeIds={['trade-hub->missing-source-node']}
        showFocusLayer={false}
        showWarningLayer={false}
        showDimLayer={false}
        onSelect={() => {}}
      />,
    );

    await waitFor(() => {
      expect(lastProps?.option).toBeTruthy();
    });

    const graphSeries = lastProps.option.series[0];
    const warningNode = graphSeries.data.find((node: any) => node.id === 'missing-source-node');
    const activeEdge = graphSeries.edges.find((edge: any) => edge.source === 'trade-hub' && edge.target === 'missing-source-node');

    expect(warningNode.itemStyle.borderColor).toBeUndefined();
    expect(warningNode.itemStyle.borderWidth).toBe(0);
    expect(warningNode.itemStyle.opacity).toBeGreaterThan(0.5);
    expect(activeEdge.lineStyle.width).toBe(1);
    expect(activeEdge.lineStyle.opacity).toBe(0.35);
  });
});
