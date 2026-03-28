import React, { Suspense, useMemo } from 'react';
import * as echarts from 'echarts/core';
import { GraphChart } from 'echarts/charts';
import { TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { GraphEdge, GraphLayoutNode } from './types';

echarts.use([GraphChart, TooltipComponent, CanvasRenderer]);

const ReactECharts = React.lazy(() => import('echarts-for-react/lib/core'));

function nodeColor(kind: GraphLayoutNode['kind']) {
  if (kind === 'pipeline_stage') return '#5f7cff';
  if (kind === 'route') return '#0f8f77';
  if (kind === 'artifact') return '#b97511';
  if (kind === 'custom_pipeline') return '#7b55d0';
  if (kind === 'market') return '#1d73d8';
  return '#41536b';
}

export function GraphHomeRenderer({
  nodes,
  edges,
  centerId,
  onSelect,
}: {
  nodes: GraphLayoutNode[];
  edges: GraphEdge[];
  centerId: string;
  onSelect: (nodeId: string) => void;
}) {
  const option = useMemo(() => ({
    backgroundColor: 'transparent',
    animationDurationUpdate: 300,
    tooltip: {
      trigger: 'item',
      formatter: (params: { data?: GraphLayoutNode }) => {
        const node = params.data;
        return node ? `${node.label}<br/>${node.subtitle || ''}` : '';
      },
    },
    series: [
      {
        type: 'graph',
        layout: 'none',
        roam: true,
        draggable: true,
        coordinateSystem: undefined,
        data: nodes.map((node) => ({
          ...node,
          x: node.x,
          y: node.y,
          symbolSize: 46 + node.importance * 4,
          itemStyle: {
            color: nodeColor(node.kind),
            opacity: node.id === centerId ? 1 : node.emphasis,
            shadowBlur: node.id === centerId ? 28 : 12,
            shadowColor: 'rgba(15,23,42,0.28)',
          },
          label: {
            show: true,
            color: '#122033',
            fontSize: node.id === centerId ? 14 : 11,
            formatter: node.label,
          },
        })),
        edges: edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          lineStyle: {
            width: edge.source === centerId || edge.target === centerId ? 3 : 1.5,
            opacity: edge.source === centerId || edge.target === centerId ? 0.9 : 0.35,
            curveness: 0.14,
          },
        })),
        emphasis: {
          focus: 'adjacency',
        },
      },
    ],
  }), [centerId, edges, nodes]);

  return (
    <Suspense fallback={<div className="chart-placeholder" style={{ height: 560 }}>图谱加载中…</div>}>
      <ReactECharts
        echarts={echarts}
        option={option}
        style={{ height: 560 }}
        onEvents={{
          click: (params: { data?: GraphLayoutNode }) => {
            if (params.data?.id) onSelect(params.data.id);
          },
        }}
      />
    </Suspense>
  );
}
