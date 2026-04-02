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

function hasMissingSourceDetail(node: GraphLayoutNode) {
  return !node.detail?.sourceContract && !['strategy', 'pipeline_stage', 'custom_pipeline'].includes(node.kind);
}

export function GraphHomeRenderer({
  nodes,
  edges,
  centerId,
  activeNodeIds = [],
  activeEdgeIds = [],
  showFocusLayer = true,
  showWarningLayer = true,
  showDimLayer = true,
  onSelect,
}: {
  nodes: GraphLayoutNode[];
  edges: GraphEdge[];
  centerId: string;
  activeNodeIds?: string[];
  activeEdgeIds?: string[];
  showFocusLayer?: boolean;
  showWarningLayer?: boolean;
  showDimLayer?: boolean;
  onSelect: (nodeId: string) => void;
}) {
  const activeNodeSet = useMemo(() => new Set(activeNodeIds), [activeNodeIds]);
  const activeEdgeSet = useMemo(() => new Set(activeEdgeIds), [activeEdgeIds]);
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
          itemMissingSource: hasMissingSourceDetail(node),
          itemStyle: {
            color: nodeColor(node.kind),
            opacity: node.id === centerId
              ? 1
              : showDimLayer && !(showFocusLayer && activeNodeSet.has(node.id))
                ? 0.16
                : showFocusLayer && activeNodeSet.has(node.id)
                  ? Math.max(node.intensity, 0.72)
                  : node.intensity,
            shadowBlur: node.id === centerId ? 28 : showFocusLayer && activeNodeSet.has(node.id) ? 16 : showDimLayer ? 4 : 12,
            shadowColor: 'rgba(15,23,42,0.28)',
            borderColor: showWarningLayer && activeNodeSet.has(node.id) && hasMissingSourceDetail(node) ? '#ff8b5b' : undefined,
            borderWidth: showWarningLayer && activeNodeSet.has(node.id) && hasMissingSourceDetail(node) ? 3 : 0,
          },
          label: {
            show: true,
            color: node.id === centerId || !showDimLayer || (showFocusLayer && activeNodeSet.has(node.id)) ? '#122033' : '#8a94a6',
            fontSize: node.id === centerId ? 14 : 11,
            formatter: node.label,
          },
        })),
        edges: edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          lineStyle: {
            width: showFocusLayer && activeEdgeSet.has(edge.id) ? 3 : 1,
            opacity: showFocusLayer && activeEdgeSet.has(edge.id) ? 0.92 : showDimLayer ? 0.08 : 0.35,
            curveness: 0.14,
          },
        })),
        emphasis: {
          focus: 'adjacency',
        },
      },
    ],
  }), [activeEdgeSet, activeNodeSet, centerId, edges, nodes, showDimLayer, showFocusLayer, showWarningLayer]);

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
