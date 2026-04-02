import type { TerminalReadModel } from '../types/contracts';
import { buildDrilldownRowId, buildWorkspacePageLink, buildWorkspaceLink, buildTerminalLink } from '../utils/focus-links';
import { buildSearchLink } from '../search/links';
import { safeDisplayValue } from '../utils/formatters';
import type { GraphEdge, GraphFocusState, GraphHomeModel, GraphLayoutNode, GraphNode, GraphPipeline } from './types';

const PIPELINE_STAGE_DEFS = [
  { id: 'pipeline-market-input', label: '市场输入', subtitle: '行情 / 事件 / 域摘要', importance: 8 },
  { id: 'pipeline-research-judgment', label: '研究判断', subtitle: 'source head / 研究主线', importance: 9 },
  { id: 'pipeline-trading-logic', label: '交易逻辑', subtitle: '信号 / 规则 / 自定义编排', importance: 8 },
  { id: 'pipeline-execution-risk', label: '执行与风控', subtitle: '终端 / 门禁 / 风险', importance: 8 },
  { id: 'pipeline-feedback', label: '复盘反馈', subtitle: '反馈 / 搜索 / 自定义管道', importance: 7 },
] as const;

function connect(
  edges: GraphEdge[],
  nodes: Map<string, GraphNode>,
  source: string,
  target: string,
  relation: GraphEdge['relation'],
  strength = 1,
) {
  if (!nodes.has(source) || !nodes.has(target)) return;
  edges.push({
    id: `${source}->${target}`,
    source,
    target,
    relation,
    strength,
  });
  nodes.get(source)?.downstream.push(target);
  nodes.get(target)?.upstream.push(source);
}

function addNode(nodes: Map<string, GraphNode>, node: Omit<GraphNode, 'upstream' | 'downstream'>) {
  nodes.set(node.id, {
    ...node,
    upstream: [],
    downstream: [],
  });
}

function slugify(value: unknown): string {
  return safeDisplayValue(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function uniqueNodeId(nodes: Map<string, GraphNode>, baseId: string): string {
  if (!nodes.has(baseId)) return baseId;
  let counter = 2;
  while (nodes.has(`${baseId}-${counter}`)) counter += 1;
  return `${baseId}-${counter}`;
}

function collectLineage(
  nodesById: Map<string, GraphNode>,
  startId: string,
  direction: 'upstream' | 'downstream',
  visited: Set<string>,
) {
  const node = nodesById.get(startId);
  if (!node) return;
  const nextIds = direction === 'upstream' ? node.upstream : node.downstream;
  nextIds.forEach((id) => {
    if (visited.has(id)) return;
    visited.add(id);
    collectLineage(nodesById, id, direction, visited);
  });
}

export function buildGraphFocusState(graph: GraphHomeModel, centerId: string): GraphFocusState {
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const resolvedCenterId = nodesById.has(centerId) ? centerId : graph.defaultCenterId;
  const activeIds = new Set<string>([resolvedCenterId]);
  collectLineage(nodesById, resolvedCenterId, 'upstream', activeIds);
  collectLineage(nodesById, resolvedCenterId, 'downstream', activeIds);
  return {
    activeNodeIds: graph.nodes.filter((node) => activeIds.has(node.id)).map((node) => node.id),
    activeEdgeIds: graph.edges
      .filter((edge) => activeIds.has(edge.source) && activeIds.has(edge.target))
      .map((edge) => edge.id),
  };
}

export function buildGraphHomeModel(model: TerminalReadModel, pipelines: GraphPipeline[]): GraphHomeModel {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const defaultFocus = model.workspace.defaultFocus || {};
  const artifactFocus = defaultFocus.artifact ? {
    artifact: defaultFocus.artifact,
    group: defaultFocus.group,
    panel: defaultFocus.panel,
    section: defaultFocus.section,
    search_scope: defaultFocus.search_scope,
    anchor: 'workspace-artifacts-focus-panel',
  } : {};

  addNode(nodes, {
    id: 'trade-hub',
    kind: 'strategy',
    label: '交易中枢',
    subtitle: '当前主控制图谱中心',
    importance: 10,
    status: model.surface.effective,
    detail: {
      sourceContract: 'graph.center',
      nextDestination: buildTerminalLink('public', { panel: 'signal-risk', section: 'action-stack' }),
    },
  });
  addNode(nodes, {
    id: 'custom-pipeline',
    kind: 'custom_pipeline',
    label: '自定义管道',
    subtitle: pipelines.length ? `${pipelines.length} 条已保存` : '暂无已保存管道',
    importance: 9,
    status: pipelines.length ? 'active' : 'idle',
  });

  PIPELINE_STAGE_DEFS.forEach((stage) => {
    addNode(nodes, {
      id: stage.id,
      kind: 'pipeline_stage',
      label: stage.label,
      subtitle: stage.subtitle,
      destination:
        stage.id === 'pipeline-market-input'
          ? buildTerminalLink('public', { anchor: 'terminal-data-regime', panel: 'data-regime' })
          : stage.id === 'pipeline-research-judgment'
            ? buildWorkspaceLink('artifacts', artifactFocus)
            : stage.id === 'pipeline-trading-logic'
              ? buildTerminalLink('public', { anchor: 'terminal-lab-review', panel: 'lab-review', artifact: defaultFocus.artifact })
              : stage.id === 'pipeline-execution-risk'
                ? buildTerminalLink('public', { anchor: 'terminal-signal-risk', panel: 'signal-risk' })
                : buildWorkspacePageLink('alignment', {}, 'alignment-actions'),
      importance: stage.importance,
      status: 'active',
      detail: {
        sourceContract: 'graph.pipelineStage',
      },
    });
    connect(edges, nodes, 'trade-hub', stage.id, 'depends_on', 1);
  });

  addNode(nodes, {
    id: 'route-terminal-public',
    kind: 'route',
    label: '公开终端',
    subtitle: '执行 / 风控 / 门禁',
    importance: 7,
    destination: buildTerminalLink('public', { anchor: 'terminal-signal-risk', panel: 'signal-risk' }),
    status: 'active',
    detail: {
      sourceContract: 'navigation.route',
      nextDestination: buildTerminalLink('public', { anchor: 'terminal-signal-risk', panel: 'signal-risk' }),
    },
  });
  addNode(nodes, {
    id: 'route-workspace-artifacts',
    kind: 'route',
    label: '工件池',
    subtitle: '研究判断 / artifact',
    importance: 7,
    destination: buildWorkspaceLink('artifacts', artifactFocus),
    status: 'active',
    detail: {
      sourceContract: 'navigation.route',
      nextDestination: buildWorkspaceLink('artifacts', artifactFocus),
    },
  });
  addNode(nodes, {
    id: 'route-workspace-contracts',
    kind: 'route',
    label: '契约层',
    subtitle: '入口拓扑 / 验收',
    importance: 7,
    destination: buildWorkspacePageLink('contracts', {}, 'contracts-source-heads'),
    status: 'active',
    detail: {
      sourceContract: 'navigation.route',
      nextDestination: buildWorkspacePageLink('contracts', {}, 'contracts-source-heads'),
    },
  });
  addNode(nodes, {
    id: 'route-search',
    kind: 'route',
    label: '搜索',
    subtitle: '跨模块定位',
    importance: 6,
    destination: buildSearchLink(defaultFocus.artifact || ''),
    status: 'active',
    detail: {
      sourceContract: 'navigation.route',
      nextDestination: buildSearchLink(defaultFocus.artifact || ''),
    },
  });

  connect(edges, nodes, 'pipeline-execution-risk', 'route-terminal-public', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-research-judgment', 'route-workspace-artifacts', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-research-judgment', 'route-workspace-contracts', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-feedback', 'route-search', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-feedback', 'custom-pipeline', 'feeds', 1);

  if (model.orchestration.guards.length) {
    addNode(nodes, {
      id: 'route-terminal-guards',
      kind: 'route',
      label: '门禁板',
      subtitle: 'orchestration / guards',
      importance: 6,
      destination: buildTerminalLink('public', { panel: 'orchestration', section: 'guards' }),
      status: 'active',
      detail: {
        sourceContract: 'orchestration.guards',
        nextDestination: buildTerminalLink('public', { panel: 'orchestration', section: 'guards' }),
      },
    });
    connect(edges, nodes, 'pipeline-execution-risk', 'route-terminal-guards', 'routes_to', 0.9);

    model.orchestration.guards.slice(0, 6).forEach((guard) => {
      const nodeId = uniqueNodeId(nodes, `guard-${slugify(guard.id || guard.label) || 'item'}`);
      addNode(nodes, {
        id: nodeId,
        kind: 'artifact',
        label: safeDisplayValue(guard.label || guard.id),
        subtitle: safeDisplayValue(guard.hint || guard.value || 'guard metric'),
        importance: 5,
        destination: buildTerminalLink('public', { panel: 'orchestration', section: 'guards' }),
        status: safeDisplayValue(guard.tone),
        detail: {
          sourceContract: 'orchestration.guards',
          gateValue: safeDisplayValue(guard.value),
          sourceReason: safeDisplayValue(guard.hint || guard.value),
          nextDestination: buildTerminalLink('public', { panel: 'orchestration', section: 'guards' }),
        },
      });
      connect(edges, nodes, 'route-terminal-guards', nodeId, 'validates', 0.75);
    });
  }

  if (model.signalRisk.actionQueue.length) {
    addNode(nodes, {
      id: 'route-terminal-action-stack',
      kind: 'route',
      label: '动作队列',
      subtitle: 'signal-risk / action-stack',
      importance: 7,
      destination: buildTerminalLink('public', { panel: 'signal-risk', section: 'action-stack' }),
      status: 'active',
      detail: {
        sourceContract: 'signalRisk.actionQueue',
        nextDestination: buildTerminalLink('public', { panel: 'signal-risk', section: 'action-stack' }),
      },
    });
    connect(edges, nodes, 'pipeline-execution-risk', 'route-terminal-action-stack', 'routes_to', 0.95);

    model.signalRisk.actionQueue.slice(0, 8).forEach((row, index) => {
      const symbol = safeDisplayValue(row.symbol || row.target || `queue-${index + 1}`);
      const action = safeDisplayValue(row.action || row.reason || row.done_when);
      const rowId = buildDrilldownRowId(row as unknown as Record<string, unknown>, ['symbol', 'action']);
      const queueDestination = buildTerminalLink('public', {
        panel: 'signal-risk',
        section: 'action-stack',
        artifact: safeDisplayValue(row.source_artifact) !== '—' ? safeDisplayValue(row.source_artifact) : undefined,
        symbol: safeDisplayValue(row.symbol) !== '—' ? safeDisplayValue(row.symbol) : undefined,
        row: rowId || undefined,
      });
      const nodeId = uniqueNodeId(nodes, `queue-${slugify(symbol) || index + 1}`);
      addNode(nodes, {
        id: nodeId,
        kind: 'artifact',
        label: symbol,
        subtitle: [action, safeDisplayValue(row.reason), safeDisplayValue(row.source_artifact)].filter(Boolean).join(' ｜ ') || 'action queue',
        importance: 6,
        destination: queueDestination,
        status: safeDisplayValue(row.state),
        detail: {
          sourceContract: 'signalRisk.actionQueue',
          sourceArtifact: safeDisplayValue(row.source_artifact) !== '—' ? safeDisplayValue(row.source_artifact) : undefined,
          sourceReason: [safeDisplayValue(row.reason), safeDisplayValue(row.action)].filter(Boolean).join(' ｜ ') || undefined,
          nextDestination: queueDestination,
        },
      });
      connect(edges, nodes, 'route-terminal-action-stack', nodeId, 'triggers', 0.85);
    });
  }

  if (model.feedback.projection) {
    addNode(nodes, {
      id: 'route-workspace-alignment',
      kind: 'route',
      label: '对齐页',
      subtitle: 'feedback / alignment actions',
      importance: 6,
      destination: buildWorkspacePageLink('alignment', {}, 'alignment-actions'),
      status: safeDisplayValue(model.feedback.projection.status) || 'active',
      detail: {
        sourceContract: 'feedback.projection.actions',
        nextDestination: buildWorkspacePageLink('alignment', {}, 'alignment-actions'),
      },
    });
    connect(edges, nodes, 'pipeline-feedback', 'route-workspace-alignment', 'routes_to', 0.9);

    model.feedback.projection.actions.slice(0, 6).forEach((action, index) => {
      const anchor = action.anchors?.find((row) => safeDisplayValue(row.artifact) !== '—') || model.feedback.projection?.anchors.find((row) => safeDisplayValue(row.artifact) !== '—');
      const actionKey = safeDisplayValue(action.feedback_id || action.recommended_action || `feedback-${index + 1}`);
      const feedbackAnchorId = safeDisplayValue(action.feedback_id) !== '—' ? `alignment-event-${safeDisplayValue(action.feedback_id)}` : undefined;
      const feedbackDestination = buildWorkspacePageLink(
        'alignment',
        {
          artifact: safeDisplayValue(anchor?.artifact) !== '—' ? safeDisplayValue(anchor?.artifact) : undefined,
          anchor: feedbackAnchorId,
        },
        feedbackAnchorId ? 'alignment-events' : 'alignment-actions',
      );
      const nodeId = uniqueNodeId(nodes, `feedback-action-${slugify(actionKey) || index + 1}`);
      addNode(nodes, {
        id: nodeId,
        kind: 'artifact',
        label: safeDisplayValue(action.recommended_action || action.feedback_id || `反馈动作 ${index + 1}`),
        subtitle: [
          safeDisplayValue(action.feedback_id),
          safeDisplayValue(action.impact_score),
        ].filter(Boolean).join(' ｜ ') || 'feedback action',
        importance: 6,
        destination: feedbackDestination,
        status: safeDisplayValue(model.feedback.projection?.status),
        detail: {
          sourceContract: 'feedback.projection.actions',
          feedbackId: safeDisplayValue(action.feedback_id) !== '—' ? safeDisplayValue(action.feedback_id) : undefined,
          anchorArtifact: safeDisplayValue(anchor?.artifact) !== '—' ? safeDisplayValue(anchor?.artifact) : undefined,
          sourceReason: safeDisplayValue(action.recommended_action) !== '—' ? safeDisplayValue(action.recommended_action) : undefined,
          nextDestination: feedbackDestination,
        },
      });
      connect(edges, nodes, 'pipeline-feedback', nodeId, 'feeds', 0.8);
      connect(edges, nodes, 'route-workspace-alignment', nodeId, 'routes_to', 0.8);
    });
  }

  model.workspace.domains.slice(0, 4).forEach((domain) => {
    const id = `market-${domain.id}`;
    addNode(nodes, {
      id,
      kind: 'market',
      label: domain.label || domain.id,
      subtitle: domain.headline || 'domain summary',
      importance: 5,
      status: domain.warning ? 'warning' : 'ok',
    });
    connect(edges, nodes, 'pipeline-market-input', id, 'feeds', 0.8);
  });

  model.workspace.artifactRows.slice(0, 8).forEach((artifact) => {
    const id = `artifact-${artifact.id}`;
    addNode(nodes, {
      id,
      kind: 'artifact',
      label: artifact.label || artifact.id,
      subtitle: artifact.path || artifact.category || 'artifact',
      importance: artifact.id === model.workspace.defaultFocus?.artifact ? 8 : 6,
      destination: buildWorkspaceLink('artifacts', {
        artifact: artifact.id,
        anchor: 'workspace-artifacts-focus-panel',
        group: artifact.artifact_group || defaultFocus.group,
        panel: defaultFocus.panel,
        section: defaultFocus.section,
        search_scope: defaultFocus.search_scope,
      }),
      status: artifact.research_decision || artifact.status,
      detail: {
        sourceContract: 'workspace.artifactRows',
        sourceArtifact: safeDisplayValue(artifact.id),
        sourcePath: safeDisplayValue(artifact.path) !== '—' ? safeDisplayValue(artifact.path) : undefined,
        sourceReason: safeDisplayValue(artifact.research_decision || artifact.status) !== '—' ? safeDisplayValue(artifact.research_decision || artifact.status) : undefined,
        nextDestination: buildWorkspaceLink('artifacts', {
          artifact: artifact.id,
          anchor: 'workspace-artifacts-focus-panel',
          group: artifact.artifact_group || defaultFocus.group,
          panel: defaultFocus.panel,
          section: defaultFocus.section,
          search_scope: defaultFocus.search_scope,
        }),
      },
    });
    connect(edges, nodes, 'pipeline-research-judgment', id, 'validates', 0.9);
  });

  const artifactIds = new Set(model.workspace.artifactRows.map((row) => safeDisplayValue(row.id)));
  model.workspace.sourceHeads
    .filter((row) => !artifactIds.has(safeDisplayValue(row.id)))
    .slice(0, 6)
    .forEach((row) => {
      const sourceHeadId = safeDisplayValue(row.id);
      if (!sourceHeadId || sourceHeadId === '—') return;
      addNode(nodes, {
        id: `source-head-${sourceHeadId}`,
        kind: 'artifact',
        label: safeDisplayValue(row.label || row.id),
        subtitle: safeDisplayValue(row.summary || row.path || 'source head'),
        importance: 7,
        destination: buildWorkspacePageLink('contracts', {}, `contracts-source-head-${sourceHeadId}`),
        status: safeDisplayValue(row.status || row.research_decision),
        detail: {
          sourceContract: 'workspace.sourceHeads',
          sourceArtifact: sourceHeadId,
          sourcePath: safeDisplayValue(row.path) !== '—' ? safeDisplayValue(row.path) : undefined,
          sourceReason: safeDisplayValue(row.summary || row.status || row.research_decision) !== '—' ? safeDisplayValue(row.summary || row.status || row.research_decision) : undefined,
          nextDestination: buildWorkspacePageLink('contracts', {}, `contracts-source-head-${sourceHeadId}`),
        },
      });
      connect(edges, nodes, 'pipeline-research-judgment', `source-head-${sourceHeadId}`, 'feeds', 0.85);
    });

  model.workspace.publicAcceptance.checks
    .slice(0, 4)
    .forEach((row) => {
      const rowId = safeDisplayValue(row.id);
      if (!rowId || rowId === '—') return;
      addNode(nodes, {
        id: `acceptance-check-${rowId}`,
        kind: 'route',
        label: safeDisplayValue(row.label || row.id),
        subtitle: safeDisplayValue(row.headline || row.generated_at_utc || row.report_path || 'contracts acceptance'),
        importance: 6,
        destination: safeDisplayValue(row.inspector_route) !== '—'
          ? row.inspector_route
          : buildWorkspacePageLink('contracts', {}, `contracts-check-${rowId}`),
        status: safeDisplayValue(row.status || (row.ok ? 'ok' : 'pending')),
        detail: {
          sourceContract: 'workspace.publicAcceptance.checks',
          sourceArtifact: 'dashboard_public_acceptance',
          sourcePath: safeDisplayValue(row.report_path) !== '—' ? safeDisplayValue(row.report_path) : undefined,
          sourceReason: safeDisplayValue(row.headline || row.failure_reason) !== '—' ? safeDisplayValue(row.headline || row.failure_reason) : undefined,
          nextDestination: safeDisplayValue(row.inspector_route) !== '—'
            ? row.inspector_route
            : buildWorkspacePageLink('contracts', {}, `contracts-check-${rowId}`),
        },
      });
      connect(edges, nodes, 'route-workspace-contracts', `acceptance-check-${rowId}`, 'routes_to', 0.8);
    });

  return {
    nodes: Array.from(nodes.values()),
    edges,
    defaultCenterId: 'trade-hub',
  };
}

export function buildGraphLayout(model: GraphHomeModel, centerId: string): GraphLayoutNode[] {
  const nodesById = new Map(model.nodes.map((node) => [node.id, node]));
  const center = nodesById.get(centerId) || nodesById.get(model.defaultCenterId) || model.nodes[0];
  const neighbors = new Set([...(center?.upstream || []), ...(center?.downstream || [])]);
  const depthOne = model.nodes.filter((node) => node.id !== center?.id && neighbors.has(node.id));
  const depthTwo = model.nodes.filter((node) => node.id !== center?.id && !neighbors.has(node.id));

  const placeRing = (ring: GraphNode[], radius: number, depth: 1 | 2): GraphLayoutNode[] =>
    ring.map((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(ring.length, 1);
      return {
        ...node,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        z: depth === 1 ? 16 : -8,
        depth,
        intensity: depth === 1 ? 1 : 0.55,
      };
    });

  return [
    {
      ...(center as GraphNode),
      x: 0,
      y: 0,
      z: 32,
      depth: 0,
      intensity: 1.3,
    },
    ...placeRing(depthOne, 220, 1),
    ...placeRing(depthTwo, 400, 2),
  ];
}
