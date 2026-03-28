import type { TerminalReadModel } from '../types/contracts';
import type { GraphEdge, GraphHomeModel, GraphLayoutNode, GraphNode, GraphPipeline } from './types';

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

export function buildGraphHomeModel(model: TerminalReadModel, pipelines: GraphPipeline[]): GraphHomeModel {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];

  addNode(nodes, {
    id: 'trade-hub',
    kind: 'strategy',
    label: '交易中枢',
    subtitle: '当前主控制图谱中心',
    importance: 10,
    status: model.surface.effective,
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
      importance: stage.importance,
      status: 'active',
    });
    connect(edges, nodes, 'trade-hub', stage.id, 'depends_on', 1);
  });

  addNode(nodes, {
    id: 'route-terminal-public',
    kind: 'route',
    label: '公开终端',
    subtitle: '执行 / 风控 / 门禁',
    importance: 7,
    destination: '/terminal/public',
    status: 'active',
  });
  addNode(nodes, {
    id: 'route-workspace-artifacts',
    kind: 'route',
    label: '工件池',
    subtitle: '研究判断 / artifact',
    importance: 7,
    destination: '/workspace/artifacts',
    status: 'active',
  });
  addNode(nodes, {
    id: 'route-workspace-contracts',
    kind: 'route',
    label: '契约层',
    subtitle: '入口拓扑 / 验收',
    importance: 7,
    destination: '/workspace/contracts',
    status: 'active',
  });
  addNode(nodes, {
    id: 'route-search',
    kind: 'route',
    label: '搜索',
    subtitle: '跨模块定位',
    importance: 6,
    destination: '/search',
    status: 'active',
  });

  connect(edges, nodes, 'pipeline-execution-risk', 'route-terminal-public', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-research-judgment', 'route-workspace-artifacts', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-research-judgment', 'route-workspace-contracts', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-feedback', 'route-search', 'routes_to', 1);
  connect(edges, nodes, 'pipeline-feedback', 'custom-pipeline', 'feeds', 1);

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
      destination: `/workspace/artifacts?artifact=${encodeURIComponent(artifact.id)}`,
      status: artifact.research_decision || artifact.status,
    });
    connect(edges, nodes, 'pipeline-research-judgment', id, 'validates', 0.9);
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
        emphasis: depth === 1 ? 1 : 0.55,
      };
    });

  return [
    {
      ...(center as GraphNode),
      x: 0,
      y: 0,
      z: 32,
      depth: 0,
      emphasis: 1.3,
    },
    ...placeRing(depthOne, 220, 1),
    ...placeRing(depthTwo, 400, 2),
  ];
}
