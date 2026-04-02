import { useMemo, useState } from 'react';
import { ActionButton, ActionLink, FilterChip } from '../components/control-primitives';
import { Badge, KeyValueGrid, PanelCard } from '../components/ui-kit';
import { buildSearchLink } from '../search/links';
import type { TerminalReadModel } from '../types/contracts';
import { buildGraphFocusState, buildGraphHomeModel, buildGraphLayout } from '../graph/model';
import { GraphHomeRenderer } from '../graph/GraphHomeRenderer';
import { useGraphPipelineStore } from '../graph/storage';
import type { GraphNode, GraphPipeline } from '../graph/types';
import { labelFor } from '../utils/dictionary';
import { safeDisplayValue } from '../utils/formatters';

function createPipeline(name: string): GraphPipeline {
  return {
    id: `pipeline-${Math.random().toString(36).slice(2, 10)}`,
    name,
    nodeIds: [],
    layout: 'radial',
    updatedAt: new Date().toISOString(),
    isDefault: true,
  };
}

function nodeLabel(node: GraphNode | undefined) {
  return node?.label || '未选择节点';
}

const FILTER_OPTIONS: Array<{ value: 'all' | GraphNode['kind']; label: string }> = [
  { value: 'all', label: '全部' },
  { value: 'pipeline_stage', label: '管道阶段' },
  { value: 'market', label: '市场输入域' },
  { value: 'route', label: '子页面' },
  { value: 'artifact', label: '研究工件' },
  { value: 'custom_pipeline', label: '自定义管道' },
];

function kindLabel(kind: GraphNode['kind'] | undefined) {
  if (kind === 'pipeline_stage') return '管道阶段';
  if (kind === 'market') return '市场输入域';
  if (kind === 'strategy') return '交易中枢';
  if (kind === 'artifact') return '研究工件';
  if (kind === 'route') return '子页面';
  if (kind === 'custom_pipeline') return '自定义管道';
  return '—';
}

function explainNode(node: GraphNode | undefined, defaultPipelineName: string) {
  if (!node) {
    return {
      title: '为什么现在看它',
      summary: '当前还没有选中节点。',
      detail: '先从交易中枢或任意一等阶段节点开始。',
    };
  }
  if (node.kind === 'strategy') {
    return {
      title: '为什么现在看它',
      summary: '它是默认首页中心',
      detail: '所有市场输入、研究判断、交易逻辑、执行风控与复盘反馈都围绕它重新居中展开。',
    };
  }
  if (node.kind === 'pipeline_stage') {
    return {
      title: '为什么现在看它',
      summary: '它是一等阶段节点',
      detail: '这一层用来承接中枢的一级工作带，方便你判断当前应该往哪个阶段继续深跳。',
    };
  }
  if (node.kind === 'market') {
    return {
      title: '为什么现在看它',
      summary: '它是市场输入节点',
      detail: '这里承载当前域摘要/行情 headline，对后面的研究判断和交易逻辑提供输入。',
    };
  }
  if (node.kind === 'artifact') {
    if (node.detail?.sourceContract === 'signalRisk.actionQueue') {
      return {
        title: '为什么现在看它',
        summary: '它是执行动作队列里的 source-owned 行',
        detail: '这里把动作、原因、源头工件和下一跳绑定在一起，适合直接判断执行与风控链是否需要介入。',
      };
    }
    if (node.detail?.sourceContract === 'feedback.projection.actions') {
      return {
        title: '为什么现在看它',
        summary: '它是反馈投射里的推荐动作',
        detail: '这里承接用户意图、反馈编号和锚点工件，适合判断是否需要跳到对齐页继续处理。',
      };
    }
    if (node.detail?.sourceContract === 'workspace.sourceHeads') {
      return {
        title: '为什么现在看它',
        summary: '它是契约层 source head 节点',
        detail: '这里保留源头路径与主线摘要，适合直接检查当前 source-owned 合约头是否需要深跳到契约层。',
      };
    }
    return {
      title: '为什么现在看它',
      summary: '它是当前研究工件',
      detail: '这里连接研究判断与具体证据材料，适合继续跳到工件池查看上下文。',
    };
  }
  if (node.kind === 'route') {
    return {
      title: '为什么现在看它',
      summary: '它是深跳入口',
      detail: '当前链路已经可以落到具体子网页，适合从图谱视角切回可操作页面。',
    };
  }
  return {
    title: '为什么现在看它',
    summary: '它是你的自定义工作流',
    detail: `当前默认管道为「${defaultPipelineName}」，你可以把常看的节点串成自己的操作顺序。`,
  };
}

function buildSourceOwnedRows(node: GraphNode | undefined) {
  if (!node?.detail) return [];
  const rows = [
    { key: 'sourceContract', label: '信源契约', value: node.detail.sourceContract || '—' },
    { key: 'sourceArtifact', label: '源头工件', value: node.detail.sourceArtifact || '—' },
    { key: 'sourcePath', label: '源头路径', value: node.detail.sourcePath || '—' },
    { key: 'feedbackId', label: '反馈ID', value: node.detail.feedbackId || '—' },
    { key: 'anchorArtifact', label: '锚点工件', value: node.detail.anchorArtifact || '—' },
    { key: 'gateValue', label: '当前门禁', value: node.detail.gateValue || '—' },
    { key: 'sourceReason', label: '触发原因', value: node.detail.sourceReason || '—' },
    { key: 'nextDestination', label: '下一跳', value: node.detail.nextDestination || '—' },
  ].filter((row) => safeDisplayValue(row.value) !== '—');
  return rows.map((row) => ({
    ...row,
    kind: row.key.includes('Path') || row.key.includes('Destination') ? 'path' as const : undefined,
    showRaw: row.key.includes('Path') || row.key.includes('Destination'),
  }));
}

function buildImpactExplanation(node: GraphNode | undefined, upstreamNodes: GraphNode[], downstreamNodes: GraphNode[]) {
  if (!node) {
    return {
      summary: '当前还没有形成影响链。',
      detail: '请先选择一个节点。',
    };
  }
  const upstream = upstreamNodes.map((item) => item.label).join(' / ') || '交易中枢';
  const downstream = downstreamNodes.map((item) => item.label).join(' / ') || '当前无下游扩散';
  const reason = node.detail?.sourceReason || node.subtitle || '暂无附加原因';
  return {
    summary: `${upstream} → ${node.label}`,
    detail: `${reason} ｜ 下游：${downstream}`,
  };
}

function recommendNextStep(node: GraphNode | undefined, nodes: GraphNode[]) {
  if (!node) return '先回到交易中枢';
  if (node.destination) return `进入${node.label}`;
  const downstreamNodes = node.downstream
    .map((id) => nodes.find((candidate) => candidate.id === id))
    .filter(Boolean) as GraphNode[];
  const routeNode = downstreamNodes.find((candidate) => candidate.kind === 'route');
  if (routeNode) return `进入${routeNode.label}`;
  const nextNode = downstreamNodes[0];
  if (nextNode) return `继续展开到${nextNode.label}`;
  return '回到交易中枢';
}

function shouldShowResearchAuditLinks(node: GraphNode | undefined) {
  return Boolean(node && ['trade-hub', 'pipeline-feedback', 'route-search', 'route-workspace-contracts', 'custom-pipeline'].includes(node.id));
}

function normalizeHashRoute(route: string | undefined): string {
  const raw = String(route || '').trim();
  if (!raw) return '';
  if (raw.startsWith('#/')) return raw.slice(1);
  return raw;
}

export function buildMissingSourceRepairHint(node: GraphNode) {
  if (node.kind === 'market') return '先核对契约层，再搜索节点来源';
  if (node.kind === 'route') return '先进入页面确认来源，再回查检索';
  if (node.kind === 'artifact') return '先查看工件，再核对检索结果';
  return '先搜索节点，再回查来源';
}

export function buildMissingSourceRepairLinks(node: GraphNode) {
  const searchLink = buildSearchLink(node.label || '');
  if (node.kind === 'market') {
    return [
      { label: `${node.label} / 查看契约层`, to: '/workspace/contracts?page_section=contracts-source-heads' },
      { label: `${node.label} / 搜索节点`, to: searchLink },
    ];
  }
  if (node.kind === 'route' && node.destination) {
    return [
      { label: `${node.label} / 进入页面`, to: node.destination },
      { label: `${node.label} / 搜索节点`, to: searchLink },
    ];
  }
  if (node.kind === 'artifact' && node.destination) {
    return [
      { label: `${node.label} / 查看工件`, to: node.destination },
      { label: `${node.label} / 搜索节点`, to: searchLink },
    ];
  }
  return [
    { label: `${node.label} / 搜索节点`, to: searchLink },
  ];
}

export function GraphHomePage({
  model,
  centerId: controlledCenterId,
  viewMode: controlledViewMode,
  kindFilter: controlledKindFilter,
  query: controlledQuery,
  showFocusLayer: controlledShowFocusLayer,
  showWarningLayer: controlledShowWarningLayer,
  showDimLayer: controlledShowDimLayer,
  onCenterChange,
  onViewModeChange,
  onKindFilterChange,
  onQueryChange,
  onFocusLayerChange,
  onWarningLayerChange,
  onDimLayerChange,
  onResetLegendLayers,
}: {
  model: TerminalReadModel | null;
  centerId?: string;
  viewMode?: 'all' | 'focus';
  kindFilter?: 'all' | GraphNode['kind'];
  query?: string;
  showFocusLayer?: boolean;
  showWarningLayer?: boolean;
  showDimLayer?: boolean;
  onCenterChange?: (next: string) => void;
  onViewModeChange?: (next: 'all' | 'focus') => void;
  onKindFilterChange?: (next: 'all' | GraphNode['kind']) => void;
  onQueryChange?: (next: string) => void;
  onFocusLayerChange?: (next: boolean) => void;
  onWarningLayerChange?: (next: boolean) => void;
  onDimLayerChange?: (next: boolean) => void;
  onResetLegendLayers?: () => void;
}) {
  const { pipelines, defaultPipelineId, upsertPipeline, setDefaultPipeline, removePipeline } = useGraphPipelineStore();
  const [draftName, setDraftName] = useState('我的交易管道');
  const [localKindFilter, setLocalKindFilter] = useState<'all' | GraphNode['kind']>('all');
  const [localQuery, setLocalQuery] = useState('');
  const [localViewMode, setLocalViewMode] = useState<'all' | 'focus'>('all');
  const [legendLayers, setLegendLayers] = useState({
    focus: true,
    warning: true,
    dim: true,
  });
  const graph = useMemo(() => (model ? buildGraphHomeModel(model, pipelines) : null), [model, pipelines]);
  const [localCenterId, setLocalCenterId] = useState('trade-hub');
  const centerId = controlledCenterId || localCenterId;
  const viewMode = controlledViewMode || localViewMode;
  const kindFilter = controlledKindFilter || localKindFilter;
  const query = controlledQuery ?? localQuery;
  const showFocusLayer = controlledShowFocusLayer ?? legendLayers.focus;
  const showWarningLayer = controlledShowWarningLayer ?? legendLayers.warning;
  const showDimLayer = controlledShowDimLayer ?? legendLayers.dim;
  const updateCenterId = (next: string) => {
    if (onCenterChange) onCenterChange(next);
    else setLocalCenterId(next);
  };
  const updateViewMode = (next: 'all' | 'focus') => {
    if (onViewModeChange) onViewModeChange(next);
    else setLocalViewMode(next);
  };
  const updateKindFilter = (next: 'all' | GraphNode['kind']) => {
    if (onKindFilterChange) onKindFilterChange(next);
    else setLocalKindFilter(next);
  };
  const updateQuery = (next: string) => {
    if (onQueryChange) onQueryChange(next);
    else setLocalQuery(next);
  };
  const updateFocusLayer = (next: boolean) => {
    if (onFocusLayerChange) onFocusLayerChange(next);
    else setLegendLayers((current) => ({ ...current, focus: next }));
  };
  const updateWarningLayer = (next: boolean) => {
    if (onWarningLayerChange) onWarningLayerChange(next);
    else setLegendLayers((current) => ({ ...current, warning: next }));
  };
  const updateDimLayer = (next: boolean) => {
    if (onDimLayerChange) onDimLayerChange(next);
    else setLegendLayers((current) => ({ ...current, dim: next }));
  };
  const resetLegendLayers = () => {
    if (onResetLegendLayers) onResetLegendLayers();
    else setLegendLayers({ focus: true, warning: true, dim: true });
  };
  const nodes = graph?.nodes || [];
  const filteredNodes = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return nodes.filter((node) => {
      const matchesKind = kindFilter === 'all' || node.kind === kindFilter;
      const matchesQuery = !normalized
        || node.label.toLowerCase().includes(normalized)
        || (node.subtitle || '').toLowerCase().includes(normalized);
      return matchesKind && matchesQuery;
    });
  }, [kindFilter, nodes, query]);
  const focusState = useMemo(() => (graph ? buildGraphFocusState(graph, centerId) : null), [graph, centerId]);
  const visibleNodeIds = useMemo(() => {
    if (viewMode === 'focus' && focusState) return new Set(focusState.activeNodeIds);
    return new Set([centerId, ...filteredNodes.map((node) => node.id)]);
  }, [centerId, filteredNodes, focusState, viewMode]);
  const visibleNodes = useMemo(() => nodes.filter((node) => visibleNodeIds.has(node.id)), [nodes, visibleNodeIds]);
  const edges = (graph?.edges || []).filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));
  const layoutNodes = graph ? buildGraphLayout({ ...graph, nodes: visibleNodes, edges }, centerId) : [];
  const selectedNode = nodes.find((node) => node.id === centerId) || nodes[0];
  const defaultPipeline = pipelines.find((pipeline) => pipeline.id === defaultPipelineId) || null;
  const defaultPipelineName = defaultPipeline?.name || '未创建默认管道';
  const upstreamNodes = selectedNode ? selectedNode.upstream.map((id) => nodes.find((node) => node.id === id)).filter(Boolean) as GraphNode[] : [];
  const downstreamNodes = selectedNode ? selectedNode.downstream.map((id) => nodes.find((node) => node.id === id)).filter(Boolean) as GraphNode[] : [];
  const nodeExplanation = explainNode(selectedNode, defaultPipelineName);
  const sourceOwnedRows = buildSourceOwnedRows(selectedNode);
  const impactExplanation = buildImpactExplanation(selectedNode, upstreamNodes, downstreamNodes);
  const recommendation = recommendNextStep(selectedNode, nodes);
  const missingSourceNodes = useMemo(() => {
    if (!focusState) return [];
    return focusState.activeNodeIds
      .map((id) => nodes.find((node) => node.id === id))
      .filter((node): node is GraphNode => Boolean(node))
      .filter((node) => !node.detail?.sourceContract && !['strategy', 'pipeline_stage', 'custom_pipeline'].includes(node.kind));
  }, [focusState, nodes]);
  const terminalLink = '/terminal/public';
  const workspaceLink = '/workspace/artifacts';
  const cpaLink = '/cpa';
  const searchLink = buildSearchLink();
  const addToPipeline = () => {
    if (!selectedNode) return;
    const pipeline = defaultPipeline || createPipeline(draftName);
    const nodeIds = pipeline.nodeIds.includes(selectedNode.id)
      ? pipeline.nodeIds
      : [...pipeline.nodeIds, selectedNode.id];
    upsertPipeline({
      ...pipeline,
      name: pipeline.name || draftName,
      nodeIds,
      updatedAt: new Date().toISOString(),
      isDefault: true,
    });
    setDefaultPipeline(pipeline.id);
  };

  const reorder = (from: number, to: number) => {
    if (!defaultPipeline) return;
    const next = [...defaultPipeline.nodeIds];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    upsertPipeline({
      ...defaultPipeline,
      nodeIds: next,
      updatedAt: new Date().toISOString(),
    });
  };

  if (!model || !graph) {
    return (
      <div className="workspace-grid single-column">
        <PanelCard title="图谱化主页" kicker="graph-home / loading" meta="等待 dashboard snapshot">
          <div className="empty-block">图谱主页加载中…</div>
        </PanelCard>
      </div>
    );
  }

  const researchAuditCases = model.workspace.researchAuditCases || [];

  return (
    <div className="workspace-grid two-column graph-home-layout">
      <PanelCard title="图谱化主页" kicker="2.5d graph / trade hub + custom pipeline" meta="点击节点后重新居中，右侧查看影响链与深跳入口。">
        <div className="graph-home-toolbar">
          <div className="chip-row">
            <span className="summary-chip">中心：{nodeLabel(selectedNode)}</span>
            <span className="summary-chip">默认管道：{defaultPipelineName}</span>
            <span className="summary-chip">推荐下一跳：{recommendation}</span>
          </div>
          <div className="button-row graph-home-quicklinks">
            <ActionButton onClick={() => updateCenterId('trade-hub')} disabled={centerId === 'trade-hub'}>
              回到交易中枢
            </ActionButton>
            <ActionButton onClick={() => updateViewMode(viewMode === 'focus' ? 'all' : 'focus')}>
              {viewMode === 'focus' ? '返回全图' : '锁定当前链'}
            </ActionButton>
            <ActionLink to={terminalLink}>去操作终端</ActionLink>
            <ActionLink to={workspaceLink}>去研究工作区</ActionLink>
            <ActionLink className="graph-home-cpa-link" to={cpaLink}>去 CPA 管理</ActionLink>
            <ActionLink to={searchLink}>打开全局搜索</ActionLink>
          </div>
          <div className="chip-row">
            <span className="summary-chip">节点数：{visibleNodes.length}/{nodes.length}</span>
            <span className="summary-chip">边数：{edges.length}</span>
            {focusState ? <span className="summary-chip">当前影响链：{focusState.activeNodeIds.length} 节点 / {focusState.activeEdgeIds.length} 边</span> : null}
            {missingSourceNodes.length ? <span className="summary-chip">缺失信源：{missingSourceNodes.length}</span> : null}
          </div>
          <div className="chip-row" data-testid="graph-home-legend">
            <FilterChip active={showFocusLayer} onClick={() => updateFocusLayer(!showFocusLayer)}>
              当前链高亮
            </FilterChip>
            <FilterChip active={showWarningLayer} onClick={() => updateWarningLayer(!showWarningLayer)}>
              缺失信源
            </FilterChip>
            <FilterChip active={showDimLayer} onClick={() => updateDimLayer(!showDimLayer)}>
              非当前链降噪
            </FilterChip>
            <ActionButton onClick={resetLegendLayers}>恢复默认图层</ActionButton>
          </div>
          <div className="chip-row">
            {FILTER_OPTIONS.map((option) => (
              <FilterChip key={option.value} active={kindFilter === option.value} onClick={() => updateKindFilter(option.value)}>
                {option.label}
              </FilterChip>
            ))}
          </div>
          <div>
            <input
              className="search-input"
              value={query}
              onChange={(event) => updateQuery(event.target.value)}
              placeholder="过滤节点：标签 / 副标题"
            />
          </div>
        </div>
        <GraphHomeRenderer
          nodes={layoutNodes}
          edges={edges}
          centerId={centerId}
          activeNodeIds={focusState?.activeNodeIds}
          activeEdgeIds={focusState?.activeEdgeIds}
          showFocusLayer={showFocusLayer}
          showWarningLayer={showWarningLayer}
          showDimLayer={showDimLayer}
          onSelect={updateCenterId}
        />
      </PanelCard>

      <div className="graph-home-side">
        <PanelCard title={nodeLabel(selectedNode)} kicker="当前节点详情" meta={selectedNode?.subtitle || '图谱详情'}>
          <KeyValueGrid
            rows={[
              { key: 'kind', label: '节点类型', value: kindLabel(selectedNode?.kind) },
              { key: 'status', label: '状态', value: selectedNode?.status || '—', kind: 'badge' as const },
              { key: 'importance', label: '重要度', value: selectedNode?.importance || '—', kind: 'number' as const },
              { key: 'upstream', label: '上游节点数', value: selectedNode?.upstream.length || 0, kind: 'number' as const },
              { key: 'downstream', label: '下游节点数', value: selectedNode?.downstream.length || 0, kind: 'number' as const },
              { key: 'destination', label: '相关页面', value: selectedNode?.destination || '—' },
            ]}
          />
          {sourceOwnedRows.length ? (
            <div className="graph-home-source-details">
              <div className="graph-home-explanation">
                <span className="graph-home-explanation-kicker">source-owned 上下文</span>
                <strong>当前节点携带源头契约字段</strong>
              </div>
              <KeyValueGrid rows={sourceOwnedRows} />
            </div>
          ) : null}
          <div className="graph-home-explanation">
            <span className="graph-home-explanation-kicker">{nodeExplanation.title}</span>
            <strong>{nodeExplanation.summary}</strong>
            <p className="graph-home-explanation-copy">{nodeExplanation.detail}</p>
          </div>
          <div className="graph-home-explanation">
            <span className="graph-home-explanation-kicker">为什么影响当前中心</span>
            <strong>{impactExplanation.summary}</strong>
            <p className="graph-home-explanation-copy">{impactExplanation.detail}</p>
          </div>
          {missingSourceNodes.length ? (
            <div className="graph-home-explanation" data-testid="graph-home-missing-source">
              <span className="graph-home-explanation-kicker">当前链缺失信源</span>
              <strong>缺失信源节点：{missingSourceNodes.length}</strong>
              <div className="stack-list">
                {missingSourceNodes.map((node) => (
                  <article className="stack-button" key={node.id}>
                    <strong>{`${node.label} / ${kindLabel(node.kind)}`}</strong>
                    <small>{buildMissingSourceRepairHint(node)}</small>
                    <div className="button-row workspace-handoff-links">
                      {buildMissingSourceRepairLinks(node).map((link) => (
                        <ActionLink key={`${node.id}-${link.label}`} to={link.to}>{link.label}</ActionLink>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            </div>
          ) : null}
          {upstreamNodes.length ? (
            <div className="empty-block">上游：{upstreamNodes.map((node) => node.label).join(' / ')}</div>
          ) : null}
          {downstreamNodes.length ? (
            <div className="empty-block">下游：{downstreamNodes.map((node) => node.label).join(' / ')}</div>
          ) : null}
          {shouldShowResearchAuditLinks(selectedNode) && researchAuditCases.length ? (
            <div className="graph-home-audit-links">
              <div className="empty-block">研究审计直达：{researchAuditCases.length} 条</div>
              {researchAuditCases.map((row, index) => (
                <div key={`${row.case_id || row.query || index}`} className="graph-home-audit-case">
                  <strong>{row.query || row.case_id || `case-${index + 1}`}</strong>
                  <div className="button-row workspace-handoff-links">
                    {row.search_route ? <ActionLink to={normalizeHashRoute(row.search_route)}>检索 / {row.query || row.case_id || 'case'}</ActionLink> : null}
                    {row.workspace_route ? <ActionLink to={normalizeHashRoute(row.workspace_route)}>工件 / {row.result_artifact || row.case_id || 'artifact'}</ActionLink> : null}
                    {row.raw_path ? <ActionLink to={`/workspace/raw?artifact=${encodeURIComponent(row.raw_path)}`}>原始层 / {row.raw_path}</ActionLink> : null}
                  </div>
                </div>
              ))}
            </div>
          ) : null}
          <div className="button-row workspace-handoff-links">
            <ActionButton onClick={addToPipeline}>加入自定义管道</ActionButton>
            {selectedNode?.destination ? <ActionLink to={selectedNode.destination}>进入子网页</ActionLink> : null}
          </div>
        </PanelCard>

        <PanelCard title="自定义管道" kicker="local-storage / draggable" meta={defaultPipeline ? defaultPipeline.name : '尚未创建'}>
          {!defaultPipeline ? (
            <div className="graph-pipeline-create">
              <input
                className="search-input"
                value={draftName}
                onChange={(event) => setDraftName(event.target.value)}
                placeholder="输入管道名称"
              />
              <ActionButton onClick={() => {
                const pipeline = createPipeline(draftName);
                upsertPipeline(pipeline);
                setDefaultPipeline(pipeline.id);
              }}
              >
                创建默认管道
              </ActionButton>
            </div>
          ) : (
            <div className="graph-pipeline-stack">
              <div className="chip-row">
                <span className="summary-chip">交易中枢</span>
                <span className="summary-chip">自定义管道</span>
              </div>
              <ol className="graph-pipeline-list">
                {defaultPipeline.nodeIds.map((nodeId, index) => {
                  const node = nodes.find((item) => item.id === nodeId);
                  return (
                    <li
                      key={nodeId}
                      className="graph-pipeline-item"
                      draggable
                      onDragStart={(event) => event.dataTransfer.setData('text/plain', String(index))}
                      onDragOver={(event) => event.preventDefault()}
                      onDrop={(event) => {
                        const from = Number(event.dataTransfer.getData('text/plain'));
                        reorder(from, index);
                      }}
                    >
                      <div className="graph-pipeline-item-copy" data-testid={`graph-pipeline-item-${nodeId}`}>
                        <strong>{nodeLabel(node)}</strong>
                        <div className="chip-row">
                          {node?.detail?.sourceContract ? (
                            <span className="summary-chip">{node.detail.sourceContract}</span>
                          ) : null}
                          {node?.status ? (
                            <Badge value={node.status}>{labelFor(String(node.status))}</Badge>
                          ) : null}
                        </div>
                      </div>
                      <div className="button-row">
                        <ActionButton onClick={() => updateCenterId(nodeId)}>定位</ActionButton>
                      </div>
                    </li>
                  );
                })}
              </ol>
              <div className="button-row workspace-handoff-links">
                <ActionButton onClick={() => setDefaultPipeline(defaultPipeline.id)}>设为默认视图</ActionButton>
                <ActionButton onClick={() => removePipeline(defaultPipeline.id)}>删除管道</ActionButton>
              </div>
            </div>
          )}
        </PanelCard>
      </div>
    </div>
  );
}
