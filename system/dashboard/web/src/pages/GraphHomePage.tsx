import { useMemo, useState } from 'react';
import { ActionButton, ActionLink, FilterChip } from '../components/control-primitives';
import { KeyValueGrid, PanelCard } from '../components/ui-kit';
import { buildSearchLink } from '../search/links';
import type { TerminalReadModel } from '../types/contracts';
import { buildGraphHomeModel, buildGraphLayout } from '../graph/model';
import { GraphHomeRenderer } from '../graph/GraphHomeRenderer';
import { useGraphPipelineStore } from '../graph/storage';
import type { GraphNode, GraphPipeline } from '../graph/types';

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

export function GraphHomePage({ model }: { model: TerminalReadModel | null }) {
  const { pipelines, defaultPipelineId, upsertPipeline, setDefaultPipeline, removePipeline } = useGraphPipelineStore();
  const [draftName, setDraftName] = useState('我的交易管道');
  const [kindFilter, setKindFilter] = useState<'all' | GraphNode['kind']>('all');
  const [query, setQuery] = useState('');
  const graph = useMemo(() => (model ? buildGraphHomeModel(model, pipelines) : null), [model, pipelines]);
  const [centerId, setCenterId] = useState('trade-hub');
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
  const visibleNodeIds = new Set([centerId, ...filteredNodes.map((node) => node.id)]);
  const edges = (graph?.edges || []).filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));
  const layoutNodes = graph ? buildGraphLayout({ ...graph, nodes: nodes.filter((node) => visibleNodeIds.has(node.id)), edges }, centerId) : [];
  const selectedNode = nodes.find((node) => node.id === centerId) || nodes[0];
  const defaultPipeline = pipelines.find((pipeline) => pipeline.id === defaultPipelineId) || null;
  const defaultPipelineName = defaultPipeline?.name || '未创建默认管道';
  const upstreamNodes = selectedNode ? selectedNode.upstream.map((id) => nodes.find((node) => node.id === id)).filter(Boolean) as GraphNode[] : [];
  const downstreamNodes = selectedNode ? selectedNode.downstream.map((id) => nodes.find((node) => node.id === id)).filter(Boolean) as GraphNode[] : [];
  const nodeExplanation = explainNode(selectedNode, defaultPipelineName);
  const recommendation = recommendNextStep(selectedNode, nodes);
  const terminalLink = '/terminal/public';
  const workspaceLink = '/workspace/artifacts';
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
            <ActionButton onClick={() => setCenterId('trade-hub')} disabled={centerId === 'trade-hub'}>
              回到交易中枢
            </ActionButton>
            <ActionLink to={terminalLink}>去操作终端</ActionLink>
            <ActionLink to={workspaceLink}>去研究工作区</ActionLink>
            <ActionLink to={searchLink}>打开全局搜索</ActionLink>
          </div>
          <div className="chip-row">
            <span className="summary-chip">节点数：{filteredNodes.length}/{nodes.length}</span>
            <span className="summary-chip">边数：{edges.length}</span>
          </div>
          <div className="chip-row">
            {FILTER_OPTIONS.map((option) => (
              <FilterChip key={option.value} active={kindFilter === option.value} onClick={() => setKindFilter(option.value)}>
                {option.label}
              </FilterChip>
            ))}
          </div>
          <div>
            <input
              className="search-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="过滤节点：标签 / 副标题"
            />
          </div>
        </div>
        <GraphHomeRenderer nodes={layoutNodes} edges={edges} centerId={centerId} onSelect={setCenterId} />
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
          <div className="graph-home-explanation">
            <span className="graph-home-explanation-kicker">{nodeExplanation.title}</span>
            <strong>{nodeExplanation.summary}</strong>
            <p className="graph-home-explanation-copy">{nodeExplanation.detail}</p>
          </div>
          {upstreamNodes.length ? (
            <div className="empty-block">上游：{upstreamNodes.map((node) => node.label).join(' / ')}</div>
          ) : null}
          {downstreamNodes.length ? (
            <div className="empty-block">下游：{downstreamNodes.map((node) => node.label).join(' / ')}</div>
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
                      <span>{nodeLabel(node)}</span>
                      <div className="button-row">
                        <ActionButton onClick={() => setCenterId(nodeId)}>定位</ActionButton>
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
