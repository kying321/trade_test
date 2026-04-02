import { Link } from 'react-router-dom';
import { Badge, DrilldownSection, KeyValueGrid, PanelCard, PathText, ValueText } from '../../components/ui-kit';
import { labelFor } from '../../utils/dictionary';
import { safeDisplayValue } from '../../utils/formatters';
import { buildWorkspacePageLink, type SharedFocusState } from '../../utils/focus-links';
import type { TerminalReadModel } from '../../types/contracts';

type OpsAuditsPageProps = {
  model: TerminalReadModel;
  focus: SharedFocusState;
};

type TopologyTier = 'primary' | 'fallback' | 'probe';

const TOPOLOGY_TIER_ORDER: TopologyTier[] = ['primary', 'fallback', 'probe'];

function classifyTopologyTier(role: string | undefined): TopologyTier {
  const raw = safeDisplayValue(role || '').toLowerCase();
  if (raw.includes('探针')) return 'probe';
  if (raw.includes('回退')) return 'fallback';
  return 'primary';
}

function topologyTierLabel(tier: TopologyTier): string {
  if (tier === 'probe') return '探针层';
  if (tier === 'fallback') return '回退入口层';
  return '主入口层';
}

function topologySectionLabel(tier: TopologyTier): string {
  if (tier === 'probe') return '探针区';
  if (tier === 'fallback') return '回退链';
  return '主入口链';
}

function shouldRenderTopologyFlow(tier: TopologyTier): boolean {
  return tier === 'primary' || tier === 'fallback';
}

export function OpsAuditsPage({ model }: OpsAuditsPageProps) {
  const publicTopology = model.workspace.publicTopology || [];
  const acceptance = model.workspace.publicAcceptance;
  const summary = acceptance.summary;

  return (
    <section className="ops-audits-page" aria-label="ops-audits-page">
      <div className="workspace-grid single-column">
        <PanelCard
          title="公开入口拓扑"
          kicker="audit / topology"
          meta="只看根域、Pages、旧 OpenClaw 与 gateway 的对外角色和退化关系。"
        >
          {publicTopology.length ? (
            <div className="stack-list">
              {TOPOLOGY_TIER_ORDER.map((tier) => {
                const rows = publicTopology.filter((row) => classifyTopologyTier(row.role) === tier);
                if (!rows.length) return null;
                return (
                  <section className={`topology-band topology-band-${tier}`} key={tier}>
                    <div className="topology-band-head">
                      <strong>{topologySectionLabel(tier)}</strong>
                      <span>{`${rows.length} 项`}</span>
                    </div>
                    {shouldRenderTopologyFlow(tier) ? (
                      <div className="topology-band-flow">
                        {rows.map((row) => (
                          <div className="topology-flow-step" key={`${row.id}-flow`}>
                            <span className="topology-flow-node"><ValueText value={row.label || row.id} showRaw={false} /></span>
                            <span className="topology-flow-arrow" aria-hidden="true">→</span>
                            <span className="topology-flow-node"><ValueText value={row.fallback || '无'} showRaw={false} /></span>
                          </div>
                        ))}
                      </div>
                    ) : null}
                    <div className="topology-node-grid">
                      {rows.map((row, index) => (
                        <article className={`topology-node-card topology-node-card-${tier}`} key={row.id}>
                          <div className="topology-node-head">
                            <div className="topology-node-headcopy">
                              <span className="topology-node-index">{`节点 ${index + 1}`}</span>
                              <span className={`topology-node-tier topology-node-tier-${tier}`}>{topologyTierLabel(tier)}</span>
                            </div>
                            <Badge value={row.role}>{labelFor(safeDisplayValue(row.role || '—'))}</Badge>
                          </div>
                          <strong className="topology-node-title"><ValueText value={row.label || row.id} showRaw={false} /></strong>
                          <div className="topology-node-url"><PathText value={row.url || '—'} showRaw /></div>
                          <div className="topology-node-note">{safeDisplayValue(row.notes || row.source || row.fallback)}</div>
                          <KeyValueGrid
                            rows={[
                              { key: 'status', label: '预期状态', value: row.expected_status || '—', showRaw: true },
                              { key: 'source', label: '控制源', value: row.source || '—', showRaw: true },
                            ]}
                          />
                          <div className="topology-node-chain">
                            <span className="topology-node-chain-label">退化至</span>{' '}
                            <strong><ValueText value={row.fallback || '无'} showRaw={false} /></strong>
                          </div>
                        </article>
                      ))}
                    </div>
                  </section>
                );
              })}
            </div>
          ) : (
            <div className="empty-block">未声明公开入口拓扑</div>
          )}
        </PanelCard>

        <PanelCard
          title="公开面验收"
          kicker="audit / acceptance"
          meta="把 topology、workspace routes 与 graph home 等公开验收链路从研究工作区中拆出。"
        >
          {summary ? (
            <>
              <div className="contracts-acceptance-status-strip" data-testid="ops-audits-status-strip">
                <div className="metric-strip">
                  {acceptance.checks.map((row) => (
                    <article
                      key={row.id}
                      className={`metric-tile contracts-acceptance-status-card tone-${safeDisplayValue(row.status || 'neutral')}`}
                    >
                      <span className="metric-label">{row.label}</span>
                      <strong className="metric-value">{labelFor(safeDisplayValue(row.status || '—'))}</strong>
                      <span className="metric-hint">{safeDisplayValue(row.generated_at_utc || row.headline || row.report_path)}</span>
                      <div className="button-row contracts-acceptance-status-actions">
                        <Link className="button" to={row.inspector_route || buildWorkspacePageLink('contracts', {}, `contracts-check-${safeDisplayValue(row.id)}`)}>
                          查看检查
                        </Link>
                      </div>
                    </article>
                  ))}
                </div>
              </div>

              <DrilldownSection
                title="穿透层 1 / 验收总览"
                summary={`${safeDisplayValue(summary.status)} ｜ ${safeDisplayValue(summary.generated_at_utc)}`}
                meta={<Badge value={summary.status} />}
                defaultOpen
              >
                <KeyValueGrid
                  rows={[
                    { key: 'status', label: '状态', value: summary.status || '—', showRaw: true },
                    { key: 'change_class', label: '变更级别', value: summary.change_class || '—', showRaw: true },
                    { key: 'generated_at_utc', label: '生成时间', value: summary.generated_at_utc || '—', showRaw: true },
                    { key: 'report_path', label: '报告路径', value: summary.report_path || '—', kind: 'path', showRaw: true },
                    { key: 'topology_status', label: '拓扑状态', value: summary.topology_status || '—', showRaw: true },
                    { key: 'workspace_status', label: '工作区状态', value: summary.workspace_status || '—', showRaw: true },
                    { key: 'graph_home_status', label: '图谱主页状态', value: summary.graph_home_status || '—', showRaw: true },
                    { key: 'frontend_public', label: '公开入口', value: summary.frontend_public || '—', kind: 'path', showRaw: true },
                  ]}
                />
              </DrilldownSection>

              <DrilldownSection
                title="穿透层 2 / 子链状态"
                summary={`${acceptance.checks.length} checks / ${acceptance.subcommands.length} subcommands`}
                defaultOpen
              >
                <div className="drill-list">
                  {acceptance.checks.map((row) => (
                    <article className="panel-card" key={row.id}>
                      <div className="panel-card-head">
                        <div className="panel-card-copy">
                          <p className="panel-kicker">acceptance check</p>
                          <h2 className="panel-card-title">{row.label}</h2>
                          <p className="panel-note panel-card-meta">{safeDisplayValue(row.headline || row.report_path || row.failure_reason)}</p>
                        </div>
                        <div className="panel-actions">
                          <Badge value={row.status} />
                        </div>
                      </div>
                      <KeyValueGrid
                        rows={[
                          { key: 'status', label: '状态', value: row.status || '—', showRaw: true },
                          { key: 'generated_at_utc', label: '生成时间', value: row.generated_at_utc || '—', showRaw: true },
                          { key: 'report_path', label: '报告路径', value: row.report_path || '—', kind: 'path', showRaw: true },
                          { key: 'failure_reason', label: '失败原因', value: row.failure_reason || '—', showRaw: true },
                        ]}
                      />
                    </article>
                  ))}
                </div>
              </DrilldownSection>
            </>
          ) : (
            <div className="empty-block">公开面验收暂不可用</div>
          )}
        </PanelCard>
      </div>
    </section>
  );
}
