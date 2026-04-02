import { Link } from 'react-router-dom';
import { KeyValueGrid, MetricStrip, PanelCard } from '../../components/ui-kit';
import { buildOpsPageLink } from '../../navigation/route-contract';
import { buildWorkspaceLink, type SharedFocusState } from '../../utils/focus-links';
import { safeDisplayValue, compactPath } from '../../utils/formatters';
import type { TerminalReadModel } from '../../types/contracts';

type OpsWorkflowPageProps = {
  model: TerminalReadModel;
  focus: SharedFocusState;
};

export function OpsWorkflowPage({ model, focus }: OpsWorkflowPageProps) {
  const holdHead = model.workspace.sourceHeads.find((row) => safeDisplayValue(row.id) === 'hold_selection_handoff')
    || model.workspace.sourceHeads[0]
    || null;
  const proposalArtifact = safeDisplayValue(holdHead?.id);
  const proposalPath = safeDisplayValue(holdHead?.path);
  const gateMetrics = model.signalRisk.gateScores;
  const rollbackChain = model.workspace.fallbackChain || [];

  return (
    <section className="ops-workflow-page" aria-label="ops-workflow-page">
      <div className="workspace-grid single-column">
        <PanelCard
          title="提案上下文"
          kicker="workflow / proposal context"
          meta="先把当前候选、基线和 source-owned 推荐摘要固定在制度页。"
          actions={proposalArtifact !== '—' ? <Link className="button" to={buildWorkspaceLink('contracts', { artifact: proposalArtifact })}>查看契约层</Link> : null}
        >
          <KeyValueGrid
            rows={[
              { key: 'change_class', label: '变更级别', value: model.meta.change_class || '—', showRaw: true },
              { key: 'artifact', label: '当前提案工件', value: proposalArtifact, showRaw: true },
              { key: 'active_baseline', label: '当前基线', value: holdHead?.active_baseline || '—', showRaw: true },
              { key: 'local_candidate', label: '本地候选', value: holdHead?.local_candidate || '—', showRaw: true },
              { key: 'return_watch', label: '收益观察', value: safeDisplayValue(Reflect.get(holdHead || {}, 'return_watch')) || '—', showRaw: true },
              { key: 'recommended_brief', label: '推荐摘要', value: holdHead?.recommended_brief || holdHead?.summary || '—', showRaw: true },
              { key: 'path', label: '源路径', value: proposalPath, kind: 'path', showRaw: true },
            ]}
          />
        </PanelCard>

        <PanelCard
          title="审批门禁"
          kicker="workflow / approval gates"
          meta="当前只消费已有 gate metrics，不在前端重造审批状态机。"
          actions={<Link className="button" to={buildOpsPageLink('risk', focus)}>回风险驾驶舱</Link>}
        >
          <MetricStrip items={gateMetrics} showRawValues />
        </PanelCard>

        <PanelCard
          title="回放与回滚"
          kicker="workflow / replay + rollback"
          meta="回放与回滚先以现有 fallback/source 路径表达，后续再接更细 workflow store。"
          actions={(
            <div className="button-row">
              <Link className="button" to={buildOpsPageLink('traces', focus)}>进入链路追踪</Link>
              {proposalArtifact !== '—' ? <Link className="button" to={buildWorkspaceLink('raw', { artifact: proposalArtifact })}>查看原始层</Link> : null}
            </div>
          )}
        >
          <div className="stack-list">
            {rollbackChain.length ? rollbackChain.map((item, index) => (
              <div className="empty-block" key={`${item}-${index}`}>
                <strong>#{index + 1}</strong> {compactPath(item)}
              </div>
            )) : <div className="empty-block">未声明回退链</div>}
          </div>
        </PanelCard>
      </div>
    </section>
  );
}
