import { Link } from 'react-router-dom';
import { MetricStrip, PanelCard } from '../components/ui-kit';
import type { TerminalReadModel } from '../types/contracts';
import type { ConversationFeedbackProjection, MetricItem } from '../types/contracts';
import type { SourceHeadEntry } from '../types/contracts';
import { buildWorkspacePageLink } from '../utils/focus-links';
import { labelFor } from '../utils/dictionary';
import { compactPath, safeDisplayValue, statusTone } from '../utils/formatters';

type OverviewPageProps = {
  model: TerminalReadModel | null;
};

function hasNonEmptyValue(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  return value !== null && value !== undefined && value !== '';
}

function buildHoldSelectionMetrics(row: SourceHeadEntry & { id: string }): MetricItem[] {
  const useReturnCandidate = hasNonEmptyValue(row.return_candidate);
  const returnCandidate = safeDisplayValue(row.return_candidate);
  const returnWatch = safeDisplayValue(row.return_watch);
  const returnFocus = useReturnCandidate ? returnCandidate : returnWatch;
  const returnLabel = useReturnCandidate
    ? labelFor('overview_return_candidate_metric')
    : labelFor('overview_return_watch_metric');
  return [
    {
      id: 'overview-active-baseline',
      label: labelFor('overview_active_baseline_metric'),
      value: row.active_baseline,
      tone: statusTone(row.status),
    },
    {
      id: 'overview-local-candidate',
      label: labelFor('overview_local_candidate_metric'),
      value: row.local_candidate,
      tone: statusTone(row.local_candidate),
    },
    {
      id: 'overview-transfer-watch',
      label: labelFor('overview_transfer_watch_metric'),
      value: row.transfer_watch,
      tone: statusTone(row.transfer_watch),
    },
    {
      id: 'overview-return-focus',
      label: returnLabel,
      value: returnFocus,
      tone: statusTone(returnFocus),
    },
  ];
}

function buildFeedbackMetrics(projection: ConversationFeedbackProjection): MetricItem[] {
  const summary = projection.summary || {};
  return [
    {
      id: 'feedback-alignment',
      label: 'alignment_score',
      value: summary.alignment_score ?? '—',
      tone: statusTone((summary.alignment_score ?? 0) >= 70 ? 'positive' : (summary.alignment_score ?? 0) < 45 ? 'negative' : 'warning'),
    },
    {
      id: 'feedback-blocker',
      label: 'blocker_pressure',
      value: summary.blocker_pressure ?? '—',
      tone: statusTone((summary.blocker_pressure ?? 0) >= 60 ? 'negative' : (summary.blocker_pressure ?? 0) <= 35 ? 'positive' : 'warning'),
    },
    {
      id: 'feedback-execution',
      label: 'execution_clarity',
      value: summary.execution_clarity ?? '—',
      tone: statusTone((summary.execution_clarity ?? 0) >= 65 ? 'positive' : (summary.execution_clarity ?? 0) < 45 ? 'negative' : 'warning'),
    },
    {
      id: 'feedback-readability',
      label: 'readability_pressure',
      value: summary.readability_pressure ?? '—',
      tone: statusTone((summary.readability_pressure ?? 0) >= 60 ? 'negative' : (summary.readability_pressure ?? 0) <= 35 ? 'positive' : 'warning'),
    },
  ];
}

function buildCommodityReasoningMetrics(model: TerminalReadModel | null): MetricItem[] {
  if (!model) return [];
  const micro = model.dataRegime.microCapture.filter((metric) =>
    ['commodity-scenario', 'commodity-chain', 'commodity-range'].includes(metric.id),
  );
  const repair = model.signalRisk.repairPlan.filter((metric) =>
    ['commodity-summary', 'commodity-boundary'].includes(metric.id),
  );
  return [...micro, ...repair];
}

export function OverviewPage({ model }: OverviewPageProps) {
  const holdSelectionHandoff = model?.workspace.sourceHeads.find((row) => row.id === 'hold_selection_handoff')
    || model?.workspace.sourceHeads[0]
    || null;
  const sourceHeadLink = holdSelectionHandoff
    ? buildWorkspacePageLink(
      'contracts',
      { artifact: safeDisplayValue(holdSelectionHandoff.id) },
      'contracts-source-heads',
    )
    : '/workspace/contracts';
  const requestedInternal = model?.surface.requested === 'internal';
  const feedbackProjection = model?.feedback.projection || null;
  const feedbackLink = buildWorkspacePageLink(
    'alignment',
    { view: 'internal' },
    'alignment-summary',
  );
  const commodityReasoningMetrics = buildCommodityReasoningMetrics(model);

  return (
    <section className="overview-page">
      <div className="overview-hero">
        <h2>总览</h2>
        <p>先看系统态势，再分流进入操作终端或研究工作区。</p>
      </div>
      <div className="overview-entry-grid">
        <Link className="panel-card overview-entry-card" to="/terminal/public">
          <p className="panel-kicker">一级域</p>
          <strong>操作终端</strong>
          <span>{model?.orchestration.topMetrics[3]?.value ? `当前值：${String(model.orchestration.topMetrics[3].value)}` : '进入调度 / 门禁 / 风险链路'}</span>
        </Link>
        <Link className="panel-card overview-entry-card" to="/workspace/artifacts">
          <p className="panel-kicker">一级域</p>
          <strong>研究工作区</strong>
          <span>{model?.workspace.artifactRows.length ? `工件数：${model.workspace.artifactRows.length}` : '进入工件 / 回测 / 比较'}</span>
        </Link>
        <Link className="panel-card overview-entry-card" to="/workspace/contracts">
          <p className="panel-kicker">入口</p>
          <strong>契约验收</strong>
          <span>查看公开面验收与入口拓扑</span>
        </Link>
      </div>
      <PanelCard
        title="研究主线摘要"
        kicker="source-owned / canonical"
        meta={holdSelectionHandoff
          ? `${safeDisplayValue(holdSelectionHandoff.label || holdSelectionHandoff.id)} ｜ ${compactPath(holdSelectionHandoff.path, 4)}`
          : '当前快照未暴露 hold_selection_handoff 主头'}
        actions={<Link className="button" to={sourceHeadLink}>查看源头主线</Link>}
      >
        {holdSelectionHandoff ? (
          <MetricStrip items={buildHoldSelectionMetrics(holdSelectionHandoff)} />
        ) : (
          <div className="empty-block">未找到研究主线主头，请先刷新 source heads 快照。</div>
        )}
      </PanelCard>
      {commodityReasoningMetrics.length ? (
        <PanelCard
          title="国内商品推理线"
          kicker="reasoning / visible-priority"
          meta="首屏固定摘要：主情景 / 主传导链 / 范围 / 边界 / 失效条件"
        >
          <MetricStrip items={commodityReasoningMetrics} showRawValues />
        </PanelCard>
      ) : null}
      {requestedInternal ? (
        <PanelCard
          title="方向对齐投射"
          kicker="internal_only / feedback_projection"
          meta={model.surface.effective === 'internal' ? '内部高价值反馈摘要' : '仅内部可见'}
          actions={<Link className="button" to={feedbackLink}>进入对齐页</Link>}
        >
          {model.surface.effective === 'internal' && feedbackProjection ? (
            <>
              <MetricStrip items={buildFeedbackMetrics(feedbackProjection)} />
              <div className="empty-block">{safeDisplayValue(feedbackProjection.summary.headline || '暂无高价值反馈')}</div>
            </>
          ) : (
            <div className="empty-block">内部对齐投射暂不可用</div>
          )}
        </PanelCard>
      ) : null}
    </section>
  );
}
