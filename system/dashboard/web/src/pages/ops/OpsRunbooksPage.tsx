import { Link } from 'react-router-dom';
import { DenseTable, MetricStrip, PanelCard } from '../../components/ui-kit';
import { buildOpsPageLink } from '../../navigation/route-contract';
import { buildWorkspaceLink, type SharedFocusState } from '../../utils/focus-links';
import { safeDisplayValue } from '../../utils/formatters';
import type { TerminalReadModel } from '../../types/contracts';

type OpsRunbooksPageProps = {
  model: TerminalReadModel;
  focus: SharedFocusState;
};

export function OpsRunbooksPage({ model, focus }: OpsRunbooksPageProps) {
  const view = model.view.terminal;
  const actionLog = model.orchestration.actionLog as unknown as Array<Record<string, unknown>>;
  const checklist = model.orchestration.checklist as unknown as Array<Record<string, unknown>>;
  const backlog = model.signalRisk.optimizationBacklog;
  const activeArtifact = safeDisplayValue(focus.artifact);

  return (
    <section className="ops-runbooks-page" aria-label="ops-runbooks-page">
      <div className="workspace-grid single-column">
        <PanelCard
          title="值班动作"
          kicker="runbook / recent actions"
          meta="先看最近调度动作与执行前信号，避免在当前页直接下发命令。"
          actions={<Link className="button" to={buildOpsPageLink('risk', focus)}>回风险驾驶舱</Link>}
        >
          <DenseTable
            columns={view.orchestration.actionLogColumns}
            rows={actionLog}
          />
        </PanelCard>

        <PanelCard
          title="处置清单"
          kicker="precheck / manual checklist"
          meta="把当前人工待办从旧 terminal 长页拆出，集中在动作域消费。"
          actions={<Link className="button" to={buildOpsPageLink('workflow', focus)}>进入工作流</Link>}
        >
          <DenseTable
            columns={view.orchestration.checklistColumns}
            rows={checklist}
          />
        </PanelCard>

        <PanelCard
          title="修复计划"
          kicker="repair / follow-up"
          meta="修复优先级与后续优化在这里汇总，不在总览或风险页重复堆叠。"
          actions={(
            <div className="button-row">
              <Link className="button" to={buildOpsPageLink('traces', focus)}>进入链路追踪</Link>
              {activeArtifact !== '—' ? <Link className="button" to={buildWorkspaceLink('artifacts', { artifact: activeArtifact })}>查看焦点工件</Link> : null}
            </div>
          )}
        >
          <MetricStrip items={model.signalRisk.repairPlan} showRawValues={view.signalRisk.repairPlanStrip.showRawValues} />
          <DenseTable
            columns={view.signalRisk.optimizationBacklogColumns}
            rows={backlog}
          />
        </PanelCard>
      </div>
    </section>
  );
}
