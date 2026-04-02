import { Link } from 'react-router-dom';
import { Badge, DenseTable, MetricStrip, PanelCard } from '../../components/ui-kit';
import { labelFor } from '../../utils/dictionary';
import { safeDisplayValue } from '../../utils/formatters';
import { buildWorkspaceLink, type SharedFocusState } from '../../utils/focus-links';
import { buildOpsPageLink } from '../../navigation/route-contract';
import type { TerminalReadModel } from '../../types/contracts';

type OpsRiskPageProps = {
  model: TerminalReadModel;
  focus: SharedFocusState;
};

function previewRows(rows: Array<Record<string, unknown>>, limit = 4) {
  return rows.slice(0, limit);
}

export function OpsRiskPage({ model, focus }: OpsRiskPageProps) {
  const view = model.view.terminal;
  const alertCount = model.orchestration.alerts.length;
  const focusSlots = model.signalRisk.focusSlots as unknown as Array<Record<string, unknown>>;
  const controlChain = model.signalRisk.controlChain as unknown as Array<Record<string, unknown>>;
  const actionQueue = model.signalRisk.actionQueue as unknown as Array<Record<string, unknown>>;
  const primaryAlert = model.orchestration.alerts[0] || null;
  const activeArtifact = safeDisplayValue(focus.artifact);

  return (
    <section className="ops-risk-page" aria-label="ops-risk-page">
      <div className="terminal-rhythm-layout">
        <section className="terminal-rhythm-band terminal-rhythm-state" aria-label="ops-risk-observe">
          <PanelCard
            title="风险观察"
            kicker="observe / gate + alert"
            meta="先看是否可继续观察、当前最危险项和门禁概况。"
            actions={<Badge value={primaryAlert?.tone || 'neutral'}>{primaryAlert ? `告警 ${alertCount}` : '无显著告警'}</Badge>}
          >
            {primaryAlert ? (
              <div className="empty-block">{primaryAlert.title}｜{safeDisplayValue(primaryAlert.summary || primaryAlert.detail || '等待进一步诊断')}</div>
            ) : (
              <div className="empty-block">当前没有显著告警，继续检查门禁链与焦点槽位。</div>
            )}
            <MetricStrip items={model.signalRisk.gateScores} showRawValues={view.signalRisk.gateScoresStrip.showRawValues} />
            <MetricStrip items={model.orchestration.guards} showRawValues={view.orchestration.guardsStrip.showRawValues} />
          </PanelCard>
        </section>

        <section className="terminal-rhythm-band terminal-rhythm-focus" aria-label="ops-risk-diagnose">
          <PanelCard
            title="风险诊断"
            kicker="diagnose / focus + chain"
            meta="聚焦当前触发槽位与控制链，确认异常是噪音还是结构性问题。"
          >
            <DenseTable
              columns={view.signalRisk.focusSlotsTableColumns}
              rows={previewRows(focusSlots)}
            />
            <DenseTable
              columns={view.signalRisk.controlChainTableColumns}
              rows={previewRows(controlChain)}
            />
          </PanelCard>
        </section>

        <section className="terminal-rhythm-band terminal-rhythm-action" aria-label="ops-risk-act">
          <PanelCard
            title="动作分流"
            kicker="act / next page"
            meta="执行动作统一进入 runbooks / workflow，深追踪进入 traces。"
          >
            <div className="overview-entry-grid">
              <Link className="panel-card overview-entry-card" to={buildOpsPageLink('audits', focus)}>
                <p className="panel-kicker">diagnose</p>
                <strong>审计诊断</strong>
                <span>route audit / calibration / dependency trend</span>
              </Link>
              <Link className="panel-card overview-entry-card" to={buildOpsPageLink('runbooks', focus)}>
                <p className="panel-kicker">act</p>
                <strong>处置手册</strong>
                <span>runbook / precheck / playbook bundle</span>
              </Link>
              <Link className="panel-card overview-entry-card" to={buildOpsPageLink('workflow', focus)}>
                <p className="panel-kicker">governance</p>
                <strong>工作流</strong>
                <span>proposal / approval / replay / rollback</span>
              </Link>
              <Link className="panel-card overview-entry-card" to={buildOpsPageLink('traces', focus)}>
                <p className="panel-kicker">trace</p>
                <strong>链路追踪</strong>
                <span>trace context / event stream / artifacts</span>
              </Link>
              {activeArtifact !== '—' ? (
                <Link className="panel-card overview-entry-card" to={buildWorkspaceLink('artifacts', { artifact: activeArtifact })}>
                  <p className="panel-kicker">evidence</p>
                  <strong>焦点工件</strong>
                  <span>{labelFor(activeArtifact)}</span>
                </Link>
              ) : null}
            </div>
          </PanelCard>
        </section>

        <section className="terminal-rhythm-band terminal-rhythm-evidence" aria-label="ops-risk-evidence">
          <PanelCard
            title="当前动作栈"
            kicker="evidence / queue + repair"
            meta="保留最小动作证据，不在当前页复写完整旧终端明细。"
          >
            <DenseTable
              columns={view.signalRisk.actionQueueTableColumns}
              rows={previewRows(actionQueue)}
            />
            <MetricStrip items={model.signalRisk.repairPlan} showRawValues={view.signalRisk.repairPlanStrip.showRawValues} />
          </PanelCard>
        </section>
      </div>
    </section>
  );
}
