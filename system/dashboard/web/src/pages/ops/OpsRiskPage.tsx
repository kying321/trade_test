import { Link } from 'react-router-dom';
import { Badge, DenseTable, KeyValueGrid, MetricStrip, PanelCard } from '../../components/ui-kit';
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

function jin10Payload(model: TerminalReadModel) {
  return (model.workspace.artifactPayloads?.jin10_mcp_snapshot?.payload as Record<string, unknown> | undefined) || {};
}

function axiosPayload(model: TerminalReadModel) {
  return (model.workspace.artifactPayloads?.axios_site_snapshot?.payload as Record<string, unknown> | undefined) || {};
}

function polymarketPayload(model: TerminalReadModel) {
  return (model.workspace.artifactPayloads?.polymarket_gamma_snapshot?.payload as Record<string, unknown> | undefined) || {};
}

function externalPayload(model: TerminalReadModel) {
  return (model.workspace.artifactPayloads?.external_intelligence_snapshot?.payload as Record<string, unknown> | undefined) || {};
}

export function OpsRiskPage({ model, focus }: OpsRiskPageProps) {
  const view = model.view.terminal;
  const alertCount = model.orchestration.alerts.length;
  const focusSlots = model.signalRisk.focusSlots as unknown as Array<Record<string, unknown>>;
  const controlChain = model.signalRisk.controlChain as unknown as Array<Record<string, unknown>>;
  const actionQueue = model.signalRisk.actionQueue as unknown as Array<Record<string, unknown>>;
  const primaryAlert = model.orchestration.alerts[0] || null;
  const activeArtifact = safeDisplayValue(focus.artifact);
  const external = externalPayload(model);
  const externalSummary = (external.summary as Record<string, unknown> | undefined) || {};
  const jin10 = jin10Payload(model);
  const jin10Summary = (jin10.summary as Record<string, unknown> | undefined) || {};
  const axios = axiosPayload(model);
  const axiosSummary = (axios.summary as Record<string, unknown> | undefined) || {};
  const polymarket = polymarketPayload(model);
  const polymarketSummary = (polymarket.summary as Record<string, unknown> | undefined) || {};

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

        <section className="terminal-rhythm-band terminal-rhythm-state" aria-label="ops-risk-market-input">
          <PanelCard
            title="市场输入"
            kicker="observe / external intelligence"
            meta="把 research sidecar 外部情报收敛到风险观察页，不把它们提升为 live source-of-truth。"
          >
            {Object.keys(external).length ? (
              <PanelCard
                title="外部情报带"
                kicker="aggregated / jin10 + axios + polymarket"
                meta={safeDisplayValue(external.recommended_brief || '外部情报汇总')}
                actions={<Link className="button" to={buildWorkspaceLink('artifacts', { artifact: 'external_intelligence_snapshot', group: 'system_anchor' })}>查看外部情报工件</Link>}
              >
                <KeyValueGrid
                  rows={[
                    { key: 'sources_total', label: '情报源', value: externalSummary.sources_total ?? '—' },
                    { key: 'calendar_total', label: '日历事件', value: externalSummary.calendar_total ?? '—' },
                    { key: 'flash_total', label: '最新快讯', value: externalSummary.flash_total ?? '—' },
                    { key: 'news_total', label: '新闻条目', value: externalSummary.axios_news_total ?? '—' },
                    { key: 'polymarket_markets_total', label: '预测市场', value: externalSummary.polymarket_markets_total ?? '—' },
                    {
                      key: 'polymarket_yes_price_avg',
                      label: '平均 Yes 概率',
                      value: typeof externalSummary.polymarket_yes_price_avg === 'number'
                        ? `${(Number(externalSummary.polymarket_yes_price_avg) * 100).toFixed(1)}%`
                        : safeDisplayValue(externalSummary.polymarket_yes_price_avg),
                    },
                    {
                      key: 'quote_watch',
                      label: '盯盘品种',
                      value: Array.isArray(externalSummary.quote_watch)
                        ? externalSummary.quote_watch.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—').join(', ')
                        : '—',
                    },
                    { key: 'takeaway', label: '重点', value: safeDisplayValue(external.takeaway), showRaw: true },
                  ]}
                />
              </PanelCard>
            ) : null}
            <div className="overview-entry-grid">
              {Object.keys(jin10).length ? (
                <PanelCard
                  title="Jin10 研究侧边车"
                  kicker="calendar / flash / quote watch"
                  meta={safeDisplayValue(jin10.recommended_brief || '财经日历 / 快讯 / 盯盘摘要')}
                  actions={<Link className="button" to={buildWorkspaceLink('artifacts', { artifact: 'jin10_mcp_snapshot', group: 'system_anchor' })}>查看 Jin10 工件</Link>}
                >
                  <KeyValueGrid
                    rows={[
                      { key: 'calendar_total', label: '日历事件', value: jin10Summary.calendar_total ?? '—' },
                      { key: 'high_importance_count', label: '高重要事件', value: jin10Summary.high_importance_count ?? '—' },
                      { key: 'flash_total', label: '最新快讯', value: jin10Summary.flash_total ?? '—' },
                      {
                        key: 'quote_watch',
                        label: '盯盘品种',
                        value: Array.isArray(jin10Summary.quote_watch)
                          ? jin10Summary.quote_watch
                            .map((row) => safeDisplayValue((row as Record<string, unknown>)?.name || (row as Record<string, unknown>)?.code))
                            .filter((value) => value && value !== '—')
                            .join(', ')
                          : '—',
                      },
                    ]}
                  />
                  {safeDisplayValue(jin10.takeaway) !== '—' ? <div className="empty-block">重点：{safeDisplayValue(jin10.takeaway)}</div> : null}
                </PanelCard>
              ) : null}
              {Object.keys(axios).length ? (
                <PanelCard
                  title="Axios 研究侧边车"
                  kicker="news sitemap / headlines"
                  meta={safeDisplayValue(axios.recommended_brief || '新闻条目 / 本地与全国分布 / 关键词')}
                  actions={<Link className="button" to={buildWorkspaceLink('artifacts', { artifact: 'axios_site_snapshot', group: 'system_anchor' })}>查看 Axios 工件</Link>}
                >
                  <KeyValueGrid
                    rows={[
                      { key: 'news_total', label: 'Axios 条目', value: axiosSummary.news_total ?? '—' },
                      { key: 'local_total', label: '本地新闻', value: axiosSummary.local_total ?? '—' },
                      { key: 'national_total', label: '全国新闻', value: axiosSummary.national_total ?? '—' },
                      {
                        key: 'top_keywords',
                        label: '关键词',
                        value: Array.isArray(axiosSummary.top_keywords)
                          ? axiosSummary.top_keywords.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—').join(', ')
                          : '—',
                      },
                    ]}
                  />
                  {safeDisplayValue(axios.takeaway) !== '—' ? <div className="empty-block">重点：{safeDisplayValue(axios.takeaway)}</div> : null}
                </PanelCard>
              ) : null}
              {Object.keys(polymarket).length ? (
                <PanelCard
                  title="Polymarket 情绪侧边车"
                  kicker="prediction market / sentiment"
                  meta={safeDisplayValue(polymarket.recommended_brief || '预测市场 / 情绪分布 / 热门主题')}
                  actions={<Link className="button" to={buildWorkspaceLink('artifacts', { artifact: 'polymarket_gamma_snapshot', group: 'system_anchor' })}>查看 Polymarket 工件</Link>}
                >
                  <KeyValueGrid
                    rows={[
                      { key: 'markets_total', label: '活跃市场', value: polymarketSummary.markets_total ?? '—' },
                      { key: 'bullish_count', label: '偏多合约', value: polymarketSummary.bullish_count ?? '—' },
                      { key: 'bearish_count', label: '偏空合约', value: polymarketSummary.bearish_count ?? '—' },
                      {
                        key: 'yes_price_avg',
                        label: '平均 Yes 概率',
                        value: typeof polymarketSummary.yes_price_avg === 'number'
                          ? `${(Number(polymarketSummary.yes_price_avg) * 100).toFixed(1)}%`
                          : safeDisplayValue(polymarketSummary.yes_price_avg),
                      },
                      {
                        key: 'top_categories',
                        label: '热门主题',
                        value: Array.isArray(polymarketSummary.top_categories)
                          ? polymarketSummary.top_categories.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—').join(', ')
                          : '—',
                      },
                    ]}
                  />
                  {safeDisplayValue(polymarket.takeaway) !== '—' ? <div className="empty-block">重点：{safeDisplayValue(polymarket.takeaway)}</div> : null}
                </PanelCard>
              ) : null}
            </div>
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
