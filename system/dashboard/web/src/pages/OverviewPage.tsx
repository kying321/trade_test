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

function buildJin10Metrics(model: TerminalReadModel | null): MetricItem[] {
  if (!model) return [];
  const payload = model.workspace.artifactPayloads?.jin10_mcp_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  return [
    {
      id: 'jin10-calendar-total',
      label: '日历事件',
      value: summary.calendar_total ?? '—',
      tone: statusTone(summary.calendar_total && Number(summary.calendar_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'jin10-high-importance',
      label: '高重要事件',
      value: summary.high_importance_count ?? '—',
      tone: statusTone(summary.high_importance_count && Number(summary.high_importance_count) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'jin10-flash-total',
      label: '最新快讯',
      value: summary.flash_total ?? '—',
      tone: statusTone(summary.flash_total && Number(summary.flash_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'jin10-quote-watch',
      label: '盯盘品种',
      value: Array.isArray(summary.quote_watch)
        ? summary.quote_watch
          .map((row) => safeDisplayValue((row as Record<string, unknown>)?.name || (row as Record<string, unknown>)?.code))
          .filter((value) => value && value !== '—')
          .join(', ')
        : '—',
      tone: statusTone(Array.isArray(summary.quote_watch) && summary.quote_watch.length ? 'positive' : 'warning'),
    },
  ];
}

function buildJin10Highlights(model: TerminalReadModel | null) {
  const payload = model?.workspace.artifactPayloads?.jin10_mcp_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  const highImportanceTitles = Array.isArray(summary.high_importance_titles)
    ? summary.high_importance_titles
      .map((value) => safeDisplayValue(value))
      .filter((value) => value && value !== '—')
    : [];
  const latestFlashBriefs = Array.isArray(summary.latest_flash_briefs)
    ? summary.latest_flash_briefs
      .map((value) => safeDisplayValue(value))
      .filter((value) => value && value !== '—')
    : [];
  const quoteWatch = Array.isArray(summary.quote_watch)
    ? summary.quote_watch.map((row) => {
      const record = (row || {}) as Record<string, unknown>;
      const name = safeDisplayValue(record.name || record.code);
      const close = safeDisplayValue(record.close);
      const ups = safeDisplayValue(record.ups_percent);
      return [name, close !== '—' ? close : '', ups !== '—' ? `${ups}%` : ''].filter(Boolean).join(' ');
    }).filter((value) => value && value !== '—')
    : [];

  return {
    recommendedBrief: safeDisplayValue(payload?.recommended_brief),
    takeaway: safeDisplayValue(payload?.takeaway),
    highImportanceTitles,
    latestFlashBriefs,
    quoteWatch,
  };
}

function buildAxiosMetrics(model: TerminalReadModel | null): MetricItem[] {
  if (!model) return [];
  const payload = model.workspace.artifactPayloads?.axios_site_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  return [
    {
      id: 'axios-news-total',
      label: 'Axios 条目',
      value: summary.news_total ?? '—',
      tone: statusTone(summary.news_total && Number(summary.news_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'axios-local-total',
      label: '本地新闻',
      value: summary.local_total ?? '—',
      tone: statusTone(summary.local_total && Number(summary.local_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'axios-national-total',
      label: '全国新闻',
      value: summary.national_total ?? '—',
      tone: statusTone(summary.national_total && Number(summary.national_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'axios-keywords',
      label: '关键词',
      value: Array.isArray(summary.top_keywords)
        ? summary.top_keywords.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—').join(', ')
        : '—',
      tone: statusTone(Array.isArray(summary.top_keywords) && summary.top_keywords.length ? 'positive' : 'warning'),
    },
  ];
}

function buildAxiosHighlights(model: TerminalReadModel | null) {
  const payload = model?.workspace.artifactPayloads?.axios_site_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  const topTitles = Array.isArray(summary.top_titles)
    ? summary.top_titles.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const topKeywords = Array.isArray(summary.top_keywords)
    ? summary.top_keywords.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  return {
    recommendedBrief: safeDisplayValue(payload?.recommended_brief),
    takeaway: safeDisplayValue(payload?.takeaway),
    topTitles,
    topKeywords,
  };
}

function buildPolymarketMetrics(model: TerminalReadModel | null): MetricItem[] {
  if (!model) return [];
  const payload = model.workspace.artifactPayloads?.polymarket_gamma_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  const yesAvg = typeof summary.yes_price_avg === 'number'
    ? `${(Number(summary.yes_price_avg) * 100).toFixed(1)}%`
    : safeDisplayValue(summary.yes_price_avg);
  return [
    {
      id: 'polymarket-markets-total',
      label: '活跃市场',
      value: summary.markets_total ?? '—',
      tone: statusTone(summary.markets_total && Number(summary.markets_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'polymarket-bullish-count',
      label: '偏多合约',
      value: summary.bullish_count ?? '—',
      tone: statusTone(summary.bullish_count && Number(summary.bullish_count) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'polymarket-bearish-count',
      label: '偏空合约',
      value: summary.bearish_count ?? '—',
      tone: statusTone(summary.bearish_count && Number(summary.bearish_count) > 0 ? 'warning' : 'positive'),
    },
    {
      id: 'polymarket-yes-price-avg',
      label: '平均 Yes 概率',
      value: yesAvg,
      tone: statusTone(summary.yes_price_avg && Number(summary.yes_price_avg) > 0 ? 'positive' : 'warning'),
    },
  ];
}

function buildPolymarketHighlights(model: TerminalReadModel | null) {
  const payload = model?.workspace.artifactPayloads?.polymarket_gamma_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  const topTitles = Array.isArray(summary.top_titles)
    ? summary.top_titles.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const topCategories = Array.isArray(summary.top_categories)
    ? summary.top_categories.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  return {
    recommendedBrief: safeDisplayValue(payload?.recommended_brief),
    takeaway: safeDisplayValue(payload?.takeaway),
    topTitles,
    topCategories,
  };
}

function buildExternalIntelligenceMetrics(model: TerminalReadModel | null): MetricItem[] {
  if (!model) return [];
  const payload = model.workspace.artifactPayloads?.external_intelligence_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  const polyAvg = typeof summary.polymarket_yes_price_avg === 'number'
    ? `${(Number(summary.polymarket_yes_price_avg) * 100).toFixed(1)}%`
    : safeDisplayValue(summary.polymarket_yes_price_avg);
  return [
    {
      id: 'external-sources-total',
      label: '情报源',
      value: summary.sources_total ?? '—',
      tone: statusTone(summary.sources_total && Number(summary.sources_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'external-calendar-total',
      label: '日历事件',
      value: summary.calendar_total ?? '—',
      tone: statusTone(summary.calendar_total && Number(summary.calendar_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'external-flash-total',
      label: '最新快讯',
      value: summary.flash_total ?? '—',
      tone: statusTone(summary.flash_total && Number(summary.flash_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'external-news-total',
      label: '新闻条目',
      value: summary.axios_news_total ?? '—',
      tone: statusTone(summary.axios_news_total && Number(summary.axios_news_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'external-poly-total',
      label: '预测市场',
      value: summary.polymarket_markets_total ?? '—',
      tone: statusTone(summary.polymarket_markets_total && Number(summary.polymarket_markets_total) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'external-poly-avg',
      label: '平均 Yes 概率',
      value: polyAvg,
      tone: statusTone(summary.polymarket_yes_price_avg && Number(summary.polymarket_yes_price_avg) > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'external-quote-watch',
      label: '盯盘品种',
      value: Array.isArray(summary.quote_watch)
        ? summary.quote_watch.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—').join(', ')
        : '—',
      tone: statusTone(Array.isArray(summary.quote_watch) && summary.quote_watch.length ? 'positive' : 'warning'),
    },
  ];
}

function buildExternalIntelligenceHighlights(model: TerminalReadModel | null) {
  const payload = model?.workspace.artifactPayloads?.external_intelligence_snapshot?.payload as Record<string, unknown> | undefined;
  const summary = (payload?.summary as Record<string, unknown> | undefined) || {};
  const topTitles = Array.isArray(summary.top_titles)
    ? summary.top_titles.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const topKeywords = Array.isArray(summary.top_keywords)
    ? summary.top_keywords.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const polymarketTopCategories = Array.isArray(summary.polymarket_top_categories)
    ? summary.polymarket_top_categories.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const quoteWatch = Array.isArray(summary.quote_watch)
    ? summary.quote_watch.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  return {
    recommendedBrief: safeDisplayValue(payload?.recommended_brief),
    takeaway: safeDisplayValue(payload?.takeaway),
    topTitles,
    topKeywords,
    polymarketTopCategories,
    quoteWatch,
  };
}

function buildOverviewStateMetrics(model: TerminalReadModel | null): MetricItem[] {
  if (!model) return [];
  return [
    {
      id: 'overview-state-surface',
      label: '当前视图',
      value: model.surface.effectiveLabel || '—',
      tone: statusTone(model.surface.effective),
    },
    {
      id: 'overview-state-change-class',
      label: '变更级别',
      value: labelFor(String(model.meta.change_class || 'RESEARCH_ONLY')),
      tone: statusTone(model.meta.change_class),
    },
    {
      id: 'overview-state-artifacts',
      label: '工件数',
      value: model.workspace.artifactRows.length,
      tone: statusTone(model.workspace.artifactRows.length > 0 ? 'positive' : 'warning'),
    },
    {
      id: 'overview-state-source-heads',
      label: '源头主头',
      value: model.workspace.sourceHeads.length,
      tone: statusTone(model.workspace.sourceHeads.length > 0 ? 'positive' : 'warning'),
    },
  ];
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
  const externalIntelligenceMetrics = buildExternalIntelligenceMetrics(model);
  const externalIntelligenceHighlights = buildExternalIntelligenceHighlights(model);
  const jin10Metrics = buildJin10Metrics(model);
  const jin10Highlights = buildJin10Highlights(model);
  const axiosMetrics = buildAxiosMetrics(model);
  const axiosHighlights = buildAxiosHighlights(model);
  const polymarketMetrics = buildPolymarketMetrics(model);
  const polymarketHighlights = buildPolymarketHighlights(model);
  const overviewStateMetrics = buildOverviewStateMetrics(model);
  const externalLink = buildWorkspacePageLink(
    'artifacts',
    { artifact: 'external_intelligence_snapshot', group: 'system_anchor' },
  );
  const jin10Link = buildWorkspacePageLink(
    'artifacts',
    { artifact: 'jin10_mcp_snapshot', group: 'system_anchor' },
  );
  const axiosLink = buildWorkspacePageLink(
    'artifacts',
    { artifact: 'axios_site_snapshot', group: 'system_anchor' },
  );
  const polymarketLink = buildWorkspacePageLink(
    'artifacts',
    { artifact: 'polymarket_gamma_snapshot', group: 'system_anchor' },
  );

  return (
    <section className="overview-page">
      <div className="overview-hero">
        <h2>总览</h2>
        <p>先看系统态势，再分流进入操作终端或研究工作区。</p>
      </div>
      <section className="overview-rhythm-band" aria-label="overview-rhythm-state">
        <h3>当前状态</h3>
        <p>系统状态 / 入口 / 路由总览</p>
        {overviewStateMetrics.length ? <MetricStrip items={overviewStateMetrics} /> : null}
      </section>
      <section className="overview-rhythm-band" aria-label="overview-rhythm-focus">
        <h3>当前焦点</h3>
        <p>国内商品推理线和主线冲突统一在同一焦点带，避免首屏并列抢占。</p>
        <div data-search-anchor="overview-mainline-summary" id="overview-mainline-summary">
          <PanelCard
            title="当前焦点"
            kicker="研究主线摘要"
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
        </div>
        {commodityReasoningMetrics.length ? (
        <div data-search-anchor="overview-commodity-reasoning" id="overview-commodity-reasoning">
          <PanelCard
            title="国内商品推理线"
            kicker="reasoning / visible-priority"
            meta="首屏固定摘要：主情景 / 主传导链 / 范围 / 边界 / 失效条件"
          >
            <MetricStrip items={commodityReasoningMetrics} showRawValues />
          </PanelCard>
        </div>
        ) : null}
        {externalIntelligenceMetrics.length ? (
        <div data-search-anchor="overview-external-intelligence" id="overview-external-intelligence">
          <PanelCard
            title="外部情报带"
            kicker="research sidecar / aggregated"
            meta={externalIntelligenceHighlights.recommendedBrief !== '—' ? externalIntelligenceHighlights.recommendedBrief : 'Jin10 / Axios 外部情报摘要'}
            actions={<Link className="button" to={externalLink}>查看外部情报工件</Link>}
          >
            <MetricStrip items={externalIntelligenceMetrics} />
            {externalIntelligenceHighlights.takeaway !== '—' ? <div className="empty-block">重点：{externalIntelligenceHighlights.takeaway}</div> : null}
            {externalIntelligenceHighlights.topTitles.length ? <div className="empty-block">重点标题：{externalIntelligenceHighlights.topTitles.join(' ｜ ')}</div> : null}
            {externalIntelligenceHighlights.topKeywords.length ? <div className="empty-block">关键词：{externalIntelligenceHighlights.topKeywords.join(' ｜ ')}</div> : null}
            {externalIntelligenceHighlights.polymarketTopCategories.length ? <div className="empty-block">Polymarket 主题：{externalIntelligenceHighlights.polymarketTopCategories.join(' ｜ ')}</div> : null}
            {externalIntelligenceHighlights.quoteWatch.length ? <div className="empty-block">盯盘摘要：{externalIntelligenceHighlights.quoteWatch.join(' ｜ ')}</div> : null}
          </PanelCard>
        </div>
        ) : null}
        {jin10Metrics.length ? (
        <div data-search-anchor="overview-jin10-sidecar" id="overview-jin10-sidecar">
          <PanelCard
            title="Jin10 研究侧边车"
            kicker="research sidecar / macro calendar / flash"
            meta={jin10Highlights.recommendedBrief !== '—' ? jin10Highlights.recommendedBrief : '财经日历 / 快讯 / 盯盘摘要'}
            actions={<Link className="button" to={jin10Link}>查看 Jin10 工件</Link>}
          >
            <MetricStrip items={jin10Metrics} />
            {jin10Highlights.takeaway !== '—' ? <div className="empty-block">重点：{jin10Highlights.takeaway}</div> : null}
            {jin10Highlights.highImportanceTitles.length ? <div className="empty-block">高重要日历：{jin10Highlights.highImportanceTitles.join(' ｜ ')}</div> : null}
            {jin10Highlights.latestFlashBriefs.length ? <div className="empty-block">最新快讯：{jin10Highlights.latestFlashBriefs.join(' ｜ ')}</div> : null}
            {jin10Highlights.quoteWatch.length ? <div className="empty-block">盯盘摘要：{jin10Highlights.quoteWatch.join(' ｜ ')}</div> : null}
          </PanelCard>
        </div>
        ) : null}
        {axiosMetrics.length ? (
        <div data-search-anchor="overview-axios-sidecar" id="overview-axios-sidecar">
          <PanelCard
            title="Axios 研究侧边车"
            kicker="research sidecar / news sitemap"
            meta={axiosHighlights.recommendedBrief !== '—' ? axiosHighlights.recommendedBrief : '新闻条目 / 本地与全国分布 / 关键词'}
            actions={<Link className="button" to={axiosLink}>查看 Axios 工件</Link>}
          >
            <MetricStrip items={axiosMetrics} />
            {axiosHighlights.takeaway !== '—' ? <div className="empty-block">重点：{axiosHighlights.takeaway}</div> : null}
            {axiosHighlights.topTitles.length ? <div className="empty-block">重点标题：{axiosHighlights.topTitles.join(' ｜ ')}</div> : null}
            {axiosHighlights.topKeywords.length ? <div className="empty-block">关键词：{axiosHighlights.topKeywords.join(' ｜ ')}</div> : null}
          </PanelCard>
        </div>
        ) : null}
        {polymarketMetrics.length ? (
        <div data-search-anchor="overview-polymarket-sidecar" id="overview-polymarket-sidecar">
          <PanelCard
            title="Polymarket 情绪侧边车"
            kicker="research sidecar / prediction market"
            meta={polymarketHighlights.recommendedBrief !== '—' ? polymarketHighlights.recommendedBrief : '预测市场 / 情绪分布 / 热门主题'}
            actions={<Link className="button" to={polymarketLink}>查看 Polymarket 工件</Link>}
          >
            <MetricStrip items={polymarketMetrics} />
            {polymarketHighlights.takeaway !== '—' ? <div className="empty-block">重点：{polymarketHighlights.takeaway}</div> : null}
            {polymarketHighlights.topTitles.length ? <div className="empty-block">热点问题：{polymarketHighlights.topTitles.join(' ｜ ')}</div> : null}
            {polymarketHighlights.topCategories.length ? <div className="empty-block">热门主题：{polymarketHighlights.topCategories.join(' ｜ ')}</div> : null}
          </PanelCard>
        </div>
        ) : null}
      </section>
      <section className="overview-rhythm-band" aria-label="overview-rhythm-action">
        <h3>下一步</h3>
        <p>选择入口进入操作终端、研究工作区或契约验收。</p>
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
          <Link className="panel-card overview-entry-card overview-entry-card-emphasis" to="/cpa" aria-label="进入 CPA 管理">
            <p className="panel-kicker">高频管理</p>
            <strong>CPA 管理</strong>
            <span>认证文件 / 配额 / 删除保护，直接处理 auth-files 与误删防护。</span>
          </Link>
        </div>
      </section>
      {requestedInternal ? (
        <section className="overview-rhythm-band" aria-label="overview-rhythm-evidence">
          <h3>证据</h3>
        <div data-search-anchor="overview-alignment-projection" id="overview-alignment-projection">
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
        </div>
        </section>
      ) : null}
    </section>
  );
}
