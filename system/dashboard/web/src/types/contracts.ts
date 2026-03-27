export type SurfaceKey = 'public' | 'internal';
export type Tone = 'positive' | 'warning' | 'negative' | 'neutral';

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
  [key: string]: JsonValue;
}

export interface SnapshotMeta {
  generated_at_utc?: string;
  artifact_payload_count?: number;
  catalog_count?: number;
  backtest_artifact_count?: number;
  change_class?: string;
  surface_key?: string;
  redaction_level?: string;
}

export interface InterfaceCatalogItem {
  id: string;
  label?: string;
  kind?: string;
  url?: string;
  description?: string;
}

export interface SourceHeadEntry {
  label?: string;
  status?: string;
  research_decision?: string;
  recommended_brief?: string;
  takeaway?: string;
  summary?: string;
  path?: string;
  active_baseline?: unknown;
  local_candidate?: unknown;
  transfer_watch?: unknown;
  return_candidate?: unknown;
  return_watch?: unknown;
  generated_at_utc?: string;
  locator?: string;
}

export interface ArtifactSummary {
  id?: string;
  label?: string;
  path?: string;
  locator?: string;
  category?: string;
  payload_key?: string | null;
  artifact_layer?: string;
  artifact_group?: string;
  action?: string;
  status?: string;
  change_class?: string;
  generated_at_utc?: string;
  research_decision?: string;
  recommended_brief?: string;
  takeaway?: string;
  top_level_keys?: string[];
}

export interface ArtifactPayloadEntry {
  label?: string;
  path?: string;
  locator?: string;
  category?: string;
  payload?: unknown;
  summary?: ArtifactSummary;
}

export interface CatalogRow {
  id?: string;
  label?: string;
  path?: string;
  locator?: string;
  category?: string;
  payload_key?: string | null;
  artifact_layer?: string;
  artifact_group?: string;
  action?: string;
  status?: string;
  change_class?: string;
  generated_at_utc?: string;
  research_decision?: string;
  recommended_brief?: string;
  takeaway?: string;
  top_level_keys?: string[];
}

export interface DomainSummaryRow {
  id: string;
  label?: string;
  count?: number;
  payload_count?: number;
  positive?: number;
  warning?: number;
  negative?: number;
  neutral?: number;
  headline?: string;
}

export interface BacktestArtifact {
  id: string;
  label?: string;
  path?: string;
  locator?: string;
  start?: string;
  end?: string;
  total_return?: number;
  annual_return?: number;
  max_drawdown?: number;
  win_rate?: number;
  profit_factor?: number;
  expectancy?: number;
  trades?: number;
  violations?: number;
  positive_window_ratio?: number;
  equity_curve_sample?: Array<{ date?: string; equity?: number }>;
}

export interface SurfaceContract {
  label?: string;
  route_key?: string;
  snapshot_path?: string;
  availability?: string;
  redaction_level?: string;
  fallback_snapshot_path?: string;
  deploy_policy?: string;
  notes?: string;
}

export interface PublicTopologyEntry {
  id: string;
  label?: string;
  role?: string;
  url?: string;
  expected_status?: string;
  source?: string;
  fallback?: string;
  notes?: string;
}

export interface PublicAcceptanceSummary {
  artifact_id: string;
  status?: string;
  change_class?: string;
  generated_at_utc?: string;
  report_path?: string;
  artifact_path?: string;
  topology_status?: string;
  workspace_status?: string;
  topology_generated_at_utc?: string;
  workspace_generated_at_utc?: string;
  frontend_public?: string;
  root_public_entry?: string;
  public_snapshot_fetch_count?: number;
  internal_snapshot_fetch_count?: number;
  root_contracts_screenshot_path?: string;
  pages_contracts_screenshot_path?: string;
  orderflow_filter_route?: string;
  orderflow_source_available?: boolean;
  orderflow_active_artifact?: string;
  orderflow_visible_artifacts?: string;
}

export interface PublicAcceptanceCheckRow {
  id: string;
  label: string;
  status?: string;
  ok?: boolean;
  generated_at_utc?: string;
  report_path?: string;
  returncode?: number;
  failure_reason?: string;
  headline?: string;
  frontend_public?: string;
  tls_fallback_used?: boolean;
  root_public_entry?: string;
  root_overview_screenshot_path?: string;
  pages_overview_screenshot_path?: string;
  root_contracts_screenshot_path?: string;
  pages_contracts_screenshot_path?: string;
  requested_surface?: string;
  effective_surface?: string;
  snapshot_endpoint_observed?: string;
  public_snapshot_fetch_count?: number;
  internal_snapshot_fetch_count?: number;
  orderflow_filter_route?: string;
  orderflow_source_available?: boolean;
  orderflow_active_artifact?: string;
  orderflow_visible_artifacts?: string;
  stdout?: string;
  stderr?: string;
}

export interface PublicAcceptanceSubcommandRow {
  id: string;
  label: string;
  returncode?: number;
  payload_present?: boolean;
  stdout_bytes?: number;
  stderr_bytes?: number;
  cmd?: string;
  cwd?: string;
  stdout?: string;
  stderr?: string;
}

export interface DashboardSnapshot {
  action?: string;
  ok?: boolean;
  status?: string;
  change_class?: string;
  generated_at_utc?: string;
  surface?: string;
  workspace?: string;
  ui_routes?: Record<string, string>;
  meta?: SnapshotMeta;
  artifact_payloads?: Record<string, ArtifactPayloadEntry>;
  catalog?: CatalogRow[];
  domain_summary?: DomainSummaryRow[];
  interface_catalog?: InterfaceCatalogItem[];
  public_topology?: PublicTopologyEntry[];
  source_heads?: Record<string, SourceHeadEntry>;
  experience_contract?: Record<string, unknown>;
  surface_contracts?: Record<string, SurfaceContract>;
  backtest_artifacts?: BacktestArtifact[];
  workspace_default_focus?: WorkspaceDefaultFocus;
  conversation_feedback_projection?: ConversationFeedbackProjection | null;
}

export interface LoadedSurface {
  snapshot: DashboardSnapshot;
  requestedSurface: SurfaceKey;
  effectiveSurface: SurfaceKey;
  snapshotPath: string;
  warnings: string[];
}

export interface MetricItem {
  id: string;
  label: string;
  value: unknown;
  hint?: string;
  tone?: Tone;
}

export interface FreshnessRow {
  id: string;
  label: string;
  timestamp?: string;
  ageMinutes?: number | null;
  state: 'fresh' | 'warning' | 'stale' | 'unknown';
  artifact?: string;
}

export interface ActionLogRow {
  id: string;
  ts?: string;
  label: string;
  status?: string;
  detail?: string;
  tone?: Tone;
}

export interface OperatorAlertField {
  key: string;
  label: string;
  value: unknown;
  kind?: ViewValueKind;
  showRaw?: boolean;
}

export interface OperatorAlertSection {
  id: string;
  title: string;
  summary?: string;
  tone?: Tone;
  defaultOpen?: boolean;
  rows: OperatorAlertField[];
}

export interface OperatorAlertLink {
  id: string;
  label: string;
  to: string;
  tone?: Tone;
}

export interface FocusPointer {
  artifact?: string;
  panel?: string;
  section?: string;
  row?: string;
}

export interface ConversationFeedbackAnchor {
  route?: string;
  artifact?: string;
  component?: string;
}

export interface ConversationFeedbackSummary {
  alignment_score?: number;
  blocker_pressure?: number;
  execution_clarity?: number;
  readability_pressure?: number;
  drift_state?: string;
  headline?: string;
}

export interface ConversationFeedbackEvent {
  feedback_id: string;
  created_at_utc?: string;
  source?: string;
  domain?: string;
  headline?: string;
  summary?: string;
  recommended_action?: string;
  alignment_delta?: number;
  blocker_delta?: number;
  execution_delta?: number;
  readability_delta?: number;
  impact_score?: number;
  confidence?: number;
  status?: string;
  anchors?: ConversationFeedbackAnchor[];
}

export interface ConversationFeedbackAction {
  feedback_id?: string;
  recommended_action?: string;
  impact_score?: number;
  anchors?: ConversationFeedbackAnchor[];
}

export interface ConversationFeedbackTrendPoint {
  feedback_id?: string;
  created_at_utc?: string;
  alignment_score?: number;
  blocker_pressure?: number;
  execution_clarity?: number;
  readability_pressure?: number;
  drift_state?: string;
}

export interface ConversationFeedbackProjection {
  status?: string;
  change_class?: string;
  visibility?: string;
  generated_at_utc?: string;
  summary: ConversationFeedbackSummary;
  events: ConversationFeedbackEvent[];
  actions: ConversationFeedbackAction[];
  trends: ConversationFeedbackTrendPoint[];
  anchors: ConversationFeedbackAnchor[];
}

export interface ConversationFeedbackDomainGroup {
  domain: string;
  count: number;
  topHeadline?: string;
}

export interface WorkspaceDefaultFocus extends FocusPointer {
  group?: string;
  search_scope?: string;
}

export interface OperatorAlert {
  id: string;
  tone: Tone;
  title: string;
  summary?: string;
  detail?: string;
  highlights?: OperatorAlertField[];
  sections?: OperatorAlertSection[];
  links?: OperatorAlertLink[];
}

export interface LaneCard {
  state?: string;
  count?: number;
  priority_total?: number;
  brief?: string;
  head_symbol?: string;
  head_action?: string;
}

export interface FocusSlot {
  slot?: string;
  area?: string;
  target?: string;
  symbol?: string;
  action?: string;
  reason?: string;
  state?: string;
  priority_score?: number;
  priority_tier?: string;
  queue_rank?: number;
  blocker_detail?: string;
  done_when?: string;
  source_artifact?: string;
  source_status?: string;
  source_as_of?: string;
  source_age_minutes?: number | null;
  source_recency?: string;
  source_health?: string;
}

export interface ControlChainRow {
  stage?: string;
  label?: string;
  mission?: string;
  source_status?: string;
  source_brief?: string;
  source_decision?: string;
  source_artifact?: string;
  source_artifact_key?: string;
  next_target_artifact?: string;
  blocking_reason?: string;
  interface_fields?: string[];
  optimization_brief?: string;
}

export interface ResearchCandidateRow {
  id: string;
  label: string;
  status?: string;
  decision?: string;
  changeClass?: string;
  path?: string;
  takeaway?: string;
}

export interface SurfaceView {
  requested: SurfaceKey;
  effective: SurfaceKey;
  snapshotPath: string;
  warnings: string[];
  requestedLabel: string;
  effectiveLabel: string;
  redactionLevel: string;
  availability: string;
}

export interface SurfaceContractEntry extends SurfaceContract {
  id: string;
}

export type ViewValueKind = 'text' | 'badge' | 'number' | 'percent' | 'sparkline' | 'path';

export interface ViewFieldSchema {
  key: string;
  label: string;
  showRaw?: boolean;
  kind?: ViewValueKind;
}

export interface ViewColumnSchema extends ViewFieldSchema {}

export interface ViewMetricStripSchema {
  showRawValues: boolean;
}

export interface ViewDrilldownSchema {
  titleKey: string;
  statusKey?: string;
  summaryKeys?: string[];
  defaultOpenCount?: number;
  rowIdentityKeys?: string[];
  fields: ViewFieldSchema[];
}

export interface TerminalViewSchema {
  terminal: {
    orchestration: {
      topMetricsStrip: ViewMetricStripSchema;
      guardsStrip: ViewMetricStripSchema;
      actionLogColumns: ViewColumnSchema[];
      checklistColumns: ViewColumnSchema[];
    };
    dataRegime: {
      sourceConfidenceStrip: ViewMetricStripSchema;
      microCaptureStrip: ViewMetricStripSchema;
      regimeSummaryColumns: ViewColumnSchema[];
      regimeMatrixColumns: ViewColumnSchema[];
    };
    signalRisk: {
      gateScoresStrip: ViewMetricStripSchema;
      focusSlotsTableColumns: ViewColumnSchema[];
      focusSlotsDrilldown: ViewDrilldownSchema;
      controlChainTableColumns: ViewColumnSchema[];
      controlChainDrilldown: ViewDrilldownSchema;
      actionQueueTableColumns: ViewColumnSchema[];
      actionQueueDrilldown: ViewDrilldownSchema;
      repairPlanStrip: ViewMetricStripSchema;
      optimizationBacklogColumns: ViewColumnSchema[];
    };
    labReview: {
      researchHeadsStrip: ViewMetricStripSchema;
      manifestsStrip: ViewMetricStripSchema;
      backtestPoolColumns: ViewColumnSchema[];
      comparisonColumns: ViewColumnSchema[];
      candidateColumns: ViewColumnSchema[];
      stressSummaryColumns: ViewColumnSchema[];
    };
  };
  workspace: {
    artifacts: {
      list: {
        titleField: ViewFieldSchema;
        pathField: ViewFieldSchema;
      };
      inspector: {
        titleField: ViewFieldSchema;
        pathField: ViewFieldSchema;
      };
      metaFields: ViewFieldSchema[];
      topLevelKeyField: ViewFieldSchema;
      payloadField: ViewFieldSchema;
    };
    backtests: {
      poolColumns: ViewColumnSchema[];
      comparisonColumns: ViewColumnSchema[];
    };
    contracts: {
      surfaceAssertionFields: ViewFieldSchema[];
      surfacePublishFields: ViewFieldSchema[];
      interfaceColumns: ViewColumnSchema[];
      sourceHeadSummaryFields: {
        label: ViewFieldSchema;
        summary: ViewFieldSchema;
      };
      sourceHeadDetailFields: ViewFieldSchema[];
    };
  };
}

export interface TerminalReadModel {
  surface: SurfaceView;
  meta: SnapshotMeta;
  uiRoutes: Record<string, string>;
  experienceContract: Record<string, unknown>;
  navigation: {
    terminalPrimaryArtifact: Record<string, string>;
    terminalHandoffs: Record<string, OperatorAlertLink[]>;
  };
  view: TerminalViewSchema;
  orchestration: {
    alerts: OperatorAlert[];
    topMetrics: MetricItem[];
    freshness: FreshnessRow[];
    guards: MetricItem[];
    actionLog: ActionLogRow[];
    checklist: ActionLogRow[];
  };
  dataRegime: {
    sourceConfidence: MetricItem[];
    microCapture: MetricItem[];
    regimeSummary: Array<Record<string, unknown>>;
    regimeMatrix: Array<Record<string, unknown>>;
  };
  signalRisk: {
    laneCards: LaneCard[];
    focusSlots: FocusSlot[];
    gateScores: MetricItem[];
    controlChain: ControlChainRow[];
    actionQueue: FocusSlot[];
    repairPlan: MetricItem[];
    optimizationBacklog: Array<Record<string, unknown>>;
  };
  labReview: {
    researchHeads: MetricItem[];
    backtests: BacktestArtifact[];
    comparisonRows: Array<Record<string, unknown>>;
    manifests: MetricItem[];
    candidateRows: ResearchCandidateRow[];
    stressSummary: Array<Record<string, unknown>>;
  };
  feedback: {
    projection: ConversationFeedbackProjection | null;
    domainGroups: ConversationFeedbackDomainGroup[];
  };
  workspace: {
    domains: DomainSummaryRow[];
    interfaces: InterfaceCatalogItem[];
    publicTopology: PublicTopologyEntry[];
    publicAcceptance: {
      summary: PublicAcceptanceSummary | null;
      checks: PublicAcceptanceCheckRow[];
      subcommands: PublicAcceptanceSubcommandRow[];
    };
    surfaceContracts: SurfaceContractEntry[];
    fallbackChain: string[];
    sourceHeads: Array<SourceHeadEntry & { id: string }>;
    defaultFocus?: WorkspaceDefaultFocus;
    artifactRows: Array<CatalogRow & { id: string }>;
    artifactPrimaryFocus: Record<string, FocusPointer>;
    artifactHandoffs: Record<string, OperatorAlertLink[]>;
    artifactPayloads: Record<string, ArtifactPayloadEntry>;
    raw: DashboardSnapshot;
  };
}
