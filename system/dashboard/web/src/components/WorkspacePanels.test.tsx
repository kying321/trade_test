import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import { WorkspacePanels } from './WorkspacePanels';

function mockWindowWidth(width: number) {
  Object.defineProperty(window, 'innerWidth', {
    configurable: true,
    writable: true,
    value: width,
  });
}

function createModel(): TerminalReadModel {
  return {
    surface: {} as TerminalReadModel['surface'],
    meta: {} as TerminalReadModel['meta'],
    uiRoutes: {} as TerminalReadModel['uiRoutes'],
    experienceContract: {} as TerminalReadModel['experienceContract'],
    navigation: {
      terminalPrimaryArtifact: {},
      terminalHandoffs: {},
    },
    view: {
      workspace: {
        artifacts: {
          list: { titleField: { showRaw: false }, pathField: { showRaw: false } },
          inspector: { titleField: { showRaw: false }, pathField: { showRaw: false } },
          metaFields: [],
          topLevelKeyField: { showRaw: false },
          payloadField: { kind: 'text', showRaw: false },
        },
        contracts: {
          surfaceAssertionFields: [],
          surfacePublishFields: [],
          interfaceColumns: [],
          sourceHeadSummaryFields: {
            label: { showRaw: false },
            summary: { showRaw: false },
          },
          sourceHeadDetailFields: [],
        },
      },
    } as unknown as TerminalReadModel['view'],
    orchestration: {
      alerts: [],
      topMetrics: [],
      freshness: [],
      guards: [],
      actionLog: [],
      checklist: [],
    },
    dataRegime: {
      sourceConfidence: [],
      microCapture: [],
      regimeSummary: [],
      regimeMatrix: [],
    },
    signalRisk: {
      laneCards: [],
      focusSlots: [],
      gateScores: [],
      controlChain: [],
      actionQueue: [],
      repairPlan: [],
      optimizationBacklog: [],
    },
    labReview: {
      researchHeads: [],
      backtests: [],
      comparisonRows: [],
      manifests: [],
      candidateRows: [],
      stressSummary: [],
    },
    feedback: {
      projection: null,
      domainGroups: [],
    },
    workspace: {
      domains: [],
      interfaces: [],
      publicTopology: [],
      publicAcceptance: {
        summary: {
          artifact_id: 'dashboard_public_acceptance',
          status: 'ok',
          generated_at_utc: '2026-04-08T00:00:00Z',
        } as unknown as TerminalReadModel['workspace']['publicAcceptance']['summary'],
        checks: [
          {
            id: 'workspace_routes_smoke',
            label: '工作区五页面烟测',
            status: 'ok',
            generated_at_utc: '2026-04-08T00:00:00Z',
          },
        ],
        subcommands: [
          {
            id: 'workspace_routes_smoke',
            label: '工作区五页面子命令',
            returncode: 0,
            cmd: 'python3 smoke',
            stdout: 'ok',
            stderr: '',
          },
        ],
      },
      researchAuditCases: [],
      surfaceContracts: [
        {
          id: 'public',
          label: '公开面',
          availability: 'active',
          redaction_level: 'public_summary',
        },
      ],
      fallbackChain: ['/workspace/raw'],
      sourceHeads: [
        {
          id: 'operator_panel',
          label: '操作面板',
          summary: 'operator panel',
          status: 'ok',
          path: '/tmp/operator_panel.json',
        },
      ],
      defaultFocus: {
        artifact: 'price_action_breakout_pullback',
        group: 'research_mainline',
        search_scope: 'title',
      },
      artifactRows: [
        {
          id: 'price_action_breakout_pullback',
          label: 'ETH 主线',
          path: '/review/eth.json',
          category: 'research',
          artifact_group: 'research_mainline',
          research_decision: 'selected',
          status: 'ok',
        },
      ],
      artifactPrimaryFocus: {},
      artifactHandoffs: {},
      artifactPayloads: {},
      raw: { alpha: 1, beta: 'two' } as unknown as TerminalReadModel['workspace']['raw'],
    },
  };
}

describe('WorkspacePanels', () => {
  beforeEach(() => {
    mockWindowWidth(800);
    if (!HTMLElement.prototype.scrollIntoView) {
      HTMLElement.prototype.scrollIntoView = vi.fn();
    }
  });

  it('在手机端为 artifacts 页提供页内 jumpbar 与可折叠筛选区', () => {
    render(
      <MemoryRouter initialEntries={['/workspace/artifacts']}>
        <WorkspacePanels model={createModel()} section="artifacts" />
      </MemoryRouter>,
    );

    const quickNav = screen.getByRole('navigation', { name: 'workspace-mobile-quick-nav' });
    expect(within(quickNav).getByRole('link', { name: '当前焦点' })).toBeTruthy();
    expect(within(quickNav).getByRole('link', { name: '下一步动作' })).toBeTruthy();
    expect(within(quickNav).getByRole('link', { name: '目标池' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '展开筛选与范围' })).toBeTruthy();
    expect(screen.getByLabelText('workspace-rhythm-state').getAttribute('hidden')).not.toBeNull();

    fireEvent.click(screen.getByRole('button', { name: '展开筛选与范围' }));
    expect(screen.getByRole('button', { name: '收起筛选与范围' })).toBeTruthy();
    expect(screen.getByLabelText('workspace-rhythm-state').getAttribute('hidden')).toBeNull();
    expect(screen.getByPlaceholderText('搜索工件 / 路径 / 状态代码')).toBeTruthy();
  });

  it('在手机端把 contracts 验收状态区压成状态轨并保留页内 jumpbar', () => {
    render(
      <MemoryRouter initialEntries={['/workspace/contracts']}>
        <WorkspacePanels model={createModel()} section="contracts" />
      </MemoryRouter>,
    );

    const quickNav = screen.getByRole('navigation', { name: 'workspace-mobile-quick-nav' });
    expect(within(quickNav).getByRole('link', { name: '验收总览' })).toBeTruthy();
    expect(within(quickNav).getByRole('link', { name: '子链状态' })).toBeTruthy();
    expect(screen.getByTestId('contracts-acceptance-status-strip').getAttribute('data-mobile-layout')).toBe('true');
  });

  it('raw 页在手机端提供 jumpbar 与 JSON 阅读工具栏', () => {
    render(
      <MemoryRouter initialEntries={['/workspace/raw']}>
        <WorkspacePanels model={createModel()} section="raw" />
      </MemoryRouter>,
    );

    const quickNav = screen.getByRole('navigation', { name: 'workspace-mobile-quick-nav' });
    expect(within(quickNav).getByRole('link', { name: '原始摘要' })).toBeTruthy();
    expect(within(quickNav).getByRole('link', { name: '全量原始快照' })).toBeTruthy();
    expect(screen.getAllByRole('button', { name: '复制 JSON' }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('textbox', { name: '搜索 JSON 键 / 值' }).length).toBeGreaterThan(0);
  });
});
