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
        summary: null,
        checks: [],
        subcommands: [],
      },
      researchAuditCases: [],
      surfaceContracts: [],
      fallbackChain: [],
      sourceHeads: [],
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
      raw: {},
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
    expect(screen.queryByRole('textbox')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: '展开筛选与范围' }));
    expect(screen.getByRole('button', { name: '收起筛选与范围' })).toBeTruthy();
    expect(screen.getByRole('textbox')).toBeTruthy();
  });
});
