import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { buildTerminalReadModel } from '../adapters/read-model';
import type { ResolvedTheme, ThemePreference } from '../app-shell/GlobalTopbar';
import { useTerminalStore } from '../store/use-terminal-store';
import type { DashboardSnapshot, LoadedSurface } from '../types/contracts';

let mockedThemePreference: ThemePreference = 'system';
let mockedResolvedTheme: ResolvedTheme = 'dark';
const mockSetThemePreference = vi.fn();
const DEFAULT_SET_SURFACE = useTerminalStore.getState().setSurface;

vi.mock('../hooks/use-ui-theme', () => ({
  useUiTheme: () => ({
    preference: mockedThemePreference,
    resolvedTheme: mockedResolvedTheme,
    setPreference: mockSetThemePreference,
  }),
}));

const snapshot: DashboardSnapshot = {
  status: 'ok',
  change_class: 'RESEARCH_ONLY',
  meta: {
    generated_at_utc: '2026-03-28T03:00:00Z',
    change_class: 'RESEARCH_ONLY',
  },
  ui_routes: {
    terminal_public: '#/terminal/public',
    terminal_internal: '#/terminal/internal',
    workspace_artifacts: '#/workspace/artifacts',
    operator_panel: '/operator_task_visual_panel.html',
  },
  experience_contract: {
    mode: 'read_only_snapshot',
    live_path_touched: false,
    fallback_chain: [],
  },
  source_heads: {
    hold_selection_handoff: {
      label: 'hold selection handoff',
      status: 'ok',
      summary: 'hold16 baseline',
      path: 'review/latest_hold_selection.json',
    },
  },
  domain_summary: [],
  interface_catalog: [],
  public_topology: [],
  surface_contracts: {},
  backtest_artifacts: [],
  artifact_payloads: {},
  catalog: [
    {
      id: 'hold_selection_handoff',
      payload_key: 'hold_selection_handoff',
      artifact_group: 'research_hold_transfer',
      category: 'research',
      label: 'hold_selection_handoff',
      status: 'ok',
      path: 'review/latest_hold_selection.json',
    },
  ],
  workspace_default_focus: {
    artifact: 'hold_selection_handoff',
    group: 'research_hold_transfer',
    search_scope: 'title',
  },
};

function installPassiveStore(snapshotValue: DashboardSnapshot = snapshot) {
  const loaded: LoadedSurface = {
    snapshot: snapshotValue,
    requestedSurface: 'public',
    effectiveSurface: 'public',
    snapshotPath: '/data/fenlie_dashboard_snapshot.json',
    warnings: [],
  };
  useTerminalStore.setState({
    requestedSurface: 'public',
    effectiveSurface: 'public',
    snapshotPath: loaded.snapshotPath,
    snapshot: loaded.snapshot,
    model: buildTerminalReadModel(loaded),
    loading: false,
    error: '',
    refresh: vi.fn(async () => undefined),
    setSurface: DEFAULT_SET_SURFACE,
  });
}

async function renderApp() {
  let view: ReturnType<typeof render> | null = null;
  await act(async () => {
    view = render(<App />);
  });
  if (!view) throw new Error('render failure');
  return view;
}

async function openSearchAndQuery(value: string) {
  await act(async () => {
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
  });
  await act(async () => {
    fireEvent.change(screen.getByRole('textbox'), { target: { value } });
  });
}

beforeEach(() => {
  mockedThemePreference = 'system';
  mockedResolvedTheme = 'dark';
  mockSetThemePreference.mockReset();
  window.location.hash = '#/overview';
  installPassiveStore();
});

afterEach(() => {
  cleanup();
});

describe('global search ui', () => {
  it('topbar 仅暴露全局契约，不再混入页面职责芯片', async () => {
    await renderApp();
    const topbar = document.querySelector<HTMLElement>('header.global-topbar');
    expect(topbar).toBeTruthy();
    if (!topbar) throw new Error('global topbar not found');
    const summaryStrip = within(topbar).getByLabelText('global-summary') as HTMLElement;

    expect(within(topbar).queryByText('职责')).toBeNull();
    expect(within(topbar).queryByText(/职责/)).toBeNull();
    expect(within(topbar).getByRole('navigation', { name: 'primary-domains' })).toBeTruthy();
    expect(within(topbar).getByRole('button', { name: '打开全局搜索' })).toBeTruthy();
    expect(summaryStrip.textContent || '').toMatch(/快照|只读|generated/i);
  });

  it('toggles overlay visibility when the topbar search trigger is clicked twice', async () => {
    await renderApp();

    const trigger = screen.getByRole('button', { name: '打开全局搜索' });
    await act(async () => {
      fireEvent.click(trigger);
    });
    expect(screen.getByRole('dialog', { name: '全局搜索' })).toBeTruthy();

    await act(async () => {
      fireEvent.click(trigger);
    });

    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: '全局搜索' })).toBeNull();
    });
  });

  it('toggles overlay visibility when Cmd/Ctrl+K is pressed twice', async () => {
    await renderApp();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'k', metaKey: true });
    });
    expect(screen.getByRole('dialog', { name: '全局搜索' })).toBeTruthy();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'k', metaKey: true });
    });

    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: '全局搜索' })).toBeNull();
    });
  });

  it('closes overlay on Cmd/Ctrl+K even when the search input is focused', async () => {
    await renderApp();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'k', metaKey: true });
    });

    const input = screen.getByRole('textbox');
    await act(async () => {
      input.focus();
      fireEvent.keyDown(input, { key: 'k', metaKey: true });
    });

    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: '全局搜索' })).toBeNull();
    });
  });

  it('supports Cmd/Ctrl+K, ArrowDown and Enter for result navigation', async () => {
    await renderApp();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'k', metaKey: true });
    });
    expect(screen.getByRole('dialog', { name: '全局搜索' })).toBeTruthy();

    await act(async () => {
      fireEvent.change(screen.getByRole('textbox'), { target: { value: '国内商品推理线' } });
    });
    expect(screen.getAllByText('国内商品推理线').length).toBeGreaterThan(0);

    await act(async () => {
      fireEvent.keyDown(window, { key: 'ArrowDown' });
    });
    await act(async () => {});
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Enter' });
    });

    await waitFor(() => {
      expect(window.location.hash).toContain('/terminal/public');
      expect(window.location.hash).toContain('anchor=terminal-commodity-reasoning');
    });
  });

  it('prefills /search route and jumps from result list', async () => {
    window.location.hash = '#/search?q=hold_selection_handoff&scope=artifact';
    await renderApp();

    const input = screen.getByRole('textbox');
    expect((input as HTMLInputElement).value).toBe('hold_selection_handoff');
    const result = screen.getAllByRole('button', { name: /持有选择主头/i })[0];
    await act(async () => {
      fireEvent.click(result);
    });

    await waitFor(() => {
      expect(window.location.hash).toContain('/workspace/artifacts');
      expect(window.location.hash).toContain('artifact=hold_selection_handoff');
    });
  });

  it('separates entity rows from action links in overlay search results', async () => {
    await renderApp();
    await openSearchAndQuery('hold_selection_handoff');

    const entityRow = screen.getAllByRole('button', { name: /持有选择主头/i })[0];
    expect(entityRow.className).toContain('control-entity-row');
    const viewAll = screen.getByRole('link', { name: /查看全部结果/i });
    expect(viewAll.className).toContain('control-action-link');
    expect(screen.getAllByText(/标题命中|说明命中|路径命中|关键词命中/).length).toBeGreaterThan(0);
    expect(document.querySelector('.search-result-button .clamp-toggle')).toBeNull();
  });

  it('keeps scope switching explicit between all/module/route/artifact', async () => {
    window.location.hash = '#/search';
    await renderApp();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '模块' }));
    });
    await waitFor(() => {
      expect(window.location.hash).toContain('scope=module');
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '路由' }));
    });
    await waitFor(() => {
      expect(window.location.hash).toContain('scope=route');
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '工件' }));
    });
    await waitFor(() => {
      expect(window.location.hash).toContain('scope=artifact');
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '全部' }));
    });
    await waitFor(() => {
      expect(window.location.hash).toContain('/search');
      expect(window.location.hash).not.toContain('scope=');
    });
  });

  it('preserves deep-link params for route/module/artifact search navigation', async () => {
    await renderApp();

    await openSearchAndQuery('总览');
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '路由' }));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /总览/i }));
    });
    await waitFor(() => {
      expect(window.location.hash).toContain('/overview');
      expect(window.location.hash).not.toContain('anchor=');
    });

    await openSearchAndQuery('国内商品推理线');
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '模块' }));
    });
    await act(async () => {
      fireEvent.click(screen.getAllByRole('button', { name: /国内商品推理线/i })[0]);
    });
    await waitFor(() => {
      expect(window.location.hash).toContain('anchor=');
      expect(window.location.hash).toContain('/overview');
    });

    await openSearchAndQuery('hold_selection_handoff');
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '工件' }));
    });
    await act(async () => {
      fireEvent.click(screen.getAllByRole('button', { name: /持有选择主头/i })[0]);
    });
    await waitFor(() => {
      expect(window.location.hash).toContain('/workspace/artifacts');
      expect(window.location.hash).toContain('artifact=hold_selection_handoff');
    });
  });
});
