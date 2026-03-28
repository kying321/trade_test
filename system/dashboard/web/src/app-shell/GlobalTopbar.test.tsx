import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { GlobalTopbar, type ThemePreference } from './GlobalTopbar';

function renderTopbar({
  currentPath = '/overview',
  themePreference = 'system',
}: {
  currentPath?: string;
  themePreference?: ThemePreference;
} = {}) {
  return render(
    <MemoryRouter>
      <GlobalTopbar
        currentPath={currentPath}
        primaryNav={[
          { id: 'overview', label: '总览', to: '/overview' },
          { id: 'workspace', label: '研究工作区', to: '/workspace/artifacts' },
        ]}
        globalSummary={{
          changeClass: 'RESEARCH_ONLY',
          chips: [
            { label: '模式', value: 'read_only_snapshot' },
            { label: '快照', value: 'generated:2026-03-28T03:00:00Z' },
          ],
        }}
        themePreference={themePreference}
        resolvedTheme="dark"
        onThemeChange={vi.fn()}
        onOpenSearch={vi.fn()}
      />,
    </MemoryRouter>,
  );
}

describe('GlobalTopbar', () => {
  it('仅承载全局职责并暴露稳定语义契约', () => {
    renderTopbar();

    expect(screen.queryByText('职责')).toBeNull();
    expect(screen.getByRole('navigation', { name: 'primary-domains' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '打开全局搜索' })).toBeTruthy();
    expect(within(screen.getByLabelText('global-summary')).getByText(/快照|只读|generated/i)).toBeTruthy();
  });

  it('将品牌、主域导航、全局工具与环境条拆成独立层级区域', () => {
    renderTopbar();

    const brandRegion = screen.getByRole('region', { name: 'global-brand-context' });
    expect(within(brandRegion).getByText('Fenlie / 控制台')).toBeTruthy();
    expect(within(brandRegion).getByText('只读操作与研究面')).toBeTruthy();

    const controlDeck = screen.getByRole('region', { name: 'global-tool-strip' });
    expect(within(controlDeck).getByRole('group', { name: 'theme-switcher' })).toBeTruthy();
    expect(within(controlDeck).getByRole('button', { name: '打开全局搜索' })).toBeTruthy();

    const summaryStrip = screen.getByLabelText('global-summary');
    expect(within(summaryStrip).getByText('模式：read_only_snapshot')).toBeTruthy();
    expect(within(summaryStrip).getByText('快照：generated:2026-03-28T03:00:00Z')).toBeTruthy();
    expect(within(summaryStrip).getByText('变更级别：RESEARCH_ONLY')).toBeTruthy();
  });

  it('为主导航保留当前激活态并暴露为独立导航区', () => {
    renderTopbar({ currentPath: '/workspace/artifacts', themePreference: 'dark' });

    const nav = screen.getByRole('navigation', { name: 'primary-domains' });
    expect(within(nav).getByRole('link', { name: '总览' })).toBeTruthy();
    const activeLink = within(nav).getByRole('link', { name: '研究工作区' });
    expect(activeLink.className).toContain('active');
  });
});
