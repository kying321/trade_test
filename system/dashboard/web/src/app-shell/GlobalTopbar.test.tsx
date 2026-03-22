import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { GlobalTopbar, type GlobalTopbarDomainContext, type ThemePreference } from './GlobalTopbar';

function renderTopbar({
  currentPath = '/overview',
  themePreference = 'system',
}: {
  currentPath?: string;
  themePreference?: ThemePreference;
} = {}) {
  const domainContext: GlobalTopbarDomainContext = {
    kicker: '研究域',
    title: 'ETHUSDT 15m 主线',
    detail: '聚焦 breakout-pullback 与退出/风险验证',
    tone: 'research',
  };

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
            { label: '公开面', value: 'fuuu.fun' },
            { label: '快照', value: '只读轮询' },
          ],
        }}
        themePreference={themePreference}
        resolvedTheme="dark"
        onThemeChange={vi.fn()}
        domainContext={domainContext}
      />,
    </MemoryRouter>,
  );
}

describe('GlobalTopbar', () => {
  it('将品牌、全局控制和域上下文拆成独立层级区域', () => {
    renderTopbar();

    const brandRegion = screen.getByRole('region', { name: 'global-brand-context' });
    expect(within(brandRegion).getByText('Fenlie / 控制台')).toBeTruthy();
    expect(within(brandRegion).getByText('只读操作与研究面')).toBeTruthy();

    const controlDeck = screen.getByRole('region', { name: 'global-control-deck' });
    expect(within(controlDeck).getByRole('group', { name: 'theme-switcher' })).toBeTruthy();
    expect(within(controlDeck).getByText('变更级别：RESEARCH_ONLY')).toBeTruthy();
    expect(within(controlDeck).getByText('公开面：fuuu.fun')).toBeTruthy();

    const domainRegion = screen.getByRole('region', { name: 'global-domain-context' });
    expect(within(domainRegion).getByText('研究域')).toBeTruthy();
    expect(within(domainRegion).getByText('ETHUSDT 15m 主线')).toBeTruthy();
    expect(within(domainRegion).getByText('聚焦 breakout-pullback 与退出/风险验证')).toBeTruthy();
  });

  it('为主导航保留当前激活态并暴露为独立导航区', () => {
    renderTopbar({ currentPath: '/workspace/artifacts', themePreference: 'dark' });

    const nav = screen.getByRole('navigation', { name: 'primary-domains' });
    expect(within(nav).getByRole('link', { name: '总览' })).toBeTruthy();
    const activeLink = within(nav).getByRole('link', { name: '研究工作区' });
    expect(activeLink.className).toContain('active');
  });
});
