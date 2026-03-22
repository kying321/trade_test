import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AppShell } from './AppShell';

describe('AppShell', () => {
  it('renders four-zone shell layout with topbar sidebar stage and inspector', () => {
    render(
      <AppShell
        topbar={<div>顶栏</div>}
        sidebar={<nav aria-label="context-nav">侧栏</nav>}
        header={<div>上下文头</div>}
        notice={<div>域提示</div>}
        inspector={<div>对象检查器</div>}
      >
        <div>主舞台内容</div>
      </AppShell>,
    );

    expect(screen.getByText('顶栏')).toBeTruthy();
    expect(screen.getByRole('navigation', { name: 'context-nav' })).toBeTruthy();
    expect(screen.getByText('上下文头')).toBeTruthy();
    expect(screen.getByText('域提示')).toBeTruthy();
    expect(screen.getByText('主舞台内容')).toBeTruthy();
    expect(screen.getByText('对象检查器')).toBeTruthy();
  });

  it('marks shell and sidebar as collapsed when the left rail is reduced to a 32px handle', () => {
    const { container } = render(
      <AppShell
        topbar={<div>顶栏</div>}
        sidebar={<nav aria-label="context-nav">侧栏</nav>}
        header={<div>上下文头</div>}
        inspector={<div>对象检查器</div>}
        sidebarCollapsed
      >
        <div>主舞台内容</div>
      </AppShell>,
    );

    expect(container.querySelector('.app-shell-grid')?.getAttribute('data-sidebar-collapsed')).toBe('true');
    expect(container.querySelector('.context-sidebar')?.className).toContain('collapsed');
  });
});
