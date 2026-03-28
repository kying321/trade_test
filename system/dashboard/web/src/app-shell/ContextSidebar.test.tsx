import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { ContextSidebar } from './ContextSidebar';

function renderSidebar({ collapsed = false }: { collapsed?: boolean } = {}) {
  const onToggleCollapse = vi.fn();
  const view = render(
    <MemoryRouter>
      <ContextSidebar
        title="研究工作区"
        subtitle="Fenlie Research Workspace"
        description="把研究主线、阶段导航和只读回退拆成明确分区。"
        collapsed={collapsed}
        onToggleCollapse={onToggleCollapse}
        items={[
          { id: 'artifacts', label: '工件池', to: '/workspace/artifacts', description: '主线与归档入口' },
          { id: 'alignment', label: '对齐页', to: '/workspace/alignment?view=internal', description: '内部方向对齐投射' },
        ]}
        pageSections={[
          { id: 'summary', label: '当前摘要', to: '/workspace/artifacts?page_section=summary', level: 1, groupId: 'main', groupLabel: '研究主线' },
          { id: 'events', label: '事件列表', to: '/workspace/artifacts?page_section=events', level: 2, groupId: 'main', groupLabel: '研究主线' },
          { id: 'actions', label: '动作栈', to: '/workspace/artifacts?page_section=actions', level: 1, groupId: 'ops', groupLabel: '动作追踪' },
        ]}
        utilityLinks={[{ label: '搜索 / Search', to: '/search' }]}
        activeSectionId="summary"
      />
    </MemoryRouter>,
  );

  return { ...view, onToggleCollapse };
}

describe('ContextSidebar', () => {
  it('把品牌、主导航、长页目录和只读入口拆成独立区块', () => {
    renderSidebar();

    const brandRegion = screen.getByRole('region', { name: 'sidebar-brand-context' });
    expect(within(brandRegion).getByText('研究工作区')).toBeTruthy();
    expect(within(brandRegion).getByText('Fenlie Research Workspace')).toBeTruthy();

    const nav = screen.getByRole('navigation', { name: 'context-nav' });
    expect(within(nav).getByRole('link', { name: '工件池 主线与归档入口' })).toBeTruthy();
    expect(within(nav).getByRole('link', { name: '对齐页 内部方向对齐投射' })).toBeTruthy();

    const toc = screen.getByRole('navigation', { name: 'page-sections-nav' });
    expect(within(toc).getByText('阶段内目录')).toBeTruthy();
    expect(within(toc).getByText('研究主线')).toBeTruthy();
    expect(within(toc).getByText('动作追踪')).toBeTruthy();

    const utilities = screen.getByRole('region', { name: 'sidebar-utilities' });
    expect(within(utilities).getByRole('link', { name: '搜索 / Search' })).toBeTruthy();
    expect(within(utilities).getByRole('link', { name: '公开快照' })).toBeTruthy();
    expect(within(utilities).getByRole('link', { name: '运维面板' })).toBeTruthy();
  });

  it('收缩到 32px 竖条时只保留展开把手', () => {
    const { container, onToggleCollapse } = renderSidebar({ collapsed: true });

    const toggle = screen.getByRole('button', { name: '展开侧边导航' });
    fireEvent.click(toggle);
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);

    expect(container.querySelector('.context-sidebar-inner')?.getAttribute('data-collapsed')).toBe('true');
    expect(screen.queryByRole('navigation', { name: 'context-nav' })).toBeNull();
    expect(screen.queryByRole('navigation', { name: 'page-sections-nav' })).toBeNull();
    expect(screen.queryByRole('region', { name: 'sidebar-utilities' })).toBeNull();
  });
});
