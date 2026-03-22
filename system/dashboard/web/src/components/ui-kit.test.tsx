import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Badge, ClampText, DenseTable, MetricStrip, PathText } from './ui-kit';

function mockClampOverflow({ overflows }: { overflows: boolean }) {
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function scrollHeight(this: HTMLElement) {
    if (this.classList.contains('clamp-copy')) return overflows ? 64 : 32;
    return 32;
  });
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function clientHeight(this: HTMLElement) {
    if (this.classList.contains('clamp-copy')) return 32;
    return 32;
  });
  vi.spyOn(HTMLElement.prototype, 'scrollWidth', 'get').mockImplementation(function scrollWidth(this: HTMLElement) {
    if (this.classList.contains('clamp-copy')) return overflows ? 220 : 120;
    return 120;
  });
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function clientWidth(this: HTMLElement) {
    if (this.classList.contains('clamp-copy')) return 120;
    return 120;
  });
}

describe('ui-kit source-owned 审计显示', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('在审计模式下为指标值显示 raw 副标题', () => {
    render(
      <MetricStrip
        items={[
          {
            id: 'policy',
            label: '策略',
            value: 'shadow_policy_blocked',
          },
        ]}
        showRawValues
      />,
    );

    expect(screen.getByText('影子策略阻断')).toBeTruthy();
    expect(screen.getByText('shadow_policy_blocked')).toBeTruthy();
  });

  it('仅在列声明 showRaw 时显示 raw 副标题', () => {
    const row = { decision: 'clear_guardian_review_blockers_before_promotion' };

    const { rerender } = render(
      <DenseTable
        columns={[{ key: 'decision', label: '决策' }]}
        rows={[row]}
      />,
    );

    expect(screen.getByText('晋级前先清除守门复核阻塞')).toBeTruthy();
    expect(screen.queryByText('clear_guardian_review_blockers_before_promotion')).toBeNull();

    rerender(
      <DenseTable
        columns={[{ key: 'decision', label: '决策', showRaw: true }]}
        rows={[row]}
      />,
    );

    expect(screen.getByText('clear_guardian_review_blockers_before_promotion')).toBeTruthy();
  });

  it('在路径审计模式下显示 compact 主路径和 raw 全路径', () => {
    render(
      <PathText
        value="/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260319T192500Z_operator_task_visual_panel.json"
        showRaw
      />,
    );

    expect(screen.getByText('…/system/output/review/20260319T192500Z_operator_task_visual_panel.json')).toBeTruthy();
    expect(screen.getByText('/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260319T192500Z_operator_task_visual_panel.json')).toBeTruthy();
  });

  it('仅在真实溢出时显示展开 / 收起切换', async () => {
    mockClampOverflow({ overflows: true });
    render(
      <ClampText maxLines={2} raw="shadow_policy_blocked">
        这是一个超长的共享文本摘要，用来验证默认两行截断后才出现展开按钮，而不是无条件常驻。
      </ClampText>,
    );

    const expand = await screen.findByRole('button', { name: '展开完整内容' });
    fireEvent.click(expand);

    expect(screen.getByRole('button', { name: '收起完整内容' })).toBeTruthy();
    expect(document.querySelector('.clamp-copy')?.className).not.toContain('is-clamped');
  });

  it('没有真实溢出时不显示切换按钮', () => {
    mockClampOverflow({ overflows: false });
    render(
      <ClampText maxLines={2}>
        短文本
      </ClampText>,
    );

    expect(screen.queryByRole('button', { name: '展开完整内容' })).toBeNull();
    expect(screen.queryByRole('button', { name: '收起完整内容' })).toBeNull();
  });

  it('布局变化后会重新测量并移除误判的展开按钮', async () => {
    let overflows = true;
    const resizeObserverState: { callback: ResizeObserverCallback | null } = { callback: null };

    vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function scrollHeight(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return overflows ? 64 : 32;
      return 32;
    });
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function clientHeight(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return 32;
      return 32;
    });
    vi.spyOn(HTMLElement.prototype, 'scrollWidth', 'get').mockImplementation(function scrollWidth(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return overflows ? 180 : 120;
      return 120;
    });
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function clientWidth(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return 120;
      return 120;
    });

    class MockResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        resizeObserverState.callback = callback;
      }
      observe() {}
      disconnect() {}
      unobserve() {}
    }

    vi.stubGlobal('ResizeObserver', MockResizeObserver);

    render(
      <ClampText maxLines={2} raw="研究地图">
        研究地图
      </ClampText>,
    );

    expect(screen.getByRole('button', { name: '展开完整内容' })).toBeTruthy();

    overflows = false;
    act(() => {
      resizeObserverState.callback?.([] as ResizeObserverEntry[], {} as ResizeObserver);
    });

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: '展开完整内容' })).toBeNull();
    });
  });

  it('badge 对 JSX 复合 children 不应插入数组逗号', () => {
    render(
      <Badge value="ok">
        视图：{'匹配'}
      </Badge>,
    );

    expect(screen.getByText('视图：匹配')).toBeTruthy();
    expect(screen.queryByText('视图：, 匹配')).toBeNull();
  });

  it('展开后的长文本在重新测量时不应闪退回收', async () => {
    const resizeObserverState: { callback: ResizeObserverCallback | null } = { callback: null };

    vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function scrollHeight(this: HTMLElement) {
      if (!this.classList.contains('clamp-copy')) return 32;
      return 64;
    });
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function clientHeight(this: HTMLElement) {
      if (!this.classList.contains('clamp-copy')) return 32;
      return this.classList.contains('is-clamped') ? 32 : 64;
    });
    vi.spyOn(HTMLElement.prototype, 'scrollWidth', 'get').mockImplementation(function scrollWidth(this: HTMLElement) {
      if (!this.classList.contains('clamp-copy')) return 120;
      return 220;
    });
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function clientWidth(this: HTMLElement) {
      if (!this.classList.contains('clamp-copy')) return 120;
      return this.classList.contains('is-clamped') ? 120 : 220;
    });

    class MockResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        resizeObserverState.callback = callback;
      }
      observe() {}
      disconnect() {}
      unobserve() {}
    }

    vi.stubGlobal('ResizeObserver', MockResizeObserver);

    render(
      <ClampText maxLines={2} raw="review/20260322T094636Z_operator_task_visual_panel.json">
        review/20260322T094636Z_operator_task_visual_panel.json
      </ClampText>,
    );

    fireEvent.click(screen.getByRole('button', { name: '展开完整内容' }));
    expect(document.querySelector('.clamp-text')?.getAttribute('data-expanded')).toBe('true');
    expect(screen.getByRole('button', { name: '收起完整内容' })).toBeTruthy();

    act(() => {
      resizeObserverState.callback?.([] as ResizeObserverEntry[], {} as ResizeObserver);
    });

    await waitFor(() => {
      expect(document.querySelector('.clamp-text')?.getAttribute('data-expanded')).toBe('true');
      expect(screen.getByRole('button', { name: '收起完整内容' })).toBeTruthy();
    });
  });
});
