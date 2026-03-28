import { cloneElement, isValidElement, useEffect, useMemo, useState, type ReactElement, type ReactNode } from 'react';
import { useShellBreakpoint, type ShellTier } from '../hooks/use-shell-breakpoint';

type InspectorMode = 'rail' | 'drawer' | 'inline';

type InspectorPresentationProps = {
  mode?: InspectorMode;
  collapsed?: boolean;
  shellTier?: ShellTier;
};

type AppShellProps = {
  topbar: ReactNode;
  sidebar: ReactNode;
  header: ReactNode;
  notice?: ReactNode;
  inspector?: ReactNode;
  children: ReactNode;
  sidebarCollapsed?: boolean;
};

function enhanceInspectorNode(inspector: ReactNode, mode: InspectorMode, collapsed: boolean, shellTier: ShellTier): ReactNode {
  if (!isValidElement(inspector) || typeof inspector.type === 'string') return inspector;
  return cloneElement(inspector as ReactElement<InspectorPresentationProps>, {
    mode,
    collapsed,
    shellTier,
  });
}

export function AppShell({ topbar, sidebar, header, notice, inspector, children, sidebarCollapsed = false }: AppShellProps) {
  const shellTier = useShellBreakpoint();
  const inspectorMode: InspectorMode = shellTier.tier === 'm' ? 'drawer' : shellTier.tier === 's' ? 'inline' : 'rail';
  const [drawerExpanded, setDrawerExpanded] = useState(false);

  useEffect(() => {
    if (inspectorMode !== 'drawer') {
      setDrawerExpanded(false);
    }
  }, [inspectorMode]);

  const shouldEmbedInspector = Boolean(inspector) && inspectorMode !== 'rail';
  const shouldRenderRail = Boolean(inspector) && inspectorMode === 'rail';
  const inspectorCollapsed = inspectorMode === 'drawer' ? !drawerExpanded : false;

  const inspectorNode = useMemo(
    () => enhanceInspectorNode(inspector, inspectorMode, inspectorCollapsed, shellTier.tier),
    [inspector, inspectorCollapsed, inspectorMode, shellTier.tier],
  );

  return (
    <div className="app-shell app-shell-v1" data-shell-tier={shellTier.tier}>
      <a className="skip-link" href="#app-main-content">跳到主内容</a>
      <header className="global-topbar">{topbar}</header>
      <div className="app-shell-grid" data-shell-tier={shellTier.tier} data-sidebar-collapsed={sidebarCollapsed ? 'true' : 'false'}>
        <aside className={`context-sidebar ${sidebarCollapsed ? 'collapsed' : ''}`.trim()}>{sidebar}</aside>
        <main className="main-stage app-shell-stage" id="app-main-content">
          {header}
          {notice}
          {shouldEmbedInspector ? (
            <section
              className="inspector-embedded"
              data-mode={inspectorMode}
              data-collapsed={inspectorCollapsed ? 'true' : 'false'}
            >
              {inspectorMode === 'drawer' ? (
                <button
                  type="button"
                  className="inspector-drawer-toggle"
                  aria-expanded={drawerExpanded}
                  aria-controls="inspector-drawer-panel"
                  aria-label={drawerExpanded ? '收起对象检查器' : '展开对象检查器'}
                  onClick={() => setDrawerExpanded((open) => !open)}
                >
                  {drawerExpanded ? '收起对象检查器' : '展开对象检查器'}
                </button>
              ) : null}
              <div id="inspector-drawer-panel" hidden={inspectorCollapsed}>
                {inspectorNode}
              </div>
            </section>
          ) : null}
          <div className="stage-body">{children}</div>
        </main>
        {shouldRenderRail ? <aside className="inspector-rail">{inspectorNode}</aside> : null}
      </div>
    </div>
  );
}
