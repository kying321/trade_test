import type { ReactNode } from 'react';

type AppShellProps = {
  topbar: ReactNode;
  sidebar: ReactNode;
  header: ReactNode;
  notice?: ReactNode;
  inspector?: ReactNode;
  children: ReactNode;
  sidebarCollapsed?: boolean;
};

export function AppShell({ topbar, sidebar, header, notice, inspector, children, sidebarCollapsed = false }: AppShellProps) {
  return (
    <div className="app-shell app-shell-v1">
      <a className="skip-link" href="#app-main-content">跳到主内容</a>
      <header className="global-topbar">{topbar}</header>
      <div className="app-shell-grid" data-sidebar-collapsed={sidebarCollapsed ? 'true' : 'false'}>
        <aside className={`context-sidebar ${sidebarCollapsed ? 'collapsed' : ''}`.trim()}>{sidebar}</aside>
        <main className="main-stage app-shell-stage" id="app-main-content">
          {header}
          {notice}
          <div className="stage-body">{children}</div>
        </main>
        <aside className="inspector-rail">{inspector}</aside>
      </div>
    </div>
  );
}
