import { Link, NavLink } from 'react-router-dom';
import { ClampText } from '../components/ui-kit';
import type { NavItem } from '../pages/page-types';

type PageSectionNavItem = {
  id: string;
  label: string;
  to: string;
  level: 1 | 2;
  groupId?: string;
  groupLabel?: string;
};

type ContextSidebarProps = {
  title: string;
  subtitle: string;
  description?: string;
  items: NavItem[];
  pageSections?: PageSectionNavItem[];
  activeSectionId?: string;
  collapsed: boolean;
  onToggleCollapse: () => void;
  canCollapse?: boolean;
  utilityLinks?: Array<{ label: string; to: string }>;
};

export function ContextSidebar({
  title,
  subtitle,
  description = '参考 Bloomberg / Aladdin 的物理隔离方式，严格源头所有，只读优先。',
  items,
  pageSections = [],
  activeSectionId,
  collapsed,
  onToggleCollapse,
  canCollapse = true,
  utilityLinks = [],
}: ContextSidebarProps) {
  const toggleLabel = collapsed ? '展开侧边导航' : '收起侧边导航';
  const sectionGroups = pageSections.reduce<Array<{ id: string; label: string; items: PageSectionNavItem[] }>>((groups, item) => {
    if (!item.groupId || !item.groupLabel) return groups;
    const existing = groups.find((group) => group.id === item.groupId);
    if (existing) {
      existing.items.push(item);
      return groups;
    }
    groups.push({
      id: item.groupId,
      label: item.groupLabel,
      items: [item],
    });
    return groups;
  }, []);
  const useGroupedSections = sectionGroups.length >= 2;

  return (
    <div className={`context-sidebar-inner ${collapsed ? 'collapsed' : ''}`.trim()} data-collapsed={collapsed ? 'true' : 'false'}>
      {canCollapse ? (
        <button
          type="button"
          className={`sidebar-collapse-toggle ${collapsed ? 'collapsed' : ''}`.trim()}
          aria-label={toggleLabel}
          title={toggleLabel}
          onClick={onToggleCollapse}
        >
          <span className="sidebar-collapse-glyph" aria-hidden="true">{collapsed ? '›' : '‹'}</span>
          {!collapsed ? <span className="sidebar-collapse-copy">回缩导航</span> : null}
        </button>
      ) : null}
      {!collapsed ? (
        <>
          <section className="sidebar-section sidebar-section-brand" aria-label="sidebar-brand-context">
            <div className="brand-block">
              <p className="brand-kicker">
                <ClampText raw={title} expandable={false}>{title}</ClampText>
              </p>
              <h1>
                <ClampText raw={subtitle} expandable={false}>{subtitle}</ClampText>
              </h1>
              <p>
                <ClampText raw={description} expandable={false}>
                  {description}
                </ClampText>
              </p>
            </div>
          </section>
          <section className="sidebar-section sidebar-section-nav" aria-label="sidebar-primary-nav">
            <nav className="nav-stack" aria-label="context-nav">
              {items.map((item) => (
                <NavLink
                  key={item.id}
                  aria-label={item.description ? `${item.label} ${item.description}` : item.label}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`.trim()}
                  to={item.to}
                >
                  <span className="nav-link-copy">
                    <strong>
                      <ClampText raw={item.label} expandable={false}>{item.label}</ClampText>
                    </strong>
                    {item.description ? (
                      <small>
                        <ClampText raw={item.description} expandable={false}>{item.description}</ClampText>
                      </small>
                    ) : null}
                  </span>
                </NavLink>
              ))}
            </nav>
          </section>
          {pageSections.length >= 3 ? (
            <section className="sidebar-section sidebar-section-toc" aria-label="sidebar-page-sections">
              <nav className="page-sections-nav" aria-label="page-sections-nav">
                <div className="context-subnav-head">
                  <span className="context-subnav-kicker">长页跳转</span>
                  <strong>阶段内目录</strong>
                </div>
                {useGroupedSections ? (
                  <div className="page-sections-groups">
                    {sectionGroups.map((group) => {
                      const groupActive = group.items.some((item) => item.id === activeSectionId);
                      return (
                        <section className={`page-sections-group ${groupActive ? 'active' : ''}`.trim()} key={group.id}>
                          <div className="page-sections-group-head">
                            <strong>{group.label}</strong>
                            <span>{group.items.length}</span>
                          </div>
                          <div className="page-sections-list">
                            {group.items.map((item) => (
                              <Link
                                key={item.id}
                                className={`page-section-link level-${item.level} ${activeSectionId === item.id ? 'active' : ''}`.trim()}
                                to={item.to}
                              >
                                <ClampText raw={item.label} expandable={false}>{item.label}</ClampText>
                              </Link>
                            ))}
                          </div>
                        </section>
                      );
                    })}
                  </div>
                ) : (
                  <div className="page-sections-list">
                    {pageSections.map((item) => (
                      <Link
                        key={item.id}
                        className={`page-section-link level-${item.level} ${activeSectionId === item.id ? 'active' : ''}`.trim()}
                        to={item.to}
                      >
                        <ClampText raw={item.label} expandable={false}>{item.label}</ClampText>
                      </Link>
                    ))}
                  </div>
                )}
              </nav>
            </section>
          ) : null}
          <section className="sidebar-section sidebar-section-utilities" aria-label="sidebar-utilities">
            <div className="rail-footnote">
              {utilityLinks.map((item) => (
                <Link key={item.to} to={item.to}>{item.label}</Link>
              ))}
              <a href="/data/fenlie_dashboard_snapshot.json" target="_blank" rel="noreferrer">公开快照</a>
              <a href="/operator_task_visual_panel.html" target="_blank" rel="noreferrer">运维面板</a>
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}
