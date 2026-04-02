import { Link, NavLink } from 'react-router-dom';
import { ClampText } from '../components/ui-kit';
type SidebarNavItem = {
  id: string;
  label: string;
  to: string;
};

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
  items: SidebarNavItem[];
  pageSections?: PageSectionNavItem[];
  activeSectionId?: string;
  collapsed: boolean;
  onToggleCollapse: () => void;
  canCollapse?: boolean;
  utilityLinks?: Array<{ label: string; to: string }>;
};

type SidebarTaskNavProps = {
  items: SidebarNavItem[];
  pageSections: PageSectionNavItem[];
  activeSectionId?: string;
};

function SidebarDomainIdentity({ title, subtitle }: Pick<ContextSidebarProps, 'title' | 'subtitle'>) {
  return (
    <section className="sidebar-section sidebar-domain-identity" aria-label="sidebar-domain-identity">
      <div className="brand-block">
        <p className="brand-kicker">
          <ClampText raw={title} expandable={false}>{title}</ClampText>
        </p>
        <h1>
          <ClampText raw={subtitle} expandable={false}>{subtitle}</ClampText>
        </h1>
      </div>
    </section>
  );
}

function SidebarTaskNav({ items, pageSections, activeSectionId }: SidebarTaskNavProps) {
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
  const showNav = items.length > 0;
  const showToc = pageSections.length >= 3;

  if (!showNav && !showToc) return null;

  return (
    <section className="sidebar-section sidebar-task-nav" aria-label="sidebar-task-nav">
      {showNav ? (
        <nav className="nav-stack" aria-label="context-nav">
          {items.map((item) => (
            <NavLink
              key={item.id}
              aria-label={item.label}
              className={({ isActive }) => `nav-link control-entity-row ${isActive ? 'active' : ''}`.trim()}
              to={item.to}
            >
              <span className="nav-link-copy">
                <strong>
                  <ClampText raw={item.label} expandable={false}>{item.label}</ClampText>
                </strong>
              </span>
            </NavLink>
          ))}
        </nav>
      ) : null}
      {showToc ? (
        <section className="sidebar-page-toc" aria-label="sidebar-page-sections">
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
                            aria-current={activeSectionId === item.id ? 'location' : undefined}
                            className={`page-section-link control-filter-chip level-${item.level} ${activeSectionId === item.id ? 'active' : ''}`.trim()}
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
                    aria-current={activeSectionId === item.id ? 'location' : undefined}
                    className={`page-section-link control-filter-chip level-${item.level} ${activeSectionId === item.id ? 'active' : ''}`.trim()}
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
    </section>
  );
}

function SidebarUtilities({ utilityLinks }: Pick<ContextSidebarProps, 'utilityLinks'>) {
  const normalizedLinks = utilityLinks.some((item) => item.to === '/cpa')
    ? utilityLinks
    : [...utilityLinks, { label: 'CPA 管理 / Auth & Quota', to: '/cpa' }];

  return (
    <section className="sidebar-section sidebar-utilities" aria-label="sidebar-utilities">
      <nav className="rail-footnote" aria-label="sidebar-utility-nav">
        {normalizedLinks.map((item) => (
          <NavLink
            key={`${item.to}-${item.label}`}
            className={({ isActive }) => `utility-link ${item.to === '/cpa' ? 'utility-link-emphasis' : ''} ${isActive ? 'active' : ''}`.trim()}
            to={item.to}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </section>
  );
}

export function ContextSidebar({
  title,
  subtitle,
  items,
  pageSections = [],
  activeSectionId,
  collapsed,
  onToggleCollapse,
  canCollapse = true,
  utilityLinks = [],
}: ContextSidebarProps) {
  const toggleLabel = collapsed ? '展开侧边导航' : '收起侧边导航';

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
          <SidebarDomainIdentity title={title} subtitle={subtitle} />
          <SidebarTaskNav items={items} pageSections={pageSections} activeSectionId={activeSectionId} />
          <SidebarUtilities utilityLinks={utilityLinks} />
        </>
      ) : null}
    </div>
  );
}
