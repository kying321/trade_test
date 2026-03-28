import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';

export type ContextHeaderSectionItem =
  | {
      type: 'link';
      label: string;
      value: string;
      detail?: string;
      to: string;
      active?: boolean;
    }
  | {
      type: 'fact';
      label: string;
      value: string;
      detail?: string;
      tone?: 'neutral' | 'positive' | 'warning' | 'negative';
    };

export type ContextHeaderSection = {
  id: string;
  title: string;
  items: ContextHeaderSectionItem[];
};

type ContextHeaderRhythmSection = {
  title?: string;
  items: ContextHeaderSectionItem[];
};

type ContextHeaderProps = {
  title: string;
  subtitle: string;
  breadcrumbs?: Array<{ label: string; current?: boolean }>;
  actions?: ReactNode;
  sections?: ContextHeaderSection[];
  stateSection?: ContextHeaderRhythmSection;
  focusSection?: ContextHeaderRhythmSection;
  actionSection?: ContextHeaderRhythmSection;
  evidenceSection?: ContextHeaderRhythmSection;
  compact?: boolean;
};

function normalizeRhythmSections({
  stateSection,
  focusSection,
  actionSection,
  evidenceSection,
  sections,
}: {
  stateSection?: ContextHeaderRhythmSection;
  focusSection?: ContextHeaderRhythmSection;
  actionSection?: ContextHeaderRhythmSection;
  evidenceSection?: ContextHeaderRhythmSection;
  sections: ContextHeaderSection[];
}): ContextHeaderSection[] {
  const explicitSlots = [stateSection, focusSection, actionSection, evidenceSection];
  const hasExplicitSlots = explicitSlots.some(Boolean);
  if (hasExplicitSlots) {
    return [
      stateSection ? { id: 'state', title: stateSection.title || '当前状态', items: stateSection.items } : null,
      focusSection ? { id: 'focus', title: focusSection.title || '当前焦点', items: focusSection.items } : null,
      actionSection ? { id: 'action', title: actionSection.title || '下一步', items: actionSection.items } : null,
      evidenceSection ? { id: 'evidence', title: evidenceSection.title || '证据', items: evidenceSection.items } : null,
    ].filter((section): section is ContextHeaderSection => Boolean(section));
  }
  if (!sections.length) return [];
  const normalizedTitles = ['当前状态', '当前焦点', '下一步', '证据'];
  return sections.map((section, index) => ({
    ...section,
    title: normalizedTitles[index] || section.title,
  }));
}

export function ContextHeader({
  title,
  subtitle,
  breadcrumbs = [],
  actions,
  sections = [],
  stateSection,
  focusSection,
  actionSection,
  evidenceSection,
  compact = false,
}: ContextHeaderProps) {
  const normalizedSections = normalizeRhythmSections({
    stateSection,
    focusSection,
    actionSection,
    evidenceSection,
    sections,
  });
  return (
    <section className={`context-header ${compact ? 'compact' : ''}`.trim()}>
      <div className="context-header-head">
        <div className="context-header-copy">
          <p className="panel-kicker">{title}</p>
          <h2>{subtitle}</h2>
        </div>
        <div className="context-header-meta">
          {breadcrumbs.length ? (
            <div className="context-breadcrumbs">
              {breadcrumbs.map((item, index) => (
                <span key={`${item.label}-${index}`} className={item.current ? 'current' : ''}>
                  {item.label}
                  {index < breadcrumbs.length - 1 ? ' / ' : ''}
                </span>
              ))}
            </div>
          ) : null}
          {actions}
        </div>
      </div>
      {normalizedSections.length ? (
        <div className="context-header-controls">
          {normalizedSections.map((section) => (
            <section className="context-control-block" key={section.id}>
              <div className="context-control-head">
                <span className="context-control-title">{section.title}</span>
              </div>
              <div className="context-control-list">
                {section.items.map((item, index) => {
                  const body = (
                    <>
                      <span className="context-control-item-label">{item.label}</span>
                      <strong className="context-control-item-value">{item.value}</strong>
                      {item.detail ? <small className="context-control-item-detail">{item.detail}</small> : null}
                    </>
                  );

                  if (item.type === 'link') {
                    return (
                      <Link className={`chip-button context-control-link ${item.active ? 'active' : ''}`.trim()} key={`${section.id}-${item.label}-${index}`} to={item.to}>
                        {body}
                      </Link>
                    );
                  }

                  return (
                    <div className={`context-control-fact tone-${item.tone || 'neutral'}`.trim()} key={`${section.id}-${item.label}-${index}`}>
                      {body}
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      ) : null}
    </section>
  );
}
