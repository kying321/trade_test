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

type ContextHeaderProps = {
  title: string;
  subtitle: string;
  breadcrumbs?: Array<{ label: string; current?: boolean }>;
  actions?: ReactNode;
  sections?: ContextHeaderSection[];
  compact?: boolean;
};

export function ContextHeader({ title, subtitle, breadcrumbs = [], actions, sections = [], compact = false }: ContextHeaderProps) {
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
      {sections.length ? (
        <div className="context-header-controls">
          {sections.map((section) => (
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
