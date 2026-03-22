import React from "react";
import { Badge } from "./ui";

export function PageHeader({ eyebrow, title, description, badges = [] }) {
  return (
    <div className="page-header">
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h2 className="content-title">{title}</h2>
        {description ? <p className="content-copy">{description}</p> : null}
      </div>
      {badges.length ? (
        <div className="page-header-badges">
          {badges.map((item) => (
            <Badge key={`${item.label}-${item.value}`} value={item.value}>
              {item.label}
            </Badge>
          ))}
        </div>
      ) : null}
    </div>
  );
}
