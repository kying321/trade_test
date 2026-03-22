import React, { useEffect, useMemo, useState } from "react";
import { describeNodeType, summarizeNode, safeDisplayValue } from "../lib/formatters";
import { bilingualLabel, zhPath } from "../lib/i18n";
import { labelFor } from "../utils/dictionary";

const t = (key) => labelFor(key);

function readNodeAtPath(payload, path = []) {
  return path.reduce((current, segment) => {
    if (current === null || current === undefined) return undefined;
    return current[segment];
  }, payload);
}

function buildFieldRows(node, path = [], query = "") {
  if (!node || typeof node !== "object") return [];
  const entries = Array.isArray(node) ? node.map((value, index) => [String(index), value]) : Object.entries(node);
  return entries
    .map(([key, value]) => ({
      id: [...path, key].join(".") || key,
      key,
      path: [...path, key],
      pathLabel: [...path, key].join("."),
      type: describeNodeType(value),
      summary: summarizeNode(value),
      expandable: Boolean(value) && typeof value === "object",
      value,
    }))
    .filter((row) => !query || JSON.stringify(row).toLowerCase().includes(query.toLowerCase()));
}

export function FieldExplorer({ payload }) {
  const [path, setPath] = useState([]);
  const [query, setQuery] = useState("");
  const [selectedRowId, setSelectedRowId] = useState("");

  useEffect(() => {
    setPath([]);
    setQuery("");
    setSelectedRowId("");
  }, [payload]);

  const currentNode = readNodeAtPath(payload, path);
  const rows = useMemo(() => buildFieldRows(currentNode, path, query), [currentNode, path, query]);
  const isLeaf = currentNode === null || typeof currentNode !== "object";

  useEffect(() => {
    if (!rows.length) {
      setSelectedRowId("");
      return;
    }
    if (!rows.some((row) => row.id === selectedRowId)) setSelectedRowId(rows[0].id);
  }, [rows, selectedRowId]);

  const selectedRow = rows.find((row) => row.id === selectedRowId) || null;

  return (
    <div className="field-layout">
      <div className="field-browser-card">
        <div className="field-toolbar">
          <div>
            <div className="note-title">{t("legacy_field_path_title")}</div>
            <div className="breadcrumb-track">
              <button className="breadcrumb-button" onClick={() => setPath([])} disabled={!path.length}>{t("legacy_root_node")}</button>
              {path.map((segment, index) => (
                <React.Fragment key={`${segment}-${index}`}>
                  <span className="breadcrumb-sep">/</span>
                  <button className="breadcrumb-button" onClick={() => setPath(path.slice(0, index + 1))} disabled={index === path.length - 1}>{bilingualLabel(segment)}</button>
                </React.Fragment>
              ))}
            </div>
          </div>
          <div className="field-toolbar-actions">
            <input
              type="search"
              className="search-input search-input-compact"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t("legacy_field_filter_placeholder")}
            />
            <button className="ghost-button" disabled={!path.length} onClick={() => setPath((current) => current.slice(0, -1))} data-testid="field-back-button">{t("legacy_field_back_button")}</button>
          </div>
        </div>

        {isLeaf ? (
          <div className="note-block">
            <div className="note-title">{t("legacy_leaf_value_title")}</div>
            <div className="field-leaf-value">{safeDisplayValue(currentNode)}</div>
          </div>
        ) : (
          <div className="field-table-shell">
            <div className="field-table-head">
              <span>{t("legacy_field_key_column")}</span>
              <span>{t("kind")}</span>
              <span>{t("summary")}</span>
              <span>{t("action")}</span>
            </div>
            <div className="field-table-body">
              {rows.map((row) => (
                <div
                  key={row.id}
                  className={`field-row ${selectedRowId === row.id ? "field-row-active" : ""}`.trim()}
                  onClick={() => setSelectedRowId(row.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedRowId(row.id);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <span>
                    <strong>{bilingualLabel(row.key)}</strong>
                    {String(row.key) !== String(bilingualLabel(row.key)) ? <span className="field-raw-key">{row.key}</span> : null}
                  </span>
                  <span>{row.type}</span>
                  <span>{row.summary}</span>
                  <span>
                    {row.expandable ? (
                      <button
                        className="ghost-button field-action-button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setPath(row.path);
                        }}
                        data-testid={`field-drill-${row.pathLabel.replaceAll('.', '-')}`}
                      >
                        {t("legacy_field_enter_button")}
                      </button>
                    ) : (
                      <span className="metric-hint">{t("legacy_leaf_value_title")}</span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="field-preview-card">
        <div className="panel-header">
          <h3>{t("legacy_field_preview_title")}</h3>
          <span className="metric-hint">{selectedRow ? zhPath(selectedRow.path) : zhPath(path) || t("legacy_root_node")}</span>
        </div>
        {selectedRow ? (
          <>
            <div className="note-block">
              <div className="note-title">{t("legacy_field_alias_title")}</div>
              <div>{bilingualLabel(selectedRow.key)}</div>
            </div>
            <div className="note-block">
              <div className="note-title">{t("kind")}</div>
              <div>{selectedRow.type}</div>
            </div>
            <div className="note-block">
              <div className="note-title">{t("summary")}</div>
              <div>{selectedRow.summary}</div>
            </div>
            <pre className="json-viewer json-viewer-inline">{JSON.stringify(selectedRow.value, null, 2)}</pre>
          </>
        ) : (
          <pre className="json-viewer json-viewer-inline">{JSON.stringify(currentNode ?? {}, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}
