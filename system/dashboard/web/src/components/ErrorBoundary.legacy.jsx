import React, { Component } from "react";
import { labelFor } from "../utils/dictionary";

const t = (key) => labelFor(key);

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || t("legacy_unknown_render_error") };
  }

  componentDidCatch(error) {
    console.error("Fenlie dashboard render error:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-shell fallback-shell">
          <div className="fallback-card">
            <p className="eyebrow">{t("fallback_shell_kicker")}</p>
            <h1>{t("fallback_shell_title")}</h1>
            <p>{t("error_message_label")}：{this.state.message}</p>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => window.location.reload()}>
                {t("reload_page")}
              </button>
              <a className="ghost-link" href="/data/fenlie_dashboard_snapshot.json" target="_blank" rel="noreferrer">
                {t("open_public_snapshot")}
              </a>
              <a className="ghost-link" href="/operator_task_visual_panel.html" target="_blank" rel="noreferrer">
                {t("open_operator_panel")}
              </a>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
