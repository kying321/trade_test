export function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: Math.abs(num) < 10 && digits > 0 ? Math.min(2, digits) : 0,
  });
}

export function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return `${(num * 100).toFixed(digits)}%`;
}

export function safeDisplayValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    const primitiveOnly = value.every((item) => item === null || ["string", "number", "boolean"].includes(typeof item));
    return primitiveOnly ? value.map((item) => safeDisplayValue(item)).join(", ") : `[${value.length} items]`;
  }
  if (typeof value === "object") {
    if (typeof value.brief === "string") return value.brief;
    if (value.symbol || value.action || value.state) return [value.symbol, value.action, value.state].filter(Boolean).join(" / ");
    return `{${Object.keys(value).length} keys}`;
  }
  return String(value);
}

export function statusTone(value) {
  const text = String(value || "").toLowerCase();
  if (!text) return "neutral";
  if (["ok", "allowed", "active", "passed", "promising_small_sample", "mixed_positive", "ready", "live"].some((key) => text.includes(key))) {
    return "positive";
  }
  if (["blocked", "demoted", "failed", "negative", "harmful", "overblocking", "error", "not_promising"].some((key) => text.includes(key))) {
    return "negative";
  }
  if (["watch", "candidate", "mixed", "partial", "warning", "inconclusive", "local", "defer", "pending"].some((key) => text.includes(key))) {
    return "warning";
  }
  return "neutral";
}

export function badgeClass(value) {
  return `badge badge-${statusTone(value)}`;
}

export function cardList(items = []) {
  return items.filter((item) => item && item.value !== undefined && item.value !== null && item.value !== "");
}

export function describeNodeType(value) {
  if (Array.isArray(value)) return `array(${value.length})`;
  if (value === null) return "null";
  return typeof value;
}

export function summarizeNode(value) {
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    const primitiveOnly = value.every((item) => item === null || ["string", "number", "boolean"].includes(typeof item));
    return primitiveOnly ? value.map((item) => safeDisplayValue(item)).join(", ") : `[${value.length} items]`;
  }
  if (value && typeof value === "object") {
    if (typeof value.brief === "string") return value.brief;
    return `{${Object.keys(value).length} keys}`;
  }
  return safeDisplayValue(value);
}

export function downsampleCurve(curve = []) {
  if (!Array.isArray(curve)) return [];
  return curve.map((point) => ({
    date: point.date ?? point.ts ?? point.time ?? "",
    equity: Number(point.equity ?? point.value ?? 0),
  }));
}
