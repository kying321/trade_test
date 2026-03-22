import sharedTermContract from "../contracts/dashboard-term-contract.json";
import { displayText } from "../utils/dictionary";

const LABELS = sharedTermContract.labels || {};

function fallbackLabel(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replaceAll("-", " / ")
    .trim();
}

function normalizeValue(value) {
  return String(value ?? "").trim();
}

function exactSharedLabel(value) {
  const raw = normalizeValue(value);
  if (!raw) return "";
  return LABELS[raw] || LABELS[raw.toLowerCase()] || "";
}

function translateShared(value) {
  const raw = normalizeValue(value);
  if (!raw) return "";
  const exact = exactSharedLabel(raw);
  if (exact) return exact;
  const rendered = displayText(raw);
  return rendered && rendered !== "—" ? rendered : fallbackLabel(raw);
}

function formatBilingual(primary, raw) {
  const text = String(primary || "");
  const origin = String(raw || "");
  if (!origin || !text || text === origin) return text || origin;
  return `${text}`;
}

export function translateCategory(value) {
  return translateShared(value);
}

export function translateKind(value) {
  return translateShared(value);
}

export function translateArtifactLabel(value) {
  return translateShared(value);
}

export function translateSourceHeadLabel(id, fallback) {
  const exact = exactSharedLabel(id);
  if (exact) return exact;
  return translateShared(fallback || id);
}

export function translateStatus(value) {
  return translateShared(value);
}

export function translateFieldLabel(value) {
  return translateShared(value);
}

export function zhLabel(value) {
  return translateShared(value);
}

export function zhChangeClass(value) {
  return translateShared(value);
}

export function bilingualLabel(value) {
  return formatBilingual(zhLabel(value), value);
}

export function zhPath(path = []) {
  if (!Array.isArray(path) || !path.length) return "";
  return path.map((segment) => bilingualLabel(segment)).join(" / ");
}
