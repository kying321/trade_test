#!/usr/bin/env bash
set -euo pipefail

repo=""
audit_output_dir="system/output/review"
step_summary_out=""
comment_out=""
emit_github_warning=0

branch_gov_audit_script="${FENLIE_BRANCH_GOV_AUDIT_SCRIPT:-}"
gov_summary_script="${FENLIE_GOV_SUMMARY_SCRIPT:-}"
gov_comment_script="${FENLIE_GOV_COMMENT_SCRIPT:-}"

usage() {
  cat <<'USAGE'
Usage:
  system/scripts/run_governance_audit_advisory.sh [--repo owner/name] [--audit-output-dir PATH]
                                                  [--step-summary-out PATH] [--comment-out PATH]
                                                  [--emit-github-warning]

Runs the branch governance audit in advisory mode, then renders optional step-summary
and PR-comment outputs. This runner is intentionally non-blocking and always exits 0.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      shift
      repo="${1:-}"
      ;;
    --audit-output-dir)
      shift
      audit_output_dir="${1:-system/output/review}"
      ;;
    --step-summary-out)
      shift
      step_summary_out="${1:-}"
      ;;
    --comment-out)
      shift
      comment_out="${1:-}"
      ;;
    --emit-github-warning)
      emit_github_warning=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg '$1'" >&2
      usage
      exit 2
      ;;
  esac
  shift || true
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  echo "ERROR: must run inside git repository." >&2
  exit 2
fi

if [[ -z "$branch_gov_audit_script" ]]; then
  branch_gov_audit_script="${repo_root}/system/scripts/branch_governance_audit.sh"
fi
if [[ -z "$gov_summary_script" ]]; then
  gov_summary_script="${repo_root}/system/scripts/render_governance_audit_summary.py"
fi
if [[ -z "$gov_comment_script" ]]; then
  gov_comment_script="${repo_root}/system/scripts/render_governance_audit_comment.py"
fi

audit_stdout="$(mktemp)"
audit_stderr="$(mktemp)"
summary_rc=0
comment_rc=0

set +e
bash "${branch_gov_audit_script}" ${repo:+--repo "$repo"} --output-dir "${audit_output_dir}" >"${audit_stdout}" 2>"${audit_stderr}"
audit_rc=$?
set -e

cat "${audit_stdout}"
if [[ -s "${audit_stderr}" ]]; then
  cat "${audit_stderr}" >&2
fi

latest_audit_json="$(grep '_branch_governance_audit\.json$' "${audit_stdout}" | tail -n 1 || true)"
if [[ -z "${latest_audit_json}" || ! -f "${latest_audit_json}" ]]; then
  latest_audit_json="$(ls -1t "${repo_root}/${audit_output_dir}"/*_branch_governance_audit.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -n "${latest_audit_json}" && -f "${latest_audit_json}" ]]; then
  if [[ -n "${step_summary_out}" || "${emit_github_warning}" -eq 1 ]]; then
    summary_cmd=(python3 "${gov_summary_script}" --audit-json "${latest_audit_json}")
    if [[ -n "${step_summary_out}" ]]; then
      summary_cmd+=(--step-summary-out "${step_summary_out}")
    fi
    if [[ "${emit_github_warning}" -eq 1 ]]; then
      summary_cmd+=(--emit-github-warning)
    fi
    set +e
    "${summary_cmd[@]}"
    summary_rc=$?
    set -e
    if [[ "${summary_rc}" -ne 0 ]]; then
      echo "::warning::governance audit advisory summary render failed rc=${summary_rc}"
    fi
  fi

  if [[ -n "${comment_out}" ]]; then
    set +e
    python3 "${gov_comment_script}" --audit-json "${latest_audit_json}" --comment-out "${comment_out}"
    comment_rc=$?
    set -e
    if [[ "${comment_rc}" -ne 0 ]]; then
      echo "::warning::governance audit advisory comment render failed rc=${comment_rc}"
    fi
  fi
else
  echo "::warning::governance audit advisory did not produce an audit json artifact rc=${audit_rc}"
fi

echo "[governance-audit-advisory] non-blocking audit_rc=${audit_rc} summary_rc=${summary_rc} comment_rc=${comment_rc}"
exit 0
