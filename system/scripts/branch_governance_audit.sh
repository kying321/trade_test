#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/branch_governance_audit.sh [--repo owner/name] [--output-dir PATH]

Non-destructive audit:
  1) Verify strict branch protection posture.
  2) Report expired hotfix branches (no close/delete).
USAGE
}

repo=""
output_dir="system/output/review"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      shift
      repo="${1:-}"
      ;;
    --output-dir)
      shift
      output_dir="${1:-system/output/review}"
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

ts="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${repo_root}/${output_dir}"
artifact_json="${repo_root}/${output_dir}/${ts}_branch_governance_audit.json"
artifact_md="${repo_root}/${output_dir}/${ts}_branch_governance_audit.md"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

set +e
bash "${repo_root}/system/scripts/github_branch_protection.sh" check --profile strict ${repo:+--repo "$repo"} >"${tmpdir}/protection.out" 2>"${tmpdir}/protection.err"
protect_rc=$?
set -e

set +e
bash "${repo_root}/system/scripts/hotfix_reaper.sh" --mode report ${repo:+--repo "$repo"} >"${tmpdir}/reaper.out" 2>"${tmpdir}/reaper.err"
reaper_rc=$?
set -e

protect_pass=false
reaper_pass=false
if [[ "$protect_rc" -eq 0 ]]; then
  protect_pass=true
fi
if [[ "$reaper_rc" -eq 0 ]]; then
  reaper_pass=true
fi

overall_pass=false
if [[ "$protect_pass" == "true" && "$reaper_pass" == "true" ]]; then
  overall_pass=true
fi

jq -n \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson protect_rc "$protect_rc" \
  --argjson reaper_rc "$reaper_rc" \
  --argjson protect_pass "$protect_pass" \
  --argjson reaper_pass "$reaper_pass" \
  --argjson overall_pass "$overall_pass" \
  --arg protection_stdout "$(cat "${tmpdir}/protection.out")" \
  --arg protection_stderr "$(cat "${tmpdir}/protection.err")" \
  --arg reaper_stdout "$(cat "${tmpdir}/reaper.out")" \
  --arg reaper_stderr "$(cat "${tmpdir}/reaper.err")" \
  '{
    generated_at_utc: $generated_at,
    checks: {
      branch_protection_strict: {
        pass: $protect_pass,
        return_code: $protect_rc,
        stdout: $protection_stdout,
        stderr: $protection_stderr
      },
      hotfix_reaper_report: {
        pass: $reaper_pass,
        return_code: $reaper_rc,
        stdout: $reaper_stdout,
        stderr: $reaper_stderr
      }
    },
    overall_pass: $overall_pass
  }' > "$artifact_json"

cat > "$artifact_md" <<MD
# Branch Governance Audit (${ts})

- overall_pass: \`${overall_pass}\`
- branch_protection_strict_rc: \`${protect_rc}\`
- hotfix_reaper_report_rc: \`${reaper_rc}\`

## Artifacts
- JSON: ${artifact_json}
MD

echo "$artifact_json"
echo "$artifact_md"

if [[ "$overall_pass" != "true" ]]; then
  exit 2
fi
