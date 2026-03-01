#!/usr/bin/env bash
set -euo pipefail

gov_primary_branches_csv() {
  printf '%s' "${GOV_PRIMARY_BRANCHES:-main,pi,lie}"
}

gov_primary_branches_lines() {
  gov_primary_branches_csv \
    | tr ',' '\n' \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
    | awk 'NF>0 && !seen[$0]++'
}

gov_primary_branches_join() {
  local sep="${1:-, }"
  local out=""
  local first=1
  local branch

  while IFS= read -r branch; do
    if [[ $first -eq 1 ]]; then
      out="$branch"
      first=0
    else
      out+="${sep}${branch}"
    fi
  done < <(gov_primary_branches_lines)

  printf '%s' "$out"
}

gov_primary_branches_count() {
  gov_primary_branches_lines | awk 'END {print NR}'
}

gov_is_primary_branch() {
  local needle="$1"
  local branch

  while IFS= read -r branch; do
    if [[ "$branch" == "$needle" ]]; then
      return 0
    fi
  done < <(gov_primary_branches_lines)

  return 1
}

gov_primary_branches_regex_group() {
  local out=""
  local first=1
  local branch
  local escaped

  while IFS= read -r branch; do
    escaped="$(printf '%s' "$branch" | sed -E 's/[][(){}.^$*+?|\\/]/\\\\&/g')"
    if [[ $first -eq 1 ]]; then
      out="$escaped"
      first=0
    else
      out+="|${escaped}"
    fi
  done < <(gov_primary_branches_lines)

  printf '%s' "$out"
}

gov_hotfix_branch_regex() {
  local group
  group="$(gov_primary_branches_regex_group)"
  printf '^hotfix/(%s)/([A-Za-z0-9._-]+)/([0-9]{12})$' "$group"
}

gov_hotfix_pattern_human() {
  printf 'hotfix/<%s>/<ticket>/<expires_utc_yyyymmddhhmm>' "$(gov_primary_branches_join '|')"
}
