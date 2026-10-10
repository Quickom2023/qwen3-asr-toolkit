#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${DEPLOY_HOST:-ubuntu@s2t.quickom.net}"
REMOTE_DIR="${DEPLOY_DIR:-/home/ubuntu/qwen3-asr-toolkit}"
BRANCH="${DEPLOY_BRANCH:-prod}"
SSH_KEY="${DEPLOY_SSH_KEY:-$HOME/.ssh/bw-deploy-datagram}"

if [[ -n "${DEPLOY_SSH_AUTH_SOCK:-}" ]]; then
  export SSH_AUTH_SOCK="$DEPLOY_SSH_AUTH_SOCK"
fi

if [[ ! -r "$SSH_KEY" ]]; then
  echo "SSH key is not readable: $SSH_KEY" >&2
  exit 1
fi

ssh -A -i "$SSH_KEY" -o BatchMode=yes "$REMOTE_HOST" "bash -s -- '$REMOTE_DIR' '$BRANCH'" <<'REMOTE'
set -euo pipefail

repo_dir="$1"
branch="$2"
cd "$repo_dir"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$branch" ]]; then
  echo "Expected branch '$branch', found '$current_branch'." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked server files have local changes; deploy stopped without modifying them." >&2
  git status --short
  exit 1
fi

git fetch origin "$branch"
git merge --ff-only "origin/$branch"
asr-env/bin/pip install --no-deps -e .
sudo systemctl restart quickom-asr-api.service

for attempt in {1..30}; do
  if systemctl is-active --quiet quickom-asr-api.service; then
    break
  fi
  if [[ "$attempt" == 30 ]]; then
    echo "API service did not become active after restart." >&2
    sudo journalctl -u quickom-asr-api.service -n 40 --no-pager >&2
    exit 1
  fi
  sleep 2
done

printf 'Deployed commit: '
git rev-parse --short HEAD
printf 'API service: '
systemctl is-active quickom-asr-api.service
REMOTE
