#!/usr/bin/env bash
set -e
REPO_NAME="LY-Project"

if [ -z "$(git rev-parse --is-inside-work-tree 2>/dev/null)" ]; then
  git init
fi

git add .
git commit -m "Initial commit - LY-Project (churn+RMAB)" || echo "Nothing to commit"

if command -v gh >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
else
  echo "gh CLI not found. Create repo on GitHub and then run:"
  echo "  git remote add origin https://github.com/<your-username>/$REPO_NAME.git"
  echo "  git branch -M main"
  echo "  git push -u origin main"
fi
