#!/bin/bash
# Pull, without the wall of "would be overwritten by merge".
#
#     bash scripts/pull_safely.sh
#
# The cluster checkout has its own copies of files that are tracked here, and
# local edits to files this branch changes. A plain `git pull` refuses on both
# and names every file, which has to be cleared by hand before anything moves.
# This clears it the safe way -- nothing is deleted, everything goes into a
# dated backup directory beside the repository and is listed at the end.
#
#   1  locally MODIFIED tracked files -> copied aside, then the edit discarded
#   2  UNTRACKED files the pull would overwrite -> moved aside
#   3  git pull
#
# Set BACKUP_DIR to put the copies somewhere else. DRY_RUN=1 shows what it
# would touch and changes nothing.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BK="${BACKUP_DIR:-../pull_backup_$(date +%Y-%m-%d_%H%M)}"
DRY="${DRY_RUN:-0}"
moved=0

echo "repo:   $(pwd)"
echo "branch: $BRANCH"
echo "backup: $BK"
[ "$DRY" = "1" ] && echo "DRY RUN -- nothing will be changed"
echo ""

git fetch origin "$BRANCH" || { echo "fetch failed; nothing changed."; exit 1; }

echo "1. locally modified files that the pull would overwrite"
while IFS= read -r f; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    # Only the ones the incoming commits actually touch. An edit git is not
    # going to trip over is left exactly as it is.
    git diff --name-only HEAD "origin/$BRANCH" -- "$f" | grep -q . || continue
    echo "   copied aside, edit discarded: $f"
    if [ "$DRY" != "1" ]; then
        mkdir -p "$BK/$(dirname "$f")" && cp -p "$f" "$BK/$f" && git checkout -- "$f"
    fi
    moved=$((moved + 1))
done < <(git diff --name-only)
[ "$moved" = "0" ] && echo "   none"

echo ""
echo "2. untracked files the pull would overwrite"
before=$moved
while IFS= read -r f; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 && continue   # tracked: not this case
    echo "   moved aside: $f"
    if [ "$DRY" != "1" ]; then
        mkdir -p "$BK/$(dirname "$f")" && mv "$f" "$BK/$f"
    fi
    moved=$((moved + 1))
done < <(git diff --name-only HEAD "origin/$BRANCH")
[ "$moved" = "$before" ] && echo "   none"

echo ""
if [ "$DRY" = "1" ]; then
    echo "DRY RUN -- stopping before the pull. $moved file(s) would be set aside."
    exit 0
fi

echo "3. git pull"
git pull --ff-only origin "$BRANCH"
rc=$?

echo ""
if [ "$rc" -eq 0 ]; then
    echo "Pulled. HEAD is now $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
else
    echo "The pull still failed (exit $rc). Read its message -- everything this"
    echo "script sets aside is in $BK and nothing was deleted."
fi
if [ "$moved" -gt 0 ]; then
    echo ""
    echo "$moved file(s) set aside in $BK"
    echo "Nothing was deleted. To compare one against what arrived:"
    echo "   diff $BK/<path> <path>"
fi
exit $rc
