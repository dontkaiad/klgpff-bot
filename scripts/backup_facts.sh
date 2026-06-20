#!/usr/bin/env bash
#
# Periodic ENCRYPTED backup of facts/ and prompts/.
#
# Archives facts/ + prompts/ into a tar.gz, encrypts it symmetrically with gpg
# (passphrase from env BACKUP_PASSPHRASE), writes it to backups/archive/ and
# keeps only the 14 newest archives.
#
# Requires: bash, tar, gpg. Passphrase MUST be provided via env:
#   BACKUP_PASSPHRASE=... ./scripts/backup_facts.sh
#
# Meant to run on the SERVER via cron (do NOT run from CI). Example crontab line
# — daily at 04:00 UTC; sources .env so BACKUP_PASSPHRASE is available:
#
#   0 4 * * * cd /home/<user>/heylark/klgpff-bot && set -a && . ./.env && set +a && ./scripts/backup_facts.sh >> backups/archive/backup.log 2>&1
#
# Restore (see README → Backups):
#   gpg -d backups/archive/facts-YYYYMMDD-HHMMSS.tar.gz.gpg | tar -xzf -
#
set -euo pipefail

# Repo root = parent of this scripts/ dir, regardless of where cron invokes us.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

: "${BACKUP_PASSPHRASE:?BACKUP_PASSPHRASE is not set (export it or source .env)}"

ARCHIVE_DIR="backups/archive"
mkdir -p "$ARCHIVE_DIR"

# Only archive dirs that actually exist (avoids tar failing under set -e).
PATHS=()
[ -d facts ] && PATHS+=(facts)
[ -d prompts ] && PATHS+=(prompts)
if [ ${#PATHS[@]} -eq 0 ]; then
  echo "nothing to back up (facts/ and prompts/ both absent)"
  exit 0
fi

TS="$(date -u +%Y%m%d-%H%M%S)"
OUT="$ARCHIVE_DIR/facts-$TS.tar.gz.gpg"

tar -czf - "${PATHS[@]}" \
  | gpg --batch --yes --pinentry-mode loopback \
        --symmetric --cipher-algo AES256 \
        --passphrase "$BACKUP_PASSPHRASE" \
        -o "$OUT"

echo "💾 encrypted backup written: $OUT"

# Retention: keep the 14 newest archives, prune the rest.
ls -1t "$ARCHIVE_DIR"/facts-*.tar.gz.gpg 2>/dev/null | tail -n +15 | while read -r old; do
  rm -f -- "$old"
  echo "🗑️  pruned old backup: $old"
done
