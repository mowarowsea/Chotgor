#!/usr/bin/env bash
# debug/ の生ログを月単位で tar.gz へ退避する運用スクリプト。
#
# 当月と前月はそのまま残し、それ以前の月だけを圧縮 → 検証 → 元データ削除する。
# 検証（tar 内エントリ数と実体の一致）が通らなかった月は元データを消さずに次へ進む。
# 月初に1回叩けば足りる想定。対象月が無ければ何もせず終了するので、何度実行してもよい。
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

DEBUG_DIR="debug"
ARCHIVE_DIR="$DEBUG_DIR/_archive"
LOG_FILE="$ARCHIVE_DIR/archive.log"

mkdir -p "$ARCHIVE_DIR"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"; }

# 当月・前月は退避対象から外す
keep_current=$(date '+%Y-%m')
keep_prev=$(date -d "$(date '+%Y-%m-01') -1 month" '+%Y-%m')

months=$(find "$DEBUG_DIR" -mindepth 1 -maxdepth 1 -type d ! -name '_archive' \
              -printf '%TY-%Tm\n' 2>/dev/null | sort -u)

if [ -z "$months" ]; then
  log "対象なし（debug/ に生ログのディレクトリが無い）"
  exit 0
fi

for month in $months; do
  if [ "$month" = "$keep_current" ] || [ "$month" = "$keep_prev" ]; then
    continue
  fi

  ym=${month//-/}
  next_month=$(date -d "$month-01 +1 month" '+%Y-%m-01')
  list=$(mktemp)
  find "$DEBUG_DIR" -mindepth 1 -maxdepth 1 -type d ! -name '_archive' \
       -newermt "$month-01" ! -newermt "$next_month" | sort > "$list"

  dir_count=$(wc -l < "$list")
  if [ "$dir_count" -eq 0 ]; then
    rm -f "$list"
    continue
  fi

  # 同じ月を再度退避する場合に既存アーカイブを壊さないよう連番を振る
  tarball="$ARCHIVE_DIR/debug_$ym.tar.gz"
  seq=2
  while [ -e "$tarball" ]; do
    tarball="$ARCHIVE_DIR/debug_${ym}_$seq.tar.gz"
    seq=$((seq + 1))
  done

  log "圧縮開始 $month dirs=$dir_count -> $tarball"
  if ! tar -czf "$tarball" -T "$list"; then
    log "ERROR 圧縮失敗 $month（元データは残す）"
    rm -f "$list"
    continue
  fi

  entries_in_tar=$(tar -tzf "$tarball" | wc -l)
  entries_on_fs=$(xargs -a "$list" -d '\n' find 2>/dev/null | wc -l)
  if [ "$entries_in_tar" -ne "$entries_on_fs" ]; then
    log "ERROR 検証不一致 $month tar=$entries_in_tar fs=$entries_on_fs（元データは残す）"
    rm -f "$list"
    continue
  fi

  xargs -a "$list" -d '\n' rm -rf
  size_mb=$(( $(stat -c %s "$tarball") / 1048576 ))
  log "完了 $month entries=$entries_in_tar archive=${size_mb}MB 元データ削除済み"
  rm -f "$list"
done
