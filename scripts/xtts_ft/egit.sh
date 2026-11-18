#!/usr/bin/env bash
# XTTS ince ayar döngüsü: train_xpu.py her epoch sonunda 42 ile çıkar (sürücü bellek sızıntısı), burada yeniden başlatılır.
# Kullanım: egit.sh <veri_klasörü> <çıktı_klasörü> <epoch>
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; VERI=$1; OUT=$2; EP=$3; LOG="$OUT/egitim.log"
mkdir -p "$OUT"
for i in $(seq 1 $((EP * 3))); do
  echo "=== çalıştırma $i $(date '+%F %T') ===" >> "$LOG"
  "$ROOT/envs/xtts/bin/python" -u "$ROOT/scripts/xtts_ft/train_xpu.py" --veri "$VERI" --cikti "$OUT" --epoch "$EP" --batch "${XTTS_BATCH:-1}" --biriktir "${XTTS_BIRIKTIR:-8}" >> "$LOG" 2>&1
  rc=$?
  [ $rc -eq 0 ] && { echo "=== bitti $(date '+%F %T') ===" >> "$LOG"; break; }
  [ $rc -eq 42 ] || { echo "=== hata (çıkış $rc) $(date '+%F %T') ===" >> "$LOG"; sleep 90; }
  sleep 3
done
