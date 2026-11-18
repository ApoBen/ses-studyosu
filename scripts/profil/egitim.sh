#!/usr/bin/env bash
# Profil RVC eğitimi (kullanım: egitim.sh <profil>): XPU, bf16, batch 6, 200 epoch hedefi, her 2 epoch'ta checkpoint + temiz yeniden başlatma.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; . "$ROOT/scripts/profil/$1.env"; ST="$ROOT/outputs/.$AD"
LOG="$ROOT/outputs/train_$MODEL.log"; EXP="$ROOT/external/Applio/logs/$MODEL"
export APPLIO_CONFIG_DIR=/tmp/applio_config; mkdir -p $APPLIO_CONFIG_DIR; echo '{"precision": "bf16"}' > $APPLIO_CONFIG_DIR/config.json
cd "$ROOT/external/Applio" || exit 1
for i in $(seq 1 150); do
  [ -f "$ST/durdur" ] && break
  ls "$EXP"/${MODEL}_${TOPLAM_EPOCH:-200}e_*.pth >/dev/null 2>&1 && break
  echo "=== Run $i $(date) ===" >> "$LOG"
  "$ROOT/envs/applio/bin/python" -u core.py train --model-name $MODEL --vocoder HiFi-GAN --save-every-epoch 2 --save-only-latest \
    --total-epoch ${TOPLAM_EPOCH:-200} --sample-rate 40000 --batch-size 6 --gpu 0 --pretrained >> "$LOG" 2>&1
  sleep 2
done
