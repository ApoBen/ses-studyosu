#!/usr/bin/env bash
# Profil veri hazırlığı (idempotent, aşama işaretli). Kullanım: hazirlik.sh <profil> [cpu|xpu]; ayarlar scripts/profil/<profil>.env
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; . "$ROOT/scripts/profil/$1.env"; DEV="${2:-cpu}"
case "$CUMLELER" in /*) ;; *) CUMLELER="$ROOT/$CUMLELER" ;; esac
SRC="$KAYNAK"; PROF="$ROOT/profiles/$AD"; ST="$ROOT/outputs/.$AD"
LOG="$ROOT/outputs/${AD}_hazirlik.log"; XPY="$ROOT/envs/xtts/bin/python"; APY="$ROOT/envs/applio/bin/python"
mkdir -p "$ST" "$PROF/kaynak"; exec >> "$LOG" 2>&1
echo "=== [$(date '+%F %T')] hazırlık başlıyor (demucs: $DEV) ==="
if [ ! -f "$ST/1_ayir" ]; then                 # Demucs: 5 dk'lık parçalar (tek seferde OOM dersi)
  W="$PROF/kaynak/tmp"; rm -rf "$W"; mkdir -p "$W/in"
  ffmpeg -v error -i "$SRC" -ar 44100 -ac 2 -f segment -segment_time 300 "$W/in/p%03d.wav"
  for f in "$W"/in/p*.wav; do
    OMP_NUM_THREADS=4 "$XPY" -m demucs --two-stems=vocals -n htdemucs -d "$DEV" -o "$W/sep" "$f"; rm "$f"
    echo "  demucs $(basename "$f") $(date +%T)"
  done
  ls "$W"/sep/htdemucs/p*/vocals.wav | sort | sed "s/^/file '/; s/$/'/" > "$W/list.txt"
  ffmpeg -v error -y -f concat -safe 0 -i "$W/list.txt" -c copy "$PROF/kaynak/vocals.wav"; rm -rf "$W"
  touch "$ST/1_ayir"; echo "  vokal ayrıldı $(date +%T)"
fi
if [ ! -f "$ST/2_veri" ]; then
  rm -rf "$PROF/dataset/wavs_44k"
  if [ "${HAZIRLA_MOD:-kume}" = tek ]; then   # tek karakterli replik kaydı: kümeleme yerine istenmeyen ses elemesi
    EVAL_THREADS=4 "$XPY" "$ROOT/scripts/profil/hazirla_tek.py" "$PROF/kaynak/vocals.wav" "$PROF" $AD --istenmeyen "${ISTENMEYEN:-}"
  else
    EVAL_THREADS=4 "$XPY" "$ROOT/scripts/profil/hazirla_kume.py" "$PROF/kaynak/vocals.wav" "$PROF" $AD
  fi
  touch "$ST/2_veri"
fi
export APPLIO_CONFIG_DIR=/tmp/applio_config; mkdir -p $APPLIO_CONFIG_DIR; echo '{"precision": "bf16"}' > $APPLIO_CONFIG_DIR/config.json
cd "$ROOT/external/Applio"
if [ ! -f "$ST/3_applio" ]; then
  "$APY" core.py preprocess --model-name $MODEL --dataset-path "$PROF/dataset/wavs_44k" --sample-rate 40000 --cpu-cores 4
  "$APY" core.py extract --model-name $MODEL --f0-method rmvpe --embedder-model contentvec --sample-rate 40000 --gpu - --cpu-cores 4
  "$APY" rvc/train/process/extract_index.py "logs/$MODEL" Auto
  ls logs/$MODEL/extracted >/dev/null && ls logs/$MODEL/$MODEL.index >/dev/null
  touch "$ST/3_applio"; echo "  applio ön işleme+özellik+index tamam $(date +%T)"
fi
if [ ! -f "$ST/4_test" ]; then
  EVAL_THREADS=4 "$XPY" "$ROOT/scripts/eval/xtts_test_uret.py" "$CUMLELER" "$PROF/ref/ref.wav" "$ROOT/outputs/eval_$AD/xtts"
  EVAL_CENTROID="$PROF/ref/centroid.npy" "$XPY" "$ROOT/scripts/eval/benzerlik.py" "$ROOT/outputs/eval_$AD/xtts" | grep '"ort"' > "$ST/xtts_taban.json"
  touch "$ST/4_test"
fi
touch "$ST/hazirlik_tamam"; echo "=== [$(date '+%F %T')] hazırlık tamam ==="
