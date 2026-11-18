#!/usr/bin/env bash
# Profil hattı denetleyicisi (rvc-profil@<profil>.timer, 10 dk). Kullanım: kontrol.sh <profil>; ayarlar <profil>.env
# Sıra: hazırlık (CPU) -> GPU boşsa RVC eğitimi -> her 10e değerlendirme (20 cümle) -> durma + ince ayar + dışa aktarma.
# Takılırsa masaüstü bildirimi gönderir (en fazla 2 kez).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; P=$1; . "$ROOT/scripts/profil/$P.env"
ST="$ROOT/outputs/.$AD"; BLOG="$ROOT/outputs/${AD}_bekci.log"; EXP="$ROOT/external/Applio/logs/$MODEL"; LOG="$ROOT/outputs/train_$MODEL.log"
export PATH="/usr/bin:/bin:$HOME/.local/bin"; mkdir -p "$ST"; now=$(date +%s)
say() { echo "[$(date '+%F %T')] $*" >> "$BLOG"; }
ai() { printf '\n%s\n' "$*" >> "$ROOT/outputs/${AD}_gunluk.md"; }
notify() { notify-send -u critical "$GORUNEN_AD hattı" "$1" 2>/dev/null || true; }
exec 7>"$ST/kontrol.lock"; flock -n 7 || exit 0
sorun() {  # takılma: masaüstü bildirimi + günlük (en fazla 2 bildirim, aralarında en az 2 saat)
  n=$(cat "$ST/sorun_sayisi" 2>/dev/null || echo 0); last=$(cat "$ST/sorun_son" 2>/dev/null || echo 0)
  [ "$n" -lt 2 ] && [ $((now - last)) -ge 7200 ] || return 0
  echo $((n+1)) > "$ST/sorun_sayisi"; echo "$now" > "$ST/sorun_son"
  say "SORUN: $1"; notify "Sorun: $1 (günlük: $BLOG)"
}
[ -f "$ST/bitti" ] && exit 0
if [ -f "$ST/durdur" ]; then
  ls "$ROOT/profiles/$AD/model/ayarlar.json" >/dev/null 2>&1 && { touch "$ST/bitti"; systemctl --user disable --now "rvc-profil@$P.timer"; }
  exit 0
fi
# 1) Hazırlık
if [ ! -f "$ST/hazirlik_tamam" ]; then
  pgrep -f "profil/hazirlik.sh $P" >/dev/null && exit 0
  h=$(cat "$ST/hazirlik_deneme" 2>/dev/null || echo 0)
  if [ "$h" -ge 2 ]; then sorun "$GORUNEN_AD hazırlığı 2 kez başarısız (outputs/${AD}_hazirlik.log)"; exit 0; fi
  echo $((h+1)) > "$ST/hazirlik_deneme"
  dev=cpu; pgrep -f "rvc/train/train.py|train_xpu.py" >/dev/null || dev=xpu
  say "hazırlık başlatılıyor (deneme $((h+1)), demucs=$dev)"
  systemd-run --user -q --unit="$AD-hazirlik-$now" -p MemoryMax=7G -p Nice=15 "$ROOT/scripts/profil/hazirlik.sh" "$P" $dev
  exit 0
fi
[ -f "$ST/ai_hazirlik" ] || { ai "### $GORUNEN_AD hazırlık tamam ($(date '+%F %T')) — otomatik: $(tr -d '\n' < "$ROOT/profiles/$AD/dataset/ozet.json") | eğitimsiz XTTS taban: $(cat "$ST/xtts_taban.json")"; touch "$ST/ai_hazirlik"; say "hazırlık tamam"; }
# 2) Eğitim (GPU'yu başka iş kullanmıyorsa)
running=0; pgrep -f "profil/egitim.sh $P|core.py train --model-name $MODEL|rvc/train/train.py $MODEL" >/dev/null && running=1
if [ "$running" -eq 0 ] && ! ls "$EXP"/${MODEL}_${TOPLAM_EPOCH:-200}e_*.pth >/dev/null 2>&1; then
  if pgrep -f "rvc/train/train.py|train_xpu.py|xtts_ft/egit.sh" >/dev/null; then say "GPU meşgul, bekleniyor"; exit 0; fi
  [ -f "$ST/egitim_basladi" ] || { echo "$now" > "$ST/egitim_basladi"; ai "### $GORUNEN_AD RVC eğitimi başladı ($(date '+%F %T')) — log: outputs/train_$MODEL.log"; notify "$GORUNEN_AD eğitimi başladı."; }
  say "eğitim başlatılıyor"
  systemd-run --user -q --unit="$AD-egitim-$now" "$ROOT/scripts/profil/egitim.sh" "$P"
  exit 0
fi
# 3) Takılma bekçisi
newest=$(ls -t "$EXP"/${MODEL}_*e_*.pth 2>/dev/null | head -1)
ref=$([ -n "$newest" ] && stat -c %Y "$newest" || cat "$ST/egitim_basladi")
age=$(( (now - ref) / 60 )); log_age=$(( (now - $(stat -c %Y "$LOG" 2>/dev/null || echo $now)) / 60 ))
if [ "$age" -ge 40 ]; then
  [ "$log_age" -ge 15 ] && { say "takılma: $age dk ilerleme yok, log $log_age dk donuk -> train.py öldürülüyor"; pkill -9 -f "rvc/train/train.py $MODEL"; }
  [ "$age" -ge 70 ] && sorun "$GORUNEN_AD eğitimi $age dakikadır yeni checkpoint üretmiyor"
  exit 0
fi
# 4) Değerlendirme + durma (20 cümle, plato eşiği 0,01, en iyiye 0,01 içindeki en geç epoch)
YET=$(awk -F, 'NR>1{print $5}' "$ROOT/profiles/$AD/dataset/manifest.csv" | sort -n | awk '{a[NR]=$1} END{printf "%.3f", a[int(NR/2)+1]-0.005}')
MODEL=$MODEL DOSYA="$DOSYA" PROFIL="$ROOT/profiles/$AD" EV="$ROOT/outputs/eval_$AD" ST="$ST" BLOG="$BLOG" \
EVAL_CENTROID="$ROOT/profiles/$AD/ref/centroid.npy" MIN_EPOCH="${MIN_EPOCH:-50}" MAX_EPOCH="${TOPLAM_EPOCH:-200}" KAPAT_SERVIS="" KAPAT_TIMER="" OLDUR_EK="profil/egitim.sh $P" YETERLI="$YET" AD="$GORUNEN_AD" \
  "$ROOT/scripts/eval/rvc_degerlendir.sh"
