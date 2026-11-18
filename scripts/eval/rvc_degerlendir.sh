#!/usr/bin/env bash
# Her yeni 10'luk checkpoint'i (10e, 20e, ...) değerlendirir; yeterli/plato olunca eğitimi durdurur,
# ince ayar yapar ve modeli profil klasörüne dışa aktarır. scripts/profil/kontrol.sh her 10 dk çağırır.
# Ölçüt: test cümleleri (XTTS, profil referansı) -> RVC (CPU) -> XTTS konuşmacı gömmesi ile profil merkezine (centroid) ort. kosinüs.
# Tavan: gerçek kayıtların benzerlik medyanı (kontrol.sh YETERLI olarak hesaplar).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Parametreler ortam değişkenleriyle verilir (kontrol.sh ayarlar): MODEL, AD, DOSYA, PROFIL, EV, EVAL_CENTROID zorunludur.
: "${MODEL:?MODEL gerekli}" "${AD:?AD gerekli}" "${DOSYA:?DOSYA gerekli}" "${PROFIL:?PROFIL gerekli}" "${EVAL_CENTROID:?EVAL_CENTROID gerekli}"
EXP="$ROOT/external/Applio/logs/$MODEL"; EV="${EV:-$ROOT/outputs/eval}"; CSV="$EV/sonuclar.csv"
ST="${ST:-$ROOT/outputs/.bekci}"; BLOG="${BLOG:-$ROOT/outputs/bekci.log}"
export EVAL_CENTROID
KAPAT_SERVIS="${KAPAT_SERVIS-}"; KAPAT_TIMER="${KAPAT_TIMER-}"
OLDUR_EK="${OLDUR_EK:-}"
MIN_EPOCH="${MIN_EPOCH:-60}"; MAX_EPOCH="${MAX_EPOCH:-200}"; PLATO_PENCERE=3; PLATO_ESIK="${PLATO_ESIK:-0.01}"; SECIM_TOL="${SECIM_TOL:-0.01}"; YETERLI="${YETERLI:-0.855}"
APY="$ROOT/envs/applio/bin/python"; XPY="$ROOT/envs/xtts/bin/python"
export APPLIO_CONFIG_DIR=/tmp/applio_eval_config EVAL_THREADS=3
mkdir -p "$APPLIO_CONFIG_DIR"; echo '{"precision": "fp32"}' > "$APPLIO_CONFIG_DIR/config.json"
say() { echo "[$(date '+%F %T')] [değerlendirme] $*" >> "$BLOG"; }
lim() { systemd-run --user --scope -q -p MemoryMax=6G nice -n 19 "$@"; }
[ -f "$ST/durdur" ] && exit 0
exec 8>"$ST/degerlendir.lock"; flock -n 8 || exit 0
[ -f "$CSV" ] || echo "epoch,ort,min,zaman" > "$CSV"
[ -f "$EXP/$MODEL.index" ] || { cd "$ROOT/external/Applio" && nice -n 19 "$APY" rvc/train/process/extract_index.py "$EXP" Auto >/dev/null 2>&1; }

# 1) Değerlendirilmemiş 10'luk checkpoint'ler
for p in $(ls "$EXP"/${MODEL}_*e_*.pth 2>/dev/null); do
  n=$(basename "$p" | sed -E "s/${MODEL}_([0-9]+)e_.*/\1/")
  [ $((n % 10)) -eq 0 ] || continue
  grep -q "^$n," "$CSV" && continue
  [ $(( $(date +%s) - $(stat -c %Y "$p") )) -ge 60 ] || continue
  rm -rf "$EV/e$n"
  lim "$APY" "$ROOT/scripts/eval/rvc_cevir.py" "$p" "$EXP/$MODEL.index" "$EV/xtts" "$EV/e$n" >/dev/null 2>&1
  r=$(lim "$XPY" "$ROOT/scripts/eval/benzerlik.py" "$EV/e$n" 2>/dev/null | grep '"ort"')
  [ -n "$r" ] || { say "epoch $n ölçülemedi"; continue; }
  o=$(echo "$r" | sed -E 's/.*"ort": ([0-9.]+).*/\1/'); m=$(echo "$r" | sed -E 's/.*"min": ([0-9.]+).*/\1/')
  echo "$n,$o,$m,$(date '+%F %T')" >> "$CSV"; say "epoch $n benzerlik ort=$o min=$m"
done

# 2) Durma kuralı
karar=$("$APY" -I - "$CSV" $MIN_EPOCH $PLATO_PENCERE $PLATO_ESIK $YETERLI $SECIM_TOL $MAX_EPOCH <<'PY'
import csv, sys
rows = sorted((int(r["epoch"]), float(r["ort"])) for r in csv.DictReader(open(sys.argv[1])))
mn, w, eps, ok, tol = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
mx = int(sys.argv[7])
if not rows: print("devam"); sys.exit()
last = rows[-1][0]; top = max(s for _, s in rows)
# Gürültü payı içinde en iyiye eşit olanlardan EN GEÇ epoch (daha uzun eğitim = daha temiz ses)
best = max((r for r in rows if r[1] >= top - tol), key=lambda r: r[0])
if last >= mx: print(f"dur {best[0]} {mx} epoch (üst sınır)"); sys.exit()
if last >= max(mn - 20, 20) and len(rows) >= 2 and rows[-1][1] >= ok and rows[-2][1] >= ok - 0.005:
    print(f"dur {best[0]} yeterli: son iki ölçüm {rows[-2][1]:.3f}, {rows[-1][1]:.3f} (tavan ~0,86)"); sys.exit()
if last >= mn and len(rows) > w:
    onceki = max(s for _, s in rows[:-w]); son = max(s for _, s in rows[-w:])
    if son < onceki + eps:
        print(f"dur {best[0]} plato: son {w} ölçümün en iyisi {son:.3f}, önceki en iyi {onceki:.3f}"); sys.exit()
print("devam")
PY
)
[ "${karar%% *}" = "dur" ] || exit 0
read -r _ BEST NEDEN <<< "$karar"
touch "$ST/durdur"; say "DURDURMA KARARI: $NEDEN -> en iyi epoch $BEST"

# 3) Eğitimi durdur (bekçi ve devam servisi 'durdur' bayrağına uyar)
[ -n "$KAPAT_SERVIS" ] && systemctl --user disable $KAPAT_SERVIS >/dev/null 2>&1
[ -n "$OLDUR_EK" ] && pkill -f "$OLDUR_EK"; pkill -f "core.py train --model-name $MODEL"
sleep 2; pkill -f "rvc/train/train.py $MODEL"; sleep 5; pkill -9 -f "rvc/train/train.py $MODEL"
say "eğitim durduruldu"

# 4) İnce ayar: en iyi epoch'ta index_rate x protect taraması
BP=$(ls "$EXP"/${MODEL}_${BEST}e_*.pth); IA="$EV/ince_ayar"; rm -rf "$IA"; mkdir -p "$IA"
for ir in 0.0 0.5 0.75 1.0; do for pr in 0.33 0.5; do
  lim "$APY" "$ROOT/scripts/eval/rvc_cevir.py" "$BP" "$EXP/$MODEL.index" "$EV/xtts" "$IA/ir${ir}_pr${pr}" $ir $pr 0 >/dev/null 2>&1
done; done
lim "$XPY" "$ROOT/scripts/eval/benzerlik.py" "$IA"/ir* 2>/dev/null | grep '"ort"' > "$IA/skorlar.jsonl"
read -r SK DIR <<< "$("$APY" -I -c "import json,sys; b=max((json.loads(l) for l in open(sys.argv[1])), key=lambda r: r['ort']); print(b['ort'], b['klasor'])" "$IA/skorlar.jsonl")"
IR=$(basename "$DIR" | sed -E 's/ir([0-9.]+)_pr.*/\1/'); PR=$(basename "$DIR" | sed -E 's/.*_pr([0-9.]+)/\1/')
say "ince ayar: en iyi index_rate=$IR protect=$PR benzerlik=$SK"

# 5) Profil klasörüne dışa aktar
MD="$PROFIL/model"; mkdir -p "$MD" "$PROFIL/ornek"
cp "$BP" "$MD/$DOSYA.pth"; cp "$EXP/$MODEL.index" "$MD/$DOSYA.index"; cp "$DIR"/*.wav "$PROFIL/ornek/"
cat > "$MD/ayarlar.json" <<J
{"ad": "$AD", "kaynak_model": "$(basename "$BP")", "epoch": $BEST, "index_rate": $IR, "protect": $PR, "pitch": 0,
 "f0_method": "rmvpe", "sample_rate": 40000, "benzerlik": $SK, "durma_nedeni": "$NEDEN"}
J
[ -n "$KAPAT_TIMER" ] && systemctl --user disable --now $KAPAT_TIMER >/dev/null 2>&1
{ echo; echo "### $AD hazır ($(date '+%F %T')) — otomatik değerlendirme"
  echo "- Durma: $NEDEN. En iyi epoch: $BEST. Ölçümler: \`$CSV\`."
  echo "- İnce ayar: index_rate=$IR, protect=$PR, pitch 0, rmvpe -> benzerlik $SK (tavan/yeterli eşiği: $YETERLI)."
  echo "- Model: \`$MD/$DOSYA.pth\`, \`$DOSYA.index\`, \`ayarlar.json\`. Dinleme örnekleri: \`$PROFIL/ornek/\`."
  echo "- Eğitim durduruldu. GPU boş."; } >> "$ROOT/outputs/${DOSYA}_rapor.md"
notify-send -u critical "$AD hazır" "Epoch $BEST, benzerlik $SK. Örnekler: $PROFIL/ornek/" 2>/dev/null || true
say "$AD dışa aktarıldı, servisler kapatıldı."
