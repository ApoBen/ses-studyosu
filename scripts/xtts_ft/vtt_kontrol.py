"""Whisper metinlerini YouTube altyazısıyla (VTT) çapraz kontrol eder; uyuşmayan parçaları eğitimden çıkarır.

Kullanım (herhangi bir python): vtt_kontrol.py <tts_klasörü> <manifest.csv> <kaynak_adı> <altyazı.vtt> [eşik]
- Parçanın kaynak videodaki zamanı manifest'ten (baslangic_sn, sure_sn) alınır; ±1,5 sn pencerede altyazı kelimeleri toplanır.
- kapsama = Whisper kelimelerinden altyazı penceresinde (bulanık) bulunanların oranı.
- kapsama < eşik (vars. 0,6) ise parça metadata.csv'den çıkarılır (vtt_elenen.csv'ye yazılır). Orijinal: metadata_ham.csv.
- Bu kaynağa ait olmayan parçalar (ör. başka bir videodan gelen bölümler) olduğu gibi kalır.
"""
import csv, os, re, shutil, sys
from difflib import SequenceMatcher

tts, manifest, kaynak, vtt = sys.argv[1:5]
esik = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6


def kelimeler(t):
    t = t.replace("İ", "i").replace("I", "ı").lower()
    return re.findall(r"[0-9a-zçğıöşüâîû]+", re.sub(r"[’'`]", "", t))


def sn(x):
    h, m, s = x.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)


cues = []
for blok in open(vtt, encoding="utf-8").read().split("\n\n"):
    m = re.search(r"(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)[^\n]*\n(.+)", blok, re.S)
    if m:
        cues.append((sn(m.group(1)), sn(m.group(2)), kelimeler(re.sub(r"<[^>]+>", "", m.group(3)))))
man = {r["dosya"]: r for r in csv.DictReader(open(manifest))}

ham = os.path.join(tts, "metadata_ham.csv"); meta = os.path.join(tts, "metadata.csv")
if not os.path.exists(ham):
    shutil.copy(meta, ham)
rows = list(csv.DictReader(open(ham), delimiter="|"))
kalan, elenen, sayac = [], [], 0
for r in rows:
    ad = os.path.basename(r["audio_file"]); m = man.get(ad)
    if not m or m["kaynak"] != kaynak:
        kalan.append(r); continue
    sayac += 1
    a = float(m["baslangic_sn"]); b = a + float(m["sure_sn"])
    pencere = [w for s, e, ws in cues if e >= a - 1.5 and s <= b + 1.5 for w in ws]
    wh = kelimeler(r["text"])
    bulunan = sum(1 for w in wh if any(w == p or SequenceMatcher(None, w, p).ratio() >= 0.75 for p in pencere))
    kapsama = bulunan / max(len(wh), 1)
    if kapsama < esik:
        elenen.append((r["audio_file"], round(kapsama, 2), r["text"], " ".join(pencere)))
    else:
        kalan.append(r)
with open(meta, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|"); w.writeheader(); w.writerows(kalan)
with open(os.path.join(tts, f"vtt_elenen_{kaynak}.csv"), "w", newline="") as f:
    csv.writer(f, delimiter="|").writerows(elenen)
print(f"{kaynak}: {sayac} parça kontrol edildi, {len(elenen)} elendi (kapsama<{esik}); metadata: {len(kalan)} satır")
for e in elenen[:5]:
    print("  ", e[1], "| whisper:", e[2][:60], "| altyazı:", e[3][:60])
