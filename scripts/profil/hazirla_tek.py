"""Tek karakterli replik kayıtları için veri hazırlığı (kümeleme yerine).
Kullanım: hazirla_tek.py <vocals.wav> <profil_klasörü> <ön_ek> --istenmeyen BAS-SON [--min-sure 1.0] [--aykiri 0.45]

Replikler kısa ve duygusal olarak çok değişken (bağırma, gülme, efekt): kümeleme aynı kişiyi parçalıyor.
- Parçalar: sessizlikten bölünür (d=0.25 sn), <=12 sn.
- İstenmeyen ses (ör. arka plandaki başka bir konuşmacı) bilinen bir zaman aralığından örneklenir; ona karakterden daha yakın parçalar elenir.
- Karakter merkezi: istenmeyen olmayan parçaların ortalaması (2 tur); merkeze benzerliği < aykiri olanlar (efekt/gürültü) elenir.
- ref.wav: merkeze en yakın parçalar birleştirilerek ~12 sn.
"""
import argparse, csv, json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from xtts_ortak import tts
import numpy as np, soundfile as sf, torch, torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("voc"); ap.add_argument("prof"); ap.add_argument("pre")
ap.add_argument("--istenmeyen", default="", help="istenmeyen sesin bilinen aralığı, ör. 0.08-3.9 (boşsa sadece aykırı elemesi)")
ap.add_argument("--min-sure", type=float, default=1.0)
ap.add_argument("--aykiri", type=float, default=0.45)
a = ap.parse_args()
m = tts().synthesizer.tts_model
audio, sr = sf.read(a.voc, dtype="float32")
if audio.ndim > 1: audio = audio.mean(1)
a16 = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000)


def emb(s, e):
    with torch.inference_mode():
        v = m.get_speaker_embedding(a16[int(s * 16000):int(e * 16000)].unsqueeze(0), 16000).flatten().float().numpy()
    return v / np.linalg.norm(v)


r = subprocess.run(["ffmpeg", "-hide_banner", "-i", a.voc, "-af", "silencedetect=noise=-40dB:d=0.25", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
spans, end = [], 0.0
for line in r.splitlines():
    if "silence_start:" in line: spans.append((end, float(line.split("silence_start:")[1].split()[0])))
    elif "silence_end:" in line: end = float(line.split("silence_end:")[1].split()[0].strip("|"))
spans.append((end, len(audio) / sr))
chunks = []
for s, e in spans:
    d = e - s
    if d < a.min_sure: continue
    n = max(1, int(np.ceil(d / 12))); st = d / n
    chunks += [(s + i * st, st) for i in range(n)]
E = np.stack([emb(s, s + d) for s, d in chunks])
if a.istenmeyen.strip():
    i0, i1 = map(float, a.istenmeyen.split("-"))
    sim_ist = E @ emb(i0, i1)
else:  # istenmeyen ses yok: tüm parçalar karakter adayı, sadece aykırılar elenir
    sim_ist = np.full(len(E), -1.0)
cen = E.mean(0); cen /= np.linalg.norm(cen)
for _ in range(2):  # istenmeyene daha yakın olanları dışarıda bırakarak merkezi yenile
    karakter = (E @ cen) > sim_ist
    cen = E[karakter].mean(0); cen /= np.linalg.norm(cen)
sim = E @ cen
tut = (sim > sim_ist) & (sim >= a.aykiri)
os.makedirs(f"{a.prof}/dataset/wavs_44k", exist_ok=True); os.makedirs(f"{a.prof}/ref", exist_ok=True)
np.save(f"{a.prof}/ref/centroid.npy", cen)
rows = []
for k, ((s, d), ok) in enumerate(zip(chunks, tut)):
    if not ok: continue
    name = f"{a.pre}_{k:05d}_{s:.1f}.wav"
    seg = audio[int(s * sr):int((s + d) * sr)]
    if sr != 44100: seg = torchaudio.functional.resample(torch.from_numpy(seg), sr, 44100).numpy()
    sf.write(f"{a.prof}/dataset/wavs_44k/{name}", seg, 44100, subtype="PCM_16")
    rows.append([name, a.pre, round(s, 2), round(d, 2), round(float(sim[k]), 3)])
with open(f"{a.prof}/dataset/manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dosya", "kaynak", "baslangic_sn", "sure_sn", "hedefe_benzerlik"]); w.writerows(rows)
with open(f"{a.prof}/dataset/elenen.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["baslangic_sn", "sure_sn", "karakter_benzerlik", "istenmeyen_benzerlik"])
    for k, ((s, d), ok) in enumerate(zip(chunks, tut)):
        if not ok: w.writerow([round(s, 2), round(d, 2), round(float(sim[k]), 3), round(float(sim_ist[k]), 3)])
# ref: merkeze en yakın parçalardan ~12 sn (aralarına 0,3 sn sessizlik)
sec, top = [], 0.0
for k in np.argsort(-sim):
    if tut[k] and chunks[k][1] >= 1.5:
        sec.append(k); top += chunks[k][1]
        if top >= 12: break
parcalar = []
for k in sorted(sec):
    s, d = chunks[k]; parcalar += [audio[int(s * sr):int((s + d) * sr)], np.zeros(int(0.3 * sr), dtype=np.float32)]
ref = torchaudio.functional.resample(torch.from_numpy(np.concatenate(parcalar)), sr, 24000).numpy()
sf.write(f"{a.prof}/ref/ref.wav", ref, 24000)
ozet = {"yontem": "tek karakter (istenmeyen ses elemesi)", "toplam_konusma_dk": round(sum(d for _, d in chunks) / 60, 1),
        "parca": len(chunks), "istenmeyen_elenen": int((sim <= sim_ist).sum()), "aykiri_elenen": int(((sim > sim_ist) & (sim < a.aykiri)).sum()),
        "secilen_parca": len(rows), "secilen_dk": round(sum(r[3] for r in rows) / 60, 1),
        "benzerlik_medyan": round(float(np.median([r[4] for r in rows])), 3), "ref_sn": round(len(ref) / 24000, 1),
        "ref_benzerlik": round(float(np.mean(sim[sec])), 3)}
json.dump(ozet, open(f"{a.prof}/dataset/ozet.json", "w"), ensure_ascii=False, indent=1); print(ozet)
