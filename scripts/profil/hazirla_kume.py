"""Kullanım: hazirla.py <vocals.wav> <profil_klasörü> <ön_ek>
Sessizlikten konuşma parçaları (<=12 sn) -> XTTS konuşmacı gömmesi (CPU) -> kümeleme -> baskın konuşmacı merkezi.
Merkeze benzerliği >=0,80 olan parçalar dataset/wavs_44k/ altına (44,1 kHz mono), manifest.csv, ref/ref.wav, ref/centroid.npy."""
import os, sys, json, subprocess, csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from xtts_ortak import tts
import numpy as np, soundfile as sf, torch, torchaudio
voc, prof, pre = sys.argv[1:4]
m = tts().synthesizer.tts_model
audio, sr = sf.read(voc, dtype="float32")
if audio.ndim > 1: audio = audio.mean(1)
a16 = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000)
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", voc, "-af", "silencedetect=noise=-40dB:d=0.3", "-f", "null", "-"], capture_output=True, text=True).stderr
spans, end = [], 0.0
for line in r.splitlines():
    if "silence_start:" in line: spans.append((end, float(line.split("silence_start:")[1].split()[0])))
    elif "silence_end:" in line: end = float(line.split("silence_end:")[1].split()[0].strip("|"))
spans.append((end, len(audio) / sr))
chunks = []
for a, b in spans:
    d = b - a
    if d < 1.5: continue
    n = max(1, int(np.ceil(d / 12))); st = d / n
    chunks += [(a + i * st, st) for i in range(n)]
E = []
for a, d in chunks:
    with torch.inference_mode():
        v = m.get_speaker_embedding(a16[int(a*16000):int((a+d)*16000)].unsqueeze(0), 16000).flatten().float().numpy()
    E.append(v / np.linalg.norm(v))
E = np.stack(E)
cl = [[i] for i in range(len(E))]; C = [E[i].copy() for i in range(len(E))]
while len(cl) > 1:
    M = np.stack([c / np.linalg.norm(c) for c in C]); S = M @ M.T; np.fill_diagonal(S, -1)
    i, j = np.unravel_index(S.argmax(), S.shape)
    if S[i, j] < 0.78: break
    if i > j: i, j = j, i
    cl[i] += cl.pop(j); C[i] = C[i] + C.pop(j)
cl.sort(key=lambda c: -sum(chunks[i][1] for i in c))
cen = E[cl[0]].mean(0); cen /= np.linalg.norm(cen)
sims = E @ cen
os.makedirs(f"{prof}/dataset/wavs_44k", exist_ok=True); os.makedirs(f"{prof}/ref", exist_ok=True)
np.save(f"{prof}/ref/centroid.npy", cen)
rows = []
for k, ((a, d), s) in enumerate(zip(chunks, sims)):
    if s < 0.80: continue
    name = f"{pre}_{k:05d}_{a:.1f}.wav"
    seg = audio[int(a*sr):int((a+d)*sr)]
    seg = torchaudio.functional.resample(torch.from_numpy(seg), sr, 44100).numpy() if sr != 44100 else seg
    sf.write(f"{prof}/dataset/wavs_44k/{name}", seg, 44100, subtype="PCM_16")
    rows.append([name, pre, round(a, 2), round(d, 2), round(float(s), 3)])
with open(f"{prof}/dataset/manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dosya", "kaynak", "baslangic_sn", "sure_sn", "hedefe_benzerlik"]); w.writerows(rows)
best = max((i for i in range(len(chunks)) if chunks[i][1] >= 8), key=lambda i: sims[i], default=int(np.argmax(sims)))
a, d = chunks[best]
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-t", str(d), "-i", voc, "-ac", "1", "-ar", "24000", f"{prof}/ref/ref.wav"])
ozet = {"toplam_konusma_dk": round(sum(d for _, d in chunks) / 60, 1), "kume_sayisi": len(cl),
        "kumeler_dk": [round(sum(chunks[i][1] for i in c) / 60, 1) for c in cl[:6]],
        "secilen_parca": len(rows), "secilen_dk": round(sum(r[3] for r in rows) / 60, 1),
        "benzerlik_medyan": round(float(np.median([r[4] for r in rows])), 3), "ref": [round(a, 1), round(d, 1), round(float(sims[best]), 3)]}
json.dump(ozet, open(f"{prof}/dataset/ozet.json", "w"), ensure_ascii=False, indent=1); print(ozet)
