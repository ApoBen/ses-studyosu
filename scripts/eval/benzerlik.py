"""Kullanım: benzerlik.py <klasör>...  -> her klasör için hedef merkeze ortalama kosinüs benzerliği (JSON satırı)."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xtts_ortak import emb, CENTROID
import numpy as np
for d in sys.argv[1:]:
    s = [float(emb(f) @ CENTROID) for f in sorted(glob.glob(f"{d}/*.wav"))]
    print(json.dumps({"klasor": d, "ort": round(float(np.mean(s)), 4), "min": round(min(s), 4), "n": len(s), "hata": round(float(np.std(s, ddof=1) / np.sqrt(len(s))), 4)}), flush=True)
