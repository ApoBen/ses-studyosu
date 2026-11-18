"""Test cümlelerini bir kez XTTS (CPU) ile referans seste üretir: <çıktı_klasör>/NN.wav"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xtts_ortak import tts, ROOT
# Kullanım: xtts_test_uret.py <cumleler.txt> <ref.wav> <çıktı_klasör>
if len(sys.argv) < 4:
    sys.exit(__doc__ + "\nKullanım: xtts_test_uret.py <cumleler.txt> <ref.wav> <çıktı_klasör>")
cum, ref, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
for i, s in enumerate(open(cum).read().split("\n")):
    s = s.strip()
    if s and not os.path.exists(f"{out}/{i:02d}.wav"):
        tts().tts_to_file(text=s, speaker_wav=ref, language="tr", file_path=f"{out}/{i:02d}.wav")
        print("üretildi", i, flush=True)
