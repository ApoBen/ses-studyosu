"""XTTS ince ayar verisi: her parçayı 22,05 kHz mono'ya çevirir ve Whisper ile yazıya döker.
Kullanım (envs/whisper): transkript.py <wav_klasörü> <çıktı_klasörü> <konuşmacı_adı>
Çıktı: <çıktı>/wavs/*.wav, <çıktı>/metadata.csv (coqui biçimi: audio_file|text|speaker_name), <çıktı>/elenen.csv"""
import csv, glob, os, subprocess, sys, time
from faster_whisper import WhisperModel
src, out, spk = sys.argv[1:4]
os.makedirs(f"{out}/wavs", exist_ok=True)
m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4,
                 download_root=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "whisper"))
meta_p = f"{out}/metadata.csv"
done = set()
if os.path.exists(meta_p):
    done = {r["audio_file"] for r in csv.DictReader(open(meta_p), delimiter="|")}
else:
    open(meta_p, "w").write("audio_file|text|speaker_name\n")
meta = open(meta_p, "a"); rej = open(f"{out}/elenen.csv", "a")
files = sorted(glob.glob(f"{src}/*.wav")); t0 = time.time()
for k, f in enumerate(files):
    rel = f"wavs/{os.path.basename(f)}"
    if rel in done: continue
    dst = f"{out}/{rel}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f, "-ac", "1", "-ar", "22050", dst], check=True)
    segs, _ = m.transcribe(dst, language="tr", beam_size=5, vad_filter=False, condition_on_previous_text=False)
    segs = list(segs)
    text = " ".join(s.text.strip() for s in segs).strip()
    lp = min((s.avg_logprob for s in segs), default=-9); ns = max((s.no_speech_prob for s in segs), default=1)
    if not text or lp < -1.0 or ns > 0.6 or len(text) > 250:
        rej.write(f"{rel}|{lp:.2f}|{ns:.2f}|{text}\n"); rej.flush(); os.remove(dst); continue
    meta.write(f"{rel}|{text.replace('|', ' ')}|{spk}\n"); meta.flush()
    if k % 25 == 0: print(f"{k}/{len(files)} {time.time()-t0:.0f}s: {text[:70]}", flush=True)
print("bitti", len(files), "dosya", f"{time.time()-t0:.0f}s")
