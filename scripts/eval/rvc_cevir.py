"""Kullanım: rvc_cevir.py <model.pth> <index|-> <girdi_klasör> <çıktı_klasör> [index_rate protect pitch]
GPU eğitimde meşgul olduğu için çıkarım CPU'da yapılır."""
import os, sys, glob
import torch
torch.xpu.is_available = lambda: False          # XPU'yu gizle -> CPU
torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "3")))
APPLIO = os.environ.get("APPLIO_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "Applio"))
ARGS = [os.path.abspath(a) if i < 4 and a != "-" else a for i, a in enumerate(sys.argv[1:])]
os.chdir(APPLIO); sys.path.insert(0, APPLIO)
from rvc.infer.infer import VoiceConverter
pth, idx, src, dst = ARGS[:4]
ir, pr, pitch = (float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (0.75, 0.5, 0)
vol = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0  # volume_envelope: 1 = hedef modelin ses zarfı, <1 kaynağa karışır
os.makedirs(dst, exist_ok=True)
vc = VoiceConverter()
for f in sorted(glob.glob(f"{src}/*.wav")):
    vc.convert_audio(audio_input_path=f, audio_output_path=f"{dst}/{os.path.basename(f)}", model_path=pth,
                     index_path="" if idx == "-" else idx, pitch=pitch, f0_method="rmvpe",
                     index_rate=ir if idx != "-" else 0.0, protect=pr, volume_envelope=vol, export_format="WAV")
print("tamam", dst)
