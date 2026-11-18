"""XTTS (CPU) yükleme ve konuşmacı gömmesi. Benzerlik merkezi EVAL_CENTROID ortam değişkeniyle verilir (profiles/<profil>/ref/centroid.npy)."""
import os
os.environ.setdefault("COQUI_TOS_AGREED", "1")
import numpy as np, soundfile as sf, torch, torchaudio
torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "3")))
from TTS.api import TTS
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CENTROID = np.load(os.environ["EVAL_CENTROID"]) if os.environ.get("EVAL_CENTROID") else None
_tts = None
def tts():
    global _tts
    if _tts is None:
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    return _tts
def emb(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    a16 = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).unsqueeze(0)
    with torch.inference_mode():
        v = tts().synthesizer.tts_model.get_speaker_embedding(a16, 16000).flatten().float().numpy()
    return v / np.linalg.norm(v)
