"""XTTS motoru, kalıcı worker JSON protokolü, önbellek ve metin bölme testleri."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pytest
import soundfile as sf

from core.tts.xtts_engine import XTTSEngine, split_text_for_xtts
import core.tts.xtts_worker as xtts_worker


def test_split_text_for_xtts_word_boundaries():
    short = "Merhaba, bugün hava çok güzel."
    assert split_text_for_xtts(short, max_chars=250) == [short]
    assert split_text_for_xtts("   ", max_chars=250) == []

    words = [f"kelime{i}" for i in range(60)]
    long_text = " ".join(words)
    assert len(long_text) > 250

    chunks = split_text_for_xtts(long_text, max_chars=250)
    assert len(chunks) >= 2
    assert all(len(c) <= 250 for c in chunks)
    assert " ".join(chunks) == long_text


def test_xtts_engine_persistent_worker_and_chunking(tmp_path: Path, monkeypatch):
    ref_wav = tmp_path / "ref.wav"
    sf.write(ref_wav, np.zeros(1600, dtype=np.float32), 24000)

    start_counter = tmp_path / "starts.txt"
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        f"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

with open({str(start_counter)!r}, "a", encoding="utf-8") as f:
    f.write("start\\n")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("text") == "FAIL":
        print(json.dumps({{"error": "sentetik hata"}}), flush=True)
        continue
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)
    # 0.05 sn sabit örnek üret
    sf.write(out_path, np.ones(1200, dtype=np.float32) * 0.25, 24000)
    print(json.dumps({{"wav_path": str(out_path)}}), flush=True)
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("core.tts.xtts_engine.WORKER_PATH", fake_worker)

    engine = XTTSEngine(device="cpu", xtts_python=Path(sys.executable))
    try:
        audio1, sr1 = engine.synthesize_phrase("Birinci cümle.", speaker_wav=ref_wav, language="tr")
        audio2, sr2 = engine.synthesize_phrase("İkinci cümle.", speaker_wav=ref_wav, language="tr")
        assert sr1 == 24000 and sr2 == 24000
        assert len(audio1) == 1200
        assert len(audio2) == 1200

        # Worker yalnızca 1 kez başlatılmış olmalı
        assert start_counter.read_text(encoding="utf-8").splitlines() == ["start"]

        # 250 karakterden uzun metin bölünerek 2+ istekte sentezlenip birleştirilmeli
        long_text = " ".join([f"deneme{i}" for i in range(50)])
        assert len(long_text) > 250
        audio_long, sr_long = engine.synthesize_phrase(long_text, speaker_wav=ref_wav, language="tr")
        assert sr_long == 24000
        assert len(audio_long) == 1200 * len(split_text_for_xtts(long_text, max_chars=250))

        # Hata yanıtı RuntimeError yükseltmeli
        with pytest.raises(RuntimeError, match="sentetik hata"):
            engine.synthesize_phrase("FAIL", speaker_wav=ref_wav, language="tr")
    finally:
        engine.close()


def test_xtts_worker_caches_speaker_latents(tmp_path: Path):
    ref1 = tmp_path / "ref1.wav"
    ref2 = tmp_path / "ref2.wav"
    sf.write(ref1, np.zeros(1600, dtype=np.float32), 24000)
    sf.write(ref2, np.zeros(1600, dtype=np.float32), 24000)

    class FakeModel:
        def __init__(self):
            self.cond_calls: list[str] = []
            self.infer_calls: list[dict] = []

        def get_conditioning_latents(self, audio_path):
            path_str = audio_path[0] if isinstance(audio_path, list) else audio_path
            self.cond_calls.append(path_str)
            return ("gpt_latent", f"spk_{len(self.cond_calls)}")

        def inference(self, text, language, gpt_cond_latent, speaker_embedding, speed=1.0, **kwargs):
            self.infer_calls.append({"text": text, "language": language, "speed": speed})
            return {"wav": np.zeros(1200, dtype=np.float32)}

    fake_model = FakeModel()
    cache: dict = {}

    r1 = xtts_worker.handle_request(
        {"text": "Merhaba", "speaker_wav": str(ref1), "language": "tr", "speed": 1.0},
        model=fake_model,
        latent_cache=cache,
    )
    r2 = xtts_worker.handle_request(
        {"text": "Nasılsın", "speaker_wav": str(ref1), "language": "tr", "speed": 1.1},
        model=fake_model,
        latent_cache=cache,
    )
    r3 = xtts_worker.handle_request(
        {"text": "İyiyim", "speaker_wav": str(ref2), "language": "tr", "speed": 1.0},
        model=fake_model,
        latent_cache=cache,
    )

    try:
        assert "wav_path" in r1 and Path(r1["wav_path"]).exists()
        assert "wav_path" in r2 and Path(r2["wav_path"]).exists()
        assert "wav_path" in r3 and Path(r3["wav_path"]).exists()
        # ref1 için get_conditioning_latents yalnızca 1 kez, ref2 için 1 kez çağrılmalı
        assert len(fake_model.cond_calls) == 2
        assert len(fake_model.infer_calls) == 3
    finally:
        for resp in (r1, r2, r3):
            if "wav_path" in resp:
                Path(resp["wav_path"]).unlink(missing_ok=True)
