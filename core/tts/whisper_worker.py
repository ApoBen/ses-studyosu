"""faster-whisper kalıcı worker (envs/whisper içinde çalışır).
stdin: {"wav_path": "..."} satırları -> stdout: {"kelimeler": [[kelime, bas, son, olasilik], ...], "metin": "..."}"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--download-root", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    from faster_whisper import WhisperModel

    m = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads, download_root=a.download_root)
    print(json.dumps({"hazir": True}), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req.get("command") == "shutdown":
                break
            segs, _ = m.transcribe(req["wav_path"], language=req.get("language", "tr"), word_timestamps=True,
                                   beam_size=5, vad_filter=False, condition_on_previous_text=False)
            words = [[w.word.strip(), round(w.start, 3), round(w.end, 3), round(w.probability, 3)]
                     for s in segs for w in s.words]
            resp = {"kelimeler": words, "metin": " ".join(w[0] for w in words)}
        except Exception as e:  # noqa: BLE001
            resp = {"error": str(e)}
        print(json.dumps(resp, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
