"""XTTS-v2 (coqui-tts) motor sarmalayıcısı.

İzole `envs/xtts` ortamında kalıcı bir worker süreci (`xtts_worker.py`)
başlatır ve stdin/stdout üzerinden satır bazlı JSON ile haberleşir.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import threading
import numpy as np
import soundfile as sf

from core.tts.base import BaseTTSEngine

DEFAULT_XTTS_PYTHON = Path(__file__).resolve().parents[2] / "envs" / "xtts" / "bin" / "python"
WORKER_PATH = Path(__file__).resolve().with_name("xtts_worker.py")
XTTS_MAX_CHARS_TR = 250


def split_text_for_xtts(text: str, max_chars: int = XTTS_MAX_CHARS_TR) -> list[str]:
    """XTTS Türkçe karakter sınırı (~250) için metni kelime sınırlarından böler."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in cleaned.split(" "):
        if len(word) > max_chars:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_len = 0
            for i in range(0, len(word), max_chars):
                chunks.append(word[i : i + max_chars])
            continue

        added_len = len(word) if not current_words else len(word) + 1
        if current_len + added_len <= max_chars:
            current_words.append(word)
            current_len += added_len
        else:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


class XTTSEngine(BaseTTSEngine):
    """XTTS-v2 TTS motoru (idiap/coqui-ai-TTS)."""

    name = "xtts-v2"

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str | None = None,
        xtts_python: Path | None = None,
        gpt_checkpoint: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.gpt_checkpoint = str(gpt_checkpoint) if gpt_checkpoint else None
        self.device = device
        self.xtts_python = Path(xtts_python) if xtts_python is not None else DEFAULT_XTTS_PYTHON
        self._tts = None
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _start_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.xtts_python.exists():
            raise RuntimeError(
                "coqui-tts kurulu değil. İzole ortam için `envs/xtts` kurulumunu yapın."
            )
        cmd = self._worker_cmd()

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def _worker_cmd(self) -> list[str]:
        cmd = [str(self.xtts_python), "-u", str(WORKER_PATH), "--model", self.model_name]
        if self.device:
            cmd.extend(["--device", self.device])
        if self.gpt_checkpoint:
            cmd.extend(["--gpt-checkpoint", self.gpt_checkpoint])
        return cmd

    def load(self) -> None:
        if self._tts is not None:
            return
        if self._proc is not None and self._proc.poll() is None:
            return

        # Eğer xtts_python özel olarak verilmediyse ve mevcut ortamda TTS varsa doğrudan kullan
        if self.xtts_python == DEFAULT_XTTS_PYTHON and not self.gpt_checkpoint:
            try:
                from TTS.api import TTS  # type: ignore

                target_device = self.device or "cpu"
                self._tts = TTS(self.model_name).to(target_device)
                return
            except ImportError:
                pass

        self._start_worker()

    def _request_worker(
        self,
        text: str,
        speaker_wav: Path,
        language: str,
        speed: float,
        **extra,
    ) -> tuple[np.ndarray, int]:
        with self._lock:
            self._start_worker()
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            payload = {
                "text": text,
                "speaker_wav": str(speaker_wav),
                "language": language,
                "speed": float(speed),
                **extra,
            }
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()

            resp_line = self._proc.stdout.readline()
            if not resp_line:
                rc = self._proc.poll()
                self._proc = None
                raise RuntimeError(f"XTTS worker beklenmedik şekilde kapandı (kod={rc}).")

            resp = json.loads(resp_line)
            if "error" in resp:
                raise RuntimeError(f"XTTS worker hatası: {resp['error']}")

            out_path = Path(resp["wav_path"])
            try:
                audio, sr = sf.read(out_path, dtype="float32")
            finally:
                out_path.unlink(missing_ok=True)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), int(sr)

    def synthesize_phrase(
        self,
        text: str,
        speaker_wav: Path | None = None,
        language: str = "tr",
        speed: float = 1.0,
        emotion: str | None = None,
    ) -> tuple[np.ndarray, int]:
        if speaker_wav is None or not Path(speaker_wav).exists():
            raise ValueError("XTTS-v2 için geçerli bir referans ses dosyası (speaker_wav) gereklidir.")

        chunks = split_text_for_xtts(text)
        if not chunks:
            return np.zeros(0, dtype=np.float32), 24000

        self.load()

        audio_parts: list[np.ndarray] = []
        sample_rate = 24000

        for chunk in chunks:
            if self._tts is not None:
                wav = self._tts.tts(
                    text=chunk,
                    speaker_wav=str(speaker_wav),
                    language=language,
                    speed=speed,
                )
                part = np.asarray(wav, dtype=np.float32)
                sr = 24000
            else:
                part, sr = self._request_worker(
                    text=chunk,
                    speaker_wav=Path(speaker_wav),
                    language=language,
                    speed=speed,
                )
            sample_rate = sr
            if part.size > 0:
                audio_parts.append(part)

        if not audio_parts:
            return np.zeros(0, dtype=np.float32), sample_rate
        if len(audio_parts) == 1:
            return audio_parts[0], sample_rate
        return np.concatenate(audio_parts), sample_rate

    def sentezle(self, text: str, speaker_wav: Path | str, language: str = "tr", speed: float = 1.0,
                 out_dir: Path | str | None = None, **ornekleme) -> str:
        """Tek parçayı (bölmeden) worker ile sentezler; wav yolunu döner.
        ornekleme: temperature, repetition_penalty, top_p, top_k, length_penalty, seed (guvenli_sentez için)."""
        part, sr = self._request_worker(text=text, speaker_wav=Path(speaker_wav), language=language,
                                        speed=speed, **ornekleme)
        d = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="xtts_"))
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"parca_{abs(hash((text, str(ornekleme)))) % 10**8}.wav"
        sf.write(out, part, sr)
        return str(out)

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
        self._proc = None

    def __enter__(self) -> XTTSEngine:
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
