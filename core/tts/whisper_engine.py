"""Whisper (faster-whisper, CPU int8) istemcisi: kalıcı worker'ı envs/whisper içinde çalıştırır."""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from core.tts.dogrulama import Kelime

KOK = Path(__file__).resolve().parents[2]
WHISPER_PYTHON = KOK / "envs" / "whisper" / "bin" / "python"
WORKER = Path(__file__).with_name("whisper_worker.py")


class WhisperEngine:
    def __init__(self, model: str = "large-v3-turbo", python: Path | None = None,
                 download_root: Path | None = None, threads: int = 4) -> None:
        self.cmd = [str(python or WHISPER_PYTHON), "-u", str(WORKER), "--model", model,
                    "--download-root", str(download_root or KOK / "models" / "whisper"), "--threads", str(threads)]
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _baslat(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                      encoding="utf-8", bufsize=1)
        hazir = self._proc.stdout.readline()
        if not hazir or not json.loads(hazir).get("hazir"):
            raise RuntimeError("Whisper worker başlatılamadı.")

    def kelimeler(self, wav_path: str, language: str = "tr") -> list[Kelime]:
        with self._lock:
            self._baslat()
            self._proc.stdin.write(json.dumps({"wav_path": str(wav_path), "language": language}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                self._proc = None
                raise RuntimeError("Whisper worker kapandı.")
            r = json.loads(line)
            if "error" in r:
                raise RuntimeError(f"Whisper hatası: {r['error']}")
            return [Kelime(*w) for w in r["kelimeler"]]

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n"); self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        self._proc = None
