"""XTTS + Whisper doğrulamalı sentez: hatalı parçayı farklı tohumla yeniden üretir, fazlalığı kırpar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import soundfile as sf

from core.tts.dogrulama import Kelime, degerlendir, kirp


class SentezMotoru(Protocol):
    def sentezle(self, text: str, **kw) -> str: ...


class KelimeMotoru(Protocol):
    def kelimeler(self, wav_path: str) -> list[Kelime]: ...


@dataclass
class GuvenliSonuc:
    wav_path: str
    kabul: bool
    deneme_sayisi: int
    hata_sayisi: int
    duyulan: str


def guvenli_sentezle(metin: str, xtts: SentezMotoru, whisper: KelimeMotoru, *, deneme: int = 3,
                     temperature: float = 0.65, repetition_penalty: float = 5.0, tohum: int = 1000,
                     **xtts_kw) -> GuvenliSonuc:
    en_iyi = None
    for i in range(deneme):
        wav_path = xtts.sentezle(metin, temperature=temperature, repetition_penalty=repetition_penalty,
                                 seed=tohum + i, **xtts_kw)
        info = sf.info(wav_path)
        d = degerlendir(metin, whisper.kelimeler(wav_path), info.duration)
        if en_iyi is None or d.hata_sayisi < en_iyi[1].hata_sayisi:
            en_iyi = (wav_path, d, i + 1)
        if d.kabul:
            break
    wav_path, d, n = en_iyi
    a, sr = sf.read(wav_path, dtype="float32")
    out = str(Path(wav_path).with_name(Path(wav_path).stem + "_dogrulandi.wav"))
    sf.write(out, kirp(a, sr, d.kirp_bas, d.kirp_son), sr)
    return GuvenliSonuc(out, d.kabul, i + 1, d.hata_sayisi, d.duyulan)
