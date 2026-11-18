"""TTS motorları için ortak arayüz ve öbek-bazlı sentez (Yol 1) yardımcıları."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from core.text.prosody_planner import ProsodyPlan


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    plan: ProsodyPlan


class BaseTTSEngine(ABC):
    """Tüm TTS motorlarının (XTTS-v2, Chatterbox, F5-TTS vb.) uyguladığı ortak arayüz."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Modeli belleğe/GPU'ya yükler."""

    @abstractmethod
    def synthesize_phrase(
        self,
        text: str,
        speaker_wav: Path | None = None,
        language: str = "tr",
        speed: float = 1.0,
        emotion: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Tek bir prozodik öbeği sentezler ve (float32 mono audio, sample_rate) döner."""

    def synthesize_plan(
        self,
        plan: ProsodyPlan,
        speaker_wav: Path | None = None,
        language: str = "tr",
    ) -> SynthesisResult:
        """Yol 1: Prozodi planındaki her öbeği ayrı sentezleyip planlanan duraklamalarla birleştirir."""
        chunks: list[np.ndarray] = []
        sample_rate = 24000

        for phrase in plan.phrases:
            if not phrase.text.strip():
                continue
            wav, sr = self.synthesize_phrase(
                text=phrase.text,
                speaker_wav=speaker_wav,
                language=language,
                speed=phrase.rate,
                emotion=phrase.emotion,
            )
            sample_rate = sr
            if wav.size > 0:
                chunks.append(wav.astype(np.float32))
            if phrase.pause_after_ms > 0:
                silence_len = int(sample_rate * (phrase.pause_after_ms / 1000.0))
                chunks.append(np.zeros(silence_len, dtype=np.float32))

        if not chunks:
            return SynthesisResult(audio=np.zeros(0, dtype=np.float32), sample_rate=sample_rate, plan=plan)

        combined = np.concatenate(chunks)
        return SynthesisResult(audio=combined, sample_rate=sample_rate, plan=plan)

