"""Sesten sese dönüştürme (VC) motorları için ortak arayüz."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseVCEngine(ABC):
    """RVC / Applio / Seed-VC motorları için ortak arayüz."""

    name: str = "base"

    @abstractmethod
    def convert(
        self,
        source_wav: Path,
        output_wav: Path,
        pitch_shift: int = 0,
        **kwargs,
    ) -> Path:
        """Kaynak sesi hedef ses profiline dönüştürür."""

