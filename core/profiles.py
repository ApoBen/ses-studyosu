"""Ses profili yönetimi (profiles/<ad>/profile.json, ref/, dataset/, rvc/, tts/)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"


@dataclass
class ConsentRecord:
    granted_by: str
    granted_at: str
    scope: str = "personal_local_only"


@dataclass
class VoiceProfile:
    name: str
    language: str = "tr"
    sample_rate: int = 24000
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    models: dict[str, str] = field(default_factory=dict)
    consent: ConsentRecord | None = None

    @property
    def root(self) -> Path:
        return PROFILES_DIR / self.name

    @property
    def ref_dir(self) -> Path:
        return self.root / "ref"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def rvc_dir(self) -> Path:
        return self.root / "rvc"

    @property
    def tts_dir(self) -> Path:
        return self.root / "tts"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.ref_dir, self.dataset_dir, self.rvc_dir, self.tts_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self) -> Path:
        self.ensure_dirs()
        meta_path = self.root / "profile.json"
        meta_path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def list_reference_wavs(self) -> list[Path]:
        if not self.ref_dir.exists():
            return []
        return sorted(self.ref_dir.glob("*.wav"))


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.name for p in PROFILES_DIR.iterdir() if p.is_dir() and (p / "profile.json").exists())


def load_profile(name: str) -> VoiceProfile:
    meta_path = PROFILES_DIR / name / "profile.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    consent_raw = data.pop("consent", None)
    consent = ConsentRecord(**consent_raw) if consent_raw else None
    return VoiceProfile(**data, consent=consent)

