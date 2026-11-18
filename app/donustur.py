"""RVC (Applio) ile ses kaydını hedef ses profiline dönüştürme katmanı (CPU)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from app.profiller import VARSAYILAN_KOK, profil_bilgisi_getir

APPLIO_EVAL_CONFIG_DIR = "/tmp/applio_eval_config"


def _varsayilan_komut_calistir(cmd: list[str], env: dict[str, str]) -> None:
    cfg_dir = Path(env.get("APPLIO_CONFIG_DIR", APPLIO_EVAL_CONFIG_DIR))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text('{"precision": "fp32"}', encoding="utf-8")
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)


class RvcDonusturucu:
    """Seçilen profilin RVC modelini (`ayarlar.json` parametreleriyle) CPU üzerinde çalıştırır."""

    def __init__(
        self,
        kok_dizin: Path | str = VARSAYILAN_KOK,
        applio_python: Path | None = None,
        rvc_cevir_betik: Path | None = None,
        komut_calistirici: Callable[[list[str], dict[str, str]], None] | None = None,
    ) -> None:
        self.kok = Path(kok_dizin)
        self.applio_python = applio_python or (self.kok / "envs" / "applio" / "bin" / "python")
        self.rvc_cevir_betik = rvc_cevir_betik or (self.kok / "scripts" / "eval" / "rvc_cevir.py")
        self._calistir = komut_calistirici or _varsayilan_komut_calistir

    def donustur(
        self,
        kaynak_ses: Path | str,
        profil: str,
        pitch: int | None = None,
        cikti_klasor: Path | str | None = None,
    ) -> Path:
        """Girdi ses dosyasını seçilen profilin sesine dönüştürür ve çıktı `.wav` yolunu döner."""
        kaynak = Path(kaynak_ses)
        if not kaynak.exists():
            raise FileNotFoundError(f"Dönüştürülecek ses dosyası bulunamadı: {kaynak}")

        bilgi = profil_bilgisi_getir(profil, self.kok)
        if not bilgi.hazir_mi or bilgi.model_pth is None:
            raise ValueError(f"'{profil}' profili henüz hazır değil veya model dosyası (.pth) eksik.")

        ayarlar = bilgi.ayarlar
        index_rate = float(ayarlar.get("index_rate", 0.5))
        protect = float(ayarlar.get("protect", 0.5))
        hedef_pitch = int(pitch) if pitch is not None else int(ayarlar.get("pitch", 0))
        volume_envelope = float(ayarlar.get("volume_envelope", 1.0))
        index_arg = str(bilgi.model_index.resolve()) if bilgi.model_index and bilgi.model_index.exists() else "-"

        calisma_koku = (
            Path(cikti_klasor)
            if cikti_klasor is not None
            else Path(tempfile.mkdtemp(prefix=f"rvc_{profil}_"))
        )
        girdi_dizin = calisma_koku / "girdi"
        cikti_dizin = calisma_koku / "cikti"
        girdi_dizin.mkdir(parents=True, exist_ok=True)
        cikti_dizin.mkdir(parents=True, exist_ok=True)

        girdi_wav = girdi_dizin / "kaynak.wav"
        if kaynak.suffix.lower() == ".wav":
            shutil.copy2(kaynak, girdi_wav)
        else:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(kaynak), "-ac", "1", "-ar", "44100", str(girdi_wav)],
                check=True,
            )

        cmd = [
            str(self.applio_python),
            str(self.rvc_cevir_betik),
            str(bilgi.model_pth.resolve()),
            index_arg,
            str(girdi_dizin.resolve()),
            str(cikti_dizin.resolve()),
            str(index_rate),
            str(protect),
            str(hedef_pitch),
            str(volume_envelope),
        ]
        env: dict[str, str] = {}
        for anahtar in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "XDG_CACHE_HOME"):
            deger = os.getenv(anahtar)
            if deger is not None:
                env[anahtar] = deger
        env["APPLIO_CONFIG_DIR"] = APPLIO_EVAL_CONFIG_DIR

        self._calistir(cmd, env)

        uretilen = cikti_dizin / "kaynak.wav"
        if not uretilen.exists():
            wavlar = sorted(cikti_dizin.glob("*.wav"))
            if not wavlar:
                raise RuntimeError("RVC dönüştürme tamamlandı ancak çıktı .wav dosyası üretilmedi.")
            uretilen = wavlar[0]

        return uretilen
