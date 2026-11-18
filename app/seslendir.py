"""Metinden seçilen ses profiliyle konuşma üretme katmanı (XTTS doğrulamalı + RVC, CPU, tek iş kuyruğu)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
from typing import Protocol
import numpy as np
import soundfile as sf

from core.text.normalize import normalize_turkish_text
from core.tts.guvenli_sentez import GuvenliSonuc, KelimeMotoru, SentezMotoru, guvenli_sentezle
from core.tts.whisper_engine import WhisperEngine
from core.tts.xtts_engine import XTTSEngine, split_text_for_xtts

from app.donustur import RvcDonusturucu
from app.profiller import VARSAYILAN_KOK, profil_bilgisi_getir, referans_ses_bul


class DonusturucuArayuzu(Protocol):
    def donustur(self, kaynak_ses: Path | str, profil: str, pitch: int | None = None) -> Path: ...


@dataclass
class SeslendirmeSonucu:
    wav_path: str
    normalize_metin: str
    parca_sayisi: int
    toplam_deneme: int
    toplam_hata: int
    duyulan_metin: str


class Seslendirici:
    """Metni normalize eder, parçalara böler, XTTS+Whisper ile doğrular ve RVC ile hedef sese çevirir.

    İşlem ağır olduğu için tek iş kilidi (`_kilit`) ile aynı anda yalnızca bir sentez yürütülür.
    """

    def __init__(
        self,
        kok_dizin: Path | str = VARSAYILAN_KOK,
        xtts: SentezMotoru | None = None,
        whisper: KelimeMotoru | None = None,
        donusturucu: DonusturucuArayuzu | None = None,
    ) -> None:
        self.kok = Path(kok_dizin)
        self._xtts = xtts
        self._whisper = whisper
        self._donusturucu = donusturucu
        self._kilit = threading.Lock()

    def _motorlari_hazirla(self) -> tuple[SentezMotoru, KelimeMotoru, DonusturucuArayuzu]:
        if self._xtts is None:
            self._xtts = XTTSEngine(device="cpu", gpt_checkpoint=None)
        if self._whisper is None:
            self._whisper = WhisperEngine()
        if self._donusturucu is None:
            self._donusturucu = RvcDonusturucu(kok_dizin=self.kok)
        return self._xtts, self._whisper, self._donusturucu

    def seslendir(
        self,
        metin: str,
        profil: str,
        hiz: float = 1.0,
        temperature: float = 0.65,
        deneme: int = 3,
    ) -> SeslendirmeSonucu:
        """Metni seçilen ses profiliyle sentezler (tek iş kuyruğu)."""
        ham = metin.strip()
        if not ham:
            raise ValueError("Seslendirilecek metin boş olamaz.")

        bilgi = profil_bilgisi_getir(profil, self.kok)
        if not bilgi.hazir_mi:
            raise ValueError(f"'{profil}' modeli henüz hazır değil.")

        ref_wav = referans_ses_bul(profil, self.kok)
        if ref_wav is None or not ref_wav.exists():
            raise FileNotFoundError(f"'{profil}' profili için referans ses dosyası (ref.wav) bulunamadı.")

        norm_metin = normalize_turkish_text(ham)
        parcalar = split_text_for_xtts(norm_metin)
        if not parcalar:
            raise ValueError("Metin normalize edildikten sonra boş kaldı.")

        with self._kilit:
            xtts, whisper, donusturucu = self._motorlari_hazirla()
            gecici_klasor = Path(tempfile.mkdtemp(prefix=f"seslendir_{profil}_"))

            ses_dizileri: list[np.ndarray] = []
            orneklem_hizi = 24000
            toplam_deneme = 0
            toplam_hata = 0
            duyulanlar: list[str] = []

            for parca in parcalar:
                sonuc: GuvenliSonuc = guvenli_sentezle(
                    parca,
                    xtts,
                    whisper,
                    deneme=deneme,
                    temperature=temperature,
                    repetition_penalty=5.0,
                    speaker_wav=str(ref_wav),
                    speed=float(hiz),
                    out_dir=str(gecici_klasor),
                )
                toplam_deneme += sonuc.deneme_sayisi
                toplam_hata += sonuc.hata_sayisi
                if sonuc.duyulan:
                    duyulanlar.append(sonuc.duyulan)

                ses, sr = sf.read(sonuc.wav_path, dtype="float32")
                if ses.ndim > 1:
                    ses = ses.mean(axis=1)
                orneklem_hizi = int(sr)
                ses_dizileri.append(ses)

            birlesik_xtts_yolu = gecici_klasor / "xtts_birlesik.wav"
            if len(ses_dizileri) == 1:
                birlesik_ses = ses_dizileri[0]
            else:
                duraklama = np.zeros(int(0.15 * orneklem_hizi), dtype=np.float32)
                aralikli: list[np.ndarray] = []
                for idx, dizi in enumerate(ses_dizileri):
                    if idx > 0:
                        aralikli.append(duraklama)
                    aralikli.append(dizi)
                birlesik_ses = np.concatenate(aralikli)

            sf.write(birlesik_xtts_yolu, birlesik_ses, orneklem_hizi)
            rvc_wav = donusturucu.donustur(birlesik_xtts_yolu, profil=profil)

            return SeslendirmeSonucu(
                wav_path=str(rvc_wav),
                normalize_metin=norm_metin,
                parca_sayisi=len(parcalar),
                toplam_deneme=toplam_deneme,
                toplam_hata=toplam_hata,
                duyulan_metin=" ".join(duyulanlar).strip(),
            )
