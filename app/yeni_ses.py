"""Yeni ses profili oluşturma, .env + test cümleleri üretimi ve çalıştırıcı arayüzü."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
from typing import Protocol

from app.profiller import VARSAYILAN_KOK

_PROFIL_ADI_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ARALIK_RE = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$")

VARSAYILAN_TEST_CUMLELERI: list[str] = [
    "Merhaba, bugün hava gerçekten çok güzel.",
    "Yarın sabah erkenden yola çıkmayı düşünüyorum.",
    "Bu kitabı okudun mu, yoksa hâlâ başlamadın mı?",
    "Toplantı saat üçte başlayacak, lütfen geç kalma.",
    "Kahvemi hazırladım ve pencerenin önüne oturdum.",
    "Peki sence bu planın gerçekten işe yarar mı?",
    "Her şeyi yeniden düşünmemiz gerekiyor.",
    "Çocuklar bahçede oynarken biz de çay içtik.",
    "Bunu bir daha yaparsan çok üzülürüm.",
    "Haftaya bu zamanlar tatilde olacağız!",
    "Neden bu kadar geç kaldın, merak ettim.",
    "İstasyona vardığımızda tren çoktan gitmişti.",
    "Sanırım bu akşam yağmur yağacak.",
    "Bilgisayarı yeniden başlatınca sorun düzeldi.",
    "Ne güzel bir sürpriz bu, çok teşekkür ederim!",
    "Kütüphanede sessiz olmamız gerekiyor.",
    "Yemekten sonra biraz yürüyüş yapalım mı?",
    "Sınav sonuçları yarın açıklanacak.",
    "Bu şehirde her köşede başka bir hikâye var.",
    "Görüşmek üzere, kendine iyi bak.",
]


class Calistirici(Protocol):
    """Profil eğitim hattını başlatan çalıştırıcı arayüzü."""

    def baslat(self, profil: str) -> None: ...


class SystemdCalistirici:
    """Gerçek ortamda `systemctl --user enable --now rvc-profil@<profil>.timer` çalıştırır."""

    def baslat(self, profil: str) -> None:
        temiz = profil_adi_dogrula(profil)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", f"rvc-profil@{temiz}.timer"],
            check=True,
            capture_output=True,
            text=True,
        )


@dataclass
class SahteCalistirici:
    """Test ortamında systemctl çağırmadan başlatılan profilleri kaydeder."""

    baslatilanlar: list[str] = field(default_factory=list)

    def baslat(self, profil: str) -> None:
        temiz = profil_adi_dogrula(profil)
        self.baslatilanlar.append(temiz)


@dataclass
class YeniSesSonucu:
    profil: str
    gorunen_ad: str
    kaynak_yolu: Path
    env_yolu: Path
    cumleler_yolu: Path
    altyazi_yolu: Path | None
    toplam_epoch: int
    hazirla_mod: str


def profil_adi_dogrula(ad: str) -> str:
    """Profil adını doğrular: boş olamaz, küçük harfle başlamalı ve yalnız `a-z0-9_` içermeli."""
    if not isinstance(ad, str):
        raise ValueError("Profil adı metin olmalıdır.")
    temiz = ad.strip()
    if not _PROFIL_ADI_RE.fullmatch(temiz):
        raise ValueError(
            "Geçersiz profil adı. Yalnızca küçük İngilizce harf (a-z), rakam (0-9) ve alt çizgi (_) kullanın; "
            "isim küçük harfle başlamalıdır (örnek: ahmet, ornek_ses)."
        )
    return temiz


def dosya_adi_uret(profil: str) -> str:
    """Model dosya ön ekini üretir (örn. `ornek_ses` -> `OrnekSes`)."""
    temiz = profil_adi_dogrula(profil)
    return "".join(parca.capitalize() for parca in temiz.split("_") if parca)


def _mod_coz(mod: str) -> str:
    m = mod.strip().lower()
    if m in ("tek", "tek kişi konuşuyor"):
        return "tek"
    if m in ("kume", "birden çok konuşmacı (en baskın olanı seç)"):
        return "kume"
    raise ValueError(f"Geçersiz hazırlık modu: {mod}")


def _epoch_dogrula(otomatik_epoch: bool, toplam_epoch: int | None) -> int:
    if otomatik_epoch:
        return 200
    if toplam_epoch is None:
        raise ValueError("Elle epoch seçiminde TOPLAM_EPOCH belirtilmelidir.")
    deger = int(toplam_epoch)
    if deger < 20 or deger > 200 or deger % 10 != 0:
        raise ValueError("TOPLAM_EPOCH 20 ile 200 arasında ve 10'un katı olmalıdır.")
    return deger


def env_icerigi_uret(
    *,
    profil: str,
    gorunen_ad: str,
    kaynak_yolu: Path | str,
    cumleler_yolu: Path | str,
    hazirla_mod: str = "kume",
    istenmeyen: str = "",
    otomatik_epoch: bool = True,
    toplam_epoch: int | None = 200,
    min_epoch: int = 50,
) -> str:
    """`scripts/profil/<profil>.env` dosyasının metin içeriğini üretir."""
    temiz_profil = profil_adi_dogrula(profil)
    temiz_gorunen = (gorunen_ad.strip() or temiz_profil.replace("_", " ").title()).replace('"', "")
    dosya_onek = dosya_adi_uret(temiz_profil)
    mod_kod = _mod_coz(hazirla_mod)
    ust_epoch = _epoch_dogrula(otomatik_epoch, toplam_epoch)
    alt_epoch = min(int(min_epoch), ust_epoch)

    temiz_istenmeyen = istenmeyen.strip()
    if temiz_istenmeyen and not _ARALIK_RE.fullmatch(temiz_istenmeyen):
        raise ValueError("İstenmeyen ses aralığı 'bas-son' saniye biçiminde olmalıdır (örnek: 0.08-3.9).")

    satirlar = [
        f"# {temiz_gorunen} ses profili yapılandırması",
        f"AD={temiz_profil}",
        f"MODEL={temiz_profil}",
        f'GORUNEN_AD="{temiz_gorunen}"',
        f"DOSYA={dosya_onek}",
        f'KAYNAK="{Path(kaynak_yolu).resolve()}"',
        f'CUMLELER="{Path(cumleler_yolu).resolve()}"',
        f"MIN_EPOCH={alt_epoch}",
        f"TOPLAM_EPOCH={ust_epoch}",
        f"HAZIRLA_MOD={mod_kod}",
    ]
    if temiz_istenmeyen or mod_kod == "tek":
        satirlar.append(f"ISTENMEYEN={temiz_istenmeyen}")
    return "\n".join(satirlar) + "\n"


def yeni_ses_olustur_ve_baslat(
    *,
    profil_adi: str,
    gorunen_ad: str,
    ses_dosyasi: Path | str,
    mod: str = "Birden çok konuşmacı (en baskın olanı seç)",
    istenmeyen: str = "",
    altyazi_dosyasi: Path | str | None = None,
    epoch_modu: str = "Otomatik (önerilen)",
    toplam_epoch: int = 200,
    kok_dizin: Path | str = VARSAYILAN_KOK,
    calistirici: Calistirici | None = None,
    cumleler: list[str] | None = None,
) -> YeniSesSonucu:
    """Yeni ses profilini hazırlar, dosyaları kopyalar, `.env` üretir ve zamanlayıcıyı başlatır."""
    kok = Path(kok_dizin)
    temiz_profil = profil_adi_dogrula(profil_adi)
    temiz_gorunen = gorunen_ad.strip() or temiz_profil.replace("_", " ").title()

    kaynak_girdi = Path(ses_dosyasi)
    if not kaynak_girdi.exists():
        raise FileNotFoundError(f"Kaynak ses dosyası bulunamadı: {kaynak_girdi}")

    otomatik_mi = epoch_modu.strip().lower().startswith("otomatik")
    ust_epoch = _epoch_dogrula(otomatik_mi, toplam_epoch)
    mod_kod = _mod_coz(mod)

    # 1) Kaynak ses dosyasını profiles/<profil>/kaynak/ altına kopyala
    kaynak_klasor = kok / "profiles" / temiz_profil / "kaynak"
    kaynak_klasor.mkdir(parents=True, exist_ok=True)
    kaynak_kopya_yolu = kaynak_klasor / kaynak_girdi.name
    if kaynak_girdi.resolve() != kaynak_kopya_yolu.resolve():
        shutil.copy2(kaynak_girdi, kaynak_kopya_yolu)

    # 2) İsteğe bağlı .vtt/.txt altyazı/metin dosyasını profiles/<profil>/kaynak/ altına kopyala
    hedef_altyazi: Path | None = None
    if altyazi_dosyasi:
        altyazi_girdi = Path(altyazi_dosyasi)
        if altyazi_girdi.exists():
            hedef_altyazi = kaynak_klasor / altyazi_girdi.name
            if altyazi_girdi.resolve() != hedef_altyazi.resolve():
                shutil.copy2(altyazi_girdi, hedef_altyazi)

    # 3) scripts/profil/<profil>_cumleler.txt (20 test cümlesi) ve <profil>.env üret
    profil_betik_klasor = kok / "scripts" / "profil"
    profil_betik_klasor.mkdir(parents=True, exist_ok=True)
    cumleler_yolu = profil_betik_klasor / f"{temiz_profil}_cumleler.txt"
    secilen_cumleler = cumleler if cumleler and len(cumleler) == 20 else VARSAYILAN_TEST_CUMLELERI
    cumleler_yolu.write_text("\n".join(secilen_cumleler) + "\n", encoding="utf-8")

    env_yolu = profil_betik_klasor / f"{temiz_profil}.env"
    env_metni = env_icerigi_uret(
        profil=temiz_profil,
        gorunen_ad=temiz_gorunen,
        kaynak_yolu=kaynak_kopya_yolu,
        cumleler_yolu=cumleler_yolu,
        hazirla_mod=mod_kod,
        istenmeyen=istenmeyen,
        otomatik_epoch=otomatik_mi,
        toplam_epoch=ust_epoch,
    )
    env_yolu.write_text(env_metni, encoding="utf-8")

    # 4) İşaret klasörünü hazırla (durum "hazırlanıyor" olarak görünsün)
    durum_klasor = kok / "outputs" / f".{temiz_profil}"
    durum_klasor.mkdir(parents=True, exist_ok=True)

    # 5) Çalıştırıcı üzerinden başlat
    aktif_calistirici = calistirici if calistirici is not None else SystemdCalistirici()
    aktif_calistirici.baslat(temiz_profil)

    return YeniSesSonucu(
        profil=temiz_profil,
        gorunen_ad=temiz_gorunen,
        kaynak_yolu=kaynak_kopya_yolu,
        env_yolu=env_yolu,
        cumleler_yolu=cumleler_yolu,
        altyazi_yolu=hedef_altyazi,
        toplam_epoch=ust_epoch,
        hazirla_mod=mod_kod,
    )
