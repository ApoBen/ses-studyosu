"""Ses profilleri ve modelleri listeleme, durum çıkarımı ve ölçüm okuma katmanı."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import fcntl
import json
from pathlib import Path
import re
from typing import Any

VARSAYILAN_KOK = Path(__file__).resolve().parents[1]

_EPOCH_LOG_RE = re.compile(r"\bepoch=(\d+)\b")
_CKPT_EPOCH_RE = re.compile(r"_(\d+)e_\d+s\.pth$")


@dataclass
class ProfilBilgisi:
    kod: str
    gorunen_ad: str
    durum: str
    hazir_mi: bool
    ayarlar: dict[str, Any] = field(default_factory=dict)
    ozet: dict[str, Any] = field(default_factory=dict)
    sonuclar: list[dict[str, Any]] = field(default_factory=list)
    model_pth: Path | None = None
    model_index: Path | None = None
    ref_wav: Path | None = None
    ornekler: list[Path] = field(default_factory=list)

    @property
    def en_iyi_epoch(self) -> int | None:
        if "epoch" in self.ayarlar:
            try:
                return int(self.ayarlar["epoch"])
            except (TypeError, ValueError):
                pass
        if self.sonuclar:
            en_iyi = max(self.sonuclar, key=lambda r: float(r.get("ort", 0.0)))
            return int(en_iyi["epoch"])
        return None

    @property
    def benzerlik(self) -> float | None:
        for anahtar in ("benzerlik", "benzerlik_20cumle"):
            if anahtar in self.ayarlar:
                try:
                    return float(self.ayarlar[anahtar])
                except (TypeError, ValueError):
                    pass
        if self.sonuclar:
            return max(float(r.get("ort", 0.0)) for r in self.sonuclar)
        return None

    @property
    def veri_dk(self) -> float | None:
        for anahtar in ("secilen_dk", "toplam_konusma_dk"):
            if anahtar in self.ozet:
                try:
                    return float(self.ozet[anahtar])
                except (TypeError, ValueError):
                    pass
        return None


def _env_sozlugu_oku(env_yolu: Path) -> dict[str, str]:
    if not env_yolu.exists():
        return {}
    sonuc: dict[str, str] = {}
    for satir in env_yolu.read_text(encoding="utf-8", errors="replace").splitlines():
        satir = satir.strip()
        if not satir or satir.startswith("#"):
            continue
        if "#" in satir and not satir.split("#", 1)[0].count('"') % 2:
            satir = satir.split("#", 1)[0].strip()
        if "=" not in satir:
            continue
        k, v = satir.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        sonuc[k] = v
    return sonuc


def _json_oku(yol: Path) -> dict[str, Any]:
    if not yol.exists():
        return {}
    try:
        veri = json.loads(yol.read_text(encoding="utf-8"))
        return veri if isinstance(veri, dict) else {}
    except Exception:
        return {}


def _dosya_kilitli_mi(kilit_yolu: Path) -> bool:
    if not kilit_yolu.exists():
        return False
    try:
        with kilit_yolu.open("r+") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return False
    except Exception:
        return False


def _son_epoch_bul(profil: str, model_adi: str, kok: Path) -> int:
    # 1) İşaret klasöründe açık epoch dosyası varsa
    st_epoch = kok / "outputs" / f".{profil}" / "epoch"
    if st_epoch.exists():
        try:
            return int(st_epoch.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    en_yuksek = 0

    # 2) Eğitim log dosyasından (outputs/train_<model>.log)
    for aday_ad in {model_adi, profil}:
        log_yolu = kok / "outputs" / f"train_{aday_ad}.log"
        if log_yolu.exists():
            try:
                with log_yolu.open("rb") as f:
                    f.seek(0, 2)
                    boyut = f.tell()
                    f.seek(max(0, boyut - 65536))
                    parca = f.read().decode("utf-8", errors="replace")
                eslesmeler = _EPOCH_LOG_RE.findall(parca)
                if eslesmeler:
                    en_yuksek = max(en_yuksek, max(int(x) for x in eslesmeler))
            except Exception:
                pass

    # 3) Applio checkpoint dosyalarından
    for aday_ad in {model_adi, profil}:
        exp_dir = kok / "external" / "Applio" / "logs" / aday_ad
        if exp_dir.is_dir():
            for pth in exp_dir.glob(f"{aday_ad}_*e_*.pth"):
                m = _CKPT_EPOCH_RE.search(pth.name)
                if m:
                    en_yuksek = max(en_yuksek, int(m.group(1)))

    # 4) Ölçüm sonuçlarından
    olcumler = sonuclar_oku(profil, kok)
    if olcumler:
        en_yuksek = max(en_yuksek, max(int(r["epoch"]) for r in olcumler))

    return en_yuksek if en_yuksek > 0 else 1


def durum_cikar(profil: str, kok_dizin: Path | str = VARSAYILAN_KOK) -> str:
    """Profilin mevcut durumunu işaret dosyalarından ve çıktılardan çıkarır.

    Dönen değerler:
    - ``hazır``
    - ``hata``
    - ``ölçülüyor``
    - ``eğitiliyor (epoch N)``
    - ``eğitim bekliyor``
    - ``hazırlanıyor``
    """
    kok = Path(kok_dizin)
    ayarlar_yolu = kok / "profiles" / profil / "model" / "ayarlar.json"
    st = kok / "outputs" / f".{profil}"

    if ayarlar_yolu.exists() or (st / "bitti").exists():
        return "hazır"

    if (st / "hata").exists():
        return "hata"

    if not (st / "hazirlik_tamam").exists() and (st / "hazirlik_deneme").exists():
        try:
            deneme = int((st / "hazirlik_deneme").read_text(encoding="utf-8").strip())
            if deneme >= 2:
                return "hata"
        except ValueError:
            pass

    if (st / "olculuyor").exists() or (st / "durdur").exists() or _dosya_kilitli_mi(st / "degerlendir.lock"):
        return "ölçülüyor"

    env_bilgi = _env_sozlugu_oku(kok / "scripts" / "profil" / f"{profil}.env")
    model_adi = env_bilgi.get("MODEL", profil)

    if (st / "egitim_basladi").exists():
        epoch_n = _son_epoch_bul(profil, model_adi, kok)
        return f"eğitiliyor (epoch {epoch_n})"

    if (st / "hazirlik_tamam").exists() or (st / "4_test").exists():
        return "eğitim bekliyor"

    return "hazırlanıyor"


def sonuclar_oku(profil: str, kok_dizin: Path | str = VARSAYILAN_KOK) -> list[dict[str, Any]]:
    """Profilin epoch bazlı benzerlik ölçümlerini (`outputs/eval_<profil>/sonuclar.csv`) okur."""
    kok = Path(kok_dizin)
    csv_yolu = kok / "outputs" / f"eval_{profil}" / "sonuclar.csv"
    if not csv_yolu.exists():
        return []

    satirlar: list[dict[str, Any]] = []
    try:
        with csv_yolu.open("r", encoding="utf-8") as f:
            okuyucu = csv.DictReader(f)
            for r in okuyucu:
                try:
                    satirlar.append(
                        {
                            "epoch": int(r["epoch"]),
                            "ort": float(r["ort"]),
                            "min": float(r.get("min") or 0.0),
                            "zaman": str(r.get("zaman") or ""),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        if satirlar:
            satirlar.sort(key=lambda x: x["epoch"])
            return satirlar
    except Exception:
        pass

    return []


def referans_ses_bul(profil: str, kok_dizin: Path | str = VARSAYILAN_KOK) -> Path | None:
    """XTTS için kullanılacak referans ses dosyasını (`profiles/<profil>/ref/ref.wav`) bulur."""
    kok = Path(kok_dizin)
    ref_wav = kok / "profiles" / profil / "ref" / "ref.wav"
    if ref_wav.exists():
        return ref_wav
    ref_klasor = kok / "profiles" / profil / "ref"
    if ref_klasor.is_dir():
        wavlar = sorted(ref_klasor.glob("*.wav"))
        if wavlar:
            return wavlar[0]
    return None


def profil_bilgisi_getir(profil: str, kok_dizin: Path | str = VARSAYILAN_KOK) -> ProfilBilgisi:
    """Tek bir profilin tüm ayar, özet, model ve durum bilgilerini toplar."""
    kok = Path(kok_dizin)
    prof_klasor = kok / "profiles" / profil
    model_klasor = prof_klasor / "model"
    ayarlar = _json_oku(model_klasor / "ayarlar.json")
    ozet = _json_oku(prof_klasor / "dataset" / "ozet.json")
    env_bilgi = _env_sozlugu_oku(kok / "scripts" / "profil" / f"{profil}.env")

    gorunen_ad = (
        str(ayarlar.get("ad") or "")
        or env_bilgi.get("GORUNEN_AD", "")
        or profil.replace("_", " ").title()
    )

    model_pth: Path | None = None
    model_index: Path | None = None
    if model_klasor.is_dir():
        pth_list = sorted(model_klasor.glob("*.pth"))
        idx_list = sorted(model_klasor.glob("*.index"))
        if pth_list:
            model_pth = pth_list[0]
        if idx_list:
            model_index = idx_list[0]

    ornek_klasor = prof_klasor / "ornek"
    ornekler = sorted(ornek_klasor.glob("*.wav")) if ornek_klasor.is_dir() else []

    durum = durum_cikar(profil, kok)
    hazir_mi = durum == "hazır" and model_pth is not None and model_pth.exists()

    return ProfilBilgisi(
        kod=profil,
        gorunen_ad=gorunen_ad,
        durum=durum,
        hazir_mi=hazir_mi,
        ayarlar=ayarlar,
        ozet=ozet,
        sonuclar=sonuclar_oku(profil, kok),
        model_pth=model_pth,
        model_index=model_index,
        ref_wav=referans_ses_bul(profil, kok),
        ornekler=ornekler,
    )


def profilleri_listele(kok_dizin: Path | str = VARSAYILAN_KOK) -> list[ProfilBilgisi]:
    """Sistemdeki tüm ses profillerini tek kuralla listeler."""
    kok = Path(kok_dizin)
    aday_kodlar: set[str] = set()

    prof_kok = kok / "profiles"
    if prof_kok.is_dir():
        for d in prof_kok.iterdir():
            if not d.is_dir() or d.name.startswith("."):
                continue
            if (
                (d / "model" / "ayarlar.json").exists()
                or (d / "dataset" / "ozet.json").exists()
                or (d / "kaynak").is_dir()
                or (kok / "outputs" / f".{d.name}").is_dir()
            ):
                aday_kodlar.add(d.name)

    return [profil_bilgisi_getir(kod, kok) for kod in sorted(aday_kodlar)]


def hazir_modelleri_listele(kok_dizin: Path | str = VARSAYILAN_KOK) -> list[ProfilBilgisi]:
    """`profiles/*/model/ayarlar.json` ve `.pth` dosyası hazır olan modelleri listeler."""
    return [p for p in profilleri_listele(kok_dizin) if p.hazir_mi]
