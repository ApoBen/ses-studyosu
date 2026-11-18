"""Ses Stüdyosu (Görev A1 / A1-ek) birim testleri.

Gerçek model yüklemez, GPU/XPU veya systemctl çalıştırmaz; tüm dış bağımlılıklar
sahte (mock/fake) nesnelerle ve nötr profil adlarıyla test edilir.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from core.tts.dogrulama import Kelime

from app.profiller import (
    durum_cikar,
    hazir_modelleri_listele,
    profilleri_listele,
    referans_ses_bul,
    sonuclar_oku,
)
from app.yeni_ses import (
    VARSAYILAN_TEST_CUMLELERI,
    SahteCalistirici,
    env_icerigi_uret,
    profil_adi_dogrula,
    yeni_ses_olustur_ve_baslat,
)
from app.donustur import RvcDonusturucu
from app.seslendir import Seslendirici
from app.ui import _durum_ozeti_markdown, _modeller_tablosu_uret, build_ui


def _sahte_wav_yaz(yol: Path, sure_sn: float = 1.0, sr: int = 24000) -> Path:
    yol.parent.mkdir(parents=True, exist_ok=True)
    sf.write(yol, np.full(int(sure_sn * sr), 0.1, dtype=np.float32), sr)
    return yol


def test_profil_adi_dogrulama():
    assert profil_adi_dogrula("ornek_ses") == "ornek_ses"
    assert profil_adi_dogrula("konusmaci1") == "konusmaci1"
    assert profil_adi_dogrula("ahmet") == "ahmet"

    for gecersiz in [
        "",
        "   ",
        "Ahmet",
        "ornek-ses",
        "ornek ses",
        "çağrı",
        "şahin",
        "ığdır",
        "../ust_dizin",
        "_gizli",
        "123",
    ]:
        with pytest.raises(ValueError):
            profil_adi_dogrula(gecersiz)


def test_env_uretimi_ve_toplam_epoch_10un_kati(tmp_path: Path):
    kaynak = tmp_path / "profiles" / "ahmet" / "kaynak" / "ses.opus"
    cumleler = tmp_path / "scripts" / "profil" / "ahmet_cumleler.txt"

    # Otomatik epoch -> varsayılan TOPLAM_EPOCH=200, MIN_EPOCH=50
    env_oto = env_icerigi_uret(
        profil="ahmet",
        gorunen_ad="Ahmet Bey",
        kaynak_yolu=kaynak,
        cumleler_yolu=cumleler,
        hazirla_mod="kume",
        otomatik_epoch=True,
    )
    assert "AD=ahmet" in env_oto
    assert "MODEL=ahmet" in env_oto
    assert 'GORUNEN_AD="Ahmet Bey"' in env_oto
    assert "DOSYA=Ahmet" in env_oto
    assert f'KAYNAK="{kaynak}"' in env_oto
    assert f'CUMLELER="{cumleler}"' in env_oto
    assert "MIN_EPOCH=50" in env_oto
    assert "TOPLAM_EPOCH=200" in env_oto
    assert "HAZIRLA_MOD=kume" in env_oto

    # Elle seçilen epoch (10'un katı, 20-200 arası) ve tek konuşmacı + istenmeyen aralık
    env_elle = env_icerigi_uret(
        profil="ornek_ses",
        gorunen_ad="Örnek Ses",
        kaynak_yolu=kaynak,
        cumleler_yolu=cumleler,
        hazirla_mod="tek",
        istenmeyen="0.08-3.9",
        otomatik_epoch=False,
        toplam_epoch=80,
    )
    assert "DOSYA=OrnekSes" in env_elle
    assert "TOPLAM_EPOCH=80" in env_elle
    assert "MIN_EPOCH=50" in env_elle
    assert "HAZIRLA_MOD=tek" in env_elle
    assert "ISTENMEYEN=0.08-3.9" in env_elle

    # 10'un katı olmayan veya sınır dışı TOPLAM_EPOCH reddedilmeli
    for gecersiz_epoch in [10, 15, 55, 210, 0, -20]:
        with pytest.raises(ValueError):
            env_icerigi_uret(
                profil="ahmet",
                gorunen_ad="Ahmet",
                kaynak_yolu=kaynak,
                cumleler_yolu=cumleler,
                otomatik_epoch=False,
                toplam_epoch=gecersiz_epoch,
            )


def test_durum_cikarimi_isaret_dosyalariyla(tmp_path: Path):
    # 1) Hazırlanıyor
    st = tmp_path / "outputs" / ".deneme"
    st.mkdir(parents=True)
    (st / "1_ayir").touch()
    assert durum_cikar("deneme", tmp_path) == "hazırlanıyor"

    # 2) Hata (hazirlik_deneme >= 2 ve hazirlik_tamam yok)
    (st / "hazirlik_deneme").write_text("2\n", encoding="utf-8")
    assert durum_cikar("deneme", tmp_path) == "hata"

    # 3) Eğitim bekliyor (hazirlik_tamam var, egitim_basladi yok)
    (st / "hazirlik_tamam").touch()
    assert durum_cikar("deneme", tmp_path) == "eğitim bekliyor"

    # 4) Eğitiliyor (epoch N)
    (st / "egitim_basladi").write_text("1700000000\n", encoding="utf-8")
    (tmp_path / "outputs" / "train_deneme.log").write_text(
        "deneme | epoch=12 | step=100 | batch=10/50\ndeneme | epoch=14 | step=120 | batch=20/50\n",
        encoding="utf-8",
    )
    assert durum_cikar("deneme", tmp_path) == "eğitiliyor (epoch 14)"

    # 5) Ölçülüyor (durdur işareti var ama ayarlar.json/bitti henüz yok; ince ayar / değerlendirme aşaması)
    (st / "durdur").touch()
    assert durum_cikar("deneme", tmp_path) == "ölçülüyor"

    # 6) Hazır (ayarlar.json oluştuğunda)
    model_dir = tmp_path / "profiles" / "deneme" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "ayarlar.json").write_text('{"ad": "Deneme", "epoch": 50}', encoding="utf-8")
    assert durum_cikar("deneme", tmp_path) == "hazır"


def test_model_ve_profil_listesi_tek_kural(tmp_path: Path):
    # Şablon ornek.env (profiles/ornek veya outputs/.ornek yok) aktif profil sayılmamalı
    betik_dir = tmp_path / "scripts" / "profil"
    betik_dir.mkdir(parents=True)
    (betik_dir / "ornek.env").write_text('AD=ornek_ses\nGORUNEN_AD="Örnek Ses"\n', encoding="utf-8")

    # Hazır profil: ornek_ses (standart yollar: profiles/<p>/model, ref/ref.wav, outputs/eval_<p>/sonuclar.csv)
    ornek_model = tmp_path / "profiles" / "ornek_ses" / "model"
    ornek_model.mkdir(parents=True)
    (ornek_model / "OrnekSes.pth").write_bytes(b"pth")
    (ornek_model / "OrnekSes.index").write_bytes(b"idx")
    (ornek_model / "ayarlar.json").write_text(
        json.dumps(
            {
                "ad": "Örnek Ses",
                "epoch": 50,
                "index_rate": 0.5,
                "protect": 0.5,
                "pitch": 0,
                "benzerlik": 0.7746,
            }
        ),
        encoding="utf-8",
    )
    _sahte_wav_yaz(tmp_path / "profiles" / "ornek_ses" / "ref" / "ref.wav")
    _sahte_wav_yaz(tmp_path / "profiles" / "ornek_ses" / "ornek" / "00.wav")
    (tmp_path / "profiles" / "ornek_ses" / "dataset").mkdir(parents=True)
    (tmp_path / "profiles" / "ornek_ses" / "dataset" / "ozet.json").write_text(
        json.dumps({"secilen_dk": 31.9, "secilen_parca": 199, "benzerlik_medyan": 0.888}),
        encoding="utf-8",
    )
    eval_dir = tmp_path / "outputs" / "eval_ornek_ses"
    eval_dir.mkdir(parents=True)
    (eval_dir / "sonuclar.csv").write_text(
        "epoch,ort,min,zaman\n10,0.7384,0.6537,2026-11-18 07:40:28\n20,0.7603,0.6848,2026-11-18 07:51:28\n",
        encoding="utf-8",
    )

    # Eğitilen profil: konusmaci1
    st_k1 = tmp_path / "outputs" / ".konusmaci1"
    st_k1.mkdir(parents=True)
    (st_k1 / "hazirlik_tamam").touch()
    (st_k1 / "egitim_basladi").touch()
    (tmp_path / "outputs" / "train_konusmaci1.log").write_text("konusmaci1 | epoch=4 | step=200\n", encoding="utf-8")
    (tmp_path / "profiles" / "konusmaci1" / "dataset").mkdir(parents=True)
    (tmp_path / "profiles" / "konusmaci1" / "dataset" / "ozet.json").write_text(
        json.dumps({"secilen_dk": 9.8, "secilen_parca": 236}),
        encoding="utf-8",
    )

    profiller = profilleri_listele(tmp_path)
    kodlar = [p.kod for p in profiller]
    assert kodlar == ["konusmaci1", "ornek_ses"]

    hazirlar = hazir_modelleri_listele(tmp_path)
    assert [m.kod for m in hazirlar] == ["ornek_ses"]
    assert hazirlar[0].gorunen_ad == "Örnek Ses"
    assert len(hazirlar[0].ornekler) == 1
    assert referans_ses_bul("ornek_ses", tmp_path) == tmp_path / "profiles" / "ornek_ses" / "ref" / "ref.wav"

    olcumler = sonuclar_oku("ornek_ses", tmp_path)
    assert len(olcumler) == 2
    assert olcumler[0]["epoch"] == 10
    assert abs(olcumler[1]["ort"] - 0.7603) < 1e-4


def test_varsayilan_cumleler_ornek_dosyasiyla_uyumlu():
    kok = Path(__file__).resolve().parents[1]
    ornek_cumleler_yolu = kok / "scripts" / "profil" / "ornek_cumleler.txt"
    assert ornek_cumleler_yolu.exists()
    beklenen = [
        s.strip()
        for s in ornek_cumleler_yolu.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    assert len(beklenen) == 20
    assert VARSAYILAN_TEST_CUMLELERI == beklenen


def test_yeni_ses_baslat_sahte_calistirici_ile(tmp_path: Path):
    kaynak_ses = _sahte_wav_yaz(tmp_path / "dis_kaynak" / "konusma.wav")
    altyazi = tmp_path / "dis_kaynak" / "altyazi.vtt"
    altyazi.write_text("WEBVTT\n\n00:00.000 --> 00:02.000\nMerhaba dünya\n", encoding="utf-8")

    calistirici = SahteCalistirici()
    sonuc = yeni_ses_olustur_ve_baslat(
        profil_adi="kerem",
        gorunen_ad="Kerem",
        ses_dosyasi=kaynak_ses,
        mod="Birden çok konuşmacı (en baskın olanı seç)",
        istenmeyen="",
        altyazi_dosyasi=altyazi,
        epoch_modu="Elle seç",
        toplam_epoch=120,
        kok_dizin=tmp_path,
        calistirici=calistirici,
    )

    assert sonuc.profil == "kerem"
    assert calistirici.baslatilanlar == ["kerem"]
    assert (tmp_path / "profiles" / "kerem" / "kaynak" / "konusma.wav").exists()
    assert (tmp_path / "profiles" / "kerem" / "kaynak" / "altyazi.vtt").exists()

    env_yolu = tmp_path / "scripts" / "profil" / "kerem.env"
    cumleler_yolu = tmp_path / "scripts" / "profil" / "kerem_cumleler.txt"
    assert env_yolu.exists()
    assert cumleler_yolu.exists()
    assert len([s for s in cumleler_yolu.read_text(encoding="utf-8").splitlines() if s.strip()]) == 20

    env_metni = env_yolu.read_text(encoding="utf-8")
    assert "AD=kerem" in env_metni
    assert "TOPLAM_EPOCH=120" in env_metni
    assert "HAZIRLA_MOD=kume" in env_metni


def test_donustur_sahte_calistirici_ile(tmp_path: Path):
    model_dir = tmp_path / "profiles" / "ornek_ses" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "OrnekSes.pth").write_bytes(b"pth")
    (model_dir / "OrnekSes.index").write_bytes(b"idx")
    (model_dir / "ayarlar.json").write_text(
        json.dumps({"ad": "Örnek Ses", "index_rate": 0.5, "protect": 0.5, "pitch": 0}),
        encoding="utf-8",
    )

    girdi_wav = _sahte_wav_yaz(tmp_path / "girdi.wav", sure_sn=0.5)
    cagrilar: list[tuple[list[str], dict[str, str]]] = []

    def sahte_komut_calistir(cmd: list[str], env: dict[str, str]) -> None:
        cagrilar.append((cmd, env))
        cikti_klasor = Path(cmd[5])
        cikti_klasor.mkdir(parents=True, exist_ok=True)
        girdi_klasor = Path(cmd[4])
        for f in girdi_klasor.glob("*.wav"):
            _sahte_wav_yaz(cikti_klasor / f.name, sure_sn=0.5)

    donusturucu = RvcDonusturucu(kok_dizin=tmp_path, komut_calistirici=sahte_komut_calistir)
    cikti_yolu = donusturucu.donustur(girdi_wav, profil="ornek_ses", pitch=2)

    assert Path(cikti_yolu).exists()
    assert len(cagrilar) == 1
    cmd, env = cagrilar[0]
    assert env.get("APPLIO_CONFIG_DIR") == "/tmp/applio_eval_config"
    assert cmd[2].endswith("OrnekSes.pth")
    assert cmd[3].endswith("OrnekSes.index")
    assert cmd[6] == "0.5"
    assert cmd[7] == "0.5"
    assert cmd[8] == "2"


def test_seslendir_sahte_motorlarla(tmp_path: Path):
    model_dir = tmp_path / "profiles" / "ornek_ses" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "OrnekSes.pth").write_bytes(b"pth")
    (model_dir / "OrnekSes.index").write_bytes(b"idx")
    (model_dir / "ayarlar.json").write_text(
        json.dumps({"ad": "Örnek Ses", "index_rate": 0.5, "protect": 0.5, "pitch": 0}),
        encoding="utf-8",
    )
    _sahte_wav_yaz(tmp_path / "profiles" / "ornek_ses" / "ref" / "ref.wav")

    class SahteXTTS:
        def __init__(self) -> None:
            self.metinler: list[str] = []

        def sentezle(self, text: str, **kw) -> str:
            self.metinler.append(text)
            out = tmp_path / f"xtts_{len(self.metinler)}.wav"
            _sahte_wav_yaz(out, sure_sn=1.2)
            return str(out)

    class SahteWhisper:
        def kelimeler(self, wav_path: str) -> list[Kelime]:
            return [
                Kelime("on", 0.0, 0.3, 0.99),
                Kelime("kilometre", 0.3, 0.7, 0.99),
                Kelime("yürüdük.", 0.7, 1.1, 0.99),
            ]

    class SahteRvc:
        def __init__(self) -> None:
            self.cagrilar: list[tuple[Path, str]] = []

        def donustur(self, kaynak_wav: str | Path, profil: str, pitch: int | None = None) -> Path:
            self.cagrilar.append((Path(kaynak_wav), profil))
            hedef = tmp_path / "rvc_cikti.wav"
            _sahte_wav_yaz(hedef, sure_sn=1.1)
            return hedef

    xtts = SahteXTTS()
    whisper = SahteWhisper()
    rvc = SahteRvc()

    seslendirici = Seslendirici(
        kok_dizin=tmp_path,
        xtts=xtts,
        whisper=whisper,
        donusturucu=rvc,
    )
    sonuc = seslendirici.seslendir("10 km yürüdük.", profil="ornek_ses", hiz=1.0)

    assert Path(sonuc.wav_path).exists()
    assert xtts.metinler == ["on kilometre yürüdük."]
    assert len(rvc.cagrilar) == 1
    assert rvc.cagrilar[0][1] == "ornek_ses"


def test_ui_4_sekme_icerir(tmp_path: Path):
    demo = build_ui(kok_dizin=tmp_path, calistirici=SahteCalistirici())
    cfg = demo.get_config_file()
    tab_etiketleri = [
        c.get("props", {}).get("label")
        for c in cfg.get("components", [])
        if c.get("type") == "tabitem"
    ]
    assert tab_etiketleri == ["Yeni Ses", "Seslendir", "Dönüştür", "Modeller"]


def test_modeller_tablosu_ve_durumlari_tek_kural(tmp_path: Path):
    # 1. Hazır profil: ornek_ses
    m1 = tmp_path / "profiles" / "ornek_ses" / "model"
    m1.mkdir(parents=True)
    (m1 / "OrnekSes.pth").write_bytes(b"pth")
    (m1 / "ayarlar.json").write_text('{"ad": "Örnek Ses", "epoch": 100, "benzerlik": 0.785}', encoding="utf-8")

    # 2. Hazır profil: ahmet
    m2 = tmp_path / "profiles" / "ahmet" / "model"
    m2.mkdir(parents=True)
    (m2 / "Ahmet.pth").write_bytes(b"pth")
    (m2 / "ayarlar.json").write_text('{"ad": "Ahmet", "epoch": 50, "benzerlik": 0.774}', encoding="utf-8")

    # 3. Eğitilen profil: konusmaci1
    st3 = tmp_path / "outputs" / ".konusmaci1"
    st3.mkdir(parents=True)
    (st3 / "hazirlik_tamam").touch()
    (st3 / "egitim_basladi").touch()
    (tmp_path / "outputs" / "train_konusmaci1.log").write_text("konusmaci1 | epoch=10 | step=100\n", encoding="utf-8")
    (tmp_path / "profiles" / "konusmaci1" / "dataset").mkdir(parents=True)
    (tmp_path / "profiles" / "konusmaci1" / "dataset" / "ozet.json").write_text('{"secilen_dk": 12.5}', encoding="utf-8")

    df = _modeller_tablosu_uret(tmp_path)
    profiller = set(df["Profil"].tolist())
    assert profiller == {"ahmet", "konusmaci1", "ornek_ses"}

    durumlar = dict(zip(df["Profil"], df["Durum"]))
    assert durumlar["ornek_ses"] == "hazır"
    assert durumlar["ahmet"] == "hazır"
    assert durumlar["konusmaci1"] == "eğitiliyor (epoch 10)"

    md = _durum_ozeti_markdown(tmp_path)
    assert "Örnek Ses" in md and "Ahmet" in md and "Konusmaci1" in md


def test_seslendir_uzun_metin_bolme_ve_tek_is_kuyrugu(tmp_path: Path):
    import threading
    import time

    model_dir = tmp_path / "profiles" / "ornek_ses" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "OrnekSes.pth").write_bytes(b"pth")
    (model_dir / "ayarlar.json").write_text('{"ad": "Örnek Ses"}', encoding="utf-8")
    _sahte_wav_yaz(tmp_path / "profiles" / "ornek_ses" / "ref" / "ref.wav")

    aktif_sayac = 0
    maks_ayni_anda = 0
    sayac_kilit = threading.Lock()
    cagri_sayisi = 0

    class YavasSahteXTTS:
        def sentezle(self, text: str, **kw) -> str:
            nonlocal aktif_sayac, maks_ayni_anda, cagri_sayisi
            with sayac_kilit:
                aktif_sayac += 1
                cagri_sayisi += 1
                maks_ayni_anda = max(maks_ayni_anda, aktif_sayac)
                idx = cagri_sayisi
            time.sleep(0.03)
            out = tmp_path / f"parca_{idx}.wav"
            _sahte_wav_yaz(out, sure_sn=0.4)
            with sayac_kilit:
                aktif_sayac -= 1
            return str(out)

    class HerSeyiKabulEdenWhisper:
        def kelimeler(self, wav_path: str) -> list[Kelime]:
            return [Kelime("merhaba", 0.0, 0.3, 0.99)]

    class SahteRvc:
        def donustur(self, kaynak_wav: str | Path, profil: str, pitch: int | None = None) -> Path:
            hedef = tmp_path / f"out_{Path(kaynak_wav).stem}.wav"
            _sahte_wav_yaz(hedef, sure_sn=0.4)
            return hedef

    seslendirici = Seslendirici(
        kok_dizin=tmp_path,
        xtts=YavasSahteXTTS(),
        whisper=HerSeyiKabulEdenWhisper(),
        donusturucu=SahteRvc(),
    )

    uzun_metin = " ".join(["merhaba"] * 45)
    assert len(uzun_metin) > 250
    sonuc = seslendirici.seslendir(uzun_metin, profil="ornek_ses", deneme=1)
    assert sonuc.parca_sayisi >= 2
    assert Path(sonuc.wav_path).exists()

    # İki iş parçacığı aynı anda seslendir çağırdığında tek iş kuyruğu (_kilit) devreye girmeli
    hatalar: list[Exception] = []

    def isci():
        try:
            seslendirici.seslendir("merhaba", profil="ornek_ses", deneme=1)
        except Exception as e:
            hatalar.append(e)

    t1 = threading.Thread(target=isci)
    t2 = threading.Thread(target=isci)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not hatalar
    assert maks_ayni_anda == 1
