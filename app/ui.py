"""Ses Stüdyosu — Gradio Blocks kullanıcı arayüzü (4 sekme: Yeni Ses, Seslendir, Dönüştür, Modeller)."""

from __future__ import annotations

from pathlib import Path
import gradio as gr
import pandas as pd

from app.donustur import RvcDonusturucu
from app.profiller import (
    VARSAYILAN_KOK,
    ProfilBilgisi,
    hazir_modelleri_listele,
    profil_bilgisi_getir,
    profilleri_listele,
    sonuclar_oku,
)
from app.seslendir import Seslendirici
from app.yeni_ses import Calistirici, yeni_ses_olustur_ve_baslat


def _hazir_model_secenekleri(kok: Path) -> list[tuple[str, str]]:
    hazirlar = hazir_modelleri_listele(kok)
    return [(f"{p.gorunen_ad} ({p.kod})", p.kod) for p in hazirlar]


def _tum_profil_secenekleri(kok: Path) -> list[tuple[str, str]]:
    profiller = profilleri_listele(kok)
    return [(f"{p.gorunen_ad} ({p.kod}) — {p.durum}", p.kod) for p in profiller]


def _modeller_tablosu_uret(kok: Path) -> pd.DataFrame:
    profiller = profilleri_listele(kok)
    satirlar: list[dict[str, object]] = []
    for p in profiller:
        satirlar.append(
            {
                "Profil": p.kod,
                "Görünen Ad": p.gorunen_ad,
                "Durum": p.durum,
                "Veri (dk)": f"{p.veri_dk:.1f}" if p.veri_dk is not None else "-",
                "Epoch": str(p.en_iyi_epoch) if p.en_iyi_epoch is not None else "-",
                "Benzerlik": f"{p.benzerlik:.4f}" if p.benzerlik is not None else "-",
                "Örnek Sayısı": len(p.ornekler),
            }
        )
    if not satirlar:
        return pd.DataFrame(
            columns=["Profil", "Görünen Ad", "Durum", "Veri (dk)", "Epoch", "Benzerlik", "Örnek Sayısı"]
        )
    return pd.DataFrame(satirlar)


def _durum_ozeti_markdown(kok: Path) -> str:
    profiller = profilleri_listele(kok)
    if not profiller:
        return "_Henüz kayıtlı ses profili bulunmuyor._"
    satirlar = ["### Mevcut Ses Profilleri ve Durumları"]
    for p in profiller:
        ek = []
        if p.veri_dk is not None:
            ek.append(f"veri: {p.veri_dk:.1f} dk")
        if p.en_iyi_epoch is not None:
            ek.append(f"epoch: {p.en_iyi_epoch}")
        if p.benzerlik is not None:
            ek.append(f"benzerlik: {p.benzerlik:.3f}")
        detay = f" ({', '.join(ek)})" if ek else ""
        satirlar.append(f"- **{p.gorunen_ad}** (`{p.kod}`): **{p.durum}**{detay}")
    return "\n".join(satirlar)


def _benzerlik_grafigi_df(profil: str | None, kok: Path) -> pd.DataFrame:
    if not profil:
        return pd.DataFrame({"epoch": [], "ort": []})
    olcumler = sonuclar_oku(profil, kok)
    if not olcumler:
        return pd.DataFrame({"epoch": [], "ort": []})
    return pd.DataFrame(
        [{"epoch": int(r["epoch"]), "ort": float(r["ort"])} for r in olcumler]
    )


def _ornek_secenekleri(profil: str | None, kok: Path) -> list[tuple[str, str]]:
    if not profil:
        return []
    bilgi: ProfilBilgisi = profil_bilgisi_getir(profil, kok)
    return [(w.name, str(w)) for w in bilgi.ornekler]


def build_ui(
    kok_dizin: Path | str = VARSAYILAN_KOK,
    calistirici: Calistirici | None = None,
    seslendirici: Seslendirici | None = None,
    donusturucu: RvcDonusturucu | None = None,
) -> gr.Blocks:
    """Ses Stüdyosu Gradio Blocks arayüzünü oluşturur."""
    kok = Path(kok_dizin)
    aktif_seslendirici = seslendirici or Seslendirici(kok_dizin=kok)
    aktif_donusturucu = donusturucu or RvcDonusturucu(kok_dizin=kok)

    hazir_secenekler = _hazir_model_secenekleri(kok)
    varsayilan_hazir = hazir_secenekler[0][1] if hazir_secenekler else None

    tum_secenekler = _tum_profil_secenekleri(kok)
    varsayilan_profil = tum_secenekler[0][1] if tum_secenekler else None
    ilk_ornekler = _ornek_secenekleri(varsayilan_profil, kok)
    ilk_ornek_deger = ilk_ornekler[0][1] if ilk_ornekler else None

    with gr.Blocks(title="Ses Stüdyosu — Lokal Türkçe Ses Klonlama ve TTS") as demo:
        gr.Markdown(
            "# Ses Stüdyosu\n"
            "Ham ses kaydından otomatik veri seti ve RVC ses modeli oluşturma, "
            "Türkçe tonlamalı ve Whisper doğrulamalı metin seslendirme (TTS) ve sesten sese dönüştürme (VC)."
        )

        # 1. SEKME: YENİ SES
        with gr.Tab("Yeni Ses"):
            gr.Markdown(
                "Bir ses veya video kaydını sürükleyip bırakın. Sistem müziği ayırır, konuşmacıyı seçer, "
                "RVC modelini eğitir ve her 10 epoch'ta benzerliği ölçerek en uygun noktada otomatik durur."
            )
            with gr.Row():
                with gr.Column():
                    kaynak_dosya = gr.File(
                        label="Kaynak Ses / Video Kaydı (Sürükle-Bırak)",
                        file_types=["audio", "video", ".wav", ".mp3", ".opus", ".flac", ".m4a", ".ogg", ".mp4"],
                    )
                    profil_adi_girdi = gr.Textbox(
                        label="Profil Adı (küçük harf, a-z0-9_)",
                        placeholder="ornek: ahmet_bey",
                    )
                    gorunen_ad_girdi = gr.Textbox(
                        label="Görünen Ad",
                        placeholder="ornek: Ahmet Bey",
                    )
                    konusmaci_modu = gr.Radio(
                        choices=[
                            "Tek kişi konuşuyor",
                            "Birden çok konuşmacı (en baskın olanı seç)",
                        ],
                        value="Birden çok konuşmacı (en baskın olanı seç)",
                        label="Konuşmacı Modu",
                    )
                with gr.Column():
                    epoch_modu = gr.Radio(
                        choices=["Otomatik (önerilen)", "Elle seç"],
                        value="Otomatik (önerilen)",
                        label="Epoch Seçimi",
                    )
                    epoch_kaydirici = gr.Slider(
                        minimum=20,
                        maximum=200,
                        step=10,
                        value=200,
                        label="Toplam Epoch Üst Sınırı (TOPLAM_EPOCH)",
                        visible=False,
                    )
                    with gr.Accordion("Gelişmiş Ayarlar", open=False):
                        istenmeyen_girdi = gr.Textbox(
                            label="İstenmeyen Ses Aralığı (sn, 'bas-son' — tek modunda)",
                            placeholder="Örn: 0.08-3.9",
                        )
                        altyazi_girdi = gr.File(
                            label="Altyazı / Metin Yükleme (.vtt veya .txt — İsteğe Bağlı)",
                            file_types=[".vtt", ".txt", ".srt"],
                        )
                    baslat_btn = gr.Button("Başlat", variant="primary")
                    yeni_ses_sonuc = gr.Textbox(label="Başlatma Durumu", interactive=False, lines=3)

            yeni_ses_canli_durum = gr.Markdown(value=_durum_ozeti_markdown(kok))

        # 2. SEKME: SESLENDİR
        with gr.Tab("Seslendir"):
            with gr.Row():
                with gr.Column():
                    seslendir_model = gr.Dropdown(
                        label="Ses Seçimi",
                        choices=hazir_secenekler,
                        value=varsayilan_hazir,
                    )
                    seslendir_metin = gr.Textbox(
                        label="Metin",
                        lines=5,
                        placeholder="Seslendirilecek Türkçe metni buraya yazın...",
                        value="Merhaba, bugün hava çok güzel. Yarın sabah erkenden buluşalım.",
                    )
                    seslendir_hiz = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        step=0.05,
                        value=1.0,
                        label="Konuşma Hızı",
                    )
                    olustur_btn = gr.Button("Oluştur", variant="primary")
                with gr.Column():
                    seslendir_oynatici = gr.Audio(
                        label="Üretilen Ses",
                        type="filepath",
                        interactive=False,
                    )
                    seslendir_indir = gr.File(
                        label="Ses Dosyasını İndir (.wav)",
                        interactive=False,
                    )
                    seslendir_durum = gr.Textbox(
                        label="Sentez ve Doğrulama Bilgisi",
                        interactive=False,
                        lines=3,
                    )

        # 3. SEKME: DÖNÜŞTÜR
        with gr.Tab("Dönüştür"):
            with gr.Row():
                with gr.Column():
                    donustur_girdi = gr.Audio(
                        label="Kaynak Ses Kaydı",
                        type="filepath",
                    )
                    donustur_model = gr.Dropdown(
                        label="Hedef Ses Seçimi",
                        choices=hazir_secenekler,
                        value=varsayilan_hazir,
                    )
                    donustur_pitch = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        step=1,
                        value=0,
                        label="Perde Kaydırma (Pitch — Yarı Ton)",
                    )
                    donustur_btn = gr.Button("Dönüştür", variant="primary")
                with gr.Column():
                    donustur_oynatici = gr.Audio(
                        label="Dönüştürülen Ses",
                        type="filepath",
                        interactive=False,
                    )
                    donustur_indir = gr.File(
                        label="Dönüştürülen Dosyayı İndir (.wav)",
                        interactive=False,
                    )
                    donustur_durum = gr.Textbox(
                        label="Dönüştürme Durumu",
                        interactive=False,
                    )

        # 4. SEKME: MODELLER
        with gr.Tab("Modeller"):
            modeller_ozet_md = gr.Markdown(value=_durum_ozeti_markdown(kok))
            modeller_tablo = gr.Dataframe(
                value=_modeller_tablosu_uret(kok),
                label="Ses Profilleri ve Modeller",
                interactive=False,
            )
            with gr.Row():
                with gr.Column():
                    incelenen_profil = gr.Dropdown(
                        label="Profil Seç (Benzerlik Grafiği ve Örnekler)",
                        choices=tum_secenekler,
                        value=varsayilan_profil,
                    )
                    benzerlik_grafik = gr.LinePlot(
                        value=_benzerlik_grafigi_df(varsayilan_profil, kok),
                        x="epoch",
                        y="ort",
                        title="Epoch Başına Benzerlik (sonuclar.csv)",
                        x_title="Epoch",
                        y_title="Ortalama Kosinüs Benzerliği",
                    )
                with gr.Column():
                    ornek_secim = gr.Dropdown(
                        label="Örnek Ses Seç (profiles/<profil>/ornek/)",
                        choices=ilk_ornekler,
                        value=ilk_ornek_deger,
                    )
                    ornek_oynatici = gr.Audio(
                        label="Örnek Ses Dinle",
                        value=ilk_ornek_deger,
                        type="filepath",
                        interactive=False,
                    )

        # --- OLAY BAĞLANTILARI ---
        def _epoch_gorunurluk_degistir(secim: str):
            elle_mi = secim == "Elle seç"
            return gr.update(visible=elle_mi)

        epoch_modu.change(
            _epoch_gorunurluk_degistir,
            inputs=[epoch_modu],
            outputs=[epoch_kaydirici],
        )

        def _yeni_ses_baslat_ui(
            dosya: str | None,
            p_adi: str,
            g_ad: str,
            mod: str,
            istenmeyen: str,
            altyazi: str | None,
            e_modu: str,
            t_epoch: int,
        ):
            if not dosya:
                return (
                    "Hata: Lütfen bir kaynak ses veya video dosyası yükleyin.",
                    _durum_ozeti_markdown(kok),
                    _modeller_tablosu_uret(kok),
                )
            try:
                sonuc = yeni_ses_olustur_ve_baslat(
                    profil_adi=p_adi,
                    gorunen_ad=g_ad,
                    ses_dosyasi=dosya,
                    mod=mod,
                    istenmeyen=istenmeyen,
                    altyazi_dosyasi=altyazi,
                    epoch_modu=e_modu,
                    toplam_epoch=int(t_epoch),
                    kok_dizin=kok,
                    calistirici=calistirici,
                )
                mesaj = (
                    f"'{sonuc.gorunen_ad}' ({sonuc.profil}) hattı başlatıldı.\n"
                    f"Kaynak: {sonuc.kaynak_yolu}\n"
                    f"Ayarlar: {sonuc.env_yolu} (Üst sınır: {sonuc.toplam_epoch} epoch, mod: {sonuc.hazirla_mod})"
                )
            except Exception as exc:
                mesaj = f"Hata: {exc}"
            return (
                mesaj,
                _durum_ozeti_markdown(kok),
                _modeller_tablosu_uret(kok),
            )

        baslat_btn.click(
            _yeni_ses_baslat_ui,
            inputs=[
                kaynak_dosya,
                profil_adi_girdi,
                gorunen_ad_girdi,
                konusmaci_modu,
                istenmeyen_girdi,
                altyazi_girdi,
                epoch_modu,
                epoch_kaydirici,
            ],
            outputs=[yeni_ses_sonuc, yeni_ses_canli_durum, modeller_tablo],
        )

        def _seslendir_ui(profil: str | None, metin: str, hiz: float):
            if not profil:
                return None, None, "Hata: Lütfen hazır bir ses modeli seçin."
            try:
                sonuc = aktif_seslendirici.seslendir(metin=metin, profil=profil, hiz=float(hiz))
                bilgi_metni = (
                    f"Tamamlandı ({sonuc.parca_sayisi} parça, {sonuc.toplam_deneme} deneme, "
                    f"hata={sonuc.toplam_hata}).\nNormalize: {sonuc.normalize_metin}"
                )
                return sonuc.wav_path, sonuc.wav_path, bilgi_metni
            except Exception as exc:
                return None, None, f"Hata: {exc}"

        olustur_btn.click(
            _seslendir_ui,
            inputs=[seslendir_model, seslendir_metin, seslendir_hiz],
            outputs=[seslendir_oynatici, seslendir_indir, seslendir_durum],
        )

        def _donustur_ui(kaynak_ses: str | None, profil: str | None, pitch: int):
            if not kaynak_ses:
                return None, None, "Hata: Lütfen dönüştürülecek ses kaydını yükleyin."
            if not profil:
                return None, None, "Hata: Lütfen hedef ses modelini seçin."
            try:
                cikti_wav = aktif_donusturucu.donustur(
                    kaynak_ses=kaynak_ses,
                    profil=profil,
                    pitch=int(pitch),
                )
                return str(cikti_wav), str(cikti_wav), f"Dönüştürme tamamlandı: {cikti_wav.name}"
            except Exception as exc:
                return None, None, f"Hata: {exc}"

        donustur_btn.click(
            _donustur_ui,
            inputs=[donustur_girdi, donustur_model, donustur_pitch],
            outputs=[donustur_oynatici, donustur_indir, donustur_durum],
        )

        def _profil_incele_degisti(secilen_profil: str | None):
            df = _benzerlik_grafigi_df(secilen_profil, kok)
            ornekler = _ornek_secenekleri(secilen_profil, kok)
            ilk = ornekler[0][1] if ornekler else None
            return df, gr.update(choices=ornekler, value=ilk), ilk

        incelenen_profil.change(
            _profil_incele_degisti,
            inputs=[incelenen_profil],
            outputs=[benzerlik_grafik, ornek_secim, ornek_oynatici],
        )

        ornek_secim.change(
            lambda yol: yol,
            inputs=[ornek_secim],
            outputs=[ornek_oynatici],
        )

        # 10 saniyede bir otomatik durum yenileme
        zamanlayici = gr.Timer(10.0)

        def _durumlari_yenile(secili_inceleme: str | None, secili_seslendir: str | None, secili_donustur: str | None):
            ozet_md = _durum_ozeti_markdown(kok)
            tablo_df = _modeller_tablosu_uret(kok)
            hazirlar = _hazir_model_secenekleri(kok)
            hazir_kodlar = {k for _, k in hazirlar}
            yeni_seslendir = secili_seslendir if secili_seslendir in hazir_kodlar else (hazirlar[0][1] if hazirlar else None)
            yeni_donustur = secili_donustur if secili_donustur in hazir_kodlar else (hazirlar[0][1] if hazirlar else None)

            tum = _tum_profil_secenekleri(kok)
            tum_kodlar = {k for _, k in tum}
            aktif_inceleme = secili_inceleme if secili_inceleme in tum_kodlar else (tum[0][1] if tum else None)
            grafik_df = _benzerlik_grafigi_df(aktif_inceleme, kok)

            return (
                ozet_md,
                ozet_md,
                tablo_df,
                gr.update(choices=hazirlar, value=yeni_seslendir),
                gr.update(choices=hazirlar, value=yeni_donustur),
                gr.update(choices=tum, value=aktif_inceleme),
                grafik_df,
            )

        zamanlayici.tick(
            _durumlari_yenile,
            inputs=[incelenen_profil, seslendir_model, donustur_model],
            outputs=[
                yeni_ses_canli_durum,
                modeller_ozet_md,
                modeller_tablo,
                seslendir_model,
                donustur_model,
                incelenen_profil,
                benzerlik_grafik,
            ],
        )

    return demo
