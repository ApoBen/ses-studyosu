"""Whisper ile sentez doğrulama: fazladan hece/kelime yakalama, kırpma ve yeniden üretme."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from core.tts.dogrulama import Kelime, degerlendir, kirp, karsilastirma_metni
from core.tts.guvenli_sentez import guvenli_sentezle
import core.tts.xtts_worker as xtts_worker


def K(*items):
    return [Kelime(w, s, e, 1.0) for w, s, e in items]


def test_karsilastirma_metni_turkce_kucuk_harf_ve_noktalama():
    assert karsilastirma_metni("İstanbul'a GİDİYORUM, Işık!") == ["istanbula", "gidiyorum", "ışık"]


def test_ayri_yazilan_ek_birlesir():
    s = degerlendir("Şimdi Jenny'nin odasına gidelim.",
                    K(("Şimdi", 0, .3), ("Jenny", .3, .6), ("nin", .6, .8), ("odasına", .8, 1.2), ("gidelim.", 1.2, 1.6)), sure=1.8)
    assert s.kabul and s.hata_sayisi == 0


def test_rakamla_yazilan_sayi_eslesir():
    s = degerlendir("en az yüz metre", K(("en", 0, .2), ("az", .2, .4), ("100", .4, .7), ("metre", .7, 1.0)), sure=1.2)
    assert s.kabul


def test_sondaki_fazla_ses_kirpilir():
    # "vav" sesi kelime olarak çıkmıyor; son kelimenin bitişinden sonra kırpılmalı
    s = degerlendir("yeni bir soluk getiriyorum.", K(("yeni", 0.0, 0.3), ("bir", 0.3, 0.5),
                                                     ("soluk", 0.5, 0.9), ("getiriyorum.", 0.9, 1.5)), sure=2.0)
    assert s.kabul and s.hata_sayisi == 0
    assert abs(s.kirp_son - 1.65) < 1e-6


def test_sondaki_fazla_kelime_kirpilir():
    s = degerlendir("teşekkür ederim.", K(("teşekkür", 0.0, 1.0), ("ederim", 1.0, 1.4), ("tamamdır.", 1.4, 1.9)), sure=2.0)
    assert s.kabul and s.fazla_son == ["tamamdır"]
    assert abs(s.kirp_son - 1.45) < 1e-6


def test_ortadaki_fazla_kelime_reddedilir():
    s = degerlendir("Kapıyı kilitlemeyi unutmayın. Kendinize iyi bakın.",
                    K(("Kapıyı", 0, .5), ("kilitlemeyi", .5, 1), ("unutmayın", 1, 1.4), ("ona.", 1.4, 1.6),
                      ("Kendinize", 1.7, 2.2), ("iyi", 2.2, 2.4), ("bakın.", 2.4, 2.8)), sure=3.0)
    assert not s.kabul and s.hata_sayisi == 1


def test_eksik_kelime_reddedilir():
    s = degerlendir("kalem ve iki defter", K(("kalem", 0, .4), ("iki", .5, .7), ("defter", .7, 1.0)), sure=1.2)
    assert not s.kabul and s.hata_sayisi == 1


def test_isimlerde_kucuk_yazim_farki_tolere_edilir():
    s = degerlendir("Şimdi Jenny'nin odasına gidelim.",
                    K(("Şimdi", 0, .3), ("Jeninin", .3, .8), ("odasına", .8, 1.2), ("gidelim.", 1.2, 1.6)), sure=1.8)
    assert s.kabul


def test_kirp_basi_ve_sonu_keser_ve_yumusatir():
    sr = 1000
    wav = np.ones(2000, dtype=np.float32)
    out = kirp(wav, sr, 0.5, 1.5, fade_ms=10)
    assert len(out) == 1000
    assert out[0] == 0.0 and out[-1] < 0.2 and out[500] == 1.0


class FakeXTTS:
    def __init__(self, wavs):
        self.wavs = list(wavs); self.calls = []

    def sentezle(self, text, **kw):
        self.calls.append(kw)
        return self.wavs.pop(0)


class FakeWhisper:
    def __init__(self, cevaplar):
        self.cevaplar = dict(cevaplar)

    def kelimeler(self, wav_path):
        return self.cevaplar[Path(wav_path).name]


def _wav(tmp_path, name, sure=2.0, sr=24000):
    p = tmp_path / name
    sf.write(p, np.full(int(sure * sr), 0.1, dtype=np.float32), sr)
    return str(p)


def test_guvenli_sentez_hatalida_yeniden_uretir_ve_kirpar(tmp_path):
    kotu, iyi = _wav(tmp_path, "a.wav"), _wav(tmp_path, "b.wav")
    xtts = FakeXTTS([kotu, iyi])
    wh = FakeWhisper({"a.wav": K(("kalem", 0, .4), ("ona", .4, .6), ("defter", .6, 1.0)),
                      "b.wav": K(("kalem", 0, .4), ("defter", .5, 1.0))})
    sonuc = guvenli_sentezle("kalem defter", xtts, wh, deneme=3, temperature=0.5)
    assert sonuc.kabul and sonuc.deneme_sayisi == 2
    assert len(xtts.calls) == 2 and xtts.calls[0]["temperature"] == 0.5
    assert xtts.calls[0]["seed"] != xtts.calls[1]["seed"]
    a, sr = sf.read(sonuc.wav_path)
    assert abs(len(a) / sr - 1.15) < 0.01


def test_guvenli_sentez_hepsi_hataliysa_en_iyisini_doner(tmp_path):
    w = [_wav(tmp_path, f"{i}.wav") for i in range(2)]
    wh = FakeWhisper({"0.wav": K(("x", 0, .3), ("y", .3, .6), ("z", .6, .9)), "1.wav": K(("kalem", 0, .4), ("z", .4, .9))})
    sonuc = guvenli_sentezle("kalem defter", FakeXTTS(w), wh, deneme=2)
    assert not sonuc.kabul and sonuc.hata_sayisi == 1


def test_xtts_worker_ornekleme_ayarlarini_iletir(tmp_path):
    ref = tmp_path / "ref.wav"; sf.write(ref, np.zeros(1600, dtype=np.float32), 24000)

    class M:
        def get_conditioning_latents(self, audio_path):
            return "g", "s"

        def inference(self, **kw):
            self.kw = kw
            return {"wav": np.zeros(100, dtype=np.float32)}

    m = M()
    r = xtts_worker.handle_request({"text": "merhaba", "speaker_wav": str(ref), "temperature": 0.4,
                                    "repetition_penalty": 6.0, "seed": 7}, m, {})
    assert "wav_path" in r
    assert m.kw["temperature"] == 0.4 and m.kw["repetition_penalty"] == 6.0


def test_xtts_worker_ince_ayarli_gpt_yukler(tmp_path):
    import torch

    class G(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w = torch.nn.Linear(2, 2)

    class M:
        gpt = G()

    yeni = G(); torch.nn.init.constant_(yeni.w.weight, 3.0)
    p = tmp_path / "gpt.pth"; torch.save(yeni.state_dict(), p)
    m = M()
    xtts_worker.load_gpt_checkpoint(m, str(p))
    assert float(m.gpt.w.weight[0, 0]) == 3.0


def test_xtts_engine_gpt_parametresini_iletir():
    from core.tts.xtts_engine import XTTSEngine
    e = XTTSEngine(device="cpu", gpt_checkpoint="/yol/gpt.pth")
    assert "--gpt-checkpoint" in e._worker_cmd() and "/yol/gpt.pth" in e._worker_cmd()


def test_son_kelimeye_yapisan_ek_hata():
    from core.tts.dogrulama import Kelime, degerlendir
    k = [Kelime("merhaba", 0.0, 0.5), Kelime("haberim", 0.6, 1.0), Kelime("vardı", 1.1, 1.5)]
    assert not degerlendir("Merhaba haberim var.", k, 2.0).kabul
    k[-1] = Kelime("var", 1.1, 1.4)
    assert degerlendir("Merhaba haberim var.", k, 2.0).kabul


def test_kisa_kelime_birlesikle_yutulmaz():
    from core.tts.dogrulama import Kelime, degerlendir
    k = [Kelime("kitabın", 0.0, 0.6), Kelime("anlamı", 0.7, 1.1)]
    assert not degerlendir("kitabın ne anlamı", k, 1.5).kabul
