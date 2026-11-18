"""Türkçe metin normalizasyonu, işaretleme, kelime vurgusu ve prozodi planlayıcı testleri."""

from core.text.markup import parse_markup
from core.text.normalize import normalize_turkish_text
from core.text.prosody_planner import plan_prosody
from core.text.stress import find_stressed_syllable, syllabify


def test_normalize_numbers_dates_currency():
    assert normalize_turkish_text("17.11.2026") == "on yedi Kasım iki bin yirmi altı"
    assert normalize_turkish_text("saat 19:52'de") == "saat on dokuz elli ikide"
    assert normalize_turkish_text("%20 indirim") == "yüzde yirmi indirim"
    assert normalize_turkish_text("1.250 TL") == "bin iki yüz elli Türk lirası"
    assert normalize_turkish_text("Prof. Dr. Ahmet") == "profesör doktor Ahmet"


def test_normalize_g1_table_and_rules():
    assert normalize_turkish_text("5'te geldim") == "beşte geldim"
    assert normalize_turkish_text("1990'larda") == "bin dokuz yüz doksanlarda"
    assert normalize_turkish_text("saat 19:52'de") == "saat on dokuz elli ikide"
    assert normalize_turkish_text("2026 yılında") == "iki bin yirmi altı yılında"
    assert normalize_turkish_text("17.11.2026") == "on yedi Kasım iki bin yirmi altı"
    assert normalize_turkish_text("10.5 km") == "on virgül beş kilometre"
    assert normalize_turkish_text("TBMM'de") == "te be me mede"
    assert normalize_turkish_text("THY ve PTT") == "te he ye ve pe te te"
    assert normalize_turkish_text("NATO ve ASELSAN") == "NATO ve ASELSAN"
    assert normalize_turkish_text("10. kat") == "onuncu kat"
    assert normalize_turkish_text("Saat 10.") == "Saat on."
    assert normalize_turkish_text("5 kg ve 20 °C") == "beş kilogram ve yirmi santigrat derece"
    assert normalize_turkish_text("5 sn. bekle") == "beş saniye bekle"
    assert normalize_turkish_text("Sn. Ahmet Bey") == "sayın Ahmet Bey"


def test_normalize_g1_ek():
    assert normalize_turkish_text("3 sn sürdü") == "üç saniye sürdü"
    assert normalize_turkish_text("10 dk bekle") == "on dakika bekle"
    assert normalize_turkish_text("2 sa sürdü") == "iki saat sürdü"
    assert normalize_turkish_text("ABD'nin") == "a be denin"
    assert normalize_turkish_text("ODTÜ, NASA ve TÜBİTAK") == "ODTÜ, NASA ve TÜBİTAK"
    assert normalize_turkish_text("-5 derece") == "eksi beş derece"
    assert normalize_turkish_text("-10 °C") == "eksi on santigrat derece"
    assert normalize_turkish_text("2020-2026") == "iki bin yirmi-iki bin yirmi altı"
    assert normalize_turkish_text("NATO'ya") == "NATOya"
    assert normalize_turkish_text("ODTÜ'de") == "ODTÜde"


def test_markup_parser():
    parsed = parse_markup("Ali kitabı **dün** aldı.<es 500>Peki <yavaş>yavaşça</yavaş>.")
    assert "dün" in parsed.plain_text
    assert any(s.emphasis and s.text == "dün" for s in parsed.spans)
    assert any(s.pause_after_ms == 500 for s in parsed.spans)
    assert any(s.rate < 1.0 and s.text == "yavaşça" for s in parsed.spans)


def test_syllabify_and_stress():
    syls, idx = find_stressed_syllable("kitaplar")
    assert syls == ["ki", "tap", "lar"]
    assert idx == 2  # son hece

    syls, idx = find_stressed_syllable("Ankara")
    assert syls == ["An", "ka", "ra"]
    assert idx == 0  # istisna: AN-ka-ra

    syls, idx = find_stressed_syllable("İstanbul")
    assert syls == ["İs", "tan", "bul"]
    assert idx == 1  # istisna: İs-TAN-bul

    syls, idx = find_stressed_syllable("gelmedim")
    assert syls == ["gel", "me", "dim"]
    assert idx == 0  # -me olumsuzluk eki vurgusuz -> GEL-me-dim


def test_stress_g2_morphological_table():
    assert find_stressed_syllable("elma") == (["el", "ma"], 1)
    assert find_stressed_syllable("kültür") == (["kül", "tür"], 1)
    assert find_stressed_syllable("müdür") == (["mü", "dür"], 1)
    assert find_stressed_syllable("kedimi") == (["ke", "di", "mi"], 2)
    assert find_stressed_syllable("okuma") == (["o", "ku", "ma"], 2)
    assert find_stressed_syllable("öğretmendi") == (["öğ", "ret", "men", "di"], 2)
    assert find_stressed_syllable("gelmiyor") == (["gel", "mi", "yor"], 0)
    assert find_stressed_syllable("gelmedim") == (["gel", "me", "dim"], 0)


def test_stress_g2_suffixes_ambiguity_and_exceptions(monkeypatch):
    assert find_stressed_syllable("arabayla") == (["a", "ra", "bay", "la"], 2)
    assert find_stressed_syllable("gelirken") == (["ge", "lir", "ken"], 1)
    assert find_stressed_syllable("öğretmendir") == (["öğ", "ret", "men", "dir"], 2)

    # Bilinmeyen/analiz edilemeyen kelimede son hece varsayılanı
    assert find_stressed_syllable("zorbakalma") == (["zor", "ba", "kal", "ma"], 3)

    # Belirsiz durumda isim tercihi + WordPlan.stress_ambiguous
    plan = plan_prosody("okuma alışkanlığı")
    okuma_wp = plan.phrases[0].words[0]
    assert okuma_wp.stressed_syllable == 2
    assert okuma_wp.stress_ambiguous is True

    # İstisna sözlüğü: yaygın zarflar ve iller
    for adverb in ("şimdi", "yarın", "bugün", "nerede", "burada", "şurada", "sonra", "önce", "belki", "hemen", "yalnız", "ancak"):
        _, idx = find_stressed_syllable(adverb)
        assert idx == 0, f"{adverb} ilk hece vurgulu olmalı"
    assert find_stressed_syllable("Mersin") == (["Mer", "sin"], 0)
    assert find_stressed_syllable("Malatya") == (["Ma", "lat", "ya"], 1)

    # zeyrek kurulu değilse son hece varsayılanına düşmeli
    monkeypatch.setattr("core.text.stress._get_morph_analyzer", lambda: None)
    assert find_stressed_syllable("gelmedim") == (["gel", "me", "dim"], 2)


def test_prosody_planner_focus_and_phrases():
    plan = plan_prosody("Ali kitabı dün aldı.")
    assert len(plan.phrases) == 1
    words = plan.phrases[0].words
    # "dün" yüklemden hemen önce olduğu için en yüksek prominence skoruna sahip olmalı
    dun = next(w for w in words if w.text == "dün")
    kitabi = next(w for w in words if w.text == "kitabı")
    aldi = next(w for w in words if w.text == "aldı")
    assert dun.is_predicate is False
    assert aldi.is_predicate is True
    assert dun.is_focus is True
    assert dun.prominence > kitabi.prominence
    assert dun.prominence > aldi.prominence

    # Virgüllü cümle öbeklere bölünmeli
    plan2 = plan_prosody("Sabah kalktım, kahvemi hazırladım.")
    assert len(plan2.phrases) == 2
    assert plan2.phrases[0].boundary_tone == "H-"
    assert plan2.phrases[0].pause_after_ms == 200
    assert plan2.phrases[1].boundary_tone == "L%"


def test_prosody_g3_predicates_focus_and_questions():
    # 1. "Kadın yedi elma aldı." -> Kadın ve yedi yüklem değil, sadece aldı yüklem
    p1 = plan_prosody("Kadın yedi elma aldı.")
    w1 = {w.text: w for w in p1.phrases[0].words}
    assert w1["Kadın"].is_predicate is False
    assert w1["yedi"].is_predicate is False
    assert w1["aldı"].is_predicate is True
    assert w1["elma"].is_focus is True
    assert w1["elma"].prominence > w1["aldı"].prominence

    # 2. "Ali dün mü geldi?" -> odak dün'de, mü vurgusuz/negatif, yes_no sorusu, sonda L%
    p2 = plan_prosody("Ali dün mü geldi?")
    assert p2.phrases[0].question_type == "yes_no"
    assert p2.phrases[0].boundary_tone == "L%"
    w2 = {w.text: w for w in p2.phrases[0].words}
    assert w2["dün"].is_focus is True
    assert w2["dün"].tone == "H*"
    assert w2["dün"].prominence > w2["Ali"].prominence
    assert w2["mü"].is_focus is False
    assert w2["mü"].prominence < 0

    # Çekimli soru eki de negatif ve odaksız olmalı
    p2b = plan_prosody("Yarın geliyor musun?")
    w2b = {w.text: w for w in p2b.phrases[0].words}
    assert p2b.phrases[0].question_type == "yes_no"
    assert p2b.phrases[0].boundary_tone == "L%"
    assert w2b["musun"].is_focus is False
    assert w2b["musun"].prominence < 0

    # 3. İsim cümleleri: "Hava çok güzel." / "Ali öğretmen."
    p3a = plan_prosody("Hava çok güzel.")
    w3a = {w.text: w for w in p3a.phrases[0].words}
    assert w3a["güzel"].is_predicate is True
    assert w3a["çok"].is_focus is True
    assert w3a["çok"].prominence > w3a["güzel"].prominence

    p3b = plan_prosody("Ali öğretmen.")
    w3b = {w.text: w for w in p3b.phrases[0].words}
    assert w3b["öğretmen"].is_predicate is True
    assert w3b["Ali"].is_focus is True
    assert w3b["Ali"].prominence > w3b["öğretmen"].prominence

    # 4. Soru kelimeli soru: "Bu paketi kim getirdi?" -> wh, sonda L%, kim zirvede
    p4 = plan_prosody("Bu paketi kim getirdi?")
    assert p4.phrases[0].question_type == "wh"
    assert p4.phrases[0].boundary_tone == "L%"
    w4 = {w.text: w for w in p4.phrases[0].words}
    assert w4["kim"].is_focus is True
    assert w4["kim"].prominence == max(w.prominence for w in p4.phrases[0].words)

    # 5. Devrik cümle: yüklemden sonraki öğeler -1 sıkıştırma alır
    p5 = plan_prosody("Geldi dün Ali.")
    w5 = {w.text: w for w in p5.phrases[0].words}
    assert w5["Geldi"].is_predicate is True
    assert w5["dün"].prominence < w5["Geldi"].prominence or w5["dün"].prominence <= 0.5
    assert w5["Ali"].prominence <= 0.5


def test_prosody_g3_markup_positions(monkeypatch):
    # <es 500> işaretin bulunduğu kelime sınırında öbeği bölmeli
    p1 = plan_prosody("Peki<es 500>sonra ne oldu?")
    assert len(p1.phrases) >= 2
    assert p1.phrases[0].text == "Peki"
    assert p1.phrases[0].pause_after_ms >= 500

    # **dün** sadece işaretlenen kelimeye uygulanmalı, metindeki diğer "dün"lere değil
    p2 = plan_prosody("Dün Ali **dün** geldi.")
    words2 = p2.phrases[0].words
    assert words2[0].text == "Dün"
    assert words2[2].text == "dün"
    assert words2[2].prominence >= words2[0].prominence + 2.5

    # <duygu=...> sadece kendi aralığındaki kelimelere uygulanmalı
    p3 = plan_prosody("Merhaba. <duygu=heyecanli>Harika bir haber bu!</duygu> İyi günler.")
    assert p3.phrases[0].emotion is None
    assert all(w.emotion is None for w in p3.phrases[0].words)
    assert p3.phrases[1].emotion == "heyecanli"
    assert all(w.emotion == "heyecanli" for w in p3.phrases[1].words)
    assert p3.phrases[2].emotion is None
    assert all(w.emotion is None for w in p3.phrases[2].words)

    # Stanza yoksa regex yedeği çalışmalı
    monkeypatch.setattr("core.text.prosody_planner._get_stanza_nlp", lambda: None)
    p_fallback = plan_prosody("Ali kitabı aldı.")
    assert any(w.is_predicate for w in p_fallback.phrases[0].words)


def test_prosody_g3_ek():
    # 1. "Ben bile anladım." -> odak Ben, bile düşük/işlev skoru almalı
    p1 = plan_prosody("Ben bile anladım.")
    w1 = {w.text: w for w in p1.phrases[0].words}
    assert w1["Ben"].is_focus is True
    assert w1["bile"].is_focus is False
    assert w1["Ben"].prominence > w1["bile"].prominence
    assert w1["bile"].prominence <= 0.5

    # 2. "Sen de mi geldin?" -> de negatif/düşük skor alsın, odak Sen
    p2 = plan_prosody("Sen de mi geldin?")
    w2 = {w.text: w for w in p2.phrases[0].words}
    assert w2["Sen"].is_focus is True
    assert w2["de"].is_focus is False
    assert w2["de"].prominence <= 0.5
    assert w2["Sen"].prominence > w2["de"].prominence

    # 3. "Annem sabah erkenden pazara gitti, ama hiçbir şey almadı."
    # Her yan cümlenin kendi yüklemi ve odağı olsun; 2. yan cümlede şey (veya hiçbir şey) odak olmalı
    p3 = plan_prosody("Annem sabah erkenden pazara gitti, ama hiçbir şey almadı.")
    assert len(p3.phrases) == 2
    w3_1 = {w.text: w for w in p3.phrases[0].words}
    w3_2 = {w.text: w for w in p3.phrases[1].words}
    assert w3_1["gitti"].is_predicate is True
    assert w3_1["pazara"].is_focus is True
    assert w3_2["almadı"].is_predicate is True
    assert w3_2["şey"].is_focus is True
    assert w3_2["şey"].prominence > w3_2["almadı"].prominence

    # 4. "Ali **dün** geldi, Ayşe dün gitmedi." -> 2. dün de yüklem öncesi odak olarak puan almalı
    p4 = plan_prosody("Ali **dün** geldi, Ayşe dün gitmedi.")
    assert len(p4.phrases) == 2
    w4_1 = {w.text: w for w in p4.phrases[0].words}
    w4_2 = {w.text: w for w in p4.phrases[1].words}
    assert w4_1["dün"].is_focus is True
    assert w4_2["gitmedi"].is_predicate is True
    assert w4_2["dün"].is_focus is True
    assert w4_2["dün"].prominence > w4_2["Ayşe"].prominence
    assert w4_2["dün"].prominence > w4_2["gitmedi"].prominence


