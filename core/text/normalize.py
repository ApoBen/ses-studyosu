"""Türkçe metin normalizasyonu: sayılar, tarihler, saatler, kısaltmalar, birimler, para birimleri."""

from __future__ import annotations

import re

UNITS_TR = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS_TR = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
SCALES_TR = ["", "bin", "milyon", "milyar", "trilyon"]

LETTER_READINGS_TR: dict[str, str] = {
    "A": "a", "B": "be", "C": "ce", "Ç": "çe", "D": "de", "E": "e",
    "F": "fe", "G": "ge", "Ğ": "yumuşak ge", "H": "he", "I": "ı", "İ": "i",
    "J": "je", "K": "ke", "L": "le", "M": "me", "N": "ne", "O": "o",
    "Ö": "ö", "P": "pe", "Q": "ku", "R": "re", "S": "se", "Ş": "şe",
    "T": "te", "U": "u", "Ü": "ü", "V": "ve", "W": "dablyu", "X": "iks",
    "Y": "ye", "Z": "ze",
}
UPPER_ABBR_WHITELIST: set[str] = {
    "NATO", "ASELSAN", "ODTÜ", "TÜBİTAK", "UNESCO",
    "NASA", "TOKİ", "İSKİ", "BOTAŞ", "HAVELSAN", "TUSAŞ",
}

# Yaygın Türkçe noktalı kısaltmalar (küçük harf -> okunuş; sn. bağlama göre ayrıca çözülür)
ABBREVIATIONS: dict[str, str] = {
    "vb.": "ve benzeri",
    "vs.": "vesaire",
    "örn.": "örneğin",
    "bkz.": "bakınız",
    "dr.": "doktor",
    "prof.": "profesör",
    "doç.": "doçent",
    "yrd.": "yardımcı",
    "av.": "avukat",
    "müh.": "mühendis",
    "mah.": "mahallesi",
    "cad.": "caddesi",
    "bulv.": "bulvarı",
    "sok.": "sokak",
    "sk.": "sokak",
    "no.": "numara",
    "apt.": "apartmanı",
    "tel.": "telefon",
    "st.": "saat",
    "dk.": "dakika",
    "kg.": "kilogram",
    "gr.": "gram",
    "km.": "kilometre",
    "cm.": "santimetre",
    "mm.": "milimetre",
    "tl.": "türk lirası",
    "yy.": "yüzyıl",
    "m.ö.": "milattan önce",
    "m.s.": "milattan sonra",
    "t.c.": "türkiye cumhuriyeti",
    "a.ş.": "anonim şirket",
    "ltd.": "limited",
    "şti.": "şirketi",
}

UNIT_WORDS_TR: dict[str, str] = {
    "km": "kilometre",
    "kg": "kilogram",
    "cm": "santimetre",
    "mm": "milimetre",
    "ml": "mililitre",
    "lt": "litre",
    "gr": "gram",
    "sn": "saniye",
    "dk": "dakika",
    "sa": "saat",
    "m": "metre",
    "g": "gram",
    "l": "litre",
    "°c": "santigrat derece",
}

MONTHS_TR: dict[int, str] = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}

CURRENCY_SYMBOLS: dict[str, str] = {
    "₺": "Türk lirası",
    "TL": "Türk lirası",
    "$": "dolar",
    "€": "avro",
    "£": "sterlin",
}

_UNSIGNED_NUM_PATTERN = r"(?:\d+(?:\.\d{3})*(?:,\d+)?|\d+(?:\.\d+)?)"
_SIGNED_NUM_PATTERN = rf"(?:(?<![\w.,])-{_UNSIGNED_NUM_PATTERN}|\b{_UNSIGNED_NUM_PATTERN})"

_DATE_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_TIME_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})(?:'([a-zçğıöşüA-ZÇĞİÖŞÜ]+))?\b")
_SN_SAYIN_RE = re.compile(r"(?<!\w)[sS][nN]\.\s+(?=[A-ZÇĞİÖŞÜ])")
_SN_SANIYE_RE = re.compile(r"(?<!\w)sn\.(?!\w)", re.IGNORECASE)
_PERCENT_RE = re.compile(r"%\s*(\d+(?:[.,]\d+)?)")
_CURRENCY_PREFIX_RE = re.compile(r"([₺$€£])\s*(\d+(?:\.\d{3})*(?:,\d+)?)")
_CURRENCY_SUFFIX_RE = re.compile(rf"({_SIGNED_NUM_PATTERN})\s*(TL|₺|\$|€|£)\b", re.IGNORECASE)
_UNIT_RE = re.compile(
    rf"({_SIGNED_NUM_PATTERN})\s*(°[cC]|km|kg|cm|mm|ml|lt|gr|sn|dk|sa|m|g|l)\b(?![.a-zA-ZçğıöşüÇĞİÖŞÜ])"
)
_ORDINAL_DOT_RE = re.compile(r"\b(\d+)\.(?=\s+[a-zçğıöşüA-ZÇĞİÖŞÜ])")
_ORDINAL_APOS_RE = re.compile(r"\b(\d+)'([iıuü]?nc[iıuü])\b", re.IGNORECASE)
_NUMBER_RE = re.compile(rf"({_SIGNED_NUM_PATTERN})(?:'([a-zçğıöşüA-ZÇĞİÖŞÜ]+))?\b")
_UPPER_ABBR_RE = re.compile(r"\b([A-ZÇĞİÖŞÜ]{2,})(?:'([a-zçğıöşüA-ZÇĞİÖŞÜ]+))?\b")


def _three_digits_to_words(n: int) -> list[str]:
    words: list[str] = []
    hundreds = n // 100
    tens = (n % 100) // 10
    units = n % 10
    if hundreds == 1:
        words.append("yüz")
    elif hundreds > 1:
        words.append(UNITS_TR[hundreds])
        words.append("yüz")
    if tens > 0:
        words.append(TENS_TR[tens])
    if units > 0:
        words.append(UNITS_TR[units])
    return words


def sayi_oku(n: int) -> str:
    """Tam sayıyı Türkçe boşluklu yazıyla okur (ör: 2026 -> 'iki bin yirmi altı')."""
    if n == 0:
        return "sıfır"
    if n < 0:
        return f"eksi {sayi_oku(-n)}"

    groups: list[int] = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000

    parts: list[str] = []
    for scale_idx in range(len(groups) - 1, -1, -1):
        grp = groups[scale_idx]
        if grp == 0:
            continue
        if scale_idx == 1 and grp == 1:
            parts.append("bin")
        else:
            parts.extend(_three_digits_to_words(grp))
            if scale_idx > 0:
                parts.append(SCALES_TR[scale_idx])
    return " ".join(parts)


def _to_ordinal_tr(n: int) -> str:
    """Tam sayıyı boşluklu Türkçe sıra sayısına çevirir (ör: 10 -> 'onuncu')."""
    base = sayi_oku(n)
    tokens = base.split()
    last = tokens[-1]
    if last == "dört":
        tokens[-1] = "dördüncü"
        return " ".join(tokens)

    vowels = [ch for ch in last if ch in "aeıioöuü"]
    last_v = vowels[-1] if vowels else "i"
    ends_with_vowel = last[-1] in "aeıioöuü"

    harmony = {
        "a": "ı", "ı": "ı",
        "e": "i", "i": "i",
        "o": "u", "u": "u",
        "ö": "ü", "ü": "ü",
    }
    v = harmony[last_v]
    suffix = f"nc{v}" if ends_with_vowel else f"{v}nc{v}"
    tokens[-1] = f"{last}{suffix}"
    return " ".join(tokens)


def _parse_number_words(raw: str) -> str:
    """Türkçe tam veya ondalık sayı string'ini boşluklu kelimelere çevirir."""
    cleaned = raw.strip()
    if cleaned.startswith("-"):
        return f"eksi {_parse_number_words(cleaned[1:])}"
    if "." in cleaned and "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    if "." in cleaned:
        int_part, frac_part = cleaned.split(".", 1)
        return f"{sayi_oku(int(int_part))} virgül {sayi_oku(int(frac_part))}"
    return sayi_oku(int(cleaned))


def _replace_dates(text: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{sayi_oku(day)} {MONTHS_TR[month]} {sayi_oku(year)}"
        return match.group(0)

    return _DATE_RE.sub(_sub, text)


def _replace_times(text: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = match.group(3)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return match.group(0)
        hour_w = sayi_oku(hour)
        base = f"{hour_w} sıfır sıfır" if minute == 0 else f"{hour_w} {sayi_oku(minute)}"
        return f"{base}{suffix.lower()}" if suffix else base

    return _TIME_RE.sub(_sub, text)


def _replace_abbreviations(text: str) -> str:
    text = _SN_SAYIN_RE.sub("sayın ", text)
    text = _SN_SANIYE_RE.sub("saniye", text)

    for abbr in sorted(ABBREVIATIONS.keys(), key=len, reverse=True):
        pattern = re.compile(r"(?<!\w)" + re.escape(abbr), re.IGNORECASE)
        text = pattern.sub(ABBREVIATIONS[abbr], text)
    return text


def _replace_upper_abbreviations(text: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        token = match.group(1)
        suffix = match.group(2)
        if token in UPPER_ABBR_WHITELIST:
            return f"{token}{suffix.lower()}" if suffix else token
        if not (2 <= len(token) <= 5):
            return match.group(0)
        spelled = " ".join(LETTER_READINGS_TR.get(ch, ch.lower()) for ch in token)
        return f"{spelled}{suffix.lower()}" if suffix else spelled

    return _UPPER_ABBR_RE.sub(_sub, text)


def _replace_currency_percent_units(text: str) -> str:
    text = _PERCENT_RE.sub(lambda m: f"yüzde {_parse_number_words(m.group(1))}", text)
    text = _CURRENCY_PREFIX_RE.sub(
        lambda m: f"{_parse_number_words(m.group(2))} {CURRENCY_SYMBOLS.get(m.group(1), m.group(1))}",
        text,
    )
    text = _CURRENCY_SUFFIX_RE.sub(
        lambda m: f"{_parse_number_words(m.group(1))} {CURRENCY_SYMBOLS.get(m.group(2).upper(), CURRENCY_SYMBOLS.get(m.group(2), m.group(2)))}",
        text,
    )
    text = _UNIT_RE.sub(
        lambda m: f"{_parse_number_words(m.group(1))} {UNIT_WORDS_TR.get(m.group(2).lower(), m.group(2))}",
        text,
    )
    return text


def normalize_turkish_text(text: str) -> str:
    """Türkçe metni TTS için okunabilir yazım biçimine normalleştirir."""
    text = text.strip()
    if not text:
        return ""

    text = _replace_dates(text)
    text = _replace_times(text)
    text = _replace_currency_percent_units(text)
    text = _replace_abbreviations(text)
    text = _replace_upper_abbreviations(text)
    text = _ORDINAL_APOS_RE.sub(lambda m: _to_ordinal_tr(int(m.group(1))), text)
    text = _ORDINAL_DOT_RE.sub(lambda m: _to_ordinal_tr(int(m.group(1))), text)
    text = _NUMBER_RE.sub(
        lambda m: f"{_parse_number_words(m.group(1))}{m.group(2).lower()}" if m.group(2) else _parse_number_words(m.group(1)),
        text,
    )

    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()
