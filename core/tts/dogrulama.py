"""Sentez doğrulama: Whisper kelimeleri ile beklenen metni hizalar.

XTTS bazen cümle sonuna anlamsız bir hece ("vav") ya da metinde olmayan kelimeler ekler.
- Sondaki (veya baştaki) fazlalık kırpılarak düzeltilir.
- Ortadaki fazla/eksik/yanlış kelime düzeltilemez: parça yeniden üretilmelidir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from core.text.normalize import sayi_oku

SON_PAY = 0.15   # son kelimeden sonra bırakılan pay (sn); 0,08 son ünsüzü kesiyordu ("var" -> "vağ")
BAS_PAY = 0.05
BENZERLIK_ESIGI = 0.7  # özel isimlerde küçük yazım farkı (Jenny'nin / Jeninin)


@dataclass
class Kelime:
    kelime: str
    bas: float
    son: float
    olasilik: float = 1.0


@dataclass
class Degerlendirme:
    kabul: bool
    hata_sayisi: int
    kirp_bas: float
    kirp_son: float
    fazla_son: list[str] = field(default_factory=list)
    fazla_bas: list[str] = field(default_factory=list)
    duyulan: str = ""


def karsilastirma_metni(metin: str) -> list[str]:
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    metin = re.sub(r"[’'`]", "", metin)
    # Whisper sayıları rakamla yazar ("100"); beklenen metin okunuşla ("yüz") gelir.
    metin = re.sub(r"\d+", lambda m: f" {sayi_oku(int(m.group()))} ", metin)
    return re.findall(r"[0-9a-zçğıöşüâîû]+", metin)


def _esit(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= BENZERLIK_ESIGI


def _esit_birlesik(a: str, b: str) -> bool:
    """Birleştirmede sıkı eşleşme: fazla kelimeyi yutmasın ("unutmayın"+"ona" != "unutmayın")."""
    return a == b or (abs(len(a) - len(b)) <= 1 and SequenceMatcher(None, a, b).ratio() >= 0.85)  # <=2 "kitabın ne"yi "kitabın" sayıyordu


def _hizala(beklenen: list[str], duyulan: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein hizalaması: (işlem, beklenen_i, duyulan_j). işlem: esit/degis/eksik/fazla/bitisik.
    'bitisik': ayrı yazılmış ek veya birleşik kelime ("jenny nin" = "jennynin"), hata sayılmaz."""
    n, m = len(beklenen), len(duyulan)
    INF = 10**9
    d = np.full((n + 1, m + 1), INF, dtype=np.int64)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if _esit(beklenen[i - 1], duyulan[j - 1]) else 1
            v = min(d[i - 1, j - 1] + c, d[i - 1, j] + 1, d[i, j - 1] + 1)
            if j >= 2 and _esit_birlesik(beklenen[i - 1], duyulan[j - 2] + duyulan[j - 1]):
                v = min(v, d[i - 1, j - 2])
            if i >= 2 and _esit_birlesik(beklenen[i - 2] + beklenen[i - 1], duyulan[j - 1]):
                v = min(v, d[i - 2, j - 1])
            d[i, j] = v
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0 if _esit(beklenen[i - 1], duyulan[j - 1]) else 1
            if d[i, j] == d[i - 1, j - 1] + c:
                ops.append(("esit" if c == 0 else "degis", i - 1, j - 1)); i -= 1; j -= 1; continue
            if j >= 2 and _esit_birlesik(beklenen[i - 1], duyulan[j - 2] + duyulan[j - 1]) and d[i, j] == d[i - 1, j - 2]:
                ops.append(("esit", i - 1, j - 1)); ops.append(("bitisik", None, j - 2)); i -= 1; j -= 2; continue
            if i >= 2 and _esit_birlesik(beklenen[i - 2] + beklenen[i - 1], duyulan[j - 1]) and d[i, j] == d[i - 2, j - 1]:
                ops.append(("esit", i - 1, j - 1)); ops.append(("bitisik", i - 2, None)); i -= 2; j -= 1; continue
        if i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("eksik", i - 1, None)); i -= 1
        else:
            ops.append(("fazla", None, j - 1)); j -= 1
    return ops[::-1]


def degerlendir(beklenen_metin: str, kelimeler: list[Kelime], sure: float) -> Degerlendirme:
    beklenen = karsilastirma_metni(beklenen_metin)
    duyulan_k = [(k, w) for k in kelimeler for w in karsilastirma_metni(k.kelime)]
    duyulan = [w for _, w in duyulan_k]
    ops = _hizala(beklenen, duyulan)
    eslesen = [j for op, _, j in ops if op in ("esit", "degis", "bitisik") and j is not None]
    if not eslesen:
        return Degerlendirme(False, max(len(beklenen), 1), 0.0, sure, duyulan=" ".join(duyulan))
    ilk, son = eslesen[0], eslesen[-1]
    fazla_bas = [duyulan[j] for op, _, j in ops if op == "fazla" and j < ilk]
    fazla_son = [duyulan[j] for op, _, j in ops if op == "fazla" and j > son]
    hata = sum(1 for op, _, j in ops if op in ("degis", "eksik") or (op == "fazla" and ilk < j < son))
    # Son kelimeye yapışan uydurma ek ("var" -> "vardı"): bulanık eşleşme bunu kabul eder ama kelime içinden kırpılamaz,
    # bu yüzden hata sayılır (başka tohumla yeniden üretilir).
    son_b = next((i for op, i, j in reversed(ops) if op == "esit" and j == son), None)
    if son_b is not None and duyulan[son] != beklenen[son_b] and duyulan[son].startswith(beklenen[son_b]):
        hata += 1
    kirp_bas = max(0.0, duyulan_k[ilk][0].bas - BAS_PAY) if fazla_bas else 0.0
    # Whisper kelime sonunu erken işaretler: SON_PAY son ünsüzü korur; kelimeye dönüşmeyen uydurma sesi ("vav") yine keser.
    kirp_son = min(sure, duyulan_k[son][0].son + SON_PAY)
    if fazla_son:  # fazla kelime varsa onun başını geçme
        j0 = min(j for op, _, j in ops if op == "fazla" and j > son)
        kirp_son = min(kirp_son, max(duyulan_k[son][0].son + 0.05, duyulan_k[j0][0].bas - 0.02))
    return Degerlendirme(hata == 0, hata, kirp_bas, kirp_son, fazla_son, fazla_bas, " ".join(duyulan))


def kirp(wav: np.ndarray, sr: int, bas: float, son: float, fade_ms: float = 15) -> np.ndarray:
    out = np.array(wav[int(round(bas * sr)):int(round(son * sr))], dtype=np.float32)
    f = min(int(sr * fade_ms / 1000), len(out) // 2)
    if f > 0:
        out[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
        out[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return out
