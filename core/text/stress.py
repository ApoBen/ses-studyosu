"""Türkçe hece ayırma ve morfolojik analiz tabanlı kelime vurgusu kuralları + istisna sözlüğü."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

VOWELS = set("aeıioöuüAEIİOÖUÜ")

# Vurgusuz ek etiketleri (Zeyrek/Zemberek morfem kimlikleri)
_ALWAYS_UNSTRESSED_TAGS = {"Neg", "Ques", "Cop", "While", "Ins", "Ly"}
_COPULA_TAM_TAGS = {"Past", "Narr", "Cond"}
_FINITE_TAM_TAGS = {"Prog1", "Prog2", "Fut", "Aor", "Past", "Narr", "Desr", "Cond", "Opt", "Neces", "Imp"}
_PRES_COPULA_PERSON_TAGS = {"A1sg", "A1pl", "A2sg", "A2pl"}

_POS_PRIORITY = {
    "Noun": 0,
    "Adj": 1,
    "Adv": 1,
    "Pron": 1,
    "Num": 1,
    "Det": 1,
    "Conj": 1,
    "Postp": 1,
    "Interj": 1,
    "Verb": 2,
    "Ques": 3,
}

DEFAULT_EXCEPTIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "stress_exceptions_tr.txt"

_MORPH_ANALYZER: Any | None = None
_MORPH_INIT_TRIED = False


def _get_morph_analyzer() -> Any | None:
    """Zeyrek MorphAnalyzer örneğini tembel yükler; yoksa None döner."""
    global _MORPH_ANALYZER, _MORPH_INIT_TRIED
    if _MORPH_INIT_TRIED:
        return _MORPH_ANALYZER
    _MORPH_INIT_TRIED = True
    try:
        logging.getLogger("zeyrek").setLevel(logging.ERROR)
        import zeyrek

        _MORPH_ANALYZER = zeyrek.MorphAnalyzer()
    except Exception:
        _MORPH_ANALYZER = None
    return _MORPH_ANALYZER


def tr_lower(text: str) -> str:
    """Türkçe I/İ kurallarına uygun küçük harfe çevirme."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def syllabify(word: str) -> list[str]:
    """Türkçe kelimeyi hecelere ayırır (her hecede bir ünlü)."""
    clean = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ]", "", word)
    if not clean:
        return []

    vowel_positions = [i for i, ch in enumerate(clean) if ch in VOWELS]
    if not vowel_positions:
        return [clean]
    if len(vowel_positions) == 1:
        return [clean]

    syllables: list[str] = []
    start = 0
    for idx in range(len(vowel_positions) - 1):
        v1 = vowel_positions[idx]
        v2 = vowel_positions[idx + 1]
        consonants_between = v2 - v1 - 1
        if consonants_between == 0:
            split_at = v1 + 1
        elif consonants_between == 1:
            split_at = v1 + 1
        else:
            # İki veya daha fazla ünsüz varsa son ünsüz sonraki heceye geçer
            split_at = v2 - 1
        syllables.append(clean[start:split_at])
        start = split_at

    syllables.append(clean[start:])
    return syllables


def load_stress_exceptions(path: Path | None = None) -> dict[str, int]:
    """İstisna vurgulu kelimeler sözlüğünü yükler."""
    target = path or DEFAULT_EXCEPTIONS_PATH
    exceptions: dict[str, int] = {}
    if not target.exists():
        return exceptions
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                exceptions[tr_lower(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                continue
    return exceptions


_EXCEPTIONS_CACHE: dict[str, int] | None = None


def get_stress_exceptions() -> dict[str, int]:
    global _EXCEPTIONS_CACHE
    if _EXCEPTIONS_CACHE is None:
        _EXCEPTIONS_CACHE = load_stress_exceptions()
    return _EXCEPTIONS_CACHE


def _syllable_spans(syllables: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for syl in syllables:
        spans.append((pos, pos + len(syl)))
        pos += len(syl)
    return spans


def _char_to_syllable(char_idx: int, spans: list[tuple[int, int]]) -> int:
    for idx, (start, end) in enumerate(spans):
        if start <= char_idx < end:
            return idx
    return max(0, len(spans) - 1)


def _stress_from_analysis(parse: Any, syllables: list[str], spans: list[tuple[int, int]]) -> int | None:
    """Tek bir morfolojik analizden vurgulu hece indeksini çıkarır."""
    morphemes = [(m[0].id_, str(m[1])) for m in parse.morphemes]
    total_len = sum(len(surf) for _, surf in morphemes)
    expected_len = spans[-1][1] if spans else 0
    if total_len != expected_len:
        return None

    char_pos = 0
    prev_tag = ""
    saw_zero_verb = False
    saw_tam = False
    saw_pres_copula = False

    for tag, surf in morphemes:
        if tag == "Verb" and prev_tag == "Zero":
            saw_zero_verb = True
        if tag == "Pres" and saw_zero_verb:
            saw_pres_copula = True

        is_unstressed = False
        if surf:
            if tag in _ALWAYS_UNSTRESSED_TAGS:
                is_unstressed = True
            elif tag in _COPULA_TAM_TAGS and (saw_zero_verb or saw_tam):
                is_unstressed = True
            elif tag in _PRES_COPULA_PERSON_TAGS and saw_pres_copula:
                is_unstressed = True

        if is_unstressed:
            surf_lower = tr_lower(surf)
            v_idxs = [i for i, ch in enumerate(surf_lower) if ch in VOWELS]
            if tag == "Ins" and len(v_idxs) > 1:
                offset = v_idxs[-1]
            elif v_idxs:
                offset = v_idxs[0]
            else:
                offset = 0
            syl_idx = _char_to_syllable(char_pos + offset, spans)
            return max(0, syl_idx - 1)

        if tag in _FINITE_TAM_TAGS:
            saw_tam = True

        char_pos += len(surf)
        prev_tag = tag

    return len(syllables) - 1


def _pos_rank(parse: Any) -> int:
    pos_val = getattr(getattr(parse, "pos", None), "value", str(getattr(parse, "pos", "")))
    return _POS_PRIORITY.get(pos_val, 3)


def analyze_word_stress(
    word: str,
    exceptions: dict[str, int] | None = None,
) -> tuple[list[str], int, bool]:
    """Kelimenin hecelerini, vurgulu hece indeksini ve analiz belirsizliğini döner."""
    syllables = syllabify(word)
    if not syllables:
        return [], 0, False
    if len(syllables) == 1:
        return syllables, 0, False

    exc_map = exceptions if exceptions is not None else get_stress_exceptions()
    lower = tr_lower(re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ']", "", word))
    stem = lower.split("'")[0]

    # 1. İstisna sözlüğü kontrolü (tam kelime veya kesme öncesi kök)
    for key in (lower, stem):
        if key in exc_map:
            idx = exc_map[key]
            if idx < 0:
                idx = max(0, len(syllables) + idx)
            return syllables, min(idx, len(syllables) - 1), False

    # 2. Morfolojik analiz ile vurgusuz ek tespiti
    analyzer = _get_morph_analyzer()
    if analyzer is not None:
        clean_word = tr_lower("".join(syllables))
        try:
            parses = analyzer._parse(clean_word)
        except Exception:
            parses = []

        if parses:
            spans = _syllable_spans(syllables)
            candidates: list[tuple[int, int]] = []
            for parse in parses:
                idx = _stress_from_analysis(parse, syllables, spans)
                if idx is not None:
                    candidates.append((_pos_rank(parse), idx))

            if candidates:
                distinct_indices = {idx for _, idx in candidates}
                best_rank, best_idx = min(candidates, key=lambda item: item[0])
                _ = best_rank
                return syllables, best_idx, len(distinct_indices) > 1

    # 3. Yedek kural: analiz başarısızsa veya zeyrek yoksa son hece vurgulu
    return syllables, len(syllables) - 1, False


def find_stressed_syllable(word: str, exceptions: dict[str, int] | None = None) -> tuple[list[str], int]:
    """Kelimenin hecelerini ve vurgulu hece indeksini (0-tabanlı) döner."""
    syllables, idx, _ = analyze_word_stress(word, exceptions=exceptions)
    return syllables, idx

