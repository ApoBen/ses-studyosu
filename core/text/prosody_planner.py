"""Türkçe prozodi planlayıcı: öbekleme, vurgu skorlama, sınır tonları ve duraklamalar."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from core.text.markup import MarkupSpan, parse_markup
from core.text.normalize import normalize_turkish_text
from core.text.stress import analyze_word_stress, tr_lower

FUNCTION_WORDS = {
    "ve", "ile", "veya", "ya", "yahut", "ama", "fakat", "lakin", "ancak",
    "çünkü", "ki", "de", "da", "bu", "şu", "o", "bir", "her", "hiç",
    "için", "gibi", "kadar", "göre", "doğru", "karşı", "rağmen", "üzere",
    "mı", "mi", "mu", "mü",
}

QUESTION_WORDS = {
    "kim", "kimi", "kime", "kimde", "kimden", "kimin",
    "ne", "neyi", "neye", "nede", "neden", "niye", "niçin",
    "nerede", "nereye", "nereden", "neresi",
    "nasıl", "hangi", "hangisi", "kaç", "kaçıncı", "ne zaman",
}

QUESTION_PARTICLES = {"mı", "mi", "mu", "mü"}
_QUESTION_PARTICLE_RE = re.compile(
    r"^(?:m[ıiuü])(?:y?[ıiuü]m|s[ıiuü]n|y?[ıiuü]z|s[ıiuü]n[ıiuü]z|"
    r"yd[ıiuü]m?|yd[ıiuü]n|yd[ıiuü]k|yd[ıiuü]n[ıiuü]z|"
    r"ym[ıiuü]ş|ys[ae]|d[ıiuü]r|l[ae]r)?$"
)
FOCUS_PARTICLES = {"de", "da", "bile", "sadece", "yalnız", "hatta", "üstelik"}
PHRASE_BREAK_CONJUNCTIONS = {"ama", "fakat", "ancak", "lakin", "çünkü", "yoksa", "veya"}

# Türkçe yaygın fiil çekim ekleri (Stanza olmadan hafif yüklem tespiti için yedek sezgi)
_PREDICATE_SUFFIX_RE = re.compile(
    r"(?:yor|yordu|yorum|yorsun|yoruz|yorsunuz|yorlar|"
    r"dı|di|du|dü|tı|ti|tu|tü|"
    r"dım|dim|dum|düm|tım|tim|tum|tüm|"
    r"dın|din|dun|dün|tın|tin|tun|tün|"
    r"dık|dik|duk|dük|tık|tik|tuk|tük|"
    r"acak|ecek|acağım|eceğim|acaksın|eceksin|"
    r"mış|miş|muş|müş|mıştır|miştir|muştur|müştür|"
    r"malı|meli|malıyım|meliyim|abilir|ebilir)$"
)

PUNCT_PAUSE_MS: dict[str, tuple[str, int]] = {
    "...": ("L-...", 700),
    ".": ("L%", 500),
    ",": ("H-", 200),
    ";": ("L-", 350),
    ":": ("M%", 300),
    "?": ("H%", 500),
    "!": ("HL%", 500),
}

_STANZA_NLP: Any | None = None
_STANZA_INIT_ATTEMPTED = False


def _get_stanza_nlp() -> Any | None:
    """Stanza Türkçe bağımlılık ayrıştırıcısını tembel yükler; yoksa None döner."""
    global _STANZA_NLP, _STANZA_INIT_ATTEMPTED
    if _STANZA_INIT_ATTEMPTED:
        return _STANZA_NLP
    _STANZA_INIT_ATTEMPTED = True
    try:
        import stanza

        _STANZA_NLP = stanza.Pipeline(
            "tr",
            processors="tokenize,mwt,pos,lemma,depparse",
            download_method=None,
            verbose=False,
        )
    except Exception:
        _STANZA_NLP = None
    return _STANZA_NLP


@dataclass
class WordPlan:
    text: str
    syllables: list[str]
    stressed_syllable: int
    prominence: float
    is_function: bool = False
    is_predicate: bool = False
    is_focus: bool = False
    stress_ambiguous: bool = False
    rate: float = 1.0
    emotion: str | None = None
    pronunciation: str | None = None
    tone: str | None = None


@dataclass
class PhrasePlan:
    text: str
    words: list[WordPlan] = field(default_factory=list)
    boundary_tone: str = "M%"
    pause_after_ms: int = 0
    rate: float = 1.0
    emotion: str | None = None
    question_type: str = "none"


@dataclass
class ProsodyPlan:
    raw_text: str
    normalized_text: str
    phrases: list[PhrasePlan] = field(default_factory=list)


@dataclass
class _SpanRange:
    start: int
    end: int
    emphasis: bool = False
    rate: float = 1.0
    emotion: str | None = None
    pronunciation: str | None = None


@dataclass
class _PauseMarker:
    offset: int
    pause_ms: int


@dataclass
class _SentenceToken:
    raw: str
    clean: str
    start: int  # cümle içi karakter başlangıcı
    end: int    # cümle içi karakter bitişi
    doc_start: int  # normalize edilmiş tam metin içi başlangıç
    doc_end: int    # normalize edilmiş tam metin içi bitiş
    is_predicate: bool = False
    is_aux_q: bool = False


def _is_question_particle(word_lower: str) -> bool:
    return word_lower in QUESTION_PARTICLES or bool(_QUESTION_PARTICLE_RE.match(word_lower))


def _looks_like_predicate(word_lower: str) -> bool:
    clean = re.sub(r"[^a-zçğıöşü]", "", word_lower)
    if len(clean) < 3 or clean in FUNCTION_WORDS or _is_question_particle(clean):
        return False
    return bool(_PREDICATE_SUFFIX_RE.search(clean))


def _normalize_with_spans(
    spans: list[MarkupSpan],
) -> tuple[str, list[_SpanRange], list[_PauseMarker]]:
    """İşaretleme parçalarını ayrı ayrı normalize edip tam karakter aralıklarını korur."""
    parts: list[str] = []
    span_ranges: list[_SpanRange] = []
    pause_markers: list[_PauseMarker] = []
    cursor = 0

    for span in spans:
        raw_chunk = span.text
        had_leading_ws = bool(raw_chunk) and raw_chunk[0].isspace()
        had_trailing_ws = bool(raw_chunk) and raw_chunk[-1].isspace()
        norm_chunk = normalize_turkish_text(raw_chunk) if raw_chunk.strip() else ""

        if norm_chunk:
            starts_with_punct = norm_chunk[0] in ".,!?;:)"
            if parts and not starts_with_punct:
                last_char = parts[-1][-1]
                if had_leading_ws or not last_char.isspace():
                    parts.append(" ")
                    cursor += 1
            start = cursor
            parts.append(norm_chunk)
            cursor += len(norm_chunk)
            end = cursor

            if (
                span.emphasis
                or span.rate != 1.0
                or span.emotion is not None
                or span.pronunciation is not None
            ):
                span_ranges.append(
                    _SpanRange(
                        start=start,
                        end=end,
                        emphasis=span.emphasis,
                        rate=span.rate,
                        emotion=span.emotion,
                        pronunciation=span.pronunciation,
                    )
                )

            if had_trailing_ws and not (parts and parts[-1].endswith(" ")):
                parts.append(" ")
                cursor += 1

        if span.pause_after_ms > 0:
            pause_markers.append(_PauseMarker(offset=cursor, pause_ms=span.pause_after_ms))

    normalized = "".join(parts).strip()
    return normalized, span_ranges, pause_markers


def _split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Metni cümlelere böler ve (cümle_metni, başlangıç_ofseti, bitiş_ofseti) döner."""
    if not text.strip():
        return []
    result: list[tuple[str, int, int]] = []
    start = 0
    for m in re.finditer(r"(?<=[.!?…])\s+|(?<=\.\.\.)\s+", text):
        seg = text[start : m.start()].strip()
        if seg:
            seg_start = text.index(seg, start)
            result.append((seg, seg_start, seg_start + len(seg)))
        start = m.end()
    tail = text[start:].strip()
    if tail:
        seg_start = text.index(tail, start)
        result.append((tail, seg_start, seg_start + len(tail)))
    return result


def _split_into_phrases_with_offsets(
    sentence: str,
    pause_offsets_in_sentence: list[tuple[int, int]],
) -> list[tuple[str, int, int, str, int]]:
    """Cümleyi öbeklere böler: [(metin, cümle_içi_başlangıç, cümle_içi_bitiş, sınır_tonu, duraklama_ms)]."""
    # Önce duraklama işaretlerinin (<es N>) bulunduğu noktalara göre alt bölümlere ayır
    sorted_pauses = sorted(pause_offsets_in_sentence, key=lambda x: x[0])
    segments: list[tuple[str, int, int, int]] = []  # (text, rel_start, rel_end, explicit_pause)
    cur = 0
    for p_off, p_ms in sorted_pauses:
        if cur < p_off <= len(sentence):
            sub = sentence[cur:p_off].strip()
            if sub:
                sub_start = sentence.index(sub, cur)
                segments.append((sub, sub_start, sub_start + len(sub), p_ms))
            elif segments:
                prev_txt, ps, pe, prev_ms = segments[-1]
                segments[-1] = (prev_txt, ps, pe, max(prev_ms, p_ms))
            cur = p_off
    if cur < len(sentence):
        sub = sentence[cur:].strip()
        if sub:
            sub_start = sentence.index(sub, cur)
            segments.append((sub, sub_start, sub_start + len(sub), 0))

    if not segments:
        segments = [(sentence, 0, len(sentence), 0)]

    final_chunks: list[tuple[str, int, int, str, int]] = []
    for seg_text, seg_start, seg_end, explicit_pause in segments:
        tokens = re.split(r"(\.\.\.|[.,;:!?])", seg_text)
        raw_chunks: list[tuple[str, str, int, int]] = []
        buf = ""
        buf_start = seg_start
        local_pos = seg_start

        for tok in tokens:
            if tok in PUNCT_PAUSE_MS:
                if buf.strip():
                    stripped = buf.strip()
                    s_idx = sentence.index(stripped, buf_start)
                    raw_chunks.append((stripped, tok, s_idx, s_idx + len(stripped)))
                elif raw_chunks:
                    prev_txt, _, ps, pe = raw_chunks[-1]
                    raw_chunks[-1] = (prev_txt, tok, ps, pe)
                buf = ""
                local_pos += len(tok)
                buf_start = local_pos
            else:
                buf += tok
                local_pos += len(tok)
        if buf.strip():
            stripped = buf.strip()
            s_idx = sentence.index(stripped, buf_start)
            raw_chunks.append((stripped, "", s_idx, s_idx + len(stripped)))

        for idx_c, (chunk_text, punct, c_start, c_end) in enumerate(raw_chunks):
            is_last_in_seg = idx_c == len(raw_chunks) - 1
            words = chunk_text.split()
            if len(words) >= 8:
                split_idx = None
                for i in range(3, len(words) - 2):
                    if tr_lower(words[i]) in PHRASE_BREAK_CONJUNCTIONS:
                        split_idx = i
                        break
                if split_idx is not None:
                    left = " ".join(words[:split_idx])
                    right = " ".join(words[split_idx:])
                    left_start = c_start
                    right_start = sentence.index(words[split_idx], c_start)
                    final_chunks.append((left, left_start, left_start + len(left), "H-", 160))
                    tone, pause = PUNCT_PAUSE_MS.get(punct, ("M%", 0))
                    if is_last_in_seg and explicit_pause > 0:
                        pause = max(pause, explicit_pause)
                    final_chunks.append((right, right_start, c_end, tone, pause))
                    continue

            tone, pause = PUNCT_PAUSE_MS.get(punct, ("M%", 0))
            if is_last_in_seg and explicit_pause > 0:
                pause = max(pause, explicit_pause)
            final_chunks.append((chunk_text, c_start, c_end, tone, pause))

    return final_chunks


def _analyze_sentence_tokens(
    sentence: str,
    sent_doc_start: int,
) -> tuple[list[_SentenceToken], int | None]:
    """Cümledeki kelimeleri çıkarır ve ana yüklem indeksini (Stanza varsa root) bulur."""
    tokens: list[_SentenceToken] = []
    for m in re.finditer(r"\S+", sentence):
        raw_w = m.group(0)
        clean_w = tr_lower(re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]", "", raw_w))
        tokens.append(
            _SentenceToken(
                raw=raw_w,
                clean=clean_w,
                start=m.start(),
                end=m.end(),
                doc_start=sent_doc_start + m.start(),
                doc_end=sent_doc_start + m.end(),
            )
        )

    nlp = _get_stanza_nlp()
    pred_idx: int | None = None

    if nlp is not None and sentence.strip():
        try:
            doc = nlp(sentence)
            stanza_words = [w for sent in doc.sentences for w in sent.words]
            for idx, tok in enumerate(tokens):
                for sw in stanza_words:
                    sw_start = getattr(sw, "start_char", None)
                    sw_end = getattr(sw, "end_char", None)
                    if sw_start is None or sw_end is None:
                        continue
                    if sw_start < tok.end and sw_end > tok.start:
                        if sw.deprel == "root" or (
                            sw.deprel in ("conj", "parataxis")
                            and getattr(sw, "upos", None) in ("VERB", "AUX")
                        ):
                            tok.is_predicate = True
                            if pred_idx is None:
                                pred_idx = idx
                        if sw.deprel == "aux:q":
                            tok.is_aux_q = True
        except Exception:
            pred_idx = None

    if pred_idx is None:
        # Stanza yoksa veya root bulunamadıysa regex yedeği
        for idx in range(len(tokens) - 1, -1, -1):
            if _looks_like_predicate(tokens[idx].clean):
                tokens[idx].is_predicate = True
                pred_idx = idx
                break

    return tokens, pred_idx


def _detect_question_type(sentence: str, tokens: list[_SentenceToken]) -> str:
    """Cümle soru tipini döndürür: 'yes_no', 'wh' veya 'none'."""
    if "?" not in sentence:
        return "none"
    if any(t.is_aux_q or _is_question_particle(t.clean) for t in tokens):
        return "yes_no"
    if any(t.clean in QUESTION_WORDS for t in tokens):
        return "wh"
    return "yes_no"


def plan_prosody(raw_text: str) -> ProsodyPlan:
    """Metni normalize eder, işaretlemeyi çözer ve Türkçe prozodi planı üretir."""
    parsed = parse_markup(raw_text)
    normalized, span_ranges, pause_markers = _normalize_with_spans(parsed.spans)

    phrases: list[PhrasePlan] = []
    seen_words: set[str] = set()

    for sentence, sent_start, sent_end in _split_sentences_with_offsets(normalized):
        sent_tokens, pred_idx = _analyze_sentence_tokens(sentence, sent_start)
        q_type = _detect_question_type(sentence, sent_tokens)

        # Cümle düzeyinde odak indeksleri: her yüklemin hemen önündeki kelime
        # (arada mI veya odak edatı varsa ondan önceki)
        pred_indices = [i for i, t in enumerate(sent_tokens) if t.is_predicate]
        last_pred_idx = pred_indices[-1] if pred_indices else pred_idx
        pre_pred_focus_indices: set[int] = set()
        for p_i in pred_indices:
            if p_i > 0:
                candidate = p_i - 1
                while candidate >= 0 and (
                    sent_tokens[candidate].is_aux_q
                    or _is_question_particle(sent_tokens[candidate].clean)
                    or sent_tokens[candidate].clean in FOCUS_PARTICLES
                ):
                    candidate -= 1
                if candidate >= 0:
                    pre_pred_focus_indices.add(candidate)

        # Bu cümle içindeki <es N> duraklama sınırlarını topla
        sent_pauses = [
            (pm.offset - sent_start, pm.pause_ms)
            for pm in pause_markers
            if sent_start < pm.offset <= sent_end
        ]

        phrase_chunks = _split_into_phrases_with_offsets(sentence, sent_pauses)

        for phrase_idx, (phrase_text, p_start, p_end, boundary_tone, pause_ms) in enumerate(phrase_chunks):
            is_last_phrase = phrase_idx == len(phrase_chunks) - 1
            if is_last_phrase and boundary_tone == "H%" and q_type in ("yes_no", "wh"):
                boundary_tone = "L%"

            # Öbeğe düşen cümle token'ları (noktalama ayrıldığı için örtüşme ile eşleştir)
            phrase_tok_indices = [
                idx for idx, t in enumerate(sent_tokens) if t.start < p_end and t.end > p_start
            ]

            word_plans: list[WordPlan] = []
            phrase_emotions: list[str] = []

            for idx in phrase_tok_indices:
                tok = sent_tokens[idx]
                # Öbek sınırındaki noktalamayı WordPlan.text dışında tut (ör. "aldı." -> "aldı")
                ov_start = max(tok.start, p_start) - tok.start
                ov_end = min(tok.end, p_end) - tok.start
                raw_w = tok.raw[ov_start:ov_end] or tok.raw
                clean_w = tok.clean
                is_q_part = tok.is_aux_q or _is_question_particle(clean_w)
                is_focus_particle = clean_w in FOCUS_PARTICLES
                is_func = clean_w in FUNCTION_WORDS or is_q_part or is_focus_particle
                is_pred = tok.is_predicate
                next_tok = sent_tokens[idx + 1] if idx + 1 < len(sent_tokens) else None
                next_is_q_part = (
                    next_tok is not None
                    and (next_tok.is_aux_q or _is_question_particle(next_tok.clean))
                )
                next_clean = next_tok.clean if next_tok is not None else ""

                # Karakter konumuna göre işaretleme eşleştirme
                has_emphasis = False
                word_rate = 1.0
                word_emotion: str | None = None
                word_pron: str | None = None
                for sr in span_ranges:
                    if tok.doc_start < sr.end and tok.doc_end > sr.start:
                        if sr.emphasis:
                            has_emphasis = True
                        if sr.rate != 1.0:
                            word_rate = sr.rate
                        if sr.emotion is not None:
                            word_emotion = sr.emotion
                            phrase_emotions.append(sr.emotion)
                        if sr.pronunciation is not None:
                            word_pron = sr.pronunciation

                score = 0.0
                is_focus = False
                word_tone: str | None = None

                if is_q_part:
                    # Soru ekleri asla odak olmasın, skorları negatif olsun
                    score = -0.5
                    if pred_idx is not None and idx > pred_idx:
                        score -= 1.0
                else:
                    if not is_func:
                        score += 1.0
                    else:
                        score -= 0.5

                    # Yüklem öncesi odak kuralı
                    if idx in pre_pred_focus_indices:
                        score += 2.0
                        is_focus = True

                    # Soru kelimesi
                    if clean_w in QUESTION_WORDS:
                        score += 2.0
                        is_focus = True

                    # mI soru ekinden hemen önceki kelime: H* yükselişi + odak (odak edatı hariç)
                    if next_is_q_part and not is_focus_particle:
                        if idx not in pre_pred_focus_indices:
                            score += 2.0
                        is_focus = True
                        word_tone = "H*"

                    if next_clean in FOCUS_PARTICLES and not is_focus_particle:
                        score += 1.0

                    # Odak sonrası sıkıştırma: yüklem ve son yüklem sonrası (devrik bölge)
                    if is_pred and idx > 0:
                        score -= 1.0
                    if last_pred_idx is not None and idx > last_pred_idx:
                        score -= 1.0

                    # Virgül öncesi hafif yükselme
                    if boundary_tone == "H-" and idx == phrase_tok_indices[-1]:
                        score += 0.5

                    # Yeni bilgi
                    if clean_w and clean_w not in seen_words and not is_func:
                        score += 0.5
                        seen_words.add(clean_w)

                    # Kullanıcı vurgusu (**kelime**) - sadece işaretlenen konumda
                    if has_emphasis:
                        score += 3.0
                        is_focus = True

                syllables, stressed_idx, stress_ambiguous = analyze_word_stress(raw_w)
                word_plans.append(
                    WordPlan(
                        text=raw_w,
                        syllables=syllables,
                        stressed_syllable=stressed_idx,
                        prominence=round(score, 2),
                        is_function=is_func,
                        is_predicate=is_pred,
                        is_focus=is_focus,
                        stress_ambiguous=stress_ambiguous,
                        rate=word_rate,
                        emotion=word_emotion,
                        pronunciation=word_pron,
                        tone=word_tone,
                    )
                )

            phrase_emotion = phrase_emotions[0] if phrase_emotions else None
            phrases.append(
                PhrasePlan(
                    text=phrase_text,
                    words=word_plans,
                    boundary_tone=boundary_tone,
                    pause_after_ms=pause_ms,
                    emotion=phrase_emotion,
                    question_type=q_type if is_last_phrase or q_type != "none" else "none",
                )
            )

    return ProsodyPlan(
        raw_text=raw_text,
        normalized_text=normalized,
        phrases=phrases,
    )
