"""Hafif prozodi işaretleme dili (SSML benzeri) ayrıştırıcısı.

Desteklenen etiketler:
  - **kelime**                    -> kullanıcı vurgusu
  - <es 500>                      -> 500 ms duraklama
  - <yavaş>metin</yavaş>          -> yavaş konuşma hızı (rate=0.82)
  - <hızlı>metin</hızlı>          -> hızlı konuşma hızı (rate=1.18)
  - <duygu=heyecanli>metin</duygu> -> duygu etiketi
  - <oku=İs-TAN-bul>İstanbul</oku> -> özel telaffuz/vurgu geçersiz kılma
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class MarkupSpan:
    """İşaretlenmiş metin parçası."""

    text: str
    emphasis: bool = False
    rate: float = 1.0
    emotion: str | None = None
    pronunciation: str | None = None
    pause_after_ms: int = 0


@dataclass
class ParsedMarkup:
    """Ayrıştırılmış metin ve parça listesi."""

    plain_text: str
    spans: list[MarkupSpan] = field(default_factory=list)


_TOKEN_RE = re.compile(
    r"(<es\s+(\d+)\s*/?>)"
    r"|(\*\*(.+?)\*\*)"
    r"|(<yavaş>(.*?)</yavaş>)"
    r"|(<hızlı>(.*?)</hızlı>)"
    r"|(<duygu=([a-zA-ZçğıöşüÇĞİÖŞÜ_-]+)>(.*?)</duygu>)"
    r"|(<oku=([^>]+)>(.*?)</oku>)",
    re.DOTALL,
)


def parse_markup(raw_text: str) -> ParsedMarkup:
    """İşaretleme etiketlerini ayrıştırıp düz metin ve MarkupSpan listesi döner."""
    spans: list[MarkupSpan] = []
    plain_parts: list[str] = []
    pos = 0

    for match in _TOKEN_RE.finditer(raw_text):
        if match.start() > pos:
            chunk = raw_text[pos : match.start()]
            if chunk:
                spans.append(MarkupSpan(text=chunk))
                plain_parts.append(chunk)

        if match.group(1):  # <es 500>
            pause_ms = int(match.group(2))
            if spans:
                spans[-1].pause_after_ms += pause_ms
            else:
                spans.append(MarkupSpan(text="", pause_after_ms=pause_ms))
            plain_parts.append(" ")
        elif match.group(3):  # **kelime**
            inner = match.group(4).strip()
            spans.append(MarkupSpan(text=inner, emphasis=True))
            plain_parts.append(inner)
        elif match.group(5):  # <yavaş>...</yavaş>
            inner = match.group(6)
            spans.append(MarkupSpan(text=inner, rate=0.82))
            plain_parts.append(inner)
        elif match.group(7):  # <hızlı>...</hızlı>
            inner = match.group(8)
            spans.append(MarkupSpan(text=inner, rate=1.18))
            plain_parts.append(inner)
        elif match.group(9):  # <duygu=...>...</duygu>
            emotion = match.group(10).lower()
            inner = match.group(11)
            spans.append(MarkupSpan(text=inner, emotion=emotion))
            plain_parts.append(inner)
        elif match.group(12):  # <oku=...>...</oku>
            pron = match.group(13).strip()
            inner = match.group(14).strip()
            spans.append(MarkupSpan(text=inner, pronunciation=pron))
            plain_parts.append(inner)

        pos = match.end()

    if pos < len(raw_text):
        tail = raw_text[pos:]
        if tail:
            spans.append(MarkupSpan(text=tail))
            plain_parts.append(tail)

    plain_text = re.sub(r"\s{2,}", " ", "".join(plain_parts)).strip()
    return ParsedMarkup(plain_text=plain_text, spans=spans)

