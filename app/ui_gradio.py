"""Lokal Türkçe Ses Modeli — Gradio Arayüzü (TTS, VC, Profil Oluştur)."""

from __future__ import annotations

import json
import gradio as gr

from core.profiles import ConsentRecord, VoiceProfile, list_profiles
from core.text.prosody_planner import plan_prosody


def preview_prosody(raw_text: str) -> tuple[str, str]:
    """Metni normalize eder ve Türkçe prozodi planını okunabilir biçimde gösterir."""
    if not raw_text.strip():
        return "", "{}"
    plan = plan_prosody(raw_text)
    lines: list[str] = [f"Normalize: {plan.normalized_text}", ""]
    for idx, phrase in enumerate(plan.phrases, start=1):
        lines.append(f"[Öbek {idx}] ton={phrase.boundary_tone}, duraklama={phrase.pause_after_ms}ms")
        for w in phrase.words:
            syls = "-".join(
                s.upper() if i == w.stressed_syllable else s
                for i, s in enumerate(w.syllables)
            )
            tags = []
            if w.is_focus:
                tags.append("ODAK")
            if w.is_predicate:
                tags.append("YÜKLEM")
            if w.is_function:
                tags.append("İŞLEV")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"  • {w.text} ({syls}) | skor={w.prominence:+.1f}{tag_str}")
    summary = "\n".join(lines)
    raw_json = json.dumps(
        {
            "normalized_text": plan.normalized_text,
            "phrases": [
                {
                    "text": p.text,
                    "boundary_tone": p.boundary_tone,
                    "pause_after_ms": p.pause_after_ms,
                    "words": [
                        {
                            "text": w.text,
                            "syllables": w.syllables,
                            "stressed_syllable": w.stressed_syllable,
                            "prominence": w.prominence,
                            "is_focus": w.is_focus,
                        }
                        for w in p.words
                    ],
                }
                for p in plan.phrases
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    return summary, raw_json


def create_profile_ui(name: str, granted_by: str, scope: str) -> str:
    clean = name.strip().lower().replace(" ", "_")
    if not clean:
        return "Hata: Profil adı boş olamaz."
    if not granted_by.strip():
        return "Hata: İzin veren kişi bilgisi (KVKK/etik kayıt) zorunludur."
    profile = VoiceProfile(
        name=clean,
        consent=ConsentRecord(
            granted_by=granted_by.strip(),
            granted_at="manual-ui",
            scope=scope.strip() or "personal_local_only",
        ),
    )
    path = profile.save()
    return f"Profil oluşturuldu: {path}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Lokal Türkçe Ses Modeli") as demo:
        gr.Markdown("# Lokal Türkçe Ses Modeli\nTTS + Türkçe Prozodi Planlayıcı + VC + Ses Profili")

        with gr.Tab("TTS & Prozodi"):
            tts_input = gr.Textbox(
                label="Metin (işaretleme destekli: **vurgu**, <es 500>, <yavaş>...</yavaş>)",
                lines=4,
                value="Ali kitabı **dün** aldı. Yarın Ankara'ya gidecek, sonra seni arayacak.",
            )
            with gr.Row():
                plan_btn = gr.Button("Prozodi Planını Önizle", variant="primary")
            with gr.Row():
                plan_summary = gr.Textbox(label="Prozodi Planı Özeti", lines=14)
                plan_json = gr.Code(label="JSON Çıktısı", language="json")
            plan_btn.click(preview_prosody, inputs=[tts_input], outputs=[plan_summary, plan_json])

        with gr.Tab("Ses Dönüştürme (VC)"):
            gr.Markdown("Applio / RVC entegrasyonu Faz 1'de bağlanacaktır.")
            gr.Dropdown(label="Hedef Ses Profili", choices=list_profiles())

        with gr.Tab("Ses Profili Oluştur"):
            prof_name = gr.Textbox(label="Profil Adı (örn: ahmet)")
            prof_consent = gr.Textbox(label="İzin Veren Kişi (KVKK/etik kaydı)")
            prof_scope = gr.Textbox(label="Kullanım Kapsamı", value="personal_local_only")
            create_btn = gr.Button("Profili Oluştur")
            prof_status = gr.Textbox(label="Durum")
            create_btn.click(
                create_profile_ui,
                inputs=[prof_name, prof_consent, prof_scope],
                outputs=[prof_status],
            )
    return demo


if __name__ == "__main__":
    build_ui().launch()

