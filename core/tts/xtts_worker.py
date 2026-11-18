"""XTTS-v2 kalıcı çalışan (worker) süreci.

`envs/xtts` ortamında bir kez başlatılır, modeli bir kez yükler ve
stdin/stdout üzerinden satır bazlı JSON protokolüyle yanıt verir.
İstek:  {"text": str, "speaker_wav": str, "language": str, "speed": float}
Yanıt:  {"wav_path": str} veya {"error": str}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import soundfile as sf


def _detect_worker_device() -> str:
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _load_xtts_model(model_name: str, device: str) -> Any:
    """TTS API üzerinden XTTS-v2 modelini yükler ve hedef cihaza taşır."""
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    from TTS.api import TTS  # type: ignore

    tts = TTS(model_name).to(device)
    synthesizer = getattr(tts, "synthesizer", None)
    tts_model = getattr(synthesizer, "tts_model", None) if synthesizer is not None else None
    if tts_model is None:
        raise RuntimeError("XTTS synthesizer.tts_model yüklenemedi.")
    return tts_model


def load_gpt_checkpoint(model: Any, path: str) -> None:
    """İnce ayarlı GPT ağırlıklarını (scripts/xtts_ft/train_xpu.py çıktısı) yüklenmiş XTTS modeline uygular."""
    import torch  # type: ignore

    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("gpt", state) if isinstance(state, dict) else state
    # Eğitimde gpt_inference sarmalayıcısı yok; çıkarım modelinde var (aynı ağırlıkları paylaşır).
    res = model.gpt.load_state_dict(state, strict=False)
    eksik = [k for k in getattr(res, "missing_keys", []) if "gpt_inference" not in k and k != "gpt.wte.weight"]  # HF GPT2 wte kullanılmıyor (inputs_embeds)
    if eksik:
        raise RuntimeError(f"GPT ağırlıkları eksik: {eksik[:5]}")


def _to_numpy_audio(wav: Any) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def handle_request(
    req: dict[str, Any],
    model: Any,
    latent_cache: dict[tuple[str, int], tuple[Any, Any]],
) -> dict[str, str]:
    """Tek bir sentez isteğini işler ve geçici wav dosyasının yolunu döner."""
    text = str(req.get("text", "")).strip()
    if not text:
        return {"error": "Boş metin sentezlenemez."}

    speaker_wav_raw = req.get("speaker_wav")
    if not speaker_wav_raw:
        return {"error": "speaker_wav alanı gereklidir."}

    speaker_path = Path(str(speaker_wav_raw))
    if not speaker_path.exists():
        return {"error": f"Referans ses bulunamadı: {speaker_path}"}

    language = str(req.get("language", "tr"))
    speed = float(req.get("speed", 1.0))

    cache_key = (str(speaker_path.resolve()), speaker_path.stat().st_mtime_ns)
    if cache_key not in latent_cache:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(speaker_path)]
        )
        latent_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
    else:
        gpt_cond_latent, speaker_embedding = latent_cache[cache_key]

    # Örnekleme ayarları: düşük temperature + yüksek repetition_penalty, sondaki uydurma heceleri azaltır.
    sampling = {k: float(req[k]) for k in ("temperature", "repetition_penalty", "top_p", "length_penalty") if k in req}
    if "top_k" in req:
        sampling["top_k"] = int(req["top_k"])
    if "seed" in req:
        try:
            import torch  # type: ignore

            torch.manual_seed(int(req["seed"]))
        except ImportError:
            pass

    out = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        speed=speed,
        **sampling,
    )
    wav = _to_numpy_audio(out["wav"])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)
    sf.write(out_path, wav, 24000)
    return {"wav_path": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="XTTS-v2 kalıcı JSON worker")
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS model adı",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Hedef cihaz (xpu, cuda, cpu). Boşsa otomatik algılanır.",
    )
    parser.add_argument("--gpt-checkpoint", default=None, help="İnce ayarlı GPT ağırlıkları (.pth)")
    args = parser.parse_args(argv)

    device = args.device or _detect_worker_device()
    with contextlib.redirect_stdout(sys.stderr):
        model = _load_xtts_model(args.model, device)
        if args.gpt_checkpoint:
            load_gpt_checkpoint(model, args.gpt_checkpoint)
            if hasattr(model.gpt, "init_gpt_for_inference"):
                model.gpt.init_gpt_for_inference(kv_cache=True, use_deepspeed=False)
            model.to(device)

    latent_cache: dict[tuple[str, int], tuple[Any, Any]] = {}

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("command") == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                break
            with contextlib.redirect_stdout(sys.stderr):
                resp = handle_request(req, model=model, latent_cache=latent_cache)
        except Exception as exc:
            resp = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(resp, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
