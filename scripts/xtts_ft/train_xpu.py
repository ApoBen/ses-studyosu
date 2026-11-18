"""XTTS-v2 GPT ince ayarı — Intel Arc (XPU) için kendi döngümüz (coqui Trainer sadece CUDA destekliyor).

Kullanım (envs/xtts):
  train_xpu.py --veri profiles/<profil>/tts --cikti models/xtts_ft/<profil> --epoch 8 [--batch 2] [--lr 5e-6] [--adim N]

- Sadece GPT transformer + başlıklar eğitilir; koşullandırma kodlayıcıları, DVAE ve HiFi-GAN donuk.
- Bellek: Adafactor (AdamW 8 GB'a sığmaz), bf16 autocast.
- Level Zero host belleği sızıntısı için: her epoch sonunda checkpoint (gpt + optimizer + sayaç) ve çıkış kodu 42;
  dışarıdaki döngü yeniden başlatır, betik kaldığı yerden devam eder. Bitince çıkış 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
import torch
from transformers.optimization import Adafactor
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

ROOT = Path(__file__).resolve().parents[2]


def _xtts_klasoru() -> Path:
    """XTTS-v2 ağırlıklarının (model.pth, config.json, vocab.json, mel_stats.pth) yerel klasörü: XTTS_DIR veya bilinen önbellekler."""
    if os.environ.get("XTTS_DIR"):
        return Path(os.environ["XTTS_DIR"])
    adaylar = sorted(Path.home().glob(".cache/huggingface/hub/models--*XTTS-v2/snapshots/*")) + \
        [Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"]
    for a in adaylar:
        if (a / "model.pth").exists():
            return a
    sys.exit("XTTS-v2 ağırlıkları bulunamadı; XTTS_DIR ortam değişkenini model klasörüne ayarlayın.")


S = _xtts_klasoru()
DVAE = Path(os.environ.get("XTTS_DVAE", ROOT / "models/xtts_ft/dvae.pth"))

ap = argparse.ArgumentParser()
ap.add_argument("--veri", required=True)
ap.add_argument("--cikti", required=True)
ap.add_argument("--epoch", type=int, default=8)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--biriktir", type=int, default=4, help="gradyan biriktirme adımı (etkin batch = batch*biriktir)")
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--max-wav", type=int, default=255995, help="örnek sayısı (22,05 kHz); uzun klipler XPU belleğini aşıyor")
ap.add_argument("--adim", type=int, default=0, help=">0 ise sadece bu kadar adım (ölçüm/duman testi)")
ap.add_argument("--cihaz", default="xpu")
a = ap.parse_args()
out = Path(a.cikti); out.mkdir(parents=True, exist_ok=True)
dev = torch.device(a.cihaz)
torch.manual_seed(1)

model_args = GPTArgs(
    max_conditioning_length=132300, min_conditioning_length=66150, debug_loading_failures=False,
    max_wav_length=a.max_wav, max_text_length=200, mel_norm_file=str(S / "mel_stats.pth"),
    dvae_checkpoint=str(DVAE), xtts_checkpoint=str(S / "model.pth"), tokenizer_file=str(S / "vocab.json"),
    gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True)
audio = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
config = GPTTrainerConfig(output_path=str(out), model_args=model_args, audio=audio, batch_size=a.batch,
                          eval_batch_size=a.batch, num_loader_workers=0, num_eval_loader_workers=0, lr=a.lr)
ds = BaseDatasetConfig(formatter="coqui", dataset_name="ft", path=a.veri, meta_file_train="metadata.csv", language="tr")
config.datasets = [ds]
train_s, eval_s = load_tts_samples(config, eval_split=True, eval_split_max_size=20, eval_split_size=0.05)

model = GPTTrainer.init_from_config(config)
gpt = model.xtts.gpt
for p in model.parameters():
    p.requires_grad_(False)
egit = []
for n, p in gpt.named_parameters():
    if not (n.startswith("conditioning_encoder") or n.startswith("conditioning_perceiver")):
        p.requires_grad_(True); egit.append(p)
model.xtts.hifigan_decoder.to("cpu")  # eğitimde gerekmez
model.to(dev)
model.xtts.hifigan_decoder.to("cpu")
opt = Adafactor(egit, lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=1e-2)

durum_p = out / "durum.json"; ck_p = out / "son.pth"
durum = json.loads(durum_p.read_text()) if durum_p.exists() else {"epoch": 0, "adim": 0, "kayiplar": []}
if ck_p.exists():
    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    gpt.load_state_dict(ck["gpt"]); opt.load_state_dict(ck["opt"])
    print(f"devam: epoch {durum['epoch']}, adım {durum['adim']}", flush=True)
if durum["epoch"] >= a.epoch:
    print("zaten bitti"); sys.exit(0)

# Çöken örnek listesi: önceki çalıştırma bir eğitim batch'inde öldüyse (sürücü OOM), o batch'in dosyaları atlanır.
sb_p = out / "son_batch.txt"; atla_p = out / "atla.txt"
sc_p = out / "son_cokme.txt"
if sb_p.exists():  # aynı batch iki kez üst üste çöktürdüyse atla (tek çökme başka bir GPU kullanıcısından olabilir)
    b = sb_p.read_text()
    if sc_p.exists() and sc_p.read_text() == b:
        with open(atla_p, "a") as f: f.write(b)
        sc_p.unlink()
    else:
        sc_p.write_text(b)
    sb_p.unlink()
atla = set(atla_p.read_text().split()) if atla_p.exists() else set()
if atla:
    train_s = [x for x in train_s if x["audio_file"] not in atla]
    print(f"atlanan (çöken) örnek: {len(atla)}", flush=True)
train_dl = model.get_data_loader(config, is_eval=False, samples=train_s, verbose=False)
eval_dl = model.get_data_loader(config, is_eval=True, samples=eval_s, verbose=False)


def adim_kaybi(batch, train=True):
    batch = model.format_batch(batch)
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch = model.format_batch_on_device(batch)
    if not train:
        batch["cond_idxs"] = None
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        _, ld = model.train_step(batch, None)
    return ld


def degerlendir():
    gpt.eval(); t = []
    with torch.no_grad():
        for b in eval_dl:
            v = float(adim_kaybi(b, train=False)["loss_mel_ce"])
            if v == v and v != float("inf"): t.append(v)
    gpt.train()
    return sum(t) / max(len(t), 1)


print(f"eğitim {len(train_s)} / değerlendirme {len(eval_s)} örnek, eğitilen {sum(p.numel() for p in egit)/1e6:.0f}M parametre", flush=True)
if durum["epoch"] == 0 and durum["adim"] == 0 and not durum["kayiplar"]:
    durum["kayiplar"].append({"epoch": 0, "eval_mel": degerlendir()})
    print("başlangıç eval_mel", durum["kayiplar"][-1]["eval_mel"], flush=True)

model.eval(); gpt.train()
t0 = time.time(); top = 0.0; n = 0; atlanan = 0
opt.zero_grad(set_to_none=True)
for i, batch in enumerate(train_dl):
    sb_p.write_text("\n".join(batch["filenames"]) + "\n")
    ld = adim_kaybi(batch)
    if torch.isfinite(ld["loss"]):  # bozuk örnek (NaN/inf) ağırlıkları zehirlemesin
        (ld["loss"] / a.biriktir).backward()
        top += float(ld["loss_mel_ce"]); n += 1
    else:
        atlanan += 1
    if (i + 1) % a.biriktir == 0:
        gn = torch.nn.utils.clip_grad_norm_(egit, 1.0)
        if torch.isfinite(gn):
            opt.step(); durum["adim"] += 1
        else:
            atlanan += 1
        opt.zero_grad(set_to_none=True)
    if i % 20 == 0:
        mem = torch.xpu.max_memory_allocated() / 2**30 if dev.type == "xpu" else 0
        print(f"e{durum['epoch']+1} b{i}/{len(train_dl)} mel_ce {top/max(n,1):.3f} atlanan {atlanan} {(time.time()-t0)/(i+1):.2f}s/b bellek {mem:.1f}GB", flush=True)
    if a.adim and i + 1 >= a.adim:
        print(f"ölçüm bitti: {(time.time()-t0)/(i+1):.2f} s/batch, tepe bellek {mem:.1f} GB"); sys.exit(0)

sb_p.unlink(missing_ok=True); sc_p.unlink(missing_ok=True)
durum["epoch"] += 1
ev = degerlendir()
durum["kayiplar"].append({"epoch": durum["epoch"], "train_mel": top / max(n, 1), "eval_mel": ev,
                          "sure_dk": round((time.time() - t0) / 60, 1)})
print(f"epoch {durum['epoch']} bitti: train_mel {top/max(n,1):.3f} eval_mel {ev:.3f}", flush=True)
torch.save({"gpt": gpt.state_dict(), "opt": opt.state_dict()}, str(ck_p) + ".tmp"); os.replace(str(ck_p) + ".tmp", ck_p)
torch.save(gpt.state_dict(), out / f"gpt_e{durum['epoch']}.pth")
if ev <= min(k["eval_mel"] for k in durum["kayiplar"]):
    torch.save(gpt.state_dict(), out / "gpt_en_iyi.pth"); durum["en_iyi_epoch"] = durum["epoch"]
durum_p.write_text(json.dumps(durum, indent=1))
sys.exit(0 if durum["epoch"] >= a.epoch else 42)
