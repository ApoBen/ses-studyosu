# Ses Stüdyosu — Lokal Türkçe Ses Klonlama, TTS ve Ses Dönüştürme

**[Türkçe](#türkçe) | [English](#english)**

---

## Türkçe

**Ses Stüdyosu**, ham bir ses veya video kaydından otomatik olarak kişiye özel ses modeli oluşturan, ardından Türkçe metinleri doğru tonlama ve Whisper doğrulamasıyla o sesle okuyan (TTS) veya mevcut bir ses kaydını hedef sese dönüştüren (VC) tamamen yerel (lokal) ve açık kaynaklı bir stüdyodur.

> **Önemli Not:** Bu depo **hiçbir ses modeli, ses kaydı, transkript veya örnek ses içermez.** Uygulama yalnızca kod ve şablon yapılandırma dosyalarından oluşur; kullanıcı yalnızca kendi sesini veya açık izni olan sesleri kullanarak yerel ortamında model oluşturur.

### Ne Yapar?

1. **Yeni Ses Modeli Oluşturma:** Sürükleyip bıraktığınız ses/video kaydından müziği ayırır, hedef konuşmacıyı seçer, RVC modelini eğitir ve her 10 epoch'ta benzerliği ölçerek en uygun noktada otomatik durup ince ayar yapar.
2. **Türkçe Metin Seslendirme (TTS):** Türkçe sayı, kısaltma, tarih ve noktalama normalizasyonu yapar; XTTS-v2 ile üretilen sesi `faster-whisper` ile doğrulayarak hatalı/fazla heceleri kırpar veya yeniden üretir, ardından RVC modeliyle hedef sese dönüştürür.
3. **Sesten Sese Dönüştürme (VC):** Herhangi bir konuşma kaydını seçilen ses modelinin tınısına dönüştürür.
4. **Model ve Eğitim İzleme:** Profillerin hazırlık/eğitim durumlarını, veri seti özetlerini, epoch başına kosinüs benzerliği grafiğini (`outputs/eval_<profil>/sonuclar.csv`) ve örnek ses çıktılarını (`profiles/<profil>/ornek/`) canlı olarak gösterir.

### Özellikler

- **Ham Kayıttan Otomatik Veri Seti:** Demucs (`htdemucs`) ile 5 dakikalık parçalar hâlinde vokal ayırma, sessizlikten bölme, XTTS konuşmacı gömmeleriyle baskın konuşmacı kümeleme (`kume`) veya bilinen aralıktaki istenmeyen anlatıcı/yapay zekâ sesini eleme (`tek`).
- **Otomatik Değerlendirme ve Durma:** Eğitim sırasında her 10 epoch'ta 20 test cümlesi üzerinden hedef konuşmacı merkezine (`profiles/<profil>/ref/centroid.npy`) kosinüs benzerliği ölçülür; plato veya yeterli eşiğine ulaşıldığında eğitim otomatik durdurulur ve `index_rate × protect` ızgara aramasıyla en iyi ayarlar `profiles/<profil>/model/ayarlar.json` olarak dışa aktarılır.
- **Türkçe Tonlama ve Whisper Doğrulama:**
  - `core/text/`: Türkçe sayı/tarih/saat/birim/kısaltma normalizasyonu, `zeyrek` morfolojik analiziyle kelime vurgusu ve `stanza` bağımlılık ayrıştırmasıyla yüklem-öncesi odak ve soru tonu planlaması.
  - `core/tts/guvenli_sentez.py`: XTTS-v2 çıktısını `faster-whisper` (`large-v3-turbo`) ile hizalayıp kontrol eder; ortada kelime hatası varsa farklı tohumla yeniden üretir, baştaki/sondaki fazlalıkları yumuşak geçişle kırpar.
- **Intel Arc XPU Desteği:** RVC (Applio) ve XTTS-v2 iş yükleri Intel Arc GPU (`xpu`, `bf16`) üzerinde çalışacak şekilde uyarlanmıştır (`patches/applio_xpu.diff`). Eğitim GPU'da sürerken arayüzdeki Seslendir ve Dönüştür işlemleri GPU'ya dokunmadan güvenle CPU üzerinde çalışır.

### Ekran Görüntüsü

> _Ekran görüntüsü yeri (`docs/app/ekran_goruntusu.png` eklenecektir)_
>
> `![Ses Stüdyosu Arayüzü](docs/app/ekran_goruntusu.png)`

### Kurulum

Proje, bağımlılık çakışmalarını önlemek için dört ayrı Python sanal ortamı kullanır:

1. **Ana ortam (`.venv` — Arayüz, Türkçe NLP ve doğrulama mantığı):**
   ```bash
   python3.11 -m venv .venv
   ./.venv/bin/pip install -e ".[app,nlp,dev]"
   ```
2. **XTTS ortamı (`envs/xtts` — Python 3.11, `coqui-tts`, Intel XPU PyTorch):**
   ```bash
   python3.11 -m venv envs/xtts
   ./envs/xtts/bin/pip install torch==2.15.0+xpu torchaudio==2.11.1+xpu --index-url https://download.pytorch.org/whl/xpu
   ./envs/xtts/bin/pip install coqui-tts==0.28.0 demucs soundfile
   ```
3. **Applio / RVC ortamı (`envs/applio` — Python 3.11 + Intel Arc XPU yaması):**
   ```bash
   git clone https://github.com/IAHispano/Applio.git external/Applio
   git -C external/Applio apply ../../patches/applio_xpu.diff

   python3.11 -m venv envs/applio
   ./envs/applio/bin/pip install torch==2.15.0+xpu torchaudio==2.11.1+xpu --index-url https://download.pytorch.org/whl/xpu
   ./envs/applio/bin/pip install -r external/Applio/requirements-engine.txt
   ./envs/applio/bin/python external/Applio/core.py prerequisites
   ```
4. **Whisper ortamı (`envs/whisper` — Python 3.11, `faster-whisper` CPU int8):**
   ```bash
   python3.11 -m venv envs/whisper
   ./envs/whisper/bin/pip install faster-whisper==1.2.1 ctranslate2==4.9.0 soundfile
   ```

### Kullanım

Web arayüzünü başlatmak için:

```bash
./.venv/bin/python -m app
```

Tarayıcınızda `http://127.0.0.1:7860` adresini açın:
- **Yeni Ses:** Kaynak ses/video dosyasını sürükleyin, profil adını (`ornek_ses`, `konusmaci1`, `ahmet` gibi `a-z0-9_` biçiminde) ve konuşmacı modunu seçip **Başlat** düğmesine basın.
- **Seslendir:** Hazır modellerden birini seçin, Türkçe metni girin ve **Oluştur** düğmesine basın.
- **Dönüştür:** Bir ses kaydı yükleyip hedef ses modeline dönüştürün.
- **Modeller:** Profillerin durumlarını, benzerlik grafiklerini ve `profiles/<profil>/ornek/` altındaki örnek sesleri inceleyin.

#### Komut Satırı / Şablon Profil Örneği

Arayüzdeki **Yeni Ses** sekmesi profil dosyalarını otomatik üretir. Elle yapılandırmak isterseniz örnek şablon dosyalarını inceleyebilirsiniz:
- `scripts/profil/ornek.env` — örnek profil yapılandırma şablonu (`AD=ornek_ses`, `KAYNAK`, `CUMLELER`, `MIN_EPOCH`, `TOPLAM_EPOCH`, `HAZIRLA_MOD`, `ISTENMEYEN`).
- `scripts/profil/ornek_cumleler.txt` — benzerlik değerlendirmesinde kullanılan 20 adet standart Türkçe test cümlesi.

Testleri çalıştırmak için:

```bash
./.venv/bin/python -m pytest -q
```

#### Tonlama ince ayarı (isteğe bağlı, XTTS GPT)
Konuşma ritmini ve vurguyu hedef sese yaklaştırmak için XTTS'in GPT kısmı kısa bir ince ayarla eğitilebilir (`scripts/xtts_ft/`):
1. `transkript.py <wav_klasörü> <çıktı> <ad>`: Whisper ile coqui biçiminde `metadata.csv`; varsa altyazıyla çapraz kontrol: `vtt_kontrol.py`.
2. `egit.sh <veri> <çıktı> <epoch>`: Intel Arc XPU döngüsü (batch 1 × 8 birikim, bf16, Adafactor; ~6 GB). XTTS ağırlıkları otomatik bulunur ya da `XTTS_DIR` ile verilir; `dvae.pth` XTTS-v2 deposundan indirilip `models/xtts_ft/` altına konur (ya da `XTTS_DVAE`).
3. Az veriyle (10–30 dk) ezber erken başlar: genelde 2–4 epoch yeterlidir. Seçimi aynı cümleyi farklı epoch'larla dinleyerek yapın.

### Etik ve Hukuki Uyarı

- **Bu depo hiçbir ses modeli, ses kaydı veya örnek ses içermez.**
- **Yalnızca kendi sesinizi veya sahibinden açık, belgelenebilir izin aldığınız sesleri kullanın.**
- Eğitilmiş ses modellerini ve üretilen ses kayıtlarını kullanma, saklama ve paylaşma sorumluluğu tamamen kullanıcıya aittir.
- **Ses taklidi (deepfake) yoluyla dolandırıcılık, kimliğe bürünme, dezenformasyon veya kişilik haklarını ihlal eden her türlü kullanım kesinlikle yasaktır** ve ilgili yasalar kapsamında suç teşkil eder.

### Güvenlik

- **Resmî Kaynak:** Bu yazılımın tek resmî kaynağı projenin kendi açık kaynak deposudur. Üçüncü taraf sitelerden veya belirsiz paketlerden indirilen kopyalara güvenmeyin.
- **SHA256 Doğrulaması:** Sürümlerle (releases) birlikte `SHA256` sağlama toplamları yayınlanır; indirdiğiniz arşiv veya dosyaları çalıştırmadan önce `sha256sum` ile doğrulayın.
- **GPU ve Kaynak Kullanımı:** Uygulama GPU'yu **yalnızca aktif eğitim veya sentez sırasında** kullanır; arka planda gizli bir işlem çalıştırmaz. Uygulama boştayken Intel GPU'larda `intel_gpu_top`, NVIDIA GPU'larda `nvidia-smi` komutuyla GPU kullanımının boş (%0) olduğunu doğrudan kontrol edebilirsiniz.
- **Denetlenebilir Açık Kod:** Tüm kaynak kodu açıktır ve incelenebilir; şüpheli bir değişiklik veya güvenlik açığı görürseniz lütfen depo üzerinden bildirin.

### Lisans

- **Proje Kodu:** [GNU General Public License v3.0 veya sonrası (`GPL-3.0-or-later`)](LICENSE) — değiştirilmiş sürümler yapılan değişiklikleri açıkça belirtmek ve kaynak kodunu aynı lisans (`GPL-3.0-or-later`) altında sunmak zorundadır.
- **XTTS-v2 Model Ağırlıkları:** [Coqui Public Model License 1.0.0 (CPML)](https://coqui.ai/cpml) — ticari olmayan (non-commercial) kullanım içindir, kurulumda ayrı olarak indirilir.
- **coqui-tts Kütüphanesi:** Mozilla Public License 2.0 (`MPL-2.0`).
- **Applio (RVC):** MIT Lisansı (`MIT`).
- **Whisper / faster-whisper:** MIT Lisansı (`MIT`).
- **Gradio:** Apache License 2.0 (`Apache-2.0`).

---

## English

**Voice Studio (Ses Stüdyosu)** is a fully local, open-source Turkish voice cloning, text-to-speech (TTS), and voice conversion (VC) studio. Drop an audio or video recording to automatically build a clean speaker dataset and train an RVC voice model, then synthesize Turkish text with natural prosody and Whisper verification or convert existing recordings into the target voice.

> **Important Notice:** This repository **contains no voice models, audio recordings, transcripts, or sample voices.** It provides only source code and neutral configuration templates; users train models locally using their own voice or recordings for which they hold explicit permission.

### What It Does

1. **Automated Voice Profile Training:** Separates background music from raw recordings, extracts the target speaker, trains an RVC model, evaluates speaker similarity every 10 epochs, stops automatically at peak/plateau, and fine-tunes inference hyperparameters.
2. **Verified Turkish Text-to-Speech (TTS):** Normalizes Turkish text (numbers, dates, abbreviations, units), synthesizes speech with XTTS-v2, verifies word alignment with `faster-whisper` (re-generating on word errors and trimming trailing hallucinations), and converts timbre with RVC.
3. **Voice Conversion (VC):** Converts any input speech recording to the selected target voice on CPU without interrupting ongoing GPU training.
4. **Model & Training Dashboard:** Live monitoring of preparation/training stages, dataset statistics, per-epoch cosine similarity plots (`outputs/eval_<profile>/sonuclar.csv`), and sample audio playback (`profiles/<profile>/ornek/`).

### Key Features

- **Automatic Dataset Preparation from Raw Audio:** Chunked vocal separation via Demucs (`htdemucs`), silence-based segmentation, speaker embedding extraction, and either dominant-speaker clustering (`kume`) or unwanted narrator filtering (`tek`).
- **Automatic Stopping & Hyperparameter Tuning:** Evaluates 20 test sentences every 10 epochs against the target speaker centroid (`profiles/<profile>/ref/centroid.npy`), stops automatically upon plateau/target similarity, and runs a grid search over `index_rate × protect` to export `profiles/<profile>/model/ayarlar.json`.
- **Turkish Prosody & Whisper Verification:** Morphological stress detection (`zeyrek`), dependency-based focus/predicate planning (`stanza`), and closed-loop Whisper verification (`core/tts/guvenli_sentez.py`).
- **Intel Arc XPU Support:** Includes an XPU (`bf16`) training patch for Applio (`patches/applio_xpu.diff`) with automatic host-memory leak recovery.

### Screenshot

> _Screenshot placeholder (`docs/app/ekran_goruntusu.png`)_
>
> `![Voice Studio UI](docs/app/ekran_goruntusu.png)`

### Installation

The project uses four isolated Python 3.11 virtual setups:

1. **`.venv` (Gradio UI, Turkish NLP & verification logic):**
   ```bash
   python3.11 -m venv .venv
   ./.venv/bin/pip install -e ".[app,nlp,dev]"
   ```
2. **`envs/xtts` (XTTS-v2 & Demucs with Intel XPU PyTorch):**
   ```bash
   python3.11 -m venv envs/xtts
   ./envs/xtts/bin/pip install torch==2.15.0+xpu torchaudio==2.11.1+xpu --index-url https://download.pytorch.org/whl/xpu
   ./envs/xtts/bin/pip install coqui-tts==0.28.0 demucs soundfile
   ```
3. **`envs/applio` (Applio RVC + Intel Arc XPU patch):**
   ```bash
   git clone https://github.com/IAHispano/Applio.git external/Applio
   git -C external/Applio apply ../../patches/applio_xpu.diff

   python3.11 -m venv envs/applio
   ./envs/applio/bin/pip install torch==2.15.0+xpu torchaudio==2.11.1+xpu --index-url https://download.pytorch.org/whl/xpu
   ./envs/applio/bin/pip install -r external/Applio/requirements-engine.txt
   ./envs/applio/bin/python external/Applio/core.py prerequisites
   ```
4. **`envs/whisper` (`faster-whisper` CPU int8 verification worker):**
   ```bash
   python3.11 -m venv envs/whisper
   ./envs/whisper/bin/pip install faster-whisper==1.2.1 ctranslate2==4.9.0 soundfile
   ```

### Usage

Launch the Gradio web studio (defaults to `127.0.0.1:7860`):

```bash
./.venv/bin/python -m app
```

- **Yeni Ses (New Voice):** Drag and drop an audio/video file, enter a neutral profile identifier (e.g., `ornek_ses`, `konusmaci1`, `ahmet`), pick the speaker mode, and click **Başlat**.
- **Profil Şablonu (Profile Template):** The UI generates profile files automatically, or you can copy the example template at `scripts/profil/ornek.env` along with the 20 evaluation sentences in `scripts/profil/ornek_cumleler.txt`.

Run the test suite:

```bash
./.venv/bin/python -m pytest -q
```

#### Prosody fine-tuning (optional, XTTS GPT)
To bring rhythm and stress closer to the target voice, XTTS's GPT part can be briefly fine-tuned (`scripts/xtts_ft/`):
1. `transkript.py <wav_dir> <out> <name>`: Whisper transcripts as coqui `metadata.csv`; optional subtitle cross-check with `vtt_kontrol.py`.
2. `egit.sh <data> <out> <epochs>`: Intel Arc XPU loop (batch 1 × 8 accumulation, bf16, Adafactor; ~6 GB). XTTS weights are found automatically or set via `XTTS_DIR`; download `dvae.pth` from the XTTS-v2 repo into `models/xtts_ft/` (or set `XTTS_DVAE`).
3. With little data (10–30 min) overfitting starts early: 2–4 epochs are usually enough. Pick by listening to the same sentence across epochs.

### Ethics & Legal Notice

- **This repository does not include any voice models, audio recordings, or sample voices.**
- **Only use your own voice or voices for which you have explicit, verifiable consent from the speaker.**
- Users bear full legal and ethical responsibility for any trained voice models and generated audio.
- **Using voice cloning for fraud, impersonation, non-consensual content, or deception is strictly prohibited.**

### Security

- **Official Source Only:** Only download this project from its official repository.
- **SHA256 Checksums:** Release artifacts are accompanied by `SHA256` checksums; always verify downloaded files before use (`sha256sum`).
- **GPU Activity Verification:** The studio uses the GPU **only** during active training or synthesis. When idle, you can verify that the GPU is completely idle (`0%`) using `intel_gpu_top` (Intel) or `nvidia-smi` (NVIDIA).
- **Auditable Source:** The entire codebase is open for inspection; please report any suspicious changes or security issues.

### License

- **Repository Code:** [GNU General Public License v3.0 or later (`GPL-3.0-or-later`)](LICENSE) — modified versions must state changes and distribute source code under the same license.
- **XTTS-v2 Model Weights:** [Coqui Public Model License 1.0.0 (CPML)](https://coqui.ai/cpml) (non-commercial, downloaded separately).
- **coqui-tts Library:** Mozilla Public License 2.0 (`MPL-2.0`).
- **Applio (RVC):** MIT License (`MIT`).
- **Whisper / faster-whisper:** MIT License (`MIT`).
- **Gradio:** Apache License 2.0 (`Apache-2.0`).
