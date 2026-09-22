<div align="center">

<img src="app/static/logo.png" alt="NinjaSubs Logo" width="220" />

**Smart subtitle aggregator for Stremio with an advanced Arabic text & BiDi processing engine.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#-english) · [العربية](#-العربية)

</div>

---

## 🇬🇧 English

A lightweight, self-hosted [Stremio](https://stremio.com) / [Nuvio](https://nuvio.tv/) subtitle addon built with **Python 3.11+**, **FastAPI**, and **Uvicorn**. It aggregates subtitles from **five providers**, unpacks archives safely in memory, and streams native subtitles with a dedicated **Arabic Language Engine** for BiDi, punctuation, diacritics and numeral normalization.

### Highlights

- **Multi-provider aggregation** — SubDL, SubSource, OpenSubtitles, YIFYSubtitles and SubtitleCat queried in parallel (`asyncio.gather`) with per-provider timeouts and resilient fallback.
- **Arabic Language Engine** — context-aware RTL alignment, reversal repair, optional Tashkeel (diacritics) removal, Eastern-Arabic numerals and in-dialogue HI cleanup.
- **Stateless, per-user configuration** — every preference and API key is serialized into a URL-safe Base64 token embedded in the manifest URL; nothing is stored server-side.
- **Informative match badges** — compose the subtitle label from match score, provider, release filename and uploader.
- **Low footprint** — in-memory ZIP extraction (Zip Slip protected), LRU disk cache, and strict memory limits.

### Features & Preferences

Every setting below is an independent, opt-in preference available in the [configuration UI](#configuration-ui). Choices are serialized into the stateless manifest token and included in the cache key, so toggling one never serves stale subtitles.

#### Arabic Language Engine
* **Arabic RTL Alignment Fix**: Fixes inverted punctuation, brackets, and quotes in Arabic subtitles to prevent misplaced periods and leading dashes (e.g. <bdi dir="rtl">`مرحبا. -`</bdi> &rarr; <bdi dir="rtl">`- مرحبا.`</bdi>).
* **Strip Arabic Diacritics (Tashkeel)**: Removes heavy Harakat while selectively preserving Shadda, Tanween, and feminine Kasra (e.g. <bdi dir="rtl">`أَنْتِ الَّتِي عَلَّمْتِ`</bdi> &rarr; <bdi dir="rtl">`أنتِ التي علّمتِ`</bdi>).
* **Normalize Arabic Commas**: Converts Latin commas in Arabic dialogue into proper Arabic commas (e.g. <bdi dir="rtl">`نعم , لا`</bdi> &rarr; <bdi dir="rtl">`نعم، لا`</bdi>).
* **Convert Numbers to Eastern Arabic**: Converts Western digits to Eastern Arabic numerals while preserving timestamps, tags, and alphanumeric terms (e.g. <bdi dir="rtl">`قبل 3 أيام`</bdi> &rarr; <bdi dir="rtl">`قبل ٣ أيام`</bdi>).

#### Dialogue & Clean-up

- **Exclude HI (Hearing Impaired)**: Filters out tracks containing sound effects and audio descriptions.
- **Strip In-dialogue HI Labels**: Cleans sound effects, audio cues, and speaker names while keeping dialogue intact (e.g. `[LAUGHS] JOHN: Hello there!` → `Hello there!`).
- **Remove Ads**: Strips promotional links, websites, and social handles, then re-indexes SRT cues (e.g. Watch free at [www.example.com](https://www.example.com) — @promo_channel → (cue removed)).
- **Keep Translator Credits**: Preserves translator attribution lines while cleanly dropping attached spam and links.

#### Text Formatting & Timing

- **Clean Tags**: Balances and closes unclosed formatting tags (`<i>`/`<b>`) and removes unsupported wrappers (e.g. `<i>- Hello! <custom>world</custom>` → `<i>- Hello! world</i>`).
- **Strip Text Colors**: Removes HTML font color tags and ASS color codes (`{\c&H...&}`) to enforce the player's native styling (e.g. `<font color="#ff0000">Hello</font>` → `Hello`).
- **Normalize Spacing**: Collapses duplicate spaces and removes spaces before punctuation (e.g. `word  ,  next` → `word, next`).
- **Clean Symbols & Breaks**: Converts stray double hyphens into ellipses and removes stray `<br>` tags (e.g. `-- Wait <br>` → `... Wait`).
- **Fix Display Timing**: Clamps micro-overlaps (< 500ms) between consecutive cues to stop player flickering (e.g. `00:00:01,000 --> 00:00:03,000` & `00:00:02,800 --> 00:00:05,000` → clamped to `00:00:02,800`).

### Subtitle Providers

| Provider | Access | Default | Notes |
| :--- | :--- | :---: | :--- |
| **SubDL** | API key | ✅ on | Primary source, broad release coverage |
| **SubSource** | API key | ✅ on | Strong community uploads |
| **OpenSubtitles** | API key | ⛔ off | Optional; quota-aware |
| **YIFYSubtitles** | Keyless | ⛔ off | Movies only (IMDb lookup) |
| **SubtitleCat** | Keyless | ⛔ off | Machine-translated `.srt` files |

### Arabic Language Engine

All Arabic processing is applied at **serve time** (per user) and is fully toggleable:

- **RTL punctuation & BiDi alignment** — appends a Right-to-Left Mark (RLM, `U+200F`) after trailing neutral punctuation so sentence-ending marks stay on the left; repairs legacy "reverse RTL" hacks and mirrored brackets/quotes, and fixes pre-reversed dialogue dashes (`"نص -" → "- نص."`).
- **Clean syntax & formatting** — independent toggles for tag repair, spacing, symbols (`-- → ...`), Latin→Arabic commas, unsafe/unclosed HTML `tags`, and display-timing overlap clamping (< 500 ms).
- **Strip Arabic diacritics (Tashkeel)** — removes Harakat while keeping **Shadda**, **all Tanween types**, and the **feminine Kasra** (`أنتِ`, `لكِ`, `علّمتِ`).
- **Convert to Eastern Arabic numerals** — `3 أيام` → `٣ أيام`, while protecting tags, timestamps and Latin/alphanumeric tokens (`AK-47`, `MP4`, `Windows 11`).
- **Strip in-dialogue HI labels** — removes `[MUSIC]`, `(SIGHS)`, `JOHN:` artifacts while keeping dialogue.
- **Encoding safety** — transcodes CP1256 / ISO-8859-6 to clean UTF-8.

### Quick Start (Docker)

```bash
cp .env.example .env
# set BASE_URL to your LAN address, e.g. http://192.168.1.50:7000

docker compose up -d --build
docker compose logs -f
```

Open `http://<HOST_IP>:7000/configure`, enter your keys, then **Install to Stremio** or **Copy Link**.

### Configuration UI

The web UI is a 3-step wizard and a stateless token:

1. **Providers** — pick sources and paste API keys.
2. **Preferences** — language engine, cleaning toggles, badge preview.
3. **Install** — addon host, Stremio install actions and the generated manifest URL.

### API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/configure` | `GET` | Configuration wizard |
| `/{config}/configure` | `GET` | Prefilled configuration wizard |
| `/manifest.json` | `GET` | Default Stremio v3 manifest |
| `/{config}/manifest.json` | `GET` | User-configured manifest |
| `/subtitles/{type}/{id}.json` | `GET` | Subtitle search (server keys) |
| `/{config}/subtitles/{type}/{id}.json` | `GET` | Subtitle search (user keys) |
| `/sub/{sub_id}.srt` | `GET` | Serve UTF-8 subtitle |
| `/health` | `GET` | Health + cache statistics |

### Local Development

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7000 --reload

pytest -q                          # tests
ruff check .                       # lint
mypy app                           # types
```

---

## 🇸🇦 العربية

إضافة تُرجمة خفيفة وذاتية الاستضافة لـ [Stremio](https://stremio.com) و[Nuvio](https://nuvio.tv/)، مبنية بـ **Python 3.11+** و**FastAPI** و**Uvicorn**. تجمع الترجمة من **خمسة مصادر**، وتفكّ ضغط الملفات داخل الذاكرة بأمان، وتبثّ الترجمة مع **محرّك خاص باللغة العربية** لمعالجة الاتجاه (BiDi) وعلامات الترقيم والتشكيل والأرقام.

### أبرز المزايا

- **تجميع من عدة مصادر** — SubDL وSubSource وOpenSubtitles وYIFYSubtitles وSubtitleCat بالتوازي (`asyncio.gather`) مع مهلة لكل مصدر واحتياطي عند الفشل.
- **محرّك اللغة العربية** — ضبط اتجاه النص من اليمين لليسار، وإصلاح النص المعكوس، وإزالة التشكيل (اختياري)، وتحويل الأرقام إلى الأرقام العربية المشرقية، وتنظيف وسوم الصوت الوصفية.
- **إعدادات لكل مستخدم بدون حفظ** — تُرمّز كل الإعدادات ومفاتيح الـ API داخل رابط الـ manifest (Base64) ولا يُخزَّن شيء على الخادم.
- **شارات مطابقة غنية** — تتكوّن من نسبة المطابقة والمصدر واسم الإصدار واسم الرافع.
- **استهلاك منخفض** — فكّ ضغط داخل الذاكرة (مع حماية Zip Slip) وذاكرة تخزين LRU مؤقتة.

### مصادر الترجمة

| المصدر | الوصول | الافتراضي | ملاحظات |
| :--- | :--- | :---: | :--- |
| **SubDL** | مفتاح API | ✅ مُفعّل | المصدر الأساسي وواسع التغطية |
| **SubSource** | مفتاح API | ✅ مُفعّل | ترجمات مجتمعية قوية |
| **OpenSubtitles** | مفتاح API | ⛔ معطّل | اختياري مع مراعاة الحصة |
| **YIFYSubtitles** | بدون مفتاح | ⛔ معطّل | للأفلام فقط |
| **SubtitleCat** | بدون مفتاح | ⛔ معطّل | ملفات مترجمة آليًا |

### محرّك اللغة العربية

- **ضبط الاتجاه والترقيم** — إضافة علامة RLM (`U+200F`) بعد علامات الترقيم النهائية، وإصلاح العلامات المعكوسة والأقواس والاقتباسات، وتصحيح الشرطات المقلوبة (`"نص -" ← "- نص."`).
- **تنظيف الصياغة** — خيارات مستقلة للوسوم والمسافات والرموز (`-- ← ...`) والفواصل اللاتينية، وضبط تداخل التوقيت (< 500ms).
- **إزالة التشكيل (Tashkeel)** — مع الحفاظ على **الشدة** و**جميع أنواع التنوين** و**كسرة المؤنث** (`أنتِ`، `لكِ`، `علّمتِ`).
- **تحويل الأرقام** — `3 أيام` ← `٣ أيام` مع حماية الوسوم والتوقيت والرموز اللاتينية (`AK-47`، `MP4`، `Windows 11`).
- **تنظيف وسوم الصوت الوصفية** — إزالة `[MUSIC]` و`(SIGHS)` و`JOHN:` مع الإبقاء على الحوار.
- **أمان الترميز** — تحويل CP1256 / ISO-8859-6 إلى UTF-8 نظيف.

### التشغيل السريع (Docker)

```bash
cp .env.example .env
# اضبط BASE_URL على عنوان الشبكة المحلية، مثال: http://192.168.1.50:7000

docker compose up -d --build
```

ثم افتح `http://<HOST_IP>:7000/configure`، أدخل المفاتيح، واضغط **Install to Stremio** أو **Copy Link**.

### الترخيص

هذا المشروع مُرخّص تحت **رخصة MIT** — انظر ملف [LICENSE](LICENSE).

> ملاحظة: خط الواجهة «Serif Black» (باستخدام `app/static/fonts/SerifBlackItalic.ttf`) مُرخّص من مصدره للاستخدام الشخصي فقط.

---

<div align="center">

**Free & Open Source for the community · v1.0.0**

[github.com/donsaud/NinjaSubs](https://github.com/donsaud/NinjaSubs)

</div>
