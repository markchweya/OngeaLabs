# Ongea

**Type something. Choose a voice. Hear it spoken.**

Ongea (Kiswahili for *speak*) is a text-to-speech studio for Kiswahili,
English, German, French and Polish. The current studio runs entirely in your web
browser: there is nothing to install or sign up for, and nothing you type is
sent to a server.

**[Try the studio](https://markchweya.github.io/OngeaLabs/)**

[![CI](https://github.com/markchweya/OngeaLabs/actions/workflows/ci.yml/badge.svg)](https://github.com/markchweya/OngeaLabs/actions/workflows/ci.yml)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)

---

## Using it

1. Open the studio and type or paste your text. The voice follows the
   language you write in: Kiswahili text goes to a Kiswahili voice, and so
   on.
2. To choose a different voice, open the voice menu at the top. Each
   language lists its own voices, and the one you pick stays that
   language's voice.
3. Press the waveform button. The voice starts speaking after the first
   sentence, while the rest are still being made.
4. Replay the take, scrub through its waveform, or download it as WAV or
   M4A.

The first time you use a voice, its model is downloaded once and kept by
your browser: about 92 MB for all the English voices together, and 20 to
77 MB for each of the others. After that it speaks straight away, even
offline on the full site.

### Voices

Every language has voices of its own, trained on people speaking it, so
Kiswahili sounds like Kiswahili and not like English read aloud.

| Language | Voices | Model |
|---|---|---|
| Kiswahili | Juma, Baraka | [Meta MMS](https://huggingface.co/facebook/mms-tts-swh), [Piper](https://huggingface.co/rhasspy/piper-voices) |
| English (US and UK) | 16, from soft to deep, plus blends you mix yourself | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| Deutsch | Thorsten, Kerstin, Ramona, Eva, Karlsson, Lukas | [Piper](https://huggingface.co/rhasspy/piper-voices), [Meta MMS](https://huggingface.co/facebook/mms-tts-deu) |
| Français | Élise, Jessica, Pierre, Gilles, Antoine | [Piper](https://huggingface.co/rhasspy/piper-voices), [Meta MMS](https://huggingface.co/facebook/mms-tts-fra) |
| Polski | Gosia, Darek, Mateusz | [Piper](https://huggingface.co/rhasspy/piper-voices) |

The tone controls set pace, pitch, warmth and clarity for every voice.

## Where things live

Ongea has two homes, and it helps to know which is which.

| | What it is | Where |
|---|---|---|
| **The live studio** | The in-browser studio described above | [ongealabs.olkeri.space](https://ongealabs.olkeri.space), built in the olkeri.space repository |
| **The preview** | A static copy of the live studio, without the shared voice library | [markchweya.github.io/OngeaLabs](https://markchweya.github.io/OngeaLabs/), served from this repository's `gh-pages` branch |
| **This repository's code** | The original OngeaLabs studio: a React front end and an optional Python voice API | The `main` branch, below |

The rest of this README is about the code on `main`.

## The original studio (this repository)

A React and TypeScript front end built with Vite, plus a small FastAPI
service that runs the Meta MMS models with PyTorch. The front end works on
its own and shows the voice list. Audio needs the voice API.

### Run the front end

Needs Node.js 22 (see `.nvmrc`).

```bash
npm ci
npm run dev
```

Then open http://localhost:5173.

### Run the voice API (optional)

Needs Python 3.10 or later. The first run downloads the models.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
npm run api                      # http://127.0.0.1:8001
```

The front end calls `GET /api/voices` for the speaker list and
`POST /api/synthesize` to make audio. A request carries `language`, `voice`,
`pace`, `pitch`, `warmth` and `clarity`, and returns `ongealabs.wav`.

### Configuration

| Variable | Used by | What it does |
|---|---|---|
| `VITE_API_BASE_URL` | Front end | Where the voice API lives. Defaults to `http://127.0.0.1:8001` in development. |
| `ONGEA_TTS_VOICES_JSON` | API | The voice list as JSON: a list, or `{ "voices": [...] }`. |
| `ONGEA_TTS_DB` or `VOICE_DATABASE_PATH` | API | An SQLite database with a `voices`, `tts_voices`, `voice_profiles` or `speakers` table. |

Without either API setting, voices come from `backend/voices.json`.

### Project layout

```
src/                   React front end (App.tsx is the studio)
public/                Icons
backend/main.py        FastAPI app: /api/voices, /api/synthesize
backend/tts_engine.py  Meta MMS loading, per-voice settings, WAV output
backend/voices.json    Voice list for local development
.github/workflows/     CI, Pages, releases, monitoring (below)
```

### Checks

```bash
npm run lint       # ESLint
npm run build      # TypeScript check, then production build
python -m py_compile backend/*.py
```

## Automation

| Workflow | When it runs | What it does |
|---|---|---|
| [CI](.github/workflows/ci.yml) | Every pull request and push to `main` | Lints, type-checks and builds the front end, and compiles the API. |
| [Pages](.github/workflows/pages.yml) | By hand, or when the workflow changes | Deploys the `gh-pages` branch through GitHub Actions. It skips the deploy until Settings > Pages > Source is set to "GitHub Actions" (until then GitHub serves that branch directly); after switching, allow `main` under Settings > Environments > github-pages. |
| [Release](.github/workflows/release.yml) | When a release is published | Builds the front end, attaches it to the release, and attaches signed [SLSA level 3](https://slsa.dev) provenance. |
| [Datadog Synthetics](.github/workflows/datadog-synthetics.yml) | Each push to `main`, and daily | Runs Datadog browser tests against the live studio. Off until the `DD_API_KEY` and `DD_APP_KEY` secrets are added; the file explains the setup. |

### Checking a release download

Every release asset comes with provenance. With
[slsa-verifier](https://github.com/slsa-framework/slsa-verifier):

```bash
slsa-verifier verify-artifact ongea-web-v1.0.0.tar.gz \
  --provenance-path ongea-web-v1.0.0.tar.gz.intoto.jsonl \
  --source-uri github.com/markchweya/OngeaLabs \
  --source-tag v1.0.0
```

## Credits

Ongea's own work is the studio and its speech engine: the in-browser
worker that streams speech sentence by sentence, the Piper runner,
language detection, the voice shaper, and the voice library that breeds
new English voices from listeners' ratings. The voices themselves are
open models trained by others on the datasets below. Thank you to
everyone who recorded, collected and released them.

### Models

| Voices | Model | By | Licence |
|---|---|---|---|
| English (16) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | hexgrad | Apache 2.0 |
| Juma, Lukas, Antoine | [MMS-TTS](https://huggingface.co/facebook/mms-tts) | Meta AI | CC BY-NC 4.0 (non-commercial) |
| All other voices | [Piper voices](https://huggingface.co/rhasspy/piper-voices) | Rhasspy / Open Home Foundation | MIT, with each voice's data licence below |

### Datasets

| Voice | Dataset | By | Licence |
|---|---|---|---|
| Baraka | [A Kiswahili Dataset for Development of Text-To-Speech System](https://data.mendeley.com/datasets/vbvj6j6pm9/1) | Mendeley Data, 2021 | CC BY 4.0 |
| Thorsten | [Thorsten-Voice](https://github.com/thorstenMueller/Thorsten-Voice) | Thorsten Müller | CC0 |
| Kerstin | [dataset-voice-kerstin](https://github.com/rhasspy/dataset-voice-kerstin) | Rhasspy | CC0 |
| Ramona, Eva, Karlsson | [The M-AILABS Speech Dataset](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/) | M-AILABS | BSD-style |
| Élise | [SIWIS French Speech Synthesis Database](https://datashare.is.ed.ac.uk/handle/10283/2353) | Idiap / University of Edinburgh | CC BY 4.0 |
| Jessica, Pierre | [UPMC voice data](https://github.com/marytts/upmc-pierre-data) | MaryTTS | CC BY-SA 4.0 |
| Gilles | [CSS10 French](https://www.kaggle.com/datasets/bryanpark/french-single-speaker-speech-dataset) | Kyubyong Park, Tommy Mulc | CC0 |
| Gosia, Darek | [OHF voice datasets](https://github.com/OHF-Voice/voice-datasets) | Open Home Foundation | CC0 |
| Mateusz | [The MC Speech Dataset](https://www.kaggle.com/datasets/czyzi0/the-mc-speech-dataset) | Mateusz Czyżnikiewicz | CC0 |

Pronunciation for the Piper voices comes from
[espeak-ng](https://github.com/espeak-ng/espeak-ng) (GPL 3). Meta MMS
credit: Pratap et al., *Scaling Speech Technology to 1,000+ Languages*,
Meta AI, 2023.

## Licences

- **The code** in this repository is under the [Apache License 2.0](LICENSE).
- **The voices** keep their own licences, listed under Credits. The MMS
  voices (Juma, Lukas, Antoine) may not be used commercially; Jessica and
  Pierre are share-alike.

## Contributing

Issues and pull requests are welcome. Please keep each pull request to one
change, and run the checks above before opening it. CI runs them again on
every pull request.

Made by OngeaLabs.
