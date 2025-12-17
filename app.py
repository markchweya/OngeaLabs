# app.py
"""
ONGEA LABS — Streamlit-Cloud friendly ChatGPT-like TTS Studio (MMS/VITS + optional Coqui)

What this version fixes for Streamlit Cloud:
- Avoids heavy imports at module import-time (loads torch/transformers only when needed)
- Uses cache_resource safely (model cached per (model_id, lang_code))
- Writes outputs into ./outputs (works on Streamlit Cloud ephemeral FS)
- Has a visible Build Stamp (so you can confirm redeploys)
- Sidebar truly disappears (no reserved space) when you toggle it inside ⚙️ Settings
- Training code is kept CLI-only (won’t run inside Streamlit Cloud)

Run locally:
  streamlit run app.py

Optional CLI training (local only; not for Streamlit Cloud):
  python app.py --train --lang swh
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import re
import json
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

import numpy as np


# =========================
# PROJECT SETTINGS
# =========================

HF_DATASET_NAME = "michsethowusu/swahili-words-speech-text-parallel"
HF_DATASET_CONFIG = None
TRAIN_SPLIT = "train"
EVAL_SPLIT = None

DEFAULT_AUDIO_COL = "audio"
DEFAULT_TEXT_COL = "text"

LANGUAGES: Dict[str, str] = {
    "Swahili (Kiswahili) — KE/TZ/UG": "swh",
    "Amharic (አማርኛ) — ET": "amh",
    "Somali (Soomaaliga) — SO/ET/DJ": "som",
    "Yoruba (Yorùbá) — NG/Benin": "yor",
    "Shona — ZW": "sna",
    "Xhosa (isiXhosa) — ZA": "xho",
    "Afrikaans — ZA/NA": "afr",
    "Lingala — CD/CG": "lin",
    "Kongo (Kikongo) — CD/CG/AO": "kon",
    "Luo (Dholuo) — KE": "luo",
    "Gikuyu (Agĩkũyũ) — KE": "kik",
    "Ameru (Kĩmĩrũ / Meru) — KE": "mer",
    "Kamba (Kikamba) — KE": "kam",
    "Ekegusii (Kisii) — KE": "guz",
    "Luhya (Luluhya) — KE": "luy",
    "Kalenjin — KE": "kln",
    "Maasai (Maa) — KE/TZ": "mas",
    "Taita (Kidawida / Dawida) — KE": "dav",
}

PROJECT_NAME = "ongea-labs-streamlit"
OUTPUT_DIR = Path("./outputs") / PROJECT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000

# text normalization
LOWERCASE = True
STRIP_PUNCT = False


# =========================
# VOICE LIBRARY
# =========================

VOICE_LIBRARY_BY_LANG: Dict[str, Dict[str, str]] = {
    "swh": {
        "Ongea Labs Swahili Male / Neutral (Meta Base)": "facebook/mms-tts-swh",
        "Ongea Labs Swahili Female (Mozilla Lady)": "Benjamin-png/swahili-mms-tts-mozilla-lady-voice-finetuned",
        "Ongea Labs Swahili Studio (Fine-tuned)": "Benjamin-png/swahili-mms-tts-finetuned",
        "Ongea Labs Swahili Narrator (OpenBible)": "bookbot/vits-base-sw-KE-OpenBible",
        "Ongea Labs Swahili SALAMA (Prosody-rich)": "EYEDOL/SALAMA_TTS",
    },
    "amh": {"Ongea Labs Amharic (Meta MMS Base)": "facebook/mms-tts-amh"},
    "som": {"Ongea Labs Somali (Meta MMS Base)": "facebook/mms-tts-som"},
    "yor": {"Ongea Labs Yoruba (Meta MMS Base)": "facebook/mms-tts-yor"},
    "sna": {"Ongea Labs Shona (Meta MMS Base)": "facebook/mms-tts-sna"},
    "xho": {"Ongea Labs Xhosa (Meta MMS Base)": "facebook/mms-tts-xho"},
    "afr": {"Ongea Labs Afrikaans (Meta MMS Base)": "facebook/mms-tts-afr"},
    "lin": {"Ongea Labs Lingala (Meta MMS Base)": "facebook/mms-tts-lin"},
    "kon": {"Ongea Labs Kongo (Meta MMS Base)": "facebook/mms-tts-kon"},
    "luo": {
        "Ongea Labs Luo (CLEAR YourTTS, Coqui/HF)": "coqui:CLEAR-Global/YourTTS-Luo",
        "Ongea Labs Luo (CLEAR XTTS, Coqui/HF)": "coqui:CLEAR-Global/XTTS-Luo",
        "Ongea Labs Luo (Meta MMS Base)": "facebook/mms-tts-luo",
    },
    "kik": {"Ongea Labs Gikuyu (Meta MMS Base)": "facebook/mms-tts-kik"},
    "mer": {"Ongea Labs Ameru/Meru (Meta MMS Base)": "facebook/mms-tts-mer"},
    "kam": {"Ongea Labs Kamba (Meta MMS Base)": "facebook/mms-tts-kam"},
    "guz": {"Ongea Labs Ekegusii/Kisii (Meta MMS Base)": "facebook/mms-tts-guz"},
    "luy": {"Ongea Labs Luhya (Meta MMS Base)": "facebook/mms-tts-luy"},
    "kln": {"Ongea Labs Kalenjin (Meta MMS Base)": "facebook/mms-tts-kln"},
    "mas": {"Ongea Labs Maasai (Meta MMS Base)": "facebook/mms-tts-mas"},
    "dav": {"Ongea Labs Taita/Dawida (Meta MMS Base)": "facebook/mms-tts-dav"},
}


# =========================
# UTILITIES
# =========================

def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = t.strip()
    if LOWERCASE:
        t = t.lower()
    t = re.sub(r"\s+", " ", t)
    if STRIP_PUNCT:
        t = re.sub(r"[^\w\s']", "", t)
    return t

def split_by_punctuation(text: str) -> List[Tuple[str, str]]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r"\.\.\.+", "…", text)
    parts = re.findall(r"([^,.;:!?…]+)([,.;:!?…]?)", text)
    out = []
    for chunk, punct in parts:
        c = chunk.strip()
        if c:
            out.append((c, punct))
    return out

def _to_1d_float32(audio) -> np.ndarray:
    try:
        import torch
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().float().numpy()
    except Exception:
        pass
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    return audio

def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Try soundfile first (best), fallback to scipy
    try:
        import soundfile as sf
        sf.write(str(path), audio, sr, subtype="PCM_16", format="WAV")
        return
    except Exception:
        pass

    try:
        from scipy.io.wavfile import write as wavwrite
        y = np.clip(audio, -1.0, 1.0)
        y16 = (y * 32767.0).astype(np.int16)
        wavwrite(str(path), sr, y16)
    except Exception as e:
        raise RuntimeError("Could not write WAV. Install soundfile (recommended).") from e


# =========================
# TTS LOADING + SYNTHESIS
# =========================

COQUI_PREFIX = "coqui:"

@dataclass
class VoiceBundle:
    engine: str                 # "hf_vits" | "coqui"
    processor: Any = None       # HF processor
    model: Any = None           # HF model or Coqui TTS
    sr: int = TARGET_SR
    model_id: str = ""
    lang_code: str = ""

def _maybe_unidecode(text: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(text)
    except Exception:
        return text

def _get_hf_classes():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForTextToWaveform
        return AutoProcessor, AutoModelForTextToWaveform
    except Exception:
        from transformers import VitsModel
        return AutoProcessor, VitsModel

def _encode_inputs(processor, text: str):
    import torch
    try:
        inputs = processor(text=text, return_tensors="pt")
    except TypeError:
        inputs = processor(text=text, return_tensors="pt", normalize=False)
    ids = inputs.get("input_ids", None)
    if ids is not None and isinstance(ids, torch.Tensor) and (ids.numel() == 0 or ids.shape[-1] == 0):
        raise ValueError("Tokenizer produced empty input_ids.")
    return inputs

def _safe_load_hf_vits(model_id: str, lang_code: Optional[str] = None) -> VoiceBundle:
    """
    Streamlit-friendly load:
    - CPU only
    - avoids device_map auto
    - tries direct model_id first
    """
    import torch
    AutoProcessor, ModelClass = _get_hf_classes()

    last_err = None
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = ModelClass.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")
        model.eval()
        sr = int(getattr(processor, "sampling_rate", TARGET_SR) or TARGET_SR)
        return VoiceBundle(engine="hf_vits", processor=processor, model=model, sr=sr, model_id=model_id, lang_code=lang_code or "")
    except Exception as e:
        last_err = e

    # If someone passed a base repo + lang_code format, try MMS subfolder style
    if lang_code:
        try:
            base_repo = "facebook/mms-tts"
            sub = f"models/{lang_code}"
            processor = AutoProcessor.from_pretrained(base_repo, subfolder=sub)
            model = ModelClass.from_pretrained(base_repo, subfolder=sub, low_cpu_mem_usage=True)
            model.to("cpu")
            model.eval()
            sr = int(getattr(processor, "sampling_rate", TARGET_SR) or TARGET_SR)
            return VoiceBundle(engine="hf_vits", processor=processor, model=model, sr=sr, model_id=model_id, lang_code=lang_code)
        except Exception as e:
            last_err = e

    raise last_err

def _safe_load_coqui_hf(model_id: str) -> VoiceBundle:
    real_id = model_id[len(COQUI_PREFIX):].strip()
    try:
        from huggingface_hub import snapshot_download
        from TTS.api import TTS as CoquiTTS
    except Exception as e:
        raise RuntimeError("Coqui requested but not installed. Add: TTS huggingface_hub") from e

    local_dir = Path(snapshot_download(repo_id=real_id))
    cfgs = list(local_dir.rglob("config.json")) + list(local_dir.rglob("*config*.json"))
    if not cfgs:
        raise RuntimeError(f"No config.json found in {real_id}")
    config_path = str(cfgs[0])

    ckpts = []
    for ext in ("*.pth", "*.pt", "*.bin", "*.safetensors"):
        ckpts += list(local_dir.rglob(ext))
    ckpts = [p for p in ckpts if p.is_file()]
    if not ckpts:
        raise RuntimeError(f"No checkpoint found in {real_id}")

    model_path = str(ckpts[0])
    tts = CoquiTTS(model_path=model_path, config_path=config_path, progress_bar=False, gpu=False)

    out_sr = TARGET_SR
    try:
        out_sr = int(getattr(tts.synthesizer, "output_sample_rate", TARGET_SR))
    except Exception:
        pass

    return VoiceBundle(engine="coqui", model=tts, sr=out_sr, model_id=model_id)

def _safe_load_model(model_id: str, lang_code: Optional[str] = None) -> VoiceBundle:
    if model_id.startswith(COQUI_PREFIX):
        return _safe_load_coqui_hf(model_id)
    return _safe_load_hf_vits(model_id, lang_code=lang_code)

def synthesize_raw(bundle: VoiceBundle, text: str) -> Tuple[np.ndarray, int]:
    text = clean_text(text)
    if not text:
        raise ValueError("Empty text.")

    if bundle.engine == "coqui":
        wave = None
        try:
            wave = bundle.model.tts(text)
        except Exception:
            wave = bundle.model.tts(_maybe_unidecode(text))
        audio = _to_1d_float32(wave)
        sr = int(bundle.sr or TARGET_SR)
        if audio.size == 0:
            raise ValueError("Coqui returned empty audio.")
        m = float(np.max(np.abs(audio)))
        if m > 1.0:
            audio = audio / m
        return np.clip(audio, -1.0, 1.0), sr

    processor, model = bundle.processor, bundle.model
    inputs = _encode_inputs(processor, text)

    import torch
    with torch.no_grad():
        out = model(**inputs)

    wave = None
    if isinstance(out, dict) and "waveform" in out:
        wave = out["waveform"]
    else:
        wave = getattr(out, "waveform", None)
        if wave is None:
            wave = out[0]

    audio = _to_1d_float32(wave)
    sr = int(getattr(processor, "sampling_rate", TARGET_SR) or TARGET_SR)

    if audio.size == 0:
        raise ValueError("Model returned empty audio.")
    m = float(np.max(np.abs(audio)))
    if m > 1.0:
        audio = audio / m
    return np.clip(audio, -1.0, 1.0), sr

def synthesize_human(bundle: VoiceBundle, text: str) -> Tuple[np.ndarray, int]:
    chunks = split_by_punctuation(text)
    if not chunks:
        raise ValueError("Empty text.")

    audios = []
    pause = {",": 0.18, ";": 0.22, ":": 0.22, ".": 0.38, "!": 0.42, "?": 0.42, "…": 0.55}
    sr_final = int(bundle.sr or TARGET_SR)

    for chunk_text, punct in chunks:
        a, sr = synthesize_raw(bundle, chunk_text)
        sr_final = int(sr or sr_final)
        audios.append(a)
        dur = pause.get(punct, 0.0)
        if dur > 0:
            audios.append(np.zeros(int(sr_final * dur), dtype=np.float32))

    audio = np.concatenate(audios) if len(audios) > 1 else audios[0]
    m = float(np.max(np.abs(audio))) if audio.size else 1.0
    if m > 1.0:
        audio = audio / m
    return np.clip(audio, -1.0, 1.0), sr_final

def apply_tone(audio: np.ndarray, sr: int, speed: float, pitch_semitones: float) -> np.ndarray:
    # Optional: works if librosa is installed. Otherwise returns original.
    if speed == 1.0 and pitch_semitones == 0.0:
        return audio
    try:
        import librosa
        y = audio.astype(np.float32)
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=float(speed))
        if pitch_semitones != 0.0:
            y = librosa.effects.pitch_shift(y, sr=int(sr), n_steps=float(pitch_semitones))
        m = float(np.max(np.abs(y))) if y.size else 1.0
        if m > 1.0:
            y = y / m
        return np.clip(y, -1.0, 1.0)
    except Exception:
        return audio


# =========================
# STREAMLIT UI (Cloud-safe)
# =========================

def _init_state(st):
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"  # dark|light
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True

    if "lang_name" not in st.session_state:
        st.session_state.lang_name = list(LANGUAGES.keys())[0]
    if "voice_name" not in st.session_state:
        st.session_state.voice_name = None

    if "mode" not in st.session_state:
        st.session_state.mode = "Ongea"  # Ongea|Batch
    if "speak_text" not in st.session_state:
        st.session_state.speak_text = ""
    if "batch_lines" not in st.session_state:
        st.session_state.batch_lines = ""

    if "speed" not in st.session_state:
        st.session_state.speed = 1.0
    if "pitch" not in st.session_state:
        st.session_state.pitch = 0.0

    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest_idx" not in st.session_state:
        st.session_state.latest_idx = None

def get_voices_for(lang_code: str, lang_name: str):
    voices = VOICE_LIBRARY_BY_LANG.get(lang_code, {})
    if not voices:
        voices = {f"Ongea Labs {lang_name.split('—')[0].strip()} (Meta MMS Base)": f"facebook/mms-tts-{lang_code}"}
    return voices

def inject_css(st, theme: str, sidebar_open: bool):
    sidebar_css = ""
    if not sidebar_open:
        sidebar_css = """
<style>
section[data-testid="stSidebar"]{display:none !important;}
[data-testid="stSidebar"]{display:none !important;}
</style>
"""

    st.markdown(
        f"""
<style>
#MainMenu, footer, header {{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none !important;}}
[data-testid="stStatusWidget"]{{display:none !important;}}
[data-testid="stHeader"]{{display:none !important;}}
[data-testid="stDecoration"]{{display:none !important;}}

/* Hide Streamlit built-in sidebar toggle */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarToggleButton"],
button[title="Open sidebar"],
button[title="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"] {{
  display:none !important;
}}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {{
  --radius: 22px;
  --shadow: 0 22px 60px rgba(12, 24, 48, 0.12);
  --shadow2: 0 12px 28px rgba(12, 24, 48, 0.10);
  --soft: rgba(255,255,255,0.10);

  --font: "Plus Jakarta Sans", Inter, system-ui, -apple-system, Segoe UI, Roboto;
  --font2: Inter, system-ui, -apple-system, Segoe UI;
}}

:root {{
  --bgA: {"#dff7fb" if theme=="light" else "#050814"};
  --bgB: {"#eef4ff" if theme=="light" else "#0b1633"};
  --bgC: {"#f6ecff" if theme=="light" else "#052f2a"};

  --card: {"rgba(255,255,255,0.66)" if theme=="light" else "rgba(12,16,32,0.62)"};
  --card2: {"rgba(255,255,255,0.80)" if theme=="light" else "rgba(12,16,32,0.78)"};
  --input: {"rgba(255,255,255,0.86)" if theme=="light" else "rgba(10,12,24,0.78)"};

  --text: {"#0b1220" if theme=="light" else "#eaf2ff"};
  --muted: {"rgba(11,18,32,0.62)" if theme=="light" else "rgba(234,242,255,0.70)"};
  --soft2: {"rgba(11,18,32,0.14)" if theme=="light" else "rgba(234,242,255,0.14)"};
}}

html, body {{
  font-family: var(--font) !important;
  color: var(--text) !important;
  background: transparent !important;
}}

[data-testid="stAppViewContainer"]{{
  background:
    radial-gradient(1200px 700px at 10% 0%, rgba(25,182,173,{"0.35" if theme=="light" else "0.22"}) 0%, transparent 55%),
    radial-gradient(1000px 650px at 92% 5%, rgba(107,102,255,{"0.32" if theme=="light" else "0.18"}) 0%, transparent 55%),
    radial-gradient(900px 700px at 70% 90%, rgba(255,77,125,{"0.20" if theme=="light" else "0.10"}) 0%, transparent 55%),
    linear-gradient(135deg, var(--bgA), var(--bgB), var(--bgC)) !important;
}}

.block-container{{
  max-width: 1280px;
  padding: 1.25rem 1.2rem 1.4rem 1.2rem !important;
}}

[data-testid="stSidebar"] > div{{
  background: var(--card2) !important;
  backdrop-filter: blur(14px);
  border-right: 1px solid var(--soft) !important;
}}

.oge-card{{
  background: var(--card);
  border: 1px solid var(--soft);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  backdrop-filter: blur(14px);
}}
.oge-card2{{
  background: var(--card2);
  border: 1px solid var(--soft);
  border-radius: var(--radius);
  box-shadow: var(--shadow2);
  backdrop-filter: blur(14px);
}}
.oge-pad{{ padding: 1.1rem 1.1rem; }}

label{{ font-family: var(--font2) !important; font-weight: 700 !important; }}

div[data-baseweb="select"] > div{{
  background: var(--input) !important;
  border: 1px solid var(--soft2) !important;
  border-radius: 16px !important;
  min-height: 44px !important;
}}
div[data-baseweb="select"] span, div[data-baseweb="select"] input{{
  color: var(--text) !important;
  font-weight: 650 !important;
}}

textarea{{
  background: var(--input) !important;
  border: 1px solid var(--soft2) !important;
  border-radius: 18px !important;
  color: var(--text) !important;
  font-size: 1.02rem !important;
  line-height: 1.35 !important;
}}
textarea::placeholder{{ color: rgba(120,130,150,0.70) !important; }}

.stButton>button{{
  border-radius: 999px !important;
  border: 1px solid rgba(25,182,173,0.45) !important;
  background: linear-gradient(135deg, rgba(25,182,173,0.22), rgba(107,102,255,0.16)) !important;
  padding: 0.7rem 1.05rem !important;
  font-weight: 900 !important;
  box-shadow: var(--shadow2);
}}
.stButton>button:hover{{ transform: translateY(-1px) scale(1.01); }}

audio {{ width: 100% !important; border-radius: 999px !important; }}
</style>
{sidebar_css}
""",
        unsafe_allow_html=True,
    )

def get_voice_loader(st):
    @st.cache_resource(show_spinner=False)
    def load_voice(model_id: str, lang_code: str) -> VoiceBundle:
        return _safe_load_model(model_id, lang_code=lang_code)
    return load_voice

def render_sidebar(st):
    sb = st.sidebar

    sb.markdown(
        """
<div style="display:flex;align-items:center;gap:12px;padding:6px 6px 10px 6px;">
  <div style="width:50px;height:50px;border-radius:18px;background:linear-gradient(135deg,rgba(25,182,173,0.35),rgba(107,102,255,0.25));
  border:1px solid rgba(255,255,255,0.10);display:flex;align-items:center;justify-content:center;">🎙️</div>
  <div>
    <div style="font-weight:900;font-size:1.1rem;line-height:1.1;">Ongea Labs</div>
    <div style="opacity:.75;font-weight:700;">Chat TTS Studio</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = sb.columns([0.58, 0.42], gap="small")
    with c1:
        if sb.button("+ New chat", use_container_width=True):
            st.session_state.speak_text = ""
            st.session_state.batch_lines = ""
            st.session_state.latest_idx = None
    with c2:
        if sb.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.latest_idx = None

    sb.text_input("Search (visual only)", value="", key="sb_search")

    sb.markdown("<hr style='opacity:.15;margin:16px 0;'/>", unsafe_allow_html=True)
    sb.markdown("<div style='font-weight:900;margin-bottom:.35rem;'>History</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        sb.markdown("<div style='opacity:.75;'>No speech yet.</div>", unsafe_allow_html=True)
        return

    for item in reversed(st.session_state.history[-15:]):
        sb.markdown(
            f"<div style='padding:.7rem;border-radius:16px;border:1px solid rgba(255,255,255,0.10);"
            f"margin-bottom:.6rem;background:rgba(0,0,0,0.08);'>"
            f"<div style='font-weight:900;margin-bottom:.2rem;'>{item['label']}</div>"
            f"<div style='opacity:.75;font-size:.86rem;margin-bottom:.35rem;'>{item['ts']} • {item['voice_name']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        sb.audio(item["wav_path"], format="audio/wav")
        try:
            sb.download_button(
                "Download WAV",
                data=Path(item["wav_path"]).read_bytes(),
                file_name=Path(item["wav_path"]).name,
                mime="audio/wav",
                use_container_width=True,
                key=f"sb_dl_{item['id']}",
            )
        except Exception:
            pass

def top_right_settings(st):
    pop = st.popover("⚙️  Settings")
    with pop:
        # Sidebar toggle (single control)
        if st.button(("Hide sidebar" if st.session_state.sidebar_open else "Show sidebar"), use_container_width=True):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()

        # Theme toggle
        if st.button(("Switch to Light" if st.session_state.theme == "dark" else "Switch to Dark"), use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

        st.markdown("---")

        # Language/voice
        lang_keys = list(LANGUAGES.keys())
        st.session_state.lang_name = st.selectbox(
            "Language",
            lang_keys,
            index=lang_keys.index(st.session_state.lang_name),
            key="lang_select_top",
        )
        lang_code = LANGUAGES[st.session_state.lang_name]
        voices = get_voices_for(lang_code, st.session_state.lang_name)
        voice_names = list(voices.keys())
        if st.session_state.voice_name not in voice_names:
            st.session_state.voice_name = voice_names[0]

        st.session_state.voice_name = st.selectbox(
            "Voice / Model",
            voice_names,
            index=voice_names.index(st.session_state.voice_name),
            key="voice_select_top",
        )

        st.session_state.speed = st.slider("Speed", 0.75, 1.50, float(st.session_state.speed), 0.05, key="speed_top")
        st.session_state.pitch = st.slider("Pitch (semitones)", -4.0, 4.0, float(st.session_state.pitch), 0.5, key="pitch_top")

        st.markdown("---")
        if st.button("Clear cached models (forces reload)", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cleared model cache. Now generate again.")
        st.caption("Tip: If Cloud seems ‘stuck’, reboot app + clear cache in Streamlit settings.")

def studio_view(st):
    # Build stamp (helps you confirm the deployed file actually changed)
    try:
        mtime = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        mtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Build stamp: {mtime}")

    st.markdown(
        """
<div style="text-align:center;margin:2.2rem 0 1.1rem 0;">
  <div style="font-size:2.7rem;font-weight:900;letter-spacing:-0.02em;">What’s on your mind today?</div>
  <div style="margin-top:.55rem;opacity:.75;font-weight:700;font-size:1.05rem;">
    Type text → get natural speech audio. Use ⚙️ for language, voice, speed & pitch.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.session_state.mode = st.radio(
        "Mode",
        ["Ongea", "Batch"],
        index=0 if st.session_state.mode == "Ongea" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="mode_radio_main",
    )

    st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)

    if st.session_state.mode == "Ongea":
        st.session_state.speak_text = st.text_area(
            "Text",
            value=st.session_state.speak_text,
            height=190,
            placeholder="Example:\nHabari! Karibu kwenye Ongea Labs.\nAndika maandishi yako hapa…",
            label_visibility="collapsed",
            key="speak_text_area",
        )
        _, colB = st.columns([0.82, 0.18], gap="small")
        with colB:
            if st.button("Send", use_container_width=True, key="send_one"):
                _do_generate_speak(st)
    else:
        st.session_state.batch_lines = st.text_area(
            "Lines",
            value=st.session_state.batch_lines,
            height=210,
            placeholder="Line 1: Habari! Karibu kwenye Ongea.\nLine 2: Leo tutaongea kuhusu...\nLine 3: Asante kwa kusikiliza.",
            label_visibility="collapsed",
            key="batch_lines_area",
        )
        _, colB = st.columns([0.82, 0.18], gap="small")
        with colB:
            if st.button("Send", use_container_width=True, key="send_batch"):
                _do_generate_batch(st)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
    st.markdown("<div style='font-weight:900;font-size:1.15rem;margin-bottom:.35rem;'>Latest</div>", unsafe_allow_html=True)

    if st.session_state.latest_idx is not None and st.session_state.history:
        it = st.session_state.history[st.session_state.latest_idx]
        st.audio(it["wav_path"], format="audio/wav")
        try:
            st.download_button(
                "Download WAV",
                data=Path(it["wav_path"]).read_bytes(),
                file_name=Path(it["wav_path"]).name,
                mime="audio/wav",
                use_container_width=True,
                key="latest_dl_btn",
            )
        except Exception:
            pass
        st.caption(f"{it['ts']} • {it['lang_code']} • {it['voice_name']} • SR={it['sr']}")
    else:
        st.markdown("<div style='opacity:.75;'>Nothing yet.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def _do_generate_speak(st):
    text = st.session_state.speak_text.strip()
    if not text:
        st.warning("Paste some text first.")
        return

    lang_code = LANGUAGES[st.session_state.lang_name]
    voices = get_voices_for(lang_code, st.session_state.lang_name)
    voice_name = st.session_state.voice_name
    model_id = voices[voice_name]
    load_voice = get_voice_loader(st)

    with st.spinner("Loading voice (first time can take a bit)…"):
        try:
            bundle = load_voice(model_id, lang_code)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.info("On Streamlit Cloud: confirm requirements.txt + runtime.txt (Python 3.11 recommended).")
            st.stop()

    with st.spinner("Generating audio…"):
        try:
            audio, sr = synthesize_human(bundle, text)
            audio = apply_tone(audio, sr, speed=float(st.session_state.speed), pitch_semitones=float(st.session_state.pitch))

            out_wav = OUTPUT_DIR / "app_outputs" / f"{lang_code}_speech_{len(st.session_state.history)+1:03d}.wav"
            write_wav(out_wav, audio, sr)

            item = {
                "id": f"{lang_code}_s_{len(st.session_state.history)+1:03d}",
                "ts": datetime.now().strftime("%H:%M:%S"),
                "lang_code": lang_code,
                "voice_name": voice_name,
                "label": "Ongea",
                "wav_path": str(out_wav),
                "sr": int(sr),
            }
            st.session_state.history.append(item)
            st.session_state.latest_idx = len(st.session_state.history) - 1
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate speech: {e}")

def _do_generate_batch(st):
    raw = st.session_state.batch_lines
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if not lines:
        st.warning("Paste one sentence per line first.")
        return

    lang_code = LANGUAGES[st.session_state.lang_name]
    voices = get_voices_for(lang_code, st.session_state.lang_name)
    voice_name = st.session_state.voice_name
    model_id = voices[voice_name]
    load_voice = get_voice_loader(st)

    with st.spinner("Loading voice (first time can take a bit)…"):
        try:
            bundle = load_voice(model_id, lang_code)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.info("On Streamlit Cloud: confirm requirements.txt + runtime.txt (Python 3.11 recommended).")
            st.stop()

    with st.spinner("Generating batch…"):
        try:
            for i, ln in enumerate(lines, start=1):
                audio, sr = synthesize_human(bundle, ln)
                audio = apply_tone(audio, sr, speed=float(st.session_state.speed), pitch_semitones=float(st.session_state.pitch))
                p = OUTPUT_DIR / "app_outputs" / f"{lang_code}_batch_{len(st.session_state.history)+1:03d}_{i:02d}.wav"
                write_wav(p, audio, sr)

                item = {
                    "id": f"{lang_code}_b_{len(st.session_state.history)+1:03d}_{i:02d}",
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "lang_code": lang_code,
                    "voice_name": voice_name,
                    "label": f"Batch • Line {i}",
                    "wav_path": str(p),
                    "sr": int(sr),
                }
                st.session_state.history.append(item)

            st.session_state.latest_idx = len(st.session_state.history) - 1
            st.rerun()
        except Exception as e:
            st.error(f"Batch failed: {e}")

def run_app():
    import streamlit as st
    st.set_page_config(page_title="Ongea", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")
    _init_state(st)

    inject_css(st, st.session_state.theme, st.session_state.sidebar_open)

    if st.session_state.sidebar_open:
        render_sidebar(st)

    topL, topR = st.columns([0.72, 0.28], gap="small")
    with topR:
        top_right_settings(st)

    studio_view(st)


# =========================
# OPTIONAL: TRAINING (CLI ONLY)
# =========================
# NOTE: This is intentionally left minimal; training on Streamlit Cloud is not recommended.
def run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)

def launch_training(lang_code: str):
    raise RuntimeError(
        "Training is disabled in this Streamlit-first file. "
        "Use your separate training script locally/colab."
    )


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--lang", type=str, default="swh")
    args = parser.parse_args()

    if args.train:
        launch_training(args.lang)
    else:
        run_app()
