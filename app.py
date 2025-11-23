# app.py
"""
ONGEA v5.3 — Pan-African Text-to-Speech Studio (FAILSAFE BUILD)
Languages: Swahili, Amharic, Somali, Yoruba, Shona, Xhosa, Afrikaans, Lingala, Kongo
ONE FILE. No external assets.

Run:
    streamlit run app.py
Train:
    python app.py --train --lang swh

Why this version won't white-screen:
- UI renders BEFORE any heavy CSS.
- No components.html() CSS injection.
- No parent.document JS parallax.
- No render-time model probing.
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

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
}

PROJECT_NAME = "ongea-v5-mms-tts-multi"
OUTPUT_DIR = Path("./outputs") / PROJECT_NAME
CONVERTED_DIR = OUTPUT_DIR / "training_checkpoint_with_discriminator"
FINETUNE_REPO = Path("./finetune-hf-vits")

TARGET_SR = 16000
MIN_AUDIO_SEC = 1.0
MAX_AUDIO_SEC = 15.0

LEARNING_RATE = 2e-4
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
MAX_STEPS = 8000
WARMUP_STEPS = 200
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 500

LOWERCASE = True
STRIP_PUNCT = False  # keep punctuation for prosody splitting


# =========================
# VOICE LIBRARY (by language)
# =========================

VOICE_LIBRARY_BY_LANG: Dict[str, Dict[str, str]] = {
    "swh": {
        "Ongea Swahili Male / Neutral (Meta Base)": "facebook/mms-tts-swh",
        "Ongea Swahili Female (Mozilla Lady)": "Benjamin-png/swahili-mms-tts-mozilla-lady-voice-finetuned",
        "Ongea Swahili Studio (Fine-tuned)": "Benjamin-png/swahili-mms-tts-finetuned",
        "Ongea Swahili Narrator (OpenBible)": "bookbot/vits-base-sw-KE-OpenBible",
        "Ongea Swahili SALAMA (Prosody-rich)": "EYEDOL/SALAMA_TTS",
    },
    # ✅ MMS Amharic is flaky on Windows/Py3.13 -> add backup that works without uroman
    "amh": {
        "Ongea Amharic (Meta MMS Base)": "facebook/mms-tts-amh",
        "Ongea Amharic (SpeechT5 Backup)": "AddisuSeteye/speecht5_tts_amharic",
    },
    "som": {"Ongea Somali (Meta MMS Base)": "facebook/mms-tts-som"},
    "yor": {"Ongea Yoruba (Meta MMS Base)": "facebook/mms-tts-yor"},
    "sna": {"Ongea Shona (Meta MMS Base)": "facebook/mms-tts-sna"},
    "xho": {"Ongea Xhosa (Meta MMS Base)": "facebook/mms-tts-xho"},
    "afr": {"Ongea Afrikaans (Meta MMS Base)": "facebook/mms-tts-afr"},
    "lin": {"Ongea Lingala (Meta MMS Base)": "facebook/mms-tts-lin"},
    "kon": {"Ongea Kongo (Meta MMS Base)": "facebook/mms-tts-kon"},
}


# =========================
# UTILITIES
# =========================

def run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)

def ensure_repo():
    if FINETUNE_REPO.exists():
        return
    run(["git", "clone", "https://github.com/ylacombe/finetune-hf-vits.git", str(FINETUNE_REPO)])

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


# =========================
# TRAINING (lazy imports)
# =========================

def detect_columns(ds) -> Tuple[str, str]:
    cols = set(ds.column_names)
    audio_col = DEFAULT_AUDIO_COL if DEFAULT_AUDIO_COL in cols else None
    text_col = DEFAULT_TEXT_COL if DEFAULT_TEXT_COL in cols else None

    if audio_col is None:
        for c in cols:
            if "audio" in c or "speech" in c or "wav" in c:
                audio_col = c
                break
    if text_col is None:
        for c in cols:
            if "text" in c or "transcript" in c or "sentence" in c:
                text_col = c
                break

    if audio_col is None or text_col is None:
        raise ValueError(f"Could not auto-detect columns. Found: {cols}")
    return audio_col, text_col

def load_and_prepare_dataset() -> Tuple[Any, Optional[Any], str, str]:
    from datasets import load_dataset, Audio
    ds_train = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=TRAIN_SPLIT)
    ds_eval = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=EVAL_SPLIT) if EVAL_SPLIT else None

    audio_col, text_col = detect_columns(ds_train)
    ds_train = ds_train.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))
    if ds_eval is not None:
        ds_eval = ds_eval.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

    def _norm(ex):
        ex[text_col] = clean_text(ex[text_col])
        return ex

    ds_train = ds_train.map(_norm)
    if ds_eval is not None:
        ds_eval = ds_eval.map(_norm)

    def _keep(ex):
        a = ex[audio_col]
        if a is None or a.get("array") is None:
            return False
        dur = len(a["array"]) / a["sampling_rate"]
        if dur < MIN_AUDIO_SEC or dur > MAX_AUDIO_SEC:
            return False
        if ex[text_col] is None or ex[text_col].strip() == "":
            return False
        return True

    ds_train = ds_train.filter(_keep)
    if ds_eval is not None:
        ds_eval = ds_eval.filter(_keep)

    return ds_train, ds_eval, audio_col, text_col

def maybe_convert_discriminator(lang_code: str) -> str:
    lang_dir = CONVERTED_DIR / lang_code
    if (lang_dir / "config.json").exists():
        return str(lang_dir)
    ensure_repo()
    lang_dir.mkdir(parents=True, exist_ok=True)
    run([
        "python",
        "convert_original_discriminator_checkpoint.py",
        "--language_code", lang_code,
        "--pytorch_dump_folder_path", str(lang_dir),
    ], cwd=FINETUNE_REPO)
    return str(lang_dir)

def build_finetune_config(model_path: str, audio_col: str, text_col: str, lang_code: str) -> Dict[str, Any]:
    import torch
    return {
        "project_name": f"{PROJECT_NAME}-{lang_code}",
        "model_name_or_path": model_path,
        "output_dir": str(OUTPUT_DIR / lang_code),
        "push_to_hub": False,
        "dataset_name": HF_DATASET_NAME,
        "dataset_config_name": HF_DATASET_CONFIG,
        "train_split_name": TRAIN_SPLIT,
        "eval_split_name": EVAL_SPLIT,
        "audio_column_name": audio_col,
        "text_column_name": text_col,
        "sampling_rate": TARGET_SR,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "max_steps": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "logging_steps": LOGGING_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "do_train": True,
        "do_eval": bool(EVAL_SPLIT),
        "fp16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "max_audio_length_in_seconds": MAX_AUDIO_SEC,
        "min_audio_length_in_seconds": MIN_AUDIO_SEC,
    }

def launch_training(lang_code: str):
    _, _, audio_col, text_col = load_and_prepare_dataset()
    model_path = maybe_convert_discriminator(lang_code)

    out_dir = OUTPUT_DIR / lang_code
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_finetune_config(model_path, audio_col, text_col, lang_code)
    cfg_path = out_dir / "finetune_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    ensure_repo()
    train_script = FINETUNE_REPO / "run_vits_finetuning.py"
    run([
        "accelerate", "launch",
        str(train_script),
        "--config", str(cfg_path),
    ], cwd=FINETUNE_REPO)


# =========================
# TTS LOADING + SYNTHESIS
# =========================

BASE_MMS_REPO = "facebook/mms-tts"

def _get_model_classes():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForTextToWaveform
        return AutoProcessor, AutoModelForTextToWaveform
    except Exception:
        from transformers import VitsModel
        return AutoProcessor, VitsModel

def _load_speecht5_bundle(model_id: str):
    """
    SpeechT5 needs:
      - SpeechT5Processor
      - SpeechT5ForTextToSpeech
      - Vocoder (HiFiGAN)
      - A speaker embedding (xvector); we use CMU Arctic default.
    """
    import torch
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset

    processor = SpeechT5Processor.from_pretrained(model_id)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Default speaker embedding:
    # (works fine for single-speaker fine-tunes too)
    xvec_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    spk = torch.tensor(xvec_ds[0]["xvector"]).unsqueeze(0)

    model.to("cpu").eval()
    vocoder.to("cpu").eval()

    return {
        "kind": "speecht5",
        "processor": processor,
        "model": model,
        "vocoder": vocoder,
        "speaker_embedding": spk,
        "sampling_rate": 16000,
    }

def _safe_load_model(model_id: str, lang_code: Optional[str] = None):
    """
    Robust loader:
      1) SpeechT5 special-case
      2) try model_id directly
      3) fallback to monorepo: facebook/mms-tts subfolder=models/<lang_code>
    """
    import torch

    # 1) SpeechT5 special-case (backup)
    if "speecht5_tts_amharic" in model_id.lower() or "speecht5" in model_id.lower():
        return _load_speecht5_bundle(model_id)

    # 2/3) VITS/MMS path
    AutoProcessor, ModelClass = _get_model_classes()
    last_err = None

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = ModelClass.from_pretrained(model_id, low_cpu_mem_usage=False, device_map=None)
        if any(getattr(p, "is_meta", False) for p in model.parameters()):
            raise RuntimeError("Model loaded with meta tensors.")
        model.to("cpu"); model.eval()
        return {"kind": "vits", "processor": processor, "model": model}
    except Exception as e:
        last_err = e

    if lang_code:
        try:
            sub = f"models/{lang_code}"
            processor = AutoProcessor.from_pretrained(BASE_MMS_REPO, subfolder=sub)
            model = ModelClass.from_pretrained(BASE_MMS_REPO, subfolder=sub, low_cpu_mem_usage=False, device_map=None)
            if any(getattr(p, "is_meta", False) for p in model.parameters()):
                raise RuntimeError("Model loaded with meta tensors.")
            model.to("cpu"); model.eval()
            return {"kind": "vits", "processor": processor, "model": model}
        except Exception as e:
            last_err = e

    raise last_err

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

def _encode_inputs_vits(processor, text: str):
    """
    Encode for VITS/MMS. If tokenizer yields empty ids, raise clear error.
    """
    text = clean_text(text)
    if not text:
        raise ValueError("Empty text.")
    inputs = processor(text=text, return_tensors="pt")
    ids = inputs.get("input_ids", None)
    if ids is None or ids.numel() == 0 or ids.shape[-1] == 0:
        raise ValueError(
            "Tokenizer produced empty input_ids for this text. "
            "This is a known MMS Amharic issue on some Windows/Python setups. "
            "Try the SpeechT5 Amharic voice."
        )
    return inputs

def synthesize_raw(bundle: Dict[str, Any], text: str, *, model_id: str = "", lang_code: str = "") -> Tuple[np.ndarray, int]:
    """
    Raw synthesis for both kinds:
      - kind=vits (MMS/VITS)
      - kind=speecht5 (backup)
    """
    import torch

    kind = bundle.get("kind", "vits")

    if kind == "speecht5":
        processor = bundle["processor"]
        model = bundle["model"]
        vocoder = bundle["vocoder"]
        spk = bundle["speaker_embedding"]

        text = clean_text(text)
        if not text:
            raise ValueError("Empty text.")

        inputs = processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            speech = model.generate_speech(
                input_ids=input_ids,
                speaker_embeddings=spk,
                vocoder=vocoder
            )

        audio = _to_1d_float32(speech)
        sr = int(bundle.get("sampling_rate", TARGET_SR))
        if audio.size == 0:
            raise ValueError("Model returned empty audio.")
        m = float(np.max(np.abs(audio)))
        if m > 1.0:
            audio = audio / m
        return np.clip(audio, -1.0, 1.0), sr

    # VITS/MMS
    processor = bundle["processor"]
    model = bundle["model"]

    inputs = _encode_inputs_vits(processor, text)
    with torch.no_grad():
        out = model(**inputs)

    if isinstance(out, dict) and "waveform" in out:
        wave = out["waveform"]
    else:
        wave = getattr(out, "waveform", None) or out[0]

    audio = _to_1d_float32(wave)
    sr = int(getattr(processor, "sampling_rate", TARGET_SR))
    if audio.size == 0:
        raise ValueError("Model returned empty audio.")
    m = float(np.max(np.abs(audio)))
    if m > 1.0:
        audio = audio / m
    return np.clip(audio, -1.0, 1.0), sr

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

def synthesize_human(bundle: Dict[str, Any], text: str, *, model_id: str = "", lang_code: str = "") -> Tuple[np.ndarray, int]:
    chunks = split_by_punctuation(text)
    if not chunks:
        raise ValueError("Empty text.")

    audios = []
    sr_final = TARGET_SR
    pause = {",":0.18, ";":0.22, ":":0.22, ".":0.38, "!":0.42, "?":0.42, "…":0.55}

    for chunk_text, punct in chunks:
        a, sr = synthesize_raw(bundle, chunk_text, model_id=model_id, lang_code=lang_code)
        sr_final = sr_final or sr
        audios.append(a)
        dur = pause.get(punct, 0.0)
        if dur > 0:
            audios.append(np.zeros(int(sr * dur), dtype=np.float32))

    audio = np.concatenate(audios) if len(audios) > 1 else audios[0]
    m = float(np.max(np.abs(audio))) if audio.size else 1.0
    if m > 1.0:
        audio = audio / m
    return np.clip(audio, -1.0, 1.0), sr_final

def apply_tone(audio: np.ndarray, sr: int, speed: float, pitch_semitones: float) -> np.ndarray:
    try:
        import librosa
        y = audio.astype(np.float32)
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed)
        if pitch_semitones != 0.0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_semitones)
        m = float(np.max(np.abs(y))) if y.size else 1.0
        if m > 1.0:
            y = y / m
        return np.clip(y, -1.0, 1.0)
    except Exception:
        return audio

def write_wav(path: Path, audio: np.ndarray, sr: int):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16", format="WAV")


# =========================
# STREAMLIT UI (FAILSAFE)
# =========================

SAFE_CSS = """
<style>
:root{
  --bg1:#050814; --bg2:#0B1633; --bg3:#052F2A;
  --glass:rgba(255,255,255,0.08);
  --accent:#00E0B8; --accent2:#F9C74F;
  --txt:#EAF2FF; --muted:rgba(234,242,255,0.72);
}

/* Main background - MINIMAL to avoid conflicts */
[data-testid="stAppViewContainer"]{
  background: linear-gradient(135deg,#050814,#0B1633,#052F2A) !important;
  color: var(--txt) !important;
}

[data-testid="stMain"]{
  background: transparent !important;
}

.block-container{
  background: transparent !important;
  max-width:1120px; 
  padding-top:0.9rem;
}

/* Ensure text is visible */
body, p, span, h1, h2, h3, h4, h5, h6{
  color: var(--txt) !important;
}

.glass{
  background: rgba(11, 22, 51, 0.7) !important;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:1.05rem 1.15rem;
  box-shadow:0 10px 35px rgba(0,0,0,0.30);
  transition:transform .25s ease, box-shadow .25s ease, border .25s ease;
}
.glass:hover{
  transform: translateY(-3px);
  border-color:rgba(0,224,184,0.7);
  box-shadow:0 14px 45px rgba(0,224,184,0.20);
}

.ongea-title{
  font-size:3.1rem; font-weight:900; letter-spacing:.02em;
  color: #00E0B8 !important;
}
.ongea-sub{color:var(--muted);font-size:1.07rem;margin-top:.2rem}

#ongea-nav{
  display:flex; gap:.45rem; padding:.55rem .6rem; margin:.8rem 0 .9rem 0;
  background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,.10);
  border-radius:16px;
  box-shadow:0 8px 24px rgba(0,0,0,.28);
}
#ongea-nav .stButton button{
  width:100%;
  background:transparent !important;
  border:1px solid rgba(255,255,255,.12) !important;
  color:var(--txt) !important; border-radius:12px !important;
  font-weight:800 !important; letter-spacing:.02em;
  padding:.55rem .8rem !important;
  transition:all .18s ease !important;
}
#ongea-nav .stButton button:hover{
  transform:translateY(-1px) scale(1.02);
  border-color:rgba(0,224,184,.9) !important;
  box-shadow:0 10px 26px rgba(0,224,184,.22);
}
#ongea-nav .active .stButton button{
  background:linear-gradient(135deg, rgba(0,224,184,.18), rgba(249,199,79,.12)) !important;
  border-color:rgba(0,224,184,.9) !important;
}

.stButton button{
  background:linear-gradient(135deg,rgba(0,224,184,0.22),rgba(249,199,79,0.14)) !important;
  border:1px solid rgba(0,224,184,0.7) !important;
  color:var(--txt) !important; border-radius:12px !important; font-weight:800 !important;
  transition:all .18s ease !important; box-shadow:0 10px 24px rgba(0,0,0,0.25);
}
.stButton button:hover{transform:translateY(-2px) scale(1.02);}

textarea{
  background:rgba(8,12,25,0.98) !important;
  border:1px solid rgba(255,255,255,0.12) !important;
  color:var(--txt) !important; border-radius:14px !important;
}

div[data-baseweb="select"] > div{
  background:rgba(8,12,25,0.9) !important;
  border:1px solid rgba(255,255,255,.14) !important;
  border-radius:12px !important;
  color:var(--txt) !important;
  transition:border .18s ease, box-shadow .18s ease;
}
div[data-baseweb="select"] > div:hover{
  border-color:rgba(0,224,184,.85) !important;
  box-shadow:0 0 0 2px rgba(0,224,184,.18);
}
div[data-baseweb="select"] span{color:var(--txt) !important;}
div[data-baseweb="popover"]{background:rgba(8,12,25,0.98) !important;}
</style>
"""

def inject_theme(st):
    st.markdown(SAFE_CSS, unsafe_allow_html=True)

def header_block(st):
    st.markdown(
        """
<div class="glass">
  <div class="ongea-title">Ongea</div>
  <div class="ongea-sub">Pan-African TTS Studio • Meta MMS + Community Voices</div>
</div>
        """,
        unsafe_allow_html=True
    )

def get_voice_loader():
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def load_voice(model_id: str, lang_code: str):
        return _safe_load_model(model_id, lang_code=lang_code)
    return load_voice

def _init_state(st):
    if "page" not in st.session_state:
        st.session_state.page = "Speak"
    if "lang_name" not in st.session_state:
        st.session_state.lang_name = list(LANGUAGES.keys())[0]

def render_nav(st):
    pages = ["Speak", "Batch", "Fine-tune", "About"]
    st.markdown('<div id="ongea-nav">', unsafe_allow_html=True)
    cols = st.columns(len(pages))
    for i, p in enumerate(pages):
        active = (st.session_state.page == p)
        with cols[i]:
            st.markdown('<div class="active">' if active else "<div>", unsafe_allow_html=True)
            if st.button(p, key=f"nav_{p}", use_container_width=True):
                st.session_state.page = p
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def language_panel(st):
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    lang_name = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.lang_name),
        key="lang_select",
        help="Pick a language. Voices update automatically."
    )
    st.session_state.lang_name = lang_name
    st.markdown("</div>", unsafe_allow_html=True)
    return lang_name

def page_speak(st):
    lang_name = language_panel(st)
    lang_code = LANGUAGES[lang_name]
    voices = VOICE_LIBRARY_BY_LANG.get(lang_code, {})
    if not voices:
        voices = {f"Ongea {lang_name} (Meta MMS Base)": f"facebook/mms-tts-{lang_code}"}

    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        text = st.text_area(
            f"Enter text ({lang_name.split('—')[0].strip()}):",
            height=180,
            placeholder="Type something... then hit Speak."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        speak = c1.button("🔊 Speak", use_container_width=True)
        clear = c2.button("🧹 Clear", use_container_width=True)
        demo  = c3.button("✨ Demo", use_container_width=True)

    with colB:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        voice_name = st.selectbox("Voice / Model", list(voices.keys()), key="voice_select")
        speed = st.slider("Speed", 0.75, 1.50, 1.0, 0.05)
        pitch = st.slider("Pitch (semitones)", -4.0, 4.0, 0.0, 0.5)
        st.caption("Human punctuation pauses ON.")
        st.markdown("</div>", unsafe_allow_html=True)

    if demo:
        demos = {
            "swh": "Habari! Karibu kwenye Ongea. Hii ni sauti ya Kiswahili yenye ubora wa juu.",
            "amh": "ሰላም! ወደ ኦንጌአ እንኳን በደህና መጡ። ይህ የአማርኛ ድምፅ ሙከራ ነው።",
            "som": "Salaan! Ku soo dhawow Ongea. Tani waa tijaabo cod Soomaali ah.",
            "yor": "Báwo ni! Káàbọ̀ sí Ongea. Èyí jẹ́ àdánwò ohun Yorùbá.",
            "sna": "Mhoro! Mauya ku Ongea. Iyi inyaya yekuedza mutauro weShona.",
            "xho": "Molo! Wamkelekile ku Ongea. Le yisampuli yesiXhosa.",
            "afr": "Hallo! Welkom by Ongea. Hierdie is ’n Afrikaans stemtoets.",
            "lin": "Mbote! Boyei malamu na Ongea. Oyo ezali exemple ya mongongo ya Lingala.",
            "kon": "Mbote! Wiza malembe na Ongea. Yai kele kiteso ya mongongo ya Kikongo.",
        }
        text = demos.get(lang_code, "Hello from Ongea!")

    if clear:
        st.rerun()

    if speak:
        try:
            load_voice = get_voice_loader()
            model_id = voices[voice_name]

            with st.spinner(f"Loading {voice_name}..."):
                bundle = load_voice(model_id, lang_code)

            # ✅ If Amharic MMS tokenization fails, auto-fallback to SpeechT5 backup
            if lang_code == "amh" and bundle.get("kind") == "vits":
                try:
                    _ = _encode_inputs_vits(bundle["processor"], text)
                except Exception:
                    st.warning("Amharic MMS tokenizer failed here. Switching to SpeechT5 backup voice...")
                    backup_id = voices.get("Ongea Amharic (SpeechT5 Backup)")
                    bundle = load_voice(backup_id, lang_code)
                    model_id = backup_id

            with st.spinner("Generating speech (human pauses)..."):
                audio, sr = synthesize_human(bundle, text, model_id=model_id, lang_code=lang_code)
                audio = apply_tone(audio, sr, speed=speed, pitch_semitones=pitch)

                out_wav = OUTPUT_DIR / "app_outputs" / f"{lang_code}_speech.wav"
                write_wav(out_wav, audio, sr)

            st.success("Done!")
            st.audio(str(out_wav), format="audio/wav")
            st.download_button(
                "Download WAV",
                data=out_wav.read_bytes(),
                file_name=f"ongea_{lang_code}.wav",
                mime="audio/wav",
                use_container_width=True
            )
        except Exception as e:
            st.error(
                f"Could not load voice `{model_id}`.\n\n"
                f"Ongea tried the per-language repo and the MMS monorepo fallback.\n\n"
                f"Error: {e}"
            )
            st.exception(e)

def page_batch(st):
    lang_name = language_panel(st)
    lang_code = LANGUAGES[lang_name]
    voices = VOICE_LIBRARY_BY_LANG.get(lang_code, {})
    if not voices:
        voices = {f"Ongea {lang_name} (Meta MMS Base)": f"facebook/mms-tts-{lang_code}"}

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🎬 Batch Studio")
    lines = st.text_area("Lines (one sentence per line)", height=220)
    voice_name = st.selectbox("Voice", list(voices.keys()), key="batch_voice")
    speed = st.slider("Speed", 0.75, 1.50, 1.0, 0.05, key="batch_speed")
    pitch = st.slider("Pitch (semitones)", -4.0, 4.0, 0.0, 0.5, key="batch_pitch")
    go = st.button("Generate Batch", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if go:
        try:
            load_voice = get_voice_loader()
            model_id = voices[voice_name]
            with st.spinner(f"Loading {voice_name}..."):
                bundle = load_voice(model_id, lang_code)

            # ✅ Same Amharic fallback in batch
            if lang_code == "amh" and bundle.get("kind") == "vits":
                try:
                    some_text = next((l for l in lines.split("\n") if l.strip()), "")
                    _ = _encode_inputs_vits(bundle["processor"], some_text)
                except Exception:
                    st.warning("Amharic MMS tokenizer failed. Switching to SpeechT5 backup voice...")
                    backup_id = voices.get("Ongea Amharic (SpeechT5 Backup)")
                    bundle = load_voice(backup_id, lang_code)
                    model_id = backup_id

            outs = []
            for i, ln in enumerate([l for l in lines.split("\n") if l.strip()]):
                audio, sr = synthesize_human(bundle, ln, model_id=model_id, lang_code=lang_code)
                audio = apply_tone(audio, sr, speed=speed, pitch_semitones=pitch)
                p = OUTPUT_DIR / "app_outputs" / f"{lang_code}_batch_{i+1:02d}.wav"
                write_wav(p, audio, sr)
                outs.append(p)

            st.success(f"Generated {len(outs)} clips.")
            for p in outs:
                st.markdown(f"**{p.name}**")
                st.audio(str(p), format="audio/wav")
                st.download_button(
                    f"Download {p.name}",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="audio/wav"
                )
        except Exception as e:
            st.error(str(e))
            st.exception(e)

def page_train(st):
    lang_name = language_panel(st)
    lang_code = LANGUAGES[lang_name]
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🧪 Fine-tuning (Local)")
    st.code(f"python app.py --train --lang {lang_code}")
    st.write(f"Outputs go to: `{OUTPUT_DIR / lang_code}`")
    st.markdown("</div>", unsafe_allow_html=True)

def page_about(st):
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("About Ongea v5.3")
    st.write(
        "Ongea is a Pan-African Text-to-Speech studio built on Meta MMS + community voices.\n\n"
        "**Upgrades:**\n"
        "• 9 African languages\n"
        "• Robust MMS loader with monorepo fallback\n"
        "• Human-like punctuation pauses\n"
        "• SpeechT5 Amharic backup for Windows MMS tokenizer issues\n"
        "• Failsafe UI (no whitescreen)\n\n"
        "No assets required — models download automatically."
    )
    st.markdown("</div>", unsafe_allow_html=True)

def run_app():
    try:
        import streamlit as st
        st.set_page_config(page_title="Ongea — Pan-African TTS", layout="centered")

        _init_state(st)

        # ✅ Render visible UI FIRST (before theme)
        st.markdown("## ✅ Ongea is loading...")
        st.caption("If you see a dark/blank screen, refresh the page or try disabling JavaScript console.")

        # Try to inject theme, but don't crash if it fails
        try:
            inject_theme(st)
        except Exception as e:
            st.warning(f"⚠️ Theme not applied (non-critical): {str(e)[:100]}")

        # Try header, but ensure nav still works if it fails
        try:
            header_block(st)
        except Exception as e:
            st.warning(f"⚠️ Header error: {str(e)[:100]}")

        try:
            render_nav(st)
        except Exception as e:
            st.warning(f"⚠️ Navigation error: {str(e)[:100]}")

        # Main page content with error handling
        try:
            page = st.session_state.page
            if page == "Speak":
                page_speak(st)
            elif page == "Batch":
                page_batch(st)
            elif page == "Fine-tune":
                page_train(st)
            else:
                page_about(st)
        except Exception as e:
            st.error(f"❌ Page rendering error: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

        st.caption("Ongea • Pan-African Text-to-Speech • Meta MMS + Community Voices")

    except Exception as e:
        import streamlit as st
        st.error("🔥 Critical app initialization error:")
        import traceback
        st.text(traceback.format_exc())


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
