# app.py
"""
ONGEA v8.2 — Modern Glass TTS Studio (Light + Dark) • Speak/Batch Toggle • Sidebar History (Menu Toggle)
- Clean, modern UI (mobile-style glass) with big text area + settings on the side
- Speak/Batch toggled (no tabs), no Demo/Clear buttons
- One Menu dropdown (top-right) controls: Studio / Fine-tune / About / Theme / Sidebar
- Fixes duplicate slider IDs (unique keys) + removes stray st.code HTML blocks (and hard-hides code blocks just in case)
- Smaller Language / Voice inputs (styled)
- Sidebar can be expanded even when empty (shows “No speech yet”)

Run:
    streamlit run app.py

Train:
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
STRIP_PUNCT = False


# =========================
# VOICE LIBRARY
# =========================

VOICE_LIBRARY_BY_LANG: Dict[str, Dict[str, str]] = {
    "swh": {
        "Ongea Swahili Male / Neutral (Meta Base)": "facebook/mms-tts-swh",
        "Ongea Swahili Female (Mozilla Lady)": "Benjamin-png/swahili-mms-tts-mozilla-lady-voice-finetuned",
        "Ongea Swahili Studio (Fine-tuned)": "Benjamin-png/swahili-mms-tts-finetuned",
        "Ongea Swahili Narrator (OpenBible)": "bookbot/vits-base-sw-KE-OpenBible",
        "Ongea Swahili SALAMA (Prosody-rich)": "EYEDOL/SALAMA_TTS",
    },
    "amh": {"Ongea Amharic (Meta MMS Base)": "facebook/mms-tts-amh"},
    "som": {"Ongea Somali (Meta MMS Base)": "facebook/mms-tts-som"},
    "yor": {"Ongea Yoruba (Meta MMS Base)": "facebook/mms-tts-yor"},
    "sna": {"Ongea Shona (Meta MMS Base)": "facebook/mms-tts-sna"},
    "xho": {"Ongea Xhosa (Meta MMS Base)": "facebook/mms-tts-xho"},
    "afr": {"Ongea Afrikaans (Meta MMS Base)": "facebook/mms-tts-afr"},
    "lin": {"Ongea Lingala (Meta MMS Base)": "facebook/mms-tts-lin"},
    "kon": {"Ongea Kongo (Meta MMS Base)": "facebook/mms-tts-kon"},
    "luo": {
        "Ongea Luo (CLEAR YourTTS, HF/Coqui)": "coqui:CLEAR-Global/YourTTS-Luo",
        "Ongea Luo (CLEAR XTTS, HF/Coqui)": "coqui:CLEAR-Global/XTTS-Luo",
        "Ongea Luo (Meta MMS Base)": "facebook/mms-tts-luo",
    },
    "kik": {"Ongea Gikuyu (Meta MMS Base)": "facebook/mms-tts-kik"},
    "mer": {"Ongea Ameru/Meru (Meta MMS Base)": "facebook/mms-tts-mer"},
    "kam": {"Ongea Kamba (Meta MMS Base)": "facebook/mms-tts-kam"},
    "guz": {"Ongea Ekegusii/Kisii (Meta MMS Base)": "facebook/mms-tts-guz"},
    "luy": {"Ongea Luhya (Meta MMS Base)": "facebook/mms-tts-luy"},
    "kln": {"Ongea Kalenjin (Meta MMS Base)": "facebook/mms-tts-kln"},
    "mas": {"Ongea Maasai (Meta MMS Base)": "facebook/mms-tts-mas"},
    "dav": {"Ongea Taita/Dawida (Meta MMS Base)": "facebook/mms-tts-dav"},
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
                audio_col = c; break
    if text_col is None:
        for c in cols:
            if "text" in c or "transcript" in c or "sentence" in c:
                text_col = c; break
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
        ex[text_col] = clean_text(ex[text_col]); return ex

    ds_train = ds_train.map(_norm)
    if ds_eval is not None:
        ds_eval = ds_eval.map(_norm)

    def _keep(ex):
        a = ex[audio_col]
        if a is None or a.get("array") is None: return False
        dur = len(a["array"]) / a["sampling_rate"]
        if dur < MIN_AUDIO_SEC or dur > MAX_AUDIO_SEC: return False
        if ex[text_col] is None or ex[text_col].strip() == "": return False
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
        "python","convert_original_discriminator_checkpoint.py",
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
    run(["accelerate","launch",str(train_script),"--config",str(cfg_path)], cwd=FINETUNE_REPO)


# =========================
# TTS LOADING + SYNTHESIS
# =========================

BASE_MMS_REPO = "facebook/mms-tts"
COQUI_PREFIX = "coqui:"

@dataclass
class VoiceBundle:
    engine: str
    processor: Any = None
    model: Any = None
    sr: int = TARGET_SR
    model_id: str = ""
    lang_code: str = ""

def _get_model_classes():
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

def _maybe_unidecode(text: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(text)
    except Exception:
        return text

def _safe_load_hf_vits(model_id: str, lang_code: Optional[str] = None) -> VoiceBundle:
    import torch
    AutoProcessor, ModelClass = _get_model_classes()
    last_err = None

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = ModelClass.from_pretrained(model_id, low_cpu_mem_usage=False, device_map=None)
        if any(getattr(p, "is_meta", False) for p in model.parameters()):
            raise RuntimeError("Model loaded with meta tensors.")
        model.to("cpu"); model.eval()
        return VoiceBundle(engine="hf_vits", processor=processor, model=model,
                           sr=getattr(processor, "sampling_rate", TARGET_SR),
                           model_id=model_id, lang_code=lang_code or "")
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
            return VoiceBundle(engine="hf_vits", processor=processor, model=model,
                               sr=getattr(processor, "sampling_rate", TARGET_SR),
                               model_id=model_id, lang_code=lang_code)
        except Exception as e:
            last_err = e

    raise last_err

def _safe_load_coqui_hf(model_id: str) -> VoiceBundle:
    real_id = model_id[len(COQUI_PREFIX):].strip()
    try:
        from huggingface_hub import snapshot_download
        from TTS.api import TTS as CoquiTTS
    except Exception as e:
        raise RuntimeError("Coqui requested but not installed. pip install TTS huggingface_hub") from e

    local_dir = Path(snapshot_download(repo_id=real_id))
    cfgs = list(local_dir.rglob("config.json")) + list(local_dir.rglob("*config*.json"))
    if not cfgs:
        raise RuntimeError(f"No config.json found in {real_id}")
    config_path = str(cfgs[0])

    ckpts = []
    for ext in ("*.pth","*.pt","*.bin","*.safetensors"):
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

def synthesize_raw(bundle: VoiceBundle, text: str) -> Tuple[np.ndarray, int]:
    import numpy as np
    text = clean_text(text)
    if not text:
        raise ValueError("Empty text.")

    if bundle.engine == "coqui":
        try:
            wave = bundle.model.tts(text)
        except Exception:
            wave = bundle.model.tts(_maybe_unidecode(text))
        audio = _to_1d_float32(wave)
        sr = int(bundle.sr or TARGET_SR)
        if audio.size == 0:
            raise ValueError("Coqui returned empty audio.")
        m = float(np.max(np.abs(audio)))
        if m > 1.0: audio = audio / m
        return np.clip(audio, -1.0, 1.0), sr

    processor, model = bundle.processor, bundle.model
    inputs = _encode_inputs(processor, text)
    import torch
    with torch.no_grad():
        out = model(**inputs)

    wave = out["waveform"] if isinstance(out, dict) and "waveform" in out else getattr(out, "waveform", None) or out[0]
    audio = _to_1d_float32(wave)
    sr = int(getattr(processor, "sampling_rate", TARGET_SR))

    if audio.size == 0:
        raise ValueError("Model returned empty audio.")
    m = float(np.max(np.abs(audio)))
    if m > 1.0: audio = audio / m
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

def synthesize_human(bundle: VoiceBundle, text: str) -> Tuple[np.ndarray, int]:
    chunks = split_by_punctuation(text)
    if not chunks:
        raise ValueError("Empty text.")
    audios = []
    sr_final = bundle.sr or TARGET_SR
    pause = {",":0.18, ";":0.22, ":":0.22, ".":0.38, "!":0.42, "?":0.42, "…":0.55}

    for chunk_text, punct in chunks:
        a, sr = synthesize_raw(bundle, chunk_text)
        sr_final = sr_final or sr
        audios.append(a)
        dur = pause.get(punct, 0.0)
        if dur > 0:
            audios.append(np.zeros(int(sr * dur), dtype=np.float32))

    audio = np.concatenate(audios) if len(audios) > 1 else audios[0]
    m = float(np.max(np.abs(audio))) if audio.size else 1.0
    if m > 1.0: audio = audio / m
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
        if m > 1.0: y = y / m
        return np.clip(y, -1.0, 1.0)
    except Exception:
        return audio

def write_wav(path: Path, audio: np.ndarray, sr: int):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16", format="WAV")


# =========================
# STREAMLIT UI (Modern Glass)
# =========================

def _init_state(st):
    if "view" not in st.session_state:
        st.session_state.view = "studio"  # studio | finetune | about
    if "theme" not in st.session_state:
        st.session_state.theme = "light"  # light | dark
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = False

    if "lang_name" not in st.session_state:
        st.session_state.lang_name = list(LANGUAGES.keys())[0]
    if "voice_name" not in st.session_state:
        st.session_state.voice_name = None

    if "mode" not in st.session_state:
        st.session_state.mode = "Speak"  # Speak | Batch

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
        voices = {f"Ongea {lang_name.split('—')[0].strip()} (Meta MMS Base)": f"facebook/mms-tts-{lang_code}"}
    return voices

def get_voice_loader():
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def load_voice(model_id: str, lang_code: str):
        return _safe_load_model(model_id, lang_code=lang_code)
    return load_voice

def show_overlay(st, text="Crafting voice..."):
    ph = st.empty()
    ph.markdown(
        f"""
<div class="oge-overlay">
  <div class="oge-overlay-card">
    <div class="oge-dots"><span></span><span></span><span></span></div>
    <div class="oge-overlay-txt">{text}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )
    return ph

def hide_overlay(ph):
    try:
        ph.empty()
    except Exception:
        pass

def inject_css(st, theme: str, sidebar_open: bool):
    # Hide ALL code blocks (kills the stray <div class="oge-brand"> snippet if anything outputs it)
    sidebar_css = ""
    if not sidebar_open:
        sidebar_css = """
<style>
  [data-testid="stSidebar"]{transform: translateX(-110%) !important; width:0 !important; min-width:0 !important;}
</style>
"""

    st.markdown(f"""
<style>
/* ---- Streamlit chrome off ---- */
#MainMenu, footer, header {{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none !important;}}
[data-testid="stStatusWidget"]{{display:none !important;}}
[data-testid="stHeader"]{{display:none !important;}}
[data-testid="stDecoration"]{{display:none !important;}}

/* HARD KILL CODE BLOCKS (fix stray debug) */
[data-testid="stCodeBlock"]{{display:none !important;}}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root{{
  --radius: 22px;
  --radius2: 18px;
  --shadow: 0 22px 60px rgba(12, 24, 48, 0.12);
  --shadow2: 0 12px 28px rgba(12, 24, 48, 0.10);
  --stroke: rgba(255,255,255,0.55);
  --stroke2: rgba(14, 20, 40, 0.10);

  --accent1: #19b6ad; /* teal */
  --accent2: #6b66ff; /* indigo */
  --accent3: #ff4d7d; /* pink */
  --ok: #14b8a6;
  --warn: #f59e0b;

  --font: "Plus Jakarta Sans", Inter, system-ui, -apple-system, Segoe UI, Roboto;
  --font2: Inter, system-ui, -apple-system, Segoe UI;
}}

{"/* LIGHT THEME */" if theme=="light" else "/* DARK THEME */"}
:root{{
  --bgA: {"#dff7fb" if theme=="light" else "#050814"};
  --bgB: {"#eef4ff" if theme=="light" else "#0b1633"};
  --bgC: {"#f6ecff" if theme=="light" else "#052f2a"};

  --card: {"rgba(255,255,255,0.66)" if theme=="light" else "rgba(12,16,32,0.62)"};
  --card2: {"rgba(255,255,255,0.80)" if theme=="light" else "rgba(12,16,32,0.78)"};
  --input: {"rgba(255,255,255,0.86)" if theme=="light" else "rgba(10,12,24,0.78)"};

  --text: {"#0b1220" if theme=="light" else "#eaf2ff"};
  --muted: {"rgba(11,18,32,0.62)" if theme=="light" else "rgba(234,242,255,0.70)"};
  --soft: {"rgba(11,18,32,0.10)" if theme=="light" else "rgba(234,242,255,0.10)"};
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
/* sidebar scroll allowed */
[data-testid="stSidebar"]{{height:100vh !important;}}
[data-testid="stSidebar"] > div{{height:100vh !important; overflow:auto !important;}}

/* ---- Typography ---- */
h1,h2,h3,h4,h5,h6,p,span,div{{ color: var(--text) !important; }}
.oge-muted{{ color: var(--muted) !important; }}

/* ---- Cards ---- */
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

/* ---- Header ---- */
.oge-header{{
  display:flex; align-items:center; justify-content:space-between;
  gap: 1rem;
  margin-bottom: 1.0rem;
}}
.oge-brand{{
  display:flex; align-items:center; gap: 0.9rem;
}}
.oge-logo{{
  width: 52px; height: 52px; border-radius: 18px;
  background: linear-gradient(135deg, rgba(25,182,173,0.35), rgba(107,102,255,0.25));
  border: 1px solid var(--soft);
  display:flex; align-items:center; justify-content:center;
  box-shadow: var(--shadow2);
}}
.oge-title{{
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.0;
}}
.oge-sub{{
  font-family: var(--font2);
  font-size: 0.98rem;
  margin-top: 0.2rem;
  color: var(--muted);
}}

.oge-menu-btn .stButton>button{{
  border-radius: 999px !important;
  border: 1px solid var(--soft) !important;
  background: var(--card2) !important;
  padding: 0.55rem 1.0rem !important;
  font-weight: 800 !important;
  box-shadow: var(--shadow2);
}}
.oge-menu-btn .stButton>button:hover{{ transform: translateY(-1px); }}

/* ---- Inputs (smaller selectboxes) ---- */
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
/* placeholder contrast */
textarea::placeholder{{ color: rgba(120,130,150,0.70) !important; }}

/* ---- Toggle (Speak/Batch) ---- */
.oge-toggle-wrap{{
  background: var(--card2);
  border: 1px solid var(--soft);
  border-radius: 999px;
  padding: 0.25rem;
  display:inline-flex;
  gap: 0.25rem;
  box-shadow: var(--shadow2);
}}
/* style streamlit radio group as pills */
div[role="radiogroup"]{{ gap: 0.35rem !important; }}
div[role="radiogroup"] > label{{
  background: transparent !important;
  border-radius: 999px !important;
  padding: 0.35rem 0.8rem !important;
  border: 1px solid transparent !important;
}}
div[role="radiogroup"] > label:hover{{ background: rgba(25,182,173,0.10) !important; }}
/* hide the actual circle */
div[role="radiogroup"] > label input{{ display:none !important; }}
/* selected pill */
div[role="radiogroup"] > label:has(input:checked){{
  background: linear-gradient(135deg, rgba(25,182,173,0.22), rgba(107,102,255,0.16)) !important;
  border: 1px solid rgba(25,182,173,0.40) !important;
}}
div[role="radiogroup"] > label span{{ font-weight: 850 !important; }}

/* ---- Buttons ---- */
.stButton>button{{
  border-radius: 999px !important;
  border: 1px solid rgba(25,182,173,0.45) !important;
  background: linear-gradient(135deg, rgba(25,182,173,0.22), rgba(107,102,255,0.16)) !important;
  padding: 0.7rem 1.05rem !important;
  font-weight: 900 !important;
  box-shadow: var(--shadow2);
}}
.stButton>button:hover{{ transform: translateY(-1px) scale(1.01); }}

/* ---- Sliders ---- */
[data-testid="stSlider"] > div{{ padding-top: 0.2rem; }}
/* audio player */
audio {{
  width: 100% !important;
  border-radius: 999px !important;
}}

/* ---- Overlay ---- */
.oge-overlay{{
  position: fixed; inset: 0;
  display:flex; align-items:flex-end; justify-content:flex-end;
  padding: 1.2rem;
  z-index: 9999;
}}
.oge-overlay-card{{
  background: var(--card2);
  border: 1px solid var(--soft);
  border-radius: 18px;
  padding: 0.85rem 0.95rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(14px);
  width: 280px;
}}
.oge-dots{{ display:flex; gap:8px; margin-bottom: 0.4rem; }}
.oge-dots span{{
  width: 8px; height: 8px; border-radius: 999px;
  background: rgba(25,182,173,0.9);
  animation: oge-bounce 0.8s infinite ease-in-out;
}}
.oge-dots span:nth-child(2){{ animation-delay: 0.12s; background: rgba(107,102,255,0.85); }}
.oge-dots span:nth-child(3){{ animation-delay: 0.24s; background: rgba(255,77,125,0.75); }}
@keyframes oge-bounce{{ 0%,100%{{ transform: translateY(0); opacity:.65; }} 50%{{ transform: translateY(-6px); opacity:1; }} }}
.oge-overlay-txt{{ color: var(--muted); font-weight: 700; }}

/* spacing fixes: remove big empty gaps from markdown */
[data-testid="stMarkdownContainer"] p{{ margin: 0.2rem 0 0.35rem 0; }}

</style>
{sidebar_css}
""", unsafe_allow_html=True)


def render_sidebar(st):
    sb = st.sidebar
    sb.markdown("<div style='padding:0.6rem 0.2rem;'><div style='font-weight:900;font-size:1.05rem;'>History</div><div style='color:rgba(90,100,120,0.75);font-size:0.9rem;margin-top:0.15rem;'>Your generated clips appear here.</div></div>", unsafe_allow_html=True)
    sb.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    if not st.session_state.history:
        sb.markdown("<div style='padding:0.9rem;border-radius:16px;border:1px solid rgba(0,0,0,0.06);background:rgba(255,255,255,0.55);backdrop-filter:blur(10px);'>No speech yet.</div>", unsafe_allow_html=True)
        sb.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
        sb.markdown("<div style='padding:0.9rem;border-radius:16px;border:1px solid rgba(0,0,0,0.06);background:rgba(255,255,255,0.55);backdrop-filter:blur(10px);'>Tip: Keep each sentence short for clean results.</div>", unsafe_allow_html=True)
        return

    for item in reversed(st.session_state.history[-30:]):
        sb.markdown("<div style='padding:0.85rem;border-radius:16px;border:1px solid rgba(0,0,0,0.06);background:rgba(255,255,255,0.55);backdrop-filter:blur(10px);margin-bottom:0.7rem;'>", unsafe_allow_html=True)
        sb.markdown(
            f"<div style='font-weight:900;margin-bottom:0.2rem;'>{item['label']}</div>"
            f"<div style='color:rgba(90,100,120,0.75);font-size:0.86rem;margin-bottom:0.45rem;'>{item['ts']} • {item['voice_name']}</div>",
            unsafe_allow_html=True
        )
        sb.audio(item["wav_path"], format="audio/wav")
        sb.download_button(
            "Download WAV",
            data=Path(item["wav_path"]).read_bytes(),
            file_name=Path(item["wav_path"]).name,
            mime="audio/wav",
            use_container_width=True,
            key=f"sb_dl_{item['id']}"
        )
        sb.markdown("</div>", unsafe_allow_html=True)

    if sb.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest_idx = None
        st.rerun()


def header_bar(st):
    # Left brand
    st.markdown("""
<div class="oge-header">
  <div class="oge-brand">
    <div class="oge-logo">🎙️</div>
    <div>
      <div class="oge-title">Ongea</div>
      <div class="oge-sub">African TTS Studio</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Right menu (popover)
    cL, cR = st.columns([1, 0.33], gap="small")
    with cR:
        st.markdown('<div class="oge-menu-btn">', unsafe_allow_html=True)
        pop = st.popover("☰ Menu")
        st.markdown('</div>', unsafe_allow_html=True)

    with pop:
        st.markdown("<div style='font-weight:900;font-size:1.02rem;margin-bottom:0.35rem;'>Ongea Menu</div>", unsafe_allow_html=True)

        view = st.radio(
            "View",
            ["studio", "finetune", "about"],
            index=["studio","finetune","about"].index(st.session_state.view),
            key="menu_view",
            label_visibility="collapsed",
        )
        st.session_state.view = view

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        th = st.radio(
            "Theme",
            ["light", "dark"],
            index=["light","dark"].index(st.session_state.theme),
            key="menu_theme",
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.theme = th

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        if st.button(("Hide sidebar" if st.session_state.sidebar_open else "Show sidebar"), use_container_width=True):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()


def studio_top_controls(st):
    lang_keys = list(LANGUAGES.keys())

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        st.markdown("<div style='font-weight:900;margin-bottom:0.25rem;'>Language</div>", unsafe_allow_html=True)
        lang_name = st.selectbox(
            "Language",
            lang_keys,
            index=lang_keys.index(st.session_state.lang_name),
            key="lang_select",
            label_visibility="collapsed",
        )
        st.session_state.lang_name = lang_name

    lang_code = LANGUAGES[st.session_state.lang_name]
    voices = get_voices_for(lang_code, st.session_state.lang_name)
    voice_names = list(voices.keys())

    if st.session_state.voice_name not in voice_names:
        st.session_state.voice_name = voice_names[0]

    with c2:
        st.markdown("<div style='font-weight:900;margin-bottom:0.25rem;'>Voice / Model</div>", unsafe_allow_html=True)
        voice_name = st.selectbox(
            "Voice / Model",
            voice_names,
            index=voice_names.index(st.session_state.voice_name),
            key="voice_select",
            label_visibility="collapsed",
        )
        st.session_state.voice_name = voice_name

    # Speak/Batch toggle
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='oge-toggle-wrap'>", unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["Speak", "Batch"],
        index=0 if st.session_state.mode == "Speak" else 1,
        key="mode_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.mode = mode

    return lang_code, voices


def studio_layout(st, lang_code: str, voices: Dict[str, str]):
    # Two-column layout: left big text, right settings
    left, right = st.columns([2.25, 1.0], gap="large")

    # Shared settings (single sliders => no duplicate ID errors)
    # unique keys fixed:
    with right:
        st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:900;font-size:1.25rem;margin-bottom:0.15rem;'>Settings</div>", unsafe_allow_html=True)
        st.markdown("<div class='oge-muted' style='font-size:0.95rem;margin-bottom:0.6rem;'>Tone + output controls</div>", unsafe_allow_html=True)

        st.session_state.speed = st.slider(
            "Speed",
            0.75, 1.50,
            float(st.session_state.speed),
            0.05,
            key="slider_speed_global",
        )
        st.session_state.pitch = st.slider(
            "Pitch (semitones)",
            -4.0, 4.0,
            float(st.session_state.pitch),
            0.5,
            key="slider_pitch_global",
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    if st.session_state.mode == "Speak":
        with left:
            st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
            st.markdown("<div style='font-weight:900;font-size:1.25rem;margin-bottom:0.2rem;'>Speak</div>", unsafe_allow_html=True)
            st.markdown("<div class='oge-muted' style='margin-bottom:0.75rem;'>Paste text → generate one clean clip.</div>", unsafe_allow_html=True)
            txt = st.text_area(
                "Text",
                value=st.session_state.speak_text,
                height=260,
                placeholder="Example:\nHabari! Karibu kwenye Ongea.\nAndika maandishi yako hapa…",
                key="speak_text_area",
                label_visibility="collapsed",
            )
            st.session_state.speak_text = txt
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            btn = st.button("Generate speech", use_container_width=True, key="btn_generate_speak")
            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

            st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
            st.markdown("<div style='font-weight:900;font-size:1.15rem;margin-bottom:0.2rem;'>Latest</div>", unsafe_allow_html=True)
            st.markdown("<div class='oge-muted' style='margin-bottom:0.6rem;'>Your most recent clip appears here.</div>", unsafe_allow_html=True)

            if st.session_state.latest_idx is not None:
                it = st.session_state.history[st.session_state.latest_idx]
                st.audio(it["wav_path"], format="audio/wav")
                st.download_button(
                    "Download WAV",
                    data=Path(it["wav_path"]).read_bytes(),
                    file_name=Path(it["wav_path"]).name,
                    mime="audio/wav",
                    use_container_width=True,
                    key="latest_dl_btn",
                )
            else:
                st.markdown("<div class='oge-muted'>Nothing yet.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if btn:
            _do_generate_speak(st, lang_code, voices)

    else:
        with left:
            st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
            st.markdown("<div style='font-weight:900;font-size:1.25rem;margin-bottom:0.2rem;'>Batch Studio</div>", unsafe_allow_html=True)
            st.markdown("<div class='oge-muted' style='margin-bottom:0.75rem;'>One line = one clip. Paste everything here.</div>", unsafe_allow_html=True)
            lines = st.text_area(
                "Lines",
                value=st.session_state.batch_lines,
                height=310,
                placeholder="Line 1: Habari! Karibu kwenye Ongea.\nLine 2: Leo tutaongea kuhusu...\nLine 3: Asante kwa kusikiliza.",
                key="batch_lines_area",
                label_visibility="collapsed",
            )
            st.session_state.batch_lines = lines
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            btn = st.button("Generate batch", use_container_width=True, key="btn_generate_batch")
            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

            st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
            st.markdown("<div style='font-weight:900;font-size:1.05rem;margin-bottom:0.25rem;'>Tips</div>", unsafe_allow_html=True)
            st.markdown("""
<div class="oge-muted" style="line-height:1.35;">
• Keep each line short (best quality)<br/>
• Use punctuation for natural pauses<br/>
• Batch outputs are saved as WAV<br/>
</div>
""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if btn:
            _do_generate_batch(st, lang_code, voices)

        # show last few generated batch clips inline (nice)
        if st.session_state.history:
            st.markdown("<div style='height:1.0rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-weight:900;font-size:1.1rem;margin-bottom:0.4rem;'>Recent clips</div>", unsafe_allow_html=True)
            cols = st.columns(3, gap="medium")
            recent = list(reversed(st.session_state.history[-6:]))
            for i, it in enumerate(recent):
                with cols[i % 3]:
                    st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
                    st.markdown(f"<div style='font-weight:900;margin-bottom:0.3rem;'>{it['label']}</div>", unsafe_allow_html=True)
                    st.audio(it["wav_path"], format="audio/wav")
                    st.download_button(
                        "Download",
                        data=Path(it["wav_path"]).read_bytes(),
                        file_name=Path(it["wav_path"]).name,
                        mime="audio/wav",
                        use_container_width=True,
                        key=f"recent_dl_{it['id']}",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)


def _do_generate_speak(st, lang_code: str, voices: Dict[str, str]):
    try:
        text = st.session_state.speak_text.strip()
        if not text:
            st.warning("Paste some text first.")
            return

        voice_name = st.session_state.voice_name
        model_id = voices[voice_name]
        load_voice = get_voice_loader()

        overlay = show_overlay(st, "Loading voice model…")
        bundle = load_voice(model_id, lang_code)
        hide_overlay(overlay)

        overlay = show_overlay(st, "Generating speech…")
        audio, sr = synthesize_human(bundle, text)
        audio = apply_tone(audio, sr, speed=st.session_state.speed, pitch_semitones=st.session_state.pitch)

        out_wav = OUTPUT_DIR / "app_outputs" / f"{lang_code}_speech_{len(st.session_state.history)+1:03d}.wav"
        write_wav(out_wav, audio, sr)
        hide_overlay(overlay)

        item = {
            "id": f"{lang_code}_s_{len(st.session_state.history)+1:03d}",
            "ts": datetime.now().strftime("%H:%M:%S"),
            "lang_code": lang_code,
            "voice_name": voice_name,
            "label": "Speak",
            "wav_path": str(out_wav),
            "sr": sr,
        }
        st.session_state.history.append(item)
        st.session_state.latest_idx = len(st.session_state.history) - 1
        st.rerun()

    except Exception as e:
        try:
            hide_overlay(overlay)
        except Exception:
            pass
        st.error(f"Could not generate speech: {e}")
        st.exception(e)

def _do_generate_batch(st, lang_code: str, voices: Dict[str, str]):
    try:
        raw = st.session_state.batch_lines
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if not lines:
            st.warning("Paste one sentence per line first.")
            return

        voice_name = st.session_state.voice_name
        model_id = voices[voice_name]
        load_voice = get_voice_loader()

        overlay = show_overlay(st, "Loading voice model…")
        bundle = load_voice(model_id, lang_code)
        hide_overlay(overlay)

        overlay = show_overlay(st, "Generating batch…")
        outs = []
        for i, ln in enumerate(lines, start=1):
            audio, sr = synthesize_human(bundle, ln)
            audio = apply_tone(audio, sr, speed=st.session_state.speed, pitch_semitones=st.session_state.pitch)
            p = OUTPUT_DIR / "app_outputs" / f"{lang_code}_batch_{len(st.session_state.history)+1:03d}_{i:02d}.wav"
            write_wav(p, audio, sr)
            outs.append(p)

            item = {
                "id": f"{lang_code}_b_{len(st.session_state.history)+1:03d}_{i:02d}",
                "ts": datetime.now().strftime("%H:%M:%S"),
                "lang_code": lang_code,
                "voice_name": voice_name,
                "label": f"Batch • Line {i}",
                "wav_path": str(p),
                "sr": sr,
            }
            st.session_state.history.append(item)

        hide_overlay(overlay)
        st.success(f"Generated {len(outs)} clips.")
        st.session_state.latest_idx = len(st.session_state.history) - 1
        st.rerun()

    except Exception as e:
        try:
            hide_overlay(overlay)
        except Exception:
            pass
        st.error(f"Batch failed: {e}")
        st.exception(e)


def finetune_view(st, lang_code: str):
    st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
    st.markdown("<div style='font-weight:900;font-size:1.35rem;margin-bottom:0.25rem;'>Fine-tuning (Local)</div>", unsafe_allow_html=True)
    st.markdown("<div class='oge-muted' style='margin-bottom:0.75rem;'>Run training locally using your HF dataset + finetune-hf-vits.</div>", unsafe_allow_html=True)
    st.markdown("**Command:**")
    st.code(f"python app.py --train --lang {lang_code}")  # harmless; code blocks hidden by CSS to avoid stray debug look
    st.markdown(f"**Outputs:** `{OUTPUT_DIR / lang_code}`")
    st.markdown("</div>", unsafe_allow_html=True)

def about_view(st):
    st.markdown('<div class="oge-card oge-pad">', unsafe_allow_html=True)
    st.markdown("<div style='font-weight:900;font-size:1.35rem;margin-bottom:0.25rem;'>About</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="oge-muted" style="line-height:1.45;">
Ongea is a clean text-to-speech studio for African voices.
<br/><br/>
<b>What it does</b><br/>
• Speak: generate one clean clip from pasted text<br/>
• Batch: generate one WAV per line (studio workflow)<br/>
• Tone: speed + pitch controls applied at export<br/>
• History: all generated clips saved during the session<br/><br/>
<b>Engines</b><br/>
• Meta MMS (HF VITS) + community voices<br/>
• Optional Coqui voices via <code>coqui:</code> prefix
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def run_app():
    import streamlit as st
    st.set_page_config(page_title="Ongea", page_icon="🎙️", layout="wide", initial_sidebar_state="collapsed")
    _init_state(st)

    inject_css(st, st.session_state.theme, st.session_state.sidebar_open)

    if st.session_state.sidebar_open:
        render_sidebar(st)

    header_bar(st)

    # Studio always shows language/voice controls at top (even in about/fine-tune)
    lang_code, voices = studio_top_controls(st)

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    if st.session_state.view == "studio":
        studio_layout(st, lang_code, voices)
    elif st.session_state.view == "finetune":
        finetune_view(st, lang_code)
    else:
        about_view(st)


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
