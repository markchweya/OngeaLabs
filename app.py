# app.py
"""
ONGEA LABS — ChatGPT-style TTS Studio (Light + Dark) • Sidebar Chats • Bottom Composer
Fix in this version:
- ✅ No more StreamlitDuplicateElementKey: settings widgets have unique keys per location (sidebar vs top popover)
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

PROJECT_NAME = "ongea-chat-ui"
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
        "Ongea Labs Luo (CLEAR YourTTS, HF/Coqui)": "coqui:CLEAR-Global/YourTTS-Luo",
        "Ongea Labs Luo (CLEAR XTTS, HF/Coqui)": "coqui:CLEAR-Global/XTTS-Luo",
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
# STREAMLIT UI (ChatGPT-style)
# =========================

def _init_state(st):
    ss = st.session_state
    ss.setdefault("theme", "dark")  # dark | light
    ss.setdefault("sidebar_collapsed", False)

    ss.setdefault("view", "studio")  # studio | finetune | about
    ss.setdefault("mode", "Ongea")   # Ongea | Batch

    ss.setdefault("lang_name", list(LANGUAGES.keys())[0])
    ss.setdefault("voice_name", None)

    ss.setdefault("speed", 1.0)
    ss.setdefault("pitch", 0.0)

    ss.setdefault("chats", [])  # history
    ss.setdefault("active_chat_id", None)

    ss.setdefault("search_q", "")

def get_voices_for(lang_code: str, lang_name: str):
    voices = VOICE_LIBRARY_BY_LANG.get(lang_code, {})
    if not voices:
        voices = {f"Ongea Labs {lang_name.split('—')[0].strip()} (Meta MMS Base)": f"facebook/mms-tts-{lang_code}"}
    return voices

def get_voice_loader(st):
    @st.cache_resource(show_spinner=False)
    def load_voice(model_id: str, lang_code: str):
        return _safe_load_model(model_id, lang_code=lang_code)
    return load_voice

def _make_title(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "New chat"
    t = re.sub(r"\s+", " ", t)
    return (t[:36] + "…") if len(t) > 36 else t

def inject_css(st):
    theme = st.session_state.theme
    collapsed = st.session_state.sidebar_collapsed
    sbw = "84px" if collapsed else "320px"

    st.markdown(f"""
<style>
#MainMenu, footer, header {{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none !important;}}
[data-testid="stStatusWidget"]{{display:none !important;}}
[data-testid="stHeader"]{{display:none !important;}}
[data-testid="stDecoration"]{{display:none !important;}}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --sbw: {sbw};
  --radius: 14px;
  --accent: #10a37f;
  --accent2: rgba(16,163,127,0.18);
  --bg: {"#0f1115" if theme=="dark" else "#f7f7f8"};
  --panel: {"#16181d" if theme=="dark" else "#ffffff"};
  --panel2: {"#121418" if theme=="dark" else "#f3f4f6"};
  --border: {"rgba(255,255,255,0.08)" if theme=="dark" else "rgba(0,0,0,0.08)"};
  --text: {"#f3f4f6" if theme=="dark" else "#0f172a"};
  --muted: {"rgba(243,244,246,0.68)" if theme=="dark" else "rgba(15,23,42,0.60)"};
  --shadow: 0 10px 30px rgba(0,0,0,{"0.35" if theme=="dark" else "0.08"});
}}

html, body {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}}

[data-testid="stAppViewContainer"] {{
  background: var(--bg) !important;
}}

.block-container {{
  max-width: 1100px !important;
  padding-top: 1.3rem !important;
  padding-bottom: 5.5rem !important;
}}

section[data-testid="stSidebar"] {{ width: var(--sbw) !important; }}
section[data-testid="stSidebar"] > div {{
  width: var(--sbw) !important;
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebarUserContent"] {{
  padding: 1.0rem 0.85rem 1.0rem 0.85rem !important;
}}

{"" if not collapsed else """
.oge-hide-when-collapsed{display:none !important;}
.oge-only-when-collapsed{display:block !important;}
"""}
{"" if collapsed else """
.oge-only-when-collapsed{display:none !important;}
"""}

.stButton > button {{
  width: 100% !important;
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--panel2) !important;
  color: var(--text) !important;
  font-weight: 700 !important;
  padding: 0.62rem 0.75rem !important;
  box-shadow: none !important;
}}
.stButton > button:hover {{
  border-color: rgba(16,163,127,0.35) !important;
  background: rgba(16,163,127,0.10) !important;
}}
.stButton > button:active {{ transform: translateY(1px); }}

.oge-primary .stButton > button {{
  background: var(--accent2) !important;
  border-color: rgba(16,163,127,0.45) !important;
}}
.oge-primary .stButton > button:hover {{
  background: rgba(16,163,127,0.22) !important;
}}

.oge-brand {{
  display:flex; align-items:center; gap:0.65rem;
  padding: 0.15rem 0.1rem 0.85rem 0.1rem;
}}
.oge-logo {{
  width: 38px; height: 38px; border-radius: 12px;
  background: linear-gradient(135deg, rgba(16,163,127,0.25), rgba(16,163,127,0.08));
  border: 1px solid rgba(16,163,127,0.25);
  display:flex; align-items:center; justify-content:center;
}}
.oge-brand-title {{ font-size: 1.05rem; font-weight: 800; line-height: 1.05; }}
.oge-brand-sub {{ font-size: 0.86rem; color: var(--muted); }}

[data-testid="stTextInput"] input {{
  background: var(--panel2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
  padding: 0.55rem 0.7rem !important;
}}
[data-testid="stTextInput"] label {{ display:none !important; }}

.oge-nav div[role="radiogroup"] {{
  display:flex !important;
  gap: 0.45rem !important;
  flex-wrap: wrap !important;
}}
.oge-nav div[role="radiogroup"] > label {{
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  padding: 0.38rem 0.55rem !important;
  background: var(--panel2) !important;
  white-space: nowrap !important;
  margin: 0 !important;
}}
.oge-nav div[role="radiogroup"] > label:hover {{
  border-color: rgba(16,163,127,0.35) !important;
  background: rgba(16,163,127,0.10) !important;
}}
.oge-nav div[role="radiogroup"] > label input {{ display:none !important; }}
.oge-nav div[role="radiogroup"] > label:has(input:checked) {{
  background: var(--accent2) !important;
  border-color: rgba(16,163,127,0.45) !important;
}}
.oge-nav div[role="radiogroup"] > label span {{ font-weight: 800 !important; }}

.oge-chatlist div[role="radiogroup"] > label {{
  width: 100% !important;
  border-radius: 12px !important;
  padding: 0.55rem 0.65rem !important;
  background: transparent !important;
  border: 1px solid transparent !important;
}}
.oge-chatlist div[role="radiogroup"] > label:hover {{
  background: rgba(16,163,127,0.08) !important;
  border-color: rgba(16,163,127,0.18) !important;
}}
.oge-chatlist div[role="radiogroup"] > label:has(input:checked) {{
  background: rgba(16,163,127,0.12) !important;
  border-color: rgba(16,163,127,0.28) !important;
}}
.oge-chatlist div[role="radiogroup"] > label input {{ display:none !important; }}

.oge-row {{ display:flex; gap:0.6rem; }}
.oge-row > div {{ flex: 1 1 0; }}

.oge-hero {{ padding-top: 3.2rem; text-align:center; }}
.oge-hero h1 {{
  font-size: 3.0rem; font-weight: 900; letter-spacing: -0.03em;
  margin: 0 0 0.6rem 0;
}}
.oge-hero p {{ margin: 0; color: var(--muted); font-size: 1.05rem; }}

.oge-topbar {{
  display:flex; align-items:center; justify-content: space-between;
  gap: 1rem; margin: 0.1rem 0 1.0rem 0;
}}
.oge-pills {{ display:flex; gap: 0.6rem; flex-wrap: wrap; }}
.oge-pill {{
  border: 1px solid var(--border);
  background: var(--panel);
  padding: 0.45rem 0.8rem;
  border-radius: 999px;
  color: var(--text);
  font-weight: 800;
}}
.oge-pill small {{ color: var(--muted); font-weight: 700; margin-right: 0.3rem; }}

div[data-baseweb="select"] > div {{
  background: var(--panel2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}}
div[data-baseweb="select"] span, div[data-baseweb="select"] input {{
  color: var(--text) !important;
  font-weight: 700 !important;
}}

[data-testid="stChatMessage"] {{
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  background: var(--panel) !important;
  box-shadow: var(--shadow);
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
  margin: 0.15rem 0 0.35rem 0;
  line-height: 1.45;
}}

div[data-testid="stChatInput"] {{
  position: fixed;
  left: 0; right: 0; bottom: 0;
  padding: 0.95rem 0;
  background: linear-gradient(to top, rgba(0,0,0,{"0.55" if theme=="dark" else "0.06"}), transparent);
  z-index: 50;
}}
div[data-testid="stChatInput"] > div {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 1.0rem;
}}
div[data-testid="stChatInput"] textarea {{
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 16px !important;
  padding: 0.65rem 0.85rem !important;
  font-size: 1.02rem !important;
}}
div[data-testid="stChatInput"] textarea::placeholder {{
  color: rgba(148,163,184,{"0.65" if theme=="dark" else "0.9"}) !important;
}}
</style>
""", unsafe_allow_html=True)

def settings_ui(st, key_prefix: str):
    """
    key_prefix is REQUIRED so the same settings panel can appear in multiple places
    (sidebar popover + top-right popover) without duplicate widget keys.
    """
    ss = st.session_state

    st.markdown("**Language**")
    lang_keys = list(LANGUAGES.keys())
    ss.lang_name = st.selectbox(
        "Language",
        lang_keys,
        index=lang_keys.index(ss.lang_name) if ss.lang_name in lang_keys else 0,
        label_visibility="collapsed",
        key=f"{key_prefix}_lang",
    )
    lang_code = LANGUAGES[ss.lang_name]

    voices = get_voices_for(lang_code, ss.lang_name)
    voice_names = list(voices.keys())
    if ss.voice_name not in voice_names:
        ss.voice_name = voice_names[0]

    st.markdown("**Voice / Model**")
    ss.voice_name = st.selectbox(
        "Voice",
        voice_names,
        index=voice_names.index(ss.voice_name),
        label_visibility="collapsed",
        key=f"{key_prefix}_voice",
    )

    st.markdown("**Mode**")
    ss.mode = st.radio(
        "Mode",
        ["Ongea", "Batch"],
        index=0 if ss.mode == "Ongea" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key=f"{key_prefix}_mode",
    )

    st.markdown("**Tone**")
    ss.speed = st.slider("Speed", 0.75, 1.50, float(ss.speed), 0.05, key=f"{key_prefix}_speed")
    ss.pitch = st.slider("Pitch (semitones)", -4.0, 4.0, float(ss.pitch), 0.5, key=f"{key_prefix}_pitch")

def sidebar_ui(st):
    sb = st.sidebar
    ss = st.session_state

    c1, c2 = sb.columns([0.82, 0.18], gap="small")
    with c1:
        sb.markdown(
            """
<div class="oge-brand">
  <div class="oge-logo">🎙️</div>
  <div class="oge-hide-when-collapsed">
    <div class="oge-brand-title">Ongea Labs</div>
    <div class="oge-brand-sub">Chat TTS Studio</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        if sb.button("«" if not ss.sidebar_collapsed else "»", key="sb_collapse"):
            ss.sidebar_collapsed = not ss.sidebar_collapsed
            st.rerun()

    sb.markdown('<div class="oge-primary">', unsafe_allow_html=True)
    if sb.button("＋ New chat" if not ss.sidebar_collapsed else "＋", key="sb_new_chat"):
        ss.active_chat_id = None
        st.rerun()
    sb.markdown("</div>", unsafe_allow_html=True)

    if not ss.sidebar_collapsed:
        sb.markdown('<div class="oge-hide-when-collapsed">', unsafe_allow_html=True)
        ss.search_q = sb.text_input("Search chats", value=ss.search_q, placeholder="Search chats", key="sb_search")

        sb.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        sb.markdown('<div class="oge-nav">', unsafe_allow_html=True)
        view = sb.radio(
            "View",
            ["🟣 Studio", "🛠️ Fine-tune", "ℹ️ About"],
            index={"studio": 0, "finetune": 1, "about": 2}[ss.view],
            horizontal=True,
            label_visibility="collapsed",
            key="sb_view_radio",
        )
        sb.markdown("</div>", unsafe_allow_html=True)
        ss.view = {"🟣 Studio": "studio", "🛠️ Fine-tune": "finetune", "ℹ️ About": "about"}[view]
        sb.markdown("</div>", unsafe_allow_html=True)

    sb.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    if not ss.sidebar_collapsed:
        sb.markdown("<div class='oge-hide-when-collapsed' style='font-weight:900; color: var(--muted); font-size:0.9rem;'>Your chats</div>", unsafe_allow_html=True)

    chats = st.session_state.chats[:]
    q = (st.session_state.search_q or "").strip().lower()
    if q:
        chats = [c for c in chats if q in (c.get("title","").lower() + " " + (c.get("text","").lower()))]
    chats = list(reversed(chats))

    if chats:
        options = [f"{c['id']}|{c['title']}" for c in chats]
        active = st.session_state.active_chat_id
        idx = 0
        if active:
            for i, c in enumerate(chats):
                if c["id"] == active:
                    idx = i
                    break

        sb.markdown('<div class="oge-chatlist">', unsafe_allow_html=True)
        pick = sb.radio(
            "Chats",
            options,
            index=idx,
            format_func=lambda s: s.split("|",1)[1],
            label_visibility="collapsed",
            key="sb_chat_pick",
        )
        sb.markdown("</div>", unsafe_allow_html=True)
        st.session_state.active_chat_id = pick.split("|", 1)[0]
    else:
        if not ss.sidebar_collapsed:
            sb.markdown(
                "<div class='oge-hide-when-collapsed' style='color: var(--muted); padding:0.6rem 0.25rem;'>No chats yet.</div>",
                unsafe_allow_html=True,
            )

    sb.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

    if not ss.sidebar_collapsed:
        sb.markdown('<div class="oge-row oge-hide-when-collapsed">', unsafe_allow_html=True)
        b1, b2 = sb.columns(2, gap="small")
        with b1:
            if sb.button(("🌞 Theme" if ss.theme == "dark" else "🌚 Theme"), key="sb_theme_btn"):
                ss.theme = "light" if ss.theme == "dark" else "dark"
                st.rerun()
        with b2:
            with sb.popover("⚙️ Settings", use_container_width=True):
                settings_ui(st, key_prefix="sbset")
        sb.markdown("</div>", unsafe_allow_html=True)

def _generate_and_store(st, user_text: str):
    ss = st.session_state

    lang_code = LANGUAGES[ss.lang_name]
    voices = get_voices_for(lang_code, ss.lang_name)
    voice_name = ss.voice_name or list(voices.keys())[0]
    model_id = voices[voice_name]

    load_voice = get_voice_loader(st)

    is_batch = (ss.mode == "Batch")

    out_dir = OUTPUT_DIR / "app_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    chat_id = f"c_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    title = _make_title(user_text)

    outs: List[Path] = []
    sr = TARGET_SR

    with st.spinner("Generating speech…"):
        bundle = load_voice(model_id, lang_code)

        if not is_batch:
            audio, sr = synthesize_human(bundle, user_text)
            audio = apply_tone(audio, sr, speed=ss.speed, pitch_semitones=ss.pitch)
            p = out_dir / f"{lang_code}_{chat_id}_ongea.wav"
            write_wav(p, audio, sr)
            outs = [p]
        else:
            lines = [ln.strip() for ln in (user_text or "").splitlines() if ln.strip()]
            if not lines:
                raise ValueError("Batch mode: paste one sentence per line.")
            for i, ln in enumerate(lines, start=1):
                audio, sr = synthesize_human(bundle, ln)
                audio = apply_tone(audio, sr, speed=ss.speed, pitch_semitones=ss.pitch)
                p = out_dir / f"{lang_code}_{chat_id}_batch_{i:02d}.wav"
                write_wav(p, audio, sr)
                outs.append(p)

    chat = {
        "id": chat_id,
        "title": title,
        "ts": datetime.now().strftime("%H:%M:%S"),
        "mode": ss.mode,
        "lang_code": lang_code,
        "lang_name": ss.lang_name,
        "voice_name": voice_name,
        "model_id": model_id,
        "speed": float(ss.speed),
        "pitch": float(ss.pitch),
        "text": user_text,
        "outputs": [str(p) for p in outs],
        "sr": int(sr),
    }
    ss.chats.append(chat)
    ss.active_chat_id = chat_id

def main_view(st):
    ss = st.session_state

    left, right = st.columns([0.78, 0.22], gap="small")
    with left:
        st.markdown(
            f"""
<div class="oge-topbar">
  <div class="oge-pills">
    <div class="oge-pill"><small>Mode:</small> {ss.mode}</div>
    <div class="oge-pill"><small>View:</small> {ss.view}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        with st.popover("⚙️", use_container_width=True):
            settings_ui(st, key_prefix="topset")

    if ss.view == "finetune":
        st.markdown("## Fine-tuning (Local)")
        st.markdown(
            "Run training locally using your HF dataset + finetune-hf-vits.\n\n"
            f"**Command:** `python app.py --train --lang {LANGUAGES[ss.lang_name]}`\n\n"
            f"**Outputs:** `{OUTPUT_DIR / LANGUAGES[ss.lang_name]}`"
        )
        return

    if ss.view == "about":
        st.markdown("## About Ongea Labs")
        st.markdown(
            """
Ongea is a clean text-to-speech studio for African voices.

**What it does**
- **Ongea**: one clean clip from pasted text
- **Batch**: one WAV per line (studio workflow)
- **Tone**: speed + pitch controls applied at export
"""
        )
        return

    active = ss.active_chat_id
    chat = None
    if active:
        for c in ss.chats:
            if c["id"] == active:
                chat = c
                break

    if not chat:
        st.markdown(
            """
<div class="oge-hero">
  <h1>What’s on your mind today?</h1>
  <p>Type text → get natural speech audio. Use ⚙️ to choose language, voice, and tone.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.chat_message("user").markdown(chat["text"])
        with st.chat_message("assistant"):
            st.markdown(
                f"<div style='color:var(--muted); font-weight:700; margin-bottom:0.35rem;'>"
                f"{chat['lang_name']} • {chat['voice_name']} • Speed {chat['speed']:.2f} • Pitch {chat['pitch']:.1f}"
                f"</div>",
                unsafe_allow_html=True,
            )
            for i, wav in enumerate(chat["outputs"], start=1):
                p = Path(wav)
                if not p.exists():
                    st.error(f"Missing audio file: {wav}")
                    continue
                st.audio(str(p), format="audio/wav")
                st.download_button(
                    "Download WAV" if len(chat["outputs"]) == 1 else f"Download WAV #{i}",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="audio/wav",
                    use_container_width=True,
                    key=f"dl_{chat['id']}_{i}",
                )

def run_app():
    import streamlit as st

    st.set_page_config(
        page_title="Ongea Labs",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state(st)
    inject_css(st)

    sidebar_ui(st)
    main_view(st)

    placeholder = "Type text to speak…" if st.session_state.mode == "Ongea" else "Batch mode: paste multiple lines (one per line)…"
    user_text = st.chat_input(placeholder)

    if user_text is not None:
        user_text = (user_text or "").strip()
        if user_text:
            try:
                _generate_and_store(st, user_text)
                st.rerun()
            except Exception as e:
                st.error(f"Could not generate speech: {e}")
                st.exception(e)

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
