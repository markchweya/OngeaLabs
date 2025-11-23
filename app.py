# app.py
"""
ONGEA v7.0 — Calm One-Page TTS Studio
WELCOME → STUDIO FLOW • NO MAIN SCROLL • FLOATING TONE POPOVER • CUSTOM LIBRARY

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
        out_sr = int(getattr(tts.synthesizer,"output_sample_rate",TARGET_SR))
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
# STREAMLIT UI
# =========================

BASE_CSS = """
<style>
:root{
  --bg1:#050814; --bg2:#0B1633; --bg3:#052F2A;
  --accent:#00E0B8; --accent2:#F9C74F;
  --txt:#EAF2FF; --muted:rgba(234,242,255,0.70);
}

/* remove Streamlit chrome */
#MainMenu, footer, header {visibility:hidden;}
[data-testid="stToolbar"]{display:none !important;}
[data-testid="stStatusWidget"]{display:none !important;}
[data-testid="stHeader"]{display:none !important;}
[data-testid="stDecoration"]{display:none !important;}

/* NO SCROLL MAIN */
html, body {height:100% !important; overflow:hidden !important;}
[data-testid="stAppViewContainer"]{
  height:100vh !important; overflow:hidden !important;
  background: radial-gradient(1200px 800px at 10% 0%, #0a1230 0%, transparent 50%),
              radial-gradient(1000px 700px at 95% 20%, #063a33 0%, transparent 55%),
              linear-gradient(135deg,var(--bg1),var(--bg2),var(--bg3)) !important;
  color: var(--txt) !important;
}
[data-testid="stMain"]{
  height:100vh !important; overflow:hidden !important; background:transparent !important;
}
.block-container{
  height:100vh !important; overflow:hidden !important;
  max-width:1280px;
  padding:1.0rem 1.2rem 0.6rem 1.2rem !important;
}

/* ALLOW SIDEBAR SCROLL (library only) */
[data-testid="stSidebar"]{
  height:100vh !important;
  transition: transform .28s ease, width .28s ease;
  background: transparent !important;
}
[data-testid="stSidebar"] > div{
  height:100vh !important;
  overflow:auto !important;
  background: linear-gradient(180deg, rgba(7,12,30,0.98), rgba(7,25,22,0.98)) !important;
  border-right:1px solid rgba(255,255,255,0.06);
}

/* typography */
body, p, span, h1, h2, h3, h4, h5, h6{color:var(--txt) !important;}
.small-muted{color:var(--muted); font-size:0.95rem}

/* Calm inputs (no boxes around sections) */
textarea{
  background:rgba(8,12,25,0.96) !important;
  border:1px solid rgba(255,255,255,0.10) !important;
  color:var(--txt) !important;
  border-radius:12px !important;
  font-size:0.98rem !important;
}
div[data-baseweb="select"] > div{
  background:rgba(8,12,25,0.88) !important;
  border:1px solid rgba(255,255,255,.12) !important;
  border-radius:12px !important;
  color:var(--txt) !important;
}

/* buttons */
.stButton button{
  background:linear-gradient(135deg,rgba(0,224,184,0.18),rgba(249,199,79,0.10)) !important;
  border:1px solid rgba(0,224,184,0.65) !important;
  color:var(--txt) !important;
  border-radius:12px !important;
  font-weight:900 !important;
  padding:0.55rem 0.9rem !important;
  transition:all .18s ease !important;
  box-shadow:0 8px 20px rgba(0,0,0,0.22);
}
.stButton button:hover{transform:translateY(-2px) scale(1.02);}

/* custom tabs calm */
div[data-testid="stTabs"]{margin-top:0.5rem !important;}
div[data-testid="stTabs"] > div[role="tablist"]{
  gap:0.6rem; padding:0.15rem 0;
  background:transparent;
  border-bottom:1px solid rgba(255,255,255,0.08);
}
div[data-testid="stTabs"] > div[role="tablist"] button{
  background:transparent !important;
  color:var(--txt) !important;
  border:none !important;
  border-bottom:2px solid transparent !important;
  border-radius:0 !important;
  font-weight:900 !important;
  padding:0.4rem 0.25rem !important;
  transition:all .18s ease !important;
}
div[data-testid="stTabs"] > div[role="tablist"] button[aria-selected="true"]{
  border-bottom:2px solid var(--accent) !important;
  color:var(--accent) !important;
}
div[data-testid="stTabs"] > div[role="tabpanel"]{
  padding-top:0.7rem !important;
}

/* Crafting overlay ONLY */
#crafting-overlay{
  position: fixed; right: 1.4rem; bottom: 1.4rem;
  width: 250px;
  background: rgba(10, 16, 36, 0.88);
  border:1px solid rgba(255,255,255,0.10);
  border-radius:16px;
  padding:0.85rem 0.95rem;
  box-shadow:0 10px 34px rgba(0,0,0,0.48);
  z-index: 9999; backdrop-filter: blur(8px);
}
.dotbar{display:flex; gap:6px; align-items:flex-end; height:18px;}
.dotbar span{
  width:8px; border-radius:6px; background: rgba(0,224,184,0.9);
  animation:bounce 0.8s infinite ease-in-out;
}
.dotbar span:nth-child(1){height:6px; animation-delay:0.0s;}
.dotbar span:nth-child(2){height:8px; animation-delay:0.1s;}
.dotbar span:nth-child(3){height:10px; animation-delay:0.2s;}
.dotbar span:nth-child(4){height:14px; animation-delay:0.3s;}
.dotbar span:nth-child(5){height:18px; animation-delay:0.4s;}
.dotbar span:nth-child(6){height:16px; animation-delay:0.5s;}
.dotbar span:nth-child(7){height:12px; animation-delay:0.6s;}
@keyframes bounce{0%,100%{transform:translateY(0);opacity:.5;}50%{transform:translateY(-6px);opacity:1;}}
#crafting-text{margin-top:0.45rem;font-size:0.93rem;color:var(--muted);}

/* sidebar cards */
.sidebar-title{
  font-size:1.1rem; font-weight:900; letter-spacing:.02em; color:var(--accent);
  margin:0.7rem 0 0.25rem 0;
}
.sidebar-card{
  background: rgba(11, 22, 51, 0.85);
  border:1px solid rgba(255,255,255,0.10);
  border-radius:14px;
  padding:0.6rem 0.7rem;
  margin-bottom:0.55rem;
  box-shadow:0 6px 18px rgba(0,0,0,0.28);
}
.sidebar-meta{font-size:0.82rem;color:var(--muted);margin-bottom:0.35rem;}

/* Floating Library toggle */
#library-toggle-wrap{
  position:fixed;
  left:1.2rem;
  top:1.2rem;
  z-index:9998;
}
#library-toggle-wrap .stButton button{
  width:auto !important;
  padding:0.45rem 0.8rem !important;
  border-radius:999px !important;
  font-weight:900 !important;
  letter-spacing:.03em;
  background:rgba(0,224,184,0.12) !important;
  border:1px solid rgba(0,224,184,0.8) !important;
  box-shadow:0 10px 30px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}

/* Welcome hero */
#welcome-wrap{
  height:92vh; display:flex; align-items:center; justify-content:center;
}
#welcome-hero{
  text-align:center; max-width:780px;
}
#welcome-title{
  font-size:4.2rem; font-weight:1000; letter-spacing:0.02em; color:var(--accent);
}
#welcome-sub{
  margin-top:0.6rem; font-size:1.05rem; color:var(--muted);
}
#welcome-cta{
  margin-top:1.6rem;
}
</style>
"""

def inject_theme(st, sidebar_open: bool):
    sidebar_css = ""
    if not sidebar_open:
        sidebar_css = """
<style>
[data-testid="stSidebar"]{
  transform: translateX(-105%) !important;
  width:0 !important; min-width:0 !important;
}
</style>
"""
    st.markdown(BASE_CSS + sidebar_css, unsafe_allow_html=True)

def show_crafting_overlay(st, text="Crafting voice..."):
    ph = st.empty()
    ph.markdown(
        f"""
<div id="crafting-overlay">
  <div class="dotbar">
    <span></span><span></span><span></span><span></span><span></span><span></span><span></span>
  </div>
  <div id="crafting-text">{text}</div>
</div>
        """,
        unsafe_allow_html=True
    )
    return ph

def hide_overlay(ph):
    try: ph.empty()
    except Exception: pass

def get_voice_loader():
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def load_voice(model_id: str, lang_code: str):
        return _safe_load_model(model_id, lang_code=lang_code)
    return load_voice

def _init_state(st):
    if "mode" not in st.session_state:
        st.session_state.mode = "welcome"  # welcome | studio
    if "lang_name" not in st.session_state:
        st.session_state.lang_name = list(LANGUAGES.keys())[0]
    if "speak_text" not in st.session_state:
        st.session_state.speak_text = ""
    if "batch_lines" not in st.session_state:
        st.session_state.batch_lines = ""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest_idx" not in st.session_state:
        st.session_state.latest_idx = None
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True
    if "speak_speed" not in st.session_state:
        st.session_state.speak_speed = 1.0
    if "speak_pitch" not in st.session_state:
        st.session_state.speak_pitch = 0.0

def get_voices_for(lang_code: str, lang_name: str):
    voices = VOICE_LIBRARY_BY_LANG.get(lang_code, {})
    if not voices:
        voices = {f"Ongea {lang_name} (Meta MMS Base)": f"facebook/mms-tts-{lang_code}"}
    return voices


# ---------- SIDEBAR LIBRARY ----------

def render_sidebar_library(st):
    sb = st.sidebar
    sb.markdown("<div class='sidebar-title'>Library</div>", unsafe_allow_html=True)
    sb.caption("Generated speeches in this session.")

    if not st.session_state.history:
        sb.markdown("<div class='sidebar-card'>No speech yet.</div>", unsafe_allow_html=True)
        return

    for item in reversed(st.session_state.history):
        sb.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        sb.markdown(
            f"<div class='sidebar-meta'>"
            f"{item['ts']} • {item['lang_name'].split('—')[0].strip()}<br>"
            f"{item['voice_name']}"
            f"</div>",
            unsafe_allow_html=True
        )
        sb.audio(item["wav_path"], format="audio/wav")
        sb.download_button(
            "Download",
            data=Path(item["wav_path"]).read_bytes(),
            file_name=Path(item["wav_path"]).name,
            mime="audio/wav",
            use_container_width=True,
            key=f"sb_dl_{item['id']}"
        )
        sb.markdown("</div>", unsafe_allow_html=True)

    sb.markdown("---")
    if sb.button("🗑️ Clear Library", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest_idx = None
        st.rerun()


# ---------- WELCOME MODE ----------

def welcome_screen(st):
    st.markdown(
        """
<div id="welcome-wrap">
  <div id="welcome-hero">
    <div id="welcome-title">Ongea</div>
    <div id="welcome-sub">
      A calm studio for African voices. Type, choose a voice, and generate.
    </div>
    <div id="welcome-cta"></div>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )
    # center CTA using columns
    _, mid, _ = st.columns([1,1.2,1])
    with mid:
        if st.button("✨ Generate Speech", use_container_width=True, key="cta_generate"):
            st.session_state.mode = "studio"
            st.rerun()

    # tiny footer hint (no tagline)
    st.caption("")


# ---------- STUDIO TOP ROW ----------

def studio_top_row(st):
    c1, c2 = st.columns([1.25, 1.0], gap="small")
    with c1:
        lang_name = st.selectbox(
            "Language",
            list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.lang_name),
            key="lang_select",
        )
        st.session_state.lang_name = lang_name
    with c2:
        lang_code = LANGUAGES[lang_name]
        voices = get_voices_for(lang_code, lang_name)
        voice_name = st.selectbox("Voice / Model", list(voices.keys()), key="voice_select")
    return lang_name, LANGUAGES[lang_name], voice_name, voices


# ---------- SPEAK TAB ----------

def speak_tab(st, lang_name, lang_code, voice_name, voices):
    colA, colB = st.columns([2.1, 1.0], gap="small")

    with colA:
        text = st.text_area(
            f"Enter text ({lang_name.split('—')[0].strip()}):",
            height=140,
            placeholder="Type something... then hit Speak.",
            key="speak_text_area",
            value=st.session_state.speak_text
        )
        st.session_state.speak_text = text

        # Tone popover (floating)
        tone_pop = st.popover("🎛️ Tone")
        with tone_pop:
            st.session_state.speak_speed = st.slider(
                "Speed", 0.75, 1.50, st.session_state.speak_speed, 0.05
            )
            st.session_state.speak_pitch = st.slider(
                "Pitch (semitones)", -4.0, 4.0, st.session_state.speak_pitch, 0.5
            )
            st.caption("Tone applies on next Speak/Batch.")

        b1, b2, b3 = st.columns(3, gap="small")
        speak = b1.button("🔊 Speak", use_container_width=True, key="btn_speak")
        clear = b2.button("🧹 Clear", use_container_width=True, key="btn_clear")
        demo  = b3.button("✨ Demo", use_container_width=True, key="btn_demo")

    with colB:
        st.subheader("🎧 Latest Speech")
        if st.session_state.latest_idx is None:
            st.markdown("<div class='small-muted'>Generate speech and it appears here instantly.</div>", unsafe_allow_html=True)
        else:
            last = st.session_state.history[st.session_state.latest_idx]
            st.caption(f"{last['ts']} • {last['lang_name'].split('—')[0].strip()}")
            st.audio(last["wav_path"], format="audio/wav")
            st.download_button(
                "Download WAV",
                data=Path(last["wav_path"]).read_bytes(),
                file_name=Path(last["wav_path"]).name,
                mime="audio/wav",
                use_container_width=True,
                key="latest_dl"
            )

    if demo:
        demos = {
            "swh": "Habari! Karibu kwenye Ongea.",
            "amh": "ሰላም! ወደ ኦንጌአ እንኳን በደህና መጡ።",
            "som": "Salaan! Ku soo dhawow Ongea.",
            "yor": "Báwo ni! Káàbọ̀ sí Ongea.",
        }
        st.session_state.speak_text = demos.get(lang_code, "Hello from Ongea!")
        st.rerun()

    if clear:
        st.session_state.speak_text = ""
        st.rerun()

    if speak:
        try:
            load_voice = get_voice_loader()
            model_id = voices[voice_name]

            overlay = show_crafting_overlay(st, "Crafting voice... (loading model)")
            bundle = load_voice(model_id, lang_code)
            hide_overlay(overlay)

            overlay = show_crafting_overlay(st, "Crafting voice... (generating speech)")
            audio, sr = synthesize_human(bundle, st.session_state.speak_text)
            audio = apply_tone(
                audio, sr,
                speed=st.session_state.speak_speed,
                pitch_semitones=st.session_state.speak_pitch
            )
            out_wav = OUTPUT_DIR / "app_outputs" / f"{lang_code}_speech_{len(st.session_state.history)+1:03d}.wav"
            write_wav(out_wav, audio, sr)
            hide_overlay(overlay)

            item = {
                "id": f"{lang_code}_{len(st.session_state.history)+1:03d}",
                "ts": datetime.now().strftime("%H:%M:%S"),
                "lang_code": lang_code,
                "lang_name": lang_name,
                "voice_name": voice_name,
                "text": st.session_state.speak_text,
                "wav_path": str(out_wav),
                "sr": sr,
            }
            st.session_state.history.append(item)
            st.session_state.latest_idx = len(st.session_state.history) - 1
            st.rerun()

        except Exception as e:
            hide_overlay(overlay) if "overlay" in locals() else None
            st.error(f"Could not generate speech: {e}")
            st.exception(e)


# ---------- BATCH / FINETUNE / ABOUT ----------

def batch_tab(st, lang_name, lang_code):
    voices = get_voices_for(lang_code, lang_name)

    st.subheader("🎬 Batch Studio")
    lines = st.text_area(
        "Lines (one sentence per line)",
        height=140,
        key="batch_lines_area",
        value=st.session_state.batch_lines
    )
    st.session_state.batch_lines = lines
    voice_name = st.selectbox("Voice", list(voices.keys()), key="batch_voice")

    tone_pop = st.popover("🎛️ Tone for Batch")
    with tone_pop:
        speed = st.slider("Speed", 0.75, 1.50, st.session_state.speak_speed, 0.05, key="batch_speed")
        pitch = st.slider("Pitch", -4.0, 4.0, st.session_state.speak_pitch, 0.5, key="batch_pitch")

    bA, bB = st.columns(2, gap="small")
    go = bA.button("Generate Batch", use_container_width=True, key="btn_batch_go")
    clear = bB.button("Clear Batch", use_container_width=True, key="btn_batch_clear")

    if clear:
        st.session_state.batch_lines = ""
        st.rerun()

    if go:
        try:
            load_voice = get_voice_loader()
            model_id = voices[voice_name]

            overlay = show_crafting_overlay(st, "Crafting voice... (loading model)")
            bundle = load_voice(model_id, lang_code)
            hide_overlay(overlay)

            overlay = show_crafting_overlay(st, "Crafting voice... (batch synthesis)")
            outs = []
            for i, ln in enumerate([l for l in lines.split("\n") if l.strip()]):
                audio, sr = synthesize_human(bundle, ln)
                audio = apply_tone(audio, sr, speed=speed, pitch_semitones=pitch)
                p = OUTPUT_DIR / "app_outputs" / f"{lang_code}_batch_{i+1:02d}.wav"
                write_wav(p, audio, sr)
                outs.append(p)
            hide_overlay(overlay)

            st.success(f"Generated {len(outs)} clips.")
            for p in outs[:2]:
                st.markdown(f"**{p.name}**")
                st.audio(str(p), format="audio/wav")
        except Exception as e:
            st.error(str(e))
            st.exception(e)

def finetune_tab(st, lang_code):
    st.subheader("🧪 Fine-tuning (Local)")
    st.code(f"python app.py --train --lang {lang_code}")
    st.write(f"Outputs go to: `{OUTPUT_DIR / lang_code}`")

def about_tab(st):
    st.subheader("About")
    st.write(
        "Ongea is a calm text-to-speech studio using Meta MMS + community voices.\n\n"
        "• Multiple African languages + Kenya pack\n"
        "• Robust MMS loader (monorepo fallback)\n"
        "• Optional Coqui HF voices via `coqui:` prefix\n"
        "• Human-like punctuation pauses\n"
        "• One-page no-scroll main UI\n"
        "• Session Library in the sidebar"
    )


def studio_screen(st):
    # Floating Library toggle (always visible)
    st.markdown("<div id='library-toggle-wrap'>", unsafe_allow_html=True)
    if st.button("📚 Library", key="toggle_library"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.sidebar_open:
        render_sidebar_library(st)

    # Calm header (NO box, NO tagline)
    st.markdown(
        "<div style='font-size:2.6rem;font-weight:1000;color:#00E0B8;margin-top:0.2rem;'>Ongea</div>",
        unsafe_allow_html=True
    )

    lang_name, lang_code, voice_name, voices = studio_top_row(st)

    tabs = st.tabs(["Speak", "Batch", "Fine-tune", "About"])
    with tabs[0]:
        speak_tab(st, lang_name, lang_code, voice_name, voices)
    with tabs[1]:
        batch_tab(st, lang_name, lang_code)
    with tabs[2]:
        finetune_tab(st, lang_code)
    with tabs[3]:
        about_tab(st)


def run_app():
    import streamlit as st
    st.set_page_config(page_title="Ongea", layout="wide")
    _init_state(st)
    inject_theme(st, st.session_state.sidebar_open)

    if st.session_state.mode == "welcome":
        welcome_screen(st)
    else:
        studio_screen(st)


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
