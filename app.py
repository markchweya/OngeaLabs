# app.py
"""
ONGEA LABS — ChatGPT-Style TTS Studio (Light + Dark)
---------------------------------------------------
What changed vs your previous UI:
- ChatGPT-style layout: left sidebar chat list + main chat + bottom composer
- Generate speech as "assistant messages" with audio + download
- Supports Ongea (single clip) and Batch (one clip per line)
- Light theme default + toggle Dark/Light
- Keeps your existing TTS + training pipeline mostly intact

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
import html

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

PROJECT_NAME = "ongea-labs-chatgpt-ui"
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

APP_OUTPUTS_DIR = OUTPUT_DIR / "app_outputs"
APP_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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
    run(
        [
            "python",
            "convert_original_discriminator_checkpoint.py",
            "--language_code",
            lang_code,
            "--pytorch_dump_folder_path",
            str(lang_dir),
        ],
        cwd=FINETUNE_REPO,
    )
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
    run(["accelerate", "launch", str(train_script), "--config", str(cfg_path)], cwd=FINETUNE_REPO)

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
        model.to("cpu")
        model.eval()
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
            model.to("cpu")
            model.eval()
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
        if m > 1.0:
            audio = audio / m
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

def synthesize_human(bundle: VoiceBundle, text: str) -> Tuple[np.ndarray, int]:
    chunks = split_by_punctuation(text)
    if not chunks:
        raise ValueError("Empty text.")
    audios = []
    sr_final = bundle.sr or TARGET_SR
    pause = {",": 0.18, ";": 0.22, ":": 0.22, ".": 0.38, "!": 0.42, "?": 0.42, "…": 0.55}

    for chunk_text, punct in chunks:
        a, sr = synthesize_raw(bundle, chunk_text)
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
# APP STATE + HELPERS
# =========================

def _init_state(st):
    if "theme" not in st.session_state:
        st.session_state.theme = "light"  # light | dark
    if "view" not in st.session_state:
        st.session_state.view = "studio"  # studio | finetune | about

    if "lang_name" not in st.session_state:
        st.session_state.lang_name = list(LANGUAGES.keys())[0]
    if "voice_name" not in st.session_state:
        st.session_state.voice_name = None

    if "mode" not in st.session_state:
        st.session_state.mode = "Ongea"  # Ongea | Batch

    if "speed" not in st.session_state:
        st.session_state.speed = 1.0
    if "pitch" not in st.session_state:
        st.session_state.pitch = 0.0

    # Chat sessions (like ChatGPT)
    if "chats" not in st.session_state:
        st.session_state.chats = {}  # chat_id -> {title, created, messages[]}
    if "chat_order" not in st.session_state:
        st.session_state.chat_order = []  # newest first
    if "current_chat_id" not in st.session_state:
        cid = _create_new_chat(st, title="New chat")
        st.session_state.current_chat_id = cid

def _create_new_chat(st, title="New chat") -> str:
    chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    st.session_state.chats[chat_id] = {
        "title": title,
        "created": datetime.now().isoformat(),
        "messages": [],  # each: {id, role, kind, text?, wav_path?, meta?}
    }
    st.session_state.chat_order.insert(0, chat_id)
    return chat_id

def _set_chat_title_from_text(st, chat_id: str, text: str):
    text = (text or "").strip()
    if not text:
        return
    title = text.replace("\n", " ").strip()
    if len(title) > 34:
        title = title[:34].rstrip() + "…"
    if st.session_state.chats[chat_id]["title"] == "New chat":
        st.session_state.chats[chat_id]["title"] = title

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

def _next_clip_path(lang_code: str) -> Path:
    # safe unique name
    return APP_OUTPUTS_DIR / f"{lang_code}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"

# =========================
# STYLING (ChatGPT-like)
# =========================

def inject_css(st, theme: str):
    # Streamlit chrome off
    # NOTE: do NOT hide sidebar; we use it as ChatGPT nav.
    st.markdown(
        f"""
<style>
#MainMenu, footer, header {{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none !important;}}
[data-testid="stStatusWidget"]{{display:none !important;}}
[data-testid="stHeader"]{{display:none !important;}}
[data-testid="stDecoration"]{{display:none !important;}}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --radius: 18px;
  --radius2: 14px;
  --shadow: 0 18px 50px rgba(12, 24, 48, 0.12);
  --shadow2: 0 10px 24px rgba(12, 24, 48, 0.10);
  --stroke: rgba(255,255,255,0.55);

  --teal: #19b6ad;
  --indigo: #6b66ff;
  --pink: #ff4d7d;

  --font: Inter, system-ui, -apple-system, Segoe UI, Roboto;
}}

{"/* LIGHT */" if theme=="light" else "/* DARK */"}
:root {{
  --bgA: {"#e9fbff" if theme=="light" else "#050814"};
  --bgB: {"#f1f5ff" if theme=="light" else "#0b1633"};
  --bgC: {"#fbf2ff" if theme=="light" else "#052f2a"};

  --card: {"rgba(255,255,255,0.72)" if theme=="light" else "rgba(12,16,32,0.70)"};
  --card2: {"rgba(255,255,255,0.86)" if theme=="light" else "rgba(12,16,32,0.82)"};
  --input: {"rgba(255,255,255,0.92)" if theme=="light" else "rgba(10,12,24,0.82)"};

  --text: {"#0b1220" if theme=="light" else "#eaf2ff"};
  --muted: {"rgba(11,18,32,0.62)" if theme=="light" else "rgba(234,242,255,0.70)"};
  --soft: {"rgba(11,18,32,0.10)" if theme=="light" else "rgba(234,242,255,0.10)"};
  --soft2: {"rgba(11,18,32,0.14)" if theme=="light" else "rgba(234,242,255,0.14)"};

  --sidebar: {"rgba(255,255,255,0.65)" if theme=="light" else "rgba(10,12,24,0.78)"};
}}

html, body {{
  font-family: var(--font) !important;
  color: var(--text) !important;
}}

[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 700px at 10% 0%, rgba(25,182,173,{"0.25" if theme=="light" else "0.16"}) 0%, transparent 55%),
    radial-gradient(1000px 650px at 92% 5%, rgba(107,102,255,{"0.22" if theme=="light" else "0.14"}) 0%, transparent 55%),
    radial-gradient(900px 700px at 70% 90%, rgba(255,77,125,{"0.14" if theme=="light" else "0.08"}) 0%, transparent 55%),
    linear-gradient(135deg, var(--bgA), var(--bgB), var(--bgC)) !important;
}}

.block-container {{
  max-width: 980px;
  padding-top: 1.15rem !important;
  padding-bottom: 2.0rem !important;
}}

[data-testid="stSidebar"] > div {{
  background: var(--sidebar) !important;
  border-right: 1px solid var(--soft) !important;
  backdrop-filter: blur(14px);
}}

h1,h2,h3,h4,h5,h6,p,span,div {{ color: var(--text) !important; }}
.oge-muted {{ color: var(--muted) !important; }}

/* Buttons */
.stButton>button {{
  border-radius: 999px !important;
  border: 1px solid rgba(25,182,173,0.45) !important;
  background: linear-gradient(135deg, rgba(25,182,173,0.22), rgba(107,102,255,0.16)) !important;
  padding: 0.62rem 0.95rem !important;
  font-weight: 800 !important;
  box-shadow: var(--shadow2);
}}
.stButton>button:hover {{ transform: translateY(-1px); }}

/* Inputs */
div[data-baseweb="select"] > div {{
  background: var(--input) !important;
  border: 1px solid var(--soft2) !important;
  border-radius: 14px !important;
  min-height: 42px !important;
}}
textarea {{
  background: var(--input) !important;
  border: 1px solid var(--soft2) !important;
  border-radius: 16px !important;
  color: var(--text) !important;
  font-size: 1.02rem !important;
  line-height: 1.38 !important;
}}
textarea::placeholder {{ color: rgba(120,130,150,0.72) !important; }}

/* Chat bubbles (we render our own divs) */
.oge-bubble {{
  padding: 0.85rem 0.95rem;
  border-radius: var(--radius);
  border: 1px solid var(--soft);
  box-shadow: var(--shadow2);
  backdrop-filter: blur(12px);
  margin: 0.25rem 0 0.25rem 0;
}}
.oge-assistant {{
  background: var(--card);
}}
.oge-user {{
  background: linear-gradient(135deg, rgba(25,182,173,0.20), rgba(107,102,255,0.12));
  border: 1px solid rgba(25,182,173,0.25);
}}
.oge-meta {{
  font-size: 0.82rem;
  color: var(--muted);
  margin-top: 0.5rem;
}}

audio {{
  width: 100% !important;
  border-radius: 999px !important;
}}
</style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# UI RENDERING
# =========================

def sidebar_nav(st):
    with st.sidebar:
        st.markdown(
            """
<div style="display:flex;align-items:center;gap:10px;padding:0.25rem 0.2rem 0.6rem 0.2rem;">
  <div style="width:40px;height:40px;border-radius:14px;display:flex;align-items:center;justify-content:center;
              background:linear-gradient(135deg, rgba(25,182,173,0.26), rgba(107,102,255,0.18));
              border:1px solid rgba(255,255,255,0.25); box-shadow:0 10px 24px rgba(0,0,0,0.10);">🎙️</div>
  <div>
    <div style="font-weight:900;font-size:1.05rem;line-height:1.0;">Ongea Labs</div>
    <div style="font-size:0.85rem;opacity:0.75;">Chat TTS Studio</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("➕ New chat", use_container_width=True, key="btn_new_chat"):
            new_id = _create_new_chat(st, title="New chat")
            st.session_state.current_chat_id = new_id
            st.rerun()

        st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
        q = st.text_input("Search chats", value="", key="chat_search", placeholder="Search…")
        st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

        # Build list
        chat_ids = st.session_state.chat_order[:]
        if q.strip():
            qq = q.strip().lower()
            chat_ids = [cid for cid in chat_ids if qq in st.session_state.chats[cid]["title"].lower()]

        if not chat_ids:
            st.markdown("<div class='oge-muted' style='padding:0.35rem 0.2rem;'>No chats found.</div>", unsafe_allow_html=True)
        else:
            def _fmt(cid):
                return st.session_state.chats[cid]["title"]

            sel = st.radio(
                "Chats",
                chat_ids,
                index=chat_ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_ids else 0,
                format_func=_fmt,
                label_visibility="collapsed",
                key="chat_list_radio",
            )
            st.session_state.current_chat_id = sel

        st.divider()

        st.session_state.view = st.radio(
            "View",
            ["studio", "finetune", "about"],
            index=["studio", "finetune", "about"].index(st.session_state.view),
            horizontal=False,
            key="view_radio",
        )

        st.divider()

        # Theme toggle
        theme_is_dark = (st.session_state.theme == "dark")
        new_dark = st.toggle("Dark mode", value=theme_is_dark, key="toggle_dark")
        st.session_state.theme = "dark" if new_dark else "light"

        # Quick settings in sidebar (like ChatGPT “settings drawer”)
        with st.expander("Voice & output settings", expanded=True):
            lang_keys = list(LANGUAGES.keys())
            st.session_state.lang_name = st.selectbox(
                "Language",
                lang_keys,
                index=lang_keys.index(st.session_state.lang_name),
                key="sb_lang",
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
                key="sb_voice",
            )

            st.session_state.mode = st.radio(
                "Mode",
                ["Ongea", "Batch"],
                index=0 if st.session_state.mode == "Ongea" else 1,
                horizontal=True,
                key="sb_mode",
            )

            st.session_state.speed = st.slider("Speed", 0.75, 1.50, float(st.session_state.speed), 0.05, key="sb_speed")
            st.session_state.pitch = st.slider("Pitch (semitones)", -4.0, 4.0, float(st.session_state.pitch), 0.5, key="sb_pitch")

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        # Clear current chat
        if st.button("🗑️ Clear this chat", use_container_width=True, key="btn_clear_chat"):
            cid = st.session_state.current_chat_id
            if cid in st.session_state.chats:
                st.session_state.chats[cid]["messages"] = []
                st.session_state.chats[cid]["title"] = "New chat"
            st.rerun()

def render_topbar(st):
    cid = st.session_state.current_chat_id
    title = st.session_state.chats.get(cid, {}).get("title", "Chat")
    st.markdown(
        f"""
<div style="display:flex;align-items:flex-end;justify-content:space-between;gap:14px;margin-bottom:0.65rem;">
  <div>
    <div style="font-weight:900;font-size:1.35rem;letter-spacing:-0.01em;">{html.escape(title)}</div>
    <div class="oge-muted" style="font-size:0.92rem;margin-top:0.15rem;">Type text → get natural speech audio.</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

def _append_message(st, chat_id: str, role: str, kind: str, text: str = "", wav_path: str = "", meta: dict = None):
    meta = meta or {}
    mid = f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    st.session_state.chats[chat_id]["messages"].append({
        "id": mid,
        "role": role,   # "user" | "assistant"
        "kind": kind,   # "text" | "audio"
        "text": text,
        "wav_path": wav_path,
        "meta": meta,
        "ts": datetime.now().strftime("%H:%M:%S"),
    })
    return mid

def render_chat_thread(st):
    cid = st.session_state.current_chat_id
    chat = st.session_state.chats.get(cid)
    if not chat:
        return

    messages = chat["messages"]
    if not messages:
        st.markdown(
            """
<div class="oge-bubble oge-assistant">
  <div style="font-weight:800;">Where should we begin?</div>
  <div class="oge-muted" style="margin-top:0.25rem;">
    Paste or type something in the box below. I’ll generate a clean voice clip.
  </div>
  <div class="oge-meta">Tip: For best results, keep sentences short and use punctuation.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    for m in messages:
        if m["role"] == "user":
            st.markdown(
                f"<div class='oge-bubble oge-user'>{html.escape(m.get('text',''))}</div>",
                unsafe_allow_html=True,
            )
        else:
            # assistant
            if m["kind"] == "text":
                st.markdown(
                    f"<div class='oge-bubble oge-assistant'>{html.escape(m.get('text',''))}</div>",
                    unsafe_allow_html=True,
                )
            else:
                meta = m.get("meta", {})
                label = meta.get("label", "Audio")
                voice = meta.get("voice_name", "")
                lang = meta.get("lang_code", "")
                st.markdown(
                    f"""
<div class="oge-bubble oge-assistant">
  <div style="font-weight:800;">{html.escape(label)}</div>
  <div class="oge-meta">{html.escape(m.get("ts",""))} • {html.escape(lang)} • {html.escape(voice)}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.audio(m["wav_path"], format="audio/wav")
                try:
                    data = Path(m["wav_path"]).read_bytes()
                    st.download_button(
                        "Download WAV",
                        data=data,
                        file_name=Path(m["wav_path"]).name,
                        mime="audio/wav",
                        use_container_width=False,
                        key=f"dl_{m['id']}",
                    )
                except Exception:
                    st.caption("Could not load file for download (missing path).")

def do_generate_from_text(st, text: str):
    cid = st.session_state.current_chat_id
    if cid not in st.session_state.chats:
        return

    text = (text or "").strip()
    if not text:
        return

    # set title if new
    _set_chat_title_from_text(st, cid, text)

    # add user message
    _append_message(st, cid, role="user", kind="text", text=text)

    # resolve voice settings
    lang_code = LANGUAGES[st.session_state.lang_name]
    voices = get_voices_for(lang_code, st.session_state.lang_name)
    voice_name = st.session_state.voice_name or list(voices.keys())[0]
    model_id = voices[voice_name]

    load_voice = get_voice_loader(st)

    # generate
    try:
        with st.spinner("Loading voice model…"):
            bundle = load_voice(model_id, lang_code)

        mode = st.session_state.mode
        speed = float(st.session_state.speed)
        pitch = float(st.session_state.pitch)

        if mode == "Ongea":
            with st.spinner("Generating speech…"):
                audio, sr = synthesize_human(bundle, text)
                audio = apply_tone(audio, sr, speed=speed, pitch_semitones=pitch)
                out = _next_clip_path(lang_code)
                write_wav(out, audio, sr)

            _append_message(
                st, cid, role="assistant", kind="audio",
                wav_path=str(out),
                meta={"label": "Ongea", "voice_name": voice_name, "lang_code": lang_code, "sr": sr},
            )

        else:
            # Batch: one line = one clip
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if not lines:
                _append_message(st, cid, role="assistant", kind="text", text="Batch mode needs at least one non-empty line.")
                return

            with st.spinner(f"Generating batch ({len(lines)} lines)…"):
                for i, ln in enumerate(lines, start=1):
                    audio, sr = synthesize_human(bundle, ln)
                    audio = apply_tone(audio, sr, speed=speed, pitch_semitones=pitch)
                    out = _next_clip_path(lang_code)
                    write_wav(out, audio, sr)

                    _append_message(
                        st, cid, role="assistant", kind="audio",
                        wav_path=str(out),
                        meta={"label": f"Batch • Line {i}", "voice_name": voice_name, "lang_code": lang_code, "sr": sr},
                    )

    except Exception as e:
        _append_message(st, cid, role="assistant", kind="text", text=f"Could not generate speech: {e}")

def finetune_view(st):
    lang_code = LANGUAGES[st.session_state.lang_name]
    st.markdown(
        f"""
<div class="oge-bubble oge-assistant">
  <div style="font-weight:900;font-size:1.15rem;">Fine-tuning (Local)</div>
  <div class="oge-muted" style="margin-top:0.35rem; line-height:1.45;">
    Run training locally using your HF dataset + finetune-hf-vits.
  </div>
  <div class="oge-meta" style="margin-top:0.55rem;"><b>Command:</b> python app.py --train --lang {html.escape(lang_code)}</div>
  <div class="oge-meta"><b>Outputs:</b> {html.escape(str(OUTPUT_DIR / lang_code))}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

def about_view(st):
    st.markdown(
        """
<div class="oge-bubble oge-assistant">
  <div style="font-weight:900;font-size:1.15rem;">About Ongea</div>
  <div class="oge-muted" style="margin-top:0.35rem; line-height:1.45;">
    Ongea is a Chat-style text-to-speech studio for African voices.
    <br/><br/>
    <b>How it works</b><br/>
    • Type your text (or paste multiple lines in Batch mode)<br/>
    • The assistant replies with audio clips you can play & download<br/>
    • Speed + pitch controls are applied during export<br/>
    <br/>
    <b>Engines</b><br/>
    • Meta MMS (HF VITS) + community voices<br/>
    • Optional Coqui voices via <code>coqui:</code> prefix
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

def run_app():
    import streamlit as st

    st.set_page_config(page_title="Ongea Labs", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")
    _init_state(st)
    inject_css(st, st.session_state.theme)

    # Sidebar (ChatGPT-like)
    sidebar_nav(st)

    # Main
    render_topbar(st)

    if st.session_state.view == "studio":
        render_chat_thread(st)

        # ChatGPT-style composer
        placeholder = "Type text to speak… (Batch mode: paste multiple lines, one per line)"
        prompt = st.chat_input(placeholder)
        if prompt is not None:
            do_generate_from_text(st, prompt)
            st.rerun()

    elif st.session_state.view == "finetune":
        finetune_view(st)
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
