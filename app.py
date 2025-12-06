import os
import io
import uuid
import wave
import math
import base64
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict


# =========================
# IMPORTANT FIX SUMMARY (what this code does)
# =========================
# 1) Removes any accidental output that would show raw HTML (no st.code/st.text of HTML).
# 2) Ensures ALL HTML blocks are rendered with unsafe_allow_html=True.
# 3) Updates LIGHT theme colors for better contrast (text/labels/sliders visible on white).
# 4) Adds CSS to force Streamlit labels/slider text to use readable colors.
# 5) Keeps one hover dropdown menu on the logo (no extra sidebar buttons).


import streamlit as st

APP_NAME = "Ongea"
APP_TAGLINE = "Calm African TTS Studio"
OUTPUT_DIR = "ongea_outputs"

# Updated light-friendly accents (still editable)
ACCENT_A = "#14b8a6"  # teal
ACCENT_B = "#7c3aed"  # violet


LANGUAGES = [
    ("sw_ke", "Swahili (Kiswahili) — KE/TZ/UG"),
    ("am_et", "Amharic (አማርኛ) — ET"),
    ("so_so", "Somali (Soomaaliga) — SO/ET/DJ"),
    ("yo_ng", "Yoruba (Yorùbá) — NG/Benin"),
    ("sn_zw", "Shona — ZW"),
    ("xh_za", "Xhosa (isiXhosa) — ZA"),
    ("af_za", "Afrikaans — ZA/NA"),
]

VOICES = [
    ("meta_sw_male_neutral", "Ongea Swahili Male / Neutral (Meta Base)"),
    ("meta_sw_female_warm", "Ongea Swahili Female / Warm (Meta Base)"),
    ("openai_alloy", "OpenAI — Alloy (fallback TTS)"),
    ("openai_verse", "OpenAI — Verse (fallback TTS)"),
]


@dataclass
class Clip:
    id: str
    created_at: str
    mode: str
    language_id: str
    language_label: str
    voice_id: str
    voice_label: str
    speed: float
    pitch: float
    text: str
    filename: str
    mime: str


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_label(options: List[Tuple[str, str]], key: str) -> str:
    for k, v in options:
        if k == key:
            return v
    return key


def svg_favicon_data_uri(accent_a: str, accent_b: str, bg: str) -> str:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128' viewBox='0 0 128 128'>
      <defs>
        <linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>
          <stop offset='0' stop-color='{accent_a}'/>
          <stop offset='1' stop-color='{accent_b}'/>
        </linearGradient>
      </defs>
      <rect x='8' y='8' width='112' height='112' rx='28' fill='{bg}'/>
      <circle cx='64' cy='64' r='36' fill='url(#g)' opacity='0.92'/>
      <path d='M64 42c-7 0-12 5-12 12v14c0 7 5 12 12 12s12-5 12-12V54c0-7-5-12-12-12Z'
            fill='white' opacity='0.95'/>
      <path d='M46 66c0 10 8 18 18 18s18-8 18-18'
            fill='none' stroke='white' stroke-width='7' stroke-linecap='round' opacity='0.95'/>
      <path d='M64 90v10' stroke='white' stroke-width='7' stroke-linecap='round' opacity='0.95'/>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def inject_favicon_and_title(data_uri: str):
    st.markdown(
        f"""
        <script>
          const link = document.querySelector("link[rel~='icon']") || document.createElement('link');
          link.rel = 'icon';
          link.href = "{data_uri}";
          document.getElementsByTagName('head')[0].appendChild(link);
          document.title = "{APP_NAME}";
        </script>
        """,
        unsafe_allow_html=True,
    )


def set_qp(view: str, theme: str, lib: int, mode: str):
    st.query_params["view"] = view
    st.query_params["theme"] = theme
    st.query_params["lib"] = str(lib)
    st.query_params["mode"] = mode


def read_qp() -> Dict[str, str]:
    qp = st.query_params
    out = {}
    for k in ["view", "theme", "lib", "mode"]:
        v = qp.get(k)
        if v is not None:
            out[k] = str(v)
    return out


def save_audio_bytes(audio_bytes: bytes, ext: str = "mp3") -> Tuple[str, str]:
    ensure_dirs()
    uid = uuid.uuid4().hex[:10]
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uid}.{ext}"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    mime = "audio/mpeg" if ext.lower() == "mp3" else "audio/wav"
    return fname, mime


def zip_files(filenames: List[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in filenames:
            src = os.path.join(OUTPUT_DIR, fn)
            if os.path.exists(src):
                z.write(src, arcname=fn)
    buf.seek(0)
    return buf.read()


def make_placeholder_wav(seconds: float = 0.65, freq: float = 440.0) -> bytes:
    sr = 22050
    n = int(sr * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            t = i / sr
            fade = min(1.0, i / (sr * 0.05), (n - i) / (sr * 0.08))
            s = int(24000 * fade * math.sin(2 * math.pi * freq * t))
            wf.writeframesraw(s.to_bytes(2, "little", signed=True))
    return buf.getvalue()


def synthesize_meta_mms(text: str, language_id: str, voice_id: str, speed: float, pitch: float):
    # Plug your real Meta/MMS pipeline here: return (audio_bytes, ext) e.g. ("mp3" or "wav")
    return None


def synthesize_openai_fallback(text: str, voice_id: str) -> Tuple[bytes, str]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing openai package. Install: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=api_key)
    voice_map = {"openai_alloy": "alloy", "openai_verse": "verse"}
    model = os.environ.get("ONGEA_OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

    resp = client.audio.speech.create(model=model, voice=voice_map.get(voice_id, "alloy"), input=text)
    return resp.read(), "mp3"


def generate_tts_audio_bytes(text: str, language_id: str, voice_id: str, speed: float, pitch: float) -> Tuple[bytes, str]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is empty.")

    if voice_id.startswith("meta_"):
        out = synthesize_meta_mms(text, language_id, voice_id, speed, pitch)
        if out is not None:
            return out
        st.warning("Meta/MMS synth is not wired yet. Generating a placeholder WAV to keep UI usable.")
        return make_placeholder_wav(), "wav"

    if voice_id.startswith("openai_"):
        return synthesize_openai_fallback(text, voice_id)

    return make_placeholder_wav(), "wav"


def css(theme: str, library_open: bool) -> str:
    dark = (theme == "dark")

    if dark:
        bgA = "#070c16"
        bgB = "#071b22"
        card = "rgba(255,255,255,0.07)"
        card2 = "rgba(255,255,255,0.11)"
        border = "rgba(255,255,255,0.14)"
        text = "rgba(255,255,255,0.93)"
        muted = "rgba(255,255,255,0.70)"
        input_bg = "rgba(255,255,255,0.07)"
        shadow = "0 18px 60px rgba(0,0,0,0.40)"
    else:
        # NEW: higher-contrast, light-friendly palette
        bgA = "#f4fbff"
        bgB = "#f3fffb"
        card = "rgba(255,255,255,0.82)"
        card2 = "rgba(255,255,255,0.95)"
        border = "rgba(15,23,42,0.12)"
        text = "#0f172a"
        muted = "rgba(15,23,42,0.66)"
        input_bg = "rgba(255,255,255,0.98)"
        shadow = "0 16px 44px rgba(15,23,42,0.10)"

    sb_tx = "0px" if library_open else "-360px"
    sb_op = "1" if library_open else "0"

    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

      :root {{
        --bgA: {bgA};
        --bgB: {bgB};
        --card: {card};
        --card2: {card2};
        --border: {border};
        --text: {text};
        --muted: {muted};
        --input: {input_bg};
        --shadow: {shadow};
        --a: {ACCENT_A};
        --b: {ACCENT_B};
        --r: 22px;
      }}

      #MainMenu, footer, header {{ visibility: hidden; }}
      div[data-testid="stToolbar"] {{ display:none; }}

      .stApp {{
        background:
          radial-gradient(1100px 650px at 12% 14%, rgba(20,184,166,0.20) 0%, transparent 62%),
          radial-gradient(1000px 650px at 90% 14%, rgba(124,58,237,0.18) 0%, transparent 64%),
          radial-gradient(980px 660px at 55% 96%, rgba(20,184,166,0.12) 0%, transparent 62%),
          linear-gradient(120deg, var(--bgA) 8%, var(--bgB) 92%);
        color: var(--text);
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      }}

      .block-container {{
        padding-top: 1.0rem !important;
        padding-bottom: 1.35rem !important;
        max-width: 1240px;
      }}

      /* Library drawer */
      section[data-testid="stSidebar"] {{
        position: fixed;
        top: 0; left: 0;
        height: 100vh;
        width: 340px !important;
        transform: translateX({sb_tx});
        opacity: {sb_op};
        transition: transform .22s ease, opacity .16s ease;
        z-index: 100;
        background: linear-gradient(180deg, var(--card2), var(--card)) !important;
        border-right: 1px solid var(--border);
        box-shadow: var(--shadow);
      }}

      /* Header */
      .oge-top {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 14px;
        margin-bottom: 14px;
      }}
      .oge-left {{
        display:flex;
        align-items:center;
        gap: 12px;
      }}
      .oge-logo {{
        width: 48px; height: 48px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(20,184,166,0.22), rgba(124,58,237,0.17));
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        display:flex; align-items:center; justify-content:center;
        user-select:none;
        position: relative;
      }}
      .oge-logo span {{ font-size: 18px; }}

      .oge-brand {{
        display:flex;
        flex-direction:column;
        line-height:1.05;
      }}
      .oge-title {{
        font-family: "Plus Jakarta Sans", Inter, sans-serif;
        font-weight: 800;
        font-size: 36px;
        letter-spacing: -0.02em;
        margin: 0;
        color: var(--text);
      }}
      .oge-sub {{
        font-size: 13px;
        color: var(--muted);
        margin-top: 4px;
      }}

      /* Hover dropdown — IMPORTANT: no code blocks, real HTML only */
      .oge-dd {{
        position: absolute;
        top: 56px;
        left: 0;
        width: 285px;
        background: var(--card2);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        padding: 10px;
        display: none;
        z-index: 9999;
      }}
      .oge-logo:hover .oge-dd {{ display: block; }}
      .oge-dd a {{
        text-decoration:none;
        color: var(--text);
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 10px;
        padding: 10px 12px;
        border-radius: 14px;
      }}
      .oge-dd a:hover {{
        background: linear-gradient(135deg, rgba(20,184,166,0.16), rgba(124,58,237,0.12));
        border: 1px solid rgba(255,255,255,0.10);
      }}
      .oge-dd small {{
        color: var(--muted);
        font-size: 12px;
      }}
      .oge-dd hr {{
        border: none;
        border-top: 1px solid var(--border);
        margin: 8px 0;
      }}

      /* Cards */
      .oge-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--r);
        box-shadow: var(--shadow);
        padding: 18px;
      }}
      .oge-card-head {{
        display:flex;
        align-items:flex-start;
        gap: 12px;
        margin-bottom: 10px;
      }}
      .oge-ic {{
        width: 38px; height: 38px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(20,184,166,0.22), rgba(124,58,237,0.18));
        border: 1px solid var(--border);
        display:flex; align-items:center; justify-content:center;
        font-size: 18px;
      }}
      .oge-card-title {{
        font-family: "Plus Jakarta Sans", Inter, sans-serif;
        font-size: 20px;
        font-weight: 800;
        margin: 0;
        color: var(--text);
      }}
      .oge-card-sub {{
        color: var(--muted);
        font-size: 13px;
        margin-top: 2px;
        line-height: 1.3;
      }}

      .oge-label {{
        font-size: 12px;
        color: var(--muted);
        margin: 0 0 6px 2px;
      }}

      /* Compact selects */
      div[data-baseweb="select"] > div {{
        background: var(--input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        min-height: 42px !important;
        box-shadow: none !important;
      }}
      div[data-baseweb="select"] span {{
        color: var(--text) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
      }}
      div[role="listbox"] {{
        background: var(--card2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: var(--shadow) !important;
      }}

      /* Inputs */
      .stTextArea textarea {{
        background: var(--input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        color: var(--text) !important;
        font-weight: 500 !important;
      }}
      .stTextArea textarea::placeholder {{
        color: rgba(15,23,42,0.45) !important;
      }}

      /* Segmented radio */
      div[data-testid="stRadio"] > label {{ display:none; }}
      div[data-testid="stRadio"] div[role="radiogroup"] {{
        background: var(--card2);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 6px;
        display: inline-flex;
        gap: 6px;
        box-shadow: var(--shadow);
      }}
      div[data-testid="stRadio"] label[data-baseweb="radio"] {{
        padding: 10px 14px;
        border-radius: 999px;
        margin: 0 !important;
      }}
      div[data-testid="stRadio"] label[data-baseweb="radio"] span {{
        color: var(--text) !important;
        font-weight: 700 !important;
      }}
      div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {{
        background: linear-gradient(135deg, rgba(20,184,166,0.16), rgba(124,58,237,0.12)) !important;
        border: 1px solid rgba(255,255,255,0.10);
      }}

      /* Slider visibility + accent */
      div[data-testid="stSlider"] * {{
        color: var(--text) !important;
        font-weight: 650 !important;
      }}
      div[data-testid="stSlider"] [data-baseweb="slider"] div {{
        box-shadow: none !important;
      }}
      /* Thumb */
      div[data-testid="stSlider"] [role="slider"] {{
        background: linear-gradient(135deg, var(--a), var(--b)) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
      }}
      /* Filled track (best-effort selectors for Streamlit/BaseWeb) */
      div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background-color: rgb"] {{
        background: linear-gradient(135deg, rgba(20,184,166,0.75), rgba(124,58,237,0.65)) !important;
      }}

      /* Primary button */
      .stButton > button {{
        background: linear-gradient(135deg, rgba(20,184,166,0.95) 0%, rgba(124,58,237,0.88) 100%) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 999px !important;
        padding: 0.72rem 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.01em !important;
        box-shadow: var(--shadow) !important;
        transition: transform .08s ease, filter .18s ease;
      }}
      .stButton > button:hover {{ filter: brightness(1.02); transform: translateY(-1px); }}
      .stButton > button:active {{ transform: translateY(0px); }}

      audio {{ width:100%; border-radius: 999px; }}
      .stCaption {{ color: var(--muted) !important; }}
    </style>
    """


def init_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "view" not in st.session_state:
        st.session_state.view = "studio"
    if "lib" not in st.session_state:
        st.session_state.lib = 0
    if "mode" not in st.session_state:
        st.session_state.mode = "Batch"

    if "language_id" not in st.session_state:
        st.session_state.language_id = LANGUAGES[0][0]
    if "voice_id" not in st.session_state:
        st.session_state.voice_id = VOICES[0][0]

    if "speed" not in st.session_state:
        st.session_state.speed = 1.00
    if "pitch" not in st.session_state:
        st.session_state.pitch = 0.00

    if "speak_text" not in st.session_state:
        st.session_state.speak_text = ""
    if "batch_text" not in st.session_state:
        st.session_state.batch_text = ""

    if "clips" not in st.session_state:
        st.session_state.clips = []
    if "latest_clip_id" not in st.session_state:
        st.session_state.latest_clip_id = None


def apply_qp_to_state():
    qp = read_qp()
    if qp.get("theme") in ("light", "dark"):
        st.session_state.theme = qp["theme"]
    if qp.get("view") in ("studio", "finetune", "about"):
        st.session_state.view = qp["view"]
    if "lib" in qp:
        try:
            st.session_state.lib = 1 if int(qp["lib"]) == 1 else 0
        except:
            st.session_state.lib = 0
    if qp.get("mode") in ("Speak", "Batch"):
        st.session_state.mode = qp["mode"]


def build_link(view=None, theme=None, lib=None, mode=None) -> str:
    v = view if view is not None else st.session_state.view
    t = theme if theme is not None else st.session_state.theme
    l = lib if lib is not None else st.session_state.lib
    m = mode if mode is not None else st.session_state.mode
    return f"?view={v}&theme={t}&lib={l}&mode={m}"


def add_clip(mode: str, text: str, filename: str, mime: str):
    clip = Clip(
        id=uuid.uuid4().hex,
        created_at=now_iso(),
        mode=mode,
        language_id=st.session_state.language_id,
        language_label=get_label(LANGUAGES, st.session_state.language_id),
        voice_id=st.session_state.voice_id,
        voice_label=get_label(VOICES, st.session_state.voice_id),
        speed=float(st.session_state.speed),
        pitch=float(st.session_state.pitch),
        text=text,
        filename=filename,
        mime=mime,
    )
    st.session_state.clips.append(clip)
    st.session_state.latest_clip_id = clip.id
    return clip


def get_latest_clip() -> Optional[Clip]:
    if not st.session_state.latest_clip_id:
        return None
    for c in reversed(st.session_state.clips):
        if c.id == st.session_state.latest_clip_id:
            return c
    return None


def render_library_drawer():
    st.sidebar.markdown("### 📚 Library")
    st.sidebar.caption("Generated speeches in this session.")

    if not st.session_state.clips:
        st.sidebar.markdown(
            "<div class='oge-card' style='color:var(--muted)'>No speech yet.</div>",
            unsafe_allow_html=True,
        )
        return

    for clip in reversed(st.session_state.clips[-40:]):
        st.sidebar.markdown(
            f"""
            <div class="oge-card" style="padding:14px; margin-bottom:12px;">
              <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
                <div style="font-weight:900; font-size:13px;">🎧 {clip.mode}</div>
                <div style="font-size:11px; color:var(--muted);">{clip.created_at}</div>
              </div>
              <div style="margin-top:6px; font-size:12px; color:var(--muted);">
                {clip.language_label} • {clip.voice_label}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        src = os.path.join(OUTPUT_DIR, clip.filename)
        if os.path.exists(src):
            data = open(src, "rb").read()
            st.sidebar.audio(data, format=clip.mime)
            st.sidebar.download_button(
                "Download",
                data=data,
                file_name=clip.filename,
                mime=clip.mime,
                key=f"dl_sb_{clip.id}",
                use_container_width=True,
            )


def render_header():
    # NOTE: this is pure HTML rendered with unsafe_allow_html=True
    lib_toggle_to = 0 if st.session_state.lib == 1 else 1
    theme_toggle_to = "dark" if st.session_state.theme == "light" else "light"

    studio_link = build_link(view="studio")
    finetune_link = build_link(view="finetune")
    about_link = build_link(view="about")

    lib_link = build_link(lib=lib_toggle_to)
    theme_link = build_link(theme=theme_toggle_to)

    theme_label = "🌙 Dark mode" if st.session_state.theme == "light" else "☀️ Light mode"
    lib_label = "📚 Open Library" if st.session_state.lib == 0 else "📚 Close Library"

    st.markdown(
        f"""
        <div class="oge-top">
          <div class="oge-left">
            <div class="oge-logo" aria-label="Ongea menu">
              <span>🎙️</span>
              <div class="oge-dd">
                <a href="{studio_link}">
                  <div>🎛 Studio<br><small>Speak + Batch</small></div><div>›</div>
                </a>
                <a href="{finetune_link}">
                  <div>🧪 Fine-tune<br><small>Local training</small></div><div>›</div>
                </a>
                <a href="{about_link}">
                  <div>ℹ️ About<br><small>What this is</small></div><div>›</div>
                </a>
                <hr/>
                <a href="{lib_link}">
                  <div>{lib_label}<br><small>Session history</small></div><div>›</div>
                </a>
                <a href="{theme_link}">
                  <div>{theme_label}<br><small>Light / dark</small></div><div>›</div>
                </a>
              </div>
            </div>

            <div class="oge-brand">
              <div class="oge-title">{APP_NAME}</div>
              <div class="oge-sub">{APP_TAGLINE}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_toolbar():
    c1, c2 = st.columns([1.2, 1.2], gap="large")

    with c1:
        st.markdown('<div class="oge-label">Language</div>', unsafe_allow_html=True)
        labels = [x[1] for x in LANGUAGES]
        ids = [x[0] for x in LANGUAGES]
        idx = ids.index(st.session_state.language_id) if st.session_state.language_id in ids else 0
        chosen = st.selectbox("Language", labels, index=idx, label_visibility="collapsed", key="lang_select")
        st.session_state.language_id = ids[labels.index(chosen)]

    with c2:
        st.markdown('<div class="oge-label">Voice / Model</div>', unsafe_allow_html=True)
        vlabels = [x[1] for x in VOICES]
        vids = [x[0] for x in VOICES]
        vidx = vids.index(st.session_state.voice_id) if st.session_state.voice_id in vids else 0
        chosen_v = st.selectbox("Voice", vlabels, index=vidx, label_visibility="collapsed", key="voice_select")
        st.session_state.voice_id = vids[vlabels.index(chosen_v)]

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    mode_key = f"mode_radio_{st.session_state.view}"
    mode = st.radio("Mode", ["Speak", "Batch"], horizontal=True, label_visibility="collapsed", key=mode_key)
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        set_qp(st.session_state.view, st.session_state.theme, st.session_state.lib, st.session_state.mode)


def render_settings_panel():
    st.markdown(
        """
        <div class="oge-card">
          <div class="oge-card-head">
            <div class="oge-ic">⚙️</div>
            <div>
              <div class="oge-card-title">Settings</div>
              <div class="oge-card-sub">Tone + output controls.</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    speed = st.slider(
        "Speed",
        0.70, 1.40,
        float(st.session_state.speed),
        0.01,
        key=f"speed_{st.session_state.view}_{st.session_state.mode}",
    )
    pitch = st.slider(
        "Pitch (semitones)",
        -6.0, 6.0,
        float(st.session_state.pitch),
        0.10,
        key=f"pitch_{st.session_state.view}_{st.session_state.mode}",
    )

    st.session_state.speed = float(speed)
    st.session_state.pitch = float(pitch)

    st.markdown("</div>", unsafe_allow_html=True)


def render_latest_panel():
    clip = get_latest_clip()
    st.markdown(
        """
        <div class="oge-card">
          <div class="oge-card-head">
            <div class="oge-ic">🎧</div>
            <div>
              <div class="oge-card-title">Latest</div>
              <div class="oge-card-sub">Your most recent clip appears here.</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    if not clip:
        st.markdown("<div style='color:var(--muted)'>No clip yet.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    src = os.path.join(OUTPUT_DIR, clip.filename)
    if os.path.exists(src):
        data = open(src, "rb").read()
        st.audio(data, format=clip.mime)
        st.download_button(
            "Download latest",
            data=data,
            file_name=clip.filename,
            mime=clip.mime,
            key="dl_latest",
            use_container_width=True,
        )
        st.caption(f"{clip.language_label} • {clip.voice_label}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_studio():
    render_toolbar()

    left, right = st.columns([2.25, 1.0], gap="large")

    with right:
        render_settings_panel()

    with left:
        if st.session_state.mode == "Speak":
            st.markdown(
                """
                <div class="oge-card">
                  <div class="oge-card-head">
                    <div class="oge-ic">🗣️</div>
                    <div>
                      <div class="oge-card-title">Speak</div>
                      <div class="oge-card-sub">Paste your script here. One clean clip.</div>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.speak_text = st.text_area(
                "Speak text",
                value=st.session_state.speak_text,
                height=320,
                placeholder="Type/paste your text here…",
                label_visibility="collapsed",
                key="speak_textarea",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            with right:
                if st.button("Generate speech", use_container_width=True, key="btn_speak"):
                    text = (st.session_state.speak_text or "").strip()
                    try:
                        with st.spinner("Generating…"):
                            audio_bytes, ext = generate_tts_audio_bytes(
                                text=text,
                                language_id=st.session_state.language_id,
                                voice_id=st.session_state.voice_id,
                                speed=st.session_state.speed,
                                pitch=st.session_state.pitch,
                            )
                            fname, mime = save_audio_bytes(audio_bytes, ext=ext)
                            add_clip("Speak", text, fname, mime)
                        st.success("Done.")
                    except Exception as e:
                        st.error(str(e))

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                render_latest_panel()

        else:
            st.markdown(
                """
                <div class="oge-card">
                  <div class="oge-card-head">
                    <div class="oge-ic">🎬</div>
                    <div>
                      <div class="oge-card-title">Batch Studio</div>
                      <div class="oge-card-sub">One line = one clip. Paste everything here.</div>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.batch_text = st.text_area(
                "Batch lines",
                value=st.session_state.batch_text,
                height=360,
                placeholder="Line 1: Habari! Karibu kwenye Ongea.\nLine 2: Leo tutaongea kuhusu…\nLine 3: Asante kwa kusikiliza.",
                label_visibility="collapsed",
                key="batch_textarea",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            with right:
                if st.button("Generate batch", use_container_width=True, key="btn_batch"):
                    raw = st.session_state.batch_text or ""
                    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                    try:
                        if not lines:
                            raise ValueError("Paste at least one non-empty line.")

                        out_files = []
                        with st.spinner(f"Generating {len(lines)} clips…"):
                            for i, line in enumerate(lines, start=1):
                                audio_bytes, ext = generate_tts_audio_bytes(
                                    text=line,
                                    language_id=st.session_state.language_id,
                                    voice_id=st.session_state.voice_id,
                                    speed=st.session_state.speed,
                                    pitch=st.session_state.pitch,
                                )
                                fname, mime = save_audio_bytes(audio_bytes, ext=ext)
                                add_clip("Batch", line, fname, mime)
                                out_files.append((fname, mime, f"Line {i}"))

                        st.session_state.batch_files = out_files
                        zip_bytes = zip_files([fn for fn, _, _ in out_files])

                        st.download_button(
                            "Download ZIP",
                            data=zip_bytes,
                            file_name=f"ongea_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            key="dl_zip",
                            use_container_width=True,
                        )
                        st.success("Batch done.")
                    except Exception as e:
                        st.error(str(e))

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                render_latest_panel()


def render_finetune():
    st.markdown(
        """
        <div class="oge-card">
          <div class="oge-card-head">
            <div class="oge-ic">🧪</div>
            <div>
              <div class="oge-card-title">Fine-tune</div>
              <div class="oge-card-sub">Wire this to your local training pipeline.</div>
            </div>
          </div>
          <div style="color:var(--muted); font-size:14px; line-height:1.6;">
            Add dataset picker, training logs, checkpoints, and “active model” cards.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about():
    st.markdown(
        """
        <div class="oge-card">
          <div class="oge-card-head">
            <div class="oge-ic">ℹ️</div>
            <div>
              <div class="oge-card-title">About Ongea</div>
              <div class="oge-card-sub">Modern Speak + Batch UI with a Library drawer.</div>
            </div>
          </div>
          <div style="color:var(--muted); font-size:14px; line-height:1.65;">
            <p><b>Studio</b> generates one clean clip from text.</p>
            <p><b>Batch</b> generates one clip per line and offers ZIP export.</p>
            <p><b>Library</b> stores clips for this session.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🎙️", layout="wide", initial_sidebar_state="collapsed")
    ensure_dirs()
    init_state()
    apply_qp_to_state()

    st.markdown(css(st.session_state.theme, bool(st.session_state.lib)), unsafe_allow_html=True)

    fav_bg = "#0b1220" if st.session_state.theme == "dark" else "#ffffff"
    inject_favicon_and_title(svg_favicon_data_uri(ACCENT_A, ACCENT_B, fav_bg))

    # sync query params so refresh preserves state
    set_qp(st.session_state.view, st.session_state.theme, st.session_state.lib, st.session_state.mode)

    render_library_drawer()
    render_header()

    if st.session_state.view == "studio":
        render_studio()
    elif st.session_state.view == "finetune":
        render_finetune()
    else:
        render_about()


if __name__ == "__main__":
    main()
