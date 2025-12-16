# app.py
# ONGEA LABS — ChatGPT-style Chat TTS Studio
# FIX: Sidebar now closes (backdrop no longer blocks clicks) + optional close button inside sidebar too.

from __future__ import annotations

import io
import math
import time
import uuid
import wave
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import streamlit as st

APP_TITLE = "Ongea Labs — Chat TTS Studio"
MAX_WIDTH = 1200
SIDEBAR_W = 340

st.set_page_config(
    page_title="Ongea Labs",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Msg:
    role: str  # "user" | "assistant"
    text: str
    meta: Dict
    audio_wav: Optional[bytes] = None


@dataclass
class Chat:
    id: str
    title: str
    created_at: float
    messages: List[Msg]


# -----------------------------
# Helpers
# -----------------------------
def _k(*parts: str) -> str:
    return "k__" + "__".join(parts)


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def ss_init():
    ss = st.session_state
    ss.setdefault("theme", "dark")     # "dark" | "light"
    ss.setdefault("sb_open", True)    # drawer open?
    ss.setdefault("mode", "Ongea")    # "Ongea" | "Batch"
    ss.setdefault("view", "studio")   # "studio" | "finetune" | "about"
    ss.setdefault("chat_search", "")
    ss.setdefault("active_chat_id", None)

    ss.setdefault("lang_name", "Swahili (Kiswahili)")
    ss.setdefault("voice_name", "Ongea Labs Swahili Male / Neutral (Meta Base)")
    ss.setdefault("speed", 1.00)
    ss.setdefault("pitch", 0.0)

    ss.setdefault("chats", [])

    if ss["active_chat_id"] is None:
        new_chat()


def _get_chats() -> List[Chat]:
    out: List[Chat] = []
    for c in st.session_state["chats"]:
        if isinstance(c, Chat):
            out.append(c)
        else:
            msgs = [Msg(**m) for m in c.get("messages", [])]
            out.append(Chat(id=c["id"], title=c["title"], created_at=c["created_at"], messages=msgs))
    return out


def _save_chats(chats: List[Chat]) -> None:
    st.session_state["chats"] = [asdict(c) for c in chats]


def set_active_chat(chat_id: str) -> None:
    st.session_state["active_chat_id"] = chat_id


def get_active_chat() -> Chat:
    chats = _get_chats()
    cid = st.session_state["active_chat_id"]
    for c in chats:
        if c.id == cid:
            return c
    new_chat()
    return get_active_chat()


def new_chat() -> None:
    chats = _get_chats()
    cid = uuid.uuid4().hex[:10]
    c = Chat(id=cid, title="New chat", created_at=time.time(), messages=[])
    chats.insert(0, c)
    _save_chats(chats)
    set_active_chat(cid)


def rename_chat_if_needed(chat: Chat, user_text: str) -> None:
    if chat.title != "New chat":
        return
    t = user_text.strip().split("\n", 1)[0].strip()
    if t:
        chat.title = t[:28]


# -----------------------------
# TTS placeholders (replace with your real engine)
# -----------------------------
LANGS = [
    "Swahili (Kiswahili)",
    "English",
    "French",
    "Spanish",
    "Arabic",
    "Portuguese",
    "German",
    "Italian",
    "Kinyarwanda",
    "Luganda",
    "Lingala",
]
VOICES = [
    "Ongea Labs Swahili Male / Neutral (Meta Base)",
    "Ongea Labs Swahili Female / Neutral (Meta Base)",
    "Ongea Labs English Male / Neutral",
    "Ongea Labs English Female / Neutral",
]


def synthesize_tts(text: str, lang: str, voice: str, speed: float, pitch: float) -> bytes:
    # Fallback: beep WAV so UI works end-to-end.
    dur = min(3.5, max(0.55, len(text.strip()) / 24.0))
    sr = 22050
    base_freq = 440.0 + (pitch * 18.0)
    freq = max(120.0, min(1200.0, base_freq))
    dur = max(0.35, dur / max(0.5, min(2.0, speed)))

    n = int(sr * dur)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            t = i / sr
            env = 1.0
            if t < 0.03:
                env = t / 0.03
            if t > dur - 0.05:
                env = max(0.0, (dur - t) / 0.05)
            s = env * math.sin(2.0 * math.pi * freq * t)
            v = int(max(-1.0, min(1.0, s)) * 32767)
            wf.writeframesraw(v.to_bytes(2, "little", signed=True))
    return buf.getvalue()


# -----------------------------
# CSS injection (robust selectors)
# -----------------------------
def inject_css(theme: str, sb_open: bool) -> None:
    if theme == "dark":
        bg2 = "#0a0e14"
        panel = "rgba(255,255,255,0.06)"
        panel2 = "rgba(255,255,255,0.08)"
        border = "rgba(255,255,255,0.10)"
        text = "rgba(255,255,255,0.92)"
        muted = "rgba(255,255,255,0.68)"
        subtle = "rgba(255,255,255,0.55)"
        input_bg = "rgba(255,255,255,0.08)"
        shadow = "0 16px 60px rgba(0,0,0,0.55)"
        hero = "rgba(255,255,255,0.95)"
        grad = (
            "radial-gradient(1200px 700px at 10% 10%, rgba(44,136,255,0.12), transparent 55%),"
            "radial-gradient(900px 600px at 90% 25%, rgba(164,85,255,0.10), transparent 55%),"
            "linear-gradient(180deg, #06080d 0%, #0b0f16 55%, #0b0f16 100%)"
        )
        backdrop = "rgba(0,0,0,0.36)"
    else:
        bg2 = "#f3f6fb"
        panel = "rgba(10,20,35,0.05)"
        panel2 = "rgba(10,20,35,0.07)"
        border = "rgba(10,20,35,0.10)"
        text = "rgba(10,20,35,0.92)"
        muted = "rgba(10,20,35,0.66)"
        subtle = "rgba(10,20,35,0.52)"
        input_bg = "rgba(10,20,35,0.06)"
        shadow = "0 16px 60px rgba(10,20,35,0.14)"
        hero = "rgba(10,20,35,0.92)"
        grad = (
            "radial-gradient(1200px 700px at 10% 10%, rgba(0,170,255,0.18), transparent 60%),"
            "radial-gradient(900px 600px at 90% 25%, rgba(186,85,255,0.16), transparent 60%),"
            "linear-gradient(180deg, #ffffff 0%, #f4f7ff 55%, #f6f8fc 100%)"
        )
        backdrop = "rgba(10,20,35,0.18)"

    sb_transform = "translateX(0)" if sb_open else "translateX(-110%)"
    sb_opacity = "1" if sb_open else "0"
    sb_pointer = "auto" if sb_open else "none"
    backdrop_display = "block" if sb_open else "none"

    st.markdown(
        f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  height: 100%;
  background: {grad} !important;
}}
header, footer {{ visibility: hidden; height: 0 !important; }}

/* Main always full width (drawer overlay) */
section[data-testid="stMain"] {{
  width: 100% !important;
  margin-left: 0 !important;
}}
div.block-container {{
  padding-top: 22px !important;
  padding-bottom: 150px !important;
  max-width: {MAX_WIDTH}px;
}}

/* Hide Streamlit's own sidebar arrow (the "extra" control) */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[title="Open sidebar"],
button[title="Close sidebar"],
button[title="Collapse sidebar"],
button[title="Expand sidebar"] {{
  display: none !important;
}}

/* Sidebar drawer */
[data-testid="stSidebar"] {{
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  height: 100vh !important;
  width: {SIDEBAR_W}px !important;
  min-width: {SIDEBAR_W}px !important;
  max-width: {SIDEBAR_W}px !important;

  transform: {sb_transform} !important;
  opacity: {sb_opacity} !important;
  pointer-events: {sb_pointer} !important;

  transition: transform 170ms ease, opacity 140ms ease !important;
  z-index: 1002 !important;
  background: {bg2} !important;
  border-right: 1px solid {border} !important;
  box-shadow: {shadow} !important;
}}
[data-testid="stSidebar"] > div {{
  padding: 14px !important;
}}

/* Backdrop (IMPORTANT: do NOT block clicks, so the ☰ button can close the sidebar) */
.ongea-backdrop {{
  position: fixed;
  inset: 0;
  background: {backdrop};
  z-index: 1001;
  display: {backdrop_display};
  pointer-events: none;  /* <-- THIS is the fix */
}}

/* Typography */
* {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}}
.ongea-title {{ color: {text}; font-weight: 780; letter-spacing: -0.02em; }}
.ongea-subtitle {{ color: {muted}; font-size: 0.92rem; }}

/* Pills */
.pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 999px;
  background: {panel};
  border: 1px solid {border};
  color: {text};
  font-weight: 700;
}}
.pill small {{ color: {muted}; font-weight: 700; }}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {{
  border-radius: 16px !important;
  border: 1px solid {border} !important;
  background: {input_bg} !important;
  color: {text} !important;
}}
div[data-testid="stSelectbox"] > div {{
  border-radius: 14px !important;
  border: 1px solid {border} !important;
  background: {input_bg} !important;
}}

/* Buttons */
div[data-testid="stButton"] > button {{
  border-radius: 14px !important;
  border: 1px solid {border} !important;
  background: {panel} !important;
  color: {text} !important;
  height: 42px !important;
  padding: 0 14px !important;
  font-weight: 750 !important;
}}
div[data-testid="stButton"] > button:hover {{ background: {panel2} !important; }}

/* Sidebar segmented radios (Studio / Fine-tune / About) */
div[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {{
  display: flex !important;
  flex-wrap: wrap;
  gap: 10px;
}}
div[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
  background: {panel};
  border: 1px solid {border};
  border-radius: 14px;
  padding: 10px 12px;
  margin: 0 !important;
}}
div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{ background: {panel2}; }}

/* Hero */
.ongea-hero {{
  text-align: center;
  margin-top: 10vh;
  margin-bottom: 7vh;
}}
.ongea-hero h1 {{
  font-size: 54px;
  line-height: 1.05;
  margin: 0;
  color: {hero};
  letter-spacing: -0.04em;
}}
.ongea-hero p {{
  margin-top: 14px;
  color: {muted};
  font-size: 1.15rem;
}}

/* Message cards */
.ongea-msg {{
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid {border};
  background: {panel};
}}
.ongea-msg-user {{ background: {panel2}; }}
.ongea-msg-meta {{ color: {subtle}; font-size: 0.9rem; margin-bottom: 8px; }}

/* Audio */
div[data-testid="stAudio"] {{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid {border};
  background: {panel};
  padding: 8px 10px;
}}

/* Fixed bottom composer */
div[data-testid="stChatInput"] {{
  position: fixed !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  bottom: 18px !important;
  width: min({MAX_WIDTH}px, calc(100vw - 40px)) !important;
  z-index: 1000 !important;
}}
div[data-testid="stChatInput"] > div {{
  background: {panel2} !important;
  border: 1px solid {border} !important;
  border-radius: 24px !important;
  box-shadow: {shadow} !important;
  padding: 8px 10px !important;
}}
div[data-testid="stChatInput"] textarea {{
  background: transparent !important;
  border: none !important;
  color: {text} !important;
}}
div[data-testid="stChatInput"] button {{
  border-radius: 18px !important;
  border: 1px solid {border} !important;
  background: {panel} !important;
}}
div[data-testid="stChatInput"] button:hover {{ background: {panel2} !important; }}

hr {{
  border: none;
  border-top: 1px solid {border};
  margin: 10px 0;
}}
</style>

<div class="ongea-backdrop"></div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Sidebar UI
# -----------------------------
def sidebar_ui():
    ss = st.session_state

    with st.sidebar:
        # Close button inside sidebar (so you can always close even if you move stuff around)
        if st.button("← Close", key=_k("sb", "close"), use_container_width=True):
            ss["sb_open"] = False
            st.rerun()

        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:12px;padding:6px 2px 8px 2px;">
              <div style="width:44px;height:44px;border-radius:14px;background:rgba(34,197,94,0.18);
                          border:1px solid rgba(255,255,255,0.10);display:flex;align-items:center;justify-content:center;">
                🎙️
              </div>
              <div>
                <div class="ongea-title" style="font-size:1.15rem;">Ongea Labs</div>
                <div class="ongea-subtitle">Chat TTS Studio</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋  New chat", key=_k("sb", "new_chat"), use_container_width=True):
            new_chat()
            ss["view"] = "studio"
            st.rerun()

        ss["chat_search"] = st.text_input(
            "Search chats",
            value=ss.get("chat_search", ""),
            key=_k("sb", "search"),
            label_visibility="collapsed",
            placeholder="Search chats",
        )

        view_labels = {"studio": "🔴  Studio", "finetune": "🛠️  Fine-tune", "about": "ℹ️  About"}
        ss["view"] = st.radio(
            "View",
            ["studio", "finetune", "about"],
            format_func=lambda x: view_labels[x],
            horizontal=True,
            key=_k("sb", "view"),
            label_visibility="collapsed",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown(
            '<div class="ongea-subtitle" style="font-weight:800;margin:6px 0 10px 2px;">Your chats</div>',
            unsafe_allow_html=True,
        )

        chats = _get_chats()
        q = (ss.get("chat_search", "") or "").strip().lower()
        shown = [c for c in chats if (not q) or (q in c.title.lower())]

        if not shown:
            st.markdown('<div class="ongea-subtitle">No chats yet.</div>', unsafe_allow_html=True)
        else:
            for c in shown[:60]:
                active = (c.id == ss["active_chat_id"])
                label = ("● " if active else "○ ") + c.title
                if st.button(label, key=_k("sb", "chat", c.id), use_container_width=True):
                    set_active_chat(c.id)
                    ss["view"] = "studio"
                    st.rerun()

        st.markdown("<hr/>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🌗 Theme", key=_k("sb", "theme"), use_container_width=True):
                ss["theme"] = "light" if ss["theme"] == "dark" else "dark"
                st.rerun()
        with c2:
            ss["mode"] = st.radio(
                "Mode",
                ["Ongea", "Batch"],
                horizontal=True,
                key=_k("sb", "mode"),
                label_visibility="collapsed",
            )


# -----------------------------
# Settings popover
# -----------------------------
def settings_popover():
    ss = st.session_state
    with st.popover("⚙️  Settings"):
        st.markdown("**Voice & Language**")
        ss["lang_name"] = st.selectbox(
            "Language",
            LANGS,
            index=LANGS.index(ss["lang_name"]) if ss["lang_name"] in LANGS else 0,
            key=_k("set", "lang"),
        )
        ss["voice_name"] = st.selectbox(
            "Voice",
            VOICES,
            index=VOICES.index(ss["voice_name"]) if ss["voice_name"] in VOICES else 0,
            key=_k("set", "voice"),
        )
        ss["speed"] = st.slider("Speed", 0.5, 2.0, float(ss["speed"]), 0.05, key=_k("set", "speed"))
        ss["pitch"] = st.slider("Pitch", -6.0, 6.0, float(ss["pitch"]), 0.1, key=_k("set", "pitch"))


# -----------------------------
# Top bar (open/close with ONE toggle)
# -----------------------------
def topbar():
    ss = st.session_state
    left, right = st.columns([0.68, 0.32], vertical_alignment="center")

    with left:
        label = "☰" if not ss["sb_open"] else "☰"
        if st.button(label, key=_k("top", "toggle"), help="Toggle sidebar"):
            ss["sb_open"] = not ss["sb_open"]
            st.rerun()

        st.markdown(
            f"""
            <span class="pill" style="margin-left:10px;"><small>Mode:</small> {ss["mode"]}</span>
            <span class="pill" style="margin-left:10px;"><small>View:</small> {ss["view"]}</span>
            """,
            unsafe_allow_html=True,
        )

    with right:
        settings_popover()


# -----------------------------
# Views
# -----------------------------
def render_about():
    st.markdown(
        """
        <div class="ongea-msg">
          <div class="ongea-title" style="font-size:1.25rem;">About Ongea</div>
          <div class="ongea-subtitle" style="margin-top:10px;">
            Chat-first TTS studio. Type text, select voice/language, generate audio, download WAV.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_finetune():
    st.markdown(
        """
        <div class="ongea-msg">
          <div class="ongea-title" style="font-size:1.25rem;">Fine-tune</div>
          <div class="ongea-subtitle" style="margin-top:10px;">
            Add your fine-tuning workflow UI here (datasets, training jobs, checkpoints, evaluation).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat():
    ss = st.session_state
    chat = get_active_chat()

    if not chat.messages:
        st.markdown(
            """
            <div class="ongea-hero">
              <h1>What’s on your mind today?</h1>
              <p>Type text → get natural speech audio. Use ⚙️ to choose language, voice, and tone.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for i, m in enumerate(chat.messages):
        if m.role == "user":
            with st.chat_message("user"):
                st.markdown(
                    f"""
                    <div class="ongea-msg ongea-msg-user">
                      <div class="ongea-msg-meta">You</div>
                      <div class="ongea-title" style="font-size:1.05rem;font-weight:700;">{escape_html(m.text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            label = (m.meta or {}).get(
                "label",
                f"{ss['lang_name']} — {ss['voice_name']} • Speed {ss['speed']:.2f} • Pitch {ss['pitch']:.1f}",
            )
            with st.chat_message("assistant"):
                st.markdown(
                    f"""
                    <div class="ongea-msg">
                      <div class="ongea-msg-meta">Ongea</div>
                      <div class="ongea-title" style="font-size:1.02rem;font-weight:780;">{escape_html(label)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if m.audio_wav:
                    st.audio(m.audio_wav, format="audio/wav")
                    st.download_button(
                        "Download WAV",
                        data=m.audio_wav,
                        file_name=f"ongea_{chat.id}_{i}.wav",
                        mime="audio/wav",
                        key=_k("dl", chat.id, str(i)),
                        use_container_width=True,
                    )


def persist_chat(updated: Chat) -> None:
    chats = _get_chats()
    for idx, c in enumerate(chats):
        if c.id == updated.id:
            chats[idx] = updated
            break
    _save_chats(chats)


def handle_send(user_text: str):
    ss = st.session_state
    text = (user_text or "").strip()
    if not text:
        return

    chat = get_active_chat()
    rename_chat_if_needed(chat, text)

    chat.messages.append(Msg(role="user", text=text, meta={}))

    label = f"{ss['lang_name']} — {ss['voice_name']} • Speed {ss['speed']:.2f} • Pitch {ss['pitch']:.1f}"
    if ss["mode"] == "Batch":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            wav = synthesize_tts(ln, ss["lang_name"], ss["voice_name"], float(ss["speed"]), float(ss["pitch"]))
            chat.messages.append(Msg(role="assistant", text="", meta={"label": label}, audio_wav=wav))
    else:
        wav = synthesize_tts(text, ss["lang_name"], ss["voice_name"], float(ss["speed"]), float(ss["pitch"]))
        chat.messages.append(Msg(role="assistant", text="", meta={"label": label}, audio_wav=wav))

    persist_chat(chat)


# -----------------------------
# Main
# -----------------------------
def main():
    ss_init()
    inject_css(st.session_state["theme"], st.session_state["sb_open"])

    # Sidebar always mounted; CSS controls visibility/overlay
    sidebar_ui()

    # Top bar with toggle (now clickable even when sidebar open)
    topbar()

    v = st.session_state["view"]
    if v == "about":
        render_about()
    elif v == "finetune":
        render_finetune()
    else:
        render_chat()

    placeholder = (
        "Type text to speak… (Batch: paste multiple lines, one per line)"
        if st.session_state["mode"] == "Batch"
        else "Type text to speak…"
    )
    user_text = st.chat_input(placeholder, key=_k("composer", "chat_input"))
    if user_text is not None:
        handle_send(user_text)
        st.rerun()


if __name__ == "__main__":
    main()
