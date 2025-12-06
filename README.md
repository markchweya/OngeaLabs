# OngeaLabs — African-Language TTS Studio

OngeaLabs is a lightweight **Text-to-Speech (TTS) studio for African languages** built in **Python + Streamlit**. It supports **single-text speech generation** and **batch line-by-line clip generation**, offers **multiple voice models per language** (Meta MMS + community voices), includes **tone controls** (speed/pitch), saves **WAV outputs** to disk, and keeps an **in-session audio library** for quick playback and reuse.

Live app: https://ongealabs.streamlit.app/

---

## Features

- **Single Text → Speech**
  - Generate one-off voice clips from a text prompt.
- **Batch Clip Generator**
  - Paste multiple lines and generate clips line-by-line (ideal for voiceovers).
- **Multi-Voice, Multi-Language**
  - Select from **Meta MMS** voices and **community voice models** per language.
- **Tone Controls**
  - Adjust **speed** and **pitch** to match narration style.
- **WAV Outputs + Session Library**
  - Saves generated audio as **.wav** files.
  - Keeps an **in-session library** for playback and quick access.
- **Local Fine-Tuning Launcher**
  - Can launch local fine-tuning using a **Hugging Face dataset** and the **finetune-hf-vits** training repo.

---

## Demo Note (Streamlit Sleep / Cold Start)

This app is hosted on Streamlit Community Cloud and may go to sleep when idle.  
If prompted, click **“Wake this app”** and retry after it starts.

---

## Tech Stack

- **Python**
- **Streamlit**
- **Meta MMS** (multilingual speech models/voices)
- **Hugging Face Datasets** (for training data)
- **finetune-hf-vits** (training workflow)

---

## Project Structure (typical)

> Your repo may differ — adjust this section to match.

