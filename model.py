import streamlit as st
import platform
import os
import subprocess
from pathlib import Path
from datetime import datetime
import json
import io

# Configure tesseract path for Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

if platform.system() == "Windows":
    os.environ["PATH"] = TESSERACT_PATH.rsplit("\\", 1)[0] + os.pathsep + os.environ.get("PATH", "")
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import langdetect
import speech_recognition as sr
import pytesseract
from PIL import Image

# Set pytesseract path
pytesseract.pytesseract.pytesseract_cmd = TESSERACT_PATH

def ocr_image(image):
    """Extract text from image using Tesseract OCR"""
    try:
        if not Path(TESSERACT_PATH).exists():
            raise FileNotFoundError(f"Tesseract not found at {TESSERACT_PATH}")
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise Exception(f"OCR Error: {str(e)}")

# Load NLLB model
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        # Return None on failure and log the exception to the app UI when called
        st.error("Failed to load the NLLB model. See logs for details.")
        st.exception(e)
        return None, None

# Load DialoGPT model
@st.cache_resource
def load_dialo():
    name = "microsoft/DialoGPT-small"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name)
    return tok, m


# Initialize speech recognizer
recognizer = sr.Recognizer()

# Language mappings
lang_map = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "ar": "arb_Arab",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "de": "deu_Latn"
}

# History storage
HISTORY_FILE = Path(__file__).parent / "translation_history.json"

def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Could not save history: {e}")

def detect_language(text):
    text = (text or "").strip()
    if not text:
        return 'en'  # Default to English if no text
    try:
        from langdetect import detect
        lang = detect(text)
        if lang:
            return lang
    except Exception:
        pass
    def has_range(s, start, end):
        return any(start <= ord(ch) <= end for ch in s)
    if has_range(text, 0x0600, 0x06FF):
        return 'ar'
    if has_range(text, 0x4E00, 0x9FFF):
        return 'zh'
    if has_range(text, 0x0400, 0x04FF):
        return 'ru'
    return 'en'  # Always return a default


# Cross-version safe rerun helper for Streamlit
def safe_rerun():
    """Rerun works across modern (st.rerun) and older (st.experimental_rerun) Streamlit versions."""
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
    except Exception:
        pass

# Sidebar history
st.sidebar.header("Translation History")
with st.sidebar.expander("View history", expanded=False):
    history = load_history()
    if history:
        for entry in reversed(history[-100:]):
            st.markdown(f"**{entry.get('time','-')}** — {entry.get('src_lang','?')} → {entry.get('tgt_lang','?')}")
            st.write(entry.get('source',''))
            st.markdown(f"**{entry.get('translation','')}**")
            st.markdown("---")
    else:
        st.write("No history yet.")
    if st.button("Clear history"):
        save_history([])
        safe_rerun()

# Main tabs
tabs = st.tabs(["Translator", "Chatbot (DialoGPT)"])

with tabs[0]:
    st.title("🌍Translation using the NLLB Model")

    # User input
    st.subheader("Input Methods")
    input_method = st.radio("Choose input method:", ["Text", "Upload File", "Voice Recording", "OCR Image"], horizontal=True)

    text = ""

    if input_method == "Text":
        text = st.text_area("Enter text to translate:", "Hello, how are you today?")

    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read().decode("utf-8")
                text = st.text_area("File contents:", file_content, height=150)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    elif input_method == "Voice Recording":
        st.info("🎤 Record your voice to translate")
        audio_data = st.audio_input("Record your message:")
        
        if audio_data is not None:
            try:
                st.info("Transcribing audio...")
                # Convert audio bytes to WAV format
                audio_bytes = io.BytesIO(audio_data.getvalue())
                audio_bytes.seek(0)
                
                # Use Google's free speech recognition
                with sr.AudioFile(audio_bytes) as source:
                    audio = recognizer.record(source)
                
                text = recognizer.recognize_google(audio)
                st.success(f"✅ Transcribed: *{text}*")
                st.audio(audio_data)
            except sr.UnknownValueError:
                st.error("Could not understand audio. Please speak clearly.")
            except sr.RequestError as e:
                st.error(f"Error transcribing audio: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    elif input_method == "OCR Image":
        st.info("📸 Upload an image to extract and translate text")
        image_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg", "bmp", "gif"])
        
        if image_file is not None:
            try:
                # Display the image
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Extract text using OCR
                st.info("Extracting text from image...")
                text = ocr_image(image)
                
                if text:
                    st.success(f"✅ Extracted Text:")
                    text = st.text_area("Extracted content:", text, height=150)
                else:
                    st.warning("No text found in image. Try a clearer image.")
            except Exception as e:
                st.error(f"Error processing image: {e}\n\nTesseract path: {TESSERACT_PATH}\nExists: {Path(TESSERACT_PATH).exists()}")

    src_choice = st.selectbox("Source language", ["Auto-detect", "English", "French", "Arabic", "Spanish", "Chinese", "German"])
    tgt_choice = st.selectbox("Target language", ["French", "Arabic", "English", "Spanish", "Chinese", "German"])

    # Map target language to NLLB code
    tgt_map = {
        "English": "eng_Latn",
        "French": "fra_Latn",
        "Arabic": "arb_Arab",
        "Spanish": "spa_Latn",
        "Chinese": "zho_Hans",
        "German": "deu_Latn"
    }
    target_lang = tgt_map[tgt_choice]

    if st.button("Translate"):
        if not text or not text.strip():
            st.error("⚠️ Please enter or upload text to translate.")
        else:
            # Load models only when needed (lazy loading)
            with st.spinner("Loading NLLB model..."):
                tokenizer, model = load_model()
            if tokenizer is None or model is None:
                st.error("Model failed to load. Try upgrading your deployment (more RAM) or use a smaller translation model.")
                
            # determine source language when user selected Auto-detect
            detected_iso = None
            if src_choice == "Auto-detect":
                detected_iso = detect_language(text) or 'en'
                st.info(f"Detected source language: {detected_iso}")
                src_lang = lang_map.get(detected_iso, "eng_Latn")
            else:
                detected_iso = None
                # map named choice -> NLLB code (reuse tgt_map keys)
                src_lang = tgt_map.get(src_choice, "eng_Latn")
                
            # Tokenize with source language (ensure text is string)
            text_str = str(text).strip() if text else ""
            if not text_str:
                st.error("No valid text to translate.")
            else:
                inputs = tokenizer(text_str, return_tensors="pt")

                # Force target language
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

                # Generate translation
                translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
                translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

                st.success(f"**Translation:** {translation}")

                # Browser TTS controls (Web Speech API)
                try:
                    src_iso = detected_iso if detected_iso else {'English':'en','French':'fr','Arabic':'ar','Spanish':'es','Chinese':'zh','German':'de'}.get(src_choice, 'en')
                except Exception:
                    src_iso = 'en'
                speech_lang_map = {
                    'en': 'en-US',
                    'fr': 'fr-FR',
                    'ar': 'ar-SA',
                    'es': 'es-ES',
                    'zh': 'zh-CN',
                    'de': 'de-DE'
                }
                tgt_iso_map = {'English':'en','French':'fr','Arabic':'ar','Spanish':'es','Chinese':'zh','German':'de'}
                src_speech_lang = speech_lang_map.get(src_iso, 'en-US')
                tgt_speech_lang = speech_lang_map.get(tgt_iso_map.get(tgt_choice, 'en'), 'en-US')

                payload = json.dumps({
                    "src": text_str,
                    "tgt": translation,
                    "srcLang": src_speech_lang,
                    "tgtLang": tgt_speech_lang,
                })

                html = f"""
                <div style="display:flex;gap:8px;align-items:center;">
                    <button id="play-src">🔊 Play Source</button>
                    <button id="play-tgt">🔊 Play Translation</button>
                </div>
                <script id="tts-payload" type="application/json">{payload}</script>
                <script>
                (function(){{
                    function safeSpeak(txt, lang){{
                        try{{
                            if(!window.speechSynthesis){{ alert('Speech synthesis not supported in this browser.'); return; }}
                            var msg = new SpeechSynthesisUtterance(txt);
                            msg.lang = lang || 'en-US';
                            window.speechSynthesis.cancel();
                            window.speechSynthesis.speak(msg);
                        }}catch(e){{ console.error('TTS error', e); alert('TTS error: '+e.message); }}
                    }}
                    var payloadEl = document.getElementById('tts-payload');
                    var data = {{}};
                    try{{ data = JSON.parse(payloadEl.textContent); }}catch(e){{ console.error('Could not parse TTS payload', e); }}
                    document.getElementById('play-src').addEventListener('click', function(){{ safeSpeak(data.src||'', data.srcLang); }});
                    document.getElementById('play-tgt').addEventListener('click', function(){{ safeSpeak(data.tgt||'', data.tgtLang); }});
                }})();
                </script>
                """
                st.components.v1.html(html, height=140)

                # Record translation to history
                try:
                    entry = {
                        "time": datetime.now().isoformat(),
                        "src_lang": detected_iso if detected_iso else src_choice,
                        "tgt_lang": tgt_choice,
                        "source": text_str,
                        "translation": translation,
                    }
                    history = load_history()
                    history.append(entry)
                    save_history(history)
                except Exception as e:
                    st.error(f"Failed to record history: {e}")
                
                # Download option
                st.subheader("📥 Download Translation")
                col1, col2 = st.columns(2)
                
                with col1:
                    filename = st.text_input("Filename (without extension):", f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    st.download_button(
                        label="Download as TXT",
                        data=translation,
                        file_name=f"{filename}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.download_button(
                        label="Download as CSV",
                        data=f"Original,Translation\n{text_str},{translation}",
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                )

with tabs[1]:
    st.header("🤖 Chat with DialoGPT-small (local)")
    st.write("Lightweight local chatbot using `microsoft/DialoGPT-small`. This runs on your machine.")

    if 'dialo_messages' not in st.session_state:
        st.session_state.dialo_messages = []
    if 'dialo_history_ids' not in st.session_state:
        st.session_state.dialo_history_ids = None

    # Show messages
    for role, msg in st.session_state.dialo_messages:
        if role == 'You':
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

    st.markdown("---")

    # Chat input
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("You:")
        submitted = st.form_submit_button("Send")
        if submitted and user_input.strip():
            # Load DialoGPT only when needed
            with st.spinner("Loading DialoGPT model..."):
                dialo_tokenizer, dialo_model = load_dialo()
            
            new_user_input_ids = dialo_tokenizer.encode(user_input + dialo_tokenizer.eos_token, return_tensors='pt')
            if st.session_state.dialo_history_ids is not None:
                bot_input_ids = torch.cat([st.session_state.dialo_history_ids, new_user_input_ids], dim=-1)
            else:
                bot_input_ids = new_user_input_ids
            st.session_state.dialo_history_ids = dialo_model.generate(bot_input_ids, max_length=1000, pad_token_id=dialo_tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.95)
            response = dialo_tokenizer.decode(st.session_state.dialo_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            st.session_state.dialo_messages.append(("You", user_input))
            st.session_state.dialo_messages.append(("Bot", response))
            safe_rerun()

    if st.button("Reset chat", key='dialo_reset'):
        st.session_state.dialo_messages = []
        st.session_state.dialo_history_ids = None
        safe_rerun()

    st.info("Note: `DialoGPT-small` is a lightweight conversational model.")