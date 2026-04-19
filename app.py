import streamlit as st
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import ast
import requests
import os
import json
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestClassifier
import scipy.signal as signal

from elevenlabs import VoiceSettings


def generate_panicked_voiceover(script, pgv_value):
    """
    Generates narration and returns a raw numpy array instead of saving an MP3.
    """
    norm_pgv = min(1.0, pgv_value / 10.0)
    stability = max(0.3, 0.71 - (norm_pgv * 0.4))
    style = min(0.8, norm_pgv * 0.8)

    voice_settings = VoiceSettings(
        stability=stability,
        similarity_boost=0.75,
        style=style,
        use_speaker_boost=True
    )

    audio_generator = eleven_client.text_to_speech.convert(
        text=script,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2", 
        output_format="pcm_8000", # REQUEST RAW PCM INSTEAD OF MP3
        voice_settings=voice_settings
    )
    
    # Read the generator into bytes, then convert to a 16-bit numpy array
    audio_bytes = b"".join(list(audio_generator))
    voice_array = np.frombuffer(audio_bytes, dtype=np.int16)
    
    return voice_array

import scipy.signal as signal
import scipy.io.wavfile as wav
import numpy as np

def mix_audio_with_rumble_numpy(voice_array, rumble_file, pgv_value, voice_sample_rate=8000):
    """Mixes audio using pure numpy/scipy. Applies a physical shake to the voice at high PGVs."""
    rumble_sr, rumble_array = wav.read(rumble_file)
    
    voice_f32 = voice_array.astype(np.float32) / 32767.0
    rumble_f32 = rumble_array.astype(np.float32) / 32767.0
    
    if voice_sample_rate != rumble_sr:
        new_length = int(len(voice_f32) * (rumble_sr / voice_sample_rate))
        voice_f32 = signal.resample(voice_f32, new_length)

    # ==========================================
    # NEW: Physical Shake (Tremolo) on the Voice
    # ==========================================
    if pgv_value > 3.0:
        # Match the violent shaking frequency of the rumble generator
        voice_shake_hz = 3.0 + (pgv_value * 1.2)
        # Scale the depth of the tremolo based on intensity
        voice_shake_depth = min(0.6, (pgv_value - 3.0) * 0.1) 
        
        t_voice = np.arange(len(voice_f32)) / rumble_sr
        
        # Create a low-frequency oscillator (LFO)
        voice_tremolo = (1.0 - voice_shake_depth) + voice_shake_depth * np.sin(2 * np.pi * voice_shake_hz * t_voice)
        
        # Apply the shake to the voice array
        voice_f32 = voice_f32 * voice_tremolo

    # Adjust volumes
    voice_f32 = voice_f32 * 0.63  
    
    rumble_boost_db = min(6.0, pgv_value * 1.5)
    rumble_factor = 10 ** (rumble_boost_db / 20.0)
    rumble_f32 = rumble_f32 * rumble_factor
    
    # Pad shorter track
    max_length = max(len(voice_f32), len(rumble_f32))
    
    if len(voice_f32) < max_length:
        padded_voice = np.zeros(max_length, dtype=np.float32)
        padded_voice[:len(voice_f32)] = voice_f32
        voice_f32 = padded_voice
        
    if len(rumble_f32) < max_length:
        padded_rumble = np.zeros(max_length, dtype=np.float32)
        padded_rumble[:len(rumble_f32)] = rumble_f32
        rumble_f32 = padded_rumble
        
    # Overlay the tracks
    mixed = voice_f32 + rumble_f32
    
    # Normalize to prevent distortion/clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = (mixed / max_val) * 0.95  
        
    # Apply fade-in and fade-out
    fade_len = int(rumble_sr * 0.5)
    if len(mixed) > fade_len * 2:
        envelope = np.ones_like(mixed)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        mixed = mixed * envelope
        
    mixed_int16 = np.int16(mixed * 32767)
    
    output_file = "earthquake_experience.wav"
    wav.write(output_file, rumble_sr, mixed_int16)
    
    return output_file
# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Seismic Selfie", page_icon="📳")
# ==========================================
# CUSTOM CSS FOR CINEMATIC UI
# ==========================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
    
    /* Global Overrides */
    .stApp {
        background: linear-gradient(135deg, #0b0c10 0%, #1a1e2b 100%);
    }
    
    /* Hide Streamlit default header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    h1 {
        background: linear-gradient(135deg, #ff6b35 0%, #ff9f1c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
    }
    
    p, li, div:not(.stMarkdown) {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(20, 25, 35, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 16px;
        padding: 1.2rem 1.5rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #ff6b35;
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(255, 107, 53, 0.2);
    }
    [data-testid="stMetric"] label {
        font-family: 'Space Mono', monospace !important;
        color: #a0a8b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem !important;
    }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        font-size: 2.2rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #ff6b35 0%, #ff3e1d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(255, 62, 29, 0.4);
        transition: all 0.2s ease;
        text-transform: uppercase;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(255, 107, 53, 0.6);
        background: linear-gradient(135deg, #ff7e4a 0%, #ff5a3a 100%) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: rgba(20, 25, 35, 0.7) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 107, 53, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Space Mono', monospace !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff6b35 !important;
        box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.2) !important;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background: rgba(20, 25, 35, 0.7) !important;
        border: 1px solid rgba(255, 107, 53, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Space Mono', monospace !important;
    }
    
    /* Success/Warning/Error boxes */
    .stAlert {
        background: rgba(20, 25, 35, 0.8) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px !important;
        border-left: 4px solid !important;
        font-family: 'Inter', sans-serif;
    }
    div[data-baseweb="notification"] {
        background: rgba(20, 25, 35, 0.9) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 107, 53, 0.2) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(20, 25, 35, 0.5) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 107, 53, 0.2) !important;
        font-family: 'Space Mono', monospace !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(8, 10, 15, 0.95) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 107, 53, 0.3) !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #ff9f1c !important;
        font-family: 'Space Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1.2rem !important;
    }
    [data-testid="stSidebar"] .stCheckbox label {
        color: #a0a8b8 !important;
        font-family: 'Space Mono', monospace !important;
    }
    
    /* Progress/Spinner */
    .stSpinner > div {
        border-top-color: #ff6b35 !important;
    }
    
    /* Custom divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff6b35, #ff9f1c, #ff6b35, transparent);
        margin: 2rem 0;
    }
    
    /* Seismograph grid background effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(255, 107, 53, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 107, 53, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        pointer-events: none;
        z-index: -1;
    }
    
    /* Animated "REC" indicator */
    .rec-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #ff3e1d;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #ff3e1d;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Glow text effect for warnings */
    .glow-warning {
        color: #ff6b35;
        text-shadow: 0 0 10px rgba(255, 107, 53, 0.7);
        font-weight: 700;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 40px !important;
        background: rgba(20, 25, 35, 0.6) !important;
    }
    audio::-webkit-media-controls-panel {
        background: rgba(30, 35, 45, 0.9) !important;
    }
    audio::-webkit-media-controls-play-button,
    audio::-webkit-media-controls-timeline,
    audio::-webkit-media-controls-current-time-display,
    audio::-webkit-media-controls-time-remaining-display,
    audio::-webkit-media-controls-mute-button,
    audio::-webkit-media-controls-volume-slider {
        filter: invert(1) hue-rotate(180deg);
    }
</style>
""", unsafe_allow_html=True)

# Load API Keys
GMAPS_API_KEY = st.secrets["GMAPS_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
ELEVEN_API_KEY = st.secrets["ELEVEN_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# ==========================================
# 2. ML MODELS (Damage Prediction + Anomaly Detector)
# ==========================================
@st.cache_resource
def load_damage_model():
    """Loads the pretrained Random Forest model."""
    try:
        model = joblib.load('damage_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ damage_model.pkl not found. Run train_offline.py first.")
        st.stop()

@st.cache_resource
def load_anomaly_detector():
    """Loads the pretrained autoencoder, scaler, and threshold."""
    try:
        autoencoder = tf.keras.models.load_model('anomaly_autoencoder.keras')
        meta = joblib.load('anomaly_meta.pkl')
        scaler = meta['scaler']
        threshold = meta['threshold']
        return autoencoder, scaler, threshold
    except FileNotFoundError:
        st.error("❌ anomaly_autoencoder.keras or anomaly_meta.pkl not found. Run train_offline.py first.")
        st.stop()

# Load models instantly (no training spinner)
damage_model = load_damage_model()
anomaly_model, anomaly_scaler, anomaly_threshold = load_anomaly_detector()

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_seismic_data():
    """Loads the massive CSV once into memory."""
    try:
        df = pd.read_csv("waveform_compute.csv")
        df['pgv_array'] = df['pgv_array'].apply(ast.literal_eval)
        return df
    except FileNotFoundError:
        st.error("⚠️ waveform_compute.csv not found! Make sure it's in the same folder.")
        st.stop()

seismic_df = load_seismic_data()

def compute_anomaly_score(model, scaler, waveform_array, threshold, sequence_length=200):
    """Returns (reconstruction_error, is_anomaly)."""
    if len(waveform_array) >= sequence_length:
        arr = waveform_array[:sequence_length]
    else:
        arr = waveform_array + [waveform_array[-1]] * (sequence_length - len(waveform_array))
    
    X = np.array(arr).reshape(1, -1).astype(np.float32)
    X_scaled = scaler.transform(X)
    recon = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - recon))
    is_anomaly = mse > threshold
    return mse, is_anomaly

def get_user_pgv_and_waveform(df, user_lat, user_lon):
    """Finds closest grid point and returns (PGV value, full waveform array)."""
    dist = abs(df['latitude'] - user_lat) + abs(df['longitude'] - user_lon)
    nearest_idx = dist.idxmin()
    user_array = df.loc[nearest_idx, 'pgv_array']
    pgv_value = float(user_array[0]) * 100.0  # convert to cm/s
    return pgv_value, user_array, nearest_idx

# ==========================================
# 4. AUDIO GENERATION (WITH ANOMALY ENHANCEMENT)
# ==========================================
def generate_seismic_rumble(pgv_value, anomaly=False, sample_rate=44100, min_duration=0.0):
    """Synthesizes earthquake audio. Scales violently with PGV."""
    base_duration = max(13.0, min(12.0, 3.0 + (pgv_value * 0.6)))
    duration = max(base_duration, min_duration)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    white_noise = np.random.normal(0, 1, len(t))
    
    # 1. DYNAMIC FILTER: Low PGV = deep 80Hz hum. High PGV = harsh 400Hz+ structural crunch.
    cutoff_freq = min(1000, 80 + (pgv_value * 40)) 
    b, a = signal.butter(2, cutoff_freq / (sample_rate / 2), 'low')
    filtered_noise = signal.filtfilt(b, a, white_noise)
    
    # 2. DYNAMIC P-WAVE: Harder initial jolt for high PGV
    p_wave = np.zeros_like(t)
    p_start = int(0.3 * sample_rate)
    p_end = p_start + int(0.15 * sample_rate)
    if p_end < len(t):
        p_wave_intensity = 0.1 + (pgv_value * 0.05)
        p_wave[p_start:p_end] = np.random.normal(0, 1, p_end - p_start) * p_wave_intensity
    
    # 3. DYNAMIC STUTTER: Low PGV = slow 3Hz rolling. High PGV = violent 12Hz+ rattling.
    shake_hz = 3.0 + (pgv_value * 1.2)
    shake_depth = min(0.9, 0.2 + (pgv_value * 0.1)) # How deep the stutter cuts
    stutter = (1.0 - shake_depth) + shake_depth * np.sin(2 * np.pi * shake_hz * t)
    
    raw_audio = (filtered_noise * stutter) + p_wave
    
    # Anomaly crack
    if anomaly:
        crack_time = int(1.2 * sample_rate)
        crack_duration = int(0.12 * sample_rate)
        crack = np.random.normal(0, 0.9, crack_duration) * np.hanning(crack_duration)
        if crack_time + crack_duration < len(raw_audio):
            raw_audio[crack_time:crack_time+crack_duration] += crack
    
    # Envelope
    fade_in_len = int(sample_rate * 0.5)
    fade_out_len = int(sample_rate * 1.5)
    envelope = np.ones_like(t)
    envelope[:fade_in_len] = np.linspace(0, 1, fade_in_len)
    if len(envelope) > fade_out_len:
        envelope[-fade_out_len:] = np.linspace(1, 0, fade_out_len)
    
    final_audio = raw_audio * envelope
    
    # 4. OVERDRIVE/SATURATION: If PGV is high, clip the waveform so it sounds destructive
    if pgv_value > 5.0:
        drive = 1.0 + (pgv_value - 5.0) * 0.5
        final_audio = np.tanh(final_audio * drive) # Non-linear distortion
    
    # Normalize
    intensity = min(1.0, (pgv_value / 8.0))
    intensity = 0.3 + (intensity * 0.7) # Ensure PGV 1 isn't silent
    
    final_audio = final_audio * intensity
    final_audio = final_audio - np.mean(final_audio)
    
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        normalized_audio = (final_audio / max_val) * 0.9
        audio_data = np.int16(normalized_audio * 32767)
    else:
        audio_data = np.zeros_like(t, dtype=np.int16)
    
    filename = "local_rumble.wav"
    wav.write(filename, sample_rate, audio_data)
    return filename

def generate_seismic_script(city, pgv_value):
    """Uses Gemini to write a cinematic 15-second voiceover script."""
    model = genai.GenerativeModel('gemini-3-flash-preview')  # Use stable model name
    pgv_value = float(pgv_value)
    if pgv_value < 2.0:
        intensity = "a mild, rolling tremor"
    elif pgv_value < 5:
        intensity = "a strong, violent jolt"
    else:
        intensity = "a catastrophic, deafening shockwave"

    prompt = f"""
    You are the dramatic voiceover narrator for an immersive audio experience called 'Seismic Selfie'.
    The listener is experiencing a simulation of the 2008 Chino Hills earthquake exactly as it felt in {city}.
    
    The scientific data shows the Peak Ground Velocity at their house was {pgv_value:.2f} cm/s, which feels like {intensity}.
    
    Write a 15-second cinematic script (maximum 35 words). 
    Use vivid, sensory language. 
    Include ellipses (...) where the narrator should pause to let the deep earthquake rumble build.
    Start the script directly; do not include stage directions or quotes.
    """
    response = model.generate_content(prompt)
    return response.text

def generate_cinematic_voiceover(script, genre="Cinematic"):
    """Generates MP3 narration via ElevenLabs."""
    audio_generator = eleven_client.text_to_speech.convert(
        text=script,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2",
        output_format="mp3_44100_128"
    )
    output_filename = "final_narration.mp3"
    save(audio_generator, output_filename)
    return output_filename

def get_street_view_image(lat, lon, api_key, size="600x400", heading="0", pitch="10", fov="90"):
    """
    Fetches a Google Street View image for the given coordinates.
    Returns a PIL Image object or None if unavailable.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lon}",
        "size": size,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Check if the response is actually an image (not a placeholder)
        content_type = response.headers.get('content-type', '')
        if 'image' in content_type:
            return Image.open(BytesIO(response.content))
    return None

from PIL import Image
import io

def compress_image(img, max_size=(512, 512), quality=70):
    img = img.copy()
    img.thumbnail(max_size)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    return Image.open(buffer)

def analyze_house_image(image, pgv_value, home_year):
    """
    Uses Gemini Vision to analyze the house image and return a safety assessment.
    Returns a dictionary with analysis results.
    """
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    # Prepare the prompt
    home_age = 2024 - home_year
    prompt = f"""
    You are a structural engineering expert analyzing a house from its Street View image.
    
    Context:
    - The house was built around {home_year} (approximately {home_age} years old).
    - During the 2008 Chino Hills earthquake, this location experienced a Peak Ground Velocity of {pgv_value:.2f} cm/s.
    - Consider 8 PSV to be very threatening and 0 PSV to be peaceful.
    
    Please analyze the visible structure and provide a concise assessment in the following JSON format:
    {{
        "visual_risk_score": <integer 0-100 where 0=safest, 100=most vulnerable>,
        "vulnerability_flags": [<list of 1-3 specific concerns visible, e.g., "soft-story garage", "unreinforced masonry", "large windows", "cripple wall", "chimney risk", "overhead wires", "tall trees near house">],
        "estimated_damage": "<one of: 'minimal', 'moderate', 'severe'>",
        "explanation": "<1-2 sentences explaining your reasoning>"
    }}
    
    Only return valid JSON. Do not include markdown formatting or extra text.
    """
    
    # Call Gemini with image
    response = model.generate_content([prompt, compress_image(image)])
    
    # Parse JSON response
    try:
        # Clean response text (remove possible markdown code blocks)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        analysis = json.loads(text.strip())
    except Exception as e:
        st.warning(f"Gemini analysis parsing issue: {e}. Using fallback.")
        analysis = {
            "visual_risk_score": 50,
            "vulnerability_flags": ["analysis unavailable"],
            "estimated_damage": "unknown",
            "explanation": "Image analysis encountered an issue. Please refer to ML model prediction."
        }
    
    return analysis
# ==========================================
# 5. STREAMLIT UI
# ==========================================

st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <span class="rec-indicator"></span>
    <h1 style="margin: 0;">Chino Hills Shake: Seismic Selfie 📳</h1>
</div>
<p style="color: #a0a8b8; font-family: 'Space Mono', monospace; margin-bottom: 2rem;">
    Initialize telemetry to simulate historical ground motion and structural fatigue. 
    Use Southern California addresses for meaningful earthquake data about Chino Hills. 
</p>
""", unsafe_allow_html=True)

st.sidebar.header("Developer Settings")
dev_mode = st.sidebar.checkbox("Dev Mode (Skip API calls to save quota)", value=True)

# --- UPGRADE 1: Compact Input Grid ---
st.markdown("### 🎯 Target Coordinates")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    address = st.text_input("Street Address", placeholder="3650 Valle Vista Drive", label_visibility="collapsed")
with col2:
    city = st.text_input("City", placeholder="Chino Hills", label_visibility="collapsed")
with col3:
    state = st.text_input("State", placeholder="CA", label_visibility="collapsed")

col_yr, _ = st.columns([1, 3])
with col_yr:
    home_year = st.number_input("Year Built:", min_value=1850, max_value=2024, value=1990)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("INITIATE SEISMIC SIMULATION", type="primary", use_container_width=True):
    if not address or not city or not state:
        st.error("⚠️ Target coordinates incomplete. Please provide a full address.")
        st.stop()
        
    full_address = f"{address.strip()}, {city.strip()}, {state.strip()}"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": full_address, "key": GMAPS_API_KEY}
    
    # --- UPGRADE 2: Terminal-style Boot Sequence ---
    with st.status("📡 Establishing Uplink & Crunching Telemetry...", expanded=True) as status:
        
        st.write("📍 Locating spatial coordinates...")
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") != "OK":
            status.update(label="❌ Geolocation Failed", state="error")
            st.stop()
            
        user_lat = data["results"][0]["geometry"]["location"]["lat"]
        user_lon = data["results"][0]["geometry"]["location"]["lng"]
        
        st.write("📈 Extracting localized waveform arrays...")
        my_pgv, waveform_array, idx = get_user_pgv_and_waveform(seismic_df, user_lat, user_lon)
        
        st.write("🧠 Running autoencoder anomaly detection...")
        recon_error, is_anomaly = compute_anomaly_score(anomaly_model, anomaly_scaler, waveform_array, anomaly_threshold)
        
        st.write("🏚️ Processing structural fracture & fatigue models...")
        home_age = 2024 - home_year
        prediction = damage_model.predict([[my_pgv, home_age]])[0]
        if prediction == 0:
            damage_status = "✅ SAFE"
            damage_subtext = "No structural damage expected"
        elif prediction == 1:
            damage_status = "⚠️ WARNING"
            damage_subtext = "Minor cosmetic/structural damage likely"
        else:
            damage_status = "🚨 DANGER"
            damage_subtext = "Severe structural failure possible"

        if not dev_mode:
            st.write("✍️ Generating contextual script via LLM...")
            script = generate_seismic_script(city, my_pgv)
            
            st.write("🎙️ Rendering dynamic voice synthesis...")
            voiceover_file = generate_panicked_voiceover(script, my_pgv)

            st.write("🔊 Synthesizing algorithmic ground rumble...")
            rumble_file = generate_seismic_rumble(my_pgv, anomaly=is_anomaly)

            st.write("🎧 Mixing final audio arrays...")
            final_mixed_audio = mix_audio_with_rumble_numpy(voiceover_file, rumble_file, my_pgv, voice_sample_rate=8000)
            
            st.write("📸 Capturing Street View geometry...")
            house_image = get_street_view_image(user_lat, user_lon, GMAPS_API_KEY)
            
            if house_image:
                st.write("🔬 Running Gemini structural vulnerability scan...")
                visual_analysis = analyze_house_image(house_image, my_pgv, home_year)
            else:
                visual_analysis = None
        else:
            st.write("⚠️ Dev Mode Active: Skipping heavy APIs. Generating raw rumble only...")
            rumble_file = generate_seismic_rumble(my_pgv, anomaly=is_anomaly)
            
        status.update(label="✅ Telemetry Processed Successfully", state="complete", expanded=False)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- UPGRADE 3: The Dashboard Layout ---
    # Top Level Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Peak Ground Velocity (PGV)", value=f"{my_pgv:.2f} cm/s")
    m2.metric(label="Anomaly Score", value=f"{recon_error:.4f}", delta="ANOMALOUS" if is_anomaly else "NORMAL", delta_color="inverse")
    m3.metric(label="ML Fatigue Prediction", value=damage_status, help=damage_subtext)

    # Tabs for deep dives
    tab1, tab2, tab3 = st.tabs(["🎧 The Experience", "🏚️ Structural Assessment", "📊 Raw Telemetry"])

    with tab1:
        st.markdown("### Immersive Simulation")
        if not dev_mode:
            st.audio(final_mixed_audio, format="audio/wav")
            with st.expander("Show Generated Script"):
                st.write(f"*{script}*")
        else:
            st.info("Dev Mode Audio (Rumble Only)")
            st.audio(rumble_file, format="audio/wav")

    with tab2:
        if not dev_mode and house_image:
            col_img, col_data = st.columns([1, 1])
            with col_img:
                st.image(house_image, caption="Target Structure Geometry", use_container_width=True)
            with col_data:
                risk_score = visual_analysis.get('visual_risk_score', 50)
                st.metric("Visual Risk Score", f"{risk_score}/100", delta="High" if risk_score > 70 else ("Moderate" if risk_score > 40 else "Low"), delta_color="inverse")
                
                flags = visual_analysis.get('vulnerability_flags', [])
                if flags:
                    st.markdown("**Identified Vulnerabilities:**")
                    for flag in flags:
                        st.markdown(f"- ⚠️ {flag}")
                
                st.info(f"**Gemini Analysis:** {visual_analysis.get('explanation', '')}")
        elif dev_mode:
             st.warning("Dev mode is enabled. Visual analysis bypassed to save API quota.")
        else:
            st.warning("No Street View geometry available for this vector.")

    with tab3:
        st.markdown("### Waveform Diagnostics")
        st.write("The autoencoder processes the raw temporal array below to flag unusual ground motion signatures.")
        if is_anomaly:
            st.error(f"**Reconstruction Error:** {recon_error:.4f} exceeds threshold ({anomaly_threshold:.4f})")
            st.markdown("The neural network failed to neatly reconstruct this waveform, indicating a highly unusual physical event localized to this grid coordinate.")
        else:
            st.success(f"**Reconstruction Error:** {recon_error:.4f} is within normal threshold ({anomaly_threshold:.4f})")
        
        # Display the raw array data simply
        st.line_chart(waveform_array[:200]) # Quick visual of the raw array