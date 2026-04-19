import streamlit as st
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import ast
import requests
import os
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import save

from sklearn.ensemble import RandomForestClassifier
import numpy as np

@st.cache_resource
def build_damage_prediction_model():
    """Trains a Random Forest model to predict structural damage."""
    # 1. Create synthetic training data
    # Features: [PGV (cm/s), Home Age (Years Old)]
    X_train = []
    y_train = [] # 0 = Safe, 1 = Minor Damage, 2 = Severe Damage
    
    # Generate 500 fake data points based on earthquake engineering logic
    for _ in range(500):
        pgv = np.random.uniform(0.5, 10.0)
        age = np.random.uniform(0, 100) # 0 to 100 years old
        
        X_train.append([pgv, age])
        
        # Heuristic rules to train the model on:
        if pgv > 5.0 and age > 30:
            y_train.append(2) # Old house, massive quake = Severe
        elif pgv > 6.0 and age > 15:
            y_train.append(1) # Medium house, big quake = Minor
        elif pgv > 8.0:
            y_train.append(1) # New house, but massive quake = Minor
        else:
            y_train.append(0) # Safe
            
    # 2. Train the Random Forest
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Load the ML model into memory
damage_model = build_damage_prediction_model()
# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Seismic Selfie", page_icon="📳")

# Load API Keys (Assuming you moved them to .streamlit/secrets.toml for safety)
# If you are still using the .txt file for Google Maps, you can swap that line back in.
GMAPS_API_KEY = open("MAPS_API_KEY.txt", "r").read().strip()
GEMINI_API_KEY = open("GEMINI_API_KEY.txt", "r").read().strip()
ELEVEN_API_KEY = open("ELEVEN_API_KEY.txt", "r").read().strip()

# Configure external SDKs
genai.configure(api_key=GEMINI_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
@st.cache_data
def load_seismic_data():
    """Loads the massive CSV once into memory."""
    try:
        df = pd.read_csv("waveform_compute.csv")
        # Convert the string representation of arrays back to Python lists
        df['pgv_array'] = df['pgv_array'].apply(ast.literal_eval)
        return df
    except FileNotFoundError:
        st.error("⚠️ full_pgv_lookup.csv not found! Make sure it's in the same folder.")
        st.stop()

# Load the data into RAM
seismic_df = load_seismic_data()

def get_user_pgv(df, user_lat, user_lon):
    """Finds the closest pre-calculated grid point to the user's house."""
    # Calculate Manhattan distance (fastest) for every row
    dist = abs(df['latitude'] - user_lat) + abs(df['longitude'] - user_lon)
    nearest_idx = dist.idxmin()
    user_array = df.loc[nearest_idx, 'pgv_array']
    # Grab the first scenario (Index 0) for the demo
    return float(user_array[0])*100.0

# ==========================================
# 3. GENERATION FUNCTIONS (AUDIO & SCRIPT)
# ==========================================
import scipy.signal as signal

def generate_seismic_rumble(pgv_value, sample_rate=44100):
    """A stabilized, gritty earthquake synth that won't clip or cut out."""
    # Ensure duration is at least 3 seconds
    duration = max(3.0, min(12.0, 3.0 + (pgv_value * 0.6)))
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 1. GROUND RUMBLE (Brown Noise)
    white_noise = np.random.normal(0, 1, len(t))
    # Using a slightly higher cutoff (80Hz) to ensure it's audible on laptop speakers
    b, a = signal.butter(2, 80 / (sample_rate / 2), 'low')
    brown_noise = signal.filtfilt(b, a, white_noise)
    
    # 2. THE P-WAVE JOLT
    p_wave = np.zeros_like(t)
    p_start = int(0.3 * sample_rate)
    p_end = p_start + int(0.15 * sample_rate)
    if p_end < len(t):
        p_wave[p_start:p_end] = np.random.normal(0, 1, p_end - p_start) * 0.3
    
    # 3. STUTTER (Smoothed to prevent silence)
    # We use a sine-based tremolo instead of a jagged sawtooth to prevent the audio from "cutting"
    stutter = 0.7 + 0.3 * np.sin(2 * np.pi * 6.0 * t) 
    
    # Combine components
    raw_audio = (brown_noise * stutter) + p_wave
    
    # 4. ENVELOPE (Fade in/out)
    # Using a simple linear ramp to avoid math errors with powers
    fade_in_len = int(sample_rate * 0.5)
    fade_out_len = int(sample_rate * 1.5)
    envelope = np.ones_like(t)
    envelope[:fade_in_len] = np.linspace(0, 1, fade_in_len)
    envelope[-fade_out_len:] = np.linspace(1, 0, fade_out_len)
    
    final_audio = raw_audio * envelope
    
    # 5. INTENSITY & NORMALIZATION (The Safety Check)
    # Scale based on your 1-8 PGV scale
    intensity = min(1.0, (pgv_value / 8.0))
    final_audio = final_audio * intensity
    
    # Safety: Remove DC offset (centers the wave at 0)
    final_audio = final_audio - np.mean(final_audio)
    
    # Safety: Check for Max to prevent DivisionByZero
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        # Normalize to 80% of max volume to prevent clipping
        normalized_audio = (final_audio / max_val) * 0.8
        audio_data = np.int16(normalized_audio * 32767)
    else:
        audio_data = np.zeros_like(t, dtype=np.int16)
        
    filename = "local_rumble.wav"
    wav.write(filename, sample_rate, audio_data)
    return filename

def generate_seismic_script(city, pgv_value):
    """Uses Gemini to write a cinematic 15-second voiceover script."""
    model = genai.GenerativeModel('gemini-3-flash-preview')
    pgv_value = float(pgv_value)
    if float(pgv_value) < 2.0:
        intensity = "a mild, rolling tremor"
    elif float(pgv_value) < 5:
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
    """Takes the script from Gemini and generates an MP3 using ElevenLabs."""
    voice_mapping = {
        "Cinematic": "Marcus", # Deep, authoritative trailer voice
        "Synthwave": "Fin",
        "Ambient": "Charlotte"
    }
    selected_voice = voice_mapping.get(genre, "Brian")
    
    audio_generator = eleven_client.text_to_speech.convert(
        text=script,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2",
        output_format="mp3_44100_128"
    )
    
    output_filename = "final_narration.mp3"
    save(audio_generator, output_filename)
    return output_filename

# ==========================================
# 4. STREAMLIT UI & MAIN LOGIC
# ==========================================
st.title("Chino Hills Shake: Your Seismic Selfie 📳")
st.markdown("Enter your address to hear exactly what the ground did under your feet 18 years ago.")

# Sidebar Controls
st.sidebar.header("Developer Settings")
dev_mode = st.sidebar.checkbox("Dev Mode (Skip API calls to save quota)", value=True)

# User Inputs
address = st.text_input("Address (e.g., 1600 Amphitheatre Parkway)")
city = st.text_input("City (e.g., Mountain View)")
state = st.text_input("State (e.g., CA)")
home_year = st.number_input("Year your home was built:", min_value=1850, max_value=2024, value=1990)
if st.button("Generate Seismic Selfie", type="primary"):
    if not address or not city or not state:
        st.warning("Please fill out the full address!")
        st.stop()
        
    full_address = f"{address.strip()}, {city.strip()}, {state.strip()}"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": full_address, "key": GMAPS_API_KEY}
    
    with st.spinner("Locating your coordinates..."):
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "OK":
            user_lat = data["results"][0]["geometry"]["location"]["lat"]
            user_lon = data["results"][0]["geometry"]["location"]["lng"]
            st.success(f"Target Locked: {user_lat:.4f}, {user_lon:.4f}")
            
            # --- The Pipeline ---
            with st.spinner("Calculating local seismic intensity..."):
                my_pgv = get_user_pgv(seismic_df, user_lat, user_lon)
                st.metric(label="Peak Ground Velocity (PGV)", value=str(my_pgv)+" cm/s")
                
            with st.spinner("Synthesizing ground rumble..."):
                rumble_audio_file = generate_seismic_rumble(my_pgv)
                
            if dev_mode:
                # Save API limits while testing UI/Audio
                st.info("🛠️ DEV MODE ON: Skipping Gemini & ElevenLabs APIs.")
                script = "Dev mode is active. You feel the ground shake... it is intense."
                voiceover_audio_file = None 
            else:
                with st.spinner("Gemini is writing your personalized script..."):
                    script = generate_seismic_script(city, my_pgv)
                    st.info(f"**The Script:** {script}")
                    
                with st.spinner("Recording cinematic voiceover..."):
                    voiceover_audio_file = generate_cinematic_voiceover(script, "Cinematic")
            
            # --- Final Output ---
            st.divider()
            st.subheader("🎧 Your Experience")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**The Ground Rumble**")
                st.audio(rumble_audio_file, format="audio/wav")
            
            with col2:
                st.write("**The Narration**")
                if voiceover_audio_file:
                    st.audio(voiceover_audio_file, format="audio/mp3")
                else:
                    st.write("*(Narration disabled in Dev Mode)*")
                    
            st.balloons()
            
        else:
            st.error(f"Could not find coordinates. API Status: {data.get('status')}")
    with st.spinner("Running ML Damage Prediction..."):
        home_age = 2024 - home_year
        
        # Predict the damage!
        prediction = damage_model.predict([[my_pgv, home_age]])[0]
        
        if prediction == 0:
            damage_status = "Safe (No structural damage expected)"
        elif prediction == 1:
            damage_status = "Warning (Minor cosmetic/structural damage likely)"
        else:
            danger_status = "DANGER (Severe structural failure possible)"
            
        st.metric(label="ML Predicted Structural Status", value=damage_status)