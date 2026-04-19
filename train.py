# train_offline.py
# Run this once on your local machine to generate pretrained models.
# Then upload the .pkl and .keras files alongside app.py.

import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Sequential
import tensorflow as tf

print("📂 Loading seismic data...")
df = pd.read_csv("waveform_compute.csv")
df['pgv_array'] = df['pgv_array'].apply(ast.literal_eval)

# ==========================================
# 1. Random Forest Damage Model
# ==========================================
print("🌲 Training Random Forest damage model...")
X_train = []
y_train = []
for _ in range(500):
    pgv = np.random.uniform(0.5, 10.0)
    age = np.random.uniform(0, 100)
    X_train.append([pgv, age])
    if pgv > 5.0 and age > 30:
        y_train.append(2)
    elif pgv > 6.0 and age > 15:
        y_train.append(1)
    elif pgv > 8.0:
        y_train.append(1)
    else:
        y_train.append(0)

damage_model = RandomForestClassifier(n_estimators=50, random_state=42)
damage_model.fit(X_train, y_train)
joblib.dump(damage_model, 'damage_model.pkl')
print("✅ Saved damage_model.pkl")

# ==========================================
# 2. Waveform Anomaly Autoencoder
# ==========================================
print("🧠 Training waveform anomaly autoencoder...")
SEQUENCE_LENGTH = 200

waveforms = []
for arr in df['pgv_array']:
    if len(arr) >= SEQUENCE_LENGTH:
        waveforms.append(arr[:SEQUENCE_LENGTH])
    else:
        padded = arr + [arr[-1]] * (SEQUENCE_LENGTH - len(arr))
        waveforms.append(padded)

X = np.array(waveforms).astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]
encoder = Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
])
decoder = Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_scaled, X_scaled,
                epochs=30,
                batch_size=32,
                validation_split=0.1,
                verbose=1,
                shuffle=True)

reconstructions = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
threshold = np.percentile(mse, 95)

autoencoder.save('anomaly_autoencoder.keras')
joblib.dump({'scaler': scaler, 'threshold': threshold}, 'anomaly_meta.pkl')
print("✅ Saved anomaly_autoencoder.keras and anomaly_meta.pkl")

print("\n🎉 All models trained and saved. You can now deploy app.py!")