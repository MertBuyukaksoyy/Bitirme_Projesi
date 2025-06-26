import streamlit as st
import numpy as np
import librosa
import keras
import pickle
import os
from main import extract_features
import tempfile

@st.cache_resource
def load_model_and_utils():
    model = keras.models.load_model('results/best_dense_model.keras')
    with open('results/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('results/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le, scaler

def process_audio(audio_path):
    try:
        features = extract_features(audio_path, augment=False)
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        pred_idx = np.argmax(model.predict(features, verbose=0), axis=1)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        return pred_label
    except Exception as e:
        st.error(f'Tahmin sırasında hata oluştu: {e}')
        return None

model, le, scaler = load_model_and_utils()

st.title('Klasik Müzik Besteci Tanıma')
st.write('Bir ses dosyası yükleyin, model tahminini göstersin.')

uploaded_file = st.file_uploader('Bir .wav dosyası yükleyin', type=['wav'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name
    pred_label = process_audio(temp_audio_path)
    if pred_label:
        st.success(f'Tahmin edilen besteci: {pred_label}')
    try:
        os.remove(temp_audio_path)
    except Exception:
        pass 