import os
import numpy as np
import librosa
import keras
import pickle
import random
from main import extract_features

model = keras.models.load_model('results/best_dense_model.keras')
with open('results/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('results/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

segment_duration = 20
try:
    with open('results/results.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if 'Kullanılan segment süresi' in line:
                segment_duration = float(line.strip().split(':')[-1])
except Exception:
    pass

csv_path = 'datasett/datasett/maestro-v3.0.0.csv'
audio_base_path = 'datasett/datasett'
import pandas as pd
df = pd.read_csv(csv_path)
df = df[df['split'] == 'train']
df = df.dropna(subset=['canonical_composer'])
grouped = df.groupby('canonical_composer')
files = []
labels = []
for composer, group in grouped:
    for _, row in group.iterrows():
        audio_path = os.path.join(audio_base_path, str(row['audio_filename']))
        if os.path.isfile(audio_path):
            files.append(audio_path)
            labels.append(composer)

test_indices = random.sample(range(len(files)), 10)
print(f"Rastgele 10 test örneğiyle tahminler (segment süresi: {segment_duration} sn):")
for idx in test_indices:
    file_path = files[idx]
    true_label = labels[idx]
    try:
        feat = extract_features(file_path, augment=False)
        feat = feat.reshape(1, -1)
        feat = scaler.transform(feat)
        pred_idx = np.argmax(model.predict(feat, verbose=0), axis=1)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        print(f"Dosya: {os.path.basename(file_path)} | Gerçek: {true_label} | Tahmin: {pred_label}")
    except Exception as e:
        print(f"Hata: {file_path} - {e}") 