import os
import numpy as np
import pandas as pd
import librosa
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
from tqdm import tqdm
import pickle

def extract_features(file_path, sr=22050, n_mfcc=30, augment=False):
    y, sr = librosa.load(file_path, sr=sr, duration=15)

    if augment:
        if random.random() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
        if random.random() < 0.5:
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            y = y + np.random.normal(0, 0.005, len(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tempo = librosa.beat.tempo(y=y, sr=sr)

    features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1),
        np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
        np.mean(rms, axis=1), np.std(rms, axis=1),
        np.mean(zcr, axis=1), np.std(zcr, axis=1),
        np.mean(mel, axis=1), np.std(mel, axis=1),
        np.mean(centroid, axis=1), np.std(centroid, axis=1),
        np.mean(bandwidth, axis=1), np.std(bandwidth, axis=1),
        np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
        tempo
    ])
    return features

def prepare_dataset(csv_path, audio_base_path, augment=False, max_per_composer=30, min_per_composer=10):
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'train']
    df = df.dropna(subset=['canonical_composer'])
    grouped = df.groupby('canonical_composer')
    X, y = [], []
    for composer, group in grouped:
        files = []
        for _, row in group.iterrows():
            audio_path = os.path.join(audio_base_path, str(row['audio_filename']))
            if os.path.isfile(audio_path):
                files.append(audio_path)
        if len(files) < min_per_composer:
            continue
        random.shuffle(files)
        files = files[:max_per_composer]
        for audio_path in tqdm(files, desc=f"{composer}"):
            try:
                feat = extract_features(audio_path, augment=augment)
                X.append(feat)
                y.append(composer)
            except Exception as e:
                print(f"Feature extraction error: {audio_path} ({composer}) - {e}")
    return np.array(X), np.array(y)

def main():
    csv_path = 'datasett/datasett/maestro-v3.0.0.csv'
    audio_base_path = 'datasett/datasett'
    X, y = prepare_dataset(csv_path, audio_base_path, augment=True, max_per_composer=30, min_per_composer=10)
    print(f"Toplam örnek: {X.shape[0] if hasattr(X, 'shape') else len(X)}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    os.makedirs('results', exist_ok=True)
    with open('results/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    with open('results/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    X_test, y_test = prepare_dataset(csv_path, audio_base_path, augment=False, max_per_composer=30, min_per_composer=10)
    X_test = scaler.transform(X_test)
    y_test = le.transform(y_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70, batch_size=4, callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    ])
    test_acc = model.evaluate(X_test, y_test, verbose=1)[1]
    print(f"Test doğruluğu: {test_acc:.2%}")
    y_pred = np.argmax(model.predict(X_test, verbose=1), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(confusion_matrix(y_test, y_pred))
    model.save('results/best_dense_model.keras')
    with open('results/results.txt', 'w', encoding='utf-8') as f:
        f.write(f'Test doğruluğu: {test_acc:.2%}\n')
        f.write(str(classification_report(y_test, y_pred, target_names=le.classes_)))
        f.write("\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred)))

if __name__ == '__main__':
    main()