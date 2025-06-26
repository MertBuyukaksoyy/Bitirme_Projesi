# src/data_loader.py
import librosa
import numpy as np


def load_audio(path, sr=22050):
    y, _ = librosa.load(path, sr=sr)
    return y


def segment_audio(y, sr, segment_duration=5):
    segment_samples = segment_duration * sr
    segments = []

    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        if end <= len(y):
            segments.append(y[start:end])

    return segments


def extract_mfcc(segment, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T
