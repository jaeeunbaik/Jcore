import librosa
import numpy as np

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def normalize_audio(audio):
    return (audio - np.mean(audio)) / np.std(audio)

def pad_audio(audio, target_length):
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), 'constant')
    return audio[:target_length]