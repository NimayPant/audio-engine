#!/usr/bin/env python3
"""
Verification script: Run Python model inference and export outputs.
C++ will load the same test inputs and compare outputs.
"""

import numpy as np
import librosa
import json
import pickle
from pathlib import Path
import struct

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
OUTPUT_DIR = "tools/extracted_features"

def extract_features_for_inference(audio_path, sr=SR):
    """Extract features matching C++ implementation."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Log-mel spectrogram (primary feature)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=32)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Statistics
    mel_mean = np.mean(S_db, axis=1)
    mel_std = np.std(S_db, axis=1)
    mel_features = np.concatenate([mel_mean, mel_std])
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    energy = np.sqrt(np.mean(y**2))
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    
    spectral_features = np.concatenate([
        [np.mean(centroid), np.std(centroid)],
        [np.mean(rolloff), np.std(rolloff)],
        [np.mean(zcr), np.std(zcr)],
        [energy, np.mean(rms), np.std(rms)],
    ])
    
    features = np.concatenate([
        mel_features,
        mfcc_mean,
        mfcc_std,
        spectral_features
    ])
    
    return features.astype(np.float32)

def run_python_inference(test_audio_path):
    """Run inference in Python and export outputs."""
    
    # Load models
    with open(f"{OUTPUT_DIR}/eq_model.pkl", 'rb') as f:
        eq_model, eq_scaler = pickle.load(f)
    
    with open(f"{OUTPUT_DIR}/genre_model.pkl", 'rb') as f:
        genre_model, genre_scaler = pickle.load(f)
    
    # Extract features
    features = extract_features_for_inference(test_audio_path)
    features_scaled = eq_scaler.transform(features.reshape(1, -1))[0]
    
    # Run EQ inference
    eq_gains = eq_model.predict(features.reshape(1, -1))[0]  # 8 values
    
    # Run genre inference
    features_genre_scaled = genre_scaler.transform(features.reshape(1, -1))[0]
    genre_probs = genre_model.predict_proba(features.reshape(1, -1))[0]  # 16 probabilities
    genre_id = genre_model.predict(features.reshape(1, -1))[0]
    
    # Export to binary format for C++ comparison
    export_data = {
        'features': features,
        'features_scaled': features_scaled,
        'eq_gains': eq_gains,
        'genre_probs': genre_probs,
        'genre_id': int(genre_id),
    }
    
    # Save as JSON (human-readable)
    with open(f"{OUTPUT_DIR}/inference_output.json", 'w') as f:
        json.dump({
            'features_shape': list(features.shape),
            'features_sample': features[:10].tolist(),
            'eq_gains': eq_gains.tolist(),
            'genre_id': int(genre_id),
            'genre_probs_top3': [
                (int(genre_model.classes_[i]), float(genre_probs[i]))
                for i in np.argsort(genre_probs)[-3:][::-1]
            ]
        }, f, indent=2)
    
    # Save as binary for strict comparison
    with open(f"{OUTPUT_DIR}/inference_output.bin", 'wb') as f:
        # Features (100 floats)
        f.write(struct.pack(f'{len(features)}f', *features))
        # EQ gains (8 floats)
        f.write(struct.pack('8f', *eq_gains))
        # Genre probabilities (16 floats)
        f.write(struct.pack('16f', *genre_probs))
        # Genre ID (1 int)
        f.write(struct.pack('I', genre_id))
    
    print("Python inference complete!")
    print(f"EQ Gains: {eq_gains}")
    print(f"Genre ID: {genre_id} (probabilities: {np.max(genre_probs):.4f})")
    print(f"Exported to {OUTPUT_DIR}/inference_output.*")
    
    return export_data

def main():
    import sys
    
    # Use first available audio file, or accept as argument
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
    else:
        # Find first MP3 in dataset
        audio_files = list(Path("data/fma_small").glob("**/*.mp3"))
        if not audio_files:
            print("No audio files found!")
            return
        test_audio = str(audio_files[0])
    
    print(f"Running inference on: {test_audio}")
    run_python_inference(test_audio)

if __name__ == "__main__":
    main()
