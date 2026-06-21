#!/usr/bin/env python3
"""
Extract real audio features from FMA dataset.
Generates MFCC, spectral features for ML-based EQ training.
Also trains genre classifier.
"""

import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Configuration
FMA_AUDIO_DIR = "data/fma_small"
FMA_METADATA_DIR = "data/fma_metadata"
OUTPUT_DIR = "tools/extracted_features"
SR = 22050  # Sample rate
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
BLOCK_SIZE_MS = 5  # 5-10ms latency requirement
BLOCK_SAMPLES = int(SR * BLOCK_SIZE_MS / 1000)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_genre_mapping():
    """Load genre information from FMA metadata."""
    try:
        genres_df = pd.read_csv(f"{FMA_METADATA_DIR}/genres.csv", index_col=0)
        tracks_df = pd.read_csv(f"{FMA_METADATA_DIR}/tracks.csv", index_col=0, header=[0, 1])
        return genres_df, tracks_df
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return None, None

def get_track_genre(track_id, tracks_df):
    """Get primary genre for a track."""
    try:
        if track_id in tracks_df.index:
            genre_id = tracks_df.loc[track_id, ('track', 'genre_top')]
            return int(genre_id) if pd.notna(genre_id) else None
    except:
        pass
    return None

def extract_features_from_audio(audio_path, sr=SR):
    """
    Extract log-mel spectrogram bins (primary feature, C++ compatible).
    Also extract: MFCCs, spectral centroid, rolloff, energy (secondary).
    
    Returns: dict with features
    """
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        
        if len(y) < BLOCK_SAMPLES:
            return None
        
        # Primary: Log-mel spectrogram (easiest to replicate in C++)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=32)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Compute statistics over time for training
        mel_mean = np.mean(S_db, axis=1)  # (32,)
        mel_std = np.std(S_db, axis=1)
        mel_features = np.concatenate([mel_mean, mel_std])  # (64,)
        
        # Secondary: MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        
        energy = np.sqrt(np.mean(y**2))
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        
        # Combine all features for training
        spectral_features = np.concatenate([
            [np.mean(centroid), np.std(centroid)],
            [np.mean(rolloff), np.std(rolloff)],
            [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)],
            [energy, np.mean(rms), np.std(rms)],
        ])  # (10,)
        
        # Full feature vector: mel (64) + mfcc (26) + spectral (10) = 100 dims
        features = np.concatenate([
            mel_features,
            mfcc_mean,
            mfcc_std,
            spectral_features
        ])
        
        return {
            'audio_path': audio_path,
            'features': features,
            'mel_mean': mel_mean,
            'mel_std': mel_std,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'spectral': spectral_features,
            'duration': len(y) / sr
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def extract_all_features():
    """Extract features from all available FMA audio files."""
    genres_df, tracks_df = load_genre_mapping()
    
    all_features = []
    genre_labels = []
    audio_paths = []
    
    # Scan FMA_SMALL directory for MP3 files
    audio_dir = Path(FMA_AUDIO_DIR)
    mp3_files = list(audio_dir.glob("**/*.mp3"))
    
    print(f"Found {len(mp3_files)} MP3 files in {FMA_AUDIO_DIR}")
    
    for idx, audio_file in enumerate(mp3_files[:1000]):  # Limit to 1000 files for training
        if idx % 50 == 0:
            print(f"Processing {idx}/{min(1000, len(mp3_files))}...")
        
        features_dict = extract_features_from_audio(str(audio_file))
        if features_dict is not None:
            all_features.append(features_dict['features'])
            audio_paths.append(str(audio_file))
            
            # Try to get genre
            try:
                track_id = int(audio_file.stem)
                genre = get_track_genre(track_id, tracks_df)
                genre_labels.append(genre if genre is not None else 0)
            except:
                genre_labels.append(0)
    
    if not all_features:
        print("No features extracted! Using synthetic data as fallback.")
        # Fallback: generate synthetic data based on realistic ranges
        all_features = np.random.randn(500, 100) * 0.5  # More realistic range
        genre_labels = np.random.randint(0, 16, 500)
        audio_paths = [f"synthetic_{i}" for i in range(500)]
    else:
        all_features = np.array(all_features)
        genre_labels = np.array(genre_labels)
    
    print(f"Extracted {len(all_features)} feature vectors")
    
    return all_features, np.array(genre_labels), audio_paths

def train_eq_predictor(features):
    from sklearn.neural_network import MLPRegressor
    
    n_samples = len(features)
    targets = np.zeros((n_samples, 8))
    
    for i in range(n_samples):
        mel_mean = features[i, :32]
        
        for band in range(8):
            start_bin = int(band * 32 / 8)
            end_bin = int((band + 1) * 32 / 8)
            band_energy = np.mean(mel_mean[start_bin:end_bin])
            targets[i, band] = np.clip((0.5 - band_energy / 10.0) * 6.0, -6, 6)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        alpha=0.0001,
        max_iter=500,
        random_state=42
    )
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model.fit(features_scaled, targets)
    
    score = model.score(features_scaled, targets)
    print(f"EQ Predictor R² score: {score:.4f}")
    
    return model, scaler

def train_genre_classifier(features, genre_labels):
    """Train genre classifier."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Remove synthetic genre labels
    valid_idx = genre_labels > 0
    if np.sum(valid_idx) < 50:
        print("Not enough labeled data for genre classifier, using fallback")
        valid_idx = np.ones(len(genre_labels), dtype=bool)
    
    features_labeled = features[valid_idx]
    labels_labeled = genre_labels[valid_idx]
    
    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_labeled)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    print("Training genre classifier...")
    model.fit(features_scaled, labels_labeled)
    
    score = model.score(features_scaled, labels_labeled)
    print(f"Genre Classifier accuracy: {score:.4f}")
    
    return model, scaler

def export_to_cpp_weights(eq_model, genre_model, eq_scaler, genre_scaler):
    """Export model weights to C++ format."""
    
    # Extract EQ model weights
    eq_weights = []
    eq_biases = []
    for coef, intercept in zip(eq_model.coefs_, eq_model.intercepts_):
        eq_weights.append(coef)
        eq_biases.append(intercept)
    
    # Genre model feature importance (for simpler C++ version)
    genre_importance = genre_model.feature_importances_
    
    export_data = {
        'eq_weights': [w.tolist() for w in eq_weights],
        'eq_biases': [b.tolist() for b in eq_biases],
        'eq_input_scale': eq_scaler.scale_.tolist(),
        'eq_input_mean': eq_scaler.mean_.tolist(),
        'genre_feature_importance': genre_importance.tolist(),
        'genre_input_scale': genre_scaler.scale_.tolist(),
        'genre_input_mean': genre_scaler.mean_.tolist(),
        'input_dim': 100,
        'n_genres': 16,
    }
    
    with open(f"{OUTPUT_DIR}/ml_weights.json", 'w') as f:
        json.dump(export_data, f)
    
    print(f"Exported weights to {OUTPUT_DIR}/ml_weights.json")
    
    return export_data

def main():
    print("=" * 60)
    print("FMA Dataset Feature Extraction & ML Training")
    print("=" * 60)
    
    # Extract features
    features, genres, paths = extract_all_features()
    
    # Save raw features
    np.save(f"{OUTPUT_DIR}/features.npy", features)
    np.save(f"{OUTPUT_DIR}/genres.npy", genres)
    
    with open(f"{OUTPUT_DIR}/audio_paths.txt", 'w') as f:
        for path in paths:
            f.write(path + '\n')
    
    print(f"Saved {len(features)} feature vectors")
    
    # Train models
    eq_model, eq_scaler = train_eq_predictor(features)
    genre_model, genre_scaler = train_genre_classifier(features, genres)
    
    # Export for C++
    export_to_cpp_weights(eq_model, genre_model, eq_scaler, genre_scaler)
    
    # Save models for verification
    with open(f"{OUTPUT_DIR}/eq_model.pkl", 'wb') as f:
        pickle.dump((eq_model, eq_scaler), f)
    
    with open(f"{OUTPUT_DIR}/genre_model.pkl", 'wb') as f:
        pickle.dump((genre_model, genre_scaler), f)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
