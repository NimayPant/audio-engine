#!/usr/bin/env python3
"""
Train Compressor and Gate Models on FMA Dataset

Trains ML models specifically optimized for:
1. CompressorPredictor: Maps audio features -> threshold & ratio
2. NoiseClassifier: Maps audio features -> signal vs noise probability

Uses real Free Music Archive audio for robust, production-ready models.
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

FMA_SMALL_PATH = Path("data/fma_small")
METADATA_PATH = Path("data/fma_metadata")
OUTPUT_DIR = Path("tools/extracted_features")
SAMPLE_RATE = 22050
MAX_FILES = 50

print("\n" + "="*70)
print(" FMA-Based Compressor & Gate Model Training")
print("="*70)

def extract_compressor_features(audio):
    """Extract 19-dimensional features for compressor training."""
    S = np.abs(librosa.stft(audio, n_fft=2048))
    # Mel spectrogram (16 bins)
    mel = librosa.feature.melspectrogram(S=S, sr=SAMPLE_RATE, n_mels=16)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)
    # Peak and RMS
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    crest = peak / (rms + 1e-6)
    features = np.concatenate([
        mel_mean,      # 16 features
        [peak, rms, crest]  # 3 features
    ])
    return features  # 19 features total

def compute_optimal_compressor_params(audio, sr=SAMPLE_RATE):
    """
    Compute 'optimal' compressor parameters using energy analysis.
    Used as training targets.
    Returns: (threshold_db, ratio)
    """
    S = np.abs(librosa.stft(audio, n_fft=2048))
    spectrogram = np.mean(S, axis=1)
    # Threshold = mean + 0.5 * std (adaptive)
    mean_power = np.mean(spectrogram)
    std_power = np.std(spectrogram)
    threshold_db = 20 * np.log10(mean_power + 0.5 * std_power + 1e-6)
    threshold_db = np.clip(threshold_db, -60, 0)
    # Ratio based on spectral density
    spectral_density = np.sum(spectrogram)
    ratio = 2.0 + 3.0 * np.clip(spectral_density / np.max(spectrogram + 1e-6), 0, 1)
    ratio = np.clip(ratio, 1.0, 20.0)
    return float(threshold_db), float(ratio)

print("\nExtracting compressor training data from FMA...")
comp_features = []
comp_thresholds = []
comp_ratios = []
mp3_files = list(FMA_SMALL_PATH.glob("*/*.mp3"))[:MAX_FILES]
print(f"Found {len(mp3_files)} audio files")
for idx, audio_file in enumerate(mp3_files):
    if idx % 100 == 0:
        print(f"  Processing: {idx}/{len(mp3_files)}", end='\r')
    try:
        audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, duration=10)
        # Extract features for multiple segments
        segment_length = int(sr * 0.5)  # 500ms segments
        for start in range(0, len(audio) - segment_length, segment_length // 2):
            segment = audio[start:start + segment_length]
            feats = extract_compressor_features(segment)
            thresh, ratio = compute_optimal_compressor_params(segment, sr)
            comp_features.append(feats)
            comp_thresholds.append(thresh)
            comp_ratios.append(ratio)
    except Exception as e:
        continue
comp_features = np.array(comp_features)
comp_thresholds = np.array(comp_thresholds)
comp_ratios = np.array(comp_ratios)
print(f"\nExtracted {len(comp_features)} compressor training samples")
print(f"Threshold range: {comp_thresholds.min():.2f} to {comp_thresholds.max():.2f} dB")
print(f"Ratio range: {comp_ratios.min():.2f} to {comp_ratios.max():.2f}")
print("\nTraining Compressor Predictor...")
# Normalize features
scaler_comp = StandardScaler()
comp_features_norm = scaler_comp.fit_transform(comp_features)
# Train MLP for threshold and ratio
comp_model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
y_comp = np.column_stack([comp_thresholds, comp_ratios])
comp_model.fit(comp_features_norm, y_comp)
print(f"Compressor model trained")
print(f"R² Score: {comp_model.score(comp_features_norm, y_comp):.4f}")
# Extract weights for C++
comp_weights = []
for layer in comp_model.coefs_:
    comp_weights.extend(layer.flatten())
comp_biases = []
for bias in comp_model.intercepts_:
    comp_biases.extend(bias)
print(f"  Weights extracted: {len(comp_weights)} coefficients")

def extract_noise_features(audio):
    """Extract 18-dimensional features for noise classification."""
    S = np.abs(librosa.stft(audio, n_fft=2048))
    # Mel spectrogram (16 bins)
    mel = librosa.feature.melspectrogram(S=S, sr=SAMPLE_RATE, n_mels=16)
    mel_mean = np.mean(mel, axis=1)
    # Zero-crossing rate (voice/percussion indicator)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    # Spectral flatness (noise vs tonality)
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio)[0])
    features = np.concatenate([
        mel_mean,  # 16 features
        [zcr, flatness]  # 2 features
    ])
    return features  # 18 features total

def get_noise_label(audio, sr=SAMPLE_RATE):
    """
    Estimate if audio is noise or signal.
    Returns: probability [0, 1] where 1 = pure noise, 0 = pure signal
    """
    S = np.abs(librosa.stft(audio, n_fft=2048))
    spectrogram = np.mean(S, axis=1)
    # Spectral flatness: high = noise, low = signal
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
    # Zero-crossing rate: high = noise/percussion, low = signal
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    # Combine indicators
    noise_prob = 0.5 * flatness + 0.3 * zcr
    noise_prob = np.clip(noise_prob, 0, 1)
    return noise_prob

print("\n Extracting noise classifier training data from FMA...")
noise_features = []
noise_labels = []
for idx, audio_file in enumerate(mp3_files):
    if idx % 100 == 0:
        print(f"  Processing: {idx}/{len(mp3_files)}", end='\r')
    try:
        audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, duration=10)
        # Extract features for multiple segments
        segment_length = int(sr * 0.5)  # 500ms segments
        for start in range(0, len(audio) - segment_length, segment_length // 2):
            segment = audio[start:start + segment_length]
            feats = extract_noise_features(segment)
            label = get_noise_label(segment, sr)
            noise_features.append(feats)
            noise_labels.append(label)
    except Exception as e:
        continue
noise_features = np.array(noise_features)
noise_labels = np.array(noise_labels)
print(f"\n Extracted {len(noise_features)} noise classification training samples")
print(f"  Noise probability range: {noise_labels.min():.2f} to {noise_labels.max():.2f}")
print("\n Training Noise Classifier...")
# Normalize features
scaler_noise = StandardScaler()
noise_features_norm = scaler_noise.fit_transform(noise_features)
# Train MLP classifier
noise_model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
noise_model.fit(noise_features_norm, noise_labels)
print(f"Noise classifier trained")
print(f"R2 Score: {noise_model.score(noise_features_norm, noise_labels):.4f}")
# Extract weights for C++
noise_weights = []
for layer in noise_model.coefs_:
    noise_weights.extend(layer.flatten())
noise_biases = []
for bias in noise_model.intercepts_:
    noise_biases.extend(bias)
print(f"  Weights extracted: {len(noise_weights)} coefficients")

print("\nGenerating C++ MLWeights.h with FMA-trained models")
# Build compressor weight arrays
comp_layer_sizes = [(19, 32), (32, 16), (16, 2)]  # Input, hidden1, hidden2, output
comp_weight_strs = []
comp_bias_strs = []
idx = 0
for layer_idx, (in_size, out_size) in enumerate(comp_layer_sizes):
    w = comp_weights[idx:idx + out_size * in_size]
    idx += len(w)
    comp_weight_strs.append(f"const float kCompW{layer_idx}[{len(w)}] = {{{', '.join(f'{x:.6f}f' for x in w)}}};\n")
idx = 0
for layer_idx, bias in enumerate(comp_model.intercepts_):
    comp_bias_strs.append(f"const float kCompB{layer_idx}[{len(bias)}] = {{{', '.join(f'{x:.6f}f' for x in bias)}}};\n")
# Build noise weight arrays
noise_layer_sizes = [(18, 32), (32, 16), (16, 1)]
noise_weight_strs = []
noise_bias_strs = []
idx = 0
for layer_idx, (in_size, out_size) in enumerate(noise_layer_sizes):
    w = noise_weights[idx:idx + out_size * in_size]
    idx += len(w)
    noise_weight_strs.append(f"const float kNoiseW{layer_idx}[{len(w)}] = {{{', '.join(f'{x:.6f}f' for x in w)}}};\n")
idx = 0
for layer_idx, bias in enumerate(noise_model.intercepts_):
    noise_bias_strs.append(f"const float kNoiseB{layer_idx}[{len(bias)}] = {{{', '.join(f'{x:.6f}f' for x in bias)}}};\n")
with open("include/MLWeights.h", "w") as f:
    f.write("#ifndef MLWEIGHTS_H\n#define MLWEIGHTS_H\n\nnamespace ml_weights {\n\n")
    f.write("static constexpr int kCompL0_In = 19, kCompL0_Out = 32;\n")
    for s in comp_weight_strs: f.write(s)
    for s in comp_bias_strs: f.write(s)
    f.write("\nstatic constexpr int kCompL1_In = 32, kCompL1_Out = 16;\n")
    f.write("\nstatic constexpr int kCompL2_In = 16, kCompL2_Out = 2;\n")
    f.write("\nstatic constexpr int kNoiseL0_In = 18, kNoiseL0_Out = 24;\n")
    for s in noise_weight_strs: f.write(s)
    for s in noise_bias_strs: f.write(s)
    f.write("\nstatic constexpr int kNoiseL1_In = 24, kNoiseL1_Out = 12;\n")
    f.write("\nstatic constexpr int kNoiseL2_In = 12, kNoiseL2_Out = 1;\n")
    f.write("\n}\n#endif\n")
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
scalers = {
    'compressor': {
        'mean': scaler_comp.mean_.tolist(),
        'scale': scaler_comp.scale_.tolist(),
        'feature_dim': 19
    },
    'noise_classifier': {
        'mean': scaler_noise.mean_.tolist(),
        'scale': scaler_noise.scale_.tolist(),
        'feature_dim': 18
    }
}
with open(OUTPUT_DIR / 'ml_scalers.json', 'w') as f:
    json.dump(scalers, f, indent=2)
print("\n" + "="*70)
print("FMA-BASED MODEL TRAINING COMPLETE")
print("="*70)
print("\nTraining Summary:")
print(f"  Compressor Model:")
print(f"    - Training samples: {len(comp_features)}")
print(f"    - Architecture: 19 → 32 → 16 → 2 (threshold + ratio)")
print(f"    - Performance: R² = {comp_model.score(comp_features_norm, y_comp):.4f}")
print(f"    - Weights: {len(comp_weights)} coefficients")

print(f"\nNoise Classifier Model:")
print(f"    - Training samples: {len(noise_features)}")
print(f"    - Architecture: 18 → 32 → 16 → 1 (noise probability)")
print(f"    - Performance: R² = {noise_model.score(noise_features_norm, noise_labels):.4f}")
print(f"    - Weights: {len(noise_weights)} coefficients")

print("\nNext Steps:")
print("  1. Copy generated weights to include/MLWeights.h")
print("  2. Update weight definitions for compressor and noise models")
print("  3. Rebuild C++ project: cd build && cmake --build . --config Debug")
print("  4. Test with: Debug/verify_cpp.exe path/to/audio.wav")

print("\nResults saved to:")
print(f"  - {OUTPUT_DIR / 'ml_scalers.json'} (feature scalers)")
print(f"  - Ready to be integrated into MLWeights.h")

print("\n" + "="*70 + "\n")
