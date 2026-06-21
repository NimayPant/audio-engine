import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import soundfile as sf
import librosa
import urllib.request
import zipfile
import tarfile
import tempfile
import shutil
import time


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super(MLP, self).__init__()
        layers = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)


def export_to_cpp(model, name):
    weights, biases = [], []
    for layer in model.model:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy())
            biases.append(layer.bias.detach().cpu().numpy())

    out_lines = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        flat_w = ', '.join(map(str, w.flatten().tolist()))
        flat_b = ', '.join(map(str, b.flatten().tolist()))
        out_lines.append(f'static constexpr float {name}_W{i}[] = {{ {flat_w} }};')
        out_lines.append(f'static constexpr float {name}_B{i}[] = {{ {flat_b} }};')
    return '\n'.join(out_lines)


def compute_features_from_wav(path, sr=44100, block_size=1024, feature_mode='cpp'):
    """
    Compute features from a wav/mp3 file.
    feature_mode: 'cpp' -> 16 spectral bins + peak/rms/crest (19 dims)
                  'mfcc' -> 16 MFCCs + peak/rms/crest (19 dims) -- Python-only model
    """
    y, fs = librosa.load(path, sr=sr, mono=True)
    features = []
    hop = block_size
    for i in range(0, len(y) - block_size + 1, hop):
        block = y[i:i+block_size]
        peak = float(np.max(np.abs(block)))
        rms = float(np.sqrt(np.mean(block**2) + 1e-12))
        crest = float(peak / (rms + 1e-12))

        if feature_mode == 'mfcc':
            # compute MFCCs per block (n_mfcc=16 to keep 19-dim with peak/rms/crest)
            mfccs = librosa.feature.mfcc(y=block, sr=sr, n_mfcc=16, n_fft=block_size, hop_length=block_size, center=False)
            # mfccs shape (n_mfcc, 1) -> flatten
            mf = mfccs.mean(axis=1).tolist()
            feat = mf + [peak, rms, crest]
        else:
            # default: compute spectral bins similar to C++ extractor
            S = np.abs(librosa.stft(block, n_fft=block_size, hop_length=block_size, center=False))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=block_size)
            mags = S.mean(axis=1)
            nyq = sr / 2.0
            log_min = np.log2(20.0)
            log_max = np.log2(nyq)
            bins = []
            numBins = 16
            for b in range(numBins):
                freq_lo = 2.0 ** (log_min + (log_max - log_min) * b / numBins)
                freq_hi = 2.0 ** (log_min + (log_max - log_min) * (b + 1) / numBins)
                idx_lo = np.searchsorted(freqs, freq_lo)
                idx_hi = np.searchsorted(freqs, freq_hi)
                idx_lo = max(0, min(idx_lo, len(mags)-1))
                idx_hi = max(idx_lo+1, min(idx_hi, len(mags)))
                val = mags[idx_lo:idx_hi].mean() if idx_hi > idx_lo else 0.0
                bins.append(float(val))
            feat = bins + [peak, rms, crest]

        features.append(feat)
    return np.array(features)


def build_dataset_from_folder(folder, sr=44100, block_size=1024, max_files=200, feature_mode='cpp'):
    X_list = []
    # recursively gather wav/mp3 files
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith(('.wav', '.mp3')):
                files.append(os.path.join(root, fn))
    for i, fpath in enumerate(files[:max_files]):
        try:
            feats = compute_features_from_wav(fpath, sr=sr, block_size=block_size, feature_mode=feature_mode)
            if feats.shape[0] > 0:
                X_list.append(feats)
        except Exception as e:
            print('Failed to process', fpath, e)
    if not X_list:
        raise RuntimeError('No audio data found in folder')
    X = np.vstack(X_list)
    return X


def download_and_prepare(data_folder='data', max_files=400, sr=44100, block_size=1024):
    os.makedirs(data_folder, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix='dsp_data_')
    print('Downloading datasets into', tmpdir)

    fma_url = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
    musan_url = 'http://www.openslr.org/resources/17/musan.tar.gz'

    fma_zip = os.path.join(tmpdir, 'fma_small.zip')
    musan_tar = os.path.join(tmpdir, 'musan.tar.gz')

    try:
        print('Downloading FMA small...')
        urllib.request.urlretrieve(fma_url, fma_zip)
        print('Extracting FMA...')
        with zipfile.ZipFile(fma_zip, 'r') as zf:
            zf.extractall(tmpdir)
    except Exception as e:
        print('FMA download/extract failed:', e)

    try:
        print('Downloading MUSAN...')
        urllib.request.urlretrieve(musan_url, musan_tar)
        print('Extracting MUSAN...')
        with tarfile.open(musan_tar, 'r:gz') as tf:
            tf.extractall(tmpdir)
    except Exception as e:
        print('MUSAN download/extract failed:', e)

    # collect audio files from extracted folders
    collected = []
    for root, _, filenames in os.walk(tmpdir):
        for fn in filenames:
            if fn.lower().endswith(('.wav', '.mp3')):
                collected.append(os.path.join(root, fn))

    if not collected:
        shutil.rmtree(tmpdir)
        raise RuntimeError('No audio files found in downloaded datasets')

    # sample files: prefer music (fma) first then musan
    music_files = [p for p in collected if 'fma' in p.lower() or 'music' in p.lower()]
    other_files = [p for p in collected if p not in music_files]

    target = []
    target.extend(music_files[:max_files//2])
    remaining = max_files - len(target)
    target.extend(other_files[:remaining])

    # copy selected files into data_folder
    print(f'Copying {len(target)} files into {data_folder} ...')
    for src in target:
        try:
            dst = os.path.join(data_folder, os.path.basename(src))
            shutil.copy2(src, dst)
        except Exception as e:
            print('Failed to copy', src, e)

    # clean up tmpdir
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    print('Dataset prepared in', data_folder)


def train_compressor(data_folder='data', out_name='comp_model', block_size=1024, feature_mode='cpp'):
    X = build_dataset_from_folder(data_folder, block_size=block_size, feature_mode=feature_mode)
    
    peak = X[:, 16]
    rms = X[:, 17]
    norm_peak = (peak - peak.min()) / (peak.ptp() + 1e-9)
    threshold = -20.0 - 10.0 * norm_peak
    
    crest = X[:, 18]
    norm_crest = (crest - crest.min()) / (crest.ptp() + 1e-9)
    ratio = 2.0 + 4.0 * norm_crest

    y = np.stack([threshold, ratio], axis=1)

    # Convert to torch
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))

    model = MLP(19, [32, 16], 2)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(50):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f'Epoch {epoch+1}: loss={total_loss/len(dataset):.6f}')

    cpp_weights = export_to_cpp(model, out_name)
    with open(f'../include/MLWeights_{out_name}.h', 'w') as wf:
        wf.write('// Auto-generated weights\n')
        wf.write(cpp_weights)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', help='data folder')
    parser.add_argument('--out', default='comp_model', help='output model name')
    parser.add_argument('--feature_mode', choices=['cpp','mfcc'], default='cpp', help='feature extraction mode')
    parser.add_argument('--block_size', type=int, default=1024, help='block size for frame features')
    args = parser.parse_args()

    data_folder = args.data
    # prepare dataset if missing
    if not os.path.exists(data_folder) or not any(fn.lower().endswith(('.wav', '.mp3')) for _, _, files in os.walk(data_folder) for fn in files):
        print('No data/ folder with audio found. Downloading sample datasets (FMA small + MUSAN)...')
        download_and_prepare(data_folder, max_files=300, block_size=args.block_size)

    comp_model = train_compressor(data_folder, out_name=args.out, block_size=args.block_size, feature_mode=args.feature_mode)
    print("Training complete.")

    # Exported header is written to include/MLWeights_comp_model.h — we can delete raw audio to save space
    try:
        print('Removing raw audio data folder to save space...')
        shutil.rmtree(data_folder)
    except Exception as e:
        print('Failed to remove data folder:', e)
