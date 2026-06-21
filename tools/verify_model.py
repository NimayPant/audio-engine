import numpy as np
import librosa
import sys
import os
import torch

def compute_spectral_bins(block, sr=44100, n_fft=1024, numBins=16):
    S = np.abs(librosa.stft(block, n_fft=n_fft, hop_length=n_fft, center=False))
    mags = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyq = sr / 2.0
    log_min = np.log2(20.0)
    log_max = np.log2(nyq)
    bins = []
    for b in range(numBins):
        freq_lo = 2.0 ** (log_min + (log_max - log_min) * b / numBins)
        freq_hi = 2.0 ** (log_min + (log_max - log_min) * (b + 1) / numBins)
        idx_lo = np.searchsorted(freqs, freq_lo)
        idx_hi = np.searchsorted(freqs, freq_hi)
        idx_lo = max(0, min(idx_lo, len(mags)-1))
        idx_hi = max(idx_lo+1, min(idx_hi, len(mags)))
        val = mags[idx_lo:idx_hi].mean() if idx_hi > idx_lo else 0.0
        bins.append(float(val))
    return np.array(bins)

def features_from_wav(path, sr=44100, block_size=1024):
    y, fs = librosa.load(path, sr=sr, mono=True)
    feats = []
    hop = block_size
    for i in range(0, len(y) - block_size + 1, hop):
        block = y[i:i+block_size]
        bins = compute_spectral_bins(block, sr=sr, n_fft=block_size, numBins=16)
        peak = float(np.max(np.abs(block)))
        rms = float(np.sqrt(np.mean(block**2) + 1e-12))
        crest = float(peak / (rms + 1e-12))
        feat = np.concatenate([bins, [peak, rms, crest]])
        feats.append(feat)
    return np.vstack(feats)

def load_weights(header_path):
    # Robust parser: find C-style float array initializers in header
    import re
    txt = open(header_path, 'r', encoding='utf-8').read()
    pattern = re.compile(r'static\s+constexpr\s+float\s+([A-Za-z0-9_]+)\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}', re.M)
    arrays = {}
    for m in pattern.finditer(txt):
        name = m.group(1)
        body = m.group(2)
        # sanitize common C++ float suffixes and whitespace
        body_clean = body.replace('f', ' ')
        body_clean = body_clean.replace('\n', ' ')
        body_clean = body_clean.replace('\r', ' ')
        # use numpy.fromstring with comma separator
        try:
            arr = np.fromstring(body_clean, sep=',', dtype=np.float32)
        except Exception:
            vals = [float(x) for x in body_clean.split(',') if x.strip()!='']
            arr = np.array(vals, dtype=np.float32)
        arrays[name] = arr
    return arrays

def run_mlp(feat, weights):
    # weights: dict mapping array names -> numpy arrays
    # infer layer dims from known architecture 19->32->16->2 using MLWeights naming
    W0 = weights.get('kCompW0')
    B0 = weights.get('kCompB0')
    W1 = weights.get('kCompW1')
    B1 = weights.get('kCompB1')
    W2 = weights.get('kCompW2')
    B2 = weights.get('kCompB2')
    if W0 is None or B0 is None or W1 is None or B1 is None or W2 is None or B2 is None:
        raise RuntimeError('Could not find expected arrays in header')
    # ensure arrays have expected sizes (truncate if extra commas/values slipped through parsing)
    if W0.size < 32*19:
        raise RuntimeError('kCompW0 too small')
    W0 = W0[:32*19].reshape(32,19)
    if B0.size < 32:
        raise RuntimeError('kCompB0 too small')
    B0 = B0[:32]
    if W1.size < 16*32:
        raise RuntimeError('kCompW1 too small')
    W1 = W1[:16*32].reshape(16,32)
    if B1.size < 16:
        raise RuntimeError('kCompB1 too small')
    B1 = B1[:16]
    if W2.size < 2*16:
        raise RuntimeError('kCompW2 too small')
    W2 = W2[:2*16].reshape(2,16)
    B2 = B2[:2]
    h0 = np.maximum(0, W0.dot(feat) + B0)
    h1 = np.maximum(0, W1.dot(h0) + B1)
    out = W2.dot(h1) + B2
    # clamp as C++ does
    out[0] = max(-60.0, min(0.0, out[0]))
    out[1] = max(1.0, min(20.0, out[1]))
    return out

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('wav')
    parser.add_argument('header')
    parser.add_argument('--export', help='path to save features+output (.npz)', default=None)
    args = parser.parse_args()

    wav = args.wav
    header = args.header
    feats = features_from_wav(wav)
    weights = load_weights(header)
    print('Parsed weight arrays:')
    for k,v in weights.items():
        print('  ', k, 'len=', v.size)
    if 'kCompW0' in weights:
        print('kCompW0 tail:', weights['kCompW0'][-10:])
    # run on first frame
    out = run_mlp(feats[0], weights)
    print('Python prediction:', out)
    if args.export:
        np.savez(args.export, features=feats[0].astype(np.float32), output=out.astype(np.float32))
        txtpath = os.path.splitext(args.export)[0] + '.txt'
        with open(txtpath, 'w') as wf:
            wf.write(' '.join(map(str, feats[0].astype(np.float32).tolist())) + '\n')
            wf.write(' '.join(map(str, out.astype(np.float32).tolist())) + '\n')
        print('Saved verification data to', args.export, 'and', txtpath)

if __name__ == "__main__":
    main()
