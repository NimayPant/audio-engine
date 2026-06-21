# Smart Audio DSP Engine & Legendary DSP VST

A highly optimized, ML-powered audio processing ecosystem built from scratch in C++ with a Node.js web interface and a JUCE-based VST3 plugin. I built this project to explore the intersection of SIMD hardware acceleration and machine learning in audio processing, demonstrating how a hybrid approach can achieve real-time latency with intelligent, genre-aware processing.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Performance & Benchmarks](#performance--benchmarks)
- [Effect Knobs and Capabilities](#effect-knobs-and-capabilities)
- [Current Downsides & Limitations](#current-downsides--limitations)
- [Future Implementations](#future-implementations)
- [How to Build and Run](#how-to-build-and-run)

## Architecture Overview
This project uses a hybrid architecture for audio processing:
1. **Frontend / API:** A Node.js Express server that handles file uploads (WAV/MP3), parses effect parameters, and dynamically sets up the environment.
2. **DSP Core (C++ CLI):** A highly optimized, SIMD-accelerated offline audio processing backend that operates on streams of floating-point audio data.
3. **ML Inference Engine:** I utilize a custom-built, lightweight machine learning inference engine in C++ (trained in Python using scikit-learn). It analyzes audio features (MFCCs, spectral roll-off, etc.) and predicts optimal settings for dynamic EQ, compression, and noise gating.
4. **VST3 Plugin (Legendary DSP):** A real-time audio plugin built with the JUCE framework, providing hardware-accelerated core DSP effects (EQ, Delay, Reverb, Gain) directly inside DAWs like FL Studio and Ableton.
5. **Format Handling:** Uses `libsndfile` for WAV and `dr_mp3` for MP3. Output MP3 encoding is seamlessly offloaded to `ffmpeg`.

## Tech Stack
- **Core Engine:** C++17
- **Hardware Acceleration:** AVX2 / SSE intrinsics
- **Audio Libraries:** libsndfile, dr_mp3
- **Plugin Framework:** JUCE 8
- **Web Interface:** Node.js, Express, Multer
- **Machine Learning:** Python, scikit-learn, librosa, numpy (for model training)

## Key Features
- **SIMD Processing:** Employs explicit AVX2 intrinsics to process 8 float samples simultaneously, dramatically improving throughput over scalar processing.
- **Custom ML Engine:** A dependency-free C++ inference engine that evaluates neural networks and classifiers using pre-trained weights without heavy external libraries.
- **Genre-Aware Processing:** Extracts standard audio features (such as Log-Mel Spectrogram and MFCCs) and classifies the genre to tailor the processing knobs to the specific style (Available via CLI).
- **Intelligent Dynamics:** ML-powered smart compressors and noise gates that dynamically adjust thresholds and ratios based on audio feature extraction.
- **Real-Time VST Plugin:** A clean, optimized VST3 plugin offering standard, real-time safe DSP effects (Biquad EQ, Delay, Reverb, Master Gain) with a custom hardware-accelerated backend.
- **Multi-Format Support:** Natively supports uploading and exporting both WAV and MP3 via the Node.js API, converting dynamically as required using a bundled FFMPEG.

## Performance & Benchmarks
I benchmarked the SIMD implementations against standard scalar implementations using a buffer of 1,048,576 samples over 100 iterations on an AVX2-capable CPU.

```text
================================================================
       DSP SIMD Benchmark Suite
================================================================
  Buffer size:  1048576 samples
  Iterations:   100
  CPU features: SSE2 SSE3 SSSE3 SSE4.1 AVX AVX2 FMA

+-----------------------------------------+
|         CPU SIMD Feature Detection      |
+-----------------------------------------+
|  SSE2        V Supported                 |
|  SSE3        V Supported                 |
|  SSSE3       V Supported                 |
|  SSE4.1      V Supported                 |
|  AVX         V Supported                 |
|  AVX2        V Supported                 |
|  FMA         V Supported                 |
+-----------------------------------------+
|  Selected path: AVX2 (8 floats/cycle)   |
+-----------------------------------------+

+-------------------------------------------------------------+
| Gain Application (data[i] *= gain)                          |
+---------------+--------------+---------------+--------------+
| Path          | Time (us)    | Throughput    | Speedup      |
+---------------+--------------+---------------+--------------+
| Scalar        |    1205.04  |   870.16 MS/s |     1.00x    |
| SSE2 (4-wide) |     757.16  |  1384.87 MS/s |     1.59x    |
| AVX2 (8-wide) |     620.70  |  1689.36 MS/s |     1.94x    |
+---------------+--------------+---------------+--------------+

+-------------------------------------------------------------+
| Peak/RMS Detection (envelope follower)                      |
+---------------+--------------+---------------+--------------+
| Path          | Time (us)    | Throughput    | Speedup      |
+---------------+--------------+---------------+--------------+
| Scalar        |    2545.28  |   411.97 MS/s |     1.00x    |
| SSE2 (4-wide) |     958.80  |  1093.63 MS/s |     2.65x    |
| AVX2 (8-wide) |     610.89  |  1716.46 MS/s |     4.17x    |
+---------------+--------------+---------------+--------------+
```
As shown, AVX2 optimization achieves up to a **4.17x speedup** for envelope following and a **1.94x speedup** for gain processing over a scalar approach.

## Effect Knobs and Capabilities
The system provides several parameters to manipulate the audio (accessible via CLI/Node.js or VST):
1. **`delay`**: Applies a simple delay line with feedback.
2. **`reverb`**: Applies an algorithmic comb-filter reverb.
3. **`eq` / `biquad-simd`**: Optimized parametric biquad filter.
4. **`gain-simd`**: Hardware-accelerated (AVX2) linear gain adjustment.
5. **`smart-compress`**: An ML-powered compressor that automatically adjusts parameters based on extracted features (Offline CLI only).
6. **`smart-gate`**: A machine-learning inspired noise gate that identifies noise floor signatures (Offline CLI only).
7. **`ml-dynamic-eq`**: ML feature extractor to rebalance the frequency spectrum dynamically (Offline CLI only).
8. **`genre-aware`**: Modifies the output by applying specific EQ and compression settings uniquely suited to the given genre classification (Offline CLI only).

## Current Downsides & Limitations
While the DSP engine is highly optimized, the current architecture has a few notable downsides:
1. **Offline ML Processing:** The ML-powered features (Smart Compression, Dynamic EQ, Genre Detection) are currently limited to offline batch processing via the Node.js API / C++ CLI. Because of the strict performance and lock-free requirements of real-time audio threads, running these directly in the VST's `processBlock` currently causes audio dropouts and distortion.
2. **High Memory Footprint in API:** The Node.js API processes audio offline by loading entire `.wav` files into RAM simultaneously. For very large audio files or high concurrency, this can cause significant memory bottlenecks.
3. **Lack of Streaming:** The web interface does not currently support streaming audio. Users must wait for the entire file to process and download before hearing the result.

## Future Implementations
To address these downsides and expand the project's capabilities, I plan to implement the following in the future:
1. **Asynchronous ML in VST:** I intend to bring the smart ML features back to the VST plugin by decoupling the inference engine from the audio thread. By utilizing a background worker thread and lock-free ring buffers (FIFOs), the VST will be able to asynchronously calculate ML targets and smoothly interpolate the DSP parameters in real-time without glitching the audio.
2. **Audio Streaming API:** Replacing the batch-processing API with a WebSocket or WebRTC streaming architecture, allowing the Node.js server to process and return audio chunks on the fly.
3. **Advanced ML Models:** Integrating more advanced models (e.g., using RTNeural for extremely lightweight neural amp modeling) and expanding the genre detection dataset.
4. **GUI Enhancements:** Expanding the JUCE UI with real-time spectrum analyzers and dynamic ML target visualizations.

## How to Build and Run

### Prerequisites
- Node.js & NPM
- Python 3.8+ (for training ML models or extracting features)
- CMake 3.15+
- Visual Studio / GCC / Clang (Must support C++17 and AVX2)

### Building the C++ Engine & VST
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release

# To build the VST plugin specifically:
cd ../vst
cmake -B build
cmake --build build --config Release
```

### Running the Web Server
```bash
cd ui
npm install
npm start
```
The application will run at `http://localhost:3000`. You can upload WAV and MP3 files via the UI to process them through the DSP backend.

### Running Python ML Tools
```bash
cd tools
pip install -r requirements.txt
python extract_features_real.py
```
