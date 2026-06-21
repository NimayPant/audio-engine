#include "../include/FeatureExtractor.h"
#include "../include/SimdOps.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace features {

static unsigned int bitReverse(unsigned int x, int log2n) {
    unsigned int result = 0;
    for (int i = 0; i < log2n; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

static bool isPowerOf2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static size_t floorPow2(size_t n) {
    size_t p = 1;
    while (p * 2 <= n) p *= 2;
    return p;
}

void computeFFT(const float* data, size_t size, float* realOut, float* imagOut) {
    size_t n = isPowerOf2(size) ? size : floorPow2(size);
    if (n > kMaxFFTSize) n = kMaxFFTSize;

    int log2n = 0;
    { size_t tmp = n; while (tmp > 1) { tmp >>= 1; ++log2n; } }

    float realBuf[kMaxFFTSize];
    float imagBuf[kMaxFFTSize];

    for (size_t i = 0; i < n; ++i) {
        unsigned int j = bitReverse(static_cast<unsigned int>(i), log2n);
        realBuf[i] = (j < size) ? data[j] : 0.0f;
        imagBuf[i] = 0.0f;
    }

    for (int stage = 1; stage <= log2n; ++stage) {
        size_t halfSpan = size_t(1) << (stage - 1);
        size_t span = halfSpan << 1;
        float angleStep = static_cast<float>(-2.0 * M_PI / span);

        for (size_t group = 0; group < n; group += span) {
            for (size_t k = 0; k < halfSpan; ++k) {
                float angle = angleStep * static_cast<float>(k);
                float twiddleReal = std::cos(angle);
                float twiddleImag = std::sin(angle);
                size_t even = group + k;
                size_t odd  = group + k + halfSpan;
                float tReal = twiddleReal * realBuf[odd] - twiddleImag * imagBuf[odd];
                float tImag = twiddleReal * imagBuf[odd] + twiddleImag * realBuf[odd];
                float eReal = realBuf[even];
                float eImag = imagBuf[even];
                realBuf[even] = eReal + tReal;
                imagBuf[even] = eImag + tImag;
                realBuf[odd]  = eReal - tReal;
                imagBuf[odd]  = eImag - tImag;
            }
        }
    }

    size_t outputSize = n / 2 + 1;
    std::memcpy(realOut, realBuf, outputSize * sizeof(float));
    std::memcpy(imagOut, imagBuf, outputSize * sizeof(float));
}

void computeMagnitudeSpectrum(const float* real, const float* imag, size_t fftSize, float* magnitudes) {
    size_t numBins = fftSize / 2 + 1;
    for (size_t i = 0; i < numBins; ++i) {
        magnitudes[i] = std::sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
}

void computeSpectralBins(const float* magnitudes, size_t fftSize, int sampleRate, float* bins, int numBins) {
    size_t numMags = fftSize / 2 + 1;
    float nyquist = static_cast<float>(sampleRate) / 2.0f;
    float freqPerBin = nyquist / static_cast<float>(numMags - 1);
    float logMin = std::log2(20.0f);
    float logMax = std::log2(nyquist);

    for (int b = 0; b < numBins; ++b) {
        float freqLo = std::pow(2.0f, logMin + (logMax - logMin) * b / numBins);
        float freqHi = std::pow(2.0f, logMin + (logMax - logMin) * (b + 1) / numBins);
        int binLo = static_cast<int>(freqLo / freqPerBin);
        int binHi = static_cast<int>(freqHi / freqPerBin);
        binLo = std::max(0, std::min(binLo, static_cast<int>(numMags - 1)));
        binHi = std::max(binLo + 1, std::min(binHi, static_cast<int>(numMags)));
        float sum = 0.0f;
        int count = 0;
        for (int i = binLo; i < binHi; ++i) {
            sum += magnitudes[i];
            ++count;
        }
        bins[b] = (count > 0) ? (sum / count) : 0.0f;
    }
}

float computeZCR(const float* data, size_t size) {
    if (size < 2) return 0.0f;
    int crossings = 0;
    for (size_t i = 1; i < size; ++i) {
        if ((data[i] >= 0.0f) != (data[i - 1] >= 0.0f)) {
            ++crossings;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(size - 1);
}

float computeSpectralFlatness(const float* magnitudes, size_t numBins) {
    if (numBins == 0) return 0.0f;
    float logSum = 0.0f;
    float linSum = 0.0f;
    int validBins = 0;
    for (size_t i = 0; i < numBins; ++i) {
        float val = std::max(magnitudes[i], 1e-10f);
        logSum += std::log(val);
        linSum += val;
        ++validBins;
    }
    if (validBins == 0 || linSum < 1e-10f) return 0.0f;
    float geometricMean = std::exp(logSum / validBins);
    float arithmeticMean = linSum / validBins;
    return geometricMean / arithmeticMean;
}

void extractCompressorFeatures(const float* block, size_t blockSize, int sampleRate, float* outFeatures) {
    size_t fftSize = isPowerOf2(blockSize) ? blockSize : floorPow2(blockSize);
    if (fftSize > kMaxFFTSize) fftSize = kMaxFFTSize;
    float realBuf[kMaxFFTSize / 2 + 1];
    float imagBuf[kMaxFFTSize / 2 + 1];
    float magBuf[kMaxFFTSize / 2 + 1];
    computeFFT(block, fftSize, realBuf, imagBuf);
    computeMagnitudeSpectrum(realBuf, imagBuf, fftSize, magBuf);
    computeSpectralBins(magBuf, fftSize, sampleRate, outFeatures, kNumSpectralBins);
    float peak, rms;
    simd::computePeakRMS(block, blockSize, peak, rms);
    outFeatures[16] = peak;
    outFeatures[17] = rms;
    outFeatures[18] = (rms > 1e-10f) ? (peak / rms) : 0.0f;
}

void extractNoiseFeatures(const float* block, size_t blockSize, int sampleRate, float* outFeatures) {
    size_t fftSize = isPowerOf2(blockSize) ? blockSize : floorPow2(blockSize);
    if (fftSize > kMaxFFTSize) fftSize = kMaxFFTSize;
    float realBuf[kMaxFFTSize / 2 + 1];
    float imagBuf[kMaxFFTSize / 2 + 1];
    float magBuf[kMaxFFTSize / 2 + 1];
    computeFFT(block, fftSize, realBuf, imagBuf);
    computeMagnitudeSpectrum(realBuf, imagBuf, fftSize, magBuf);
    computeSpectralBins(magBuf, fftSize, sampleRate, outFeatures, kNumSpectralBins);
    outFeatures[16] = computeZCR(block, blockSize);
    outFeatures[17] = computeSpectralFlatness(magBuf, fftSize / 2 + 1);
}

void computeLogMelSpectrogram(const float* block, size_t blockSize, int sampleRate, float* melOut, int numMels) {
    size_t fftSize = isPowerOf2(blockSize) ? blockSize : floorPow2(blockSize);
    if (fftSize > kMaxFFTSize) fftSize = kMaxFFTSize;
    
    float realBuf[kMaxFFTSize / 2 + 1];
    float imagBuf[kMaxFFTSize / 2 + 1];
    float magBuf[kMaxFFTSize / 2 + 1];
    
    computeFFT(block, fftSize, realBuf, imagBuf);
    computeMagnitudeSpectrum(realBuf, imagBuf, fftSize, magBuf);
    
    float nyquist = static_cast<float>(sampleRate) / 2.0f;
    float logMin = std::log2(20.0f);
    float logMax = std::log2(nyquist);
    
    size_t numBins = fftSize / 2 + 1;
    float freqPerBin = nyquist / static_cast<float>(numBins - 1);
    
    for (int m = 0; m < numMels; ++m) {
        float frac = static_cast<float>(m + 0.5f) / numMels;
        float freq = std::pow(2.0f, logMin + (logMax - logMin) * frac);
        int bin = static_cast<int>(freq / freqPerBin);
        bin = std::max(0, std::min(bin, static_cast<int>(numBins - 1)));
        
        float mag = magBuf[bin];
        float power = mag * mag;
        melOut[m] = 10.0f * std::log10(std::max(power, 1e-10f));
    }
}

void extractMLFeatures(const float* block, size_t blockSize, int sampleRate, float* features) {
    const int numMels = 32;
    const int numMFCC = 13;
    
    size_t fftSize = isPowerOf2(blockSize) ? blockSize : floorPow2(blockSize);
    if (fftSize > kMaxFFTSize) fftSize = kMaxFFTSize;
    
    float realBuf[kMaxFFTSize / 2 + 1];
    float imagBuf[kMaxFFTSize / 2 + 1];
    float magBuf[kMaxFFTSize / 2 + 1];
    
    computeFFT(block, fftSize, realBuf, imagBuf);
    computeMagnitudeSpectrum(realBuf, imagBuf, fftSize, magBuf);
    
    float melBands[numMels];
    computeLogMelSpectrogram(block, blockSize, sampleRate, melBands, numMels);
    
    for (int i = 0; i < numMels; ++i) {
        features[i] = melBands[i];
    }
    
    for (int i = 0; i < numMels; ++i) {
        features[32 + i] = 2.0f + (melBands[i] * 0.1f);
    }
    
    for (int m = 0; m < numMFCC; ++m) {
        float mfcc_val = 0.0f;
        for (int k = 0; k < numMels; ++k) {
            float angle = static_cast<float>(M_PI * m * (k + 0.5f) / numMels);
            mfcc_val += melBands[k] * std::cos(angle);
        }
        features[64 + m] = mfcc_val / (numMels / 2.0f);
    }
    
    for (int m = 0; m < numMFCC; ++m) {
        features[77 + m] = 1.0f + (std::abs(features[64 + m]) * 0.1f);
    }
    
    float peak, rms;
    simd::computePeakRMS(block, blockSize, peak, rms);
    
    float numBins = static_cast<float>(fftSize) / 2.0f + 1.0f;
    float freqPerBin = static_cast<float>(sampleRate) / (2.0f * numBins);
    float centroid = 0.0f, weightSum = 0.0f;
    for (size_t i = 1; i < numBins; ++i) {
        float freq = i * freqPerBin;
        centroid += freq * magBuf[i];
        weightSum += magBuf[i];
    }
    features[90] = (weightSum > 1e-10f) ? (centroid / weightSum) : 5000.0f;
    
    float rolloffTarget = 0.95f * weightSum;
    float rolloffSum = 0.0f;
    float rolloff = 5000.0f;
    for (size_t i = 1; i < numBins; ++i) {
        rolloffSum += magBuf[i];
        if (rolloffSum >= rolloffTarget) {
            rolloff = i * freqPerBin;
            break;
        }
    }
    features[91] = rolloff;
    
    features[92] = computeZCR(block, blockSize);
    features[93] = peak;
    features[94] = rms;
    features[95] = computeSpectralFlatness(magBuf, fftSize / 2 + 1);
    features[96] = (rms > 1e-10f) ? (peak / rms) : 1.0f;
    features[97] = std::log10(std::max(rms, 1e-10f));
    features[98] = melBands[numMels - 1] - melBands[0];
    features[99] = std::max(0.0f, features[91] / (features[90] + 1e-6f));
}

}
