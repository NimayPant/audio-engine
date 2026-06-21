#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <cstddef>

namespace features {

static constexpr size_t kMaxFFTSize = 4096;
static constexpr int kNumSpectralBins = 16;

void computeFFT(const float* data, size_t size, float* realOut, float* imagOut);

void computeMagnitudeSpectrum(const float* real, const float* imag,
                              size_t fftSize, float* magnitudes);

void computeSpectralBins(const float* magnitudes, size_t fftSize,
                         int sampleRate, float* bins, int numBins);

float computeZCR(const float* data, size_t size);

float computeSpectralFlatness(const float* magnitudes, size_t numBins);

void extractCompressorFeatures(const float* block, size_t blockSize,
                               int sampleRate, float* features);

void extractNoiseFeatures(const float* block, size_t blockSize,
                          int sampleRate, float* features);

void extractMLFeatures(const float* block, size_t blockSize,
                       int sampleRate, float* features);

void computeLogMelSpectrogram(const float* block, size_t blockSize,
                              int sampleRate, float* melOut, int numMels = 32);

}
#endif
