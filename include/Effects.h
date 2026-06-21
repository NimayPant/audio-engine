#ifndef EFFECTS_H
#define EFFECTS_H

#include <cstddef>
#include <vector>

namespace simd { struct BiquadState; }

class Effects {
public:
    static void applyDelay(std::vector<float>& channel, int sampleRate, float delayTimeMs, float feedback);
    static void applyReverb(std::vector<float>& channel, int sampleRate, float roomSize, float damping, float wetDryMix);
    static void applyGainSIMD(std::vector<float>& channel, float gain);
    static void applyGainSIMD(float* data, size_t count, float gain);
    static void applyBiquadSIMD(std::vector<float>& channel, int sampleRate, float centerFreq, float Q, float gainDb);
    static void applySmartCompression(std::vector<float>& channel, int sampleRate, float intensity);
    static void applySmartGate(std::vector<float>& channel, int sampleRate, float threshold);
    static void applySmartCompression(float* data, size_t count, int sampleRate, float intensity);
    static void applySmartGate(float* data, size_t count, int sampleRate, float threshold);
    static void applyParametricEQ(std::vector<float>& channel, int sampleRate, float centerFreq, float Q, float gainDb);
    static void applyParametricEQ(float* data, size_t count, int sampleRate, float centerFreq, float Q, float gainDb);
    static void applyMLDynamicEQ(std::vector<float>& channel, int sampleRate, float intensity = 0.5f);
    static void applyMLDynamicEQ(float* data, size_t count, int sampleRate, float intensity = 0.5f);
    static void applyGenreAwareProcessing(std::vector<float>& channel, int sampleRate, int genreId, float intensity = 1.0f);
    static void applyGenreAwareProcessing(float* data, size_t count, int sampleRate, int genreId, float intensity = 1.0f);

    // Real-Time Safe Overloads for VST
    static void applySmartCompressionRT(float* data, size_t count, int sampleRate, float intensity, float& envelope);
    static void applySmartGateRT(float* data, size_t count, int sampleRate, float threshold, float& gateGain);
    static void applyMLDynamicEQRT(float* data, size_t count, int sampleRate, float intensity, simd::BiquadState* bandStates);
    static void applyParametricEQRT(float* data, size_t count, int sampleRate, float centerFreq, float Q, float gainDb, float* state);
    static void applyGenreAwareProcessingRT(float* data, size_t count, int sampleRate, int genreId, float intensity, float& compEnvelope, float* eqStates);
};

#endif