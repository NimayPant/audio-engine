#include "../include/Effects.h"
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void Effects::applyDelay(std::vector<float>& channel, int sampleRate, float delayTimeMs, float feedback) {
    float delayTime = delayTimeMs / 1000.0f;
    int delaySamples = static_cast<int>(delayTime * sampleRate);
    if (delaySamples <= 0 || delaySamples >= (int)channel.size()) return;

    std::vector<float> buffer(delaySamples, 0.0f);
    size_t bufferIndex = 0;

    for (size_t i = 0; i < channel.size(); ++i) {
        float delayed = buffer[bufferIndex];
        buffer[bufferIndex] = channel[i] + delayed * feedback;
        channel[i] += delayed;
        bufferIndex = (bufferIndex + 1) % delaySamples;
    }
}

void Effects::applyReverb(std::vector<float>& channel, int sampleRate, float roomSize, float damping, float wetDryMix) {
    const int numCombs = 4;
    float combTimes[] = {0.0297f, 0.0371f, 0.0411f, 0.0437f};
    std::vector<std::vector<float>> combBuffers(numCombs);
    std::vector<size_t> combIndices(numCombs, 0);
    
    float feedback = 0.7f + roomSize * 0.28f;

    for (int i = 0; i < numCombs; ++i) {
        combBuffers[i].resize(static_cast<size_t>(combTimes[i] * sampleRate), 0.0f);
    }

    for (size_t i = 0; i < channel.size(); ++i) {
        float input = channel[i];
        float combsOut = 0.0f;
        
        for (int c = 0; c < numCombs; ++c) {
            float delayed = combBuffers[c][combIndices[c]];
            combBuffers[c][combIndices[c]] = input + delayed * feedback;
            combsOut += delayed;
            combIndices[c] = (combIndices[c] + 1) % combBuffers[c].size();
        }
        
        float wet = combsOut / numCombs;
        channel[i] = input * (1.0f - wetDryMix) + wet * wetDryMix;
    }
}

void Effects::applyParametricEQ(std::vector<float>& channel, int sampleRate, float centerFreq, float Q, float gainDb) {
    float A = pow(10.0f, gainDb / 40.0f);
    float omega = 2.0f * M_PI * centerFreq / sampleRate;
    float sn = sin(omega);
    float cs = cos(omega);
    float alpha = sn / (2.0f * Q);

    float b0 = 1.0f + alpha * A;
    float b1 = -2.0f * cs;
    float b2 = 1.0f - alpha * A;
    float a0 = 1.0f + alpha / A;
    float a1 = -2.0f * cs;
    float a2 = 1.0f - alpha / A;

    b0 /= a0; b1 /= a0; b2 /= a0;
    a1 /= a0; a2 /= a0;

    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (size_t i = 0; i < channel.size(); ++i) {
        float x0 = channel[i];
        float y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        channel[i] = y0;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }
}

#include "../include/SimdOps.h"
#include "../include/MLModels.h"
#include "../include/FeatureExtractor.h"

void Effects::applyGainSIMD(std::vector<float>& channel, float gain) {
    simd::applyGain(channel.data(), channel.size(), gain);
}

void Effects::applyGainSIMD(float* data, size_t count, float gain) {
    simd::applyGain(data, count, gain);
}

void Effects::applyBiquadSIMD(std::vector<float>& channel, int sampleRate, float centerFreq, float Q, float gainDb) {
    float A = pow(10.0f, gainDb / 40.0f);
    float omega = 2.0f * M_PI * centerFreq / sampleRate;
    float sn = sin(omega);
    float cs = cos(omega);
    float alpha = sn / (2.0f * Q);

    simd::BiquadCoeffs coeffs;
    float a0 = 1.0f + alpha / A;
    coeffs.b0 = (1.0f + alpha * A) / a0;
    coeffs.b1 = (-2.0f * cs) / a0;
    coeffs.b2 = (1.0f - alpha * A) / a0;
    coeffs.a1 = (-2.0f * cs) / a0;
    coeffs.a2 = (1.0f - alpha / A) / a0;

    simd::BiquadState state;
    simd::processBiquad(channel.data(), channel.size(), &coeffs, &state, 1);
}

void Effects::applyParametricEQ(float* data, size_t count, int sampleRate, float centerFreq, float Q, float gainDb) {
    float A = pow(10.0f, gainDb / 40.0f);
    float omega = 2.0f * M_PI * centerFreq / sampleRate;
    float sn = sin(omega);
    float cs = cos(omega);
    float alpha = sn / (2.0f * Q);

    float b0 = 1.0f + alpha * A;
    float b1 = -2.0f * cs;
    float b2 = 1.0f - alpha * A;
    float a0 = 1.0f + alpha / A;
    float a1 = -2.0f * cs;
    float a2 = 1.0f - alpha / A;

    b0 /= a0; b1 /= a0; b2 /= a0;
    a1 /= a0; a2 /= a0;

    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (size_t i = 0; i < count; ++i) {
        float x0 = data[i];
        float y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        data[i] = y0;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }
}

void Effects::applySmartCompression(float* data, size_t count, int sampleRate, float intensity) {
    const size_t blockSize = 512;
    float featuresArr[19];
    float envelope = 0.0f;

    for (size_t i = 0; i < count; i += blockSize) {
        size_t sz = std::min(blockSize, count - i);
        if (sz < 128) break;

        features::extractCompressorFeatures(&data[i], sz, sampleRate, featuresArr);
        auto pred = ml::CompressorPredictor::predict(featuresArr);

        float threshold = pow(10.0f, pred.thresholdDb / 20.0f);
        float attackCoeff = 1.0f - exp(-1.0f / (0.01f * sampleRate + 1e-6f));
        float releaseCoeff = 1.0f - exp(-1.0f / (0.1f * sampleRate + 1e-6f));

        for (size_t j = 0; j < sz; ++j) {
            float input = fabs(data[i + j]);
            if (input > envelope) envelope += (input - envelope) * attackCoeff;
            else envelope += (input - envelope) * releaseCoeff;

            float gainVal = 1.0f;
            if (envelope > threshold) {
                float actualRatio = 1.0f + (pred.ratio - 1.0f) * intensity;
                gainVal = (threshold + (envelope - threshold) / std::max(1.01f, actualRatio)) / (envelope + 1e-6f);
            }
            data[i + j] *= gainVal;
        }
    }
}

void Effects::applySmartGate(float* data, size_t count, int sampleRate, float manualThreshold) {
    const size_t blockSize = 512;
    float featuresArr[18];
    float gateGain = 1.0f;

    for (size_t i = 0; i < count; i += blockSize) {
        size_t sz = std::min(blockSize, count - i);
        if (sz < 128) break;

        features::extractNoiseFeatures(&data[i], sz, sampleRate, featuresArr);
        float signalProb = ml::NoiseClassifier::predict(featuresArr);

        float targetGain = (signalProb > (0.1f * manualThreshold)) ? 1.0f : 0.0f;

        for (size_t j = 0; j < sz; ++j) {
            gateGain += (targetGain - gateGain) * 0.05f;
            data[i + j] *= gateGain;
        }
    }
}

void Effects::applySmartCompression(std::vector<float>& channel, int sampleRate, float intensity) {
    const size_t blockSize = 512;
    float features[19];
    float envelopeDb = -100.0f;
    
    const float kneeWidth = 6.0f; 
    
    for (size_t i = 0; i < channel.size(); i += blockSize) {
        size_t size = std::min(blockSize, channel.size() - i);
        if (size < 128) break;

        features::extractCompressorFeatures(&channel[i], size, sampleRate, features);
        auto pred = ml::CompressorPredictor::predict(features);

        float attackCoeff = 1.0f - exp(-1.0f / (0.01f * sampleRate + 1e-6f));
        float releaseCoeff = 1.0f - exp(-1.0f / (0.1f * sampleRate + 1e-6f));
        
        float actualRatio = 1.0f + (pred.ratio - 1.0f) * intensity;
        actualRatio = std::max(1.0f, actualRatio);
        
        for (size_t j = 0; j < size; ++j) {
            float input = fabs(channel[i + j]);
            float inputDb = (input > 1e-6f) ? 20.0f * log10(input) : -120.0f;
            
            if (inputDb > envelopeDb) {
                envelopeDb += (inputDb - envelopeDb) * attackCoeff;
            } else {
                envelopeDb += (inputDb - envelopeDb) * releaseCoeff;
            }

            float gainDb = 0.0f;
            float overThreshold = envelopeDb - pred.thresholdDb;
            
            if (overThreshold > -(kneeWidth / 2.0f)) {
                if (overThreshold < (kneeWidth / 2.0f)) {
                    float interp = (overThreshold + kneeWidth / 2.0f) / kneeWidth;
                    gainDb = -(overThreshold * interp * (1.0f - 1.0f / actualRatio)) / 2.0f;
                } else {
                    gainDb = -overThreshold * (1.0f - 1.0f / actualRatio);
                }
            }
            
            float gainLinear = pow(10.0f, gainDb / 20.0f);
            channel[i + j] *= gainLinear;
        }
    }
}

void Effects::applySmartGate(std::vector<float>& channel, int sampleRate, float manualThreshold) {
    const size_t blockSize = 512;
    float features[18];
    float gateGain = 1.0f;

    for (size_t i = 0; i < channel.size(); i += blockSize) {
        size_t size = std::min(blockSize, channel.size() - i);
        if (size < 128) break;

        features::extractNoiseFeatures(&channel[i], size, sampleRate, features);
        float prob = ml::NoiseClassifier::predict(features);

        float targetGain = (prob > (0.5f * manualThreshold)) ? 1.0f : 0.0f;

        for (size_t j = 0; j < size; ++j) {
            gateGain += (targetGain - gateGain) * 0.01f;
            channel[i + j] *= gateGain;
        }
    }
}

void Effects::applyMLDynamicEQ(std::vector<float>& channel, int sampleRate, float intensity) {
    applyMLDynamicEQ(channel.data(), channel.size(), sampleRate, intensity);
}

void Effects::applyMLDynamicEQ(float* data, size_t count, int sampleRate, float intensity) {
    const size_t blockSize = 256;
    const int numBands = 8;
    
    simd::BiquadState bandStates[numBands] = {};
    
    for (size_t i = 0; i < count; i += blockSize) {
        size_t sz = std::min(blockSize, count - i);
        if (sz < 64) break;
        
        float features[100];
        features::extractMLFeatures(&data[i], sz, sampleRate, features);
        
        auto eqGains = ml::EQPredictor::predict(features);
        
        float nyq = sampleRate * 0.5f;
        float logMin = std::log2(20.0f);
        float logMax = std::log2(std::min(nyq, 20000.0f));
        
        simd::BiquadCoeffs coeffs[numBands];
        
        for (int bn = 0; bn < numBands; ++bn) {
            float frac = (bn + 0.5f) / float(numBands);
            float center = std::pow(2.0f, logMin + (logMax - logMin) * frac);
            float Q = 1.2f;
            
            float gainDb = std::max(-4.0f, std::min(4.0f, eqGains.gains[bn] * intensity));
            
            float A = pow(10.0f, gainDb / 40.0f);
            float omega = 2.0f * M_PI * center / float(sampleRate);
            float sn = sin(omega);
            float cs = cos(omega);
            float alpha = sn / (2.0f * Q);
            
            float b0 = 1.0f + alpha * A;
            float b1 = -2.0f * cs;
            float b2 = 1.0f - alpha * A;
            float a0 = 1.0f + alpha / A;
            float a1 = -2.0f * cs;
            float a2 = 1.0f - alpha / A;
            
            b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
            coeffs[bn].b0 = b0; coeffs[bn].b1 = b1; coeffs[bn].b2 = b2;
            coeffs[bn].a1 = a1; coeffs[bn].a2 = a2;
        }
        
        simd::processBiquad(&data[i], sz, coeffs, bandStates, numBands);
    }
}

void Effects::applyGenreAwareProcessing(std::vector<float>& channel, int sampleRate, int genreId, float intensity) {
    applyGenreAwareProcessing(channel.data(), channel.size(), sampleRate, genreId, intensity);
}

void Effects::applyGenreAwareProcessing(float* data, size_t count, int sampleRate, int genreId, float intensity) {
    struct GenreParams {
        float compression;
        float brightness;
        float presence;
        float warmth;
    };
    
    GenreParams params;
    
    switch (genreId) {
        case 0:
            params = {0.3f, 2.0f, 2.5f, 1.0f};
            break;
        case 1:
            params = {0.25f, 4.0f, 3.0f, -2.0f};
            break;
        case 2:
            params = {0.5f, 1.0f, 4.0f, 2.5f};
            break;
        case 3:
            params = {0.6f, -1.0f, 5.0f, 3.5f};
            break;
        case 4:
            params = {0.15f, 0.5f, 1.5f, 4.0f};
            break;
        case 8:
            params = {0.2f, -1.0f, 2.0f, 4.5f};
            break;
        case 15:
            params = {0.05f, 3.0f, 2.0f, 1.5f};
            break;
        default:
            params = {0.3f, 1.5f, 2.0f, 2.0f};
    }
    
    if (std::abs(params.brightness * intensity) > 0.01f) {
        Effects::applyParametricEQ(data, count, sampleRate, 5000.0f, 2.0f, params.brightness * intensity);
    }
    
    if (std::abs(params.presence * intensity) > 0.01f) {
        Effects::applyParametricEQ(data, count, sampleRate, 3000.0f, 1.5f, params.presence * intensity);
    }
    
    if (std::abs(params.warmth * intensity) > 0.01f) {
        Effects::applyParametricEQ(data, count, sampleRate, 250.0f, 1.0f, params.warmth * intensity);
    }
    
    if (params.compression * intensity > 0.01f) {
        Effects::applySmartCompression(data, count, sampleRate, params.compression * intensity);
    }
    
    float peak = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float absVal = std::abs(data[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak > 1e-6f) {
        float gain = 1.0f / peak;
        for (size_t i = 0; i < count; ++i) {
            data[i] *= gain;
        }
    }
}

void Effects::applySmartCompressionRT(float* data, size_t count, int sampleRate, float intensity, float& envelopeDb) {
    if (count == 0) return;
    float features[19];
    
    // We assume count is standard VST block size (e.g. 512). We extract once per block.
    features::extractCompressorFeatures(data, count, sampleRate, features);
    auto pred = ml::CompressorPredictor::predict(features);

    float attackCoeff = 1.0f - exp(-1.0f / (0.01f * sampleRate + 1e-6f));
    float releaseCoeff = 1.0f - exp(-1.0f / (0.1f * sampleRate + 1e-6f));
    
    float actualRatio = 1.0f + (pred.ratio - 1.0f) * intensity;
    actualRatio = std::max(1.0f, actualRatio);
    const float kneeWidth = 6.0f; 

    for (size_t j = 0; j < count; ++j) {
        float input = std::fabs(data[j]);
        float inputDb = (input > 1e-6f) ? 20.0f * log10(input) : -120.0f;
        
        if (inputDb > envelopeDb) {
            envelopeDb += (inputDb - envelopeDb) * attackCoeff;
        } else {
            envelopeDb += (inputDb - envelopeDb) * releaseCoeff;
        }

        float gainDb = 0.0f;
        float overThreshold = envelopeDb - pred.thresholdDb;
        
        if (overThreshold > -(kneeWidth / 2.0f)) {
            if (overThreshold < (kneeWidth / 2.0f)) {
                float interp = (overThreshold + kneeWidth / 2.0f) / kneeWidth;
                gainDb = -(overThreshold * interp * (1.0f - 1.0f / actualRatio)) / 2.0f;
            } else {
                gainDb = -overThreshold * (1.0f - 1.0f / actualRatio);
            }
        }
        
        float gainLinear = pow(10.0f, gainDb / 20.0f);
        data[j] *= gainLinear;
    }
}

void Effects::applySmartGateRT(float* data, size_t count, int sampleRate, float threshold, float& gateGain) {
    if (count == 0) return;
    float features[18];
    features::extractNoiseFeatures(data, count, sampleRate, features);
    float prob = ml::NoiseClassifier::predict(features);

    float targetGain = (prob > (0.5f * threshold)) ? 1.0f : 0.0f;

    for (size_t j = 0; j < count; ++j) {
        gateGain += (targetGain - gateGain) * 0.01f;
        data[j] *= gateGain;
    }
}

void Effects::applyMLDynamicEQRT(float* data, size_t count, int sampleRate, float intensity, simd::BiquadState* bandStates) {
    if (count == 0) return;
    const int numBands = 8;
    
    float features[100];
    features::extractMLFeatures(data, count, sampleRate, features);
    auto eqGains = ml::EQPredictor::predict(features);
    
    float nyq = sampleRate * 0.5f;
    float logMin = std::log2(20.0f);
    float logMax = std::log2(std::min(nyq, 20000.0f));
    
    simd::BiquadCoeffs coeffs[numBands];
    
    for (int bn = 0; bn < numBands; ++bn) {
        float frac = (bn + 0.5f) / float(numBands);
        float center = std::pow(2.0f, logMin + (logMax - logMin) * frac);
        float Q = 1.2f;
        
        float gainDb = std::max(-4.0f, std::min(4.0f, eqGains.gains[bn] * intensity));
        
        float A = pow(10.0f, gainDb / 40.0f);
        float omega = 2.0f * M_PI * center / float(sampleRate);
        float sn = sin(omega);
        float cs = cos(omega);
        float alpha = sn / (2.0f * Q);
        
        float b0 = 1.0f + alpha * A;
        float b1 = -2.0f * cs;
        float b2 = 1.0f - alpha * A;
        float a0 = 1.0f + alpha / A;
        float a1 = -2.0f * cs;
        float a2 = 1.0f - alpha / A;
        
        b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
        coeffs[bn].b0 = b0; coeffs[bn].b1 = b1; coeffs[bn].b2 = b2;
        coeffs[bn].a1 = a1; coeffs[bn].a2 = a2;
    }
    
    simd::processBiquad(data, count, coeffs, bandStates, numBands);
}

void Effects::applyParametricEQRT(float* data, size_t count, int sampleRate, float centerFreq, float Q, float gainDb, float* state) {
    float A = pow(10.0f, gainDb / 40.0f);
    float omega = 2.0f * M_PI * centerFreq / sampleRate;
    float sn = sin(omega);
    float cs = cos(omega);
    float alpha = sn / (2.0f * Q);

    float b0 = 1.0f + alpha * A;
    float b1 = -2.0f * cs;
    float b2 = 1.0f - alpha * A;
    float a0 = 1.0f + alpha / A;
    float a1 = -2.0f * cs;
    float a2 = 1.0f - alpha / A;

    b0 /= a0; b1 /= a0; b2 /= a0;
    a1 /= a0; a2 /= a0;

    float x1 = state[0], x2 = state[1], y1 = state[2], y2 = state[3];
    for (size_t i = 0; i < count; ++i) {
        float x0 = data[i];
        float y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        data[i] = y0;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }
    state[0] = x1; state[1] = x2; state[2] = y1; state[3] = y2;
}

void Effects::applyGenreAwareProcessingRT(float* data, size_t count, int sampleRate, int genreId, float intensity, float& compEnvelope, float* eqStates) {
    struct GenreParams {
        float compression;
        float brightness;
        float presence;
        float warmth;
    };
    GenreParams params;
    
    switch (genreId) {
        case 0: params = {0.3f, 2.0f, 2.5f, 1.0f}; break;
        case 1: params = {0.25f, 4.0f, 3.0f, -2.0f}; break;
        case 2: params = {0.5f, 1.0f, 4.0f, 2.5f}; break;
        case 3: params = {0.6f, -1.0f, 5.0f, 3.5f}; break;
        case 4: params = {0.15f, 0.5f, 1.5f, 4.0f}; break;
        case 8: params = {0.2f, -1.0f, 2.0f, 4.5f}; break;
        case 15: params = {0.05f, 3.0f, 2.0f, 1.5f}; break;
        default: params = {0.3f, 1.5f, 2.0f, 2.0f};
    }
    
    if (std::abs(params.brightness * intensity) > 0.01f) {
        applyParametricEQRT(data, count, sampleRate, 5000.0f, 2.0f, params.brightness * intensity, &eqStates[0]);
    }
    if (std::abs(params.presence * intensity) > 0.01f) {
        applyParametricEQRT(data, count, sampleRate, 3000.0f, 1.5f, params.presence * intensity, &eqStates[4]);
    }
    if (std::abs(params.warmth * intensity) > 0.01f) {
        applyParametricEQRT(data, count, sampleRate, 250.0f, 1.0f, params.warmth * intensity, &eqStates[8]);
    }
    if (params.compression * intensity > 0.01f) {
        applySmartCompressionRT(data, count, sampleRate, params.compression * intensity, compEnvelope);
    }
    
    float peak = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float absVal = std::abs(data[i]);
        if (absVal > peak) peak = absVal;
    }
    if (peak > 1.0f) {
        float gain = 1.0f / peak;
        simd::applyGain(data, count, gain);
    }
}