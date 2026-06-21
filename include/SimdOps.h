#ifndef SIMDOPS_H
#define SIMDOPS_H

#include <cstddef>
#include <cstdint>

namespace simd {

struct CpuFeatures {
    bool sse2  = false;
    bool sse3  = false;
    bool ssse3 = false;
    bool sse41 = false;
    bool avx   = false;
    bool avx2  = false;
    bool fma   = false;
};

const CpuFeatures& detectCpuFeatures();
const char* getCpuFeatureString();

#ifdef _MSC_VER
    __declspec(align(32))
#endif
struct BiquadCoeffs {
    float b0, b1, b2;
    float a1, a2;
    float _pad[3]; 
} 
#ifdef __GNUC__
    __attribute__((aligned(32)))
#endif
;

#ifdef _MSC_VER
    __declspec(align(16))
#endif
struct BiquadState {
    float x1 = 0.0f, x2 = 0.0f;
    float y1 = 0.0f, y2 = 0.0f;
}
#ifdef __GNUC__
    __attribute__((aligned(16)))
#endif
;

void applyGain_AVX2(float* data, size_t count, float gain);
void applyGain_SSE2(float* data, size_t count, float gain);
void applyGain_Scalar(float* data, size_t count, float gain);
void applyGain(float* data, size_t count, float gain);

void processBiquad_Scalar(float* data, size_t count, const BiquadCoeffs* coeffs, BiquadState* states, int numStages);
void processBiquad(float* data, size_t count, const BiquadCoeffs* coeffs, BiquadState* states, int numStages);

void computePeakRMS_AVX2(const float* data, size_t count, float& outPeak, float& outRMS);
void computePeakRMS_SSE2(const float* data, size_t count, float& outPeak, float& outRMS);
void computePeakRMS_Scalar(const float* data, size_t count, float& outPeak, float& outRMS);
void computePeakRMS(const float* data, size_t count, float& outPeak, float& outRMS);

}

#endif
