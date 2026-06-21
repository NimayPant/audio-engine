#include "../include/SimdOps.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif

#include <emmintrin.h>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(_MSC_VER)
#include <immintrin.h>
#endif

namespace simd {

static std::atomic<bool> g_featuresDetected{false};
static CpuFeatures g_features;
static char g_featureString[256] = {0};

static bool osSupportsAVX() {
#ifdef _MSC_VER
  unsigned long long xcr0 = _xgetbv(0);
  return (xcr0 & 0x6) == 0x6;
#elif defined(__GNUC__) || defined(__clang__)
  unsigned int eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (eax & 0x6) == 0x6;
#else
  return false;
#endif
}

const CpuFeatures &detectCpuFeatures() {
  if (g_featuresDetected.load(std::memory_order_acquire)) {
    return g_features;
  }

  CpuFeatures f;

#ifdef _MSC_VER
  int cpuInfo[4] = {0};
  __cpuid(cpuInfo, 0);
  int maxLeaf = cpuInfo[0];

  if (maxLeaf >= 1) {
    __cpuid(cpuInfo, 1);
    unsigned int ecx = cpuInfo[2];
    unsigned int edx = cpuInfo[3];

    f.sse2 = (edx >> 26) & 1;
    f.sse3 = (ecx >> 0) & 1;
    f.ssse3 = (ecx >> 9) & 1;
    f.sse41 = (ecx >> 19) & 1;
    f.fma = (ecx >> 12) & 1;

    bool cpuAVX = (ecx >> 28) & 1;
    bool osxsave = (ecx >> 27) & 1;
    f.avx = cpuAVX && osxsave && osSupportsAVX();
  }

  if (maxLeaf >= 7 && f.avx) {
    __cpuidex(cpuInfo, 7, 0);
    f.avx2 = (cpuInfo[1] >> 5) & 1;
  }
#else
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    f.sse2 = (edx >> 26) & 1;
    f.sse3 = (ecx >> 0) & 1;
    f.ssse3 = (ecx >> 9) & 1;
    f.sse41 = (ecx >> 19) & 1;
    f.fma = (ecx >> 12) & 1;

    bool cpuAVX = (ecx >> 28) & 1;
    bool osxsave = (ecx >> 27) & 1;
    f.avx = cpuAVX && osxsave && osSupportsAVX();
  }

  if (f.avx && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    f.avx2 = (ebx >> 5) & 1;
  }
#endif

  g_features = f;

  char *p = g_featureString;
  int remaining = sizeof(g_featureString);
  int written = 0;

  auto append = [&](const char *name, bool supported) {
    if (supported && remaining > 0) {
      int n = snprintf(p, remaining, "%s%s", (p != g_featureString ? " " : ""),
                       name);
      if (n > 0) {
        p += n;
        remaining -= n;
        written += n;
      }
    }
  };

  append("SSE2", f.sse2);
  append("SSE3", f.sse3);
  append("SSSE3", f.ssse3);
  append("SSE4.1", f.sse41);
  append("AVX", f.avx);
  append("AVX2", f.avx2);
  append("FMA", f.fma);

  if (p == g_featureString) {
    snprintf(g_featureString, sizeof(g_featureString), "(none)");
  }

  g_featuresDetected.store(true, std::memory_order_release);
  return g_features;
}

const char *getCpuFeatureString() {
  detectCpuFeatures();
  return g_featureString;
}

void applyGain_AVX2(float *data, size_t count, float gain) {
#if defined(__AVX2__) || defined(_MSC_VER)
  __m256 vGain = _mm256_set1_ps(gain);
  size_t i = 0;
  const size_t simdEnd = count & ~size_t(7);

  for (; i < simdEnd; i += 8) {
    __m256 samples = _mm256_loadu_ps(&data[i]);
    __m256 result = _mm256_mul_ps(samples, vGain);
    _mm256_storeu_ps(&data[i], result);
  }

  for (; i < count; ++i) {
    data[i] *= gain;
  }
#else
  applyGain_Scalar(data, count, gain);
#endif
}

void applyGain_SSE2(float *data, size_t count, float gain) {
  __m128 vGain = _mm_set1_ps(gain);
  size_t i = 0;
  const size_t simdEnd = count & ~size_t(3);

  for (; i < simdEnd; i += 4) {
    __m128 samples = _mm_loadu_ps(&data[i]);
    __m128 result = _mm_mul_ps(samples, vGain);
    _mm_storeu_ps(&data[i], result);
  }

  for (; i < count; ++i) {
    data[i] *= gain;
  }
}

void applyGain_Scalar(float *data, size_t count, float gain) {
  for (size_t i = 0; i < count; ++i) {
    data[i] *= gain;
  }
}

using GainFunc = void (*)(float *, size_t, float);

static GainFunc resolveGainFunc() {
  const auto &features = detectCpuFeatures();
  if (features.avx2)
    return applyGain_AVX2;
  if (features.sse2)
    return applyGain_SSE2;
  return applyGain_Scalar;
}

static std::atomic<GainFunc> g_gainFunc{nullptr};

void applyGain(float *data, size_t count, float gain) {
  GainFunc fn = g_gainFunc.load(std::memory_order_acquire);
  if (!fn) {
    fn = resolveGainFunc();
    g_gainFunc.store(fn, std::memory_order_release);
  }
  fn(data, count, gain);
}

void computePeakRMS_AVX2(const float *data, size_t count, float &outPeak,
                         float &outRMS) {
#if defined(__AVX2__) || defined(_MSC_VER)
  if (count == 0) {
    outPeak = 0.0f;
    outRMS = 0.0f;
    return;
  }

  __m256 absMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  __m256 vPeak = _mm256_setzero_ps();
  __m256 vSumSq = _mm256_setzero_ps();

  size_t i = 0;
  const size_t simdEnd = count & ~size_t(7);

  for (; i < simdEnd; i += 8) {
    __m256 samples = _mm256_loadu_ps(&data[i]);
    __m256 absSamples = _mm256_and_ps(samples, absMask);
    vPeak = _mm256_max_ps(vPeak, absSamples);
    __m256 squared = _mm256_mul_ps(samples, samples);
    vSumSq = _mm256_add_ps(vSumSq, squared);
  }

  __m128 peakHi = _mm256_extractf128_ps(vPeak, 1);
  __m128 peakLo = _mm256_castps256_ps128(vPeak);
  __m128 peak4 = _mm_max_ps(peakHi, peakLo);
  __m128 peak2 = _mm_max_ps(peak4, _mm_movehl_ps(peak4, peak4));
  __m128 peak1 = _mm_max_ss(peak2, _mm_shuffle_ps(peak2, peak2, 1));
  float peak = _mm_cvtss_f32(peak1);

  __m128 sumHi = _mm256_extractf128_ps(vSumSq, 1);
  __m128 sumLo = _mm256_castps256_ps128(vSumSq);
  __m128 sum4 = _mm_add_ps(sumHi, sumLo);
  __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
  float sumSq = _mm_cvtss_f32(sum1);

  for (; i < count; ++i) {
    float abs_val = std::fabs(data[i]);
    if (abs_val > peak)
      peak = abs_val;
    sumSq += data[i] * data[i];
  }

  outPeak = peak;
  outRMS = std::sqrt(sumSq / static_cast<float>(count));
#else
  computePeakRMS_Scalar(data, count, outPeak, outRMS);
#endif
}

void computePeakRMS_SSE2(const float *data, size_t count, float &outPeak,
                         float &outRMS) {
  if (count == 0) {
    outPeak = 0.0f;
    outRMS = 0.0f;
    return;
  }
  __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  __m128 vPeak = _mm_setzero_ps();
  __m128 vSumSq = _mm_setzero_ps();

  size_t i = 0;
  const size_t simdEnd = count & ~size_t(3);

  for (; i < simdEnd; i += 4) {
    __m128 samples = _mm_loadu_ps(&data[i]);
    __m128 absSamples = _mm_and_ps(samples, absMask);
    vPeak = _mm_max_ps(vPeak, absSamples);
    __m128 squared = _mm_mul_ps(samples, samples);
    vSumSq = _mm_add_ps(vSumSq, squared);
  }

  __m128 peak2 = _mm_max_ps(vPeak, _mm_movehl_ps(vPeak, vPeak));
  __m128 peak1 = _mm_max_ss(peak2, _mm_shuffle_ps(peak2, peak2, 1));
  float peak = _mm_cvtss_f32(peak1);
  __m128 sum2 = _mm_add_ps(vSumSq, _mm_movehl_ps(vSumSq, vSumSq));
  __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
  float sumSq = _mm_cvtss_f32(sum1);

  for (; i < count; ++i) {
    float abs_val = std::fabs(data[i]);
    if (abs_val > peak)
      peak = abs_val;
    sumSq += data[i] * data[i];
  }

  outPeak = peak;
  outRMS = std::sqrt(sumSq / static_cast<float>(count));
}

void computePeakRMS_Scalar(const float *data, size_t count, float &outPeak,
                           float &outRMS) {
  if (count == 0) {
    outPeak = 0.0f;
    outRMS = 0.0f;
    return;
  }
  float peak = 0.0f;
  float sumSq = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    float abs_val = std::fabs(data[i]);
    if (abs_val > peak)
      peak = abs_val;
    sumSq += data[i] * data[i];
  }
  outPeak = peak;
  outRMS = std::sqrt(sumSq / static_cast<float>(count));
}

using PeakRmsFunc = void (*)(const float *, size_t, float &, float &);

static PeakRmsFunc resolvePeakRmsFunc() {
  const auto &features = detectCpuFeatures();
  if (features.avx2)
    return computePeakRMS_AVX2;
  if (features.sse2)
    return computePeakRMS_SSE2;
  return computePeakRMS_Scalar;
}

static std::atomic<PeakRmsFunc> g_peakRmsFunc{nullptr};

void computePeakRMS(const float *data, size_t count, float &outPeak,
                    float &outRMS) {
  PeakRmsFunc fn = g_peakRmsFunc.load(std::memory_order_acquire);
  if (!fn) {
    fn = resolvePeakRmsFunc();
    g_peakRmsFunc.store(fn, std::memory_order_release);
  }
  fn(data, count, outPeak, outRMS);
}

void processBiquad_Scalar(float *data, size_t count, const BiquadCoeffs *coeffs,
                          BiquadState *states, int numStages) {
  for (int stage = 0; stage < numStages; ++stage) {
    const BiquadCoeffs &c = coeffs[stage];
    BiquadState &s = states[stage];
    for (size_t i = 0; i < count; ++i) {
      float x0 = data[i];
      float y0 =
          c.b0 * x0 + c.b1 * s.x1 + c.b2 * s.x2 - c.a1 * s.y1 - c.a2 * s.y2;
      s.x2 = s.x1;
      s.x1 = x0;
      s.y2 = s.y1;
      s.y1 = y0;
      data[i] = y0;
    }
  }
}

void processBiquad(float *data, size_t count, const BiquadCoeffs *coeffs,
                   BiquadState *states, int numStages) {
  processBiquad_Scalar(data, count, coeffs, states, numStages);
}

}
