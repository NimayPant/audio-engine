#include "../include/Benchmark.h"
#include "../include/SimdOps.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace benchmark {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

static double timeFunction(std::function<void()> fn, int iterations) {
    fn();
    fn();
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto end = Clock::now();
    Duration elapsed = end - start;
    return elapsed.count() / iterations;
}

static std::vector<float> generateTestBuffer(int size) {
    std::vector<float> buf(size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        buf[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * i / 44100.0)
               + 0.3f * std::sin(2.0 * M_PI * 880.0 * i / 44100.0)
               + 0.2f * dist(rng);
    }
    return buf;
}

static void printHeader(const char* testName) {
    std::cout << "\n+-------------------------------------------------------------+\n";
    std::cout << "| " << std::left << std::setw(60) << testName << "|\n";
    std::cout << "+---------------+--------------+---------------+--------------+\n";
    std::cout << "| Path          | Time (us)    | Throughput    | Speedup      |\n";
    std::cout << "+---------------+--------------+---------------+--------------+\n";
}

static void printRow(const char* path, double timeUs, int bufferSize, double baselineUs) {
    double throughput = static_cast<double>(bufferSize) / timeUs;
    double speedup = baselineUs / timeUs;

    std::cout << "| " << std::left << std::setw(14) << path
              << "| " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << timeUs << "  "
              << "| " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << throughput << " MS/s "
              << "| " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << speedup << "x    "
              << "|\n";
}

static void printFooter() {
    std::cout << "+---------------+--------------+---------------+--------------+\n";
}

static void benchmarkGain(int bufferSize, int iterations) {
    auto original = generateTestBuffer(bufferSize);
    printHeader("Gain Application (data[i] *= gain)");

    auto buf = original;
    double scalarTime = timeFunction([&]() {
        std::memcpy(buf.data(), original.data(), bufferSize * sizeof(float));
        simd::applyGain_Scalar(buf.data(), bufferSize, 0.75f);
    }, iterations);
    printRow("Scalar", scalarTime, bufferSize, scalarTime);

    buf = original;
    double sse2Time = timeFunction([&]() {
        std::memcpy(buf.data(), original.data(), bufferSize * sizeof(float));
        simd::applyGain_SSE2(buf.data(), bufferSize, 0.75f);
    }, iterations);
    printRow("SSE2 (4-wide)", sse2Time, bufferSize, scalarTime);

    const auto& features = simd::detectCpuFeatures();
    if (features.avx2) {
        buf = original;
        double avx2Time = timeFunction([&]() {
            std::memcpy(buf.data(), original.data(), bufferSize * sizeof(float));
            simd::applyGain_AVX2(buf.data(), bufferSize, 0.75f);
        }, iterations);
        printRow("AVX2 (8-wide)", avx2Time, bufferSize, scalarTime);
    } else {
        std::cout << "| AVX2 (8-wide) |     N/A      |     N/A       |     N/A      |\n";
    }
    printFooter();
}

static void benchmarkPeakRMS(int bufferSize, int iterations) {
    auto original = generateTestBuffer(bufferSize);
    float peak, rms;
    printHeader("Peak/RMS Detection (envelope follower)");

    double scalarTime = timeFunction([&]() {
        simd::computePeakRMS_Scalar(original.data(), bufferSize, peak, rms);
    }, iterations);
    printRow("Scalar", scalarTime, bufferSize, scalarTime);

    double sse2Time = timeFunction([&]() {
        simd::computePeakRMS_SSE2(original.data(), bufferSize, peak, rms);
    }, iterations);
    printRow("SSE2 (4-wide)", sse2Time, bufferSize, scalarTime);

    const auto& features = simd::detectCpuFeatures();
    if (features.avx2) {
        double avx2Time = timeFunction([&]() {
            simd::computePeakRMS_AVX2(original.data(), bufferSize, peak, rms);
        }, iterations);
        printRow("AVX2 (8-wide)", avx2Time, bufferSize, scalarTime);
    } else {
        std::cout << "| AVX2 (8-wide) |     N/A      |     N/A       |     N/A      |\n";
    }
    printFooter();

    simd::computePeakRMS(original.data(), bufferSize, peak, rms);
    std::cout << "  Detected: Peak = " << std::fixed << std::setprecision(4)
              << peak << "  RMS = " << rms << "\n";
}

static void benchmarkBiquad(int bufferSize, int iterations) {
    auto original = generateTestBuffer(bufferSize);
    simd::BiquadCoeffs coeffs[4];
    simd::BiquadState states[4];
    float freqs[] = {100.0f, 1000.0f, 4000.0f, 10000.0f};
    float sampleRate = 44100.0f;

    for (int i = 0; i < 4; ++i) {
        float omega = 2.0f * static_cast<float>(M_PI) * freqs[i] / sampleRate;
        float sn = std::sin(omega);
        float cs = std::cos(omega);
        float alpha = sn / (2.0f * 0.707f);
        float A = std::pow(10.0f, 3.0f / 40.0f);
        float a0 = 1.0f + alpha / A;
        coeffs[i].b0 = (1.0f + alpha * A) / a0;
        coeffs[i].b1 = (-2.0f * cs) / a0;
        coeffs[i].b2 = (1.0f - alpha * A) / a0;
        coeffs[i].a1 = (-2.0f * cs) / a0;
        coeffs[i].a2 = (1.0f - alpha / A) / a0;
        states[i] = simd::BiquadState{};
    }

    printHeader("Biquad IIR Filter (4 cascaded stages)");

    auto buf = original;
    simd::BiquadState scalarStates[4];
    double scalarTime = timeFunction([&]() {
        std::memcpy(buf.data(), original.data(), bufferSize * sizeof(float));
        std::memcpy(scalarStates, states, sizeof(states));
        simd::processBiquad_Scalar(buf.data(), bufferSize, coeffs, scalarStates, 4);
    }, iterations);
    printRow("Scalar", scalarTime, bufferSize, scalarTime);

    printFooter();
}

void printCpuFeatures() {
    const auto& f = simd::detectCpuFeatures();
    std::cout << "\n+-----------------------------------------+\n";
    std::cout << "|         CPU SIMD Feature Detection      |\n";
    std::cout << "+-----------------------------------------+\n";

    auto row = [](const char* name, bool supported) {
        std::cout << "|  " << std::left << std::setw(12) << name
                  << (supported ? "V Supported" : "X Not available")
                  << std::setw(supported ? 17 : 13) << "" << "|\n";
    };

    row("SSE2",   f.sse2);
    row("SSE3",   f.sse3);
    row("SSSE3",  f.ssse3);
    row("SSE4.1", f.sse41);
    row("AVX",    f.avx);
    row("AVX2",   f.avx2);
    row("FMA",    f.fma);

    std::cout << "+-----------------------------------------+\n";
    std::cout << "|  Selected path: ";
    if (f.avx2) std::cout << "AVX2 (8 floats/cycle)      |\n";
    else if (f.sse2) std::cout << "SSE2 (4 floats/cycle)      |\n";
    else std::cout << "Scalar (1 float/cycle)     |\n";
    std::cout << "+-----------------------------------------+\n";
}

void runAllBenchmarks(int bufferSize, int iterations) {
    std::cout << "\n================================================================\n";
    std::cout << "       DSP SIMD Benchmark Suite\n";
    std::cout << "================================================================\n";
    std::cout << "  Buffer size:  " << bufferSize << " samples\n";
    std::cout << "  Iterations:   " << iterations << "\n";
    std::cout << "  CPU features: " << simd::getCpuFeatureString() << "\n";

    printCpuFeatures();
    benchmarkGain(bufferSize, iterations);
    benchmarkPeakRMS(bufferSize, iterations);
    benchmarkBiquad(bufferSize, iterations);

    std::cout << "\n================================================================\n";
    std::cout << "  Benchmark complete.\n";
    std::cout << "================================================================\n\n";
}

}
