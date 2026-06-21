#include "../include/MLModels.h"
#include "../include/MLWeights.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace ml {

void MLP::forward(const float* input, int inDim, int outDim, const float* weights, const float* bias, float* output, bool useRelu) {
    for (int i = 0; i < outDim; ++i) {
        float sum = bias[i];
        for (int j = 0; j < inDim; ++j) {
            sum += input[j] * weights[i * inDim + j];
        }
        if (useRelu) {
            output[i] = std::max(0.0f, sum);
        } else {
            output[i] = sum;
        }
    }
}

void MLP::sigmoid(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

CompressorPredictor::Prediction CompressorPredictor::predict(const float* features) {
    float hidden0[ml_weights::kCompL0_Out];
    float hidden1[ml_weights::kCompL1_Out];
    float output[ml_weights::kCompL2_Out];

    MLP::forward(features, ml_weights::kCompL0_In, ml_weights::kCompL0_Out, ml_weights::kCompW0, ml_weights::kCompB0, hidden0);
    MLP::forward(hidden0, ml_weights::kCompL1_In, ml_weights::kCompL1_Out, ml_weights::kCompW1, ml_weights::kCompB1, hidden1);
    MLP::forward(hidden1, ml_weights::kCompL2_In, ml_weights::kCompL2_Out, ml_weights::kCompW2, ml_weights::kCompB2, output, false);

    Prediction p;
    p.thresholdDb = std::max(-60.0f, std::min(0.0f, output[0]));
    p.ratio = std::max(1.0f, std::min(20.0f, output[1]));
    return p;
}

float NoiseClassifier::predict(const float* features) {
    float hidden0[ml_weights::kNoiseL0_Out];
    float hidden1[ml_weights::kNoiseL1_Out];
    float output[ml_weights::kNoiseL2_Out];

    MLP::forward(features, ml_weights::kNoiseL0_In, ml_weights::kNoiseL0_Out, ml_weights::kNoiseW0, ml_weights::kNoiseB0, hidden0);
    MLP::forward(hidden0, ml_weights::kNoiseL1_In, ml_weights::kNoiseL1_Out, ml_weights::kNoiseW1, ml_weights::kNoiseB1, hidden1);
    MLP::forward(hidden1, ml_weights::kNoiseL2_In, ml_weights::kNoiseL2_Out, ml_weights::kNoiseW2, ml_weights::kNoiseB2, output, false);

    MLP::sigmoid(output, 1);
    return output[0];
}

const char* GenreClassifier::getGenreName(int genreId) {
    static const char* genres[] = {
        "Pop", "Electronic", "Rock", "Hip-Hop", "Folk", "Experimental",
        "International", "Indie", "Jazz", "Soul", "Punk", "Metal",
        "RnB", "Dance", "Latin", "Classical"
    };
    if (genreId < 0 || genreId >= 16) return "Unknown";
    return genres[genreId];
}

GenreClassifier::Prediction GenreClassifier::predict(const float* features) {
    Prediction pred;
    pred.genreId = 0;
    pred.confidence = 0.0f;
    std::memset(pred.genreProbs, 0, sizeof(pred.genreProbs));

    float bassSum = 0.0f, trebleSum = 0.0f;
    for (int i = 0;  i < 8;  ++i) bassSum   += features[i];
    for (int i = 24; i < 32; ++i) trebleSum += features[i];
    float bassAvg   = bassSum   / 8.0f;
    float trebleAvg = trebleSum / 8.0f;

    float centroid  = features[90];
    float rolloff   = features[91];
    float zcr       = features[92];
    float rms       = features[94];
    float flatness  = features[95];
    float mfcc0     = features[64];
    float mfcc1     = features[65];

    pred.genreProbs[0] = 0.08f
        + 0.12f * (centroid > 1500.0f && centroid < 4500.0f ? 1.0f : 0.0f)
        + 0.05f * (flatness < 0.3f ? 1.0f : 0.0f);

    pred.genreProbs[1] = 0.03f
        + 0.35f * flatness
        + 0.10f * std::min(1.0f, zcr * 5.0f)
        + 0.05f * (centroid > 3000.0f ? 1.0f : 0.0f);

    pred.genreProbs[2] = 0.05f
        + 0.12f * (bassAvg > -30.0f ? 1.0f : 0.0f)
        + 0.08f * (rms > 0.15f ? 1.0f : 0.0f)
        + 0.05f * (trebleAvg > -40.0f && centroid > 1000.0f ? 1.0f : 0.0f);

    pred.genreProbs[3] = 0.04f
        + 0.20f * (bassAvg > -25.0f && centroid < 2500.0f ? 1.0f : 0.0f)
        + 0.10f * std::min(1.0f, zcr * 3.0f)
        + 0.05f * (mfcc0 > 5.0f ? 1.0f : 0.0f);

    pred.genreProbs[4] = 0.04f
        + 0.15f * (flatness < 0.15f ? 1.0f : 0.0f)
        + 0.10f * (centroid > 500.0f && centroid < 2500.0f ? 1.0f : 0.0f)
        + 0.05f * (rms < 0.15f ? 1.0f : 0.0f);

    pred.genreProbs[5] = 0.03f
        + 0.25f * (flatness > 0.5f ? 1.0f : 0.0f)
        + 0.10f * (centroid > 6000.0f || centroid < 300.0f ? 1.0f : 0.0f);

    pred.genreProbs[6] = 0.04f
        + 0.10f * (flatness < 0.2f && centroid > 800.0f && centroid < 3000.0f ? 1.0f : 0.0f);

    pred.genreProbs[7] = 0.04f
        + 0.10f * (centroid > 1000.0f && centroid < 4000.0f ? 1.0f : 0.0f)
        + 0.05f * (flatness < 0.25f ? 1.0f : 0.0f);

    pred.genreProbs[8] = 0.04f
        + 0.20f * (flatness < 0.12f ? 1.0f : 0.0f)
        + 0.10f * (zcr < 0.05f ? 1.0f : 0.0f)
        + 0.08f * (centroid > 800.0f && centroid < 3500.0f ? 1.0f : 0.0f);

    pred.genreProbs[9] = 0.04f
        + 0.12f * (centroid < 2500.0f && flatness < 0.2f ? 1.0f : 0.0f)
        + 0.08f * (mfcc1 < 0.0f ? 1.0f : 0.0f);

    pred.genreProbs[10] = 0.03f
        + 0.25f * std::min(1.0f, zcr * 4.0f)
        + 0.10f * (rms > 0.2f ? 1.0f : 0.0f);

    pred.genreProbs[11] = 0.03f
        + 0.20f * (bassAvg > -20.0f && rms > 0.25f ? 1.0f : 0.0f)
        + 0.10f * std::min(1.0f, zcr * 3.5f);

    pred.genreProbs[12] = 0.04f
        + 0.12f * (zcr < 0.04f ? 1.0f : 0.0f)
        + 0.10f * (centroid < 3000.0f && flatness < 0.25f ? 1.0f : 0.0f);

    pred.genreProbs[13] = 0.03f
        + 0.15f * flatness
        + 0.10f * std::min(1.0f, zcr * 4.0f)
        + 0.08f * (rms > 0.2f ? 1.0f : 0.0f);

    pred.genreProbs[14] = 0.04f
        + 0.12f * std::min(1.0f, zcr * 3.0f)
        + 0.08f * (centroid > 1000.0f && centroid < 3500.0f ? 1.0f : 0.0f);

    pred.genreProbs[15] = 0.04f
        + 0.25f * (flatness < 0.08f ? 1.0f : 0.0f)
        + 0.10f * (rms < 0.10f ? 1.0f : 0.0f)
        + 0.08f * (trebleAvg > bassAvg ? 1.0f : 0.0f);

    float sum = 0.0f;
    const float power = 2.5f;
    for (int i = 0; i < 16; ++i) {
        if (pred.genreProbs[i] < 0.0f) pred.genreProbs[i] = 0.001f;
        pred.genreProbs[i] = std::pow(pred.genreProbs[i], power);
        sum += pred.genreProbs[i];
    }

    int maxIdx = 0;
    float maxProb = 0.0f;
    for (int i = 0; i < 16; ++i) {
        pred.genreProbs[i] /= (sum + 1e-9f);
        if (pred.genreProbs[i] > maxProb) {
            maxProb = pred.genreProbs[i];
            maxIdx = i;
        }
    }

    pred.genreId = maxIdx;
    pred.confidence = maxProb;
    return pred;
}

EQPredictor::EQGains EQPredictor::predict(const float* features) {
    EQGains gains{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float global_mean = 0.0f;
    for (int i = 0; i < 32; ++i) global_mean += features[i];
    global_mean /= 32.0f;

    for (int band = 0; band < 8; ++band) {
        int start_bin = band * 4;
        float band_mean = 0.0f;
        for (int b = start_bin; b < start_bin + 4; ++b) band_mean += features[b];
        band_mean /= 4.0f;
        
        float relative_energy = band_mean - global_mean;
        float target_offset = 0.0f;
        if (band < 2) target_offset = 3.0f;
        else if (band > 5) target_offset = -2.0f;

        float diff = target_offset - relative_energy;
        gains.gains[band] = std::max(-8.0f, std::min(8.0f, diff * 0.5f));
    }
    return gains;
}

}
