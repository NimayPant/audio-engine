#ifndef MLMODELS_H
#define MLMODELS_H

#include <vector>
#include <string>

namespace ml {

class MLP {
public:
    static void forward(const float* input, int inDim, int outDim, 
                       const float* weights, const float* bias, 
                       float* output, bool useRelu = true);
    
    static void sigmoid(float* data, int size);
};

class CompressorPredictor {
public:
    struct Prediction {
        float thresholdDb;
        float ratio;
    };

    static Prediction predict(const float* features);
};

class NoiseClassifier {
public:
    static float predict(const float* features);
};

class GenreClassifier {
public:
    struct Prediction {
        int genreId;
        float confidence;
        float genreProbs[16];
    };

    static Prediction predict(const float* features);
    static const char* getGenreName(int genreId);
};

class EQPredictor {
public:
    struct EQGains {
        float gains[8];
    };

    static EQGains predict(const float* features);
};

}

#endif
