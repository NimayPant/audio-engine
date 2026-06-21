#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <sndfile.h>
#include "../include/FeatureExtractor.h"
#include "../include/MLModels.h"
#include "../include/AudioProcessor.h"
#include <filesystem>

namespace stdfs = std::filesystem;

void exportBinaryResults(const char* filename,
                         const float* features, int featureDim,
                         const ml::EQPredictor::EQGains& eqGains,
                         const ml::GenreClassifier::Prediction& genre) {
    std::ofstream file(filename, std::ios::binary);
    
    file.write(reinterpret_cast<const char*>(features), featureDim * sizeof(float));
    file.write(reinterpret_cast<const char*>(eqGains.gains), 8 * sizeof(float));
    file.write(reinterpret_cast<const char*>(genre.genreProbs), 16 * sizeof(float));
    file.write(reinterpret_cast<const char*>(&genre.genreId), sizeof(int));
    
    file.close();
    std::cout << "Binary results exported to: " << filename << std::endl;
}

void exportJSONResults(const char* filename,
                       const float* features, int featureDim,
                       const ml::EQPredictor::EQGains& eqGains,
                       const ml::GenreClassifier::Prediction& genre) {
    std::ofstream file(filename);
    
    file << "{\n";
    file << "  \"features_shape\": [" << featureDim << "],\n";
    file << "  \"features_sample\": [";
    
    for (int i = 0; i < std::min(10, featureDim); ++i) {
        if (i > 0) file << ", ";
        file << features[i];
    }
    file << "],\n";
    
    file << "  \"eq_gains\": [";
    for (int i = 0; i < 8; ++i) {
        if (i > 0) file << ", ";
        file << eqGains.gains[i];
    }
    file << "],\n";
    
    file << "  \"genre_id\": " << genre.genreId << ",\n";
    file << "  \"genre_name\": \"" << ml::GenreClassifier::getGenreName(genre.genreId) << "\",\n";
    file << "  \"genre_confidence\": " << genre.confidence << ",\n";
    
    file << "  \"genre_probs_top3\": [\n";
    int topIndices[3] = {0, 0, 0};
    float topProbs[3] = {-1, -1, -1};
    
    for (int i = 0; i < 16; ++i) {
        if (genre.genreProbs[i] > topProbs[0]) {
            topProbs[2] = topProbs[1];
            topIndices[2] = topIndices[1];
            topProbs[1] = topProbs[0];
            topIndices[1] = topIndices[0];
            topProbs[0] = genre.genreProbs[i];
            topIndices[0] = i;
        } else if (genre.genreProbs[i] > topProbs[1]) {
            topProbs[2] = topProbs[1];
            topIndices[2] = topIndices[1];
            topProbs[1] = genre.genreProbs[i];
            topIndices[1] = i;
        } else if (genre.genreProbs[i] > topProbs[2]) {
            topProbs[2] = genre.genreProbs[i];
            topIndices[2] = i;
        }
    }
    
    for (int j = 0; j < 3; ++j) {
        if (j > 0) file << ",\n";
        file << "    {\"genre\": \"" << ml::GenreClassifier::getGenreName(topIndices[j])
             << "\", \"probability\": " << topProbs[j] << "}";
    }
    file << "\n  ]\n";
    file << "}\n";
    
    file.close();
    std::cout << "JSON results exported to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: verify_cpp <audio_file>\n";
        return 1;
    }
    
    std::string audioPath = argv[1];
    
    std::cout << "==========================================================\n";
    std::cout << "C++ ML Verification Script\n";
    std::cout << "==========================================================\n";
    
    AudioProcessor processor;
    
    if (!processor.loadAudio(audioPath)) {
        std::cerr << "Failed to load audio file!\n";
        return 1;
    }
    
    std::cout << "Loaded: " << processor.leftChannel.size() << " samples @ "
              << processor.sampleRate << " Hz\n";
    
    const size_t blockSize = 1024;
    float features[100] = {0};
    if (processor.leftChannel.size() >= blockSize) {
        features::extractMLFeatures(processor.leftChannel.data(), blockSize, processor.sampleRate, features);
    }
    
    if (processor.leftChannel.size() >= blockSize) {
        std::cout << "\nFeatures extracted:\n";
        std::cout << "  Input dimension: 100\n";
        std::cout << "  First 5 features: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << features[i] << " ";
        }
        std::cout << "\n";
        
        std::cout << "\nRunning EQ prediction...\n";
        auto eqGains = ml::EQPredictor::predict(features);
        
        std::cout << "  8-Band EQ Gains (dB):\n";
        for (int i = 0; i < 8; ++i) {
            float centerFreq = 20.0f * std::pow(10.0f, (i + 0.5f) * 3.0f / 8.0f);
            std::cout << "    Band " << i << " (" << centerFreq << " Hz): " 
                      << eqGains.gains[i] << " dB\n";
        }
        
        std::cout << "\nRunning genre classification...\n";
        auto genrePred = ml::GenreClassifier::predict(features);
        
        std::cout << "  Predicted genre: " << ml::GenreClassifier::getGenreName(genrePred.genreId)
                  << " (confidence: " << genrePred.confidence << ")\n";
        
        std::cout << "  Top genres:\n";
        int topIdx[3] = {0, 0, 0};
        float topProbs[3] = {-1, -1, -1};
        
        for (int i = 0; i < 16; ++i) {
            if (genrePred.genreProbs[i] > topProbs[0]) {
                topProbs[2] = topProbs[1]; topIdx[2] = topIdx[1];
                topProbs[1] = topProbs[0]; topIdx[1] = topIdx[0];
                topProbs[0] = genrePred.genreProbs[i]; topIdx[0] = i;
            } else if (genrePred.genreProbs[i] > topProbs[1]) {
                topProbs[2] = topProbs[1]; topIdx[2] = topIdx[1];
                topProbs[1] = genrePred.genreProbs[i]; topIdx[1] = i;
            } else if (genrePred.genreProbs[i] > topProbs[2]) {
                topProbs[2] = genrePred.genreProbs[i]; topIdx[2] = i;
            }
        }
        
        for (int j = 0; j < 3; ++j) {
            std::cout << "    " << (j + 1) << ". " 
                      << ml::GenreClassifier::getGenreName(topIdx[j])
                      << " (" << topProbs[j] << ")\n";
        }
        
        std::string exeDir = stdfs::path(argv[0]).parent_path().string();
        stdfs::path projectRoot = stdfs::path(exeDir).parent_path().parent_path();
        std::string featDir = (projectRoot / "tools" / "extracted_features").string();
        stdfs::create_directories(featDir);
        std::string binOut = (projectRoot / "tools" / "extracted_features" / "inference_output.bin").string();
        std::string jsonOut = (projectRoot / "tools" / "extracted_features" / "cpp_inference_output.json").string();
        
        exportBinaryResults(binOut.c_str(), features, 100, eqGains, genrePred);
        exportJSONResults(jsonOut.c_str(), features, 100, eqGains, genrePred);
        
        std::cout << "\nVerification complete!\n";
    } else {
        std::cerr << "Audio file is too short for feature extraction!\n";
        return 1;
    }
    
    std::cout << "==========================================================\n";
    return 0;
}
