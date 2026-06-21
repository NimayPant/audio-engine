#include <iostream>
#include <string>
#include <vector>
#include "../include/AudioProcessor.h"
#include "../include/Benchmark.h"

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        std::string firstArg = argv[1];
        if (firstArg == "--benchmark") {
            benchmark::runAllBenchmarks();
            return 0;
        }
        if (firstArg == "--detect-cpu") {
            benchmark::printCpuFeatures();
            return 0;
        }
    }

    if (argc < 3) {
        std::cout << "Usage: dsp input.wav output.wav [--effect param1 param2 ...]" << std::endl;
        std::cout << "Advanced Engine Flags:" << std::endl;
        std::cout << "  --benchmark             Run SIMD performance tests" << std::endl;
        std::cout << "  --detect-cpu            Show CPU SIMD capabilities" << std::endl;
        std::cout << "Effects:" << std::endl;
        std::cout << "  --delay ms feedback" << std::endl;
        std::cout << "  --gain-simd gain" << std::endl;
        std::cout << "  --biquad-simd freq Q gainDb" << std::endl;
        std::cout << "  --smart-compress        ML-powered compression" << std::endl;
        std::cout << "  --smart-gate            ML-powered noise gate" << std::endl;
        std::cout << "  --ml-dynamic-eq         ML-powered dynamic EQ" << std::endl;
        std::cout << "  --genre-aware           Genre-optimized processing" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    AudioProcessor processor;
    if (!processor.loadAudio(inputFile)) {
        std::cerr << "Failed to load audio file: " << inputFile << std::endl;
        return 1;
    }

    std::vector<std::pair<std::string, std::vector<float>>> effects;
    int i = 3;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg.length() > 2 && arg.substr(0, 2) == "--") {
            std::string effect = arg.substr(2);
            std::vector<float> params;
                ++i;
                while (i < argc) {
                    std::string candidate = argv[i];
                    if (candidate.size() >= 2 && candidate[0] == '-' && candidate[1] == '-') break;
                    try {
                        params.push_back(std::stof(candidate));
                        ++i;
                    } catch (...) {
                        break;
                    }
                }
            effects.push_back({effect, params});
        } else {
            ++i;
        }
    }

    for (const auto& eff : effects) {
        float p[4] = {0,0,0,0};
        for (size_t j = 0; j < eff.second.size() && j < 4; ++j) {
            p[j] = eff.second[j];
        }
        processor.applyEffect(eff.first, p[0], p[1], p[2], p[3]);
    }

    bool saveSuccess = false;
    size_t dot = outputFile.rfind('.');
    std::string ext = "";
    if (dot != std::string::npos) {
        ext = outputFile.substr(dot + 1);
        for (auto &c : ext) c = std::tolower(c);
    }

    if (ext == "mp3") {
        saveSuccess = processor.saveAsMP3(outputFile);
    } else {
        saveSuccess = processor.saveAudio(outputFile);
    }

    if (!saveSuccess) {
        std::cerr << "Failed to save audio file: " << outputFile << std::endl;
        return 1;
    }

    std::cout << "Successfully processed audio." << std::endl;
    return 0;
}