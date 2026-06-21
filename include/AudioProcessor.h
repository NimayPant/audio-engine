#ifndef AUDIOPROCESSOR_H
#define AUDIOPROCESSOR_H

#include <vector>
#include <string>
#include <sndfile.h>

class AudioProcessor {
public:
    AudioProcessor();
    ~AudioProcessor();

    bool loadAudio(const std::string& filename);
    bool loadWAV(const std::string& filename);
    bool loadMP3(const std::string& filename);
    bool saveAudio(const std::string& filename);
    bool saveAsMP3(const std::string& filename, int bitrate = 192);

    void applyEffect(const std::string& effect, float param1 = 0.0f, float param2 = 0.0f, float param3 = 0.0f, float param4 = 0.0f);

    std::vector<float> leftChannel;
    std::vector<float> rightChannel;
    int sampleRate;
    int channels;
    
private:
    std::string getFileExtension(const std::string& filename);
};

#endif