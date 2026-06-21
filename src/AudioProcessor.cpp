#include "../include/AudioProcessor.h"
#include "../include/Effects.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#define DR_MP3_IMPLEMENTATION
#include "../include/dr_mp3.h"

AudioProcessor::AudioProcessor() : sampleRate(44100), channels(2) {}

AudioProcessor::~AudioProcessor() {}

std::string AudioProcessor::getFileExtension(const std::string &filename) {
  size_t dot = filename.rfind('.');
  if (dot == std::string::npos)
    return "";

  std::string ext = filename.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext;
}

bool AudioProcessor::loadAudio(const std::string &filename) {
  std::string ext = getFileExtension(filename);

  if (ext == "mp3") {
    return loadMP3(filename);
  } else {
    return loadWAV(filename);
  }
}

bool AudioProcessor::loadWAV(const std::string &filename) {
  SF_INFO sfinfo;
  SNDFILE *file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
  if (!file) {
    std::cerr << "Error opening WAV file: " << filename << std::endl;
    return false;
  }

  sampleRate = sfinfo.samplerate;
  channels = sfinfo.channels;

  std::vector<float> buffer(sfinfo.frames * channels);
  sf_count_t framesRead = sf_readf_float(file, buffer.data(), sfinfo.frames);
  sf_close(file);

  if (framesRead != sfinfo.frames) {
    std::cerr << "Error reading frames from file" << std::endl;
    return false;
  }

  leftChannel.resize(sfinfo.frames);
  rightChannel.resize(sfinfo.frames);

  for (sf_count_t i = 0; i < sfinfo.frames; ++i) {
    leftChannel[i] = buffer[i * channels];
    rightChannel[i] =
        (channels > 1) ? buffer[i * channels + 1] : buffer[i * channels];
  }

  return true;
}

bool AudioProcessor::loadMP3(const std::string &filename) {
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, filename.c_str(), NULL)) {
        std::cerr << "Failed to open MP3 file: " << filename << std::endl;
        return false;
    }

    sampleRate = mp3.sampleRate;
    channels   = mp3.channels;

    drmp3_uint64 totalFrames = drmp3_get_pcm_frame_count(&mp3);
    
    drmp3_seek_to_pcm_frame(&mp3, 0);

    std::vector<float> buffer(totalFrames * channels);
    drmp3_uint64 framesRead = drmp3_read_pcm_frames_f32(&mp3, totalFrames, buffer.data());
    
    drmp3_uninit(&mp3);

    if (framesRead == 0 && totalFrames > 0) {
        std::cerr << "Error: failed to read MP3 frames after seeking" << std::endl;
        return false;
    }

    leftChannel.resize(framesRead);
    rightChannel.resize(framesRead);
    for (drmp3_uint64 i = 0; i < framesRead; ++i) {
        leftChannel[i]  = buffer[i * channels];
        rightChannel[i] = (channels > 1) ? buffer[i * channels + 1] : buffer[i * channels];
    }
    return true;
}

bool AudioProcessor::saveAudio(const std::string &filename) {
  if (leftChannel.empty())
    return false;

  SF_INFO sfinfo;
  sfinfo.samplerate = sampleRate;
  sfinfo.channels = channels;
  sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

  SNDFILE *file = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
  if (!file) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return false;
  }

  auto limit = [](float x) {
    if (std::abs(x) < 0.9f)
      return x;
    return (x > 0) ? (0.9f + 0.1f * std::tanh((x - 0.9f) / 0.1f))
                   : (-0.9f + 0.1f * std::tanh((x + 0.9f) / 0.1f));
  };

  std::vector<float> buffer(leftChannel.size() * channels);
  for (size_t i = 0; i < leftChannel.size(); ++i) {
    buffer[i * channels] = limit(leftChannel[i]);
    if (channels > 1) {
      buffer[i * channels + 1] = limit(rightChannel[i]);
    }
  }

  sf_count_t framesWritten =
      sf_writef_float(file, buffer.data(), leftChannel.size());
  sf_close(file);

  if (framesWritten != static_cast<sf_count_t>(leftChannel.size())) {
    std::cerr << "Error writing frames to file" << std::endl;
    return false;
  }

  return true;
}

bool AudioProcessor::saveAsMP3(const std::string &filename, int bitrate) {
  std::string tmpWav = filename + ".temp.wav";
  if (!saveAudio(tmpWav)) {
    return false;
  }

  std::string cmd = "ffmpeg -y -hide_banner -loglevel error -i \"" + tmpWav + "\" -q:a 5 \"" + filename + "\"";
  int result = system(cmd.c_str());

  std::remove(tmpWav.c_str());

  return result == 0;
}


void AudioProcessor::applyEffect(const std::string &effect, float param1,
                                 float param2, float param3, float param4) {
  if (effect == "delay") {
    Effects::applyDelay(leftChannel, sampleRate, param1, param2);
    if (channels > 1)
      Effects::applyDelay(rightChannel, sampleRate, param1, param2);
  } else if (effect == "reverb") {
    Effects::applyReverb(leftChannel, sampleRate, param1, param2, param3);
    if (channels > 1)
      Effects::applyReverb(rightChannel, sampleRate, param1, param2, param3);
  } else if (effect == "eq") {
    Effects::applyParametricEQ(leftChannel, sampleRate, param1, param2, param3);
    if (channels > 1)
      Effects::applyParametricEQ(rightChannel, sampleRate, param1, param2,
                                 param3);
  } else if (effect == "gain-simd") {
    Effects::applyGainSIMD(leftChannel, param1);
    if (channels > 1)
      Effects::applyGainSIMD(rightChannel, param1);
  } else if (effect == "smart-compress") {
    Effects::applySmartCompression(leftChannel, sampleRate, param1);
    if (channels > 1)
      Effects::applySmartCompression(rightChannel, sampleRate, param1);
  } else if (effect == "smart-gate") {
    Effects::applySmartGate(leftChannel, sampleRate, param1);
    if (channels > 1)
      Effects::applySmartGate(rightChannel, sampleRate, param1);
  } else if (effect == "ml-dynamic-eq") {
    Effects::applyMLDynamicEQ(leftChannel, sampleRate, param1);
    if (channels > 1)
      Effects::applyMLDynamicEQ(rightChannel, sampleRate, param1);
  } else if (effect == "genre-aware") {
    Effects::applyGenreAwareProcessing(leftChannel, sampleRate,
                                       static_cast<int>(param1), param2);
    if (channels > 1)
      Effects::applyGenreAwareProcessing(rightChannel, sampleRate,
                                         static_cast<int>(param1), param2);
  } else if (effect == "biquad-simd") {
    Effects::applyBiquadSIMD(leftChannel, sampleRate, param1, param2, param3);
    if (channels > 1)
      Effects::applyBiquadSIMD(rightChannel, sampleRate, param1, param2,
                               param3);
  } else {
    std::cerr << "Unknown effect: " << effect << std::endl;
  }
}