#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "../include/Effects.h"
#include "../include/SimdOps.h"
#include <atomic>
#include <array>

class DspPluginProcessor : public juce::AudioProcessor
{
public:
    DspPluginProcessor();
    ~DspPluginProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState parameters;

    std::array<float, 2048> visualizerBuffer;
    std::atomic<int> visualizerWritePos { 0 };

private:
    juce::AudioBuffer<float> delayBuffer;
    int delayWritePos = 0;

    static constexpr int numCombs = 4;
    juce::AudioBuffer<float> combBuffersL[numCombs];
    juce::AudioBuffer<float> combBuffersR[numCombs];
    int combIndicesL[numCombs] = {0};
    int combIndicesR[numCombs] = {0};
    
    simd::BiquadState eqStateL, eqStateR;

    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DspPluginProcessor)
};
