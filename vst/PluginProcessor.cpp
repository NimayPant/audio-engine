#include "PluginProcessor.h"
#include "PluginEditor.h"

juce::AudioProcessorValueTreeState::ParameterLayout DspPluginProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    params.push_back(std::make_unique<juce::AudioParameterFloat>("master_gain", "Master Gain", -24.0f, 24.0f, 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("eq_freq", "EQ Freq", 20.0f, 20000.0f, 1000.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("eq_q", "EQ Q", 0.1f, 10.0f, 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("eq_gain", "EQ Gain", -24.0f, 24.0f, 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("delay_time", "Delay Time", 0.0f, 2000.0f, 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("delay_fb", "Delay Feedback", 0.0f, 1.0f, 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("rev_size", "Reverb Size", 0.0f, 1.0f, 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("rev_mix", "Reverb Mix", 0.0f, 1.0f, 0.0f));
    return { params.begin(), params.end() };
}

DspPluginProcessor::DspPluginProcessor()
     : AudioProcessor (BusesProperties()
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
       parameters(*this, nullptr, "PARAMETERS", createParameterLayout())
{
    visualizerBuffer.fill(0.0f);
    visualizerWritePos.store(0);
}

DspPluginProcessor::~DspPluginProcessor()
{
}

void DspPluginProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    visualizerBuffer.fill(0.0f);
    visualizerWritePos.store(0);
    
    delayBuffer.setSize(2, sampleRate * 2.5);
    delayBuffer.clear();
    delayWritePos = 0;
    
    float combTimes[] = {0.0297f, 0.0371f, 0.0411f, 0.0437f};
    for (int i = 0; i < numCombs; ++i) {
        int combSamples = static_cast<int>(combTimes[i] * sampleRate);
        combBuffersL[i].setSize(1, combSamples);
        combBuffersL[i].clear();
        combBuffersR[i].setSize(1, combSamples);
        combBuffersR[i].clear();
        combIndicesL[i] = 0;
        combIndicesR[i] = 0;
    }
    
    eqStateL = simd::BiquadState();
    eqStateR = simd::BiquadState();
}

void DspPluginProcessor::releaseResources()
{
}

bool DspPluginProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    return true;
}

void DspPluginProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    if (totalNumInputChannels < 2) return;

    float* leftData = buffer.getWritePointer(0);
    float* rightData = buffer.getWritePointer(1);
    size_t numSamples = buffer.getNumSamples();

    float eqFreq = parameters.getRawParameterValue("eq_freq")->load();
    float eqQ = parameters.getRawParameterValue("eq_q")->load();
    float eqGain = parameters.getRawParameterValue("eq_gain")->load();
    float delayTimeMs = parameters.getRawParameterValue("delay_time")->load();
    float delayFb = parameters.getRawParameterValue("delay_fb")->load();
    float revSize = parameters.getRawParameterValue("rev_size")->load();
    float revMix = parameters.getRawParameterValue("rev_mix")->load();
    float masterGainDb = parameters.getRawParameterValue("master_gain")->load();

    float A = pow(10.0f, eqGain / 40.0f);
    float omega = 2.0f * 3.14159265358979323846f * eqFreq / getSampleRate();
    float sn = sin(omega);
    float cs = cos(omega);
    float alpha = sn / (2.0f * eqQ);

    simd::BiquadCoeffs coeffs;
    float a0 = 1.0f + alpha / A;
    coeffs.b0 = (1.0f + alpha * A) / a0;
    coeffs.b1 = (-2.0f * cs) / a0;
    coeffs.b2 = (1.0f - alpha * A) / a0;
    coeffs.a1 = (-2.0f * cs) / a0;
    coeffs.a2 = (1.0f - alpha / A) / a0;

    simd::processBiquad(leftData, numSamples, &coeffs, &eqStateL, 1);
    simd::processBiquad(rightData, numSamples, &coeffs, &eqStateR, 1);

    if (delayTimeMs > 0.1f) {
        int delaySamples = static_cast<int>((delayTimeMs / 1000.0f) * getSampleRate());
        for (size_t i = 0; i < numSamples; ++i) {
            float dl = delayBuffer.getSample(0, delayWritePos);
            float dr = delayBuffer.getSample(1, delayWritePos);
            
            delayBuffer.setSample(0, delayWritePos, leftData[i] + dl * delayFb);
            delayBuffer.setSample(1, delayWritePos, rightData[i] + dr * delayFb);
            
            leftData[i] += dl;
            rightData[i] += dr;
            
            delayWritePos = (delayWritePos + 1) % delayBuffer.getNumSamples();
        }
    }
    
    if (revMix > 0.01f) {
        float fb = 0.7f + revSize * 0.28f;
        for (size_t i = 0; i < numSamples; ++i) {
            float inL = leftData[i], inR = rightData[i];
            float outL = 0.0f, outR = 0.0f;
            
            for (int c = 0; c < numCombs; ++c) {
                float dl = combBuffersL[c].getSample(0, combIndicesL[c]);
                combBuffersL[c].setSample(0, combIndicesL[c], inL + dl * fb);
                outL += dl;
                combIndicesL[c] = (combIndicesL[c] + 1) % combBuffersL[c].getNumSamples();
                
                float dr = combBuffersR[c].getSample(0, combIndicesR[c]);
                combBuffersR[c].setSample(0, combIndicesR[c], inR + dr * fb);
                outR += dr;
                combIndicesR[c] = (combIndicesR[c] + 1) % combBuffersR[c].getNumSamples();
            }
            
            leftData[i] = inL * (1.0f - revMix) + (outL / numCombs) * revMix;
            rightData[i] = inR * (1.0f - revMix) + (outR / numCombs) * revMix;
        }
    }

    if (std::abs(masterGainDb) > 0.01f) {
        float mg = std::pow(10.0f, masterGainDb / 20.0f);
        simd::applyGain(leftData, numSamples, mg);
        simd::applyGain(rightData, numSamples, mg);
    }

    for (size_t i = 0; i < numSamples; ++i) {
        float mono = (leftData[i] + rightData[i]) * 0.5f;
        int pos = visualizerWritePos.load(std::memory_order_relaxed);
        visualizerBuffer[pos] = mono;
        visualizerWritePos.store((pos + 1) % 2048, std::memory_order_relaxed);
    }
}

juce::AudioProcessorEditor* DspPluginProcessor::createEditor()
{
    return new DspPluginEditor(*this);
}

void DspPluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void DspPluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xmlState));
}

bool DspPluginProcessor::hasEditor() const
{
    return true; 
}

const juce::String DspPluginProcessor::getName() const
{
    return "Legendary DSP";
}

bool DspPluginProcessor::acceptsMidi() const
{
    return false;
}

bool DspPluginProcessor::producesMidi() const
{
    return false;
}

bool DspPluginProcessor::isMidiEffect() const
{
    return false;
}

double DspPluginProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int DspPluginProcessor::getNumPrograms()
{
    return 1;
}

int DspPluginProcessor::getCurrentProgram()
{
    return 0;
}

void DspPluginProcessor::setCurrentProgram (int index)
{
}

const juce::String DspPluginProcessor::getProgramName (int index)
{
    return {};
}

void DspPluginProcessor::changeProgramName (int index, const juce::String& newName)
{
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DspPluginProcessor();
}
