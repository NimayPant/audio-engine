#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"

class FlatLookAndFeel : public juce::LookAndFeel_V4
{
public:
    FlatLookAndFeel()
    {
        setColour(juce::Slider::thumbColourId, juce::Colour(0xff888888));
        setColour(juce::Slider::trackColourId, juce::Colour(0xff4b7bec));
        setColour(juce::Slider::backgroundColourId, juce::Colour(0xff333333));
        setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff2a2a2a));
        setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff444444));
        setColour(juce::ComboBox::textColourId, juce::Colour(0xffe0e0e0));
        setColour(juce::Label::textColourId, juce::Colour(0xff888888));
    }
    
    void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float minSliderPos, float maxSliderPos,
                          const juce::Slider::SliderStyle style, juce::Slider& slider) override
    {
        auto trackRect = juce::Rectangle<float>(static_cast<float>(x), static_cast<float>(y + height / 2 - 2), static_cast<float>(width), 4.0f);
        g.setColour(slider.findColour(juce::Slider::backgroundColourId));
        g.fillRect(trackRect);
        
        auto fillRect = trackRect.withWidth(sliderPos - trackRect.getX());
        g.setColour(slider.findColour(juce::Slider::trackColourId));
        g.fillRect(fillRect);
        
        g.setColour(slider.findColour(juce::Slider::thumbColourId));
        g.fillRect(sliderPos - 4.0f, static_cast<float>(y + height / 2 - 8), 8.0f, 16.0f);
    }
};

class VisualizerComponent : public juce::Component, public juce::Timer
{
public:
    VisualizerComponent(DspPluginProcessor& p) : processor(p)
    {
        startTimerHz(60);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff000000));
        g.setColour(juce::Colour(0xff222222));
        g.drawRect(getLocalBounds(), 1);

        juce::Path wavePath;
        int writePos = processor.visualizerWritePos.load(std::memory_order_relaxed);
        
        float width = static_cast<float>(getWidth());
        float height = static_cast<float>(getHeight());
        float centerY = height / 2.0f;
        
        wavePath.startNewSubPath(0, centerY);
        
        for (int i = 0; i < width; ++i)
        {
            int bufferIndex = (writePos - static_cast<int>(width) + i + 2048) % 2048;
            float sample = processor.visualizerBuffer[bufferIndex];
            float y = centerY - (sample * centerY * 0.8f);
            wavePath.lineTo(static_cast<float>(i), y);
        }

        g.setColour(juce::Colour(0xff4b7bec));
        g.strokePath(wavePath, juce::PathStrokeType(2.0f));
    }

    void timerCallback() override
    {
        repaint();
    }

private:
    DspPluginProcessor& processor;
};

class DspPluginEditor : public juce::AudioProcessorEditor
{
public:
    DspPluginEditor(DspPluginProcessor&);
    ~DspPluginEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    DspPluginProcessor& audioProcessor;
    FlatLookAndFeel customLookAndFeel;
    VisualizerComponent visualizer;

    juce::Slider eqFreqSlider, eqQSlider, eqGainSlider, masterGainSlider;
    juce::Slider delayTimeSlider, delayFbSlider, revSizeSlider, revMixSlider;

    juce::Label eqFreqLabel, eqQLabel, eqGainLabel, masterGainLabel;
    juce::Label delayTimeLabel, delayFbLabel, revSizeLabel, revMixLabel;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> eqFreqAttach, eqQAttach, eqGainAttach, masterGainAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> delayTimeAttach, delayFbAttach, revSizeAttach, revMixAttach;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DspPluginEditor)
};
