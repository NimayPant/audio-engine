#include "PluginProcessor.h"
#include "PluginEditor.h"

DspPluginEditor::DspPluginEditor(DspPluginProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p), visualizer(p)
{
    setLookAndFeel(&customLookAndFeel);

    auto setupSlider = [this](juce::Slider& slider, juce::Label& label, const juce::String& text) {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
        addAndMakeVisible(slider);
        
        label.setText(text, juce::dontSendNotification);
        label.setFont(juce::Font(10.0f, juce::Font::bold));
        addAndMakeVisible(label);
    };

    setupSlider(eqFreqSlider, eqFreqLabel, "EQ FREQ");
    setupSlider(eqQSlider, eqQLabel, "EQ Q");
    setupSlider(eqGainSlider, eqGainLabel, "EQ GAIN");
    setupSlider(masterGainSlider, masterGainLabel, "MASTER GAIN");

    setupSlider(delayTimeSlider, delayTimeLabel, "DELAY TIME");
    setupSlider(delayFbSlider, delayFbLabel, "DELAY FB");
    setupSlider(revSizeSlider, revSizeLabel, "REVERB SIZE");
    setupSlider(revMixSlider, revMixLabel, "REVERB MIX");

    addAndMakeVisible(visualizer);

    eqFreqAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "eq_freq", eqFreqSlider);
    eqQAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "eq_q", eqQSlider);
    eqGainAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "eq_gain", eqGainSlider);
    masterGainAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "master_gain", masterGainSlider);

    delayTimeAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "delay_time", delayTimeSlider);
    delayFbAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "delay_fb", delayFbSlider);
    revSizeAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "rev_size", revSizeSlider);
    revMixAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(audioProcessor.parameters, "rev_mix", revMixSlider);

    setSize(800, 500);
}

DspPluginEditor::~DspPluginEditor()
{
    setLookAndFeel(nullptr);
}

void DspPluginEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff121212));
    
    g.setColour(juce::Colour(0xff151515));
    g.fillRect(0, 0, getWidth(), 160);
    g.setColour(juce::Colour(0xff333333));
    g.drawLine(0, 160, static_cast<float>(getWidth()), 160);

    g.setColour(juce::Colour(0xff333333));
    g.fillRect(0, 160, getWidth(), getHeight() - 160);

    int moduleWidth = getWidth() / 2;
    int moduleHeight = getHeight() - 160;

    for (int i = 0; i < 2; ++i)
    {
        int x = i * moduleWidth;
        juce::Rectangle<int> moduleRect(x + 1, 161, moduleWidth - 2, moduleHeight - 2);
        g.setColour(juce::Colour(0xff1e1e1e));
        g.fillRect(moduleRect);

        g.setColour(juce::Colour(0xffe0e0e0));
        g.setFont(juce::Font(14.0f, juce::Font::bold));
        juce::String title = (i == 0) ? "EQUALIZER" : "SPACE (DELAY & REVERB)";
        g.drawText(title, x + 24, 180, moduleWidth - 48, 20, juce::Justification::left, false);
    }
}

void DspPluginEditor::resized()
{
    visualizer.setBounds(24, 24, getWidth() - 48, 120);

    int moduleWidth = getWidth() / 2;
    int startY = 220;
    int spacing = 60;
    int padding = 24;

    auto layoutSlider = [&](int col, int index, juce::Label& label, juce::Component& comp) {
        int x = col * moduleWidth + padding;
        int y = startY + (index * spacing);
        int w = moduleWidth - (padding * 2);
        
        label.setBounds(x, y, w, 20);
        comp.setBounds(x, y + 20, w, 20);
    };

    layoutSlider(0, 0, eqFreqLabel, eqFreqSlider);
    layoutSlider(0, 1, eqQLabel, eqQSlider);
    layoutSlider(0, 2, eqGainLabel, eqGainSlider);
    layoutSlider(0, 3, masterGainLabel, masterGainSlider);

    layoutSlider(1, 0, delayTimeLabel, delayTimeSlider);
    layoutSlider(1, 1, delayFbLabel, delayFbSlider);
    layoutSlider(1, 2, revSizeLabel, revSizeSlider);
    layoutSlider(1, 3, revMixLabel, revMixSlider);
}
