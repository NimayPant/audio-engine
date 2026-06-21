cmake_minimum_required(VERSION 3.15)
project(DSP_Plugin VERSION 1.0.0)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    juce
    URL https://github.com/juce-framework/JUCE/archive/refs/tags/7.0.5.zip
)
FetchContent_MakeAvailable(juce)

juce_add_plugin(DSP_VST
    COMPANY_NAME "NimayTheLegend"
    PLUGIN_MANUFACTURER_CODE "Nims"
    PLUGIN_CODE "Ldsp"
    FORMATS VST3 Standalone
    PRODUCT_NAME "Legendary DSP"
    VERSION "1.0.0"
)

juce_generate_juce_header(DSP_VST)


set(BACKEND_SOURCES
    ../src/Effects.cpp
    ../src/SimdOps.cpp
    ../src/FeatureExtractor.cpp
    ../src/MLModels.cpp
)

target_sources(DSP_VST PRIVATE
    PluginProcessor.cpp
    PluginEditor.cpp
    ${BACKEND_SOURCES}
)

target_compile_features(DSP_VST PUBLIC cxx_std_17)


if(MSVC)
    target_compile_options(DSP_VST PRIVATE /arch:AVX2)
else()
    target_compile_options(DSP_VST PRIVATE -mavx2 -mfma)
endif()


target_compile_definitions(DSP_VST PUBLIC JUCE_VST3_CAN_REPLACE_VST2=0)

target_include_directories(DSP_VST PRIVATE 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/../include
)

target_link_libraries(DSP_VST PRIVATE juce::juce_audio_utils juce::juce_dsp)
