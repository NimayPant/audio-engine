cmake_minimum_required(VERSION 3.10)
project(DSP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
endif()

find_package(SndFile REQUIRED)

add_executable(dsp 
    src/main.cpp 
    src/AudioProcessor.cpp 
    src/Effects.cpp
    src/SimdOps.cpp
    src/Benchmark.cpp
    src/FeatureExtractor.cpp
    src/MLModels.cpp
)
target_include_directories(dsp PRIVATE ${PROJECT_SOURCE_DIR}/include ${SndFile_INCLUDE_DIRS})
target_link_libraries(dsp SndFile::sndfile)

add_executable(verify_cpp
    tools/verify_cpp.cpp
    src/AudioProcessor.cpp
    src/Effects.cpp
    src/SimdOps.cpp
    src/FeatureExtractor.cpp
    src/MLModels.cpp
)
target_include_directories(verify_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include ${SndFile_INCLUDE_DIRS})
target_link_libraries(verify_cpp SndFile::sndfile)