cmake_minimum_required(VERSION 3.10)
project(DSP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
endif()

find_package(SndFile QUIET)
if(NOT SndFile_FOUND)
    find_path(SndFile_INCLUDE_DIR NAMES sndfile.h)
    find_library(SndFile_LIBRARY NAMES sndfile)
    if(SndFile_INCLUDE_DIR AND SndFile_LIBRARY)
        add_library(SndFile::sndfile UNKNOWN IMPORTED)
        set_target_properties(SndFile::sndfile PROPERTIES
            IMPORTED_LOCATION "${SndFile_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SndFile_INCLUDE_DIR}"
        )
        set(SndFile_INCLUDE_DIRS ${SndFile_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "SndFile not found! Please install libsndfile.")
    endif()
endif()

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