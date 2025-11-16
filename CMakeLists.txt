cmake_minimum_required(VERSION 3.20)
project(OSS20B_Server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find required packages
find_package(CUDAToolkit REQUIRED)


# Set CUDA architecture for RTX 5090 Blackwell (SM 90a) and compatibility
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90") # Support multiple architectures including Blackwell

# CUDA flags - minimal set to avoid compilation conflicts
# RTX 5090 Blackwell architecture optimizations will be handled by CMAKE_CUDA_ARCHITECTURES

# TensorRT paths - use environment variable or default
if(DEFINED ENV{TENSORRT_ROOT})
    set(TensorRT_ROOT $ENV{TENSORRT_ROOT} CACHE PATH "TensorRT installation directory")
else()
    set(TensorRT_ROOT "C:/models/tensorRT" CACHE PATH "TensorRT installation directory")
endif()
set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT}/include")
set(TensorRT_LIB_DIR "${TensorRT_ROOT}/lib")

# Find TensorRT libraries (version 10.x uses versioned names)
find_library(TensorRT_nvinfer_LIBRARY NAMES nvinfer_10 nvinfer HINTS ${TensorRT_LIB_DIR})
find_library(TensorRT_nvonnxparser_LIBRARY NAMES nvonnxparser_10 nvonnxparser HINTS ${TensorRT_LIB_DIR})
find_library(TensorRT_nvinfer_plugin_LIBRARY NAMES nvinfer_plugin_10 nvinfer_plugin HINTS ${TensorRT_LIB_DIR})

if(NOT TensorRT_nvinfer_LIBRARY)
    message(FATAL_ERROR "TensorRT nvinfer library not found. Please set TensorRT_ROOT correctly.")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${TensorRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files - refactored into focused modules
set(ENGINE_SOURCES
    src/inference/engine_core.cpp
    src/inference/request_processor.cpp
    src/inference/tensorrt_engine.cpp
    src/cpp/memory_manager.cpp
    src/cpp/cuda_graphs.cpp
    src/cpp/memory_pool.cpp
)

set(ENGINE_HEADERS
    src/inference/engine_core.hpp
    src/inference/request_processor.hpp
    src/inference/tensorrt_engine.hpp
    src/cpp/memory_manager.hpp
    src/cpp/tokenizer.hpp
    src/cpp/kv_cache.hpp
    src/cpp/cuda_graphs.hpp
    src/cpp/memory_pool.hpp
    src/cpp/simple_json.hpp
)

# Create shared library for Python binding
add_library(oss20b_engine SHARED ${ENGINE_SOURCES} ${ENGINE_HEADERS})

# Link libraries with TensorRT
target_link_libraries(oss20b_engine
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
    ${TensorRT_nvinfer_plugin_LIBRARY}
)


# Set library properties
set_target_properties(oss20b_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Test executables removed - using TensorRT-only components

# TensorRT inference executable removed - using complete_inference instead

# Removed tensorrt_inference target linking

# Create complete inference executable with consolidated TensorRT-only components
add_executable(complete_inference
    src/inference/complete_inference.cpp
    src/inference/engine_core.cpp
    src/inference/request_processor.cpp
    src/inference/tensorrt_engine.cpp
    src/cpp/tokenizer.cpp
    src/cpp/memory_manager.cpp
    src/cpp/memory_pool.cpp
    src/cpp/dynamic_batcher.cpp
    src/cpp/cuda_graphs.cpp
    src/cpp/kv_cache.cpp
    src/cpp/kv_cache_kernels.cu
)

# Create streaming inference example
add_executable(streaming_inference_example
    src/inference/streaming_inference_example.cpp
    src/inference/engine_core.cpp
    src/inference/tensorrt_engine.cpp
    src/cpp/tokenizer.cpp
    src/cpp/memory_manager.cpp
    src/cpp/memory_pool.cpp
    src/cpp/cuda_graphs.cpp
    src/cpp/kv_cache.cpp
    src/cpp/kv_cache_kernels.cu
)

# Create speculative decoding example
add_executable(speculative_decoding_example
    src/inference/speculative_decoding.cpp
    src/inference/engine_core.cpp
    src/inference/tensorrt_engine.cpp
    src/cpp/tokenizer.cpp
    src/cpp/memory_manager.cpp
    src/cpp/memory_pool.cpp
    src/cpp/cuda_graphs.cpp
    src/cpp/kv_cache.cpp
    src/cpp/kv_cache_kernels.cu
)

# Create engine diagnostic tool
add_executable(engine_diagnostic
    src/cpp/engine_diagnostic.cpp
)

target_link_libraries(complete_inference
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
    ${TensorRT_nvinfer_plugin_LIBRARY}
)

target_link_libraries(streaming_inference_example
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
    ${TensorRT_nvinfer_plugin_LIBRARY}
)

target_link_libraries(speculative_decoding_example
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
    ${TensorRT_nvinfer_plugin_LIBRARY}
)

target_link_libraries(engine_diagnostic
    CUDA::cudart
    ${TensorRT_nvinfer_LIBRARY}
)


# Compiler-specific options
if(MSVC)
    target_compile_options(oss20b_engine PRIVATE /W4)
    target_compile_options(complete_inference PRIVATE /W4)
    target_compile_options(engine_diagnostic PRIVATE /W4)
else()
    target_compile_options(oss20b_engine PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(complete_inference PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(engine_diagnostic PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS oss20b_engine
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${ENGINE_HEADERS}
    DESTINATION include
)
