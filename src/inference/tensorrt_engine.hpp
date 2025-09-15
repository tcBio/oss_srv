/**
 * @file tensorrt_engine.hpp
 * @brief TensorRT engine wrapper for high-performance GPU inference
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * This file implements a TensorRT engine wrapper that handles engine loading,
 * execution context management, CUDA buffer allocation, and inference execution
 * with dynamic shape support for transformer models.
 * 
 * Key Features:
 * - TensorRT engine deserialization and initialization
 * - Dynamic shape binding for variable sequence lengths (1-256 tokens)
 * - CUDA buffer management with device memory allocation
 * - Asynchronous inference execution with CUDA streams
 * - FP16 precision optimization for Blackwell Blackwell architecture
 * 
 * Performance Optimizations:
 * - Pre-allocated GPU buffers for minimal allocation overhead
 * - CUDA stream synchronization for overlapped operations
 * - Optimized tensor binding for dynamic input shapes
 * 
 * @author Brian Worthington
 * @date 2025
 * @version 1.0
 * 
 * @copyright MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * @note Requires NVIDIA Blackwell, CUDA 12.8+, TensorRT 10.8+
 * @warning Optimized for Blackwell architecture - may need adjustments for other GPUs
 * 
 * Repository: https://github.com/tcBio/oss_srv
 */

#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();
    
    // Engine management
    bool loadEngine(const std::string& engine_path);
    bool buildEngine(const std::string& onnx_path, const std::string& engine_path);
    void cleanup();
    
    // Inference execution
    bool executeInference(const std::vector<int32_t>& input_tokens, std::vector<float>& output_logits);
    
    // Buffer management
    bool allocateBuffers();
    bool copyInputToDevice(const std::vector<int32_t>& input_tokens);
    bool copyOutputFromDevice(std::vector<float>& output_logits, size_t sequence_length);
    
    // Engine properties
    bool isLoaded() const { return engine_ != nullptr && context_ != nullptr; }
    size_t getMaxBatchSize() const;
    size_t getMaxSequenceLength() const;
    
private:
    // TensorRT objects
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // CUDA resources
    cudaStream_t cuda_stream_;
    void* input_buffer_device_ = nullptr;
    void* output_buffer_device_ = nullptr;
    
    // Engine metadata
    int input_binding_index_ = -1;
    int output_binding_index_ = -1;
    size_t input_buffer_size_ = 0;
    size_t output_buffer_size_ = 0;
    int actual_seq_length_ = 2048;
    
    // Helper methods
    bool setupBindings();
    void logEngineInfo();
};
