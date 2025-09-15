/**
 * @file tensorrt_runner.hpp
 * @brief Legacy TensorRT runner interface
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Legacy TensorRT execution interface - superseded by tensorrt_engine.hpp in refactored architecture.
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
 * @warning Optimized for Blackwell architecture
 * 
 * Repository: https://github.com/tcBio/oss_srv
 */

#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>

class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // Only print warnings and errors to reduce noise
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TensorRTRunner {
private:
    std::unique_ptr<TensorRTLogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t cuda_stream_;
    
    // Buffer management
    void* input_buffer_device_ = nullptr;
    void* output_buffer_device_ = nullptr;
    
    // Model metadata
    int max_sequence_length_ = 0;
    int actual_seq_length_ = 0;  // Actual allocated sequence length
    int vocab_size_ = 0;
    int batch_size_ = 1;
    
    // Binding indices
    int input_binding_index_ = -1;
    int output_binding_index_ = -1;
    
public:
    TensorRTRunner();
    ~TensorRTRunner();
    
    bool loadEngine(const std::string& engine_path);
    bool executeInference(const std::vector<int32_t>& input_tokens, 
                         std::vector<float>& output_logits);
    bool executeBatchInference(const std::vector<std::vector<int32_t>>& batch_input_tokens,
                              std::vector<std::vector<float>>& batch_output_logits);
    bool extractOutputLogits(size_t batch_size, std::vector<std::vector<float>>& batch_output_logits);
    
    // Getters
    int getMaxSequenceLength() const { return max_sequence_length_; }
    int getVocabSize() const { return vocab_size_; }
    int getBatchSize() const { return batch_size_; }
    
private:
    bool initializeBuffers();
    void cleanup();
    bool extractEngineMetadata();
    bool copyInputToDevice(const std::vector<int32_t>& input_tokens);
    bool copyOutputFromDevice(std::vector<float>& output_logits, size_t actual_seq_len);
    bool copyBatchInputToDevice(const std::vector<std::vector<int32_t>>& batch_input_tokens);
    bool copyBatchOutputFromDevice(size_t batch_size, size_t seq_len, 
                                  std::vector<std::vector<float>>& batch_output_logits);
    // Removed reallocateBuffersForSequenceLength - not needed
};