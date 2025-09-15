/**
 * @file tensorrt_engine.cpp
 * @brief TensorRT engine wrapper implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of TensorRT engine operations with dynamic shape support and GPU optimization.
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

#include "tensorrt_engine.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>

TensorRTEngine::TensorRTEngine() {
    cudaStreamCreate(&cuda_stream_);
}

TensorRTEngine::~TensorRTEngine() {
    cleanup();
    cudaStreamDestroy(cuda_stream_);
}

bool TensorRTEngine::loadEngine(const std::string& engine_path) {
    std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;
    
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    // Create a simple logger for TensorRT
    class SimpleLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    static SimpleLogger logger;
    
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return false;
    }
    
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    if (!setupBindings()) {
        return false;
    }
    
    if (!allocateBuffers()) {
        return false;
    }
    
    logEngineInfo();
    std::cout << "TensorRT engine loaded successfully" << std::endl;
    return true;
}

bool TensorRTEngine::setupBindings() {
    int num_bindings = engine_->getNbIOTensors();
    std::cout << "Engine has " << num_bindings << " bindings" << std::endl;
    
    for (int i = 0; i < num_bindings; i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_binding_index_ = i;
            std::cout << "Input binding: " << name << " (index " << i << ")" << std::endl;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_binding_index_ = i;
            std::cout << "Output binding: " << name << " (index " << i << ")" << std::endl;
        }
    }
    
    if (input_binding_index_ == -1 || output_binding_index_ == -1) {
        std::cerr << "Failed to find input/output bindings" << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTEngine::allocateBuffers() {
    // Calculate buffer sizes based on maximum dimensions
    const char* input_name = engine_->getIOTensorName(input_binding_index_);
    const char* output_name = engine_->getIOTensorName(output_binding_index_);
    
    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name);
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_name);
    
    // Calculate sizes (assuming max sequence length)
    input_buffer_size_ = actual_seq_length_ * sizeof(int32_t);
    output_buffer_size_ = actual_seq_length_ * 200000 * sizeof(float); // vocab_size = 200000
    
    // Allocate device memory
    cudaError_t status;
    status = cudaMalloc(&input_buffer_device_, input_buffer_size_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    status = cudaMalloc(&output_buffer_device_, output_buffer_size_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer: " << cudaGetErrorString(status) << std::endl;
        cudaFree(input_buffer_device_);
        return false;
    }
    
    std::cout << "Allocated buffers - Input: " << input_buffer_size_ << " bytes, Output: " << output_buffer_size_ << " bytes" << std::endl;
    return true;
}

bool TensorRTEngine::executeInference(const std::vector<int32_t>& input_tokens, std::vector<float>& output_logits) {
    if (!context_ || !engine_) {
        std::cerr << "TensorRT not initialized" << std::endl;
        return false;
    }
    
    // Set optimization profile first (required for dynamic shapes in TensorRT 10.x)
    if (!context_->setOptimizationProfileAsync(0, cuda_stream_)) {
        std::cerr << "Failed to set optimization profile" << std::endl;
        return false;
    }
    
    // Synchronize to ensure profile is set before setting shapes
    cudaError_t sync_status = cudaStreamSynchronize(cuda_stream_);
    if (sync_status != cudaSuccess) {
        std::cerr << "Failed to synchronize after setting optimization profile: " << cudaGetErrorString(sync_status) << std::endl;
        return false;
    }
    
    // Validate input length against buffer capacity
    if (input_tokens.size() > static_cast<size_t>(actual_seq_length_)) {
        std::cerr << "Input tokens (" << input_tokens.size() << ") exceed buffer capacity (" << actual_seq_length_ << ")" << std::endl;
        return false;
    }
    
    // Set input tensor shape for dynamic dimensions
    const char* input_tensor_name = engine_->getIOTensorName(input_binding_index_);
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 2;
    input_dims.d[0] = 1;  // batch size
    input_dims.d[1] = input_tokens.size();  // Use actual input length
    
    if (!context_->setInputShape(input_tensor_name, input_dims)) {
        std::cerr << "Failed to set input tensor shape" << std::endl;
        return false;
    }
    
    // Validate that all input dimensions are specified
    if (!context_->allInputDimensionsSpecified()) {
        std::cerr << "Not all input dimensions are specified" << std::endl;
        return false;
    }
    
    // Set tensor addresses for TensorRT 10.x API
    if (!context_->setTensorAddress(input_tensor_name, input_buffer_device_)) {
        std::cerr << "Failed to set input tensor address" << std::endl;
        return false;
    }
    
    const char* output_tensor_name = engine_->getIOTensorName(output_binding_index_);
    if (!context_->setTensorAddress(output_tensor_name, output_buffer_device_)) {
        std::cerr << "Failed to set output tensor address" << std::endl;
        return false;
    }
    
    // Copy input to device
    if (!copyInputToDevice(input_tokens)) {
        return false;
    }
    
    // Execute inference
    bool success = context_->enqueueV3(cuda_stream_);
    if (!success) {
        std::cerr << "TensorRT inference execution failed" << std::endl;
        return false;
    }
    
    // Synchronize CUDA stream
    cudaError_t sync_status2 = cudaStreamSynchronize(cuda_stream_);
    if (sync_status2 != cudaSuccess) {
        std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(sync_status2) << std::endl;
        return false;
    }
    
    // Copy output back to host
    if (!copyOutputFromDevice(output_logits, input_tokens.size())) {
        return false;
    }
    
    return true;
}

bool TensorRTEngine::copyInputToDevice(const std::vector<int32_t>& input_tokens) {
    size_t copy_size = input_tokens.size() * sizeof(int32_t);
    
    cudaError_t status = cudaMemcpyAsync(
        input_buffer_device_, 
        input_tokens.data(), 
        copy_size, 
        cudaMemcpyHostToDevice, 
        cuda_stream_
    );
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy input to device: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTEngine::copyOutputFromDevice(std::vector<float>& output_logits, size_t sequence_length) {
    size_t vocab_size = 200000;  // OSS-20B vocabulary size
    size_t output_size = vocab_size;  // Last token logits only
    
    output_logits.resize(output_size);
    
    // Copy only the last token's logits
    size_t offset = (sequence_length - 1) * vocab_size * sizeof(float);
    size_t copy_size = vocab_size * sizeof(float);
    
    cudaError_t status = cudaMemcpyAsync(
        output_logits.data(),
        static_cast<char*>(output_buffer_device_) + offset,
        copy_size,
        cudaMemcpyDeviceToHost,
        cuda_stream_
    );
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy output from device: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    return true;
}

void TensorRTEngine::cleanup() {
    if (input_buffer_device_) {
        cudaFree(input_buffer_device_);
        input_buffer_device_ = nullptr;
    }
    
    if (output_buffer_device_) {
        cudaFree(output_buffer_device_);
        output_buffer_device_ = nullptr;
    }
    
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

size_t TensorRTEngine::getMaxBatchSize() const {
    return 1;  // Currently supporting batch size 1
}

size_t TensorRTEngine::getMaxSequenceLength() const {
    return actual_seq_length_;
}

void TensorRTEngine::logEngineInfo() {
    std::cout << "=== TensorRT Engine Info ===" << std::endl;
    std::cout << "Max batch size: " << getMaxBatchSize() << std::endl;
    std::cout << "Max sequence length: " << getMaxSequenceLength() << std::endl;
    std::cout << "Input buffer size: " << input_buffer_size_ << " bytes" << std::endl;
    std::cout << "Output buffer size: " << output_buffer_size_ << " bytes" << std::endl;
    std::cout << "============================" << std::endl;
}
