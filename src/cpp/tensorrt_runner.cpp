/**
 * @file tensorrt_runner.cpp
 * @brief Legacy TensorRT runner implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Legacy TensorRT execution implementation - superseded by tensorrt_engine.cpp in refactored architecture.
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

#include "tensorrt_runner.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

TensorRTRunner::TensorRTRunner() : cuda_stream_(nullptr) {
    // Initialize logger
    logger_ = std::make_unique<TensorRTLogger>();
    
    // Create CUDA stream
    cudaStreamCreate(&cuda_stream_);
}

TensorRTRunner::~TensorRTRunner() {
    cleanup();
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
    }
}

bool TensorRTRunner::loadEngine(const std::string& engine_path) {
    std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;
    
    // Check if file exists
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    
    // Read engine file
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();
    
    std::cout << "Engine file size: " << engine_size << " bytes" << std::endl;
    
    try {
        // Create TensorRT runtime
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_));
        if (!runtime_) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        // Deserialize engine
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), engine_size));
        if (!engine_) {
            std::cerr << "Failed to deserialize CUDA engine" << std::endl;
            return false;
        }
        
        // Create execution context
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        std::cout << "TensorRT engine loaded successfully" << std::endl;
        
        // Extract metadata and initialize buffers
        if (!extractEngineMetadata()) {
            std::cerr << "Failed to extract engine metadata" << std::endl;
            return false;
        }
        
        if (!initializeBuffers()) {
            std::cerr << "Failed to initialize GPU buffers" << std::endl;
            return false;
        }
        
        std::cout << "Engine metadata:" << std::endl;
        std::cout << "  Max sequence length: " << max_sequence_length_ << std::endl;
        std::cout << "  Vocabulary size: " << vocab_size_ << std::endl;
        std::cout << "  Batch size: " << batch_size_ << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during TensorRT initialization: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRTRunner::extractEngineMetadata() {
    if (!engine_) {
        return false;
    }
    
    // Get number of IO tensors (TensorRT 10.x API)
    int num_io_tensors = engine_->getNbIOTensors();
    std::cout << "Engine has " << num_io_tensors << " IO tensors" << std::endl;
    
    for (int i = 0; i < num_io_tensors; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(tensor_name);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
        bool is_input = (io_mode == nvinfer1::TensorIOMode::kINPUT);
        
        std::cout << "Tensor " << i << ": " << tensor_name 
                  << " (input: " << is_input << ")" << std::endl;
        std::cout << "  Dimensions: [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        if (is_input) {
            input_binding_index_ = i;
            if (dims.nbDims >= 2) {
                batch_size_ = static_cast<int>(dims.d[0]);
                // Handle dynamic shape (-1) by setting a reasonable default
                if (dims.d[1] == -1) {
                    max_sequence_length_ = 256;  // Default sequence length for testing
                    std::cout << "  Dynamic input sequence length detected, using default: " << max_sequence_length_ << std::endl;
                } else {
                    max_sequence_length_ = static_cast<int>(dims.d[1]);
                }
            }
        } else {
            output_binding_index_ = i;
            if (dims.nbDims >= 3) {
                vocab_size_ = static_cast<int>(dims.d[2]); // Usually [batch, seq_len, vocab_size]
            }
        }
    }
    
    if (input_binding_index_ == -1 || output_binding_index_ == -1) {
        std::cerr << "Could not find input/output tensors" << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTRunner::initializeBuffers() {
    if (!engine_) {
        return false;
    }
    
    // Calculate buffer sizes - fix size mismatch issue
    // Use consistent sequence length for allocation and usage
    // Ensure we have a valid sequence length (fallback to 256 if max_sequence_length_ is 0)
    actual_seq_length_ = (max_sequence_length_ > 0) ? std::min(max_sequence_length_, 256) : 256;
    size_t input_size = batch_size_ * actual_seq_length_ * sizeof(int32_t);
    size_t output_size = batch_size_ * actual_seq_length_ * vocab_size_ * sizeof(float);
    
    std::cout << "Debug: max_sequence_length_=" << max_sequence_length_ << ", actual_seq_length_=" << actual_seq_length_ << std::endl;
    std::cout << "Attempting to allocate GPU memory:" << std::endl;
    std::cout << "  Input size: " << input_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Output size: " << output_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Total: " << (input_size + output_size) / (1024*1024) << " MB" << std::endl;
    
    // Allocate GPU memory
    cudaError_t status;
    
    // Add comprehensive CUDA error checking and validation
    std::cout << "Allocating input buffer: " << input_size << " bytes" << std::endl;
    status = cudaMalloc(&input_buffer_device_, input_size);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer: " << cudaGetErrorString(status) << std::endl;
        std::cerr << "Requested size: " << input_size << " bytes (" << input_size / (1024*1024) << " MB)" << std::endl;
        return false;
    }
    
    std::cout << "Allocating output buffer: " << output_size << " bytes" << std::endl;
    status = cudaMalloc(&output_buffer_device_, output_size);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer: " << cudaGetErrorString(status) << std::endl;
        std::cerr << "Requested size: " << output_size << " bytes (" << output_size / (1024*1024) << " MB)" << std::endl;
        cudaFree(input_buffer_device_);
        input_buffer_device_ = nullptr;
        return false;
    }
    
    // Verify allocations succeeded
    if (!input_buffer_device_ || !output_buffer_device_) {
        std::cerr << "Buffer allocation returned null pointers" << std::endl;
        cleanup();
        return false;
    }
    
    // Tensor addresses will be set per-inference in executeInference method
    
    std::cout << "GPU buffers allocated successfully" << std::endl;
    std::cout << "  Input buffer: " << input_size << " bytes" << std::endl;
    std::cout << "  Output buffer: " << output_size << " bytes" << std::endl;
    
    return true;
}

// Removed reallocateBuffersForSequenceLength - not needed with proper dynamic shape handling

bool TensorRTRunner::executeInference(const std::vector<int32_t>& input_tokens, 
                                     std::vector<float>& output_logits) {
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
    
    // Set input tensor shape for dynamic dimensions (engine supports this)
    const char* input_tensor_name = engine_->getIOTensorName(input_binding_index_);
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 2;
    input_dims.d[0] = 1;  // batch size
    input_dims.d[1] = input_tokens.size();  // Use actual input length
    
    std::cout << "Setting input shape: [" << input_dims.d[0] << ", " << input_dims.d[1] << "] for " << input_tokens.size() << " tokens" << std::endl;
    
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
    
    // Execute inference (TensorRT 10.x API) with validation
    std::cout << "Executing TensorRT inference..." << std::endl;
    
    // Validate context and stream before execution
    if (!context_) {
        std::cerr << "Execution context is null" << std::endl;
        return false;
    }
    
    if (!cuda_stream_) {
        std::cerr << "CUDA stream is null" << std::endl;
        return false;
    }
    
    bool success = context_->enqueueV3(cuda_stream_);
    if (!success) {
        std::cerr << "TensorRT inference execution failed" << std::endl;
        std::cerr << "Check tensor shapes and buffer allocations" << std::endl;
        return false;
    }
    
    // Synchronize CUDA stream with error checking
    cudaError_t sync_status2 = cudaStreamSynchronize(cuda_stream_);
    if (sync_status2 != cudaSuccess) {
        std::cerr << "CUDA stream synchronization failed: " << cudaGetErrorString(sync_status2) << std::endl;
        return false;
    }
    
    std::cout << "TensorRT inference completed successfully" << std::endl;
    
    // Copy output back to host (use actual input size)
    if (!copyOutputFromDevice(output_logits, input_tokens.size())) {
        return false;
    }
    
    return true;
}

bool TensorRTRunner::executeBatchInference(const std::vector<std::vector<int32_t>>& batch_input_tokens,
                                          std::vector<std::vector<float>>& batch_output_logits) {
    if (!context_ || !engine_) {
        std::cerr << "TensorRT not initialized" << std::endl;
        return false;
    }
    
    if (batch_input_tokens.empty()) {
        std::cerr << "Empty batch provided" << std::endl;
        return false;
    }
    
    size_t batch_size = batch_input_tokens.size();
    if (batch_size > static_cast<size_t>(batch_size_)) {
        std::cerr << "Batch size " << batch_size << " exceeds maximum " << batch_size_ << std::endl;
        return false;
    }
    
    // Set optimization profile first (required for dynamic shapes in TensorRT 10.x)
    if (!context_->setOptimizationProfileAsync(0, cuda_stream_)) {
        std::cerr << "Failed to set optimization profile" << std::endl;
        return false;
    }
    
    // Synchronize to ensure profile is set before setting shapes
    cudaError_t sync_status3 = cudaStreamSynchronize(cuda_stream_);
    if (sync_status3 != cudaSuccess) {
        std::cerr << "Failed to synchronize after setting optimization profile: " << cudaGetErrorString(sync_status3) << std::endl;
        return false;
    }
    
    // Set input tensor shape for dynamic dimensions (TensorRT 10.x API)
    const char* input_tensor_name = engine_->getIOTensorName(input_binding_index_);
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 2;
    input_dims.d[0] = batch_size;
    input_dims.d[1] = actual_seq_length_;  // Set actual sequence length
    
    std::cout << "Setting input shape: [" << input_dims.d[0] << ", " << input_dims.d[1] << "]" << std::endl;
    
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
    
    // Copy batch input to device
    if (!copyBatchInputToDevice(batch_input_tokens)) {
        return false;
    }
    
    // Execute inference (TensorRT 10.x API)
    bool success = context_->enqueueV3(cuda_stream_);
    if (!success) {
        std::cerr << "TensorRT batch inference execution failed" << std::endl;
        return false;
    }
    
    // Synchronize CUDA stream
    cudaStreamSynchronize(cuda_stream_);
    
    // Copy batch output back to host
    size_t seq_len = batch_input_tokens[0].size(); // Assume all sequences have same length
    if (!copyBatchOutputFromDevice(batch_size, seq_len, batch_output_logits)) {
        return false;
    }
    
    return true;
}

bool TensorRTRunner::copyInputToDevice(const std::vector<int32_t>& input_tokens) {
    if (!input_buffer_device_) {
        return false;
    }
    
    // Copy only the actual input tokens (no padding needed for dynamic shapes)
    size_t input_size = input_tokens.size() * sizeof(int32_t);
    
    // Add CUDA error checking
    cudaError_t status = cudaMemcpyAsync(
        input_buffer_device_, 
        input_tokens.data(), 
        input_size,
        cudaMemcpyHostToDevice,
        cuda_stream_
    );
    
    std::cout << "Copying " << input_tokens.size() << " tokens (" << input_size << " bytes) to device" << std::endl;
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy input to device: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTRunner::copyBatchInputToDevice(const std::vector<std::vector<int32_t>>& batch_input_tokens) {
    if (!input_buffer_device_) {
        return false;
    }
    
    size_t total_size = batch_size_ * max_sequence_length_;
    std::vector<int32_t> batch_input(total_size, 0); // Initialize with padding
    
    for (size_t batch_idx = 0; batch_idx < batch_input_tokens.size(); ++batch_idx) {
        const auto& tokens = batch_input_tokens[batch_idx];
        size_t copy_size = std::min(tokens.size(), static_cast<size_t>(max_sequence_length_));
        
        std::copy(tokens.begin(), tokens.begin() + copy_size,
                 batch_input.begin() + batch_idx * max_sequence_length_);
    }
    
    cudaError_t status = cudaMemcpyAsync(
        input_buffer_device_,
        batch_input.data(),
        total_size * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        cuda_stream_
    );
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy batch input to device: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTRunner::copyOutputFromDevice(std::vector<float>& output_logits, size_t actual_seq_len) {
    if (!output_buffer_device_) {
        std::cerr << "Output buffer not allocated" << std::endl;
        return false;
    }
    
    // Calculate output size using actual sequence length from input
    size_t output_size = batch_size_ * actual_seq_len * vocab_size_;
    output_logits.resize(output_size);
    
    std::cout << "Copying " << output_size << " floats (" << output_size * sizeof(float) << " bytes) from device" << std::endl;
    
    cudaError_t status = cudaMemcpyAsync(
        output_logits.data(),
        output_buffer_device_,
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        cuda_stream_
    );
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy output from device: " << cudaGetErrorString(status) << std::endl;
        std::cerr << "Output size: " << output_size << " floats (" << output_size * sizeof(float) << " bytes)" << std::endl;
        return false;
    }
    
    // Synchronize to ensure copy completes
    status = cudaStreamSynchronize(cuda_stream_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to synchronize output copy: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    std::cout << "Successfully copied " << output_logits.size() << " output values" << std::endl;
    return true;
}

bool TensorRTRunner::copyBatchOutputFromDevice(size_t batch_size, size_t seq_len, 
                                              std::vector<std::vector<float>>& batch_output_logits) {
    if (!output_buffer_device_) {
        return false;
    }
    
    size_t total_output_size = batch_size * seq_len * vocab_size_;
    std::vector<float> batch_output(total_output_size);
    
    cudaError_t status = cudaMemcpyAsync(
        batch_output.data(),
        output_buffer_device_,
        total_output_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        cuda_stream_
    );
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy batch output from device: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Reshape into batch format
    batch_output_logits.resize(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        size_t seq_vocab_size = seq_len * vocab_size_;
        batch_output_logits[i].resize(seq_vocab_size);
        std::copy(
            batch_output.begin() + i * seq_vocab_size,
            batch_output.begin() + (i + 1) * seq_vocab_size,
            batch_output_logits[i].begin()
        );
    }
    
    return true;
}

bool TensorRTRunner::extractOutputLogits(size_t batch_size, std::vector<std::vector<float>>& batch_output_logits) {
    // This method is called after executeBatchInference to extract specific logits
    // For now, we'll assume the logits are already extracted in executeBatchInference
    return true;
}

void TensorRTRunner::cleanup() {
    // Clean up GPU buffers
    if (input_buffer_device_) {
        cudaFree(input_buffer_device_);
        input_buffer_device_ = nullptr;
    }
    if (output_buffer_device_) {
        cudaFree(output_buffer_device_);
        output_buffer_device_ = nullptr;
    }
    
    // Clean up TensorRT objects (smart pointers handle this automatically)
    context_.reset();
    engine_.reset();
    runtime_.reset();
}