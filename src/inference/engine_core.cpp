/**
 * @file engine_core.cpp
 * @brief Core TensorRT inference engine implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of engine orchestration with TensorRT lifecycle management and async processing.
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

#include "engine_core.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <cublas_v2.h>

EngineCore::EngineCore() : gen_(rd_()), stop_workers_(false) {}

EngineCore::~EngineCore() {
    shutdown();
}

bool EngineCore::initialize(const EngineConfig& config) {
    config_ = config;
    
    try {
        std::cout << "Initializing OSS-20B TensorRT Inference Engine..." << std::endl;
        std::cout << "Model path: " << config.model_path << std::endl;
        std::cout << "Max batch size: " << config.max_batch_size << std::endl;
        std::cout << "Max sequence length: " << config.max_sequence_length << std::endl;
        std::cout << "Precision: " << config.precision << std::endl;
        
        if (!initializeCUDA()) {
            return false;
        }
        
        if (!initializeComponents()) {
            return false;
        }
        
        initializeThreadPool();
        
        initialized_ = true;
        std::cout << "OSS-20B TensorRT Inference Engine initialized successfully!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

bool EngineCore::initializeCUDA() {
    // Initialize CUDA context and enable TF32 for Blackwell Blackwell
    cudaError_t cuda_status = cudaSetDevice(config_.gpu_device_id);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }
    
    // Enable TF32 for 2x speedup on Ampere/Blackwell architectures
    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        cublasDestroy(cublas_handle);
        std::cout << "TF32 tensor operations enabled for Blackwell Blackwell" << std::endl;
    } else {
        std::cout << "Warning: Could not enable TF32 operations" << std::endl;
    }
    
    std::cout << "CUDA device " << config_.gpu_device_id << " initialized with TF32 enabled" << std::endl;
    
    // Create CUDA streams for async operations
    cudaError_t stream_status;
    stream_status = cudaStreamCreate(&compute_stream_);
    if (stream_status != cudaSuccess) {
        std::cerr << "Failed to create compute stream: " << cudaGetErrorString(stream_status) << std::endl;
        return false;
    }
    
    stream_status = cudaStreamCreate(&memory_stream_);
    if (stream_status != cudaSuccess) {
        std::cerr << "Failed to create memory stream: " << cudaGetErrorString(stream_status) << std::endl;
        cudaStreamDestroy(compute_stream_);
        return false;
    }
    
    streams_initialized_ = true;
    std::cout << "CUDA streams initialized for async operations" << std::endl;
    
    return true;
}

bool EngineCore::initializeComponents() {
    // Initialize TensorRT engine
    std::cout << "Creating TensorRT engine..." << std::endl;
    
    // Check if this is a TensorRT engine file
    if (config_.model_path.find(".engine") != std::string::npos) {
        tensorrt_engine_ = std::make_unique<TensorRTEngine>();
        if (!tensorrt_engine_->loadEngine(config_.model_path)) {
            std::cerr << "Failed to load TensorRT engine from: " << config_.model_path << std::endl;
            return false;
        }
        std::cout << "TensorRT engine loaded successfully" << std::endl;
    } else {
        std::cerr << "Invalid model path. Expected .engine file, got: " << config_.model_path << std::endl;
        return false;
    }
    
    // Initialize memory manager
    std::cout << "Initializing memory manager..." << std::endl;
    memory_manager_ = std::make_unique<MemoryManager>();
    if (!memory_manager_->initialize(config_.max_batch_size, config_.max_sequence_length)) {
        std::cerr << "Failed to initialize memory manager" << std::endl;
        return false;
    }
    std::cout << "Memory manager initialized" << std::endl;
    
    // Initialize tokenizer
    std::cout << "Initializing tokenizer..." << std::endl;
    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->loadFromDirectory("tokenizer")) {
        std::cerr << "Failed to load tokenizer from directory" << std::endl;
        return false;
    }
    std::cout << "Tokenizer initialized" << std::endl;
    
    // Initialize KV cache
    std::cout << "Initializing KV cache..." << std::endl;
    kv_cache_ = std::make_unique<KVCache>();
    if (!kv_cache_->initialize(config_.max_batch_size, config_.max_sequence_length, 24, 64, 45)) {
        std::cerr << "Failed to initialize KV cache" << std::endl;
        return false;
    }
    std::cout << "KV cache initialized" << std::endl;
    
    // Initialize CUDA graphs for inference pipeline optimization
    if (config_.use_cuda_graph) {
        cuda_graphs_ = std::make_unique<CUDAGraphManager>();
        size_t vocab_size = 200000;  // OSS-20B vocabulary size
        if (!cuda_graphs_->initialize(config_.max_batch_size, config_.max_sequence_length, vocab_size)) {
            std::cerr << "Warning: Failed to initialize CUDA graphs, continuing without graph optimization" << std::endl;
            cuda_graphs_.reset();
        } else {
            std::cout << "CUDA graphs initialized for inference pipeline optimization" << std::endl;
        }
    } else {
        std::cout << "CUDA graphs disabled in configuration" << std::endl;
    }
    
    return true;
}

void EngineCore::initializeThreadPool() {
    // Initialize thread pool for CPU parallelism
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Default fallback
    
    stop_workers_ = false;
    worker_threads_.reserve(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back([this]() {
            while (!stop_workers_) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(task_mutex_);
                    task_condition_.wait(lock, [this]() { 
                        return stop_workers_ || !task_queue_.empty(); 
                    });
                    
                    if (stop_workers_ && task_queue_.empty()) {
                        return;
                    }
                    
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                task();
            }
        });
    }
    
    std::cout << "Thread pool initialized with " << num_threads << " worker threads" << std::endl;
}

InferenceResult EngineCore::executeInference(const InferenceRequest& request) {
    InferenceResult result;
    
    if (!initialized_) {
        result.error_message = "Engine not initialized";
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Tokenize input
        auto input_tokens = tokenizer_->tokenize(request.prompt);
        if (input_tokens.empty()) {
            result.error_message = "Failed to tokenize input";
            return result;
        }
        
        std::vector<int32_t> current_tokens = input_tokens;
        std::vector<int32_t> generated_tokens;
        
        // Generation loop with streaming support
        for (int i = 0; i < request.max_tokens; i++) {
            std::vector<float> logits;
            bool inference_success = tensorrt_engine_->executeInference(current_tokens, logits);

            if (!inference_success || logits.empty()) {
                result.error_message = "TensorRT inference failed";
                return result;
            }

            // Sample next token using CPU sampling (TensorRT-only)
            int32_t next_token = sampleTokenCPU(logits, request.temperature, request.top_p);

            generated_tokens.push_back(next_token);
            current_tokens.push_back(next_token);

            // Streaming callback: send token as it's generated
            if (request.stream && request.stream_callback) {
                std::string token_text = tokenizer_->detokenize({next_token});
                bool is_final = (next_token == tokenizer_->getEOSTokenId()) || (i == request.max_tokens - 1);
                request.stream_callback(next_token, token_text, is_final);
            }

            // Check for EOS token
            if (next_token == tokenizer_->getEOSTokenId()) {
                break;
            }
        }
        
        // Detokenize generated tokens
        result.generated_text = tokenizer_->detokenize(generated_tokens);
        result.generated_tokens = generated_tokens;
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Exception during inference: ") + e.what();
        return result;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.total_time_ms = static_cast<float>(duration.count());
    
    return result;
}

void EngineCore::shutdown() {
    if (!initialized_) {
        return;
    }
    
    cleanupThreadPool();
    cleanupStreams();
    
    // Reset components
    cuda_graphs_.reset();
    kv_cache_.reset();
    tokenizer_.reset();
    memory_manager_.reset();
    tensorrt_engine_.reset();
    
    initialized_ = false;
    std::cout << "OSS-20B TensorRT Inference Engine shutdown complete" << std::endl;
}

void EngineCore::cleanupThreadPool() {
    // Stop worker threads
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        stop_workers_ = true;
    }
    task_condition_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void EngineCore::cleanupStreams() {
    if (streams_initialized_) {
        cudaStreamDestroy(compute_stream_);
        cudaStreamDestroy(memory_stream_);
        streams_initialized_ = false;
    }
}

// CPU-based token sampling for TensorRT-only inference
int32_t sampleTokenCPU(const std::vector<float>& logits, float temperature, float top_p) {
    if (logits.empty()) return 0;
    
    std::vector<float> scaled_logits = logits;
    if (temperature != 1.0f && temperature > 0.0f) {
        for (float& logit : scaled_logits) {
            logit /= temperature;
        }
    }
    
    // Softmax with numerical stability
    float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
    for (float& logit : scaled_logits) {
        logit = std::exp(logit - max_logit);
    }
    
    float sum = 0.0f;
    for (float logit : scaled_logits) {
        sum += logit;
    }
    
    for (float& logit : scaled_logits) {
        logit /= sum;
    }
    
    // Simple random sampling
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float random_val = dis(gen);
    float cumulative = 0.0f;
    
    for (size_t i = 0; i < scaled_logits.size(); i++) {
        cumulative += scaled_logits[i];
        if (random_val <= cumulative) {
            return static_cast<int32_t>(i);
        }
    }
    
    return static_cast<int32_t>(scaled_logits.size() - 1);
}
