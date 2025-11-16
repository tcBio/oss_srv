/**
 * @file engine_core.hpp
 * @brief Core TensorRT inference engine orchestration and management
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * This file implements the main engine orchestration layer that coordinates
 * TensorRT execution, memory management, tokenization, and request processing
 * for high-performance transformer model inference.
 * 
 * Key Features:
 * - TensorRT engine lifecycle management with dynamic shapes
 * - Asynchronous request processing with thread-safe queues
 * - Optimized memory allocation and GPU buffer management
 * - CPU-based token sampling for inference output processing
 * - CUDA stream coordination for overlapped operations
 * 
 * Performance Targets:
 * - Throughput: 100+ tokens/second sustained
 * - Latency: <50ms first token, <10ms subsequent tokens
 * - Memory: Efficient GPU buffer pooling with minimal fragmentation
 * - Batch Size: Support for 1-8 concurrent requests
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
 * @warning This implementation is optimized for Blackwell architecture
 * 
 * Repository: https://github.com/tcBio/oss_srv
 */

#pragma once
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <functional>
#include <random>
#include <cuda_runtime.h>

#include "tensorrt_engine.hpp"
#include "../cpp/memory_manager.hpp"
#include "../cpp/tokenizer.hpp"
#include "../cpp/kv_cache.hpp"
#include "../cpp/cuda_graphs.hpp"

struct EngineConfig {
    std::string model_path;
    int max_batch_size = 1;
    int max_sequence_length = 2048;
    std::string precision = "fp16";
    bool use_cuda_graph = true;
    int gpu_device_id = 0;
};

// Streaming callback function type
// Called for each generated token during streaming inference
// Parameters: token_id, token_text, is_final
using StreamCallback = std::function<void(int32_t, const std::string&, bool)>;

struct InferenceRequest {
    std::string prompt;
    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 1.0f;
    std::vector<std::string> stop_sequences;
    bool stream = false;
    std::string request_id;

    // Optional callback for streaming responses
    StreamCallback stream_callback = nullptr;
};

struct InferenceResult {
    bool success = false;
    std::string generated_text;
    std::vector<int32_t> generated_tokens;
    float inference_time_ms = 0.0f;
    float total_time_ms = 0.0f;
    std::string error_message;
};

class EngineCore {
public:
    EngineCore();
    ~EngineCore();
    
    bool initialize(const EngineConfig& config);
    void shutdown();
    
    // Core inference functionality
    InferenceResult executeInference(const InferenceRequest& request);
    
    // Component access
    TensorRTEngine* getTensorRTEngine() { return tensorrt_engine_.get(); }
    MemoryManager* getMemoryManager() { return memory_manager_.get(); }
    Tokenizer* getTokenizer() { return tokenizer_.get(); }
    
    // Configuration
    const EngineConfig& getConfig() const { return config_; }
    
private:
    // Core components
    std::unique_ptr<TensorRTEngine> tensorrt_engine_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<CUDAGraphManager> cuda_graphs_;
    
    // Configuration
    EngineConfig config_;
    bool initialized_ = false;
    
    // CUDA streams
    cudaStream_t compute_stream_;
    cudaStream_t memory_stream_;
    bool streams_initialized_ = false;
    
    // Thread pool
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex task_mutex_;
    std::condition_variable task_condition_;
    std::atomic<bool> stop_workers_;
    
    // Random number generation
    std::random_device rd_;
    std::mt19937 gen_;
    
    // Helper methods
    bool initializeCUDA();
    bool initializeComponents();
    void initializeThreadPool();
    void cleanupStreams();
    void cleanupThreadPool();
};

// CPU-based token sampling for TensorRT-only inference
int32_t sampleTokenCPU(const std::vector<float>& logits, float temperature = 1.0f, float top_p = 1.0f);
