/**
 * @file dynamic_batcher.hpp
 * @brief Dynamic batching system for request optimization
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Dynamic request batching to optimize GPU utilization and improve inference throughput.
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

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <future>
#include "../inference/engine_core.hpp"

// Dynamic batching for continuous request processing
struct BatchingRequest {
    InferenceRequest request;
    std::promise<InferenceResult> promise;
    std::chrono::steady_clock::time_point arrival_time;
    size_t current_token_count = 0;
    std::vector<int32_t> generated_tokens;
    bool is_completed = false;
    
    BatchingRequest(InferenceRequest req) 
        : request(std::move(req)), arrival_time(std::chrono::steady_clock::now()) {}
};

class DynamicBatcher {
public:
    struct BatcherConfig {
        size_t max_batch_size = 16;           // Blackwell can handle larger batches
        size_t max_sequence_length = 4096;    // OSS-20B context length
        float batch_timeout_ms = 5.0f;        // Wait time to fill batch
        size_t min_batch_size = 1;            // Process immediately if >= min
        bool enable_preemption = true;        // Allow preempting long sequences
        float target_latency_ms = 100.0f;     // Target per-token latency
    };

private:
    BatcherConfig config_;
    std::queue<std::shared_ptr<BatchingRequest>> pending_requests_;
    std::vector<std::shared_ptr<BatchingRequest>> active_requests_;
    
    std::mutex request_mutex_;
    std::condition_variable batch_condition_;
    std::thread batcher_thread_;
    std::atomic<bool> shutdown_requested_;
    
    // TensorRT inference engine core
    std::unique_ptr<EngineCore> engine_;
    
public:
    DynamicBatcher(BatcherConfig config = BatcherConfig{});
    ~DynamicBatcher();
    
    bool initialize(const EngineConfig& engine_config);
    void shutdown();
    
    // Submit request for dynamic batching
    std::future<InferenceResult> submitRequest(const InferenceRequest& request);
    
    // Get current batch statistics
    size_t getPendingCount() const;
    size_t getActiveCount() const;
    float getAverageBatchSize() const;
    float getThroughput() const;

private:
    void batchingLoop();
    void processBatch();
    std::vector<std::shared_ptr<BatchingRequest>> formBatch();
    void updateActiveRequests(const std::vector<InferenceResult>& results);
    bool shouldPreempt(const std::shared_ptr<BatchingRequest>& req) const;
    void completeRequest(std::shared_ptr<BatchingRequest> req, const InferenceResult& result);
    
    // Performance tracking
    std::atomic<size_t> total_requests_processed_;
    std::atomic<size_t> total_tokens_generated_;
    std::chrono::steady_clock::time_point start_time_;
};