/**
 * @file request_processor.hpp
 * @brief Asynchronous request processing and batching system
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Thread-safe request queue management with batching support for concurrent inference requests.
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
#include "engine_core.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>

struct BatchedRequest {
    InferenceRequest request;
    std::promise<InferenceResult> promise;
    std::chrono::steady_clock::time_point timestamp;
};

class RequestProcessor {
public:
    RequestProcessor(EngineCore* engine_core);
    ~RequestProcessor();
    
    bool initialize();
    void shutdown();
    
    // Process single request
    std::future<InferenceResult> processRequest(const InferenceRequest& request);
    
    // Batch processing
    InferenceResult executeBatchedInference(const std::vector<InferenceRequest>& requests);
    
    // Queue management
    size_t getQueueSize() const;
    void clearQueue();
    
private:
    EngineCore* engine_core_;
    
    // Request queue
    std::queue<BatchedRequest> request_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    
    // Processing thread
    std::thread processing_thread_;
    std::atomic<bool> stop_processing_;
    
    // Configuration
    size_t max_batch_size_ = 4;
    std::chrono::milliseconds batch_timeout_{50};
    
    // Processing methods
    void processingLoop();
    std::vector<BatchedRequest> collectBatch();
    void processBatch(std::vector<BatchedRequest>& batch);
};
