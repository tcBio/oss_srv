/**
 * @file dynamic_batcher.cpp
 * @brief Dynamic batcher implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of dynamic batching with request aggregation and batch size optimization.
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

#include "dynamic_batcher.hpp"
#include <algorithm>
#include <iostream>

DynamicBatcher::DynamicBatcher(BatcherConfig config) 
    : config_(config), shutdown_requested_(false), 
      total_requests_processed_(0), total_tokens_generated_(0),
      start_time_(std::chrono::steady_clock::now()) {
}

DynamicBatcher::~DynamicBatcher() {
    shutdown();
}

bool DynamicBatcher::initialize(const EngineConfig& engine_config) {
    // Initialize TensorRT inference engine core
    engine_ = std::make_unique<EngineCore>();
    if (!engine_->initialize(engine_config)) {
        std::cerr << "Failed to initialize engine core for dynamic batching" << std::endl;
        return false;
    }
    
    std::cout << "Dynamic Batcher initialized:" << std::endl;
    std::cout << "  Max batch size: " << config_.max_batch_size << std::endl;
    std::cout << "  Batch timeout: " << config_.batch_timeout_ms << "ms" << std::endl;
    std::cout << "  Min batch size: " << config_.min_batch_size << std::endl;
    
    // Start batching thread
    shutdown_requested_ = false;
    batcher_thread_ = std::thread(&DynamicBatcher::batchingLoop, this);
    
    return true;
}

void DynamicBatcher::shutdown() {
    shutdown_requested_ = true;
    batch_condition_.notify_all();
    
    if (batcher_thread_.joinable()) {
        batcher_thread_.join();
    }
    
    // Complete any remaining requests with error
    std::lock_guard<std::mutex> lock(request_mutex_);
    while (!pending_requests_.empty()) {
        auto req = pending_requests_.front();
        pending_requests_.pop();
        
        InferenceResult error_result;
        error_result.error_message = "Shutdown requested";
        req->promise.set_value(error_result);
    }
}

std::future<InferenceResult> DynamicBatcher::submitRequest(const InferenceRequest& request) {
    auto batching_request = std::make_shared<BatchingRequest>(request);
    auto future = batching_request->promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        pending_requests_.push(batching_request);
    }
    batch_condition_.notify_one();
    
    return future;
}

size_t DynamicBatcher::getPendingCount() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(request_mutex_));
    return pending_requests_.size();
}

size_t DynamicBatcher::getActiveCount() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(request_mutex_));
    return active_requests_.size();
}

float DynamicBatcher::getAverageBatchSize() const {
    if (total_requests_processed_ == 0) return 0.0f;
    return static_cast<float>(total_requests_processed_) / std::max(1.0f, static_cast<float>(total_tokens_generated_));
}

float DynamicBatcher::getThroughput() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    if (duration == 0) return 0.0f;
    return static_cast<float>(total_tokens_generated_) / duration;
}

void DynamicBatcher::batchingLoop() {
    while (!shutdown_requested_) {
        std::unique_lock<std::mutex> lock(request_mutex_);
        batch_condition_.wait_for(lock, std::chrono::milliseconds(static_cast<int>(config_.batch_timeout_ms)), 
                                 [this]() { return shutdown_requested_ || !pending_requests_.empty(); });
        
        if (shutdown_requested_) break;
        
        if (!pending_requests_.empty()) {
            processBatch();
        }
    }
}

void DynamicBatcher::processBatch() {
    // Simple implementation - process one request at a time
    if (pending_requests_.empty()) return;
    
    auto request = pending_requests_.front();
    pending_requests_.pop();
    
    // Execute inference
    auto result = engine_->executeInference(request->request);
    
    // Complete the request
    request->promise.set_value(result);
    total_requests_processed_++;
    if (!result.generated_tokens.empty()) {
        total_tokens_generated_ += result.generated_tokens.size();
    }
}

std::vector<std::shared_ptr<BatchingRequest>> DynamicBatcher::formBatch() {
    std::vector<std::shared_ptr<BatchingRequest>> batch;
    return batch; // Stub implementation
}

void DynamicBatcher::updateActiveRequests(const std::vector<InferenceResult>& results) {
    // Stub implementation
}

bool DynamicBatcher::shouldPreempt(const std::shared_ptr<BatchingRequest>& req) const {
    return false; // Stub implementation
}

void DynamicBatcher::completeRequest(std::shared_ptr<BatchingRequest> req, const InferenceResult& result) {
    req->promise.set_value(result);
}
