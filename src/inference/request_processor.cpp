/**
 * @file request_processor.cpp
 * @brief Request processor implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of async request processing with worker threads and batch optimization.
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

#include "request_processor.hpp"
#include <iostream>
#include <algorithm>

RequestProcessor::RequestProcessor(EngineCore* engine_core) 
    : engine_core_(engine_core), stop_processing_(false) {}

RequestProcessor::~RequestProcessor() {
    shutdown();
}

bool RequestProcessor::initialize() {
    if (!engine_core_) {
        std::cerr << "RequestProcessor: Engine core is null" << std::endl;
        return false;
    }
    
    stop_processing_ = false;
    processing_thread_ = std::thread(&RequestProcessor::processingLoop, this);
    
    std::cout << "RequestProcessor initialized" << std::endl;
    return true;
}

void RequestProcessor::shutdown() {
    stop_processing_ = true;
    queue_condition_.notify_all();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    clearQueue();
    std::cout << "RequestProcessor shutdown complete" << std::endl;
}

std::future<InferenceResult> RequestProcessor::processRequest(const InferenceRequest& request) {
    BatchedRequest batched_request;
    batched_request.request = request;
    batched_request.timestamp = std::chrono::steady_clock::now();
    
    auto future = batched_request.promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(batched_request));
    }
    queue_condition_.notify_one();
    
    return future;
}

InferenceResult RequestProcessor::executeBatchedInference(const std::vector<InferenceRequest>& requests) {
    InferenceResult result;
    
    if (requests.empty()) {
        result.error_message = "No requests to process";
        return result;
    }
    
    // For now, process first request only (batching not fully implemented)
    result = engine_core_->executeInference(requests[0]);
    
    return result;
}

size_t RequestProcessor::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return request_queue_.size();
}

void RequestProcessor::clearQueue() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Reject all pending requests
    while (!request_queue_.empty()) {
        auto& batched_request = request_queue_.front();
        InferenceResult result;
        result.error_message = "Request cancelled due to shutdown";
        batched_request.promise.set_value(result);
        request_queue_.pop();
    }
}

void RequestProcessor::processingLoop() {
    while (!stop_processing_) {
        auto batch = collectBatch();
        
        if (!batch.empty()) {
            processBatch(batch);
        }
    }
}

std::vector<BatchedRequest> RequestProcessor::collectBatch() {
    std::vector<BatchedRequest> batch;
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for requests or timeout
    queue_condition_.wait_for(lock, batch_timeout_, [this]() {
        return stop_processing_ || !request_queue_.empty();
    });
    
    if (stop_processing_) {
        return batch;
    }
    
    // Collect requests for batch
    auto deadline = std::chrono::steady_clock::now() + batch_timeout_;
    
    while (!request_queue_.empty() && 
           batch.size() < max_batch_size_ && 
           std::chrono::steady_clock::now() < deadline) {
        
        batch.push_back(std::move(request_queue_.front()));
        request_queue_.pop();
    }
    
    return batch;
}

void RequestProcessor::processBatch(std::vector<BatchedRequest>& batch) {
    if (batch.empty()) {
        return;
    }
    
    try {
        // Process each request individually for now
        for (auto& batched_request : batch) {
            auto result = engine_core_->executeInference(batched_request.request);
            batched_request.promise.set_value(result);
        }
        
    } catch (const std::exception& e) {
        // Set error for all requests in batch
        InferenceResult error_result;
        error_result.error_message = std::string("Batch processing error: ") + e.what();
        
        for (auto& batched_request : batch) {
            batched_request.promise.set_value(error_result);
        }
    }
}
