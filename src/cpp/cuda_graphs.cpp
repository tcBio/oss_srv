/**
 * @file cuda_graphs.cpp
 * @brief CUDA graphs implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of CUDA graph optimization with kernel fusion and launch overhead reduction.
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

#include "cuda_graphs.hpp"
#include <iostream>
#include <algorithm>

CUDAGraphManager::CUDAGraphManager() 
    : initialized_(false), max_batch_size_(0), max_seq_len_(0), vocab_size_(0),
      capture_stream_(nullptr) {
}

CUDAGraphManager::~CUDAGraphManager() {
    cleanup();
}

bool CUDAGraphManager::initialize(size_t max_batch_size, size_t max_seq_len, size_t vocab_size) {
    if (initialized_) {
        cleanup();
    }
    
    max_batch_size_ = max_batch_size;
    max_seq_len_ = max_seq_len;
    vocab_size_ = vocab_size;
    
    std::cout << "Initializing CUDA Graph Manager for Blackwell Blackwell:" << std::endl;
    std::cout << "  Max batch size: " << max_batch_size << std::endl;
    std::cout << "  Max sequence length: " << max_seq_len << std::endl;
    std::cout << "  Vocabulary size: " << vocab_size << std::endl;
    
    // Create capture stream
    cudaError_t status = cudaStreamCreate(&capture_stream_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to create capture stream: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Create execution streams for different batch sizes
    exec_streams_.resize(max_batch_size_ + 1);
    for (size_t i = 1; i <= max_batch_size_; ++i) {
        status = cudaStreamCreate(&exec_streams_[i]);
        if (status != cudaSuccess) {
            std::cerr << "Failed to create execution stream " << i << ": " << cudaGetErrorString(status) << std::endl;
            cleanup();
            return false;
        }
    }
    
    // Pre-allocate graph instances for common batch sizes
    graph_instances_.resize(max_batch_size_ + 1);
    for (size_t i = 1; i <= max_batch_size_; ++i) {
        graph_instances_[i] = std::make_unique<GraphInstance>();
        graph_instances_[i]->batch_size = i;
        graph_instances_[i]->stream = exec_streams_[i];
    }
    
    initialized_ = true;
    std::cout << "CUDA Graph Manager initialized successfully" << std::endl;
    return true;
}

void CUDAGraphManager::cleanup() {
    // Destroy all graph instances
    for (auto& instance : graph_instances_) {
        if (instance) {
            destroyGraphInstance(instance.get());
        }
    }
    graph_instances_.clear();
    
    // Destroy execution streams
    for (auto stream : exec_streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    exec_streams_.clear();
    
    // Destroy capture stream
    if (capture_stream_) {
        cudaStreamDestroy(capture_stream_);
        capture_stream_ = nullptr;
    }
    
    initialized_ = false;
}

bool CUDAGraphManager::captureInferenceGraph(size_t batch_size,
                                            void** input_buffers,
                                            void** output_buffers,
                                            cudaStream_t stream,
                                            std::function<void()> inference_func) {
    if (!initialized_ || batch_size == 0 || batch_size > max_batch_size_) {
        std::cerr << "Invalid batch size for graph capture: " << batch_size << std::endl;
        return false;
    }
    
    auto* instance = getOrCreateGraphInstance(batch_size);
    if (!instance) {
        std::cerr << "Failed to get graph instance for batch size " << batch_size << std::endl;
        return false;
    }
    
    std::cout << "Capturing CUDA graph for batch size " << batch_size << std::endl;
    
    // Warm up - run the inference function a few times to ensure everything is loaded
    for (int i = 0; i < 3; ++i) {
        inference_func();
        cudaStreamSynchronize(stream);
    }
    
    // Begin graph capture
    cudaError_t status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (status != cudaSuccess) {
        std::cerr << "Failed to begin graph capture: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Execute the inference function to capture all CUDA operations
    try {
        inference_func();
    } catch (const std::exception& e) {
        std::cerr << "Exception during graph capture: " << e.what() << std::endl;
        cudaStreamEndCapture(stream, &instance->graph);  // End capture even on error
        return false;
    }
    
    // End graph capture
    status = cudaStreamEndCapture(stream, &instance->graph);
    if (status != cudaSuccess) {
        std::cerr << "Failed to end graph capture: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Instantiate the captured graph
    status = cudaGraphInstantiate(&instance->exec_graph, instance->graph, nullptr, nullptr, 0);
    if (status != cudaSuccess) {
        std::cerr << "Failed to instantiate graph: " << cudaGetErrorString(status) << std::endl;
        if (instance->graph) {
            cudaGraphDestroy(instance->graph);
            instance->graph = nullptr;
        }
        return false;
    }
    
    instance->is_valid = true;
    std::cout << "Successfully captured CUDA graph for batch size " << batch_size << std::endl;
    
    // Print graph statistics
    size_t num_nodes;
    status = cudaGraphGetNodes(instance->graph, nullptr, &num_nodes);
    if (status == cudaSuccess) {
        std::cout << "  Graph contains " << num_nodes << " nodes" << std::endl;
    }
    
    return true;
}

bool CUDAGraphManager::executeGraph(size_t batch_size) {
    if (!initialized_ || batch_size == 0 || batch_size > max_batch_size_) {
        return false;
    }
    
    auto* instance = getOrCreateGraphInstance(batch_size);
    if (!instance || !instance->is_valid) {
        return false;
    }
    
    // Launch the captured graph
    cudaError_t status = cudaGraphLaunch(instance->exec_graph, instance->stream);
    if (status != cudaSuccess) {
        std::cerr << "Failed to launch graph for batch size " << batch_size 
                  << ": " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    return true;
}

bool CUDAGraphManager::updateGraphInputs(size_t batch_size, void** input_buffers) {
    if (!initialized_ || batch_size == 0 || batch_size > max_batch_size_) {
        return false;
    }
    
    auto* instance = getOrCreateGraphInstance(batch_size);
    if (!instance || !instance->is_valid) {
        return false;
    }
    
    // For now, we'll recreate the graph with new inputs
    // In a production system, you'd use cudaGraphKernelNodeSetParams for updating
    // specific kernel parameters without full recapture
    
    // This is a simplified implementation - in practice you'd want to update
    // specific nodes rather than marking for recapture
    instance->is_valid = false;
    
    return true;
}

bool CUDAGraphManager::isGraphCaptured(size_t batch_size) const {
    if (!initialized_ || batch_size == 0 || batch_size > max_batch_size_) {
        return false;
    }
    
    if (batch_size < graph_instances_.size() && graph_instances_[batch_size]) {
        return graph_instances_[batch_size]->is_valid;
    }
    
    return false;
}

size_t CUDAGraphManager::getMemoryUsage() const {
    if (!initialized_) return 0;
    
    // CUDA graphs have minimal memory overhead
    // Main memory usage is in the captured kernels and their parameters
    size_t total_usage = 0;
    
    for (const auto& instance : graph_instances_) {
        if (instance && instance->is_valid) {
            // Rough estimate - actual usage depends on captured operations
            total_usage += instance->batch_size * max_seq_len_ * sizeof(float) * 2; // Input + Output
        }
    }
    
    return total_usage;
}

CUDAGraphManager::GraphInstance* CUDAGraphManager::getOrCreateGraphInstance(size_t batch_size) {
    if (batch_size >= graph_instances_.size()) {
        return nullptr;
    }
    
    auto& instance = graph_instances_[batch_size];
    if (!instance) {
        instance = std::make_unique<GraphInstance>();
        instance->batch_size = batch_size;
        instance->stream = exec_streams_[batch_size];
    }
    
    return instance.get();
}

void CUDAGraphManager::destroyGraphInstance(GraphInstance* instance) {
    if (!instance) return;
    
    if (instance->exec_graph) {
        cudaGraphExecDestroy(instance->exec_graph);
        instance->exec_graph = nullptr;
    }
    
    if (instance->graph) {
        cudaGraphDestroy(instance->graph);
        instance->graph = nullptr;
    }
    
    instance->is_valid = false;
}