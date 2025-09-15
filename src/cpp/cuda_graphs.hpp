/**
 * @file cuda_graphs.hpp
 * @brief CUDA graph optimization for inference acceleration
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * CUDA graph capture and replay for minimizing kernel launch overhead in inference.
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

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>

class CUDAGraphManager {
public:
    CUDAGraphManager();
    ~CUDAGraphManager();
    
    bool initialize(size_t max_batch_size, size_t max_seq_len, size_t vocab_size);
    void cleanup();
    
    // Capture inference graph for a specific batch size
    bool captureInferenceGraph(size_t batch_size, 
                              void** input_buffers, 
                              void** output_buffers,
                              cudaStream_t stream,
                              std::function<void()> inference_func);
    
    // Execute captured graph
    bool executeGraph(size_t batch_size);
    
    // Update graph inputs for new data
    bool updateGraphInputs(size_t batch_size, void** input_buffers);
    
    // Check if graph is captured for given batch size
    bool isGraphCaptured(size_t batch_size) const;
    
    // Get memory usage
    size_t getMemoryUsage() const;
    
private:
    struct GraphInstance {
        cudaGraph_t graph;
        cudaGraphExec_t exec_graph;
        cudaStream_t stream;
        size_t batch_size;
        bool is_valid;
        
        GraphInstance() : graph(nullptr), exec_graph(nullptr), 
                         stream(nullptr), batch_size(0), is_valid(false) {}
    };
    
    bool initialized_;
    size_t max_batch_size_;
    size_t max_seq_len_;
    size_t vocab_size_;
    
    // Store graphs for different batch sizes
    std::vector<std::unique_ptr<GraphInstance>> graph_instances_;
    
    // CUDA streams for graph capture and execution
    cudaStream_t capture_stream_;
    std::vector<cudaStream_t> exec_streams_;
    
    // Helper methods
    GraphInstance* getOrCreateGraphInstance(size_t batch_size);
    void destroyGraphInstance(GraphInstance* instance);
};