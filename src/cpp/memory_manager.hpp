/**
 * @file memory_manager.hpp
 * @brief GPU memory management and buffer allocation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Optimized GPU memory allocation with buffer pooling and efficient memory reuse strategies.
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
#include <mutex>
#include <list>
#include "memory_pool.hpp"

struct MemoryBlock {
    void* ptr;
    size_t size;
    size_t offset;
    bool in_use;
    
    MemoryBlock(void* p, size_t s, size_t o) : ptr(p), size(s), offset(o), in_use(false) {}
};

class MemoryManager {
private:
    // GPU memory pools with proper alignment
    void* input_pool_ = nullptr;
    void* output_pool_ = nullptr;
    void* kv_cache_pool_ = nullptr;
    void* pinned_host_buffer_ = nullptr;
    
    size_t input_pool_size_ = 0;
    size_t output_pool_size_ = 0;
    size_t kv_cache_pool_size_ = 0;
    size_t pinned_buffer_size_ = 0;
    
    // Free list memory management for efficient allocation
    std::list<MemoryBlock> input_free_blocks_;
    std::list<MemoryBlock> output_free_blocks_;
    std::list<MemoryBlock> kv_cache_free_blocks_;
    
    // Advanced memory pools for optimized allocation
    std::unique_ptr<CUDAMemoryPool> cuda_pool_;
    std::unique_ptr<HostMemoryPool> host_pool_;
    
    // Thread safety
    std::mutex allocation_mutex_;
    
    int max_batch_size_ = 0;
    int max_sequence_length_ = 0;
    int vocab_size_ = 199036;  // Correct OSS-20B vocab size
    
    // Memory alignment constants
    static constexpr size_t MEMORY_ALIGNMENT = 256;  // GPU memory alignment
    
public:
    MemoryManager();
    ~MemoryManager();
    
    bool initialize(int max_batch_size, int max_sequence_length);
    bool allocateBuffers(int max_batch_size, int max_sequence_length);
    
    // Memory allocation/deallocation with free list management
    void* allocateInputBuffer(size_t size);
    void* allocateOutputBuffer(size_t size);
    void* allocateKVCacheBuffer(size_t size);
    
    void deallocateInputBuffer(void* buffer);
    void deallocateOutputBuffer(void* buffer);
    void deallocateKVCacheBuffer(void* buffer);
    
    // Pinned memory for fast transfers
    void* allocatePinnedBuffer(size_t size);
    void deallocatePinnedBuffer(void* buffer);
    
    // Memory utilities (both sync and async)
    void copyInputToGPU(const std::vector<int32_t>& tokens, void* gpu_buffer);
    void copyOutputFromGPU(void* gpu_buffer, std::vector<float>& logits, int vocab_size);
    void copyInputToGPUAsync(const std::vector<int32_t>& tokens, void* gpu_buffer, cudaStream_t stream = 0);
    void copyOutputFromGPUAsync(void* gpu_buffer, std::vector<float>& logits, int vocab_size, cudaStream_t stream = 0);
    
    // Memory pool interface (optimized allocation)
    void* allocateFromPool(size_t size, size_t alignment = 256);
    void deallocateToPool(void* ptr);
    void* allocateHostFromPool(size_t size, size_t alignment = 64);
    void deallocateHostToPool(void* ptr);
    
    // Memory info and utilities
    size_t getTotalAllocatedMemory() const;
    size_t getAvailableMemory() const;
    float getMemoryUtilization() const;
    void defragmentPool();  // Coalesce free blocks
    void printPoolStats() const;
    
private:
    void cleanup();
    void* allocateFromFreeList(std::list<MemoryBlock>& free_list, void* pool, size_t requested_size);
    void returnToFreeList(std::list<MemoryBlock>& free_list, void* buffer);
    size_t alignSize(size_t size) const;
    void initializeFreeList(std::list<MemoryBlock>& free_list, void* pool, size_t pool_size);
};
