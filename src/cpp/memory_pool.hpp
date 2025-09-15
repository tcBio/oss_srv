/**
 * @file memory_pool.hpp
 * @brief Memory pool management for efficient allocation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Memory pool implementation with pre-allocation and efficient reuse for GPU buffers.
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
#include <unordered_map>

class CUDAMemoryPool {
public:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        
        MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
    };
    
    CUDAMemoryPool();
    ~CUDAMemoryPool();
    
    bool initialize(size_t initial_pool_size);
    void cleanup();
    
    // Allocate memory from pool
    void* allocate(size_t size, size_t alignment = 256);
    
    // Free memory back to pool
    void deallocate(void* ptr);
    
    // Get pool statistics
    struct PoolStats {
        size_t total_allocated;
        size_t total_in_use;
        size_t num_blocks;
        size_t num_free_blocks;
        size_t largest_free_block;
    };
    
    PoolStats getStats() const;
    
    // Defragment the pool (merge adjacent free blocks)
    void defragment();
    
private:
    bool initialized_;
    std::mutex pool_mutex_;
    
    // Memory pool data
    void* pool_base_;
    size_t pool_size_;
    
    // Block management
    std::vector<std::unique_ptr<MemoryBlock>> blocks_;
    std::unordered_map<void*, size_t> ptr_to_block_index_;
    
    // Helper methods
    MemoryBlock* findFreeBlock(size_t size, size_t alignment);
    void splitBlock(size_t block_index, size_t needed_size);
    void mergeAdjacentFreeBlocks();
    size_t alignSize(size_t size, size_t alignment);
};

// Host memory pool for CPU-side allocations
class HostMemoryPool {
public:
    HostMemoryPool();
    ~HostMemoryPool();
    
    bool initialize(size_t initial_pool_size);
    void cleanup();
    
    void* allocate(size_t size, size_t alignment = 64);
    void deallocate(void* ptr);
    
    size_t getTotalAllocated() const;
    size_t getTotalInUse() const;
    
private:
    bool initialized_;
    std::mutex pool_mutex_;
    
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
    size_t total_allocated_;
    size_t total_in_use_;
};