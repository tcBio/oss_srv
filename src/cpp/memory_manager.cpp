/**
 * @file memory_manager.cpp
 * @brief Memory manager implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of GPU memory management with CUDA buffer allocation and pool optimization.
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

#include "memory_manager.hpp"
#include <iostream>
#include <algorithm>
#include <cassert>

MemoryManager::MemoryManager() = default;

MemoryManager::~MemoryManager() {
    cleanup();
}

bool MemoryManager::initialize(int max_batch_size, int max_sequence_length) {
    return allocateBuffers(max_batch_size, max_sequence_length);
}

bool MemoryManager::allocateBuffers(int max_batch_size, int max_sequence_length) {
    max_batch_size_ = max_batch_size;
    max_sequence_length_ = max_sequence_length;
    
    std::cout << "Allocating optimized GPU memory pools with free list management..." << std::endl;
    
    // Calculate buffer sizes with correct OSS-20B parameters and efficient allocation
    input_pool_size_ = alignSize(max_batch_size * max_sequence_length * sizeof(int32_t) * 4); // 4x for headroom
    
    // Output pool: Only need logits for the last position (major memory saving)
    output_pool_size_ = alignSize(max_batch_size * vocab_size_ * sizeof(float) * 2); // 2x for double buffering
    
    // KV cache size with correct OSS-20B architecture
    // OSS-20B: 24 layers * 64 heads * 45 head_dim (2880/64) * max_seq_len * batch_size * 2 (K,V) * fp16
    int num_layers = 24;  // OSS-20B has 24 layers (not 32!)
    int num_heads = 64;   // 64 attention heads (not 32!)
    int head_dim = 45;    // 2880 / 64 = 45 (not 64!)
    kv_cache_pool_size_ = alignSize(num_layers * 2 * num_heads * head_dim * max_sequence_length * max_batch_size * sizeof(float) / 2); // fp16
    
    // Allocate pinned host memory for fast transfers
    pinned_buffer_size_ = alignSize(max_batch_size * max_sequence_length * sizeof(int32_t));
    
    // Allocate GPU memory with proper error handling
    cudaError_t status;
    
    status = cudaMalloc(&input_pool_, input_pool_size_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate input pool: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    status = cudaMalloc(&output_pool_, output_pool_size_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate output pool: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    status = cudaMalloc(&kv_cache_pool_, kv_cache_pool_size_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate KV cache pool: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Allocate pinned host memory for async transfers
    status = cudaHostAlloc(&pinned_host_buffer_, pinned_buffer_size_, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host buffer: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Initialize free lists for efficient memory management
    initializeFreeList(input_free_blocks_, input_pool_, input_pool_size_);
    initializeFreeList(output_free_blocks_, output_pool_, output_pool_size_);
    initializeFreeList(kv_cache_free_blocks_, kv_cache_pool_, kv_cache_pool_size_);
    
    std::cout << "Memory pools allocated:" << std::endl;
    std::cout << "  Input pool: " << (input_pool_size_ / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Output pool: " << (output_pool_size_ / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  KV cache pool: " << (kv_cache_pool_size_ / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Total: " << ((input_pool_size_ + output_pool_size_ + kv_cache_pool_size_) / 1024 / 1024) << " MB" << std::endl;
    
    // Initialize advanced memory pools for optimized allocation
    size_t total_cuda_pool_size = input_pool_size_ + output_pool_size_ + kv_cache_pool_size_ + (512 * 1024 * 1024); // +512MB for general use
    cuda_pool_ = std::make_unique<CUDAMemoryPool>();
    if (cuda_pool_->initialize(total_cuda_pool_size)) {
        std::cout << "Advanced CUDA memory pool initialized: " << (total_cuda_pool_size / 1024 / 1024) << " MB" << std::endl;
    } else {
        std::cout << "Warning: CUDA memory pool initialization failed, continuing with standard allocation" << std::endl;
        cuda_pool_.reset();
    }
    
    // Initialize host memory pool
    size_t host_pool_size = 256 * 1024 * 1024; // 256MB host pool
    host_pool_ = std::make_unique<HostMemoryPool>();
    if (host_pool_->initialize(host_pool_size)) {
        std::cout << "Host memory pool initialized: " << (host_pool_size / 1024 / 1024) << " MB" << std::endl;
    } else {
        std::cout << "Warning: Host memory pool initialization failed" << std::endl;
        host_pool_.reset();
    }
    
    return true;
}

void* MemoryManager::allocateInputBuffer(size_t size) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    size_t aligned_size = alignSize(size);
    return allocateFromFreeList(input_free_blocks_, input_pool_, aligned_size);
}

void* MemoryManager::allocateOutputBuffer(size_t size) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    size_t aligned_size = alignSize(size);
    return allocateFromFreeList(output_free_blocks_, output_pool_, aligned_size);
}

void* MemoryManager::allocateKVCacheBuffer(size_t size) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    size_t aligned_size = alignSize(size);
    return allocateFromFreeList(kv_cache_free_blocks_, kv_cache_pool_, aligned_size);
}

void MemoryManager::deallocateInputBuffer(void* buffer) {
    if (!buffer) return;
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    returnToFreeList(input_free_blocks_, buffer);
}

void MemoryManager::deallocateOutputBuffer(void* buffer) {
    if (!buffer) return;
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    returnToFreeList(output_free_blocks_, buffer);
}

void MemoryManager::deallocateKVCacheBuffer(void* buffer) {
    if (!buffer) return;
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    returnToFreeList(kv_cache_free_blocks_, buffer);
}

void* MemoryManager::allocatePinnedBuffer(size_t size) {
    void* buffer = nullptr;
    cudaError_t status = cudaHostAlloc(&buffer, alignSize(size), cudaHostAllocDefault);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate pinned buffer: " << cudaGetErrorString(status) << std::endl;
        return nullptr;
    }
    return buffer;
}

void MemoryManager::deallocatePinnedBuffer(void* buffer) {
    if (buffer) {
        cudaFreeHost(buffer);
    }
}

void MemoryManager::copyInputToGPU(const std::vector<int32_t>& tokens, void* gpu_buffer) {
    if (!gpu_buffer) return;
    
    size_t copy_size = tokens.size() * sizeof(int32_t);
    cudaMemcpy(gpu_buffer, tokens.data(), copy_size, cudaMemcpyHostToDevice);
}

void MemoryManager::copyInputToGPUAsync(const std::vector<int32_t>& tokens, void* gpu_buffer, cudaStream_t stream) {
    if (!gpu_buffer) return;
    
    size_t copy_size = tokens.size() * sizeof(int32_t);
    cudaMemcpyAsync(gpu_buffer, tokens.data(), copy_size, cudaMemcpyHostToDevice, stream);
}

void MemoryManager::copyOutputFromGPU(void* gpu_buffer, std::vector<float>& logits, int vocab_size) {
    if (!gpu_buffer) return;
    
    logits.resize(vocab_size);
    size_t copy_size = vocab_size * sizeof(float);
    cudaMemcpy(logits.data(), gpu_buffer, copy_size, cudaMemcpyDeviceToHost);
}

void MemoryManager::copyOutputFromGPUAsync(void* gpu_buffer, std::vector<float>& logits, int vocab_size, cudaStream_t stream) {
    if (!gpu_buffer) return;
    
    logits.resize(vocab_size);
    size_t copy_size = vocab_size * sizeof(float);
    cudaMemcpyAsync(logits.data(), gpu_buffer, copy_size, cudaMemcpyDeviceToHost, stream);
}

size_t MemoryManager::getTotalAllocatedMemory() const {
    return input_pool_size_ + output_pool_size_ + kv_cache_pool_size_;
}

size_t MemoryManager::getAvailableMemory() const {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

float MemoryManager::getMemoryUtilization() const {
    size_t total_memory = input_pool_size_ + output_pool_size_ + kv_cache_pool_size_;
    size_t used_memory = 0;
    
    // Calculate used memory from all free lists
    for (const auto& block : input_free_blocks_) {
        if (block.in_use) used_memory += block.size;
    }
    for (const auto& block : output_free_blocks_) {
        if (block.in_use) used_memory += block.size;
    }
    for (const auto& block : kv_cache_free_blocks_) {
        if (block.in_use) used_memory += block.size;
    }
    
    return total_memory > 0 ? static_cast<float>(used_memory) / total_memory : 0.0f;
}


size_t MemoryManager::alignSize(size_t size) const {
    return ((size + MEMORY_ALIGNMENT - 1) / MEMORY_ALIGNMENT) * MEMORY_ALIGNMENT;
}

void MemoryManager::initializeFreeList(std::list<MemoryBlock>& free_list, void* pool, size_t pool_size) {
    free_list.clear();
    free_list.emplace_back(pool, pool_size, 0);
}

void* MemoryManager::allocateFromFreeList(std::list<MemoryBlock>& free_list, void* pool, size_t requested_size) {
    // Find first fit
    for (auto it = free_list.begin(); it != free_list.end(); ++it) {
        if (!it->in_use && it->size >= requested_size) {
            // Split block if necessary
            if (it->size > requested_size) {
                size_t remaining_size = it->size - requested_size;
                void* remaining_ptr = static_cast<char*>(it->ptr) + requested_size;
                free_list.emplace(std::next(it), remaining_ptr, remaining_size, it->offset + requested_size);
                it->size = requested_size;
            }
            it->in_use = true;
            return it->ptr;
        }
    }
    
    std::cerr << "Out of memory in pool, requested: " << requested_size << " bytes" << std::endl;
    return nullptr;
}

void MemoryManager::returnToFreeList(std::list<MemoryBlock>& free_list, void* buffer) {
    // Find the block to return
    for (auto it = free_list.begin(); it != free_list.end(); ++it) {
        if (it->ptr == buffer && it->in_use) {
            it->in_use = false;
            
            // Coalesce with next block if possible
            auto next = std::next(it);
            if (next != free_list.end() && !next->in_use && 
                static_cast<char*>(it->ptr) + it->size == next->ptr) {
                it->size += next->size;
                free_list.erase(next);
            }
            
            // Coalesce with previous block if possible
            if (it != free_list.begin()) {
                auto prev = std::prev(it);
                if (!prev->in_use && static_cast<char*>(prev->ptr) + prev->size == it->ptr) {
                    prev->size += it->size;
                    free_list.erase(it);
                }
            }
            return;
        }
    }
    std::cerr << "Warning: Attempted to return buffer not found in free list" << std::endl;
}

void MemoryManager::defragmentPool() {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    // This would implement memory defragmentation - complex but not needed for Phase 1
}

void MemoryManager::cleanup() {
    if (input_pool_) {
        cudaFree(input_pool_);
        input_pool_ = nullptr;
    }
    
    if (output_pool_) {
        cudaFree(output_pool_);
        output_pool_ = nullptr;
    }
    
    if (kv_cache_pool_) {
        cudaFree(kv_cache_pool_);
        kv_cache_pool_ = nullptr;
    }
    
    if (pinned_host_buffer_) {
        cudaFreeHost(pinned_host_buffer_);
        pinned_host_buffer_ = nullptr;
    }
    
    input_free_blocks_.clear();
    output_free_blocks_.clear();
    kv_cache_free_blocks_.clear();
}

// Advanced memory pool interface
void* MemoryManager::allocateFromPool(size_t size, size_t alignment) {
    if (cuda_pool_) {
        return cuda_pool_->allocate(size, alignment);
    }
    
    // Fallback to regular CUDA allocation
    void* ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, size);
    return (status == cudaSuccess) ? ptr : nullptr;
}

void MemoryManager::deallocateToPool(void* ptr) {
    if (cuda_pool_) {
        cuda_pool_->deallocate(ptr);
    } else {
        cudaFree(ptr);
    }
}

void* MemoryManager::allocateHostFromPool(size_t size, size_t alignment) {
    if (host_pool_) {
        return host_pool_->allocate(size, alignment);
    }
    
    // Fallback to regular host allocation (MSVC compatible)
    return _aligned_malloc(size, alignment);
}

void MemoryManager::deallocateHostToPool(void* ptr) {
    if (host_pool_) {
        host_pool_->deallocate(ptr);
    } else {
        _aligned_free(ptr);
    }
}

void MemoryManager::printPoolStats() const {
    std::cout << "=== Memory Pool Statistics ===" << std::endl;
    
    if (cuda_pool_) {
        auto stats = cuda_pool_->getStats();
        std::cout << "CUDA Pool:" << std::endl;
        std::cout << "  Total allocated: " << (stats.total_allocated / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Currently in use: " << (stats.total_in_use / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Utilization: " << (100.0f * stats.total_in_use / stats.total_allocated) << "%" << std::endl;
        std::cout << "  Free blocks: " << stats.num_free_blocks << std::endl;
        std::cout << "  Largest free block: " << (stats.largest_free_block / 1024 / 1024) << " MB" << std::endl;
    } else {
        std::cout << "CUDA Pool: Not initialized" << std::endl;
    }
    
    if (host_pool_) {
        std::cout << "Host Pool:" << std::endl;
        std::cout << "  Total allocated: " << (host_pool_->getTotalAllocated() / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Currently in use: " << (host_pool_->getTotalInUse() / 1024 / 1024) << " MB" << std::endl;
    } else {
        std::cout << "Host Pool: Not initialized" << std::endl;
    }
    
    std::cout << "===============================" << std::endl;
}
