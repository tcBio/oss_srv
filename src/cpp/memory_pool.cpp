/**
 * @file memory_pool.cpp
 * @brief Memory pool implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of memory pool with allocation strategies and fragmentation reduction.
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

#include "memory_pool.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

// CUDA Memory Pool Implementation
CUDAMemoryPool::CUDAMemoryPool() 
    : initialized_(false), pool_base_(nullptr), pool_size_(0) {
}

CUDAMemoryPool::~CUDAMemoryPool() {
    cleanup();
}

bool CUDAMemoryPool::initialize(size_t initial_pool_size) {
    if (initialized_) {
        cleanup();
    }
    
    // Align pool size to 1MB boundary for optimal performance
    pool_size_ = ((initial_pool_size + 1048576 - 1) / 1048576) * 1048576;
    
    std::cout << "Initializing CUDA memory pool:" << std::endl;
    std::cout << "  Requested size: " << (initial_pool_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Aligned size: " << (pool_size_ / 1024 / 1024) << " MB" << std::endl;
    
    // Allocate the entire pool at once
    cudaError_t status = cudaMalloc(&pool_base_, pool_size_);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA memory pool: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Create initial single free block covering entire pool
    blocks_.emplace_back(std::make_unique<MemoryBlock>(pool_base_, pool_size_));
    ptr_to_block_index_[pool_base_] = 0;
    
    initialized_ = true;
    std::cout << "CUDA memory pool initialized successfully" << std::endl;
    
    return true;
}

void CUDAMemoryPool::cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (pool_base_) {
        cudaFree(pool_base_);
        pool_base_ = nullptr;
    }
    
    blocks_.clear();
    ptr_to_block_index_.clear();
    pool_size_ = 0;
    initialized_ = false;
}

void* CUDAMemoryPool::allocate(size_t size, size_t alignment) {
    if (!initialized_ || size == 0) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t aligned_size = alignSize(size, alignment);
    MemoryBlock* block = findFreeBlock(aligned_size, alignment);
    
    if (!block) {
        // Pool is full - could implement expansion here
        std::cerr << "CUDA memory pool exhausted, requested: " << size << " bytes" << std::endl;
        return nullptr;
    }
    
    // Mark block as in use
    block->in_use = true;
    
    // Split block if it's significantly larger than needed
    size_t block_index = std::distance(blocks_.begin(), 
        std::find_if(blocks_.begin(), blocks_.end(), 
            [block](const auto& b) { return b.get() == block; }));
    
    if (block->size > aligned_size + 1024) { // Leave some padding
        splitBlock(block_index, aligned_size);
    }
    
    return block->ptr;
}

void CUDAMemoryPool::deallocate(void* ptr) {
    if (!ptr || !initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    auto it = ptr_to_block_index_.find(ptr);
    if (it == ptr_to_block_index_.end()) {
        std::cerr << "Attempting to deallocate pointer not from memory pool" << std::endl;
        return;
    }
    
    size_t block_index = it->second;
    if (block_index >= blocks_.size()) {
        std::cerr << "Invalid block index in deallocation" << std::endl;
        return;
    }
    
    blocks_[block_index]->in_use = false;
    
    // Merge adjacent free blocks to reduce fragmentation
    mergeAdjacentFreeBlocks();
}

CUDAMemoryPool::PoolStats CUDAMemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(pool_mutex_));
    
    PoolStats stats = {};
    stats.total_allocated = pool_size_;
    stats.num_blocks = blocks_.size();
    
    for (const auto& block : blocks_) {
        if (block->in_use) {
            stats.total_in_use += block->size;
        } else {
            stats.num_free_blocks++;
            stats.largest_free_block = std::max(stats.largest_free_block, block->size);
        }
    }
    
    return stats;
}

void CUDAMemoryPool::defragment() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    mergeAdjacentFreeBlocks();
}

CUDAMemoryPool::MemoryBlock* CUDAMemoryPool::findFreeBlock(size_t size, size_t alignment) {
    for (auto& block : blocks_) {
        if (!block->in_use && block->size >= size) {
            // Check alignment
            uintptr_t addr = reinterpret_cast<uintptr_t>(block->ptr);
            if (addr % alignment == 0) {
                return block.get();
            }
        }
    }
    return nullptr;
}

void CUDAMemoryPool::splitBlock(size_t block_index, size_t needed_size) {
    if (block_index >= blocks_.size()) {
        return;
    }
    
    auto& original_block = blocks_[block_index];
    if (original_block->size <= needed_size) {
        return;
    }
    
    // Create new block for remaining space
    void* new_ptr = static_cast<char*>(original_block->ptr) + needed_size;
    size_t new_size = original_block->size - needed_size;
    
    blocks_.emplace_back(std::make_unique<MemoryBlock>(new_ptr, new_size));
    ptr_to_block_index_[new_ptr] = blocks_.size() - 1;
    
    // Update original block size
    original_block->size = needed_size;
}

void CUDAMemoryPool::mergeAdjacentFreeBlocks() {
    // Sort blocks by address to find adjacent blocks
    std::vector<size_t> free_indices;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (!blocks_[i]->in_use) {
            free_indices.push_back(i);
        }
    }
    
    std::sort(free_indices.begin(), free_indices.end(), 
        [this](size_t a, size_t b) {
            return blocks_[a]->ptr < blocks_[b]->ptr;
        });
    
    // Merge adjacent blocks (simplified implementation)
    for (size_t i = 0; i < free_indices.size(); ++i) {
        auto& current_block = blocks_[free_indices[i]];
        
        for (size_t j = i + 1; j < free_indices.size(); ++j) {
            auto& next_block = blocks_[free_indices[j]];
            
            // Check if blocks are adjacent
            if (static_cast<char*>(current_block->ptr) + current_block->size == next_block->ptr) {
                // Merge blocks
                current_block->size += next_block->size;
                
                // Remove the merged block
                ptr_to_block_index_.erase(next_block->ptr);
                blocks_.erase(blocks_.begin() + free_indices[j]);
                
                // Update indices
                for (auto& entry : ptr_to_block_index_) {
                    if (entry.second > free_indices[j]) {
                        entry.second--;
                    }
                }
                
                break;
            }
        }
    }
}

size_t CUDAMemoryPool::alignSize(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// Host Memory Pool Implementation
HostMemoryPool::HostMemoryPool() 
    : initialized_(false), total_allocated_(0), total_in_use_(0) {
}

HostMemoryPool::~HostMemoryPool() {
    cleanup();
}

bool HostMemoryPool::initialize(size_t initial_pool_size) {
    if (initialized_) {
        cleanup();
    }
    
    std::cout << "Host memory pool initialized with " << (initial_pool_size / 1024 / 1024) << " MB" << std::endl;
    
    // For host memory, we'll use a simple allocation strategy
    total_allocated_ = 0;
    total_in_use_ = 0;
    initialized_ = true;
    
    return true;
}

void HostMemoryPool::cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Free all allocated blocks
    for (auto& block : blocks_) {
        if (block.ptr) {
            free(block.ptr);
        }
    }
    
    blocks_.clear();
    total_allocated_ = 0;
    total_in_use_ = 0;
    initialized_ = false;
}

void* HostMemoryPool::allocate(size_t size, size_t alignment) {
    if (!initialized_ || size == 0) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Look for existing free block
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            total_in_use_ += block.size;
            return block.ptr;
        }
    }
    
    // Allocate new block
    void* ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        return nullptr;
    }
    
    blocks_.push_back({ptr, size, true});
    total_allocated_ += size;
    total_in_use_ += size;
    
    return ptr;
}

void HostMemoryPool::deallocate(void* ptr) {
    if (!ptr || !initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    for (auto& block : blocks_) {
        if (block.ptr == ptr && block.in_use) {
            block.in_use = false;
            total_in_use_ -= block.size;
            return;
        }
    }
    
    std::cerr << "Attempting to deallocate unknown host pointer" << std::endl;
}

size_t HostMemoryPool::getTotalAllocated() const {
    return total_allocated_;
}

size_t HostMemoryPool::getTotalInUse() const {
    return total_in_use_;
}