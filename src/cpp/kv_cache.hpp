/**
 * @file kv_cache.hpp
 * @brief Key-Value cache management for transformer attention
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * KV cache implementation for efficient transformer attention computation and memory management.
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
#include <memory>
#include <cuda_runtime.h>

// Quantization modes for KV cache
enum class QuantizationMode {
    NONE = 0,      // FP16 (no quantization)
    INT8 = 1,      // INT8 quantization (4x memory reduction)
    INT4 = 2       // INT4 quantization (8x memory reduction) 
};

class KVCache {
public:
    struct CacheState {
        void* keys_cache = nullptr;      // GPU memory for keys
        void* values_cache = nullptr;    // GPU memory for values
        void* keys_scale = nullptr;      // Quantization scales for keys
        void* values_scale = nullptr;    // Quantization scales for values
        size_t current_seq_len = 0;      // Current sequence length in cache
        size_t max_seq_len = 0;          // Maximum sequence length
        size_t batch_size = 1;           // Batch size
        bool is_valid = false;           // Whether cache contains valid data
        QuantizationMode quant_mode = QuantizationMode::INT8;  // Quantization mode
    };

    KVCache() = default;
    ~KVCache() { cleanup(); }

    bool initialize(size_t batch_size, size_t max_seq_len, size_t num_layers, 
                   size_t num_heads, size_t head_dim, QuantizationMode quant_mode = QuantizationMode::INT8);
    
    void cleanup();
    
    // Reset cache for new sequence
    void reset();
    
    // Get cache state for TensorRT execution
    CacheState& getCacheState() { return cache_state_; }
    
    // Update cache after inference
    void updateCacheLength(size_t new_seq_len);
    
    // Check if we can use cached computation
    bool canUseCache(size_t input_seq_len) const;
    
    // Store and retrieve operations for cache management
    bool store(size_t layer, size_t batch_idx, size_t seq_pos, 
               const float* keys, const float* values, size_t key_value_size);
    bool retrieve(size_t layer, size_t batch_idx, size_t seq_pos,
                  float* keys, float* values, size_t key_value_size);
    
    // Calculate memory offset for cache access
    size_t calculateOffset(size_t layer, size_t batch_idx, size_t seq_pos) const;
    
    // Quantization operations for Blackwell Blackwell optimization
    void quantizeKeys(const float* fp16_keys, int8_t* int8_keys, float* scales, size_t size);
    void quantizeValues(const float* fp16_values, int8_t* int8_values, float* scales, size_t size);
    void dequantizeKeys(const int8_t* int8_keys, const float* scales, float* fp16_keys, size_t size);
    void dequantizeValues(const int8_t* int8_values, const float* scales, float* fp16_values, size_t size);
    
    // Memory usage reporting
    size_t getMemoryUsage() const;
    size_t getMemorySavings() const;
    
private:
    CacheState cache_state_;
    size_t cache_size_bytes_ = 0;
    size_t num_layers_ = 24;     // OSS-20B has 24 layers
    size_t num_heads_ = 64;      // OSS-20B has 64 attention heads  
    size_t head_dim_ = 45;       // 2880 / 64 = 45 per head
    bool initialized_ = false;
};