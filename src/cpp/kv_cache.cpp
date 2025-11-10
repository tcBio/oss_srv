/**
 * @file kv_cache.cpp
 * @brief KV cache implementation
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Implementation of key-value cache with GPU memory optimization for attention mechanisms.
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

#include "kv_cache.hpp"
#include <iostream>
#include <cstring>

// CUDA kernel declarations
extern "C" {
void launchQuantizeINT8(const float* input, int8_t* output, float* scales, size_t size, cudaStream_t stream);
void launchDequantizeINT8(const int8_t* input, const float* scales, float* output, size_t size, cudaStream_t stream);
void launchQuantizeINT4(const float* input, int8_t* output, float* scales, size_t size, cudaStream_t stream);
void launchDequantizeINT4(const int8_t* input, const float* scales, float* output, size_t size, cudaStream_t stream);
}

bool KVCache::initialize(size_t batch_size, size_t max_seq_len, size_t num_layers,
                        size_t num_heads, size_t head_dim, QuantizationMode quant_mode) {
    if (initialized_) {
        cleanup();
    }

    cache_state_.batch_size = batch_size;
    cache_state_.max_seq_len = max_seq_len;
    cache_state_.quant_mode = quant_mode;
    cache_state_.is_valid = false;

    num_layers_ = num_layers;
    num_heads_ = num_heads;
    head_dim_ = head_dim;

    // Calculate memory requirements
    // Each layer needs: batch_size * max_seq_len * num_heads * head_dim elements
    size_t elements_per_layer = batch_size * max_seq_len * num_heads * head_dim;
    size_t total_elements = elements_per_layer * num_layers;

    size_t element_size;
    switch (quant_mode) {
        case QuantizationMode::NONE:
            element_size = sizeof(float);  // FP32 or FP16
            break;
        case QuantizationMode::INT8:
            element_size = sizeof(int8_t);
            break;
        case QuantizationMode::INT4:
            element_size = sizeof(int8_t) / 2;  // Pack 2 values per byte
            break;
    }

    cache_size_bytes_ = total_elements * element_size * 2;  // *2 for keys and values

    std::cout << "Initializing KV Cache:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Max sequence length: " << max_seq_len << std::endl;
    std::cout << "  Num layers: " << num_layers << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Quantization mode: " << (quant_mode == QuantizationMode::NONE ? "NONE" :
                                             quant_mode == QuantizationMode::INT8 ? "INT8" : "INT4") << std::endl;
    std::cout << "  Cache memory: " << cache_size_bytes_ / (1024*1024) << " MB" << std::endl;

    // Allocate GPU memory for keys cache
    cudaError_t status = cudaMalloc(&cache_state_.keys_cache, total_elements * element_size);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate keys cache: " << cudaGetErrorString(status) << std::endl;
        cleanup();
        return false;
    }

    // Allocate GPU memory for values cache
    status = cudaMalloc(&cache_state_.values_cache, total_elements * element_size);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate values cache: " << cudaGetErrorString(status) << std::endl;
        cleanup();
        return false;
    }

    // Allocate scale factors for quantization (if using quantization)
    if (quant_mode != QuantizationMode::NONE) {
        size_t num_scales = total_elements / 128;  // One scale per 128 elements (typical block size)

        status = cudaMalloc(&cache_state_.keys_scale, num_scales * sizeof(float));
        if (status != cudaSuccess) {
            std::cerr << "Failed to allocate keys scale: " << cudaGetErrorString(status) << std::endl;
            cleanup();
            return false;
        }

        status = cudaMalloc(&cache_state_.values_scale, num_scales * sizeof(float));
        if (status != cudaSuccess) {
            std::cerr << "Failed to allocate values scale: " << cudaGetErrorString(status) << std::endl;
            cleanup();
            return false;
        }
    }

    // Initialize cache to zero
    cudaMemset(cache_state_.keys_cache, 0, total_elements * element_size);
    cudaMemset(cache_state_.values_cache, 0, total_elements * element_size);

    initialized_ = true;
    std::cout << "KV Cache initialized successfully" << std::endl;
    return true;
}

void KVCache::cleanup() {
    if (cache_state_.keys_cache) {
        cudaFree(cache_state_.keys_cache);
        cache_state_.keys_cache = nullptr;
    }
    if (cache_state_.values_cache) {
        cudaFree(cache_state_.values_cache);
        cache_state_.values_cache = nullptr;
    }
    if (cache_state_.keys_scale) {
        cudaFree(cache_state_.keys_scale);
        cache_state_.keys_scale = nullptr;
    }
    if (cache_state_.values_scale) {
        cudaFree(cache_state_.values_scale);
        cache_state_.values_scale = nullptr;
    }
    
    cache_state_.is_valid = false;
    initialized_ = false;
}

void KVCache::reset() {
    cache_state_.current_seq_len = 0;
    cache_state_.is_valid = false;
}

void KVCache::updateCacheLength(size_t new_seq_len) {
    cache_state_.current_seq_len = new_seq_len;
    cache_state_.is_valid = true;
}

bool KVCache::canUseCache(size_t input_seq_len) const {
    return cache_state_.is_valid && 
           (cache_state_.current_seq_len + input_seq_len <= cache_state_.max_seq_len);
}

bool KVCache::store(size_t layer, size_t batch_idx, size_t seq_pos,
                   const float* keys, const float* values, size_t key_value_size) {
    if (!initialized_) return false;

    size_t offset = calculateOffset(layer, batch_idx, seq_pos);
    size_t byte_size = key_value_size * sizeof(float);

    // Store keys
    void* keys_dest = static_cast<char*>(cache_state_.keys_cache) + offset * sizeof(float);
    cudaError_t status = cudaMemcpy(keys_dest, keys, byte_size, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        std::cerr << "Failed to store keys: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    // Store values
    void* values_dest = static_cast<char*>(cache_state_.values_cache) + offset * sizeof(float);
    status = cudaMemcpy(values_dest, values, byte_size, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        std::cerr << "Failed to store values: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    return true;
}

bool KVCache::retrieve(size_t layer, size_t batch_idx, size_t seq_pos,
                      float* keys, float* values, size_t key_value_size) {
    if (!initialized_ || !cache_state_.is_valid) return false;

    size_t offset = calculateOffset(layer, batch_idx, seq_pos);
    size_t byte_size = key_value_size * sizeof(float);

    // Retrieve keys
    void* keys_src = static_cast<char*>(cache_state_.keys_cache) + offset * sizeof(float);
    cudaError_t status = cudaMemcpy(keys, keys_src, byte_size, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        std::cerr << "Failed to retrieve keys: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    // Retrieve values
    void* values_src = static_cast<char*>(cache_state_.values_cache) + offset * sizeof(float);
    status = cudaMemcpy(values, values_src, byte_size, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        std::cerr << "Failed to retrieve values: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    return true;
}

size_t KVCache::calculateOffset(size_t layer, size_t batch_idx, size_t seq_pos) const {
    return layer * cache_state_.batch_size * cache_state_.max_seq_len * num_heads_ * head_dim_ +
           batch_idx * cache_state_.max_seq_len * num_heads_ * head_dim_ +
           seq_pos * num_heads_ * head_dim_;
}

void KVCache::quantizeKeys(const float* fp16_keys, int8_t* int8_keys, float* scales, size_t size) {
    if (cache_state_.quant_mode == QuantizationMode::INT8) {
        launchQuantizeINT8(fp16_keys, int8_keys, scales, size, nullptr);
    } else if (cache_state_.quant_mode == QuantizationMode::INT4) {
        launchQuantizeINT4(fp16_keys, int8_keys, scales, size, nullptr);
    }
    cudaDeviceSynchronize();  // Ensure completion
}

void KVCache::quantizeValues(const float* fp16_values, int8_t* int8_values, float* scales, size_t size) {
    if (cache_state_.quant_mode == QuantizationMode::INT8) {
        launchQuantizeINT8(fp16_values, int8_values, scales, size, nullptr);
    } else if (cache_state_.quant_mode == QuantizationMode::INT4) {
        launchQuantizeINT4(fp16_values, int8_values, scales, size, nullptr);
    }
    cudaDeviceSynchronize();  // Ensure completion
}

void KVCache::dequantizeKeys(const int8_t* int8_keys, const float* scales, float* fp16_keys, size_t size) {
    if (cache_state_.quant_mode == QuantizationMode::INT8) {
        launchDequantizeINT8(int8_keys, scales, fp16_keys, size, nullptr);
    } else if (cache_state_.quant_mode == QuantizationMode::INT4) {
        launchDequantizeINT4(int8_keys, scales, fp16_keys, size, nullptr);
    }
    cudaDeviceSynchronize();  // Ensure completion
}

void KVCache::dequantizeValues(const int8_t* int8_values, const float* scales, float* fp16_values, size_t size) {
    if (cache_state_.quant_mode == QuantizationMode::INT8) {
        launchDequantizeINT8(int8_values, scales, fp16_values, size, nullptr);
    } else if (cache_state_.quant_mode == QuantizationMode::INT4) {
        launchDequantizeINT4(int8_values, scales, fp16_values, size, nullptr);
    }
    cudaDeviceSynchronize();  // Ensure completion
}

size_t KVCache::getMemoryUsage() const {
    return cache_size_bytes_;
}

size_t KVCache::getMemorySavings() const {
    // Return estimated memory savings from quantization
    if (cache_state_.quant_mode == QuantizationMode::INT8) {
        return cache_size_bytes_ / 2;  // 50% savings with INT8
    } else if (cache_state_.quant_mode == QuantizationMode::INT4) {
        return cache_size_bytes_ * 3 / 4;  // 75% savings with INT4
    }
    return 0;
}
