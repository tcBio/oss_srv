/**
 * @file kv_cache_kernels.cu
 * @brief CUDA kernels for KV cache quantization
 *
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized
 * for NVIDIA Blackwell GPU architecture.
 *
 * CUDA kernels for INT8/INT4 quantization with Blackwell optimizations.
 *
 * @author Brian Worthington
 * @date 2025
 * @version 1.0
 *
 * @copyright MIT License
 *
 * Repository: https://github.com/tcBio/oss_srv
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Block-wise quantization for INT8
// Each block of 128 elements shares one scale factor
constexpr int QUANT_BLOCK_SIZE = 128;

/**
 * Quantize FP32/FP16 values to INT8 with per-block scaling
 * Optimized for Blackwell architecture with warp-level operations
 */
__global__ void quantizeToINT8Kernel(const float* __restrict__ input,
                                     int8_t* __restrict__ output,
                                     float* __restrict__ scales,
                                     size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = tid / QUANT_BLOCK_SIZE;
    int local_tid = threadIdx.x % QUANT_BLOCK_SIZE;

    if (tid >= size) return;

    // Compute scale factor for this block (first thread in block)
    __shared__ float block_scale;
    if (local_tid == 0) {
        float max_val = 0.0f;
        for (int i = 0; i < QUANT_BLOCK_SIZE && block_id * QUANT_BLOCK_SIZE + i < size; ++i) {
            float val = fabsf(input[block_id * QUANT_BLOCK_SIZE + i]);
            max_val = fmaxf(max_val, val);
        }
        block_scale = max_val / 127.0f;  // INT8 range: -127 to 127
        if (block_scale == 0.0f) block_scale = 1.0f;  // Avoid division by zero
        scales[block_id] = block_scale;
    }
    __syncthreads();

    // Quantize the value
    float value = input[tid];
    int8_t quantized = static_cast<int8_t>(roundf(value / block_scale));
    output[tid] = quantized;
}

/**
 * Dequantize INT8 values back to FP32/FP16
 */
__global__ void dequantizeFromINT8Kernel(const int8_t* __restrict__ input,
                                         const float* __restrict__ scales,
                                         float* __restrict__ output,
                                         size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = tid / QUANT_BLOCK_SIZE;

    if (tid >= size) return;

    float scale = scales[block_id];
    int8_t quantized = input[tid];
    output[tid] = static_cast<float>(quantized) * scale;
}

/**
 * Quantize FP32/FP16 values to INT4 (4-bit) with per-block scaling
 * Two INT4 values packed into one byte
 */
__global__ void quantizeToINT4Kernel(const float* __restrict__ input,
                                     int8_t* __restrict__ output,
                                     float* __restrict__ scales,
                                     size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = tid / QUANT_BLOCK_SIZE;
    int local_tid = threadIdx.x % QUANT_BLOCK_SIZE;

    // Each thread processes 2 elements (packed into 1 byte)
    int idx = tid * 2;
    if (idx >= size) return;

    // Compute scale factor for this block
    __shared__ float block_scale;
    if (local_tid == 0) {
        float max_val = 0.0f;
        for (int i = 0; i < QUANT_BLOCK_SIZE && block_id * QUANT_BLOCK_SIZE + i < size; ++i) {
            float val = fabsf(input[block_id * QUANT_BLOCK_SIZE + i]);
            max_val = fmaxf(max_val, val);
        }
        block_scale = max_val / 7.0f;  // INT4 range: -7 to 7
        if (block_scale == 0.0f) block_scale = 1.0f;
        scales[block_id] = block_scale;
    }
    __syncthreads();

    // Quantize two values and pack into one byte
    float val1 = input[idx];
    float val2 = (idx + 1 < size) ? input[idx + 1] : 0.0f;

    int8_t quant1 = static_cast<int8_t>(roundf(val1 / block_scale));
    int8_t quant2 = static_cast<int8_t>(roundf(val2 / block_scale));

    // Clamp to 4-bit range: -7 to 7
    quant1 = max(min(quant1, (int8_t)7), (int8_t)-7);
    quant2 = max(min(quant2, (int8_t)7), (int8_t)-7);

    // Pack two 4-bit values into one byte
    int8_t packed = ((quant1 & 0x0F) << 4) | (quant2 & 0x0F);
    output[tid] = packed;
}

/**
 * Dequantize INT4 values back to FP32/FP16
 */
__global__ void dequantizeFromINT4Kernel(const int8_t* __restrict__ input,
                                         const float* __restrict__ scales,
                                         float* __restrict__ output,
                                         size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = tid / QUANT_BLOCK_SIZE;

    int idx = tid * 2;
    if (idx >= size) return;

    float scale = scales[block_id];
    int8_t packed = input[tid];

    // Unpack two 4-bit values
    int8_t quant1 = (packed >> 4) & 0x0F;
    int8_t quant2 = packed & 0x0F;

    // Sign extend from 4-bit to 8-bit
    if (quant1 & 0x08) quant1 |= 0xF0;
    if (quant2 & 0x08) quant2 |= 0xF0;

    output[idx] = static_cast<float>(quant1) * scale;
    if (idx + 1 < size) {
        output[idx + 1] = static_cast<float>(quant2) * scale;
    }
}

// C++ wrapper functions
extern "C" {

void launchQuantizeINT8(const float* input, int8_t* output, float* scales, size_t size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    quantizeToINT8Kernel<<<blocks, threads, 0, stream>>>(input, output, scales, size);
}

void launchDequantizeINT8(const int8_t* input, const float* scales, float* output, size_t size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dequantizeFromINT8Kernel<<<blocks, threads, 0, stream>>>(input, scales, output, size);
}

void launchQuantizeINT4(const float* input, int8_t* output, float* scales, size_t size, cudaStream_t stream) {
    int threads = 256;
    int blocks = ((size / 2) + threads - 1) / threads;
    quantizeToINT4Kernel<<<blocks, threads, 0, stream>>>(input, output, scales, size);
}

void launchDequantizeINT4(const int8_t* input, const float* scales, float* output, size_t size, cudaStream_t stream) {
    int threads = 256;
    int blocks = ((size / 2) + threads - 1) / threads;
    dequantizeFromINT4Kernel<<<blocks, threads, 0, stream>>>(input, scales, output, size);
}

} // extern "C"
