/**
 * @file complete_inference.cpp
 * @brief Main inference executable and CLI interface
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Command-line interface for TensorRT inference with argument parsing and output formatting.
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

#include "engine_core.hpp"
#include "request_processor.hpp"
#include "../cpp/tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

int main(int argc, char* argv[]) {
    std::cout << "Starting complete_inference..." << std::endl;
    std::cout.flush();
    
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <prompt> <max_tokens> <temperature> <top_p>" << std::endl;
        return 1;
    }
    
    std::cout << "Arguments parsed successfully" << std::endl;
    std::cout.flush();
    
    std::string engine_path = argv[1];
    std::string prompt = argv[2];
    int max_tokens = std::stoi(argv[3]);
    float temperature = std::stof(argv[4]);
    float top_p = std::stof(argv[5]);
    
    std::cout << "=== OSS-20B Complete Inference Pipeline ===" << std::endl;
    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << "Top-p: " << top_p << std::endl;
    std::cout << std::endl;
    
    try {
        // Initialize engine configuration
        EngineConfig config;
        config.model_path = engine_path;
        config.max_batch_size = 1;
        config.max_sequence_length = 2048;
        config.precision = "fp16";
        config.use_cuda_graph = true;
        config.gpu_device_id = 0;
        
        std::cout << "1. Initializing OSS-20B TensorRT Engine Core..." << std::endl;
        EngineCore engine_core;
        if (!engine_core.initialize(config)) {
            std::cout << "GENERATED_TEXT: Failed to initialize engine core" << std::endl;
            std::cout << "TOKENS_GENERATED: 0" << std::endl;
            std::cout << "INFERENCE_TIME_MS: 0" << std::endl;
            std::cout << "THROUGHPUT_TOKENS_PER_SEC: 0" << std::endl;
            return 1;
        }
        
        std::cout << "2. Initializing Request Processor..." << std::endl;
        RequestProcessor processor(&engine_core);
        if (!processor.initialize()) {
            std::cout << "GENERATED_TEXT: Failed to initialize request processor" << std::endl;
            std::cout << "TOKENS_GENERATED: 0" << std::endl;
            std::cout << "INFERENCE_TIME_MS: 0" << std::endl;
            std::cout << "THROUGHPUT_TOKENS_PER_SEC: 0" << std::endl;
            return 1;
        }
        
        std::cout << "3. Creating inference request..." << std::endl;
        InferenceRequest request;
        request.prompt = prompt;
        request.max_tokens = max_tokens;
        request.temperature = temperature;
        request.top_p = top_p;
        request.stream = false;
        request.request_id = "complete_inference_001";
        
        std::cout << "4. Starting inference..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process the request
        auto future_result = processor.processRequest(request);
        auto result = future_result.get();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        if (result.success) {
            // Calculate metrics
            int tokens_generated = result.generated_tokens.size();
            double throughput = 0.0;
            if (result.total_time_ms > 0) {
                throughput = (tokens_generated * 1000.0) / result.total_time_ms;
            } else if (total_time_ms > 0) {
                throughput = (tokens_generated * 1000.0) / total_time_ms;
            } else {
                throughput = tokens_generated * 100.0; // Fallback for very fast execution
            }
            
            std::cout << "=== Inference Complete ===" << std::endl;
            std::cout << "Generated tokens: " << tokens_generated << std::endl;
            std::cout << "Inference time: " << result.total_time_ms << " ms" << std::endl;
            std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
            
            // Output results in the expected format
            std::cout << "GENERATED_TEXT: " << result.generated_text << std::endl;
            std::cout << "TOKENS_GENERATED: " << tokens_generated << std::endl;
            std::cout << "INFERENCE_TIME_MS: " << result.total_time_ms << std::endl;
            std::cout << "THROUGHPUT_TOKENS_PER_SEC: " << throughput << std::endl;
        } else {
            std::cout << "GENERATED_TEXT: Inference failed: " << result.error_message << std::endl;
            std::cout << "TOKENS_GENERATED: 0" << std::endl;
            std::cout << "INFERENCE_TIME_MS: " << total_time_ms << std::endl;
            std::cout << "THROUGHPUT_TOKENS_PER_SEC: 0" << std::endl;
        }
        
        // Shutdown components
        processor.shutdown();
        engine_core.shutdown();
        
        return result.success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cout << "GENERATED_TEXT: Exception during inference: " << e.what() << std::endl;
        std::cout << "TOKENS_GENERATED: 0" << std::endl;
        std::cout << "INFERENCE_TIME_MS: 0" << std::endl;
        std::cout << "THROUGHPUT_TOKENS_PER_SEC: 0" << std::endl;
        return 1;
    }
}
