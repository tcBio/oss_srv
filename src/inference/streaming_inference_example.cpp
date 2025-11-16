/**
 * @file streaming_inference_example.cpp
 * @brief Example of streaming inference with SSE-style callbacks
 *
 * OSS-20B TensorRT Inference Server
 * Demonstrates how to use the streaming API with callbacks
 *
 * @author Brian Worthington
 * @date 2025
 */

#include "engine_core.hpp"
#include "streaming_server.hpp"
#include <iostream>
#include <chrono>
#include <thread>

using namespace oss_srv;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.engine> <prompt>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];

    // Initialize engine
    EngineConfig config;
    config.model_path = model_path;
    config.max_batch_size = 1;
    config.max_sequence_length = 2048;
    config.precision = "fp16";
    config.use_cuda_graph = true;

    EngineCore engine;
    if (!engine.initialize(config)) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }

    std::cout << "Engine initialized successfully" << std::endl;
    std::cout << "\nPrompt: " << prompt << std::endl;
    std::cout << "Streaming response:" << std::endl;
    std::cout << "─────────────────────────────────────" << std::endl;

    // Create stream manager
    StreamManager stream_manager;
    std::string request_id = "req_001";

    stream_manager.createStream(request_id);

    // Create streaming request
    InferenceRequest request;
    request.prompt = prompt;
    request.max_tokens = 100;
    request.temperature = 0.7f;
    request.top_p = 0.9f;
    request.stream = true;
    request.request_id = request_id;

    // Set up streaming callback
    request.stream_callback = StreamingHelper::createSSECallback(stream_manager, request_id);

    // Run inference in a separate thread
    std::thread inference_thread([&engine, &request]() {
        auto result = engine.executeInference(request);
        if (!result.success) {
            std::cerr << "\nError: " << result.error_message << std::endl;
        }
    });

    // Read and display streamed tokens
    int token_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    bool first_token = true;
    float ttft_ms = 0.0f;

    while (true) {
        StreamManager::StreamToken token;
        if (stream_manager.popToken(request_id, token, std::chrono::milliseconds(5000))) {

            if (first_token) {
                auto first_token_time = std::chrono::high_resolution_clock::now();
                auto ttft = std::chrono::duration_cast<std::chrono::milliseconds>(
                    first_token_time - start_time);
                ttft_ms = static_cast<float>(ttft.count());
                first_token = false;
            }

            // Print token as it arrives (no newline for streaming effect)
            std::cout << token.token_text << std::flush;
            token_count++;

            if (token.is_final) {
                break;
            }
        } else {
            // Timeout or stream closed
            break;
        }
    }

    // Wait for inference thread to complete
    inference_thread.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print metrics
    std::cout << "\n─────────────────────────────────────" << std::endl;
    std::cout << "\n📊 Streaming Metrics:" << std::endl;
    std::cout << "  Tokens generated: " << token_count << std::endl;
    std::cout << "  Time to first token (TTFT): " << ttft_ms << " ms" << std::endl;
    std::cout << "  Total time: " << total_time.count() << " ms" << std::endl;
    if (token_count > 0 && total_time.count() > 0) {
        float throughput = (token_count * 1000.0f) / total_time.count();
        std::cout << "  Throughput: " << throughput << " tokens/sec" << std::endl;
    }

    stream_manager.closeStream(request_id);
    engine.shutdown();

    return 0;
}
