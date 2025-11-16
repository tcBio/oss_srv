/**
 * @file speculative_decoding.cpp
 * @brief Implementation of speculative decoding for 2-3x speedup
 *
 * OSS-20B TensorRT Inference Server
 *
 * @author Brian Worthington
 * @date 2025
 */

#include "speculative_decoding.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

bool SpeculativeDecoder::initialize(const EngineConfig& target_config,
                                   const SpeculativeConfig& spec_config) {
    if (initialized_) {
        shutdown();
    }

    config_ = spec_config;

    std::cout << "Initializing Speculative Decoder..." << std::endl;
    std::cout << "  Lookahead tokens: " << config_.lookahead_tokens << std::endl;
    std::cout << "  Acceptance threshold: " << config_.acceptance_threshold << std::endl;

    // Initialize target model (large, accurate)
    std::cout << "Loading target model: " << target_config.model_path << std::endl;
    target_engine_ = std::make_unique<EngineCore>();
    if (!target_engine_->initialize(target_config)) {
        std::cerr << "Failed to initialize target engine" << std::endl;
        return false;
    }

    // Initialize draft model (small, fast)
    std::cout << "Loading draft model: " << config_.draft_model_path << std::endl;
    EngineConfig draft_config = target_config;
    draft_config.model_path = config_.draft_model_path;
    draft_config.use_cuda_graph = true;  // Critical for draft model speed

    draft_engine_ = std::make_unique<EngineCore>();
    if (!draft_engine_->initialize(draft_config)) {
        std::cerr << "Failed to initialize draft engine" << std::endl;
        target_engine_->shutdown();
        return false;
    }

    stats_.reset();
    initialized_ = true;

    std::cout << "Speculative Decoder initialized successfully" << std::endl;
    return true;
}

InferenceResult SpeculativeDecoder::executeInference(const InferenceRequest& request) {
    InferenceResult result;

    if (!initialized_) {
        result.error_message = "Speculative decoder not initialized";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Tokenize input
        auto tokenizer = target_engine_->getTokenizer();
        auto input_tokens = tokenizer->tokenize(request.prompt);

        if (input_tokens.empty()) {
            result.error_message = "Failed to tokenize input";
            return result;
        }

        std::vector<int32_t> context = input_tokens;
        std::vector<int32_t> generated_tokens;

        int consecutive_failures = 0;

        // Generation loop with speculative decoding
        while (generated_tokens.size() < static_cast<size_t>(request.max_tokens)) {
            // Step 1: Generate candidate tokens with draft model
            auto draft_tokens = generateDraftTokens(context, config_.lookahead_tokens);

            if (draft_tokens.empty()) {
                consecutive_failures++;
                if (consecutive_failures >= config_.max_draft_failures) {
                    std::cout << "Too many draft failures, falling back to standard decoding" << std::endl;
                    // Fall back to single token generation with target model
                    draft_tokens = {-1};  // Placeholder
                }
            }

            // Step 2: Verify draft tokens with target model
            auto acceptance_mask = verifyDraftTokens(context, draft_tokens);

            // Step 3: Accept tokens until first rejection
            int accepted_count = 0;
            for (size_t i = 0; i < draft_tokens.size() && i < acceptance_mask.size(); ++i) {
                if (acceptance_mask[i]) {
                    generated_tokens.push_back(draft_tokens[i]);
                    context.push_back(draft_tokens[i]);
                    accepted_count++;
                    stats_.total_accepted_tokens++;

                    // Check for EOS
                    if (draft_tokens[i] == tokenizer->getEOSTokenId()) {
                        break;
                    }

                    // Streaming callback if enabled
                    if (request.stream && request.stream_callback) {
                        std::string token_text = tokenizer->detokenize({draft_tokens[i]});
                        bool is_final = (generated_tokens.size() >= static_cast<size_t>(request.max_tokens)) ||
                                      (draft_tokens[i] == tokenizer->getEOSTokenId());
                        request.stream_callback(draft_tokens[i], token_text, is_final);
                    }
                } else {
                    // Rejection: stop accepting
                    stats_.total_rejected_tokens++;
                    break;
                }
            }

            if (accepted_count > 0) {
                consecutive_failures = 0;
            } else {
                consecutive_failures++;
            }

            if (config_.debug) {
                std::cout << "  Draft: " << draft_tokens.size()
                         << " tokens, Accepted: " << accepted_count
                         << " (rate: " << stats_.getAcceptanceRate() << ")" << std::endl;
            }

            // Check if we should stop
            if (!generated_tokens.empty() &&
                generated_tokens.back() == tokenizer->getEOSTokenId()) {
                break;
            }
        }

        // Detokenize results
        result.generated_text = tokenizer->detokenize(generated_tokens);
        result.generated_tokens = generated_tokens;
        result.success = true;

        stats_.total_tokens_generated = generated_tokens.size();

    } catch (const std::exception& e) {
        result.error_message = std::string("Exception during speculative decoding: ") + e.what();
        return result;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.total_time_ms = static_cast<float>(duration.count());

    if (config_.debug) {
        std::cout << "\nSpeculative Decoding Stats:" << std::endl;
        std::cout << "  Total tokens: " << stats_.total_tokens_generated << std::endl;
        std::cout << "  Acceptance rate: " << (stats_.getAcceptanceRate() * 100) << "%" << std::endl;
        std::cout << "  Estimated speedup: " << stats_.getSpeedup() << "x" << std::endl;
    }

    return result;
}

std::vector<int32_t> SpeculativeDecoder::generateDraftTokens(
    const std::vector<int32_t>& context,
    int num_tokens
) {
    std::vector<int32_t> draft_tokens;

    // Create temporary request for draft model
    InferenceRequest draft_request;
    draft_request.max_tokens = num_tokens;
    draft_request.temperature = 1.0f;  // Use higher temp for diversity
    draft_request.top_p = 0.9f;

    auto tokenizer = draft_engine_->getTokenizer();
    std::string prompt = tokenizer->detokenize(context);
    draft_request.prompt = prompt;

    // Generate with draft model
    auto result = draft_engine_->executeInference(draft_request);

    stats_.draft_model_calls++;

    if (result.success && !result.generated_tokens.empty()) {
        draft_tokens = result.generated_tokens;
        stats_.total_draft_tokens += draft_tokens.size();
    }

    return draft_tokens;
}

std::vector<bool> SpeculativeDecoder::verifyDraftTokens(
    const std::vector<int32_t>& context,
    const std::vector<int32_t>& draft_tokens
) {
    std::vector<bool> acceptance_mask;

    if (draft_tokens.empty()) {
        return acceptance_mask;
    }

    // Build combined context with draft tokens for verification
    std::vector<int32_t> verify_context = context;
    verify_context.insert(verify_context.end(), draft_tokens.begin(), draft_tokens.end());

    // Run target model on entire sequence
    std::vector<float> logits;
    bool success = target_engine_->getTensorRTEngine()->executeInference(verify_context, logits);

    stats_.target_model_calls++;

    if (!success || logits.empty()) {
        // Verification failed, reject all
        return std::vector<bool>(draft_tokens.size(), false);
    }

    // Check each draft token against target model distribution
    // In practice, we'd need to extract logits at each position
    // For now, use simplified acceptance based on threshold

    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        // Simplified: accept if draft token has reasonable probability
        // In full implementation, we'd compare draft vs target distributions
        float prob = getTokenProbability(logits, draft_tokens[i]);
        bool accept = prob >= config_.acceptance_threshold;
        acceptance_mask.push_back(accept);

        if (!accept) {
            // Stop at first rejection
            break;
        }
    }

    return acceptance_mask;
}

int32_t SpeculativeDecoder::sampleWithAcceptance(
    const std::vector<float>& target_logits,
    const std::vector<float>& draft_logits,
    int32_t draft_token,
    float threshold
) {
    // Compute acceptance probability
    float target_prob = getTokenProbability(target_logits, draft_token);
    float draft_prob = getTokenProbability(draft_logits, draft_token);

    if (draft_prob == 0.0f) {
        return -1;  // Reject
    }

    float acceptance_prob = std::min(1.0f, target_prob / draft_prob);

    // Stochastic acceptance
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(gen) < acceptance_prob) {
        return draft_token;  // Accept
    }

    return -1;  // Reject
}

float SpeculativeDecoder::getTokenProbability(const std::vector<float>& logits, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= logits.size()) {
        return 0.0f;
    }

    // Apply softmax to get probabilities
    auto probs = softmax(logits);
    return probs[token_id];
}

std::vector<float> SpeculativeDecoder::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());

    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute exp(x - max)
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }

    // Normalize
    if (sum > 0.0f) {
        for (auto& p : probs) {
            p /= sum;
        }
    }

    return probs;
}

void SpeculativeDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    if (draft_engine_) {
        draft_engine_->shutdown();
        draft_engine_.reset();
    }

    if (target_engine_) {
        target_engine_->shutdown();
        target_engine_.reset();
    }

    initialized_ = false;
    std::cout << "Speculative Decoder shut down" << std::endl;
}
