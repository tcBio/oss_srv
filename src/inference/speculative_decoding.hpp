/**
 * @file speculative_decoding.hpp
 * @brief Speculative decoding for 2-3x inference speedup
 *
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized
 * for NVIDIA Blackwell GPU architecture.
 *
 * Speculative decoding uses a small "draft" model to generate candidate tokens
 * quickly, then verifies them with the larger "target" model in parallel.
 * This can achieve 2-3x speedup with no quality degradation.
 *
 * Key idea:
 * 1. Draft model generates K candidate tokens (fast, low quality)
 * 2. Target model verifies all K tokens in parallel (1 forward pass)
 * 3. Accept tokens until first mismatch
 * 4. Effective speedup: (K * accept_rate) / (1 + K_draft_cost)
 *
 * @author Brian Worthington
 * @date 2025
 * @version 1.0
 *
 * @copyright MIT License
 *
 * Repository: https://github.com/tcBio/oss_srv
 */

#pragma once

#include <vector>
#include <memory>
#include <string>
#include "engine_core.hpp"

/**
 * Configuration for speculative decoding
 */
struct SpeculativeConfig {
    // Draft model path (smaller, faster model)
    std::string draft_model_path;

    // Number of lookahead tokens to generate
    int lookahead_tokens = 4;  // Typically 4-8 works well

    // Accept threshold for draft tokens (0.0-1.0)
    // Lower = more conservative (higher quality)
    // Higher = more aggressive (higher speedup)
    float acceptance_threshold = 0.6f;

    // Maximum draft failures before fallback to standard decoding
    int max_draft_failures = 3;

    // Enable debug logging
    bool debug = false;
};

/**
 * Statistics for speculative decoding performance
 */
struct SpeculativeStats {
    int total_tokens_generated = 0;
    int total_draft_tokens = 0;
    int total_accepted_tokens = 0;
    int total_rejected_tokens = 0;
    int draft_model_calls = 0;
    int target_model_calls = 0;

    float getAcceptanceRate() const {
        return total_draft_tokens > 0
            ? static_cast<float>(total_accepted_tokens) / total_draft_tokens
            : 0.0f;
    }

    float getSpeedup() const {
        // Effective speedup calculation
        // Assumes draft model is ~5x faster than target model
        if (target_model_calls == 0) return 1.0f;

        float draft_cost = draft_model_calls * 0.2f;  // Draft is 5x faster
        float target_cost = target_model_calls * 1.0f;
        float baseline_cost = total_tokens_generated * 1.0f;

        return baseline_cost / (draft_cost + target_cost);
    }

    void reset() {
        total_tokens_generated = 0;
        total_draft_tokens = 0;
        total_accepted_tokens = 0;
        total_rejected_tokens = 0;
        draft_model_calls = 0;
        target_model_calls = 0;
    }
};

/**
 * Speculative decoding engine
 */
class SpeculativeDecoder {
public:
    SpeculativeDecoder() = default;
    ~SpeculativeDecoder() = default;

    /**
     * Initialize with draft and target models
     */
    bool initialize(const EngineConfig& target_config,
                   const SpeculativeConfig& spec_config);

    /**
     * Execute inference with speculative decoding
     */
    InferenceResult executeInference(const InferenceRequest& request);

    /**
     * Get performance statistics
     */
    const SpeculativeStats& getStats() const { return stats_; }

    /**
     * Reset statistics
     */
    void resetStats() { stats_.reset(); }

    /**
     * Shutdown and cleanup
     */
    void shutdown();

private:
    /**
     * Generate candidate tokens using draft model
     */
    std::vector<int32_t> generateDraftTokens(
        const std::vector<int32_t>& context,
        int num_tokens
    );

    /**
     * Verify draft tokens using target model
     */
    std::vector<bool> verifyDraftTokens(
        const std::vector<int32_t>& context,
        const std::vector<int32_t>& draft_tokens
    );

    /**
     * Sample from logits with acceptance check
     */
    int32_t sampleWithAcceptance(
        const std::vector<float>& target_logits,
        const std::vector<float>& draft_logits,
        int32_t draft_token,
        float threshold
    );

    /**
     * Calculate token probability from logits
     */
    float getTokenProbability(const std::vector<float>& logits, int32_t token_id);

    /**
     * Apply softmax to logits
     */
    std::vector<float> softmax(const std::vector<float>& logits);

    // Components
    std::unique_ptr<EngineCore> draft_engine_;   // Small, fast model
    std::unique_ptr<EngineCore> target_engine_;  // Large, accurate model

    SpeculativeConfig config_;
    SpeculativeStats stats_;
    bool initialized_ = false;
};

/**
 * Helper functions for speculative decoding
 */
namespace SpeculativeHelper {

/**
 * Estimate optimal lookahead tokens based on model sizes
 */
inline int estimateOptimalLookahead(size_t draft_params, size_t target_params) {
    // Heuristic: lookahead ~ sqrt(target_params / draft_params)
    // Clamped to reasonable range [2, 8]
    if (draft_params == 0) return 4;  // Default

    float ratio = static_cast<float>(target_params) / draft_params;
    int lookahead = static_cast<int>(std::sqrt(ratio));

    return std::max(2, std::min(8, lookahead));
}

/**
 * Calculate expected speedup given acceptance rate
 */
inline float calculateExpectedSpeedup(float acceptance_rate, int lookahead,
                                     float draft_speedup = 5.0f) {
    // Effective tokens per iteration
    float effective_tokens = lookahead * acceptance_rate;

    // Cost per iteration
    float cost = lookahead / draft_speedup + 1.0f;  // draft + verify

    // Speedup vs baseline (1 token per iteration)
    return effective_tokens / cost;
}

/**
 * Recommend configuration based on model characteristics
 */
inline SpeculativeConfig recommendConfig(
    const std::string& draft_model_path,
    size_t draft_params_billions,
    size_t target_params_billions
) {
    SpeculativeConfig config;
    config.draft_model_path = draft_model_path;

    // Adjust lookahead based on model size ratio
    config.lookahead_tokens = estimateOptimalLookahead(
        draft_params_billions, target_params_billions
    );

    // Conservative threshold for production
    config.acceptance_threshold = 0.6f;

    // Increase failures tolerance for larger ratios
    config.max_draft_failures = (target_params_billions / draft_params_billions) * 2;

    return config;
}

} // namespace SpeculativeHelper
