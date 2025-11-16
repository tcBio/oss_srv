/**
 * @file streaming_server.hpp
 * @brief SSE and WebSocket streaming server for real-time inference
 *
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized
 * for NVIDIA Blackwell GPU architecture.
 *
 * Provides Server-Sent Events (SSE) and WebSocket support for streaming
 * token-by-token inference responses to clients.
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

#include <string>
#include <sstream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "engine_core.hpp"

namespace oss_srv {

/**
 * SSE (Server-Sent Events) message builder
 */
class SSEMessage {
public:
    SSEMessage& event(const std::string& event_type) {
        event_ = event_type;
        return *this;
    }

    SSEMessage& data(const std::string& data) {
        data_ = data;
        return *this;
    }

    SSEMessage& id(const std::string& id) {
        id_ = id;
        return *this;
    }

    std::string build() const {
        std::ostringstream oss;
        if (!event_.empty()) {
            oss << "event: " << event_ << "\n";
        }
        if (!id_.empty()) {
            oss << "id: " << id_ << "\n";
        }
        if (!data_.empty()) {
            oss << "data: " << data_ << "\n";
        }
        oss << "\n";
        return oss.str();
    }

private:
    std::string event_;
    std::string data_;
    std::string id_;
};

/**
 * Stream manager for handling concurrent streaming requests
 */
class StreamManager {
public:
    struct StreamToken {
        int32_t token_id;
        std::string token_text;
        bool is_final;
        std::string request_id;
    };

    StreamManager() : stop_flag_(false) {}

    /**
     * Create a new stream for a request
     */
    std::string createStream(const std::string& request_id) {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        active_streams_[request_id] = std::queue<StreamToken>();
        return request_id;
    }

    /**
     * Push a token to a stream
     */
    void pushToken(const std::string& request_id, int32_t token_id,
                   const std::string& token_text, bool is_final) {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        auto it = active_streams_.find(request_id);
        if (it != active_streams_.end()) {
            it->second.push({token_id, token_text, is_final, request_id});
            stream_condition_.notify_all();
        }
    }

    /**
     * Pop next token from a stream (blocking)
     */
    bool popToken(const std::string& request_id, StreamToken& token,
                  std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(streams_mutex_);

        auto deadline = std::chrono::steady_clock::now() + timeout;

        while (true) {
            auto it = active_streams_.find(request_id);
            if (it == active_streams_.end()) {
                return false;  // Stream not found
            }

            if (!it->second.empty()) {
                token = it->second.front();
                it->second.pop();
                return true;
            }

            if (stop_flag_) {
                return false;
            }

            // Wait for token or timeout
            if (stream_condition_.wait_until(lock, deadline) == std::cv_status::timeout) {
                return false;
            }
        }
    }

    /**
     * Close a stream
     */
    void closeStream(const std::string& request_id) {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        active_streams_.erase(request_id);
    }

    /**
     * Get number of active streams
     */
    size_t getActiveStreamCount() const {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        return active_streams_.size();
    }

    /**
     * Shutdown all streams
     */
    void shutdown() {
        stop_flag_ = true;
        stream_condition_.notify_all();
    }

private:
    mutable std::mutex streams_mutex_;
    std::condition_variable stream_condition_;
    std::map<std::string, std::queue<StreamToken>> active_streams_;
    std::atomic<bool> stop_flag_;
};

/**
 * Streaming helper functions
 */
class StreamingHelper {
public:
    /**
     * Convert token to SSE message
     */
    static std::string tokenToSSE(int32_t token_id, const std::string& token_text, bool is_final) {
        SSEMessage msg;

        if (is_final) {
            msg.event("done").data("{\"finish_reason\":\"stop\"}");
        } else {
            std::string json = "{\"token_id\":" + std::to_string(token_id) +
                             ",\"text\":\"" + escapeJSON(token_text) + "\"}";
            msg.event("token").data(json);
        }

        return msg.build();
    }

    /**
     * Create streaming callback for SSE
     */
    static StreamCallback createSSECallback(StreamManager& manager, const std::string& request_id) {
        return [&manager, request_id](int32_t token_id, const std::string& token_text, bool is_final) {
            manager.pushToken(request_id, token_id, token_text, is_final);
        };
    }

private:
    static std::string escapeJSON(const std::string& str) {
        std::ostringstream oss;
        for (char c : str) {
            switch (c) {
                case '"':  oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default:   oss << c; break;
            }
        }
        return oss.str();
    }
};

} // namespace oss_srv
