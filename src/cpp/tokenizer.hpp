/**
 * @file tokenizer.hpp
 * @brief Text tokenization and vocabulary management
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Tokenizer interface for text preprocessing and token ID conversion for model input.
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
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    bool loadFromDirectory(const std::string& tokenizer_dir);
    
    std::vector<int32_t> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int32_t>& tokens);
    
    int32_t getBOSTokenId() const { return bos_token_id_; }
    int32_t getEOSTokenId() const { return eos_token_id_; }
    int32_t getPADTokenId() const { return pad_token_id_; }
    
    size_t getVocabSize() const { return vocab_size_; }
    
    // Test tokenizer with known test cases
    bool runTests();

private:
    std::unordered_map<std::string, int32_t> str_to_token_;
    std::unordered_map<int32_t, std::string> token_to_str_;
    
    int32_t bos_token_id_ = 199998;
    int32_t eos_token_id_ = 200002;
    int32_t pad_token_id_ = 199999;
    size_t vocab_size_ = 199998;
    
public:
    bool loadVocabulary(const std::string& vocab_file);
    bool loadSpecialTokens(const std::string& special_tokens_file);

private:
    
    // Simple word-based tokenization 
    std::vector<std::string> tokenizeToWords(const std::string& text);
};