/**
 * @file simple_json.hpp
 * @brief Lightweight JSON parsing utilities
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Simple JSON parser for configuration and data serialization without external dependencies.
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
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

class SimpleJSON {
public:
    static std::unordered_map<std::string, std::string> parseFile(const std::string& filename) {
        std::unordered_map<std::string, std::string> result;
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "SimpleJSON: Cannot open file: " << filename << std::endl;
            return result;
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::cout << "SimpleJSON: Loading file of size " << file_size << " bytes" << std::endl;
        
        // Read entire file into string
        std::string content;
        content.reserve(file_size);
        content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        std::cout << "SimpleJSON: File loaded, content length: " << content.length() << std::endl;
        
        // Parse JSON with better error handling
        size_t pos = 0;
        int entries_parsed = 0;
        
        while (pos < content.length() && entries_parsed < 1000000) { // Safety limit
            // Find opening quote for key
            size_t key_start = content.find('"', pos);
            if (key_start == std::string::npos) break;
            key_start++;
            
            // Find closing quote for key
            size_t key_end = content.find('"', key_start);
            if (key_end == std::string::npos) break;
            
            std::string key = content.substr(key_start, key_end - key_start);
            
            // Find colon
            size_t colon = content.find(':', key_end);
            if (colon == std::string::npos) break;
            colon++;
            
            // Skip whitespace
            while (colon < content.length() && std::isspace(content[colon])) {
                colon++;
            }
            
            std::string value;
            if (content[colon] == '"') {
                // String value
                colon++;
                size_t value_end = content.find('"', colon);
                if (value_end == std::string::npos) break;
                value = content.substr(colon, value_end - colon);
                pos = value_end + 1;
            } else {
                // Number value
                size_t value_end = colon;
                while (value_end < content.length() && 
                       (std::isdigit(content[value_end]) || content[value_end] == '-' || content[value_end] == '.')) {
                    value_end++;
                }
                value = content.substr(colon, value_end - colon);
                pos = value_end;
            }
            
            result[key] = value;
            entries_parsed++;
            
            // Progress indicator for large files
            if (entries_parsed % 10000 == 0) {
                std::cout << "SimpleJSON: Parsed " << entries_parsed << " entries..." << std::endl;
            }
        }
        
        std::cout << "SimpleJSON: Completed parsing, " << result.size() << " entries loaded" << std::endl;
        return result;
    }
    
    static int getInt(const std::unordered_map<std::string, std::string>& json, const std::string& key, int defaultValue = 0) {
        auto it = json.find(key);
        if (it != json.end() && !it->second.empty()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }
    
    static std::string getString(const std::unordered_map<std::string, std::string>& json, const std::string& key, const std::string& defaultValue = "") {
        auto it = json.find(key);
        if (it != json.end()) {
            return it->second;
        }
        return defaultValue;
    }
};