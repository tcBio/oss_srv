/**
 * @file engine_diagnostic.cpp
 * @brief TensorRT engine diagnostic and validation tools
 * 
 * OSS-20B TensorRT Inference Server
 * High-performance inference server for OpenAI's GPT-OSS-20B model optimized 
 * for NVIDIA Blackwell GPU with Blackwell architecture.
 * 
 * Diagnostic utilities for TensorRT engine validation, performance analysis, and debugging.
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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime.h>

class TensorRTEngineDiagnostic {
private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    
public:
    TensorRTEngineDiagnostic() : runtime_(nullptr), engine_(nullptr) {}
    
    ~TensorRTEngineDiagnostic() {
        cleanup();
    }
    
    bool loadEngine(const std::string& engine_path) {
        // Create TensorRT logger
        class Logger : public nvinfer1::ILogger {
            void log(Severity severity, const char* msg) noexcept override {
                if (severity <= Severity::kWARNING) {
                    std::cout << "[TensorRT] " << msg << std::endl;
                }
            }
        } logger;
        
        // Create runtime
        runtime_ = nvinfer1::createInferRuntime(logger);
        if (!runtime_) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        // Read engine file
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open engine file: " << engine_path << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        // Deserialize engine
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }
        
        std::cout << "Successfully loaded TensorRT engine: " << engine_path << std::endl;
        return true;
    }
    
    void analyzeEngine() {
        if (!engine_) {
            std::cerr << "No engine loaded" << std::endl;
            return;
        }
        
        std::cout << "\n=== TensorRT Engine Analysis ===" << std::endl;
        
        // Basic engine info
        std::cout << "Engine Name: " << (engine_->getName() ? engine_->getName() : "Unknown") << std::endl;
        std::cout << "Number of Optimization Profiles: " << engine_->getNbOptimizationProfiles() << std::endl;
        std::cout << "Number of IO Tensors: " << engine_->getNbIOTensors() << std::endl;
        std::cout << "Device Memory Size: " << engine_->getDeviceMemorySize() << " bytes" << std::endl;
        std::cout << "Has Implicit Batch Dimension: " << (engine_->hasImplicitBatchDimension() ? "Yes" : "No") << std::endl;
        
        // Analyze IO tensors
        std::cout << "\n=== IO Tensor Analysis ===" << std::endl;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
            nvinfer1::DataType data_type = engine_->getTensorDataType(tensor_name);
            nvinfer1::Dims dims = engine_->getTensorShape(tensor_name);
            
            std::cout << "Tensor " << i << ": " << tensor_name << std::endl;
            std::cout << "  Mode: " << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << std::endl;
            std::cout << "  Data Type: " << static_cast<int>(data_type) << std::endl;
            std::cout << "  Shape: [";
            for (int j = 0; j < dims.nbDims; ++j) {
                std::cout << dims.d[j];
                if (j < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            // Check if tensor has dynamic dimensions
            bool has_dynamic = false;
            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    has_dynamic = true;
                    break;
                }
            }
            std::cout << "  Has Dynamic Dimensions: " << (has_dynamic ? "Yes" : "No") << std::endl;
            std::cout << std::endl;
        }
        
        // Analyze optimization profiles
        std::cout << "=== Optimization Profile Analysis ===" << std::endl;
        int num_profiles = engine_->getNbOptimizationProfiles();
        
        if (num_profiles == 0) {
            std::cout << "❌ NO OPTIMIZATION PROFILES FOUND" << std::endl;
            std::cout << "This engine does not support dynamic shapes!" << std::endl;
        } else {
            for (int profile_idx = 0; profile_idx < num_profiles; ++profile_idx) {
                std::cout << "Profile " << profile_idx << ":" << std::endl;
                
                for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
                    const char* tensor_name = engine_->getIOTensorName(i);
                    nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
                    
                    if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
                        nvinfer1::Dims min_dims = engine_->getProfileShape(tensor_name, profile_idx, nvinfer1::OptProfileSelector::kMIN);
                        nvinfer1::Dims opt_dims = engine_->getProfileShape(tensor_name, profile_idx, nvinfer1::OptProfileSelector::kOPT);
                        nvinfer1::Dims max_dims = engine_->getProfileShape(tensor_name, profile_idx, nvinfer1::OptProfileSelector::kMAX);
                        
                        std::cout << "  Input '" << tensor_name << "':" << std::endl;
                        std::cout << "    Min: [";
                        for (int j = 0; j < min_dims.nbDims; ++j) {
                            std::cout << min_dims.d[j];
                            if (j < min_dims.nbDims - 1) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                        
                        std::cout << "    Opt: [";
                        for (int j = 0; j < opt_dims.nbDims; ++j) {
                            std::cout << opt_dims.d[j];
                            if (j < opt_dims.nbDims - 1) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                        
                        std::cout << "    Max: [";
                        for (int j = 0; j < max_dims.nbDims; ++j) {
                            std::cout << max_dims.d[j];
                            if (j < max_dims.nbDims - 1) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                    }
                }
                std::cout << std::endl;
            }
        }
        
        // Compatibility check
        std::cout << "=== Compatibility Analysis ===" << std::endl;
        checkDynamicShapeCompatibility();
    }
    
private:
    void checkDynamicShapeCompatibility() {
        if (engine_->getNbOptimizationProfiles() == 0) {
            std::cout << "❌ CRITICAL: Engine has no optimization profiles" << std::endl;
            std::cout << "   This engine cannot support dynamic input shapes" << std::endl;
            std::cout << "   Recommendation: Rebuild engine with optimization profiles" << std::endl;
            return;
        }
        
        // Check if input dimensions support our target range (1-256 sequence length)
        bool supports_target_range = false;
        
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
            
            if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
                // Check profile 0 (the one we're trying to use)
                nvinfer1::Dims min_dims = engine_->getProfileShape(tensor_name, 0, nvinfer1::OptProfileSelector::kMIN);
                nvinfer1::Dims max_dims = engine_->getProfileShape(tensor_name, 0, nvinfer1::OptProfileSelector::kMAX);
                
                // Assume input format is [batch_size, sequence_length]
                if (min_dims.nbDims >= 2 && max_dims.nbDims >= 2) {
                    int min_seq_len = static_cast<int>(min_dims.d[1]);
                    int max_seq_len = static_cast<int>(max_dims.d[1]);
                    
                    std::cout << "Input '" << tensor_name << "' sequence length range: " 
                              << min_seq_len << " - " << max_seq_len << std::endl;
                    
                    if (min_seq_len <= 3 && max_seq_len >= 256) {
                        supports_target_range = true;
                        std::cout << "✅ COMPATIBLE: Supports our target range (1-256 tokens)" << std::endl;
                    } else {
                        std::cout << "❌ INCOMPATIBLE: Does not support our target range (1-256 tokens)" << std::endl;
                    }
                }
            }
        }
        
        if (!supports_target_range) {
            std::cout << "\n❌ RECOMMENDATION: Engine needs to be rebuilt with proper optimization profiles" << std::endl;
            std::cout << "   Required: min=[1, 1], opt=[1, 128], max=[1, 256] for sequence dimension" << std::endl;
        } else {
            std::cout << "\n✅ Engine appears compatible with dynamic shapes" << std::endl;
            std::cout << "   The runtime issue may be in the inference code implementation" << std::endl;
        }
    }
    
    void cleanup() {
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
        if (runtime_) {
            delete runtime_;
            runtime_ = nullptr;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_path>" << std::endl;
        return 1;
    }
    
    std::string engine_path = argv[1];
    
    TensorRTEngineDiagnostic diagnostic;
    
    if (!diagnostic.loadEngine(engine_path)) {
        return 1;
    }
    
    diagnostic.analyzeEngine();
    
    return 0;
}
