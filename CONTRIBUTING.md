# Contributing to OSS_SRV

First off, thank you for considering contributing to OSS_SRV! It's people like you that make OSS_SRV such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include GPU info, CUDA version, and TensorRT version**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and explain which behavior you expected to see instead**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Follow the C++17 style guide
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Development Process

### Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/oss_srv.git
cd oss_srv

# Install dependencies
sudo apt-get install cmake g++ cuda-toolkit-12-8

# Set up TensorRT
export TENSORRT_ROOT=/path/to/tensorrt

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### Coding Guidelines

#### C++ Style

* Use C++17 features
* Follow Google C++ Style Guide (mostly)
* Use meaningful variable and function names
* Comment complex logic
* Use RAII for resource management
* Prefer `std::unique_ptr` and `std::shared_ptr` over raw pointers

**Example:**

```cpp
// Good
class MemoryManager {
public:
    bool initialize(size_t size) {
        buffer_ = std::make_unique<float[]>(size);
        return buffer_ != nullptr;
    }

private:
    std::unique_ptr<float[]> buffer_;
};

// Bad
class MemoryManager {
public:
    bool initialize(size_t size) {
        buffer = new float[size];  // Manual memory management
        return buffer != NULL;  // Use nullptr in C++
    }

private:
    float* buffer;
};
```

#### CUDA Style

* Use `__restrict__` for kernel parameters when appropriate
* Prefer warp-level primitives for Blackwell optimization
* Always check CUDA errors
* Use async operations with streams

**Example:**

```cuda
__global__ void optimizedKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        output[tid] = input[tid] * 2.0f;
    }
}

// Check for errors
cudaError_t status = cudaGetLastError();
if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
}
```

#### Python Style

* Follow PEP 8
* Use type hints
* Write docstrings for all public functions
* Use `black` for formatting

### Testing

* Write unit tests for new features
* Ensure all tests pass before submitting PR
* Add integration tests for complex features

```bash
# Run C++ tests (when available)
cd build
ctest

# Run Python tests
cd python
pytest tests/
```

### Benchmarking

If your change affects performance:

* Run benchmarks before and after
* Include results in PR description
* Use the provided benchmark suite

```bash
python benchmarks/benchmark.py --model model.engine --compare-all
```

## Areas We Need Help

### High Priority

* 🐍 **Python bindings with pybind11**
* 🌐 **HTTP/gRPC server implementation**
* 📊 **Prometheus metrics exporter**
* 🧪 **Comprehensive test suite**

### Medium Priority

* 🔬 **Benchmark comparisons with vLLM, TRT-LLM**
* 📝 **Documentation improvements**
* 🎨 **Example applications**
* 🐳 **Kubernetes deployment guide**

### Low Priority

* 🌍 **Internationalization**
* 🎨 **Web UI for inference**
* 📱 **Mobile deployment (Jetson)**

## Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

**Examples:**

```
feat: Add INT4 KV cache quantization

Implements block-wise INT4 quantization for KV cache with
minimal quality degradation (<1% PPL increase).

Closes #123
```

```
fix: Resolve memory leak in CUDA graphs

Fixed issue where graph instances weren't properly destroyed
on cleanup, causing GPU memory leak.

Fixes #456
```

## Pull Request Process

1. **Update the README.md** with details of changes if applicable
2. **Update the documentation** in the `docs/` folder
3. **Add tests** for your changes
4. **Ensure the test suite passes**
5. **Update the CHANGELOG.md** (if applicable)
6. **Request review** from maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change needed? What problem does it solve?

## How Has This Been Tested?
Describe the tests you ran

## Types of changes
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)

## Checklist:
- [ ] My code follows the code style of this project
- [ ] I have updated the documentation accordingly
- [ ] I have added tests to cover my changes
- [ ] All new and existing tests passed
```

## Community

* **GitHub Discussions**: Ask questions, share ideas
* **Discord** (coming soon): Real-time chat with the community
* **Twitter**: [@oss_srv](https://twitter.com/oss_srv) for updates

## Recognition

Contributors will be:
* Listed in the README
* Mentioned in release notes
* Invited to the contributor Discord channel (coming soon)

## Questions?

Don't hesitate to ask! Open an issue or reach out to the maintainers.

---

Thank you for contributing to OSS_SRV! 🚀
