# OSS_SRV - Blackwell-Native Inference Server
# Optimized for NVIDIA Blackwell GPUs (RTX 5090, GB200)

# Base image with CUDA 12.8 and Ubuntu 22.04
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

LABEL maintainer="Brian Worthington <brian@example.com>"
LABEL description="High-performance C++ inference server for OSS-20B optimized for Blackwell"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TENSORRT_ROOT=/opt/tensorrt
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    ca-certificates \
    libcublas-12-8 \
    libcurand-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT 10.8 (download from NVIDIA or mount as volume)
# Option 1: Download during build (requires NVIDIA account)
# RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.Linux.x86_64-gnu.cuda-12.8.tar.gz \
#     && tar -xzf TensorRT-10.8.0.Linux.x86_64-gnu.cuda-12.8.tar.gz -C /opt \
#     && mv /opt/TensorRT-10.8.0 /opt/tensorrt \
#     && rm TensorRT-10.8.0.Linux.x86_64-gnu.cuda-12.8.tar.gz

# Option 2: Copy from local (recommended)
# COPY tensorrt/ /opt/tensorrt/

# Option 3: Mount as volume at runtime (most flexible)
# docker run -v /path/to/tensorrt:/opt/tensorrt ...

# Set TensorRT library path
ENV LD_LIBRARY_PATH=/opt/tensorrt/lib:${LD_LIBRARY_PATH}

# Create application directory
WORKDIR /app

# Copy source code
COPY CMakeLists.txt /app/
COPY src/ /app/src/
COPY .gitignore /app/

# Build the project
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
          -DTensorRT_ROOT=/opt/tensorrt \
          .. && \
    make -j$(nproc) && \
    cp complete_inference /app/ && \
    cp engine_diagnostic /app/ && \
    cp liboss20b_engine.so /app/ && \
    cd /app && rm -rf build

# Create directories for models and data
RUN mkdir -p /app/models /app/tokenizer /app/logs

# Set permissions
RUN chmod +x /app/complete_inference /app/engine_diagnostic

# Expose ports (for future HTTP/gRPC server)
EXPOSE 8000 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ -f /app/complete_inference ] || exit 1

# Default command (can be overridden)
# Usage: docker run oss_srv /app/complete_inference /app/models/model.engine "prompt" 100 0.7 0.9
CMD ["/bin/bash"]
